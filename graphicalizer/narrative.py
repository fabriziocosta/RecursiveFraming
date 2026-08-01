"""Ontology-aware narrative generation for graph substructures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .core import DEFAULT_MODEL, MissingDependencyError, Ontology
from .subgraphs import subgraph_to_payload


SUBGRAPH_NARRATIVE_SYSTEM_PROMPT = """You are a scientific narrative component. Given a connected graph substructure and its ontology, write a plausible, concise narrative of the information contained in that subgraph.

Use every relevant node attribute, edge attribute, ontology type, raw mention, and available node context. Explain relationships and preserve edge direction. Ground exclusively in the supplied subgraph and ontology. Do not introduce entities, relations, causal claims, or unsupported facts. Do not describe the graph mechanically; express its scientific meaning in coherent prose.

Aim approximately for the requested word count. Do not pad the narrative. If the subgraph is ambiguous or the evidence is limited, state that uncertainty briefly."""


@dataclass(frozen=True)
class SubgraphNarrativeConfig:
    """Controls the requested effort of the narrative pass."""

    target_words: int = 120

    def __post_init__(self) -> None:
        if self.target_words < 1:
            raise ValueError("target_words must be at least 1.")

    def to_mapping(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SubgraphNarrativeResult:
    """Narrative and provenance returned for one sampled subgraph."""

    narrative: str
    requested_words: int
    actual_words: int
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    uncertainty: str = ""
    notes: str = ""

    def to_mapping(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubgraphNarrativePrompt:
    """Serializable template or rendered prompt for the narrative pass."""

    name: str = "subgraph-narrator"
    version: str = "0.1.0"
    system: str = SUBGRAPH_NARRATIVE_SYSTEM_PROMPT
    rendered: bool = False
    render_context: Mapping[str, Any] = field(default_factory=dict)
    user_prefix: str = ""

    @classmethod
    def default(cls) -> "SubgraphNarrativePrompt":
        return cls()

    def render(
        self,
        ontology: Ontology,
        config: SubgraphNarrativeConfig,
        model: Optional[str] = None,
    ) -> "SubgraphNarrativePrompt":
        """Create an inspectable prompt containing runtime ontology and effort."""
        if self.rendered:
            raise ValueError("This prompt configuration has already been rendered.")
        ontology_context = json.dumps(
            {
                "metadata": ontology.metadata,
                **ontology.prompt_schema(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        rendered_user_prefix = (
            "RUNTIME NARRATIVE EFFORT\n"
            "Treat the narrative_config in the request payload as an approximate "
            "target, not a reason to add unsupported facts.\n\n"
            "ONTOLOGY\n"
            "Use the exact entity and relation identifiers supplied here when "
            "interpreting typed graph attributes:\n"
            f"{ontology_context}"
        )
        return SubgraphNarrativePrompt(
            name=self.name,
            version=self.version,
            system=self.system,
            rendered=True,
            render_context={
                "template_name": self.name,
                "template_version": self.version,
                "model": model,
                "target_words": config.target_words,
                "ontology": {
                    "id": ontology.metadata.get("id"),
                    "version": ontology.metadata.get("version"),
                },
            },
            user_prefix=rendered_user_prefix,
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "system": self.system,
            "rendered": self.rendered,
            "render_context": dict(self.render_context),
            "user_prefix": self.user_prefix,
        }

    def to_yaml(self) -> str:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("Install PyYAML to serialize narrative prompts.") from exc
        return yaml.safe_dump(self.to_mapping(), allow_unicode=True, sort_keys=False)

    def write_yaml(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SubgraphNarrativePrompt":
        required = {"name", "version", "system"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Narrative prompt is missing: {', '.join(missing)}")
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            system=str(data["system"]),
            rendered=bool(data.get("rendered", False)),
            render_context=dict(data.get("render_context", {})),
            user_prefix=str(data.get("user_prefix", "")),
        )

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "SubgraphNarrativePrompt":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("Install PyYAML to load narrative prompts.") from exc
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Narrative prompt YAML must contain a mapping.")
        return cls.from_mapping(data)


class OpenAISubgraphNarrator:
    """Generate a grounded narrative through a structured LLM client.

    The historical class name is retained for compatibility. The client only
    needs the internal responses.parse contract, so it also works with the
    Ollama adapter.
    """

    def __init__(
        self,
        ontology: Ontology,
        *,
        model: str = DEFAULT_MODEL,
        prompt: Optional[SubgraphNarrativePrompt] = None,
        client: Any = None,
        prompt_snapshot_path: Optional[str | Path] = None,
        request_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty.")
        self.ontology = ontology
        self.model = model
        self.prompt_template = prompt or SubgraphNarrativePrompt.default()
        self.prompt_snapshot_path = (
            Path(prompt_snapshot_path) if prompt_snapshot_path else None
        )
        self.request_options = dict(request_options or {})
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise MissingDependencyError("Install the openai package to call the API.") from exc
            client = OpenAI()
        self.client = client
        self._schema = self._make_schema()

    @classmethod
    def from_graphicalizer(
        cls,
        graphicalizer: Any,
        *,
        prompt: Optional[SubgraphNarrativePrompt] = None,
        prompt_snapshot_path: Optional[str | Path] = None,
    ) -> "OpenAISubgraphNarrator":
        """Reuse the graphicalizer's ontology, model, and configured API client."""
        return cls(
            graphicalizer.ontology,
            model=graphicalizer.llm_client.model,
            prompt=prompt,
            client=graphicalizer.llm_client.client,
            prompt_snapshot_path=prompt_snapshot_path,
            request_options=getattr(graphicalizer.llm_client, "request_options", None),
        )

    @staticmethod
    def _make_schema() -> Any:
        try:
            from pydantic import BaseModel, ConfigDict
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("Install pydantic for structured API outputs.") from exc

        class StrictModel(BaseModel):
            model_config = ConfigDict(extra="forbid")

        class Narrative(StrictModel):
            narrative: str
            uncertainty: str
            notes: str

        return Narrative

    @staticmethod
    def _parsed(response: Any) -> Any:
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("OpenAI returned no parsed narrative output.")
        return parsed

    def narrate(
        self,
        graph: Any,
        config: Optional[SubgraphNarrativeConfig] = None,
    ) -> SubgraphNarrativeResult:
        """Generate a narrative using all serialized information in the graph."""
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot narrate an empty subgraph.")
        settings = config or SubgraphNarrativeConfig()
        prompt = (
            self.prompt_template
            if self.prompt_template.rendered
            else self.prompt_template.render(self.ontology, settings, model=self.model)
        )
        if self.prompt_snapshot_path is not None:
            prompt.write_yaml(self.prompt_snapshot_path)
        payload = {
            "subgraph": subgraph_to_payload(graph),
            "ontology": self.ontology.prompt_schema(),
            "narrative_config": settings.to_mapping(),
        }
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": prompt.system},
                {
                    "role": "user",
                    "content": self._user_content(prompt.user_prefix, payload),
                },
            ],
            text_format=self._schema,
            **self.request_options,
        )
        parsed = self._parsed(response)
        narrative = (parsed.narrative or "").strip()
        if not narrative:
            raise RuntimeError("OpenAI returned an empty narrative.")
        edge_ids = tuple(edge["id"] for edge in payload["subgraph"]["edges"])
        node_ids = tuple(node["id"] for node in payload["subgraph"]["nodes"])
        return SubgraphNarrativeResult(
            narrative=narrative,
            requested_words=settings.target_words,
            actual_words=len(narrative.split()),
            node_ids=node_ids,
            edge_ids=edge_ids,
            uncertainty=(parsed.uncertainty or "").strip(),
            notes=(parsed.notes or "").strip(),
        )

    @staticmethod
    def _user_content(prefix: str, payload: Mapping[str, Any]) -> str:
        """Keep stable ontology guidance before the changing graph payload."""
        if not prefix.strip():
            return json.dumps(payload, ensure_ascii=False)
        return json.dumps(
            {"_runtime_instructions": prefix, **payload},
            ensure_ascii=False,
        )


SubgraphNarrator = OpenAISubgraphNarrator


__all__ = [
    "OpenAISubgraphNarrator",
    "SubgraphNarrator",
    "SUBGRAPH_NARRATIVE_SYSTEM_PROMPT",
    "SubgraphNarrativeConfig",
    "SubgraphNarrativePrompt",
    "SubgraphNarrativeResult",
]
