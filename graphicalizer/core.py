"""LLM-assisted graphicalization of scientific text.

The module implements a three-pass workflow:

1. Extract important entities and free-form relations from text.
2. Build and validate a graph before imposing any ontology.
3. Cast the extracted vocabulary into a supplied entity/relation ontology.
4. Generate grounded explanatory context for each final graph node.
5. Attach the ontology assignments and context to the validated graph.

Provider-specific code is isolated in small structured-output adapters. The
graph builder, ontology loader, validator, and orchestration layer depend only
on :class:`StructuredLLMClient` and can be used with a fake client in tests
without making an API call.

Runtime dependencies for the OpenAI client are ``openai``, ``pydantic``,
``networkx``, and ``PyYAML``. Graph rendering uses the Graphviz ``dot``
executable when available and falls back to ``pygraphviz`` when its Graphviz
bindings are installed. Dependencies are imported lazily where possible so
this module can still be imported for local schema and validation work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from .embeddings import NodeContextEmbedder, NodeEmbeddingConfig


# Cheap, fast default for early extraction experiments. Override explicitly
# when a higher-capability model is needed.
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_OLLAMA_MODEL = "llama3.2"


class GraphicalizerError(RuntimeError):
    """Base exception for graphicalization failures."""


class MissingDependencyError(GraphicalizerError):
    """Raised when an optional runtime dependency is needed but unavailable."""


class LLMResponseError(GraphicalizerError):
    """Raised when the model does not return the expected structured object."""


class GraphValidationError(GraphicalizerError):
    """Raised when strict graph validation is requested and validation fails."""

    def __init__(self, report: "ValidationReport") -> None:
        self.report = report
        super().__init__("Graph validation failed: " + "; ".join(report.issues))


class CastingValidationError(GraphicalizerError):
    """Raised when ontology casting is structurally or semantically invalid."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("Ontology casting validation failed: " + "; ".join(self.issues))


@dataclass(frozen=True)
class FreeEntity:
    """An entity identified before ontology casting."""

    entity_id: str
    mention_text: str
    free_type: str
    description: str = ""
    evidence: str = ""
    confidence: Optional[float] = None


@dataclass(frozen=True)
class FreeRelation:
    """A relation identified before ontology casting."""

    relation_id: str
    source_id: str
    target_id: str
    free_type: str
    evidence: str = ""
    confidence: Optional[float] = None


@dataclass(frozen=True)
class FreeGraphExtraction:
    """The model's unconstrained entity/relation extraction."""

    entities: Tuple[FreeEntity, ...] = ()
    relations: Tuple[FreeRelation, ...] = ()
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OntologyEntity:
    """An entity assigned to a type from the supplied ontology."""

    entity_id: str
    ontology_type: Optional[str]
    canonical_name: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)
    confidence: Optional[float] = None


@dataclass(frozen=True)
class OntologyRelation:
    """A relation assigned to a type from the supplied ontology."""

    relation_id: str
    source_id: str
    target_id: str
    ontology_type: Optional[str]
    attributes: Mapping[str, str] = field(default_factory=dict)
    confidence: Optional[float] = None


@dataclass(frozen=True)
class OntologyCasting:
    """The second-pass ontology assignments."""

    entities: Tuple[OntologyEntity, ...] = ()
    relations: Tuple[OntologyRelation, ...] = ()
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodeContext:
    """Grounded explanatory context generated for one final graph node."""

    entity_id: str
    summary: str
    evidence: Tuple[str, ...] = ()
    uncertainty: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodeContextConfig:
    """Controls the optional third-pass node-context annotation."""

    max_sentences: int = 3
    max_evidence_items: int = 2
    include_evidence: bool = True
    include_uncertainty: bool = True

    def __post_init__(self) -> None:
        if self.max_sentences < 1:
            raise ValueError("max_sentences must be at least 1.")
        if self.max_evidence_items < 0:
            raise ValueError("max_evidence_items must be non-negative.")

    def to_mapping(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "NodeContextConfig":
        return cls(
            max_sentences=int(data.get("max_sentences", 3)),
            max_evidence_items=int(data.get("max_evidence_items", 2)),
            include_evidence=bool(data.get("include_evidence", True)),
            include_uncertainty=bool(data.get("include_uncertainty", True)),
        )


@dataclass(frozen=True)
class ValidationReport:
    """Connectivity and sparsity checks for a graph."""

    valid: bool
    connected: bool
    sparse: bool
    node_count: int
    edge_count: int
    density: float
    max_neighbor_count: int
    hub_nodes: Tuple[str, ...] = ()
    issues: Tuple[str, ...] = ()
    density_ok: bool = True
    hubs_ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphicalizationResult:
    """All intermediate and final artifacts produced by the pipeline.

    The raw stages preserve the original model output, while the normalized
    stages are the topology passed to ontology casting. The graph field is the
    final ontology-typed graph.
    """

    raw_extraction: FreeGraphExtraction
    normalized_extraction: FreeGraphExtraction
    raw_graph: Any
    normalized_graph: Any
    raw_validation: ValidationReport
    normalized_validation: ValidationReport
    ontology_casting: OntologyCasting
    graph: Any
    final_validation: ValidationReport
    normalization_notes: Tuple[str, ...] = ()
    node_contexts: Tuple[NodeContext, ...] = ()
    normalization_report: Mapping[str, Any] = field(default_factory=dict)
    run_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def free_extraction(self) -> FreeGraphExtraction:
        """Backward-compatible alias for the normalized extraction."""
        return self.normalized_extraction

    @property
    def typed_graph(self) -> Any:
        """Preferred explicit name for the final graph."""
        return self.graph


class StructuredLLMClient(Protocol):
    """Interface required by :class:`LLMGraphicalizer`."""

    def extract_free_graph(self, text: str) -> FreeGraphExtraction:
        """Extract unconstrained entities and relations from text."""

    def cast_to_ontology(
        self,
        text: str,
        extraction: FreeGraphExtraction,
        ontology: "Ontology",
    ) -> OntologyCasting:
        """Assign ontology types without changing the extracted topology."""

    def enrich_node_contexts(
        self,
        text: str,
        extraction: FreeGraphExtraction,
        casting: OntologyCasting,
        ontology: "Ontology",
        context_config: NodeContextConfig,
    ) -> Tuple[NodeContext, ...]:
        """Generate grounded context for each final graph node."""


@dataclass(frozen=True)
class Ontology:
    """Loaded entity and relation ontology with compact prompt helpers."""

    entity_types: Mapping[str, Mapping[str, Any]]
    relation_types: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Ontology":
        return cls(
            entity_types=dict(data.get("entity_types", {})),
            relation_types=dict(data.get("relation_types", {})),
            metadata=dict(data.get("ontology", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Ontology":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingDependencyError("Install PyYAML to load ontology YAML files.") from exc

        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, Mapping):
            raise ValueError("Ontology YAML must contain a mapping at its root.")
        return cls.from_mapping(data)

    def prompt_schema(self) -> Dict[str, Any]:
        """Return only the ontology material needed by the casting prompt."""
        return {
            "entity_types": self.entity_types,
            "relation_types": self.relation_types,
        }

    def unknown_entity_types(self, casting: OntologyCasting) -> Tuple[str, ...]:
        values = {
            entity.ontology_type
            for entity in casting.entities
            if entity.ontology_type is not None
        }
        return tuple(sorted(values - set(self.entity_types)))

    def unknown_relation_types(self, casting: OntologyCasting) -> Tuple[str, ...]:
        values = {
            relation.ontology_type
            for relation in casting.relations
            if relation.ontology_type is not None
        }
        return tuple(sorted(values - set(self.relation_types)))


def validate_free_extraction(extraction: FreeGraphExtraction) -> Tuple[str, ...]:
    """Return structural issues in the model's free-form extraction."""
    issues: List[str] = []
    entity_ids = [entity.entity_id for entity in extraction.entities]
    relation_ids = [relation.relation_id for relation in extraction.relations]
    duplicate_entities = sorted(
        entity_id for entity_id in set(entity_ids) if entity_ids.count(entity_id) > 1
    )
    duplicate_relations = sorted(
        relation_id for relation_id in set(relation_ids) if relation_ids.count(relation_id) > 1
    )
    if duplicate_entities:
        issues.append("duplicate entity ids: " + ", ".join(duplicate_entities))
    if duplicate_relations:
        issues.append("duplicate relation ids: " + ", ".join(duplicate_relations))
    known_entities = set(entity_ids)
    for relation in extraction.relations:
        missing = [
            endpoint
            for endpoint in (relation.source_id, relation.target_id)
            if endpoint not in known_entities
        ]
        if missing:
            issues.append(
                f"relation {relation.relation_id!r} references unknown endpoint(s): "
                + ", ".join(sorted(set(missing)))
            )
    return tuple(issues)


def validate_ontology_casting(
    extraction: FreeGraphExtraction,
    casting: OntologyCasting,
    ontology: Ontology,
    *,
    require_types: bool = False,
) -> Tuple[str, ...]:
    """Return structural and ontology-vocabulary issues in a casting."""
    issues: List[str] = []
    expected_entities = {entity.entity_id for entity in extraction.entities}
    actual_entities = [entity.entity_id for entity in casting.entities]
    expected_relations = {relation.relation_id for relation in extraction.relations}
    actual_relations = [relation.relation_id for relation in casting.relations]

    if set(actual_entities) != expected_entities or len(actual_entities) != len(expected_entities):
        issues.append(
            "entity casting ids do not exactly match extraction "
            f"(missing={sorted(expected_entities - set(actual_entities))}, "
            f"extra={sorted(set(actual_entities) - expected_entities)})"
        )
    if set(actual_relations) != expected_relations or len(actual_relations) != len(expected_relations):
        issues.append(
            "relation casting ids do not exactly match extraction "
            f"(missing={sorted(expected_relations - set(actual_relations))}, "
            f"extra={sorted(set(actual_relations) - expected_relations)})"
        )

    for entity in casting.entities:
        if entity.ontology_type is None:
            if require_types:
                issues.append(f"entity {entity.entity_id!r} has no ontology type")
        elif entity.ontology_type not in ontology.entity_types:
            issues.append(
                f"entity {entity.entity_id!r} has unknown ontology type "
                f"{entity.ontology_type!r}"
            )

    extraction_relations = {
        relation.relation_id: relation for relation in extraction.relations
    }
    for relation in casting.relations:
        source = extraction_relations.get(relation.relation_id)
        if source is not None and (
            relation.source_id != source.source_id
            or relation.target_id != source.target_id
        ):
            issues.append(
                f"relation {relation.relation_id!r} changed endpoints during casting"
            )
        if relation.ontology_type is None:
            if require_types:
                issues.append(f"relation {relation.relation_id!r} has no ontology type")
        elif relation.ontology_type not in ontology.relation_types:
            issues.append(
                f"relation {relation.relation_id!r} has unknown ontology type "
                f"{relation.ontology_type!r}"
            )
    return tuple(issues)


class NetworkXGraphBuilder:
    """Build a directed NetworkX graph while preserving model provenance.

    One directed edge is stored per source/target pair. If the model emits
    multiple relation mentions for that pair, they are retained in the
    ``raw_relations`` and ``ontology_relations`` edge attributes rather than
    silently overwriting one another.
    """

    def __init__(self) -> None:
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingDependencyError("Install networkx to build graphicalizer graphs.") from exc
        self.nx = nx

    def build(
        self,
        extraction: FreeGraphExtraction,
        casting: Optional[OntologyCasting] = None,
    ) -> Any:
        graph = self.nx.MultiDiGraph()
        casting_entities = {item.entity_id: item for item in (casting.entities if casting else ())}
        casting_relations = {item.relation_id: item for item in (casting.relations if casting else ())}

        for entity in extraction.entities:
            typed = casting_entities.get(entity.entity_id)
            attributes: Dict[str, Any] = {
                "mention_text": entity.mention_text,
                "free_type": entity.free_type,
                "description": entity.description,
                "evidence": entity.evidence,
                "confidence": entity.confidence,
            }
            if typed is not None:
                attributes.update(
                    {
                        "ontology_type": typed.ontology_type,
                        "canonical_name": typed.canonical_name,
                        "ontology_attributes": dict(typed.attributes),
                        "ontology_confidence": typed.confidence,
                    }
                )
            graph.add_node(entity.entity_id, **attributes)

        for relation in extraction.relations:
            if relation.source_id not in graph or relation.target_id not in graph:
                raise GraphicalizerError(
                    f"Relation {relation.relation_id!r} references an unknown endpoint."
                )
            edge: Dict[str, Any] = {
                "raw_relations": [
                    {
                        "relation_id": relation.relation_id,
                        "free_type": relation.free_type,
                        "evidence": relation.evidence,
                        "confidence": relation.confidence,
                    }
                ],
                "ontology_relations": [],
            }
            typed = casting_relations.get(relation.relation_id)
            if typed is not None:
                edge["ontology_relations"].append(
                    {
                        "relation_id": typed.relation_id,
                        "ontology_type": typed.ontology_type,
                        "attributes": dict(typed.attributes),
                        "confidence": typed.confidence,
                    }
                )
            graph.add_edge(
                relation.source_id,
                relation.target_id,
                key=relation.relation_id,
                **edge,
            )
        return graph


class GraphvizError(GraphicalizerError):
    """Raised when DOT generation or Graphviz rendering fails."""


def _dot_escape(value: Any) -> str:
    """Escape a value for a quoted DOT string."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _node_dot_label(
    node_id: Any,
    attributes: Mapping[str, Any],
    *,
    ontology_only: bool = False,
) -> str:
    """Build a two-line label: ontology type followed by the raw mention."""
    raw_entity = attributes.get("mention_text") or attributes.get("canonical_name") or node_id
    ontology_type = attributes.get("ontology_type")
    free_type = attributes.get("free_type")
    if ontology_only and not ontology_type:
        raise GraphicalizerError(f"Node {node_id!r} has no ontology_type label.")
    if ontology_type:
        lines = [str(ontology_type), str(raw_entity)]
    elif free_type:
        lines = [str(free_type), str(raw_entity)]
    else:
        lines = [str(raw_entity)]
    return "\n".join(lines)


def _edge_dot_label(
    attributes: Mapping[str, Any],
    *,
    ontology_only: bool = False,
) -> str:
    """Build a label from typed relation names, falling back to free names."""
    typed = attributes.get("ontology_relations", []) or []
    typed_names = [item.get("ontology_type") for item in typed if item.get("ontology_type")]
    if typed_names:
        return ", ".join(typed_names)

    if ontology_only:
        raise GraphicalizerError("Edge has no ontology relation label.")

    raw = attributes.get("raw_relations", []) or []
    raw_names = [item.get("free_type") for item in raw if item.get("free_type")]
    return ", ".join(raw_names)


def validate_ontology_labeled_graph(graph: Any) -> Any:
    """Enforce ontology labels on every node and relation edge.

    The function returns the graph for convenient pipeline use and raises a
    ``GraphicalizerError`` if any extracted object remains uncategorized.
    """
    missing_nodes = [
        str(node_id)
        for node_id, attributes in graph.nodes(data=True)
        if not str(attributes.get("ontology_type") or "").strip()
    ]
    missing_edges: List[str] = []
    if graph.is_multigraph():
        edge_rows = graph.edges(data=True, keys=True)
        edge_rows = ((source, target, attributes) for source, target, _key, attributes in edge_rows)
    else:
        edge_rows = graph.edges(data=True)
    for source, target, attributes in edge_rows:
        raw_relations = attributes.get("raw_relations", []) or []
        typed_relations = attributes.get("ontology_relations", []) or []
        raw_ids = {item.get("relation_id") for item in raw_relations}
        typed_ids = {item.get("relation_id") for item in typed_relations}
        if not typed_relations or raw_ids != typed_ids or any(
            not str(item.get("ontology_type") or "").strip()
            for item in typed_relations
        ):
            missing_edges.append(f"{source!r}->{target!r}")

    if missing_nodes or missing_edges:
        problems = []
        if missing_nodes:
            problems.append("nodes without ontology_type: " + ", ".join(missing_nodes))
        if missing_edges:
            problems.append("edges without complete ontology relation labels: " + ", ".join(missing_edges))
        raise GraphicalizerError("Ontology labeling requirement failed; " + "; ".join(problems))
    return graph


def graph_to_dot(
    graph: Any,
    *,
    graph_name: str = "GraphicalizedGraph",
    include_edge_labels: bool = True,
    require_ontology_labels: bool = False,
) -> str:
    """Convert a NetworkX graph into Graphviz DOT text.

    The function uses only graph attributes and does not require ``pydot``.
    Nodes expose the ontology type and original mention text; edges expose
    ontology relation names or free-form relation names. Internal node ids
    remain available in the graph structure but are not printed in labels.
    """
    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise TypeError("graph_to_dot expects a NetworkX-like graph.")

    if require_ontology_labels:
        validate_ontology_labeled_graph(graph)

    directed = bool(graph.is_directed())
    graph_keyword = "digraph" if directed else "graph"
    connector = "->" if directed else "--"
    lines = [f'{graph_keyword} "{_dot_escape(graph_name)}" {{', "  rankdir=LR;"]

    for node_id, attributes in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
        label = _node_dot_label(
            node_id,
            attributes,
            ontology_only=require_ontology_labels,
        )
        lines.append(
            f'  "{_dot_escape(node_id)}" '
            f'[label="{_dot_escape(label)}", shape=box];'
        )

    if graph.is_multigraph():
        edge_iterator = graph.edges(data=True, keys=True)
        edge_rows = ((source, target, attributes) for source, target, _key, attributes in edge_iterator)
    else:
        edge_rows = graph.edges(data=True)
    for source, target, attributes in sorted(
        edge_rows,
        key=lambda item: (
            str(item[0]),
            str(item[1]),
            _edge_dot_label(item[2], ontology_only=require_ontology_labels),
        ),
    ):
        label = _edge_dot_label(attributes, ontology_only=require_ontology_labels)
        label_attribute = f', label="{_dot_escape(label)}"' if include_edge_labels and label else ""
        lines.append(
            f'  "{_dot_escape(source)}" {connector} "{_dot_escape(target)}"'
            f' [color="#555555"{label_attribute}];'
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_dot(
    graph: Any,
    path: str | Path,
    *,
    graph_name: str = "GraphicalizedGraph",
    include_edge_labels: bool = True,
    require_ontology_labels: bool = False,
) -> Path:
    """Write a NetworkX graph as a DOT file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        graph_to_dot(
            graph,
            graph_name=graph_name,
            include_edge_labels=include_edge_labels,
            require_ontology_labels=require_ontology_labels,
        ),
        encoding="utf-8",
    )
    return output_path


def render_graph(
    graph: Any,
    output_path: str | Path,
    *,
    graph_name: str = "GraphicalizedGraph",
    output_format: Optional[str] = None,
    dot_command: str = "dot",
    include_edge_labels: bool = True,
    require_ontology_labels: bool = False,
) -> Path:
    """Render a graph through the Graphviz ``dot`` executable.

    ``output_format`` may be ``png``, ``svg``, ``pdf``, or another format
    supported by the installed Graphviz version. If omitted, it is inferred
    from the output filename suffix.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_format = output_format or output.suffix.lstrip(".") or "png"
    if not output_format:
        raise ValueError("output_format must not be empty.")
    dot_source = graph_to_dot(
        graph,
        graph_name=graph_name,
        include_edge_labels=include_edge_labels,
        require_ontology_labels=require_ontology_labels,
    )
    dot_executable = shutil.which(dot_command)
    if dot_executable is None:
        try:
            import pygraphviz as pgv
        except ImportError as exc:
            raise GraphvizError(
                f"Graphviz executable {dot_command!r} was not found on PATH and "
                "pygraphviz is not installed. Install Graphviz or pygraphviz to render."
            ) from exc
        try:
            graphviz_graph = pgv.AGraph(string=dot_source)
            graphviz_graph.layout(prog="dot")
            graphviz_graph.draw(str(output))
        except Exception as exc:  # pragma: no cover - Graphviz binding dependent
            raise GraphvizError(f"pygraphviz Graphviz rendering failed: {exc}") from exc
        return output

    completed = subprocess.run(
        [dot_executable, f"-T{output_format}", "-o", str(output)],
        input=dot_source,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or "unknown Graphviz error"
        raise GraphvizError(f"Graphviz dot failed: {error}")
    return output


class GraphValidator:
    """Validate weak connectivity, density, and excessive hub formation.

    Validation uses an undirected projection for structural sparsity because a
    bidirectional pair still represents one local connection. Connectivity is
    weak connectivity for directed graphs. A hub is a node connected to more
    than ``max_degree_fraction`` of all other nodes, with a minimum threshold
    of two neighbors for very small graphs.
    """

    def __init__(
        self,
        *,
        max_density: float = 0.40,
        max_degree_fraction: float = 0.50,
        max_hub_nodes: int = 0,
    ) -> None:
        if not 0 < max_density <= 1:
            raise ValueError("max_density must be in (0, 1].")
        if not 0 < max_degree_fraction <= 1:
            raise ValueError("max_degree_fraction must be in (0, 1].")
        if max_hub_nodes < 0:
            raise ValueError("max_hub_nodes must be non-negative.")
        self.max_density = max_density
        self.max_degree_fraction = max_degree_fraction
        self.max_hub_nodes = max_hub_nodes

    def validate(self, graph: Any) -> ValidationReport:
        import networkx as nx

        if not isinstance(graph, nx.Graph):
            raise TypeError("GraphValidator currently expects a NetworkX graph.")

        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        issues: List[str] = []
        if node_count == 0:
            connected = False
            density = 0.0
            neighbor_counts: Dict[Any, int] = {}
            issues.append("graph has no entities")
        else:
            connected = (
                self._weakly_connected(graph)
                if graph.is_directed()
                else self._connected(graph)
            )
            if not connected:
                issues.append("graph is disconnected")
            undirected = self._undirected_projection(graph)
            density = self._density(undirected)
            neighbor_counts = {
                node: undirected.degree(node) for node in undirected.nodes()
            }

        density_ok = edge_count == 0 if node_count <= 1 else density <= self.max_density
        if not density_ok:
            issues.append(f"graph density {density:.3f} exceeds {self.max_density:.3f}")

        if node_count <= 1:
            hub_nodes: List[str] = []
        else:
            degree_limit = max(
                2,
                int(self.max_degree_fraction * (node_count - 1)),
            )
            hub_nodes = [
                str(node)
                for node, degree in neighbor_counts.items()
                if degree > degree_limit
            ]
        hubs_ok = len(hub_nodes) <= self.max_hub_nodes
        if not hubs_ok:
            issues.append(
                f"graph contains {len(hub_nodes)} hub node(s): {', '.join(hub_nodes)}"
            )

        return ValidationReport(
            valid=connected and density_ok and hubs_ok,
            connected=connected,
            sparse=density_ok,
            node_count=node_count,
            edge_count=edge_count,
            density=density,
            max_neighbor_count=max(neighbor_counts.values(), default=0),
            hub_nodes=tuple(hub_nodes),
            issues=tuple(issues),
            density_ok=density_ok,
            hubs_ok=hubs_ok,
        )

    @staticmethod
    def _weakly_connected(graph: Any) -> bool:
        import networkx as nx

        return nx.is_weakly_connected(graph)

    @staticmethod
    def _connected(graph: Any) -> bool:
        import networkx as nx

        return nx.is_connected(graph)

    @staticmethod
    def _undirected_projection(graph: Any) -> Any:
        import networkx as nx

        return nx.Graph(graph.to_undirected())

    @staticmethod
    def _density(graph: Any) -> float:
        import networkx as nx

        return nx.density(graph)


@dataclass(frozen=True)
class GraphicalizerPrompt:
    """Explicit, versioned prompt configuration for the three-pass workflow.

    Prompt text is data rather than hidden behavior: it can be edited,
    serialized for an experiment, restored later, and passed into the API
    client at construction time.
    """

    name: str
    version: str
    free_extraction_system: str
    ontology_casting_system: str
    rendered: bool = False
    render_context: Mapping[str, Any] = field(default_factory=dict)
    node_context_system: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Prompt name must not be empty.")
        if not self.version.strip():
            raise ValueError("Prompt version must not be empty.")
        if not self.free_extraction_system.strip():
            raise ValueError("Free-extraction prompt must not be empty.")
        if not self.ontology_casting_system.strip():
            raise ValueError("Ontology-casting prompt must not be empty.")
        if not self.node_context_system.strip():
            object.__setattr__(self, "node_context_system", NODE_CONTEXT_SYSTEM_PROMPT)

    @classmethod
    def default(cls) -> "GraphicalizerPrompt":
        """Return the repository's domain-independent prompt configuration."""
        return cls(
            name="ontology-aware-graphicalizer",
            version="0.4.0",
            free_extraction_system=FREE_EXTRACTION_SYSTEM_PROMPT,
            ontology_casting_system=ONTOLOGY_CASTING_SYSTEM_PROMPT,
            node_context_system=NODE_CONTEXT_SYSTEM_PROMPT,
            rendered=False,
            render_context={},
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "free_extraction_system": self.free_extraction_system,
            "ontology_casting_system": self.ontology_casting_system,
            "node_context_system": self.node_context_system,
            "rendered": self.rendered,
            "render_context": dict(self.render_context),
        }

    def render(
        self,
        ontology: Ontology,
        extraction_density: ExtractionDensityConfig,
        model: Optional[str] = None,
        context_config: Optional[NodeContextConfig] = None,
    ) -> "GraphicalizerPrompt":
        """Generate concrete prompts from this template and runtime context.

        The resulting prompt contains the ontology and density configuration
        that the object was initialized with, making the effective prompt
        inspectable and serializable for reproducible experiments.
        """
        if self.rendered:
            raise ValueError("This prompt configuration has already been rendered.")

        density = extraction_density.to_mapping()
        context_settings = context_config or NodeContextConfig()
        ontology_ids = {
            "entity_type_ids": sorted(ontology.entity_types),
            "relation_type_ids": sorted(ontology.relation_types),
        }
        density_context = json.dumps(density, ensure_ascii=False, indent=2, sort_keys=True)
        ontology_id_context = json.dumps(ontology_ids, ensure_ascii=False, indent=2, sort_keys=True)
        ontology_context = json.dumps(
            {
                "metadata": ontology.metadata,
                **ontology.prompt_schema(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        render_context = {
            "template_name": self.name,
            "template_version": self.version,
            "ontology": {
                "id": ontology.metadata.get("id"),
                "version": ontology.metadata.get("version"),
            },
            "entity_type_ids": ontology_ids["entity_type_ids"],
            "relation_type_ids": ontology_ids["relation_type_ids"],
            "extraction_density": density,
            "node_context": context_settings.to_mapping(),
            "model": model,
        }

        free_prompt = (
            f"{self.free_extraction_system}\n\n"
            "RUNTIME EXTRACTION DENSITY\n"
            "Use these effort settings to produce a sufficiently detailed graph; "
            "do not invent unsupported entities or relations to hit a quota. "
            "If the first pass is below the configured minimum, a grounded "
            "expansion pass may follow:\n"
            f"{density_context}\n\n"
            "ONTOLOGY AWARENESS FOR LATER CASTING\n"
            "Pass 1 remains free-form and must not force mentions into these "
            "identifiers, but the following vocabulary will be used in pass 2:\n"
            f"{ontology_id_context}"
        )
        casting_prompt = (
            f"{self.ontology_casting_system}\n\n"
            "RUNTIME EXTRACTION DENSITY REFERENCE\n"
            "The first pass was configured with these approximate targets. "
            "Do not add or remove graph objects during casting:\n"
            f"{density_context}\n\n"
            "TARGET ONTOLOGY\n"
            "Use only the exact dictionary keys under entity_types and relation_types "
            "as ontology_type values. Never use a label, synonym, inverse value, "
            "description, or free-form relation name as an ontology_type. Every "
            "non-null type must match one of those keys exactly:\n"
            f"{ontology_context}"
        )
        node_context_prompt = (
            f"{self.node_context_system}\n\n"
            "RUNTIME NODE-CONTEXT CONFIGURATION\n"
            f"{json.dumps(context_settings.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
            "TARGET ONTOLOGY\n"
            "Use the exact ontology identifiers supplied in the graph payload; "
            "do not introduce new entities or relations.\n"
            f"{ontology_context}"
        )
        return GraphicalizerPrompt(
            name=self.name,
            version=self.version,
            free_extraction_system=free_prompt,
            ontology_casting_system=casting_prompt,
            node_context_system=node_context_prompt,
            rendered=True,
            render_context=render_context,
        )

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize this prompt configuration as stable JSON."""
        return json.dumps(
            self.to_mapping(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GraphicalizerPrompt":
        required = {
            "name",
            "version",
            "free_extraction_system",
            "ontology_casting_system",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Prompt configuration is missing: {', '.join(missing)}")
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            free_extraction_system=str(data["free_extraction_system"]),
            ontology_casting_system=str(data["ontology_casting_system"]),
            node_context_system=str(data.get("node_context_system", NODE_CONTEXT_SYSTEM_PROMPT)),
            rendered=bool(data.get("rendered", False)),
            render_context=dict(data.get("render_context", {})),
        )

    @classmethod
    def from_json(cls, serialized: str) -> "GraphicalizerPrompt":
        data = json.loads(serialized)
        if not isinstance(data, Mapping):
            raise ValueError("Serialized prompt must contain a JSON object.")
        return cls.from_mapping(data)

    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Persist the prompt configuration to a UTF-8 JSON file."""
        Path(path).write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "GraphicalizerPrompt":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_yaml(self) -> str:
        """Serialize the prompt template or resolved prompt as YAML."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingDependencyError("Install PyYAML to serialize prompt YAML.") from exc
        return yaml.safe_dump(
            self.to_mapping(),
            allow_unicode=True,
            sort_keys=False,
        )

    @classmethod
    def from_yaml(cls, serialized: str) -> "GraphicalizerPrompt":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingDependencyError("Install PyYAML to load prompt YAML.") from exc
        data = yaml.safe_load(serialized) or {}
        if not isinstance(data, Mapping):
            raise ValueError("Serialized prompt YAML must contain a mapping.")
        return cls.from_mapping(data)

    def write_yaml(self, path: str | Path) -> None:
        """Persist this prompt configuration as a versioned YAML snapshot."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "GraphicalizerPrompt":
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ExtractionDensityConfig:
    """Separate effort controls for free-form graph extraction.

    ``entities_per_word=0.1`` requests roughly one entity per ten words.
    ``relations_per_entity=2.0`` requests roughly two relations per extracted
    entity. The minimum fraction and retry count control a grounded expansion
    pass when the first extraction is undersized.
    """

    entities_per_word: float = 0.1
    relations_per_entity: float = 2.0
    minimum_entity_fraction: float = 0.75
    density_retries: int = 2

    def __post_init__(self) -> None:
        if self.entities_per_word < 0:
            raise ValueError("entities_per_word must be non-negative.")
        if self.relations_per_entity < 0:
            raise ValueError("relations_per_entity must be non-negative.")
        if not 0 <= self.minimum_entity_fraction <= 1:
            raise ValueError("minimum_entity_fraction must be between 0 and 1.")
        if self.density_retries < 0:
            raise ValueError("density_retries must be non-negative.")

    def target_counts(self, text: str) -> Dict[str, int]:
        """Calculate approximate counts for one input document."""
        word_count = len(text.split())
        entity_count = round(word_count * self.entities_per_word)
        if word_count and self.entities_per_word > 0:
            entity_count = max(1, entity_count)
        relation_count = round(entity_count * self.relations_per_entity)
        return {
            "word_count": word_count,
            "target_entity_count": entity_count,
            "minimum_entity_count": math.ceil(
                entity_count * self.minimum_entity_fraction
            ),
            "target_relation_count": relation_count,
        }

    def request_payload(self, text: str) -> Dict[str, Any]:
        """Return the structured user-side control data sent to pass one."""
        return {
            "source_text": text,
            "extraction_density": self.to_mapping(),
            "target_counts": self.target_counts(text),
        }

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "entities_per_word": self.entities_per_word,
            "relations_per_entity": self.relations_per_entity,
            "minimum_entity_fraction": self.minimum_entity_fraction,
            "density_retries": self.density_retries,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_mapping(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExtractionDensityConfig":
        return cls(
            entities_per_word=float(data.get("entities_per_word", 0.1)),
            relations_per_entity=float(data.get("relations_per_entity", 2.0)),
            minimum_entity_fraction=float(data.get("minimum_entity_fraction", 0.75)),
            density_retries=int(data.get("density_retries", 2)),
        )

    @classmethod
    def from_json(cls, serialized: str) -> "ExtractionDensityConfig":
        data = json.loads(serialized)
        if not isinstance(data, Mapping):
            raise ValueError("Serialized extraction density must contain a JSON object.")
        return cls.from_mapping(data)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ExtractionDensityConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class GraphicalizerConfig:
    """Single public configuration for the complete graphicalizer pipeline."""

    provider: str = "openai"
    model: str = DEFAULT_MODEL
    extraction_density: ExtractionDensityConfig = field(
        default_factory=ExtractionDensityConfig
    )
    node_context: NodeContextConfig = field(default_factory=NodeContextConfig)
    prompt_template_path: Optional[str | Path] = None
    prompt_snapshot_path: Optional[str | Path] = None
    disconnected_policy: str = "largest_component"
    context_policy: str = "all_nodes"
    casting_retries: int = 2
    strict_validation: bool = False
    max_density: float = 0.40
    max_degree_fraction: float = 0.50
    max_hub_nodes: int = 0

    def __post_init__(self) -> None:
        if self.provider not in {"openai", "ollama"}:
            raise ValueError("provider must be either 'openai' or 'ollama'.")
        if not self.model.strip():
            raise ValueError("model must not be empty.")
        if self.disconnected_policy not in {"largest_component", "raise", "preserve"}:
            raise ValueError(
                "disconnected_policy must be 'largest_component', 'raise', or 'preserve'."
            )
        if self.context_policy not in {"all_nodes", "disabled"}:
            raise ValueError("context_policy must be either 'all_nodes' or 'disabled'.")
        if self.casting_retries < 0:
            raise ValueError("casting_retries must be non-negative.")

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "extraction_density": self.extraction_density.to_mapping(),
            "node_context": self.node_context.to_mapping(),
            "prompt_template_path": (
                str(self.prompt_template_path)
                if self.prompt_template_path is not None
                else None
            ),
            "prompt_snapshot_path": (
                str(self.prompt_snapshot_path)
                if self.prompt_snapshot_path is not None
                else None
            ),
            "disconnected_policy": self.disconnected_policy,
            "context_policy": self.context_policy,
            "casting_retries": self.casting_retries,
            "strict_validation": self.strict_validation,
            "max_density": self.max_density,
            "max_degree_fraction": self.max_degree_fraction,
            "max_hub_nodes": self.max_hub_nodes,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GraphicalizerConfig":
        return cls(
            provider=str(data.get("provider", "openai")),
            model=str(data.get("model", DEFAULT_MODEL)),
            extraction_density=ExtractionDensityConfig.from_mapping(
                data.get("extraction_density", {})
            ),
            node_context=NodeContextConfig.from_mapping(
                data.get("node_context", {})
            ),
            prompt_template_path=data.get("prompt_template_path"),
            prompt_snapshot_path=data.get("prompt_snapshot_path"),
            disconnected_policy=str(
                data.get("disconnected_policy", "largest_component")
            ),
            context_policy=str(data.get("context_policy", "all_nodes")),
            casting_retries=int(data.get("casting_retries", 2)),
            strict_validation=bool(data.get("strict_validation", False)),
            max_density=float(data.get("max_density", 0.40)),
            max_degree_fraction=float(data.get("max_degree_fraction", 0.50)),
            max_hub_nodes=int(data.get("max_hub_nodes", 0)),
        )


class OpenAIResponsesClient:
    """Structured-output client for the three LLM passes.

    The official Python SDK's Responses API parser is used with Pydantic
    schemas. The API key is read by the SDK from ``OPENAI_API_KEY`` unless an
    already-configured client is injected.
    """

    provider = "openai"

    def __init__(
        self,
        prompt: GraphicalizerPrompt,
        ontology: Ontology,
        extraction_density: Optional[ExtractionDensityConfig] = None,
        model: str = DEFAULT_MODEL,
        client: Any = None,
        prompt_snapshot_path: Optional[str | Path] = None,
        node_context_config: Optional[NodeContextConfig] = None,
        casting_retries: int = 2,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.ontology = ontology
        self.extraction_density = extraction_density or ExtractionDensityConfig()
        self.node_context_config = node_context_config or NodeContextConfig()
        if casting_retries < 0:
            raise ValueError("casting_retries must be non-negative.")
        self.casting_retries = casting_retries
        self.request_options = dict(options or {})
        self.prompt_template = prompt
        self.model = model
        self.prompt = (
            prompt
            if prompt.rendered
            else prompt.render(
                self.ontology,
                self.extraction_density,
                model=self.model,
                context_config=self.node_context_config,
            )
        )
        self.prompt_snapshot_path = Path(prompt_snapshot_path) if prompt_snapshot_path else None
        if self.prompt_snapshot_path is not None and not self.prompt.rendered:
            raise ValueError("Internal error: prompt snapshot must be a rendered prompt.")
        if self.prompt_snapshot_path is not None:
            self.prompt.write_yaml(self.prompt_snapshot_path)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise MissingDependencyError(
                    "Install the openai package to call the API."
                ) from exc
            client = OpenAI()
        self.client = client
        self._free_schema = self._make_free_schema()
        self._casting_schema = self._make_casting_schema()
        self._node_context_schema = self._make_node_context_schema()

    @staticmethod
    def _make_free_schema() -> Any:
        try:
            from pydantic import BaseModel, ConfigDict, Field
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingDependencyError("Install pydantic for structured API outputs.") from exc

        class StrictModel(BaseModel):
            model_config = ConfigDict(extra="forbid")

        class Entity(StrictModel):
            id: str = Field(description="Stable identifier used by relations, e.g. e1")
            mention_text: str
            free_type: str
            description: str
            evidence: str
            confidence: Optional[float]

        class Relation(StrictModel):
            id: str = Field(description="Stable relation identifier, e.g. r1")
            source_id: str
            target_id: str
            free_type: str
            evidence: str
            confidence: Optional[float]

        class Extraction(StrictModel):
            entities: List[Entity]
            relations: List[Relation]
            notes: str

        return Extraction

    @staticmethod
    def _make_casting_schema() -> Any:
        try:
            from pydantic import BaseModel, ConfigDict
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingDependencyError("Install pydantic for structured API outputs.") from exc

        class StrictModel(BaseModel):
            model_config = ConfigDict(extra="forbid")

        class Attribute(StrictModel):
            key: str
            value: str

        class EntityCasting(StrictModel):
            id: str
            ontology_type: Optional[str]
            canonical_name: str
            attributes: List[Attribute]
            confidence: Optional[float]

        class RelationCasting(StrictModel):
            id: str
            source_id: str
            target_id: str
            ontology_type: Optional[str]
            attributes: List[Attribute]
            confidence: Optional[float]

        class Casting(StrictModel):
            entities: List[EntityCasting]
            relations: List[RelationCasting]
            notes: str

        return Casting

    @staticmethod
    def _make_node_context_schema() -> Any:
        try:
            from pydantic import BaseModel, ConfigDict
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingDependencyError("Install pydantic for structured API outputs.") from exc

        class StrictModel(BaseModel):
            model_config = ConfigDict(extra="forbid")

        class Context(StrictModel):
            id: str
            summary: str
            evidence: List[str]
            uncertainty: str

        class Contexts(StrictModel):
            contexts: List[Context]
            notes: str

        return Contexts

    @staticmethod
    def _parsed(response: Any) -> Any:
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMResponseError("OpenAI returned no parsed structured output.")
        return parsed

    def _parse_response(self, **request: Any) -> Any:
        return self.client.responses.parse(**request, **self.request_options)

    @staticmethod
    def _ontology_casting_from_parsed(parsed: Any) -> OntologyCasting:
        return OntologyCasting(
            entities=tuple(
                OntologyEntity(
                    entity_id=item.id,
                    ontology_type=item.ontology_type,
                    canonical_name=item.canonical_name or "",
                    attributes={
                        attribute.key: attribute.value
                        for attribute in item.attributes
                    },
                    confidence=item.confidence,
                )
                for item in parsed.entities
            ),
            relations=tuple(
                OntologyRelation(
                    relation_id=item.id,
                    source_id=item.source_id,
                    target_id=item.target_id,
                    ontology_type=item.ontology_type,
                    attributes={
                        attribute.key: attribute.value
                        for attribute in item.attributes
                    },
                    confidence=item.confidence,
                )
                for item in parsed.relations
            ),
            notes=parsed.notes or "",
        )

    @staticmethod
    def _fallback_unknown_casting_types(
        casting: OntologyCasting,
        ontology: Ontology,
    ) -> OntologyCasting:
        """Keep a graph usable when the model invents a final type.

        Repair prompts normally correct unknown types. This final guard keeps a
        grounded graph from failing solely because a model emitted a novel
        label: the original proposal is preserved in an attribute.
        """
        fallback_entity = (
            "investigation_entity"
            if "investigation_entity" in ontology.entity_types
            else next(iter(ontology.entity_types), None)
        )
        fallback_relation = (
            "related_to"
            if "related_to" in ontology.relation_types
            else next(iter(ontology.relation_types), None)
        )
        entities = []
        changed = False
        for entity in casting.entities:
            if entity.ontology_type in ontology.entity_types:
                entities.append(entity)
                continue
            attributes = dict(entity.attributes)
            attributes.setdefault(
                "unmapped_ontology_type",
                str(entity.ontology_type or ""),
            )
            entities.append(
                OntologyEntity(
                    entity_id=entity.entity_id,
                    ontology_type=fallback_entity,
                    canonical_name=entity.canonical_name,
                    attributes=attributes,
                    confidence=entity.confidence,
                )
            )
            changed = True
        relations = []
        for relation in casting.relations:
            if relation.ontology_type in ontology.relation_types:
                relations.append(relation)
                continue
            attributes = dict(relation.attributes)
            attributes.setdefault(
                "unmapped_ontology_type",
                str(relation.ontology_type or ""),
            )
            relations.append(
                OntologyRelation(
                    relation_id=relation.relation_id,
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    ontology_type=fallback_relation,
                    attributes=attributes,
                    confidence=relation.confidence,
                )
            )
            changed = True
        if not changed:
            return casting
        note = casting.notes.strip()
        fallback_note = "Applied safe fallback type(s) for unmapped model labels."
        return OntologyCasting(
            entities=tuple(entities),
            relations=tuple(relations),
            notes=f"{note} {fallback_note}".strip(),
        )

    def _free_graph_from_parsed(self, parsed: Any) -> FreeGraphExtraction:
        return FreeGraphExtraction(
            entities=tuple(
                FreeEntity(
                    entity_id=item.id,
                    mention_text=item.mention_text,
                    free_type=item.free_type,
                    description=item.description or "",
                    evidence=item.evidence or "",
                    confidence=item.confidence,
                )
                for item in parsed.entities
            ),
            relations=tuple(
                FreeRelation(
                    relation_id=item.id,
                    source_id=item.source_id,
                    target_id=item.target_id,
                    free_type=item.free_type,
                    evidence=item.evidence,
                    confidence=item.confidence,
                )
                for item in parsed.relations
            ),
            notes=parsed.notes or "",
        )

    def _extract_free_graph_once(
        self,
        *,
        system_prompt: str,
        request_payload: Mapping[str, Any],
    ) -> FreeGraphExtraction:
        response = self._parse_response(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(request_payload, ensure_ascii=False),
                },
            ],
            text_format=self._free_schema,
        )
        return self._free_graph_from_parsed(self._parsed(response))

    @staticmethod
    def _expansion_preserves_previous(
        previous: FreeGraphExtraction,
        candidate: FreeGraphExtraction,
    ) -> bool:
        previous_entities = {entity.entity_id for entity in previous.entities}
        candidate_entities = {entity.entity_id for entity in candidate.entities}
        if not previous_entities <= candidate_entities:
            return False
        previous_relations = {
            relation.relation_id: (relation.source_id, relation.target_id)
            for relation in previous.relations
        }
        candidate_relations = {
            relation.relation_id: (relation.source_id, relation.target_id)
            for relation in candidate.relations
        }
        return all(
            candidate_relations.get(relation_id) == endpoints
            for relation_id, endpoints in previous_relations.items()
        )

    def extract_free_graph(self, text: str) -> FreeGraphExtraction:
        extraction_request = self.extraction_density.request_payload(text)
        extraction = self._extract_free_graph_once(
            system_prompt=self.prompt.free_extraction_system,
            request_payload=extraction_request,
        )
        minimum_entity_count = extraction_request["target_counts"]["minimum_entity_count"]
        for attempt in range(self.extraction_density.density_retries):
            validation_issues = validate_free_extraction(extraction)
            if len(extraction.entities) >= minimum_entity_count and not validation_issues:
                break
            expansion_payload = {
                **extraction_request,
                "current_extraction": extraction.to_dict(),
                "current_entity_count": len(extraction.entities),
                "current_relation_count": len(extraction.relations),
                "current_validation_issues": validation_issues,
                "required_minimum_entity_count": minimum_entity_count,
                "expansion_attempt": attempt + 1,
            }
            expansion_system = (
                f"{self.prompt.free_extraction_system}\n\n"
                "DENSITY EXPANSION\n"
                "The previous extraction is below the requested minimum. Return a "
                "complete replacement extraction that preserves every existing id "
                "and supported fact, while adding distinct entities and relations "
                "grounded in the source text. Aim for at least the required minimum "
                "entity count. Every added entity must participate in a supported "
                "relation; do not invent facts, duplicate mentions, or add links "
                "solely to satisfy the count. Ensure every relation endpoint refers "
                "to an entity returned in the same extraction. Correct any listed "
                "validation issues.\n"
                f"Current validation issues: {json.dumps(validation_issues, ensure_ascii=False)}\n"
                f"Required minimum entity count: {minimum_entity_count}\n"
                f"Expansion attempt: {attempt + 1}"
            )
            candidate = self._extract_free_graph_once(
                system_prompt=expansion_system,
                request_payload=expansion_payload,
            )
            candidate_issues = validate_free_extraction(candidate)
            if (
                not candidate_issues
                and self._expansion_preserves_previous(extraction, candidate)
            ):
                extraction = candidate
        return extraction

    def cast_to_ontology(
        self,
        text: str,
        extraction: FreeGraphExtraction,
        ontology: Ontology,
    ) -> OntologyCasting:
        if ontology.prompt_schema() != self.ontology.prompt_schema():
            raise ValueError(
                "The casting ontology differs from the ontology used to render the prompt."
            )
        payload = {
            "source_text": text,
            "free_extraction": extraction.to_dict(),
            "ontology": ontology.prompt_schema(),
        }
        response = self._parse_response(
            model=self.model,
            input=[
                {"role": "system", "content": self.prompt.ontology_casting_system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            text_format=self._casting_schema,
        )
        casting = self._ontology_casting_from_parsed(self._parsed(response))
        issues = validate_ontology_casting(
            extraction,
            casting,
            ontology,
            require_types=True,
        )

        for attempt in range(self.casting_retries):
            if not issues:
                break
            repair_payload = {
                **payload,
                "previous_casting": casting.to_dict(),
                "validation_issues": issues,
                "repair_attempt": attempt + 1,
                "required_entity_ids": [
                    entity.entity_id for entity in extraction.entities
                ],
                "required_relation_ids": [
                    relation.relation_id for relation in extraction.relations
                ],
            }
            repair_system = (
                f"{self.prompt.ontology_casting_system}\n\n"
                "CASTING REPAIR\n"
                "The previous casting violated the ontology contract. Return a "
                "complete replacement, preserving every id and endpoint. Correct "
                "all listed issues. Use only exact keys from ontology.entity_types "
                "and ontology.relation_types; use related_to when no specific "
                "relation is justified; never use inverse values or free-form "
                "labels. Do not return null ontology types because every graph "
                "object must be ontology-labeled. Return exactly one entity casting "
                "for every required_entity_id and exactly one relation casting for "
                "every required_relation_id; do not omit an item even if its type "
                "is uncertain.\n"
                f"Validation issues: {json.dumps(issues, ensure_ascii=False)}"
            )
            response = self._parse_response(
                model=self.model,
                input=[
                    {"role": "system", "content": repair_system},
                    {
                        "role": "user",
                        "content": json.dumps(repair_payload, ensure_ascii=False),
                    },
                ],
                text_format=self._casting_schema,
            )
            casting = self._ontology_casting_from_parsed(self._parsed(response))
            issues = validate_ontology_casting(
                extraction,
                casting,
                ontology,
                require_types=True,
            )
        if issues:
            casting = self._fallback_unknown_casting_types(casting, ontology)
        return casting

    def enrich_node_contexts(
        self,
        text: str,
        extraction: FreeGraphExtraction,
        casting: OntologyCasting,
        ontology: Ontology,
        context_config: NodeContextConfig,
    ) -> Tuple[NodeContext, ...]:
        """Generate grounded context for every node in the final graph."""
        typed_entities = {
            entity.entity_id: entity
            for entity in casting.entities
        }
        typed_relations = {
            relation.relation_id: relation
            for relation in casting.relations
        }
        relation_rows = []
        for relation in extraction.relations:
            typed = typed_relations.get(relation.relation_id)
            relation_rows.append(
                {
                    "id": relation.relation_id,
                    "source_id": relation.source_id,
                    "target_id": relation.target_id,
                    "raw_relation": relation.free_type,
                    "ontology_relation": typed.ontology_type if typed else None,
                    "evidence": relation.evidence,
                }
            )

        nodes = []
        for entity in extraction.entities:
            typed = typed_entities.get(entity.entity_id)
            nodes.append(
                {
                    "id": entity.entity_id,
                    "raw_mention": entity.mention_text,
                    "raw_type": entity.free_type,
                    "ontology_type": typed.ontology_type if typed else None,
                    "canonical_name": typed.canonical_name if typed else "",
                    "incoming_relations": [
                        row for row in relation_rows if row["target_id"] == entity.entity_id
                    ],
                    "outgoing_relations": [
                        row for row in relation_rows if row["source_id"] == entity.entity_id
                    ],
                }
            )

        payload = {
            "source_text": text,
            "nodes": nodes,
            "relations": relation_rows,
            "ontology": ontology.prompt_schema(),
            "context_config": context_config.to_mapping(),
        }
        response = self._parse_response(
            model=self.model,
            input=[
                {"role": "system", "content": self.prompt.node_context_system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            text_format=self._node_context_schema,
        )
        parsed = self._parsed(response)
        return tuple(
            NodeContext(
                entity_id=item.id,
                summary=item.summary,
                evidence=(
                    tuple(item.evidence[: context_config.max_evidence_items])
                    if context_config.include_evidence
                    else ()
                ),
                uncertainty=(
                    item.uncertainty or ""
                    if context_config.include_uncertainty
                    else ""
                ),
            )
            for item in parsed.contexts
        )


class LLMGraphicalizer:
    """Orchestrate extraction, topology normalization, casting, and context.

    The model is allowed to return a disconnected graph so that the raw
    result remains inspectable. By default, the deterministic normalization
    step retains the largest weakly connected component before ontology
    casting. This removes unsupported peripheral nodes without inventing
    relations between them.
    """

    @classmethod
    def from_llm_client(
        cls,
        ontology: Ontology,
        llm_client: StructuredLLMClient,
        config: Optional[GraphicalizerConfig] = None,
        *,
        embedding_model: Any = None,
        embedding_config: Optional[NodeEmbeddingConfig] = None,
    ) -> "LLMGraphicalizer":
        """Build the orchestrator around any StructuredLLMClient implementation."""
        settings = config or GraphicalizerConfig()
        return cls(
            llm_client=llm_client,
            ontology=ontology,
            validator=GraphValidator(
                max_density=settings.max_density,
                max_degree_fraction=settings.max_degree_fraction,
                max_hub_nodes=settings.max_hub_nodes,
            ),
            strict_validation=settings.strict_validation,
            disconnected_policy=settings.disconnected_policy,
            node_context_config=settings.node_context,
            context_policy=settings.context_policy,
            embedding_model=embedding_model,
            embedding_config=embedding_config,
        )

    @classmethod
    def from_provider(
        cls,
        ontology: Ontology,
        config: Optional[GraphicalizerConfig] = None,
        *,
        provider: Optional[str] = None,
        client: Any = None,
        host: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
        prompt_template: Optional[GraphicalizerPrompt] = None,
        embedding_model: Any = None,
        embedding_config: Optional[NodeEmbeddingConfig] = None,
    ) -> "LLMGraphicalizer":
        """Construct the graphicalizer using the selected LLM provider."""
        selected_provider = provider or (config.provider if config is not None else "openai")
        settings = config or (
            GraphicalizerConfig(provider="ollama", model=DEFAULT_OLLAMA_MODEL)
            if selected_provider == "ollama"
            else GraphicalizerConfig()
        )
        if selected_provider == "openai":
            return cls.from_openai(
                ontology,
                settings,
                client=client,
                options=options,
                prompt_template=prompt_template,
                embedding_model=embedding_model,
                embedding_config=embedding_config,
            )
        if selected_provider == "ollama":
            return cls.from_ollama(
                ontology,
                settings,
                client=client,
                host=host,
                options=options,
                prompt_template=prompt_template,
                embedding_model=embedding_model,
                embedding_config=embedding_config,
            )
        raise ValueError(
            "Unsupported LLM provider "
            f"{selected_provider!r}; choose 'openai' or 'ollama'."
        )

    @classmethod
    def from_ollama(
        cls,
        ontology: Ontology,
        config: Optional[GraphicalizerConfig] = None,
        *,
        client: Any = None,
        host: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
        prompt_template: Optional[GraphicalizerPrompt] = None,
        embedding_model: Any = None,
        embedding_config: Optional[NodeEmbeddingConfig] = None,
    ) -> "LLMGraphicalizer":
        """Build the pipeline against a local Ollama chat server."""
        settings = config or GraphicalizerConfig(
            provider="ollama",
            model=DEFAULT_OLLAMA_MODEL,
        )
        if prompt_template is not None:
            template = prompt_template
        elif settings.prompt_template_path is not None:
            template = GraphicalizerPrompt.from_yaml_file(
                settings.prompt_template_path
            )
        else:
            template = GraphicalizerPrompt.default()
        llm_client = OllamaStructuredClient(
            prompt=template,
            ontology=ontology,
            extraction_density=settings.extraction_density,
            model=settings.model,
            client=client,
            prompt_snapshot_path=settings.prompt_snapshot_path,
            node_context_config=settings.node_context,
            casting_retries=settings.casting_retries,
            host=host,
            options=options,
        )
        return cls.from_llm_client(
            ontology,
            llm_client,
            settings,
            embedding_model=embedding_model,
            embedding_config=embedding_config,
        )

    @classmethod
    def from_openai(
        cls,
        ontology: Ontology,
        config: Optional[GraphicalizerConfig] = None,
        *,
        client: Any = None,
        options: Optional[Mapping[str, Any]] = None,
        prompt_template: Optional[GraphicalizerPrompt] = None,
        embedding_model: Any = None,
        embedding_config: Optional[NodeEmbeddingConfig] = None,
    ) -> "LLMGraphicalizer":
        """Build the complete pipeline from one configuration object."""
        settings = config or GraphicalizerConfig()
        if prompt_template is not None:
            template = prompt_template
        elif settings.prompt_template_path is not None:
            template = GraphicalizerPrompt.from_yaml_file(
                settings.prompt_template_path
            )
        else:
            template = GraphicalizerPrompt.default()
        llm_client = OpenAIResponsesClient(
            prompt=template,
            ontology=ontology,
            extraction_density=settings.extraction_density,
            model=settings.model,
            client=client,
            prompt_snapshot_path=settings.prompt_snapshot_path,
            node_context_config=settings.node_context,
            casting_retries=settings.casting_retries,
            options=options,
        )
        return cls(
            llm_client=llm_client,
            ontology=ontology,
            validator=GraphValidator(
                max_density=settings.max_density,
                max_degree_fraction=settings.max_degree_fraction,
                max_hub_nodes=settings.max_hub_nodes,
            ),
            strict_validation=settings.strict_validation,
            disconnected_policy=settings.disconnected_policy,
            node_context_config=settings.node_context,
            context_policy=settings.context_policy,
            embedding_model=embedding_model,
            embedding_config=embedding_config,
        )

    def __init__(
        self,
        llm_client: StructuredLLMClient,
        ontology: Ontology,
        *,
        graph_builder: Optional[NetworkXGraphBuilder] = None,
        validator: Optional[GraphValidator] = None,
        strict_validation: bool = False,
        disconnected_policy: str = "largest_component",
        node_context_config: Optional[NodeContextConfig] = None,
        context_policy: str = "per_node",
        embedding_model: Any = None,
        embedding_config: Optional[NodeEmbeddingConfig] = None,
    ) -> None:
        valid_policies = {"largest_component", "raise", "preserve"}
        if disconnected_policy not in valid_policies:
            choices = ", ".join(sorted(valid_policies))
            raise ValueError(f"disconnected_policy must be one of: {choices}")
        if context_policy == "per_node":
            context_policy = "all_nodes"
        elif context_policy == "none":
            context_policy = "disabled"
        if context_policy not in {"all_nodes", "disabled"}:
            raise ValueError("context_policy must be either 'all_nodes' or 'disabled'.")
        if embedding_model is not None and context_policy != "all_nodes":
            raise ValueError(
                "embedding_model requires context_policy='all_nodes' so every "
                "node has a node_context.summary to embed."
            )
        self.llm_client = llm_client
        self.ontology = ontology
        self.graph_builder = graph_builder or NetworkXGraphBuilder()
        self.validator = validator or GraphValidator()
        self.strict_validation = strict_validation
        self.disconnected_policy = disconnected_policy
        self.node_context_config = node_context_config or NodeContextConfig()
        self.context_policy = context_policy
        self.node_embedder = (
            NodeContextEmbedder(embedding_model, embedding_config)
            if embedding_model is not None
            else None
        )

    @staticmethod
    def _validate_node_contexts(
        contexts: Sequence[NodeContext],
        graph: Any,
        config: NodeContextConfig,
    ) -> Tuple[NodeContext, ...]:
        """Require exactly one non-empty context for every final node."""
        expected_ids = tuple(str(node_id) for node_id in graph.nodes())
        expected = set(expected_ids)
        seen: set[str] = set()
        for context in contexts:
            if context.entity_id in seen:
                raise LLMResponseError(
                    f"Node-context pass returned duplicate entity id {context.entity_id!r}."
                )
            if context.entity_id not in expected:
                raise LLMResponseError(
                    f"Node-context pass returned unknown entity id {context.entity_id!r}."
                )
            if not context.summary.strip():
                raise LLMResponseError(
                    f"Node-context pass returned an empty summary for {context.entity_id!r}."
                )
            sentence_count = len(
                [
                    sentence
                    for sentence in re.split(
                        r"(?<=[.!?])\s+",
                        context.summary.strip(),
                    )
                    if sentence
                ]
            )
            if sentence_count > config.max_sentences:
                raise LLMResponseError(
                    f"Node-context summary for {context.entity_id!r} has "
                    f"{sentence_count} sentences; maximum is {config.max_sentences}."
                )
            if not config.include_evidence and context.evidence:
                raise LLMResponseError(
                    f"Node-context pass returned evidence for {context.entity_id!r} "
                    "although evidence output is disabled."
                )
            if len(context.evidence) > config.max_evidence_items:
                raise LLMResponseError(
                    f"Node-context pass returned too many evidence items for "
                    f"{context.entity_id!r}."
                )
            seen.add(context.entity_id)
        missing = [entity_id for entity_id in expected_ids if entity_id not in seen]
        if missing:
            raise LLMResponseError(
                "Node-context pass omitted entity id(s): " + ", ".join(missing)
            )
        by_id = {context.entity_id: context for context in contexts}
        return tuple(by_id[entity_id] for entity_id in expected_ids)

    def _normalize_extraction(
        self,
        extraction: FreeGraphExtraction,
        raw_graph: Any,
        raw_validation: ValidationReport,
    ) -> Tuple[FreeGraphExtraction, Tuple[str, ...], Mapping[str, Any]]:
        """Apply the configured policy to disconnected model output."""
        import networkx as nx

        if raw_graph.is_directed():
            components = list(nx.weakly_connected_components(raw_graph))
        else:
            components = list(nx.connected_components(raw_graph))
        if raw_validation.connected or self.disconnected_policy == "preserve":
            node_ids = tuple(entity.entity_id for entity in extraction.entities)
            relation_ids = tuple(relation.relation_id for relation in extraction.relations)
            return (
                extraction,
                (),
                {
                    "policy": self.disconnected_policy,
                    "component_count": len(components),
                    "retained_entity_ids": node_ids,
                    "discarded_entity_ids": (),
                    "retained_relation_ids": relation_ids,
                    "discarded_relation_ids": (),
                },
            )
        if self.disconnected_policy == "raise":
            raise GraphValidationError(raw_validation)

        if not components:
            return (
                extraction,
                ("could not normalize an empty graph",),
                {
                    "policy": self.disconnected_policy,
                    "component_count": 0,
                    "retained_entity_ids": (),
                    "discarded_entity_ids": (),
                    "retained_relation_ids": (),
                    "discarded_relation_ids": tuple(
                        relation.relation_id for relation in extraction.relations
                    ),
                },
            )

        # Make ties reproducible by choosing the component with the smallest
        # sorted node-id tuple after comparing component size.
        retained_nodes = min(
            components,
            key=lambda component: (
                -len(component),
                tuple(sorted(str(node) for node in component)),
            ),
        )
        retained_ids = set(retained_nodes)
        entities = tuple(
            entity for entity in extraction.entities if entity.entity_id in retained_ids
        )
        relations = tuple(
            relation
            for relation in extraction.relations
            if relation.source_id in retained_ids and relation.target_id in retained_ids
        )
        discarded_entities = len(extraction.entities) - len(entities)
        discarded_relations = len(extraction.relations) - len(relations)
        note = (
            "retained largest weakly connected component with "
            f"{len(entities)} node(s); discarded {discarded_entities} disconnected "
            f"entity/entities and {discarded_relations} relation(s)"
        )
        normalized = FreeGraphExtraction(
            entities=entities,
            relations=relations,
            notes=extraction.notes,
        )
        retained_relation_ids = tuple(relation.relation_id for relation in relations)
        discarded_relation_ids = tuple(
            relation.relation_id
            for relation in extraction.relations
            if relation.relation_id not in set(retained_relation_ids)
        )
        report = {
            "policy": self.disconnected_policy,
            "component_count": len(components),
            "retained_entity_ids": tuple(entity.entity_id for entity in entities),
            "discarded_entity_ids": tuple(
                entity.entity_id
                for entity in extraction.entities
                if entity.entity_id not in retained_ids
            ),
            "retained_relation_ids": retained_relation_ids,
            "discarded_relation_ids": discarded_relation_ids,
        }
        return normalized, (note,), report

    def graphicalize(self, text: str) -> GraphicalizationResult:
        if not text.strip():
            raise ValueError("Text must not be empty.")

        raw_extraction = self.llm_client.extract_free_graph(text)
        extraction_issues = validate_free_extraction(raw_extraction)
        if extraction_issues:
            raise LLMResponseError(
                "Free extraction validation failed: " + "; ".join(extraction_issues)
            )
        raw_graph = self.graph_builder.build(raw_extraction)
        raw_validation = self.validator.validate(raw_graph)
        normalized_extraction, normalization_notes, normalization_report = (
            self._normalize_extraction(
                raw_extraction,
                raw_graph,
                raw_validation,
            )
        )
        normalized_graph = self.graph_builder.build(normalized_extraction)
        normalized_validation = self.validator.validate(normalized_graph)
        if self.strict_validation and not normalized_validation.valid:
            raise GraphValidationError(normalized_validation)

        ontology_casting = self.llm_client.cast_to_ontology(
            text,
            normalized_extraction,
            self.ontology,
        )
        casting_issues = validate_ontology_casting(
            normalized_extraction,
            ontology_casting,
            self.ontology,
            require_types=True,
        )
        if casting_issues:
            raise CastingValidationError(casting_issues)

        graph = self.graph_builder.build(normalized_extraction, ontology_casting)
        validate_ontology_labeled_graph(graph)
        final_validation = self.validator.validate(graph)
        if self.strict_validation and not final_validation.valid:
            raise GraphValidationError(final_validation)
        if self.context_policy == "all_nodes":
            node_contexts = self._validate_node_contexts(
                self.llm_client.enrich_node_contexts(
                    text,
                    normalized_extraction,
                    ontology_casting,
                    self.ontology,
                    self.node_context_config,
                ),
                graph,
                self.node_context_config,
            )
            for context in node_contexts:
                graph.nodes[context.entity_id].update(
                    {
                        "node_context": context.summary,
                        "node_context_evidence": list(context.evidence),
                        "node_context_uncertainty": context.uncertainty,
                    }
                )
        else:
            node_contexts = ()

        if self.node_embedder is not None:
            self.node_embedder.enrich_graph(graph, node_contexts)

        prompt = getattr(self.llm_client, "prompt", None)
        run_metadata = {
            "provider": getattr(self.llm_client, "provider", None),
            "model": getattr(self.llm_client, "model", None),
            "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "prompt_sha256": (
                hashlib.sha256(
                    (
                        prompt.free_extraction_system
                        + prompt.ontology_casting_system
                        + prompt.node_context_system
                    ).encode("utf-8")
                ).hexdigest()
                if prompt is not None
                else None
            ),
            "node_context_embeddings": (
                graph.graph.get("node_context_embeddings")
                if self.node_embedder is not None
                else None
            ),
        }
        graph.graph["run_metadata"] = dict(run_metadata)
        return GraphicalizationResult(
            raw_extraction=raw_extraction,
            normalized_extraction=normalized_extraction,
            raw_graph=raw_graph,
            normalized_graph=normalized_graph,
            raw_validation=raw_validation,
            normalized_validation=normalized_validation,
            ontology_casting=ontology_casting,
            graph=graph,
            final_validation=final_validation,
            normalization_notes=normalization_notes,
            node_contexts=node_contexts,
            normalization_report=normalization_report,
            run_metadata=run_metadata,
        )

    def run(self, text: str) -> GraphicalizationResult:
        """Preferred concise alias for graphicalize."""
        return self.graphicalize(text)


FREE_EXTRACTION_SYSTEM_PROMPT = """
You are a domain-independent scientific and technical information-extraction
component. Read the supplied text and identify only the most important
entities and relations needed to explain the process, system, or phenomenon
described.

Pass 1 is deliberately ontology-light. Invent concise, domain-specific entity
and relation types when useful; do not force mentions into a predefined
ontology. Use the runtime density target to choose a sufficiently detailed
graph. Aim for the requested entity count when the text supports it, while
avoiding duplicates and unsupported claims. Select entity classes and relation
types that explain the supplied text, regardless of domain. Every relation must
connect two extracted entity ids and be supported by an evidence phrase from
the text. Do not create unsupported causal claims.

The graph should be connected when the text supports a connected interpretation
and should not contain isolated nodes. Select only entities that participate in
at least one supported relation; omit peripheral mentions that cannot be linked
to the central process. Do not invent a relation merely to connect components.
Avoid generic hubs such as `study`, `patient`, or `disease` unless that node has
a specific scientific role. Use stable ids e1, e2, ... and r1, r2, ... .
""".strip()


ONTOLOGY_CASTING_SYSTEM_PROMPT = """
You are an ontology-mapping component for a typed knowledge graph.
Given the source text, a free-form extraction, and an ontology, assign each
extracted entity exactly one ontology entity type when justified. Assign each
extracted relation exactly one ontology relation type when justified. Use the
exact dictionary keys under entity_types and relation_types as ontology_type
values; never use a label, synonym, inverse value, description, or free-form
relation name. If no specific relation is justified, use `related_to` rather
than inventing a relation type. Preserve all entity ids, relation ids, and endpoints. Every
returned graph object must have one exact ontology type because untyped objects
cannot enter the final graph. Do not add entities or relations and do not alter
the graph topology. Keep canonical names close to the wording supported by the
text, and place specific details in attributes.
""".strip()


NODE_CONTEXT_SYSTEM_PROMPT = """
You are a scientific context-annotation component for a typed knowledge graph.
For every supplied graph node, write a concise explanation grounded only in the
source abstract. Explain what the entity is in this abstract and how its
incoming and outgoing typed relations connect it to the described process.
Use the ontology type and raw mention as anchors, but preserve the meaning
supported by the source text.

Return exactly one context object for every supplied node id. Do not add,
remove, rename, or infer graph entities or relations. Do not introduce facts
that are absent from the abstract. Keep the summary within the configured
sentence limit. Provide short evidence excerpts when requested and state
uncertainty instead of guessing when the abstract is ambiguous.
""".strip()


def load_text(path: str | Path) -> str:
    """Load a UTF-8 text document for graphicalization."""
    return Path(path).read_text(encoding="utf-8")


def load_ontology(path: str | Path) -> Ontology:
    """Load an ontology YAML file."""
    return Ontology.from_yaml(path)


Graphicalizer = LLMGraphicalizer


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_MODEL",
    "FreeEntity",
    "FreeRelation",
    "FreeGraphExtraction",
    "Ontology",
    "OntologyEntity",
    "OntologyRelation",
    "OntologyCasting",
    "NodeContext",
    "NodeContextConfig",
    "GraphicalizerConfig",
    "ValidationReport",
    "GraphicalizationResult",
    "StructuredLLMClient",
    "NetworkXGraphBuilder",
    "GraphvizError",
    "graph_to_dot",
    "write_dot",
    "render_graph",
    "validate_ontology_labeled_graph",
    "GraphValidator",
    "validate_free_extraction",
    "validate_ontology_casting",
    "GraphicalizerPrompt",
    "ExtractionDensityConfig",
    "OpenAIResponsesClient",
    "OllamaResponsesAdapter",
    "OllamaStructuredClient",
    "LLMGraphicalizer",
    "Graphicalizer",
    "GraphicalizerError",
    "MissingDependencyError",
    "LLMResponseError",
    "GraphValidationError",
    "CastingValidationError",
    "load_text",
    "load_ontology",
]


class OllamaResponsesAdapter:
    """Expose Ollama chat structured outputs through the internal response contract."""

    def __init__(
        self,
        client: Any,
        *,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.client = client
        self.options = dict(options or {})
        self.responses = self

    def parse(
        self,
        *,
        model: str,
        input: Sequence[Mapping[str, Any]],
        text_format: Any,
    ) -> Any:
        messages = [
            {
                "role": str(message["role"]),
                "content": str(message.get("content", "")),
            }
            for message in input
        ]
        request: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "format": text_format.model_json_schema(),
            "stream": False,
        }
        if self.options:
            request["options"] = self.options
        response = self.client.chat(**request)
        message = getattr(response, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content is None and isinstance(response, Mapping):
            response_message = response.get("message", {})
            content = response_message.get("content")
        if not content:
            raise LLMResponseError("Ollama returned no structured response content.")
        try:
            parsed = text_format.model_validate_json(content)
        except Exception as exc:
            raise LLMResponseError(
                "Ollama returned content that does not match the requested schema."
            ) from exc
        return SimpleNamespace(output_parsed=parsed)


class OllamaStructuredClient(OpenAIResponsesClient):
    """Structured-output LLM client backed by a local Ollama server.

    Ollama's chat API receives the Pydantic JSON schema through its format
    parameter. The rest of the extraction, casting, validation, and context
    workflow is shared with the provider-independent client contract.
    """

    provider = "ollama"

    def __init__(
        self,
        prompt: GraphicalizerPrompt,
        ontology: Ontology,
        extraction_density: Optional[ExtractionDensityConfig] = None,
        model: str = DEFAULT_OLLAMA_MODEL,
        client: Any = None,
        prompt_snapshot_path: Optional[str | Path] = None,
        node_context_config: Optional[NodeContextConfig] = None,
        casting_retries: int = 2,
        host: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if client is None:
            try:
                from ollama import Client
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise MissingDependencyError(
                    "Install the Ollama extra with: python -m pip install -e '.[ollama]'"
                ) from exc
            client = Client(host=host) if host else Client()
        if isinstance(client, OllamaResponsesAdapter):
            adapter = client
            raw_client = client.client
        else:
            adapter = OllamaResponsesAdapter(client, options=options)
            raw_client = client
        super().__init__(
            prompt=prompt,
            ontology=ontology,
            extraction_density=extraction_density,
            model=model,
            client=adapter,
            prompt_snapshot_path=prompt_snapshot_path,
            node_context_config=node_context_config,
            casting_retries=casting_retries,
        )
        self.ollama_client = raw_client
