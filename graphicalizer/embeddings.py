"""Embedding enrichment for node-context summaries.

Embedding models are deliberately injected by the caller.  This keeps model
loading, credentials, and device selection outside the graphicalizer while
supporting common model interfaces such as SentenceTransformers and simple
callables used in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Protocol, Sequence


class EmbeddingModel(Protocol):
    """Minimal protocol for an injected embedding model."""

    def encode(self, texts: Sequence[str]) -> Any:
        """Return one numeric vector per input text."""


@dataclass(frozen=True)
class NodeEmbeddingConfig:
    """Controls how node-context vectors are stored."""

    normalize: bool = True
    model_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be a bool.")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def _to_list(value: Any) -> Any:
    """Convert array-like model output to ordinary Python containers."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _coerce_vectors(raw: Any, expected_count: int) -> list[list[float]]:
    """Validate and normalize the outer shape of model output."""
    if isinstance(raw, Mapping):
        if "data" in raw:
            raw = raw["data"]
        elif "embeddings" in raw:
            raw = raw["embeddings"]
    raw = _to_list(raw)
    if expected_count == 1 and raw and isinstance(raw[0], (int, float)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or len(raw) != expected_count:
        raise ValueError(
            "Embedding model must return one vector per node-context summary; "
            f"expected {expected_count}, received "
            f"{len(raw) if isinstance(raw, (list, tuple)) else type(raw).__name__}."
        )

    vectors: list[list[float]] = []
    dimension: int | None = None
    for index, vector in enumerate(raw):
        if isinstance(vector, Mapping) and "embedding" in vector:
            vector = vector["embedding"]
        vector = _to_list(vector)
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError(f"Embedding vector {index} is empty or not a sequence.")
        values = [float(item) for item in vector]
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"Embedding vector {index} contains a non-finite value.")
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise ValueError(
                "Embedding model returned vectors with inconsistent dimensions."
            )
        vectors.append(values)
    return vectors


def _model_identifier(model: Any, configured: str) -> str:
    if configured.strip():
        return configured.strip()
    for attribute in ("model_name", "model", "name"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return type(model).__name__


class NodeContextEmbedder:
    """Vectorize each ``NodeContext.summary`` and attach it to graph nodes."""

    def __init__(
        self,
        embedding_model: Any,
        config: NodeEmbeddingConfig | None = None,
    ) -> None:
        if embedding_model is None:
            raise ValueError("embedding_model must be provided explicitly.")
        self.embedding_model = embedding_model
        self.config = config or NodeEmbeddingConfig()
        self.model_id = _model_identifier(embedding_model, self.config.model_id)

    def _encode(self, summaries: Sequence[str]) -> list[list[float]]:
        model = self.embedding_model
        if hasattr(model, "encode"):
            raw = model.encode(list(summaries))
        elif hasattr(model, "embed_documents"):
            raw = model.embed_documents(list(summaries))
        elif hasattr(model, "embed"):
            raw = model.embed(list(summaries))
        elif callable(model):
            raw = model(list(summaries))
        else:
            raise TypeError(
                "embedding_model must expose encode, embed_documents, embed, or be callable."
            )
        vectors = _coerce_vectors(raw, len(summaries))
        if not self.config.normalize:
            return vectors
        normalized: list[list[float]] = []
        for vector in vectors:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                normalized.append(vector)
            else:
                normalized.append([value / norm for value in vector])
        return normalized

    def enrich_graph(self, graph: Any, node_contexts: Sequence[Any]) -> Any:
        """Attach vectors to ``graph`` and return the same graph object."""
        contexts = {context.entity_id: context for context in node_contexts}
        node_ids = list(graph.nodes())
        missing = [node_id for node_id in node_ids if node_id not in contexts]
        if missing:
            raise ValueError(
                "Cannot embed graph nodes without node contexts: "
                + ", ".join(str(node_id) for node_id in missing)
            )
        summaries = [str(contexts[node_id].summary) for node_id in node_ids]
        vectors = self._encode(summaries)
        dimensions = len(vectors[0]) if vectors else 0
        for node_id, context, vector in zip(
            node_ids,
            (contexts[node_id] for node_id in node_ids),
            vectors,
        ):
            graph.nodes[node_id].update(
                {
                    "node_context_embedding": vector,
                    "node_context_embedding_model": self.model_id,
                    "node_context_embedding_dimension": dimensions,
                }
            )
        graph.graph["node_context_embeddings"] = {
            "field": "node_context_embedding",
            "model": self.model_id,
            "dimension": dimensions,
            "normalized": self.config.normalize,
        }
        return graph


__all__ = ["EmbeddingModel", "NodeEmbeddingConfig", "NodeContextEmbedder"]
