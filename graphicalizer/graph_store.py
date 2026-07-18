"""Persistence for individual NetworkX graphicalization results."""

from __future__ import annotations

from pathlib import Path
import pickle
import re
from typing import Any
from uuid import uuid4


class NetworkXGraphStore:
    """Save and load one NetworkX graph per serialized ``.gpickle`` file.

    Pickle is used because the final graph is a ``MultiDiGraph`` whose node
    attributes include vectors, mappings, and provenance lists.  Only load
    files from trusted locations because pickle is not a safe interchange
    format for untrusted input.
    """

    suffix = ".gpickle"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_graph_id(graph_id: str) -> str:
        value = graph_id.strip()
        if not value:
            raise ValueError("graph_id must not be empty.")
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
        return value.strip("._") or "graph"

    def save(self, graph: Any, graph_id: str | None = None) -> Path:
        """Serialize ``graph`` and return its path."""
        if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
            raise TypeError("graph must be a NetworkX-compatible graph.")
        identifier = self._safe_graph_id(graph_id or uuid4().hex)
        path = self.directory / f"{identifier}{self.suffix}"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        return path

    def load(self, graph_id_or_path: str | Path) -> Any:
        """Load a graph by id or by an explicit serialized path."""
        candidate = Path(graph_id_or_path)
        if candidate.parent == Path("."):
            identifier = (
                candidate.stem
                if candidate.suffix == self.suffix
                else candidate.name
            )
            candidate = self.directory / f"{self._safe_graph_id(identifier)}{self.suffix}"
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        with candidate.open("rb") as stream:
            graph = pickle.load(stream)
        if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
            raise TypeError(
                f"Serialized file does not contain a NetworkX graph: {candidate}"
            )
        return graph

    def list(self) -> tuple[Path, ...]:
        """Return serialized graph files in deterministic order."""
        return tuple(sorted(self.directory.glob(f"*{self.suffix}")))


__all__ = ["NetworkXGraphStore"]
