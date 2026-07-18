"""Batch processing of abstract text files into serialized NetworkX graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Tuple

from .graph_store import NetworkXGraphStore


@dataclass(frozen=True)
class BatchGraphicalizationResult:
    """Summary of a folder processing run."""

    discovered: int
    processed: int
    failed: int
    saved_paths: Tuple[Path, ...] = ()
    failures: Tuple[Mapping[str, str], ...] = ()
    manifest_path: Path | None = None

    def to_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["saved_paths"] = [str(path) for path in self.saved_paths]
        result["failures"] = [dict(item) for item in self.failures]
        result["manifest_path"] = (
            str(self.manifest_path) if self.manifest_path is not None else None
        )
        return result


def process_abstract_folder(
    abstract_folder: str | Path,
    graphicalizer: Any,
    graph_store: NetworkXGraphStore,
    *,
    pattern: str = "*.txt",
    graph_id_prefix: str = "abstract",
    manifest_path: str | Path | None = None,
    continue_on_error: bool = True,
) -> BatchGraphicalizationResult:
    """Graphicalize and persist every matching abstract in a folder.

    Files are processed in deterministic path order. Each successful result is
    saved as an individual NetworkX ``.gpickle`` file. A JSON manifest records
    source paths, graph paths, counts, hashes, and any per-file failures.
    """
    folder = Path(abstract_folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    if not pattern.strip():
        raise ValueError("pattern must not be empty.")
    if not graph_id_prefix.strip():
        raise ValueError("graph_id_prefix must not be empty.")

    paths = tuple(sorted(path for path in folder.glob(pattern) if path.is_file()))
    saved_paths: list[Path] = []
    failures: list[Mapping[str, str]] = []
    records: list[dict[str, Any]] = []

    for index, abstract_path in enumerate(paths):
        graph_id = f"{graph_id_prefix}_{index:04d}_{abstract_path.stem}"
        record: dict[str, Any] = {
            "index": index,
            "graph_id": graph_id,
            "source_path": str(abstract_path),
        }
        try:
            text = abstract_path.read_text(encoding="utf-8")
            source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            record["source_sha256"] = source_sha256
            result = graphicalizer.run(text)
            graph = result.graph
            graph.graph.update(
                {
                    "batch_index": index,
                    "batch_graph_id": graph_id,
                    "source_path": str(abstract_path),
                    "source_sha256": source_sha256,
                }
            )
            graph_path = graph_store.save(graph, graph_id)
            saved_paths.append(graph_path)
            record.update(
                {
                    "status": "saved",
                    "graph_path": str(graph_path),
                    "node_count": graph.number_of_nodes(),
                    "edge_count": graph.number_of_edges(),
                }
            )
        except Exception as exc:
            failure = {
                "source_path": str(abstract_path),
                "graph_id": graph_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            record.update({"status": "failed", **failure})
            if not continue_on_error:
                records.append(record)
                _write_manifest(
                    manifest_path,
                    graph_store,
                    folder,
                    pattern,
                    records,
                    len(saved_paths),
                    failures,
                )
                raise
        records.append(record)

    resolved_manifest_path = _write_manifest(
        manifest_path,
        graph_store,
        folder,
        pattern,
        records,
        len(saved_paths),
        failures,
    )
    return BatchGraphicalizationResult(
        discovered=len(paths),
        processed=len(saved_paths),
        failed=len(failures),
        saved_paths=tuple(saved_paths),
        failures=tuple(failures),
        manifest_path=resolved_manifest_path,
    )


def _write_manifest(
    manifest_path: str | Path | None,
    graph_store: NetworkXGraphStore,
    folder: Path,
    pattern: str,
    records: list[dict[str, Any]],
    processed: int,
    failures: list[Mapping[str, str]],
) -> Path:
    path = (
        Path(manifest_path)
        if manifest_path is not None
        else graph_store.directory / "manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "abstract_folder": str(folder),
        "pattern": pattern,
        "graph_folder": str(graph_store.directory),
        "discovered": len(records),
        "processed": processed,
        "failed": len(failures),
        "records": records,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = ["BatchGraphicalizationResult", "process_abstract_folder"]
