"""Batch processing of text files or PubMed corpus rows into graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Tuple

import pandas as pd

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


def _load_manifest_records(path: Path, key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    """Load prior records keyed by their stable source identity."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    records = payload.get("records", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return {}
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        key = tuple(str(record.get(field, "")) for field in key_fields)
        if all(key):
            result[key] = dict(record)
    return result


def _record_is_reusable(record: Mapping[str, Any], source_sha256: str) -> bool:
    """Return whether a saved graph still represents the current source."""
    if record.get("status") != "saved" or record.get("source_sha256") != source_sha256:
        return False
    graph_path = record.get("graph_path")
    return bool(graph_path) and Path(str(graph_path)).exists()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist a manifest without leaving a partially-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def process_abstract_folder(
    abstract_folder: str | Path,
    graphicalizer: Any,
    graph_store: NetworkXGraphStore,
    *,
    pattern: str = "*.txt",
    graph_id_prefix: str = "abstract",
    manifest_path: str | Path | None = None,
    continue_on_error: bool = True,
    resume: bool = True,
    retry_failed: bool = False,
    verbose: bool = True,
) -> BatchGraphicalizationResult:
    """Graphicalize and persist every matching abstract in a folder.

    Files are processed in deterministic path order. Each successful result is
    saved as an individual NetworkX ``.gpickle`` file. A JSON manifest records
    source paths, graph paths, counts, hashes, and any per-file failures. The
    manifest is updated after each item, so a rerun resumes without duplicate
    LLM calls; set ``retry_failed=True`` to retry unchanged failures.
    """
    folder = Path(abstract_folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    if not pattern.strip():
        raise ValueError("pattern must not be empty.")
    if not graph_id_prefix.strip():
        raise ValueError("graph_id_prefix must not be empty.")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool.")

    paths = tuple(sorted(path for path in folder.glob(pattern) if path.is_file()))
    paths_to_process: Any = paths
    if verbose:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            print("Progress bar unavailable; install tqdm to enable it.")
        else:
            paths_to_process = tqdm(
                paths,
                desc="Processing abstracts",
                unit="abstract",
            )
    resolved_manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else graph_store.directory / "manifest.json"
    )
    prior_records = (
        _load_manifest_records(resolved_manifest_path, ("source_path",))
        if resume
        else {}
    )
    # Keep prior records in the manifest even when a later run uses a narrower
    # source selection; re-expanding the selection can then still resume.
    records_by_key: dict[tuple[str, ...], dict[str, Any]] = dict(prior_records)
    saved_paths: list[Path] = []
    failures: list[Mapping[str, str]] = []

    def checkpoint() -> None:
        current_records = [
            records_by_key[(str(path),)]
            for path in paths
            if (str(path),) in records_by_key
        ]
        all_records = list(records_by_key.values())
        current_failures = [record for record in all_records if record.get("status") == "failed"]
        current_saved_paths = [
            Path(str(record["graph_path"]))
            for record in all_records
            if record.get("status") == "saved" and record.get("graph_path")
        ]
        _write_manifest(
            resolved_manifest_path,
            graph_store,
            folder,
            pattern,
            all_records,
            len(current_saved_paths),
            current_failures,
        )

    try:
        for index, abstract_path in enumerate(paths_to_process):
            graph_id = f"{graph_id_prefix}_{index:04d}_{abstract_path.stem}"
            source_key = (str(abstract_path),)
            prior = prior_records.get(source_key)
            record: dict[str, Any] = {
                "index": index,
                "graph_id": graph_id,
                "source_path": str(abstract_path),
            }
            try:
                text = abstract_path.read_text(encoding="utf-8")
                source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                record["source_sha256"] = source_sha256
                if prior and _record_is_reusable(prior, source_sha256):
                    record = {**prior, "index": index, "source_path": str(abstract_path)}
                    saved_paths.append(Path(str(record["graph_path"])))
                    records_by_key[source_key] = record
                    checkpoint()
                    continue
                if (
                    prior
                    and prior.get("status") == "failed"
                    and not retry_failed
                    and prior.get("source_sha256") == source_sha256
                ):
                    record = {**prior, "index": index, "source_path": str(abstract_path)}
                    failures.append(record)
                    records_by_key[source_key] = record
                    checkpoint()
                    continue
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
                    records_by_key[source_key] = record
                    checkpoint()
                    raise
            records_by_key[source_key] = record
            checkpoint()
    except KeyboardInterrupt:
        checkpoint()
        raise

    current_records = [records_by_key[(str(path),)] for path in paths if (str(path),) in records_by_key]
    saved_paths = [
        Path(str(record["graph_path"]))
        for record in current_records
        if record.get("status") == "saved" and record.get("graph_path")
    ]
    failures = [record for record in current_records if record.get("status") == "failed"]
    return BatchGraphicalizationResult(
        discovered=len(paths),
        processed=len(saved_paths),
        failed=len(failures),
        saved_paths=tuple(saved_paths),
        failures=tuple(failures),
        manifest_path=resolved_manifest_path,
    )


def process_corpus_articles(
    corpus: pd.DataFrame | str | Path,
    graphicalizer: Any,
    graph_store: NetworkXGraphStore,
    *,
    corpus_path: str | Path | None = None,
    graph_id_prefix: str = "pubmed",
    manifest_path: str | Path | None = None,
    continue_on_error: bool = True,
    resume: bool = True,
    retry_failed: bool = False,
    verbose: bool = True,
) -> BatchGraphicalizationResult:
    """Graphicalize rows from the canonical PubMed corpus DataFrame/Parquet.

    The input must contain ``pathogen``, ``pmid``, ``title``, and ``abstract``.
    Rows are processed in their existing order, which should be established by
    :func:`graphicalizer.load_corpus_articles` when deterministic IDs matter.
    The manifest is updated after each item and is used to resume unchanged
    rows without duplicate LLM calls.
    """
    if isinstance(corpus, (str, Path)):
        source = Path(corpus)
        frame = pd.read_parquet(source)
        resolved_corpus_path = str(source)
    else:
        frame = corpus.copy()
        resolved_corpus_path = str(corpus_path) if corpus_path is not None else ""
    required = {"pathogen", "pmid", "title", "abstract"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Corpus is missing required columns: {', '.join(missing)}")
    if not graph_id_prefix.strip():
        raise ValueError("graph_id_prefix must not be empty.")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool.")

    rows = list(frame.itertuples(index=False))
    rows_to_process: Any = rows
    if verbose:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            print("Progress bar unavailable; install tqdm to enable it.")
        else:
            rows_to_process = tqdm(rows, desc="Processing corpus", unit="abstract")

    resolved_manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else graph_store.directory / "manifest.json"
    )
    prior_records = (
        _load_manifest_records(resolved_manifest_path, ("pathogen", "pmid"))
        if resume
        else {}
    )
    # Preserve prior rows outside a temporary filter so they remain resumable
    # if a later run includes them again.
    records_by_key: dict[tuple[str, ...], dict[str, Any]] = dict(prior_records)
    saved_paths: list[Path] = []
    failures: list[Mapping[str, str]] = []

    def checkpoint() -> None:
        current_records = [
            records_by_key[(str(row.pathogen), str(row.pmid))]
            for row in rows
            if (str(row.pathogen), str(row.pmid)) in records_by_key
        ]
        all_records = list(records_by_key.values())
        current_failures = [record for record in all_records if record.get("status") == "failed"]
        current_saved_paths = [
            Path(str(record["graph_path"]))
            for record in all_records
            if record.get("status") == "saved" and record.get("graph_path")
        ]
        _write_corpus_manifest(
            resolved_manifest_path,
            graph_store,
            resolved_corpus_path,
            all_records,
            len(current_saved_paths),
            current_failures,
        )

    try:
        for index, row in enumerate(rows_to_process):
            pathogen = str(row.pathogen)
            pmid = str(row.pmid)
            graph_id = f"{graph_id_prefix}_{index:04d}_{pathogen}_{pmid}"
            source_key = (pathogen, pmid)
            prior = prior_records.get(source_key)
            source_ref = f"{resolved_corpus_path}#{pathogen}/{pmid}" if resolved_corpus_path else f"{pathogen}/{pmid}"
            record: dict[str, Any] = {
                "index": index,
                "graph_id": graph_id,
                "source_path": source_ref,
                "pathogen": pathogen,
                "pmid": pmid,
                "title": str(row.title or ""),
                "publication_date": str(getattr(row, "publication_date", "") or ""),
            }
            try:
                text = str(row.abstract or "")
                if not text.strip():
                    raise ValueError("empty abstract")
                source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                record["source_sha256"] = source_sha256
                if prior and _record_is_reusable(prior, source_sha256):
                    record = {**prior, "index": index, "pathogen": pathogen, "pmid": pmid}
                    saved_paths.append(Path(str(record["graph_path"])))
                    records_by_key[source_key] = record
                    checkpoint()
                    continue
                if (
                    prior
                    and prior.get("status") == "failed"
                    and not retry_failed
                    and prior.get("source_sha256") == source_sha256
                ):
                    record = {**prior, "index": index, "pathogen": pathogen, "pmid": pmid}
                    failures.append(record)
                    records_by_key[source_key] = record
                    checkpoint()
                    continue
                result = graphicalizer.run(text)
                graph = result.graph
                graph.graph.update(
                    {
                        "batch_index": index,
                        "batch_graph_id": graph_id,
                        "corpus_path": resolved_corpus_path,
                        "pathogen": pathogen,
                        "pmid": pmid,
                        "title": str(row.title or ""),
                        "publication_date": str(getattr(row, "publication_date", "") or ""),
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
                    "source_path": source_ref,
                    "graph_id": graph_id,
                    "pathogen": pathogen,
                    "pmid": pmid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                record.update({"status": "failed", **failure})
                if not continue_on_error:
                    records_by_key[source_key] = record
                    checkpoint()
                    raise
            records_by_key[source_key] = record
            checkpoint()
    except KeyboardInterrupt:
        checkpoint()
        raise

    current_records = [
        records_by_key[(str(row.pathogen), str(row.pmid))]
        for row in rows
        if (str(row.pathogen), str(row.pmid)) in records_by_key
    ]
    saved_paths = [
        Path(str(record["graph_path"]))
        for record in current_records
        if record.get("status") == "saved" and record.get("graph_path")
    ]
    failures = [record for record in current_records if record.get("status") == "failed"]
    return BatchGraphicalizationResult(
        discovered=len(rows),
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
    _write_json_atomic(path, payload)
    return path


def _write_corpus_manifest(
    manifest_path: str | Path | None,
    graph_store: NetworkXGraphStore,
    corpus_path: str,
    records: list[dict[str, Any]],
    processed: int,
    failures: list[Mapping[str, str]],
) -> Path:
    path = Path(manifest_path) if manifest_path is not None else graph_store.directory / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_path": corpus_path,
        "graph_folder": str(graph_store.directory),
        "discovered": len(records),
        "processed": processed,
        "failed": len(failures),
        "records": records,
    }
    _write_json_atomic(path, payload)
    return path


__all__ = ["BatchGraphicalizationResult", "process_abstract_folder", "process_corpus_articles"]
