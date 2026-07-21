"""Discover zoonotic bacterial species from PubMed titles and abstracts."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from .pubmed import PubMedClient


DEFAULT_ZOONOSIS_QUERY = (
    '("zoonosis"[Title/Abstract] OR "zoonotic"[Title/Abstract] '
    'OR "zoonoses"[Title/Abstract] OR spillover[Title/Abstract] '
    'OR "animal-to-human"[Title/Abstract] OR "animal to human"[Title/Abstract])'
)

EXTRACTION_COLUMNS = [
    "stage",
    "pmid",
    "title",
    "publication_date",
    "associated_with_zoonosis",
    "bacterial_species",
    "evidence_phrases",
    "rationale",
    "confidence",
    "review_required",
    "classification_status",
    "last_error",
    "extraction_hash",
    "extracted_at",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(payload: Any) -> str:
    value = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{id(frame)}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{id(payload)}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _merge_checkpoint(path: Path, updates: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_parquet(path)
        merged = pd.concat([existing, updates], ignore_index=True)
    else:
        merged = updates.copy()
    if not merged.empty:
        merged = merged.drop_duplicates(subset=list(keys), keep="last").reset_index(drop=True)
    _atomic_write_parquet(merged, path)
    return merged


def collect_zoonosis_articles(
    client: PubMedClient,
    output_dir: str | Path,
    *,
    query: str = DEFAULT_ZOONOSIS_QUERY,
    max_results: int = 5000,
    search_page_size: int = 1000,
    fetch_batch_size: int = 200,
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Search PubMed and persist a resumable title/abstract article table."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    query_hash = _stable_hash({"query": query, "max_results": max_results})
    search_path = output / "zoonosis_pubmed_search.json"
    articles_path = output / "zoonosis_pubmed_articles.parquet"

    search = _read_json(search_path) if resume else {}
    search_matches = search.get("query_hash") == query_hash
    if not search_matches:
        total, pmids = client.search_all(
            query,
            max_results=max_results,
            page_size=search_page_size,
            sort="relevance",
        )
        search = {
            "query": query,
            "query_hash": query_hash,
            "total_matches": total,
            "retrieved_pmids": len(pmids),
            "pmids": [str(pmid) for pmid in pmids],
            "updated_at": _now(),
        }
        _write_json(search_path, search)
    pmids = [str(pmid) for pmid in search.get("pmids", [])]
    articles = (
        pd.read_parquet(articles_path)
        if resume and search_matches and articles_path.exists()
        else pd.DataFrame()
    )
    if articles.empty:
        existing_pmids = set()
    else:
        completed_statuses = {"ok", "no_abstract"}
        existing_pmids = set(
            articles.loc[
                articles.get("fetch_status", pd.Series(index=articles.index, dtype=str)).isin(completed_statuses),
                "pmid",
            ].astype(str)
        )
    pending = [pmid for pmid in pmids if pmid not in existing_pmids]

    for start in range(0, len(pending), fetch_batch_size):
        batch = pending[start : start + fetch_batch_size]
        try:
            fetched = {article.pmid: article for article in client.fetch_many(batch, batch_size=fetch_batch_size)}
            fetch_error = ""
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            fetched = {}
            fetch_error = f"{type(exc).__name__}: {exc}"
        rows = []
        for pmid in batch:
            article = fetched.get(pmid)
            if article is None:
                rows.append({
                    "pmid": pmid,
                    "title": "",
                    "abstract": "",
                    "publication_date": "",
                    "fetch_status": "network_error" if fetch_error else "not_returned",
                    "fetch_error": fetch_error,
                    "fetched_at": _now(),
                })
            else:
                rows.append({
                    **asdict(article),
                    "fetch_status": "ok" if article.abstract.strip() else "no_abstract",
                    "fetch_error": "",
                    "fetched_at": _now(),
                })
        articles = _merge_checkpoint(articles_path, pd.DataFrame(rows), ["pmid"])
        if verbose:
            print(f"Fetched {min(start + len(batch), len(pending))}/{len(pending)} PubMed records")

    if articles.empty:
        return pd.DataFrame(columns=["pmid", "title", "abstract", "publication_date", "fetch_status"])
    return articles.sort_values("pmid").reset_index(drop=True)


def _json_object(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM response did not contain a JSON object.")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON was not an object.")
    return value


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    seen: set[str] = set()
    result = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item)).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _normalize_extraction(raw: Mapping[str, Any]) -> dict[str, Any]:
    names = _string_list(raw.get("bacterial_species", raw.get("species_names", [])))
    phrases = _string_list(raw.get("evidence_phrases", []))
    confidence = raw.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    associated = bool(raw.get("associated_with_zoonosis", bool(names)))
    if not names:
        associated = False
    return {
        "associated_with_zoonosis": associated,
        "bacterial_species": json.dumps(names, ensure_ascii=False),
        "evidence_phrases": json.dumps(phrases, ensure_ascii=False),
        "rationale": str(raw.get("rationale", "")).strip(),
        "confidence": confidence,
        "review_required": bool(raw.get("review_required", False)),
        "classification_status": "classified",
        "last_error": "",
    }


def extract_bacterial_species(
    llm: Any,
    *,
    title: str,
    abstract: str = "",
    stage: str = "title",
    max_tokens: int = 512,
    retries: int = 3,
) -> dict[str, Any]:
    """Ask an LLM for bacterial species explicitly associated with zoonosis."""
    if stage not in {"title", "abstract"}:
        raise ValueError("stage must be 'title' or 'abstract'.")
    source = f"TITLE:\n{title.strip()}"
    if stage == "abstract":
        source += f"\n\nABSTRACT:\n{abstract.strip()}"
    prompt = f"""Review this PubMed {stage} for zoonotic bacterial evidence.

Extract only bacterial species explicitly associated with zoonosis, animal-to-human
transmission, spillover, or a zoonotic infection in the supplied text. Do not return
viruses, parasites, fungi, hosts, diseases, genera without a species, or names that
appear only as background/comparison organisms. Return an empty list when no bacterial
species is explicitly associated.

Return JSON with exactly these keys:
associated_with_zoonosis (boolean), bacterial_species (array of scientific species names),
evidence_phrases (array of short exact phrases), rationale (short string),
confidence (number from 0 to 1), review_required (boolean).

{source}
"""
    last_error = ""
    for attempt in range(max(1, retries)):
        try:
            raw = _json_object(
                llm.complete(
                    prompt,
                    system_prompt="You extract conservative, evidence-grounded bacterial species names from biomedical text.",
                    max_tokens=max_tokens,
                    temperature=0,
                )
            )
            return _normalize_extraction(raw)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 >= max(1, retries):
                break
    return {
        "associated_with_zoonosis": False,
        "bacterial_species": "[]",
        "evidence_phrases": "[]",
        "rationale": f"Extraction failed: {last_error}",
        "confidence": None,
        "review_required": True,
        "classification_status": "failed",
        "last_error": last_error,
    }


def run_species_extraction_stage(
    articles: pd.DataFrame,
    llm: Any,
    output_path: str | Path,
    *,
    stage: str,
    model: str = "unknown",
    max_tokens: int = 512,
    retries: int = 3,
    save_every: int = 10,
    max_llm_calls: int | None = None,
    resume: bool = True,
    retry_failed: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run one title/abstract extraction stage with an interrupt-safe checkpoint."""
    if stage not in {"title", "abstract"}:
        raise ValueError("stage must be 'title' or 'abstract'.")
    output = Path(output_path)
    prompt_hash = _stable_hash({"stage": stage, "model": model, "max_tokens": max_tokens})
    existing = pd.read_parquet(output) if resume and output.exists() else pd.DataFrame(columns=EXTRACTION_COLUMNS)
    if existing.empty:
        current = existing
    else:
        hash_mask = existing.get(
            "extraction_hash", pd.Series(index=existing.index, dtype=str)
        ).astype(str).eq(prompt_hash)
        stage_mask = existing.get("stage", pd.Series(index=existing.index, dtype=str)).eq(stage)
        current = existing[hash_mask & stage_mask].copy()
    done = set()
    if not current.empty:
        completed = current["classification_status"].eq("classified")
        if not retry_failed:
            completed = completed | current["classification_status"].eq("failed")
        done = set(current.loc[completed, "pmid"].astype(str))
    pending = articles[~articles["pmid"].astype(str).isin(done)].copy()
    updates: list[dict[str, Any]] = []
    calls = 0
    try:
        for row in pending.itertuples(index=False):
            if max_llm_calls is not None and calls >= max_llm_calls:
                break
            result = extract_bacterial_species(
                llm,
                title=str(getattr(row, "title", "") or ""),
                abstract=str(getattr(row, "abstract", "") or ""),
                stage=stage,
                max_tokens=max_tokens,
                retries=retries,
            )
            calls += 1
            updates.append({
                "stage": stage,
                "pmid": str(row.pmid),
                "title": str(getattr(row, "title", "") or ""),
                "publication_date": str(getattr(row, "publication_date", "") or ""),
                **result,
                "extraction_hash": prompt_hash,
                "extracted_at": _now(),
            })
            if verbose:
                print(f"{stage.title()} extraction {calls}/{len(pending)}: PMID {row.pmid}")
            if len(updates) >= max(1, save_every):
                _merge_checkpoint(output, pd.DataFrame(updates, columns=EXTRACTION_COLUMNS), ["pmid"])
                updates.clear()
    finally:
        if updates:
            _merge_checkpoint(output, pd.DataFrame(updates, columns=EXTRACTION_COLUMNS), ["pmid"])
    if not output.exists():
        return pd.DataFrame(columns=EXTRACTION_COLUMNS)
    return pd.read_parquet(output)


def _decode_list(value: Any) -> list[str]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return _string_list(decoded)


def build_bacterial_species_summary(
    title_extractions: pd.DataFrame,
    abstract_extractions: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build and optionally persist one row per bacterial species name."""
    evidence_rows = []
    for frame in (title_extractions, abstract_extractions):
        if frame is None or frame.empty:
            continue
        for row in frame.itertuples(index=False):
            names = _decode_list(getattr(row, "bacterial_species", "[]"))
            if not bool(getattr(row, "associated_with_zoonosis", False)):
                continue
            for name in names:
                evidence_rows.append({
                    "bacterial_species": name,
                    "species_key": re.sub(r"\s+", " ", name).strip().casefold(),
                    "stage": str(getattr(row, "stage", "")),
                    "pmid": str(getattr(row, "pmid", "")),
                    "title": str(getattr(row, "title", "")),
                })
    columns = [
        "bacterial_species", "evidence_levels", "title_evidence_count",
        "abstract_evidence_count", "evidence_count", "pmids", "example_titles",
        "updated_at",
    ]
    if not evidence_rows:
        summary = pd.DataFrame(columns=columns)
    else:
        evidence = pd.DataFrame(evidence_rows)
        rows = []
        for key, group in evidence.groupby("species_key", sort=True):
            display_name = sorted(group["bacterial_species"].unique(), key=str.casefold)[0]
            title_rows = group[group["stage"].eq("title")]
            abstract_rows = group[group["stage"].eq("abstract")]
            rows.append({
                "bacterial_species": display_name,
                "evidence_levels": json.dumps(sorted(group["stage"].unique())),
                "title_evidence_count": int(len(title_rows)),
                "abstract_evidence_count": int(len(abstract_rows)),
                "evidence_count": int(len(group)),
                "pmids": json.dumps(sorted(group["pmid"].unique())),
                "example_titles": json.dumps(sorted(group["title"].unique())[:5], ensure_ascii=False),
                "updated_at": _now(),
            })
        summary = pd.DataFrame(rows, columns=columns).sort_values("bacterial_species").reset_index(drop=True)
    if output_path is not None:
        _atomic_write_parquet(summary, Path(output_path))
    return summary


__all__ = [
    "DEFAULT_ZOONOSIS_QUERY",
    "build_bacterial_species_summary",
    "collect_zoonosis_articles",
    "extract_bacterial_species",
    "run_species_extraction_stage",
]
