"""Resumable PubMed corpus collection and pathogen-specific screening."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Protocol, Sequence

import fcntl
import pandas as pd

from .pubmed import PubMedArticle, PubMedClient


DEFAULT_ANIMAL_TERMS = (
    "animal",
    "host",
    "wildlife",
    "livestock",
    "veterinary",
    "infection",
)
DEFAULT_ZOONOSIS_TERMS = (
    "zoonosis",
    "zoonotic",
    "spillover",
    "animal-to-human",
    "human infection",
    "human disease",
    "transmission",
)
SEARCH_TYPES = ("animal", "zoonosis")
CATEGORY_1 = 1
CATEGORY_2 = 2
CATEGORY_3 = 3


@dataclass(frozen=True)
class PathogenSpec:
    """Canonical pathogen name and PubMed aliases."""

    name: str
    aliases: tuple[str, ...] = ()

    def all_terms(self) -> tuple[str, ...]:
        values = [self.name, *self.aliases]
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            clean = str(value).strip()
            key = clean.casefold()
            if clean and key not in seen:
                seen.add(key)
                result.append(clean)
        if not result:
            raise ValueError("A pathogen must have a non-empty name.")
        return tuple(result)


@dataclass(frozen=True)
class SearchBundle:
    """Named PubMed evidence-term bundle loaded from a YAML asset."""

    search_type: str
    terms: tuple[str, ...]
    label: str = ""
    description: str = ""
    version: str = "1"

    def __post_init__(self) -> None:
        search_type = self.search_type.strip()
        if not search_type:
            raise ValueError("search_type must not be empty.")
        normalized = tuple(str(term).strip() for term in self.terms if str(term).strip())
        if not normalized:
            raise ValueError(f"Search bundle {search_type!r} must contain terms.")
        seen: set[str] = set()
        deduplicated: list[str] = []
        for term in normalized:
            key = term.casefold()
            if key not in seen:
                seen.add(key)
                deduplicated.append(term)
        object.__setattr__(self, "search_type", search_type)
        object.__setattr__(self, "terms", tuple(deduplicated))


def load_search_bundle(
    path: str | Path,
    *,
    expected_search_type: str | None = None,
) -> SearchBundle:
    """Load and validate one generic search-term bundle from YAML."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - base dependency
        raise RuntimeError("Install PyYAML to load search bundles.") from exc

    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Search bundle YAML must contain a mapping: {source}")
    terms = data.get("terms")
    if isinstance(terms, str) or not isinstance(terms, Sequence):
        raise ValueError(f"Search bundle terms must be a list: {source}")
    bundle = SearchBundle(
        search_type=str(data.get("search_type", source.stem)),
        terms=tuple(str(term) for term in terms),
        label=str(data.get("label", "")),
        description=str(data.get("description", "")),
        version=str(data.get("version", "1")),
    )
    if expected_search_type is not None and bundle.search_type != expected_search_type:
        raise ValueError(
            f"Expected search bundle type {expected_search_type!r}, "
            f"got {bundle.search_type!r} from {source}."
        )
    return bundle


def normalize_search_bundles(
    search_bundles: Mapping[str, SearchBundle | Sequence[str]] | Sequence[SearchBundle] | None = None,
    *,
    animal_terms: Sequence[str] = DEFAULT_ANIMAL_TERMS,
    zoonosis_terms: Sequence[str] = DEFAULT_ZOONOSIS_TERMS,
) -> tuple[SearchBundle, ...]:
    """Normalize generic search bundles while retaining legacy term arguments."""
    if search_bundles is None:
        candidates: Sequence[SearchBundle] = (
            SearchBundle("animal", tuple(animal_terms)),
            SearchBundle("zoonosis", tuple(zoonosis_terms)),
        )
    elif isinstance(search_bundles, Mapping):
        mapped: list[SearchBundle] = []
        for search_type, value in search_bundles.items():
            if isinstance(value, SearchBundle):
                if value.search_type != str(search_type):
                    raise ValueError(
                        f"Search bundle key {search_type!r} does not match "
                        f"bundle type {value.search_type!r}."
                    )
                mapped.append(value)
            else:
                terms = (value,) if isinstance(value, str) else tuple(value)
                mapped.append(SearchBundle(str(search_type), terms))
        candidates = tuple(mapped)
    else:
        candidates = tuple(search_bundles)

    if not candidates:
        raise ValueError("search_bundles must contain at least one bundle.")
    seen: set[str] = set()
    for bundle in candidates:
        if not isinstance(bundle, SearchBundle):
            raise TypeError("search_bundles sequences must contain SearchBundle values.")
        key = bundle.search_type.casefold()
        if key in seen:
            raise ValueError(f"Search bundle types must be unique: {bundle.search_type!r}")
        seen.add(key)
    return tuple(candidates)


@dataclass(frozen=True)
class CorpusCollectionResult:
    """Outputs and summary metadata from a corpus collection run."""

    corpus: pd.DataFrame
    search_runs: pd.DataFrame
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class ScreeningRunResult:
    """Outputs and summary metadata from an LLM screening run."""

    screening: pd.DataFrame
    failures: pd.DataFrame
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class RefinementRunResult:
    """Outputs from the first-stage LLM relevance/refinement run."""

    refined_corpus: pd.DataFrame
    refinement: pd.DataFrame
    failures: pd.DataFrame
    manifest: Mapping[str, Any]


class TerminalLLMError(RuntimeError):
    """Raised when continuing an LLM run is not useful."""


class ChatCompleter(Protocol):
    """Minimal AMR-compatible chat-completion interface."""

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Return the assistant text for one completion."""


class OpenAIChatCompleter:
    """OpenAI Chat Completions adapter matching :class:`ChatCompleter`."""

    def __init__(self, model: str, *, client: Any = None, options: Mapping[str, Any] | None = None):
        if not model.strip():
            raise ValueError("model must not be empty.")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Install the openai extra to use OpenAIChatCompleter.") from exc
            client = OpenAI()
        self.model = model
        self.client = client
        self.options = dict(options or {})

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.options,
            **kwargs,
        }
        response = self.client.chat.completions.create(**request)
        choices = getattr(response, "choices", None) or []
        if not choices or getattr(choices[0], "message", None) is None:
            raise RuntimeError("OpenAI returned no chat completion choices.")
        content = getattr(choices[0].message, "content", None)
        if not content:
            raise RuntimeError("OpenAI returned an empty chat completion.")
        return str(content)


@contextmanager
def _checkpoint_lock(path: Path):
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _where(frame: pd.DataFrame, column: str, value: Any) -> pd.Series:
    """Return an index-aligned equality mask even for legacy checkpoints."""
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).eq(str(value))


def _merge_checkpoint(path: Path, updates: pd.DataFrame, key_columns: Sequence[str]) -> pd.DataFrame:
    with _checkpoint_lock(path):
        current = _read_parquet(path)
        merged = pd.concat([current, updates], ignore_index=True) if not updates.empty else current
        if not merged.empty:
            merged = merged.drop_duplicates(subset=list(key_columns), keep="last").reset_index(drop=True)
        _atomic_write_parquet(merged, path)
        return merged


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def normalize_pathogens(pathogens: Mapping[str, Sequence[str]] | Sequence[PathogenSpec]) -> tuple[PathogenSpec, ...]:
    """Normalize notebook-friendly pathogen configuration."""
    if isinstance(pathogens, Mapping):
        specs = tuple(PathogenSpec(str(name), tuple(str(alias) for alias in aliases)) for name, aliases in pathogens.items())
    else:
        specs = tuple(
            item if isinstance(item, PathogenSpec) else PathogenSpec(str(item))
            for item in pathogens
        )
    if not specs:
        raise ValueError("pathogens must contain at least one pathogen.")
    names: set[str] = set()
    for spec in specs:
        key = spec.name.casefold().strip()
        if not key or key in names:
            raise ValueError(f"Pathogen names must be unique and non-empty: {spec.name!r}")
        names.add(key)
        spec.all_terms()
    return specs


def build_pathogen_search_query(
    pathogen: PathogenSpec,
    search_type: str,
    *,
    search_bundles: Mapping[str, SearchBundle | Sequence[str]] | Sequence[SearchBundle] | None = None,
    animal_terms: Sequence[str] = DEFAULT_ANIMAL_TERMS,
    zoonosis_terms: Sequence[str] = DEFAULT_ZOONOSIS_TERMS,
) -> str:
    """Build one recall-oriented pathogen search query."""
    bundles = {
        bundle.search_type: bundle
        for bundle in normalize_search_bundles(
            search_bundles,
            animal_terms=animal_terms,
            zoonosis_terms=zoonosis_terms,
        )
    }
    if search_type not in bundles:
        raise ValueError(f"search_type must be one of {tuple(bundles)}.")
    terms = bundles[search_type].terms
    pathogen_clause = PubMedClient.keyword_query(pathogen.all_terms(), operator="OR")
    evidence_clause = PubMedClient.keyword_query(terms, operator="OR")
    return f"({pathogen_clause}) AND ({evidence_clause})"


def _path_rows(pathogens: Sequence[PathogenSpec], **kwargs: Any) -> list[dict[str, Any]]:
    return [asdict(spec) | kwargs for spec in pathogens]


def collect_pubmed_corpus(
    client: PubMedClient,
    pathogens: Mapping[str, Sequence[str]] | Sequence[PathogenSpec],
    output_dir: str | Path,
    *,
    search_bundles: Mapping[str, SearchBundle | Sequence[str]] | Sequence[SearchBundle] | None = None,
    animal_terms: Sequence[str] = DEFAULT_ANIMAL_TERMS,
    zoonosis_terms: Sequence[str] = DEFAULT_ZOONOSIS_TERMS,
    max_results_per_query: int | None = None,
    search_page_size: int = 1000,
    fetch_batch_size: int = 200,
    resume: bool = True,
    verbose: bool = True,
) -> CorpusCollectionResult:
    """Collect and checkpoint a deduplicated pathogen abstract corpus."""
    specs = normalize_pathogens(pathogens)
    bundles = normalize_search_bundles(
        search_bundles,
        animal_terms=animal_terms,
        zoonosis_terms=zoonosis_terms,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "pathogens": [{"name": spec.name, "aliases": list(spec.aliases)} for spec in specs],
        "search_bundles": [asdict(bundle) for bundle in bundles],
        "max_results_per_query": max_results_per_query,
        "search_page_size": search_page_size,
        "fetch_batch_size": fetch_batch_size,
    }
    config_hash = _stable_hash(config)
    search_runs_path = output / "search_runs.parquet"
    search_pmids_path = output / "search_pmids.parquet"
    corpus_path = output / "corpus_articles.parquet"
    manifest_path = output / "run_manifest.json"

    if not resume:
        for path in (search_runs_path, search_pmids_path, corpus_path, manifest_path):
            path.unlink(missing_ok=True)

    search_runs = _read_parquet(search_runs_path) if resume else pd.DataFrame()
    search_pmids = _read_parquet(search_pmids_path) if resume else pd.DataFrame()
    query_rows: list[dict[str, Any]] = []
    pmid_rows: list[dict[str, Any]] = []
    for spec in specs:
        for bundle in bundles:
            search_type = bundle.search_type
            query = build_pathogen_search_query(
                spec,
                search_type,
                search_bundles=bundles,
                animal_terms=animal_terms,
                zoonosis_terms=zoonosis_terms,
            )
            query_key = _stable_hash({"config_hash": config_hash, "pathogen": spec.name, "search_type": search_type, "query": query})
            complete = (
                not search_runs.empty
                and (_where(search_runs, "query_key", query_key) & _where(search_runs, "status", "complete")).any()
            )
            if complete:
                continue
            try:
                total, pmids = client.search_all(
                    query,
                    max_results=max_results_per_query,
                    page_size=search_page_size,
                )
                now = datetime.now(timezone.utc).isoformat()
                query_rows.append({
                    "query_key": query_key,
                    "config_hash": config_hash,
                    "pathogen": spec.name,
                    "search_type": search_type,
                    "query": query,
                    "status": "complete",
                    "total_matches": total,
                    "retrieved_pmids": len(pmids),
                    "updated_at": now,
                    "error": "",
                })
                pmid_rows.extend(
                    {
                        "query_key": query_key,
                        "config_hash": config_hash,
                        "pathogen": spec.name,
                        "search_type": search_type,
                        "pmid": str(pmid),
                    }
                    for pmid in pmids
                )
                if verbose:
                    print(f"{spec.name} / {search_type}: {len(pmids)} PMIDs from {total} matches")
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                query_rows.append({
                    "query_key": query_key,
                    "config_hash": config_hash,
                    "pathogen": spec.name,
                    "search_type": search_type,
                    "query": query,
                    "status": "failed",
                    "total_matches": None,
                    "retrieved_pmids": 0,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                if verbose:
                    print(f"Search failed for {spec.name} / {search_type}: {exc}")
            if query_rows:
                search_runs = _merge_checkpoint(search_runs_path, pd.DataFrame(query_rows), ["query_key"])
                query_rows.clear()
            if pmid_rows:
                search_pmids = _merge_checkpoint(search_pmids_path, pd.DataFrame(pmid_rows), ["query_key", "pmid"])
                pmid_rows.clear()

    current_pmids = search_pmids[_where(search_pmids, "config_hash", config_hash)].copy()
    if current_pmids.empty:
        candidates = pd.DataFrame(columns=["config_hash", "pathogen", "pmid", "source_search_types", "source_queries"])
    else:
        run_lookup = search_runs.set_index("query_key")["query"] if not search_runs.empty else pd.Series(dtype=str)
        grouped = current_pmids.groupby(["config_hash", "pathogen", "pmid"], sort=True)
        rows = []
        for (hash_value, pathogen, pmid), group in grouped:
            query_keys = group["query_key"].astype(str).tolist()
            rows.append({
                "config_hash": hash_value,
                "pathogen": pathogen,
                "pmid": str(pmid),
                "source_search_types": json.dumps(sorted(set(group["search_type"])), ensure_ascii=False),
                "source_queries": json.dumps(sorted({str(run_lookup.get(key, "")) for key in query_keys if run_lookup.get(key, "")}), ensure_ascii=False),
            })
        candidates = pd.DataFrame(rows)

    corpus = _read_parquet(corpus_path) if resume else pd.DataFrame()
    current_corpus = corpus[_where(corpus, "config_hash", config_hash)].copy() if not corpus.empty else pd.DataFrame()
    existing_by_key = {
        (str(row.pathogen), str(row.pmid)): row._asdict()
        for row in current_corpus.itertuples(index=False)
    }
    pending = []
    for row in candidates.itertuples(index=False):
        key = (str(row.pathogen), str(row.pmid))
        prior = existing_by_key.get(key)
        if prior is None or prior.get("fetch_status") not in {"ok", "no_abstract"}:
            pending.append((key, row))
        elif prior.get("source_search_types") != row.source_search_types or prior.get("source_queries") != row.source_queries:
            prior["source_search_types"] = row.source_search_types
            prior["source_queries"] = row.source_queries

    for start in range(0, len(pending), fetch_batch_size):
        batch = pending[start : start + fetch_batch_size]
        pmids = [key[1] for key, _ in batch]
        try:
            fetched = {article.pmid: article for article in client.fetch_many(pmids, batch_size=min(fetch_batch_size, 200))}
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            fetched = {}
            fetch_error = f"{type(exc).__name__}: {exc}"
        else:
            fetch_error = ""
        updates = []
        for (pathogen, pmid), candidate in batch:
            article = fetched.get(pmid)
            if article is None:
                updates.append({
                    **candidate._asdict(),
                    "title": "",
                    "abstract": "",
                    "journal": "",
                    "publication_date": "",
                    "doi": "",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "fetch_status": "network_error" if fetch_error else "not_returned",
                    "fetch_error": fetch_error,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
            else:
                updates.append({
                    **candidate._asdict(),
                    **article.to_mapping(),
                    "fetch_status": "ok" if article.abstract.strip() else "no_abstract",
                    "fetch_error": "",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
        if updates:
            current_corpus = _merge_checkpoint(corpus_path, pd.DataFrame(updates), ["config_hash", "pathogen", "pmid"])
            corpus = current_corpus
            existing_by_key.update({(str(row["pathogen"]), str(row["pmid"])): row for row in updates})

    if existing_by_key:
        current_corpus = pd.DataFrame(list(existing_by_key.values()))
        current_corpus = _merge_checkpoint(corpus_path, current_corpus, ["config_hash", "pathogen", "pmid"])
    else:
        current_corpus = _read_parquet(corpus_path)

    manifest = {
        "run_type": "pubmed_pathogen_zoonosis_corpus",
        "config": config,
        "config_hash": config_hash,
        "paths": {
            "search_runs": str(search_runs_path),
            "search_pmids": str(search_pmids_path),
            "corpus_articles": str(corpus_path),
        },
        "search_queries": int(len(search_runs[_where(search_runs, "config_hash", config_hash)])) if not search_runs.empty else 0,
        "search_failures": int(_where(search_runs, "status", "failed").sum()) if not search_runs.empty else 0,
        "corpus_rows": int(len(current_corpus[_where(current_corpus, "config_hash", config_hash)])) if not current_corpus.empty else 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    return CorpusCollectionResult(current_corpus, search_runs, manifest)


def load_corpus_articles(
    corpus_path: str | Path,
    *,
    pathogens: Sequence[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    search_types: Sequence[str] | None = None,
    require_abstract: bool = True,
) -> pd.DataFrame:
    """Load and filter the canonical PubMed corpus Parquet file.

    Filtering happens before downstream graphicalization, so a large corpus
    can be narrowed by pathogen, publication year, and search provenance.
    """
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("start_year must not be greater than end_year.")
    source = Path(corpus_path)
    frame = pd.read_parquet(source)
    required = {"pathogen", "pmid", "abstract", "publication_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Corpus Parquet is missing required columns: {', '.join(missing)}")

    filtered = frame.copy()
    if pathogens is not None:
        wanted = {str(pathogen).strip().casefold() for pathogen in pathogens if str(pathogen).strip()}
        filtered = filtered[filtered["pathogen"].astype(str).str.casefold().isin(wanted)]
    years = filtered["publication_date"].map(_publication_year)
    filtered = filtered.assign(publication_year=years)
    if start_year is not None:
        filtered = filtered[filtered["publication_year"].fillna(-1) >= start_year]
    if end_year is not None:
        filtered = filtered[filtered["publication_year"].fillna(-1) <= end_year]
    if search_types is not None:
        wanted_types = {str(search_type).strip() for search_type in search_types if str(search_type).strip()}
        if not wanted_types:
            raise ValueError("search_types must contain at least one non-empty value.")
        def contains_search_type(value: Any) -> bool:
            try:
                values = json.loads(str(value))
            except (TypeError, ValueError, json.JSONDecodeError):
                values = []
            return bool(wanted_types.intersection(str(item) for item in values))
        filtered = filtered[filtered.get("source_search_types", pd.Series("[]", index=filtered.index)).map(contains_search_type)]
    if require_abstract:
        filtered = filtered[
            filtered["abstract"].fillna("").astype(str).str.strip().ne("")
            & filtered.get("fetch_status", pd.Series("ok", index=filtered.index)).astype(str).eq("ok")
        ]
    return filtered.sort_values(
        [column for column in ("pathogen", "publication_year", "pmid") if column in filtered.columns],
        na_position="last",
    ).reset_index(drop=True)


SCREENING_SYSTEM_PROMPT = """
You are a careful biomedical evidence reviewer screening one PubMed abstract for one target pathogen.
Use only the supplied title and abstract. Do not transfer human infection, zoonosis, spillover, or
animal-infection evidence from another pathogen mentioned nearby. The target pathogen must be the
subject, agent, host-associated organism, or explicitly included member of the reported study group.

Return JSON with exactly these keys:
category (integer 1, 2, or 3), target_pathogen_supported (boolean),
animal_infection_supported (boolean), zoonosis_supported (boolean), evidence_phrases (array of strings),
rationale (short string), confidence (number from 0 to 1), review_required (boolean).

Use category 3 only when the abstract explicitly attributes zoonosis, spillover, animal-to-human
transmission, human infection, or human disease to the target pathogen. Use category 1 for clear
animal-infection evidence without zoonosis evidence in this abstract. Use category 2 when there is
animal/pathogen context but the evidence is indirect, mixed, or requires review. Set review_required
when attribution is ambiguous, the target pathogen is not clearly linked to the finding, or confidence
is below 0.75. Do not make claims about evidence outside this abstract.
""".strip()


REFINEMENT_SYSTEM_PROMPT = """
You are a careful biomedical retrieval-refinement reviewer. Decide whether one PubMed abstract is
actually about the supplied target pathogen. Use only the title and abstract. Do not transfer
evidence from another pathogen mentioned in the same abstract, even when the other pathogen is
related or appears in a comparison, review, background statement, or list of organisms.

Return JSON with exactly these keys:
keep (boolean), target_pathogen_supported (boolean), evidence_relevant (boolean),
evidence_phrases (array of strings), rationale (short string), confidence (number from 0 to 1),
review_required (boolean).

Set keep=true only when the target pathogen is clearly the subject, agent, or explicitly included
study organism and the abstract is materially relevant to the configured retrieval corpus. Set
keep=false for incidental keyword matches, unrelated pathogens, or abstracts where attribution is
not supported. Set review_required=true when the attribution is ambiguous or confidence is below
0.75. A confident rejection is not review_required.
""".strip()


def build_screening_prompt(pathogen: str, aliases: Sequence[str], title: str, abstract: str) -> str:
    return json.dumps(
        {
            "target_pathogen": pathogen,
            "target_aliases": list(aliases),
            "title": title,
            "abstract": abstract or "[no abstract available]",
        },
        ensure_ascii=False,
        indent=2,
    )


def _json_object(text: str) -> Mapping[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object.")
    value = json.loads(match.group(0))
    if not isinstance(value, Mapping):
        raise ValueError("LLM response JSON must be an object.")
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "yes", "1"}


def _is_terminal_llm_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "insufficient_quota",
            "exceeded your current quota",
            "request_headers_too_large",
            "request headers are too large",
        )
    )


def build_refinement_prompt(pathogen: str, aliases: Sequence[str], title: str, abstract: str) -> str:
    """Build the JSON payload for first-stage pathogen relevance refinement."""
    return json.dumps(
        {
            "target_pathogen": pathogen,
            "target_aliases": list(aliases),
            "title": title,
            "abstract": abstract or "[no abstract available]",
        },
        ensure_ascii=False,
        indent=2,
    )


def normalize_refinement_output(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a relevance decision and identify rows safe to retain."""
    keep = _bool(raw.get("keep"))
    target_supported = _bool(raw.get("target_pathogen_supported"))
    evidence_relevant = _bool(raw.get("evidence_relevant"))
    evidence = raw.get("evidence_phrases", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence = [str(item).strip() for item in evidence if str(item).strip()]
    confidence = raw.get("confidence")
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = None
    review_required = _bool(raw.get("review_required"))
    if keep and (not target_supported or not evidence_relevant or confidence is None or confidence < 0.75):
        review_required = True
    valid = confidence is not None
    accepted = bool(keep and target_supported and evidence_relevant and not review_required and valid)
    return {
        "refinement_keep": keep,
        "target_pathogen_supported": target_supported,
        "evidence_relevant": evidence_relevant,
        "evidence_phrases": evidence,
        "rationale": str(raw.get("rationale", "")).strip(),
        "confidence": confidence,
        "review_required": review_required,
        "accepted": accepted,
        "classification_status": "classified" if valid else "invalid_response",
    }


def refine_abstract(
    llm: ChatCompleter,
    pathogen: str,
    aliases: Sequence[str],
    title: str,
    abstract: str,
    *,
    system_prompt: str = REFINEMENT_SYSTEM_PROMPT,
    max_tokens: int = 768,
    retries: int = 3,
    backoff_factor: float = 2.0,
) -> dict[str, Any]:
    """Refine one candidate abstract with bounded retry/backoff."""
    if retries < 1:
        raise ValueError("retries must be at least 1.")
    if backoff_factor < 1:
        raise ValueError("backoff_factor must be at least 1.")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = llm.complete(
                build_refinement_prompt(pathogen, aliases, title, abstract),
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=0,
            )
            return {
                **normalize_refinement_output(_json_object(raw)),
                "attempt_count": attempt + 1,
            }
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if _is_terminal_llm_error(exc):
                raise TerminalLLMError(str(exc)) from exc
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff_factor**attempt)
    return {
        "refinement_keep": False,
        "target_pathogen_supported": False,
        "evidence_relevant": False,
        "evidence_phrases": [],
        "rationale": f"LLM failure: {last_error}",
        "confidence": None,
        "review_required": True,
        "accepted": False,
        "classification_status": "failed",
        "last_error": f"{type(last_error).__name__}: {last_error}",
        "attempt_count": retries,
    }


def normalize_screening_output(raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        category = int(raw.get("category"))
    except (TypeError, ValueError):
        category = None
    if category not in {CATEGORY_1, CATEGORY_2, CATEGORY_3}:
        category = None
    evidence = raw.get("evidence_phrases", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence = [str(item).strip() for item in evidence if str(item).strip()]
    confidence = raw.get("confidence")
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = None
    review_required = _bool(raw.get("review_required"))
    target_supported = _bool(raw.get("target_pathogen_supported"))
    animal_supported = _bool(raw.get("animal_infection_supported"))
    zoonosis_supported = _bool(raw.get("zoonosis_supported"))
    if category is None or not target_supported or confidence is None or confidence < 0.75:
        review_required = True
    if zoonosis_supported and category != CATEGORY_3:
        review_required = True
    if category == CATEGORY_3 and not zoonosis_supported:
        review_required = True
    return {
        "llm_category": category,
        "target_pathogen_supported": target_supported,
        "animal_infection_supported": animal_supported,
        "zoonosis_supported": zoonosis_supported,
        "evidence_phrases": evidence,
        "rationale": str(raw.get("rationale", "")).strip(),
        "confidence": confidence,
        "review_required": review_required,
        "classification_status": "classified" if category is not None else "invalid_response",
    }


def classify_abstract(
    llm: ChatCompleter,
    pathogen: str,
    aliases: Sequence[str],
    title: str,
    abstract: str,
    *,
    system_prompt: str = SCREENING_SYSTEM_PROMPT,
    max_tokens: int = 1024,
    retries: int = 3,
    backoff_factor: float = 2.0,
) -> dict[str, Any]:
    """Classify one abstract with bounded retry/backoff and normalization."""
    if retries < 1:
        raise ValueError("retries must be at least 1.")
    if backoff_factor < 1:
        raise ValueError("backoff_factor must be at least 1.")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = llm.complete(
                build_screening_prompt(pathogen, aliases, title, abstract),
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=0,
            )
            return {
                **normalize_screening_output(_json_object(raw)),
                "attempt_count": attempt + 1,
            }
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if _is_terminal_llm_error(exc):
                raise TerminalLLMError(str(exc)) from exc
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff_factor**attempt)
    return {
        "llm_category": None,
        "target_pathogen_supported": False,
        "animal_infection_supported": False,
        "zoonosis_supported": False,
        "evidence_phrases": [],
        "rationale": f"LLM failure: {last_error}",
        "confidence": None,
        "review_required": True,
        "classification_status": "failed",
        "last_error": f"{type(last_error).__name__}: {last_error}",
        "attempt_count": retries,
    }


def screen_pubmed_corpus(
    corpus: pd.DataFrame | str | Path,
    pathogens: Mapping[str, Sequence[str]] | Sequence[PathogenSpec],
    llm: ChatCompleter,
    output_dir: str | Path,
    *,
    model: str = "unknown",
    system_prompt: str = SCREENING_SYSTEM_PROMPT,
    max_tokens: int = 1024,
    retries: int = 3,
    max_llm_calls: int | None = None,
    save_every: int = 10,
    resume: bool = True,
    retry_failed: bool = False,
    verbose: bool = True,
) -> ScreeningRunResult:
    """Screen every available corpus abstract with resumable checkpoints."""
    specs = normalize_pathogens(pathogens)
    aliases = {spec.name: spec.all_terms() for spec in specs}
    corpus_df = pd.read_parquet(corpus) if isinstance(corpus, (str, Path)) else corpus.copy()
    required = {"config_hash", "pathogen", "pmid", "title", "abstract", "fetch_status", "publication_date"}
    missing = required - set(corpus_df.columns)
    if missing:
        raise KeyError(f"corpus is missing required columns: {sorted(missing)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    screening_hash = _stable_hash({"model": model, "prompt_hash": prompt_hash, "max_tokens": max_tokens})
    screening_path = output / "llm_screening.parquet"
    failures_path = output / "llm_failures.parquet"
    manifest_path = output / "run_manifest.json"

    if not resume:
        for path in (screening_path, failures_path):
            path.unlink(missing_ok=True)
    existing = _read_parquet(screening_path) if resume else pd.DataFrame()
    current_existing = existing[_where(existing, "screening_hash", screening_hash)].copy() if not existing.empty else pd.DataFrame()
    done_keys = set()
    if not current_existing.empty and "classification_status" in current_existing:
        for row in current_existing.itertuples(index=False):
            if row.classification_status == "classified" or (not retry_failed and row.classification_status == "failed"):
                done_keys.add((str(row.pathogen), str(row.pmid)))
    pending = corpus_df[(corpus_df["fetch_status"] == "ok") & corpus_df["abstract"].fillna("").astype(str).str.strip().ne("")].copy()
    pending = pending[~pending.apply(lambda row: (str(row.pathogen), str(row.pmid)) in done_keys, axis=1)]
    calls = 0
    updates: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        for row in pending.itertuples(index=False):
            if max_llm_calls is not None and calls >= max_llm_calls:
                break
            spec_aliases = aliases.get(str(row.pathogen), (str(row.pathogen),))
            stop_after = False
            try:
                verdict = classify_abstract(
                    llm,
                    str(row.pathogen),
                    spec_aliases,
                    str(row.title or ""),
                    str(row.abstract or ""),
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    retries=retries,
                )
            except TerminalLLMError as exc:
                verdict = {
                    "llm_category": None,
                    "target_pathogen_supported": False,
                    "animal_infection_supported": False,
                    "zoonosis_supported": False,
                    "evidence_phrases": [],
                    "rationale": f"Terminal LLM failure: {exc}",
                    "confidence": None,
                    "review_required": True,
                    "classification_status": "terminal_failure",
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "attempt_count": 1,
                }
                stop_after = True
            calls += 1
            updates.append({
                "screening_hash": screening_hash,
                "corpus_config_hash": str(row.config_hash),
                "pathogen": str(row.pathogen),
                "pmid": str(row.pmid),
                "title": str(row.title or ""),
                "abstract": str(row.abstract or ""),
                "publication_date": str(row.publication_date or ""),
                "publication_year": _publication_year(row.publication_date),
                "model": model,
                "prompt_hash": prompt_hash,
                "screened_at": datetime.now(timezone.utc).isoformat(),
                **verdict,
            })
            if verbose:
                print(f"Screened {calls}/{len(pending)}: {row.pathogen} / PMID {row.pmid}")
            if len(updates) >= max(1, save_every):
                _merge_checkpoint(screening_path, pd.DataFrame(updates), ["screening_hash", "pathogen", "pmid"])
                updates.clear()
            if stop_after:
                break
    except KeyboardInterrupt:
        raise
    finally:
        if updates:
            _merge_checkpoint(screening_path, pd.DataFrame(updates), ["screening_hash", "pathogen", "pmid"])

    screening = _read_parquet(screening_path)
    failures = screening[_where(screening, "screening_hash", screening_hash)].copy()
    failures = failures[failures["classification_status"].isin(["failed", "invalid_response", "terminal_failure"])] if "classification_status" in failures else failures
    if not failures.empty:
        _atomic_write_parquet(failures, failures_path)
    else:
        failures_path.unlink(missing_ok=True)
    screening_manifest = {
        "run_type": "pubmed_pathogen_zoonosis_screening",
        "screening_hash": screening_hash,
        "model": model,
        "prompt_hash": prompt_hash,
        "started_at": started,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_rows": int(len(corpus_df)),
        "candidate_rows": int(len(pending)),
        "llm_calls": calls,
        "classified_rows": int(_where(screening, "classification_status", "classified").sum()) if not screening.empty else 0,
        "failed_rows": int(len(failures)),
        "paths": {"screening": str(screening_path), "failures": str(failures_path)},
    }
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
    manifest["screening"] = screening_manifest
    _write_json(manifest_path, manifest)
    return ScreeningRunResult(screening, failures, screening_manifest)


def refine_pubmed_corpus(
    corpus: pd.DataFrame | str | Path,
    pathogens: Mapping[str, Sequence[str]] | Sequence[PathogenSpec],
    llm: ChatCompleter,
    output_dir: str | Path,
    *,
    model: str = "unknown",
    system_prompt: str = REFINEMENT_SYSTEM_PROMPT,
    max_tokens: int = 768,
    retries: int = 3,
    max_llm_calls: int | None = None,
    save_every: int = 10,
    resume: bool = True,
    retry_failed: bool = False,
    verbose: bool = True,
) -> RefinementRunResult:
    """Keep only LLM-confirmed target-pathogen abstracts for stage two."""
    specs = normalize_pathogens(pathogens)
    aliases = {spec.name: spec.all_terms() for spec in specs}
    corpus_df = pd.read_parquet(corpus) if isinstance(corpus, (str, Path)) else corpus.copy()
    required = {"pathogen", "pmid", "title", "abstract", "fetch_status", "publication_date"}
    missing = required - set(corpus_df.columns)
    if missing:
        raise KeyError(f"corpus is missing required columns: {sorted(missing)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corpus_identity = [
        {
            "pathogen": str(row.pathogen),
            "pmid": str(row.pmid),
            "abstract_sha256": hashlib.sha256(str(row.abstract or "").encode("utf-8")).hexdigest(),
        }
        for row in corpus_df.itertuples(index=False)
    ]
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    refinement_hash = _stable_hash(
        {
            "corpus_identity": corpus_identity,
            "model": model,
            "prompt_hash": prompt_hash,
            "max_tokens": max_tokens,
        }
    )
    refinement_path = output / "llm_refinement.parquet"
    failures_path = output / "refinement_failures.parquet"
    refined_path = output / "refined_corpus_articles.parquet"
    manifest_path = output / "run_manifest.json"
    if not resume:
        for path in (refinement_path, failures_path, refined_path):
            path.unlink(missing_ok=True)

    existing = _read_parquet(refinement_path) if resume else pd.DataFrame()
    current_existing = (
        existing[_where(existing, "refinement_hash", refinement_hash)].copy()
        if not existing.empty
        else pd.DataFrame()
    )
    done_keys: set[tuple[str, str]] = set()
    if not current_existing.empty and "classification_status" in current_existing:
        for row in current_existing.itertuples(index=False):
            if row.classification_status == "classified" or (
                not retry_failed and row.classification_status in {"failed", "invalid_response"}
            ):
                done_keys.add((str(row.pathogen), str(row.pmid)))
    pending = corpus_df[
        (corpus_df["fetch_status"] == "ok")
        & corpus_df["abstract"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    pending = pending[
        ~pending.apply(
            lambda row: (str(row.pathogen), str(row.pmid)) in done_keys,
            axis=1,
        )
    ]
    calls = 0
    updates: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        for row in pending.itertuples(index=False):
            if max_llm_calls is not None and calls >= max_llm_calls:
                break
            spec_aliases = aliases.get(str(row.pathogen), (str(row.pathogen),))
            stop_after = False
            try:
                verdict = refine_abstract(
                    llm,
                    str(row.pathogen),
                    spec_aliases,
                    str(row.title or ""),
                    str(row.abstract or ""),
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    retries=retries,
                )
            except TerminalLLMError as exc:
                verdict = {
                    "refinement_keep": False,
                    "target_pathogen_supported": False,
                    "evidence_relevant": False,
                    "evidence_phrases": [],
                    "rationale": f"Terminal LLM failure: {exc}",
                    "confidence": None,
                    "review_required": True,
                    "accepted": False,
                    "classification_status": "terminal_failure",
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "attempt_count": 1,
                }
                stop_after = True
            calls += 1
            updates.append(
                {
                    "refinement_hash": refinement_hash,
                    "corpus_config_hash": str(getattr(row, "config_hash", "")),
                    "pathogen": str(row.pathogen),
                    "pmid": str(row.pmid),
                    "title": str(row.title or ""),
                    "abstract": str(row.abstract or ""),
                    "publication_date": str(row.publication_date or ""),
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "refined_at": datetime.now(timezone.utc).isoformat(),
                    **verdict,
                }
            )
            if verbose:
                print(f"Refined {calls}/{len(pending)}: {row.pathogen} / PMID {row.pmid}")
            if len(updates) >= max(1, save_every):
                _merge_checkpoint(
                    refinement_path,
                    pd.DataFrame(updates),
                    ["refinement_hash", "pathogen", "pmid"],
                )
                updates.clear()
            if stop_after:
                break
    except KeyboardInterrupt:
        raise
    finally:
        if updates:
            _merge_checkpoint(
                refinement_path,
                pd.DataFrame(updates),
                ["refinement_hash", "pathogen", "pmid"],
            )

    refinement = _read_parquet(refinement_path)
    current_refinement = refinement[_where(refinement, "refinement_hash", refinement_hash)].copy()
    failures = current_refinement[
        current_refinement["classification_status"].isin(
            ["failed", "invalid_response", "terminal_failure"]
        )
    ].copy() if not current_refinement.empty else pd.DataFrame()
    if not failures.empty:
        _atomic_write_parquet(failures, failures_path)
    else:
        failures_path.unlink(missing_ok=True)

    accepted = current_refinement[
        (current_refinement["classification_status"] == "classified")
        & current_refinement["accepted"].fillna(False).astype(bool)
    ].copy() if not current_refinement.empty else pd.DataFrame()
    decision_columns = [
        "pathogen",
        "pmid",
        "refinement_hash",
        "refinement_keep",
        "target_pathogen_supported",
        "evidence_relevant",
        "evidence_phrases",
        "rationale",
        "confidence",
        "review_required",
        "accepted",
        "model",
        "prompt_hash",
        "refined_at",
    ]
    decision_columns = [column for column in decision_columns if column in accepted.columns]
    if accepted.empty:
        refined_corpus = corpus_df.iloc[0:0].copy()
    else:
        refined_corpus = corpus_df.merge(
            accepted[decision_columns],
            on=["pathogen", "pmid"],
            how="inner",
            suffixes=("", "_refinement"),
        )
    _atomic_write_parquet(refined_corpus, refined_path)

    refinement_manifest = {
        "run_type": "pubmed_pathogen_retrieval_refinement",
        "refinement_hash": refinement_hash,
        "model": model,
        "prompt_hash": prompt_hash,
        "started_at": started,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_rows": int(len(corpus_df)),
        "candidate_rows": int(len(pending)),
        "llm_calls": calls,
        "accepted_rows": int(len(refined_corpus)),
        "rejected_rows": int(
            ((current_refinement["classification_status"] == "classified")
             & ~current_refinement["accepted"].fillna(False).astype(bool)).sum()
        ) if not current_refinement.empty else 0,
        "failed_rows": int(len(failures)),
        "paths": {
            "refinement": str(refinement_path),
            "failures": str(failures_path),
            "refined_corpus": str(refined_path),
        },
    }
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
    manifest["refinement"] = refinement_manifest
    _write_json(manifest_path, manifest)
    return RefinementRunResult(refined_corpus, refinement, failures, refinement_manifest)


def load_refinement_run(
    corpus: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    refinement_hash: str | None = None,
) -> RefinementRunResult:
    """Reconstruct a refinement result from its persisted checkpoints.

    This is useful after a notebook kernel interruption, when
    :func:`refine_pubmed_corpus` has flushed its in-memory batch but did not
    return a ``RefinementRunResult`` to the caller.
    """
    corpus_df = pd.read_parquet(corpus) if isinstance(corpus, (str, Path)) else corpus.copy()
    required = {"pathogen", "pmid"}
    missing = required - set(corpus_df.columns)
    if missing:
        raise KeyError(f"corpus is missing required columns: {sorted(missing)}")

    output = Path(output_dir)
    refinement_path = output / "llm_refinement.parquet"
    if not refinement_path.exists():
        raise FileNotFoundError(
            f"No refinement checkpoint found at {refinement_path}. "
            "Run the refinement cell first."
        )
    refinement = _read_parquet(refinement_path)
    if refinement.empty:
        raise ValueError(f"Refinement checkpoint is empty: {refinement_path}")

    top_level_manifest: dict[str, Any] = {}
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        try:
            top_level_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            top_level_manifest = {}

    selected_hash = refinement_hash
    if selected_hash is None and "refined_at" in refinement:
        latest = refinement.sort_values("refined_at").iloc[-1]
        selected_hash = str(latest["refinement_hash"])
    if selected_hash is None:
        selected_hash = str(
            top_level_manifest.get("refinement", {}).get("refinement_hash", "")
        ) or None
    if selected_hash is None:
        hashes = refinement["refinement_hash"].dropna().astype(str).unique()
        if len(hashes) != 1:
            raise ValueError(
                "Multiple refinement checkpoints are present; pass refinement_hash explicitly."
            )
        selected_hash = str(hashes[0])

    current_refinement = refinement[
        _where(refinement, "refinement_hash", selected_hash)
    ].copy()
    if current_refinement.empty:
        raise KeyError(f"No rows found for refinement_hash={selected_hash!r}")

    failures = current_refinement[
        current_refinement["classification_status"].isin(
            ["failed", "invalid_response", "terminal_failure"]
        )
    ].copy() if "classification_status" in current_refinement else pd.DataFrame()
    accepted = current_refinement[
        (current_refinement["classification_status"] == "classified")
        & current_refinement["accepted"].fillna(False).astype(bool)
    ].copy() if {"classification_status", "accepted"}.issubset(current_refinement.columns) else pd.DataFrame()

    decision_columns = [
        "pathogen",
        "pmid",
        "refinement_hash",
        "refinement_keep",
        "target_pathogen_supported",
        "evidence_relevant",
        "evidence_phrases",
        "rationale",
        "confidence",
        "review_required",
        "accepted",
        "model",
        "prompt_hash",
        "refined_at",
    ]
    decision_columns = [column for column in decision_columns if column in accepted.columns]
    if accepted.empty:
        refined_corpus = corpus_df.iloc[0:0].copy()
    else:
        refined_corpus = corpus_df.merge(
            accepted[decision_columns],
            on=["pathogen", "pmid"],
            how="inner",
            suffixes=("", "_refinement"),
        )

    refinement_manifest = dict(top_level_manifest.get("refinement", {}))
    refinement_manifest.setdefault("refinement_hash", selected_hash)
    refinement_manifest.setdefault("paths", {})
    refinement_manifest["paths"].setdefault("refinement", str(refinement_path))
    refinement_manifest["reconstructed_from_checkpoints"] = True
    return RefinementRunResult(
        refined_corpus=refined_corpus,
        refinement=current_refinement,
        failures=failures,
        manifest=refinement_manifest,
    )


def _publication_year(value: Any) -> int | None:
    match = re.search(r"(\d{4})", str(value or ""))
    return int(match.group(1)) if match else None


def derive_category_counts(screening: pd.DataFrame, output_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive corpus-relative categories, timeline, and total/yearly counts."""
    work = screening.copy()
    if work.empty:
        work = pd.DataFrame(
            columns=[
                "pathogen",
                "pmid",
                "publication_year",
                "classification_status",
                "review_required",
                "target_pathogen_supported",
                "animal_infection_supported",
                "zoonosis_supported",
                "final_category",
                "predates_first_confirmed_zoonosis",
            ]
        )
        timeline = pd.DataFrame(columns=["pathogen", "first_confirmed_zoonosis_year"])
        counts = pd.DataFrame(columns=["pathogen", "publication_year", "category", "count"])
        pathogen_table = pd.DataFrame(
            columns=[
                "pathogen",
                "category_1_count",
                "category_2_count",
                "category_3_count",
                "total_counted",
                "first_confirmed_zoonosis_year",
            ]
        )
        if output_dir is not None:
            output = Path(output_dir)
            _atomic_write_parquet(work, output / "llm_screening_with_categories.parquet")
            _atomic_write_parquet(timeline, output / "zoonosis_timeline.parquet")
            _atomic_write_parquet(counts, output / "category_counts.parquet")
            _atomic_write_parquet(pathogen_table, output / "pathogen_category_table.parquet")
        return work, timeline, counts
    work["publication_year"] = work["publication_year"].map(_publication_year)
    valid = work[
        (work["classification_status"] == "classified")
        & (~work["review_required"].fillna(True).astype(bool))
        & (work["target_pathogen_supported"].fillna(False).astype(bool))
    ].copy()
    positive = valid[valid["zoonosis_supported"].fillna(False).astype(bool)]
    timeline = (
        positive.groupby("pathogen", as_index=False)["publication_year"]
        .min()
        .rename(columns={"publication_year": "first_confirmed_zoonosis_year"})
    )
    work = work.merge(timeline, on="pathogen", how="left")
    work["final_category"] = pd.NA
    animal_only = (
        work["classification_status"].eq("classified")
        & ~work["review_required"].fillna(True).astype(bool)
        & work["target_pathogen_supported"].fillna(False).astype(bool)
        & work["animal_infection_supported"].fillna(False).astype(bool)
        & ~work["zoonosis_supported"].fillna(False).astype(bool)
    )
    work.loc[animal_only & work["first_confirmed_zoonosis_year"].isna(), "final_category"] = CATEGORY_1
    work.loc[animal_only & work["first_confirmed_zoonosis_year"].notna(), "final_category"] = CATEGORY_2
    work.loc[
        work["classification_status"].eq("classified")
        & ~work["review_required"].fillna(True).astype(bool)
        & work["target_pathogen_supported"].fillna(False).astype(bool)
        & work["zoonosis_supported"].fillna(False).astype(bool),
        "final_category",
    ] = CATEGORY_3
    work["predates_first_confirmed_zoonosis"] = (
        work["final_category"].eq(CATEGORY_2)
        & work["publication_year"].notna()
        & work["first_confirmed_zoonosis_year"].notna()
        & (work["publication_year"] < work["first_confirmed_zoonosis_year"])
    )
    counted = work[work["final_category"].notna()].copy()
    counted["final_category"] = counted["final_category"].astype(int)
    yearly = (
        counted.dropna(subset=["publication_year"])
        .groupby(["pathogen", "publication_year", "final_category"], as_index=False)
        .size()
        .rename(columns={"size": "count", "final_category": "category"})
    )
    totals = counted.groupby(["pathogen", "final_category"], as_index=False).size().rename(columns={"size": "count", "final_category": "category"})
    totals["publication_year"] = pd.NA
    counts = pd.concat([totals[yearly.columns], yearly], ignore_index=True) if not yearly.empty else totals
    pathogens = pd.DataFrame({"pathogen": sorted(work["pathogen"].dropna().astype(str).unique())})
    if counted.empty:
        pathogen_table = pathogens.copy()
        for category in (CATEGORY_1, CATEGORY_2, CATEGORY_3):
            pathogen_table[f"category_{category}_count"] = 0
        pathogen_table["total_counted"] = 0
    else:
        category_pivot = (
            counted.assign(counted_value=1)
            .pivot_table(
                index="pathogen",
                columns="final_category",
                values="counted_value",
                aggfunc="sum",
                fill_value=0,
            )
            .rename(columns={
                CATEGORY_1: "category_1_count",
                CATEGORY_2: "category_2_count",
                CATEGORY_3: "category_3_count",
            })
            .reset_index()
        )
        pathogen_table = pathogens.merge(category_pivot, on="pathogen", how="left")
        for category in (CATEGORY_1, CATEGORY_2, CATEGORY_3):
            column = f"category_{category}_count"
            if column not in pathogen_table:
                pathogen_table[column] = 0
        pathogen_table["total_counted"] = pathogen_table[
            ["category_1_count", "category_2_count", "category_3_count"]
        ].sum(axis=1)
    pathogen_table = pathogen_table.merge(timeline, on="pathogen", how="left")
    count_columns = [
        "pathogen",
        "category_1_count",
        "category_2_count",
        "category_3_count",
        "total_counted",
        "first_confirmed_zoonosis_year",
    ]
    pathogen_table = pathogen_table[count_columns]
    if output_dir is not None:
        output = Path(output_dir)
        _atomic_write_parquet(work, output / "llm_screening_with_categories.parquet")
        _atomic_write_parquet(timeline, output / "zoonosis_timeline.parquet")
        _atomic_write_parquet(counts, output / "category_counts.parquet")
        _atomic_write_parquet(pathogen_table, output / "pathogen_category_table.parquet")
        manifest_path = output / "run_manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
        manifest["categories"] = {
            "counted_rows": int(len(counted)),
            "category_counts": counts.to_dict(orient="records"),
            "timeline_rows": timeline.to_dict(orient="records"),
            "pathogen_category_table": pathogen_table.to_dict(orient="records"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(manifest_path, manifest)
    return work, timeline, counts


__all__ = [
    "CATEGORY_1",
    "CATEGORY_2",
    "CATEGORY_3",
    "DEFAULT_ANIMAL_TERMS",
    "DEFAULT_ZOONOSIS_TERMS",
    "REFINEMENT_SYSTEM_PROMPT",
    "SCREENING_SYSTEM_PROMPT",
    "ChatCompleter",
    "CorpusCollectionResult",
    "OpenAIChatCompleter",
    "PathogenSpec",
    "RefinementRunResult",
    "SearchBundle",
    "ScreeningRunResult",
    "TerminalLLMError",
    "build_pathogen_search_query",
    "build_refinement_prompt",
    "build_screening_prompt",
    "classify_abstract",
    "collect_pubmed_corpus",
    "derive_category_counts",
    "load_corpus_articles",
    "load_refinement_run",
    "normalize_pathogens",
    "normalize_refinement_output",
    "normalize_search_bundles",
    "normalize_screening_output",
    "load_search_bundle",
    "screen_pubmed_corpus",
    "refine_abstract",
    "refine_pubmed_corpus",
]
