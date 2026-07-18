# Notebook 01 — PubMed pathogen/zoonosis corpus screening

Source notebook: [`notebooks/01_pubmed_abstract_download.ipynb`](../notebooks/01_pubmed_abstract_download.ipynb)

## Purpose

This notebook builds a recall-oriented PubMed corpus for one or more named
pathogens. It loads two generic search bundles from YAML and performs one
search for every bundle/pathogen combination. The current assets define:

- an **animal evidence** bundle, using terms such as `animal`, `host`,
  `wildlife`, `livestock`, `veterinary`, and `infection`;
- a **zoonosis evidence** bundle, using terms such as `zoonosis`, `zoonotic`,
  `spillover`, `animal-to-human`, `human infection`, `human disease`, and
  `transmission`.

The notebook's search-term source files are:

```text
assets/search_bundles/pubmed_animal_evidence.yaml
assets/search_bundles/pubmed_zoonosis_evidence.yaml
```

The retrieval code accepts arbitrary named bundles, so another domain can
reuse the PubMed/checkpoint machinery by replacing these YAML files and the
pathogen/entity configuration. Category derivation remains intentionally
specific to the current animal-infection/zoonosis study design.

The two result sets are deduplicated by `(pathogen, PMID)`. Every candidate
abstract is then passed to a pathogen-aware LLM screen. The screen is required
to decide whether the evidence is actually about the target pathogen, rather
than transferring a human-infection or zoonosis statement from another
pathogen mentioned in the same abstract.

The final category is corpus-relative:

| Category | Meaning |
| --- | --- |
| 1 | The abstract confirms animal infection, but no reviewable abstract in the searched corpus confirms zoonosis for that pathogen. |
| 2 | The abstract confirms animal infection without zoonosis evidence, and another reviewable abstract for the same pathogen confirms zoonosis somewhere in the corpus. |
| 3 | The abstract explicitly supports zoonosis, spillover, human infection, or human disease caused by the target pathogen. |

Category 1 must be reported as **“no zoonosis evidence in the searched
corpus.”** It is not evidence that the pathogen can never undergo zoonosis.
Failed, invalid, ambiguous, and `review_required` classifications are excluded
from final category counts.

## Prerequisites

Install the base project and the OpenAI extra in the environment used by the
notebook:

```bash
python -m pip install -e ".[openai]"
```

The workflow also requires `pandas` and `pyarrow` for Parquet checkpoints.
Network access is required for NCBI E-utilities and the LLM provider.

Set the following environment variables before running the relevant cells:

| Variable | Required | Description |
| --- | --- | --- |
| `PUBMED_EMAIL` | Yes | Contact email sent to NCBI with every E-utilities request. The client validates that it is non-empty and contains `@`. |
| `PUBMED_API_KEY` | No | Optional NCBI API key. If absent, the request is made in the same way except that no `api_key` parameter is sent. |
| `OPENAI_API_KEY` | Yes for the screening cell | API credential used by `OpenAIChatCompleter`. |
| `OPENAI_MODEL` | No | Chat-completion model name; defaults to `gpt-4o-mini`. |

An NCBI API key is not needed for correctness. It can provide a higher request
rate limit, but leaving it unset does not disable searching. The notebook does
require `PUBMED_EMAIL` so that requests have a contact identity.

## Pathogen and search-bundle configuration

`PATHOGENS` maps one canonical pathogen name to optional aliases:

```python
PATHOGENS = {
    "Nipah virus": ["Nipah", "NiV"],
}
```

The canonical name is used in output rows and category aggregation. The
canonical name and aliases are OR-expanded in the PubMed query. Empty aliases
are removed and case-insensitive duplicates are collapsed. Add aliases only
when they are sufficiently specific to the pathogen; broad abbreviations can
reduce precision and increase the LLM review burden.

Each search-bundle YAML file has this shape:

```yaml
search_type: animal
label: Animal and host infection evidence
version: "1.0"
description: Recall-oriented terms for animal infection evidence.
terms:
  - animal
  - host
  - infection
```

`search_type` is the stable key used in query provenance and must be unique
within a run. `terms` must be a non-empty list; duplicate terms are removed
case-insensitively. `label`, `description`, and `version` are descriptive
metadata retained in the configuration hash/manifest. Notebook 01 validates
that the configured files have the expected `animal` and `zoonosis` types.

For each pathogen and search type, the generated query has this shape:

```text
(alias_1[Title/Abstract] OR alias_2[Title/Abstract] ...)
AND
(evidence_term_1[Title/Abstract] OR evidence_term_2[Title/Abstract] ...)
```

The search intentionally does not use aggressive `NOT` clauses. Recall is
prioritized at retrieval time, while pathogen-specific attribution and
incidental keyword matches are resolved during screening.

## Configuration parameters

### Search and persistence

| Variable | Default | Description |
| --- | ---: | --- |
| `SEARCH_BUNDLE_DIR` | `assets/search_bundles` | Directory containing YAML term bundles. |
| `SEARCH_BUNDLES` | Two YAML-loaded bundles | Mapping from `search_type` to validated `SearchBundle`. Add or replace bundles here for another retrieval domain. |
| `PUBMED_MAX_RESULTS_PER_QUERY` | `5000` | Maximum PMIDs retrieved from each search. `None` in the library means all available results. |
| `PUBMED_SEARCH_PAGE_SIZE` | `1000` | PMIDs requested per `esearch` page. PubMed pagination is handled by the client. |
| `PUBMED_FETCH_BATCH_SIZE` | `200` | PMIDs sent per abstract fetch batch. Larger values are split automatically by the PubMed layer. |
| `RESUME` | `True` | Reuse completed query and fetch checkpoints. Set to `False` to remove the collection checkpoints and start a fresh collection configuration. |
| `OUTPUT_DIR` | `outputs/pubmed_screening` | Directory for Parquet checkpoints and the JSON run manifest. |

The PubMed client retries transient request errors three times with exponential
backoff. Query results are checkpointed after each search, and abstract fetches
are checkpointed in batches. Writes use temporary files and atomic replacement
so an interrupted write does not leave a partially written Parquet file.

### LLM screening

| Variable | Default | Description |
| --- | ---: | --- |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model passed to the chat-completion adapter. |
| `LLM_MAX_TOKENS` | `1024` | Maximum completion tokens per abstract. |
| `LLM_RETRIES` | `3` | Attempts for malformed responses or retryable provider errors. |
| `LLM_MAX_CALLS` | `None` | Optional cap useful for smoke tests or staged runs. |
| `LLM_SAVE_EVERY` | `10` | Number of successful/failed decisions accumulated before merging the screening checkpoint. |
| `RETRY_FAILED_LLM_ROWS` | `False` | If `True`, previously failed classification rows are eligible for another attempt. |

The LLM adapter follows the repository's AMR-compatible `.complete(...)`
contract. The adapter sends a system prompt plus a JSON user payload containing
the canonical pathogen, aliases, title, and abstract. The classifier does not
need to know the PubMed search bucket; it must judge the abstract itself.

## Notebook stages

### 1. Imports and project paths

The notebook locates the repository by looking for the `assets` directory and
adds the project root to `sys.path`. This allows the cells to import the
reusable PubMed and screening functions from `graphicalizer`.

### 2. Query preview

The preview cell prints one generated query per configured bundle for every pathogen. Inspect this
before collecting results. In particular, check that aliases are unambiguous
and that the evidence terms are neither too narrow nor unintentionally broad.

### 3. Corpus collection

`collect_pubmed_corpus(...)` runs every configured search bundle, records query
provenance, deduplicates overlapping PMIDs, and fetches article metadata and
abstract text. A PMID appearing in multiple bundles produces one corpus row
with all source buckets in its provenance.

The notebook prints corpus size, search failures, and fetch status. A row with
no abstract can be retained for provenance but is not sent to the LLM screen.

### 4. LLM screening

The screening cell requires `OPENAI_API_KEY`. It skips classifications already
completed with the same corpus and prompt configuration hashes. Retryable
failures are retained in `llm_failures.parquet`; terminal provider failures
(for example quota errors) are recorded and the run stops after checkpointing
the current progress.

The normalized decision contains:

- `llm_category`: the model's proposed category, when valid;
- `target_pathogen_supported`: whether the evidence is attributed to the
  target pathogen;
- `animal_infection_supported` and `zoonosis_supported`;
- `evidence_phrases` and `rationale`;
- `confidence` in the normalized range `[0, 1]`;
- `review_required` and `classification_status`;
- model, prompt hash, timestamps, and attempt count.

Responses with an invalid category, missing target attribution, missing
confidence, confidence below `0.75`, or inconsistent evidence flags are marked
for review. They are retained but not counted automatically.

### 5. Category derivation

`derive_category_counts(...)` first identifies the earliest publication year
with a valid, target-attributed, non-review-required zoonosis-positive record
for each pathogen. It then derives categories 1 and 2 from that pathogen-level
timeline and category 3 from the abstract-level zoonosis flag.

Consequently, category 1/category 2 cannot be finalized correctly from an
individual abstract in isolation. They require the complete screened corpus.
The `predates_first_confirmed_zoonosis` field is `True` only when a category 2
record has a known publication year earlier than the first confirmed zoonosis
year.

## Durable outputs

All files below are written under `OUTPUT_DIR` and ignored by Git:

| File | Contents |
| --- | --- |
| `search_runs.parquet` | One row per pathogen/search-bundle/query configuration, including status, result count, and errors. |
| `search_pmids.parquet` | PMID-level search hits with query keys and provenance. |
| `corpus_articles.parquet` | One deduplicated row per `(pathogen, PMID)`, including title, abstract, publication date, fetch status, search buckets, and queries. |
| `llm_screening.parquet` | One normalized LLM decision per `(pathogen, PMID)` and screening hash. |
| `llm_failures.parquet` | Failed or terminal classification rows for inspection/retry. |
| `llm_screening_with_categories.parquet` | Screening rows augmented with final category and timeline fields. |
| `zoonosis_timeline.parquet` | Earliest confirmed zoonosis year per pathogen. |
| `category_counts.parquet` | Total and publication-year counts by pathogen and category. |
| `run_manifest.json` | Configuration hashes, query/result counts, model and prompt hashes, timestamps, paths, failures, and category summaries. |

Parquet is used because it preserves typed tabular data and supports efficient
checkpoint merges. The JSON manifest is the human-readable run summary, not the
source of truth for individual records.

## Resume and reproducibility

With `RESUME=True`, completed search queries are skipped when their query key
and configuration hash match. Existing fetched corpus rows and classifications
are similarly reused only when their hashes match. Changing pathogens, aliases,
term bundles, result limits, page/batch sizes, model, or prompt content creates
a new logical run without silently mixing incompatible records.

For a clean collection run, set `RESUME=False` for the collection cell. To
retry failed LLM rows after fixing credentials or a provider issue, set
`RETRY_FAILED_LLM_ROWS=True` and keep the same output directory/configuration.

Keyboard interrupts preserve the latest checkpoint. A later run can continue
from it. Failed rows remain visible rather than being mistaken for negative
biological evidence.

## Interpretation and limitations

- PubMed keyword retrieval is not exhaustive literature review.
- A keyword hit does not establish that the term refers to the target pathogen.
- The LLM is an evidence-attribution aid, not a replacement for expert review.
- Abstracts may omit important experimental or epidemiological detail.
- Publication dates can be incomplete or represented at different granularities.
- Category 1 is explicitly bounded by the searched corpus, aliases, terms,
  result cap, and publication database state at collection time.
- Counts exclude unresolved and review-required records, so totals may be lower
  than the retrieved corpus size.
- Category 2 is a temporal comparison within the corpus, not proof that the
  earlier abstract established absence of zoonosis at that historical time.

The quality-review cell should be used to inspect sample decisions and all
review-required records before making scientific claims.
