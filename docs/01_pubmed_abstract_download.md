# Notebook 01 — PubMed corpus collection and retrieval refinement

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
pathogen/entity configuration. Category assignment is deliberately deferred to
the second notebook.

The two result sets are deduplicated by `(pathogen, PMID)`. Every candidate
abstract is then passed to a pathogen-aware LLM refinement pass. Only abstracts
with clear target-pathogen attribution, sufficient relevance, confidence at
least `0.75`, and `review_required=False` are retained for the second
notebook. Rejected, ambiguous, and failed decisions remain available for audit
and retry.

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
| `OPENAI_API_KEY` | Yes for the refinement cell | API credential used by `OpenAIChatCompleter`. |
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

The canonical name is used in output rows and later category aggregation. The
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

### LLM refinement

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
the canonical pathogen, aliases, title, and abstract. The refinement decision
must judge target attribution, not merely the presence of a keyword.

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

### 4. LLM retrieval refinement

The refinement cell requires `OPENAI_API_KEY`. It skips decisions already
completed with the same corpus and prompt configuration hashes. Retryable
failures are retained in `refinement_failures.parquet`; terminal provider
failures are recorded and the run stops after checkpointing current progress.

The normalized decision contains `refinement_keep`,
`target_pathogen_supported`, `evidence_relevant`, `evidence_phrases`,
`rationale`, `confidence`, `review_required`, `accepted`, model, prompt
hash, timestamp, and attempt count. Only accepted rows are copied to
`refined_corpus_articles.parquet`; category 1/2/3 logic is handled by notebook
01b.

## Durable outputs

All files below are written under `OUTPUT_DIR` and ignored by Git:

| File | Contents |
| --- | --- |
| `search_runs.parquet` | One row per pathogen/search-bundle/query configuration, including status, result count, and errors. |
| `search_pmids.parquet` | PMID-level search hits with query keys and provenance. |
| `corpus_articles.parquet` | One deduplicated row per `(pathogen, PMID)`, including title, abstract, publication date, fetch status, search buckets, and queries. |
| `llm_refinement.parquet` | One target-pathogen relevance decision per `(pathogen, PMID)` and refinement hash. |
| `refinement_failures.parquet` | Failed, invalid, or terminal refinement rows for inspection/retry. |
| `refined_corpus_articles.parquet` | Accepted corpus rows plus refinement evidence and provenance columns; input to notebook 01b. |
| `run_manifest.json` | Configuration hashes, query/result counts, model and prompt hashes, timestamps, paths, failures, and category summaries. |

Parquet is used because it preserves typed tabular data and supports efficient
checkpoint merges. The JSON manifest is the human-readable run summary, not the
source of truth for individual records.

## Resume and reproducibility

With `RESUME=True`, completed search queries are skipped when their query key
and configuration hash match. Existing fetched corpus rows and refinement decisions
are similarly reused only when their hashes match. Changing pathogens, aliases,
term bundles, result limits, page/batch sizes, model, or prompt content creates
a new logical run without silently mixing incompatible records.

For a clean collection run, set `RESUME=False` for the collection cell. To
retry failed refinement rows after fixing credentials or a provider issue, set
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
- Accepted rows are bounded by the searched corpus, aliases, terms, result cap,
  refinement prompt/model, and publication database state at collection time.
- Rejected, unresolved, and review-required rows remain available for audit but
  are excluded from the refined corpus.

The quality-review cell should be used to inspect sample refinement decisions
and all review-required records before sending the accepted corpus to notebook
01b.
