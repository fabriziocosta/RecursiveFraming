# Notebook 02 — PubMed pathogen category assignment

Source notebook: notebooks/02_pubmed_pathogen_category.ipynb

## Purpose

This notebook is the second stage of PubMed processing. It reads only the
accepted rows written by notebook 01, asks the LLM to classify abstract-level
animal-infection and zoonosis evidence, and derives the corpus-relative
categories:

| Category | Meaning |
| --- | --- |
| 1 | Animal infection is confirmed, but no reviewable abstract in the searched corpus confirms zoonosis for that pathogen. |
| 2 | Animal infection is confirmed in this abstract, while another reviewable abstract for the pathogen confirms zoonosis elsewhere in the corpus. |
| 3 | The abstract explicitly attributes zoonosis, spillover, human infection, or human disease to the target pathogen. |

Category 1 means “no zoonosis evidence in the searched corpus.” It is not
proof that the pathogen has never undergone zoonosis.

## Input and configuration

Notebook 01 must be run first. Its output is:

    outputs/pubmed_screening/refined_corpus_articles.parquet

This file contains only rows accepted by the target-pathogen refinement pass.
The category notebook does not re-run keyword retrieval or relevance
refinement.

| Variable | Default | Description |
| --- | --- | --- |
| REFINED_CORPUS_PATH | outputs/pubmed_screening/refined_corpus_articles.parquet | Accepted corpus input. |
| OUTPUT_DIR | outputs/pubmed_screening | Shared checkpoint/output directory. |
| CORPUS_PATHOGENS | None | Optional pathogen filter for a focused run. |
| CORPUS_START_YEAR | None | Optional inclusive publication-year lower bound. |
| CORPUS_END_YEAR | None | Optional inclusive publication-year upper bound. |
| PATHOGENS | Coxiella burnetii aliases | Canonical names and aliases used by the classifier prompt. They must cover the pathogens in the refined corpus. |
| OPENAI_MODEL | gpt-4o-mini | Chat-completion model. |
| LLM_MAX_TOKENS | 1024 | Completion token limit. |
| LLM_RETRIES | 3 | Retry count for retryable errors or malformed output. |
| LLM_MAX_CALLS | None | Optional call cap for smoke tests/staged runs. |
| LLM_SAVE_EVERY | 10 | Checkpoint merge interval. |
| RESUME | True | Reuse completed category decisions with matching hashes. |
| RETRY_FAILED_LLM_ROWS | False | Retry previously failed category decisions when enabled. |

OPENAI_API_KEY is required for the classification cell. The PubMed API key is
not needed because PubMed retrieval has already completed in notebook 01.

## Category semantics and derivation

The LLM returns abstract-level fields:

- llm_category
- target_pathogen_supported
- animal_infection_supported
- zoonosis_supported
- evidence_phrases
- rationale
- confidence
- review_required

The final category is not accepted directly from the model for categories 1
and 2. derive_category_counts first finds the earliest valid,
target-attributed, non-review-required zoonosis-positive publication year for
each pathogen. It then derives categories 1 and 2 using that pathogen-level
timeline and category 3 using the abstract-level zoonosis flag.

Rows with failed classification, invalid output, review_required=True, or
missing target attribution are retained in the screening checkpoint but
excluded from final counts. The predates_first_confirmed_zoonosis field is true
for category 2 rows whose known publication year precedes the earliest
confirmed zoonosis year.

## Pathogen-level output table

The main summary is pathogen_category_table.parquet, with one row per pathogen
and these columns:

| Column | Description |
| --- | --- |
| pathogen | Canonical pathogen name. |
| category_1_count | Count of valid final category 1 abstracts. |
| category_2_count | Count of valid final category 2 abstracts. |
| category_3_count | Count of valid final category 3 abstracts. |
| total_counted | Sum of the three category counts. |
| first_confirmed_zoonosis_year | Earliest valid zoonosis-positive publication year in the screened corpus. |

This table is the direct association between pathogen and category requested
by the workflow. It is corpus-relative and should be accompanied by the
search/refinement configuration and review counts.

## Durable outputs

All outputs are local and ignored by Git:

| File | Contents |
| --- | --- |
| llm_screening.parquet | Resumable abstract-level category decisions. |
| llm_failures.parquet | Failed or terminal category decisions. |
| llm_screening_with_categories.parquet | Category decisions with derived fields. |
| zoonosis_timeline.parquet | Earliest confirmed zoonosis year by pathogen. |
| category_counts.parquet | Total and yearly category counts. |
| pathogen_category_table.parquet | One pathogen-level row with category counts and first-zoonosis year. |
| run_manifest.json | Collection, refinement, and category-stage metadata and hashes. |

## Resume behavior

Category rows are skipped when the corpus, model, prompt, and token
configuration hashes match an existing completed decision. Changing the
refined corpus, aliases, model, prompt, or token limit creates a new logical
screening hash. Failed rows can be retried explicitly with
RETRY_FAILED_LLM_ROWS=True.

## Limitations

- The category is an abstract-level evidence label, not a clinical or
  biological ground truth.
- Category 1 is bounded by the selected PubMed corpus and its
  retrieval/refinement configuration.
- Publication dates can be incomplete or represented at different granularities.
- Counts exclude unresolved and review-required rows.
- Expert review remains necessary for ambiguous, high-impact, or
  publication-level conclusions.
