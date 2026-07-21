# Notebook 00 — zoonotic bacterial species discovery

Source notebook: [`notebooks/00_zoonotic_bacterial_species.ipynb`](../notebooks/00_zoonotic_bacterial_species.ipynb)

This notebook searches PubMed for records containing zoonosis-related terms and
asks an LLM to extract bacterial species explicitly associated with zoonosis.
It has two extraction levels:

1. titles only, to discover high-precision bacterial candidates;
2. abstracts, by default for title-positive records, to validate and enrich the
   species evidence. Set `ABSTRACT_INPUT_MODE = 'all_articles'` when recall is
   more important than LLM cost.

The extractor is deliberately conservative: it excludes viruses, parasites,
fungi, hosts, diseases, and genus-only names. Each LLM response is stored with
its evidence phrases, rationale, confidence, and review flag.

## Outputs

All intermediate files are under:

    outputs/pubmed_screening/zoonosis_species/

The main checkpoints are:

- `zoonosis_pubmed_search.json` — query and retrieved PMIDs;
- `zoonosis_pubmed_articles.parquet` — fetched titles and abstracts;
- `title_species_extraction.parquet` — title-stage LLM decisions;
- `abstract_species_extraction.parquet` — abstract-stage LLM decisions.

The final one-row-per-species dataframe is persisted to:

    assets/zoonotic_bacterial_species.parquet

Its key columns are `bacterial_species`, `evidence_levels`, `pmids`, and the
title/abstract evidence counts. The recovery cell reads the extraction
checkpoints directly and rebuilds this Parquet, so it remains runnable after a
manual interrupt.

## Configuration

`PUBMED_EMAIL` is required by NCBI and defaults to the project contact used by
the other PubMed notebooks. `OPENAI_API_KEY` is required for the LLM stages.
`PUBMED_MAX_RESULTS`, `LLM_MAX_CALLS`, and `LLM_SAVE_EVERY` can be reduced for
an initial test run. Set `RESUME = True` to reuse completed checkpoints.
