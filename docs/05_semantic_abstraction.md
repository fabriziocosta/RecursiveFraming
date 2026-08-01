# Notebook 05 — Single-abstract semantic abstraction

Source notebook: [`notebooks/05_semantic_abstraction.ipynb`](../notebooks/05_semantic_abstraction.ipynb)

## Purpose

This is the interactive, single-document experiment. It loads one row from the
canonical PubMed corpus Parquet file, extracts a free graph, casts it into the microbiology ontology,
generates grounded node context, attaches embeddings to node summaries, saves
the graph, renders it, samples a connected subgraph, and asks the LLM to write
a short scientific narrative for that subgraph.

Use this notebook to inspect prompt behavior and graph quality before running
the batch workflow in notebook 06. It is not a corpus-wide resumable runner.

## Prerequisites

Install the provider and embedding extras needed by the selected configuration:

```bash
python -m pip install -e ".[openai,embeddings,render]"
```

For OpenAI, set `OPENAI_API_KEY`. For Ollama, run the local Ollama service and
make the selected model available. Graph rendering requires either the native
Graphviz `dot` executable or the project's supported Python fallback binding.

The notebook expects `outputs/pubmed_screening/corpus_articles.parquet`, which
is produced by notebook 02 and ignored by Git. The Parquet file contains one
deduplicated row per `(pathogen, PMID)` and is now the canonical abstract store
for downstream graphicalization.

## Input selection and paths

| Variable | Default | Description |
| --- | --- | --- |
| `PROJECT_ROOT` | Repository root discovered from `assets` | Root used to resolve all relative paths. |
| `ASSETS_ROOT` | `PROJECT_ROOT / "assets"` | Input asset directory. |
| `CORPUS_PATH` | `outputs/pubmed_screening/corpus_articles.parquet` | Canonical PubMed corpus Parquet file. |
| `CORPUS_PATHOGENS` | `None` | Optional list of canonical pathogen names to retain. |
| `CORPUS_START_YEAR` | `None` | Optional inclusive publication-year lower bound. |
| `CORPUS_END_YEAR` | `None` | Optional inclusive publication-year upper bound. |
| `PUBMED_ABSTRACT_INDEX` | `0` | Zero-based index into the filtered corpus rows. |
| `ABSTRACT_REF` | Derived | Stable `pathogen/PMID` reference for the selected row. |
| `ASSEMBLED_ONTOLOGY_PATH` | `assets/ontologies/entity_ontology_microbiology_assembled.yaml` | Preferred ontology. |
| `ONTOLOGY_PATH` | Base ontology fallback | Uses `entity_ontology_microbiology.yaml` when the assembled ontology is absent. |
| `PROMPT_PATH` | `assets/prompts/graphicalizer_prompt_template.yaml` | Graphicalizer prompt template. |
| `PROMPT_SNAPSHOT_PATH` | `outputs/prompts/ontology-aware-graphicalizer-0.4.0.yaml` | Runtime prompt snapshot for provenance. |
| `GRAPH_OUTPUT_PATH` | `outputs/selected_ontology_graph.svg` | Rendered full-graph SVG. |

The selected abstract is based on the deterministic order returned by
`load_corpus_articles`, not on filesystem paths. Changing filters or adding
new corpus rows can change the index. For a stable experiment, record the
corpus path, filters, selected pathogen/PMID, and source SHA-256 printed by the
notebook.

## LLM and embedding parameters

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `"openai"` | Provider adapter: `"openai"` or `"ollama"`. |
| `LLM_MODEL` | `gpt-4o-mini` for OpenAI; `gemma4:12b-mlx` for Ollama | Model used for extraction, ontology casting, and node context. |
| `LLM_OPTIONS` | Provider-specific mapping | OpenAI uses `max_output_tokens`; Ollama uses `num_ctx` and `num_predict`. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | SentenceTransformers model used for node-context summaries. |
| `NARRATIVE_COLUMNS` | `80` | Display wrapping width only; it does not change model input. |

The embedding model is loaded before graphicalization and is injected through
`NodeEmbeddingConfig(normalize=True, model_id=...)`. Embeddings are attached to
the final graph's node context metadata. Loading the model may download model
weights on the first run.

## Graphicalizer controls

The notebook creates:

```python
ExtractionDensityConfig(
    entities_per_word=0.05,
    relations_per_entity=1.5,
    minimum_entity_fraction=0.75,
    density_retries=2,
)
```

These values are effort targets, not guarantees. `entities_per_word` requests
approximately one entity per twenty words; `relations_per_entity` requests
roughly 1.5 relations per extracted entity. If the first extraction is too
small, `minimum_entity_fraction` and `density_retries` govern grounded retry
expansion.

The node-context configuration is:

| Setting | Value | Meaning |
| --- | ---: | --- |
| `max_sentences` | `3` | Maximum summary sentence count. |
| `max_evidence_items` | `2` | Maximum evidence items retained per node. |
| `include_evidence` | `True` | Include source-grounded evidence. |
| `include_uncertainty` | `True` | Preserve uncertainty notes. |

The `GraphicalizerConfig` uses `disconnected_policy="largest_component"`,
`context_policy="all_nodes"`, and `casting_retries=2`. The disconnected policy
means disconnected extracted material is reduced to the largest component for
the final graph. Set the policy explicitly in a separate experiment if that
discarding behavior is not appropriate.

## Processing stages and validation

`graphicalizer.run(abstract_text)` performs the complete pipeline:

1. Extract free-form entities and relations from the abstract.
2. Validate and normalize the extraction.
3. Cast entity and relation types into the selected ontology.
4. Generate grounded context for final graph nodes.
5. Build the typed NetworkX graph and attach provenance/embeddings.

The diagnostics cell prints target counts, source hash, raw/normalized/final
validation reports, normalization details, relation count, node-context count,
and embedding metadata. Treat validation output as part of the result, not as
incidental logging. A successful model response can still be normalized or
repaired before it becomes the final graph.

## Persisting and loading the graph

The graph is saved through `NetworkXGraphStore` in `outputs/graphs` with ID
`pubmed_<pathogen>_<PMID>`, for example `pubmed_Coxiella_burnetii_<PMID>.gpickle`.
The store writes via a
temporary file and replacement. Graph files use Python pickle because node
attributes include mappings, provenance, and vectors; only load graph files
from trusted locations.

The full rendered graph is written to `GRAPH_OUTPUT_PATH`. The notebook also
prints the DOT representation and reloads the persisted graph to verify its
node and edge counts.

## Subgraph sampling and narrative

| Variable | Default | Description |
| --- | ---: | --- |
| `SUBGRAPH_NODE_COUNT` | `3` | Number of nodes requested in the connected sample. |
| `SUBGRAPH_SEED` | `None` | Random seed. Set an integer for reproducible sampling. |
| `SUBGRAPH_OUTPUT_PATH` | `outputs/selected_random_subgraph.svg` | Rendered subgraph SVG. |
| `NARRATIVE_WORDS` | `100` | Approximate target length for the narrative. |
| `NARRATIVE_PROMPT_PATH` | `assets/prompts/subgraph_narrative_prompt_template.yaml` | Narrative prompt template. |
| `NARRATIVE_SNAPSHOT_PATH` | `outputs/prompts/subgraph-narrator-0.1.0.yaml` | Rendered prompt snapshot. |

Sampling uses an undirected connectivity view but preserves the original graph
direction and attributes in the returned subgraph. The requested count must
fit within the selected component. The narrative prompt is instructed to use
only supplied graph and ontology evidence and to state uncertainty when the
subgraph is ambiguous.

## Assumptions and limitations

- Corpus rows contain UTF-8-compatible abstract text and enough scientific
  context for extraction.
- The ontology identifiers and prompt template are compatible.
- LLM output is probabilistic; record model, prompt snapshot, source hash, and
  configuration when comparing runs.
- Embedding similarity is based on generated node summaries, not raw abstracts.
- A graph is an interpretation of the text, not a database of independently
  verified facts.
- The largest-component policy can remove disconnected facts or caveats.
- SVG rendering is a visualization aid and is not the persisted graph source.

## Common failures

- **No corpus rows found:** run notebook 02 to create
  `outputs/pubmed_screening/corpus_articles.parquet`, then check the corpus
  filters and current working directory.
- **Missing assembled ontology:** run notebook 04, or allow the documented base
  ontology fallback.
- **Provider authentication failure:** set `OPENAI_API_KEY` or switch to a
  running local Ollama provider.
- **Embedding import/model failure:** install the embeddings extra and verify
  model-cache/network access.
- **Graphviz rendering failure:** install Graphviz or use the supported Python
  rendering fallback.
- **Validation or node-context errors:** inspect the printed reports and prompt
  snapshots; lowering model output pressure or using a more capable model can
  help, but any repaired result should be reviewed.
