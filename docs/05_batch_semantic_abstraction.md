# Notebook 05 — Batch semantic abstraction

Source notebook: [`notebooks/05_batch_semantic_abstraction.ipynb`](../notebooks/05_batch_semantic_abstraction.ipynb)

## Purpose

This notebook applies the graphicalizer to filtered rows in the canonical
`corpus_articles.parquet` DataFrame and saves one NetworkX graph per abstract.
The loop is implemented in `graphicalizer.batch.process_corpus_articles`; the
notebook configures the provider, ontology, embeddings, and corpus filters,
then prints a run summary.

Use notebook 03 first when tuning prompts or inspecting a single graph. Use
this notebook after those choices are acceptable for the full local corpus.

## Prerequisites

Install the provider, embedding, and rendering dependencies:

```bash
python -m pip install -e ".[openai,embeddings,render]"
```

The import cell checks for `sentence_transformers` and, if missing, installs
the editable `embeddings` extra into the active notebook kernel. Automatic
installation requires an environment with package-install permissions and
network access; pre-installing the extra is more reproducible.

For the OpenAI provider set `OPENAI_API_KEY`. For Ollama, start the local
service and install the selected model. Graph rendering is not invoked by this
notebook, but the graphicalizer dependencies still need to be installed.

## Inputs and configuration

| Variable | Default | Description |
| --- | --- | --- |
| `ASSETS_ROOT` | `PROJECT_ROOT / "assets"` | Root for ontology and prompt assets. |
| `CORPUS_PATH` | `outputs/pubmed_screening/corpus_articles.parquet` | Canonical PubMed corpus Parquet file. |
| `CORPUS_PATHOGENS` | `None` | Optional list of canonical pathogen names to retain. |
| `CORPUS_START_YEAR` | `None` | Optional inclusive publication-year lower bound. |
| `CORPUS_END_YEAR` | `None` | Optional inclusive publication-year upper bound. |
| `GRAPH_FOLDER` | `outputs/graphs` | Destination for serialized graph files and the manifest. |
| `GRAPH_ID_PREFIX` | `pubmed` | Prefix used in generated graph IDs. |
| `CONTINUE_ON_ERROR` | `True` | Continue processing later files after one file fails. |
| `ONTOLOGY_PATH` | Assembled ontology, then base fallback | Ontology used for casting. |
| `LLM_PROVIDER` | `"openai"` | `"openai"` or `"ollama"`. |
| `LLM_MODEL` | Provider-dependent | Model used for all graphicalization passes. |
| `LLM_OPTIONS` | Provider-dependent | Capacity settings passed to the provider adapter. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | SentenceTransformers model for node summaries. |

The batch density configuration is fixed in the notebook at:

```python
entities_per_word=0.05
relations_per_entity=1.5
minimum_entity_fraction=0.75
density_retries=2
```

The node-context configuration uses the library defaults. `context_policy` is
set to `"all_nodes"`, so the run attempts to generate context for every final
graph node.

`load_corpus_articles(...)` applies the pathogen/year filters before graph
processing and retains only rows with a non-empty abstract and successful
PubMed fetch status. This keeps selection separate from graphicalization and
allows the same corpus to support many filtered runs.

## File ordering and graph IDs

Rows are sorted deterministically by pathogen, publication year, and PMID by
`load_corpus_articles`. For the row at index `i`, the graph ID is:

```text
{GRAPH_ID_PREFIX}_{i:04d}_{pathogen}_{pmid}
```

For example, a *Coxiella burnetii* row may become
`pubmed_0000_Coxiella_burnetii_<PMID>`. Graph IDs depend on the filtered corpus contents and
ordering. Adding rows or changing filters can change the numeric prefix of
later records; preserve the manifest if IDs are used downstream.

## Batch behavior

`process_corpus_articles(...)` performs the following for each filtered row:

1. Read the `abstract` value from the DataFrame row.
2. Compute the source SHA-256 hash.
3. Run `graphicalizer.run(text)`.
4. Add batch index, graph ID, corpus path, pathogen, PMID, publication date,
   title, and source hash to graph metadata.
5. Save the graph as a `.gpickle` file through `NetworkXGraphStore`.
6. Record status, output path, node count, edge count, or failure details.

The function processes rows in the order returned by
`load_corpus_articles`. With
`CONTINUE_ON_ERROR=True`, one bad abstract does not prevent later abstracts
from running. With `False`, the function writes the current manifest and then
raises at the first error.

This is a batch loop, not a content-hash resume system. On rerun, matching
files are processed again and their graph paths are replaced. The writes are
atomic, but the LLM calls are repeated. If resumability is needed, add a
separate driver that compares source hashes and configuration/prompt hashes
before invoking the batch function.

## Outputs

| Path | Contents |
| --- | --- |
| `outputs/graphs/<graph_id>.gpickle` | One serialized NetworkX graph per successful abstract. |
| `outputs/graphs/manifest.json` | Corpus path, graph folder, discovered/processed/failed counts, and per-row records. |

Each successful manifest record includes `source_sha256`, `graph_path`,
`node_count`, and `edge_count`. Each failure includes `source_path`, `graph_id`,
`error_type`, and `error`. The graph directory and persisted graphs are ignored
by Git.

The notebook prints:

- number of discovered files;
- number of successfully processed files;
- number of failures;
- manifest path; and
- each failure record when present.

## Assumptions and limitations

- Every selected corpus row contains one abstract/document.
- The corpus Parquet file is local and intentionally outside version control.
- Corpus ordering and PMIDs are stable enough for deterministic IDs within a run.
- Every graph can fit within the provider and model context limits; the code
  does not truncate source text or graph payloads.
- The generated graph is a model-assisted interpretation and should be
  validated before scientific use.
- Embedding model identity is recorded in graph metadata, but changing the
  embedding model makes vector comparisons across runs invalid.
- Pickle files must only be loaded from trusted locations.

## Common failures and recovery

- **No files processed:** check `CORPUS_PATH` and the pathogen/year filters.
- **One or more LLM failures:** inspect the manifest, fix credentials/model or
  prompt issues, and rerun. The batch function will attempt all files again.
- **`LLMResponseError` or missing node IDs:** inspect the prompt snapshot and
  model output capacity. The manifest preserves the failing source and error.
- **Out-of-memory during embeddings:** use a smaller embedding model or process
  a smaller input folder.
- **Provider rate limiting:** use provider-specific throttling/options or split
  the input folder into smaller runs.
- **Corrupt graph file:** rerun the corresponding corpus row;
  `NetworkXGraphStore.save` writes
  through a temporary file and replaces the target only after serialization
  completes.

## Relation to notebook 03

Notebook 04 is the diagnostic path: it exposes validation reports, renders the
full graph, samples a connected subgraph, and generates a narrative. Notebook
04 intentionally omits those interactive displays and focuses on durable
per-abstract graph files plus a manifest. Use the same ontology, provider,
model, prompt templates, density settings, and embedding model when comparing
single-document and batch results.
