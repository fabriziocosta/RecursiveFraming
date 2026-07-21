# Executive Summary

Scientific knowledge is fundamentally hierarchical. Researchers begin by observing primitive entities and relationships, discover recurring patterns, and progressively introduce new concepts that explain increasingly complex phenomena. Current artificial intelligence systems, however, largely operate with fixed representations or static ontologies, limiting their ability to construct new abstractions from data.

This paper proposes a new computational framework for **recursive semantic abstraction**, in which scientific text is transformed into a hierarchy of increasingly abstract graph representations. Rather than treating language models as systems that must be continuously fine-tuned, we deliberately isolate them as **modular semantic interfaces** whose sole responsibility is to convert between natural language and structured graph representations. All learning occurs outside the language model through graph-based machine learning and recursive concept induction.

The process begins by converting text into a **base graph** whose nodes represent primitive entities and whose edges represent typed relationships. From this graph, standard graph-learning techniques identify predictive subgraphs relevant to a supervised task. These subgraphs are clustered into families that represent recurring semantic structures. Each cluster defines a **new interpretation operator** that maps collections of lower-level graph structures into a higher-level concept.

The collection of learned interpretation operators generates an **interpretation graph**, whose nodes are no longer primitive entities but learned semantic concepts. This interpretation graph then becomes the input for the next iteration of learning, allowing progressively higher levels of abstraction to emerge. The framework therefore constructs a hierarchy of abstract graphs, where each level represents an increasingly expressive semantic description of the underlying data while maintaining complete traceability to the original text.

A key contribution of the framework is the strict separation between **language understanding** and **representation learning**. Because the language model remains unchanged throughout the process, advances in foundation models can be incorporated immediately without retraining the system. Likewise, different language models can be selected according to cost, speed, or capability without altering the underlying learning architecture. Domain adaptation occurs entirely through the evolution of the ontology and the learned interpretation operators rather than through parameter updates to the language model.

The framework is illustrated through the problem of **early prediction of bacterial zoonotic emergence**. Scientific publications are transformed into graph representations, predictive motifs are identified from historical literature, and recurring patterns are elevated into higher-order concepts that may reveal biological, ecological, or epidemiological signals preceding the formal recognition of zoonotic transmission. Beyond improving predictive performance, the resulting concepts provide explicit, human-interpretable explanations of why predictions are made.

More broadly, this work introduces a computational view of scientific discovery in which learning is the recursive construction of interpretation operators over abstract graphs. The resulting system does not simply classify documents or extract facts; it incrementally develops its own hierarchy of semantic concepts while preserving interpretability, modularity, and scientific transparency.

## Prototype layout

Reusable inputs live under the assets directory:

- outputs/pubmed_screening/ — canonical PubMed corpus and screening checkpoints
- assets/ontologies/ — entity and relation ontologies
- assets/prompts/ — versioned prompt templates

Implementation code lives under the graphicalizer package.

The preferred interface is configuration-driven:

    from graphicalizer import GraphicalizerConfig, LLMGraphicalizer, Ontology

    ontology = Ontology.from_yaml(
        "assets/ontologies/entity_ontology_microbiology.yaml"
    )
    config = GraphicalizerConfig(
        provider="openai",
        model="gpt-5-nano",
        prompt_template_path="assets/prompts/graphicalizer_prompt_template.yaml",
        prompt_snapshot_path="outputs/prompts/run.yaml",
        casting_retries=2,
        node_context_retries=2,
    )
    graphicalizer = LLMGraphicalizer.from_provider(ontology, config)
    result = graphicalizer.run(text)

To attach vectors for each node's generated `node_context.summary`, inject an
embedding model explicitly and save the final NetworkX graph as one file:

    from sentence_transformers import SentenceTransformer
    from graphicalizer import NetworkXGraphStore, NodeEmbeddingConfig

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    graphicalizer = LLMGraphicalizer.from_provider(
        ontology,
        config,
        embedding_model=embedding_model,
        embedding_config=NodeEmbeddingConfig(model_id="all-MiniLM-L6-v2"),
    )
    result = graphicalizer.run(text)
    store = NetworkXGraphStore("outputs/graphs")
    store.save(result.graph, "article-001")
    graph = store.load("article-001")

Set `provider="ollama"` and choose a locally installed model such as
`llama3.2` to use Ollama. The Ollama server must be running locally.
Provider-specific capacity settings are passed with `options`; for example,
Ollama accepts `num_ctx` and `num_predict`, while OpenAI accepts
`max_output_tokens`. The pipeline does not truncate the source text or graph
payload before sending it.

The ontology-casting pass validates every returned type against the ontology
keys. By default, two repair attempts are made when the model emits an invalid
casting; set `casting_retries=0` to disable repair calls. The node-context pass
also retries only omitted node IDs by default; set `node_context_retries=0` to
disable that recovery.

## Installation

Install the package in editable mode for local development:

    python -m pip install -e ".[dev]"

Install the provider extra you intend to use:

    python -m pip install -e ".[openai]"
    python -m pip install -e ".[ollama]"

For graph rendering through the Python fallback binding:

    python -m pip install -e ".[render]"

For the SentenceTransformers example above:

    python -m pip install -e ".[embeddings]"

Native Graphviz rendering also requires the Graphviz dot executable on the
system path.

## PubMed abstracts

The [zoonotic bacterial species notebook](notebooks/00_zoonotic_bacterial_species.ipynb)
searches zoonosis-related PubMed records and runs title- and abstract-level LLM
extraction. It writes resumable checkpoints under
`outputs/pubmed_screening/zoonosis_species/` and the one-row-per-species result
to `assets/zoonotic_bacterial_species.parquet`.

The [PubMed collection notebook](notebooks/01_pubmed_abstract_download.ipynb)
uses NCBI E-utilities to build a resumable pathogen/zoonosis corpus, then
refines target-pathogen attribution with an LLM. The
[category notebook](notebooks/02_pubmed_pathogen_category.ipynb) classifies
the accepted corpus and writes pathogen-level category tables. Configure the
pathogen aliases in the notebook and edit the reusable search bundles under
`assets/search_bundles/`; the notebook defaults to `xfcosta@gmail.com` for
`PUBMED_EMAIL` (override it if needed) and requires
`OPENAI_API_KEY` before running it. Parquet checkpoints and the run manifest
are written under `outputs/pubmed_screening/`, which is intentionally ignored
by Git.

Detailed notebook assumptions, parameters, outputs, and troubleshooting notes
are documented in [`docs/`](docs/README.md).
