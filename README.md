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

- assets/abstracts/ — source scientific texts
- assets/ontologies/ — entity and relation ontologies
- assets/prompts/ — versioned prompt templates

Implementation code lives under the graphicalizer package.

The preferred interface is configuration-driven:

    from graphicalizer import GraphicalizerConfig, LLMGraphicalizer, Ontology

    ontology = Ontology.from_yaml(
        "assets/ontologies/entity_ontology_microbiology.yaml"
    )
    config = GraphicalizerConfig(
        model="gpt-4o-mini",
        prompt_template_path="assets/prompts/graphicalizer_prompt_template.yaml",
        prompt_snapshot_path="outputs/prompts/run.yaml",
    )
    graphicalizer = LLMGraphicalizer.from_openai(ontology, config)
    result = graphicalizer.run(text)

## Installation

Install the package in editable mode for local development:

    python -m pip install -e ".[dev]"

For graph rendering through the Python fallback binding:

    python -m pip install -e ".[render]"

Native Graphviz rendering also requires the Graphviz dot executable on the
system path.
