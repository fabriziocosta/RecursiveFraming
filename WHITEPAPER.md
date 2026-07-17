# Self-Expanding Semantic Graphs

## A Framework for Recursive Semantic Abstraction Using Interpretation Operators

### White Paper

## 1. Introduction

Scientific discovery is fundamentally a process of abstraction. Researchers begin with observations of individual entities and relationships, identify recurring patterns, and eventually introduce new concepts that capture increasingly complex phenomena. These concepts subsequently become part of the language of science, enabling further discoveries at higher levels of abstraction.

Current artificial intelligence systems perform remarkably well at extracting information from text and making predictions from large datasets. However, they generally assume that the semantic representation is fixed. Knowledge graphs rely on predefined ontologies, language models operate over static vocabularies, and machine learning algorithms learn predictive functions over predetermined feature spaces.

This paper proposes a different paradigm.

Rather than assuming a fixed semantic representation, we present a framework that **learns new semantic concepts through recursive graph abstraction**. Scientific text is transformed into structured graph representations, predictive graph motifs are discovered using graph machine learning, and recurring motifs are promoted into new semantic concepts through learned interpretation operators. These higher-order concepts then become the vocabulary for the next level of representation, allowing the system to progressively build richer semantic models while preserving complete interpretability.

The central idea is that learning does not occur inside the language model. Instead, learning occurs through the recursive construction of increasingly abstract graph representations.

---

## 2. Motivation

Large Language Models possess remarkable linguistic competence, yet adapting them to specialised scientific domains typically requires fine-tuning, reinforcement learning, or extensive prompt engineering.

These approaches suffer from several limitations.

First, they couple semantic knowledge to model parameters, making the resulting systems difficult to interpret.

Second, improvements in language models often require rebuilding downstream pipelines.

Third, domain knowledge becomes embedded inside opaque neural weights rather than explicit semantic structures.

We argue that language understanding and scientific learning should be separated.

The language model should serve only as a **semantic interface** between natural language and structured representations. Domain adaptation, abstraction, and scientific concept formation should instead occur through explicit graph-based learning algorithms operating over symbolic representations.

This separation produces a modular architecture in which language models are interchangeable components rather than repositories of learned scientific knowledge.

---

## 3. Design Principles

The proposed framework is built around five principles.

### Principle 1 — Modular Language Models

The language model is treated as a black box.

Its responsibilities are limited to:

* extracting primitive entities and relationships,
* mapping them to the current ontology,
* translating graph structures back into natural language.

The model itself is never modified.

---

### Principle 2 — Learning Outside the LLM

All adaptation occurs through graph learning.

Graph embeddings, motif discovery, clustering, feature selection and ontology evolution are entirely independent of the underlying language model.

Consequently, newer or cheaper language models can be substituted without changing the learning architecture.

---

### Principle 3 — Explicit Interpretability

Every prediction can be traced back through:

* interpretation operators,
* graph motifs,
* primitive graph structures,
* original text.

Interpretability is therefore intrinsic rather than an afterthought.

---

### Principle 4 — Recursive Abstraction

Concepts are not predefined.

They emerge from recurring predictive graph structures.

Each newly learned concept becomes available for constructing even higher-order concepts.

---

### Principle 5 — Provenance Preservation

Every abstraction maintains links to the graph structures and textual evidence from which it originated.

No semantic information is discarded.

---

## 4. System Architecture

The framework consists of two fundamentally different graph representations.

### 4.1 Base Graph

The Base Graph is the direct semantic representation extracted from text.

Nodes correspond to primitive entities.

Edges correspond to typed relationships.

Each node contains:

* ontology type,
* original textual span,
* contextual description,
* generic semantic description,
* vector embedding.

Importantly, the Base Graph remains stable throughout the learning process.

It is the semantic equivalent of raw observations.

---

### 4.2 Interpretation Graph

The Interpretation Graph is learned.

Its nodes no longer represent primitive entities.

Instead, each node represents the application of an **interpretation operator** to a subgraph of the previous representation.

Consequently, an interpretation graph is an abstraction of another graph rather than another parsing of the original text.

This distinction is fundamental.

The system reasons primarily over interpretation graphs while retaining the Base Graph solely for provenance and traceability.

---

## 5. Learning Interpretation Operators

Interpretation operators are the central mathematical objects of the framework.

Initially, candidate local graph structures are generated mechanically using graphlets, neighbourhoods, paths or other local graph constructions.

These structures become candidate features for supervised graph learning.

Feature selection identifies which graph structures contribute most strongly to prediction.

Importance scores are propagated back onto the graph, producing connected predictive motifs.

These motifs constitute the raw material from which new semantic concepts are constructed.

---

## 6. From Motifs to Concepts

Predictive motifs are clustered according to structural and semantic similarity.

Each cluster represents a family of related graph structures.

Rather than viewing clustering as the end of the learning process, we interpret each cluster as defining a new **interpretation operator**.

The operator recognises instances of that family within previously unseen graphs.

Initially this recognition can be implemented using Large Language Models.

Each representative motif is translated back into natural language.

The language model is then prompted to determine whether a new subgraph belongs to one of the learned concept families.

Thus the LLM acts as a modular semantic recogniser rather than a learned classifier.

The ontology evolves only through the addition of new interpretation operators.

The language model itself remains unchanged.

---

## 7. Recursive Semantic Abstraction

Once interpretation operators have been learned, they generate a new graph representation.

Learning now proceeds on this higher-level graph.

Motif discovery, feature selection and prediction are repeated using concepts rather than primitive entities.

This recursive process produces a hierarchy of semantic representations:

Text

↓

Base Graph

↓

Interpretation Graph 1

↓

Interpretation Graph 2

↓

⋯

Each level represents increasingly abstract semantic structure while maintaining complete traceability to the original observations.

The framework therefore models scientific discovery as repeated semantic abstraction rather than repeated parameter optimisation.

---

## 8. Application: Early Detection of Zoonotic Emergence

To demonstrate the framework, consider the scientific literature describing bacterial pathogens.

Each publication is converted into a Base Graph representing its biological entities, ecological descriptors, experimental observations and causal relationships.

Historical publications provide a temporal sequence spanning periods before and after formal recognition of zoonotic transmission.

A supervised prediction task identifies graph motifs associated with future zoonotic emergence.

These motifs are elevated into higher-order concepts representing recurring biological or ecological mechanisms.

Subsequent iterations operate directly over these learned concepts, allowing increasingly sophisticated representations of pathogen behaviour to emerge.

Beyond prediction, the framework produces explicit semantic explanations describing the mechanisms underlying each prediction.

---

## 9. Advantages

The proposed architecture offers several advantages over conventional approaches.

* Language models remain modular and interchangeable.
* Domain adaptation occurs without fine-tuning.
* Scientific concepts emerge explicitly rather than implicitly.
* Every abstraction is interpretable and traceable.
* The ontology evolves continuously through data-driven learning.
* Improvements in foundation models immediately improve extraction quality without requiring retraining.
* Computational effort is concentrated in graph learning rather than repeated optimisation of very large language models.

---

## 10. Future Directions

Several research questions naturally emerge.

How should interpretation operators be represented mathematically?

How should concept quality be measured?

When should a motif cluster become a semantic concept?

How should recursive abstraction be regularised to prevent semantic drift?

Can multiple scientific domains share interpretation operators?

These questions define a broader research programme centred on recursive semantic abstraction.

---

## 11. Conclusion

This paper proposes a new perspective on machine learning for scientific discovery.

Rather than adapting increasingly larger language models, we argue that intelligence should emerge through the recursive construction of interpretation operators over abstract graphs.

Language models become semantic interfaces.

Graph learning becomes the engine of scientific abstraction.

The ontology becomes a living collection of learned interpretation operators.

Viewed in this way, concept formation is no longer an informal human activity but a computational process that incrementally constructs richer semantic representations while preserving transparency, provenance and interpretability.

The resulting framework provides a principled foundation for machine-assisted scientific discovery in which learning, explanation and abstraction evolve together.
