# Notebook documentation

The notebooks are deliberately thin orchestration layers around the reusable
functions in `graphicalizer/`. Run them in the following order for the full
microbiology workflow:

1. [`01_pubmed_abstract_download.md`](01_pubmed_abstract_download.md) — build
   and screen a PubMed pathogen/zoonosis corpus.
2. [`02_assemble_microbiology_ontology.md`](02_assemble_microbiology_ontology.md)
   — assemble a compact prompt-facing microbiology ontology.
3. [`03_semantic_abstraction.md`](03_semantic_abstraction.md) — graphicalize
   and inspect one abstract interactively.
4. [`04_batch_semantic_abstraction.md`](04_batch_semantic_abstraction.md) —
   graphicalize every local abstract and persist one graph per source.

The documents below describe the assumptions, configuration variables, inputs,
outputs, external services, and failure modes for each notebook. Generated
abstracts, graphs, and screening checkpoints are local research artifacts and
are ignored by Git.
