# Notebook documentation

The notebooks are deliberately thin orchestration layers around the reusable
functions in `graphicalizer/`. Run them in the following order for the full
microbiology workflow:

1. [`01_pubmed_abstract_download.md`](01_pubmed_abstract_download.md) — collect
   PubMed candidates and refine target-pathogen attribution.
2. [`01b_pubmed_pathogen_category.md`](01b_pubmed_pathogen_category.md) —
   assign corpus-relative categories and produce the pathogen table.
3. [`02_assemble_microbiology_ontology.md`](02_assemble_microbiology_ontology.md)
   — assemble a compact prompt-facing microbiology ontology.
4. [`03_semantic_abstraction.md`](03_semantic_abstraction.md) — graphicalize
   and inspect one abstract interactively.
5. [`04_batch_semantic_abstraction.md`](04_batch_semantic_abstraction.md) —
   graphicalize every local abstract and persist one graph per source.

The documents below describe the assumptions, configuration variables, inputs,
outputs, external services, and failure modes for each notebook. The canonical
abstract corpus, graphs, and screening checkpoints are local research artifacts
and are ignored by Git.
