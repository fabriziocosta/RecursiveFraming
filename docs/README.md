# Notebook documentation

The notebooks are deliberately thin orchestration layers around the reusable
functions in `graphicalizer/`. Run them in the following order for the full
microbiology workflow:

1. [`00_configuration.md`](00_configuration.md) — load the shared YAML configuration.
2. [`01_zoonotic_bacterial_species.md`](01_zoonotic_bacterial_species.md) — discover
   bacterial species associated with zoonosis from PubMed titles and abstracts.
3. [`02_pubmed_abstract_download.md`](02_pubmed_abstract_download.md) — collect
   PubMed candidates and refine target-pathogen attribution.
4. [`03_pubmed_pathogen_category.md`](03_pubmed_pathogen_category.md) —
   assign corpus-relative categories and produce the pathogen table.
5. [`04_assemble_microbiology_ontology.md`](04_assemble_microbiology_ontology.md)
   — assemble a compact prompt-facing microbiology ontology.
6. [`05_semantic_abstraction.md`](05_semantic_abstraction.md) — graphicalize
   and inspect one abstract interactively.
7. [`06_batch_semantic_abstraction.md`](06_batch_semantic_abstraction.md) —
   graphicalize every local abstract and persist one graph per source.

The documents below describe the assumptions, configuration variables, inputs,
outputs, external services, and failure modes for each notebook. The canonical
abstract corpus, graphs, and screening checkpoints are local research artifacts
and are ignored by Git.
