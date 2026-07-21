# Notebook documentation

The notebooks are deliberately thin orchestration layers around the reusable
functions in `graphicalizer/`. Run them in the following order for the full
microbiology workflow:

1. [`00_zoonotic_bacterial_species.md`](00_zoonotic_bacterial_species.md) — discover
   bacterial species associated with zoonosis from PubMed titles and abstracts.
2. [`01_pubmed_abstract_download.md`](01_pubmed_abstract_download.md) — collect
   PubMed candidates and refine target-pathogen attribution.
3. [`02_pubmed_pathogen_category.md`](02_pubmed_pathogen_category.md) —
   assign corpus-relative categories and produce the pathogen table.
4. [`03_assemble_microbiology_ontology.md`](03_assemble_microbiology_ontology.md)
   — assemble a compact prompt-facing microbiology ontology.
5. [`04_semantic_abstraction.md`](04_semantic_abstraction.md) — graphicalize
   and inspect one abstract interactively.
6. [`05_batch_semantic_abstraction.md`](05_batch_semantic_abstraction.md) —
   graphicalize every local abstract and persist one graph per source.

The documents below describe the assumptions, configuration variables, inputs,
outputs, external services, and failure modes for each notebook. The canonical
abstract corpus, graphs, and screening checkpoints are local research artifacts
and are ignored by Git.
