# Notebook 03 — Assemble a filtered microbiology ontology

Source notebook: [`notebooks/03_assemble_microbiology_ontology.ipynb`](../notebooks/03_assemble_microbiology_ontology.ipynb)

## Purpose

This notebook creates a compact, prompt-compatible microbiology ontology for
the graphicalization notebooks. It combines the repository's generic relation
seed with the base microbiology ontology and selected terms from external
ontology resources. External resources are downloaded, filtered by domain
patterns, ranked by hierarchy and resource priority, and bounded by a global
term budget.

Run this notebook before notebook 03 when the assembled ontology is desired.
Notebook 04 automatically falls back to the checked-in base ontology if the
assembled file does not exist.

## Inputs and external resources

The assembly function reads the base files under:

```text
assets/ontologies/entity_ontology_generic.yaml
assets/ontologies/entity_ontology_microbiology.yaml
```

It downloads enabled resources defined in
`graphicalizer.ontology_assembly.RESOURCE_SPECS`, currently including IDO,
PHIPO, ENVO, GenEpiO, and OMP. NCBI Taxonomy is disabled by default and can be
enabled with `INCLUDE_NCBITAXON=True`.

Network access is therefore required on the first run. Downloads are cached in
`assets/ontologies/external/`; an existing cached file is reused without a new
download. The resource URLs, local paths, byte hashes, and available-term
counts are stored in the assembled ontology metadata and term catalog.

## Configuration parameters

| Variable | Default in notebook | Description |
| --- | ---: | --- |
| `TERM_BUDGET` | `500` | Maximum number of retained external terms across all enabled resources. The generic/base ontology entries are preserved separately. |
| `ABSTRACTION_LEVEL` | `2` | Controls term specificity: `0`/`"specific"` favors leaf concepts, `1`/`"balanced"` is intermediate, and `2`/`"broad"` favors broader concepts and useful ancestors. |
| `INCLUDE_NCBITAXON` | `False` | Whether to enable the NCBI Taxonomy resource. Keep disabled unless taxonomic terms are needed and the additional download/term selection is acceptable. |

`TERM_BUDGET` must be positive. The budget is global across external sources,
not a separate allowance per source. A larger budget provides more references
to the LLM but increases ontology size and prompt length.

## Execution stages

### 1. Project path discovery

The notebook searches the current directory and its parent for `assets`. This
supports running Jupyter from either the repository root or `notebooks/`.

### 2. Assembly

`assemble_microbiology_ontology(PROJECT_ROOT, ...)` performs these operations:

1. Resolve enabled resource specifications.
2. Download or reuse each raw OBO/OWL resource.
3. Parse terms and retain terms matching the configured microbiology patterns.
4. Rank matching terms using resource priority, hierarchy, and abstraction
   level.
5. Allocate the bounded external term catalog.
6. Merge generic and microbiology entity/relation definitions.
7. Attach selected external reference terms to compatible entity types.
8. Write the assembled ontology and machine-readable catalog.

The function returns an `OntologyAssemblyResult` with the assembled ontology
path, catalog path, raw resource paths, selected-term count, and normalized
abstraction level.

### 3. Prompt-facing ontology preview

The final cell loads the assembled YAML and prints only `entity_types` and
`relation_types`. This is the portion used to constrain ontology casting in
the graphicalizer prompt. The full assembled YAML also contains ontology
metadata, assembly provenance, and the selected external term catalog.

## Outputs

The notebook writes or updates:

| Path | Contents |
| --- | --- |
| `assets/ontologies/entity_ontology_microbiology_assembled.yaml` | Prompt-compatible assembled ontology, metadata, and selected external terms. |
| `assets/ontologies/microbiology_external_terms.json` | Selected term catalog plus source metadata and hashes. |
| `assets/ontologies/external/*` | Cached raw ontology downloads. |

These outputs live under `assets`, so they should be treated as project data
and reviewed before committing. If external ontology files are intentionally
local-only in a deployment, add the relevant paths to `.gitignore` rather than
assuming that the notebook will do so.

## Assumptions

- External ontology files are trusted and parseable OBO or OWL documents.
- The configured regular-expression patterns are an approximate relevance
  filter, not a formal semantic mapping.
- A bounded term catalog is preferable to including every term in an external
  ontology.
- Existing cached downloads correspond to the resource URL/version expected by
  the project. The catalog records a SHA-256 hash so changes can be audited.
- The base ontology remains the authority for entity and relation identifiers;
  external terms act as references and vocabulary support.
- A broader abstraction level may improve generalization but can remove useful
  pathogen-specific detail from the prompt-facing catalog.

## Choosing an abstraction level

Use `0` (`specific`) when the extraction task needs fine-grained terms and the
prompt budget is generous. Use `1` (`balanced`) for a compromise. Use `2`
(`broad`) when the graph should emphasize higher-level concepts and ancestors.
The notebook currently selects `2`, matching the goal of exploring reusable
microbiology and zoonosis abstractions.

## Failure modes and troubleshooting

- **Download failure:** check network access, upstream URL availability, and
  whether a stale partial file exists in `assets/ontologies/external/`.
- **Missing PyYAML:** install the base project dependencies or the development
  environment before running the preview cell.
- **Too few selected terms:** increase `TERM_BUDGET`, broaden the resource
  patterns, or choose a different abstraction level.
- **Unexpected prompt vocabulary:** inspect
  `microbiology_external_terms.json` and the printed prompt-facing mapping;
  the external catalog is selected by patterns and ranking, not manually
  curated per run.
- **Ontology changes after upstream updates:** compare the source metadata and
  hashes recorded in the assembled YAML/catalog before comparing downstream
  graph results.
