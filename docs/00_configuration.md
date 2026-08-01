# Notebook 00 — shared configuration

Source notebook: [`notebooks/00_configuration.ipynb`](../notebooks/00_configuration.ipynb)

The numbered workflow notebooks read their parameters from
[`configs/notebook_config.yaml`](../configs/notebook_config.yaml). Notebook 00
loads and validates the YAML, then displays its sections. Edit the YAML before
running the workflow; credentials remain environment variables and are never
stored in the configuration file.

The sections are organized for notebooks 01 through 06: zoonotic bacterial
species discovery, PubMed corpus collection, pathogen categorization, ontology
assembly, single-abstract semantic abstraction, and batch semantic abstraction.
