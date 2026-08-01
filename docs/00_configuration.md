# Notebook 00 — shared configuration

Source notebook: [`notebooks/00_configuration.ipynb`](../notebooks/00_configuration.ipynb)

Notebook 00 is the editable source for the workflow parameters. Its sectioned
Python cells contain the explicit values for notebooks 01 through 06, and its
final cell writes them to [`configs/notebook_config.yaml`](../configs/notebook_config.yaml).
The numbered workflow notebooks read that generated YAML. Credentials remain
environment variables and are never stored in the configuration file.

The sections are organized for notebooks 01 through 06: zoonotic bacterial
species discovery, PubMed corpus collection, pathogen categorization, ontology
assembly, single-abstract semantic abstraction, and batch semantic abstraction.
