"""Load the shared YAML configuration used by the research notebooks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG_RELATIVE_PATH = Path("configs") / "notebook_config.yaml"

__all__ = [
    "CONFIG_RELATIVE_PATH",
    "configured_env",
    "debug_pathogens",
    "find_project_root",
    "limit_debug_abstracts",
    "load_notebook_config",
    "resolve_config_path",
]


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a notebook or script working directory."""
    candidate = Path(start or Path.cwd()).resolve()
    for directory in (candidate, *candidate.parents):
        if (directory / "assets").is_dir() and (directory / "graphicalizer").is_dir():
            return directory
    raise FileNotFoundError("Could not locate the repository root.")


def load_notebook_config(project_root: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the shared notebook YAML configuration."""
    root = find_project_root(project_root)
    path = root / CONFIG_RELATIVE_PATH
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Notebook configuration must be a YAML mapping: {path}")
    if "common" not in config:
        raise ValueError(f"Notebook configuration is missing the common section: {path}")
    return config


def configured_env(config: Mapping[str, Any], key: str, default: str = "") -> str:
    """Read an environment variable named by a common config entry."""
    variable_name = str(config.get(key, ""))
    return os.environ.get(variable_name, default) if variable_name else default


def debug_pathogens(common: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return the explicitly configured debug pathogens and aliases."""
    raw = common.get("debug_pathogens", {})
    return {
        str(pathogen): [str(alias) for alias in aliases]
        for pathogen, aliases in raw.items()
    }


def limit_debug_abstracts(frame: Any, common: Mapping[str, Any]) -> Any:
    """Limit a corpus to the configured debug sample, grouped by pathogen."""
    if not bool(common.get("debug_mode", False)):
        return frame
    per_pathogen = int(common.get("debug_abstracts_per_pathogen", 5))
    if per_pathogen < 1:
        raise ValueError("debug_abstracts_per_pathogen must be positive.")
    if "pathogen" not in frame.columns:
        return frame.head(per_pathogen * len(debug_pathogens(common))).reset_index(drop=True)
    return (
        frame.sort_values(["pathogen", "pmid"], kind="stable")
        .groupby("pathogen", sort=False, group_keys=False)
        .head(per_pathogen)
        .reset_index(drop=True)
    )


def resolve_config_path(project_root: str | Path, relative_path: str | Path) -> Path:
    """Resolve a repository-relative path stored in YAML."""
    return Path(project_root) / Path(relative_path)
