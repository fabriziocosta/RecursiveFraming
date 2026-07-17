"""Download, filter, and assemble a compact microbiology ontology catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET


USER_AGENT = "RecursiveFraming ontology assembler/0.1"

RESOURCE_SPECS: dict[str, dict[str, Any]] = {
    "ido": {
        "url": "https://raw.githubusercontent.com/infectious-disease-ontology/infectious-disease-ontology/master/ido.owl",
        "format": "owl",
        "priority": 5,
        "patterns": [
            r"\binfectious disease\b",
            r"\binfection\b",
            r"\bpathogen\b",
            r"\bhost\b",
            r"\btransmission\b",
            r"\bexposure\b",
            r"\boutbreak\b",
            r"\bsurveillance\b",
            r"\breservoir\b",
        ],
    },
    "phipo": {
        "url": "https://purl.obolibrary.org/obo/phipo.obo",
        "format": "obo",
        "priority": 4,
        "patterns": [
            r"\bpathogen\b",
            r"\bhost\b",
            r"\bvirulence\b",
            r"\battenuat",
            r"\binfection\b",
            r"\bsusceptib",
        ],
    },
    "envo": {
        "url": "https://purl.obolibrary.org/obo/envo.obo",
        "format": "obo",
        "priority": 3,
        "patterns": [
            r"\benvironment",
            r"\bhabitat\b",
            r"\bwater\b",
            r"\bsoil\b",
            r"\bfood\b",
            r"\bplant\b",
            r"\bsap\b",
            r"\bsurface\b",
            r"\breservoir\b",
            r"\bsample\b",
        ],
    },
    "genepio": {
        "url": "https://raw.githubusercontent.com/GenEpiO/genepio/master/genepio.obo",
        "format": "obo",
        "priority": 4,
        "patterns": [
            r"\boutbreak\b",
            r"\bcase\b",
            r"\bspecimen\b",
            r"\bisolate\b",
            r"\bpathogen\b",
            r"\btransmission\b",
            r"\bfood\b",
            r"\bgenom",
            r"\bsequence\b",
            r"\bassay\b",
        ],
    },
    "omp": {
        "url": "https://purl.obolibrary.org/obo/omp.obo",
        "format": "obo",
        "priority": 3,
        "patterns": [
            r"\bvirulence\b",
            r"\bpathogen\b",
            r"\bhost\b",
            r"\bgrowth\b",
            r"\btemperature\b",
            r"\bpH\b",
            r"\bsalinity\b",
            r"\boxygen\b",
            r"\bresistance\b",
            r"\bmorphology\b",
            r"\binfection\b",
            r"\bbiofilm\b",
            r"\bmotility\b",
        ],
    },
    "ncbitaxon": {
        "url": "https://purl.obolibrary.org/obo/ncbitaxon.obo",
        "format": "obo",
        "priority": 5,
        "enabled": False,
        "patterns": [
            r"\bnipah virus\b",
            r"\bpteropus\b",
            r"\bhomo sapiens\b",
            r"\bparamyxoviridae\b",
        ],
    },
}

PROMPT_REFERENCE_PATTERNS: dict[str, list[str]] = {
    "pathogen": [r"\bpathogen\b", r"\binfectious agent\b", r"\bvirus\b"],
    "host": [r"\bhost\b", r"\borganism\b"],
    "reservoir_host": [r"\breservoir\b", r"\bmaintenance host\b"],
    "exposure": [r"\bexposure\b", r"\bcontact event\b"],
    "transmission_route": [r"\btransmission\b", r"\broute\b"],
    "outbreak": [r"\boutbreak\b", r"\bcase cluster\b"],
    "specimen": [r"\bspecimen\b", r"\bsample\b"],
    "assay": [r"\bassay\b", r"\blaboratory test\b"],
    "genomic_sequence": [r"\bsequence\b", r"\bgenome\b"],
    "geographic_location": [r"\bgeographic\b", r"\blocation\b", r"\bregion\b"],
}


@dataclass(frozen=True)
class OntologyAssemblyResult:
    """Paths and counts produced by :func:`assemble_microbiology_ontology`."""

    ontology_path: Path
    catalog_path: Path
    raw_paths: tuple[Path, ...]
    selected_terms: int
    abstraction_level: int


def _download(path: Path, url: str) -> Path:
    if path.exists():
        return path
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.read())
    return path


def _parse_obo(path: Path) -> dict[str, dict[str, Any]]:
    terms: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    in_term = False
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line == "[Term]":
            current = {}
            in_term = True
            continue
        if line.startswith("["):
            current = None
            in_term = False
            continue
        if not line or line.startswith("!") or current is None or not in_term:
            continue
        key, separator, value = line.partition(": ")
        if not separator:
            continue
        if key in {"id", "name", "def", "is_a"}:
            current[key] = value
        elif key == "synonym":
            current.setdefault("synonym", []).append(value)
        elif key == "is_obsolete" and value == "true":
            current["is_obsolete"] = True
        if key == "id":
            terms[value] = current
    return terms


def _parse_owl(path: Path) -> dict[str, dict[str, Any]]:
    rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    root = ET.fromstring(path.read_bytes())
    terms: dict[str, dict[str, Any]] = {}
    for element in root.iter():
        if not element.tag.endswith("Class"):
            continue
        iri = element.attrib.get(f"{{{rdf}}}about") or element.attrib.get(f"{{{rdf}}}ID")
        if not iri:
            continue
        term_id = iri.rsplit("/obo/", 1)[-1].rsplit("#", 1)[-1]
        term: dict[str, Any] = {"id": term_id, "synonym": []}
        for child in element:
            local_name = child.tag.rsplit("}", 1)[-1]
            value = (child.text or "").strip()
            if local_name == "subClassOf":
                parent_iri = child.attrib.get(f"{{{rdf}}}resource")
                if parent_iri:
                    term["is_a"] = parent_iri.rsplit("/obo/", 1)[-1].rsplit("#", 1)[-1]
            elif local_name == "label" and value:
                term["name"] = value
            elif local_name in {"IAO_0000115", "definition"} and value:
                term["def"] = value
            elif "Synonym" in local_name and value:
                term["synonym"].append(value)
            elif local_name in {"deprecated", "is_obsolete"} and value.lower() == "true":
                term["is_obsolete"] = True
        if term.get("name"):
            terms[term_id] = term
    return terms


def _parse_resource(path: Path, resource_format: str) -> dict[str, dict[str, Any]]:
    return _parse_owl(path) if resource_format == "owl" else _parse_obo(path)


def _clean_quoted(value: str) -> str:
    match = re.match(r'"(.*?)"', value)
    return match.group(1) if match else value


def _ancestors(term_id: str, terms: Mapping[str, Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    pending = [term_id]
    while pending:
        current_id = pending.pop()
        parent_line = terms.get(current_id, {}).get("is_a")
        if not parent_line:
            continue
        parent_id = parent_line.split(" ! ", 1)[0].strip()
        if parent_id not in result:
            result.add(parent_id)
            pending.append(parent_id)
    return result


def _filter_terms(terms: Mapping[str, Mapping[str, Any]], patterns: list[str]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    expressions = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    matched: set[str] = set()
    for term_id, term in terms.items():
        if term.get("is_obsolete") or "name" not in term:
            continue
        text = " ".join([term.get("name", ""), *term.get("synonym", [])])
        if any(expression.search(text) for expression in expressions):
            matched.add(term_id)
    selected = matched | {parent for term_id in matched for parent in _ancestors(term_id, terms)}
    return {term_id: dict(terms[term_id]) for term_id in selected if term_id in terms}, matched


def _depth(term_id: str, terms: Mapping[str, Mapping[str, Any]], memo: dict[str, int]) -> int:
    if term_id in memo:
        return memo[term_id]
    parent_line = terms.get(term_id, {}).get("is_a")
    if not parent_line:
        memo[term_id] = 0
        return 0
    parent_id = parent_line.split(" ! ", 1)[0].strip()
    memo[term_id] = _depth(parent_id, terms, memo) + 1
    return memo[term_id]


def _normalise_abstraction_level(level: int | str) -> int:
    if isinstance(level, str):
        names = {"specific": 0, "balanced": 1, "broad": 2}
        if level not in names:
            raise ValueError("abstraction_level must be 0, 1, 2, specific, balanced, or broad")
        return names[level]
    if level not in {0, 1, 2}:
        raise ValueError("abstraction_level must be 0, 1, or 2")
    return level


def rank_terms(
    terms: Mapping[str, Mapping[str, Any]],
    matched: set[str],
    *,
    resource_priority: int = 0,
    abstraction_level: int | str = 1,
) -> list[str]:
    """Rank filtered terms, with higher levels preferring broader concepts."""

    level = _normalise_abstraction_level(abstraction_level)
    children: dict[str, int] = {}
    for term in terms.values():
        parent_line = term.get("is_a")
        if parent_line:
            parent_id = parent_line.split(" ! ", 1)[0].strip()
            children[parent_id] = children.get(parent_id, 0) + 1
    depth_memo: dict[str, int] = {}
    scored: list[tuple[float, str]] = []
    for term_id, term in terms.items():
        depth = _depth(term_id, terms, depth_memo)
        breadth = math.log1p(children.get(term_id, 0))
        direct_match = term_id in matched
        direct_weight = {0: 10, 1: 4, 2: 1}[level]
        score = (direct_weight if direct_match else 0) + resource_priority
        score += (2 * breadth - depth) if level == 2 else (breadth + 0.2 * depth) if level == 1 else 1.5 * depth
        scored.append((score, term_id))
    return [term_id for _, term_id in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _compact_term(source: str, term_id: str, term: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": term_id,
        "source": source,
        "label": term.get("name"),
        "definition": _clean_quoted(term.get("def", "")),
        "synonyms": [_clean_quoted(value) for value in term.get("synonym", [])],
        "parent": term.get("is_a", "").split(" ! ", 1)[0].strip() or None,
    }


def _reference_terms(catalog: Mapping[str, Mapping[str, Mapping[str, Any]]], entity_key: str, limit: int = 8) -> list[dict[str, str]]:
    expressions = [re.compile(pattern, re.IGNORECASE) for pattern in PROMPT_REFERENCE_PATTERNS.get(entity_key, [])]
    references = []
    for source, terms in catalog.items():
        for term_id, term in terms.items():
            text = " ".join([term.get("label", ""), *term.get("synonyms", [])])
            if any(expression.search(text) for expression in expressions):
                references.append({"id": f"{source}:{term_id}", "label": term["label"]})
    return sorted(references, key=lambda item: (item["label"].lower(), item["id"]))[:limit]


def assemble_microbiology_ontology(
    project_root: str | Path,
    *,
    term_budget: int = 500,
    abstraction_level: int | str = "balanced",
    include_ncbitaxon: bool = False,
) -> OntologyAssemblyResult:
    """Download and assemble a compact, prompt-compatible microbiology ontology.

    ``term_budget`` is the maximum number of external terms retained across all
    resources. ``abstraction_level`` accepts ``0``/``"specific"`` for leaf
    concepts, ``1``/``"balanced"``, or ``2``/``"broad"`` for higher-level
    concepts and their useful ancestors.
    """

    if term_budget < 1:
        raise ValueError("term_budget must be positive")
    root = Path(project_root)
    ontology_root = root / "assets" / "ontologies"
    raw_root = ontology_root / "external"
    base_path = ontology_root / "entity_ontology_microbiology.yaml"
    assembled_path = ontology_root / "entity_ontology_microbiology_assembled.yaml"
    catalog_path = ontology_root / "microbiology_external_terms.json"
    level = _normalise_abstraction_level(abstraction_level)

    resources = {
        name: spec
        for name, spec in RESOURCE_SPECS.items()
        if spec.get("enabled", True) and (name != "ncbitaxon" or include_ncbitaxon)
    }
    filtered: dict[str, dict[str, dict[str, Any]]] = {}
    matched_by_resource: dict[str, set[str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    raw_paths: list[Path] = []
    ranked: dict[str, list[str]] = {}

    for name, spec in resources.items():
        raw_path = _download(raw_root / f"{name}.{spec['format']}", spec["url"])
        raw_paths.append(raw_path)
        raw_bytes = raw_path.read_bytes()
        terms = _parse_resource(raw_path, spec["format"])
        selected, matched = _filter_terms(terms, spec["patterns"])
        filtered[name] = selected
        matched_by_resource[name] = matched
        ranked[name] = rank_terms(
            selected,
            matched & set(selected),
            resource_priority=spec.get("priority", 0),
            abstraction_level=level,
        )
        metadata[name] = {
            "url": spec["url"],
            "path": str(raw_path.relative_to(root)),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "available_terms": len(terms),
        }

    initial_quota = 1 if term_budget >= len(ranked) else 0
    selected_ids: dict[str, list[str]] = {
        name: ids[:initial_quota] for name, ids in ranked.items()
    }
    selected_keys = {(name, term_id) for name, ids in selected_ids.items() for term_id in ids}
    remaining = [
        (name, term_id)
        for name, ids in ranked.items()
        for term_id in ids[initial_quota:]
    ]
    remaining.sort(key=lambda item: (ranked[item[0]].index(item[1]), item[0], item[1]))
    for name, term_id in remaining:
        if len(selected_keys) >= term_budget:
            break
        selected_keys.add((name, term_id))
        selected_ids[name].append(term_id)

    catalog: dict[str, dict[str, dict[str, Any]]] = {
        name: {
            term_id: _compact_term(name, term_id, filtered[name][term_id])
            for term_id in ids
        }
        for name, ids in selected_ids.items()
    }

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install PyYAML to assemble the ontology.") from exc
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    for entity_key, entity_type in base.get("entity_types", {}).items():
        references = _reference_terms(catalog, entity_key)
        if references:
            entity_type["reference_terms"] = references
    base["ontology"]["id"] = "microbiology-zoonosis-assembled"
    base["ontology"]["version"] = "0.2.0"
    base["ontology"]["assembly"] = {
        "method": "download, relevance-filter, hierarchy-rank, and retain a bounded term catalog",
        "term_budget": term_budget,
        "abstraction_level": level,
        "sources": metadata,
    }
    base["external_terms"] = catalog
    assembled_path.parent.mkdir(parents=True, exist_ok=True)
    assembled_path.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding="utf-8")
    catalog_path.write_text(
        json.dumps({"sources": metadata, "terms": catalog}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return OntologyAssemblyResult(
        ontology_path=assembled_path,
        catalog_path=catalog_path,
        raw_paths=tuple(raw_paths),
        selected_terms=len(selected_keys),
        abstraction_level=level,
    )


__all__ = ["OntologyAssemblyResult", "assemble_microbiology_ontology", "rank_terms"]
