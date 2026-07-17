"""Utilities for selecting and serializing connected graph substructures."""

from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Optional


def sample_random_connected_subgraph(
    graph: Any,
    num_nodes: int = 3,
    *,
    seed: Optional[int] = None,
    start_node: Any = None,
) -> Any:
    """Sample an induced connected subgraph using a seeded random walk.

    Connectivity is evaluated on the undirected projection, while the
    returned subgraph preserves the original graph direction and attributes.
    If the walk reaches a dead end, it restarts from a visited node with an
    unvisited frontier; this keeps the requested node count attainable without
    introducing disconnected nodes.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install networkx to sample graph substructures.") from exc

    if not hasattr(graph, "number_of_nodes") or not hasattr(graph, "subgraph"):
        raise TypeError("graph must be a NetworkX-like graph.")
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    if num_nodes > graph.number_of_nodes():
        raise ValueError("num_nodes cannot exceed the graph node count.")

    undirected = graph.to_undirected(as_view=True) if graph.is_directed() else graph
    if start_node is None:
        rng = random.Random(seed)
        start_node = rng.choice(list(graph.nodes()))
    elif start_node not in graph:
        raise ValueError(f"start_node {start_node!r} is not in the graph.")
    else:
        rng = random.Random(seed)

    component = nx.node_connected_component(undirected, start_node)
    if len(component) < num_nodes:
        raise ValueError(
            f"The component containing {start_node!r} has {len(component)} nodes; "
            f"{num_nodes} requested."
        )

    selected = [start_node]
    selected_set = {start_node}
    current = start_node
    while len(selected) < num_nodes:
        candidates = [
            node for node in undirected.neighbors(current) if node not in selected_set
        ]
        if not candidates:
            frontier = [
                node
                for node in selected
                if any(
                    neighbor not in selected_set
                    for neighbor in undirected.neighbors(node)
                )
            ]
            if not frontier:
                raise RuntimeError("Random walk could not reach the requested node count.")
            current = rng.choice(frontier)
            candidates = [
                node
                for node in undirected.neighbors(current)
                if node not in selected_set
            ]
        current = rng.choice(candidates)
        selected.append(current)
        selected_set.add(current)

    return graph.subgraph(selected).copy()


def subgraph_to_payload(graph: Any) -> Dict[str, Any]:
    """Serialize all node and edge information in a subgraph for an LLM."""
    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise TypeError("graph must be a NetworkX-like graph.")

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [clean(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    nodes = [
        {"id": str(node_id), "attributes": clean(dict(attributes))}
        for node_id, attributes in graph.nodes(data=True)
    ]
    if graph.is_multigraph():
        edges = [
            {
                "id": str(key),
                "source_id": str(source),
                "target_id": str(target),
                "attributes": clean(dict(attributes)),
            }
            for source, target, key, attributes in graph.edges(data=True, keys=True)
        ]
    else:
        edges = [
            {
                "id": f"{source}->{target}",
                "source_id": str(source),
                "target_id": str(target),
                "attributes": clean(dict(attributes)),
            }
            for source, target, attributes in graph.edges(data=True)
        ]
    return {
        "directed": bool(graph.is_directed()),
        "nodes": nodes,
        "edges": edges,
    }


__all__ = ["sample_random_connected_subgraph", "subgraph_to_payload"]

