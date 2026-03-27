"""NetworkX DiGraph construction from CausalGraph objects.

Provides utilities to convert CausalAudit's Pydantic graph model into
NetworkX directed graphs for analysis, and to persist/restore graphs as JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from causalaudit.data.schema import CausalEdge, CausalGraph

logger = logging.getLogger("causalaudit.graph")


def build_nx_graph(graph: CausalGraph) -> nx.DiGraph:
    """Convert a ``CausalGraph`` to a NetworkX ``DiGraph``.

    Each node is labelled with its variable name.  Each directed edge carries
    ``strength`` and ``p_value`` attributes from the source ``CausalEdge``.

    Parameters
    ----------
    graph:
        The CausalAudit graph model to convert.

    Returns
    -------
    nx.DiGraph
        A directed graph with edge attributes ``strength`` and ``p_value``.
    """
    g = nx.DiGraph()
    g.add_nodes_from(graph.nodes)

    for edge in graph.edges:
        g.add_edge(
            edge.cause,
            edge.effect,
            strength=edge.strength,
            p_value=edge.p_value,
            is_significant=edge.is_significant,
        )

    logger.debug(
        "Built DiGraph: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges()
    )
    return g


def save_graph_json(graph: CausalGraph, path: Path) -> None:
    """Persist a ``CausalGraph`` to a JSON file.

    Parameters
    ----------
    graph:
        The graph to serialise.
    path:
        Destination file path.
    """
    path = Path(path)
    payload: dict[str, Any] = graph.model_dump()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("CausalGraph saved → %s", path)


def load_graph_json(path: Path) -> CausalGraph:
    """Load a ``CausalGraph`` from a JSON file produced by ``save_graph_json``.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    CausalGraph
        The deserialised graph.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file content is not a valid ``CausalGraph``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph = CausalGraph.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Failed to load graph from '{path}': {exc}") from exc

    logger.info(
        "CausalGraph loaded from %s (%d nodes, %d edges)",
        path, len(graph.nodes), len(graph.edges),
    )
    return graph


def compute_graph_stats(graph: CausalGraph) -> dict[str, Any]:
    """Compute summary statistics for a ``CausalGraph``.

    Returns a dict with keys:
    - ``n_nodes``, ``n_edges``, ``n_significant_edges``
    - ``mean_strength``, ``max_strength``, ``min_strength``
    - ``is_dag``: whether the NetworkX graph is a DAG (acyclic).
    """
    g = build_nx_graph(graph)

    strengths = [e.strength for e in graph.edges]
    sig_edges = [e for e in graph.edges if e.is_significant]

    stats: dict[str, Any] = {
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "n_significant_edges": len(sig_edges),
        "mean_strength": float(sum(strengths) / len(strengths)) if strengths else 0.0,
        "max_strength": float(max(strengths)) if strengths else 0.0,
        "min_strength": float(min(strengths)) if strengths else 0.0,
        "is_dag": nx.is_directed_acyclic_graph(g),
    }
    logger.debug("Graph stats: %s", stats)
    return stats
