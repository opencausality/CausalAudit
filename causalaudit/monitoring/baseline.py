"""Baseline causal graph construction and persistence.

The baseline graph represents the expected causal structure of a healthy,
production-quality model.  It is built once from historical inference logs
and stored on disk so that future windows can be compared against it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from causalaudit.config import Settings, get_settings
from causalaudit.data.schema import CausalGraph, InferenceLog
from causalaudit.discovery.algorithms import discover_causal_graph

logger = logging.getLogger("causalaudit.monitoring")


def build_baseline(
    log: InferenceLog,
    settings: Settings | None = None,
) -> CausalGraph:
    """Build a baseline causal graph from historical inference logs.

    Discovers the causal structure between all feature columns and the label
    column using independence tests.  The resulting graph represents the
    reference causal structure for drift comparison.

    Parameters
    ----------
    log:
        Historical inference log containing feature and label columns.
    settings:
        Application settings.  Uses the global singleton when omitted.

    Returns
    -------
    CausalGraph
        The discovered baseline causal structure.
    """
    settings = settings or get_settings()

    logger.info(
        "Building baseline causal graph from %d rows, target='%s'",
        log.n_rows, log.label_column,
    )

    df = pd.DataFrame(log.columns)
    edges = discover_causal_graph(
        data=df,
        target_column=log.label_column,
        significance_level=settings.significance_level,
    )

    # Filter to significant edges above the confidence threshold
    filtered_edges = [
        e for e in edges
        if e.is_significant and e.strength >= settings.confidence_threshold
    ]

    logger.info(
        "Baseline: %d / %d edges passed significance+strength filters",
        len(filtered_edges), len(edges),
    )

    nodes = _extract_nodes(filtered_edges, log.label_column, log.feature_columns)

    graph = CausalGraph(
        nodes=nodes,
        edges=filtered_edges,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        model_version="baseline",
    )
    logger.info(
        "Baseline graph built: %d nodes, %d edges",
        len(graph.nodes), len(graph.edges),
    )
    return graph


def save_baseline(graph: CausalGraph, path: Path) -> None:
    """Persist a baseline ``CausalGraph`` to a JSON file.

    Parameters
    ----------
    graph:
        The baseline graph to save.
    path:
        Destination file path.  Parent directories must exist.
    """
    path = Path(path)
    payload = graph.model_dump()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Baseline graph saved → %s", path)


def load_baseline(path: Path) -> CausalGraph:
    """Load a previously saved baseline ``CausalGraph`` from disk.

    Parameters
    ----------
    path:
        Path to a JSON file written by ``save_baseline``.

    Returns
    -------
    CausalGraph
        The reconstructed baseline graph.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the JSON cannot be parsed into a valid ``CausalGraph``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph = CausalGraph.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Failed to parse baseline file '{path}': {exc}") from exc

    logger.info(
        "Baseline graph loaded from %s (%d nodes, %d edges)",
        path, len(graph.nodes), len(graph.edges),
    )
    return graph


def _extract_nodes(
    edges: list,
    target: str,
    features: list[str],
) -> list[str]:
    """Collect unique node names from edges, ensuring the target is included."""
    node_set: set[str] = {target}
    for edge in edges:
        node_set.add(edge.cause)
        node_set.add(edge.effect)
    # Add feature columns that appear in edges
    return sorted(node_set)
