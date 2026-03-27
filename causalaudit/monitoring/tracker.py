"""Current-window causal graph construction.

Mirrors ``baseline.py`` but is intended for the most-recent window of
inference logs.  The resulting graph is compared against the baseline in
the drift detection step.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from causalaudit.config import Settings, get_settings
from causalaudit.data.schema import CausalGraph, InferenceLog
from causalaudit.discovery.algorithms import discover_causal_graph

logger = logging.getLogger("causalaudit.monitoring")


def build_current_graph(
    log: InferenceLog,
    settings: Settings | None = None,
) -> CausalGraph:
    """Build a causal graph from recent inference logs for drift comparison.

    Uses the same discovery algorithm as ``build_baseline`` so that the
    resulting graphs are structurally comparable.

    Parameters
    ----------
    log:
        Recent inference log (the "current" production window).
    settings:
        Application settings.  Uses the global singleton when omitted.

    Returns
    -------
    CausalGraph
        The causal graph inferred from the current window.
    """
    settings = settings or get_settings()

    logger.info(
        "Building current causal graph from %d rows, target='%s'",
        log.n_rows, log.label_column,
    )

    df = pd.DataFrame(log.columns)
    edges = discover_causal_graph(
        data=df,
        target_column=log.label_column,
        significance_level=settings.significance_level,
    )

    # Apply the same confidence threshold as baseline for fair comparison
    filtered_edges = [
        e for e in edges
        if e.is_significant and e.strength >= settings.confidence_threshold
    ]

    logger.info(
        "Current: %d / %d edges passed significance+strength filters",
        len(filtered_edges), len(edges),
    )

    node_set: set[str] = {log.label_column}
    for edge in filtered_edges:
        node_set.add(edge.cause)
        node_set.add(edge.effect)

    graph = CausalGraph(
        nodes=sorted(node_set),
        edges=filtered_edges,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        model_version="current",
    )
    logger.info(
        "Current graph built: %d nodes, %d edges",
        len(graph.nodes), len(graph.edges),
    )
    return graph
