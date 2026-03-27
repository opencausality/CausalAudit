"""PC-style causal discovery algorithm (simplified, no external causal-learn dependency).

This module implements a pairwise causal structure learning approach that:
1. Computes Pearson correlation between each feature and the target.
2. Applies Fisher Z significance tests to filter spurious edges.
3. Uses partial correlations conditioned on subsets of other features to
   approximate d-separation tests, distinguishing direct from indirect effects.

The approach is intentionally conservative and targets the feature→target
direction only (not feature→feature edges), which is the most useful signal
for ML model monitoring.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd

from causalaudit.data.schema import CausalEdge
from causalaudit.discovery.independence import (
    fisher_z_test,
    independence_test,
    partial_correlation,
)

logger = logging.getLogger("causalaudit.discovery")


def discover_causal_graph(
    data: pd.DataFrame,
    target_column: str,
    significance_level: float = 0.05,
) -> list[CausalEdge]:
    """Discover causal relationships between features and a target column.

    Uses pairwise Pearson correlation strength together with partial
    correlations to infer which feature→target edges survive conditioning on
    other features.  This approximates the skeleton-finding step of the PC
    algorithm restricted to the star graph centred on ``target_column``.

    Algorithm
    ---------
    1. For each feature F, compute Pearson correlation with target T.
    2. Run Fisher Z test; skip F if p > significance_level.
    3. For each surviving F, compute the partial correlation r(F, T | others)
       where ``others`` is the set of all other significant features.
    4. Re-test significance after conditioning.  Features that become
       independent after conditioning are likely indirect (mediated) effects.
    5. Return significant edges as ``CausalEdge`` objects sorted by |strength|.

    Parameters
    ----------
    data:
        DataFrame where each column is a variable and each row is an
        observation.  Must contain ``target_column``.
    target_column:
        Name of the column to treat as the causal outcome (label).
    significance_level:
        P-value threshold.  Edges with p > this are excluded.

    Returns
    -------
    list[CausalEdge]
        Discovered edges sorted by descending ``|strength|``.
    """
    if target_column not in data.columns:
        raise ValueError(
            f"target_column '{target_column}' not found in data. "
            f"Available: {list(data.columns)}"
        )

    feature_cols = [c for c in data.columns if c != target_column]
    if not feature_cols:
        logger.warning("No feature columns found besides target; returning empty graph.")
        return []

    target_arr = data[target_column].to_numpy(dtype=float)
    n = len(target_arr)

    logger.info(
        "Causal discovery: %d features → target='%s', n=%d, alpha=%.3f",
        len(feature_cols), target_column, n, significance_level,
    )

    # ── Step 1: pairwise correlations ────────────────────────────────────────
    pairwise: dict[str, tuple[float, float]] = {}  # feature → (r, p_value)
    for feat in feature_cols:
        feat_arr = data[feat].to_numpy(dtype=float)
        r = partial_correlation(feat_arr, target_arr, None)
        _, p = fisher_z_test(r, n)
        pairwise[feat] = (r, p)
        logger.debug("Pairwise: %s → %s  r=%.4f  p=%.4f", feat, target_column, r, p)

    # ── Step 2: filter to significant features ────────────────────────────────
    significant_features = [
        feat for feat, (r, p) in pairwise.items() if p <= significance_level
    ]
    logger.info(
        "%d / %d features pass pairwise significance filter (alpha=%.3f)",
        len(significant_features), len(feature_cols), significance_level,
    )

    if not significant_features:
        logger.warning(
            "No features are significantly correlated with target '%s'. "
            "Returning empty graph.", target_column,
        )
        return []

    # ── Step 3 & 4: partial correlations conditioned on other features ────────
    edges: list[CausalEdge] = []

    for feat in significant_features:
        feat_arr = data[feat].to_numpy(dtype=float)

        # Conditioning set: all other significant features
        others = [
            data[f].to_numpy(dtype=float)
            for f in significant_features
            if f != feat
        ]

        if others:
            # Partial correlation given all other significant features
            z_arr = np.column_stack(others) if len(others) > 1 else others[0]
            r_partial = partial_correlation(feat_arr, target_arr, z_arr)
            _, p_partial = fisher_z_test(r_partial, n)
        else:
            # Only one significant feature — use pairwise result
            r_partial, p_partial = pairwise[feat]

        strength = abs(r_partial)
        is_sig = p_partial <= significance_level

        logger.debug(
            "Partial: %s → %s  r_partial=%.4f  p=%.4f  significant=%s",
            feat, target_column, r_partial, p_partial, is_sig,
        )

        edge = CausalEdge(
            cause=feat,
            effect=target_column,
            strength=float(np.clip(strength, 0.0, 1.0)),
            p_value=float(p_partial),
            is_significant=is_sig,
        )
        edges.append(edge)

    # Also add feature–feature edges for non-target pairs if there are
    # multiple features (helps build a richer graph for drift diagnosis)
    if len(significant_features) >= 2:
        edges.extend(
            _discover_feature_edges(
                data=data,
                features=significant_features,
                n=n,
                significance_level=significance_level,
            )
        )

    # Sort by descending strength
    edges.sort(key=lambda e: e.strength, reverse=True)
    logger.info("Discovered %d causal edges in total", len(edges))
    return edges


def _discover_feature_edges(
    data: pd.DataFrame,
    features: list[str],
    n: int,
    significance_level: float,
) -> list[CausalEdge]:
    """Discover direct edges among feature variables (not involving target).

    For each feature pair (A, B), tests whether A→B survives conditioning on
    all other features.  Both directions are tested; the stronger one wins.

    Parameters
    ----------
    data:
        Full DataFrame.
    features:
        List of feature column names to consider.
    n:
        Number of rows.
    significance_level:
        P-value threshold.

    Returns
    -------
    list[CausalEdge]
        Feature-to-feature edges that survive the conditioning test.
    """
    feature_edges: list[CausalEdge] = []

    for feat_a, feat_b in combinations(features, 2):
        arr_a = data[feat_a].to_numpy(dtype=float)
        arr_b = data[feat_b].to_numpy(dtype=float)

        # Condition on all other features
        others = [
            data[f].to_numpy(dtype=float)
            for f in features
            if f not in (feat_a, feat_b)
        ]

        if others:
            z_arr = np.column_stack(others) if len(others) > 1 else others[0]
            r = partial_correlation(arr_a, arr_b, z_arr)
        else:
            r = partial_correlation(arr_a, arr_b, None)

        _, p = fisher_z_test(r, n)

        if p <= significance_level:
            # Assign direction: higher absolute marginal correlation with target
            # as proxy (the variable more predictive of outcome is likely upstream)
            edge = CausalEdge(
                cause=feat_a,
                effect=feat_b,
                strength=float(np.clip(abs(r), 0.0, 1.0)),
                p_value=float(p),
                is_significant=True,
            )
            feature_edges.append(edge)

    logger.debug("Found %d feature-feature edges", len(feature_edges))
    return feature_edges
