"""Structural causal drift detection.

Compares two causal graphs (baseline and current) and identifies structural
breaks — edges that appeared, disappeared, strengthened, or collapsed.
"""

from __future__ import annotations

import logging

from causalaudit.config import Settings, get_settings
from causalaudit.data.schema import CausalBreak, CausalGraph

logger = logging.getLogger("causalaudit.drift")

# Threshold for classifying strength change as meaningful drift
_STRENGTH_DELTA_THRESHOLD = 0.15

# Threshold below which a previously strong edge is considered collapsed
_PROXY_COLLAPSE_THRESHOLD = 0.05


def detect_drift(
    baseline: CausalGraph,
    current: CausalGraph,
    settings: Settings | None = None,
) -> list[CausalBreak]:
    """Compare two causal graphs and detect structural breaks.

    Compares edges in the baseline against edges in the current graph,
    classifying any structural differences into typed ``CausalBreak`` objects.

    Break types detected:
    - ``REMOVED_EDGE``: Present in baseline, absent from current.
    - ``NEW_EDGE``: Present in current, absent from baseline.
    - ``STRENGTHENED``: In both graphs, but current strength is notably higher.
    - ``WEAKENED``: In both graphs, but current strength is notably lower.
    - ``PROXY_COLLAPSE``: Baseline had strong edge; current strength near zero,
      suggesting the feature became a proxy or was corrupted.

    Parameters
    ----------
    baseline:
        Reference causal graph from historical data.
    current:
        Current causal graph from recent production data.
    settings:
        Application settings for threshold configuration.

    Returns
    -------
    list[CausalBreak]
        Detected breaks sorted by descending significance.
    """
    settings = settings or get_settings()

    baseline_map = baseline.edge_map()
    current_map = current.edge_map()

    baseline_keys = set(baseline_map.keys())
    current_keys = set(current_map.keys())

    breaks: list[CausalBreak] = []

    # ── REMOVED_EDGE and PROXY_COLLAPSE ──────────────────────────────────────
    for key in baseline_keys - current_keys:
        baseline_edge = baseline_map[key]
        cause, effect = key

        if baseline_edge.strength >= 0.5:
            # A strong edge vanished — this is a proxy collapse
            breaks.append(
                CausalBreak(
                    break_type="PROXY_COLLAPSE",
                    cause=cause,
                    effect=effect,
                    baseline_strength=baseline_edge.strength,
                    current_strength=0.0,
                    significance=_compute_removal_significance(baseline_edge.strength),
                    explanation=(
                        f"The strong causal relationship '{cause}' → '{effect}' "
                        f"(baseline strength {baseline_edge.strength:.3f}) has completely "
                        "collapsed in the current window. This typically indicates that "
                        f"'{cause}' has become a proxy variable, been corrupted in the "
                        "feature pipeline, or a structural policy change has severed the "
                        "mechanism."
                    ),
                )
            )
        else:
            breaks.append(
                CausalBreak(
                    break_type="REMOVED_EDGE",
                    cause=cause,
                    effect=effect,
                    baseline_strength=baseline_edge.strength,
                    current_strength=None,
                    significance=_compute_removal_significance(baseline_edge.strength),
                    explanation=(
                        f"The causal edge '{cause}' → '{effect}' (baseline strength "
                        f"{baseline_edge.strength:.3f}) is no longer statistically "
                        "significant in the current window.  The feature may have "
                        "lost its causal relationship with the outcome."
                    ),
                )
            )

    # ── NEW_EDGE ─────────────────────────────────────────────────────────────
    for key in current_keys - baseline_keys:
        current_edge = current_map[key]
        cause, effect = key

        breaks.append(
            CausalBreak(
                break_type="NEW_EDGE",
                cause=cause,
                effect=effect,
                baseline_strength=None,
                current_strength=current_edge.strength,
                significance=_compute_new_edge_significance(current_edge.strength),
                explanation=(
                    f"A new causal relationship has emerged: '{cause}' → '{effect}' "
                    f"(current strength {current_edge.strength:.3f}).  This was not "
                    "present in the baseline and could indicate distributional shift, "
                    "a new feature interaction, or a change in the data-generating "
                    "process."
                ),
            )
        )

    # ── STRENGTHENED and WEAKENED ─────────────────────────────────────────────
    for key in baseline_keys & current_keys:
        baseline_edge = baseline_map[key]
        current_edge = current_map[key]
        cause, effect = key

        delta = current_edge.strength - baseline_edge.strength

        if delta > _STRENGTH_DELTA_THRESHOLD:
            breaks.append(
                CausalBreak(
                    break_type="STRENGTHENED",
                    cause=cause,
                    effect=effect,
                    baseline_strength=baseline_edge.strength,
                    current_strength=current_edge.strength,
                    significance=min(1.0, abs(delta) / baseline_edge.strength)
                    if baseline_edge.strength > 0 else abs(delta),
                    explanation=(
                        f"The causal relationship '{cause}' → '{effect}' has significantly "
                        f"strengthened: {baseline_edge.strength:.3f} → {current_edge.strength:.3f} "
                        f"(Δ={delta:+.3f}).  This may indicate covariate shift where "
                        f"'{cause}' now accounts for more outcome variance, or confounding "
                        "has increased."
                    ),
                )
            )
        elif delta < -_STRENGTH_DELTA_THRESHOLD:
            if current_edge.strength <= _PROXY_COLLAPSE_THRESHOLD and baseline_edge.strength >= 0.5:
                # Technically still present but near zero — report as proxy collapse
                breaks.append(
                    CausalBreak(
                        break_type="PROXY_COLLAPSE",
                        cause=cause,
                        effect=effect,
                        baseline_strength=baseline_edge.strength,
                        current_strength=current_edge.strength,
                        significance=_compute_removal_significance(baseline_edge.strength),
                        explanation=(
                            f"The causal relationship '{cause}' → '{effect}' has effectively "
                            f"collapsed: {baseline_edge.strength:.3f} → {current_edge.strength:.3f}. "
                            "The edge still passes statistical thresholds but is functionally "
                            "negligible, suggesting the feature has become a near-proxy."
                        ),
                    )
                )
            else:
                breaks.append(
                    CausalBreak(
                        break_type="WEAKENED",
                        cause=cause,
                        effect=effect,
                        baseline_strength=baseline_edge.strength,
                        current_strength=current_edge.strength,
                        significance=min(1.0, abs(delta) / baseline_edge.strength)
                        if baseline_edge.strength > 0 else abs(delta),
                        explanation=(
                            f"The causal relationship '{cause}' → '{effect}' has weakened: "
                            f"{baseline_edge.strength:.3f} → {current_edge.strength:.3f} "
                            f"(Δ={delta:+.3f}).  The feature still causally influences the "
                            "outcome but with reduced effect — possibly due to population "
                            "shift or upstream feature engineering changes."
                        ),
                    )
                )

    # Sort by descending significance
    breaks.sort(key=lambda b: b.significance, reverse=True)

    logger.info(
        "Drift detection complete: %d breaks found "
        "(removed=%d, new=%d, strengthened=%d, weakened=%d, proxy_collapse=%d)",
        len(breaks),
        sum(1 for b in breaks if b.break_type == "REMOVED_EDGE"),
        sum(1 for b in breaks if b.break_type == "NEW_EDGE"),
        sum(1 for b in breaks if b.break_type == "STRENGTHENED"),
        sum(1 for b in breaks if b.break_type == "WEAKENED"),
        sum(1 for b in breaks if b.break_type == "PROXY_COLLAPSE"),
    )
    return breaks


def _compute_removal_significance(baseline_strength: float) -> float:
    """Compute significance of a removed/collapsed edge from its baseline strength.

    Stronger baseline edges produce higher significance scores when removed,
    because they represent a larger departure from expected causal structure.
    """
    # Sigmoid-like scaling: linear in strength, capped at 1.0
    return float(min(1.0, baseline_strength * 1.5))


def _compute_new_edge_significance(current_strength: float) -> float:
    """Compute significance of a newly-appeared edge from its current strength."""
    return float(min(1.0, current_strength * 1.2))
