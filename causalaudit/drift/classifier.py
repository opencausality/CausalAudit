"""Causal drift severity classification and root-cause identification.

Takes the raw list of ``CausalBreak`` objects from the detector and
produces a single degradation score and a human-readable root-cause
explanation.
"""

from __future__ import annotations

import logging

from causalaudit.data.schema import CausalBreak, CausalGraph

logger = logging.getLogger("causalaudit.drift")

# Weight per break type for degradation scoring
_BREAK_WEIGHTS: dict[str, float] = {
    "PROXY_COLLAPSE": 1.0,
    "REMOVED_EDGE": 0.8,
    "WEAKENED": 0.5,
    "STRENGTHENED": 0.4,
    "NEW_EDGE": 0.3,
}


def compute_degradation_score(
    breaks: list[CausalBreak],
    baseline: CausalGraph,
) -> float:
    """Compute a [0, 1] severity score for the overall causal drift.

    The score combines:
    - The weighted sum of per-break significance values.
    - Normalised by the number of baseline edges (so adding more edges does
      not inflate the score).
    - Break type weights (proxy collapses are worse than new edges).

    Parameters
    ----------
    breaks:
        List of detected causal breaks.
    baseline:
        The baseline causal graph used as the reference denominator.

    Returns
    -------
    float
        Degradation score in [0, 1].  0 = no drift, 1 = complete structural
        collapse.
    """
    if not breaks:
        return 0.0

    n_baseline_edges = max(1, len(baseline.edges))

    weighted_sum = sum(
        _BREAK_WEIGHTS.get(b.break_type, 0.5) * b.significance
        for b in breaks
    )

    # Normalise: expect at most ~2x weighted breaks per baseline edge
    normaliser = n_baseline_edges * 2.0
    raw_score = weighted_sum / normaliser

    # Clip to [0, 1]
    score = float(min(1.0, raw_score))
    logger.debug(
        "Degradation score: %.4f (weighted_sum=%.3f, n_baseline_edges=%d)",
        score, weighted_sum, n_baseline_edges,
    )
    return score


def classify_root_cause(
    breaks: list[CausalBreak],
    baseline: CausalGraph,
) -> str:
    """Identify the primary root cause of model degradation.

    Selects the most significant break and generates a targeted explanation.

    Parameters
    ----------
    breaks:
        Detected causal breaks, ideally sorted by descending significance.
    baseline:
        The baseline causal graph for context.

    Returns
    -------
    str
        A human-readable primary root-cause explanation.  Returns a safe
        message if no breaks are present.
    """
    if not breaks:
        return (
            "No significant causal drift was detected.  The model's causal "
            "structure is consistent with the baseline."
        )

    # The first break (highest significance after sorting) is the primary driver
    primary = breaks[0]

    # Tally break types for context
    type_counts: dict[str, int] = {}
    for b in breaks:
        type_counts[b.break_type] = type_counts.get(b.break_type, 0) + 1

    secondary_summary = ", ".join(
        f"{count} {btype.lower().replace('_', ' ')} break(s)"
        for btype, count in sorted(type_counts.items(), key=lambda x: -x[1])
        if not (btype == primary.break_type and count == 1)
    )

    primary_narrative = _narrative_for_break(primary, baseline)

    if secondary_summary:
        return (
            f"{primary_narrative}  Additionally, {secondary_summary} "
            "were detected across the causal structure."
        )
    return primary_narrative


def _narrative_for_break(break_: CausalBreak, baseline: CausalGraph) -> str:
    """Generate a root-cause narrative for a single primary break."""
    b = break_

    if b.break_type == "PROXY_COLLAPSE":
        return (
            f"PRIMARY CAUSE — PROXY COLLAPSE: The feature '{b.cause}' has lost its "
            f"causal mechanism with '{b.effect}'.  Baseline strength was "
            f"{b.baseline_strength:.3f}; it is now effectively zero.  "
            "This is the most severe form of causal drift and typically results "
            "from data pipeline corruption, a policy change that decoupled the "
            "feature from the outcome, or feature staleness (e.g., stale cached values)."
        )

    if b.break_type == "REMOVED_EDGE":
        return (
            f"PRIMARY CAUSE — REMOVED EDGE: The causal relationship "
            f"'{b.cause}' → '{b.effect}' (baseline strength {b.baseline_strength:.3f}) "
            "is no longer statistically detectable.  The feature may have undergone "
            "a distribution shift that breaks its predictive mechanism, or the "
            "relationship has been confounded by a newly introduced variable."
        )

    if b.break_type == "WEAKENED":
        delta = (b.current_strength or 0.0) - (b.baseline_strength or 0.0)
        return (
            f"PRIMARY CAUSE — WEAKENED EDGE: The causal relationship "
            f"'{b.cause}' → '{b.effect}' has degraded significantly "
            f"({b.baseline_strength:.3f} → {b.current_strength:.3f}, Δ={delta:+.3f}).  "
            "This suggests the feature still carries causal signal but with reduced "
            "efficacy — investigate upstream data quality, feature engineering changes, "
            "or population-level distributional shift."
        )

    if b.break_type == "STRENGTHENED":
        delta = (b.current_strength or 0.0) - (b.baseline_strength or 0.0)
        return (
            f"PRIMARY CAUSE — STRENGTHENED EDGE: The causal relationship "
            f"'{b.cause}' → '{b.effect}' has unexpectedly intensified "
            f"({b.baseline_strength:.3f} → {b.current_strength:.3f}, Δ={delta:+.3f}).  "
            "While this may seem positive, unexpected strengthening often indicates "
            "spurious correlation from sample bias, population shift, or a newly "
            "introduced confound."
        )

    if b.break_type == "NEW_EDGE":
        return (
            f"PRIMARY CAUSE — NEW CAUSAL EDGE: A causal relationship "
            f"'{b.cause}' → '{b.effect}' (strength {b.current_strength:.3f}) has "
            "emerged that was absent from the baseline.  New edges indicate a "
            "structural change in the data-generating process — investigate whether "
            "a new feature interaction, confounding variable, or data pipeline change "
            "has been introduced."
        )

    return f"Primary causal break: {b.break_type} on '{b.cause}' → '{b.effect}'."
