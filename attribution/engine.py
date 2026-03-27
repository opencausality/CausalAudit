"""Degradation attribution and recommendation generation.

Translates a ``DriftReport`` into ranked, actionable recommendations for
ML engineers and data scientists.  Recommendations are ordered by estimated
impact on restoring model reliability.
"""

from __future__ import annotations

import logging

from causalaudit.data.schema import CausalBreak, DriftReport

logger = logging.getLogger("causalaudit.attribution")

# Score thresholds for determining retrain scope
_FULL_RETRAIN_THRESHOLD = 0.6
_TARGETED_UPDATE_THRESHOLD = 0.3


def attribute_degradation(
    report: DriftReport,
    performance_delta: float | None = None,
) -> list[str]:
    """Generate ranked, actionable recommendations from a drift report.

    Analyses the causal breaks in the report and produces a prioritised list
    of concrete next steps, ranging from targeted feature investigation to
    full model retraining.

    Parameters
    ----------
    report:
        The completed drift report, including break list and degradation score.
    performance_delta:
        Optional observed performance drop (e.g., -0.05 means 5% accuracy
        regression).  When provided, recommendations are calibrated to the
        observed severity.

    Returns
    -------
    list[str]
        Ordered list of recommendations, most impactful first.
    """
    recommendations: list[str] = []
    breaks = report.breaks
    score = report.degradation_score

    if not breaks:
        recommendations.append(
            "No causal drift detected.  Continue monitoring on the current schedule."
        )
        logger.info("No breaks found; returning no-drift recommendation.")
        return recommendations

    logger.info(
        "Attributing degradation: score=%.3f, breaks=%d, performance_delta=%s",
        score, len(breaks), performance_delta,
    )

    # ── Classify most-impactful breaks ────────────────────────────────────────
    proxy_collapses = [b for b in breaks if b.break_type == "PROXY_COLLAPSE"]
    removed_edges = [b for b in breaks if b.break_type == "REMOVED_EDGE"]
    weakened_edges = [b for b in breaks if b.break_type == "WEAKENED"]
    new_edges = [b for b in breaks if b.break_type == "NEW_EDGE"]
    strengthened_edges = [b for b in breaks if b.break_type == "STRENGTHENED"]

    # ── Proxy collapse recommendations ────────────────────────────────────────
    if proxy_collapses:
        collapsed_features = [b.cause for b in proxy_collapses]
        recommendations.append(
            f"CRITICAL — Immediately audit the data pipeline for feature(s): "
            f"{', '.join(collapsed_features)}.  These features have completely lost "
            "their causal mechanism and are likely corrupted, stale, or have had their "
            "meaning changed by an upstream system.  Do NOT retrain until the root "
            "cause of the collapse is identified and fixed."
        )
        recommendations.append(
            f"Run data quality checks on {', '.join(collapsed_features)}: verify "
            "value distributions, check for encoding changes, confirm the ETL pipeline "
            "has not introduced silent failures (e.g., constant imputation, schema "
            "mismatches)."
        )

    # ── Removed edge recommendations ─────────────────────────────────────────
    if removed_edges:
        removed_features = [b.cause for b in removed_edges]
        recommendations.append(
            f"Investigate feature(s) {', '.join(removed_features)}: their causal "
            "relationships with the target have vanished.  Check for population shift "
            "(e.g., new customer segment), policy changes (e.g., new regulatory rules), "
            "or upstream feature-engineering modifications."
        )

    # ── Weakened edge recommendations ────────────────────────────────────────
    if weakened_edges:
        weakened_features = sorted(weakened_edges, key=lambda b: b.significance, reverse=True)
        top_weakened = [b.cause for b in weakened_features[:3]]
        recommendations.append(
            f"Re-examine feature importance for: {', '.join(top_weakened)}.  "
            "These features retain some causal signal but at reduced strength — "
            "consider whether re-collecting fresh training data in the current "
            "population would help, or whether feature engineering needs updating."
        )

    # ── New edge recommendations ──────────────────────────────────────────────
    if new_edges:
        new_causes = [b.cause for b in new_edges]
        recommendations.append(
            f"New causal relationships have emerged involving: {', '.join(new_causes)}.  "
            "Verify these are genuine causal signals and not spurious correlations from "
            "sample bias.  If genuine, consider incorporating these features more "
            "prominently in the model."
        )

    # ── Strengthened edge recommendations ────────────────────────────────────
    if strengthened_edges:
        strengthened_causes = [b.cause for b in strengthened_edges]
        recommendations.append(
            f"Unexpected strengthening of causal relationships for: "
            f"{', '.join(strengthened_causes)}.  Verify there is no selection bias, "
            "confounding, or data leakage introducing spurious correlation."
        )

    # ── Retrain scope recommendation ─────────────────────────────────────────
    recommendations.extend(
        _retrain_scope_recommendation(score, breaks, performance_delta)
    )

    # ── Monitoring recommendation ─────────────────────────────────────────────
    if score > 0.2:
        recommendations.append(
            "Increase monitoring frequency: run CausalAudit daily (or per-batch) "
            "until causal structure stabilises.  Set up automated alerts for "
            "degradation_score > 0.3."
        )

    # ── Performance-aware recommendation ─────────────────────────────────────
    if performance_delta is not None and performance_delta < -0.05:
        recommendations.append(
            f"Observed performance drop of {abs(performance_delta):.1%} correlates with "
            "the detected causal drift.  Prioritise the proxy collapse and removed-edge "
            "fixes first — they are most likely to account for performance regression."
        )

    logger.info("Generated %d recommendations", len(recommendations))
    return recommendations


def _retrain_scope_recommendation(
    score: float,
    breaks: list[CausalBreak],
    performance_delta: float | None,
) -> list[str]:
    """Return recommendations about the scope of model retraining."""
    recs: list[str] = []

    if score >= _FULL_RETRAIN_THRESHOLD:
        affected_features = sorted({b.cause for b in breaks})
        recs.append(
            f"RETRAIN RECOMMENDED — The causal structure has changed substantially "
            f"(degradation score {score:.2f}).  A full model retrain on fresh data "
            f"collected under the current distribution is recommended.  "
            f"Affected features: {', '.join(affected_features)}."
        )
    elif score >= _TARGETED_UPDATE_THRESHOLD:
        high_sig_features = sorted({
            b.cause for b in breaks if b.significance >= 0.5
        })
        recs.append(
            f"TARGETED UPDATE RECOMMENDED — Moderate causal drift detected "
            f"(degradation score {score:.2f}).  A targeted update — retraining on "
            f"a fresh slice of data with focus on features "
            f"{', '.join(high_sig_features) if high_sig_features else 'identified above'} "
            "— may restore model performance without a full retrain."
        )
    else:
        recs.append(
            f"LOW DRIFT — The causal structure shows minor drift "
            f"(degradation score {score:.2f}).  Monitor closely but full retraining "
            "is not immediately necessary.  Consider refreshing the baseline graph "
            "to capture the gradual evolution."
        )

    return recs
