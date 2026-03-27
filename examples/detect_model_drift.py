"""Example: Detect causal drift between baseline and production model logs.

Demonstrates the full CausalAudit pipeline:
1. Load inference logs from two time windows (baseline vs. current)
2. Build causal graphs from each
3. Detect structural breaks
4. Generate a human-readable drift report

Run:
    python examples/detect_model_drift.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from causalaudit.attribution.engine import compute_degradation_score
from causalaudit.config import Settings, configure_logging
from causalaudit.data.loader import load_inference_log
from causalaudit.data.schema import DriftReport
from causalaudit.drift.detector import detect_drift
from causalaudit.graph.builder import build_causal_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
configure_logging()

BASELINE = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_logs" / "baseline_inference.csv"
CURRENT = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_logs" / "drifted_inference.csv"


def main() -> None:
    """Run causal drift detection between baseline and drifted model logs."""
    settings = Settings()

    # ── Load inference logs ───────────────────────────────────────────────────
    print(f"Loading baseline: {BASELINE.name}")
    baseline_log = load_inference_log(BASELINE)
    print(f"  {baseline_log.n_rows} rows, {len(baseline_log.columns)} columns")
    print(f"  Features: {baseline_log.feature_columns}")

    print(f"\nLoading current: {CURRENT.name}")
    current_log = load_inference_log(CURRENT)
    print(f"  {current_log.n_rows} rows, {len(current_log.columns)} columns")

    # ── Build causal graphs ───────────────────────────────────────────────────
    print("\nBuilding baseline causal graph...")
    baseline_graph = build_causal_graph(baseline_log)
    print(f"  {len(baseline_graph.nodes)} nodes, {len(baseline_graph.edges)} edges")
    print(f"  Significant edges: {len(baseline_graph.significant_edges())}")

    print("\nBuilding current causal graph...")
    current_graph = build_causal_graph(current_log)
    print(f"  {len(current_graph.nodes)} nodes, {len(current_graph.edges)} edges")
    print(f"  Significant edges: {len(current_graph.significant_edges())}")

    # ── Detect structural breaks ──────────────────────────────────────────────
    print("\nDetecting causal drift...")
    breaks = detect_drift(baseline_graph, current_graph, settings)

    degradation = compute_degradation_score(breaks, baseline_graph)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("CAUSAL DRIFT ANALYSIS")
    print("═" * 70)
    print(f"\nDegradation score: {degradation:.2f} / 1.00")
    print(f"Structural breaks detected: {len(breaks)}")

    if not breaks:
        print("\n✅ No causal structural breaks detected. Model is stable.")
        return

    # Show breaks sorted by significance
    sorted_breaks = sorted(breaks, key=lambda b: b.significance, reverse=True)

    print("\nStructural Breaks (sorted by severity):")
    print()
    for i, brk in enumerate(sorted_breaks, 1):
        symbol = "🔴" if brk.significance > 0.7 else "🟡" if brk.significance > 0.3 else "🟢"
        print(f"{i}. {symbol} [{brk.break_type}] {brk.cause} → {brk.effect}")
        print(f"   Significance: {brk.significance:.2f}")
        if brk.baseline_strength is not None:
            print(f"   Baseline strength: {brk.baseline_strength:.3f}")
        if brk.current_strength is not None:
            print(f"   Current strength:  {brk.current_strength:.3f}")
        print(f"   {brk.explanation}")
        print()

    # ── Baseline vs current edge comparison ───────────────────────────────────
    print("Causal Graph Comparison:")
    print()
    baseline_edges = {(e.cause, e.effect): e for e in baseline_graph.significant_edges()}
    current_edges = {(e.cause, e.effect): e for e in current_graph.significant_edges()}

    all_edge_keys = set(baseline_edges.keys()) | set(current_edges.keys())

    for cause, effect in sorted(all_edge_keys):
        b = baseline_edges.get((cause, effect))
        c = current_edges.get((cause, effect))

        if b and c:
            delta = c.strength - b.strength
            direction = "↑" if delta > 0.05 else "↓" if delta < -0.05 else "≈"
            print(f"  {cause} → {effect}: {b.strength:.3f} → {c.strength:.3f} {direction}")
        elif b:
            print(f"  {cause} → {effect}: {b.strength:.3f} → [REMOVED] ❌")
        else:
            print(f"  {cause} → {effect}: [NEW] → {c.strength:.3f} ⚠️")

    # ── Save report ────────────────────────────────────────────────────────────
    report = DriftReport(
        baseline_graph=baseline_graph,
        current_graph=current_graph,
        breaks=sorted_breaks,
        degradation_score=degradation,
        root_cause=sorted_breaks[0].explanation if sorted_breaks else "No drift detected.",
        recommendations=[
            "Investigate PROXY_COLLAPSE or REMOVED_EDGE breaks first.",
            "Compare feature distributions between time windows.",
            "Re-train if degradation_score > 0.5.",
        ],
    )

    output = Path("drift_report.json")
    output.write_text(report.model_dump_json(indent=2))
    print(f"\nFull report saved → {output}")


if __name__ == "__main__":
    main()
