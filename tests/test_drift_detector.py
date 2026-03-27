"""Tests for structural drift detection."""

from __future__ import annotations

import pytest

from causalaudit.data.schema import CausalBreak, CausalEdge, CausalGraph
from causalaudit.drift.detector import detect_drift


class TestDetectDrift:
    def test_no_drift_returns_empty_breaks(self, baseline_graph: CausalGraph) -> None:
        """Comparing a graph to itself returns no breaks."""
        breaks = detect_drift(baseline_graph, baseline_graph)
        assert breaks == []

    def test_detects_weakened_edge(
        self, baseline_graph: CausalGraph, drifted_graph: CausalGraph
    ) -> None:
        """feature_a → prediction weakened from 0.8 to 0.2."""
        breaks = detect_drift(baseline_graph, drifted_graph)
        weakened = [b for b in breaks if b.break_type in ("WEAKENED", "PROXY_COLLAPSE")]
        assert len(weakened) >= 1
        targets = [(b.cause, b.effect) for b in weakened]
        assert ("feature_a", "prediction") in targets

    def test_detects_removed_edge(self) -> None:
        """An edge in baseline that's absent from current is REMOVED_EDGE."""
        from datetime import datetime, timezone

        baseline = CausalGraph(
            nodes=["a", "b"],
            edges=[CausalEdge(cause="a", effect="b", strength=0.7, p_value=0.01, is_significant=True)],
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
        current = CausalGraph(
            nodes=["a", "b"],
            edges=[],
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
        breaks = detect_drift(baseline, current)
        assert any(b.break_type == "REMOVED_EDGE" and b.cause == "a" and b.effect == "b" for b in breaks)

    def test_detects_new_edge(self) -> None:
        """An edge in current that was absent from baseline is NEW_EDGE."""
        from datetime import datetime, timezone

        baseline = CausalGraph(
            nodes=["a", "b"],
            edges=[],
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
        current = CausalGraph(
            nodes=["a", "b"],
            edges=[CausalEdge(cause="a", effect="b", strength=0.8, p_value=0.001, is_significant=True)],
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
        breaks = detect_drift(baseline, current)
        assert any(b.break_type == "NEW_EDGE" and b.cause == "a" and b.effect == "b" for b in breaks)

    def test_returns_causal_break_instances(
        self, baseline_graph: CausalGraph, drifted_graph: CausalGraph
    ) -> None:
        """All returned items are CausalBreak instances."""
        breaks = detect_drift(baseline_graph, drifted_graph)
        for b in breaks:
            assert isinstance(b, CausalBreak)

    def test_break_significance_in_range(
        self, baseline_graph: CausalGraph, drifted_graph: CausalGraph
    ) -> None:
        """Break significance scores are in [0, 1]."""
        breaks = detect_drift(baseline_graph, drifted_graph)
        for b in breaks:
            assert 0.0 <= b.significance <= 1.0
