"""Tests for causal graph discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalaudit.data.loader import load_inference_log
from causalaudit.data.schema import CausalGraph
from causalaudit.graph.builder import build_causal_graph


class TestBuildCausalGraph:
    def test_returns_causal_graph(self, baseline_log_path: Path) -> None:
        """build_causal_graph returns a CausalGraph instance."""
        log = load_inference_log(baseline_log_path)
        graph = build_causal_graph(log)
        assert isinstance(graph, CausalGraph)

    def test_nodes_include_features(self, baseline_log_path: Path) -> None:
        """Graph nodes include all feature columns."""
        log = load_inference_log(baseline_log_path)
        graph = build_causal_graph(log)
        for col in log.feature_columns:
            assert col in graph.nodes

    def test_edges_are_causal_edges(self, baseline_log_path: Path) -> None:
        """All edges in the graph reference valid nodes."""
        log = load_inference_log(baseline_log_path)
        graph = build_causal_graph(log)
        for edge in graph.edges:
            assert edge.cause in graph.nodes
            assert edge.effect in graph.nodes

    def test_significant_edges_method(self, baseline_log_path: Path) -> None:
        """significant_edges() filters correctly."""
        log = load_inference_log(baseline_log_path)
        graph = build_causal_graph(log)
        sig = graph.significant_edges()
        assert all(e.is_significant for e in sig)

    def test_edge_strength_in_range(self, baseline_log_path: Path) -> None:
        """All edge strengths are in [0, 1]."""
        log = load_inference_log(baseline_log_path)
        graph = build_causal_graph(log)
        for edge in graph.edges:
            assert 0.0 <= edge.strength <= 1.0

    def test_graph_timestamp_set(self, baseline_log_path: Path) -> None:
        """Graph timestamp is a non-empty string."""
        log = load_inference_log(baseline_log_path)
        graph = build_causal_graph(log)
        assert isinstance(graph.timestamp, str)
        assert len(graph.timestamp) > 0
