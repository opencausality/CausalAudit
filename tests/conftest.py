"""Shared pytest fixtures for CausalAudit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from causalaudit.data.schema import CausalBreak, CausalEdge, CausalGraph, DriftReport

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LOGS_DIR = FIXTURES_DIR / "sample_logs"


@pytest.fixture
def baseline_log_path() -> Path:
    return LOGS_DIR / "baseline_inference.csv"


@pytest.fixture
def drifted_log_path() -> Path:
    return LOGS_DIR / "drifted_inference.csv"


@pytest.fixture
def baseline_graph() -> CausalGraph:
    """Baseline causal graph with 3 edges."""
    return CausalGraph(
        nodes=["feature_a", "feature_b", "prediction"],
        edges=[
            CausalEdge(cause="feature_a", effect="prediction", strength=0.8, p_value=0.001, is_significant=True),
            CausalEdge(cause="feature_b", effect="prediction", strength=0.6, p_value=0.01, is_significant=True),
            CausalEdge(cause="feature_a", effect="feature_b", strength=0.4, p_value=0.03, is_significant=True),
        ],
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        model_version="v1.0",
    )


@pytest.fixture
def drifted_graph() -> CausalGraph:
    """Current graph where feature_a → prediction has weakened."""
    return CausalGraph(
        nodes=["feature_a", "feature_b", "prediction"],
        edges=[
            CausalEdge(cause="feature_a", effect="prediction", strength=0.2, p_value=0.1, is_significant=False),
            CausalEdge(cause="feature_b", effect="prediction", strength=0.75, p_value=0.001, is_significant=True),
            CausalEdge(cause="feature_a", effect="feature_b", strength=0.45, p_value=0.02, is_significant=True),
        ],
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        model_version="v1.1",
    )
