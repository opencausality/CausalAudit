"""Tests for the data loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from causalaudit.data.loader import load_inference_log
from causalaudit.data.schema import InferenceLog


class TestLoadInferenceLog:
    def test_loads_baseline_fixture(self, baseline_log_path: Path) -> None:
        """Loads the baseline inference log fixture correctly."""
        log = load_inference_log(baseline_log_path)
        assert isinstance(log, InferenceLog)
        assert log.n_rows > 0

    def test_loads_drifted_fixture(self, drifted_log_path: Path) -> None:
        """Loads the drifted inference log fixture correctly."""
        log = load_inference_log(drifted_log_path)
        assert isinstance(log, InferenceLog)
        assert log.n_rows > 0

    def test_columns_contain_features(self, baseline_log_path: Path) -> None:
        """Loaded log has feature columns."""
        log = load_inference_log(baseline_log_path)
        assert len(log.columns) > 0

    def test_label_column_in_columns(self, baseline_log_path: Path) -> None:
        """label_column is present in columns dict."""
        log = load_inference_log(baseline_log_path)
        assert log.label_column in log.columns

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError or similar."""
        with pytest.raises(Exception):
            load_inference_log(tmp_path / "nonexistent.csv")

    def test_inference_log_n_rows_property(self, baseline_log_path: Path) -> None:
        """n_rows property matches actual column length."""
        log = load_inference_log(baseline_log_path)
        first_col = next(iter(log.columns.values()))
        assert log.n_rows == len(first_col)
