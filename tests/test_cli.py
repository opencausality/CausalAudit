"""CLI tests for CausalAudit."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from causalaudit.cli import app

runner = CliRunner()


class TestAuditCommand:
    def test_audit_baseline_vs_drifted(
        self, baseline_log_path: Path, drifted_log_path: Path
    ) -> None:
        """audit command compares baseline and drifted logs."""
        result = runner.invoke(
            app,
            ["audit", str(baseline_log_path), str(drifted_log_path)],
        )
        assert result.exit_code == 0
        assert "drift" in result.output.lower() or "Break" in result.output or "Causal" in result.output

    def test_audit_saves_report(
        self, baseline_log_path: Path, drifted_log_path: Path, tmp_path: Path
    ) -> None:
        """--output flag saves the drift report JSON."""
        output = tmp_path / "drift_report.json"
        result = runner.invoke(
            app,
            ["audit", str(baseline_log_path), str(drifted_log_path), "--output", str(output)],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_audit_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        """Missing file causes non-zero exit."""
        result = runner.invoke(
            app, ["audit", str(tmp_path / "no.csv"), str(tmp_path / "no2.csv")]
        )
        assert result.exit_code != 0


class TestVersionFlag:
    def test_version_flag(self) -> None:
        """--version flag shows version info."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "CausalAudit" in result.output or "0." in result.output
