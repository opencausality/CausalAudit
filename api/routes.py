"""FastAPI route definitions for the CausalAudit REST API.

Endpoints:
  GET  /health          — Liveness check
  POST /audit           — Run full causal drift audit
  POST /baseline        — Build and return a baseline graph from uploaded CSV
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from causalaudit.attribution.engine import attribute_degradation
from causalaudit.config import get_settings
from causalaudit.data.loader import load_inference_logs
from causalaudit.data.schema import DriftReport
from causalaudit.drift.classifier import classify_root_cause, compute_degradation_score
from causalaudit.drift.detector import detect_drift
from causalaudit.exceptions import CausalAuditError, InsufficientDataError
from causalaudit.monitoring.baseline import build_baseline
from causalaudit.monitoring.tracker import build_current_graph

logger = logging.getLogger("causalaudit.api")

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────


@router.get("/health", summary="Liveness check")
async def health() -> JSONResponse:
    """Return a simple health-check payload."""
    from causalaudit import __version__

    return JSONResponse(
        {
            "status": "ok",
            "service": "causalaudit",
            "version": __version__,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    )


# ── Audit ─────────────────────────────────────────────────────────────────────


@router.post("/audit", summary="Run causal drift audit", response_model=DriftReport)
async def audit(
    baseline_csv: Annotated[UploadFile, File(description="Baseline inference log CSV")],
    current_csv: Annotated[UploadFile, File(description="Current inference log CSV")],
    target: Annotated[str, Form(description="Name of the label / outcome column")],
    explain: Annotated[bool, Form(description="Use LLM to generate narrative explanation")] = False,
) -> DriftReport:
    """Run a full causal drift audit comparing baseline and current inference logs.

    Accepts two CSV files (baseline and current) plus the target column name.
    Returns a ``DriftReport`` containing detected structural breaks, a
    degradation score, root cause explanation, and recommendations.

    Parameters
    ----------
    baseline_csv:
        CSV file of historical (healthy) model inference records.
    current_csv:
        CSV file of recent production inference records.
    target:
        Column name of the prediction / label variable.
    explain:
        If True, use the configured LLM to generate a narrative explanation.

    Raises
    ------
    HTTPException 400:
        If either CSV is invalid, missing the target column, or has too few rows.
    HTTPException 500:
        For unexpected internal errors.
    """
    settings = get_settings()

    try:
        baseline_content = await baseline_csv.read()
        current_content = await current_csv.read()

        baseline_log = _load_csv_bytes(baseline_content, target, "baseline")
        current_log = _load_csv_bytes(current_content, target, "current")

    except (CausalAuditError, ValueError) as exc:
        logger.warning("Audit request rejected — data load error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        logger.info("Starting causal audit (target=%s)", target)
        baseline_graph = build_baseline(baseline_log, settings)
        current_graph = build_current_graph(current_log, settings)
        breaks = detect_drift(baseline_graph, current_graph, settings)
        degradation_score = compute_degradation_score(breaks, baseline_graph)
        root_cause = classify_root_cause(breaks, baseline_graph)
        model_used = ""

        if explain and breaks:
            from causalaudit.llm.adapter import LLMAdapter
            from causalaudit.llm.prompts import DRIFT_EXPLANATION_PROMPT

            try:
                adapter = LLMAdapter(settings)
                breaks_desc = "\n".join(
                    f"- {b.break_type}: {b.cause} → {b.effect} "
                    f"(significance={b.significance:.2f})"
                    for b in breaks[:10]
                )
                prompt = DRIFT_EXPLANATION_PROMPT.format(
                    breaks_description=breaks_desc,
                    target=target,
                    baseline_period=baseline_graph.timestamp,
                    current_period=current_graph.timestamp,
                )
                adapter.complete(prompt)
                model_used = adapter.provider_info()["model"]
            except Exception as llm_exc:
                logger.warning("LLM explanation failed (non-fatal): %s", llm_exc)

        report = DriftReport(
            baseline_graph=baseline_graph,
            current_graph=current_graph,
            breaks=breaks,
            degradation_score=degradation_score,
            root_cause=root_cause,
            recommendations=attribute_degradation(
                DriftReport(
                    baseline_graph=baseline_graph,
                    current_graph=current_graph,
                    breaks=breaks,
                    degradation_score=degradation_score,
                    root_cause=root_cause,
                    recommendations=[],
                )
            ),
            model_used=model_used,
        )

        logger.info(
            "Audit complete: degradation_score=%.3f, breaks=%d",
            report.degradation_score, len(report.breaks),
        )
        return report

    except Exception as exc:
        logger.exception("Unexpected error during audit: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


# ── Baseline ──────────────────────────────────────────────────────────────────


@router.post("/baseline", summary="Build a baseline causal graph")
async def baseline_endpoint(
    data_csv: Annotated[UploadFile, File(description="Historical inference log CSV")],
    target: Annotated[str, Form(description="Name of the label / outcome column")],
) -> JSONResponse:
    """Build and return a baseline causal graph from an uploaded CSV.

    Useful for pre-computing and caching the baseline outside of a full audit.
    """
    settings = get_settings()

    try:
        content = await data_csv.read()
        log = _load_csv_bytes(content, target, "baseline")
    except (CausalAuditError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        graph = build_baseline(log, settings)
        return JSONResponse(graph.model_dump())
    except Exception as exc:
        logger.exception("Failed to build baseline: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_csv_bytes(content: bytes, target: str, name: str) -> "object":
    """Write CSV bytes to a temporary in-memory path and load via loader."""
    import tempfile
    from pathlib import Path

    from causalaudit.data.loader import load_inference_logs

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        log = load_inference_logs(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)

    return log
