"""Pydantic data models for CausalAudit.

These models form the core data contracts flowing through the pipeline:
  InferenceLog → CausalGraph → CausalBreak → DriftReport
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Input data models ─────────────────────────────────────────────────────────


class InferenceLog(BaseModel):
    """A batch of model inference records suitable for causal analysis.

    Attributes
    ----------
    columns:
        Feature columns as {column_name: [float, ...]}.  All lists must share
        the same length.
    label_column:
        Name of the prediction / outcome column (must exist in ``columns``).
    timestamps:
        Optional ISO-8601 timestamp strings, one per row.
    """

    columns: dict[str, list[float]]
    label_column: str
    timestamps: list[str] | None = None

    @field_validator("label_column")
    @classmethod
    def _label_in_columns(cls, v: str, info: "object") -> str:  # type: ignore[override]
        # Pydantic v2 passes FieldValidationInfo; we access data via info.data
        data = getattr(info, "data", {})
        cols = data.get("columns", {})
        if cols and v not in cols:
            raise ValueError(
                f"label_column '{v}' not found in columns: {list(cols.keys())}"
            )
        return v

    @property
    def n_rows(self) -> int:
        """Return the number of rows in the log."""
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    @property
    def feature_columns(self) -> list[str]:
        """Return all column names excluding the label column."""
        return [c for c in self.columns if c != self.label_column]


# ── Graph models ──────────────────────────────────────────────────────────────


class CausalEdge(BaseModel):
    """A directed causal relationship discovered from data.

    Attributes
    ----------
    cause:
        The originating variable name.
    effect:
        The downstream variable name.
    strength:
        Absolute partial correlation (0.0–1.0).  Higher = stronger.
    p_value:
        P-value from the Fisher Z significance test.
    is_significant:
        True when ``p_value`` is below the configured significance threshold.
    """

    cause: str
    effect: str
    strength: float = Field(ge=0.0, le=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    is_significant: bool


class CausalGraph(BaseModel):
    """A snapshot of causal structure at a point in time.

    Attributes
    ----------
    nodes:
        All variable names present in the graph.
    edges:
        Discovered causal edges.
    timestamp:
        ISO-8601 string indicating when this graph was computed.
    model_version:
        Optional tag identifying which model version produced these logs.
    """

    nodes: list[str]
    edges: list[CausalEdge]
    timestamp: str
    model_version: str = ""

    def edge_map(self) -> dict[tuple[str, str], CausalEdge]:
        """Return a dict keyed by (cause, effect) for fast lookup."""
        return {(e.cause, e.effect): e for e in self.edges}

    def significant_edges(self) -> list[CausalEdge]:
        """Return only edges marked as statistically significant."""
        return [e for e in self.edges if e.is_significant]


# ── Drift models ──────────────────────────────────────────────────────────────


class CausalBreak(BaseModel):
    """A detected structural change between the baseline and current graphs.

    Break types:
    - ``NEW_EDGE``: A causal edge appeared that was not present at baseline.
    - ``REMOVED_EDGE``: A baseline edge is no longer present in current data.
    - ``STRENGTHENED``: An edge became significantly stronger.
    - ``WEAKENED``: An edge became significantly weaker.
    - ``PROXY_COLLAPSE``: A previously strong edge dropped near zero (feature
      may have become a proxy or been corrupted).

    Attributes
    ----------
    break_type:
        Category of structural change.
    cause / effect:
        The variables involved in the changed edge.
    baseline_strength:
        Edge strength in the baseline graph (None for NEW_EDGE).
    current_strength:
        Edge strength in the current graph (None for REMOVED_EDGE).
    significance:
        How significant this break is, in [0, 1].  Computed from the
        magnitude of change relative to the baseline.
    explanation:
        Human-readable description of what changed and why it matters.
    """

    break_type: Literal["NEW_EDGE", "REMOVED_EDGE", "STRENGTHENED", "WEAKENED", "PROXY_COLLAPSE"]
    cause: str
    effect: str
    baseline_strength: float | None = None
    current_strength: float | None = None
    significance: float = Field(ge=0.0, le=1.0)
    explanation: str


class DriftReport(BaseModel):
    """Full causal drift analysis report.

    Attributes
    ----------
    baseline_graph:
        Causal structure learned from historical data.
    current_graph:
        Causal structure learned from recent data.
    breaks:
        All detected structural breaks, sorted by significance.
    degradation_score:
        Aggregate severity score in [0, 1].  0 = no drift, 1 = complete
        structural collapse.
    root_cause:
        Primary explanation of the dominant causal break.
    recommendations:
        Ordered list of actionable next steps.
    model_used:
        LLM model used for narrative explanation (empty if no LLM was called).
    """

    baseline_graph: CausalGraph
    current_graph: CausalGraph
    breaks: list[CausalBreak]
    degradation_score: float = Field(ge=0.0, le=1.0)
    root_cause: str
    recommendations: list[str]
    model_used: str = ""
