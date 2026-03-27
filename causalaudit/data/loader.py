"""Inference log loading utilities.

Provides functions to read CSV inference logs into ``InferenceLog`` objects
and to split them temporally for baseline/current comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from causalaudit.data.schema import InferenceLog
from causalaudit.exceptions import DataLoadError, InsufficientDataError

logger = logging.getLogger("causalaudit.data")

# Minimum number of rows required for meaningful causal discovery.
MIN_ROWS = 50


def load_inference_logs(path: Path, label_column: str) -> InferenceLog:
    """Load a CSV file of inference records into an ``InferenceLog``.

    The CSV must have:
    - At least ``MIN_ROWS`` rows.
    - A column matching ``label_column``.
    - All columns must be numeric (or castable to float).

    Parameters
    ----------
    path:
        Filesystem path to the CSV file.
    label_column:
        Name of the column containing the model's output label / prediction.

    Returns
    -------
    InferenceLog
        Validated inference log ready for causal discovery.

    Raises
    ------
    DataLoadError
        If the file cannot be read, is missing ``label_column``, or contains
        non-numeric data that cannot be coerced.
    InsufficientDataError
        If the CSV has fewer than ``MIN_ROWS`` rows after loading.
    """
    logger.info("Loading inference log from %s", path)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataLoadError(f"Failed to read CSV at '{path}': {exc}") from exc

    logger.debug("Loaded %d rows × %d columns from %s", len(df), len(df.columns), path)

    if label_column not in df.columns:
        raise DataLoadError(
            f"label_column '{label_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if len(df) < MIN_ROWS:
        raise InsufficientDataError(
            f"Dataset has only {len(df)} rows; at least {MIN_ROWS} are required "
            "for reliable causal discovery."
        )

    # Coerce all columns to numeric; non-numeric values become NaN
    non_numeric: list[str] = []
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any() and df[col].dtype == object:
            non_numeric.append(col)
        df[col] = coerced

    if non_numeric:
        raise DataLoadError(
            f"Columns contain non-numeric values that cannot be coerced: {non_numeric}. "
            "Encode categorical features before running CausalAudit."
        )

    # Drop rows with any NaN after coercion
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        logger.warning(
            "Dropped %d rows containing NaN values (%d rows remain)",
            before - len(df), len(df),
        )

    if len(df) < MIN_ROWS:
        raise InsufficientDataError(
            f"After dropping NaN rows only {len(df)} rows remain; "
            f"at least {MIN_ROWS} are required."
        )

    columns: dict[str, list[float]] = {
        col: df[col].tolist() for col in df.columns
    }

    log = InferenceLog(columns=columns, label_column=label_column)
    logger.info(
        "Loaded %d rows, %d features + label '%s'",
        log.n_rows, len(log.feature_columns), label_column,
    )
    return log


def split_temporal(
    log: InferenceLog,
    split_ratio: float = 0.5,
) -> tuple[InferenceLog, InferenceLog]:
    """Split an ``InferenceLog`` into baseline and current halves.

    The split is strictly temporal: the first ``split_ratio`` fraction of
    rows becomes the baseline, the remainder becomes the current window.
    This preserves any time ordering present in the data.

    Parameters
    ----------
    log:
        The full inference log to split.
    split_ratio:
        Fraction of rows to use as the baseline (0.0 < split_ratio < 1.0).

    Returns
    -------
    tuple[InferenceLog, InferenceLog]
        ``(baseline, current)`` where baseline covers the earlier rows.

    Raises
    ------
    InsufficientDataError
        If either split produces fewer than ``MIN_ROWS`` rows.
    """
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")

    n = log.n_rows
    split_idx = int(n * split_ratio)

    if split_idx < MIN_ROWS:
        raise InsufficientDataError(
            f"Baseline split would have only {split_idx} rows; "
            f"at least {MIN_ROWS} required."
        )
    if n - split_idx < MIN_ROWS:
        raise InsufficientDataError(
            f"Current split would have only {n - split_idx} rows; "
            f"at least {MIN_ROWS} required."
        )

    def _slice(start: int, end: int) -> InferenceLog:
        sliced_cols = {
            col: vals[start:end] for col, vals in log.columns.items()
        }
        sliced_ts: list[str] | None = None
        if log.timestamps is not None:
            sliced_ts = log.timestamps[start:end]
        return InferenceLog(
            columns=sliced_cols,
            label_column=log.label_column,
            timestamps=sliced_ts,
        )

    baseline = _slice(0, split_idx)
    current = _slice(split_idx, n)

    logger.info(
        "Split %d rows → baseline=%d, current=%d (ratio=%.2f)",
        n, baseline.n_rows, current.n_rows, split_ratio,
    )
    return baseline, current
