"""Statistical independence tests for causal discovery.

Implements partial correlation computation and Fisher's Z transformation
test, which are the core statistical primitives used by the PC-style
causal discovery algorithm in ``algorithms.py``.

References
----------
- Fisher (1915): "Frequency distribution of the values of the correlation
  coefficient in samples from an indefinitely large population."
- Spirtes, Glymour & Scheines (2000): "Causation, Prediction, and Search."
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger("causalaudit.discovery")


def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
) -> float:
    """Compute partial correlation between x and y controlling for z.

    When ``z`` is None, returns the standard Pearson correlation coefficient.
    When ``z`` is provided, removes the linear effect of z from both x and y
    via OLS regression residuals, then returns the correlation of the
    residuals.

    Parameters
    ----------
    x:
        First variable, shape (n,).
    y:
        Second variable, shape (n,).
    z:
        Conditioning variable(s).  Shape (n,) for a single variable or
        (n, k) for k conditioning variables.  Pass None for marginal
        correlation.

    Returns
    -------
    float
        Partial correlation coefficient in [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if z is None:
        # Standard Pearson correlation
        corr_matrix = np.corrcoef(x, y)
        return float(corr_matrix[0, 1])

    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    n = len(x)
    # Add intercept column
    Z = np.column_stack([np.ones(n), z])

    def _residuals(v: np.ndarray) -> np.ndarray:
        """Return OLS residuals of regressing v on Z."""
        try:
            coeffs, _, _, _ = np.linalg.lstsq(Z, v, rcond=None)
            return v - Z @ coeffs
        except np.linalg.LinAlgError:
            # Fall back to zero residuals if matrix is singular
            return np.zeros_like(v)

    res_x = _residuals(x)
    res_y = _residuals(y)

    # Correlation of residuals
    std_x = np.std(res_x)
    std_y = np.std(res_y)

    if std_x < 1e-10 or std_y < 1e-10:
        # One variable is constant after conditioning — no relationship
        return 0.0

    corr = float(np.corrcoef(res_x, res_y)[0, 1])
    # Clip to valid range to guard against floating-point overshoot
    return float(np.clip(corr, -1.0, 1.0))


def fisher_z_test(r: float, n: int) -> tuple[float, float]:
    """Fisher Z transformation significance test for a correlation coefficient.

    Tests H0: the population correlation is zero.

    The Fisher Z transformation is:
        Z = 0.5 * ln((1 + r) / (1 - r))

    Under H0 with large n, Z ~ N(0, 1 / sqrt(n - 3)).

    Parameters
    ----------
    r:
        Sample correlation coefficient in (-1, 1).
    n:
        Sample size.

    Returns
    -------
    tuple[float, float]
        ``(z_statistic, p_value)`` where p_value is two-tailed.
    """
    # Clamp r away from ±1 to avoid log(0)
    r_clamped = float(np.clip(r, -0.9999, 0.9999))

    if n <= 3:
        # Not enough degrees of freedom
        return 0.0, 1.0

    z_stat = 0.5 * np.log((1.0 + r_clamped) / (1.0 - r_clamped))
    se = 1.0 / np.sqrt(n - 3)
    z_score = z_stat / se

    # Two-tailed p-value from standard normal
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
    return float(z_score), p_value


def independence_test(
    x: np.ndarray,
    y: np.ndarray,
    conditioning: list[np.ndarray] | None = None,
) -> tuple[bool, float]:
    """Test whether x and y are conditionally independent given a conditioning set.

    Uses partial correlation as the test statistic, then applies the Fisher Z
    test to assess significance.

    Parameters
    ----------
    x:
        First variable, shape (n,).
    y:
        Second variable, shape (n,).
    conditioning:
        List of conditioning variables, each of shape (n,).  Pass None or
        an empty list to test marginal independence.

    Returns
    -------
    tuple[bool, float]
        ``(is_independent, p_value)`` where ``is_independent`` is True when
        p_value > 0.05 (i.e., we fail to reject independence).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if conditioning and len(conditioning) > 0:
        z = np.column_stack([np.asarray(c, dtype=float) for c in conditioning])
        r = partial_correlation(x, y, z)
    else:
        r = partial_correlation(x, y, None)

    _, p_value = fisher_z_test(r, n)

    # H0: independent.  Reject H0 (i.e., conclude dependence) when p < 0.05.
    is_independent = p_value > 0.05

    logger.debug(
        "Independence test: r=%.4f, n=%d, p=%.4f, independent=%s",
        r, n, p_value, is_independent,
    )
    return is_independent, p_value
