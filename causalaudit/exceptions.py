"""CausalAudit exception hierarchy.

All library-specific exceptions inherit from ``CausalAuditError`` so that
callers can catch the entire family with a single ``except`` clause.
"""

from __future__ import annotations


class CausalAuditError(Exception):
    """Base exception for all CausalAudit errors."""


class ProviderError(CausalAuditError):
    """Raised when no LLM provider is reachable or a call fails after retries."""


class DataLoadError(CausalAuditError):
    """Raised when an inference log file cannot be loaded or parsed."""


class InsufficientDataError(CausalAuditError):
    """Raised when a dataset has fewer than 50 rows, which is too few for
    reliable causal discovery via independence tests.
    """
