"""Application configuration via Pydantic settings.

Reads from environment variables and ``.env`` files.  Provider resolution
follows the priority chain: explicit config → env var → Ollama default.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("causalaudit")

# ── Supported LLM providers ──────────────────────────────────────────────────


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    MISTRAL = "mistral"
    TOGETHER_AI = "together_ai"


# ── Default model per provider ────────────────────────────────────────────────

DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OLLAMA: "llama3.1",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-5-20250929",
    LLMProvider.GROQ: "llama-3.1-70b-versatile",
    LLMProvider.MISTRAL: "mistral-large-latest",
    LLMProvider.TOGETHER_AI: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
}

# ── Settings ──────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Central configuration for CausalAudit.

    Values are loaded in this priority order:
    1. Explicit constructor arguments
    2. Environment variables (prefixed ``CAUSALAUDIT_``)
    3. ``.env`` file in the working directory
    4. Built-in defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="CAUSALAUDIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM settings ─────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="LLM provider to use for drift explanation.",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Model name. Defaults to provider-specific recommendation.",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Low values yield consistent explanations.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max retry attempts for failed LLM calls.",
    )

    # ── Causal discovery settings ─────────────────────────────────────────
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum edge strength for including a causal edge.",
    )
    significance_level: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="P-value threshold for statistical significance of edges.",
    )
    baseline_window: int = Field(
        default=1000,
        ge=50,
        description="Minimum number of rows required for baseline computation.",
    )

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # ── API server ───────────────────────────────────────────────────────
    api_host: str = Field(default="127.0.0.1", description="API listen address.")
    api_port: int = Field(default=8000, description="API listen port.")

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log level: {v}")
        return v

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def resolved_model(self) -> str:
        """Return the model name, falling back to the provider default."""
        if self.llm_model:
            return self.llm_model
        return DEFAULT_MODELS[self.llm_provider]

    @property
    def litellm_model(self) -> str:
        """Return the model string expected by LiteLLM.

        Ollama models must be prefixed with ``ollama/``.
        """
        model = self.resolved_model
        if self.llm_provider == LLMProvider.OLLAMA and not model.startswith("ollama/"):
            return f"ollama/{model}"
        return model


# ── Singleton accessor ────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()


def configure_logging(settings: Settings | None = None) -> None:
    """Set up Python logging based on application settings."""
    settings = settings or get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "litellm", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.debug("CausalAudit logging configured at %s level", settings.log_level)
