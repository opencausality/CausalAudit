"""LLM adapter — unified interface to all providers via LiteLLM.

Design rules (mirrored from WhyNet spec):
- Always use ``litellm.completion()`` — never call provider SDKs directly.
- Resolve provider priority: explicit config → env var (CAUSALAUDIT_LLM_PROVIDER) → Ollama default.
- Ollama models are prefixed ``ollama/`` for LiteLLM.
- Retry with exponential backoff (3 attempts).
- Log which model was used for every call.
- Raise ``ProviderError`` if no provider is reachable after all retries.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import litellm

from causalaudit.config import Settings, get_settings
from causalaudit.exceptions import ProviderError

logger = logging.getLogger("causalaudit.llm")

# Suppress LiteLLM's own verbose logging
litellm.suppress_debug_info = True


class LLMAdapter:
    """Thin wrapper around LiteLLM that handles retries, model resolution,
    and transparent logging for every call.

    Parameters
    ----------
    settings:
        Application settings.  Uses the global singleton when omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = self._settings.litellm_model
        self._temperature = self._settings.llm_temperature
        self._max_retries = self._settings.llm_max_retries
        logger.info(
            "LLMAdapter initialised → model=%s, provider=%s",
            self._model, self._settings.llm_provider.value,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Send a prompt to the configured LLM and return the text response.

        Parameters
        ----------
        prompt:
            The user message / main prompt.
        system:
            Optional system prompt prepended to the conversation.
        temperature:
            Override instance-level temperature for this call.
        max_tokens:
            Upper bound on response length.

        Returns
        -------
        str
            The model's response text (stripped of leading/trailing whitespace).

        Raises
        ------
        ProviderError
            If the call fails after all retry attempts.
        """
        messages = self._build_messages(prompt, system)
        temp = temperature if temperature is not None else self._temperature

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    "LLM call attempt %d/%d → model=%s",
                    attempt, self._max_retries, self._model,
                )
                response = litellm.completion(
                    model=self._model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                content = response.choices[0].message.content or ""
                logger.info(
                    "LLM call succeeded → model=%s, tokens_used=%s",
                    self._model,
                    getattr(response.usage, "total_tokens", "?"),
                )
                return content.strip()

            except Exception as exc:
                last_error = exc
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s …
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, self._max_retries, exc, wait,
                )
                if attempt < self._max_retries:
                    time.sleep(wait)

        raise ProviderError(
            f"All {self._max_retries} LLM call attempts failed. "
            f"Provider={self._settings.llm_provider.value}, model={self._model}. "
            f"Last error: {last_error}"
        )

    # ── Provider introspection ────────────────────────────────────────────

    def provider_info(self) -> dict[str, str]:
        """Return a summary dict of the current provider configuration."""
        return {
            "provider": self._settings.llm_provider.value,
            "model": self._model,
            "temperature": str(self._temperature),
            "max_retries": str(self._max_retries),
        }

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        prompt: str,
        system: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the LiteLLM messages list."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages
