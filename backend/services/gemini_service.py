"""Gemini gateway.

Responsibilities
----------------
* Single integration point for the new google-genai SDK (`from google import genai`).
* JSON-only generation with: schema-constrained output, JSON extraction/repair,
  Pydantic validation, and a retry loop.
* Exponential backoff on rate limits / transient errors.
* Graceful degradation: if no API key is configured the service reports
  `enabled = False` and agents transparently fall back to heuristic analysis.

The rest of the codebase never imports the SDK directly — only this class.
"""
from __future__ import annotations

import time
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import Settings, get_settings
from core.json_utils import extract_json, validate
from core.logging_config import get_logger

logger = get_logger("gemini")

T = TypeVar("T", bound=BaseModel)


class GeminiUnavailable(RuntimeError):
    """Raised when an LLM call cannot be satisfied (no key, or repeated failure)."""


class GeminiService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None
        self._types = None
        self._disabled = False   # set on fatal auth errors to stop wasteful retries
        if self.settings.llm_enabled:
            try:
                from google import genai
                from google.genai import types

                self._client = genai.Client(api_key=self.settings.gemini_api_key)
                self._types = types
                logger.info("Gemini enabled (model=%s)", self.settings.gemini_model)
            except Exception:  # pragma: no cover - import/credential issues
                logger.exception("Gemini init failed; falling back to heuristic mode")
                self._client = None
        else:
            logger.info("No GEMINI_API_KEY set — running in deterministic heuristic mode")

    @property
    def enabled(self) -> bool:
        return self._client is not None and not self._disabled

    # ------------------------------------------------------------------ #
    def _generate_text(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            response_mime_type="application/json",
        )
        resp = self._client.models.generate_content(
            model=self.settings.gemini_model,
            contents=user_prompt,
            config=config,
        )
        return getattr(resp, "text", "") or ""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model_cls: Type[T],
        temperature: float = 0.4,
    ) -> T:
        """Return a validated `model_cls` instance, repairing/retrying as needed."""
        if not self.enabled:
            raise GeminiUnavailable("LLM disabled")

        last_err: Exception | None = None
        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                raw = self._generate_text(system_prompt, user_prompt, temperature)
                data = extract_json(raw)
                return validate(data, model_cls)
            except ValidationError as exc:
                last_err = exc
                logger.warning(
                    "Validation failed (attempt %s/%s) for %s: %s",
                    attempt, self.settings.llm_max_retries, model_cls.__name__, exc,
                )
                # On retry, nudge the model to fix its own output.
                user_prompt = (
                    f"{user_prompt}\n\nYour previous response did not match the "
                    f"required schema. Return ONLY valid JSON with the exact keys."
                )
            except Exception as exc:  # network / rate-limit / parse
                last_err = exc
                if _is_fatal_auth(exc):
                    # Invalid/expired key or permission denied: not transient.
                    # Disable the LLM for the rest of the process so every
                    # subsequent agent skips straight to heuristic mode.
                    self._disabled = True
                    logger.error(
                        "Gemini auth error - disabling LLM for this process, "
                        "using heuristic mode. (%s)", str(exc)[:160],
                    )
                    raise GeminiUnavailable("Gemini authentication failed") from exc
                if _is_rate_limited(exc):
                    backoff = min(2 ** attempt, 30)
                    logger.warning("Rate limited; backing off %ss", backoff)
                    time.sleep(backoff)
                else:
                    logger.warning(
                        "LLM call failed (attempt %s/%s): %s",
                        attempt, self.settings.llm_max_retries, exc,
                    )
                    time.sleep(min(attempt, 5))

        raise GeminiUnavailable(f"Exhausted retries for {model_cls.__name__}: {last_err}")


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate" in text or "quota" in text or "resource_exhausted" in text


def _is_fatal_auth(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "api_key_invalid" in text
        or "api key not found" in text
        or "permission_denied" in text
        or "unauthenticated" in text
        or "401" in text
        or "403" in text
    )


# Process-wide singleton (cheap; the SDK client is thread-safe for our usage).
_service: GeminiService | None = None


def get_gemini_service() -> GeminiService:
    global _service
    if _service is None:
        _service = GeminiService()
    return _service
