"""LLM abstraction layer supporting Gemini and OpenAI with heuristic fallback."""
from __future__ import annotations

import json
import time
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import Settings, get_settings
from core.logging_config import get_logger
from core.json_utils import extract_json, validate

logger = get_logger("llm")

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when an LLM call cannot be satisfied (no key, or repeated failure)."""


class BaseLLMService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._disabled = False  # set on fatal auth errors

    @property
    def enabled(self) -> bool:
        raise NotImplementedError

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model_cls: Type[T],
        temperature: float = 0.4,
    ) -> T:
        raise NotImplementedError


class GeminiLLMService(BaseLLMService):
    def __init__(self, settings: Settings | None = None):
        super().__init__(settings)
        self._client = None
        self._types = None
        if self.settings.llm_enabled and self.settings.gemini_api_key:
            try:
                from google import genai
                from google.genai import types

                self._client = genai.Client(api_key=self.settings.gemini_api_key)
                self._types = types
                logger.info("Gemini LLM enabled (model=%s)", self.settings.gemini_model)
            except Exception:
                logger.exception("Gemini init failed; falling back to heuristic")
                self._client = None
        else:
            logger.info("No GEMINI_API_KEY set — running in heuristic mode for Gemini path")

    @property
    def enabled(self) -> bool:
        return self._client is not None and not self._disabled

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
        if not self.enabled:
            raise LLMUnavailable("Gemini LLM disabled")

        last_err: Exception | None = None
        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                raw = self._generate_text(system_prompt, user_prompt, temperature)
                data = extract_json(raw)
                return validate(data, model_cls)
            except ValidationError as exc:
                last_err = exc
                logger.warning(
                    "Gemini validation failed (attempt %s/%s) for %s: %s",
                    attempt,
                    self.settings.llm_max_retries,
                    model_cls.__name__,
                    exc,
                )
                user_prompt = (
                    f"{user_prompt}\n\nYour previous response did not match the "
                    f"required schema. Return ONLY valid JSON with the exact keys."
                )
            except Exception as exc:
                last_err = exc
                if _is_fatal_auth(exc):
                    self._disabled = True
                    logger.error(
                        "Gemini auth error - disabling LLM for this process: %s",
                        str(exc)[:160],
                    )
                    raise LLMUnavailable("Gemini authentication failed") from exc
                if _is_rate_limited(exc):
                    backoff = min(2 ** attempt, 30)
                    logger.warning("Gemini rate limited; backing off %ss", backoff)
                    time.sleep(backoff)
                else:
                    logger.warning(
                        "Gemini LLM call failed (attempt %s/%s): %s",
                        attempt,
                        self.settings.llm_max_retries,
                        exc,
                    )
                    time.sleep(min(attempt, 5))
        raise LLMUnavailable(f"Exhausted retries for {model_cls.__name__}: {last_err}")


class OpenAILLMService(BaseLLMService):
    def __init__(self, settings: Settings | None = None):
        super().__init__(settings)
        self._client = None
        if self.settings.openai_enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.settings.openai_api_key)
                logger.info(
                    "OpenAI LLM enabled (model=%s)", self.settings.openai_model
                )
            except Exception:
                logger.exception("OpenAI init failed; falling back to heuristic")
                self._client = None
        else:
            logger.info("No usable OPENAI_API_KEY set — skipping OpenAI LLM")

    @property
    def enabled(self) -> bool:
        return self._client is not None and not self._disabled

    def _generate_text(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        # Use chat.completions with response_format JSON object
        resp = self._client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model_cls: Type[T],
        temperature: float = 0.4,
    ) -> T:
        if not self.enabled:
            raise LLMUnavailable("OpenAI LLM disabled")

        last_err: Exception | None = None
        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                raw = self._generate_text(system_prompt, user_prompt, temperature)
                data = extract_json(raw)
                return validate(data, model_cls)
            except ValidationError as exc:
                last_err = exc
                logger.warning(
                    "OpenAI validation failed (attempt %s/%s) for %s: %s",
                    attempt,
                    self.settings.llm_max_retries,
                    model_cls.__name__,
                    exc,
                )
                user_prompt = (
                    f"{user_prompt}\n\nYour previous response did not match the "
                    f"required schema. Return ONLY valid JSON with the exact keys."
                )
            except Exception as exc:
                last_err = exc
                # Simple retry handling for OpenAI; treat 401/403 as fatal
                if getattr(exc, "status_code", None) in (401, 403):
                    self._disabled = True
                    logger.error(
                        "OpenAI auth error - disabling LLM for this process: %s",
                        str(exc)[:160],
                    )
                    raise LLMUnavailable("OpenAI authentication failed") from exc
                if getattr(exc, "status_code", None) == 429:
                    backoff = min(2 ** attempt, 30)
                    logger.warning("OpenAI rate limited; backing off %ss", backoff)
                    time.sleep(backoff)
                else:
                    logger.warning(
                        "OpenAI LLM call failed (attempt %s/%s): %s",
                        attempt,
                        self.settings.llm_max_retries,
                        exc,
                    )
                    time.sleep(min(attempt, 5))
        raise LLMUnavailable(f"Exhausted retries for {model_cls.__name__}: {last_err}")


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("429" in text) or ("rate" in text) or ("quota" in text) or ("resource_exhausted" in text)


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


class LLMService:
    """Facade that picks the first available LLM provider (OpenAI > Gemini)."""
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.openai = OpenAILLMService(self.settings)
        self.gemini = GeminiLLMService(self.settings)

    @property
    def enabled(self) -> bool:
        return self.openai.enabled or self.gemini.enabled

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model_cls: Type[T],
        temperature: float = 0.4,
    ) -> T:
        # Try OpenAI first, then Gemini
        if self.openai.enabled:
            try:
                return self.openai.generate_json(
                    system_prompt, user_prompt, model_cls, temperature
                )
            except LLMUnavailable:
                pass  # fall through to Gemini
        if self.gemini.enabled:
            return self.gemini.generate_json(
                system_prompt, user_prompt, model_cls, temperature
            )
        raise LLMUnavailable("No LLM provider enabled")


# Process-wide singleton
_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _service
    if _service is None:
        _service = LLMService()
    return _service