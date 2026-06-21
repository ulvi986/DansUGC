"""Robust JSON extraction, repair and validation.

LLMs frequently wrap JSON in prose or markdown fences, emit trailing commas,
or use single quotes. This module extracts the JSON payload and repairs the
common failure modes. It prefers the `json-repair` library when installed and
falls back to a self-contained repairer otherwise.
"""
from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from core.logging_config import get_logger

logger = get_logger("json_utils")

try:  # optional dependency, gracefully degraded
    from json_repair import repair_json as _lib_repair_json  # type: ignore
except Exception:  # pragma: no cover
    _lib_repair_json = None

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _slice_to_braces(text: str) -> str:
    """Return substring spanning the first '{' to the last matching '}'/']'."""
    start = min(
        [i for i in (text.find("{"), text.find("[")) if i != -1] or [-1]
    )
    if start == -1:
        return text
    end_brace = text.rfind("}")
    end_bracket = text.rfind("]")
    end = max(end_brace, end_bracket)
    return text[start : end + 1] if end > start else text[start:]


def _fallback_repair(text: str) -> str:
    """Self-contained repairs for the most common LLM JSON defects."""
    s = text
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Smart quotes -> straight quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # Python literals -> JSON literals
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return s


def extract_json(text: str) -> Any:
    """Best-effort parse of a JSON object/array out of arbitrary model text."""
    if text is None:
        raise ValueError("No text to parse")
    candidate = _slice_to_braces(_strip_fences(text)).strip()

    # 1) straight parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 2) library repair
    if _lib_repair_json is not None:
        try:
            repaired = _lib_repair_json(candidate)
            return json.loads(repaired) if isinstance(repaired, str) else repaired
        except Exception:  # pragma: no cover
            pass

    # 3) self-contained repair
    try:
        return json.loads(_fallback_repair(candidate))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse JSON from model output: {exc}") from exc


def validate(data: Any, model_cls: Type[T]) -> T:
    """Coerce a dict into a pydantic model (raises ValidationError on failure)."""
    if not isinstance(data, dict):
        raise ValidationError.from_exception_data(model_cls.__name__, [])
    return model_cls.model_validate(data)


def parse_and_validate(text: str, model_cls: Type[T]) -> T:
    return validate(extract_json(text), model_cls)
