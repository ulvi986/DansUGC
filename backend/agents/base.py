"""BaseAgent — the LLM-or-heuristic execution contract.

Every agent:
  * owns a dedicated system prompt (no giant shared prompt),
  * declares its strict Pydantic output schema,
  * implements a deterministic `heuristic()` fallback,
  * returns validated output plus a provenance tag ("llm" | "heuristic").

`analyze()` tries Gemini first (when enabled) and transparently degrades to the
heuristic on any failure, so the pipeline can never crash on a single agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel

from core.logging_config import get_logger
from services.llm_service import LLMService, LLMUnavailable

logger = get_logger("agent")


@dataclass
class AgentResult:
    output: BaseModel
    source: str          # "llm" | "heuristic"
    confidence: float


class BaseAgent:
    name: str = "base"
    output_model: Type[BaseModel]
    system_prompt: str = ""
    base_temperature: float = 0.4

    def __init__(self, llm: LLMService):
        self.llm = llm

    # --- to be implemented by subclasses ------------------------------------
    def build_prompt(self, ctx: dict) -> str:
        raise NotImplementedError

    def heuristic(self, ctx: dict) -> dict:
        raise NotImplementedError

    # --- shared execution ---------------------------------------------------
    def analyze(self, ctx: dict, temperature: float | None = None) -> AgentResult:
        temp = self.base_temperature if temperature is None else temperature
        if self.llm.enabled:
            try:
                out = self.llm.generate_json(
                    self.system_prompt, self.build_prompt(ctx), self.output_model, temp
                )
                return AgentResult(out, "llm", _confidence_of(out))
            except LLMUnavailable as exc:
                logger.warning("[%s] LLM unavailable, using heuristic: %s", self.name, exc)
            except Exception:  # defensive: never let one agent kill the run
                logger.exception("[%s] unexpected LLM error, using heuristic", self.name)

        out = self.output_model(**self.heuristic(ctx))
        return AgentResult(out, "heuristic", _confidence_of(out))


def _confidence_of(model: BaseModel) -> float:
    val = getattr(model, "confidence", None)
    if isinstance(val, (int, float)):
        return float(val)
    val = getattr(model, "confidence_score", None)
    if isinstance(val, (int, float)):
        return float(val)
    return 0.5
