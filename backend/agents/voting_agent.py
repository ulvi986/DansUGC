"""Agent 8 — Voting / Consensus.

Runs any voting-eligible agent N independent times (with varied sampling
temperature, when the LLM is enabled) and fuses the outputs into a single
consensus result:

  * booleans / categoricals  -> confidence-weighted majority vote
  * numerics                 -> confidence-weighted mean
  * lists                    -> items appearing in a majority of samples
  * dicts                    -> taken from the highest-confidence sample

It also returns a `consensus_agreement` in [0,1] measuring how strongly the
samples agreed — surfaced in the UI and folded into the run confidence.

In heuristic mode the samples are identical, so agreement is 1.0 and the
mechanism is a no-op — the same code path serves both modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel

from agents.base import BaseAgent
from core.logging_config import get_logger

logger = get_logger("voting_agent")


@dataclass
class Consensus:
    output: BaseModel
    agreement: float
    confidence: float
    source: str
    samples: int


class VotingAgent:
    name = "voting_agent"

    def __init__(self, samples: int = 3):
        self.samples = max(1, samples)

    def _temperatures(self, base: float) -> list[float]:
        if self.samples == 1:
            return [base]
        lo, hi = max(0.0, base - 0.25), min(1.0, base + 0.35)
        step = (hi - lo) / (self.samples - 1)
        return [round(lo + i * step, 2) for i in range(self.samples)]

    def consensus(self, agent: BaseAgent, ctx: dict) -> Consensus:
        temps = self._temperatures(agent.base_temperature)
        results = [agent.analyze(ctx, temperature=t) for t in temps]
        outputs = [r.output for r in results]
        confidences = [max(0.01, r.confidence) for r in results]
        model_cls = type(outputs[0])

        merged, agreement = _merge(outputs, confidences, model_cls)
        source = "llm" if any(r.source == "llm" for r in results) else "heuristic"
        confidence = round(sum(confidences) / len(confidences), 3)
        out = model_cls(**merged)
        return Consensus(out, round(agreement, 3), confidence, source, len(results))


def _merge(
    outputs: list[BaseModel], weights: list[float], model_cls: Type[BaseModel]
) -> tuple[dict, float]:
    merged: dict = {}
    agreements: list[float] = []
    total_w = sum(weights)

    for field in model_cls.model_fields:
        values = [getattr(o, field) for o in outputs]
        sample = values[0]

        if isinstance(sample, bool):
            w_true = sum(w for v, w in zip(values, weights) if v)
            winner = w_true >= (total_w - w_true)
            merged[field] = winner
            win_w = w_true if winner else (total_w - w_true)
            agreements.append(win_w / total_w if total_w else 1.0)

        elif isinstance(sample, (int, float)):
            merged[field] = round(
                sum(v * w for v, w in zip(values, weights)) / total_w, 3
            ) if total_w else sample

        elif isinstance(sample, str):
            tally: dict[str, float] = {}
            for v, w in zip(values, weights):
                tally[v] = tally.get(v, 0.0) + w
            winner = max(tally, key=tally.get)
            merged[field] = winner
            agreements.append(tally[winner] / total_w if total_w else 1.0)

        elif isinstance(sample, list):
            merged[field] = _merge_lists(values)

        elif isinstance(sample, dict):
            # take dict from the highest-confidence sample
            best_idx = max(range(len(outputs)), key=lambda i: weights[i])
            merged[field] = values[best_idx]

        else:
            merged[field] = sample

    agreement = sum(agreements) / len(agreements) if agreements else 1.0
    return merged, agreement


def _merge_lists(value_lists: list[list]) -> list:
    """Keep items present in a majority of samples; else union (order-preserving)."""
    n = len(value_lists)
    counts: dict = {}
    order: list = []
    for lst in value_lists:
        for item in lst:
            key = str(item).lower()
            if key not in counts:
                counts[key] = 0
                order.append(item)
            counts[key] += 1
    majority = [it for it in order if counts[str(it).lower()] >= (n / 2)]
    return majority or order
