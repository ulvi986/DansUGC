"""Evidence-locked ad-intelligence report pipeline.

Workflow:
    Raw ads
      -> Category Validation        (categories.py)
      -> Taxonomy Extraction        (taxonomy.py — prompts + normalisation)
      -> Aggregation                (aggregation.py)
      -> Confidence Scoring         (confidence.py)
      -> Report Draft               (summary.py + briefs.py)
      -> ReportConsistencyValidator (validator.py)
      -> Final Report               (pipeline.py orchestrates)

Design principle: every *number* and every *claim verb* is produced by the
deterministic layer. The LLM (prompts.py) may only narrate values it is given.
The ReportConsistencyValidator is the final gate: a draft that contradicts the
evidence is corrected or stripped before it can be rendered.
"""
from __future__ import annotations

from .pipeline import build_report

__all__ = ["build_report"]
