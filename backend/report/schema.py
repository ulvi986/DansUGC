"""Pydantic schema for the final, validated report (BUG 10 — canonical contract).

The deterministic pipeline emits dicts; `validate_final_report` round-trips them
through these models so the API/UI gets a typed, stable contract.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class AdCategoryOut(_Lenient):
    ad_id: Any = None
    detected_product_type: str = "uncertain"
    selected_category: str = ""
    category_match_status: str = "uncertain"   # exact_match|adjacent_match|wrong_category|uncertain
    category_match_score: int = 0
    category_reason: str = ""
    should_include_in_report: bool = False
    rejection_reason: str | None = None


class ContaminationBlock(_Lenient):
    raw_ads_count: int = 0
    included_ads_count: int = 0
    rejected_ads_count: int = 0
    rejected_rate: float = 0.0
    dominant_detected_product_type: str = "uncertain"
    category_integrity_score: int = 0
    contamination_detected: bool = False
    rename_suggested: bool = False
    warnings: list[str] = Field(default_factory=list)


class RejectedAd(_Lenient):
    ad_id: Any = None
    detected_product_type: str = "uncertain"
    category_match_status: str = "wrong_category"
    category_match_score: int = 0
    rejection_reason: str = ""
    ad_text_excerpt: str = ""


class SaturationRow(_Lenient):
    pattern: str = ""
    raw_frequency: str = ""
    validated_frequency: str = ""
    saturation_tier: str = ""
    usable_for_conclusion: bool = False
    reason: str = ""


class CategoryBlock(_Lenient):
    title: str = ""
    selected: str = ""
    product_type: str = "uncertain"
    renamed: bool = False
    rename_reason: str = ""
    match_score: int = 0
    included: int = 0
    rejected: int = 0
    per_ad: list[AdCategoryOut] = Field(default_factory=list)


class EvidenceRow(_Lenient):
    ad_id: Any = None
    advertiser: str = ""
    platform: str = ""
    creative_url: str = ""
    hook_text: str = ""
    cta: str = ""
    format: str = ""
    emotion: str = ""
    first_seen: str | None = None
    last_seen: str | None = None
    days_active: int | None = None
    category_match_score: int = 0
    classification_reason: str = ""


class Pattern(_Lenient):
    name: str = ""
    dimension: str = ""
    value: str = ""
    support_count: int = 0
    frequency: float = 0.0
    raw_frequency: str = ""
    validated_frequency: str = ""
    performance_lift: float = 0.0
    has_perf: bool = False
    claim_class: str = "low_support"   # proven_winner|dominant|saturated|emerging|low_support
    label: str = ""
    verb: str = ""
    text: str = ""
    evidence_ad_ids: list[Any] = Field(default_factory=list)
    needs_evidence: bool = False


class Opportunity(_Lenient):
    name: str = ""
    usage_frequency: float = 0.0
    confidence: int = 0
    label: str = "whitespace_untested"
    has_external_benchmark: bool = False
    note: str = ""


class Insight(_Lenient):
    title: str = ""
    claim_class: str = ""
    support_count: int = 0
    frequency: float = 0.0
    evidence_rows: list[EvidenceRow] = Field(default_factory=list)
    needs_evidence: bool = False


class Brief(_Lenient):
    concept_name: str = ""
    target_audience: str = ""
    target_pain_desire: str = ""
    hook: str = ""
    first_3_seconds: str = ""
    visual_direction: str = ""
    script_outline: list[str] = Field(default_factory=list)
    cta: str = ""
    evidence_behind_it: str = ""
    risk: str = ""
    ab_test: str = ""


class ConfidenceBlock(_Lenient):
    components: dict[str, float] = Field(default_factory=dict)
    raw: float = 0.0
    ceilings: dict[str, int] = Field(default_factory=dict)
    final: float = 0.0
    band: str = "low"
    notes: list[str] = Field(default_factory=list)


class ValidationBlock(_Lenient):
    passed: bool = True
    violations: list[str] = Field(default_factory=list)


class FinalReport(_Lenient):
    mode: str = "normal"            # "normal" | "insufficient"
    category: CategoryBlock = Field(default_factory=CategoryBlock)
    contamination: ContaminationBlock = Field(default_factory=ContaminationBlock)
    executive_summary: list[str] = Field(default_factory=list)
    data_quality: list[str] = Field(default_factory=list)
    creative_dna: dict[str, Any] = Field(default_factory=dict)
    cta_section: dict[str, Any] = Field(default_factory=dict)
    hook_section: dict[str, Any] = Field(default_factory=dict)
    patterns: list[Pattern] = Field(default_factory=list)
    saturation: list[SaturationRow] = Field(default_factory=list)
    opportunities: list[Opportunity] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    strategies: list[dict[str, Any]] = Field(default_factory=list)
    briefs: list[Brief] = Field(default_factory=list)
    briefs_note: str = ""
    rejected_ads: list[RejectedAd] = Field(default_factory=list)
    confidence: ConfidenceBlock = Field(default_factory=ConfidenceBlock)
    validation: ValidationBlock = Field(default_factory=ValidationBlock)


def validate_final_report(report: dict) -> FinalReport:
    return FinalReport(**report)
