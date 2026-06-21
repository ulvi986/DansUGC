"""Pydantic contracts.

Two groups:
  1. Agent I/O schemas — the strict JSON each Gemini agent must return. They are
     deliberately lenient on *missing* fields (defaults) but strict on *types*,
     so JSON-repair + validation + retry can recover partial outputs.
  2. API DTOs — request/response models for FastAPI.

All categorical fields are plain strings (not Enums) so an unexpected label from
the model never hard-fails validation; normalisation happens in the agents.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Agent output schemas
# --------------------------------------------------------------------------- #
class TextAnalysis(_Lenient):
    hook: str = ""
    hook_type: str = "unknown"            # problem_first | question | curiosity | social_proof | offer | statement
    cta: str = ""
    cta_type: str = "none"                # download | try_free | learn_more | shop | signup | none
    pain_points: list[str] = Field(default_factory=list)
    emotional_triggers: list[str] = Field(default_factory=list)
    audience_language: list[str] = Field(default_factory=list)
    offer_positioning: str = ""
    copywriting_structure: str = ""       # PAS | AIDA | listicle | testimonial | direct | unknown
    repeated_keywords: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class VideoAnalysis(_Lenient):
    human_present: bool = False
    face_visible_first_3s: bool = False
    app_screen_first_5s: bool = False
    product_demo_present: bool = False
    cta_frame: str = "none"               # start | middle | end | none
    scene_structure: str = ""
    visual_pacing: str = "medium"         # slow | medium | fast
    hook_placement: str = "start"         # start | middle | end
    ugc_style: bool = False
    story_structure: str = ""             # problem_solution | testimonial | demo | listicle | unknown
    confidence: float = 0.4


class ImageAnalysis(_Lenient):
    human_present: bool = False
    product_screenshot: bool = False
    brand_visible: bool = False
    cta_visible: bool = False
    visual_style: str = ""                # minimal | bold | cluttered | lifestyle | screenshot
    color_style: str = ""                 # warm | cool | high_contrast | pastel | dark
    text_overlay: bool = False
    emotional_tone: str = "neutral"
    confidence: float = 0.4


class ExtractedFeatures(_Lenient):
    hook_type: str = "unknown"
    hook_strength: float = 0.0
    cta_type: str = "none"
    emotion_type: str = "neutral"
    creative_format: str = "unknown"      # ugc_video | brand_video | screenshot | lifestyle_image | ...
    human_present: bool = False
    app_screen_visible: bool = False
    ugc_style: bool = False
    video_length: float | None = None
    visual_density: str = "medium"        # low | medium | high
    face_visible: bool = False
    product_demo_present: bool = False


class PatternFinding(_Lenient):
    pattern_type: str
    pattern_value: str = ""
    platform: str | None = None
    frequency: int = 0
    sample_size: int = 0
    percentage: float = 0.0
    statement: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(_Lenient):
    hook_strength: float = 0.0
    pain_point_clarity: float = 0.0
    visual_clarity: float = 0.0
    product_demonstration: float = 0.0
    emotional_trigger: float = 0.0
    cta_strength: float = 0.0
    platform_fit: float = 0.0
    total: float = 0.0
    explanations: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.5


class Strategy(_Lenient):
    strategy_name: str = ""
    target_audience: str = ""
    core_pain_point: str = ""
    hooks: list[str] = Field(default_factory=list)
    creative_concept: str = ""
    video_script: list[str] = Field(default_factory=list)
    platform_plan: dict[str, Any] = Field(default_factory=dict)
    ab_test_plan: list[str] = Field(default_factory=list)
    why_supported: str = ""
    confidence_score: float = 0.0
    limitations: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# API DTOs
# --------------------------------------------------------------------------- #
class AnalyzeRequest(BaseModel):
    app_name: str
    platforms: list[str] = Field(default_factory=list)  # empty => all platforms


class ImportResponse(BaseModel):
    imported_ads: int
    imported_apps: int
    skipped_rows: int
    detected_columns: list[str]
    missing_optional_columns: list[str]
    message: str


class FetchRequest(BaseModel):
    app_name: str
    country: str | None = None              # default: settings.fetch_country
    include_tiktok: bool | None = None       # default: settings.fetch_tiktok


class FetchResponse(BaseModel):
    app_name: str
    advertiser_name: str | None = None
    fetched_ads: int                         # newly inserted this call
    total_ads: int                           # total stored for this app
    by_platform: dict[str, int] = Field(default_factory=dict)
    message: str


class AppSummary(BaseModel):
    app_name: str
    ad_count: int
    platforms: dict[str, int]


class AdOut(BaseModel):
    id: int
    ad_id: str | None = None
    app_name: str
    platform: str | None = None
    creative_type: str | None = None
    ad_text: str | None = None
    image_or_video_url: str | None = None
    country: str | None = None
    winner_score: float | None = None

    model_config = ConfigDict(from_attributes=True)


class RunSummary(BaseModel):
    id: int
    app_name: str
    platforms: list[str] | None = None
    status: str
    analyzed_ads_count: int
    confidence_score: float | None = None
    created_at: Any = None

    model_config = ConfigDict(from_attributes=True)


class FinalOutput(BaseModel):
    """The canonical final result shape (also written to outputs/final_strategy.json)."""

    run_id: int | None = None
    app_name: str = ""
    platforms: list[str] = Field(default_factory=list)
    analyzed_ads_count: int = 0
    winning_patterns: list[dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 0.0
    evidence_summary: str = ""
    reasoning_summary: str = ""
    limitations: list[str] = Field(default_factory=list)
    strategy: dict[str, Any] = Field(default_factory=dict)
    # Market Intelligence layer (clusters, winner patterns, gaps, DNA,
    # saturation, strategy triad, market map, executive summary).
    intelligence: dict[str, Any] = Field(default_factory=dict)
