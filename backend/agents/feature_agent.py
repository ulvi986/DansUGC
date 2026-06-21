"""Agent 4 — Feature Extraction.

Deterministically fuses the (consensus) text + visual analyses for a single ad
into the canonical structured feature row. This is a structuring step, so it is
intentionally rule-based (reproducible, explainable) rather than generative.
"""
from __future__ import annotations

from schemas.models import (
    ExtractedFeatures,
    ImageAnalysis,
    TextAnalysis,
    VideoAnalysis,
)


class FeatureExtractionAgent:
    name = "feature_agent"

    def extract(
        self,
        text: TextAnalysis,
        visual: VideoAnalysis | ImageAnalysis | None,
        creative_type: str | None,
        duration: float | None,
    ) -> ExtractedFeatures:
        is_video = isinstance(visual, VideoAnalysis)
        is_image = isinstance(visual, ImageAnalysis)

        human = bool(getattr(visual, "human_present", False))
        if is_video:
            app_screen = visual.app_screen_first_5s
            face = visual.face_visible_first_3s
            product_demo = visual.product_demo_present
            ugc = visual.ugc_style
            fmt = "ugc_video" if ugc else "brand_video"
            density = "high" if visual.visual_pacing == "fast" else "medium"
        elif is_image:
            app_screen = visual.product_screenshot
            face = visual.human_present
            product_demo = visual.product_screenshot
            ugc = visual.visual_style == "lifestyle"
            fmt = "screenshot" if visual.visual_style == "screenshot" else "lifestyle_image"
            density = "high" if visual.text_overlay else "low"
        else:
            app_screen = face = product_demo = ugc = False
            fmt = creative_type or "unknown"
            density = "medium"

        return ExtractedFeatures(
            hook_type=text.hook_type or "unknown",
            hook_strength=_hook_strength(text),
            cta_type=text.cta_type or "none",
            emotion_type=(text.emotional_triggers[0] if text.emotional_triggers else "neutral"),
            creative_format=fmt,
            human_present=human,
            app_screen_visible=bool(app_screen),
            ugc_style=bool(ugc),
            video_length=duration if is_video else None,
            visual_density=density,
            face_visible=bool(face),
            product_demo_present=bool(product_demo),
        )


def _hook_strength(text: TextAnalysis) -> float:
    base = {
        "problem_first": 0.9,
        "question": 0.75,
        "curiosity": 0.72,
        "social_proof": 0.68,
        "offer": 0.6,
        "statement": 0.45,
    }.get(text.hook_type, 0.4)
    if text.pain_points:
        base = min(1.0, base + 0.05)
    return round(base, 2)
