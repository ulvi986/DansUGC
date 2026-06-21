"""Agent 6 — Scoring.

Explainable, weighted, deterministic scoring. Each criterion contributes a
transparent share of the 100-point total, and every sub-score ships with a
plain-language explanation. Determinism is a feature here: scores are auditable
and reproducible.

Weights (sum = 100):
  Hook Strength ........... 20
  Pain Point Clarity ...... 15
  Visual Clarity .......... 15
  Product Demonstration ... 15
  Emotional Trigger ....... 15
  CTA Strength ............ 10
  Platform Fit ............ 10
"""
from __future__ import annotations

from schemas.models import (
    ExtractedFeatures,
    ImageAnalysis,
    ScoreBreakdown,
    TextAnalysis,
    VideoAnalysis,
)

WEIGHTS = {
    "hook_strength": 20,
    "pain_point_clarity": 15,
    "visual_clarity": 15,
    "product_demonstration": 15,
    "emotional_trigger": 15,
    "cta_strength": 10,
    "platform_fit": 10,
}

_STRONG_CTA = {"download", "try_free", "signup"}


class ScoringAgent:
    name = "scoring_agent"

    def score(
        self,
        text: TextAnalysis,
        visual: VideoAnalysis | ImageAnalysis | None,
        features: ExtractedFeatures,
        platform: str | None,
    ) -> ScoreBreakdown:
        expl: dict[str, str] = {}

        # 1) Hook strength
        hook_n = features.hook_strength
        expl["hook_strength"] = f"Hook type '{features.hook_type}' → strength {hook_n:.2f}."

        # 2) Pain point clarity
        pains = len(text.pain_points)
        pain_n = min(1.0, pains / 2.0)
        expl["pain_point_clarity"] = (
            f"{pains} explicit pain point(s) referenced." if pains
            else "No explicit pain point referenced."
        )

        # 3) Visual clarity
        vis_n = 0.5
        if features.human_present:
            vis_n += 0.2
        if features.app_screen_visible:
            vis_n += 0.3
        vis_n = min(1.0, vis_n)
        expl["visual_clarity"] = (
            f"human_present={features.human_present}, app_screen_visible={features.app_screen_visible}."
        )

        # 4) Product demonstration
        demo_n = 1.0 if features.product_demo_present else (0.5 if features.app_screen_visible else 0.0)
        expl["product_demonstration"] = (
            "Clear product demo." if features.product_demo_present
            else ("App screen shown without full demo." if features.app_screen_visible else "No product demo.")
        )

        # 5) Emotional trigger
        emo_n = 0.0 if features.emotion_type in ("neutral", "") else min(1.0, 0.6 + 0.2 * len(text.emotional_triggers))
        expl["emotional_trigger"] = (
            f"Emotion '{features.emotion_type}' with {len(text.emotional_triggers)} trigger(s)."
            if emo_n else "Neutral emotional tone."
        )

        # 6) CTA strength
        cta_n = 1.0 if features.cta_type in _STRONG_CTA else (0.6 if features.cta_type != "none" else 0.0)
        expl["cta_strength"] = f"CTA type '{features.cta_type}'."

        # 7) Platform fit
        fit_n = self._platform_fit(features, platform)
        expl["platform_fit"] = f"UGC={features.ugc_style} on platform '{platform}'."

        normalised = {
            "hook_strength": hook_n,
            "pain_point_clarity": pain_n,
            "visual_clarity": vis_n,
            "product_demonstration": demo_n,
            "emotional_trigger": emo_n,
            "cta_strength": cta_n,
            "platform_fit": fit_n,
        }
        weighted = {k: round(v * WEIGHTS[k], 2) for k, v in normalised.items()}
        total = round(sum(weighted.values()), 2)

        return ScoreBreakdown(
            **weighted,
            total=total,
            explanations=expl,
            confidence=round((text.confidence + (getattr(visual, "confidence", 0.4) if visual else 0.4)) / 2, 2),
        )

    @staticmethod
    def _platform_fit(features: ExtractedFeatures, platform: str | None) -> float:
        p = (platform or "").lower()
        if p == "tiktok":
            return 1.0 if features.ugc_style else 0.55
        if p == "meta":
            # Meta tolerates both; explanatory/brand creative fits slightly better.
            return 0.85 if not features.ugc_style else 0.7
        return 0.6
