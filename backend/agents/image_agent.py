"""Agent 3 — Image Analysis."""
from __future__ import annotations

from agents import lexicon
from agents.base import BaseAgent
from schemas.models import ImageAnalysis

SYSTEM_PROMPT = """You are a senior static-creative analyst.
You analyse ONE image ad and return strict JSON describing observable elements.
Only report what is supported by the provided material.

Return JSON with EXACTLY these keys:
  human_present: boolean
  product_screenshot: boolean (true if an app/product screen is shown)
  brand_visible: boolean
  cta_visible: boolean
  visual_style: one of [minimal, bold, cluttered, lifestyle, screenshot]
  color_style: one of [warm, cool, high_contrast, pastel, dark]
  text_overlay: boolean
  emotional_tone: string
  confidence: number 0..1
Return ONLY the JSON object."""


class ImageAnalysisAgent(BaseAgent):
    name = "image_agent"
    output_model = ImageAnalysis
    system_prompt = SYSTEM_PROMPT
    base_temperature = 0.4

    def build_prompt(self, ctx: dict) -> str:
        return (
            f"App: {ctx.get('app_name','')}\n"
            f"Platform: {ctx.get('platform','')}\n"
            f"Creative type: image\n"
            f"On-image / accompanying copy:\n\"\"\"\n{ctx.get('ad_text','') or '(none)'}\n\"\"\"\n"
            f"Media URL: {ctx.get('image_or_video_url','') or '(none)'}"
        )

    def heuristic(self, ctx: dict) -> dict:
        text = ctx.get("ad_text") or ""
        emotion, _ = lexicon.classify_emotion(text)
        app_words = ["app", "journal", "track", "diary", "mood", "screen"]
        screenshot = any(w in text.lower() for w in app_words)
        return {
            "human_present": False,
            "product_screenshot": screenshot,
            "brand_visible": True,
            "cta_visible": lexicon.classify_cta(text) != "none",
            "visual_style": "screenshot" if screenshot else "lifestyle",
            "color_style": "pastel",
            "text_overlay": bool(text),
            "emotional_tone": emotion,
            "confidence": 0.4,
        }
