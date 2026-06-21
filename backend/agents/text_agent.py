"""Agent 1 — Text Analysis."""
from __future__ import annotations

from agents import lexicon
from agents.base import BaseAgent
from schemas.models import TextAnalysis

SYSTEM_PROMPT = """You are a senior direct-response copywriting analyst.
You analyse a SINGLE advertising creative's text and return a strict JSON object.
Only describe what is present in the provided copy. Never invent claims.

Return JSON with EXACTLY these keys:
  hook: string (the opening hook, verbatim or paraphrased)
  hook_type: one of [problem_first, question, curiosity, social_proof, offer, statement, unknown]
  cta: string (the call to action text, or "")
  cta_type: one of [download, try_free, signup, shop, learn_more, none]
  pain_points: string[] (explicit problems the copy references)
  emotional_triggers: string[] (emotion-bearing words/phrases present)
  audience_language: string[] (phrases that reveal who is being targeted)
  offer_positioning: string (how the offer/value is framed, or "")
  copywriting_structure: one of [PAS, AIDA, listicle, testimonial, direct, unknown]
  repeated_keywords: string[] (notable repeated/important keywords)
  confidence: number 0..1
Return ONLY the JSON object, no prose, no markdown fences."""


class TextAnalysisAgent(BaseAgent):
    name = "text_agent"
    output_model = TextAnalysis
    system_prompt = SYSTEM_PROMPT
    base_temperature = 0.3

    def build_prompt(self, ctx: dict) -> str:
        return (
            f"App: {ctx.get('app_name','')}\n"
            f"Platform: {ctx.get('platform','')}\n"
            f"Ad copy:\n\"\"\"\n{ctx.get('ad_text','') or '(no text provided)'}\n\"\"\""
        )

    def heuristic(self, ctx: dict) -> dict:
        text = ctx.get("ad_text") or ""
        hook_type, strength = lexicon.classify_hook(text)
        emotion, triggers = lexicon.classify_emotion(text)
        pains = lexicon.extract_pain_points(text)
        return {
            "hook": text[:120],
            "hook_type": hook_type,
            "cta": "",
            "cta_type": lexicon.classify_cta(text),
            "pain_points": pains,
            "emotional_triggers": triggers,
            "audience_language": pains[:3],
            "offer_positioning": "offer-led" if hook_type == "offer" else "",
            "copywriting_structure": lexicon.detect_copy_structure(text),
            "repeated_keywords": lexicon.top_keywords(text),
            "confidence": round(0.45 + 0.4 * strength, 2),
        }
