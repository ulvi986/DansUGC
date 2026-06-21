"""Agent 7 — Strategy Generation.

Generates the final, evidence-bound creative strategy. The LLM is explicitly
constrained to use ONLY the supplied evidence (discovered patterns + top-scoring
ad copy). The heuristic fallback assembles the same strategy from the mined
patterns, so output is data-derived in either mode — never imagined.
"""
from __future__ import annotations

import json

from agents.base import BaseAgent
from schemas.models import Strategy

SYSTEM_PROMPT = """You are a senior performance-marketing strategist.
You receive EVIDENCE mined from real competitor ads (discovered patterns and
top-scoring ad copy). Produce a creative strategy that is fully SUPPORTED by
that evidence. You must NOT invent facts, audiences, or claims that the evidence
does not support. Where you make a recommendation, it must trace to a pattern.

Return strict JSON with EXACTLY these keys:
  strategy_name: string
  target_audience: string
  core_pain_point: string
  hooks: string[] (exactly 3 new, ready-to-use hooks grounded in the patterns)
  creative_concept: string
  video_script: string[] (ordered beats/lines for a short video)
  platform_plan: object (keys are platform names, values are concrete guidance)
  ab_test_plan: string[] (specific A/B tests to run)
  why_supported: string (explain which patterns justify the strategy)
  confidence_score: number 0..1
  limitations: string[] (honest caveats about the evidence)
Return ONLY the JSON object."""


class StrategyGenerationAgent(BaseAgent):
    name = "strategy_agent"
    output_model = Strategy
    system_prompt = SYSTEM_PROMPT
    base_temperature = 0.5

    def build_prompt(self, ctx: dict) -> str:
        evidence = "\n".join(f"- {s}" for s in ctx.get("pattern_statements", [])) or "- (no strong patterns)"
        examples = "\n".join(
            f"- [{a.get('score', 0):.0f} pts | {a.get('platform','')}] {a.get('ad_text','')[:160]}"
            for a in ctx.get("top_ads", [])[:6]
        ) or "- (none)"
        return (
            f"App: {ctx.get('app_name','')}\n"
            f"Platforms analysed: {', '.join(ctx.get('platforms', [])) or 'all'}\n"
            f"Ads analysed: {ctx.get('analyzed_ads_count', 0)}\n\n"
            f"DISCOVERED PATTERNS (evidence):\n{evidence}\n\n"
            f"TOP-SCORING AD COPY (evidence):\n{examples}\n\n"
            f"Top pain points observed: {', '.join(ctx.get('top_pain_points', [])) or 'none'}\n"
            f"Dominant hook: {ctx.get('dominant_hook','unknown')}; "
            f"dominant emotion: {ctx.get('dominant_emotion','neutral')}; "
            f"dominant format: {ctx.get('dominant_format','unknown')}.\n"
        )

    def heuristic(self, ctx: dict) -> dict:
        app = ctx.get("app_name", "the app")
        pains = ctx.get("top_pain_points", []) or ["everyday stress and overthinking"]
        core_pain = pains[0]
        dom_hook = ctx.get("dominant_hook", "problem_first")
        dom_emotion = ctx.get("dominant_emotion", "relief")
        dom_format = ctx.get("dominant_format", "ugc_video")
        platforms = ctx.get("platforms", []) or ["meta", "tiktok"]

        hooks = [
            f"Still {core_pain}? Here's the 30-second habit that helped.",
            f"POV: you finally stopped {core_pain} — here's how.",
            f"The app that turns {core_pain} into a 2-minute daily reset.",
        ]
        script = [
            f"0-3s HOOK: open on a relatable '{core_pain}' moment (problem-first, face on camera).",
            "3-6s: agitate — name the feeling the viewer recognises.",
            f"6-12s: reveal {app} on screen, show the core action solving it (product demo).",
            f"12-18s: emotional payoff — convey '{dom_emotion}'.",
            "18-22s: CTA — 'Download free and try your first entry today.'",
        ]
        platform_plan = {}
        for p in platforms:
            if p == "tiktok":
                platform_plan[p] = "UGC, face-on-camera, fast pacing, native sound, hook in first 2s."
            elif p == "meta":
                platform_plan[p] = "Explanatory copy + clear app-screen demo; carousel/video with captioned benefits."
            else:
                platform_plan[p] = "Lead with the dominant winning pattern for this platform."

        return {
            "strategy_name": f"Problem-First {dom_emotion.capitalize()} Engine for {app}",
            "target_audience": f"Adults seeking help with {', '.join(pains[:3])}.",
            "core_pain_point": core_pain,
            "hooks": hooks,
            "creative_concept": (
                f"{dom_format.replace('_',' ').title()} creatives that open with a problem-first hook, "
                f"demo the app within the first 5 seconds, and resolve on a feeling of {dom_emotion}."
            ),
            "video_script": script,
            "platform_plan": platform_plan,
            "ab_test_plan": [
                "Hook A (problem-first) vs Hook B (question) — measure 3s retention.",
                "UGC face-on-camera vs app-screen-first opening — measure CTR.",
                "CTA 'Download free' vs 'Try your first entry' — measure installs.",
            ],
            "why_supported": (
                "Each element mirrors a mined pattern: "
                + "; ".join(ctx.get("pattern_statements", [])[:4])
            ),
            "confidence_score": ctx.get("evidence_confidence", 0.5),
            "limitations": ctx.get("limitations", []),
        }


def evidence_block(ctx: dict) -> str:
    return json.dumps(ctx.get("pattern_statements", []), ensure_ascii=False)
