"""Creative brief generation (BUG 8).

Briefs are seeded from the *resolved* product category and the actual dominant
angles, so an astrology app never gets a generic "better organisation and
insight" persona. In production the CreativeBriefAgent (prompts.CREATIVE_BRIEF_PROMPT)
expands these seeds; the deterministic scaffolding here guarantees the audience
and angle are category-correct even in heuristic mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# category-correct audience + desire seeds
CATEGORY_BRIEF_SEED: dict[str, dict] = {
    "astrology_prediction": {
        "audience": ("People interested in astrology, self-discovery, relationship "
                     "prediction, destiny, birth charts and personalised life guidance."),
        "desire": "Wanting to understand their future, love life and major life events.",
        "concepts": [
            ("Your birthday is not random",
             "Your birth date may explain why your love life repeats the same pattern.",
             ["love prediction", "life-event prediction"]),
            ("The pattern in your chart",
             "Your birth chart may reveal the timing of your next big change.",
             ["timing reveal", "career vs love framing"]),
        ],
    },
    "calendar_scheduling": {
        "audience": "Busy professionals and students who want to control their time and never miss commitments.",
        "desire": "Wanting less chaos and a clear plan for the day.",
        "concepts": [
            ("The 10-second plan", "Plan your whole week before your coffee gets cold.",
             ["time-block angle", "reminder angle"]),
        ],
    },
    "language_learning": {
        "audience": "People who want to speak a new language for travel, work or relationships.",
        "desire": "Wanting to actually speak, not just memorise.",
        "concepts": [
            ("Order coffee abroad", "I lived abroad a year and still couldn't order coffee.",
             ["shame hook", "question hook"]),
        ],
    },
}

_DEFAULT_SEED = {
    "audience": "The core users the analysed advertisers are targeting.",
    "desire": "The primary desire expressed in the analysed ad copy.",
    "concepts": [("Concept", "Lead with the dominant proven angle.", ["variant a", "variant b"])],
}


@dataclass
class CreativeBrief:
    concept_name: str
    target_audience: str
    target_pain_desire: str
    hook: str
    first_3_seconds: str
    visual_direction: str
    script_outline: list[str]
    cta: str
    evidence_behind_it: str
    risk: str
    ab_test: str


def build_briefs(resolved_product_type: str, dominant_format: str,
                 evidence_note: str, n_concepts: int = 5) -> list[CreativeBrief]:
    seed = CATEGORY_BRIEF_SEED.get(resolved_product_type, _DEFAULT_SEED)
    fmt = (dominant_format or "video").replace("_", " ")
    briefs: list[CreativeBrief] = []
    concepts = list(seed["concepts"])
    # pad concepts up to n_concepts by varying the angle
    while len(concepts) < n_concepts:
        base = concepts[len(briefs) % len(seed["concepts"])]
        concepts.append((f"{base[0]} (variant {len(concepts)})", base[1], base[2]))

    for name, hook, variants in concepts[:n_concepts]:
        briefs.append(CreativeBrief(
            concept_name=name,
            target_audience=seed["audience"],
            target_pain_desire=seed["desire"],
            hook=hook,
            first_3_seconds=f"Open on the core promise; {fmt} style, app screen shown early.",
            visual_direction=f"{fmt}; show the in-app result quickly; minimal brand intro.",
            script_outline=[
                "0-3s: hook line on screen + voiceover",
                "3-7s: user takes the key action in-app",
                "7-12s: app reveals the result/payoff",
                "12-18s: emotional reaction / proof",
                "18-22s: CTA over the app demo",
            ],
            cta="Download and try it",
            evidence_behind_it=evidence_note,
            risk="May feel too mystical / low-trust; test an entertainment framing." if
                 resolved_product_type == "astrology_prediction" else
                 "Angle may not generalise beyond the analysed set; validate with a test.",
            ab_test=f"A: '{variants[0]}' vs B: '{variants[1]}'" if len(variants) > 1
                    else f"A vs B on the {variants[0]} angle",
        ))
    return briefs
