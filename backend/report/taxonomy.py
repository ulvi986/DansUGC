"""Taxonomy Extraction normalisation (BUG 4).

Closed vocabularies for every categorical dimension plus a deterministic hook
classifier that distinguishes:
  * a confidently classified hook,
  * "uncertain"  -> hook text exists but could not be mapped,
  * "unknown"    -> no hook text at all.

This split is what fixes the "Brand video · Unknown hook 80%" vs "Statement 29%"
contradiction: aggregation can now see that the field is under-classified instead
of silently promoting a 2-ad value to a headline.
"""
from __future__ import annotations

import re

HOOK_TYPES = [
    "statement", "question", "problem_first", "curiosity", "social_proof",
    "founder_story", "authority", "offer_led", "fear_loss", "transformation",
    "prediction_fortune", "uncertain", "unknown",
]

EMOTION_TYPES = [
    "relief", "excitement", "fear", "belonging", "pride", "frustration",
    "curiosity", "trust", "prediction", "neutral", "unknown",
]

CTA_TYPES = [
    "download", "try_free", "signup", "subscribe", "shop", "learn_more",
    "none", "unknown",
]

FORMAT_TYPES = [
    "brand_video", "ugc_testimonial", "ugc_demo", "talking_head",
    "screen_recording", "motion_graphics", "static_image", "carousel",
    "meme", "unknown",
]

# values that mean "we did not classify this" (drive unknown_rate)
UNCLASSIFIED = {"unknown", "uncertain", ""}
# values that are real but carry no editorial signal (excluded from "dominant")
NON_SIGNAL = {"none", "neutral"}

# hook cue lexicon (deterministic fallback classifier)
_HOOK_CUES: list[tuple[str, list[str]]] = [
    ("question", [r"\?$", r"^(do|are|can|what|why|how|did|have|is|will) "]),
    ("prediction_fortune", ["predict", "your future", "will reveal", "destiny",
                            "what your", "your birth", "fortune", "horoscope"]),
    ("problem_first", ["tired of", "struggling", "can't", "cant ", "stop ",
                       "sick of", "hate when"]),
    ("social_proof", ["million users", "rated", "reviews", "join ", "everyone",
                      "people love"]),
    ("founder_story", ["i built", "i created", "our story", "when i was",
                       "i started"]),
    ("authority", ["doctors", "experts", "scientifically", "study shows",
                   "backed by"]),
    ("offer_led", ["free trial", "% off", "discount", "limited time", "save "]),
    ("fear_loss", ["don't miss", "before it", "running out", "last chance",
                   "you're losing"]),
    ("transformation", ["before and after", "transformed", "changed my life",
                        "in 30 days", "results"]),
    ("curiosity", ["secret", "nobody tells", "the truth about", "what happens",
                   "you won't believe"]),
]


def classify_hook(hook_text: str) -> str:
    """Map hook copy to a HOOK_TYPE.

    Returns "unknown" only when there is no hook text. When text exists but no
    cue matches, returns "uncertain" (not "unknown") so aggregation treats it as
    under-classified rather than a real category.
    """
    t = (hook_text or "").strip().lower()
    if not t:
        return "unknown"
    for label, cues in _HOOK_CUES:
        for cue in cues:
            if cue.startswith("^") or cue.endswith("$") or "(" in cue:
                if re.search(cue, t):
                    return label
            elif cue in t:
                return label
    return "uncertain"


def normalise(value: str, vocab: list[str], default: str = "unknown") -> str:
    v = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return v if v in vocab else default
