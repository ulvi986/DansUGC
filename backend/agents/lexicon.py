"""Deterministic lexicons + text heuristics.

These power the *fallback* analysis when no Gemini key is configured, and also
seed/cross-check the LLM path. Everything here produces evidence that traces
back to the actual ad copy — never invented.
"""
from __future__ import annotations

import re
from collections import Counter

PROBLEM_FIRST = [
    "do you", "are you", "ever feel", "struggling", "struggle", "tired of",
    "can't", "cant", "overwhelmed", "stop", "still", "why do", "feeling",
    "stressed", "anxious", "overthinking", "trouble", "hard to",
]
QUESTION = ["?"]
OFFER = ["free", "% off", "discount", "shipping", "today", "sale", "limited", "deal", "save"]
SOCIAL_PROOF = ["people", "users", "join", "thousands", "millions", "everyone",
                "reviews", "rated", "loved", "trusted", "#1", "best"]
CURIOSITY = ["secret", "this is why", "nobody", "what happens", "the truth",
             "you won't believe", "here's how", "trick"]

CTA_TERMS = {
    "download": ["download", "install", "get the app"],
    "try_free": ["try free", "free trial", "start free", "try it"],
    "signup": ["sign up", "signup", "join", "register", "create account"],
    "shop": ["shop", "buy", "order", "purchase", "checkout"],
    "learn_more": ["learn more", "find out", "discover", "see how", "link in bio", "tap"],
}

PAIN_POINTS = [
    "anxiety", "anxious", "stress", "stressed", "overthinking", "overwhelmed",
    "sleep", "insomnia", "mood", "depression", "depressed", "lonely", "loneliness",
    "burnout", "focus", "mental health", "emotions", "emotional", "calm",
    "self-care", "journaling", "journal", "mindfulness", "therapy", "feelings",
]

EMOTIONS = {
    "relief": ["relief", "relieved", "let go", "release", "lighter", "alivio"],
    "calm": ["calm", "peace", "peaceful", "relax", "soothe", "serene"],
    "hope": ["hope", "heal", "growth", "transform"],
    "fear": ["fear", "scared", "worry", "afraid", "panic", "dread", "miedo"],
    "insecurity": ["insecure", "insecurity", "embarrassed", "ashamed", "self-conscious",
                   "inseguridad", "vergüenza"],
    "confidence": ["confidence", "confident", "empower", "take charge", "you can",
                   "self-assured", "confianza", "seguridad"],
    "trust": ["trust", "trusted", "reliable", "proven by", "confiar"],
    "curiosity": ["curious", "wonder", "find out", "discover why"],
    # NOTE: "luxury" is claim-intensity/positioning (see CLAIM_INTENSITY["premium"]),
    # not an emotion — kept out of here to avoid an emotion/claim collision.
    "aspiration": ["dream", "aspire", "elegance", "elegância", "elegancia", "belong"],
    "urgency": ["now", "today only", "hurry", "last chance", "don't miss", "ahora"],
    "joy": ["happy", "joy", "smile", "grateful", "gratitude"],
}

# Claim INTENSITY — how loudly the ad asserts quality. NOT an emotion.
# ("Increíble", "máximo", "strong", "clinically proven" live here, not in EMOTIONS.)
CLAIM_INTENSITY = {
    "incredible": ["incredible", "increíble", "increible", "incrível"],
    "maximum": ["maximum", "máximo", "maximo", "máxima", "máxima"],
    "revolutionary": ["revolutionary", "revolucionario", "revolucionária"],
    "clinically_proven": ["clinically proven", "clinically", "clínicamente", "dermatologist"],
    "fast_results": ["fast results", "resultados rápidos", "in days", "overnight"],
    "visible_results": ["visible results", "resultados visibles", "visibly", "visible"],
    "strong": ["strong", "powerful", "potente", "fuerte", "forte"],
    "premium": ["premium", "luxury", "advanced", "pro-grade"],
}

# BENEFIT / value proposition — the outcome promised. NOT an emotion.
# ("Hidratar al máximo" -> benefit hydrate, not emotion.)
BENEFITS = {
    "hydrate_skin": ["hydrate", "hydration", "hidratar", "hidrata", "hidratación", "moistur"],
    "improve_firmness": ["firmness", "firm", "firmeza", "reafirm", "lifting", "flacidez"],
    "reduce_wrinkles": ["wrinkle", "arrugas", "fine lines", "anti-aging", "antiarrugas"],
    "smoother_texture": ["smoother", "texture", "textura", "soft skin", "suave"],
    "visible_glow": ["glow", "radiant", "luminos", "brillo", "brightening"],
    "long_lasting": ["long-lasting", "lasts", "all day", "duradero", "24h", "24 h"],
    "easier_routine": ["easy routine", "simple routine", "rutina", "one step", "effortless"],
}

_WORD_RE = re.compile(r"[a-zA-Z']+")
_STOPWORDS = set(
    "the a an and or to of in for on with your you i we is are be it this that "
    "my me at as by from so but if not no yes do does can will just get got new "
    "app now free today out up all can't cant".split()
)


def _contains_any(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if t in text]


def classify_hook(text: str) -> tuple[str, float]:
    """Return (hook_type, strength 0..1) from copy. Order = priority."""
    t = (text or "").lower()
    if not t:
        return "unknown", 0.2
    if _contains_any(t, PROBLEM_FIRST):
        return "problem_first", 0.85
    if "?" in t:
        return "question", 0.7
    if _contains_any(t, CURIOSITY):
        return "curiosity", 0.7
    if _contains_any(t, SOCIAL_PROOF):
        return "social_proof", 0.65
    if _contains_any(t, OFFER):
        return "offer", 0.6
    return "statement", 0.45


def classify_cta(text: str) -> str:
    t = (text or "").lower()
    for cta_type, terms in CTA_TERMS.items():
        if _contains_any(t, terms):
            return cta_type
    return "none"


def extract_pain_points(text: str) -> list[str]:
    t = (text or "").lower()
    found = []
    for p in PAIN_POINTS:
        if p in t and p not in found:
            found.append(p)
    return found[:6]


def classify_emotion(text: str) -> tuple[str, list[str]]:
    """Return (emotion, triggers). Claim-intensity and benefit words are NOT
    emotions and are excluded, so 'increíble'/'máximo'/'strong'/'hidratar' never
    get mislabelled as emotional triggers."""
    t = (text or "").lower()
    best, triggers = "neutral", []
    for emotion, terms in EMOTIONS.items():
        hits = _contains_any(t, terms)
        if hits and len(hits) > len(triggers):
            best, triggers = emotion, hits
    return best, triggers


def classify_claim_intensity(text: str) -> tuple[str | None, list[str]]:
    """Return (claim_intensity_label | None, matched_terms)."""
    t = (text or "").lower()
    best, hits = None, []
    for label, terms in CLAIM_INTENSITY.items():
        m = _contains_any(t, terms)
        if m and len(m) > len(hits):
            best, hits = label, m
    return best, hits


def classify_benefit(text: str) -> tuple[str | None, list[str]]:
    """Return (benefit_label | None, matched_terms)."""
    t = (text or "").lower()
    best, hits = None, []
    for label, terms in BENEFITS.items():
        m = _contains_any(t, terms)
        if m and len(m) > len(hits):
            best, hits = label, m
    return best, hits


def top_keywords(text: str, k: int = 8) -> list[str]:
    words = [w.lower() for w in _WORD_RE.findall(text or "") if len(w) > 2]
    words = [w for w in words if w not in _STOPWORDS]
    return [w for w, _ in Counter(words).most_common(k)]


def detect_copy_structure(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "unknown"
    if any(p in t for p in PROBLEM_FIRST) and classify_cta(t) != "none":
        return "PAS"            # problem-agitate-solution
    if "\n" in t or "•" in t or re.search(r"\d\.", t):
        return "listicle"
    if any(s in t for s in ["i ", "my ", "me "]):
        return "testimonial"
    return "direct"
