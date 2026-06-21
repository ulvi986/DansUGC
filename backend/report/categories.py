"""Category Validation + Contamination Guard (TASK 1 & 2).

Strict reject/include gate. Every ad is classified into exact / adjacent / wrong /
uncertain against the SELECTED category, and only `should_include_in_report` ads
ever reach taxonomy / aggregation / insight generation. Rejected ads are kept
separately so they can be shown in a "Rejected / Off-category" section but never
counted.

The heuristic classifier here is the deterministic ground truth; the LLM
CategoryValidationAgent (prompts.CATEGORY_VALIDATION_PROMPT) emits the same object
and is normalised through this module, so downstream maths is identical.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# product_type -> (display label, primary-promise keyword lexicon)
# --------------------------------------------------------------------------- #
PRODUCT_LEXICON: dict[str, tuple[str, list[str]]] = {
    "language_learning": (
        "Language learning app",
        ["learn spanish", "learn english", "learn french", "learn a language",
         "language learning", "language course", "vocabulary", "grammar",
         "pronunciation", "speaking practice", "speak a new language",
         "speak like a local", "ai language tutor", "language tutor", "fluent in",
         "become fluent", "translate", "translation", "duolingo", "babbel"],
    ),
    "esim_connectivity": (
        "eSIM / travel connectivity",
        ["esim", "e-sim", "roaming", "roaming fees", "sim card", "mobile data",
         "data plan", "stay connected", "150+ countries", "travel data",
         "local sim", "connected worldwide"],
    ),
    "music_streaming": (
        "Music streaming app",
        ["apple music", "spotify", "lyrics", "real-time lyrics", "adjustable vocals",
         "sing along", "stream music", "karaoke", "playlist", "millions of songs"],
    ),
    "ecommerce_shopping": (
        "E-commerce / shopping app",
        ["$0.99", "1st order", "first order", "free shipping", "add to cart",
         "% off your order", "shop now", "flash sale", "best deals", "coupon"],
    ),
    "food_delivery": (
        "Food delivery app",
        ["food delivery", "order food", "restaurants near", "delivery fee",
         "your favourite restaurants", "groceries delivered"],
    ),
    "astrology_prediction": (
        "Astrology / life prediction app",
        ["astrolog", "horoscope", "zodiac", "birth chart", "natal", "destiny",
         "fortune", "tarot", "numerolog", "predict your life", "love events",
         "life events", "luna", "moon sign"],
    ),
    "calendar_scheduling": (
        "Calendar / scheduling app",
        ["calendar", "schedule meeting", "appointment", "agenda", "time block",
         "plan your day", "to-do", "event planner"],
    ),
    "vpn_privacy": (
        "VPN / privacy app",
        ["vpn", "encrypt your", "hide your ip", "stay private online", "no logs"],
    ),
    "dating": ("Dating app", ["dating", "find a match", "singles near", "swipe", "first date"]),
    "fitness_health": ("Fitness / health app", ["workout", "lose weight", "calorie", "gym", "meal plan"]),
    "finance": ("Finance app", ["invest", "stocks", "crypto wallet", "loan", "credit score", "trading"]),
}

# Tokens that, for a given selected product type, force wrong_category unless the
# ad is *primarily* about the selected category (TASK 1 hard-exclusion rules).
HARD_EXCLUSIONS: dict[str, list[str]] = {
    "language_learning": [
        "esim", "e-sim", "roaming", "roaming fees", "150+ countries", "mobile data",
        "sim card", "apple music", "spotify", "lyrics", "adjustable vocals",
        "sing along", "$0.99", "1st order", "first order", "free shipping",
        "food delivery", "horoscope", "vpn",
    ],
}

ADJACENCY: dict[str, set[str]] = {
    "calendar_scheduling": {"astrology_prediction"},
    "astrology_prediction": {"calendar_scheduling"},
    # NOTE: eSIM is explicitly WRONG (not adjacent) for language apps.
}

_STATUS = {"exact_match", "adjacent_match", "wrong_category", "uncertain"}
UNCERTAIN_INCLUDE_MIN = 70

SELECTED_ALIASES: dict[str, str] = {
    "language app": "language_learning", "language": "language_learning",
    "language learning app": "language_learning", "english learning app": "language_learning",
    "calendar": "calendar_scheduling", "scheduling": "calendar_scheduling",
    "astrology": "astrology_prediction", "horoscope": "astrology_prediction",
    "esim": "esim_connectivity", "vpn": "vpn_privacy", "music": "music_streaming",
    "dating": "dating", "fitness": "fitness_health", "finance": "finance",
}


@dataclass
class AdCategory:
    ad_id: object
    selected_category: str
    detected_product_type: str
    category_match_status: str          # exact_match|adjacent_match|wrong_category|uncertain
    category_match_score: int           # 0..100
    category_reason: str
    should_include_in_report: bool
    rejection_reason: str | None = None
    ad_text: str = ""


@dataclass
class CategoryResolution:
    selected_category: str
    resolved_category: str
    selected_product_type: str
    resolved_product_type: str
    renamed: bool
    rename_suggested: bool
    rename_reason: str
    # contamination guard (TASK 2)
    raw_ads_count: int
    included_ads_count: int
    rejected_ads_count: int
    rejected_rate: float
    dominant_detected_product_type: str
    category_integrity_score: int       # 0..100
    aggregate_match_score: int
    mode: str                           # "normal" | "insufficient"
    warnings: list[str] = field(default_factory=list)
    per_ad: list[AdCategory] = field(default_factory=list)


# --------------------------------------------------------------------------- #
def normalise_status(status: str) -> str:
    s = (status or "").strip().lower()
    return s if s in _STATUS else "uncertain"


def _selected_product_type(selected_category: str) -> str | None:
    key = (selected_category or "").strip().lower()
    if key in SELECTED_ALIASES:
        return SELECTED_ALIASES[key]
    for ptype in PRODUCT_LEXICON:
        if key and key in ptype:
            return ptype
    return None


def _label_for_selected(selected_category: str) -> str:
    ptype = _selected_product_type(selected_category)
    if ptype:
        return PRODUCT_LEXICON[ptype][0]
    return (selected_category or "").strip().title() or "Unknown category"


def _ad_full_text(ad: dict) -> str:
    """All copy fields an off-category product could hide in."""
    return " ".join(str(ad.get(k, "") or "")
                    for k in ("ad_text", "hook_text", "cta_text", "advertiser"))


def normalise_text(text: str) -> str:
    """Lowercase + defeat spacing/punctuation evasion before substring matching.

    Joins single-character-spaced sequences ("E S I M" -> "esim",
    "1 5 0" -> "150"), collapses spaces around '+' ("150 +" -> "150+") and
    squeezes runs of whitespace. Applied to BOTH the text and the lexicon tokens
    so matching is symmetric.
    """
    t = (text or "").lower()
    t = re.sub(r"\b(?:[a-z0-9] )+[a-z0-9]\b",
               lambda m: m.group(0).replace(" ", ""), t)
    t = re.sub(r"\s*\+\s*", "+", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _hits_by_type(norm_text: str) -> dict[str, list[str]]:
    res: dict[str, list[str]] = {}
    for ptype, (_label, lexicon) in PRODUCT_LEXICON.items():
        hits = [kw for kw in lexicon if normalise_text(kw) in norm_text]
        if hits:
            res[ptype] = hits
    return res


def detect_product_type(text: str) -> tuple[str, int, list[str]]:
    """Return (product_type, score 0..100, matched_keywords) from ad copy."""
    norm = normalise_text(text)
    if not norm.strip():
        return "uncertain", 0, []
    hits_by = _hits_by_type(norm)
    if not hits_by:
        return "uncertain", 0, []
    best_type = max(hits_by, key=lambda p: len(hits_by[p]))
    best_hits = hits_by[best_type]
    return best_type, min(100, 45 + 20 * len(best_hits)), best_hits


def _hard_exclusion_hits(text: str, selected_ptype: str | None) -> list[str]:
    norm = normalise_text(text)
    return [tok for tok in HARD_EXCLUSIONS.get(selected_ptype or "", [])
            if normalise_text(tok) in norm]


def classify_ad(ad_id, text: str, selected_category: str) -> AdCategory:
    selected_label = _label_for_selected(selected_category)
    selected_ptype = _selected_product_type(selected_category)
    norm = normalise_text(text)
    hits_by = _hits_by_type(norm)
    detected, score, hits = detect_product_type(text)

    # Strict-dominance rule: the ad counts as "primarily the selected category"
    # ONLY if the selected category is the detected type AND it strictly out-counts
    # every other product type. A tie with an off-category product does NOT qualify
    # (closes the keyword-tie bypass).
    lang_hits = len(hits_by.get(selected_ptype, [])) if selected_ptype else 0
    offcat_max = max((len(h) for pt, h in hits_by.items() if pt != selected_ptype),
                     default=0)
    primarily_selected = bool(selected_ptype and detected == selected_ptype
                              and lang_hits > offcat_max)

    # Hard exclusion: off-category primary product wins unless clearly the selected one.
    excl = _hard_exclusion_hits(text, selected_ptype)
    if excl and not primarily_selected:
        det = detected if detected != "uncertain" else "other"
        det_label = PRODUCT_LEXICON.get(det, (det, []))[0]
        return AdCategory(
            ad_id, selected_label, det, "wrong_category",
            max(0, 20 - 3 * len(excl)),
            f"Hard-exclusion tokens present ({', '.join(excl[:4])}); primary product is {det_label}, not {selected_label}.",
            False, rejection_reason=f"wrong_category: {det_label} (tokens: {', '.join(excl[:3])})",
            ad_text=text,
        )

    if detected == "uncertain":
        include = score >= UNCERTAIN_INCLUDE_MIN
        return AdCategory(
            ad_id, selected_label, "uncertain", "uncertain", score,
            "No clear primary-promise keywords detected.", include,
            rejection_reason=None if include else "uncertain_low_confidence",
            ad_text=text,
        )

    if selected_ptype is None:
        return AdCategory(ad_id, selected_label, detected, "uncertain", min(score, 40),
                          f"Selected category not in taxonomy; detected {detected}.",
                          False, rejection_reason="uncertain_low_confidence", ad_text=text)

    if detected == selected_ptype:
        return AdCategory(ad_id, selected_label, detected, "exact_match", score,
                          f"Primary promise matches {selected_label} "
                          f"(keywords: {', '.join(hits[:4])}).", True, ad_text=text)

    if detected in ADJACENCY.get(selected_ptype, set()):
        return AdCategory(ad_id, selected_label, detected, "adjacent_match",
                          min(55, score - 20),
                          f"Detected {PRODUCT_LEXICON[detected][0]} — adjacent but distinct.",
                          False, rejection_reason=f"adjacent_match: {PRODUCT_LEXICON[detected][0]}",
                          ad_text=text)

    det_label = PRODUCT_LEXICON[detected][0]
    return AdCategory(ad_id, selected_label, detected, "wrong_category",
                      max(0, 25 - 5 * len(hits)),
                      f"Primary promise is {det_label} (keywords: {', '.join(hits[:4])}), not {selected_label}.",
                      False, rejection_reason=f"wrong_category: {det_label}", ad_text=text)


def resolve_category(selected_category: str, ads: list[dict],
                     rename_threshold: float = 0.5,
                     min_included: int = 10) -> CategoryResolution:
    selected_label = _label_for_selected(selected_category)
    selected_ptype = _selected_product_type(selected_category)
    # Classify on ALL copy fields, not just ad_text: off-category payload hidden in
    # hook_text / cta_text / advertiser must not slip past the gate (Finding 3).
    per_ad = [classify_ad(a.get("ad_id", i), _ad_full_text(a), selected_category)
              for i, a in enumerate(ads)]
    raw = len(per_ad)

    detected_counts = Counter(c.detected_product_type for c in per_ad
                              if c.detected_product_type not in ("uncertain", "other"))
    dom_type, dom_share = "", 0.0
    if detected_counts and raw:
        dom_type, dom_count = detected_counts.most_common(1)[0]
        dom_share = dom_count / raw

    # Rename ONLY when a single off-category product clearly dominates (e.g. all
    # astrology). Heterogeneous contamination (eSIM+music+shopping) does NOT rename.
    resolved_ptype, resolved_label = selected_ptype, selected_label
    renamed = rename_suggested = False
    rename_reason = ""
    if dom_type and dom_type != selected_ptype and dom_share >= rename_threshold:
        renamed = rename_suggested = True
        resolved_ptype, resolved_label = dom_type, PRODUCT_LEXICON[dom_type][0]
        rename_reason = (f"{int(round(dom_share*raw))}/{raw} ads have a '{resolved_label}' "
                         f"primary promise; report renamed from '{selected_label}'.")
    elif dom_type and dom_type != selected_ptype and dom_share >= 0.30:
        rename_suggested = True
        rename_reason = (f"Largest detected product is '{PRODUCT_LEXICON[dom_type][0]}' "
                         f"({dom_share:.0%}); consider whether the selected category is correct.")

    # Re-mark inclusion against the RESOLVED category.
    included = 0
    for c in per_ad:
        keep = (c.detected_product_type == resolved_ptype)
        if c.category_match_status == "uncertain" and c.category_match_score >= UNCERTAIN_INCLUDE_MIN \
                and resolved_ptype == selected_ptype:
            keep = True
        c.should_include_in_report = keep
        if not keep and c.rejection_reason is None:
            c.rejection_reason = f"not {resolved_label}"
        if keep:
            c.rejection_reason = None
            included += 1

    rejected = raw - included
    rejected_rate = round(rejected / raw, 3) if raw else 0.0
    integrity = round(100 * included / raw) if raw else 0
    mode = "insufficient" if included < min_included else "normal"

    warnings: list[str] = []
    if rejected_rate > 0.30:
        warnings.append("Category contamination detected. A significant share of collected "
                        f"ads ({rejected}/{raw}) did not match the selected category.")
    if included < min_included:
        warnings.append(f"Only {included} of {raw} collected ads matched '{selected_label}'. "
                        "Insufficient validated ads for a reliable market report.")
    if rename_suggested and not renamed:
        warnings.append(rename_reason)

    return CategoryResolution(
        selected_category=selected_label, resolved_category=resolved_label,
        selected_product_type=selected_ptype or "uncertain",
        resolved_product_type=resolved_ptype or "uncertain",
        renamed=renamed, rename_suggested=rename_suggested, rename_reason=rename_reason,
        raw_ads_count=raw, included_ads_count=included, rejected_ads_count=rejected,
        rejected_rate=rejected_rate, dominant_detected_product_type=dom_type or "uncertain",
        category_integrity_score=integrity, aggregate_match_score=integrity, mode=mode,
        warnings=warnings, per_ad=per_ad,
    )
