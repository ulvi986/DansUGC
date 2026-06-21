"""Credibility / honesty tests for the Intelligence engine (rule 15).

Covers: confidence caps, whitespace caps, gated Creative DNA, taxonomy
(claim-intensity vs benefit vs emotion), founder de-duplication, no "winning"
language without performance data, and the disclaimer.

Runs with pytest or as a plain script.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import lexicon                                   # noqa: E402
from agents.intelligence import (                            # noqa: E402
    IntelligenceEngine, build_profiles, DISCLAIMER,
)

SKIN_COPY = "Flacidez en brazos o piel seca. Hidratar al máximo, resultados increíbles."


def _profile(i, hook, emotion, fmt, cta, score, platform="meta", has_perf=False):
    return {
        "id": i, "platform": platform, "score": float(score),
        "hook": hook, "cta": cta, "emotion": emotion, "format": fmt,
        "ugc": fmt in ("ugc_video", "ugc_demo"), "human": True,
        "app_screen": False, "product_demo": True, "face": True,
        "hook_strength": 0.7, "copy_structure": "PAS",
        "pain_points": ["flacidez", "piel seca"], "ad_text": SKIN_COPY,
        "has_perf": has_perf,
    }


def skincare_profiles(n=15, has_perf=False):
    """10 problem_first / 3 statement / 2 question; lifestyle-heavy; CTA shop in only 2."""
    profs = []
    for i in range(n):
        hook = "problem_first" if i < 10 else ("statement" if i < 13 else "question")
        emo = "insecurity" if i < 8 else ("relief" if i < 12 else "neutral")
        fmt = "lifestyle_image" if i < 9 else ("ugc_video" if i < 13 else "brand_video")
        cta = "shop" if i < 2 else "none"
        score = 70 + (i % 5) * 3
        plat = "meta" if i < 10 else "tiktok"
        profs.append(_profile(i + 1, hook, emo, fmt, cta, score, plat, has_perf))
    return profs


def _engine(n=15, has_perf=False):
    return IntelligenceEngine(skincare_profiles(n, has_perf),
                              platforms=["meta", "tiktok"], has_performance_data=has_perf)


def _walk_confidences(obj, path=""):
    """Yield (path, value) for every int 'confidence'/'*_confidence' field."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if (k == "confidence" or k.endswith("_confidence")) and isinstance(v, (int, float)):
                yield f"{path}.{k}", v
            else:
                yield from _walk_confidences(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            yield from _walk_confidences(v, f"{path}[{idx}]")


def _walk_strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "disclaimer":
                continue                      # disclaimer legitimately contains "winners"
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, str):
        yield obj


# --------------------------------------------------------------------------- #
def test_n15_caps_confidence_at_65():
    out = _engine(15).build()
    over = [(p, v) for p, v in _walk_confidences(out) if v is not None and v > 65]
    assert not over, f"confidences above 65 for N=15: {over[:5]}"


def test_n15_zero_usage_whitespace_capped_at_60():
    eng = _engine(15)
    gaps = eng.opportunity_gaps()
    zero = [g for g in gaps if g["current_usage_pct"] == 0]
    assert zero, "expected at least one absent (0%) whitespace lever"
    assert all(g["confidence"] <= 60 for g in zero), zero


def test_sparse_cta_not_in_creative_dna():
    dna = _engine(15).creative_dna()
    # shop CTA appears in only 2/15 -> must NOT be a dominant DNA component
    assert dna["cta"]["included"] is False
    assert "fragmented" in dna["formula"].lower()
    assert "strongest observed pattern" in dna["formula"].lower()
    # hook + format ARE supported and lead the formula
    assert dna["hook"]["included"] is True


def test_increible_is_claim_intensity_not_emotion():
    label, hits = lexicon.classify_claim_intensity("resultados increíbles")
    assert label == "incredible" and hits
    assert lexicon.classify_emotion("resultados increíbles")[0] == "neutral"


def test_hidratar_al_maximo_is_benefit_not_emotion():
    label, hits = lexicon.classify_benefit("hidratar al máximo")
    assert label == "hydrate_skin" and hits
    assert lexicon.classify_emotion("hidratar al máximo")[0] == "neutral"


def test_strong_is_claim_intensity_not_emotion():
    assert lexicon.classify_claim_intensity("strong results")[0] == "strong"
    assert lexicon.classify_emotion("strong results")[0] == "neutral"


def test_founder_not_duplicated_as_hook_and_format():
    ad = SimpleNamespace(id=1, platform="meta", winner_score=80,
                         ad_text="As a founder I built this", impressions=0, likes=0,
                         shares=0, comments=0)
    feats = SimpleNamespace(hook_type="founder_story", cta_type="learn_more",
                            emotion_type="trust", creative_format="founder_story",
                            ugc_style=False, human_present=True, app_screen_visible=False,
                            product_demo_present=False, face_visible=True, hook_strength=0.6)
    text_o = SimpleNamespace(copywriting_structure="testimonial", pain_points=[])
    prof = build_profiles([(ad, text_o, None, feats)])[0]
    assert prof["hook"] == "founder_story"
    assert prof["format"] == "founder_talking_head"   # NOT founder_story again


def test_no_winning_language_without_performance_data():
    out = _engine(15, has_perf=False).build()
    blob = " ".join(_walk_strings(out)).lower()
    for banned in ("winner", "winning", "this wins", "market winner", "proven winner"):
        assert banned not in blob, f"banned term '{banned}' present without performance data"


def test_disclaimer_present_when_no_performance_data():
    out = _engine(15, has_perf=False).build()
    assert out["disclaimer"] == DISCLAIMER
    assert out["has_performance_data"] is False
    assert out["meta"]["score_label"] == "Average creative-quality proxy score"


def test_performance_confidence_disabled_without_perf():
    out = _engine(15, has_perf=False).build()
    for w in out["winner_patterns"]:
        assert w["performance_confidence"] is None


def test_saturation_labels_and_small_sample_note():
    rows = _engine(15).market_saturation(top_gap=None)
    assert rows and all("saturation_label" in r for r in rows)
    assert all(r["directional"] for r in rows)   # N=15 < 20
    assert any("directional" in r["recommendation"].lower() for r in rows)


# --- QA adversarial regressions ------------------------------------------- #
def test_qa_luxury_not_emotion():
    """'luxury' is claim-intensity/positioning, must NOT classify as an emotion."""
    assert lexicon.classify_emotion("luxury")[0] == "neutral"
    assert lexicon.classify_claim_intensity("luxury")[0] == "premium"


def test_qa_founder_dedup_robust_to_spelling():
    """Case/spacing/spelling founder format variants must canonicalise, not duplicate."""
    for variant in ("founder_story", "founder", "Founder_Story",
                    "founder talking head", "founder-story"):
        ad = SimpleNamespace(id=1, platform="meta", winner_score=80, ad_text="x",
                             impressions=0, likes=0, shares=0, comments=0)
        feats = SimpleNamespace(hook_type="founder_story", cta_type="none",
                                emotion_type="trust", creative_format=variant,
                                ugc_style=False, human_present=True, app_screen_visible=False,
                                product_demo_present=False, face_visible=True, hook_strength=0.6)
        text_o = SimpleNamespace(copywriting_structure="testimonial", pain_points=[])
        prof = build_profiles([(ad, text_o, None, feats)])[0]
        assert prof["format"] == "founder_talking_head", f"variant {variant!r} not canonicalised"
        # founder must not appear duplicated across both dimensions
        assert not (prof["hook"] == "founder_story" and "founder" in prof["format"]
                    and prof["format"] != "founder_talking_head")


def test_qa_small_sample_builds_without_crash():
    """Empty winner/cluster lists at tiny N must not break build() or drop keys."""
    for n in (1, 2, 3):
        profs = [_profile(i + 1, "statement", "neutral", "brand_video", "none", 70) for i in range(n)]
        out = IntelligenceEngine(profs, ["meta"]).build()
        assert all(k in out for k in ("executive_summary", "creative_dna", "strategies", "meta"))
        assert all(k in out["executive_summary"]
                   for k in ("headline", "what_is_winning", "highest_confidence_opportunity"))


# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}"); failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
