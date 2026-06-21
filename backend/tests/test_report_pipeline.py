"""Unit tests for the evidence-locked report pipeline.

Runs with pytest *or* as a plain script:  python backend/tests/test_report_pipeline.py
(kept dependency-free because pytest is not installed in this env).

Dataset is the Luna astrology example from the brief:
  sample_size=15, brand_video=14/15, product_demo=15/15, app_screen_early=15/15,
  statement_hook=2/15, predict_emotion=2/15, download_cta=0/15, unknown_hook=12/15,
  no engagement metrics.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from report.pipeline import build_report          # noqa: E402
from report.validator import (                     # noqa: E402
    ReportConsistencyValidator, ValidatorContext,
)

LUNA_COPY = ("Positions at your birthday predict your life & love events. "
             "It's time to uncover them in Luna App!")


def luna_ads() -> list[dict]:
    ads = []
    for i in range(15):
        ads.append({
            "ad_id": i + 1,
            "ad_text": LUNA_COPY,
            "advertiser": "Luna",
            "platform": "meta",
            "creative_format": "brand_video" if i < 14 else "ugc_demo",
            "product_demo_present": True,
            "app_screen_visible": True,
            "human_present": False,
            # 2 statement, 1 question, 12 unknown
            "hook_type": "statement" if i in (0, 1) else ("question" if i == 2 else "unknown"),
            "hook_text": "",
            # 2 prediction (one brand, one ugc), 13 neutral
            "emotion_type": "prediction" if i in (0, 14) else "neutral",
            "cta_type": "none",
            "cta_text": "",
            # prediction ads score higher -> emotion shows positive lift, brand does not
            "creative_score": 80 if i in (0, 14) else 70,
        })
    return ads


def _report():
    return build_report("calendar", luna_ads(), has_performance_data=False)


def _pattern(report, dimension, value):
    for p in report["patterns"]:
        if p.get("dimension") == dimension and p.get("value") == value:
            return p
    return None


# --------------------------------------------------------------------------- #
def test_category_renamed_from_calendar_to_astrology():
    r = _report()
    title = r["category"]["title"].lower()
    assert "calendar" != title
    assert "astrolog" in title, f"expected astrology title, got {r['category']['title']}"
    assert r["category"]["renamed"] is True
    assert r["category"]["selected"].lower().startswith("calendar")


def test_exec_summary_mentions_format_pattern():
    r = _report()
    s = " ".join(r["executive_summary"]).lower()
    assert "brand-style video" in s or "brand video" in s
    assert "product demo" in s
    assert "app screen" in s
    assert "format-level pattern" in s


def test_exec_summary_excludes_statement_hook_and_download_cta():
    r = _report()
    s = " ".join(r["executive_summary"]).lower()
    assert "statement hook" not in s
    assert "download" not in s          # BUG 2/3: never mention a 0% CTA


def test_cta_section_says_no_reliable_cta():
    r = _report()
    assert r["cta_section"]["text"] == "No reliable CTA pattern detected."
    assert r["cta_section"]["download_frequency"] == 0.0
    assert r["cta_section"]["reliable"] is False


def test_hook_section_flagged_unreliable():
    r = _report()
    hook = r["hook_section"]
    assert hook["reliable"] is False
    assert hook["unknown_rate"] >= 0.5
    assert "unreliable" in hook["note"].lower()


def test_brand_video_is_saturated_not_winner():
    r = _report()
    p = _pattern(r, "format", "brand_video")
    assert p is not None
    assert p["claim_class"] == "saturated"
    assert "winner" not in p["verb"].lower()
    assert "winning" not in p["text"].lower()


def test_predict_emotion_is_low_support_high_lift():
    r = _report()
    p = _pattern(r, "emotion", "prediction")
    assert p is not None
    assert p["performance_lift"] > 0
    assert p["label"] == "low_support_high_lift"
    assert "winner" not in p["verb"].lower()


def test_founder_story_is_untested_whitespace():
    r = _report()
    fs = next((o for o in r["opportunities"] if "founder_story" in o["name"]), None)
    assert fs is not None
    assert fs["usage_frequency"] == 0
    assert fs["confidence"] <= 65
    assert fs["label"] == "whitespace_untested"


def test_no_winner_wording_anywhere_without_perf_data():
    r = _report()
    # patterns must never assert winner/winning (BUG 5)
    pat = " ".join(p["text"].lower() + " " + p["verb"].lower() for p in r["patterns"])
    assert "winner" not in pat
    assert "winning" not in pat
    # the summary may only use "winning" inside the sanctioned cautious negation
    # ("...not a full winning formula"); never as a positive claim.
    summ = " ".join(r["executive_summary"]).lower().replace("not a full winning formula", "")
    assert "winner" not in summ
    assert "winning" not in summ


def test_confidence_capped_for_small_sample_and_no_perf():
    r = _report()
    c = r["confidence"]
    assert c["final"] <= 70                     # sample < 30 cap
    assert "sample_size<30" in c["ceilings"]
    assert "no_performance_data" in c["ceilings"]
    assert c["band"] in ("directional", "low", "moderate")


def test_briefs_are_category_specific():
    r = _report()
    assert len(r["briefs"]) == 5
    aud = r["briefs"][0]["target_audience"].lower()
    assert "astrolog" in aud or "birth chart" in aud or "destiny" in aud
    assert "organization" not in aud           # the old generic persona must be gone


def test_pipeline_self_validation_passes():
    r = _report()
    # the deterministic draft should already be consistent
    assert r["validation"]["passed"] is True, r["validation"]["violations"]


def test_validator_catches_a_hostile_draft():
    """Feed a draft that violates the rules; the validator must correct it."""
    bad = {
        "executive_summary": ["The market runs on a winning Download CTA formula."],
        "strategies_text": ["Scale the proven winner."],
        "patterns": [
            {"name": "cta:download", "dimension": "cta", "value": "download",
             "support_count": 0, "frequency": 0.0, "performance_lift": 0.0,
             "claim_class": "proven_winner", "verb": "proven winner",
             "text": "Download CTA is a proven winner.", "evidence_ad_ids": []},
            {"name": "hook:statement", "dimension": "hook", "value": "statement",
             "support_count": 2, "frequency": 13.0, "performance_lift": 0.0,
             "claim_class": "dominant", "verb": "dominant pattern",
             "text": "Statement hook is a dominant pattern.", "evidence_ad_ids": []},
        ],
        "opportunities": [
            {"name": "hook:founder_story", "usage_frequency": 0.0, "confidence": 90,
             "has_external_benchmark": False, "label": "high_confidence"},
        ],
        "insights": [{"title": "x", "evidence_rows": []}],
        "strategies": [{"title": "do x", "text": "x", "linked_to": "vibes"}],
        "cta_section": {}, "creative_dna": {"cta": {}}, "category": {"title": "calendar"},
    }
    ctx = ValidatorContext(
        detected_product_type="astrology_prediction",
        resolved_product_type="astrology_prediction",
        selected_was_generic=True, download_frequency=0.0, cta_reliable=False,
        hook_unknown_rate=0.8, has_performance_data=False, any_positive_lift=False,
    )
    res = ReportConsistencyValidator(ctx).validate(bad)
    assert not res.passed
    # download CTA scrubbed from summary
    assert all("download cta" not in l.lower() for l in bad["executive_summary"])
    # proven winner downgraded
    assert all(p["claim_class"] != "proven_winner" for p in bad["patterns"])
    # cta pattern dropped, section corrected
    assert all(p["dimension"] != "cta" for p in bad["patterns"])
    assert bad["cta_section"]["text"] == "No reliable CTA pattern detected."
    # hook pattern demoted (hook unknown 80%)
    hook_p = next(p for p in bad["patterns"] if p["dimension"] == "hook")
    assert hook_p["claim_class"] == "low_support"
    # absent opportunity capped + relabelled
    opp = bad["opportunities"][0]
    assert opp["confidence"] <= 65 and opp["label"] == "whitespace_untested"
    # unlinked strategy flagged
    assert bad["strategies"][0].get("needs_link") is True


# --------------------------------------------------------------------------- #
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
