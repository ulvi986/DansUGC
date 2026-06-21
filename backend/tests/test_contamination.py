"""Contamination / category-validation tests (TASK 10 — the exact bug case).

selected_category = "language app", 15 collected ads, of which eSIM/music/shopping
ads must be rejected and excluded from every analysis section. Fewer than 10 valid
language ads remain -> the pipeline must render an "insufficient validated ads"
report and list the rejected ads separately.

Runs with pytest or as a plain script.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from report.pipeline import build_report          # noqa: E402
from report.categories import resolve_category    # noqa: E402

# 4 real language ads, 4 eSIM, 4 music, 3 shopping  -> 15 total, 4 validated
LANG = [
    "Learn Spanish in 5 minutes a day with our AI language tutor and become fluent fast.",
    "Master English grammar and pronunciation with bite-size speaking practice lessons.",
    "Build vocabulary in French — real conversation practice with an AI language tutor.",
    "Speak a new language: daily lessons, pronunciation drills and translation help.",
]
ESIM = [
    "Stay connected worldwide. Get your eSIM in minutes and use it in 150+ countries with no roaming fees.",
    "Ditch expensive roaming fees. One eSIM, mobile data in 150+ countries.",
    "No more SIM card hassle — instant eSIM data plan for travellers worldwide.",
    "Stay connected worldwide with affordable mobile data and no roaming fees.",
]
MUSIC = [
    "Sing along to real-time lyrics with adjustable vocals. Try Apple Music free for 3 months.",
    "Stream millions of songs and sing along with real-time lyrics on Apple Music.",
    "Karaoke night anywhere: adjustable vocals and lyrics, free for 3 months.",
    "Apple Music: real-time lyrics, adjustable vocals, millions of songs.",
]
SHOP = [
    "3 items from $0.99 on 1st order. Shop now with free shipping!",
    "Flash sale: best deals, $0.99 on your 1st order, free shipping today only.",
    "Get 3 items from $0.99 on your first order — add to cart now.",
]


def contaminated_ads() -> list[dict]:
    ads, i = [], 0
    for bucket, fmt, emo in ((LANG, "ugc_demo", "trust"), (ESIM, "brand_video", "relief"),
                             (MUSIC, "brand_video", "excitement"), (SHOP, "static_image", "excitement")):
        for txt in bucket:
            i += 1
            ads.append({
                "ad_id": i, "ad_text": txt, "advertiser": f"adv{i}", "platform": "meta",
                "creative_format": fmt, "product_demo_present": True,
                "app_screen_visible": True, "human_present": False,
                "hook_type": "statement", "hook_text": txt[:40],
                "emotion_type": emo, "cta_type": "try_free" if bucket is MUSIC else "none",
                "cta_text": "", "creative_score": 70,
            })
    return ads


def _report():
    return build_report("language app", contaminated_ads(), has_performance_data=False)


# --------------------------------------------------------------------------- #
def test_offcategory_ads_are_rejected():
    res = resolve_category("language app", contaminated_ads())
    by_id = {c.ad_id: c for c in res.per_ad}
    # language ads (1-4) included
    for i in (1, 2, 3, 4):
        assert by_id[i].should_include_in_report is True, by_id[i]
        assert by_id[i].category_match_status == "exact_match"
    # eSIM / music / shopping rejected as wrong_category
    for i in range(5, 16):
        assert by_id[i].should_include_in_report is False, by_id[i]
        assert by_id[i].category_match_status == "wrong_category"
        assert by_id[i].rejection_reason


def test_contamination_guard_counts():
    res = resolve_category("language app", contaminated_ads())
    assert res.raw_ads_count == 15
    assert res.included_ads_count == 4
    assert res.rejected_ads_count == 11
    assert res.rejected_rate > 0.30
    assert res.category_integrity_score < 70
    assert res.mode == "insufficient"


def test_report_is_insufficient_mode():
    r = _report()
    assert r["mode"] == "insufficient"
    summ = " ".join(r["executive_summary"]).lower()
    assert "insufficient validated" in summ
    assert "4 of 15" in summ


def test_no_offcategory_terms_anywhere_in_conclusions():
    r = _report()
    blob = " ".join(r["executive_summary"]).lower()
    blob += str(r["creative_dna"]).lower()
    blob += str(r.get("patterns", [])).lower()
    blob += str(r.get("strategies", [])).lower()
    blob += str(r.get("briefs", [])).lower()
    for bad in ("roaming", "esim", "150+ countries", "apple music", "lyrics",
                "adjustable vocals", "$0.99", "1st order"):
        assert bad not in blob, f"off-category term leaked into conclusions: {bad}"


def test_creative_dna_blocked():
    r = _report()
    assert r["creative_dna"] == {"status": "Insufficient validated ads"}


def test_no_patterns_or_briefs_generated():
    r = _report()
    assert r["patterns"] == []
    assert r["briefs"] == []
    assert "skipped" in r.get("briefs_note", "").lower()


def test_cta_not_claimed_from_offcategory():
    # try_free came only from rejected Apple Music ads -> must not surface
    r = _report()
    assert r["cta_section"]["reliable"] is False
    assert "free" not in r["cta_section"].get("text", "").lower() or \
           r["cta_section"]["text"] == "No reliable CTA pattern detected."


def test_rejected_ads_listed_separately():
    r = _report()
    rej = r["rejected_ads"]
    assert len(rej) == 11
    assert all(row["rejection_reason"] for row in rej)
    # an eSIM example is present in the rejected section, with reason
    assert any("roaming" in (row["ad_text_excerpt"] or "").lower() for row in rej)


def test_confidence_decreased_and_capped():
    r = _report()
    c = r["confidence"]
    assert c["final"] <= 50            # included < 10 cap
    assert "included<10" in c["ceilings"]
    assert "category_integrity<70" in c["ceilings"]


def test_contamination_warning_present():
    r = _report()
    warns = " ".join(r["contamination"]["warnings"]).lower()
    assert "contamination" in warns
    assert r["contamination"]["contamination_detected"] is True


# --- QA adversarial regressions (findings from the QA agent) --------------- #
def test_qa_finding2_spaced_token_evasion_rejected():
    """'E S I M' / '150 + countries' must not evade the hard-exclusion gate."""
    from report.categories import classify_ad
    t = "Translate signs abroad. E S I M ready, works in 150 + countries."
    c = classify_ad(99, t, "language app")
    assert c.category_match_status == "wrong_category"
    assert c.should_include_in_report is False


def test_qa_finding3_offcategory_payload_in_other_fields_rejected():
    """Off-category copy hidden in hook_text/cta_text/advertiser must be caught."""
    ads = [{
        "ad_id": i + 1, "ad_text": "learn spanish",
        "advertiser": "Apple Music" if i == 0 else "a",
        "creative_format": "ugc_demo", "product_demo_present": True,
        "app_screen_visible": True, "hook_type": "question",
        "hook_text": "Sing along to lyrics with adjustable vocals" if i == 0 else "learn fast",
        "emotion_type": "trust", "cta_type": "download",
        "cta_text": "Try Apple Music free", "creative_score": 75,
    } for i in range(12)]
    r = build_report("language app", ads)
    assert r["category"]["per_ad"][0]["should_include_in_report"] is False
    blob = (str(r["insights"]) + str(r["creative_dna"]) + str(r["executive_summary"])).lower()
    for bad in ("lyrics", "adjustable vocals", "apple music"):
        assert bad not in blob


def test_qa_finding1_keyword_tie_does_not_auto_include():
    """A tie between language and an off-category product must not be included."""
    from report.categories import classify_ad
    c = classify_ad(99, "Learn Spanish translate. esim roaming.", "language app")
    assert c.should_include_in_report is False


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
