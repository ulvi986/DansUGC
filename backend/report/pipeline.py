"""End-to-end evidence-locked report pipeline.

    Raw ads -> CategoryValidation (reject/include) -> Taxonomy (included only)
    -> Aggregation (included only) -> Confidence -> Draft
    -> ReportConsistencyValidator -> Final report

Rejected ads never reach the executive summary, Creative DNA, patterns, clusters,
opportunities, strategies, market map, evidence examples or creative briefs. They
appear only in `rejected_ads`. When fewer than `min_included` category-matched ads
survive, the pipeline refuses to produce a market report and renders an
"insufficient validated ads" report instead.
"""
from __future__ import annotations

from collections import Counter

from . import aggregation as agg_mod
from . import briefs as briefs_mod
from . import categories as cat_mod
from . import confidence as conf_mod
from . import summary as sum_mod
from .taxonomy import classify_hook, normalise, HOOK_TYPES, EMOTION_TYPES, CTA_TYPES, FORMAT_TYPES
from .validator import ValidatorContext, validate_report

MIN_INCLUDED = 10

_DIM_SRC = {"hook": "hook_type", "emotion": "emotion_type",
            "cta": "cta_type", "format": "creative_format"}


def normalise_ad_taxonomy(ad: dict) -> dict:
    hook_type = ad.get("hook_type") or classify_hook(ad.get("hook_text", ""))
    return {
        **ad,
        "hook_type": normalise(hook_type, HOOK_TYPES),
        "emotion_type": normalise(ad.get("emotion_type", "unknown"), EMOTION_TYPES),
        "cta_type": normalise(ad.get("cta_type", "unknown"), CTA_TYPES, default="none"),
        "creative_format": normalise(ad.get("creative_format", "unknown"), FORMAT_TYPES),
        "product_demo_present": bool(ad.get("product_demo_present", False)),
        "app_screen_visible": bool(ad.get("app_screen_visible", False)),
        "human_present": bool(ad.get("human_present", False)),
    }


def build_report(selected_category: str, ads: list[dict], *,
                 has_performance_data: bool = False,
                 has_active_duration: bool = False,
                 has_creative_score: bool = True,
                 source_quality_score: int = 40,
                 recency_score: int = 50) -> dict:
    norm_ads = [normalise_ad_taxonomy(a) for a in ads]
    resolution = cat_mod.resolve_category(selected_category, ads, min_included=MIN_INCLUDED)

    included, rejected = [], []
    for norm, cat in zip(norm_ads, resolution.per_ad):
        (included if cat.should_include_in_report else rejected).append((norm, cat))
    included_ads = [n for n, _ in included]

    contamination = _contamination_block(resolution)
    rejected_section = [_rejected_row(n, c) for n, c in rejected]
    rejected_ids = {c.ad_id for _, c in rejected}

    if resolution.mode == "insufficient":
        return _insufficient_report(resolution, included_ads, rejected_section,
                                    contamination, rejected_ids, has_performance_data)

    return _normal_report(resolution, norm_ads, included_ads, rejected_section,
                          contamination, rejected_ids, has_performance_data,
                          has_active_duration, has_creative_score,
                          source_quality_score, recency_score)


# --------------------------------------------------------------------------- #
def _normal_report(resolution, norm_ads, included_ads, rejected_section,
                   contamination, rejected_ids, has_perf, has_active_duration,
                   has_creative_score, source_quality_score, recency_score) -> dict:
    agg = agg_mod.build_aggregation(included_ads)
    n = agg.sample_size

    confidence = conf_mod.compute_confidence(
        sample_size=n, category_match_score=resolution.aggregate_match_score,
        overall_unknown_rate=agg.overall_unknown_rate(),
        has_creative_score=has_creative_score, has_active_duration=has_active_duration,
        has_performance_data=has_perf,
        category_integrity_score=resolution.category_integrity_score,
        included_ads_count=n, source_quality_score=source_quality_score,
        recency_score=recency_score,
    )

    patterns = _build_patterns(agg, included_ads, norm_ads, has_perf)
    saturation = _saturation_table(agg, included_ads, norm_ads)
    opportunities = _build_opportunities(agg, n)
    exec_summary = sum_mod.build_executive_summary(resolution, agg, has_perf)
    if resolution.warnings:
        exec_summary = resolution.warnings[:1] + exec_summary

    fmt = agg.fields["format"]
    evidence_note = (f"{fmt.dominant_count}/{n} validated ads use the dominant "
                     f"{fmt.dominant_value.replace('_',' ')} format; directional, not measured.")
    briefs = briefs_mod.build_briefs(resolution.resolved_product_type,
                                     fmt.dominant_value, evidence_note)

    report = {
        "mode": "normal",
        "category": _category_block(resolution),
        "contamination": contamination,
        "executive_summary": exec_summary,
        "data_quality": _data_quality(n, agg, has_perf, confidence, resolution),
        "creative_dna": _creative_dna(agg),
        "cta_section": _cta_section(agg),
        "hook_section": _hook_section(agg),
        "patterns": patterns,
        "saturation": saturation,
        "opportunities": opportunities,
        "insights": _insights(patterns, included_ads),
        "strategies": _strategies(patterns, opportunities),
        "strategies_text": [],
        "briefs": [vars(b) for b in briefs],
        "rejected_ads": rejected_section,
        "confidence": _confidence_block(confidence),
    }
    return _finalise(report, resolution, agg, rejected_ids, has_perf, patterns)


def _insufficient_report(resolution, included_ads, rejected_section,
                         contamination, rejected_ids, has_perf) -> dict:
    n = resolution.included_ads_count
    raw = resolution.raw_ads_count
    label = resolution.selected_category
    confidence = conf_mod.compute_confidence(
        sample_size=n, category_match_score=resolution.aggregate_match_score,
        overall_unknown_rate=1.0, has_creative_score=False, has_active_duration=False,
        has_performance_data=has_perf,
        category_integrity_score=resolution.category_integrity_score,
        included_ads_count=n,
    )
    # raw-vs-validated saturation so the UI can show why nothing is usable
    agg_inc = agg_mod.build_aggregation(included_ads) if included_ads else None
    saturation = _saturation_table(agg_inc, included_ads, included_ads) if agg_inc else []
    for s in saturation:
        s["usable_for_conclusion"] = False
        s["reason"] = f"Valid sample below {MIN_INCLUDED}"

    exec_summary = [
        f"Only {n} of {raw} collected ads matched the selected category "
        f"('{label}'). Because category contamination is high, this run should be "
        f"treated as invalid for market-level conclusions.",
        f"Insufficient validated {label.lower()} ads for a reliable market report. "
        f"The system needs more category-matched ads before generating Creative DNA, "
        f"patterns or creative briefs.",
    ]
    report = {
        "mode": "insufficient",
        "category": _category_block(resolution),
        "contamination": contamination,
        "executive_summary": exec_summary,
        "data_quality": [
            f"Validated ads: {n}/{raw} (need >= {MIN_INCLUDED}).",
            f"Rejected (off-category): {resolution.rejected_ads_count}.",
            f"Category integrity score: {resolution.category_integrity_score}/100.",
            f"Confidence capped: {confidence.final}/100 ({confidence.band}).",
        ] + confidence.notes,
        "creative_dna": {"status": "Insufficient validated ads"},
        "cta_section": {"text": "No reliable CTA pattern detected.", "reliable": False},
        "hook_section": {"note": "Insufficient validated ads for hook analysis."},
        "patterns": [],
        "saturation": saturation,
        "opportunities": [],
        "insights": [],
        "strategies": [],
        "strategies_text": [],
        "briefs": [],
        "briefs_note": "Creative brief generation skipped because too few "
                       "category-matched ads were found.",
        "rejected_ads": rejected_section,
        "confidence": _confidence_block(confidence),
    }
    return _finalise(report, resolution, agg_inc, rejected_ids, has_perf, [])


# --------------------------------------------------------------------------- #
def _finalise(report, resolution, agg, rejected_ids, has_perf, patterns) -> dict:
    ctx = ValidatorContext(
        detected_product_type=resolution.resolved_product_type,
        resolved_product_type=resolution.resolved_product_type,
        selected_was_generic=resolution.renamed or resolution.rename_suggested,
        download_frequency=_download_freq(agg) if agg else 0.0,
        cta_reliable=(agg.fields["cta"].reliable if agg else False),
        hook_unknown_rate=(agg.hook_unknown_rate() if agg else 1.0),
        has_performance_data=has_perf,
        any_positive_lift=any((p.get("performance_lift", 0) or 0) > 0 for p in patterns),
        rejected_ad_ids=set(rejected_ids),
        included_ads_count=resolution.included_ads_count,
        category_integrity_score=resolution.category_integrity_score,
        mode=resolution.mode,
        resolved_title=resolution.resolved_category,
        forbidden_tokens=cat_mod.HARD_EXCLUSIONS.get(resolution.selected_product_type, []),
    )
    result = validate_report(report, ctx)
    report["executive_summary"] = [l for l in report.get("executive_summary", []) if l.strip()]
    report["validation"] = {"passed": result.passed, "violations": result.violations,
                            "warnings": resolution.warnings}
    return report


def _contamination_block(r: cat_mod.CategoryResolution) -> dict:
    return {
        "raw_ads_count": r.raw_ads_count,
        "included_ads_count": r.included_ads_count,
        "rejected_ads_count": r.rejected_ads_count,
        "rejected_rate": r.rejected_rate,
        "dominant_detected_product_type": r.dominant_detected_product_type,
        "category_integrity_score": r.category_integrity_score,
        "contamination_detected": r.rejected_rate > 0.30,
        "rename_suggested": r.rename_suggested,
        "warnings": r.warnings,
    }


def _category_block(r: cat_mod.CategoryResolution) -> dict:
    return {
        "title": r.resolved_category, "selected": r.selected_category,
        "product_type": r.resolved_product_type, "renamed": r.renamed,
        "rename_suggested": r.rename_suggested, "rename_reason": r.rename_reason,
        "match_score": r.aggregate_match_score, "included": r.included_ads_count,
        "rejected": r.rejected_ads_count,
        "per_ad": [vars(c) for c in r.per_ad],
    }


def _rejected_row(ad: dict, cat: cat_mod.AdCategory) -> dict:
    return {
        "ad_id": cat.ad_id, "detected_product_type": cat.detected_product_type,
        "category_match_status": cat.category_match_status,
        "category_match_score": cat.category_match_score,
        "rejection_reason": cat.rejection_reason or cat.category_reason,
        "ad_text_excerpt": (ad.get("ad_text", "") or "")[:140],
    }


# --------------------------------------------------------------------------- #
def _count_value(ads, dim, value) -> int:
    src = _DIM_SRC[dim]
    return sum(1 for a in ads if str(a.get(src)) == value)


def _build_patterns(agg, included_ads, all_ads, has_perf) -> list[dict]:
    out = []
    for dim, fa in agg.fields.items():
        if fa.dominant_value in ("unknown", "uncertain", "none", "neutral"):
            continue
        lift = _lift_for(dim, fa.dominant_value, included_ads)
        cls = agg_mod.classify_pattern(fa.dominant_count, fa.dominant_frequency, lift, has_perf)
        ev_ids = [a.get("ad_id", i) for i, a in enumerate(included_ads)
                  if str(a.get(_DIM_SRC[dim])) == fa.dominant_value]
        raw_n = _count_value(all_ads, dim, fa.dominant_value)
        out.append({
            "name": f"{dim}:{fa.dominant_value}", "dimension": dim,
            "value": fa.dominant_value, "support_count": fa.dominant_count,
            "frequency": fa.dominant_frequency,
            "raw_frequency": f"{raw_n}/{len(all_ads)}",
            "validated_frequency": f"{fa.dominant_count}/{fa.total}",
            "performance_lift": lift, "has_perf": has_perf,
            "claim_class": cls["claim_class"], "label": cls["label"], "verb": cls["verb"],
            "text": f"{fa.dominant_value.replace('_',' ').title()} is a {cls['verb']} "
                    f"({fa.dominant_count}/{fa.total} validated, {fa.dominant_frequency}%).",
            "evidence_ad_ids": ev_ids,
        })
    for key, fa in agg.booleans.items():
        if fa.dominant_value != "true" or fa.dominant_count == 0:
            continue
        cls = agg_mod.classify_pattern(fa.dominant_count, fa.dominant_frequency, 0.0, has_perf)
        out.append({
            "name": key, "dimension": "format_signal", "value": "true",
            "support_count": fa.dominant_count, "frequency": fa.dominant_frequency,
            "raw_frequency": f"{fa.dominant_count}/{fa.total}",
            "validated_frequency": f"{fa.dominant_count}/{fa.total}",
            "performance_lift": 0.0, "has_perf": has_perf,
            "claim_class": cls["claim_class"], "label": cls["label"], "verb": cls["verb"],
            "text": f"{key.replace('_',' ')} is a {cls['verb']} "
                    f"({fa.dominant_count}/{fa.total} validated, {fa.dominant_frequency}%).",
            "evidence_ad_ids": [a.get("ad_id", i) for i, a in enumerate(included_ads)
                                if a.get(_BOOL_SRC[key])],
        })
    return out


def _saturation_table(agg, included_ads, all_ads) -> list[dict]:
    if not agg:
        return []
    rows = []
    usable = len(included_ads) >= MIN_INCLUDED
    for dim, fa in agg.fields.items():
        if fa.dominant_value in ("unknown", "uncertain", "none", "neutral"):
            continue
        raw_n = _count_value(all_ads, dim, fa.dominant_value)
        rows.append({
            "pattern": f"{dim}:{fa.dominant_value}",
            "raw_frequency": f"{raw_n}/{len(all_ads)}",
            "validated_frequency": f"{fa.dominant_count}/{fa.total}",
            "saturation_tier": fa.tier,
            "usable_for_conclusion": usable and fa.reliable,
            "reason": "" if (usable and fa.reliable) else
                      (f"Valid sample below {MIN_INCLUDED}" if not usable else
                       "Dominant value under-supported / under-classified"),
        })
    return rows


_BOOL_SRC = {"product_demo": "product_demo_present",
             "app_screen_early": "app_screen_visible", "human_present": "human_present"}


def _lift_for(dim, value, ads) -> float:
    src = _DIM_SRC[dim]
    have, rest = [], []
    for a in ads:
        s = a.get("creative_score")
        if s is None:
            continue
        (have if str(a.get(src)) == value else rest).append(s)
    if not have or not rest:
        return 0.0
    return round(sum(have)/len(have) - sum(rest)/len(rest), 1)


def _build_opportunities(agg, n) -> list[dict]:
    out = []
    hook = agg.fields["hook"]
    base_cap = 50 if n < MIN_INCLUDED else 60
    for absent in ("founder_story", "authority", "transformation"):
        if hook.distribution.get(absent, 0) == 0:
            out.append({
                "name": f"hook:{absent}", "usage_frequency": 0.0,
                "confidence": base_cap, "label": "whitespace_untested",
                "has_external_benchmark": False,
                "note": f"{absent.replace('_',' ').title()} is absent from the validated "
                        f"set ({n} ads). Differentiation hypothesis, unproven in this sample.",
            })
    return out


def _insights(patterns, ads) -> list[dict]:
    by_id = {a.get("ad_id", i): a for i, a in enumerate(ads)}
    out = []
    for p in patterns:
        if p["claim_class"] == "low_support":
            continue
        rows = [_evidence_row(by_id.get(i, {})) for i in p.get("evidence_ad_ids", [])[:5]]
        out.append({"title": p["text"], "claim_class": p["claim_class"],
                    "support_count": p["support_count"], "frequency": p["frequency"],
                    "evidence_rows": rows})
    return out


def _evidence_row(ad) -> dict:
    return {"ad_id": ad.get("ad_id"), "advertiser": ad.get("advertiser", ""),
            "platform": ad.get("platform", ""), "hook_text": ad.get("hook_text", ""),
            "cta": ad.get("cta_text", ""), "format": ad.get("creative_format", ""),
            "emotion": ad.get("emotion_type", "")}


def _strategies(patterns, opportunities) -> list[dict]:
    out = []
    link_map = {"saturated": "saturation_risk", "dominant": "dominant_pattern",
                "emerging": "emerging_signal", "low_support": "low_support_high_lift",
                "proven_winner": "dominant_pattern"}
    for p in patterns:
        out.append({"title": f"Leverage {p.get('value', p['name'])}", "text": p["text"],
                    "linked_to": link_map.get(p["claim_class"], "dominant_pattern")})
    for o in opportunities:
        out.append({"title": f"Test {o['name']}", "text": o["note"],
                    "linked_to": "whitespace_opportunity"})
    return out


def _download_freq(agg) -> float:
    cta = agg.fields["cta"]
    return 100.0 * cta.distribution.get("download", 0) / cta.total if cta.total else 0.0


def _data_quality(n, agg, has_perf, confidence, resolution) -> list[str]:
    out = [f"Validated sample size: {n} ads (of {resolution.raw_ads_count} collected).",
           f"Category integrity: {resolution.category_integrity_score}/100.",
           f"Mean taxonomy signal-gap rate: {agg.overall_unknown_rate():.0%}.",
           f"Direct performance data: {'yes' if has_perf else 'none'}.",
           f"Overall confidence: {confidence.final}/100 ({confidence.band})."]
    out.extend(confidence.notes)
    return out


def _creative_dna(agg) -> dict:
    if agg.sample_size < MIN_INCLUDED:
        return {"status": "Insufficient validated ads"}
    dna = {}
    for k, fa in agg.fields.items():
        if fa.dominant_count < agg_mod.RELIABLE_MIN_SUPPORT or fa.dominant_value in (
                "unknown", "uncertain", "none", "neutral"):
            dna[k] = {"reliable": False,
                      "note": "No reliable CTA pattern detected." if k == "cta"
                              else f"No reliable {k} pattern (support {fa.dominant_count})."}
            continue
        dna[k] = {"dominant": fa.dominant_value, "frequency": fa.dominant_frequency,
                  "support": fa.dominant_count, "reliable": fa.reliable, "tier": fa.tier}
    for k, fa in agg.booleans.items():
        dna[k] = {"frequency": fa.dominant_frequency, "support": fa.dominant_count,
                  "reliable": fa.reliable, "tier": fa.tier}
    return dna


def _cta_section(agg) -> dict:
    cta = agg.fields["cta"]
    download = cta.distribution.get("download", 0)
    dom_count = cta.dominant_count
    reliable = cta.reliable and dom_count >= agg_mod.RELIABLE_MIN_SUPPORT
    return {"dominant": cta.dominant_value, "download_count": download,
            "download_frequency": round(100.0 * download / cta.total, 1) if cta.total else 0.0,
            "support_count": dom_count, "reliable": reliable,
            "text": "No reliable CTA pattern detected." if not reliable
                    else f"Dominant CTA: {cta.dominant_value} ({dom_count}/{cta.total})."}


def _hook_section(agg) -> dict:
    hook = agg.fields["hook"]
    note = ("Hook classification quality is low. Hook-level conclusions are unreliable."
            if hook.unknown_rate > 0.50 else "")
    return {"dominant": hook.dominant_value, "frequency": hook.dominant_frequency,
            "support": hook.dominant_count, "unknown_rate": hook.unknown_rate,
            "reliable": hook.reliable, "note": note}


def _confidence_block(c) -> dict:
    return {"components": c.components, "raw": c.raw, "ceilings": c.ceilings,
            "final": c.final, "band": c.band, "notes": c.notes}
