"""Executive Summary builder (BUG 2, 3, 5, 6).

The summary is generated from validated aggregates, not free text, so it can only
mention patterns that clear their evidence thresholds. This is the structural fix
for "summary contradicts the data": there is no path by which a 0%-frequency CTA
or a 2-ad hook can enter the prose.
"""
from __future__ import annotations

from .aggregation import Aggregation, FieldAggregate

_FIELD_LABEL = {
    "hook": "hook", "emotion": "emotional trigger", "cta": "CTA", "format": "format",
}
_BOOL_LABEL = {
    "product_demo": "a product demo", "app_screen_early": "the app screen shown early",
    "human_present": "a human on screen",
}
_NICE = {
    "brand_video": "brand-style video", "ugc_testimonial": "UGC testimonial",
    "ugc_demo": "UGC demo", "screen_recording": "screen recording",
    "prediction": "prediction", "predict": "prediction",
}


def _nice(v: str) -> str:
    return _NICE.get(v, v.replace("_", " "))


def sample_size_language(n: int) -> list[str]:
    out: list[str] = []
    if n < 10:
        out.append(f"Low confidence: only {n} ads were analysed (below 10).")
    elif n < 30:
        out.append(f"This report is directional, not conclusive, because the sample "
                   f"size is below 30 ads (N={n}).")
    elif n < 100:
        out.append(f"Moderate confidence (N={n}); treat as indicative, not definitive.")
    return out


def build_executive_summary(resolution, agg: Aggregation,
                            has_performance_data: bool) -> list[str]:
    n = agg.sample_size
    lines: list[str] = []

    # 1) Category framing (with rename note if it happened).
    if resolution.renamed:
        lines.append(
            f"Report category resolved to '{resolution.resolved_category}': "
            f"{resolution.rename_reason}")
    else:
        lines.append(f"Category: {resolution.resolved_category} "
                     f"({resolution.included_ads_count}/{n} ads matched).")

    # 2) Reliable dominant patterns — booleans first (these are the strong ones).
    strong: list[str] = []
    for key, fa in agg.booleans.items():
        if fa.is_dominant or (fa.reliable and fa.dominant_value == "true"
                              and fa.dominant_frequency > 60):
            strong.append(f"{_BOOL_LABEL[key]} ({fa.dominant_count}/{n})")
    fmt = agg.fields.get("format")
    if fmt and fmt.is_dominant:
        strong.insert(0, f"{_nice(fmt.dominant_value)} format "
                         f"({fmt.dominant_count}/{n})")

    if strong:
        lines.append("The analysed ads are dominated by " + _join(strong) + ".")

    # 3) Weak / under-classified fields — say so explicitly, never promote them.
    weak_notes: list[str] = []
    hook = agg.fields.get("hook")
    if hook and not hook.reliable:
        if hook.unknown_rate >= 0.50:
            weak_notes.append(
                f"hooks are mostly unclassified ({hook.unknown_rate:.0%} unknown/uncertain) - "
                f"hook classification quality is low and hook-level conclusions are unreliable")
        else:
            weak_notes.append(f"no hook reaches reliable support "
                              f"(top hook only {hook.dominant_count}/{n})")
    cta = agg.fields.get("cta")
    if cta and not cta.reliable:
        weak_notes.append("no reliable CTA pattern was detected")
    if weak_notes:
        lines.append("Hook and CTA signals are weak or under-classified: "
                     + _join(weak_notes) + ".")

    # 4) Format-level vs full-formula framing.
    reliable_non_format = [k for k, fa in agg.fields.items()
                           if k != "format" and fa.reliable]
    if strong and not reliable_non_format:
        lines.append("Treat this as a format-level pattern, not a full winning formula.")

    # 5) Performance-data caveat.
    if not has_performance_data:
        lines.append("No direct performance data (spend/CTR/CPA/ROAS/installs/revenue) "
                     "was available, so creative scores reflect quality heuristics, not measured lift.")

    # 6) Sample-size language.
    lines.extend(sample_size_language(n))
    return lines


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
