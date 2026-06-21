"""Confidence Scoring (BUG 9).

Confidence is computed from measurable components and then hard-capped, so it can
never be high when the data is thin. The caps are `min()` ceilings: the weakest
dimension dominates.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def sample_size_score(n: int) -> int:
    if n < 10:
        return 20
    if n < 30:
        return 45
    if n < 100:
        return 70
    return 90


def taxonomy_quality_score(unknown_rate: float) -> int:
    pct = unknown_rate * 100
    if pct > 70:
        return 20
    if pct >= 50:
        return 40
    if pct >= 25:
        return 65
    return 85


def evidence_strength_score(*, has_frequency: bool, has_creative_score: bool,
                            has_active_duration: bool, has_performance_data: bool) -> int:
    if has_performance_data:
        return 90
    if has_active_duration:
        return 65
    if has_creative_score:
        return 55
    if has_frequency:
        return 40
    return 20


WEIGHTS = {
    "sample_size": 0.20,
    "category_match": 0.20,
    "taxonomy_quality": 0.20,
    "evidence_strength": 0.20,
    "source_quality": 0.10,
    "recency": 0.10,
}


@dataclass
class ConfidenceResult:
    components: dict[str, float]
    raw: float
    ceilings: dict[str, int]        # rule -> ceiling value actually in force
    final: float
    band: str
    notes: list[str] = field(default_factory=list)


def _band(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 60:
        return "moderate"
    if score >= 40:
        return "directional"
    return "low"


def compute_confidence(*, sample_size: int, category_match_score: int,
                       overall_unknown_rate: float, has_creative_score: bool,
                       has_active_duration: bool, has_performance_data: bool,
                       category_integrity_score: int = 100,
                       included_ads_count: int | None = None,
                       source_quality_score: int = 40,
                       recency_score: int = 50) -> ConfidenceResult:
    components = {
        "sample_size": sample_size_score(sample_size),
        "category_match": float(category_match_score),
        "taxonomy_quality": taxonomy_quality_score(overall_unknown_rate),
        "evidence_strength": evidence_strength_score(
            has_frequency=True, has_creative_score=has_creative_score,
            has_active_duration=has_active_duration,
            has_performance_data=has_performance_data),
        "source_quality": float(source_quality_score),
        "recency": float(recency_score),
    }
    raw = sum(WEIGHTS[k] * v for k, v in components.items())

    ceilings: dict[str, int] = {}
    notes: list[str] = []
    if sample_size < 30:
        ceilings["sample_size<30"] = 70
        notes.append("Confidence capped at 70: sample size below 30 ads.")
    if not has_performance_data:
        ceilings["no_performance_data"] = 75
        notes.append("Confidence capped at 75: no direct performance data (spend/CTR/CPA/ROAS/installs/revenue).")
    if category_match_score < 70:
        ceilings["category_match<70"] = 60
        notes.append("Confidence capped at 60: category match score below 70.")
    if category_integrity_score < 70:
        ceilings["category_integrity<70"] = 55
        notes.append("Confidence capped at 55: category integrity below 70 (contamination).")
    if included_ads_count is not None and included_ads_count < 10:
        ceilings["included<10"] = 50
        notes.append("Confidence capped at 50: fewer than 10 category-matched ads.")

    final = raw
    if ceilings:
        final = min([raw] + list(ceilings.values()))

    return ConfidenceResult(
        components={k: round(v, 1) for k, v in components.items()},
        raw=round(raw, 1), ceilings=ceilings, final=round(final, 1),
        band=_band(final), notes=notes,
    )


# ---- per-insight confidence caps (BUG 7, 9) ------------------------------- #
def hook_insight_cap(hook_unknown_rate: float) -> int | None:
    return 50 if hook_unknown_rate > 0.50 else None


def cta_insight_cap(download_frequency_pct: float, cta_reliable: bool) -> int | None:
    if download_frequency_pct <= 0 or not cta_reliable:
        return 30
    return None


def whitespace_opportunity_cap(has_external_benchmark: bool) -> int:
    """Absent patterns are capped at 65 unless external benchmark data exists."""
    return 100 if has_external_benchmark else 65


def apply_cap(confidence: float, cap: int | None) -> float:
    return min(confidence, cap) if cap is not None else confidence
