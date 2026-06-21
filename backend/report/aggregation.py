"""Aggregation rules (BUG 2, 4, 5, 6).

Turns per-ad taxonomy into evidence-gated field aggregates. Every aggregate
carries its support tier, unknown rate and a `reliable` flag. Nothing downstream
is allowed to call a value "dominant" unless its aggregate says so — that single
gate is what stops the executive summary from inventing a Statement hook or a
Download CTA.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .taxonomy import NON_SIGNAL, UNCLASSIFIED

# ---- thresholds (BUG 6) --------------------------------------------------- #
LOW_SUPPORT_MAX = 2          # support_count <= 2  -> low_support
EMERGING_MAX = 5             # 3..5                -> emerging
DOMINANT_MIN_FREQ = 30.0     # > this and support > 5 -> dominant
SATURATED_MIN_FREQ = 60.0    # > this              -> saturated
RELIABLE_MIN_SUPPORT = 3
RELIABLE_MAX_UNKNOWN = 0.50  # field unreliable once >= 50% unclassified


def support_tier(support_count: int, frequency_pct: float) -> str:
    if support_count <= LOW_SUPPORT_MAX:
        return "low_support"
    if frequency_pct > SATURATED_MIN_FREQ:
        return "saturated"
    if support_count > EMERGING_MAX and frequency_pct > DOMINANT_MIN_FREQ:
        return "dominant"
    if RELIABLE_MIN_SUPPORT <= support_count <= EMERGING_MAX:
        return "emerging"
    return "low_support"


@dataclass
class FieldAggregate:
    field_name: str
    total: int
    distribution: dict[str, int]
    dominant_value: str
    dominant_count: int
    dominant_frequency: float        # 0..100, of total
    unknown_count: int
    unknown_rate: float              # 0..1  (UNCLASSIFIED only — drives reliability)
    gap_count: int = 0               # UNCLASSIFIED + NON_SIGNAL — drives taxonomy quality
    tier: str = "low_support"
    reliable: bool = False
    note: str = ""

    @property
    def is_dominant(self) -> bool:
        return self.reliable and self.tier in ("dominant", "saturated")


def aggregate_field(field_name: str, values: list[str]) -> FieldAggregate:
    total = len(values)
    counts = Counter(v if v not in ("", None) else "unknown" for v in values)
    unknown_count = sum(c for v, c in counts.items() if v in UNCLASSIFIED)
    gap_count = sum(c for v, c in counts.items() if v in UNCLASSIFIED or v in NON_SIGNAL)
    unknown_rate = (unknown_count / total) if total else 1.0

    # dominant among *signal-bearing* values only
    signal = {v: c for v, c in counts.items()
              if v not in UNCLASSIFIED and v not in NON_SIGNAL}
    if signal:
        dom_value, dom_count = max(signal.items(), key=lambda kv: kv[1])
    else:
        dom_value, dom_count = "unknown", unknown_count

    freq = (100.0 * dom_count / total) if total else 0.0
    tier = support_tier(dom_count, freq) if dom_value not in UNCLASSIFIED else "low_support"
    reliable = (
        dom_value not in UNCLASSIFIED
        and dom_count >= RELIABLE_MIN_SUPPORT
        and unknown_rate < RELIABLE_MAX_UNKNOWN
    )

    note = ""
    if unknown_rate >= RELIABLE_MAX_UNKNOWN:
        note = (f"{field_name} is under-classified ({unknown_rate:.0%} unknown/uncertain); "
                f"{field_name}-level conclusions are unreliable.")
    elif not reliable:
        note = f"{field_name} dominant value has low support ({dom_count}/{total})."

    return FieldAggregate(
        field_name=field_name, total=total, distribution=dict(counts),
        dominant_value=dom_value, dominant_count=dom_count,
        dominant_frequency=round(freq, 1), unknown_count=unknown_count,
        unknown_rate=round(unknown_rate, 3), gap_count=gap_count,
        tier=tier, reliable=reliable, note=note,
    )


def aggregate_boolean(field_name: str, flags: list[bool]) -> FieldAggregate:
    """Prevalence of a boolean signal (e.g. product_demo_present)."""
    total = len(flags)
    count = sum(1 for f in flags if f)
    freq = (100.0 * count / total) if total else 0.0
    tier = support_tier(count, freq)
    reliable = count >= RELIABLE_MIN_SUPPORT
    return FieldAggregate(
        field_name=field_name, total=total,
        distribution={"true": count, "false": total - count},
        dominant_value="true" if count else "false", dominant_count=count,
        dominant_frequency=round(freq, 1), unknown_count=0, unknown_rate=0.0,
        tier=tier, reliable=reliable,
    )


@dataclass
class Aggregation:
    sample_size: int
    fields: dict[str, FieldAggregate]
    booleans: dict[str, FieldAggregate]

    def overall_unknown_rate(self) -> float:
        """Signal-gap rate across categorical fields (drives taxonomy quality).

        Counts cells that carry no usable creative signal (unclassified OR
        non-signal like 'none'/'neutral') over all cells, so a 0%-CTA field and an
        80%-unknown-hook field correctly drag taxonomy quality down instead of
        being averaged away by a well-classified format field.
        """
        cats = list(self.fields.values())
        total = sum(f.total for f in cats)
        gap = sum(f.gap_count for f in cats)
        return (gap / total) if total else 1.0

    def hook_unknown_rate(self) -> float:
        f = self.fields.get("hook")
        return f.unknown_rate if f else 1.0


def build_aggregation(ads: list[dict]) -> Aggregation:
    """`ads` are normalised per-ad taxonomy dicts (output of the taxonomy stage)."""
    n = len(ads)

    def col(key: str) -> list[str]:
        return [str(a.get(key, "unknown") or "unknown") for a in ads]

    def boolcol(key: str) -> list[bool]:
        return [bool(a.get(key, False)) for a in ads]

    fields = {
        "hook": aggregate_field("hook", col("hook_type")),
        "emotion": aggregate_field("emotion", col("emotion_type")),
        "cta": aggregate_field("cta", col("cta_type")),
        "format": aggregate_field("format", col("creative_format")),
    }
    booleans = {
        "product_demo": aggregate_boolean("product_demo", boolcol("product_demo_present")),
        "app_screen_early": aggregate_boolean("app_screen_early", boolcol("app_screen_visible")),
        "human_present": aggregate_boolean("human_present", boolcol("human_present")),
    }
    return Aggregation(sample_size=n, fields=fields, booleans=booleans)


# --------------------------------------------------------------------------- #
# Pattern labelling (BUG 5): separate frequency / lift / proof concepts.
# --------------------------------------------------------------------------- #
def classify_pattern(support_count: int, frequency_pct: float,
                     performance_lift: float, has_perf_data: bool) -> dict:
    """Return {label, claim_class, verb} obeying BUG 5 / BUG 6 wording rules."""
    tier = support_tier(support_count, frequency_pct)
    high_lift = performance_lift > 0

    if has_perf_data and high_lift and tier in ("dominant", "saturated"):
        return {"label": "proven_winner", "claim_class": "proven_winner",
                "verb": "proven winner"}
    if tier == "saturated":
        return {"label": "saturated_pattern", "claim_class": "saturated",
                "verb": "saturated (likely overused) pattern"}
    if tier == "dominant":
        return {"label": "dominant_pattern", "claim_class": "dominant",
                "verb": "dominant pattern"}
    if tier == "emerging":
        return {"label": "emerging_signal", "claim_class": "emerging",
                "verb": "emerging signal"}
    # support_count < 3
    if high_lift:
        return {"label": "low_support_high_lift", "claim_class": "low_support",
                "verb": "low-support high-lift signal"}
    return {"label": "low_support_observation", "claim_class": "low_support",
            "verb": "low-support observation"}
