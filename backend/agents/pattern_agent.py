"""Agent 5 — Pattern Mining.

THE evidence engine. Every finding is a real frequency computed over the
extracted features of the analysed ads — never invented. Produces both
categorical-distribution patterns (top hook/CTA/emotion/format) and
boolean-prevalence patterns (human present, app screen visible, UGC, …), plus
platform-specific breakdowns. Output statements look like:

    "Problem-first hooks appeared in 74% of analysed ads (20/27)."
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from schemas.models import ExtractedFeatures, PatternFinding

# (feature attribute, human label)
_CATEGORICAL = [
    ("hook_type", "hook"),
    ("cta_type", "CTA type"),
    ("emotion_type", "emotional trigger"),
    ("creative_format", "creative format"),
]
# (feature attribute, label when True)
_BOOLEAN = [
    ("human_present", "featured a human"),
    ("face_visible", "showed a face early"),
    ("app_screen_visible", "showed the app screen early"),
    ("product_demo_present", "included a product demo"),
    ("ugc_style", "used UGC-style creative"),
]

_LABELS = {
    "problem_first": "Problem-first hooks",
    "question": "Question hooks",
    "curiosity": "Curiosity hooks",
    "social_proof": "Social-proof hooks",
    "offer": "Offer-led hooks",
    "statement": "Statement hooks",
    "ugc_video": "UGC-style videos",
    "brand_video": "Brand-style videos",
    "screenshot": "App-screenshot creatives",
    "lifestyle_image": "Lifestyle images",
}


_FILLER = {"none", "unknown", "neutral", ""}


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total else 0.0


def _nice(value: str) -> str:
    return _LABELS.get(value, value.replace("_", " ").capitalize())


class PatternMiningAgent:
    name = "pattern_agent"
    MIN_PREVALENCE = 25.0  # only surface boolean patterns at/above this %

    def mine(self, items: list[tuple[ExtractedFeatures, str | None]]) -> list[PatternFinding]:
        total = len(items)
        findings: list[PatternFinding] = []
        if total == 0:
            return findings

        features = [f for f, _ in items]

        # --- categorical: most-common value per dimension -------------------
        for attr, label in _CATEGORICAL:
            values = [getattr(f, attr) for f in features if getattr(f, attr)]
            if not values:
                continue
            value, count = Counter(values).most_common(1)[0]
            if value in _FILLER:
                # Prefer a meaningful value; if none exists, don't surface a
                # filler ("neutral"/"none"/"unknown") as a headline finding.
                meaningful = [v for v in values if v not in _FILLER]
                if not meaningful:
                    continue
                value, count = Counter(meaningful).most_common(1)[0]
            pct = _pct(count, total)
            findings.append(
                PatternFinding(
                    pattern_type=f"top_{attr}",
                    pattern_value=value,
                    frequency=count,
                    sample_size=total,
                    percentage=pct,
                    statement=f"{_nice(value)} were the most common {label}, in {pct}% of analysed ads ({count}/{total}).",
                    evidence={"distribution": dict(Counter(values))},
                )
            )

        # --- boolean prevalence --------------------------------------------
        for attr, label in _BOOLEAN:
            count = sum(1 for f in features if getattr(f, attr))
            pct = _pct(count, total)
            if pct >= self.MIN_PREVALENCE:
                findings.append(
                    PatternFinding(
                        pattern_type=f"prevalence_{attr}",
                        pattern_value="true",
                        frequency=count,
                        sample_size=total,
                        percentage=pct,
                        statement=f"{pct}% of analysed ads {label} ({count}/{total}).",
                        evidence={"true": count, "false": total - count},
                    )
                )

        # --- platform-specific ---------------------------------------------
        findings.extend(self._platform_patterns(items))

        findings.sort(key=lambda p: p.percentage, reverse=True)
        return findings

    def _platform_patterns(
        self, items: list[tuple[ExtractedFeatures, str | None]]
    ) -> list[PatternFinding]:
        by_platform: dict[str, list[ExtractedFeatures]] = {}
        for feat, plat in items:
            if plat:
                by_platform.setdefault(plat, []).append(feat)

        out: list[PatternFinding] = []
        for plat, feats in by_platform.items():
            total = len(feats)
            if total < 2:
                continue
            # dominant hook & emotion on this platform
            for attr, label in (("hook_type", "hooks"), ("emotion_type", "emotional triggers")):
                vals = [getattr(f, attr) for f in feats if getattr(f, attr) not in (None, "unknown", "neutral")]
                if not vals:
                    continue
                value, count = Counter(vals).most_common(1)[0]
                pct = _pct(count, total)
                out.append(
                    PatternFinding(
                        pattern_type=f"platform_top_{attr}",
                        pattern_value=value,
                        platform=plat,
                        frequency=count,
                        sample_size=total,
                        percentage=pct,
                        statement=f"On {plat}, {_nice(value)} dominated {label} ({pct}%, {count}/{total}).",
                        evidence={"platform": plat, "distribution": dict(Counter(vals))},
                    )
                )
            ugc = sum(1 for f in feats if f.ugc_style)
            out.append(
                PatternFinding(
                    pattern_type="platform_ugc_rate",
                    pattern_value="ugc",
                    platform=plat,
                    frequency=ugc,
                    sample_size=total,
                    percentage=_pct(ugc, total),
                    statement=f"{_pct(ugc, total)}% of {plat} creatives were UGC-style ({ugc}/{total}).",
                    evidence={"platform": plat},
                )
            )
        return out
