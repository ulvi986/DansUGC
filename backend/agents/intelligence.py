"""Market Intelligence Engine — credible, evidence-locked creative analytics.

Deterministic analytics over the per-ad features the upstream agents extracted
(hook, CTA, emotion, format, UGC, demo, copy structure, creative score,
platform). No extra model calls, so every number traces to the dataset.

Credibility contract (this engine does NOT overclaim)
-----------------------------------------------------
* Unless the dataset carries REAL performance metrics (spend / CTR / CPA / ROAS /
  impressions / installs / conversions), the engine never says "winner",
  "winning" or "proven winner". It speaks in proxy terms: "strongest observed
  creative pattern", "highest-scoring creative proxy", "observed dominant
  pattern". A disclaimer is attached wherever strategy is summarised.
* Confidence is capped by sample size (N<20 -> 65, N<50 -> 80, else 90) and
  absent-pattern whitespace in small samples is capped at 60.
* A Creative-DNA component only enters the main formula if it is actually
  supported (usage >= 25% OR score-lift >= +3 pts) AND n >= 3. Sparse signals are
  reported separately as "low-frequency signals", never as dominant DNA.
* The creative score is labelled a *creative-quality proxy*, not performance.

All JSON keys consumed by the existing frontend are preserved; new sections
(platform_intelligence, creative_brief, disclaimer) are additive.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Callable, Iterable

from agents import lexicon

# --------------------------------------------------------------------------- #
# Labels & taxonomies
# --------------------------------------------------------------------------- #
_LABELS = {
    "problem_first": "Problem-first", "question": "Question", "curiosity": "Curiosity",
    "social_proof": "Social-proof", "offer": "Offer-led", "statement": "Statement",
    "claim_first": "Claim-first", "comparison": "Comparison", "before_after": "Before / after",
    "founder_story": "Founder story", "authority": "Authority positioning",
    "ugc_video": "UGC video", "brand_video": "Brand video", "demo_video": "Demo video",
    "screenshot": "App screenshot", "lifestyle_image": "Lifestyle image",
    "static_product_image": "Static product image", "carousel": "Carousel",
    "founder_talking_head": "Founder talking-head", "testimonial": "Testimonial",
    "expert_interview": "Expert interview", "product_demo": "Product demo",
    "app_screen": "App screen",
    "try_free": "Free-trial", "learn_more": "Learn-more", "download": "Download",
    "signup": "Sign-up", "shop": "Shop", "claim_offer": "Claim-offer",
    "get_discount": "Get-discount", "see_routine": "See-routine",
    # emotions / claim-intensity / benefit display
    "insecurity": "Insecurity", "relief": "Relief", "confidence": "Confidence",
    "urgency": "Urgency", "trust": "Trust", "aspiration": "Aspiration", "fear": "Fear",
}

# Underused-lever priors. founder is a HOOK lever; the FORMAT counterpart is the
# founder *talking-head* — kept distinct so founder never duplicates across both.
_LEVERS: dict[str, dict[str, float]] = {
    "hook": {
        "problem_first": 0.60, "question": 0.60, "curiosity": 0.70,
        "social_proof": 0.85, "offer": 0.60, "statement": 0.55,
        "founder_story": 0.90, "authority": 0.85, "comparison": 0.70,
        "before_after": 0.80,
    },
    "emotion": {
        "fear": 0.70, "relief": 0.55, "aspiration": 0.80, "curiosity": 0.70,
        "trust": 0.80, "urgency": 0.85, "confidence": 0.70, "insecurity": 0.65,
    },
    "cta": {
        "download": 0.50, "try_free": 0.80, "signup": 0.60,
        "shop": 0.60, "learn_more": 0.50, "get_discount": 0.55, "see_routine": 0.70,
    },
    "format": {
        "ugc_video": 0.60, "brand_video": 0.55, "demo_video": 0.80,
        "lifestyle_image": 0.55, "product_demo": 0.80, "carousel": 0.55,
        "founder_talking_head": 0.90, "before_after": 0.85, "expert_interview": 0.85,
        "testimonial": 0.75,
    },
}

_POSITIVE_EMOTIONS = {
    "relief", "calm", "aspiration", "joy", "trust", "hope",
    "confidence", "happiness", "excitement", "empowerment",
}
_FILLER = {"", "none", "unknown", "neutral", None}

DISCLAIMER = (
    "This report identifies repeated creative patterns and proxy-quality signals "
    "from the observed ad sample. It does not claim true performance winners unless "
    "spend, engagement, or conversion data is available."
)
SCORE_LABEL = "Average creative-quality proxy score"
SCORE_EXPLANATION = (
    "This score is based on observable creative factors such as hook clarity, "
    "pain-point clarity, visual clarity, product demonstration, emotional trigger, "
    "CTA strength, and platform fit. It is not a substitute for spend, engagement, "
    "or conversion data."
)


def nice(value: str | None) -> str:
    if not value:
        return "Unknown"
    return _LABELS.get(value, str(value).replace("_", " ").capitalize())


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total else 0.0


def _safe_mean(xs: Iterable[float], default: float = 0.0) -> float:
    xs = list(xs)
    return mean(xs) if xs else default


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _mode(values: Iterable[str], default: str = "unknown") -> tuple[str, int]:
    vals = [v for v in values if v not in _FILLER]
    if not vals:
        return default, 0
    return Counter(vals).most_common(1)[0]


def _mode_share(values: Iterable[str]) -> float:
    vals = [v for v in values if v not in _FILLER]
    if not vals:
        return 0.0
    _, c = Counter(vals).most_common(1)[0]
    return c / len(vals)


def _bool_agreement(flags: list[bool]) -> float:
    if not flags:
        return 1.0
    t = sum(1 for f in flags if f)
    return max(t, len(flags) - t) / len(flags)


def _topk(values: Iterable[str], total: int, k: int = 3) -> list[dict[str, Any]]:
    counts = Counter(v for v in values if v not in _FILLER)
    return [{"value": v, "count": c, "pct": _pct(c, total)} for v, c in counts.most_common(k)]


def signal_label(n: int) -> str:
    """Sample-count strength label (rule 11)."""
    if n < 3:
        return "weak signal"
    if n <= 5:
        return "directional signal"
    return "stronger observed signal"


def saturation_label(pct: float) -> str:
    """Usage-percentage saturation band (rule 7)."""
    if pct <= 15:
        return "low usage / whitespace"
    if pct <= 35:
        return "emerging pattern"
    if pct <= 55:
        return "common pattern"
    if pct <= 75:
        return "saturated pattern"
    return "heavily saturated pattern"


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class IntelligenceEngine:
    def __init__(self, profiles: list[dict[str, Any]], platforms: list[str] | None = None,
                 has_performance_data: bool | None = None):
        self.p = profiles
        self.total = len(profiles)
        self.platforms = platforms or sorted({p["platform"] for p in profiles if p.get("platform")})
        self.baseline = _safe_mean([p["score"] for p in profiles])
        if has_performance_data is None:
            has_performance_data = any(p.get("has_perf") for p in profiles)
        self.has_perf = bool(has_performance_data)

    # ---- confidence governance ------------------------------------------- #
    def _sample_cap(self) -> int:
        if self.total < 20:
            return 65
        if self.total < 50:
            return 80
        return 90

    def _cap(self, conf: float, *, whitespace_zero: bool = False) -> int:
        cap = self._sample_cap()
        if whitespace_zero and self.total < 30:
            cap = min(cap, 60)
        return int(round(min(conf, cap)))

    def _small_sample_note(self) -> str:
        return ("Small sample — saturation estimate should be treated as directional."
                if self.total < 20 else "")

    # ---- public ---------------------------------------------------------- #
    def build(self) -> dict[str, Any]:
        clusters = self.creative_clusters()
        winners = self.winner_patterns()
        gaps = self.opportunity_gaps()
        dna = self.creative_dna()
        saturation = self.market_saturation(top_gap=gaps[0] if gaps else None)
        strategies = self.strategy_triad(dna, winners, gaps, saturation)
        market_map = self.market_map()
        platform_intel = self.platform_intelligence(gaps)
        brief = self.creative_brief(dna, winners, gaps)
        executive = self.executive_summary(winners, gaps, saturation, dna)

        return {
            "disclaimer": DISCLAIMER,
            "has_performance_data": self.has_perf,
            "executive_summary": executive,
            "creative_dna": dna,
            "winner_patterns": winners,
            "creative_clusters": clusters,
            "opportunity_gaps": gaps,
            "market_saturation": saturation,
            "strategies": strategies,
            "platform_intelligence": platform_intel,
            "creative_brief": brief,
            "market_map": market_map,
            "meta": {
                "analyzed_ads": self.total,
                "platforms": self.platforms,
                "method": "deterministic analytics over agent-extracted features",
                "score_label": SCORE_LABEL,
                "score_explanation": SCORE_EXPLANATION,
                "disclaimer": DISCLAIMER,
                "confidence_cap": self._sample_cap(),
                "caveat": (
                    "Source ads carry no engagement metrics, so the strongest patterns "
                    "are measured by creative-quality score lift and consistency, not by "
                    "clicks/installs." if not self.has_perf else
                    "Performance metrics present; performance-backed claims are permitted."
                ),
            },
        }

    # ---- 1. creative clustering ----------------------------------------- #
    def creative_clusters(self) -> list[dict[str, Any]]:
        groups: dict[tuple, list[dict]] = {}
        for p in self.p:
            groups.setdefault((p["hook"], p["format"]), []).append(p)

        clusters: list[dict[str, Any]] = []
        for (hook, fmt), items in groups.items():
            n = len(items)
            if n < 2:
                continue
            cta, _ = _mode([x["cta"] for x in items], "none")
            emo, _ = _mode([x["emotion"] for x in items], "neutral")
            cta_share = _mode_share([x["cta"] for x in items])
            emo_share = _mode_share([x["emotion"] for x in items])
            ugc_agree = _bool_agreement([x["ugc"] for x in items])
            consistency = _clamp((cta_share + emo_share + ugc_agree) / 3)
            avg_score = _safe_mean([x["score"] for x in items])
            size_factor = _clamp(n / (0.3 * self.total)) if self.total else 0.0
            confidence = self._cap(100 * _clamp(
                0.45 * consistency + 0.35 * size_factor + 0.20 * (avg_score / 100)))
            clusters.append({
                "cluster_name": f"{nice(fmt)} · {nice(hook)} hook",
                "ads_count": n, "frequency_pct": _pct(n, self.total),
                "confidence": confidence, "consistency": round(consistency * 100),
                "signal": signal_label(n),
                "dominant_hook": hook, "dominant_cta": cta, "dominant_emotion": emo,
                "common_emotions": _topk([x["emotion"] for x in items], n),
                "common_ctas": _topk([x["cta"] for x in items], n),
                "visual_structure": {
                    "ugc_pct": round(100 * sum(1 for x in items if x["ugc"]) / n),
                    "product_demo_pct": round(100 * sum(1 for x in items if x["product_demo"]) / n),
                    "app_screen_pct": round(100 * sum(1 for x in items if x["app_screen"]) / n),
                },
                "avg_creative_score": round(avg_score, 1),
                "reasoning": (
                    f"{n} ads ({_pct(n, self.total)}%) share a {nice(fmt)} format with a "
                    f"{nice(hook)} hook; {round(cta_share * 100)}% use a '{cta}' CTA and "
                    f"{round(emo_share * 100)}% evoke '{emo}'. Mean creative-quality proxy "
                    f"{round(avg_score)}/100 ({signal_label(n)})."
                ),
            })
        clusters.sort(key=lambda c: (c["ads_count"], c["confidence"]), reverse=True)
        return clusters[:6]

    # ---- 2. strongest observed patterns (was "winner patterns") --------- #
    def winner_patterns(self) -> list[dict[str, Any]]:
        if self.total < 2:
            return []
        candidates: list[tuple[str, str, Callable[[dict], bool]]] = []
        for dim in ("hook", "cta", "emotion", "format"):
            for val in {p[dim] for p in self.p if p[dim] not in _FILLER}:
                candidates.append((f"{dim}:{val}", f"{nice(val)} {dim}", lambda p, d=dim, v=val: p[d] == v))
        for attr, label in (
            ("ugc", "UGC-style creative"), ("product_demo", "Product demo"),
            ("app_screen", "App screen shown early"), ("human", "Human on-screen"),
            ("face", "Face shown early"),
        ):
            candidates.append((f"trait:{attr}", label, lambda p, a=attr: bool(p[a])))

        out: list[dict[str, Any]] = []
        for key, label, fn in candidates:
            withs = [p for p in self.p if fn(p)]
            n = len(withs)
            if n < 2:
                continue
            withouts = [p for p in self.p if not fn(p)]
            avg_with = _safe_mean([p["score"] for p in withs])
            avg_without = _safe_mean([p["score"] for p in withouts], avg_with)
            lift = (avg_with - avg_without) / 100.0

            msg_c = _mode_share([p["copy_structure"] for p in withs])
            vis_c = (_mode_share([p["format"] for p in withs]) + _bool_agreement([p["ugc"] for p in withs])) / 2
            emo_c = _mode_share([p["emotion"] for p in withs])
            platform_fit = _mode_share([p["platform"] for p in withs])

            performance = _clamp(0.5 + lift)
            consistency = _clamp((msg_c + vis_c + emo_c) / 3)
            support = _clamp(n / (0.3 * self.total))
            confidence = self._cap(100 * _clamp(
                0.35 * performance + 0.30 * consistency + 0.20 * support + 0.15 * platform_fit))
            top_platform, _ = _mode([p["platform"] for p in withs], "n/a")
            example = next((p["ad_text"] for p in sorted(withs, key=lambda x: x["score"], reverse=True) if p["ad_text"]), "")

            lift_pts = round(lift * 100, 1)
            # honest, sample-aware phrasing
            if self.has_perf:
                verdict = f"strong performer (+{lift_pts}-pt lift)" if lift_pts > 0 else "no measured lift"
            elif lift_pts > 0:
                verdict = "highest-scoring creative proxy"
            else:
                verdict = "high-frequency but no score lift"

            out.append({
                "pattern": label, "key": key, "confidence": confidence,
                "ads_count": n, "frequency_pct": _pct(n, self.total),
                "score_lift": lift_pts, "avg_score_with": round(avg_with, 1),
                "avg_score_without": round(avg_without, 1),
                "signal": signal_label(n),
                "pattern_confidence": confidence,
                "performance_confidence": (confidence if self.has_perf else None),
                "confidence_breakdown": {
                    "score_proxy_lift": round(performance * 100),
                    "consistency": round(consistency * 100),
                    "repetition_support": round(support * 100),
                    "platform_fit": round(platform_fit * 100),
                },
                "evidence": [
                    f"Present in {n}/{self.total} ads ({_pct(n, self.total)}%) — {signal_label(n)}.",
                    f"Creative-quality proxy {avg_with:.0f}/100 with it vs {avg_without:.0f} without "
                    f"(proxy lift {lift_pts:+.0f} pts).",
                    f"Internally consistent — message {msg_c:.0%}, visual {vis_c:.0%}, emotional {emo_c:.0%}.",
                    f"Most concentrated on '{top_platform}' ({platform_fit:.0%} of matches).",
                ] + ([f"Top example: “{example[:120]}”"] if example else []),
                "reasoning": (
                    f"{label} is the {verdict} here, on a {lift_pts:+.0f}-pt creative-quality "
                    f"proxy lift with {consistency:.0%} cross-signal consistency. This is "
                    f"internally consistent across the observed ads, but actual performance "
                    f"cannot be confirmed without engagement or conversion metrics."
                ),
            })

        # supported signals (n>=3) lead; then positive proxy lift; then confidence —
        # so a 2-ad weak signal never headlines as the "strongest observed pattern".
        out.sort(key=lambda w: (w["ads_count"] >= 3, w["score_lift"] > 0, w["confidence"]),
                 reverse=True)
        return out[:8]

    # ---- 3. opportunity gaps / whitespace ------------------------------- #
    def opportunity_gaps(self) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []
        present = {dim: Counter(p[dim] for p in self.p) for dim in ("hook", "emotion", "cta", "format")}
        for dim, levers in _LEVERS.items():
            counts = present[dim]
            for value, weight in levers.items():
                n = counts.get(value, 0)
                freq = n / self.total if self.total else 0.0
                if freq >= 0.12:
                    continue
                whitespace_zero = (n == 0)
                confidence = self._cap(100 * _clamp((1 - freq) * weight),
                                       whitespace_zero=whitespace_zero)
                usage = _pct(n, self.total)
                why = self._why_matters(dim, value)
                risk = ("Unproven in this observed sample." if whitespace_zero
                        else "Underused here; limited in-sample evidence.")
                test = self._test_idea(dim, value)
                reason = (f"Observed whitespace: {nice(value).lower()} is absent from this sample."
                          if whitespace_zero else
                          f"Appears in only {usage}% of ads ({n}/{self.total}) — {signal_label(n)}.")
                gaps.append({
                    "type": dim, "opportunity": value, "label": nice(value),
                    "confidence": confidence, "opportunity_confidence": confidence,
                    "current_usage_pct": usage, "current_usage": f"{n}/{self.total} ads",
                    "reason": reason, "why_it_matters": why, "risk": risk, "test_idea": test,
                    "reasoning": (
                        f"{nice(value)} is underused in this sample. {why} "
                        f"Treat as a test idea, not a guaranteed opportunity — "
                        f"confidence adjusted for sample size (N={self.total})."
                    ),
                })
        gaps.sort(key=lambda g: g["confidence"], reverse=True)
        return gaps[:6]

    @staticmethod
    def _why_matters(dim: str, value: str) -> str:
        if value in ("founder_story", "founder_talking_head", "authority", "expert_interview"):
            return "Could create differentiation against repetitive problem-first ads."
        if value in ("before_after",):
            return "Strong visual proof is underused and tends to lift credibility."
        if dim == "emotion":
            return "A fresh emotional angle can break ad fatigue in a repetitive market."
        return "An underused lever can stand out in an otherwise homogeneous market."

    @staticmethod
    def _test_idea(dim: str, value: str) -> str:
        ideas = {
            "founder_story": "Founder/esthetician talking-head explaining why most routines fail.",
            "founder_talking_head": "Founder/esthetician talking-head explaining the mechanism.",
            "authority": "Expert/dermatologist framing the problem with credibility cues.",
            "before_after": "Day 1 / Day 7 split-screen showing visible improvement.",
            "social_proof": "Stack real user counts / ratings against the dominant hook.",
        }
        return ideas.get(value, f"A/B a {nice(value).lower()} variant against the dominant pattern.")

    # ---- 4. creative DNA (gated inclusion) ------------------------------ #
    def creative_dna(self) -> dict[str, Any]:
        ranked = sorted(self.p, key=lambda p: p["score"], reverse=True)
        top = ranked[: max(3, self.total // 2)] if self.total >= 4 else ranked
        nt = len(top) or 1

        def field(dim: str, default: str) -> dict[str, Any]:
            # dominant value over the FULL set (usage), with score lift over full set
            val, c_full = _mode([p[dim] for p in self.p], default)
            usage_pct = _pct(c_full, self.total)
            withs = [p["score"] for p in self.p if p[dim] == val]
            withouts = [p["score"] for p in self.p if p[dim] != val]
            lift = round(_safe_mean(withs) - _safe_mean(withouts, _safe_mean(withs)), 1)
            n_top = sum(1 for p in top if p[dim] == val)
            included = (c_full >= 3) and (usage_pct >= 25.0 or lift >= 3.0)
            return {
                "value": val, "label": nice(val),
                "support_pct": usage_pct, "n": c_full, "score_lift": lift,
                "signal": signal_label(c_full), "included": included,
                "n_top": n_top,
            }

        hook = field("hook", "problem_first")
        emotion = field("emotion", "relief")
        fmt = field("format", "ugc_video")
        cta = field("cta", "download")

        # claim-intensity / benefit observed in the top ads (not emotions)
        claim_lbl, _ = _mode([lexicon.classify_claim_intensity(p["ad_text"])[0] for p in top
                              if lexicon.classify_claim_intensity(p["ad_text"])[0]], "")
        benefit_lbl, _ = _mode([lexicon.classify_benefit(p["ad_text"])[0] for p in top
                               if lexicon.classify_benefit(p["ad_text"])[0]], "")

        pains = Counter(t for p in top for t in p.get("pain_points", []) if t)
        from_state = pains.most_common(1)[0][0] if pains else "the problem"
        to_state = benefit_lbl.replace("_", " ") if benefit_lbl else (
            emotion["value"] if emotion["value"] in _POSITIVE_EMOTIONS else "resolution")

        core = [f for f in (hook, fmt) if f["included"]]
        low_freq = [
            {"dimension": dim, "label": f["label"], "usage_pct": f["support_pct"],
             "n": f["n"], "signal": signal_label(f["n"]),
             "note": f"{f['label']} appears in {f['n']}/{self.total} ads "
                     f"({f['support_pct']}%) — {signal_label(f['n'])}, too sparse to call dominant."}
            for dim, f in (("hook", hook), ("emotion", emotion), ("format", fmt), ("cta", cta))
            if not f["included"]
        ]

        # build an honest formula sentence
        if core:
            strongest = " + ".join(f"{f['label']}" for f in core)
            parts = [f"The strongest observed pattern is {strongest}"
                     + (" format." if fmt["included"] else ".")]
        else:
            parts = ["No single component is supported strongly enough to call a dominant formula."]
        if claim_lbl or benefit_lbl:
            cl = nice(claim_lbl) if claim_lbl else None
            bn = benefit_lbl.replace("_", " ") if benefit_lbl else None
            phrase = " / ".join(x for x in (cl, bn) if x)
            parts.append(f"Claim/benefit language around “{phrase}” appears in the highest-scoring subset.")
        if not cta["included"]:
            parts.append(f"CTA usage is fragmented and too sparse to call a dominant CTA "
                         f"({cta['label']} only {cta['n']}/{self.total}).")
        formula = " ".join(parts)

        # confidence only over INCLUDED components (no sparse padding)
        inc_support = [f["support_pct"] for f in (hook, emotion, fmt, cta) if f["included"]]
        confidence = self._cap(mean(inc_support) if inc_support else hook["support_pct"])

        return {
            "hook": hook, "emotion": emotion, "format": fmt, "cta": cta,
            "claim_intensity": {"value": claim_lbl, "label": nice(claim_lbl) if claim_lbl else "—"},
            "benefit": {"value": benefit_lbl, "label": benefit_lbl.replace("_", " ") if benefit_lbl else "—"},
            "transformation": {"from": from_state, "to": to_state},
            "low_frequency_signals": low_freq,
            "confidence": confidence, "formula": formula,
            "basis": f"Derived from {self.total} ads (top {nt} by creative-quality proxy for emphasis). "
                     f"A component enters the formula only if usage ≥ 25% or proxy lift ≥ +3 pts, with n ≥ 3.",
            "disclaimer": DISCLAIMER,
        }

    # ---- 5. market saturation ------------------------------------------- #
    def market_saturation(self, top_gap: dict | None) -> list[dict[str, Any]]:
        diff = top_gap["label"] if top_gap else "an underused angle"
        note = self._small_sample_note()
        rows: list[dict[str, Any]] = []
        for dim, label in (("hook", "hook"), ("format", "format"), ("cta", "CTA"), ("emotion", "emotion")):
            val, c = _mode([p[dim] for p in self.p])
            if not c:
                continue
            sat = _pct(c, self.total)
            band = saturation_label(sat)
            risk = "high" if sat >= 56 else "medium" if sat >= 36 else "low"
            rec = (f"{sat:.0f}% of ads already use this {label} ({band}) — differentiate by testing {diff}."
                   if risk != "low" else
                   f"Still headroom on this {label} ({band}); safe to lean in while monitoring.")
            if note:
                rec = f"{rec} {note}"
            rows.append({
                "pattern": f"{nice(val)} {label}", "dimension": dim, "value": val,
                "saturation": round(sat), "saturation_label": band, "risk": risk,
                "ads_count": c, "signal": signal_label(c),
                "directional": self.total < 20, "recommendation": rec,
            })
        rows.sort(key=lambda r: r["saturation"], reverse=True)
        return rows

    # ---- 6. strategy triad (honest A/B/C) ------------------------------- #
    def strategy_triad(self, dna, winners, gaps, saturation) -> dict[str, Any]:
        top_winner = winners[0] if winners else None
        top_gap = gaps[0] if gaps else None
        most_saturated = saturation[0] if saturation else None
        dom = " + ".join(f["label"] for f in (dna["hook"], dna["format"]) if f["included"]) \
            or f"{dna['hook']['label']} hook"

        safe = {
            "name": "A — Replicate the dominant observed pattern",
            "thesis": "Enter the market with low creative risk by matching the strongest observed pattern.",
            "hypothesis": f"A {dom} baseline performs at least at market parity.",
            "creative_direction": f"Lead with {dom}; close on the clearest available CTA.",
            "plays": [
                f"Lead with a {dna['hook']['label']} hook.",
                f"Produce in {dna['format']['label']} format.",
                "Use when you need baseline ads with low differentiation risk.",
            ],
            "evidence": [f"{dom} is the strongest observed pattern in {self.total} ads."],
            "what_to_measure": "CTR / thumb-stop vs current baseline once minimum spend is reached.",
            "confidence": self._cap(dna["confidence"] * 0.9),
            "risk": "Low differentiation — you blend in with everyone running the same pattern.",
        }
        sharpen = {
            "name": "B — Sharpen the highest-scoring proxy pattern",
            "thesis": "Keep the dominant pattern but improve specificity, proof or emotional clarity.",
            "hypothesis": "Adding stronger proof to the dominant pattern raises hold/CTR.",
            "creative_direction": (f"Anchor on {dom}; amplify '{top_winner['pattern']}'."
                                   if top_winner else f"Anchor on {dom}; strengthen the first 3 seconds."),
            "plays": [
                f"Anchor on the {dom} base.",
                (f"Amplify '{top_winner['pattern']}' — {top_winner['score_lift']:+.0f}-pt proxy lift "
                 f"({top_winner['signal']})." if top_winner else "Strengthen CTA + first-3-seconds hook."),
                "Use when the market pattern is common but still has headroom.",
            ],
            "evidence": (top_winner["evidence"][:2] if top_winner else ["Based on the dominant observed pattern."]),
            "what_to_measure": "CVR / CPA vs the plain baseline after a minimum spend threshold.",
            "confidence": self._cap(mean([dna["confidence"], top_winner["confidence"]]) if top_winner else dna["confidence"]),
            "risk": "May still blend in — it stays inside the common lane, with an edge.",
        }
        contrarian = {
            "name": "C — Test a contrarian whitespace",
            "thesis": "Differentiate with an underused angle or format when the market looks repetitive.",
            "hypothesis": (f"A {top_gap['label']} angle out-stops the repetitive dominant pattern."
                           if top_gap else "A fresh angle out-stops the repetitive dominant pattern."),
            "creative_direction": (f"Build around {top_gap['label']} ({top_gap['type']}); {top_gap['test_idea']}"
                                   if top_gap else "Build around an underused emotional angle."),
            "plays": [
                (f"Avoid the over-used '{most_saturated['pattern']}' "
                 f"({most_saturated['saturation']}% — {most_saturated['saturation_label']})."
                 if most_saturated else "Avoid the dominant hook everyone repeats."),
                (f"Build around {top_gap['label']} ({top_gap['type']}), used in only "
                 f"{top_gap['current_usage_pct']}% of ads." if top_gap else "Test a different emotional angle."),
                "Use when the market looks repetitive and you want differentiation.",
            ],
            "evidence": [top_gap["reason"]] if top_gap else ["Whitespace inferred from low-frequency angles."],
            "what_to_measure": "Thumb-stop rate / CTR vs the dominant-pattern baseline.",
            "confidence": top_gap["confidence"] if top_gap else self._cap(50),
            "risk": "Higher variance — the pattern is not proven in this sample.",
        }
        return {"safe": safe, "winning": sharpen, "contrarian": contrarian, "disclaimer": DISCLAIMER}

    # ---- 7. platform intelligence --------------------------------------- #
    def platform_intelligence(self, gaps) -> list[dict[str, Any]]:
        top_gap = gaps[0]["label"] if gaps else "an underused angle"
        out: list[dict[str, Any]] = []
        for plat in self.platforms:
            items = [p for p in self.p if p.get("platform") == plat]
            n = len(items)
            if not n:
                continue
            hook, _ = _mode([x["hook"] for x in items])
            fmt, _ = _mode([x["format"] for x in items])
            cta, _ = _mode([x["cta"] for x in items])
            angle = Counter(t for x in items for t in x.get("pain_points", []) if t)
            angle_lbl = angle.most_common(1)[0][0] if angle else "general"
            proof = self._proof_proxy(items)
            ugc_share = sum(1 for x in items if x["ugc"]) / n
            native = (ugc_share >= 0.5) if plat.lower() == "tiktok" else (ugc_share < 0.5)
            interp = (f"{plat} creatives in this sample skew toward {nice(hook).lower()} "
                      f"{nice(fmt).lower()} ads"
                      + (", native to the platform's creator style." if native
                         else ", which is not fully native to the platform."))
            out.append({
                "platform": plat, "ads_count": n, "signal": signal_label(n),
                "dominant_hook": nice(hook), "dominant_format": nice(fmt),
                "dominant_cta": nice(cta), "dominant_proof": proof,
                "dominant_angle": angle_lbl, "native_style": native,
                "interpretation": interp,
                "so_what": {
                    "replicate": f"Reuse {nice(fmt).lower()} + {nice(hook).lower()} as the {plat} baseline.",
                    "avoid": f"Avoid over-repeating the dominant {nice(hook).lower()} hook into fatigue.",
                    "test_next": f"Test a {top_gap} variant to differentiate on {plat}.",
                },
            })
        return out

    @staticmethod
    def _proof_proxy(items: list[dict]) -> str:
        if sum(1 for x in items if x.get("product_demo")) / max(len(items), 1) >= 0.4:
            return "product demo"
        if sum(1 for x in items if x.get("app_screen")) / max(len(items), 1) >= 0.4:
            return "app screen proof"
        if _mode([x["hook"] for x in items])[0] == "social_proof":
            return "social proof"
        return "none observed"

    # ---- 8. ready-to-run creative brief --------------------------------- #
    def creative_brief(self, dna, winners, gaps) -> list[dict[str, Any]]:
        fmt = dna["format"]["label"]
        hook = dna["hook"]["label"]
        from_state = dna["transformation"]["from"]
        to_state = dna["transformation"]["to"]
        gap = next((g for g in gaps if g["type"] in ("hook", "format")), gaps[0] if gaps else None)

        variants = [{
            "variant": "A — Dominant pattern baseline",
            "strategy": "Replicate dominant observed pattern",
            "format": fmt,
            "hook": f"Problem-first label: “{from_state}?”",
            "visual_opening": f"Close-up on the problem ({from_state}) with a simple on-screen label.",
            "core_message": f"{to_state.title()} routine that addresses {from_state}.",
            "proof": "Product / routine demonstration.",
            "cta": dna["cta"]["label"] if dna["cta"]["included"] else "Shop routine",
            "why_this_test_exists": f"{hook} + {fmt} is the strongest observed pattern.",
            "winning_condition": "CTR +15% vs current baseline or CPA -10% after a minimum spend threshold.",
            "risk": "May blend into saturated problem-first ads.",
        }, {
            "variant": "B — Founder / authority whitespace",
            "strategy": "Contrarian whitespace",
            "format": "Founder or esthetician talking-head",
            "hook": "“Most people treat this as one problem. That’s the mistake.”",
            "visual_opening": "Founder / expert speaking directly to camera.",
            "core_message": f"Explain the mechanism behind {to_state}.",
            "proof": "Show the routine / product demonstration.",
            "cta": "See routine",
            "why_this_test_exists": (gap["reason"] if gap else
                                     "Founder/authority narrative is absent from the observed sample."),
            "winning_condition": "Higher thumb-stop rate or CTR than the problem-first baseline.",
            "risk": "Unproven in this sample.",
        }, {
            "variant": "C — Before/after proof test",
            "strategy": "Sharpen proxy pattern with proof",
            "format": "Before-after + routine demo",
            "hook": f"“7 days using this routine on {from_state}”",
            "visual_opening": "Day 1 / Day 7 split-screen.",
            "core_message": f"Visible {to_state} improvement.",
            "proof": "Before-after sequence + routine steps.",
            "cta": "Shop routine",
            "why_this_test_exists": "The market uses problem-first messaging but underuses strong visual proof.",
            "winning_condition": "Higher CVR or lower CPA than generic problem-first ads.",
            "risk": "Requires credible visuals and compliance-safe claims.",
        }]
        return variants

    # ---- market map ------------------------------------------------------ #
    def market_map(self) -> dict[str, Any]:
        def dist(dim: str) -> list[dict[str, Any]]:
            counts = Counter(p[dim] for p in self.p if p[dim] not in _FILLER)
            return [{"value": v, "label": nice(v), "count": c, "pct": _pct(c, self.total)}
                    for v, c in counts.most_common()]

        def combos(a: str, b: str, k: int = 5) -> list[dict[str, Any]]:
            counts = Counter((p[a], p[b]) for p in self.p
                             if p[a] not in _FILLER and p[b] not in _FILLER)
            return [{"combo": f"{nice(x)} + {nice(y)}", "count": c, "pct": _pct(c, self.total)}
                    for (x, y), c in counts.most_common(k)]

        return {
            "hooks": dist("hook"), "formats": dist("format"),
            "emotions": dist("emotion"), "ctas": dist("cta"),
            "dominant_combinations": {
                "hook_x_format": combos("hook", "format"),
                "format_x_emotion": combos("format", "emotion"),
                "hook_x_cta": combos("hook", "cta"),
            },
        }

    # ---- executive summary ---------------------------------------------- #
    def executive_summary(self, winners, gaps, saturation, dna) -> dict[str, Any]:
        top_winner = winners[0] if winners else None
        top_gap = gaps[0] if gaps else None
        top_sat = saturation[0] if saturation else None
        pool = [("winner", w) for w in winners] + [("gap", g) for g in gaps]
        best = max(pool, key=lambda t: t[1]["confidence"], default=None)

        def q(answer, confidence, evidence):
            return {"answer": answer, "confidence": self._cap(confidence), "evidence": evidence}

        strongest_answer = (top_winner["pattern"] + f" - {top_winner['signal']}"
                            if top_winner else f"{dna['hook']['label']} hooks in {dna['format']['label']}")

        return {
            "headline": dna["formula"],
            "disclaimer": DISCLAIMER,
            "what_is_winning": q(   # key kept for UI; wording is now proxy-honest
                strongest_answer,
                top_winner["confidence"] if top_winner else dna["confidence"],
                top_winner["evidence"][:2] if top_winner else [dna["basis"]],
            ),
            "why_is_it_winning": q(
                top_winner["reasoning"] if top_winner else
                "It is the most consistent, highest-scoring creative proxy in the set, but "
                "performance cannot be confirmed without engagement or conversion metrics.",
                top_winner["confidence"] if top_winner else dna["confidence"],
                [top_winner["evidence"][1]] if top_winner else [dna["formula"]],
            ),
            "what_is_saturated": q(
                f"{top_sat['pattern']} ({top_sat['saturation']}% of ads — {top_sat['saturation_label']}, {top_sat['risk']} risk)"
                + (" · directional, small sample" if top_sat and top_sat.get("directional") else "")
                if top_sat else "No single pattern is saturated.",
                top_sat["saturation"] if top_sat else 40,
                [top_sat["recommendation"]] if top_sat else [],
            ),
            "what_is_underused": q(
                f"{top_gap['label']} ({top_gap['type']}) — {top_gap['current_usage']}" if top_gap
                else "No clear whitespace detected.",
                top_gap["confidence"] if top_gap else 50,
                [top_gap["reason"], top_gap["why_it_matters"]] if top_gap else [],
            ),
            "what_to_test_next": q(
                (f"Run the dominant pattern against a contrarian {top_gap['label']} variant."
                 if top_gap else "A/B the dominant hook against a fresh emotional angle."),
                top_gap["confidence"] if top_gap else 55,
                ([top_gap["test_idea"]] if top_gap else []) + ([top_winner["evidence"][1]] if top_winner else []),
            ),
            "highest_confidence_opportunity": q(
                (f"{best[1]['pattern']} (strongest observed proxy pattern)" if best and best[0] == "winner"
                 else f"{best[1]['label']} (open {best[1]['type']} whitespace)" if best else "Insufficient data"),
                best[1]["confidence"] if best else 0,
                (best[1]["evidence"][:2] if best and best[0] == "winner"
                 else [best[1]["reason"]] if best else []),
            ),
        }


def build_profiles(per_ad: list[tuple]) -> list[dict[str, Any]]:
    """Adapt orchestrator per-ad tuples into intelligence profiles.

    per_ad item = (ad, text_output, visual_output|None, features)
    """
    profiles: list[dict[str, Any]] = []
    for ad, text_o, _visual_o, f in per_ad:
        fmt = f.creative_format or "unknown"
        # Founder is a HOOK angle; the visual counterpart is a talking-head FORMAT.
        # Never let a founder value land in both hook and format. Normalise first so
        # case/spacing/spelling variants ("founder", "Founder_Story", "founder-story",
        # "founder talking head") all canonicalise rather than re-duplicating.
        fmt_norm = fmt.strip().lower().replace(" ", "_").replace("-", "_")
        if "founder" in fmt_norm:
            fmt = "founder_talking_head"
        has_perf = bool(getattr(ad, "impressions", 0) or getattr(ad, "likes", 0)
                        or getattr(ad, "shares", 0) or getattr(ad, "comments", 0))
        profiles.append({
            "id": ad.id, "platform": (ad.platform or "unknown"),
            "score": float(ad.winner_score or 0.0),
            "hook": f.hook_type or "unknown", "cta": f.cta_type or "none",
            "emotion": f.emotion_type or "neutral", "format": fmt,
            "ugc": bool(f.ugc_style), "human": bool(f.human_present),
            "app_screen": bool(f.app_screen_visible),
            "product_demo": bool(f.product_demo_present), "face": bool(f.face_visible),
            "hook_strength": float(f.hook_strength or 0.0),
            "copy_structure": getattr(text_o, "copywriting_structure", "") or "unknown",
            "pain_points": [str(x).lower() for x in getattr(text_o, "pain_points", []) or []],
            "ad_text": ad.ad_text or "", "has_perf": has_perf,
        })
    return profiles
