"""ReportConsistencyValidator (BUG 3, 10).

The final gate. Runs after the draft is assembled and before rendering. It treats
the deterministic aggregates / confidence / category resolution as ground truth
and rewrites or strips any draft claim that contradicts them. Every correction is
recorded so the pipeline (and tests) can prove the gate fired.

It is defensive on purpose: even if an upstream LLM ignores its instructions and
writes "Download CTA" or "proven winner", this layer removes it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WINNER_WORDS = re.compile(r"\b(winner|winning|wins)\b", re.IGNORECASE)
_PROVEN_WINNER = re.compile(r"\bproven winners?\b", re.IGNORECASE)
_PROVEN_OR_FORMULA = re.compile(r"\b(proven|market formula)\b", re.IGNORECASE)
_DOWNLOAD_CTA = re.compile(r"\bdownload cta\b", re.IGNORECASE)

ALLOWED_STRATEGY_LINKS = {
    "dominant_pattern", "emerging_signal", "whitespace_opportunity",
    "saturation_risk", "low_support_high_lift",
}


@dataclass
class ValidatorContext:
    detected_product_type: str
    resolved_product_type: str
    selected_was_generic: bool
    download_frequency: float       # 0..100
    cta_reliable: bool
    hook_unknown_rate: float        # 0..1
    has_performance_data: bool
    any_positive_lift: bool
    rejected_ad_ids: set = field(default_factory=set)
    included_ads_count: int = 0
    category_integrity_score: int = 100
    mode: str = "normal"            # "normal" | "insufficient"
    resolved_title: str = ""
    forbidden_tokens: list = field(default_factory=list)  # off-category lexicon to scan for


@dataclass
class ValidationReport:
    report: dict
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


class ReportConsistencyValidator:
    def __init__(self, ctx: ValidatorContext):
        self.ctx = ctx
        self.v: list[str] = []

    def validate(self, report: dict) -> ValidationReport:
        self.v = []
        self._check_cta_zero(report)                 # rules 2, 3 (BUG 3)
        self._check_winner_wording(report)           # rules 3, 4 (BUG 5)
        self._check_hook_unreliable(report)          # rule 5 (BUG 4)
        self._check_low_support_claims(report)       # rule 1 (BUG 2/6)
        self._check_category_title(report)           # rule 6 (BUG 1)
        self._check_absent_opportunities(report)     # rule 7 (BUG 7)
        self._check_insight_evidence(report)         # rule 8
        self._check_strategy_links(report)           # rule 9
        self._check_proven_formula(report)           # TASK 5: ban proven/market formula
        self._check_evidence_inclusion(report)       # TASK 9.8: examples from included only
        self._check_title_matches_validated(report)  # TASK 9.9
        self._check_confidence_caps(report)          # TASK 9.7
        self._check_insufficient_block(report)       # TASK 9.10
        self._check_text_contamination(report)       # defense-in-depth backstop
        return ValidationReport(report=report, violations=list(self.v))

    # -- rule 2 & 3: no CTA claim when frequency is 0 ----------------------- #
    def _check_cta_zero(self, report: dict) -> None:
        if self.ctx.download_frequency > 0 and self.ctx.cta_reliable:
            return
        # scrub prose
        for key in ("executive_summary", "strategies_text"):
            if self._scrub_lines(report.get(key, []), _DOWNLOAD_CTA):
                self.v.append(f"Removed 'Download CTA' from {key}: download frequency is 0%.")
        # drop CTA patterns / creative-DNA CTA claims
        before = len(report.get("patterns", []))
        report["patterns"] = [p for p in report.get("patterns", [])
                              if not (p.get("dimension") == "cta")]
        if len(report.get("patterns", [])) != before:
            self.v.append("Dropped CTA pattern(s): no reliable CTA detected.")
        cta_section = report.setdefault("cta_section", {})
        cta_section["text"] = "No reliable CTA pattern detected."
        dna = report.get("creative_dna", {})
        if isinstance(dna, dict) and "cta" in dna and not self.ctx.cta_reliable:
            dna["cta"] = {"reliable": False, "note": "No reliable CTA pattern detected."}

    # -- rule 3 & 4: winner wording ---------------------------------------- #
    def _check_winner_wording(self, report: dict) -> None:
        # 'proven winner' anywhere requires direct performance data
        if not self.ctx.has_performance_data:
            for key in ("executive_summary", "strategies_text"):
                if self._scrub_lines(report.get(key, []), _PROVEN_WINNER,
                                     replacement="dominant pattern"):
                    self.v.append(f"Replaced 'proven winner' in {key}: no direct performance data.")
            for p in report.get("patterns", []):
                if p.get("claim_class") == "proven_winner":
                    p["claim_class"] = "dominant"
                    p["verb"] = "dominant pattern"
                    p["label"] = "dominant_pattern"
                    self.v.append(f"Downgraded pattern '{p.get('name')}' from proven_winner: no performance data.")
        # 'winner/winning/wins' requires positive lift OR performance data
        if not (self.ctx.any_positive_lift or self.ctx.has_performance_data):
            for key in ("executive_summary", "strategies_text"):
                if self._scrub_lines(report.get(key, []), _WINNER_WORDS,
                                     replacement="dominant"):
                    self.v.append(f"Replaced winner-wording in {key}: no positive lift / performance data.")
        for p in report.get("patterns", []):
            lift = p.get("performance_lift", 0) or 0
            text = p.get("text", "")
            if _WINNER_WORDS.search(text) and lift <= 0 and not self.ctx.has_performance_data:
                p["text"] = _WINNER_WORDS.sub("dominant", text)
                self.v.append(f"Rewrote winner-wording on pattern '{p.get('name')}': lift={lift}.")

    # -- rule 5: hook conclusions when under-classified -------------------- #
    def _check_hook_unreliable(self, report: dict) -> None:
        if self.ctx.hook_unknown_rate <= 0.50:
            return
        note = "Hook classification quality is low. Hook-level conclusions are unreliable."
        report.setdefault("hook_section", {})["note"] = note
        # demote hook patterns
        for p in report.get("patterns", []):
            if p.get("dimension") == "hook" and p.get("claim_class") in ("dominant", "saturated", "emerging"):
                p["claim_class"] = "low_support"
                p["label"] = "low_support_observation"
                p["verb"] = "low-support observation"
                self.v.append(f"Demoted hook pattern '{p.get('name')}': hook unknown rate "
                              f"{self.ctx.hook_unknown_rate:.0%}.")

    # -- rule 1: no exec-summary claim with support < 3 -------------------- #
    def _check_low_support_claims(self, report: dict) -> None:
        for p in report.get("patterns", []):
            if (p.get("support_count", 0) < 3
                    and p.get("claim_class") in ("dominant", "saturated", "proven_winner")):
                p["claim_class"] = "low_support"
                p["label"] = "low_support_observation"
                p["verb"] = "low-support observation"
                self.v.append(f"Downgraded pattern '{p.get('name')}' to low-support "
                              f"(support_count={p.get('support_count')}).")

    # -- rule 6: title must not conflict with detected product ------------- #
    def _check_category_title(self, report: dict) -> None:
        cat = report.get("category", {})
        title = (cat.get("title") or "").lower()
        if (self.ctx.detected_product_type != self.ctx.resolved_product_type
                and self.ctx.detected_product_type != "uncertain"):
            self.v.append("Category title conflicts with detected product type.")
        # generic-title guard
        if self.ctx.selected_was_generic and self.ctx.resolved_product_type != "uncertain":
            if self.ctx.resolved_product_type not in (cat.get("product_type") or ""):
                self.v.append("Category title left generic despite a specific detected product.")

    # -- rule 7: absent / whitespace opportunities ------------------------- #
    def _check_absent_opportunities(self, report: dict) -> None:
        for o in report.get("opportunities", []):
            usage = o.get("usage_frequency", 0) or 0
            if usage <= 0:
                if not o.get("has_external_benchmark") and o.get("confidence", 0) > 65:
                    o["confidence"] = 65
                    self.v.append(f"Capped absent opportunity '{o.get('name')}' confidence at 65.")
                o["label"] = "whitespace_untested"
                o.setdefault("note", "Absent from the analysed set — differentiation "
                                     "hypothesis, unproven in this sample.")

    # -- rule 8: every major insight has evidence -------------------------- #
    def _check_insight_evidence(self, report: dict) -> None:
        for ins in report.get("insights", []):
            if not ins.get("evidence_rows"):
                ins["needs_evidence"] = True
                self.v.append(f"Insight '{ins.get('title')}' has no evidence rows; flagged.")
        for p in report.get("patterns", []):
            if p.get("claim_class") in ("dominant", "saturated", "emerging") and not p.get("evidence_ad_ids"):
                p["needs_evidence"] = True
                self.v.append(f"Pattern '{p.get('name')}' presented as {p.get('claim_class')} "
                              f"without evidence ad ids.")

    # -- rule 9: every strategy linked ------------------------------------- #
    def _check_strategy_links(self, report: dict) -> None:
        for s in report.get("strategies", []):
            link = s.get("linked_to")
            if link not in ALLOWED_STRATEGY_LINKS:
                s["needs_link"] = True
                self.v.append(f"Strategy '{s.get('title', s.get('text','?'))[:40]}' "
                              f"not linked to a valid evidence type.")

    # -- TASK 5: ban 'proven' / 'market formula' without performance data --- #
    def _check_proven_formula(self, report: dict) -> None:
        if self.ctx.has_performance_data:
            return
        for key in ("executive_summary", "strategies_text"):
            if self._scrub_lines(report.get(key, []), _PROVEN_OR_FORMULA, replacement="dominant"):
                self.v.append(f"Removed 'proven'/'market formula' from {key}: no performance data.")
        for p in report.get("patterns", []):
            if _PROVEN_OR_FORMULA.search(p.get("text", "")):
                p["text"] = _PROVEN_OR_FORMULA.sub("dominant", p["text"])
                self.v.append(f"Rewrote proven/formula wording on pattern '{p.get('name')}'.")

    # -- TASK 9.8: evidence examples must belong to INCLUDED ads ------------ #
    def _check_evidence_inclusion(self, report: dict) -> None:
        rid = self.ctx.rejected_ad_ids
        if not rid:
            return
        for p in report.get("patterns", []):
            before = list(p.get("evidence_ad_ids", []))
            p["evidence_ad_ids"] = [i for i in before if i not in rid]
            if len(p["evidence_ad_ids"]) != len(before):
                self.v.append(f"Stripped rejected-ad evidence from pattern '{p.get('name')}'.")
        for ins in report.get("insights", []):
            rows = ins.get("evidence_rows", [])
            kept = [r for r in rows if r.get("ad_id") not in rid]
            if len(kept) != len(rows):
                ins["evidence_rows"] = kept
                self.v.append(f"Stripped rejected-ad evidence rows from insight '{ins.get('title','')[:30]}'.")

    # -- TASK 9.9: title must match dominant validated category ------------- #
    def _check_title_matches_validated(self, report: dict) -> None:
        cat = report.get("category", {})
        title = cat.get("title", "")
        if self.ctx.resolved_title and title != self.ctx.resolved_title:
            cat["title"] = self.ctx.resolved_title
            self.v.append("Report title did not match the validated category; corrected.")

    # -- TASK 9.7: confidence may not exceed allowed caps ------------------- #
    def _check_confidence_caps(self, report: dict) -> None:
        conf = report.get("confidence", {})
        final = conf.get("final")
        if final is None:
            return
        caps: list[int] = []
        if not self.ctx.has_performance_data:
            caps.append(75)
        if self.ctx.category_integrity_score < 70:
            caps.append(55)
        if self.ctx.included_ads_count < 10:
            caps.append(50)
        if caps and final > min(caps):
            conf["final"] = float(min(caps))
            self.v.append(f"Clamped confidence from {final} to {min(caps)} (cap rule).")

    # -- TASK 9.10: block normal sections when too few validated ads -------- #
    def _check_insufficient_block(self, report: dict) -> None:
        if self.ctx.mode != "insufficient":
            return
        dna = report.get("creative_dna")
        if isinstance(dna, dict) and dna.get("status") != "Insufficient validated ads":
            report["creative_dna"] = {"status": "Insufficient validated ads"}
            self.v.append("Blocked Creative DNA: insufficient validated ads.")
        for section in ("patterns", "opportunities", "strategies", "briefs", "insights"):
            if report.get(section):
                report[section] = []
                self.v.append(f"Blocked '{section}' section: insufficient validated ads.")

    # -- defense-in-depth: scan assembled conclusions for off-category tokens #
    def _check_text_contamination(self, report: dict) -> None:
        toks = [t.lower() for t in (self.ctx.forbidden_tokens or [])]
        if not toks:
            return

        def scan(s: str) -> list[str]:
            low = (s or "").lower()
            return [t for t in toks if t in low]

        # exec summary lines
        for i, line in enumerate(list(report.get("executive_summary", []))):
            if scan(line):
                report["executive_summary"][i] = ""
                self.v.append("Removed off-category text from executive summary (contamination backstop).")
        # insight evidence rows
        for ins in report.get("insights", []):
            kept = []
            for row in ins.get("evidence_rows", []):
                blob = " ".join(str(row.get(k, "")) for k in ("hook_text", "cta", "advertiser"))
                if scan(blob):
                    self.v.append(f"Dropped off-category evidence row (ad {row.get('ad_id')}) from insight.")
                else:
                    kept.append(row)
            ins["evidence_rows"] = kept
        # patterns / briefs text
        for p in report.get("patterns", []):
            if scan(p.get("text", "")):
                p["needs_evidence"] = True
                self.v.append(f"Flagged pattern '{p.get('name')}' carrying off-category text.")
        report["executive_summary"] = [l for l in report.get("executive_summary", []) if l.strip()]

    # -- helpers ----------------------------------------------------------- #
    def _scrub_lines(self, lines: list[str], pattern: re.Pattern,
                     replacement: str | None = None) -> bool:
        changed = False
        for i, line in enumerate(list(lines)):
            if pattern.search(line):
                changed = True
                if replacement is None:
                    lines[i] = ""      # blanked; pipeline filters empties
                else:
                    lines[i] = pattern.sub(replacement, line)
        if changed and replacement is None:
            lines[:] = [l for l in lines if l.strip()]
        return changed


def validate_report(report: dict, ctx: ValidatorContext) -> ValidationReport:
    return ReportConsistencyValidator(ctx).validate(report)
