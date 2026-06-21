"""Agent Orchestration Layer.

Flow (per spec):
  Load Ads → Text → Video/Image → Feature Extraction → Pattern Mining →
  Scoring → Voting (wraps Text/Visual/Strategy) → Strategy → Save Results.

The orchestrator persists the output of every stage so the dashboard can show
agent outputs, features, patterns, scores and the final strategy, and so runs
are fully reproducible from the database.

Chain-of-thought is never exposed: only Evidence Summary, Reasoning Summary and
Confidence Score are surfaced.
"""
from __future__ import annotations

from collections import Counter

from sqlalchemy.orm import Session

from agents.feature_agent import FeatureExtractionAgent
from agents.image_agent import ImageAnalysisAgent
from agents.intelligence import IntelligenceEngine, build_profiles
from agents.pattern_agent import PatternMiningAgent
from agents.scoring_agent import ScoringAgent
from agents.strategy_agent import StrategyGenerationAgent
from agents.text_agent import TextAnalysisAgent
from agents.video_agent import VideoAnalysisAgent
from agents.voting_agent import VotingAgent
from config import get_settings
from core.logging_config import get_logger
from database.models import (
    AgentAnalysisResult,
    Ad,
    AnalysisRun,
    ExtractedFeature,
    FinalStrategy,
    Status,
    WinningPattern,
)
from schemas.models import FinalOutput
from services.llm_service import get_llm_service

logger = get_logger("orchestrator")


class Orchestrator:
    def __init__(self, db: Session, llm):
        self.db = db
        self.settings = get_settings()
        self.text_agent = TextAnalysisAgent(llm)
        self.video_agent = VideoAnalysisAgent(llm)
        self.image_agent = ImageAnalysisAgent(llm)
        self.feature_agent = FeatureExtractionAgent()
        self.pattern_agent = PatternMiningAgent()
        self.scoring_agent = ScoringAgent()
        self.strategy_agent = StrategyGenerationAgent(llm)
        self.voting = VotingAgent(samples=self.settings.consensus_samples)

    # ------------------------------------------------------------------ #
    def execute(self, run: AnalysisRun, ads: list[Ad]) -> FinalOutput:
        logger.info("Run %s: analysing %s ads", run.id, len(ads))
        per_ad = []                       # bookkeeping for scoring + strategy
        feature_items = []                # (ExtractedFeatures, platform)
        agent_confidences = []
        agreements = []
        any_llm = False

        # --- Stage 1-4: per-ad text + visual + feature extraction ----------
        for ad in ads:
            ctx = _ad_ctx(ad)

            text_c = self.voting.consensus(self.text_agent, ctx)
            self._save_agent(run.id, ad.id, "text_agent", text_c)
            agent_confidences.append(text_c.confidence)
            agreements.append(text_c.agreement)
            any_llm |= text_c.source == "llm"

            visual_c = None
            if (ad.creative_type or "").lower() == "video":
                visual_c = self.voting.consensus(self.video_agent, ctx)
                self._save_agent(run.id, ad.id, "video_agent", visual_c)
            elif (ad.creative_type or "").lower() == "image":
                visual_c = self.voting.consensus(self.image_agent, ctx)
                self._save_agent(run.id, ad.id, "image_agent", visual_c)
            if visual_c:
                agent_confidences.append(visual_c.confidence)
                agreements.append(visual_c.agreement)
                any_llm |= visual_c.source == "llm"

            features = self.feature_agent.extract(
                text_c.output, visual_c.output if visual_c else None,
                ad.creative_type, ad.duration,
            )
            self._save_feature(run.id, ad, features)
            feature_items.append((features, ad.platform))
            per_ad.append((ad, text_c.output, visual_c.output if visual_c else None, features))

        # --- Stage 5: pattern mining ---------------------------------------
        patterns = self.pattern_agent.mine(feature_items)
        self._save_patterns(run.id, patterns)

        # --- Stage 6: scoring (with voting bookkeeping) --------------------
        for ad, text_o, visual_o, feats in per_ad:
            score = self.scoring_agent.score(text_o, visual_o, feats, ad.platform)
            ad.winner_score = score.total
            self.db.add(
                AgentAnalysisResult(
                    analysis_run_id=run.id, ad_id=ad.id, agent_name="scoring_agent",
                    output=score.model_dump(), confidence=score.confidence,
                    consensus_agreement=1.0,  # deterministic scoring => full agreement
                    status=Status.COMPLETED.value, source="heuristic",
                )
            )
        self.db.flush()

        # --- Market Intelligence layer (deterministic, evidence-driven) ----
        # Runs over the now-scored per-ad features: clusters, winner patterns,
        # opportunity gaps, creative DNA, saturation, strategy triad, market
        # map and an executive summary. No extra LLM calls.
        intelligence = IntelligenceEngine(
            build_profiles(per_ad), run.platforms or []
        ).build()

        # --- aggregates for the strategy stage -----------------------------
        agg = _aggregate(per_ad, patterns)
        mean_conf = _mean(agent_confidences, 0.5)
        mean_agree = _mean(agreements, 1.0)
        sample_factor = min(1.0, len(ads) / 20.0)
        confidence = round(
            0.5 * mean_conf + 0.3 * mean_agree + 0.2 * sample_factor, 3
        )
        if not any_llm:
            confidence = round(confidence * 0.9, 3)  # discount pure-heuristic runs

        limitations = _limitations(len(ads), any_llm, ads)

        # --- Stage 7+8: strategy via consensus -----------------------------
        strat_ctx = {
            **agg,
            "app_name": run.app_name,
            "platforms": run.platforms or [],
            "analyzed_ads_count": len(ads),
            "pattern_statements": [p.statement for p in patterns[:10]],
            "evidence_confidence": confidence,
            "limitations": limitations,
        }
        strat_c = self.voting.consensus(self.strategy_agent, strat_ctx)
        strategy = strat_c.output
        # keep the data-grounded confidence/limitations authoritative
        strategy.confidence_score = confidence
        if not strategy.limitations:
            strategy.limitations = limitations

        self.db.add(
            FinalStrategy(
                analysis_run_id=run.id,
                confidence_score=confidence,
                strategy=strategy.model_dump(),
                intelligence=intelligence,
            )
        )

        # --- summaries (no chain-of-thought) -------------------------------
        evidence_summary = " ".join(p.statement for p in patterns[:5])
        reasoning_summary = (
            f"Mined {len(ads)} ads for {run.app_name}. Dominant hook: "
            f"{agg['dominant_hook']}; dominant emotion: {agg['dominant_emotion']}; "
            f"dominant format: {agg['dominant_format']}. Scored each ad across 7 "
            f"explainable criteria; mean consensus agreement {mean_agree:.0%}. "
            f"Strategy assembled strictly from the patterns above."
        )

        run.analyzed_ads_count = len(ads)
        run.confidence_score = confidence
        run.evidence_summary = evidence_summary
        run.reasoning_summary = reasoning_summary
        run.limitations = limitations
        self.db.flush()

        return FinalOutput(
            app_name=run.app_name,
            platforms=run.platforms or [],
            analyzed_ads_count=len(ads),
            winning_patterns=[p.model_dump() for p in patterns],
            confidence_score=confidence,
            evidence_summary=evidence_summary,
            reasoning_summary=reasoning_summary,
            limitations=limitations,
            strategy=strategy.model_dump(),
            intelligence=intelligence,
        )

    # ------------------------------------------------------------------ #
    def _save_agent(self, run_id, ad_id, name, consensus) -> None:
        self.db.add(
            AgentAnalysisResult(
                analysis_run_id=run_id, ad_id=ad_id, agent_name=name,
                output=consensus.output.model_dump(),
                confidence=consensus.confidence,
                consensus_agreement=consensus.agreement,
                status=Status.COMPLETED.value, source=consensus.source,
            )
        )

    def _save_feature(self, run_id, ad: Ad, feats) -> None:
        self.db.add(
            ExtractedFeature(
                analysis_run_id=run_id, ad_id=ad.id, platform=ad.platform,
                hook_type=feats.hook_type, hook_strength=feats.hook_strength,
                cta_type=feats.cta_type, emotion_type=feats.emotion_type,
                creative_format=feats.creative_format, human_present=feats.human_present,
                app_screen_visible=feats.app_screen_visible, ugc_style=feats.ugc_style,
                video_length=feats.video_length, visual_density=feats.visual_density,
                face_visible=feats.face_visible, product_demo_present=feats.product_demo_present,
                raw=feats.model_dump(),
            )
        )

    def _save_patterns(self, run_id, patterns) -> None:
        for p in patterns:
            self.db.add(
                WinningPattern(
                    analysis_run_id=run_id, pattern_type=p.pattern_type,
                    pattern_value=p.pattern_value, platform=p.platform,
                    frequency=p.frequency, sample_size=p.sample_size,
                    percentage=p.percentage, statement=p.statement, evidence=p.evidence,
                )
            )
        self.db.flush()


# --------------------------------------------------------------------------- #
def _ad_ctx(ad: Ad) -> dict:
    return {
        "app_name": ad.app_name,
        "platform": ad.platform,
        "creative_type": ad.creative_type,
        "ad_text": ad.ad_text,
        "image_or_video_url": ad.image_or_video_url,
        "duration": ad.duration,
    }


def _aggregate(per_ad, patterns) -> dict:
    hooks = Counter(f.hook_type for _, _, _, f in per_ad if f.hook_type != "unknown")
    emotions = Counter(f.emotion_type for _, _, _, f in per_ad if f.emotion_type != "neutral")
    formats = Counter(f.creative_format for _, _, _, f in per_ad if f.creative_format != "unknown")
    pains: Counter = Counter()
    for _, text_o, _, _ in per_ad:
        pains.update(p.lower() for p in text_o.pain_points)

    top_ads = sorted(per_ad, key=lambda x: x[0].winner_score or 0, reverse=True)[:6]
    return {
        "dominant_hook": _top(hooks, "problem_first"),
        "dominant_emotion": _top(emotions, "relief"),
        "dominant_format": _top(formats, "ugc_video"),
        "top_pain_points": [p for p, _ in pains.most_common(5)],
        "top_ads": [
            {"ad_text": a.ad_text or "", "platform": a.platform, "score": a.winner_score or 0}
            for a, *_ in top_ads
        ],
    }


def _top(counter: Counter, default: str) -> str:
    return counter.most_common(1)[0][0] if counter else default


def _mean(values, default: float) -> float:
    return sum(values) / len(values) if values else default


def _limitations(n: int, any_llm: bool, ads: list[Ad]) -> list[str]:
    out: list[str] = []
    if n < 10:
        out.append(f"Small sample size (N={n}); percentages are directional, not statistically conclusive.")
    if not any_llm:
        out.append("Visual analysis ran in heuristic mode (no vision model configured); "
                   "face/app-screen timing signals are inferred from platform and copy, not pixels.")
    if not any((a.impressions or a.likes or a.shares or a.comments) for a in ads):
        out.append("Source data has no engagement metrics; winner scores reflect creative-quality "
                   "criteria, not measured performance.")
    if not out:
        out.append("Findings reflect the analysed ad set only and may not generalise to all audiences.")
    return out
