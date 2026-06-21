"""Analysis service — owns the run lifecycle and transaction boundaries.

Responsibilities:
  * validate the request (app/platform must exist — never analyse non-existent ads),
  * create + transition the AnalysisRun (pending → running → completed/failed),
  * invoke the Orchestrator,
  * persist the canonical FinalOutput to DB and outputs/final_strategy.json,
  * rebuild a FinalOutput from stored rows for historical runs.
"""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from agents.orchestrator import Orchestrator
from config import get_settings
from core.logging_config import get_logger
from database.repositories import (
    AdRepository,
    AnalysisRunRepository,
    AgentResultRepository,
    FeatureRepository,
    PatternRepository,
    StrategyRepository,
)
from schemas.models import FinalOutput
from services.llm_service import get_llm_service
from services.storage_service import sweep_orphans

logger = get_logger("analysis")


class AnalysisError(Exception):
    pass


class AnalysisService:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.ads = AdRepository(db)
        self.runs = AnalysisRunRepository(db)
        self.patterns = PatternRepository(db)
        self.strategies = StrategyRepository(db)
        self.agent_results = AgentResultRepository(db)
        self.features = FeatureRepository(db)

    # ------------------------------------------------------------------ #
    def run_analysis(self, app_name: str, platforms: list[str]) -> FinalOutput:
        ads = self.ads.for_analysis(
            app_name, platforms or None, limit=self.settings.analyze_max_ads
        )
        if not ads:
            raise AnalysisError(
                f"No ads found for app '{app_name}'"
                + (f" on {platforms}" if platforms else "")
                + ". Nothing to analyse."
            )

        run = self.runs.create(app_name, platforms)
        self.runs.mark_running(run)
        self.db.commit()
        logger.info("Run %s started (app=%s, platforms=%s, ads=%s)",
                    run.id, app_name, platforms or "all", len(ads))

        try:
            orchestrator = Orchestrator(self.db, get_llm_service())
            result = orchestrator.execute(run, ads)
            result.run_id = run.id
            self.runs.mark_completed(run)
            self.db.commit()
            self._write_output_file(run.id, result)
            logger.info("Run %s completed (confidence=%s)", run.id, result.confidence_score)
            return result
        except Exception as exc:
            self.db.rollback()
            # Re-load run after rollback and record the failure in its own tx.
            run = self.runs.get(run.id)
            if run:
                self.runs.mark_failed(run, str(exc))
                self.db.commit()
            logger.exception("Run %s failed", run.id if run else "?")
            raise AnalysisError(str(exc)) from exc
        finally:
            sweep_orphans()  # belt-and-braces temp cleanup

    # ------------------------------------------------------------------ #
    def get_run_output(self, run_id: int) -> FinalOutput:
        """Rebuild the canonical output for a historical run from stored rows."""
        run = self.runs.get(run_id)
        if not run:
            raise AnalysisError(f"Run {run_id} not found")

        patterns = self.patterns.for_run(run_id)
        strat = self.strategies.for_run(run_id)
        return FinalOutput(
            run_id=run.id,
            app_name=run.app_name,
            platforms=run.platforms or [],
            analyzed_ads_count=run.analyzed_ads_count,
            winning_patterns=[
                {
                    "pattern_type": p.pattern_type,
                    "pattern_value": p.pattern_value,
                    "platform": p.platform,
                    "frequency": p.frequency,
                    "sample_size": p.sample_size,
                    "percentage": p.percentage,
                    "statement": p.statement,
                    "evidence": p.evidence,
                }
                for p in patterns
            ],
            confidence_score=run.confidence_score or 0.0,
            evidence_summary=run.evidence_summary or "",
            reasoning_summary=run.reasoning_summary or "",
            limitations=run.limitations or [],
            strategy=(strat.strategy if strat else {}),
            intelligence=(strat.intelligence if strat and strat.intelligence else {}),
        )

    def get_agent_outputs(self, run_id: int) -> list[dict]:
        out = []
        for r in self.agent_results.for_run(run_id):
            out.append({
                "id": r.id,
                "ad_id": r.ad_id,
                "agent_name": r.agent_name,
                "confidence": r.confidence,
                "consensus_agreement": r.consensus_agreement,
                "source": r.source,
                "output": r.output,
            })
        return out

    # ------------------------------------------------------------------ #
    def _write_output_file(self, run_id: int, result: FinalOutput) -> None:
        path = self.settings.outputs_dir / "final_strategy.json"
        payload = result.model_dump()
        payload["run_id"] = run_id
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        # Also keep a per-run snapshot.
        (self.settings.outputs_dir / f"run_{run_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
