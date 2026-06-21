"""Repository Pattern.

Each repository encapsulates all persistence logic for one aggregate and hides
SQLAlchemy from the service layer. Services depend on repositories, never on the
ORM session directly — which keeps business logic database-agnostic and easy to
unit-test or re-point at PostgreSQL/MySQL/Oracle.
"""
from __future__ import annotations

from datetime import datetime
from typing import Generic, Iterable, Sequence, Type, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import (
    AgentAnalysisResult,
    Ad,
    AdCreative,
    AnalysisRun,
    App,
    ExtractedFeature,
    FinalStrategy,
    Status,
    WinningPattern,
)

M = TypeVar("M")


class BaseRepository(Generic[M]):
    model: Type[M]

    def __init__(self, db: Session):
        self.db = db

    def get(self, pk: int) -> M | None:
        return self.db.get(self.model, pk)

    def list(self, limit: int | None = None, offset: int = 0) -> Sequence[M]:
        stmt = select(self.model).offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.db.execute(stmt).scalars().all()

    def add(self, obj: M) -> M:
        self.db.add(obj)
        self.db.flush()
        return obj

    def add_all(self, objs: Iterable[M]) -> None:
        self.db.add_all(list(objs))
        self.db.flush()

    def commit(self) -> None:
        self.db.commit()


class AppRepository(BaseRepository[App]):
    model = App

    def get_by_name(self, app_name: str) -> App | None:
        return self.db.execute(
            select(App).where(App.app_name == app_name)
        ).scalar_one_or_none()

    def upsert(self, app_name: str, **fields) -> App:
        app = self.get_by_name(app_name)
        if app is None:
            app = App(app_name=app_name, **fields)
            self.db.add(app)
            self.db.flush()
        else:
            for k, v in fields.items():
                if v is not None:
                    setattr(app, k, v)
        return app

    def list_with_counts(self) -> list[dict]:
        rows = self.db.execute(
            select(Ad.app_name, Ad.platform, func.count(Ad.id))
            .group_by(Ad.app_name, Ad.platform)
        ).all()
        agg: dict[str, dict] = {}
        for app_name, platform, count in rows:
            entry = agg.setdefault(
                app_name, {"app_name": app_name, "ad_count": 0, "platforms": {}}
            )
            entry["ad_count"] += count
            if platform:
                entry["platforms"][platform] = count
        return sorted(agg.values(), key=lambda e: -e["ad_count"])


class AdRepository(BaseRepository[Ad]):
    model = Ad

    def query(
        self,
        app_name: str | None = None,
        platform: str | None = None,
        creative_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Ad]:
        stmt = select(Ad)
        if app_name:
            stmt = stmt.where(Ad.app_name == app_name)
        if platform:
            stmt = stmt.where(Ad.platform == platform)
        if creative_type:
            stmt = stmt.where(Ad.creative_type == creative_type)
        stmt = stmt.order_by(Ad.winner_score.desc().nullslast(), Ad.id).offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.db.execute(stmt).scalars().all()

    def for_analysis(
        self,
        app_name: str,
        platforms: Sequence[str] | None,
        limit: int | None = None,
    ) -> list[Ad]:
        stmt = select(Ad).where(Ad.app_name == app_name)
        if platforms:
            stmt = stmt.where(Ad.platform.in_(list(platforms)))
        # Deterministic order so a capped run always picks the same representative set.
        stmt = stmt.order_by(Ad.id)
        if limit and limit > 0:
            stmt = stmt.limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def count(self, app_name: str | None = None, platform: str | None = None) -> int:
        stmt = select(func.count(Ad.id))
        if app_name:
            stmt = stmt.where(Ad.app_name == app_name)
        if platform:
            stmt = stmt.where(Ad.platform == platform)
        return int(self.db.execute(stmt).scalar_one())

    def set_winner_score(self, ad_id: int, score: float) -> None:
        ad = self.db.get(Ad, ad_id)
        if ad:
            ad.winner_score = score


class AdCreativeRepository(BaseRepository[AdCreative]):
    model = AdCreative


class AnalysisRunRepository(BaseRepository[AnalysisRun]):
    model = AnalysisRun

    def create(self, app_name: str, platforms: Sequence[str]) -> AnalysisRun:
        run = AnalysisRun(
            app_name=app_name,
            platforms=list(platforms),
            status=Status.PENDING.value,
        )
        self.db.add(run)
        self.db.flush()
        return run

    def mark_running(self, run: AnalysisRun) -> None:
        run.status = Status.RUNNING.value
        run.started_at = datetime.utcnow()

    def mark_completed(self, run: AnalysisRun) -> None:
        run.status = Status.COMPLETED.value
        run.completed_at = datetime.utcnow()

    def mark_failed(self, run: AnalysisRun, error: str) -> None:
        run.status = Status.FAILED.value
        run.error = error[:4000]
        run.completed_at = datetime.utcnow()

    def list_recent(self, limit: int = 50) -> Sequence[AnalysisRun]:
        return self.db.execute(
            select(AnalysisRun).order_by(AnalysisRun.id.desc()).limit(limit)
        ).scalars().all()


class AgentResultRepository(BaseRepository[AgentAnalysisResult]):
    model = AgentAnalysisResult

    def for_run(self, run_id: int) -> Sequence[AgentAnalysisResult]:
        return self.db.execute(
            select(AgentAnalysisResult).where(
                AgentAnalysisResult.analysis_run_id == run_id
            )
        ).scalars().all()


class FeatureRepository(BaseRepository[ExtractedFeature]):
    model = ExtractedFeature

    def for_run(self, run_id: int) -> Sequence[ExtractedFeature]:
        return self.db.execute(
            select(ExtractedFeature).where(
                ExtractedFeature.analysis_run_id == run_id
            )
        ).scalars().all()


class PatternRepository(BaseRepository[WinningPattern]):
    model = WinningPattern

    def for_run(self, run_id: int) -> Sequence[WinningPattern]:
        return self.db.execute(
            select(WinningPattern)
            .where(WinningPattern.analysis_run_id == run_id)
            .order_by(WinningPattern.percentage.desc())
        ).scalars().all()


class StrategyRepository(BaseRepository[FinalStrategy]):
    model = FinalStrategy

    def for_run(self, run_id: int) -> FinalStrategy | None:
        return self.db.execute(
            select(FinalStrategy).where(FinalStrategy.analysis_run_id == run_id)
        ).scalar_one_or_none()
