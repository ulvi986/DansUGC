"""SQLAlchemy ORM models.

Design notes
------------
* Portable types only (String/Integer/Float/Text/DateTime/Boolean) so the schema
  migrates cleanly to PostgreSQL / MySQL / Oracle.
* JSON is stored through a `JSONEncoded` TypeDecorator backed by Text — every
  RDBMS supports Text, and we avoid dialect-specific JSON columns.
* Every table has a surrogate PK, created_at, updated_at.
* Indexes exactly as required: app_name, platform, creative_type,
  analysis_run_id, winner_score.
* Status columns use the Status string-enum (pending/running/completed/failed).
"""
from __future__ import annotations

import json
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


# --------------------------------------------------------------------------- #
# Portable JSON column
# --------------------------------------------------------------------------- #
class JSONEncoded(TypeDecorator):
    """Stores any JSON-serialisable Python value as Text. Portable everywhere."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, default=str)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


# --------------------------------------------------------------------------- #
# Core entities
# --------------------------------------------------------------------------- #
class App(Base, TimestampMixin):
    __tablename__ = "apps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    app_name: Mapped[str] = mapped_column(String(512), nullable=False, unique=True, index=True)
    developer_name: Mapped[str | None] = mapped_column(String(512))
    category: Mapped[str | None] = mapped_column(String(255))
    rating: Mapped[float | None] = mapped_column(Float)
    description: Mapped[str | None] = mapped_column(Text)

    ads: Mapped[list["Ad"]] = relationship(back_populates="app", cascade="all, delete-orphan")


class Ad(Base, TimestampMixin):
    __tablename__ = "ads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    app_id: Mapped[int | None] = mapped_column(ForeignKey("apps.id", ondelete="CASCADE"))

    ad_id: Mapped[str | None] = mapped_column(String(128))               # external id
    app_name: Mapped[str] = mapped_column(String(512), nullable=False)   # denormalised for fast filter
    platform: Mapped[str | None] = mapped_column(String(64))
    creative_type: Mapped[str | None] = mapped_column(String(64))
    advertiser_name: Mapped[str | None] = mapped_column(String(512))
    ad_text: Mapped[str | None] = mapped_column(Text)
    image_or_video_url: Mapped[str | None] = mapped_column(Text)
    ad_url: Mapped[str | None] = mapped_column(Text)
    country: Mapped[str | None] = mapped_column(String(16))

    # Engagement (optional — may be absent in source data; handled safely).
    impressions: Mapped[int | None] = mapped_column(Integer)
    likes: Mapped[int | None] = mapped_column(Integer)
    shares: Mapped[int | None] = mapped_column(Integer)
    comments: Mapped[int | None] = mapped_column(Integer)
    duration: Mapped[float | None] = mapped_column(Float)

    start_date: Mapped[str | None] = mapped_column(String(64))
    end_date: Mapped[str | None] = mapped_column(String(64))

    # Computed by the Scoring Agent during analysis.
    winner_score: Mapped[float | None] = mapped_column(Float)

    app: Mapped["App"] = relationship(back_populates="ads")
    creatives: Mapped[list["AdCreative"]] = relationship(
        back_populates="ad", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_ads_app_name", "app_name"),
        Index("ix_ads_platform", "platform"),
        Index("ix_ads_creative_type", "creative_type"),
        Index("ix_ads_winner_score", "winner_score"),
        Index("ix_ads_app_platform", "app_name", "platform"),
    )


class AdCreative(Base, TimestampMixin):
    __tablename__ = "ad_creatives"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id", ondelete="CASCADE"), index=True)
    creative_type: Mapped[str | None] = mapped_column(String(64), index=True)
    url: Mapped[str | None] = mapped_column(Text)
    duration: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default=Status.PENDING.value)

    ad: Mapped["Ad"] = relationship(back_populates="creatives")


# --------------------------------------------------------------------------- #
# Analysis entities
# --------------------------------------------------------------------------- #
class AnalysisRun(Base, TimestampMixin):
    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    app_name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    platforms: Mapped[list | None] = mapped_column(JSONEncoded)         # e.g. ["meta","tiktok"]
    status: Mapped[str] = mapped_column(String(32), default=Status.PENDING.value, index=True)
    analyzed_ads_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence_score: Mapped[float | None] = mapped_column(Float)
    evidence_summary: Mapped[str | None] = mapped_column(Text)
    reasoning_summary: Mapped[str | None] = mapped_column(Text)
    limitations: Mapped[list | None] = mapped_column(JSONEncoded)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)


class AgentAnalysisResult(Base, TimestampMixin):
    __tablename__ = "agent_analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    ad_id: Mapped[int | None] = mapped_column(ForeignKey("ads.id", ondelete="CASCADE"))
    agent_name: Mapped[str] = mapped_column(String(64), index=True)
    output: Mapped[dict | None] = mapped_column(JSONEncoded)
    confidence: Mapped[float | None] = mapped_column(Float)
    consensus_agreement: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default=Status.COMPLETED.value)
    source: Mapped[str | None] = mapped_column(String(32))  # "llm" | "heuristic"

    __table_args__ = (
        Index("ix_agent_results_run_agent", "analysis_run_id", "agent_name"),
    )


class ExtractedFeature(Base, TimestampMixin):
    __tablename__ = "extracted_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    ad_id: Mapped[int] = mapped_column(ForeignKey("ads.id", ondelete="CASCADE"), index=True)
    platform: Mapped[str | None] = mapped_column(String(64))

    hook_type: Mapped[str | None] = mapped_column(String(64))
    hook_strength: Mapped[float | None] = mapped_column(Float)
    cta_type: Mapped[str | None] = mapped_column(String(64))
    emotion_type: Mapped[str | None] = mapped_column(String(64))
    creative_format: Mapped[str | None] = mapped_column(String(64))
    human_present: Mapped[bool | None] = mapped_column(Boolean)
    app_screen_visible: Mapped[bool | None] = mapped_column(Boolean)
    ugc_style: Mapped[bool | None] = mapped_column(Boolean)
    video_length: Mapped[float | None] = mapped_column(Float)
    visual_density: Mapped[str | None] = mapped_column(String(32))
    face_visible: Mapped[bool | None] = mapped_column(Boolean)
    product_demo_present: Mapped[bool | None] = mapped_column(Boolean)
    raw: Mapped[dict | None] = mapped_column(JSONEncoded)


class WinningPattern(Base, TimestampMixin):
    __tablename__ = "winning_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    pattern_type: Mapped[str] = mapped_column(String(64), index=True)
    pattern_value: Mapped[str | None] = mapped_column(String(255))
    platform: Mapped[str | None] = mapped_column(String(64))
    frequency: Mapped[int] = mapped_column(Integer, default=0)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    percentage: Mapped[float] = mapped_column(Float, default=0.0)
    statement: Mapped[str | None] = mapped_column(Text)   # human-readable evidence line
    evidence: Mapped[dict | None] = mapped_column(JSONEncoded)


class FinalStrategy(Base, TimestampMixin):
    __tablename__ = "final_strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True, unique=True
    )
    confidence_score: Mapped[float | None] = mapped_column(Float)
    strategy: Mapped[dict | None] = mapped_column(JSONEncoded)
    # Full market-intelligence object (see agents/intelligence.py).
    intelligence: Mapped[dict | None] = mapped_column(JSONEncoded)
