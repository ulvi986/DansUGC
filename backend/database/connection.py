"""Engine / session / Base.

The only database-specific line in the whole codebase is `DATABASE_URL`.
Everything else uses portable SQLAlchemy Core/ORM constructs so the same code
runs on SQLite (MVP), PostgreSQL, MySQL or Oracle unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config import get_settings
from core.logging_config import get_logger

logger = get_logger("database")
settings = get_settings()

_engine_kwargs: dict = {"pool_pre_ping": True, "future": True}
if settings.database_url.startswith("sqlite"):
    # Needed because FastAPI serves requests from a threadpool.
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.database_url, **_engine_kwargs)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

Base = declarative_base()


def get_db() -> Iterator[Session]:
    """FastAPI dependency — yields a session, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error, always close."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Transaction rolled back")
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Importing models registers them on Base.metadata."""
    from database import models  # noqa: F401  (side-effect: register tables)

    Base.metadata.create_all(bind=engine)
    _ensure_columns()
    logger.info("Database initialised at %s", settings.database_url)


def _ensure_columns() -> None:
    """Tiny forward-only migration: add columns introduced after a DB was first
    created. `create_all` never alters existing tables, so we add them by hand.
    Safe to run on every boot — only acts when a column is missing.
    """
    from sqlalchemy import inspect, text

    additions = {
        "final_strategies": {"intelligence": "TEXT"},
    }
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in additions.items():
            if not insp.has_table(table):
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl_type in cols.items():
                if name not in existing:
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {ddl_type}'))
                    logger.info("Migrated: added %s.%s", table, name)
