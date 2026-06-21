"""Database seeding.

Creates tables and (optionally) imports the bundled sample CSV so the platform
has data to analyse on first run. Safe to run repeatedly — imports upsert.

Usage:
    python -m database.seed            # init + import sample data
    python -m database.seed --empty    # init schema only
"""
from __future__ import annotations

import sys
from pathlib import Path

from config import BACKEND_DIR
from core.logging_config import configure_logging, get_logger
from database.connection import SessionLocal, init_db
from services.import_service import ImportService

logger = get_logger("seed")

SAMPLE_CSV = BACKEND_DIR / "data" / "sample_ads.csv"


def seed(import_sample: bool = True) -> None:
    configure_logging()
    init_db()
    if not import_sample:
        logger.info("Schema initialised (no sample import).")
        return
    if not SAMPLE_CSV.exists():
        logger.warning("Sample CSV not found at %s; skipping import.", SAMPLE_CSV)
        return
    db = SessionLocal()
    try:
        resp = ImportService(db).import_csv_path(str(SAMPLE_CSV))
        logger.info(resp.message)
    finally:
        db.close()


if __name__ == "__main__":
    seed(import_sample="--empty" not in sys.argv)
