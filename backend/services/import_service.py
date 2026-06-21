"""CSV import service.

CSV is used ONLY for import. After import, every downstream analysis reads
exclusively from the database.

Robustness requirements honoured here:
  * Never crash on missing columns — unknown/absent fields are filled with None.
  * Tolerant of column-name variants via an alias map.
  * Tolerant of BOM / encoding (`utf-8-sig`) and bad rows.
  * Idempotent-ish: existing (app_name, ad_id) ads are updated, not duplicated.
"""
from __future__ import annotations

import io
import math
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.logging_config import get_logger
from database.models import Ad, AdCreative
from database.repositories import AdRepository, AppRepository
from schemas.models import ImportResponse

logger = get_logger("import")

# Canonical field -> list of accepted source column names (lower-cased).
COLUMN_ALIASES: dict[str, list[str]] = {
    "ad_id": ["ad_id", "id", "adid"],
    "app_name": ["app_name", "app", "application", "app_title"],
    "platform": ["platform", "network", "channel"],
    "creative_type": ["creative_type", "media_type", "type", "format"],
    "advertiser_name": ["advertiser_name", "advertiser", "brand"],
    "ad_text": ["ad_text", "text", "caption", "copy", "body"],
    "image_or_video_url": ["image_or_video_url", "media_url", "creative_url", "url"],
    "ad_url": ["ad_url", "landing_url", "link"],
    "country": ["country", "geo", "region"],
    "impressions": ["impressions", "views", "imps"],
    "likes": ["likes", "reactions"],
    "shares": ["shares"],
    "comments": ["comments"],
    "duration": ["duration", "video_length", "length"],
    "start_date": ["start_date", "created", "date_started"],
    "end_date": ["end_date", "date_ended"],
}

OPTIONAL_FIELDS = [
    "impressions", "likes", "shares", "comments", "duration", "end_date",
]

INT_FIELDS = {"impressions", "likes", "shares", "comments"}
FLOAT_FIELDS = {"duration"}


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return value


def _to_int(value: Any) -> int | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _to_float(value: Any) -> float | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _build_column_map(columns: list[str]) -> dict[str, str]:
    """Map canonical field -> actual column present in the dataframe."""
    lower = {c.lower().strip(): c for c in columns}
    mapping: dict[str, str] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                mapping[field] = lower[alias]
                break
    return mapping


class ImportService:
    def __init__(self, db: Session):
        self.db = db
        self.apps = AppRepository(db)
        self.ads = AdRepository(db)

    def import_csv_bytes(self, content: bytes) -> ImportResponse:
        # utf-8-sig strips the BOM seen in the source files.
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", dtype=str)
        except Exception:
            df = pd.read_csv(io.BytesIO(content), encoding="latin-1", dtype=str)
        return self._import_dataframe(df)

    def import_csv_path(self, path: str) -> ImportResponse:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        return self._import_dataframe(df)

    def _import_dataframe(self, df: pd.DataFrame) -> ImportResponse:
        detected = list(df.columns)
        colmap = _build_column_map(detected)
        if "app_name" not in colmap:
            raise ValueError("CSV must contain an 'app_name' column (or an alias of it).")

        missing_optional = [f for f in OPTIONAL_FIELDS if f not in colmap]
        imported_ads = skipped = 0
        seen_apps: set[str] = set()

        for _, row in df.iterrows():
            record = {field: row.get(src) for field, src in colmap.items()}
            app_name = _clean(record.get("app_name"))
            if not app_name:
                skipped += 1
                continue

            app = self.apps.upsert(app_name)
            seen_apps.add(app_name)

            ad = self._find_existing(app_name, _clean(record.get("ad_id")))
            is_new = ad is None
            if is_new:
                ad = Ad(app_id=app.id, app_name=app_name)
                self.db.add(ad)

            ad.app_id = app.id
            ad.ad_id = _clean(record.get("ad_id"))
            ad.platform = _norm_platform(_clean(record.get("platform")))
            ad.creative_type = _norm_creative(_clean(record.get("creative_type")))
            ad.advertiser_name = _clean(record.get("advertiser_name"))
            ad.ad_text = _clean(record.get("ad_text"))
            ad.image_or_video_url = _clean(record.get("image_or_video_url"))
            ad.ad_url = _clean(record.get("ad_url"))
            ad.country = _clean(record.get("country"))
            ad.start_date = _clean(record.get("start_date"))
            ad.end_date = _clean(record.get("end_date"))
            for f in INT_FIELDS:
                setattr(ad, f, _to_int(record.get(f)))
            for f in FLOAT_FIELDS:
                setattr(ad, f, _to_float(record.get(f)))

            self.db.flush()

            # Mirror the creative asset (one row per ad here; extensible to many).
            if is_new and ad.image_or_video_url:
                self.db.add(
                    AdCreative(
                        ad_id=ad.id,
                        creative_type=ad.creative_type,
                        url=ad.image_or_video_url,
                        duration=ad.duration,
                    )
                )
            imported_ads += 1

        self.db.commit()
        logger.info("Imported %s ads across %s apps (%s skipped)",
                    imported_ads, len(seen_apps), skipped)

        return ImportResponse(
            imported_ads=imported_ads,
            imported_apps=len(seen_apps),
            skipped_rows=skipped,
            detected_columns=detected,
            missing_optional_columns=missing_optional,
            message=(
                f"Imported {imported_ads} ads for {len(seen_apps)} app(s). "
                f"Missing optional columns handled safely: {missing_optional or 'none'}."
            ),
        )

    def _find_existing(self, app_name: str, ad_id: str | None) -> Ad | None:
        if not ad_id:
            return None
        return self.db.execute(
            select(Ad).where(Ad.app_name == app_name, Ad.ad_id == ad_id)
        ).scalar_one_or_none()


def _norm_platform(value: str | None) -> str | None:
    if not value:
        return None
    v = value.lower()
    if "tik" in v:
        return "tiktok"
    if "meta" in v or "facebook" in v or "instagram" in v or v == "fb":
        return "meta"
    return v


def _norm_creative(value: str | None) -> str | None:
    if not value:
        return None
    v = value.lower()
    if "vid" in v:
        return "video"
    if "img" in v or "image" in v or "photo" in v or "static" in v:
        return "image"
    return v
