"""Live ad-sourcing service.

Replaces the CSV-import workflow with on-demand fetching: give it an app name
and it discovers the advertiser on the Meta Ad Library (and TikTok Ad Library)
via ScrapeCreators, normalises every ad, and persists it to the database — the
exact same `Ad` rows the analysis pipeline already reads from.

The heavy lifting (page discovery, snapshot parsing, media/text extraction) is
ported from the standalone `dansugc.py` research script so behaviour is
identical to the data that produced the bundled samples.

After a fetch, the 8-agent pipeline runs unchanged — it never sees where the
ads came from.
"""
from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from sqlalchemy import select
from sqlalchemy.orm import Session

try:  # urllib3 ships with requests; import path differs by version
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None

from config import get_settings
from core.logging_config import get_logger
from database.models import Ad, AdCreative
from database.repositories import AdRepository, AppRepository
from schemas.models import FetchResponse

logger = get_logger("fetch")

SC_BASE = "https://api.scrapecreators.com"
SC_SEARCH_COMPANIES_URL = f"{SC_BASE}/v1/facebook/adLibrary/search/companies"
SC_META_ADS_URL = f"{SC_BASE}/v1/facebook/adLibrary/company/ads"


class FetchError(Exception):
    """Raised when live fetching cannot proceed (no key, no advertiser, etc.)."""


# --------------------------------------------------------------------------- #
# Small parsing helpers (ported from dansugc.py)
# --------------------------------------------------------------------------- #
def _norm(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if d.get(k) not in (None, ""):
            return d.get(k)
    return None


def _brand_query(app_name: str) -> str:
    """'Solou: AI Diary & Mood Journal' -> 'Solou'. Falls back to full name."""
    for sep in (":", " - ", " – ", "|"):
        if sep in app_name:
            head = app_name.split(sep, 1)[0].strip()
            if head:
                return head
    return app_name.strip()


def _flatten_media(value: Any) -> str:
    """TikTok video_url is sometimes a resolution-keyed dict; pick one URL."""
    if isinstance(value, dict):
        for key in ("720p", "540p", "480p", "360p"):
            if value.get(key):
                return str(value[key])
        for v in value.values():
            if v:
                return str(v)
        return ""
    return _norm(value)


def _meta_text(snapshot: dict, ad: dict) -> str:
    body = snapshot.get("body")
    text = _norm(body.get("text")) if isinstance(body, dict) else _norm(body)
    if not text:
        text = _norm(snapshot.get("title")) or _norm(snapshot.get("caption"))
    if not text:
        for card in snapshot.get("cards") or []:
            cand = _norm(card.get("body")) or _norm(card.get("link_description"))
            if cand:
                text = cand
                break
    if not text:
        text = _norm(_first(ad, "ad_text", "adText", "text"))
    return text


def _meta_creative(snapshot: dict) -> tuple[str, str]:
    """Return (creative_type, media_url) from a Meta ad snapshot."""
    video_fields = ("video_hd_url", "video_sd_url", "video_preview_image_url")
    image_fields = ("original_image_url", "resized_image_url", "image_url")

    for vid in snapshot.get("videos") or []:
        url = _first(vid, *video_fields)
        if url:
            return "video", str(url)
    for img in snapshot.get("images") or []:
        url = _first(img, *image_fields)
        if url:
            return "image", str(url)
    for card in snapshot.get("cards") or []:
        url = _first(card, *video_fields)
        if url:
            return "video", str(url)
        url = _first(card, *image_fields)
        if url:
            return "image", str(url)
    return ("video" if snapshot.get("videos") else "image"), ""


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class FetchService:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.apps = AppRepository(db)
        self.ads = AdRepository(db)
        self._session = self._build_session()
        self._api_error: str | None = None   # last hard API failure (auth/credits)

    # -- HTTP ---------------------------------------------------------------- #
    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        if Retry is not None:
            retry = Retry(
                total=4,
                backoff_factor=1.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        session.headers.update({"User-Agent": "ad-intelligence/1.0"})
        return session

    def _get_json(self, url: str, params: dict) -> dict | None:
        headers = {"x-api-key": self.settings.scrapecreators_api_key}
        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=40)
        except requests.RequestException as exc:
            logger.warning("Request error for %s: %s", url, exc)
            return None
        if resp.status_code != 200:
            logger.warning("HTTP %s from %s: %s", resp.status_code, url, resp.text[:200])
            # Surface account-level problems (bad key / no credits) distinctly so
            # the caller doesn't mistake them for "no ads found".
            if resp.status_code in (401, 402, 403, 429):
                msg = resp.text[:200]
                try:
                    msg = resp.json().get("message", msg)
                except ValueError:
                    pass
                if resp.status_code == 402:
                    self._api_error = f"ScrapeCreators account is out of credits: {msg}"
                elif resp.status_code == 429:
                    self._api_error = f"ScrapeCreators rate limit hit: {msg}"
                else:
                    self._api_error = f"ScrapeCreators rejected the API key: {msg}"
            return None
        try:
            return resp.json()
        except ValueError:
            logger.warning("Non-JSON response from %s", url)
            return None

    # -- ScrapeCreators endpoints ------------------------------------------- #
    def _find_page(self, query: str) -> dict | None:
        data = self._get_json(SC_SEARCH_COMPANIES_URL, {"query": query})
        if not data:
            return None
        results = (
            data.get("searchResults")
            or data.get("results")
            or data.get("companies")
            or []
        )
        if not results:
            return None
        qlow = query.lower()
        for r in results:
            if _norm(r.get("name")).lower() == qlow:
                return r
        return results[0]

    def _meta_ads(self, page_id: str, app_name: str, country: str) -> list[dict]:
        data = self._get_json(SC_META_ADS_URL, {"pageId": page_id})
        if not data:
            return []
        ads = data.get("ads") or data.get("results") or data.get("searchResults") or []
        rows: list[dict] = []
        for ad in ads:
            snapshot = ad.get("snapshot") or {}
            creative_type, media = _meta_creative(snapshot)
            rows.append({
                "app_name": app_name,
                "platform": "meta",
                "ad_id": _norm(_first(ad, "ad_archive_id", "adArchiveId", "id")),
                "ad_url": _norm(_first(ad, "url", "ad_snapshot_url")),
                "advertiser_name": _norm(
                    _first(ad, "page_name", "pageName") or snapshot.get("page_name")
                ),
                "ad_text": _meta_text(snapshot, ad),
                "creative_type": creative_type,
                "image_or_video_url": _flatten_media(media),
                "start_date": _norm(
                    _first(ad, "start_date_string", "start_date", "startDate")
                ),
                "country": country.upper(),
            })
        return rows

    def _tiktok_ads(self, query: str, app_name: str, country: str) -> list[dict]:
        data = self._get_json(
            self.settings.scrapecreators_tiktok_url,
            {"query": query, "region": country.upper(), "period": "180"},
        )
        if not data:
            return []
        ads = data.get("ads") or []
        if isinstance(ads, dict):  # some payloads nest the list
            ads = ads.get("ads") or ads.get("materials") or []
        rows: list[dict] = []
        for ad in ads:
            video_info = ad.get("video_info") or {}
            media = _first(video_info, "video_url", "cover", "cover_image_url") or _first(
                ad, "video_url", "cover"
            )
            rows.append({
                "app_name": app_name,
                "platform": "tiktok",
                "ad_id": _norm(_first(ad, "id", "ad_id", "material_id")),
                "ad_url": _norm(_first(ad, "url", "ad_url", "share_url")),
                "advertiser_name": _norm(_first(ad, "brand_name", "advertiser_name")),
                "ad_text": _norm(_first(ad, "ad_title", "ad_text", "text", "title")),
                "creative_type": "video",
                "image_or_video_url": _flatten_media(media),
                "start_date": _norm(
                    _first(ad, "first_shown_date", "start_date", "create_time")
                ),
                "country": country.upper(),
            })
        return rows

    # -- Public API ---------------------------------------------------------- #
    def fetch(
        self,
        app_name: str,
        country: str | None = None,
        include_tiktok: bool | None = None,
    ) -> FetchResponse:
        """Discover and persist live ads for `app_name`. Returns a summary."""
        if not self.settings.fetch_enabled:
            raise FetchError(
                "Live fetching is disabled: set SCRAPECREATORS_API_KEY in .env."
            )

        app_name = app_name.strip()
        if not app_name:
            raise FetchError("app_name is required.")

        country = (country or self.settings.fetch_country).strip() or "us"
        include_tiktok = (
            self.settings.fetch_tiktok if include_tiktok is None else include_tiktok
        )
        query = _brand_query(app_name)
        logger.info("Fetching ads for '%s' (brand query='%s', country=%s)",
                    app_name, query, country)

        collected: list[dict] = []

        page = self._find_page(query)
        page_id = _norm(page.get("page_id") or page.get("id")) if page else ""
        advertiser = _norm(page.get("name")) if page else ""
        if page_id:
            logger.info("Matched advertiser '%s' (page_id=%s)", advertiser, page_id)
            collected.extend(self._meta_ads(page_id, app_name, country))
        else:
            logger.info("No Meta advertiser page found for '%s'", query)

        if include_tiktok:
            collected.extend(self._tiktok_ads(query, app_name, country))

        # Drop ads with no usable content (no text and no media).
        collected = [
            r for r in collected
            if r.get("ad_text") or r.get("image_or_video_url")
        ]

        if not collected:
            if self._api_error:
                raise FetchError(self._api_error)
            raise FetchError(
                f"No ads found for '{app_name}'"
                f"{' (advertiser ' + advertiser + ')' if advertiser else ''}. "
                "Try the brand name as it appears on Facebook/TikTok."
            )

        imported, by_platform = self._persist(app_name, collected)
        total = self.ads.count(app_name=app_name)
        message = (
            f"Fetched {imported} new ad(s) for '{app_name}' "
            f"({', '.join(f'{k}: {v}' for k, v in by_platform.items()) or 'none'}). "
            f"{total} ad(s) now stored."
        )
        logger.info(message)
        return FetchResponse(
            app_name=app_name,
            advertiser_name=advertiser or None,
            fetched_ads=imported,
            total_ads=total,
            by_platform=by_platform,
            message=message,
        )

    # -- Persistence --------------------------------------------------------- #
    def _persist(self, app_name: str, rows: list[dict]) -> tuple[int, dict[str, int]]:
        app = self.apps.upsert(app_name)
        imported = 0
        by_platform: dict[str, int] = {}
        seen: set[str] = set()

        for r in rows:
            ad_id = r.get("ad_id") or None
            platform = r.get("platform")

            # In-batch dedupe (same ad can appear twice in a response).
            key = f"{platform}|{ad_id}" if ad_id else None
            if key and key in seen:
                continue
            if key:
                seen.add(key)

            ad = self._find_existing(app_name, ad_id)
            is_new = ad is None
            if is_new:
                ad = Ad(app_id=app.id, app_name=app_name)
                self.db.add(ad)

            ad.app_id = app.id
            ad.ad_id = ad_id
            ad.platform = platform
            ad.creative_type = r.get("creative_type") or None
            ad.advertiser_name = r.get("advertiser_name") or None
            ad.ad_text = r.get("ad_text") or None
            ad.image_or_video_url = r.get("image_or_video_url") or None
            ad.ad_url = r.get("ad_url") or None
            ad.country = r.get("country") or None
            ad.start_date = r.get("start_date") or None
            self.db.flush()

            if is_new:
                if ad.image_or_video_url:
                    self.db.add(AdCreative(
                        ad_id=ad.id,
                        creative_type=ad.creative_type,
                        url=ad.image_or_video_url,
                    ))
                imported += 1
                by_platform[platform] = by_platform.get(platform, 0) + 1

        self.db.commit()
        return imported, by_platform

    def _find_existing(self, app_name: str, ad_id: str | None) -> Ad | None:
        if not ad_id:
            return None
        return self.db.execute(
            select(Ad).where(Ad.app_name == app_name, Ad.ad_id == ad_id)
        ).scalar_one_or_none()
