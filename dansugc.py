"""
dansugc.py
==========

End-to-end pipeline for analysing AI journaling apps:

1. Scrape the Apple App Store (iTunes Search API) for AI journaling / diary /
   mental-health / mood-tracking / self-reflection apps  ->  apps.csv
2. Filter the apps down to the ones that are actually relevant to
   "AI journaling" and attach a relevance score + reason.
3. For every relevant app, collect paid ad data from Meta (Facebook/Instagram)
   Ads Library and TikTok using ScrapeCreators and Apify  ->  ads.csv

Usage
-----
    python dansugc.py                 # full pipeline, default settings
    python dansugc.py --country us    # change App Store / ad storefront
    python dansugc.py --max-apps 15   # cap how many apps we scrape ads for
    python dansugc.py --skip-ads      # only build apps.csv
    python dansugc.py --no-apify      # use only ScrapeCreators for ads

API keys are read from environment variables (a local .env file is supported):
    SCRAPECREATORS_API_KEY   - required for ad scraping
    APIFY_TOKEN              - optional, enables the Apify Meta ads actor
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import time
from typing import Any, Callable, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 ships with requests; Retry import path differs by version
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

try:
    from apify_client import ApifyClient
except Exception:  # pragma: no cover - apify is optional
    ApifyClient = None


# --------------------------------------------------------------------------- #
# Configuration / constants
# --------------------------------------------------------------------------- #

APPS_CSV = "apps.csv"
ADS_CSV = "ads.csv"

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"

SC_BASE = "https://api.scrapecreators.com"
SC_SEARCH_COMPANIES_URL = f"{SC_BASE}/v1/facebook/adLibrary/search/companies"
SC_META_ADS_URL = f"{SC_BASE}/v1/facebook/adLibrary/company/ads"
# TikTok ad-library endpoint is configurable because ScrapeCreators changes it
# occasionally; override with SCRAPECREATORS_TIKTOK_URL if needed.
SC_TIKTOK_ADS_URL = os.getenv(
    "SCRAPECREATORS_TIKTOK_URL",
    f"{SC_BASE}/v1/tiktok/ad-library/search",
)

APIFY_META_ACTOR = os.getenv("APIFY_META_ACTOR", "apify/facebook-ads-scraper")

# Search queries used to discover apps in the App Store.
SEARCH_QUERIES = [
    "AI journaling",
    "journaling",
    "mental health journaling",
    "diary app",
    "AI diary",
    "self-reflection",
    "mood tracking",
    "personal growth journal",
    "gratitude journal",
]

# Keyword weights used for relevance scoring. Strong signals score higher.
RELEVANCE_KEYWORDS = {
    "ai journal": 5,
    "ai diary": 5,
    "journal": 3,
    "journaling": 3,
    "diary": 3,
    "ai": 2,
    "mood": 2,
    "reflection": 2,
    "self-reflection": 2,
    "mental health": 2,
    "gratitude": 2,
    "mindful": 1,
    "wellbeing": 1,
    "well-being": 1,
    "personal growth": 2,
    "therapy": 1,
    "emotions": 1,
}

# Apps whose name/description match these are almost certainly off-topic.
NEGATIVE_KEYWORDS = [
    "audio bible",
    "sermon",
    "recipe",
    "workout planner",
    "crypto",
    "invoice",
    "dating",
    "video editor",
    "photo editor",
    "game",
]

RELEVANCE_THRESHOLD = 8  # minimum score to keep an app

DATE_TODAY = dt.date.today().isoformat()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dansugc")


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

def build_session() -> requests.Session:
    """A requests session with automatic retries/backoff for transient errors."""
    session = requests.Session()
    if Retry is not None:
        retry = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    session.headers.update({"User-Agent": "dansugc-research/1.0"})
    return session


SESSION = build_session()


def get_json(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
) -> dict | None:
    """GET a URL and return parsed JSON, or None on failure (logged)."""
    try:
        resp = SESSION.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        log.warning("Request error for %s: %s", url, exc)
        return None

    if resp.status_code != 200:
        log.warning("HTTP %s from %s: %s", resp.status_code, url, resp.text[:200])
        return None

    try:
        return resp.json()
    except ValueError:
        log.warning("Non-JSON response from %s", url)
        return None


def retry(fn: Callable, *, attempts: int = 3, delay: float = 2.0):
    """Run a callable with simple retry semantics for flaky third-party calls."""
    last_exc: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we want to retry anything
            last_exc = exc
            log.warning("Attempt %s/%s failed: %s", i, attempts, exc)
            if i < attempts:
                time.sleep(delay * i)
    if last_exc:
        log.error("All %s attempts failed: %s", attempts, last_exc)
    return None


# --------------------------------------------------------------------------- #
# Step 1 — App Store scraping
# --------------------------------------------------------------------------- #

def search_app_store(query: str, country: str, limit: int = 50) -> list[dict]:
    """Query the iTunes Search API for software matching `query`."""
    params = {
        "term": query,
        "entity": "software",
        "country": country,
        "limit": limit,
        "media": "software",
    }
    data = get_json(ITUNES_SEARCH_URL, params=params)
    if not data:
        return []

    rows: list[dict] = []
    for app in data.get("results", []):
        rows.append(
            {
                "app_name": app.get("trackName"),
                "app_store_url": app.get("trackViewUrl"),
                "developer_name": app.get("sellerName") or app.get("artistName"),
                "category": app.get("primaryGenreName"),
                "rating": app.get("averageUserRating"),
                "review_count": app.get("userRatingCount"),
                "description": (app.get("description") or "").replace("\n", " ").strip(),
                "search_query_used": query,
                "country": country.upper(),
                "date_collected": DATE_TODAY,
                # kept internally for ad scraping, dropped from the CSV later
                "_bundle_id": app.get("bundleId"),
            }
        )
    return rows


def collect_apps(country: str, limit: int) -> pd.DataFrame:
    """Run every search query, de-duplicate, and persist apps.csv."""
    log.info("Searching the App Store across %d queries...", len(SEARCH_QUERIES))
    all_rows: list[dict] = []
    for query in SEARCH_QUERIES:
        rows = search_app_store(query, country=country, limit=limit) or []
        log.info("  '%s' -> %d apps", query, len(rows))
        all_rows.extend(rows)
        time.sleep(0.5)  # be polite to the API

    df = pd.DataFrame(all_rows)
    if df.empty:
        log.error("No apps returned from the App Store.")
        return df

    # De-duplicate apps (same app appears for several queries). Keep the first
    # query that found it but remember it was matched by multiple keywords.
    before = len(df)
    df = df.drop_duplicates(subset=["app_store_url"], keep="first").reset_index(drop=True)
    log.info("De-duplicated apps: %d -> %d", before, len(df))

    csv_cols = [
        "app_name",
        "app_store_url",
        "developer_name",
        "category",
        "rating",
        "review_count",
        "description",
        "search_query_used",
        "country",
        "date_collected",
    ]
    df[csv_cols].to_csv(APPS_CSV, index=False, encoding="utf-8-sig")
    log.info("Wrote %d apps to %s", len(df), APPS_CSV)
    return df


# --------------------------------------------------------------------------- #
# Step 2 — App filtering / relevance scoring
# --------------------------------------------------------------------------- #

def score_relevance(app: dict) -> tuple[int, str]:
    """Return (score, reason) describing how relevant an app is to AI journaling."""
    text = f"{app.get('app_name', '')} {app.get('description', '')}".lower()
    category = str(app.get("category", "")).lower()

    score = 0
    matched: list[str] = []
    for keyword, weight in RELEVANCE_KEYWORDS.items():
        if keyword in text:
            score += weight
            matched.append(keyword)

    # Bonus when the app explicitly combines AI with journaling/diary.
    has_ai = "ai" in matched or "ai journal" in matched or "ai diary" in matched
    has_journal = any(k in matched for k in ("journal", "journaling", "diary"))
    if has_ai and has_journal:
        score += 3
        matched.append("ai+journal combo")

    # Category bump for health & lifestyle apps.
    if category in ("health & fitness", "lifestyle", "medical"):
        score += 1

    # Strong penalty for clearly unrelated apps.
    penalty = [neg for neg in NEGATIVE_KEYWORDS if neg in text]
    score -= 3 * len(penalty)

    if not matched:
        reason = "No journaling/AI signals found"
    else:
        reason = "Matched: " + ", ".join(sorted(set(matched)))
        if penalty:
            reason += f" | Penalised for: {', '.join(penalty)}"
    return score, reason


def filter_apps(df: pd.DataFrame) -> pd.DataFrame:
    """Score every app and keep the ones above the relevance threshold."""
    if df.empty:
        return df

    scores, reasons = [], []
    for _, row in df.iterrows():
        score, reason = score_relevance(row.to_dict())
        scores.append(score)
        reasons.append(reason)

    df = df.copy()
    df["relevance_score"] = scores
    df["relevance_reason"] = reasons

    relevant = (
        df[df["relevance_score"] >= RELEVANCE_THRESHOLD]
        .sort_values("relevance_score", ascending=False)
        .reset_index(drop=True)
    )
    log.info(
        "Relevance filter: %d/%d apps kept (threshold=%d)",
        len(relevant),
        len(df),
        RELEVANCE_THRESHOLD,
    )

    # Re-write apps.csv including the scoring columns so the file is transparent.
    csv_cols = [
        "app_name",
        "app_store_url",
        "developer_name",
        "category",
        "rating",
        "review_count",
        "description",
        "search_query_used",
        "country",
        "date_collected",
        "relevance_score",
        "relevance_reason",
    ]
    df.sort_values("relevance_score", ascending=False)[csv_cols].to_csv(
        APPS_CSV, index=False, encoding="utf-8-sig"
    )
    log.info("Updated %s with relevance scores", APPS_CSV)
    return relevant


# --------------------------------------------------------------------------- #
# Step 3 — Ads scraping
# --------------------------------------------------------------------------- #

def _norm(value: Any) -> str:
    return "" if value is None else str(value).strip()


def first_present(d: dict, *keys: str) -> Any:
    for k in keys:
        if d.get(k) not in (None, ""):
            return d.get(k)
    return None


def brand_query(app_name: str) -> str:
    """Derive a short brand name from a full App Store title.

    'Solou: AI Diary & Mood Journal' -> 'Solou'
    'Rosebud - AI Journal'          -> 'Rosebud'
    Falls back to the full name when no separator is present.
    """
    for sep in (":", " - ", " – ", "|"):
        if sep in app_name:
            head = app_name.split(sep, 1)[0].strip()
            if head:
                return head
    return app_name.strip()


def flatten_media(value: Any) -> str:
    """TikTok video_url is sometimes a dict keyed by resolution; pick one URL."""
    if isinstance(value, dict):
        for key in ("720p", "540p", "480p", "360p"):
            if value.get(key):
                return str(value[key])
        for v in value.values():
            if v:
                return str(v)
        return ""
    return _norm(value)


def extract_meta_text(snapshot: dict, ad: dict) -> str:
    """Meta body/title can be a {'text': ...} dict or live inside cards."""
    body = snapshot.get("body")
    if isinstance(body, dict):
        text = _norm(body.get("text"))
    else:
        text = _norm(body)
    if not text:
        text = _norm(snapshot.get("title")) or _norm(snapshot.get("caption"))
    if not text:
        for card in snapshot.get("cards") or []:
            cand = _norm(card.get("body")) or _norm(card.get("link_description"))
            if cand:
                text = cand
                break
    if not text:
        text = _norm(first_present(ad, "ad_text", "adText", "text"))
    return text


def extract_meta_creative(snapshot: dict) -> tuple[str, str]:
    """Return (creative_type, media_url) from a Meta ad snapshot.

    Looks at top-level videos/images and falls back to per-card media.
    """
    video_fields = ("video_hd_url", "video_sd_url", "video_preview_image_url")
    image_fields = ("original_image_url", "resized_image_url", "image_url")

    # Top-level videos / images.
    for vid in snapshot.get("videos") or []:
        url = first_present(vid, *video_fields)
        if url:
            return "video", str(url)
    for img in snapshot.get("images") or []:
        url = first_present(img, *image_fields)
        if url:
            return "image", str(url)

    # Per-card media (carousel / link ads).
    for card in snapshot.get("cards") or []:
        url = first_present(card, *video_fields)
        if url:
            return "video", str(url)
        url = first_present(card, *image_fields)
        if url:
            return "image", str(url)

    return ("video" if snapshot.get("videos") else "image"), ""


def sc_find_page(query: str, api_key: str) -> dict | None:
    """Find the best matching Facebook page (advertiser) for an app name."""
    data = retry(
        lambda: get_json(
            SC_SEARCH_COMPANIES_URL,
            params={"query": query},
            headers={"x-api-key": api_key},
        )
    )
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
    # Prefer an exact-ish name match, otherwise the first (most relevant) result.
    qlow = query.lower()
    for r in results:
        if _norm(r.get("name")).lower() == qlow:
            return r
    return results[0]


def sc_meta_ads(page_id: str, app_name: str, country: str, api_key: str) -> list[dict]:
    """Pull active Meta ads for a page via ScrapeCreators."""
    data = retry(
        lambda: get_json(
            SC_META_ADS_URL,
            params={"pageId": page_id},
            headers={"x-api-key": api_key},
        )
    )
    if not data:
        return []
    ads = data.get("ads") or data.get("results") or data.get("searchResults") or []
    rows: list[dict] = []
    for ad in ads:
        snapshot = ad.get("snapshot") or {}
        creative_type, media = extract_meta_creative(snapshot)
        is_active = ad.get("is_active")
        status = "active" if is_active else ("inactive" if is_active is False else "unknown")
        rows.append(
            normalize_ad(
                app_name=app_name,
                platform="meta",
                ad_id=first_present(ad, "ad_archive_id", "adArchiveId", "id"),
                ad_url=first_present(ad, "url", "ad_snapshot_url"),
                advertiser=first_present(ad, "page_name", "pageName")
                or snapshot.get("page_name"),
                ad_text=extract_meta_text(snapshot, ad),
                creative_type=creative_type,
                media_url=media,
                # Prefer the human-readable ISO date over the unix timestamp.
                start_date=first_present(ad, "start_date_string", "start_date", "startDate"),
                country=country.upper(),
                status=status,
                source="scrapecreators",
            )
        )
    return rows


def sc_tiktok_ads(query: str, app_name: str, country: str, api_key: str) -> list[dict]:
    """Pull TikTok ads via ScrapeCreators (best-effort; endpoint may vary)."""
    data = retry(
        lambda: get_json(
            SC_TIKTOK_ADS_URL,
            # period is required by the endpoint to return "top ads"; query
            # narrows the results to the app name.
            params={"query": query, "region": country.upper(), "period": "180"},
            headers={"x-api-key": api_key},
        )
    )
    if not data:
        return []
    ads = data.get("ads") or []
    if isinstance(ads, dict):  # defensive: some payloads nest the list
        ads = ads.get("ads") or ads.get("materials") or []
    rows: list[dict] = []
    for ad in ads:
        video_info = ad.get("video_info") or {}
        media = first_present(
            video_info, "video_url", "cover", "cover_image_url"
        ) or first_present(ad, "video_url", "cover")
        rows.append(
            normalize_ad(
                app_name=app_name,
                platform="tiktok",
                ad_id=first_present(ad, "id", "ad_id", "material_id"),
                ad_url=first_present(ad, "url", "ad_url", "share_url"),
                advertiser=first_present(ad, "brand_name", "advertiser_name"),
                ad_text=first_present(ad, "ad_title", "ad_text", "text", "title"),
                creative_type="video",
                media_url=media,
                start_date=first_present(ad, "first_shown_date", "start_date", "create_time"),
                country=country.upper(),
                status=first_present(ad, "status") or "active",
                source="scrapecreators",
            )
        )
    return rows


def apify_meta_ads(
    client: "ApifyClient",
    page_id: str,
    app_name: str,
    country: str,
    max_items: int = 20,
) -> list[dict]:
    """Pull Meta ads for a page via the Apify Facebook Ads Scraper actor."""
    meta_url = (
        "https://www.facebook.com/ads/library/"
        f"?active_status=active&ad_type=all&country={country.upper()}"
        f"&search_type=page&view_all_page_id={page_id}"
    )
    run_input = {"startUrls": [{"url": meta_url}], "maxItems": max_items}

    run = retry(lambda: client.actor(APIFY_META_ACTOR).call(run_input=run_input))
    if not run:
        return []

    # The apify_client .call() may return a plain dict (older versions) or a
    # pydantic `Run` object (newer versions); support both.
    if isinstance(run, dict):
        dataset_id = run.get("defaultDatasetId")
    else:
        dataset_id = getattr(run, "default_dataset_id", None) or getattr(
            run, "defaultDatasetId", None
        )
    if not dataset_id:
        return []
    items = retry(lambda: client.dataset(dataset_id).list_items().items) or []

    rows: list[dict] = []
    for ad in items:
        snapshot = ad.get("snapshot") or {}
        creative_type, media = extract_meta_creative(snapshot)
        if not media:
            media = first_present(ad, "imageUrl", "videoUrl", "image_url", "video_url")
        is_active = first_present(ad, "isActive", "is_active")
        status = "active" if is_active else ("inactive" if is_active is False else "unknown")
        rows.append(
            normalize_ad(
                app_name=app_name,
                platform="meta",
                ad_id=first_present(ad, "adArchiveId", "ad_archive_id", "id"),
                ad_url=first_present(ad, "url", "adSnapshotUrl"),
                advertiser=first_present(ad, "pageName", "page_name"),
                ad_text=extract_meta_text(snapshot, ad),
                creative_type=creative_type,
                media_url=media,
                start_date=first_present(ad, "start_date_string", "startDate", "start_date"),
                country=country.upper(),
                status=status,
                source="apify",
            )
        )
    return rows


def normalize_ad(**kwargs) -> dict:
    """Produce a row matching the ads.csv schema, filling missing keys."""
    return {
        "app_name": kwargs.get("app_name"),
        "platform": kwargs.get("platform"),
        "ad_id": _norm(kwargs.get("ad_id")),
        "ad_url": _norm(kwargs.get("ad_url")),
        "advertiser_name": _norm(kwargs.get("advertiser")),
        "ad_text": _norm(kwargs.get("ad_text")),
        "creative_type": _norm(kwargs.get("creative_type")),
        "image_or_video_url": flatten_media(kwargs.get("media_url")),
        "start_date": _norm(kwargs.get("start_date")),
        "country": kwargs.get("country"),
        "status": _norm(kwargs.get("status")),
        "source": kwargs.get("source"),
        "date_collected": DATE_TODAY,
    }


def collect_ads(
    apps: pd.DataFrame,
    country: str,
    max_apps: int,
    use_apify: bool,
) -> pd.DataFrame:
    """For each relevant app, gather Meta + TikTok ads and persist ads.csv."""
    api_key = os.getenv("SCRAPECREATORS_API_KEY")
    if not api_key:
        log.error("SCRAPECREATORS_API_KEY is not set; cannot scrape ads.")
        return pd.DataFrame()

    apify_client = None
    if use_apify:
        token = os.getenv("APIFY_TOKEN")
        if token and ApifyClient is not None:
            apify_client = ApifyClient(token)
            log.info("Apify client enabled (actor=%s)", APIFY_META_ACTOR)
        else:
            log.warning("Apify disabled (missing APIFY_TOKEN or apify_client package).")

    targets = apps.head(max_apps) if max_apps else apps
    log.info("Collecting ads for %d apps...", len(targets))

    all_ads: list[dict] = []
    for _, app in targets.iterrows():
        name = app["app_name"]
        query = brand_query(name)
        log.info("App: %s (brand query: '%s')", name, query)

        # Find the advertiser page once and reuse for both ad sources.
        page = sc_find_page(query, api_key)
        page_id = _norm(page.get("page_id") or page.get("id")) if page else ""
        if page:
            log.info("  Matched page '%s' (id=%s)", page.get("name"), page_id or "?")
        else:
            log.info("  No advertiser page found on Meta.")

        # --- Meta via ScrapeCreators ---
        if page_id:
            meta_sc = sc_meta_ads(page_id, name, country, api_key)
            log.info("  Meta/ScrapeCreators: %d ads", len(meta_sc))
            all_ads.extend(meta_sc)

        # --- Meta via Apify ---
        if apify_client and page_id:
            meta_apify = apify_meta_ads(apify_client, page_id, name, country)
            log.info("  Meta/Apify: %d ads", len(meta_apify))
            all_ads.extend(meta_apify)

        # --- TikTok via ScrapeCreators ---
        tiktok = sc_tiktok_ads(query, name, country, api_key)
        log.info("  TikTok/ScrapeCreators: %d ads", len(tiktok))
        all_ads.extend(tiktok)

        time.sleep(0.5)

    df = pd.DataFrame(all_ads)
    if df.empty:
        log.warning("No ads collected. Writing empty ads.csv with headers.")
        df = pd.DataFrame(
            columns=[
                "app_name", "platform", "ad_id", "ad_url", "advertiser_name",
                "ad_text", "creative_type", "image_or_video_url", "start_date",
                "country", "status", "source", "date_collected",
            ]
        )
    else:
        before = len(df)
        # De-duplicate: same ad can come from multiple sources/queries.
        df["_dedup"] = df["platform"] + "|" + df["ad_id"].astype(str)
        # Rows without an ad_id can't be reliably deduped; keep them all.
        has_id = df["ad_id"].astype(str).str.len() > 0
        deduped = pd.concat(
            [
                df[has_id].drop_duplicates(subset=["_dedup"], keep="first"),
                df[~has_id],
            ]
        ).drop(columns=["_dedup"]).reset_index(drop=True)
        log.info("De-duplicated ads: %d -> %d", before, len(deduped))
        df = deduped

    df.to_csv(ADS_CSV, index=False, encoding="utf-8-sig")
    log.info("Wrote %d ads to %s", len(df), ADS_CSV)
    return df


# --------------------------------------------------------------------------- #
# CLI / orchestration
# --------------------------------------------------------------------------- #

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI journaling app + ads scraper")
    parser.add_argument("--country", default="us", help="App Store / ads storefront (default: us)")
    parser.add_argument("--limit", type=int, default=50, help="Results per search query (default: 50)")
    parser.add_argument("--max-apps", type=int, default=10, help="Max relevant apps to scrape ads for (0 = all)")
    parser.add_argument("--skip-ads", action="store_true", help="Only build apps.csv")
    parser.add_argument("--no-apify", action="store_true", help="Use only ScrapeCreators for ads")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    log.info("=== dansugc pipeline starting (country=%s) ===", args.country)

    # Step 1
    apps_df = collect_apps(country=args.country, limit=args.limit)
    if apps_df.empty:
        log.error("Aborting: no apps to work with.")
        return 1

    # Step 2
    relevant = filter_apps(apps_df)
    if relevant.empty:
        log.warning("No relevant apps passed the filter; nothing to scrape ads for.")
        return 0

    log.info("Top relevant apps:")
    for _, r in relevant.head(args.max_apps or len(relevant)).iterrows():
        log.info("  [%2d] %s", r["relevance_score"], r["app_name"])

    # Step 3
    if args.skip_ads:
        log.info("--skip-ads set; stopping after apps.csv.")
        return 0

    collect_ads(
        relevant,
        country=args.country,
        max_apps=args.max_apps,
        use_apify=not args.no_apify,
    )

    log.info("=== Done. Outputs: %s, %s ===", APPS_CSV, ADS_CSV)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        sys.exit(130)
