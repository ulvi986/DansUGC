"""Central, typed application configuration.

Loaded once and cached. Reads from environment / `.env` (python-dotenv via
pydantic-settings). Every setting has a safe default so the app boots cleanly
on a fresh checkout.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root (…/ad-intelligence-platform) regardless of CWD.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    # The .env lives at project root; we also tolerate a backend-local one.
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Database -----------------------------------------------------------------
    database_url: str = "sqlite:///./ad_intelligence.db"

    # Gemini -------------------------------------------------------------------
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # OpenAI -------------------------------------------------------------------
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Live ad sourcing (ScrapeCreators / Apify) --------------------------------
    scrapecreators_api_key: str | None = None
    apify_token: str | None = None
    scrapecreators_tiktok_url: str = (
        "https://api.scrapecreators.com/v1/tiktok/ad-library/search"
    )
    fetch_country: str = "us"          # default ad storefront / region
    fetch_tiktok: bool = True          # also pull TikTok ads alongside Meta

    # Analysis engine ----------------------------------------------------------
    # consensus_samples > 1 reruns each agent N times and votes; great for
    # robustness but multiplies LLM latency/cost N×. Default 1 keeps the live
    # "fetch + analyse" flow snappy. Raise it for offline / batch quality runs.
    consensus_samples: int = 1
    llm_max_retries: int = 2
    analyze_video_binary: bool = False
    # Cap how many ads one run analyses, to bound latency when an LLM is enabled
    # (each ad triggers several LLM calls). 0 = no cap. Patterns over ~15 ads are
    # already directionally meaningful.
    analyze_max_ads: int = 15

    # Ops ----------------------------------------------------------------------
    log_level: str = "INFO"

    # Derived paths ------------------------------------------------------------
    @property
    def outputs_dir(self) -> Path:
        d = BACKEND_DIR / "outputs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def frontend_dir(self) -> Path:
        return PROJECT_ROOT / "frontend"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @staticmethod
    def _is_real_key(key: str | None) -> bool:
        """A key is usable only if set and not a leftover placeholder."""
        if not key or not key.strip():
            return False
        k = key.strip().lower()
        placeholders = ("your-openai-key", "replace", "changeme", "sk- ", "xxx")
        return not any(p in k for p in placeholders)

    @property
    def openai_enabled(self) -> bool:
        return self._is_real_key(self.openai_api_key)

    @property
    def active_provider(self) -> str:
        """Which LLM provider will actually be used (OpenAI preferred)."""
        if self.openai_enabled:
            return "openai"
        if self.llm_enabled:
            return "gemini"
        return "heuristic"

    @property
    def active_model(self) -> str:
        provider = self.active_provider
        if provider == "openai":
            return self.openai_model
        if provider == "gemini":
            return self.gemini_model
        return "heuristic"

    @property
    def fetch_enabled(self) -> bool:
        """Live ad fetching needs a ScrapeCreators key."""
        return bool(self.scrapecreators_api_key and self.scrapecreators_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
