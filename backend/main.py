"""FastAPI application — API + static dashboard.

Run from the `backend/` directory:
    uvicorn main:app --reload --port 8000

Endpoints
---------
  POST /fetch-ads                        discover + store live ads for an app name
  POST /import-csv                       import ads from a CSV file
  GET  /apps                             list apps with ad counts / platforms
  GET  /ads                              browse ads (filter by app/platform/type)
  POST /analyze                          run the full multi-agent analysis
  GET  /analysis-runs                    list past runs (history)
  GET  /analysis-runs/{id}/agents        agent outputs for a run
  GET  /strategy/{run_id}                final evidence-based result for a run
  GET  /health                           liveness + mode (llm | heuristic)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from config import get_settings
from core.logging_config import configure_logging, get_logger
from database.connection import get_db, init_db
from database.repositories import AdRepository, AnalysisRunRepository, AppRepository
from schemas.models import (
    AdOut,
    AnalyzeRequest,
    AppSummary,
    FetchRequest,
    FetchResponse,
    FinalOutput,
    ImportResponse,
    RunSummary,
)
from services.analysis_service import AnalysisError, AnalysisService
from services.fetch_service import FetchError, FetchService
from services.import_service import ImportService
from services.llm_service import get_llm_service
from services.storage_service import sweep_orphans

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    sweep_orphans()
    logger.info(
        "API ready (provider=%s, model=%s)",
        settings.active_provider,
        settings.active_model,
    )
    yield


app = FastAPI(
    title="Ad Intelligence Platform",
    description="Discover winning advertising patterns and generate evidence-based strategies.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    provider = settings.active_provider
    return {
        "status": "ok",
        "mode": "heuristic" if provider == "heuristic" else "llm",
        "provider": provider,
        "model": settings.active_model,
        "database": settings.database_url.split("://", 1)[0],
        "live_fetch": settings.fetch_enabled,
    }


# --------------------------------------------------------------------------- #
# Live fetch (app name -> real ads)
# --------------------------------------------------------------------------- #
@app.post("/fetch-ads", response_model=FetchResponse)
def fetch_ads(req: FetchRequest, db: Session = Depends(get_db)):
    if not req.app_name or not req.app_name.strip():
        raise HTTPException(400, "app_name is required.")
    try:
        return FetchService(db).fetch(
            req.app_name.strip(), req.country, req.include_tiktok
        )
    except FetchError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Fetch failed")
        raise HTTPException(500, f"Fetch failed: {exc}")


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #
@app.post("/import-csv", response_model=ImportResponse)
async def import_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    content = await file.read()
    try:
        return ImportService(db).import_csv_bytes(content)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Import failed")
        raise HTTPException(500, f"Import failed: {exc}")


# --------------------------------------------------------------------------- #
# Browse
# --------------------------------------------------------------------------- #
@app.get("/apps", response_model=list[AppSummary])
def list_apps(db: Session = Depends(get_db)):
    return [AppSummary(**row) for row in AppRepository(db).list_with_counts()]


@app.get("/ads", response_model=list[AdOut])
def list_ads(
    app_name: str | None = Query(None),
    platform: str | None = Query(None),
    creative_type: str | None = Query(None),
    limit: int = Query(200, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    ads = AdRepository(db).query(app_name, platform, creative_type, limit, offset)
    return [AdOut.model_validate(a) for a in ads]


# --------------------------------------------------------------------------- #
# Analyse
# --------------------------------------------------------------------------- #
@app.post("/analyze", response_model=FinalOutput)
def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)):
    if not req.app_name or not req.app_name.strip():
        raise HTTPException(400, "app_name is required.")
    try:
        return AnalysisService(db).run_analysis(req.app_name.strip(), req.platforms or [])
    except AnalysisError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Analysis crashed")
        raise HTTPException(500, f"Analysis failed: {exc}")


# --------------------------------------------------------------------------- #
# History / results
# --------------------------------------------------------------------------- #
@app.get("/analysis-runs", response_model=list[RunSummary])
def list_runs(db: Session = Depends(get_db)):
    return [RunSummary.model_validate(r) for r in AnalysisRunRepository(db).list_recent()]


@app.get("/analysis-runs/{run_id}/agents")
def run_agent_outputs(run_id: int, db: Session = Depends(get_db)):
    return AnalysisService(db).get_agent_outputs(run_id)


@app.get("/strategy/{run_id}", response_model=FinalOutput)
def get_strategy(run_id: int, db: Session = Depends(get_db)):
    try:
        return AnalysisService(db).get_run_output(run_id)
    except AnalysisError as exc:
        raise HTTPException(404, str(exc))


# --------------------------------------------------------------------------- #
# Static dashboard (mounted last so API routes take precedence)
# --------------------------------------------------------------------------- #
_frontend = settings.frontend_dir
if _frontend.exists():
    # html=True serves index.html at /app/ and assets at /app/src/* (relative
    # asset URLs resolve correctly under this prefix).
    app.mount("/app", StaticFiles(directory=str(_frontend), html=True), name="frontend")

    @app.get("/")
    def root():
        return RedirectResponse(url="/app/")
else:  # pragma: no cover
    @app.get("/")
    def root():
        return {"message": "Ad Intelligence Platform API. See /docs."}
