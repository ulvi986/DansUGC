# Ad Intelligence Platform

Discover **winning advertising patterns** from real Meta/TikTok ad data and
generate **evidence-based creative strategies** with a multi-agent AI system.

The platform never invents strategies. Every recommendation traces back to a
pattern mined from the analysed ads — e.g.:

> *"Problem-first hooks appeared in 74% of analysed ads (20/27)."*
> *"82% of TikTok creatives were UGC-style (17/21)."*

---

## Why this is not a toy

| Concern | How it's handled |
|---|---|
| **Storage** | Real relational schema (SQLAlchemy ORM + Repository Pattern). SQLite for the MVP; swap one `DATABASE_URL` line for PostgreSQL/MySQL/Oracle. CSV is **import-only**. |
| **Agents** | 8 specialised agents, each with its **own** system prompt returning **strict JSON**. |
| **Orchestration** | A dedicated orchestration layer runs the full pipeline and persists every stage. |
| **Consensus** | Voting agent runs N independent samples per analysis → weighted majority vote + consensus agreement. |
| **Evidence** | Pattern mining computes **real frequencies**; strategies are assembled only from those patterns. |
| **Reliability** | JSON repair + Pydantic validation + retries, rate-limit backoff, transactional writes + rollback, temp-file auto-cleanup, safe missing-column handling. |
| **Works with no API key** | Graceful degradation to deterministic heuristic analysis, so the whole pipeline runs end-to-end offline. |

---

## Architecture

```
CSV ──import──▶ SQLite (SQLAlchemy + Repository Pattern)
                     │  (all analysis reads from DB only)
                     ▼
            ┌─────────────── Orchestrator ───────────────┐
 Load Ads → │ Text → Video/Image → Feature → Pattern →    │
            │ Scoring → Voting(consensus) → Strategy      │ → Save every stage
            └────────────────────────────────────────────┘
                     ▼
        FinalOutput  +  outputs/final_strategy.json  +  Dashboard
```

### The 8 agents
1. **Text** — hooks, CTA, pain points, emotion, audience language, structure, keywords.
2. **Video** — human/face/app-screen presence, demo, pacing, hook placement, UGC vs brand, story. Falls back to text on failure; videos are streamed to a **temp** folder and **auto-deleted**.
3. **Image** — human, screenshot, brand, CTA, style, color, overlay, tone.
4. **Feature Extraction** — fuses agent outputs into structured features.
5. **Pattern Mining** — evidence-backed frequencies + platform-specific patterns.
6. **Scoring** — explainable 100-point weighted score with per-criterion explanations.
7. **Strategy** — final strategy assembled strictly from evidence.
8. **Voting** — independent samples → weighted majority + consensus scoring.

Chain-of-thought is never exposed — only **Evidence Summary**, **Reasoning
Summary** and **Confidence Score**.

---

## Project structure

```
ad-intelligence-platform/
  backend/
    main.py                  FastAPI app + endpoints + static dashboard
    config.py                typed settings (.env)
    core/                    logging, JSON repair/validation utilities
    agents/                  text, video, image, feature, pattern, scoring,
                             strategy, voting, orchestrator, base, lexicon
    database/                connection, models, repositories, seed
    services/                gemini, import, analysis, storage
    schemas/                 Pydantic contracts (agent I/O + API DTOs)
    outputs/                 final_strategy.json (+ per-run snapshots)
    data/sample_ads.csv      bundled demo data (real ads)
  frontend/
    index.html, src/         zero-build dashboard (vanilla JS)
  requirements.txt
  .env.example
  README.md
```

---

## Setup

Requires Python 3.10+.

```bash
# 1. install deps
pip install -r requirements.txt

# 2. configure (optional — runs without a key in heuristic mode)
cp .env.example .env          # then set GEMINI_API_KEY to enable Gemini

# 3. create DB + import bundled sample ads
cd backend
python -m database.seed

# 4. run the API + dashboard
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/** for the dashboard, or **/docs** for the API.

> Windows PowerShell: use `copy .env.example .env`.

---

## Usage

1. **Import** a CSV (or use the seeded sample data).
2. **Select** an app and one or more platforms.
3. **Run analysis** — the 8-agent pipeline executes and persists results.
4. Explore **Winning Patterns**, **Strategy**, **Scores**, **Agent Outputs**,
   **Evidence**, and re-open any **previous run** from history.

The system only analyses ads that exist in the DB for the selected
app/platform — it never generates strategies for non-existent ads.

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/import-csv` | Import ads from a CSV file (multipart). |
| `GET` | `/apps` | List apps with ad counts and platform breakdown. |
| `GET` | `/ads` | Browse ads (`app_name`, `platform`, `creative_type` filters). |
| `POST` | `/analyze` | Run the full multi-agent analysis (`{app_name, platforms[]}`). |
| `GET` | `/analysis-runs` | List previous runs (history). |
| `GET` | `/analysis-runs/{id}/agents` | Agent outputs for a run. |
| `GET` | `/strategy/{run_id}` | Canonical final result for a run. |
| `GET` | `/health` | Liveness + engine mode (`llm` / `heuristic`). |

### Final output shape

```json
{
  "run_id": 1,
  "app_name": "Mirror: Mental Health Journal",
  "platforms": ["meta", "tiktok"],
  "analyzed_ads_count": 43,
  "winning_patterns": [
    { "pattern_type": "top_hook_type", "pattern_value": "problem_first",
      "percentage": 62.8, "frequency": 27, "sample_size": 43,
      "statement": "Problem-first hooks were the most common hook, in 62.8% of analysed ads (27/43)." }
  ],
  "confidence_score": 0.71,
  "evidence_summary": "…",
  "reasoning_summary": "…",
  "limitations": ["…"],
  "strategy": {
    "strategy_name": "…", "target_audience": "…", "core_pain_point": "…",
    "hooks": ["…","…","…"], "creative_concept": "…",
    "video_script": ["…"], "platform_plan": { "tiktok": "…", "meta": "…" },
    "ab_test_plan": ["…"]
  }
}
```

---

## Migrating off SQLite

Change one line in `.env`:

```
# PostgreSQL
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/ad_intel
# MySQL
DATABASE_URL=mysql+pymysql://user:pass@host:3306/ad_intel
# Oracle
DATABASE_URL=oracle+oracledb://user:pass@host:1521/?service_name=XEPDB1
```

All models use portable types (JSON is stored via a Text-backed type decorator),
so no schema changes are required. Install the matching DB driver.

---

## Configuration (`.env`)

| Key | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./ad_intelligence.db` | Database connection. |
| `GEMINI_API_KEY` | *(empty)* | Enables Gemini; empty → heuristic mode. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model id. |
| `CONSENSUS_SAMPLES` | `3` | Independent samples per voting-backed agent. |
| `LLM_MAX_RETRIES` | `3` | JSON validation/retry attempts. |
| `ANALYZE_VIDEO_BINARY` | `false` | Stream video to temp for multimodal analysis. |
| `LOG_LEVEL` | `INFO` | Logging level. |
