# Project Progress

## Phase 1 — Data Pipeline ✅ Complete

- Built ingestion framework: `src/collectors/news_fetcher.py` (NewsAPI) and `src/collectors/market_data.py` (yfinance)
- `src/storage/db_manager.py` with SHA-256 deduplication via `INSERT OR IGNORE` on `headline_hash`
- `market_date` normalization: maps UTC timestamps to the trading session they can realistically affect (pre/post 16:00 EST rule)
- Log returns stored instead of simple returns for additive, near-normal price series
- `src/scheduler.py` with APScheduler wiring `news_job` (4h) and `price_job` (1h)
- `check_pipeline_health.py` for live observability without disrupting the running pipeline
- `.gitignore` configured to exclude `data/`, `logs/`, and `venv/`

## Phase 2 — NLP & Sentiment Scoring ✅ Complete

- `training/prepare_data.py`: builds structured training dataset from collected headlines
- `training/train.py`: fine-tunes ProsusAI/finbert on the prepared dataset; global seeding for MPS + NumPy reproducibility
- Fine-tuned checkpoint saved to `training/finetuned_finbert/`
- `src/inference/model_runner.py`: loads checkpoint, runs inference, writes `sentiment_label`, `confidence`, and attention-derived keyword back to DB
- `scoring_job` integrated into the scheduler (2h interval); decoupled from ingestion — drains `NULL` sentiment backlog independently
- Fixed `utils` import path and model init seed bug

## Phase 2 — Analytics & Dashboard 🔧 In Progress

- [ ] Pearson correlation between daily sentiment score and log return for each ticker
- [ ] Aggregate sentiment signal per ticker per `market_date`
- [ ] Dashboard / visualization layer

## Current Branch

`feature/auto-finbert-scoring` — automates FinBERT scoring in the scheduler; ready to merge to `main` once analytics work begins or is deferred.
