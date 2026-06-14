# FinBERT: Automated Financial Sentiment Pipeline

An automated data engineering pipeline that collects financial news headlines and equity price data on scheduled intervals, deduplicates and warehouses them in a local SQLite store, and prepares a structured time-series dataset for downstream sentiment analysis using a fine-tuned FinBERT transformer model.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![APScheduler](https://img.shields.io/badge/APScheduler-3.10+-FF6B35?style=flat)](https://apscheduler.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This is **Phase 1** of a two-phase project. The pipeline continuously ingests raw financial data and structures it for inference. The inference layer — fine-tuned FinBERT scoring, attention-weight extraction, and correlation analytics — is built in Phase 2.

| Phase | Scope | Status |
|:---|:---|:---|
| Phase 1 — Data pipeline | Ingestion · Deduplication · SQLite persistence · Observability | ✅ Complete |
| Phase 2 — NLP + Analytics | FinBERT fine-tuning · Sentiment scoring · Price correlation · Dashboard | 🔧 In progress |

---

## Key Engineering Decisions

### SHA-256 cryptographic deduplication
Every headline is hashed before insertion. The database enforces a `UNIQUE` constraint on `headline_hash`, and all inserts use `INSERT OR IGNORE` — meaning duplicates are silently skipped at the database level rather than caught in application code. This prevents repeated news alerts from inflating downstream sentiment scores and avoids expensive `SELECT` checks before every insert.

### market_date normalisation
Raw article timestamps from NewsAPI are in UTC. A headline published at 23:00 EST cannot affect that day's closing price. The pipeline maps every timestamp to the trading session it can realistically affect:

- Published before 16:00 EST on a weekday → same trading day
- Published at or after 16:00 EST, or on a weekend → next weekday open

This normalisation is what makes the (ticker, market_date) join between `sentiment_scores` and `price_data` statistically meaningful.

### Log returns over simple returns
Daily price returns are stored as log returns: `ln(close / prev_close)` rather than `(close - prev_close) / prev_close`. Log returns are additive across time periods and more normally distributed — both properties the Pearson correlation engine in Phase 2 depends on.

### Decoupled ingestion and inference
The scheduler writes headlines with `NULL` sentiment fields. The inference layer (Phase 2) reads unscored records via `get_unscored_headlines()` and writes labels back independently. Neither service needs to know the other is running. This means the pipeline accumulates data even before the model is deployed.

---

## Architecture

```
[NewsAPI]  ──── every 4h ────┐
                              ├──► [src/scheduler.py] ──► [SHA-256 hash check] ──► [SQLite]
[yfinance] ──── every 1h ────┘                                                  (sentiment_pipeline.db)
                                                                                        │
                                                              [check_pipeline_health.py]┘
                                                              (observability — safe to run in parallel)
```

**Data flow:**

1. `scheduler.py` triggers `news_job` and `price_job` on configurable intervals
2. Each collector fetches data and returns normalised records
3. `db_manager.py` applies `INSERT OR IGNORE` deduplication and writes to SQLite
4. Every job execution is logged to `job_log` with inserted/skipped counts
5. `check_pipeline_health.py` reads the database and prints a live health summary

---

## Tech Stack

| Category | Technology |
|:---|:---|
| Language | Python 3.12+ |
| Scheduler | APScheduler 3.10+ |
| Database | SQLite3 (standard library) |
| News data | NewsAPI v2 `/everything` endpoint |
| Price data | yfinance (OHLCV, auto-adjusted) |
| Deduplication | hashlib SHA-256 (standard library) |
| NLP model (Phase 2) | ProsusAI/finbert — HuggingFace Transformers |

---

## Project Structure

```
finbert/
├── config/
│   ├── settings.py               # Centralised config: tickers, intervals, paths, API keys
│   └── logging_config.py         # Console + rotating file log handler setup
├── src/
│   ├── collectors/
│   │   ├── news_fetcher.py       # NewsAPI fetcher: market_date normalisation, backoff
│   │   └── market_data.py        # yfinance OHLCV fetcher: log return calculation
│   ├── storage/
│   │   └── db_manager.py         # Schema init, INSERT OR IGNORE, health queries
│   └── scheduler.py              # APScheduler entry point: news_job + price_job
├── check_pipeline_health.py      # Observability script: DB metrics, job history, backlog
├── view_data.py                  # Quick DB table viewer for local inspection
├── data/
│   └── sentiment_pipeline.db     # SQLite database (excluded from Git via .gitignore)
├── logs/
│   └── pipeline.log              # Rotating execution log (excluded from Git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Prerequisites

- Python 3.12+
- A free [NewsAPI key](https://newsapi.org/register) (100 requests/day on free tier)
- macOS, Linux, or WSL2

### Installation

```bash
git clone https://github.com/Adam210503/finbert.git
cd finbert
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1 — Start the pipeline

```bash
export NEWSAPI_KEY=your_api_key_here
python src/scheduler.py
```

On macOS, wrap with `caffeinate` to prevent the system from sleeping and interrupting the scheduler:

```bash
NEWSAPI_KEY=your_api_key_here caffeinate -i python src/scheduler.py
```

> **Tip:** To keep your MacBook running with the display off, lower screen brightness to zero rather than closing the lid. Closing the lid triggers a hardware sleep signal that `caffeinate` cannot override.

Both jobs run immediately on startup, then repeat on their configured intervals.

### Step 2 — Monitor the pipeline

Open a second terminal window (leave the scheduler running) and run:

```bash
python check_pipeline_health.py
```

This is safe to run at any time — it only reads from the database and does not interrupt active scheduler threads.

Example output:

```
────────────────────────────────────────
Pipeline Health  —  2026-05-14 09:45:01
────────────────────────────────────────
Total headlines      :  247
  Scored             :    0  (awaiting Phase 2 inference)
  Unscored backlog   :  247

Total price records  :   42

Last news ingestion  :  2026-05-14T09:30:02Z
Last price record    :  2026-05-14

Recent job history:
  news_job   09:30:02   inserted=31  skipped=8
  price_job  09:30:05   inserted=6   skipped=0
  news_job   05:30:01   inserted=19  skipped=4
────────────────────────────────────────
```

### Step 3 — Inspect the database directly

```bash
python view_data.py
```

---

## Database Schema

### `sentiment_scores`

One row per unique headline. Sentiment fields are `NULL` until Phase 2 inference runs.

| Column | Type | Description |
|:---|:---|:---|
| `id` | INTEGER PK | Auto-increment |
| `ticker` | TEXT | AAPL / TSLA / SPY |
| `headline` | TEXT | Raw headline text |
| `source` | TEXT | Publishing outlet (e.g. reuters.com) |
| `raw_timestamp` | TEXT | Original UTC publish time (ISO 8601) |
| `market_date` | TEXT | Normalised trading date (YYYY-MM-DD) |
| `sentiment_label` | TEXT | positive / neutral / negative — filled by Phase 2 |
| `confidence` | REAL | Softmax probability of predicted class — filled by Phase 2 |
| `attention_keyword` | TEXT | Top attention-weighted token — filled by Phase 2 |
| `model_version` | TEXT | Checkpoint identifier for reproducibility |
| `headline_hash` | TEXT UNIQUE | SHA-256 hash for deduplication |

### `price_data`

One row per ticker per trading day.

| Column | Type | Description |
|:---|:---|:---|
| `id` | INTEGER PK | Auto-increment |
| `ticker` | TEXT | AAPL / TSLA / SPY |
| `market_date` | TEXT | Trading date (YYYY-MM-DD) |
| `open` | REAL | Opening price |
| `high` | REAL | Daily high |
| `low` | REAL | Daily low |
| `close` | REAL | Closing price |
| `volume` | INTEGER | Volume traded |
| `daily_return` | REAL | Log return: ln(close / prev_close) |

### `job_log`

One row per scheduler job execution. Used by `check_pipeline_health.py`.

| Column | Type | Description |
|:---|:---|:---|
| `id` | INTEGER PK | Auto-increment |
| `job_name` | TEXT | news_job / price_job |
| `ran_at` | TEXT | UTC execution timestamp |
| `inserted` | INTEGER | New records written to DB |
| `skipped` | INTEGER | Duplicates ignored |
| `error` | TEXT | Exception message if job failed, else NULL |

---

## Limitations

- **Free-tier API constraints.** NewsAPI allows 100 requests/day on the free tier. The current configuration uses 18 requests/day (3 tickers × 2 fetches/day), leaving comfortable headroom. yfinance is an unofficial wrapper with no SLA.
- **NYSE holidays not handled.** The `market_date` normalisation advances past weekends but does not account for public holidays. Headlines published on a NYSE holiday are assigned to the next calendar weekday, which may itself be a holiday.
- **No GPU inference.** Phase 2 FinBERT inference will run on CPU. Expected latency is 200–800ms per batch of 16–32 headlines — acceptable for a scheduled pipeline, not suitable for live streaming.
- **SQLite concurrency.** SQLite is single-writer. The scheduler and the health check script should not both attempt writes simultaneously. In the current architecture this is not an issue — the health check is read-only.

---

## What's Next (Phase 2)

- Fine-tune `ProsusAI/finbert` on a custom labeled dataset (Reddit PRAW + NewsAPI headlines, ~1,000+ samples)
- Deploy fine-tuned checkpoint for batch inference against the `unscored_headlines` backlog
- Implement rolling Pearson correlation between daily mean sentiment and next-day log return
- Build event study: return distributions at t+1h, t+4h, t+24h following sentiment spike events
- Streamlit dashboard with analytics panel and live observability metrics

---

## Author

**Adam Mikail**
[LinkedIn](https://www.linkedin.com/in/adammikail/) · [Email](mailto:adammikail2105@gmail.com)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
