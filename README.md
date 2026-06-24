# FinBERT: Automated Financial Sentiment Pipeline

An automated data engineering pipeline that collects financial news headlines and equity price data on scheduled intervals, deduplicates and warehouses them in a local SQLite store, and prepares a structured time-series dataset for downstream sentiment analysis using a fine-tuned FinBERT transformer model.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![APScheduler](https://img.shields.io/badge/APScheduler-3.10+-FF6B35?style=flat)](https://apscheduler.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This is a two-phase project. Phase 1 continuously ingests raw financial data; Phase 2 fine-tunes FinBERT and scores that data automatically as it arrives.

| Phase | Scope | Status |
|:---|:---|:---|
| Phase 1: Data pipeline | Ingestion, Deduplication, SQLite persistence, and Observability | ✅ Complete |
| Phase 2: NLP and Analytics | FinBERT fine-tuning, Sentiment scoring | ✅ Complete |
| Phase 2: Analytics | Price correlation and Dashboard | 🔧 In progress |

---

## Key Engineering Decisions

### SHA-256 cryptographic deduplication
Every headline is hashed before insertion. The database enforces a `UNIQUE` constraint on `headline_hash`, and all inserts use `INSERT OR IGNORE`. This means duplicates are silently skipped at the database level instead of being caught in application code, preventing repeated news alerts from inflating downstream sentiment scores and avoiding expensive `SELECT` checks before every insert.

### market_date normalization
Raw article timestamps from NewsAPI are in UTC. A headline published at 23:00 EST cannot affect that day's closing price. The pipeline maps every timestamp to the trading session it can realistically affect:

- Published before 16:00 EST on a weekday: same trading day
- Published at or after 16:00 EST, or on a weekend: next weekday open

This normalization makes the (ticker, market_date) join between `sentiment_scores` and `price_data` statistically meaningful.

### Log returns over simple returns
Daily price returns are stored as log returns: 

$$\ln\left(\frac{\text{Close}_t}{\text{Close}_{t-1}}\right)$$

instead of simple returns: 

$$\frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$$

Log returns are additive across time periods and are more normally distributed. The Pearson correlation engine in Phase 2 relies directly on these properties.

### Decoupled ingestion and inference
The scheduler writes headlines with `NULL` sentiment fields. `scoring_job` (in `src/inference/model_runner.py`) reads unscored records via `get_unscored_headlines()` and writes labels back independently via `update_sentiment()`. Ingestion and scoring don't need to know about each other — `scoring_job` runs on its own interval and simply drains whatever backlog `news_job` has built up since its last pass.

---

## Architecture

```
[NewsAPI]  ──── every 4h ────┐
                              ├──► [src/scheduler.py] ──► [SHA-256 hash check] ──► [SQLite]
[yfinance] ──── every 1h ────┘                                                  (sentiment_pipeline.db)
                                                                                        │
                              [FinBERT scoring_job] ◄── pulls NULL sentiment rows ──────┤
                              (every 2h, src/inference/model_runner.py)                 │
                                                                                        │
                                                              [check_pipeline_health.py]┘
                                                              (observability — safe to run in parallel)
```

**Data flow:**

1. `scheduler.py` triggers `news_job`, `price_job`, and `scoring_job` on configurable intervals
2. Each collector fetches data and returns normalised records
3. `db_manager.py` applies `INSERT OR IGNORE` deduplication and writes to SQLite
4. `scoring_job` pulls rows with `sentiment_label IS NULL`, runs them through the fine-tuned FinBERT checkpoint (`training/finetuned_finbert/`), and writes the label, confidence, and attention-derived keyword back in place
5. Every job execution is logged to `job_log` with inserted/skipped counts
6. `check_pipeline_health.py` reads the database and prints a live health summary

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
| Config | python-dotenv (`.env` file, gitignored) |
| NLP model | `bert-base-uncased` fine-tuned on Financial PhraseBank — HuggingFace Transformers |

---

## Project Structure

```
finbert/
├── config/
│   ├── settings.py               # Centralised config: tickers, intervals, paths, API keys (loads .env)
│   └── logging_config.py         # Console + rotating file log handler setup
├── src/
│   ├── collectors/
│   │   ├── news_fetcher.py       # NewsAPI fetcher: market_date normalisation, backoff
│   │   └── market_data.py        # yfinance OHLCV fetcher: log return calculation
│   ├── inference/
│   │   └── model_runner.py       # FinBERT scoring: get_unscored_headlines() → inference → update_sentiment()
│   ├── storage/
│   │   └── db_manager.py         # Schema init, INSERT OR IGNORE, health queries
│   └── scheduler.py              # APScheduler entry point: news_job + price_job + scoring_job
├── training/
│   ├── prepare_data.py           # Downloads Financial PhraseBank, stratified 70/15/15 split
│   ├── train.py                  # Fine-tunes bert-base-uncased on the prepared splits
│   └── finetuned_finbert/        # Saved checkpoint used by model_runner.py (excluded from Git)
├── utils/
│   └── helpers.py                # seed_everything() — reproducibility across Python/NumPy/Torch/MPS
├── check_pipeline_health.py      # Observability script: DB metrics, job history, backlog
├── view_data.py                  # Quick DB table viewer for local inspection
├── data/
│   └── sentiment_pipeline.db     # SQLite database (excluded from Git via .gitignore)
├── logs/
│   └── pipeline.log              # Rotating execution log (excluded from Git)
├── .env                           # NEWSAPI_KEY (excluded from Git via .gitignore)
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
git clone https://github.com/Adam210503/finbert-sentiment-analytics.git
cd finbert-sentiment-analytics
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1 — Configure your API key

Create a `.env` file at the project root (already gitignored, so it stays local):

```
NEWSAPI_KEY=your_api_key_here
```

`config/settings.py` loads this automatically via `python-dotenv` on every run — no need to `export` it in every terminal session.

### Step 2 — Train the FinBERT model

`scoring_job` requires a fine-tuned checkpoint at `training/finetuned_finbert/`. This directory is gitignored, so it must be generated locally before the scheduler can score anything. Skip this step if you already have a checkpoint there.

```bash
python training/prepare_data.py   # downloads Financial PhraseBank, writes training/processed_dataset/
python training/train.py          # fine-tunes bert-base-uncased, writes training/finetuned_finbert/
```

`train.py` automatically uses Apple Silicon (MPS) if available, else CPU. Expect several minutes for 3 epochs over ~1,600 training rows. Both scripts are seeded (`seed_everything(42)`) for reproducible splits and model initialization.

### Step 3 — Start the pipeline

```bash
python src/scheduler.py
```

On macOS, wrap with `caffeinate` to prevent the system from sleeping and interrupting the scheduler, and `nohup`/`disown` to keep it running after you close the terminal:

```bash
nohup caffeinate -i python src/scheduler.py > /tmp/finbert_scheduler.log 2>&1 &
disown
```

> **Tip:** To keep your MacBook running with the display off, lower screen brightness to zero rather than closing the lid. Closing the lid triggers a hardware sleep signal that `caffeinate` cannot override.

All three jobs (`news_job`, `price_job`, `scoring_job`) run immediately on startup, then repeat on their configured intervals (`NEWS_INTERVAL_HOURS`, `PRICE_INTERVAL_HOURS`, `SCORING_INTERVAL_HOURS` in `config/settings.py`).

### Step 4 — Monitor the pipeline

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
  Scored             :  247
  Unscored backlog   :    0

Total price records  :   42

Last news ingestion  :  2026-05-14T09:30:02Z
Last price record    :  2026-05-14

Recent job history:
  scoring_job 09:32:10   inserted=31  skipped=0
  news_job    09:30:02   inserted=31  skipped=8
  price_job   09:30:05   inserted=6   skipped=0
  news_job    05:30:01   inserted=19  skipped=4
────────────────────────────────────────
```

### Step 5 — Inspect the database directly

```bash
python view_data.py
```

---

## Database Schema

### `sentiment_scores`

One row per unique headline. Sentiment fields are `NULL` until `scoring_job` runs.

| Column | Type | Description |
|:---|:---|:---|
| `id` | INTEGER PK | Auto-increment |
| `ticker` | TEXT | AAPL / TSLA / SPY |
| `headline` | TEXT | Raw headline text |
| `source` | TEXT | Publishing outlet (e.g. reuters.com) |
| `raw_timestamp` | TEXT | Original UTC publish time (ISO 8601) |
| `market_date` | TEXT | Normalised trading date (YYYY-MM-DD) |
| `sentiment_label` | TEXT | positive / neutral / negative — filled by `scoring_job` |
| `confidence` | REAL | Softmax probability of predicted class — filled by `scoring_job` |
| `attention_keyword` | TEXT | Token most attended to by [CLS] in the last layer — filled by `scoring_job` |
| `model_version` | TEXT | Checkpoint directory name for reproducibility |
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
| `job_name` | TEXT | news_job / price_job / scoring_job |
| `ran_at` | TEXT | UTC execution timestamp |
| `inserted` | INTEGER | New records written to DB |
| `skipped` | INTEGER | Duplicates ignored |
| `error` | TEXT | Exception message if job failed, else NULL |

---

## Limitations

- **Free-tier API constraints.** NewsAPI allows 100 requests/day on the free tier. The current configuration uses 18 requests/day (3 tickers × 2 fetches/day), leaving comfortable headroom. yfinance is an unofficial wrapper with no SLA.
- **NYSE holidays not handled.** The `market_date` normalisation advances past weekends but does not account for public holidays. Headlines published on a NYSE holiday are assigned to the next calendar weekday, which may itself be a holiday.
- **Apple Silicon (MPS) inference only tested locally.** `model_runner.py` automatically uses MPS if available, falling back to CPU otherwise. Not yet benchmarked on CUDA.
- **`attention_keyword` is a coarse heuristic.** It's the token most attended to by `[CLS]` in the last transformer layer, which often surfaces function words (e.g. "has", "as") rather than substantive keywords. Useful as a debugging signal, not a rigorous attribution method.
- **SQLite concurrency.** SQLite is single-writer. The scheduler and the health check script should not both attempt writes simultaneously. In the current architecture this is not an issue — the health check is read-only.

---

## What's Next

- Implement rolling Pearson correlation between daily mean sentiment and next-day log return
- Build event study: return distributions at t+1h, t+4h, t+24h following sentiment spike events
- Streamlit dashboard with analytics panel and live observability metrics
- Improve `attention_keyword` extraction (e.g. attention rollout across layers, stopword filtering)

---

## Author

**Adam Mikail**
[LinkedIn](https://www.linkedin.com/in/adammikail/) · [Email](mailto:adammikail2105@gmail.com)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
