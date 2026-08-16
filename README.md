# FinBERT: Automated Financial Sentiment Pipeline

An automated data engineering pipeline that collects financial news headlines and equity price data on scheduled intervals, deduplicates and warehouses them in a local SQLite store, fine-tunes ProsusAI/finbert on a combined Financial PhraseBank + FiQA 2018 corpus, and scores live headlines with the resulting checkpoint — correlating daily sentiment signals against real log returns for AAPL, TSLA, and SPY.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![APScheduler](https://img.shields.io/badge/APScheduler-3.10+-FF6B35?style=flat)](https://apscheduler.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

| Phase | Scope | Status |
|:---|:---|:---|
| Phase 1: Data pipeline | Ingestion, deduplication, SQLite persistence, observability | ✅ Complete |
| Phase 2: NLP | FinBERT fine-tuning (Optuna HPO), evaluation, live sentiment scoring | ✅ Complete |
| Phase 2: Analytics | Rolling correlation, spike event study | ✅ Complete |
| Phase 3: Dashboard | FastAPI REST backend + React + Vite frontend | ✅ Complete |
| Phase 4: Deployment | Docker Compose containerisation | ✅ Complete |

---

## Model Card

### Training data

| Dataset | Source | Samples | Notes |
|:---|:---|:---|:---|
| Financial PhraseBank | Raw HTTP (Sentences_AllAgree.txt) | 4,840 | Expert-annotated financial news; all-agree split for highest label confidence |
| FiQA 2018 | `pauri32/fiqa-2018` (HuggingFace) | 1,213 | Financial QA with sentiment; broader language register than news-only |
| **Combined** | | **~6,050** | Shuffled, stratified 70/15/15 split |

**Label scheme:** `0 = negative`, `1 = neutral`, `2 = positive`

FiQA's native label order (`0=positive, 1=neutral, 2=negative`) is remapped to the canonical scheme during data preparation.

### Hyperparameters

Hyperparameters were selected via **Optuna** (30 trials, TPE sampler, MedianPruner). The best trial (#22) was used for the final retrain on train+validation combined.

| Parameter | Value | Source |
|:---|:---|:---|
| Base model | `ProsusAI/finbert` | Pre-trained on 4.9B financial tokens (Reuters, Bloomberg, SEC filings) |
| Learning rate | 4.49e-05 | Optuna best trial |
| Batch size | 16 | Optuna best trial |
| Epochs | 3 | Optuna best trial |
| Weight decay | 0.0836 | Optuna best trial |
| Warmup ratio | 0.176 | Optuna best trial |
| Max sequence length | 128 | Headlines average 12–15 tokens; avoids 16× memory cost of max_length=512 |
| Optimiser | AdamW | Decoupled weight decay; standard for transformer fine-tuning |
| Best model metric | Macro F1 | Weights all classes equally; accuracy is misleading on imbalanced data |
| Device | MPS (Apple Silicon) → CPU fallback | |
| Seed | 42 | Applied across Python, NumPy, PyTorch CPU/MPS, HuggingFace Transformers |

Full Optuna trial log: [`training/optuna_results.csv`](training/optuna_results.csv)

### Evaluation results

Evaluated on the held-out test split (15% of combined corpus, ~907 samples). Both models tokenised with `AutoTokenizer.from_pretrained("ProsusAI/finbert")`, max_length=128.

> **Note:** The base model's native label order (`positive=0, negative=1, neutral=2`) differs from the project's canonical scheme. A label remap `{0→2, 1→0, 2→1}` is applied to base model predictions at inference time.

| Model | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 |
|:---|:---|:---|:---|:---|:---|
| ProsusAI/finbert (base) | 0.8218 | 0.8168 | 0.8063 | 0.8382 | 0.8059 |
| **Fine-tuned** (ours, Optuna) | **0.9291** | **0.9201** | **0.8792** | **0.9569** | **0.9242** |
| Delta | +0.1073 | **+0.1033** | +0.0729 | +0.1187 | +0.1183 |

The fine-tuned model outperforms the base across every metric. The largest gains are on the neutral (+11.87 F1) and positive (+11.83 F1) classes. Negative remains the hardest class for both models — negative financial language tends to be more nuanced than clearly positive signals.

Confusion matrices: [`training/confusion_base.png`](training/confusion_base.png) · [`training/confusion_finetuned.png`](training/confusion_finetuned.png)
Full results CSV: [`training/evaluation_results.csv`](training/evaluation_results.csv)

### Limitations

- **Attention keyword is a heuristic.** The attention-derived keyword (`attention_keyword`) is the token most attended to by `[CLS]` in the final encoder layer. This frequently surfaces function words rather than substantive terms. Useful as a debugging signal, not rigorous attribution.
- **Small corpus.** ~6,050 samples is modest for transformer fine-tuning. Performance on out-of-domain financial text (e.g. earnings call transcripts, analyst reports) may degrade.
- **Label noise in FiQA.** FiQA 2018 covers financial question-answering, a slightly different domain from news headlines. Some label boundary cases may be inconsistent across datasets.

---

## Key Engineering Decisions

### SHA-256 cryptographic deduplication
Every headline is hashed before insertion. The database enforces a `UNIQUE` constraint on `headline_hash`, and all inserts use `INSERT OR IGNORE`. Duplicates are silently skipped at the database level, preventing repeated news alerts from inflating downstream sentiment scores.

### market_date normalisation
Raw article timestamps from NewsAPI are in UTC. A headline published at 23:00 EST cannot affect that day's closing price. The pipeline maps every timestamp to the trading session it can realistically affect:

- Published before 16:00 EST on a weekday → same trading day
- Published at or after 16:00 EST, or on a weekend → next weekday open

This normalisation makes the `(ticker, market_date)` join between `sentiment_scores` and `price_data` statistically meaningful.

### Log returns over simple returns
Daily price returns are stored as log returns:

$$\ln\left(\frac{\text{Close}_t}{\text{Close}_{t-1}}\right)$$

Log returns are additive across time periods and more normally distributed than simple returns. The Pearson correlation engine relies on these properties.

### Decoupled ingestion and inference
The scheduler writes headlines with `NULL` sentiment fields. `scoring_job` reads unscored records via `get_unscored_headlines()` and writes labels back independently on its own 2h interval. Ingestion and scoring have no runtime dependency on each other.

### Label remap at inference time
`ProsusAI/finbert`'s native output order (`positive=0, negative=1, neutral=2`) differs from the project's canonical scheme (`negative=0, neutral=1, positive=2`). The fine-tuned checkpoint inherits this native ordering (confirmed via `config.json`). A remap `{0→2, 1→0, 2→1}` is applied inside `model_runner.py` so all database writes always use the canonical scheme regardless of which checkpoint is loaded.

---

## Architecture

```
[NewsAPI]  ──── every 4h ────┐
                              ├──► [src/scheduler.py] ──► [SHA-256 dedup] ──► [SQLite]
[yfinance] ──── every 1h ────┘         │                               (sentiment_pipeline.db)
                                        │                                        │
                              ┌─────────┴──────────────────────────────────────┐│
                              │  news_job     (every 4h)  → sentiment_scores   ││
                              │  price_job    (every 1h)  → price_data         ││
                              │  scoring_job  (every 2h)  → sentiment labels   ││
                              │  analytics_job(every 6h)  → correlations       ││
                              │                           → spike_events        ││
                              └─────────────────────────────────────────────────┘│
                                                                                  │
                    [dashboard/api/main.py] ◄──────── FastAPI REST ───────────────┘
                           │
                    [dashboard/frontend/] ◄── React + Vite + Recharts (port 5173)
```

**Data flow:**

1. `scheduler.py` triggers all four jobs on configurable intervals; each fires immediately on startup
2. `news_job` fetches headlines from NewsAPI, `db_manager.py` applies `INSERT OR IGNORE` deduplication
3. `price_job` fetches OHLCV from yfinance (no-op outside NYSE hours); log returns computed inline
4. `scoring_job` pulls rows with `sentiment_label IS NULL`, runs FinBERT inference, writes label/confidence/attention keyword back in place
5. `analytics_job` runs `correlation.py` (7d/30d rolling Pearson) then `event_study.py` (spike events + forward returns) in sequence — always on fresh data
6. FastAPI backend serves all tables over a read-only REST API; React frontend polls it and renders live charts

---

## Tech Stack

| Category | Technology |
|:---|:---|
| Language | Python 3.11+ |
| Scheduler | APScheduler 3.10+ |
| Database | SQLite3 (standard library) |
| News data | NewsAPI v2 `/everything` endpoint |
| Price data | yfinance (OHLCV, auto-adjusted) |
| Deduplication | hashlib SHA-256 (standard library) |
| Base model | ProsusAI/finbert (pre-trained on 4.9B financial tokens) |
| Fine-tuning | HuggingFace Transformers · Trainer API · AdamW |
| HPO | Optuna (TPE sampler, MedianPruner, 30 trials) |
| Training data | Financial PhraseBank + FiQA 2018 (~6,050 samples) |
| Analysis | pandas · scipy · statsmodels |
| API backend | FastAPI · uvicorn |
| Frontend | React 18 · Vite · Recharts · Tailwind CSS |
| Infrastructure | Docker Compose (backend + dashboard services) |
| Config | python-dotenv (`.env` file, gitignored) |

---

## Project Structure

```
finbert/
├── config/
│   ├── settings.py                   # Centralised config: tickers, intervals, paths, API keys
│   └── logging_config.py             # Console + rotating file log handler
├── src/
│   ├── collectors/
│   │   ├── news_fetcher.py           # NewsAPI fetcher: market_date normalisation, backoff
│   │   └── market_data.py            # yfinance OHLCV fetcher: log return calculation
│   ├── inference/
│   │   └── model_runner.py           # FinBERT scoring: batch inference → update_sentiment()
│   ├── analysis/
│   │   ├── correlation.py            # 7d/30d rolling Pearson correlation → correlations table
│   │   └── event_study.py            # Spike event detection → spike_events table
│   ├── storage/
│   │   └── db_manager.py             # Schema init, INSERT OR IGNORE, health queries
│   └── scheduler.py                  # APScheduler entry point: news, price, scoring, analytics jobs
├── training/
│   ├── prepare_data.py               # PhraseBank + FiQA 2018 → stratified 70/15/15 split
│   ├── train.py                      # Fine-tunes ProsusAI/finbert, saves checkpoint
│   ├── tune.py                       # Optuna HPO: 30 trials → best params → final retrain
│   ├── evaluate.py                   # Base vs fine-tuned comparison on held-out test set
│   ├── evaluation_results.csv        # Exported evaluation metrics
│   ├── optuna_results.csv            # Per-trial hyperparameters and val macro F1
│   ├── confusion_base.png            # Confusion matrix — base model
│   ├── confusion_finetuned.png       # Confusion matrix — fine-tuned model
│   └── finetuned_finbert/            # Saved checkpoint (gitignored — generate locally)
├── utils/
│   └── helpers.py                    # seed_everything() for reproducibility
├── dashboard/
│   ├── api/
│   │   └── main.py                   # FastAPI REST API: /health /sentiment /correlation /events /keywords /flow /prices
│   └── frontend/
│       ├── src/
│       │   ├── components/           # TopBar, ClockStrip, TickerNav, SentimentFeed, CorrelationChart, HealthPanel, …
│       │   ├── hooks/usePolling.js   # Polling hook with configurable interval
│       │   ├── api.js                # Typed fetch wrappers for each REST endpoint
│       │   ├── constants.js          # Design token palette
│       │   └── App.jsx               # Root: ticker state, price polling, layout
│       ├── index.html
│       ├── vite.config.js
│       └── package.json
├── check_pipeline_health.py          # Observability: DB metrics, job history, backlog
├── view_data.py                      # Quick DB table viewer
├── data/
│   └── sentiment_pipeline.db         # SQLite database (gitignored)
├── logs/
│   └── pipeline.log                  # Rotating execution log (gitignored)
├── .env                              # NEWSAPI_KEY (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Runnable Scripts

| Script | Command | Output |
|:---|:---|:---|
| `training/prepare_data.py` | `python training/prepare_data.py` | Downloads PhraseBank + FiQA 2018, merges ~6,050 samples, stratified 70/15/15 split. Saves `training/processed_dataset/`. |
| `training/train.py` | `python training/train.py` | Fine-tunes ProsusAI/finbert with fixed hyperparameters. Saves checkpoint to `training/finetuned_finbert/`. |
| `training/tune.py` | `python training/tune.py` | Optuna HPO: 30 trials (TPE sampler, MedianPruner), then retrains on train+val with best params, then calls `evaluate.py`. Saves `optuna_results.csv`. |
| `training/evaluate.py` | `python training/evaluate.py` | Base vs fine-tuned comparison on held-out test set. Prints accuracy, macro F1, per-class F1. Saves `evaluation_results.csv` and confusion matrix PNGs. |
| `src/scheduler.py` | `caffeinate -i python src/scheduler.py` | Starts live pipeline. Fires all four jobs immediately then every 4h / 1h / 2h / 6h (news / price / scoring / analytics). Logs to `logs/pipeline.log`. Runs indefinitely. |
| `src/inference/model_runner.py` | `python src/inference/model_runner.py` | Manually scores all unscored headlines once. Writes label, confidence, attention keyword to DB. |
| `src/analysis/correlation.py` | `python src/analysis/correlation.py` | 7d/30d rolling Pearson correlation between daily sentiment and next-day log return. Writes to `correlations` table. Prints latest r values per ticker. |
| `src/analysis/event_study.py` | `python src/analysis/event_study.py` | Detects sentiment spikes (≥0.65 / ≤−0.65), measures t+1d/t+2d/t+5d forward returns. Writes to `spike_events` table. Prints average returns per ticker per spike type. |
| `check_pipeline_health.py` | `python check_pipeline_health.py` | Read-only snapshot: headline counts, scored/unscored, price records, last fetch times, recent job history. Safe to run while scheduler is live. |
| `view_data.py` | `python view_data.py` | Prints last 10 headlines and last 5 job log entries. |

**Shell helper (add to `~/.zshrc`):**

```bash
finbert_results              # last 20 scored headlines, all tickers
finbert_results 50           # last 50
finbert_results 20 TSLA      # last 20 TSLA headlines
finbert_results 20 TSLA positive   # last 20 positive TSLA headlines
```

---

## Docker deployment

The full system runs as three containers orchestrated by Docker Compose.

### Prerequisites

- Docker Desktop installed and running
- The fine-tuned model checkpoint at `training/finetuned_finbert/` (generate locally first with `python training/tune.py`)
- A `.env` file at the project root containing `NEWSAPI_KEY=your_key_here`

### Build and run

```bash
# 1. Build all images
docker compose build

# 2. Seed the model checkpoint into the named volume
docker run --rm \
  -v $(pwd)/training/finetuned_finbert:/source:ro \
  -v finbert_finbert-model:/dest \
  alpine sh -c "cp -r /source/. /dest/"

# 3. (Optional) Seed existing database
docker run --rm \
  -v $(pwd)/data/sentiment_pipeline.db:/source/sentiment_pipeline.db:ro \
  -v finbert_finbert-data:/dest \
  alpine sh -c "cp /source/sentiment_pipeline.db /dest/"

# 4. Start all services
docker compose up -d

# 5. Check all containers are healthy
docker compose ps
```

Dashboard is available at `https://finbert-sentinel.vercel.app/`. FastAPI docs at `http://localhost:8000/docs`.

### Logs

```bash
docker compose logs -f backend   # scheduler + scoring
docker compose logs -f api       # FastAPI request logs
docker compose logs -f frontend  # nginx access logs
```

### Stop

```bash
docker compose down              # stop containers, keep volumes
docker compose down -v           # stop containers and delete all data
```

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- A free [NewsAPI key](https://newsapi.org/register) (100 requests/day on the free tier)
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

```
NEWSAPI_KEY=your_api_key_here
```

Create this as a `.env` file at the project root (already gitignored). `config/settings.py` loads it automatically.

### Step 2 — Build the training dataset and fine-tune

```bash
python training/prepare_data.py   # downloads PhraseBank + FiQA 2018, writes processed_dataset/
python training/train.py          # fine-tunes ProsusAI/finbert, writes finetuned_finbert/
python training/evaluate.py       # compares base vs fine-tuned on held-out test set
```

`train.py` uses Apple Silicon (MPS) automatically if available, else CPU. Expect 10–20 minutes on MPS. Both scripts are seeded (`seed_everything(42)`) for reproducible splits and weight initialisation.

### Step 3 — Start the pipeline

```bash
caffeinate -i python src/scheduler.py
```

All four jobs fire immediately on startup, then repeat on their configured intervals:

| Job | Interval | What it does |
|:---|:---|:---|
| `news_job` | every 4h | Fetches headlines from NewsAPI |
| `price_job` | every 1h | Fetches OHLCV from yfinance (market hours only) |
| `scoring_job` | every 2h | Runs FinBERT on unscored headlines |
| `analytics_job` | every 6h | Recomputes rolling correlations then spike events |

To keep running after closing the terminal:

```bash
nohup caffeinate -i python src/scheduler.py > /tmp/finbert.log 2>&1 &
disown
```

### Step 4 — Start the dashboard

```bash
# Terminal 1 — FastAPI backend
uvicorn dashboard.api.main:app --reload --port 8000

# Terminal 2 — React frontend
cd dashboard/frontend && npm install && npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

### Step 5 — Monitor

```bash
python check_pipeline_health.py      # DB health snapshot
tail -f logs/pipeline.log            # live log stream
```

---

## Database Schema

### `sentiment_scores`

| Column | Type | Description |
|:---|:---|:---|
| `id` | INTEGER PK | Auto-increment |
| `ticker` | TEXT | AAPL / TSLA / SPY |
| `headline` | TEXT | Raw headline text |
| `source` | TEXT | Publishing outlet |
| `raw_timestamp` | TEXT | UTC publish time (ISO 8601) |
| `market_date` | TEXT | Normalised trading date (YYYY-MM-DD) |
| `sentiment_label` | TEXT | positive / neutral / negative |
| `confidence` | REAL | Softmax probability of predicted class |
| `attention_keyword` | TEXT | Token most attended to by [CLS] in last layer |
| `model_version` | TEXT | Checkpoint identifier (`ProsusAI/finbert-base` or `finetuned_finbert-optuna`) |
| `headline_hash` | TEXT UNIQUE | SHA-256 hash for deduplication |
| `url` | TEXT | Article URL from NewsAPI (populated for rows collected after migration) |

### `price_data`

| Column | Type | Description |
|:---|:---|:---|
| `ticker` | TEXT | AAPL / TSLA / SPY |
| `market_date` | TEXT | Trading date (YYYY-MM-DD) |
| `open` / `high` / `low` / `close` | REAL | OHLC prices |
| `volume` | INTEGER | Volume traded |
| `daily_return` | REAL | Log return: ln(close / prev_close) |

### `correlations`

| Column | Type | Description |
|:---|:---|:---|
| `ticker` | TEXT | |
| `market_date` | TEXT | |
| `mean_sentiment` | REAL | Confidence-weighted signed daily mean |
| `log_return` | REAL | Next-day log return (shifted −1) |
| `pearson_7d` | REAL | 7-day rolling Pearson r |
| `pearson_30d` | REAL | 30-day rolling Pearson r |

### `spike_events`

| Column | Type | Description |
|:---|:---|:---|
| `ticker` | TEXT | |
| `event_date` | TEXT | |
| `event_type` | TEXT | positive / negative |
| `mean_sentiment` | REAL | Score on the event day |
| `return_t1d` | REAL | Log return 1 trading day after |
| `return_t2d` | REAL | Log return 2 trading days after |
| `return_t5d` | REAL | Log return 5 trading days after |

---

## Known Constraints

- **Free-tier APIs.** NewsAPI: 100 req/day (current usage: ~18/day). yfinance: unofficial, no SLA.
- **Data volume.** ~6.5 weeks of live data at current stage. Rolling correlation and event study results should be interpreted with caution — 30-day windows will have limited coverage, and spike event counts are small.
- **NYSE holidays not handled.** The `market_date` normalisation rule in `src/collectors/news_fetcher.py` advances timestamps past weekends (Saturday and Sunday) but does not account for NYSE public holidays. A headline published on a holiday — for example, July 4th, Thanksgiving, or Christmas — is mapped to that calendar date rather than the next trading session open. This means sentiment scores for those dates are paired with the wrong price record in `correlation.py` and `event_study.py`, or not paired at all if no price record exists for that date. The affected dates in the US equity calendar are: New Year's Day, Martin Luther King Jr. Day, Presidents Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labour Day, Thanksgiving, and Christmas. A fix would involve integrating a trading calendar library such as `pandas_market_calendars` or `exchange_calendars` and replacing the weekend-only check with a full holiday-aware `next_trading_day()` function. This is deferred as a known limitation.
- **MPS only tested locally.** Inference falls back to CPU on non-Apple hardware; not benchmarked on CUDA.
- **Attention keyword is a heuristic.** Not a validated attribution method.

---

## What's Next

- [x] Docker Compose — containerise scheduler, FastAPI backend, and React frontend as separate services for one-command deployment

---

## Author

**Adam Mikail** · ML Engineering Portfolio · Project 2 of 4
[LinkedIn](https://www.linkedin.com/in/adammikail/) · [Email](mailto:adammikail2105@gmail.com)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
