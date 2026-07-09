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
| Phase 2: NLP | FinBERT fine-tuning, evaluation, live sentiment scoring | ✅ Complete |
| Phase 2: Analytics | Rolling correlation, spike event study | ✅ Complete |
| Phase 3: Dashboard | Streamlit analytics + observability panel | 🔧 In progress |
| Phase 4: Deployment | Docker Compose containerisation | ⏳ Pending |

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

| Parameter | Value | Rationale |
|:---|:---|:---|
| Base model | `ProsusAI/finbert` | Pre-trained on 4.9B financial tokens (Reuters, Bloomberg, SEC filings) |
| Learning rate | 2e-5 | BERT paper recommendation; low enough to avoid catastrophic forgetting |
| Batch size | 16 | Memory-efficient on MPS |
| Epochs | 3 | BERT paper recommends 2–4; early stopping guards against overfitting |
| Weight decay | 0.01 | L2 regularisation on non-bias/LayerNorm params |
| Max sequence length | 128 | Headlines average 12–15 tokens; avoids 16× memory cost of max_length=512 |
| Optimiser | AdamW | Decoupled weight decay; standard for transformer fine-tuning |
| Best model metric | Macro F1 | Weights all classes equally; accuracy is misleading on imbalanced data |
| Device | MPS (Apple Silicon) → CPU fallback | |
| Seed | 42 | Applied across Python, NumPy, PyTorch CPU, MPS |

### Evaluation results

Evaluated on the held-out test split (15% of combined corpus, ~907 samples). Both models tokenised with `AutoTokenizer.from_pretrained("ProsusAI/finbert")`, max_length=128.

> **Note:** The base model's native label order (`positive=0, negative=1, neutral=2`) differs from the project's canonical scheme. A label remap `{0→2, 1→0, 2→1}` is applied to base model predictions at inference time.

| Model | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 |
|:---|:---|:---|:---|:---|:---|
| ProsusAI/finbert (base) | 0.8218 | 0.8168 | 0.8063 | 0.8382 | 0.8059 |
| **Fine-tuned** (ours) | **0.9138** | **0.9021** | **0.8488** | **0.9457** | **0.9118** |
| Delta | +0.0920 | **+0.0853** | +0.0425 | +0.1075 | +0.1059 |

The fine-tuned model outperforms the base across every metric. The largest gain is on the neutral class (+10.75 F1), which is typically the hardest to classify in financial text due to its ambiguity.

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
`ProsusAI/finbert`'s native output order (`positive=0, negative=1, neutral=2`) differs from the project's canonical scheme (`negative=0, neutral=1, positive=2`). A remap `{0→2, 1→0, 2→1}` is applied inside `model_runner.py` when using the base model, so all database writes always use the canonical scheme regardless of which checkpoint is loaded.

---

## Architecture

```
[NewsAPI]  ──── every 4h ────┐
                              ├──► [src/scheduler.py] ──► [SHA-256 dedup] ──► [SQLite]
[yfinance] ──── every 1h ────┘                                           (sentiment_pipeline.db)
                                                                                 │
                    [scoring_job] ◄── NULL sentiment rows ───────────────────────┤
                    (every 2h, src/inference/model_runner.py)                    │
                                                                                 │
                    [src/analysis/correlation.py] ─── correlations table ────────┤
                    [src/analysis/event_study.py] ─── spike_events table ─────────┘
```

**Data flow:**

1. `scheduler.py` triggers `news_job`, `price_job`, and `scoring_job` on configurable intervals
2. Each collector fetches data and returns normalised records
3. `db_manager.py` applies `INSERT OR IGNORE` deduplication and writes to SQLite
4. `scoring_job` pulls rows with `sentiment_label IS NULL`, runs them through the fine-tuned FinBERT checkpoint, and writes label, confidence, and attention keyword back in place
5. `correlation.py` aggregates daily mean sentiment, joins with price data, and computes 7d/30d rolling Pearson correlation
6. `event_study.py` identifies sentiment spike events and measures t+1d/t+2d/t+5d log returns

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
| Training data | Financial PhraseBank + FiQA 2018 (~6,050 samples) |
| Analysis | pandas · scipy · statsmodels |
| Dashboard | Streamlit + Plotly |
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
│   └── scheduler.py                  # APScheduler entry point: news + price + scoring jobs
├── training/
│   ├── prepare_data.py               # PhraseBank + FiQA 2018 → stratified 70/15/15 split
│   ├── train.py                      # Fine-tunes ProsusAI/finbert, saves checkpoint
│   ├── evaluate.py                   # Base vs fine-tuned comparison on held-out test set
│   ├── evaluation_results.csv        # Exported evaluation metrics
│   ├── confusion_base.png            # Confusion matrix — base model
│   ├── confusion_finetuned.png       # Confusion matrix — fine-tuned model
│   └── finetuned_finbert/            # Saved checkpoint (gitignored — generate locally)
├── utils/
│   └── helpers.py                    # seed_everything() for reproducibility
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

`train.py` uses Apple Silicon (MPS) automatically if available, else CPU. Expect 10–20 minutes on MPS for the full ~4,200-row training split. Both scripts are seeded (`seed_everything(42)`) for reproducible splits and weight initialisation.

### Step 3 — Start the pipeline

```bash
caffeinate -i python src/scheduler.py
```

All three jobs (`news_job`, `price_job`, `scoring_job`) fire immediately on startup, then repeat on their configured intervals.

To keep it running after closing the terminal:

```bash
nohup caffeinate -i python src/scheduler.py > /tmp/finbert.log 2>&1 &
disown
```

### Step 4 — Run analytics

```bash
python src/analysis/correlation.py   # computes rolling Pearson correlation
python src/analysis/event_study.py   # identifies spike events and forward returns
```

### Step 5 — Monitor

```bash
python check_pipeline_health.py   # read-only, safe to run while scheduler is live
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
| `model_version` | TEXT | Checkpoint identifier for reproducibility |
| `headline_hash` | TEXT UNIQUE | SHA-256 hash for deduplication |

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
- **NYSE holidays not handled.** `market_date` normalisation advances past weekends but not public holidays.
- **MPS only tested locally.** Inference falls back to CPU on non-Apple hardware; not benchmarked on CUDA.
- **Attention keyword is a heuristic.** Not a validated attribution method.

---

## What's Next

- [ ] Streamlit dashboard — analytics panel (sentiment feed, correlation charts, event study, keyword table) + observability panel (API health, data freshness, error log tail)
- [ ] Docker Compose — containerise backend scheduler and Streamlit frontend as separate services

---

## Author

**Adam Mikail** · ML Engineering Portfolio · Project 2 of 4
[LinkedIn](https://www.linkedin.com/in/adammikail/) · [Email](mailto:adammikail2105@gmail.com)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
