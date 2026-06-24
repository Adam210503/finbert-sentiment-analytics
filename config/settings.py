"""
config/settings.py
──────────────────
Single source of truth for all project configuration.
Values are read from environment variables where possible so
no credentials are ever hardcoded.

Set in a .env file at the project root:
    NEWSAPI_KEY=your_key_here
"""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# ── Project root ─────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

load_dotenv(ROOT_DIR / ".env")

# ── Target assets ────────────────────────────────────────────────
TICKERS: list[str] = ["AAPL", "TSLA", "SPY"]

# Human-readable search queries for NewsAPI.
# Broader than the ticker symbol alone to capture more coverage.
TICKER_QUERIES: dict[str, str] = {
    "AAPL": "Apple stock AAPL earnings",
    "TSLA": "Tesla stock TSLA Elon Musk",
    "SPY":  "S&P 500 SPY stock market index",
}

# ── API credentials ──────────────────────────────────────────────
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")

# ── Database ─────────────────────────────────────────────────────
DB_PATH: Path = ROOT_DIR / "data" / "sentiment_pipeline.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────
LOG_DIR: Path  = ROOT_DIR / "logs"
LOG_FILE: Path = LOG_DIR / "pipeline.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Market hours (NYSE, America/New_York) ─────────────────────────
MARKET_TZ          = ZoneInfo("America/New_York")
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MIN    = 30
MARKET_CLOSE_HOUR  = 16
MARKET_CLOSE_MIN   = 0

# ── Scheduler intervals ──────────────────────────────────────────
# News: every 4 hours (free NewsAPI tier = 100 req/day;
#   3 tickers × 6 runs/day = 18 req/day — well within limit)
NEWS_INTERVAL_HOURS: int  = 4

# Prices: every hour, but the job checks market hours internally
PRICE_INTERVAL_HOURS: int = 1

# ── NewsAPI fetch config ─────────────────────────────────────────
NEWS_PAGE_SIZE: int = 20    # articles per ticker per request (max 100)
NEWS_DAYS_BACK: int = 7     # rolling window to look back for articles

# ── Rate-limit / retry ───────────────────────────────────────────
# Delay in seconds between retry attempts after HTTP 429
BACKOFF_DELAYS: list[int] = [2, 4, 8]
MAX_RETRIES: int           = 3

# ── Inference (Phase 2) ───────────────────────────────────────────
MODEL_PATH: Path          = ROOT_DIR / "training" / "finetuned_finbert"
SCORING_BATCH_SIZE: int   = 200    # headlines pulled from DB per scoring run
SCORING_INTERVAL_HOURS: int = 2
