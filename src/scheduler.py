"""
src/scheduler.py
────────────────
Main entry point for the backend pipeline.

Runs two APScheduler jobs on separate intervals:
  news_job   — fetches NewsAPI headlines every NEWS_INTERVAL_HOURS hours.
               Always runs regardless of market hours.
  price_job  — fetches yfinance OHLCV every PRICE_INTERVAL_HOURS hours.
               The collector itself checks is_market_hours() and no-ops
               if called outside trading hours, so it is safe to schedule
               at any interval.

Both jobs write to the same SQLite database via DatabaseManager.
Neither job runs inference — the inference layer (model_runner.py,
built in Phase 2) reads from the database independently.

Usage:
    python src/scheduler.py

Docker:
    CMD ["python", "src/scheduler.py"]
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ── Path setup: allow imports from project root ──────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.logging_config import setup_logging
from config.settings import (
    DB_PATH,
    LOG_FILE,
    NEWS_INTERVAL_HOURS,
    PRICE_INTERVAL_HOURS,
)
from src.collectors.market_data import fetch_all_prices
from src.collectors.news_fetcher import fetch_all_tickers
from src.storage.db_manager import DatabaseManager

setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────────────────────────

def news_job(db: DatabaseManager) -> None:
    """
    Fetch headlines from NewsAPI and persist to sentiment_scores.

    New records are inserted with NULL sentiment fields — the inference
    layer will pick them up and score them asynchronously.
    """
    ran_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    logger.info("── news_job starting (%s) ──", ran_at)

    try:
        records          = fetch_all_tickers()
        inserted, skipped = db.insert_headlines(records)
        db.log_job("news_job", ran_at, inserted, skipped)
        logger.info(
            "news_job done — %d inserted, %d skipped", inserted, skipped
        )
    except Exception as exc:
        logger.exception("news_job failed: %s", exc)
        db.log_job("news_job", ran_at, 0, 0, error=str(exc))


def price_job(db: DatabaseManager) -> None:
    """
    Fetch OHLCV data from yfinance and persist to price_data.

    The collector's is_market_hours() guard means this is a no-op when
    called outside trading hours. Log returns are computed inside the
    collector and stored alongside the OHLCV bars.
    """
    ran_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    logger.info("── price_job starting (%s) ──", ran_at)

    try:
        records           = fetch_all_prices()
        inserted, skipped = db.insert_price_data(records)
        db.log_job("price_job", ran_at, inserted, skipped)
        logger.info(
            "price_job done — %d inserted, %d skipped", inserted, skipped
        )
    except Exception as exc:
        logger.exception("price_job failed: %s", exc)
        db.log_job("price_job", ran_at, 0, 0, error=str(exc))


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Sentiment pipeline starting")
    logger.info("Database : %s", DB_PATH)
    logger.info("News job : every %dh", NEWS_INTERVAL_HOURS)
    logger.info("Price job: every %dh (market hours only)", PRICE_INTERVAL_HOURS)
    logger.info("=" * 60)

    db        = DatabaseManager(DB_PATH)
    scheduler = BlockingScheduler(timezone="UTC")

    # Run both jobs immediately on startup, then on the interval
    scheduler.add_job(
        news_job,
        trigger    = IntervalTrigger(hours=NEWS_INTERVAL_HOURS),
        args       = [db],
        id         = "news_job",
        name       = "NewsAPI headline collector",
        next_run_time = datetime.utcnow(),     # run immediately
    )

    scheduler.add_job(
        price_job,
        trigger    = IntervalTrigger(hours=PRICE_INTERVAL_HOURS),
        args       = [db],
        id         = "price_job",
        name       = "yfinance OHLCV collector",
        next_run_time = datetime.utcnow(),     # run immediately
    )

    logger.info("Scheduler started. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
