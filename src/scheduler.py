"""
src/scheduler.py
────────────────
Main entry point for the backend pipeline.

Runs four APScheduler jobs on separate intervals:
  news_job      — fetches NewsAPI headlines every NEWS_INTERVAL_HOURS hours.
                  Always runs regardless of market hours.
  price_job     — fetches yfinance OHLCV every PRICE_INTERVAL_HOURS hours.
                  The collector itself checks is_market_hours() and no-ops
                  if called outside trading hours, so it is safe to schedule
                  at any interval.
  scoring_job   — scores any pending headlines with FinBERT every
                  SCORING_INTERVAL_HOURS hours, via model_runner.py.
  analytics_job — recomputes rolling Pearson correlations (7d/30d) then
                  spike events every ANALYTICS_INTERVAL_HOURS hours.
                  Runs correlation.py first so event_study.py always
                  operates on fresh correlation data.

All jobs write to the same SQLite database via DatabaseManager.

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
    ANALYTICS_INTERVAL_HOURS,
    DB_PATH,
    LOG_FILE,
    NEWS_INTERVAL_HOURS,
    PRICE_INTERVAL_HOURS,
    SCORING_INTERVAL_HOURS,
)
from src.analysis.correlation import run_correlation_analysis
from src.analysis.event_study import run_event_study
from src.collectors.market_data import fetch_all_prices
from src.collectors.news_fetcher import fetch_all_tickers
from src.inference.model_runner import run_scoring_job
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


def analytics_job(db: DatabaseManager) -> None:
    """
    Recompute rolling correlations then spike events.
    Always runs in sequence so event study uses fresh correlation data.
    """
    ran_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    logger.info("── analytics_job starting (%s) ──", ran_at)

    try:
        run_correlation_analysis(DB_PATH)
        run_event_study(DB_PATH)
        db.log_job("analytics_job", ran_at, 0, 0)
        logger.info("analytics_job done")
    except Exception as exc:
        logger.exception("analytics_job failed: %s", exc)
        db.log_job("analytics_job", ran_at, 0, 0, error=str(exc))


def scoring_job(db: DatabaseManager) -> None:
    """
    Score any headlines pending FinBERT inference.

    Picks up everything news_job has inserted since the last run and
    writes sentiment_label/confidence/attention_keyword back in place.
    """
    ran_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    logger.info("── scoring_job starting (%s) ──", ran_at)

    try:
        scored = run_scoring_job(db)
        db.log_job("scoring_job", ran_at, scored, 0)
        logger.info("scoring_job done — %d scored", scored)
    except Exception as exc:
        logger.exception("scoring_job failed: %s", exc)
        db.log_job("scoring_job", ran_at, 0, 0, error=str(exc))


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Sentiment pipeline starting")
    logger.info("Database : %s", DB_PATH)
    logger.info("News job      : every %dh", NEWS_INTERVAL_HOURS)
    logger.info("Price job     : every %dh (market hours only)", PRICE_INTERVAL_HOURS)
    logger.info("Scoring job   : every %dh", SCORING_INTERVAL_HOURS)
    logger.info("Analytics job : every %dh", ANALYTICS_INTERVAL_HOURS)
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

    scheduler.add_job(
        scoring_job,
        trigger    = IntervalTrigger(hours=SCORING_INTERVAL_HOURS),
        args       = [db],
        id         = "scoring_job",
        name       = "FinBERT sentiment scorer",
        next_run_time = datetime.utcnow(),     # run immediately
    )

    scheduler.add_job(
        analytics_job,
        trigger       = IntervalTrigger(hours=ANALYTICS_INTERVAL_HOURS),
        args          = [db],
        id            = "analytics_job",
        name          = "Correlation + event study analytics",
        next_run_time = datetime.utcnow(),
    )

    logger.info("Scheduler started. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
