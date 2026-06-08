"""
src/storage/db_manager.py
─────────────────────────
Manages all SQLite interactions for the sentiment pipeline.

Two tables:
  sentiment_scores — one row per headline, including sentiment fields
                     which start as NULL and are filled by the inference layer.
  price_data       — one row per ticker per trading day (OHLCV + log return).

Design decisions:
  - INSERT OR IGNORE enforces deduplication via UNIQUE constraints without
    raising exceptions on duplicates. The caller receives a count of how
    many rows were actually inserted so it can log the skip rate.
  - market_date is stored as a TEXT in ISO format (YYYY-MM-DD) rather than
    a DATE type because SQLite treats DATE as TEXT internally anyway, and
    TEXT makes the join condition transparent in query output.
  - model_version stores the checkpoint directory name, allowing comparison
    of base vs fine-tuned model outputs on the same headlines later.
"""

import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────

_CREATE_SENTIMENT_SCORES = """
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker            TEXT    NOT NULL,
    headline          TEXT    NOT NULL,
    source            TEXT    NOT NULL,
    raw_timestamp     TEXT,
    market_date       TEXT,
    sentiment_label   TEXT,
    confidence        REAL,
    attention_keyword TEXT,
    model_version     TEXT,
    headline_hash     TEXT    NOT NULL UNIQUE
);
"""

_CREATE_PRICE_DATA = """
CREATE TABLE IF NOT EXISTS price_data (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT    NOT NULL,
    market_date  TEXT    NOT NULL,
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       INTEGER,
    daily_return REAL,
    UNIQUE (ticker, market_date)
);
"""

_CREATE_JOB_LOG = """
CREATE TABLE IF NOT EXISTS job_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name    TEXT    NOT NULL,
    ran_at      TEXT    NOT NULL,
    inserted    INTEGER NOT NULL DEFAULT 0,
    skipped     INTEGER NOT NULL DEFAULT 0,
    error       TEXT
);
"""

# Indices for the join used by the correlation engine
_CREATE_INDICES = """
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date
    ON sentiment_scores (ticker, market_date);

CREATE INDEX IF NOT EXISTS idx_price_ticker_date
    ON price_data (ticker, market_date);
"""


# ─────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────

class DatabaseManager:
    """Thread-safe SQLite manager using one connection per thread."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialize()
        logger.info("DatabaseManager initialised at %s", db_path)

    # ── Internal ─────────────────────────────────────────────────

    @contextmanager
    def _connect(self):
        """Yield a connection that auto-commits or rolls back."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                _CREATE_SENTIMENT_SCORES
                + _CREATE_PRICE_DATA
                + _CREATE_JOB_LOG
                + _CREATE_INDICES
            )
        logger.debug("Schema initialised (tables + indices)")

    # ── Public: inserts ───────────────────────────────────────────

    def insert_headlines(self, records: list[dict]) -> tuple[int, int]:
        """
        Insert news headline records.

        Duplicates are silently skipped via INSERT OR IGNORE on headline_hash.

        Parameters
        ----------
        records : list of dicts with keys:
            ticker, headline, source, raw_timestamp, market_date

        Returns
        -------
        (inserted, skipped) counts
        """
        if not records:
            return 0, 0

        sql = """
            INSERT OR IGNORE INTO sentiment_scores
                (ticker, headline, source, raw_timestamp, market_date, headline_hash)
            VALUES
                (:ticker, :headline, :source, :raw_timestamp, :market_date, :headline_hash)
        """

        # Attach headline_hash to each record before insert
        enriched = [
            {**r, "headline_hash": _hash(r["headline"])} for r in records
        ]

        inserted = 0
        with self._connect() as conn:
            for row in enriched:
                cursor = conn.execute(sql, row)
                inserted += cursor.rowcount

        skipped = len(records) - inserted
        logger.debug("Headlines: %d inserted, %d skipped (duplicates)", inserted, skipped)
        return inserted, skipped

    def insert_price_data(self, records: list[dict]) -> tuple[int, int]:
        """
        Insert OHLCV price records.

        Duplicates on (ticker, market_date) are silently skipped.

        Parameters
        ----------
        records : list of dicts with keys:
            ticker, market_date, open, high, low, close, volume, daily_return

        Returns
        -------
        (inserted, skipped) counts
        """
        if not records:
            return 0, 0

        sql = """
            INSERT OR IGNORE INTO price_data
                (ticker, market_date, open, high, low, close, volume, daily_return)
            VALUES
                (:ticker, :market_date, :open, :high, :low, :close, :volume, :daily_return)
        """

        inserted = 0
        with self._connect() as conn:
            for row in records:
                cursor = conn.execute(sql, row)
                inserted += cursor.rowcount

        skipped = len(records) - inserted
        logger.debug("Prices: %d inserted, %d skipped (duplicates)", inserted, skipped)
        return inserted, skipped

    def log_job(
        self,
        job_name: str,
        ran_at: str,
        inserted: int,
        skipped: int,
        error: str | None = None,
    ) -> None:
        """Record one scheduler job execution in job_log."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO job_log (job_name, ran_at, inserted, skipped, error) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_name, ran_at, inserted, skipped, error),
            )

    # ── Public: queries ───────────────────────────────────────────

    def get_unscored_headlines(self, limit: int = 200) -> list[dict]:
        """
        Return headlines where sentiment_label IS NULL.
        Called by the inference layer (model_runner.py) to fetch
        the next batch of headlines to score.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, ticker, headline, headline_hash
                FROM   sentiment_scores
                WHERE  sentiment_label IS NULL
                ORDER  BY id ASC
                LIMIT  ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_sentiment(
        self,
        headline_hash: str,
        label: str,
        confidence: float,
        attention_keyword: str,
        model_version: str,
    ) -> None:
        """
        Write inference results back to sentiment_scores.
        Called by model_runner.py after batch inference.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sentiment_scores
                SET    sentiment_label   = ?,
                       confidence        = ?,
                       attention_keyword = ?,
                       model_version     = ?
                WHERE  headline_hash = ?
                """,
                (label, confidence, attention_keyword, model_version, headline_hash),
            )

    def get_health(self) -> dict:
        """
        Return summary statistics for the observability dashboard.
        """
        with self._connect() as conn:
            total_headlines = conn.execute(
                "SELECT COUNT(*) FROM sentiment_scores"
            ).fetchone()[0]

            scored = conn.execute(
                "SELECT COUNT(*) FROM sentiment_scores WHERE sentiment_label IS NOT NULL"
            ).fetchone()[0]

            total_prices = conn.execute(
                "SELECT COUNT(*) FROM price_data"
            ).fetchone()[0]

            last_news = conn.execute(
                "SELECT raw_timestamp FROM sentiment_scores ORDER BY id DESC LIMIT 1"
            ).fetchone()

            last_price = conn.execute(
                "SELECT market_date FROM price_data ORDER BY id DESC LIMIT 1"
            ).fetchone()

            recent_jobs = conn.execute(
                """
                SELECT job_name, ran_at, inserted, skipped, error
                FROM   job_log
                ORDER  BY id DESC
                LIMIT  10
                """
            ).fetchall()

        return {
            "total_headlines": total_headlines,
            "scored_headlines": scored,
            "unscored_headlines": total_headlines - scored,
            "total_price_records": total_prices,
            "last_news_timestamp": last_news[0] if last_news else None,
            "last_price_date": last_price[0] if last_price else None,
            "recent_jobs": [dict(r) for r in recent_jobs],
        }


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _hash(text: str) -> str:
    """SHA-256 hex digest of a headline string (for deduplication)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
