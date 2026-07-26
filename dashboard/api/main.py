"""
dashboard/api/main.py
──────────────────────
Read-only FastAPI REST API over the FinBERT SQLite sentiment pipeline database.

Run with (from project root):
    uvicorn dashboard.api.main:app --reload --port 8000
"""

import sqlite3
import sys
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DB_PATH


# ── DB connection ──────────────────────────────────────────────────────────────

@contextmanager
def _db():
    """Open a per-request read-only SQLite connection and close on exit."""
    if not Path(DB_PATH).exists():
        raise HTTPException(status_code=503, detail="Database unavailable")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ── Startup ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not Path(DB_PATH).exists():
        print(f"\n  WARNING: Database not found at {DB_PATH}\n")
    else:
        conn = sqlite3.connect(DB_PATH)
        tables = ["sentiment_scores", "price_data", "correlations", "spike_events", "job_log"]
        print(f"\n  FinBERT API — {DB_PATH}")
        print("  " + "─" * 40)
        for t in tables:
            try:
                n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                print(f"  {t:<22}: {n:>6} rows")
            except Exception:
                print(f"  {t:<22}: (not found)")
        conn.close()
        print("  " + "─" * 40 + "\n")
    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinBERT Sentiment API",
    description="Read-only REST API over the FinBERT SQLite sentiment pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_ticker(conn: sqlite3.Connection, ticker: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM sentiment_scores WHERE ticker = ? LIMIT 1", (ticker,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Ticker not found")


def _to_list(rows) -> list[dict]:
    return [dict(r) for r in rows]


# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health", summary="System health snapshot")
def health():
    """
    Row counts, latest dates, inference backlog, and tracked tickers.
    """
    try:
        with _db() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM sentiment_scores"
            ).fetchone()[0]

            unscored = conn.execute(
                "SELECT COUNT(*) FROM sentiment_scores WHERE sentiment_label IS NULL"
            ).fetchone()[0]

            latest_price = conn.execute(
                "SELECT MAX(market_date) FROM price_data"
            ).fetchone()[0]

            latest_scored = conn.execute(
                "SELECT MAX(market_date) FROM sentiment_scores "
                "WHERE sentiment_label IS NOT NULL"
            ).fetchone()[0]

            corr_rows = conn.execute(
                "SELECT COUNT(*) FROM correlations"
            ).fetchone()[0]

            spike_count = conn.execute(
                "SELECT COUNT(*) FROM spike_events"
            ).fetchone()[0]

            tickers = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT ticker FROM sentiment_scores ORDER BY ticker"
                ).fetchall()
            ]

        return {
            "status": "ok",
            "headlines_total": total,
            "headlines_unscored": unscored,
            "inference_backlog": unscored,
            "latest_price_date": latest_price,
            "latest_scored_date": latest_scored,
            "correlation_rows": corr_rows,
            "spike_events": spike_count,
            "tickers": tickers,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /sentiment ─────────────────────────────────────────────────────────────

@app.get("/sentiment", summary="Scored headlines")
def sentiment(
    ticker: Annotated[
        Optional[str],
        Query(description="Filter by ticker symbol, e.g. AAPL"),
    ] = None,
    limit: Annotated[
        int,
        Query(description="Maximum rows to return (1–500)", ge=1, le=500),
    ] = 50,
    label: Annotated[
        Optional[str],
        Query(description="Filter by sentiment class: positive, neutral, negative"),
    ] = None,
):
    """
    Scored headlines sorted by market_date descending.
    `url` is populated for rows fetched after the url column migration.
    Pre-migration rows return an empty string.
    """
    try:
        with _db() as conn:
            if ticker:
                _check_ticker(conn, ticker)

            conditions = ["sentiment_label IS NOT NULL"]
            params: list = []

            if ticker:
                conditions.append("ticker = ?")
                params.append(ticker)
            if label:
                conditions.append("sentiment_label = ?")
                params.append(label)

            where = "WHERE " + " AND ".join(conditions)
            params.append(limit)

            rows = conn.execute(
                f"""
                SELECT
                    ticker,
                    headline,
                    source,
                    COALESCE(url, '') AS url,
                    market_date,
                    sentiment_label,
                    confidence,
                    attention_keyword,
                    model_version
                FROM sentiment_scores
                {where}
                ORDER BY market_date DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return {"ticker": ticker, "data": _to_list(rows)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /correlation ───────────────────────────────────────────────────────────

@app.get("/correlation", summary="Sentiment vs return time series")
def correlation(
    ticker: Annotated[str, Query(description="Ticker symbol (required), e.g. AAPL")],
    window: Annotated[
        int,
        Query(description="Rolling Pearson window: 7 or 30 (default 7)"),
    ] = 7,
):
    """
    Full time series of daily mean_sentiment + log_return + rolling Pearson r
    for the requested ticker, sorted by market_date ascending.
    Used for the dual-axis line chart.
    """
    if window not in (7, 30):
        raise HTTPException(status_code=422, detail="window must be 7 or 30")

    try:
        with _db() as conn:
            _check_ticker(conn, ticker)

            pearson_col = "pearson_7d" if window == 7 else "pearson_30d"

            rows = conn.execute(
                f"""
                SELECT
                    market_date,
                    mean_sentiment,
                    log_return,
                    {pearson_col} AS pearson_r
                FROM correlations
                WHERE ticker = ?
                ORDER BY market_date ASC
                """,
                (ticker,),
            ).fetchall()

        return {"ticker": ticker, "window": window, "data": _to_list(rows)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /events ────────────────────────────────────────────────────────────────

@app.get("/events", summary="Sentiment spike events with forward returns")
def events(
    ticker: Annotated[
        Optional[str],
        Query(description="Filter by ticker symbol"),
    ] = None,
):
    """
    All spike events with t+1d, t+2d, t+5d forward returns,
    sorted by event_date descending.
    """
    try:
        with _db() as conn:
            if ticker:
                _check_ticker(conn, ticker)

            if ticker:
                rows = conn.execute(
                    """
                    SELECT ticker, event_date, event_type, mean_sentiment,
                           return_t1d, return_t2d, return_t5d
                    FROM spike_events
                    WHERE ticker = ?
                    ORDER BY event_date DESC
                    """,
                    (ticker,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT ticker, event_date, event_type, mean_sentiment,
                           return_t1d, return_t2d, return_t5d
                    FROM spike_events
                    ORDER BY event_date DESC
                    """
                ).fetchall()

        return {"data": _to_list(rows)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /keywords ──────────────────────────────────────────────────────────────

@app.get("/keywords", summary="Top attention keywords by sentiment class")
def keywords(
    ticker: Annotated[
        Optional[str],
        Query(description="Filter by ticker symbol"),
    ] = None,
    sentiment: Annotated[
        str,
        Query(description="Sentiment class: positive, neutral, negative"),
    ] = "positive",
):
    """
    Top 10 attention_keyword values by frequency for the given
    ticker and sentiment class.
    """
    try:
        with _db() as conn:
            if ticker:
                _check_ticker(conn, ticker)

            conditions = ["attention_keyword IS NOT NULL", "sentiment_label = ?"]
            params: list = [sentiment]

            if ticker:
                conditions.append("ticker = ?")
                params.append(ticker)

            where = "WHERE " + " AND ".join(conditions)

            rows = conn.execute(
                f"""
                SELECT attention_keyword AS keyword, COUNT(*) AS count
                FROM sentiment_scores
                {where}
                GROUP BY attention_keyword
                ORDER BY count DESC
                LIMIT 10
                """,
                params,
            ).fetchall()

        return {"ticker": ticker, "sentiment": sentiment, "data": _to_list(rows)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /flow ──────────────────────────────────────────────────────────────────

@app.get("/flow", summary="Causal flow strip — today's aggregated signals")
def flow():
    """
    Headline count, mean sentiment, average 7d Pearson r, and latest
    log return — all for the most recent market date across all tickers.
    """
    try:
        with _db() as conn:
            most_recent = conn.execute(
                "SELECT MAX(market_date) FROM sentiment_scores "
                "WHERE sentiment_label IS NOT NULL"
            ).fetchone()[0]

            if most_recent is None:
                return {
                    "headlines_today": 0,
                    "mean_sentiment_today": None,
                    "pearson_7d_avg": None,
                    "latest_log_return": None,
                }

            headlines_today = conn.execute(
                "SELECT COUNT(*) FROM sentiment_scores "
                "WHERE market_date = ? AND sentiment_label IS NOT NULL",
                (most_recent,),
            ).fetchone()[0]

            # Confidence-weighted signed sentiment score for most recent date
            scored_rows = conn.execute(
                """
                SELECT sentiment_label, confidence
                FROM sentiment_scores
                WHERE market_date = ? AND sentiment_label IS NOT NULL
                """,
                (most_recent,),
            ).fetchall()

            sign_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            scores = [
                sign_map.get(r["sentiment_label"], 0.0) * (r["confidence"] or 0.0)
                for r in scored_rows
            ]
            mean_sentiment = sum(scores) / len(scores) if scores else None

            # Latest pearson_7d per ticker — reliable latest-row-per-group subquery
            pearson_row = conn.execute(
                """
                SELECT AVG(c.pearson_7d)
                FROM correlations c
                INNER JOIN (
                    SELECT ticker, MAX(market_date) AS latest_date
                    FROM correlations
                    WHERE pearson_7d IS NOT NULL
                    GROUP BY ticker
                ) latest ON c.ticker = latest.ticker
                         AND c.market_date = latest.latest_date
                WHERE c.pearson_7d IS NOT NULL
                """
            ).fetchone()
            pearson_7d_avg = pearson_row[0] if pearson_row else None

            # Latest log_return per ticker
            return_row = conn.execute(
                """
                SELECT AVG(c.log_return)
                FROM correlations c
                INNER JOIN (
                    SELECT ticker, MAX(market_date) AS latest_date
                    FROM correlations
                    WHERE log_return IS NOT NULL
                    GROUP BY ticker
                ) latest ON c.ticker = latest.ticker
                         AND c.market_date = latest.latest_date
                WHERE c.log_return IS NOT NULL
                """
            ).fetchone()
            latest_log_return = return_row[0] if return_row else None

        return {
            "headlines_today": headlines_today,
            "mean_sentiment_today": round(mean_sentiment, 4) if mean_sentiment is not None else None,
            "pearson_7d_avg": round(pearson_7d_avg, 4) if pearson_7d_avg is not None else None,
            "latest_log_return": round(latest_log_return, 4) if latest_log_return is not None else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
