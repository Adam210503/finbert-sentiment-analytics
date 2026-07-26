"""
src/analysis/correlation.py
────────────────────────────
Computes rolling Pearson correlation between daily mean sentiment score
and next-day log return for each tracked ticker.

Sentiment score per headline is confidence-weighted and signed:
  positive  → +confidence
  negative  → -confidence
  neutral   →  0

These are averaged per (ticker, market_date) to form a daily signal.
The log return is shifted by -1 so sentiment on day T is compared
against the return on day T+1.

Results are written to the `correlations` table and printed to stdout.

Usage:
    python src/analysis/correlation.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DB_PATH

# ── Schema ────────────────────────────────────────────────────────
_CREATE_CORRELATIONS = """
CREATE TABLE IF NOT EXISTS correlations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    market_date    TEXT    NOT NULL,
    mean_sentiment REAL,
    log_return     REAL,
    pearson_7d     REAL,
    pearson_30d    REAL,
    UNIQUE (ticker, market_date)
);
"""

# Minimum non-NaN data points required before emitting a rolling value.
_MIN_PERIODS_7D  = 4
_MIN_PERIODS_30D = 10


# ─────────────────────────────────────────────────────────────────
# Public
# ─────────────────────────────────────────────────────────────────

def run_correlation_analysis(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Aggregate sentiment, join price data, compute rolling correlations,
    persist to SQLite, and return the full result DataFrame.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_CORRELATIONS)
        conn.commit()
        df = _load_joined(conn)

    if df.empty:
        print("  No scored headlines joined with price data — nothing to compute.")
        return df

    df = _add_sentiment_score(df)
    daily = _aggregate_daily(df)
    daily = _attach_next_day_return(daily)
    daily = _compute_rolling_correlations(daily)
    _persist(daily, db_path)
    return daily


# ─────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────

def _load_joined(conn: sqlite3.Connection) -> pd.DataFrame:
    """Pull all scored sentiment rows joined with their matching price record."""
    sql = """
        SELECT
            s.ticker,
            s.market_date,
            s.sentiment_label,
            s.confidence,
            p.daily_return  AS log_return
        FROM sentiment_scores s
        JOIN price_data p
          ON s.ticker      = p.ticker
         AND s.market_date = p.market_date
        WHERE s.sentiment_label IS NOT NULL
          AND s.confidence      IS NOT NULL
          AND p.daily_return    IS NOT NULL
        ORDER BY s.ticker, s.market_date
    """
    return pd.read_sql_query(sql, conn)


def _add_sentiment_score(df: pd.DataFrame) -> pd.DataFrame:
    """Map (label, confidence) → signed sentiment score."""
    sign_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    df = df.copy()
    df["sentiment_score"] = df["sentiment_label"].map(sign_map) * df["confidence"]
    return df


def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Average sentiment scores to one row per (ticker, market_date)."""
    daily = (
        df.groupby(["ticker", "market_date"])
        .agg(
            mean_sentiment=("sentiment_score", "mean"),
            log_return=("log_return", "first"),  # identical for all rows in the group
        )
        .reset_index()
    )
    daily["market_date"] = pd.to_datetime(daily["market_date"])
    daily = daily.sort_values(["ticker", "market_date"]).reset_index(drop=True)
    return daily


def _attach_next_day_return(daily: pd.DataFrame) -> pd.DataFrame:
    """Shift log_return by -1 per ticker so sentiment[T] pairs with return[T+1]."""
    daily = daily.copy()
    daily["log_return"] = daily.groupby("ticker")["log_return"].shift(-1)
    return daily


def _compute_rolling_correlations(daily: pd.DataFrame) -> pd.DataFrame:
    """Add pearson_7d and pearson_30d columns via rolling window per ticker."""
    results = []
    for ticker, group in daily.groupby("ticker"):
        g = group.copy().reset_index(drop=True)
        # Drop rows where either signal is NaN before rolling (last row per ticker
        # always loses log_return after the shift above).
        valid = g.dropna(subset=["mean_sentiment", "log_return"])

        def rolling_corr(window, min_periods):
            return (
                valid["mean_sentiment"]
                .rolling(window, min_periods=min_periods)
                .corr(valid["log_return"])
                .values
            )

        g.loc[valid.index, "pearson_7d"]  = rolling_corr(7,  _MIN_PERIODS_7D)
        g.loc[valid.index, "pearson_30d"] = rolling_corr(30, _MIN_PERIODS_30D)
        results.append(g)

    return pd.concat(results, ignore_index=True)


def _persist(daily: pd.DataFrame, db_path: Path) -> None:
    """Upsert correlation rows into SQLite."""
    rows = daily[["ticker", "market_date", "mean_sentiment",
                  "log_return", "pearson_7d", "pearson_30d"]].copy()
    rows["market_date"] = rows["market_date"].dt.strftime("%Y-%m-%d")

    sql = """
        INSERT INTO correlations
            (ticker, market_date, mean_sentiment, log_return, pearson_7d, pearson_30d)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, market_date) DO UPDATE SET
            mean_sentiment = excluded.mean_sentiment,
            log_return     = excluded.log_return,
            pearson_7d     = excluded.pearson_7d,
            pearson_30d    = excluded.pearson_30d
    """
    with sqlite3.connect(db_path) as conn:
        conn.executemany(sql, rows.itertuples(index=False, name=None))
        conn.commit()


def _print_summary(daily: pd.DataFrame) -> None:
    """Print latest 7d and 30d Pearson r per ticker."""
    W   = 54
    SEP = "─" * W
    print(f"\n{SEP}")
    print(f"  {'TICKER':<8}  {'LATEST 7d Pearson r':>20}  {'LATEST 30d Pearson r':>20}")
    print(SEP)
    for ticker, group in daily.groupby("ticker"):
        last = group.dropna(subset=["pearson_7d", "pearson_30d"]).tail(1)
        if last.empty:
            print(f"  {ticker:<8}  {'insufficient data':>20}  {'insufficient data':>20}")
            continue
        r7  = last["pearson_7d"].iloc[0]
        r30 = last["pearson_30d"].iloc[0]
        print(f"  {ticker:<8}  {r7:>20.4f}  {r30:>20.4f}")
    print(SEP)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 54)
    print("  SENTIMENT ↔ RETURN CORRELATION ANALYSIS")
    print("=" * 54)

    print(f"\n  Database : {DB_PATH}")

    print("\n[1/3] Loading and joining sentiment + price data...")
    daily = run_correlation_analysis()

    if daily.empty:
        return

    rows_written = daily["pearson_7d"].notna().sum() + daily["pearson_30d"].notna().sum()
    print(f"  └─ {len(daily)} date rows across {daily['ticker'].nunique()} ticker(s)")

    print("\n[2/3] Rolling correlations computed and persisted.")
    print(f"  └─ {rows_written} non-NaN correlation values written to `correlations` table")

    print("\n[3/3] Summary:")
    _print_summary(daily)


if __name__ == "__main__":
    main()
    print("Correlation analysis complete")
