"""
src/analysis/event_study.py
────────────────────────────
Event study: measures log returns at t+1d, t+2d, and t+5d following
sentiment spike events for each tracked ticker.

A spike is defined as a day where the confidence-weighted mean sentiment
score (from the correlations table) crosses a directional threshold:
  positive spike : mean_sentiment >= +0.65
  negative spike : mean_sentiment <= -0.65

t+Nd returns are resolved by advancing N positions through the actual
trading days present in price_data, so no calendar assumptions are made.

Usage:
    python src/analysis/event_study.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DB_PATH

# ── Constants ─────────────────────────────────────────────────────
POSITIVE_THRESHOLD =  0.65
NEGATIVE_THRESHOLD = -0.65
MIN_EVENTS_WARNING =  3

# ── Schema ────────────────────────────────────────────────────────
_CREATE_SPIKE_EVENTS = """
CREATE TABLE IF NOT EXISTS spike_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    event_date     TEXT    NOT NULL,
    event_type     TEXT    NOT NULL,
    mean_sentiment REAL    NOT NULL,
    return_t1d     REAL,
    return_t2d     REAL,
    return_t5d     REAL,
    UNIQUE (ticker, event_date, event_type)
);
"""


# ─────────────────────────────────────────────────────────────────
# Public
# ─────────────────────────────────────────────────────────────────

def run_event_study(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Identify sentiment spike events, attach forward returns, persist to
    SQLite, and return the full spike_events DataFrame.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_SPIKE_EVENTS)
        conn.commit()
        sentiment = _load_sentiment(conn)
        prices    = _load_prices(conn)

    if sentiment.empty:
        print("  No data in correlations table — run correlation.py first.")
        return pd.DataFrame()

    events = _identify_spikes(sentiment)

    if events.empty:
        print("  No spike events found within the current data window.")
        return pd.DataFrame()

    events = _attach_forward_returns(events, prices)
    _persist(events, db_path)
    return events


# ─────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────

def _load_sentiment(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load mean_sentiment per (ticker, market_date) from correlations."""
    df = pd.read_sql_query(
        "SELECT ticker, market_date, mean_sentiment FROM correlations",
        conn,
    )
    df["market_date"] = pd.to_datetime(df["market_date"])
    return df


def _load_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all trading-day log returns from price_data."""
    df = pd.read_sql_query(
        "SELECT ticker, market_date, daily_return FROM price_data WHERE daily_return IS NOT NULL",
        conn,
    )
    df["market_date"] = pd.to_datetime(df["market_date"])
    df = df.sort_values(["ticker", "market_date"]).reset_index(drop=True)
    return df


def _identify_spikes(sentiment: pd.DataFrame) -> pd.DataFrame:
    """Return one row per spike event with event_type assigned."""
    pos = sentiment[sentiment["mean_sentiment"] >= POSITIVE_THRESHOLD].copy()
    pos["event_type"] = "positive"

    neg = sentiment[sentiment["mean_sentiment"] <= NEGATIVE_THRESHOLD].copy()
    neg["event_type"] = "negative"

    events = pd.concat([pos, neg], ignore_index=True)
    events = events.sort_values(["ticker", "market_date"]).reset_index(drop=True)
    events = events.rename(columns={"market_date": "event_date"})
    return events


def _attach_forward_returns(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each spike event, look up the log return N trading days ahead
    by advancing N positions through the sorted price_data dates for
    that ticker. No calendar arithmetic — uses actual trading days only.
    """
    # Build per-ticker index: date → positional index in sorted trading-day list
    ticker_dates: dict[str, list] = {}
    ticker_returns: dict[str, dict] = {}

    for ticker, group in prices.groupby("ticker"):
        dates = group["market_date"].tolist()
        rets  = group["daily_return"].tolist()
        ticker_dates[ticker]  = dates
        ticker_returns[ticker] = {d: r for d, r in zip(dates, rets)}

    def forward_return(ticker: str, event_date, offset: int):
        dates = ticker_dates.get(ticker, [])
        try:
            pos = dates.index(event_date)
        except ValueError:
            # Event date not in price_data (weekend / holiday edge case)
            # Find the next available trading day instead
            future = [d for d in dates if d > event_date]
            if not future:
                return None
            pos = dates.index(future[0]) - 1  # offset will add 1

        target_pos = pos + offset
        if target_pos >= len(dates):
            return None
        return ticker_returns[ticker].get(dates[target_pos])

    events = events.copy()
    events["return_t1d"] = events.apply(
        lambda r: forward_return(r["ticker"], r["event_date"], 1), axis=1
    )
    events["return_t2d"] = events.apply(
        lambda r: forward_return(r["ticker"], r["event_date"], 2), axis=1
    )
    events["return_t5d"] = events.apply(
        lambda r: forward_return(r["ticker"], r["event_date"], 5), axis=1
    )
    return events


def _persist(events: pd.DataFrame, db_path: Path) -> None:
    """Upsert spike events into SQLite."""
    rows = events[["ticker", "event_date", "event_type",
                   "mean_sentiment", "return_t1d", "return_t2d", "return_t5d"]].copy()
    rows["event_date"] = rows["event_date"].dt.strftime("%Y-%m-%d")

    sql = """
        INSERT INTO spike_events
            (ticker, event_date, event_type, mean_sentiment,
             return_t1d, return_t2d, return_t5d)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, event_date, event_type) DO UPDATE SET
            mean_sentiment = excluded.mean_sentiment,
            return_t1d     = excluded.return_t1d,
            return_t2d     = excluded.return_t2d,
            return_t5d     = excluded.return_t5d
    """
    with sqlite3.connect(db_path) as conn:
        conn.executemany(sql, rows.itertuples(index=False, name=None))
        conn.commit()


def _print_summary(events: pd.DataFrame) -> None:
    W   = 66
    SEP = "─" * W

    total = len(events)
    if total < MIN_EVENTS_WARNING:
        print(
            f"\n  WARNING: Insufficient spike events for meaningful analysis "
            f"— pipeline needs more data\n  ({total} event(s) found, recommend >= {MIN_EVENTS_WARNING})"
        )

    print(f"\n{SEP}")
    print(f"  {'TICKER':<6}  {'TYPE':<10}  {'N':>3}  {'t+1d':>8}  {'t+2d':>8}  {'t+5d':>8}")
    print(SEP)

    for ticker in sorted(events["ticker"].unique()):
        t_events = events[events["ticker"] == ticker]
        for etype in ["positive", "negative"]:
            subset = t_events[t_events["event_type"] == etype]
            n = len(subset)
            if n == 0:
                continue
            r1 = subset["return_t1d"].mean()
            r2 = subset["return_t2d"].mean()
            r5 = subset["return_t5d"].mean()
            fmt = lambda v: f"{v:+.4f}" if pd.notna(v) else "   N/A"
            print(f"  {ticker:<6}  {etype:<10}  {n:>3}  {fmt(r1):>8}  {fmt(r2):>8}  {fmt(r5):>8}")

    print(SEP)
    print(f"  Thresholds: positive >= {POSITIVE_THRESHOLD:+.2f}  |  negative <= {NEGATIVE_THRESHOLD:+.2f}")
    print(f"  Returns are log returns (next N trading days after event date)")
    print(SEP)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 66)
    print("  SENTIMENT SPIKE EVENT STUDY")
    print("=" * 66)
    print(f"\n  Database  : {DB_PATH}")
    print(f"  Thresholds: positive >= {POSITIVE_THRESHOLD}  |  negative <= {NEGATIVE_THRESHOLD}")

    print("\n[1/3] Loading sentiment and price data...")
    events = run_event_study()

    if events.empty:
        return

    pos_n = (events["event_type"] == "positive").sum()
    neg_n = (events["event_type"] == "negative").sum()
    print(f"  └─ {len(events)} spike events found  ({pos_n} positive, {neg_n} negative)")

    print("\n[2/3] Forward returns attached and written to `spike_events` table.")

    print("\n[3/3] Summary (mean log return per event type per ticker):")
    _print_summary(events)


if __name__ == "__main__":
    main()
    print("Event study complete")
