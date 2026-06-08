"""
src/collectors/market_data.py
──────────────────────────────
Fetches daily OHLCV price data from yfinance for all target tickers.

Key design decisions:
  - yfinance.download() batches all tickers in a single HTTP call,
    reducing overhead compared to fetching one ticker at a time.
  - daily_return is stored as the log return: ln(close / prev_close).
    Log returns are preferred over simple returns for time-series
    analysis because they are additive across periods and more
    normally distributed — both properties the correlation engine needs.
  - The job checks is_market_hours() before fetching. Price data from
    yfinance during extended hours may be incomplete. Fetching after
    16:00 EST ensures all four OHLCV values are final for the day.
  - Prices for the last PRICE_LOOKBACK_DAYS days are always re-fetched
    to handle yfinance's occasional delayed data updates, but
    INSERT OR IGNORE in db_manager means existing records are not
    overwritten.
"""

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import math

import pandas as pd
import yfinance as yf

from config.settings import (
    MARKET_CLOSE_HOUR,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MIN,
    MARKET_TZ,
    TICKERS,
)

logger = logging.getLogger(__name__)

# How many calendar days to look back each fetch.
# 14 days catches any gaps from weekends and public holidays.
PRICE_LOOKBACK_DAYS = 14


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def fetch_all_prices() -> list[dict]:
    """
    Fetch OHLCV data for all configured tickers.

    Skips the fetch entirely if called outside market hours so that
    the scheduler can run at any interval and remain safe.

    Returns
    -------
    List of normalised records ready for DB insert. Each record has:
        ticker, market_date, open, high, low, close, volume, daily_return
    """
    if not is_market_hours():
        logger.debug("Outside market hours — skipping price fetch")
        return []

    start = (date.today() - timedelta(days=PRICE_LOOKBACK_DAYS)).isoformat()
    end   = (date.today() + timedelta(days=1)).isoformat()  # yfinance end is exclusive

    logger.info("Fetching OHLCV for %s from %s to %s", TICKERS, start, end)

    try:
        # auto_adjust=True applies splits and dividends automatically
        raw = yf.download(
            tickers   = TICKERS,
            start     = start,
            end       = end,
            interval  = "1d",
            auto_adjust = True,
            progress  = False,
        )
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        return []

    if raw.empty:
        logger.warning("yfinance returned an empty DataFrame")
        return []

    records = _parse_ohlcv(raw)
    logger.info("Price fetch complete — %d records", len(records))
    return records


def is_market_hours() -> bool:
    """
    Return True if the current time falls within NYSE regular trading hours:
    Monday–Friday, 09:30–16:00 EST.

    Note: NYSE holidays are not accounted for (documented limitation).
    """
    now = datetime.now(tz=MARKET_TZ)

    # Skip weekends
    if now.weekday() >= 5:   # 5=Saturday, 6=Sunday
        return False

    open_time  = now.replace(hour=MARKET_OPEN_HOUR,  minute=MARKET_OPEN_MIN,  second=0, microsecond=0)
    close_time = now.replace(hour=MARKET_CLOSE_HOUR, minute=0, second=0, microsecond=0)

    return open_time <= now < close_time


# ─────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────

def _parse_ohlcv(raw: pd.DataFrame) -> list[dict]:
    """
    Transform the raw yfinance multi-level DataFrame into flat dicts.

    yfinance.download() with multiple tickers returns a DataFrame with
    a MultiIndex column: (field, ticker). This function normalises it
    into one dict per (ticker, date) pair and computes the log return.
    """
    records: list[dict] = []

    # If only one ticker is requested yfinance may drop the ticker level.
    # Ensure a consistent MultiIndex structure.
    if not isinstance(raw.columns, pd.MultiIndex):
        # Single ticker — re-index with the ticker name
        ticker = TICKERS[0] if len(TICKERS) == 1 else "UNKNOWN"
        raw.columns = pd.MultiIndex.from_product([raw.columns, [ticker]])

    for ticker in TICKERS:
        try:
            df = raw.xs(ticker, axis=1, level=1).copy()
        except KeyError:
            logger.warning("No data found for ticker %s", ticker)
            continue

        df = df.dropna(subset=["Close"])
        df = df.sort_index()  # ascending date order

        prev_close: float | None = None

        for idx, row in df.iterrows():
            market_date = idx.date().isoformat()
            close       = float(row["Close"])
            log_ret     = _log_return(close, prev_close)
            prev_close  = close

            records.append({
                "ticker":       ticker,
                "market_date":  market_date,
                "open":         _safe_float(row.get("Open")),
                "high":         _safe_float(row.get("High")),
                "low":          _safe_float(row.get("Low")),
                "close":        close,
                "volume":       _safe_int(row.get("Volume")),
                "daily_return": log_ret,
            })

    return records


def _log_return(close: float, prev_close: float | None) -> float | None:
    """
    Compute log return: ln(close / prev_close).

    Returns None if prev_close is unavailable or zero.

    Log returns are used instead of simple returns because:
      1. Additive across time periods.
      2. More normally distributed — better for Pearson correlation.
      3. Symmetric: a 50% loss followed by a 100% gain = 0 in log space.
    """
    if prev_close is None or prev_close <= 0:
        return None
    try:
        return math.log(close / prev_close)
    except (ValueError, ZeroDivisionError):
        return None


def _safe_float(value) -> float | None:
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
