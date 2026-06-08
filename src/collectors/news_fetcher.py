"""
src/collectors/news_fetcher.py
──────────────────────────────
Fetches financial news headlines from the NewsAPI /v2/everything endpoint.

Key design decisions:
  - One request per ticker per run (3 requests per schedule cycle).
    With NEWS_INTERVAL_HOURS=4 that is 18 requests/day — well below
    the free-tier limit of 100 requests/day.
  - market_date normalisation maps each article's publishedAt timestamp
    to the trading session it can realistically affect:
      · Published before 16:00 EST → same trading day.
      · Published after 16:00 EST or on a weekend → next trading day.
  - Exponential backoff retries on HTTP 429 (rate limited) responses.
  - Returns a list of dicts ready for DatabaseManager.insert_headlines().
"""

import logging
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from config.settings import (
    BACKOFF_DELAYS,
    MARKET_CLOSE_HOUR,
    MARKET_TZ,
    MAX_RETRIES,
    NEWS_DAYS_BACK,
    NEWS_PAGE_SIZE,
    NEWSAPI_KEY,
    TICKER_QUERIES,
    TICKERS,
)

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def fetch_all_tickers() -> list[dict]:
    """
    Fetch news for all configured tickers.

    Returns
    -------
    List of normalised records ready for DB insert. Each record has:
        ticker, headline, source, raw_timestamp, market_date
    """
    if not NEWSAPI_KEY:
        logger.warning(
            "NEWSAPI_KEY not set — skipping news fetch. "
            "Set it with: export NEWSAPI_KEY=your_key"
        )
        return []

    all_records: list[dict] = []
    for ticker in TICKERS:
        records = _fetch_ticker(ticker)
        all_records.extend(records)
        logger.info("  %s: %d headlines fetched", ticker, len(records))

    logger.info("News fetch complete — %d total records", len(all_records))
    return all_records


# ─────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────

def _fetch_ticker(ticker: str) -> list[dict]:
    """Fetch, parse and normalise articles for a single ticker."""
    query     = TICKER_QUERIES.get(ticker, ticker)
    from_date = (date.today() - timedelta(days=NEWS_DAYS_BACK)).isoformat()

    params = {
        "q":          query,
        "language":   "en",
        "sortBy":     "publishedAt",
        "pageSize":   NEWS_PAGE_SIZE,
        "from":       from_date,
        "apiKey":     NEWSAPI_KEY,
    }

    try:
        response = _get_with_backoff(NEWSAPI_URL, params)
    except requests.RequestException as exc:
        logger.error("NewsAPI request failed for %s: %s", ticker, exc)
        return []

    if response.status_code != 200:
        logger.error(
            "NewsAPI returned %d for %s: %s",
            response.status_code, ticker, response.text[:200],
        )
        return []

    data     = response.json()
    articles = data.get("articles", [])

    records = []
    for article in articles:
        title = (article.get("title") or "").strip()

        # NewsAPI sometimes returns "[Removed]" for deleted articles
        if not title or title.lower() == "[removed]":
            continue

        raw_ts = article.get("publishedAt", "")

        records.append({
            "ticker":        ticker,
            "headline":      title,
            "source":        _extract_source(article),
            "raw_timestamp": raw_ts,
            "market_date":   _market_date(raw_ts),
        })

    return records


def _get_with_backoff(url: str, params: dict) -> requests.Response:
    """
    GET request with exponential backoff on 429 responses.

    Raises
    ------
    requests.RequestException
        If all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt, delay in enumerate([0] + BACKOFF_DELAYS[:MAX_RETRIES - 1], start=1):
        if delay:
            logger.debug("Backoff: waiting %ds before retry %d", delay, attempt)
            time.sleep(delay)

        try:
            resp = requests.get(url, params=params, timeout=10)
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Request error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            continue

        if resp.status_code == 429:
            logger.warning("HTTP 429 rate limited (attempt %d/%d)", attempt, MAX_RETRIES)
            continue

        return resp

    raise requests.RequestException(
        f"All {MAX_RETRIES} attempts failed"
    ) from last_exc


def _extract_source(article: dict) -> str:
    """Return a clean source identifier, e.g. 'reuters.com'."""
    source = article.get("source", {})
    name   = source.get("name") or ""
    url    = article.get("url") or ""

    if name:
        return name.lower().replace(" ", "-")

    # Fall back to domain from URL
    if url:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lstrip("www.")
        except Exception:
            pass

    return "newsapi"


def _market_date(raw_timestamp: str) -> str:
    """
    Map a UTC publish timestamp to the trading session it can affect.

    Rules (NYSE):
      - Published before 16:00 EST on a weekday → same trading day.
      - Published at or after 16:00 EST, or on Saturday/Sunday → next weekday.

    Note: NYSE holidays are not accounted for (documented limitation).

    Parameters
    ----------
    raw_timestamp : str
        ISO 8601 string from NewsAPI (e.g. "2026-05-14T13:42:00Z").

    Returns
    -------
    str  — YYYY-MM-DD trading date.
    """
    if not raw_timestamp:
        return date.today().isoformat()

    try:
        # Parse UTC timestamp
        utc_dt = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        est_dt = utc_dt.astimezone(MARKET_TZ)
    except ValueError:
        logger.debug("Could not parse timestamp '%s'; using today", raw_timestamp)
        return date.today().isoformat()

    # If published at or after market close, shift to next calendar day
    if est_dt.hour >= MARKET_CLOSE_HOUR:
        est_dt = est_dt + timedelta(days=1)

    # Advance past weekends (0=Mon … 6=Sun)
    target = est_dt.date()
    while target.weekday() >= 5:          # 5=Sat, 6=Sun
        target += timedelta(days=1)

    return target.isoformat()
