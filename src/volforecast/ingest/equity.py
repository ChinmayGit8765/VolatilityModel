"""Equity OHLCV adapter — yfinance-based SPY/AAPL/MSFT daily data fetching.

Uses yfinance's batch `download()` API with explicit `auto_adjust=True` and `threads=False`
for correctness and thread-safety, wrapped in a tenacity retry/backoff decorator to tolerate
Yahoo Finance rate limits.

Design decisions (locked per RESEARCH.md and CONTEXT.md):
  - `auto_adjust=True` is set explicitly (not relied on as a default) because the choice to
    record split/dividend-adjusted OHLC rather than raw prices is a modelling decision that
    must be visible in source code. (Pitfall 6 from RESEARCH.md: "Don't rely on defaults for
    adjusted prices — future yfinance versions may change the default.")
  - `threads=False` prevents a known shared-global-dict race condition in yfinance 1.x when
    multiple tickers are downloaded concurrently. (RESEARCH.md anti-pattern, issue #2557.)
  - The `nospam` extra (`yfinance[nospam]`) installs `requests_ratelimiter` which yfinance
    consumes automatically to throttle Yahoo requests; this adapter does not need to configure
    it explicitly.
  - Live API calls are guarded by VOLFORECAST_NO_LIVE_API so tests never touch the network.
    The guard fires BEFORE the retry wrapper; tests monkeypatch `yf.download` (not the whole
    function) so the env guard is checked before any call is attempted.

OHLCV Contract:
  All DataFrames returned by this module conform to the canonical contract defined in base.py:
    - Index: tz-aware UTC DatetimeIndex named "date"
    - Columns: ["open", "high", "low", "close", "volume"] (lowercase, float64)
"""

from __future__ import annotations

import os

import pandas as pd
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from volforecast.ingest.base import OHLCV_COLUMNS


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _download_with_retry(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Internal retry-wrapped yf.download call.

    Separated from the public API so the VOLFORECAST_NO_LIVE_API guard can be checked
    outside the retry loop (the guard should fire once, not be re-raised on each retry).

    Uses:
    - auto_adjust=True: records adjusted OHLC (splits + dividends) — explicit documented choice
    - threads=False: disables threading to avoid shared-global-dict race in yfinance 1.x
    - progress=False: suppresses console output
    """
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,  # Adjusted for splits + dividends — explicit documented decision
        threads=False,  # Disable threading: shared-global-dict bug in yfinance 1.x
        progress=False,
    )


def download_equity_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Download split/dividend-adjusted daily OHLCV for equity tickers.

    Public entry point: checks VOLFORECAST_NO_LIVE_API guard before invoking the
    retry-wrapped `_download_with_retry` helper, then normalizes the raw yfinance output
    to the OHLCV contract per ticker.

    Args:
        tickers: List of Yahoo Finance ticker symbols, e.g. ["SPY", "AAPL", "MSFT"].
        start: Start date in "YYYY-MM-DD" format (inclusive).
        end: End date in "YYYY-MM-DD" format (exclusive in yfinance convention).

    Returns:
        dict mapping each ticker symbol to a DataFrame conforming to the OHLCV contract:
        - Index: tz-aware UTC DatetimeIndex named "date"
        - Columns: ["open", "high", "low", "close", "volume"] (float64)

    Raises:
        RuntimeError: If VOLFORECAST_NO_LIVE_API=1 is set (network guard for CI/tests).
        Exception: Retried up to 5 times by _download_with_retry; reraises on final failure.
    """
    if os.environ.get("VOLFORECAST_NO_LIVE_API") == "1":
        raise RuntimeError(
            "Live API call blocked: VOLFORECAST_NO_LIVE_API=1 is set. "
            "Inject a monkeypatched yf.download in tests; unset for local live ingestion."
        )

    raw = _download_with_retry(tickers, start, end)
    return {ticker: normalize_equity_frame(raw, ticker) for ticker in tickers}


def normalize_equity_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Reshape a yf.download output for a single ticker into the OHLCV contract.

    yf.download for multiple tickers returns a 2-level MultiIndex column DataFrame:
        (price_type, ticker) e.g. ("Open", "SPY"), ("Close", "AAPL")
    For a single ticker it returns flat columns: "Open", "High", "Low", "Close", "Volume".

    This function handles both cases and returns a canonical OHLCV DataFrame.

    Args:
        raw: Raw DataFrame from yf.download (MultiIndex or flat columns).
        ticker: Ticker symbol to extract, e.g. "SPY".

    Returns:
        pd.DataFrame with OHLCV_COLUMNS, float64 dtype, tz-aware UTC DatetimeIndex "date".
    """
    # Handle MultiIndex columns (multi-ticker download)
    if isinstance(raw.columns, pd.MultiIndex):
        # Level 0: price type ("Open", "High", etc.), Level 1: ticker
        try:
            df = raw.xs(ticker, level=1, axis=1)
        except KeyError:
            # Fallback: try level 0 as ticker
            df = raw.xs(ticker, level=0, axis=1)
    else:
        # Single-ticker download returns flat columns
        df = raw.copy()

    # Rename price type columns to canonical lowercase OHLCV names
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Keep only canonical OHLCV columns (drop Dividends, Stock Splits, etc.)
    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]]

    # Ensure all OHLCV columns are present
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"normalize_equity_frame: missing columns {missing} for ticker {ticker}. "
            f"Available: {df.columns.tolist()}"
        )
    df = df[OHLCV_COLUMNS]

    # Ensure tz-aware UTC DatetimeIndex named "date"
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    elif str(idx.tz) != "UTC":
        idx = idx.tz_convert("UTC")
    df.index = idx.normalize()
    df.index.name = "date"

    return df.astype("float64")
