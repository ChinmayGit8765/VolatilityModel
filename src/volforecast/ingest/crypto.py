"""Crypto OHLCV adapter — ccxt-based BTC/ETH daily data fetching.

Uses ccxt's unified `fetch_ohlcv` API with Binance as the default exchange
(configurable to Kraken or any other ccxt-supported exchange for geo-block resilience).

Live API calls are guarded by VOLFORECAST_NO_LIVE_API so tests never touch the network.
"""

from __future__ import annotations

import os

import ccxt
import pandas as pd

from volforecast.ingest.base import candles_to_df, drop_incomplete_candles

# One daily candle timeframe in milliseconds
_DAY_MS = 24 * 60 * 60 * 1000


def fetch_crypto_ohlcv(
    symbol: str,
    since_ms: int,
    exchange_id: str = "binance",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch closed daily OHLCV candles for a crypto symbol via ccxt."""
    """Fetch closed daily OHLCV candles for a crypto symbol via ccxt.

    Implements the since-pagination pattern: loop until `len(batch) < limit`,
    then drop the still-forming last candle.

    Args:
        symbol: ccxt symbol string, e.g. "BTC/USDT".
        since_ms: Start time in milliseconds (UTC epoch). Fetch candles from this time onward.
        exchange_id: ccxt exchange identifier. Defaults to "binance"; use "kraken" as fallback
                     if Binance is geo-blocked (HTTP 451 or connection refused).
        limit: Max candles per ccxt request. 500 is safe for Binance daily timeframe.

    Returns:
        pd.DataFrame with canonical OHLCV columns (open/high/low/close/volume, float64)
        and tz-aware UTC DatetimeIndex named "date". Only fully-closed candles included.

    Raises:
        RuntimeError: If VOLFORECAST_NO_LIVE_API=1 is set (prevents live calls in CI/tests).
        ccxt.NetworkError: On transient exchange connectivity issues.
        ccxt.ExchangeError: On exchange-reported errors (rate limits, symbol not found, etc.).
    """
    if os.environ.get("VOLFORECAST_NO_LIVE_API") == "1":
        raise RuntimeError(
            "Live API call blocked: VOLFORECAST_NO_LIVE_API=1 is set. "
            "Use fixture data in tests; unset the env var for local live ingestion."
        )

    exchange_class = getattr(ccxt, exchange_id)
    exchange: ccxt.Exchange = exchange_class({"enableRateLimit": True})

    timeframe = "1d"
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = exchange.milliseconds()

    all_candles: list[list] = []
    cursor: int | None = since_ms

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_candles.extend(batch)
        if len(batch) < limit:
            break
        # Advance cursor past the last returned candle's open-time
        cursor = batch[-1][0] + 1

    # Drop the still-forming (incomplete) candle
    closed_candles = drop_incomplete_candles(all_candles, timeframe_ms=tf_ms, now_ms=now_ms)

    return candles_to_df(closed_candles)
