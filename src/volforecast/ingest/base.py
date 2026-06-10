"""Base ingest utilities — canonical OHLCV column contract, incremental merge-dedupe,
and incomplete-candle exclusion.

This module provides the core data primitives that all ingest adapters (crypto, equity)
must conform to. It deliberately has no import-time side effects and no live API calls.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical column order for all OHLCV DataFrames produced by this package.
OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


def drop_incomplete_candles(
    candles: list[list],
    timeframe_ms: int,
    now_ms: int,
) -> list[list]:
    """Remove still-forming (incomplete) candles from a ccxt-shaped candle list.

    A daily candle opened at `open_time` is considered still-forming if
    its bar period has not yet closed:  open_time + timeframe_ms > now_ms.

    Args:
        candles: List of ccxt-shaped candles: [open_time_ms, open, high, low, close, volume].
        timeframe_ms: Duration of one candle bar in milliseconds (e.g. 86400000 for 1 day).
        now_ms: Current UTC time in milliseconds. A candle is complete only if
                open_time + timeframe_ms <= now_ms.

    Returns:
        Filtered list containing only fully-closed candles.

    Example:
        >>> day_ms = 24 * 60 * 60 * 1000
        >>> import time; now = int(time.time() * 1000)
        >>> drop_incomplete_candles([[now - day_ms * 2, 1, 2, 0.5, 1.5, 100],
        ...                          [now, 1, 2, 0.5, 1.5, 100]], day_ms, now)
        [[now - day_ms * 2, 1, 2, 0.5, 1.5, 100]]
    """
    return [c for c in candles if c[0] + timeframe_ms <= now_ms]


def candles_to_df(candles: list[list]) -> pd.DataFrame:
    """Convert ccxt-shaped candles to a canonical OHLCV DataFrame.

    Args:
        candles: List of ccxt-shaped candles: [open_time_ms, open, high, low, close, volume].

    Returns:
        pd.DataFrame with:
        - tz-aware UTC DatetimeIndex named "date" (normalized to 00:00 UTC)
        - float64 columns in OHLCV_COLUMNS order: open, high, low, close, volume
    """
    if not candles:
        return pd.DataFrame(
            columns=OHLCV_COLUMNS,
            index=pd.DatetimeIndex([], tz="UTC", name="date"),
        ).astype(float)

    df = pd.DataFrame(candles, columns=["_ts", "open", "high", "low", "close", "volume"])
    # Convert ms timestamp to tz-aware UTC DatetimeIndex
    df.index = pd.to_datetime(df["_ts"], unit="ms", utc=True).dt.normalize()
    df.index.name = "date"
    df = df.drop(columns=["_ts"])

    # Ensure canonical column order and float64 dtype
    df = df[OHLCV_COLUMNS].astype("float64")
    return df


def incremental_update(existing_path: Path, new_bars: pd.DataFrame) -> pd.DataFrame:
    """Merge new_bars into an existing parquet; deduplicate on the date index.

    This implements cache-first incremental ingestion: read the existing parquet,
    concat new bars, deduplicate keeping the last (newest) value per date, sort,
    and write back.

    Args:
        existing_path: Path to the existing parquet file. May not exist (first run).
        new_bars: DataFrame with canonical OHLCV columns and tz-aware UTC DatetimeIndex.

    Returns:
        The merged, deduplicated DataFrame (also written to existing_path).

    Notes:
        - On first run (file does not exist), writes new_bars directly.
        - On subsequent runs, existing data is loaded and merged with new_bars.
        - Duplicate index entries are resolved by keeping the last (new_bars wins for
          any overlapping dates, which supports corrections/re-runs).
        - The parent directory is created if it does not exist.
    """
    existing_path = Path(existing_path)
    existing_path.parent.mkdir(parents=True, exist_ok=True)

    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        combined = pd.concat([existing, new_bars])
    else:
        combined = new_bars.copy()

    # Deduplicate: keep the last occurrence (new_bars wins over existing for same date)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    combined.to_parquet(existing_path)
    return combined
