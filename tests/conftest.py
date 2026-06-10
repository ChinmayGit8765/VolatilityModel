"""Test configuration and shared fixtures.

Sets VOLFORECAST_NO_LIVE_API=1 at import time so no test can accidentally make
a live exchange/API call.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

# Guard: prevent any live API calls during tests.
os.environ.setdefault("VOLFORECAST_NO_LIVE_API", "1")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def crypto_fixture_df() -> pd.DataFrame:
    """Load the committed crypto OHLCV fixture parquet.

    Returns a DataFrame with:
    - tz-aware UTC DatetimeIndex named "date"
    - Columns: open, high, low, close, volume (all float64)
    """
    path = FIXTURES_DIR / "crypto_sample.parquet"
    df = pd.read_parquet(path)
    return df


@pytest.fixture
def now_utc_ms() -> int:
    """Return today's UTC midnight in milliseconds (epoch ms).

    Used to simulate the 'current time' when testing incomplete-candle exclusion.
    A candle opened today (open_time >= today midnight UTC) should be treated as
    still-forming and excluded.
    """
    today_midnight = pd.Timestamp.now(tz="UTC").normalize()
    return int(today_midnight.timestamp() * 1000)
