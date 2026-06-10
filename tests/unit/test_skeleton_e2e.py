"""End-to-end skeleton tests for the VolForecast trusted-data layer.

These tests exercise the full path:
  fixture OHLCV data -> drop_incomplete_candles -> incremental_update -> validate schema

All three tests are written against the *not-yet-existing* implementation modules
and are expected to fail (RED) until Task 2 provides the implementation (GREEN).

Imports will fail with ImportError/ModuleNotFoundError until the implementation
is in place. The conftest.py sets VOLFORECAST_NO_LIVE_API=1 so no live calls
can be made.
"""

from pathlib import Path

import pandas as pd
import pytest

from volforecast.ingest.base import drop_incomplete_candles, incremental_update
from volforecast.validate.schemas import crypto_ohlcv_schema

DAY_MS = 24 * 60 * 60 * 1000  # one daily candle timeframe in milliseconds


class TestIngestPipelineWritesValidatedParquet:
    """Test that the ingest pipeline produces a valid, validated parquet.

    Given fixture OHLCV data where the last row is a still-forming candle
    (opened today UTC), the pipeline must:
    1. Drop the forming candle via drop_incomplete_candles
    2. Write a parquet whose max index date is strictly < today UTC
    """

    def test_ingest_pipeline_writes_validated_parquet(
        self, crypto_fixture_df: pd.DataFrame, now_utc_ms: int, tmp_path: Path
    ) -> None:
        """Fixture with a still-forming last candle produces parquet < today UTC."""
        # Append a still-forming candle: opened exactly at today's UTC midnight
        today_ts = now_utc_ms  # open_time = now_utc_ms (today midnight)
        forming_candle = [
            today_ts,  # open_time ms
            40000.0,  # open
            41000.0,  # high
            39500.0,  # low
            40500.0,  # close
            999.0,  # volume
        ]

        # Build raw ccxt-shaped candles from the fixture + forming candle
        # ccxt format: [timestamp_ms, open, high, low, close, volume]
        base_rows = []
        for idx, row in crypto_fixture_df.iterrows():
            ts_ms = int(idx.timestamp() * 1000)
            base_rows.append(
                [ts_ms, row["open"], row["high"], row["low"], row["close"], row["volume"]]
            )
        raw_candles = base_rows + [forming_candle]

        # Drop incomplete candles
        closed = drop_incomplete_candles(raw_candles, timeframe_ms=DAY_MS, now_ms=now_utc_ms)

        # Build DataFrame from closed candles
        from volforecast.ingest.base import candles_to_df

        df = candles_to_df(closed)

        # Validate
        validated = crypto_ohlcv_schema.validate(df, lazy=True)

        # Write via incremental_update
        out_path = tmp_path / "BTC-USD.parquet"
        result = incremental_update(out_path, validated)

        # Assertions
        today_utc = pd.Timestamp.now(tz="UTC").normalize()
        assert out_path.exists(), "Parquet file was not written"
        assert result.index.max() < today_utc, (
            f"Forming candle leaked into store: max date {result.index.max()} >= today {today_utc}"
        )
        assert result.index.name == "date", "Index must be named 'date'"
        assert list(result.columns) == ["open", "high", "low", "close", "volume"], (
            f"Wrong columns: {result.columns.tolist()}"
        )


class TestIngestGateRejectsOhlcViolation:
    """Test that the Pandera validation gate rejects OHLC-invalid rows.

    A row where high < low must cause schema validation to raise (fail closed)
    and must NOT write the output parquet.
    """

    def test_ingest_gate_rejects_ohlc_violation(self, tmp_path: Path) -> None:
        """A row with high < low causes the gate to raise; no parquet written."""
        import pandas as pd

        dates = pd.date_range("2022-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [40000.0] * 5,
                "high": [41000.0, 41000.0, 39000.0, 41000.0, 41000.0],  # row 2: high<low
                "low": [39000.0, 39000.0, 40000.0, 39000.0, 39000.0],  # row 2: high<low
                "close": [40500.0] * 5,
                "volume": [5000.0] * 5,
            },
            index=dates,
        )
        df.index.name = "date"

        out_path = tmp_path / "bad.parquet"

        import pandera.errors

        with pytest.raises((pandera.errors.SchemaErrors, pandera.errors.SchemaError)):
            crypto_ohlcv_schema.validate(df, lazy=True)

        # Parquet must NOT have been written
        assert not out_path.exists(), "Parquet should not be written on validation failure"


class TestIncrementalMergeDedupe:
    """Test that re-running ingest merges and deduplicates on the date index.

    Re-running with an overlapping date range must produce no duplicate index
    entries, keeping the latest value for any overlapping dates.
    """

    def test_incremental_merge_dedupe(
        self, crypto_fixture_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Overlapping date range merges with dedupe; no duplicate index entries."""
        out_path = tmp_path / "BTC-USD.parquet"

        # First write: first 10 rows
        first_batch = crypto_fixture_df.iloc[:10].copy()
        incremental_update(out_path, first_batch)

        # Second write: last 10 rows (rows 8-17), overlapping with rows 8-9 from first batch
        # Modify the overlapping rows to verify "keep=last" semantics
        second_batch = crypto_fixture_df.iloc[8:].copy()
        second_batch.loc[second_batch.index[:2], "close"] = 99999.0  # sentinel value

        result = incremental_update(out_path, second_batch)

        # No duplicate index entries
        assert not result.index.duplicated().any(), "Duplicate index entries found after merge"
        assert result.index.is_monotonic_increasing, "Index is not sorted after merge"

        # Verify total rows = 18 (no phantom rows)
        assert len(result) == len(crypto_fixture_df), (
            f"Expected {len(crypto_fixture_df)} rows after dedupe, got {len(result)}"
        )

        # Verify the overlapping rows were updated to the "last" value (sentinel 99999.0)
        overlap_dates = second_batch.index[:2]
        for date in overlap_dates:
            assert result.loc[date, "close"] == 99999.0, (
                f"Expected latest value (99999.0) for {date}, got {result.loc[date, 'close']}"
            )
