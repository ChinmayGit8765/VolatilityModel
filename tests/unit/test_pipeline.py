"""End-to-end pipeline tests: ingest dispatch -> validate_asset -> processed write.

Tests are fully offline: adapters are monkeypatched to return fixture DataFrames.
No live API calls are made.

Covers:
- test_pipeline_promotes_clean_asset_to_processed: clean fixture flows through
  ingest -> validate_asset -> processed parquet written.
- test_pipeline_quarantines_and_skips_bad_asset: weekend-bad equity fixture is
  rejected by validate_asset; no processed parquet written; quarantine CSV exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _asset(symbol: str, asset_class: str, exchange: str = "binance") -> dict[str, Any]:
    return {"symbol": symbol, "asset_class": asset_class, "exchange": exchange}


def _load_fixture(name: str) -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / name)


# ---------------------------------------------------------------------------
# Test 1: clean asset promoted to processed
# ---------------------------------------------------------------------------


def test_pipeline_promotes_clean_asset_to_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean crypto fixture flows through ingest->validate->processed.

    The raw parquet is written unconditionally; after validate_asset passes,
    the validated DataFrame is also written to the processed parquet path.
    """
    from volforecast import cli
    from volforecast.config import processed_path, raw_path

    # Load clean crypto fixture (18 consecutive daily rows, all valid)
    fixture_df = _load_fixture("crypto_sample.parquet")
    assert not fixture_df.empty

    # Patch fetch_crypto_ohlcv to return the clean fixture
    import volforecast.ingest.crypto as crypto_mod

    monkeypatch.setattr(
        crypto_mod,
        "fetch_crypto_ohlcv",
        lambda symbol, since_ms, exchange_id: fixture_df,
    )

    # Patch resume_since_ms to return a fixed timestamp (skip file-based resume)
    monkeypatch.setattr(
        crypto_mod,
        "resume_since_ms",
        lambda path, default: default,
    )

    # Build asset dict
    asset = _asset("BTC/USDT", "crypto", "binance")

    # Compute paths under tmp_path
    out_path = raw_path(asset, data_root=tmp_path / "data")
    proc_path = processed_path(asset, data_root=tmp_path / "data")
    quarantine_dir = tmp_path / "data" / "quarantine"

    # Pre-create output dirs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Run the pipeline via _ingest_single_asset
    rc = cli._ingest_single_asset(
        asset=asset,
        since_ms=int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000),
        exchange_id="binance",
        start="2022-01-01",
        out_path=out_path,
        processed_out_path=proc_path,
        quarantine_path=quarantine_dir / "BTC-USD_quarantine.csv",
    )

    # Assert pipeline succeeded
    assert rc == 0, f"Pipeline returned non-zero exit code: {rc}"

    # Raw parquet must exist
    assert out_path.exists(), f"Raw parquet not written to {out_path}"

    # Processed parquet must exist (clean data promoted)
    assert proc_path.exists(), (
        f"Processed parquet not written to {proc_path} — clean asset was not promoted"
    )

    # Processed parquet must have the same number of rows as the fixture
    proc_df = pd.read_parquet(proc_path)
    assert len(proc_df) == len(fixture_df), (
        f"Processed row count {len(proc_df)} != fixture {len(fixture_df)}"
    )

    # No quarantine file should exist for a clean asset
    quarantine_files = list(quarantine_dir.glob("*.csv"))
    assert len(quarantine_files) == 0, (
        f"Unexpected quarantine file(s) for clean asset: {quarantine_files}"
    )


# ---------------------------------------------------------------------------
# Test 2: bad asset quarantined, processed NOT written
# ---------------------------------------------------------------------------


def test_pipeline_quarantines_and_skips_bad_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weekend-bad equity fixture is rejected; no processed parquet is written.

    The equity_bad_weekend fixture contains rows on non-NYSE trading sessions
    (Saturday/Sunday). validate_asset (equity_session_check) must reject it.
    The raw parquet is still written; the processed parquet must NOT be written;
    a quarantine CSV must exist.
    """
    from volforecast import cli
    from volforecast.config import processed_path, raw_path

    # Load the bad equity fixture (contains weekend rows)
    bad_df = _load_fixture("equity_bad_weekend.parquet")
    assert not bad_df.empty

    # Verify the fixture actually has weekend rows (precondition)
    has_weekend = bad_df.index.dayofweek.isin([5, 6]).any()
    assert has_weekend, "equity_bad_weekend fixture must contain Saturday/Sunday rows"

    # Patch download_equity_ohlcv to return the bad fixture
    import volforecast.ingest.equity as equity_mod

    monkeypatch.setattr(
        equity_mod,
        "download_equity_ohlcv",
        lambda tickers, start, end: {"SPY": bad_df},
    )

    # Build asset dict
    asset = _asset("SPY", "equity", "nasdaq")

    # Compute paths under tmp_path
    out_path = raw_path(asset, data_root=tmp_path / "data")
    proc_path = processed_path(asset, data_root=tmp_path / "data")
    quarantine_dir = tmp_path / "data" / "quarantine"

    # Pre-create output dirs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    quarantine_path = quarantine_dir / "SPY_quarantine.csv"

    # Run the pipeline
    rc = cli._ingest_single_asset(
        asset=asset,
        since_ms=int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000),
        exchange_id="nasdaq",
        start="2022-01-01",
        out_path=out_path,
        processed_out_path=proc_path,
        quarantine_path=quarantine_path,
    )

    # Validation failure returns 1 (gate fails closed)
    assert rc == 1, f"Pipeline should return 1 for rejected asset, got {rc}"

    # Raw parquet must still exist (raw is always written before validation)
    assert out_path.exists(), f"Raw parquet not written to {out_path}"

    # Processed parquet must NOT exist (gate fails closed)
    assert not proc_path.exists(), (
        f"Processed parquet should NOT exist for rejected asset, found at {proc_path}"
    )

    # Quarantine CSV must exist (written by validate_asset on failure)
    quarantine_files = list(quarantine_dir.glob("*.csv"))
    assert len(quarantine_files) > 0, (
        f"No quarantine CSV found in {quarantine_dir} — validate_asset did not quarantine"
    )
