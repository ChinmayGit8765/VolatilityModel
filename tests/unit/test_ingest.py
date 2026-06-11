"""Unit tests for the multi-asset ingest layer.

Tests are fully offline: all network calls are monkeypatched.
The conftest.py sets VOLFORECAST_NO_LIVE_API=1 so no live exchange/API calls can leak.

Task 1 tests:
  - test_equity_adapter_normalizes_to_ohlcv_contract
  - test_equity_adapter_retries_then_succeeds

Task 2 tests:
  - test_incremental_resume_uses_last_stored_timestamp
  - test_ingest_all_dispatches_by_asset_class
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_spy_df(n: int = 5) -> pd.DataFrame:
    """Build a minimal SPY-shaped DataFrame conforming to the OHLCV contract."""
    dates = pd.bdate_range(start="2022-01-03", periods=n, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [470.0 + i for i in range(n)],
            "high": [473.0 + i for i in range(n)],
            "low": [468.0 + i for i in range(n)],
            "close": [471.0 + i for i in range(n)],
            "volume": [80_000_000.0] * n,
        },
        index=dates,
    )
    df.index.name = "date"
    return df.astype("float64")


def _make_yf_multi_raw(tickers: list[str], n: int = 5) -> pd.DataFrame:
    """Simulate the raw MultiIndex output yf.download returns for multiple tickers.

    yf.download with multiple tickers returns a DataFrame with a 2-level column
    MultiIndex: (price_type, ticker).  e.g. ('Open', 'SPY'), ('Close', 'AAPL'), ...
    """
    import numpy as np

    dates = pd.bdate_range(start="2022-01-03", periods=n, tz="UTC")
    price_types = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([price_types, tickers])
    data = {(pt, t): np.random.uniform(100, 500, n) for pt, t in cols}
    # Ensure high >= low for sanity
    for t in tickers:
        open_ = data[("Open", t)]
        close_ = data[("Close", t)]
        data[("High", t)] = np.maximum(open_, close_) + 1.0
        data[("Low", t)] = np.minimum(open_, close_) - 1.0
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


# ── Task 1: Equity adapter ────────────────────────────────────────────────────


class TestEquityAdapterNormalizesOhlcvContract:
    """Equity adapter normalizes yf.download multi-ticker raw output to the OHLCV contract."""

    def test_equity_adapter_normalizes_to_ohlcv_contract(self) -> None:
        """Given a monkeypatched yfinance return, adapter yields per-ticker DataFrames
        with exactly OHLCV_COLUMNS, float64 dtype, and tz-aware UTC 'date' index.

        The test patches both yf.download AND unsets VOLFORECAST_NO_LIVE_API so the env
        guard (which protects against accidentally calling the real Yahoo Finance API) does
        not block the monkeypatched execution path. This is the correct test pattern: the
        env guard is a CI safety net, not a test-isolation tool — the monkeypatched
        yf.download is what provides isolation.
        """
        import os

        from volforecast.ingest.base import OHLCV_COLUMNS
        from volforecast.ingest.equity import download_equity_ohlcv

        tickers = ["SPY", "AAPL"]
        raw = _make_yf_multi_raw(tickers, n=10)

        # Temporarily unset env guard so the monkeypatched download path is exercised
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("volforecast.ingest.equity.yf.download", return_value=raw),
        ):
            os.environ.pop("VOLFORECAST_NO_LIVE_API", None)
            result = download_equity_ohlcv(tickers, start="2022-01-03", end="2022-01-31")

        assert isinstance(result, dict), "Expected dict keyed by ticker"
        for ticker in tickers:
            assert ticker in result, f"{ticker} missing from result"
            df = result[ticker]
            assert list(df.columns) == OHLCV_COLUMNS, (
                f"{ticker}: columns {df.columns.tolist()} != {OHLCV_COLUMNS}"
            )
            assert df.index.name == "date", f"{ticker}: index.name={df.index.name}"
            assert df.index.tz is not None, f"{ticker}: index is not tz-aware"
            assert str(df.index.tz) == "UTC", f"{ticker}: expected UTC tz, got {df.index.tz}"
            for col in OHLCV_COLUMNS:
                assert df[col].dtype == "float64", (
                    f"{ticker}[{col}]: dtype={df[col].dtype}, expected float64"
                )


class TestEquityAdapterRetriesThenSucceeds:
    """Equity adapter retry wrapper: a downloader that raises once then succeeds returns ok."""

    def test_equity_adapter_retries_then_succeeds(self) -> None:
        """If yf.download raises on the first call but succeeds on the second,
        download_equity_ohlcv still returns a valid per-ticker dict (retry is wired).

        The env guard is temporarily unset so the retry path is exercised with the
        monkeypatched downloader. No real network calls are made.
        """
        import os

        from volforecast.ingest.equity import download_equity_ohlcv

        tickers = ["SPY"]
        raw = _make_yf_multi_raw(tickers, n=5)

        call_count = {"n": 0}

        def flaky_download(*args, **kwargs):  # noqa: ANN001, ANN202
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("Simulated rate-limit error on first attempt")
            return raw

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("volforecast.ingest.equity.yf.download", side_effect=flaky_download),
        ):
            os.environ.pop("VOLFORECAST_NO_LIVE_API", None)
            result = download_equity_ohlcv(tickers, start="2022-01-03", end="2022-01-31")

        assert "SPY" in result, "Retry succeeded but SPY missing from result"
        assert call_count["n"] == 2, f"Expected 2 calls (1 fail + 1 succeed), got {call_count['n']}"

    def test_equity_adapter_retries_on_silent_empty_download(self) -> None:
        """WR-03: yf.download's dominant failure mode is returning an EMPTY frame
        without raising.  The adapter must treat an empty result as a retryable
        failure (so the backoff fires) and succeed once data arrives.
        """
        import os

        from volforecast.ingest.equity import download_equity_ohlcv

        tickers = ["SPY"]
        raw = _make_yf_multi_raw(tickers, n=5)

        call_count = {"n": 0}

        def silent_empty_then_ok(*args, **kwargs):  # noqa: ANN001, ANN202
            call_count["n"] += 1
            if call_count["n"] == 1:
                return pd.DataFrame()  # silent rate-limit failure: empty, no exception
            return raw

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("volforecast.ingest.equity.yf.download", side_effect=silent_empty_then_ok),
        ):
            os.environ.pop("VOLFORECAST_NO_LIVE_API", None)
            result = download_equity_ohlcv(tickers, start="2022-01-03", end="2022-01-31")

        assert "SPY" in result, "Empty-frame retry succeeded but SPY missing from result"
        assert call_count["n"] == 2, (
            f"Expected 2 calls (1 empty + 1 succeed), got {call_count['n']} — "
            "an empty download must trigger the retry path"
        )

    def test_equity_adapter_does_not_retry_programming_errors(self) -> None:
        """WR-03: non-transient exceptions (e.g. TypeError) must propagate
        immediately instead of being retried 5 times with long waits.
        """
        import os

        import pytest

        from volforecast.ingest.equity import download_equity_ohlcv

        call_count = {"n": 0}

        def buggy_download(*args, **kwargs):  # noqa: ANN001, ANN202
            call_count["n"] += 1
            raise TypeError("Simulated programming error")

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("volforecast.ingest.equity.yf.download", side_effect=buggy_download),
        ):
            os.environ.pop("VOLFORECAST_NO_LIVE_API", None)
            with pytest.raises(TypeError):
                download_equity_ohlcv(["SPY"], start="2022-01-03", end="2022-01-31")

        assert call_count["n"] == 1, (
            f"TypeError must NOT be retried; yf.download was called {call_count['n']} times"
        )


# ── Task 2: Cache-first incremental resume + multi-asset dispatch ─────────────


class TestIncrementalResumeUsesLastStoredTimestamp:
    """resume_since_ms returns next-day ms from the last stored date when parquet exists."""

    def test_incremental_resume_uses_last_stored_timestamp(self, tmp_path: Path) -> None:
        """Given an existing parquet ending at date D, resume_since_ms(parquet_path, default)
        returns a since_ms strictly greater than D (not the default_start).
        """
        from volforecast.ingest.crypto import resume_since_ms

        # Write a fixture parquet with a known last date
        last_date = pd.Timestamp("2023-06-01", tz="UTC")
        dates = pd.date_range(end=last_date, periods=5, freq="D", tz="UTC")
        df = _make_spy_df(5)
        df.index = dates
        df.index.name = "date"

        parquet_path = tmp_path / "BTC-USD.parquet"
        df.to_parquet(parquet_path)

        # Default start well in the past — should NOT be used when file exists
        default_start_ms = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
        since = resume_since_ms(parquet_path, default_start_ms)

        # since must be strictly after last_date (next day)
        expected_next_day = pd.Timestamp("2023-06-02", tz="UTC")
        expected_ms = int(expected_next_day.timestamp() * 1000)
        assert since == expected_ms, (
            f"resume_since_ms={since} expected {expected_ms} (next day after {last_date.date()})"
        )

        # Verify it does NOT fall back to default when file exists
        assert since != default_start_ms, (
            "resume_since_ms should not return default when file exists"
        )

    def test_incremental_resume_falls_back_to_default_when_missing(self, tmp_path: Path) -> None:
        """When the parquet does not exist, resume_since_ms returns default_start_ms."""
        from volforecast.ingest.crypto import resume_since_ms

        missing_path = tmp_path / "nonexistent.parquet"
        default_ms = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)

        since = resume_since_ms(missing_path, default_ms)
        assert since == default_ms, f"Expected default {default_ms} when file missing, got {since}"


class TestIngestAllDispatchesByAssetClass:
    """volforecast ingest with no --symbol dispatches each asset to the correct adapter."""

    def test_ingest_all_dispatches_by_asset_class(self, tmp_path: Path) -> None:
        """load_assets from a minimal config; each asset is routed to the correct adapter
        (ccxt for crypto, yfinance for equity); one parquet per asset is written.
        The test runs fully offline (both adapters monkeypatched).
        """
        import yaml

        from volforecast.ingest.base import OHLCV_COLUMNS

        # Write a minimal assets.yaml for the test
        config_data = {
            "assets": [
                {"symbol": "BTC/USDT", "asset_class": "crypto", "exchange": "binance"},
                {"symbol": "SPY", "asset_class": "equity", "exchange": "nasdaq"},
            ]
        }
        config_path = tmp_path / "assets.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Build mock DataFrames that each adapter would return
        crypto_df = _make_spy_df(5)
        equity_result = {"SPY": _make_spy_df(5)}

        crypto_called_with: list = []
        equity_called_with: list = []

        def mock_fetch_crypto(symbol, since_ms, exchange_id="binance", **kwargs):  # noqa: ANN001, ANN202
            crypto_called_with.append(symbol)
            return crypto_df

        def mock_download_equity(tickers, start, end):  # noqa: ANN001, ANN202
            equity_called_with.extend(tickers)
            return equity_result

        # Import and run the ingest-all logic (the CLI dispatch)
        with (
            patch("volforecast.ingest.crypto.fetch_crypto_ohlcv", side_effect=mock_fetch_crypto),
            patch(
                "volforecast.ingest.equity.download_equity_ohlcv",
                side_effect=mock_download_equity,
            ),
        ):
            from volforecast.config import load_assets, raw_path
            from volforecast.ingest.base import incremental_update
            from volforecast.ingest.crypto import fetch_crypto_ohlcv, resume_since_ms
            from volforecast.ingest.equity import download_equity_ohlcv

            assets = load_assets(config_path)
            default_start_ms = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)

            written_paths = []
            for asset in assets:
                out = raw_path(asset, data_root=tmp_path)
                out.parent.mkdir(parents=True, exist_ok=True)

                if asset["asset_class"] == "crypto":
                    since = resume_since_ms(out, default_start_ms)
                    exchange_id = asset.get("exchange", "binance")
                    df = fetch_crypto_ohlcv(
                        asset["symbol"], since_ms=since, exchange_id=exchange_id
                    )
                    incremental_update(out, df)
                    written_paths.append(out)
                elif asset["asset_class"] == "equity":
                    result = download_equity_ohlcv(
                        [asset["symbol"]], start="2022-01-01", end="2024-01-01"
                    )
                    ticker_df = result[asset["symbol"]]
                    incremental_update(out, ticker_df)
                    written_paths.append(out)

        # Each asset should have produced a parquet file
        assert len(written_paths) == 2, f"Expected 2 parquets, got {len(written_paths)}"
        for p in written_paths:
            assert p.exists(), f"Missing parquet: {p}"
            stored = pd.read_parquet(p)
            assert list(stored.columns) == OHLCV_COLUMNS, (
                f"{p.name}: columns {stored.columns.tolist()}"
            )

        # Verify dispatch was by asset_class
        assert "BTC/USDT" in crypto_called_with, "Crypto adapter not called for BTC/USDT"
        assert "SPY" in equity_called_with, "Equity adapter not called for SPY"

    def test_overlapping_rerun_no_duplicates(self, tmp_path: Path) -> None:
        """Re-running ingest over an overlapping range produces no duplicate index entries."""
        from volforecast.ingest.base import incremental_update

        df1 = _make_spy_df(10)
        out_path = tmp_path / "SPY.parquet"
        incremental_update(out_path, df1)

        # Overlap: last 5 rows of df1 with modified close
        df2 = df1.iloc[5:].copy()
        df2["close"] = 99999.0
        result = incremental_update(out_path, df2)

        assert not result.index.duplicated().any(), "Duplicate dates after overlapping re-run"
        assert len(result) == 10, f"Expected 10 rows, got {len(result)}"
        # The overlapping rows should have the new value (keep=last)
        for date in df2.index:
            assert result.loc[date, "close"] == 99999.0


# ── WR-09: equity fetches always restart from the stored inception date ───────


class TestEquityEffectiveFetchStart:
    """effective_fetch_start keeps the whole equity history on one adjustment basis."""

    def test_missing_file_returns_requested_start(self, tmp_path: Path) -> None:
        """First run (no stored parquet): the requested --start is used as-is."""
        from volforecast.ingest.equity import effective_fetch_start

        assert effective_fetch_start(tmp_path / "missing.parquet", "2022-01-01") == "2022-01-01"

    def test_later_requested_start_is_overridden_by_stored_inception(self, tmp_path: Path) -> None:
        """A --start AFTER the stored inception must be ignored: fetch from the
        stored first date so keep='last' rewrites every stored row on the
        current adjustment basis (no mixed-basis seam)."""
        from volforecast.ingest.equity import effective_fetch_start

        df = _make_spy_df(5)  # starts 2022-01-03
        path = tmp_path / "SPY.parquet"
        df.to_parquet(path)

        assert effective_fetch_start(path, "2023-06-01") == "2022-01-03"

    def test_earlier_requested_start_wins(self, tmp_path: Path) -> None:
        """A --start BEFORE the stored inception extends the history backwards."""
        from volforecast.ingest.equity import effective_fetch_start

        df = _make_spy_df(5)  # starts 2022-01-03
        path = tmp_path / "SPY.parquet"
        df.to_parquet(path)

        assert effective_fetch_start(path, "2020-01-01") == "2020-01-01"

    def test_pipeline_refetches_equity_from_stored_inception(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """_ingest_single_asset must call the equity downloader with the stored
        inception date even when the run was started with a later --start."""
        import exchange_calendars as xcals

        from volforecast import cli
        from volforecast.config import processed_path, raw_path

        # Clean equity frame over real XNYS sessions (passes the session gate)
        xnys = xcals.get_calendar("XNYS")
        sessions = xnys.sessions_in_range("2022-01-03", "2022-01-14")
        idx = sessions.tz_localize("UTC")
        n = len(idx)
        clean_df = pd.DataFrame(
            {
                "open": [150.0 + i * 0.3 for i in range(n)],
                "high": [155.0 + i * 0.3 for i in range(n)],
                "low": [148.0 + i * 0.3 for i in range(n)],
                "close": [152.0 + i * 0.5 for i in range(n)],
                "volume": [1_000_000.0 + i * 1000 for i in range(n)],
            },
            index=idx,
        )
        clean_df.index.name = "date"

        asset = {"symbol": "SPY", "asset_class": "equity", "exchange": "nasdaq"}
        out_path = raw_path(asset, data_root=tmp_path / "data")
        proc_path = processed_path(asset, data_root=tmp_path / "data")
        quarantine_dir = tmp_path / "data" / "quarantine"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Stored raw history begins 2022-01-03 (the inception date)
        clean_df.to_parquet(out_path)

        captured_start: list[str] = []

        def fake_download(tickers, start, end):  # noqa: ANN001, ANN202
            captured_start.append(start)
            return {"SPY": clean_df}

        import volforecast.ingest.equity as equity_mod

        monkeypatch.setattr(equity_mod, "download_equity_ohlcv", fake_download)

        # Re-run with a LATER --start — must be overridden by stored inception
        rc = cli._ingest_single_asset(
            asset=asset,
            since_ms=int(pd.Timestamp("2022-06-01", tz="UTC").timestamp() * 1000),
            exchange_id="nasdaq",
            start="2022-06-01",
            out_path=out_path,
            processed_out_path=proc_path,
            quarantine_path=quarantine_dir / "SPY_quarantine.csv",
        )

        assert rc == 0, f"Clean equity re-run must succeed, got rc={rc}"
        assert captured_start == ["2022-01-03"], (
            f"Equity fetch must restart from the stored inception date "
            f"(2022-01-03), got {captured_start}"
        )
