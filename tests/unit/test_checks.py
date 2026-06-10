"""Unit tests for volforecast.validate.checks.

Tests calendar-aware gap checks, OHLC consistency, and stale-row detection.
All tests are fully offline — they use committed parquet fixtures only.

TDD RED: this file is written before the implementation in checks.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helper: load fixtures
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> pd.DataFrame:
    """Load a parquet fixture from tests/fixtures/."""
    return pd.read_parquet(FIXTURES_DIR / name)


# ---------------------------------------------------------------------------
# equity_session_check — fabricated weekend row
# ---------------------------------------------------------------------------


class TestEquitySessionCheck:
    """equity_session_check must reject DataFrames containing non-XNYS rows."""

    def test_equity_weekend_row_rejected(self) -> None:
        """equity_bad_weekend.parquet (has a Saturday row) must fail the session check."""
        from volforecast.validate.checks import equity_session_check

        df = _load_fixture("equity_bad_weekend.parquet")
        result = equity_session_check(df)
        assert result.passed is False, (
            "equity_session_check should fail for a fixture containing a Saturday row"
        )
        assert len(result.offending_index) > 0, (
            "offending_index should list the bad weekend/holiday rows"
        )

    def test_equity_clean_passes_session_check(self) -> None:
        """A clean equity fixture with only valid XNYS sessions must pass."""
        import exchange_calendars as xcals

        from volforecast.validate.checks import equity_session_check

        # Build a small but correct equity fixture inline: Mon 2022-01-03 to Fri 2022-01-07
        # XNYS sessions: 2022-01-03 (Mon), 2022-01-04 (Tue), 2022-01-05 (Wed),
        #                2022-01-06 (Thu), 2022-01-07 (Fri)

        xnys = xcals.get_calendar("XNYS")
        sessions = xnys.sessions_in_range("2022-01-03", "2022-01-07")
        # sessions is a tz-naive DatetimeIndex; OHLCV index must be UTC tz-aware
        idx = sessions.tz_localize("UTC")
        df = pd.DataFrame(
            {
                "open": [100.0] * len(idx),
                "high": [105.0] * len(idx),
                "low": [98.0] * len(idx),
                "close": [102.0] * len(idx),
                "volume": [1_000_000.0] * len(idx),
            },
            index=idx,
        )
        df.index.name = "date"
        result = equity_session_check(df)
        assert result.passed is True, (
            f"equity_session_check should pass for a clean XNYS-aligned fixture; "
            f"offending_index={result.offending_index}"
        )


# ---------------------------------------------------------------------------
# crypto_gap_check — missing interior day
# ---------------------------------------------------------------------------


class TestCryptoGapCheck:
    """crypto_gap_check must detect any missing day in a 24/7 continuous range."""

    def test_crypto_gap_rejected(self) -> None:
        """crypto_gap.parquet (one interior day removed) must fail the gap check."""
        from volforecast.validate.checks import crypto_gap_check

        df = _load_fixture("crypto_gap.parquet")
        result = crypto_gap_check(df)
        assert result.passed is False, (
            "crypto_gap_check should fail for a fixture with a missing interior day"
        )
        assert len(result.offending_index) > 0, "offending_index should list the missing day(s)"

    def test_crypto_clean_passes_gap_check(self) -> None:
        """The crypto_sample.parquet clean fixture (continuous dates) must pass."""
        from volforecast.validate.checks import crypto_gap_check

        df = _load_fixture("crypto_sample.parquet")
        result = crypto_gap_check(df)
        assert result.passed is True, (
            f"crypto_gap_check should pass for the continuous crypto_sample fixture; "
            f"offending_index={result.offending_index}"
        )


# ---------------------------------------------------------------------------
# stale_row_check
# ---------------------------------------------------------------------------


class TestStaleRowCheck:
    """stale_row_check detects an excess of duplicate close values."""

    def test_stale_row_check_flags_all_same_close(self) -> None:
        """A DataFrame where every close is identical must fail the stale check."""
        from volforecast.validate.checks import stale_row_check

        dates = pd.date_range("2022-01-03", periods=20, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0] * 20,
                "high": [105.0] * 20,
                "low": [98.0] * 20,
                "close": [100.0] * 20,  # all identical — stale feed
                "volume": [1_000_000.0] * 20,
            },
            index=dates,
        )
        df.index.name = "date"
        result = stale_row_check(df)
        assert result.passed is False, (
            "stale_row_check should flag when all close values are duplicated"
        )

    def test_stale_row_check_passes_varied_close(self) -> None:
        """A DataFrame with varied close values passes the stale check."""
        from volforecast.validate.checks import stale_row_check

        df = _load_fixture("crypto_sample.parquet")
        result = stale_row_check(df)
        assert result.passed is True, (
            f"stale_row_check should pass for the varied-close crypto_sample fixture; "
            f"offending_index={result.offending_index}"
        )


# ---------------------------------------------------------------------------
# ohlc_consistency_check
# ---------------------------------------------------------------------------


class TestOhlcConsistencyCheck:
    """ohlc_consistency_check rejects rows where high < low/open/close."""

    def test_ohlc_violation_high_lt_low_rejected(self) -> None:
        """A row with high < low must fail the OHLC consistency check."""
        from volforecast.validate.checks import ohlc_consistency_check

        dates = pd.date_range("2022-01-03", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0, 105.0, 95.0, 105.0, 105.0],  # row 2: high=95 < low=98
                "low": [98.0] * 5,
                "close": [102.0] * 5,
                "volume": [1_000_000.0] * 5,
            },
            index=dates,
        )
        df.index.name = "date"
        result = ohlc_consistency_check(df)
        assert result.passed is False, (
            "ohlc_consistency_check should fail for a row where high < low"
        )
        assert len(result.offending_index) > 0

    def test_ohlc_clean_passes(self) -> None:
        """A clean OHLCV DataFrame passes the OHLC consistency check."""
        from volforecast.validate.checks import ohlc_consistency_check

        df = _load_fixture("crypto_sample.parquet")
        result = ohlc_consistency_check(df)
        assert result.passed is True, (
            f"ohlc_consistency_check should pass for the clean crypto_sample fixture; "
            f"offending_index={result.offending_index}"
        )


# ---------------------------------------------------------------------------
# test_clean_fixtures_pass — green path for both fixture types
# ---------------------------------------------------------------------------


class TestCleanFixturesPass:
    """Clean fixtures (crypto_sample.parquet) pass all applicable checks."""

    def test_clean_crypto_fixture_passes_all_checks(self) -> None:
        """crypto_sample.parquet must pass crypto_gap, stale, and OHLC checks."""
        from volforecast.validate.checks import (
            crypto_gap_check,
            ohlc_consistency_check,
            stale_row_check,
        )

        df = _load_fixture("crypto_sample.parquet")

        gap_result = crypto_gap_check(df)
        stale_result = stale_row_check(df)
        ohlc_result = ohlc_consistency_check(df)

        assert gap_result.passed, (
            f"crypto_gap_check failed on clean fixture: {gap_result.offending_index}"
        )
        assert stale_result.passed, (
            f"stale_row_check failed on clean fixture: {stale_result.offending_index}"
        )
        assert ohlc_result.passed, (
            f"ohlc_consistency_check failed on clean fixture: {ohlc_result.offending_index}"
        )
