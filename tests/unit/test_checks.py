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
    """stale_row_check detects runs of consecutive identical close values."""

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

    def test_stale_row_check_passes_cent_quantized_random_walk(self) -> None:
        """A realistic cent-quantized equity series with many NON-consecutive
        duplicate closes must pass.

        Regression test for CR-01: equity closes are quantized to cents, so over
        multi-year history many close values collide globally without the feed
        being stale.  The old global-duplicate implementation rejected this
        series (~80 % globally non-duplicated vs a 95 % threshold); the
        consecutive-run implementation must accept it.
        """
        import numpy as np

        from volforecast.validate.checks import stale_row_check

        rng = np.random.default_rng(42)
        steps = rng.normal(0.0, 1.5, 750)
        closes = np.round(150.0 + np.cumsum(steps), 2)  # cent-quantized random walk
        close_s = pd.Series(closes)

        # Precondition 1: the fixture reproduces the old failure mode — well
        # below the old 95 % global-uniqueness threshold.
        frac_globally_unique = (~close_s.duplicated(keep=False)).sum() / len(close_s)
        assert frac_globally_unique < 0.95, (
            f"Fixture must contain heavy global close duplication; "
            f"got {frac_globally_unique:.1%} globally unique"
        )
        # Precondition 2: the feed is NOT stale — no long consecutive runs.
        same_as_prev = close_s.diff() == 0
        run_lengths = same_as_prev.groupby((~same_as_prev).cumsum()).cumsum() + 1
        assert run_lengths.max() <= 5, "Fixture must not contain a stale run"

        dates = pd.date_range("2022-01-03", periods=750, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes + 1.0,
                "low": closes - 1.0,
                "close": closes,
                "volume": [1_000_000.0] * 750,
            },
            index=dates,
        )
        df.index.name = "date"

        result = stale_row_check(df)
        assert result.passed is True, (
            f"Clean cent-quantized series falsely flagged as stale: {result.reason}; "
            f"{len(result.offending_index)} offending rows"
        )

    def test_stale_row_check_flags_six_consecutive_identical_closes(self) -> None:
        """A run of 6 consecutive identical closes inside a varied series must fail."""
        from volforecast.validate.checks import stale_row_check

        closes = [100.0 + i * 0.5 for i in range(30)]
        closes[10:16] = [105.0] * 6  # 6-row stale run (max_run default is 5)
        dates = pd.date_range("2022-01-03", periods=30, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1.0 for c in closes],
                "low": [c - 1.0 for c in closes],
                "close": closes,
                "volume": [1_000_000.0] * 30,
            },
            index=dates,
        )
        df.index.name = "date"

        result = stale_row_check(df)
        assert result.passed is False, "stale_row_check must flag a 6-row run of identical closes"
        assert len(result.offending_index) > 0, (
            "offending_index must list the rows beyond the tolerated run length"
        )
        # The offending rows must lie inside the stale run
        for ts in result.offending_index:
            assert dates[10] <= ts <= dates[15], f"Offending row {ts} outside the stale run"

    def test_stale_row_check_passes_run_at_max_boundary(self) -> None:
        """A run of exactly max_run (5) consecutive identical closes still passes."""
        from volforecast.validate.checks import stale_row_check

        closes = [100.0 + i * 0.5 for i in range(30)]
        closes[10:15] = [105.0] * 5  # exactly max_run — tolerated
        dates = pd.date_range("2022-01-03", periods=30, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1.0 for c in closes],
                "low": [c - 1.0 for c in closes],
                "close": closes,
                "volume": [1_000_000.0] * 30,
            },
            index=dates,
        )
        df.index.name = "date"

        result = stale_row_check(df)
        assert result.passed is True, (
            f"A run of exactly max_run rows must be tolerated; reason={result.reason}"
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
