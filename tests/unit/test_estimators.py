"""Unit tests for src/volforecast/features/estimators.py.

All tests are offline-only (no network, no live data).  Series are
constructed inline or loaded from the committed fixture parquets.

Coverage:
- log_returns: r[t] == ln(close[t]/close[t-1]); first row NaN
- squared_returns: sr[t] == log_returns(close)^2
- realized_var: rolling count-based mean of squared returns; integer window;
  first (window-1) rows NaN; right-aligned (uses only data <= t)
- ewma_variance: EWMA with adjust=False; EWMA_LAMBDA constant == 0.94;
  values match manual recursion; positivity
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volforecast.features.estimators import (
    EWMA_LAMBDA,
    ewma_variance,
    log_returns,
    realized_var,
    squared_returns,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _close(prices: list[float]) -> pd.Series:
    """Build a UTC-indexed daily close Series."""
    dates = pd.date_range("2022-01-01", periods=len(prices), freq="D", tz="UTC")
    return pd.Series(prices, index=dates, name="close", dtype="float64")


# ---------------------------------------------------------------------------
# EWMA_LAMBDA constant
# ---------------------------------------------------------------------------


class TestEwmaLambdaConstant:
    def test_value_is_0_94(self) -> None:
        """EWMA_LAMBDA must equal 0.94 (RiskMetrics daily value)."""
        assert EWMA_LAMBDA == 0.94

    def test_type_is_float(self) -> None:
        assert isinstance(EWMA_LAMBDA, float)


# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------


class TestLogReturns:
    def test_first_row_is_nan(self) -> None:
        """First row must be NaN (no prior close)."""
        close = _close([100.0, 101.0, 102.0])
        result = log_returns(close)
        assert np.isnan(result.iloc[0])

    def test_values_match_formula(self) -> None:
        """r[t] == ln(close[t]) - ln(close[t-1])."""
        prices = [100.0, 102.0, 99.0, 105.0]
        close = _close(prices)
        result = log_returns(close)
        expected = np.log(np.array(prices, dtype=float))
        for i in range(1, len(prices)):
            assert abs(result.iloc[i] - (expected[i] - expected[i - 1])) < 1e-12

    def test_same_index_as_input(self) -> None:
        close = _close([100.0, 101.0, 102.0])
        result = log_returns(close)
        pd.testing.assert_index_equal(result.index, close.index)

    def test_non_nan_count(self) -> None:
        """n prices → n-1 non-NaN log returns."""
        close = _close([100.0 + i for i in range(10)])
        result = log_returns(close)
        assert result.isna().sum() == 1
        assert result.notna().sum() == 9

    def test_on_crypto_fixture(self) -> None:
        """Smoke test on the crypto parquet fixture."""
        df = pd.read_parquet(FIXTURES / "crypto_sample.parquet")
        result = log_returns(df["close"])
        assert isinstance(result, pd.Series)
        assert result.isna().sum() == 1  # only first row NaN


# ---------------------------------------------------------------------------
# squared_returns
# ---------------------------------------------------------------------------


class TestSquaredReturns:
    def test_equals_log_returns_squared(self) -> None:
        """squared_returns(close) == log_returns(close) ** 2."""
        close = _close([100.0, 102.0, 99.0, 105.0, 103.0])
        lr = log_returns(close)
        sr = squared_returns(close)
        # Compare non-NaN values
        mask = lr.notna()
        np.testing.assert_allclose(sr[mask].values, (lr[mask] ** 2).values)

    def test_first_row_is_nan(self) -> None:
        close = _close([100.0, 101.0, 102.0])
        assert np.isnan(squared_returns(close).iloc[0])

    def test_non_negative(self) -> None:
        close = _close([100.0 + i * 0.5 for i in range(10)])
        sr = squared_returns(close)
        assert (sr.dropna() >= 0).all()


# ---------------------------------------------------------------------------
# realized_var
# ---------------------------------------------------------------------------


class TestRealizedVar:
    def test_nan_prefix_length(self) -> None:
        """First window rows must be NaN (1 from log_returns + window-1 from rolling).

        Integer window ensures consistent NaN count regardless of calendar gaps.
        The series must be large enough for the window; we test with n=50.
        """
        close = _close([100.0 + i for i in range(50)])
        lr = log_returns(close)
        for window in (5, 10, 22):
            rv = realized_var(lr, window=window)
            # log_returns has 1 leading NaN; rolling(window) adds window-1 more.
            # Total NaN prefix = window rows.
            nan_count = rv.isna().sum()
            assert nan_count == window, (
                f"window={window}: expected {window} NaN rows, got {nan_count}"
            )

    def test_values_match_formula(self) -> None:
        """rv[t] == mean(r[t-w+1]^2, ..., r[t]^2)."""
        close = _close([100.0 * (1.01 ** i) for i in range(30)])
        lr = log_returns(close)
        window = 5
        rv = realized_var(lr, window=window)
        expected = lr.pow(2).rolling(window).mean()
        pd.testing.assert_series_equal(rv, expected)

    def test_right_aligned(self) -> None:
        """rv[t] must only use data up to and including t (no look-ahead)."""
        close = _close([100.0 * (1.005 ** i) for i in range(50)])
        lr = log_returns(close)
        window = 5
        rv = realized_var(lr, window=window)
        # Manually compute rv at index 10 using only rows 6..10
        squared = lr.pow(2)
        manual_rv = squared.iloc[6:11].mean()  # rows 6,7,8,9,10 (window=5)
        assert abs(rv.iloc[10] - manual_rv) < 1e-14

    def test_invalid_window_raises(self) -> None:
        close = _close([100.0, 101.0])
        lr = log_returns(close)
        with pytest.raises(ValueError, match="window must be >= 1"):
            realized_var(lr, window=0)

    def test_integer_window_on_equity_fixture(self) -> None:
        """Integer window produces consistent NaN prefix on an equity fixture."""
        df = pd.read_parquet(FIXTURES / "equity_sample.parquet")
        lr = log_returns(df["close"])
        window = 5
        rv = realized_var(lr, window=window)
        # NaN count == window (first LR NaN + window-1 from rolling = window total)
        assert rv.isna().sum() == window


# ---------------------------------------------------------------------------
# ewma_variance
# ---------------------------------------------------------------------------


class TestEwmaVariance:
    def test_adjust_false_matches_manual_recursion(self) -> None:
        """EWMA must match the RiskMetrics recursion h[t] = (1-lam)*r[t]^2 + lam*h[t-1]."""
        prices = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0]
        close = _close(prices)
        lr = log_returns(close)
        lam = 0.94

        # Manual recursion (initialise at first squared return)
        sq = (lr**2).values
        h = np.empty(len(sq))
        h[0] = np.nan  # first log return is NaN
        # Start from index 1 where first non-NaN log return exists
        h[1] = sq[1]  # initialise: first squared return gets full weight
        for t in range(2, len(sq)):
            h[t] = (1 - lam) * sq[t] + lam * h[t - 1]

        result = ewma_variance(lr, lam=lam)
        # Pandas ewm with adjust=False initialises differently (h[0] = sq[0] where sq[0]=NaN)
        # so compare only positions 2 onwards
        for i in range(2, len(h)):
            assert abs(result.iloc[i] - h[i]) < 1e-12, (
                f"Mismatch at index {i}: got {result.iloc[i]}, expected {h[i]}"
            )

    def test_default_lambda_is_0_94(self) -> None:
        """Default lam=0.94 (EWMA_LAMBDA) must match explicit lam=0.94."""
        close = _close([100.0 + i * 0.5 for i in range(20)])
        lr = log_returns(close)
        result_default = ewma_variance(lr)
        result_explicit = ewma_variance(lr, lam=0.94)
        pd.testing.assert_series_equal(result_default, result_explicit)

    def test_positive_and_finite(self) -> None:
        """EWMA variance must be positive and finite (except NaN at position 0)."""
        close = _close([100.0 + i * 0.3 for i in range(30)])
        lr = log_returns(close)
        result = ewma_variance(lr)
        non_nan = result.dropna()
        assert (non_nan > 0).all(), "All non-NaN EWMA values must be positive"
        assert np.isfinite(non_nan.values).all()

    def test_same_index_as_input(self) -> None:
        close = _close([100.0 + i for i in range(10)])
        lr = log_returns(close)
        result = ewma_variance(lr)
        pd.testing.assert_index_equal(result.index, lr.index)

    def test_ewm_adjust_false_explicitly(self) -> None:
        """Verify adjust=True gives DIFFERENT results (confirming adjust=False is active)."""
        close = _close([100.0 * (1.01 ** i) for i in range(30)])
        lr = log_returns(close)
        result_correct = ewma_variance(lr)
        result_wrong = lr.pow(2).ewm(alpha=1 - EWMA_LAMBDA, adjust=True).mean()
        # On a 30-point series they should differ at the start
        # (adjust=True converges to adjust=False for large n)
        assert not result_correct.equals(result_wrong), (
            "adjust=False and adjust=True should produce different results on a short series"
        )

    def test_on_crypto_fixture(self) -> None:
        """Smoke test on the crypto fixture."""
        df = pd.read_parquet(FIXTURES / "crypto_sample.parquet")
        lr = log_returns(df["close"])
        result = ewma_variance(lr)
        assert isinstance(result, pd.Series)
        assert result.dropna().gt(0).all()
