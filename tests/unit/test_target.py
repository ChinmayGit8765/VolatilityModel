"""Unit tests for src/volforecast/features/target.py.

Tests are written BEFORE the implementation (TDD RED phase).
All tests verify the canonical target definition:
- Next-day squared decimal log return as realized-variance proxy
- HORIZON constant == 1
- Correct NaN placement at tail of series
- forward_realized_var helper for 5-day secondary stability check

No fixtures or network calls — all data is constructed inline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast.features.target import HORIZON, compute_target, forward_realized_var


class TestHorizonConstant:
    """HORIZON must equal 1 (next-day default)."""

    def test_horizon_is_one(self) -> None:
        """HORIZON == 1 so callers can reference the constant without magic numbers."""
        assert HORIZON == 1


class TestComputeTarget:
    """compute_target produces next-day squared decimal log returns."""

    def _make_close(self, prices: list[float], n: int = 0) -> pd.Series:
        """Build a daily close Series with UTC DatetimeIndex."""
        if not prices:
            prices = [100.0 * (1 + 0.01 * i) for i in range(n)]
        dates = pd.date_range("2022-01-01", periods=len(prices), freq="D", tz="UTC")
        return pd.Series(prices, index=dates, name="close", dtype="float64")

    def test_last_row_is_nan(self) -> None:
        """The last row must be NaN because there is no t+1 log return."""
        close = self._make_close([100.0, 101.0, 102.0, 103.0, 104.0])
        target = compute_target(close)
        assert np.isnan(target.iloc[-1]), "Last row must be NaN (no t+1 observation)"

    def test_non_nan_count(self) -> None:
        """For n observations, compute_target returns n-1 non-NaN values and 1 NaN."""
        close = self._make_close([], n=10)
        target = compute_target(close)
        assert target.isna().sum() == 1, "Exactly 1 NaN expected (last row)"
        assert target.notna().sum() == 9, "9 non-NaN rows for 10 close prices"

    def test_value_is_squared_log_return(self) -> None:
        """Value at index t is (ln(close[t+1]/close[t]))**2."""
        prices = [100.0, 101.0, 103.0, 100.0, 99.0]
        close = self._make_close(prices)
        target = compute_target(close)

        for i in range(len(prices) - 1):
            expected = (np.log(prices[i + 1] / prices[i])) ** 2
            actual = float(target.iloc[i])
            assert abs(actual - expected) < 1e-14, (
                f"Row {i}: expected {expected}, got {actual}"
            )

    def test_units_are_decimal_variance(self) -> None:
        """For typical daily equity returns (~1%), target values should be ~1e-4.

        This catches unit errors (e.g. using percent returns instead of decimal).
        """
        # 1% daily return: (ln(1.01))^2 ≈ 9.9e-5 (decimal variance, NOT ~1.0 for percent)
        prices = [100.0, 101.0]
        close = self._make_close(prices)
        target = compute_target(close)
        assert target.iloc[0] < 0.01, (
            f"Target value {target.iloc[0]:.6e} looks too large — units may be wrong "
            "(expected ~1e-4 for 1% return, not percent-squared ~1.0)"
        )

    def test_index_matches_input(self) -> None:
        """Output Series must have the same index as the input close Series."""
        close = self._make_close([], n=15)
        target = compute_target(close)
        assert target.index.equals(close.index), "Output index must match input index"

    def test_horizon_parameter(self) -> None:
        """With horizon=5, the last 5 rows are NaN and value at t = (ln(c[t+5]/c[t]))?

        Actually: horizon-step-ahead target is sum of squared log-returns
        over the next `horizon` days. For horizon=1 it's just r_{t+1}^2.
        For horizon=5, it may be computed differently — test that last 5 are NaN.
        """
        close = self._make_close([], n=20)
        target = compute_target(close, horizon=5)
        # At minimum the last `horizon` rows should be NaN
        assert target.iloc[-5:].isna().all(), (
            "With horizon=5, last 5 rows must be NaN"
        )

    def test_no_lookahead_nan_fill(self) -> None:
        """NaN at the tail must NOT be filled with zero — Pitfall 4 guard."""
        close = self._make_close([], n=10)
        target = compute_target(close)
        # The last row should be NaN, not 0.0
        assert np.isnan(target.iloc[-1]), "Last row should be NaN, not filled with zero"
        assert target.iloc[-1] != 0.0, "NaN must not be silently replaced with 0"


class TestForwardRealizedVar:
    """forward_realized_var computes mean of next `window` squared log returns."""

    def _make_close(self, n: int = 30) -> pd.Series:
        dates = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
        rng = np.random.default_rng(seed=42)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        return pd.Series(prices, index=dates, name="close")

    def test_last_window_rows_are_nan(self) -> None:
        """The last `window` rows must be NaN (no complete forward window)."""
        close = self._make_close(30)
        result = forward_realized_var(close, window=5)
        assert result.iloc[-5:].isna().all(), "Last 5 rows must be NaN for window=5"

    def test_non_nan_count(self) -> None:
        """For n observations, forward_realized_var has n-window non-NaN values."""
        n, window = 30, 5
        close = self._make_close(n)
        result = forward_realized_var(close, window=window)
        assert result.notna().sum() == n - window, (
            f"Expected {n - window} non-NaN rows, got {result.notna().sum()}"
        )

    def test_value_is_mean_of_squared_log_returns(self) -> None:
        """Row t value equals mean of next `window` squared log returns."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        dates = pd.date_range("2022-01-01", periods=len(prices), freq="D", tz="UTC")
        close = pd.Series(prices, index=dates, name="close")
        result = forward_realized_var(close, window=3)

        log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
        sq_returns = log_returns ** 2

        # Row 0 should be mean of sq_returns[0], sq_returns[1], sq_returns[2]
        expected_row0 = np.mean(sq_returns[:3])
        assert abs(float(result.iloc[0]) - expected_row0) < 1e-14, (
            f"Row 0: expected {expected_row0}, got {result.iloc[0]}"
        )

    def test_no_nan_fill_with_zero(self) -> None:
        """Tail NaNs must NOT be filled with zero."""
        close = self._make_close(20)
        result = forward_realized_var(close, window=5)
        for i in range(-5, 0):
            assert np.isnan(result.iloc[i]), f"Row {i} should be NaN, not filled"
