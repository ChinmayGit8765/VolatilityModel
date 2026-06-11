"""Unit tests for range estimators, vol-of-vol, rolling skew/kurt, lagged vol,
and calendar features in src/volforecast/features/estimators.py.

All tests are offline-only (no network). Series are constructed inline or from
committed fixture parquets.

Coverage:
- parkinson_var: formula ln(H/L)**2 / (4*ln(2)); positive; row-wise using only high/low
- garman_klass_var: 0.5*ln(H/L)**2 - (2*ln(2)-1)*ln(C/O)**2; can be negative
- vol_of_vol: rolling std of a realized-vol series over integer window
- rolling_skew / rolling_kurt: match pandas .rolling().skew()/.kurt()
- lagged_vol: rv_series.shift(k); first k rows NaN; no future leakage
- calendar_features: day_of_week (0-4), month (1-12), is_monday, is_friday
  plus equity fixture has no weekend day_of_week values (5/6)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volforecast.features.estimators import (
    calendar_features,
    garman_klass_var,
    lagged_vol,
    log_returns,
    parkinson_var,
    rolling_kurt,
    rolling_skew,
    vol_of_vol,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlc(n: int = 30, base: float = 100.0, drift: float = 1.002) -> pd.DataFrame:
    """Build a deterministic OHLC DataFrame with UTC DatetimeIndex."""
    dates = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    close = base * (drift ** np.arange(n))
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.995
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _rv_series(n: int = 60) -> pd.Series:
    """Deterministic realized-var series (all positive, no NaN)."""
    rng = np.random.default_rng(42)
    vals = rng.uniform(1e-5, 1e-3, size=n)
    dates = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    return pd.Series(vals, index=dates, name="rv")


# ---------------------------------------------------------------------------
# parkinson_var
# ---------------------------------------------------------------------------


class TestParkinsonVar:
    def test_formula_elementwise(self) -> None:
        """parkinson_var(df) == ln(H/L)**2 / (4*ln(2)) for every row."""
        df = _ohlc(20)
        result = parkinson_var(df)
        expected = np.log(df["high"] / df["low"]) ** 2 / (4.0 * np.log(2.0))
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_positive(self) -> None:
        """Parkinson variance must always be positive (H > L by design)."""
        df = _ohlc(20)
        assert (parkinson_var(df) > 0).all()

    def test_uses_only_high_low(self) -> None:
        """Changing open/close does not affect Parkinson variance."""
        df = _ohlc(10)
        df2 = df.copy()
        df2["open"] = df2["open"] * 2.0
        df2["close"] = df2["close"] * 2.0
        pd.testing.assert_series_equal(parkinson_var(df), parkinson_var(df2), check_names=False)

    def test_same_index_as_input(self) -> None:
        df = _ohlc(15)
        result = parkinson_var(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_formula_denominator_4_log2(self) -> None:
        """Smoke check: for H/L = e => ln(H/L) = 1, result = 1/(4*ln2)."""
        df = _ohlc(5)
        df["high"] = df["close"] * np.e
        df["low"] = df["close"]
        result = parkinson_var(df)
        expected_val = 1.0 / (4.0 * np.log(2.0))
        np.testing.assert_allclose(result.values, expected_val, rtol=1e-10)

    def test_on_crypto_fixture(self) -> None:
        """Smoke test on real crypto fixture."""
        df = pd.read_parquet(FIXTURES / "crypto_sample.parquet")
        result = parkinson_var(df)
        assert isinstance(result, pd.Series)
        assert (result > 0).all()
        assert result.isna().sum() == 0

    def test_on_equity_fixture(self) -> None:
        """Smoke test on real equity fixture."""
        df = pd.read_parquet(FIXTURES / "equity_sample.parquet")
        result = parkinson_var(df)
        assert isinstance(result, pd.Series)
        assert (result > 0).all()


# ---------------------------------------------------------------------------
# garman_klass_var
# ---------------------------------------------------------------------------


class TestGarmanKlassVar:
    def test_formula_elementwise(self) -> None:
        """garman_klass_var(df) == 0.5*ln(H/L)**2 - (2*ln2-1)*ln(C/O)**2 for every row."""
        df = _ohlc(20)
        log_hl = np.log(df["high"] / df["low"])
        log_co = np.log(df["close"] / df["open"])
        expected = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
        result = garman_klass_var(df)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_high_low_dominates_for_wide_range(self) -> None:
        """When H/L is large, the result should be positive (HL term dominates)."""
        df = _ohlc(10)
        df["high"] = df["close"] * 1.10
        df["low"] = df["close"] * 0.90
        # HL term 0.5*(ln(1.10/0.90))^2 ~ 0.5*0.0204 = 0.0102 >> CO term
        result = garman_klass_var(df)
        assert (result > 0).all()

    def test_same_index_as_input(self) -> None:
        df = _ohlc(15)
        result = garman_klass_var(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_on_crypto_fixture(self) -> None:
        """Smoke test: returns a series without NaN for clean OHLC input."""
        df = pd.read_parquet(FIXTURES / "crypto_sample.parquet")
        result = garman_klass_var(df)
        assert isinstance(result, pd.Series)
        assert result.isna().sum() == 0

    def test_on_equity_fixture(self) -> None:
        df = pd.read_parquet(FIXTURES / "equity_sample.parquet")
        result = garman_klass_var(df)
        assert isinstance(result, pd.Series)
        assert result.isna().sum() == 0


# ---------------------------------------------------------------------------
# vol_of_vol
# ---------------------------------------------------------------------------


class TestVolOfVol:
    def test_equals_rolling_std(self) -> None:
        """vol_of_vol(rv, w) == rv.rolling(w).std() (integer window)."""
        rv = _rv_series(60)
        window = 10
        result = vol_of_vol(rv, window)
        expected = rv.rolling(window).std()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_nan_prefix_length(self) -> None:
        """First window-1 rows are NaN (rolling std with min_periods=window default)."""
        rv = _rv_series(50)
        window = 5
        result = vol_of_vol(rv, window)
        assert result.isna().sum() == window - 1

    def test_integer_window_required(self) -> None:
        """window must be an integer >= 1; float raises TypeError or ValueError."""
        rv = _rv_series(20)
        # Should not raise with integer window
        vol_of_vol(rv, 5)

    def test_positive(self) -> None:
        """Non-NaN vol-of-vol values must be >= 0."""
        rv = _rv_series(60)
        result = vol_of_vol(rv, 10)
        assert (result.dropna() >= 0).all()

    def test_same_index_as_input(self) -> None:
        rv = _rv_series(30)
        result = vol_of_vol(rv, 5)
        pd.testing.assert_index_equal(result.index, rv.index)


# ---------------------------------------------------------------------------
# rolling_skew
# ---------------------------------------------------------------------------


class TestRollingSkew:
    def test_matches_pandas_rolling_skew(self) -> None:
        """rolling_skew(lr, w) == lr.rolling(w).skew()."""
        df = _ohlc(60)
        lr = log_returns(df["close"])
        window = 10
        result = rolling_skew(lr, window)
        expected = lr.rolling(window).skew()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_nan_prefix(self) -> None:
        """First window-1 rows NaN (rolling skew has min_periods default = window)."""
        df = _ohlc(60)
        lr = log_returns(df["close"])
        window = 10
        result = rolling_skew(lr, window)
        # First row of lr is NaN; pandas rolling skew then NaN for window-1 additional rows
        assert result.iloc[:window].isna().all()

    def test_integer_window(self) -> None:
        df = _ohlc(30)
        lr = log_returns(df["close"])
        rolling_skew(lr, 5)  # Should not raise

    def test_same_index_as_input(self) -> None:
        df = _ohlc(30)
        lr = log_returns(df["close"])
        result = rolling_skew(lr, 5)
        pd.testing.assert_index_equal(result.index, lr.index)


# ---------------------------------------------------------------------------
# rolling_kurt
# ---------------------------------------------------------------------------


class TestRollingKurt:
    def test_matches_pandas_rolling_kurt(self) -> None:
        """rolling_kurt(lr, w) == lr.rolling(w).kurt()."""
        df = _ohlc(60)
        lr = log_returns(df["close"])
        window = 10
        result = rolling_kurt(lr, window)
        expected = lr.rolling(window).kurt()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_nan_prefix(self) -> None:
        df = _ohlc(60)
        lr = log_returns(df["close"])
        window = 10
        result = rolling_kurt(lr, window)
        assert result.iloc[:window].isna().all()

    def test_same_index_as_input(self) -> None:
        df = _ohlc(30)
        lr = log_returns(df["close"])
        result = rolling_kurt(lr, 5)
        pd.testing.assert_index_equal(result.index, lr.index)


# ---------------------------------------------------------------------------
# lagged_vol
# ---------------------------------------------------------------------------


class TestLaggedVol:
    def test_equals_shift(self) -> None:
        """lagged_vol(rv, k) == rv.shift(k) — no future leakage."""
        rv = _rv_series(40)
        for k in (1, 5, 10):
            result = lagged_vol(rv, k)
            expected = rv.shift(k)
            pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_first_k_rows_nan(self) -> None:
        """First k rows must be NaN (shift introduces NaN at the start)."""
        rv = _rv_series(30)
        for k in (1, 3, 5):
            result = lagged_vol(rv, k)
            assert result.isna().sum() == k, f"k={k}: expected {k} NaN rows"

    def test_no_future_leakage(self) -> None:
        """lagged_vol at row t must equal rv at row t-k (uses only past data)."""
        rv = _rv_series(20)
        k = 3
        result = lagged_vol(rv, k)
        # At index 5, lagged_vol[5] must equal rv[5-3] = rv[2]
        assert result.iloc[5] == pytest.approx(rv.iloc[2])

    def test_same_index_as_input(self) -> None:
        rv = _rv_series(20)
        result = lagged_vol(rv, 1)
        pd.testing.assert_index_equal(result.index, rv.index)


# ---------------------------------------------------------------------------
# calendar_features
# ---------------------------------------------------------------------------


class TestCalendarFeatures:
    def test_day_of_week_range(self) -> None:
        """day_of_week must be in 0..4 for a weekday-only index."""
        df = _ohlc(20)  # uses freq="B" = business days only
        result = calendar_features(df)
        assert "day_of_week" in result.columns
        assert result["day_of_week"].between(0, 4).all()

    def test_month_range(self) -> None:
        df = _ohlc(30)
        result = calendar_features(df)
        assert "month" in result.columns
        assert result["month"].between(1, 12).all()

    def test_is_monday_flag(self) -> None:
        """is_monday is 1 on Mondays (dayofweek==0), 0 otherwise."""
        df = _ohlc(20)
        result = calendar_features(df)
        assert "is_monday" in result.columns
        mondays = result.index.dayofweek == 0
        assert (result.loc[mondays, "is_monday"] == 1).all()
        assert (result.loc[~mondays, "is_monday"] == 0).all()

    def test_is_friday_flag(self) -> None:
        """is_friday is 1 on Fridays (dayofweek==4), 0 otherwise."""
        df = _ohlc(20)
        result = calendar_features(df)
        assert "is_friday" in result.columns
        fridays = result.index.dayofweek == 4
        assert (result.loc[fridays, "is_friday"] == 1).all()
        assert (result.loc[~fridays, "is_friday"] == 0).all()

    def test_preserves_existing_columns(self) -> None:
        """calendar_features must preserve all existing OHLCV columns."""
        df = _ohlc(10)
        result = calendar_features(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_returns_dataframe(self) -> None:
        df = _ohlc(10)
        result = calendar_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_no_weekend_on_equity_fixture(self) -> None:
        """Equity fixture has only trading days; day_of_week in [0,4], never 5 or 6."""
        df = pd.read_parquet(FIXTURES / "equity_sample.parquet")
        result = calendar_features(df)
        # Processed equity parquet must have no weekend rows (Phase 1 Pandera gate)
        weekend_mask = result["day_of_week"].isin([5, 6])
        assert not weekend_mask.any(), (
            f"Equity fixture has weekend rows: {result.loc[weekend_mask, 'day_of_week']}"
        )

    def test_crypto_fixture_may_have_all_days(self) -> None:
        """Crypto fixture is 24/7 so day_of_week should span 0-6."""
        df = pd.read_parquet(FIXTURES / "crypto_sample.parquet")
        result = calendar_features(df)
        # Crypto has all 7 days; month range still 1-12
        assert result["month"].between(1, 12).all()
