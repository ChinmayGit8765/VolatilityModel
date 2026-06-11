"""Unit tests for src/volforecast/models/ewma.py.

Tests verify:
- No-look-ahead: forecast at t uses only returns < t (shift(1) guarantee)
- Lambda configurability and defaults (0.94)
- Forecast alignment to test indices from walk_forward_splits
- Positivity and finiteness of forecasts
- EWMA reuses estimators.py (not re-derived) — import chain tested implicitly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.eval.harness import walk_forward_splits
from volforecast.features.estimators import EWMA_LAMBDA, ewma_variance, log_returns
from volforecast.models.ewma import EWMA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _close(prices: list[float]) -> pd.Series:
    """Build a UTC-indexed daily close Series."""
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="D", tz="UTC")
    return pd.Series(prices, index=dates, name="close", dtype="float64")


def _synthetic_close(n: int, seed: int = 42) -> pd.Series:
    """Build a synthetic close price series of length n."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(prices, index=dates, name="close", dtype="float64")


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestEWMAConstruction:
    def test_default_lambda_is_0_94(self) -> None:
        """EWMA() with no args must use EWMA_LAMBDA == 0.94."""
        model = EWMA()
        assert model.lam == EWMA_LAMBDA
        assert model.lam == 0.94

    def test_custom_lambda(self) -> None:
        model = EWMA(lam=0.97)
        assert model.lam == 0.97

    def test_invalid_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="lam must be in"):
            EWMA(lam=0.0)
        with pytest.raises(ValueError, match="lam must be in"):
            EWMA(lam=1.0)
        with pytest.raises(ValueError, match="lam must be in"):
            EWMA(lam=1.5)

    def test_lambda_configurable_in_range(self) -> None:
        for lam in (0.5, 0.9, 0.94, 0.97, 0.99):
            model = EWMA(lam=lam)
            assert model.lam == lam


# ---------------------------------------------------------------------------
# No-look-ahead guarantee
# ---------------------------------------------------------------------------


class TestNoLookAhead:
    """Forecast at t must use only returns strictly before t."""

    def test_forecast_is_shift_1_of_ewma(self) -> None:
        """forecast_path(lr) == ewma_variance(lr, 0.94).shift(1) exactly."""
        close = _synthetic_close(50)
        lr = log_returns(close)
        model = EWMA()
        forecasts = model.forecast_path(lr)
        expected = ewma_variance(lr, lam=0.94).shift(1)
        pd.testing.assert_series_equal(forecasts, expected)

    def test_forecast_at_t_equals_ewma_at_t_minus_1(self) -> None:
        """For each t >= 2, forecast[t] == ewma_variance(lr)[t-1]."""
        close = _synthetic_close(30)
        lr = log_returns(close)
        model = EWMA()
        forecasts = model.forecast_path(lr)
        ewma_vals = ewma_variance(lr, lam=0.94)

        for t in range(2, len(lr)):
            if not np.isnan(forecasts.iloc[t]) and not np.isnan(ewma_vals.iloc[t - 1]):
                assert abs(forecasts.iloc[t] - ewma_vals.iloc[t - 1]) < 1e-14, (
                    f"At t={t}: forecast={forecasts.iloc[t]}, "
                    f"ewma[t-1]={ewma_vals.iloc[t-1]}"
                )

    def test_first_two_forecasts_are_nan(self) -> None:
        """Positions 0 and 1 must be NaN (shift(1) of NaN at 0 propagates)."""
        close = _synthetic_close(20)
        lr = log_returns(close)
        model = EWMA()
        forecasts = model.forecast_path(lr)
        assert np.isnan(forecasts.iloc[0]), "forecast[0] must be NaN"
        assert np.isnan(forecasts.iloc[1]), "forecast[1] must be NaN after shift"

    def test_no_future_data_used(self) -> None:
        """Mutating returns at t+1..n does not affect forecast at t."""
        close = _synthetic_close(30)
        lr = log_returns(close)
        model = EWMA()
        forecasts_orig = model.forecast_path(lr)

        # Replace returns at positions t_pivot+1..n with large values
        t_pivot = 15
        lr_modified = lr.copy()
        lr_modified.iloc[t_pivot + 1 :] = 1.0  # obviously wrong values
        forecasts_modified = model.forecast_path(lr_modified)

        # forecast at t_pivot must be identical (uses only data < t_pivot+1)
        assert abs(forecasts_orig.iloc[t_pivot] - forecasts_modified.iloc[t_pivot]) < 1e-14, (
            "forecast at t_pivot changed when future data was modified — look-ahead detected"
        )


# ---------------------------------------------------------------------------
# Positivity and finiteness
# ---------------------------------------------------------------------------


class TestForecastQuality:
    def test_forecasts_positive_and_finite(self) -> None:
        """Non-NaN forecasts must be > 0 and finite."""
        close = _synthetic_close(100)
        lr = log_returns(close)
        model = EWMA()
        forecasts = model.forecast_path(lr)
        non_nan = forecasts.dropna()
        assert (non_nan > 0).all(), "All non-NaN forecasts must be positive"
        assert np.isfinite(non_nan.values).all(), "All non-NaN forecasts must be finite"

    def test_same_index_as_input(self) -> None:
        close = _synthetic_close(50)
        lr = log_returns(close)
        model = EWMA()
        forecasts = model.forecast_path(lr)
        pd.testing.assert_index_equal(forecasts.index, lr.index)

    def test_dtype_is_float64(self) -> None:
        close = _synthetic_close(20)
        lr = log_returns(close)
        model = EWMA()
        forecasts = model.forecast_path(lr)
        assert forecasts.dtype == np.float64


# ---------------------------------------------------------------------------
# Walk-forward harness compatibility
# ---------------------------------------------------------------------------


class TestWalkForwardCompatibility:
    """EWMA forecast_path integrates correctly with walk_forward_splits."""

    def test_one_forecast_per_test_index(self) -> None:
        """Each test index from the harness must correspond to a valid forecast."""
        n = 300
        close = _synthetic_close(n)
        lr = log_returns(close)
        model = EWMA()

        forecasts = model.forecast_path(lr)

        splits = list(walk_forward_splits(n, min_train=252, step=21, horizon=1))
        assert len(splits) >= 1, "Need at least 1 split for this test"

        for split in splits:
            test_forecasts = forecasts.iloc[split.test_idx]
            # All test forecasts must be non-NaN (test positions are well beyond
            # the first two NaN positions from shift(1))
            assert test_forecasts.notna().all(), (
                f"NaN forecast found in test window {split.test_idx}"
            )

    def test_forecasts_cover_all_test_indices_across_folds(self) -> None:
        """Collecting test forecasts across all folds yields n_test_total forecasts."""
        n = 300
        close = _synthetic_close(n)
        lr = log_returns(close)
        model = EWMA()
        forecasts = model.forecast_path(lr)

        all_test_forecasts = []
        all_test_targets = []
        from volforecast.features.target import compute_target  # noqa: PLC0415

        target = compute_target(close)

        splits = list(walk_forward_splits(n, min_train=252, step=21, horizon=1))
        for split in splits:
            test_fc = forecasts.iloc[split.test_idx]
            test_tgt = target.iloc[split.test_idx]
            # Drop NaN targets (end-of-series — Pitfall 4)
            valid_mask = test_tgt.notna()
            all_test_forecasts.extend(test_fc[valid_mask].tolist())
            all_test_targets.extend(test_tgt[valid_mask].tolist())

        n_total = len(all_test_forecasts)
        assert n_total > 0, "Must collect at least 1 valid forecast/target pair"
        assert len(all_test_forecasts) == len(all_test_targets)

    def test_aligned_to_test_index_date(self) -> None:
        """forecast.iloc[test_idx[0]] must correspond to the correct date in the index."""
        n = 300
        close = _synthetic_close(n)
        lr = log_returns(close)
        model = EWMA()
        forecasts = model.forecast_path(lr)

        splits = list(walk_forward_splits(n, min_train=252, step=21, horizon=1))
        split = splits[0]

        # Verify that slicing by integer position gives same VALUES as slicing by date
        # (ignore freq metadata — integer slicing drops freq even on a regular index)
        test_by_pos = forecasts.iloc[split.test_idx]
        test_dates = lr.index[split.test_idx]
        test_by_date = forecasts.loc[test_dates]
        np.testing.assert_array_equal(test_by_pos.values, test_by_date.values)

    def test_lambda_affects_forecasts(self) -> None:
        """Different lambda values must produce different forecasts."""
        n = 100
        close = _synthetic_close(n)
        lr = log_returns(close)

        fc_94 = EWMA(lam=0.94).forecast_path(lr)
        fc_97 = EWMA(lam=0.97).forecast_path(lr)

        # They should differ — higher lambda gives more weight to old variance
        assert not fc_94.equals(fc_97), (
            "lam=0.94 and lam=0.97 should produce different forecasts"
        )
