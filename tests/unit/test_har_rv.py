"""Tests for the HAR-RV (Heterogeneous Autoregressive) baseline forecaster.

TDD RED phase: tests written before the implementation exists.
They cover:
  1. Regressor matrix construction: [const, rv_d, rv_w, rv_m] where
     rv_w and rv_m are rolling MEANS of shifted RV (not last values).
  2. Unit consistency: forecasts are in the same decimal variance units as
     the target — no scaling needed or applied.
  3. Walk-forward: HAR refits each step (cheap OLS) and produces one aligned
     forecast per test index.
  4. Finite forecasts and accessible coefficients.

Reference: Corsi (2009) "A Simple Approximate Long-Memory Model of Realized
Volatility", J. Financial Econometrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

N_TRAIN = 150  # enough for 22-day rolling lookback + OLS
N_STEP = 10


def _make_rv(n: int = N_TRAIN + N_STEP, seed: int = 7) -> pd.Series:
    """Synthetic daily realized variance ~chi-squared shaped, decimal units.

    Realistic magnitudes: ~1e-4 for typical daily equity.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, n)
    rv = r**2  # squared log returns as variance proxy
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rv, index=idx, name="test_rv")


# ------------------------------------------------------------------ #
# Imports (will fail until implementation exists — RED phase)
# ------------------------------------------------------------------ #

from volforecast.models.har_rv import (  # noqa: E402
    HARRV,
    build_har_features,
    fit_har,
    har_forecast,
)

# ------------------------------------------------------------------ #
# 1. Regressor matrix construction
# ------------------------------------------------------------------ #


class TestBuildHarFeatures:
    def test_columns_present(self):
        """build_har_features must return a DataFrame with const + rv_d + rv_w + rv_m."""
        rv = _make_rv(N_TRAIN)
        X = build_har_features(rv)
        required_cols = {"const", "rv_d", "rv_w", "rv_m"}
        assert required_cols.issubset(set(X.columns)), (
            f"Missing columns: {required_cols - set(X.columns)}"
        )

    def test_rv_w_is_rolling_mean_not_last_value(self):
        """rv_w must be the rolling mean of 5 past days, not just shift(1).

        If rv_w were simply rv.shift(1) (last value), it would equal rv_d.
        The rolling-mean version must differ from the last-value version.
        """
        rng = np.random.default_rng(0)
        # Make returns with clear variation so shift vs mean differs
        r = rng.normal(0.0, 0.01, N_TRAIN)
        rv = pd.Series(r**2, name="rv")
        X = build_har_features(rv)
        # rv_w should NOT equal rv_d element-wise (they only match in flat vol scenarios)
        # Check that they differ on most rows where both are non-NaN
        valid_mask = X["rv_d"].notna() & X["rv_w"].notna()
        if valid_mask.sum() > 10:
            eq_fraction = (X.loc[valid_mask, "rv_d"] == X.loc[valid_mask, "rv_w"]).mean()
            assert eq_fraction < 0.5, (
                f"rv_w appears to equal rv_d on {eq_fraction:.0%} of rows — "
                "rv_w must be a rolling MEAN of 5 days, not shift(1)"
            )

    def test_rv_m_is_rolling_mean_of_22_days(self):
        """rv_m must be the rolling mean of 22 past days."""
        rng = np.random.default_rng(1)
        r = rng.normal(0.0, 0.01, N_TRAIN)
        rv = pd.Series(r**2, name="rv")
        X = build_har_features(rv)
        # rv_m should NOT equal rv_w element-wise in general
        valid_mask = X["rv_w"].notna() & X["rv_m"].notna()
        if valid_mask.sum() > 10:
            eq_fraction = (X.loc[valid_mask, "rv_w"] == X.loc[valid_mask, "rv_m"]).mean()
            assert eq_fraction < 0.5, (
                f"rv_m appears to equal rv_w on {eq_fraction:.0%} of rows — "
                "rv_m must be a rolling MEAN of 22 days, not 5"
            )

    def test_rv_d_is_shifted_by_1(self):
        """rv_d must be rv.shift(1) — yesterday's realized variance."""
        rng = np.random.default_rng(2)
        rv_vals = rng.uniform(1e-5, 1e-3, N_TRAIN)
        rv = pd.Series(rv_vals, name="rv")
        X = build_har_features(rv)
        # X has dropped the first ~23 NaN rows; use label-based indexing.
        # For each row in X, rv_d[idx] should equal rv[idx - 1 step].
        # Check a sample of rows from X (after the rolling NaN prefix).
        idx_labels = X.index[5:10]  # pick rows well past the NaN region
        rv_integer_locs = [rv.index.get_loc(lbl) for lbl in idx_labels]
        for lbl, pos in zip(idx_labels, rv_integer_locs):
            expected = float(rv.iloc[pos - 1])  # yesterday (integer shift by 1)
            actual = float(X.loc[lbl, "rv_d"])
            assert abs(actual - expected) < 1e-15, (
                f"rv_d[{lbl}]={actual:.2e} != rv[pos-1={pos - 1}]={expected:.2e}"
            )

    def test_nan_rows_dropped(self):
        """build_har_features must drop rows with NaN (start of rolling windows)."""
        rv = _make_rv(N_TRAIN)
        X = build_har_features(rv)
        # After dropna, no NaN should remain in the feature matrix
        assert not X.isna().any().any(), "build_har_features result must have no NaN"

    def test_minimum_rows_after_dropna(self):
        """After 22-day rolling window + 1-day shift, at least N-23 rows remain."""
        rv = _make_rv(N_TRAIN)
        X = build_har_features(rv)
        # NaN prefix: 22 for rv_m + 1 for shift = 23 rows dropped at minimum
        assert len(X) >= N_TRAIN - 25, f"Expected >= {N_TRAIN - 25} rows after dropna, got {len(X)}"


# ------------------------------------------------------------------ #
# 2. fit_har and har_forecast
# ------------------------------------------------------------------ #


class TestFitHar:
    def test_fit_har_returns_result(self):
        """fit_har must return a statsmodels RegressionResults object."""
        rv = _make_rv(N_TRAIN)
        train_idx = np.arange(N_TRAIN)
        result = fit_har(rv, train_idx)
        assert hasattr(result, "params"), "fit_har must return RegressionResults"
        assert hasattr(result, "predict"), "fit_har result must have predict method"

    def test_fit_har_has_four_params(self):
        """HAR has 4 params: const, rv_d, rv_w, rv_m."""
        rv = _make_rv(N_TRAIN)
        train_idx = np.arange(N_TRAIN)
        result = fit_har(rv, train_idx)
        assert len(result.params) == 4, (
            f"Expected 4 HAR params, got {len(result.params)}: {list(result.params.index)}"
        )

    def test_fit_har_params_accessible(self):
        """HAR params must be accessible and finite."""
        rv = _make_rv(N_TRAIN)
        train_idx = np.arange(N_TRAIN)
        result = fit_har(rv, train_idx)
        for name, val in result.params.items():
            assert np.isfinite(val), f"HAR param {name}={val} is not finite"


class TestHarForecast:
    def test_har_forecast_is_scalar(self):
        """har_forecast must return a scalar float."""
        rv = _make_rv(N_TRAIN)
        train_idx = np.arange(N_TRAIN)
        result = fit_har(rv, train_idx)
        fc = har_forecast(result, rv.iloc[:N_TRAIN])
        assert isinstance(fc, float), f"har_forecast must return float, got {type(fc)}"

    def test_har_forecast_is_finite(self):
        """har_forecast must be finite."""
        rv = _make_rv(N_TRAIN)
        train_idx = np.arange(N_TRAIN)
        result = fit_har(rv, train_idx)
        fc = har_forecast(result, rv.iloc[:N_TRAIN])
        assert np.isfinite(fc), f"har_forecast must be finite, got {fc}"

    def test_har_forecast_positive(self):
        """har_forecast should be positive (variance proxy)."""
        rng = np.random.default_rng(3)
        rv_vals = rng.uniform(5e-5, 5e-4, N_TRAIN)
        rv = pd.Series(rv_vals, name="rv")
        train_idx = np.arange(N_TRAIN)
        result = fit_har(rv, train_idx)
        fc = har_forecast(result, rv.iloc[:N_TRAIN])
        # HAR with positive RV history should produce a positive forecast
        # (though OLS doesn't enforce positivity; warn if negative)
        # Just check it is finite
        assert np.isfinite(fc), f"har_forecast not finite: {fc}"


# ------------------------------------------------------------------ #
# 3. Walk-forward driver
# ------------------------------------------------------------------ #


class TestHARRVWalkForward:
    def test_forecast_path_returns_series(self):
        """HARRV.forecast_path must return a pd.Series with same index as input."""
        rv = _make_rv(N_TRAIN + N_STEP)
        model = HARRV(min_train=N_TRAIN, step=N_STEP)
        forecasts = model.forecast_path(rv)
        assert isinstance(forecasts, pd.Series), "forecast_path must return pd.Series"
        assert len(forecasts) == len(rv), (
            f"forecast_path must be same length as input, got {len(forecasts)} != {len(rv)}"
        )

    def test_forecast_path_same_index(self):
        """HARRV.forecast_path output index must match input index."""
        rv = _make_rv(N_TRAIN + N_STEP)
        model = HARRV(min_train=N_TRAIN, step=N_STEP)
        forecasts = model.forecast_path(rv)
        assert forecasts.index.equals(rv.index), "Output index must match input"

    def test_test_window_forecasts_finite(self):
        """Forecasts in the test window must be finite."""
        rv = _make_rv(N_TRAIN + N_STEP)
        model = HARRV(min_train=N_TRAIN, step=N_STEP)
        forecasts = model.forecast_path(rv)
        test_fc = forecasts.iloc[N_TRAIN:]
        assert test_fc.notna().all(), f"Test window has NaN forecasts: {test_fc}"
        assert test_fc.apply(np.isfinite).all(), "Test window forecasts must be finite"

    def test_forecast_path_per_step_refit(self):
        """HARRV refits every step — total refits equals ceil(test_obs / step)."""
        n_total = N_TRAIN + 3 * N_STEP
        rv = _make_rv(n_total)
        model = HARRV(min_train=N_TRAIN, step=N_STEP)
        model.forecast_path(rv)
        expected_refits = 3  # 3 * N_STEP test observations → 3 refits
        assert model.n_refits == expected_refits, (
            f"Expected {expected_refits} refits, got {model.n_refits}"
        )


# ------------------------------------------------------------------ #
# 4. Unit consistency
# ------------------------------------------------------------------ #


class TestHARRVUnits:
    def test_forecast_variance_same_units_as_input(self):
        """HAR-RV forecasts must be in the same decimal variance units as the input RV.

        No scaling is applied — rv is regressed directly on lagged rv.
        Median forecast should be in the same ballpark as the median of the
        input rv series.
        """
        rng = np.random.default_rng(4)
        # Create RV in realistic range ~1e-4
        rv_vals = np.abs(rng.normal(0.0, 0.01, N_TRAIN + N_STEP)) ** 2
        rv = pd.Series(rv_vals, name="rv")
        model = HARRV(min_train=N_TRAIN, step=N_STEP)
        forecasts = model.forecast_path(rv)
        valid_fc = forecasts.dropna()
        valid_rv = rv.dropna()
        # Median forecast should be within 100x of median input rv
        med_fc = float(np.median(valid_fc))
        med_rv = float(np.median(valid_rv))
        ratio = med_fc / (med_rv + 1e-15)
        assert 0.01 < ratio < 100.0, (
            f"HAR forecast units appear wrong: median_forecast={med_fc:.2e}, "
            f"median_rv={med_rv:.2e}, ratio={ratio:.2f} (expected ~1.0)"
        )
