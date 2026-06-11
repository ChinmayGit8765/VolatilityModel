"""Tests for the GARCH(1,1) baseline forecaster.

TDD RED phase: these tests are written before the implementation exists.
They cover:
  1. Scaling path: model multiplies returns by GARCH_SCALE=100.0 before fitting.
  2. Convergence + stationarity assertions on every refit.
  3. Scale inversion: forecast variance is divided by GARCH_SCALE**2 exactly once
     → magnitude ~1e-4..1e-2 for realistic returns, NOT ~1..100.
  4. Multi-step-per-refit: a single refit covers all step test days via
     forecast(horizon=step), not one fit per day (Pitfall 5).
  5. Fallback: convergence failure increments fallback_count; no garbage emitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

RNG = np.random.default_rng(12345)
N_TRAIN = 300  # enough for GARCH convergence on synthetic data
N_STEP = 21


def _make_returns(n: int = N_TRAIN + N_STEP, seed: int = 42) -> pd.Series:
    """Synthetic decimal log returns ~N(0, 0.01) — realistic daily equity."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(r, index=idx, name="test_asset")


# ------------------------------------------------------------------ #
# Imports (will fail until implementation exists)
# ------------------------------------------------------------------ #

from volforecast.models.garch import (  # noqa: E402
    GARCH,
    GARCH_SCALE,
    GarchConvergenceError,
    GarchFitError,
    GarchNonStationaryError,
    fit_garch,
)

# ------------------------------------------------------------------ #
# 1. Module constant
# ------------------------------------------------------------------ #


class TestGarchConstants:
    def test_garch_scale_is_100(self):
        """GARCH_SCALE must equal 100.0 — module-level constant."""
        assert GARCH_SCALE == 100.0, f"Expected GARCH_SCALE=100.0, got {GARCH_SCALE}"


# ------------------------------------------------------------------ #
# 2. fit_garch: scaling path
# ------------------------------------------------------------------ #


class TestFitGarch:
    def test_fit_garch_returns_result(self):
        """fit_garch should return an ARCHModelResult without error."""
        r = _make_returns(N_TRAIN)
        res = fit_garch(r)
        # Should have params attribute (ARCHModelResult)
        assert hasattr(res, "params"), "fit_garch must return ARCHModelResult"
        assert "alpha[1]" in res.params.index, "Result must have alpha[1] param"
        assert "beta[1]" in res.params.index, "Result must have beta[1] param"

    def test_convergence_flag_zero(self):
        """fit_garch must assert convergence_flag == 0 on a well-conditioned series."""
        r = _make_returns(N_TRAIN)
        res = fit_garch(r)
        assert res.convergence_flag == 0, f"Expected convergence_flag=0, got {res.convergence_flag}"

    def test_stationarity_alpha_beta_lt_1(self):
        """fit_garch must assert alpha + beta < 1.0 on a stationary series."""
        r = _make_returns(N_TRAIN)
        res = fit_garch(r)
        alpha = res.params["alpha[1]"]
        beta = res.params["beta[1]"]
        assert alpha + beta < 1.0, f"GARCH non-stationary: alpha+beta={alpha + beta:.6f}"


# ------------------------------------------------------------------ #
# 3. Forecast variance in decimal units
# ------------------------------------------------------------------ #


class TestGarchForecastUnits:
    def test_forecast_variance_is_decimal_scale(self):
        """GARCH forecast variance must be in decimal units (~1e-4..1e-2).

        If GARCH_SCALE**2 inversion is NOT applied, the forecast variance
        would be in (100*return)^2 units — roughly 10,000x too large (~1.0).
        Verify that the returned value is well below 1.0.
        """
        r = _make_returns(N_TRAIN)
        model = GARCH()
        forecasts = model.forecast_path(r)
        # Drop NaN at start (need training window)
        valid = forecasts.dropna()
        assert len(valid) > 0, "forecast_path must produce non-NaN forecasts"
        # All forecasts must be << 1.0 (decimal variance, not percent-squared)
        assert valid.max() < 1.0, (
            f"Forecast variance max={valid.max():.4f} is too large — "
            "GARCH_SCALE**2 inversion likely missing or applied twice"
        )
        # All forecasts must be positive
        assert (valid > 0).all(), "Forecast variance must be strictly positive"
        # Realistic lower bound: daily variance is at least 1e-8 for any asset
        assert valid.min() > 1e-8, f"Forecast variance min={valid.min():.2e} is implausibly small"

    def test_forecast_variance_realistic_magnitude(self):
        """Forecast variance should be in realistic daily equity range ~1e-5..5e-3."""
        r = _make_returns(N_TRAIN, seed=99)
        model = GARCH()
        forecasts = model.forecast_path(r)
        valid = forecasts.dropna()
        med = float(np.median(valid))
        # For N(0, 0.01) returns, variance ~1e-4; GARCH should be near that
        assert 1e-6 < med < 1e-1, (
            f"Median forecast variance={med:.2e} — expected ~1e-4 for N(0,0.01) returns"
        )


# ------------------------------------------------------------------ #
# 4. Multi-step per refit (Pitfall 5)
# ------------------------------------------------------------------ #


class TestGarchMultiStepPerRefit:
    def test_single_refit_covers_all_step_days(self):
        """A single GARCH refit must produce step forecasts, not 1."""
        r = _make_returns(N_TRAIN + N_STEP)
        model = GARCH(step=N_STEP)
        forecasts = model.forecast_path(r)
        # forecast_path returns a Series aligned with input dates
        # The number of valid (non-NaN) forecasts must equal the test window size
        # (at a minimum — for the first refit window there should be N_STEP values)
        valid = forecasts.dropna()
        assert len(valid) >= N_STEP, (
            f"Expected >= {N_STEP} daily forecasts from first refit, got {len(valid)}"
        )

    def test_forecasts_cover_every_test_index(self):
        """forecast_path must return one forecast per input index (no gaps)."""
        r = _make_returns(N_TRAIN + 2 * N_STEP)
        model = GARCH(step=N_STEP, min_train=N_TRAIN)
        forecasts = model.forecast_path(r)
        # The returned series must have the same index as the input
        assert len(forecasts) == len(r), (
            f"forecast_path must return same-length series, got {len(forecasts)} != {len(r)}"
        )


# ------------------------------------------------------------------ #
# 5. Fallback path
# ------------------------------------------------------------------ #


class TestGarchFallback:
    def test_fallback_count_accessible(self):
        """GARCH class must expose fallback_count after forecast_path call."""
        r = _make_returns(N_TRAIN + N_STEP)
        model = GARCH()
        model.forecast_path(r)
        assert hasattr(model, "fallback_count"), (
            "GARCH model must expose fallback_count attribute after forecast_path"
        )

    def test_fallback_count_is_int(self):
        """fallback_count must be a non-negative integer."""
        r = _make_returns(N_TRAIN + N_STEP)
        model = GARCH()
        model.forecast_path(r)
        assert isinstance(model.fallback_count, int), "fallback_count must be int"
        assert model.fallback_count >= 0, "fallback_count must be >= 0"

    def test_no_nan_forecasts_in_test_window(self):
        """forecast_path must not emit NaN forecasts for test-window positions."""
        r = _make_returns(N_TRAIN + N_STEP)
        model = GARCH(min_train=N_TRAIN, step=N_STEP)
        forecasts = model.forecast_path(r)
        # Positions from N_TRAIN onward should have valid forecasts
        test_forecasts = forecasts.iloc[N_TRAIN:]
        assert test_forecasts.notna().all(), (
            f"forecast_path emitted NaN in test window: {test_forecasts}"
        )


# ------------------------------------------------------------------ #
# 5b. Dedicated fit-failure exceptions (WR-03)
# ------------------------------------------------------------------ #


class TestGarchFitExceptions:
    def test_exception_hierarchy(self):
        """WR-03: dedicated exceptions exist and share the GarchFitError base.

        Plain ``assert`` statements are stripped under ``python -O``; the
        enforcement must be regular raises of these dedicated types.
        """
        assert issubclass(GarchConvergenceError, GarchFitError)
        assert issubclass(GarchNonStationaryError, GarchFitError)
        assert issubclass(GarchFitError, RuntimeError)
        assert not issubclass(GarchFitError, AssertionError), (
            "fit-quality enforcement must not rely on AssertionError (stripped under python -O)"
        )

    def test_convergence_failure_triggers_ewma_fallback(self, monkeypatch):
        """forecast_path catches GarchConvergenceError and falls back to EWMA
        (no prior fit) — value-level check against the EWMA fallback values."""
        import volforecast.models.garch as garch_mod
        from volforecast.features.estimators import ewma_variance

        def _always_fails(train_slice):
            raise GarchConvergenceError("forced convergence failure (test)")

        monkeypatch.setattr(garch_mod, "fit_garch", _always_fails)

        r = _make_returns(N_TRAIN + N_STEP)
        model = GARCH(min_train=N_TRAIN, step=N_STEP)
        forecasts = model.forecast_path(r)

        # Exactly one refit window — one fallback
        assert model.fallback_count == 1, f"expected 1 fallback, got {model.fallback_count}"

        # Fallback 2 (no prior fit): forecasts equal the EWMA variance values
        expected = ewma_variance(r, lam=model.ewma_lam).values
        np.testing.assert_allclose(
            forecasts.iloc[N_TRAIN:].values,
            expected[N_TRAIN : N_TRAIN + N_STEP],
            rtol=1e-12,
        )

    def test_nonstationary_failure_triggers_fallback(self, monkeypatch):
        """GarchNonStationaryError is also caught by the fallback chain."""
        import volforecast.models.garch as garch_mod

        def _always_fails(train_slice):
            raise GarchNonStationaryError("forced non-stationarity (test)")

        monkeypatch.setattr(garch_mod, "fit_garch", _always_fails)

        r = _make_returns(N_TRAIN + N_STEP)
        model = GARCH(min_train=N_TRAIN, step=N_STEP)
        forecasts = model.forecast_path(r)
        assert model.fallback_count == 1
        assert forecasts.iloc[N_TRAIN:].notna().all()


# ------------------------------------------------------------------ #
# 5c. Narrow exception handling — no bug laundering (WR-04)
# ------------------------------------------------------------------ #


class TestGarchNoBugLaundering:
    def test_real_bug_propagates_not_laundered(self, monkeypatch):
        """WR-04 regression: a non-fit-quality exception (a real bug, e.g.
        TypeError) must PROPAGATE out of forecast_path — never silently
        converted into a 'fallback'."""
        import pytest

        import volforecast.models.garch as garch_mod

        def _buggy_fit(train_slice):
            raise TypeError("real bug: bad argument type (test)")

        monkeypatch.setattr(garch_mod, "fit_garch", _buggy_fit)

        r = _make_returns(N_TRAIN + N_STEP)
        model = GARCH(min_train=N_TRAIN, step=N_STEP)
        with pytest.raises(TypeError, match="real bug"):
            model.forecast_path(r)
        assert model.fallback_count == 0, "a real bug must not count as a fallback"

    def test_linalg_error_triggers_fallback(self, monkeypatch):
        """np.linalg.LinAlgError (arch numerical failure) IS a legitimate
        fallback trigger alongside the GarchFitError family."""
        import volforecast.models.garch as garch_mod

        def _numerical_failure(train_slice):
            raise np.linalg.LinAlgError("singular matrix (test)")

        monkeypatch.setattr(garch_mod, "fit_garch", _numerical_failure)

        r = _make_returns(N_TRAIN + N_STEP)
        model = GARCH(min_train=N_TRAIN, step=N_STEP)
        forecasts = model.forecast_path(r)
        assert model.fallback_count == 1
        assert forecasts.iloc[N_TRAIN:].notna().all()

    def test_fit_garch_does_not_suppress_unrelated_warnings(self):
        """WR-04: the warnings filter inside fit_garch must be scoped to
        arch's categories — a blanket simplefilter('ignore') would also
        swallow unrelated warnings raised while the filter is active."""
        import warnings

        import volforecast.models.garch as garch_mod

        # Wrap arch_model.fit via fit_garch by emitting an unrelated warning
        # inside the fit call: patch arch_model so its fit() warns.
        real_fit_garch = garch_mod.fit_garch
        r = _make_returns(N_TRAIN)

        from arch import arch_model as real_arch_model

        class _WarningModel:
            def __init__(self, *args, **kwargs):
                self._inner = real_arch_model(*args, **kwargs)

            def fit(self, *args, **kwargs):
                warnings.warn("unrelated deprecation (test)", DeprecationWarning, stacklevel=2)
                return self._inner.fit(*args, **kwargs)

        import arch

        original = arch.arch_model
        try:
            arch.arch_model = _WarningModel
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                real_fit_garch(r)
        finally:
            arch.arch_model = original

        messages = [str(w.message) for w in caught]
        assert any("unrelated deprecation" in m for m in messages), (
            "fit_garch blanket-suppressed an unrelated warning — the filter "
            f"must be scoped to arch categories only (caught: {messages})"
        )


# ------------------------------------------------------------------ #
# 6. GARCH class interface (mirrors EWMA.forecast_path)
# ------------------------------------------------------------------ #


class TestGarchInterface:
    def test_forecast_path_returns_series(self):
        """forecast_path must return a pd.Series with same index as input."""
        r = _make_returns(N_TRAIN + N_STEP)
        model = GARCH()
        forecasts = model.forecast_path(r)
        assert isinstance(forecasts, pd.Series), "forecast_path must return pd.Series"
        assert forecasts.index.equals(r.index), "Series index must match input"

    def test_forecast_path_dtype_float64(self):
        """forecast_path output must be float64."""
        r = _make_returns(N_TRAIN + N_STEP)
        model = GARCH()
        forecasts = model.forecast_path(r)
        assert forecasts.dtype == np.float64, f"Expected float64, got {forecasts.dtype}"
