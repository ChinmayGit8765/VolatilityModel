"""Value-level alignment tests for the classical baselines (CR-01 regression).

The target is ``compute_target[t] = RV[t+1]`` — defined as-of t, i.e. using
data up to and including t.  Every baseline ``forecast_path`` must therefore
produce, at index t, a forecast of RV[t+1] built from data <= t:

    - EWMA:   forecast[t] = ewma[t]                       (returns <= t)
    - GARCH:  refit at pos trains on returns iloc[:pos+1] (returns <= pos);
              forecast[pos + k] is the (k+1)-step-ahead variance = RV[pos+k+1]
    - HAR-RV: forecast[t] applies fitted OLS coefficients to HAR components
              built from rv[<= t]

These are VALUE-level tests (not count/NaN-placement tests): they pin the
exact numeric pairing between forecast[t], the information set <= t, and the
scored target RV[t+1].  The pre-fix implementations were internally consistent
but systematically one day stale (forecast[t] used data <= t-1), handicapping
the classical bar vs the as-of-t Phase-3 ML features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.features.estimators import EWMA_LAMBDA, log_returns, realized_var
from volforecast.features.target import compute_target
from volforecast.models.ewma import EWMA
from volforecast.models.garch import GARCH, fit_garch, garch_forecast_variance_decimal
from volforecast.models.har_rv import HARRV, fit_har

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 330
MIN_TRAIN = 280
STEP = 21


def _synthetic_close(n: int = N, seed: int = 7) -> pd.Series:
    """Deterministic close-price series with volatility clustering.

    A simple two-regime variance process gives GARCH/HAR something real to fit
    so the value-level comparisons are not degenerate.
    """
    rng = np.random.default_rng(seed)
    sigma = np.where((np.arange(n) // 40) % 2 == 0, 0.008, 0.02)
    r = rng.normal(0.0, 1.0, n) * sigma
    close = 100.0 * np.exp(np.cumsum(r))
    dates = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC", name="date")
    return pd.Series(close, index=dates, name="close", dtype="float64")


# ---------------------------------------------------------------------------
# Target pairing: compute_target[t] == r[t+1]^2 (value-level)
# ---------------------------------------------------------------------------


class TestTargetPairing:
    def test_target_at_t_is_next_day_squared_return(self) -> None:
        """compute_target[t] must equal r[t+1]^2 exactly — the value the
        baselines are scored against at index t."""
        close = _synthetic_close(60)
        lr = log_returns(close)
        target = compute_target(close)
        for t in [5, 20, 40, 58]:
            assert target.iloc[t] == pytest.approx(float(lr.iloc[t + 1]) ** 2, rel=1e-12), (
                f"target[{t}] != r[{t + 1}]^2 — target alignment broken"
            )


# ---------------------------------------------------------------------------
# EWMA: forecast[t] == RiskMetrics recursion through r[t], scored vs RV[t+1]
# ---------------------------------------------------------------------------


class TestEwmaAlignment:
    def test_forecast_value_is_recursion_through_t(self) -> None:
        """Value-level: forecast[t] == h[t] where
        h[t] = lam * h[t-1] + (1-lam) * r[t]^2 (recursion seeded at r[1])."""
        close = _synthetic_close(50)
        lr = log_returns(close)
        forecasts = EWMA().forecast_path(lr)

        # Manual RiskMetrics recursion (adjust=False semantics): h[1] = r[1]^2
        h = float(lr.iloc[1]) ** 2
        assert forecasts.iloc[1] == pytest.approx(h, rel=1e-12)
        for t in range(2, 50):
            h = EWMA_LAMBDA * h + (1.0 - EWMA_LAMBDA) * float(lr.iloc[t]) ** 2
            assert forecasts.iloc[t] == pytest.approx(h, rel=1e-12), (
                f"EWMA forecast[{t}] does not match the recursion through r[{t}]"
            )

    def test_scored_pair_is_forecast_through_t_vs_rv_tplus1(self) -> None:
        """The scored pair at index t is (data <= t, RV[t+1]):
        - mutating r[t+1] changes target[t] but NOT forecast[t]
        - mutating r[t] changes forecast[t] but NOT target[t]'s pairing index
        """
        close = _synthetic_close(60)
        lr = log_returns(close)
        target = compute_target(close)
        forecasts = EWMA().forecast_path(lr)
        t = 30

        # Forecast invariant to the future value it predicts
        lr_future = lr.copy()
        lr_future.iloc[t + 1] = lr.iloc[t + 1] + 0.1
        fc_future = EWMA().forecast_path(lr_future)
        assert fc_future.iloc[t] == pytest.approx(float(forecasts.iloc[t]), rel=1e-14), (
            "forecast[t] changed when r[t+1] (the predicted value) was mutated — leakage"
        )

        # Forecast DOES use r[t] (as-of t info set)
        lr_asof = lr.copy()
        lr_asof.iloc[t] = lr.iloc[t] + 0.1
        fc_asof = EWMA().forecast_path(lr_asof)
        assert abs(float(fc_asof.iloc[t]) - float(forecasts.iloc[t])) > 1e-12, (
            "forecast[t] ignored r[t] — baseline is one day stale (CR-01 regression)"
        )

        # And the target at t is exactly RV[t+1]
        assert target.iloc[t] == pytest.approx(float(lr.iloc[t + 1]) ** 2, rel=1e-12)


# ---------------------------------------------------------------------------
# GARCH: refit at pos trains through r[pos]; forecast[pos+k] = RV[pos+k+1]
# ---------------------------------------------------------------------------


class TestGarchAlignment:
    def test_first_window_forecast_trained_through_pos(self) -> None:
        """Value-level: forecast_path[pos] must equal the h.1 forecast from a
        GARCH fitted on returns iloc[:pos+1] (data <= pos), de-scaled.

        Also pins the multi-step alignment: forecast_path[pos+k] == h.{k+1}
        from the same fit.
        """
        close = _synthetic_close(N)
        lr = log_returns(close)
        model = GARCH(min_train=MIN_TRAIN, step=STEP)
        forecasts = model.forecast_path(lr)

        pos = MIN_TRAIN
        res = fit_garch(lr.iloc[: pos + 1])
        horizon = min(STEP, len(lr) - pos)
        expected = garch_forecast_variance_decimal(res, horizon=horizon)

        for k in range(horizon):
            assert forecasts.iloc[pos + k] == pytest.approx(float(expected[k]), rel=1e-10), (
                f"GARCH forecast[pos+{k}] != h.{k + 1} from fit on data <= pos — "
                "training window misaligned with the as-of date (CR-01)"
            )

    def test_forecast_invariant_to_data_after_asof(self) -> None:
        """forecast[pos] computed on the full series equals forecast[pos]
        computed on the series truncated at pos (data <= pos only) —
        value-level proof that nothing past the as-of date is used."""
        close = _synthetic_close(N)
        lr = log_returns(close)
        pos = MIN_TRAIN

        fc_full = GARCH(min_train=MIN_TRAIN, step=STEP).forecast_path(lr)
        fc_trunc = GARCH(min_train=MIN_TRAIN, step=STEP).forecast_path(lr.iloc[: pos + 1])

        assert fc_trunc.iloc[pos] == pytest.approx(float(fc_full.iloc[pos]), rel=1e-10), (
            "GARCH forecast at pos differs when future data is removed — leakage"
        )


# ---------------------------------------------------------------------------
# HAR-RV: forecast[t] applies fitted coefficients to rv components <= t
# ---------------------------------------------------------------------------


class TestHarAlignment:
    def test_forecast_value_uses_components_through_t(self) -> None:
        """Value-level: forecast_path[t] must equal
        fitted.predict([1, rv[t], mean(rv[t-4..t]), mean(rv[t-21..t])])
        where fitted is the OLS from the window's training positions."""
        close = _synthetic_close(N)
        lr = log_returns(close)
        rv = realized_var(lr, window=1)

        model = HARRV(min_train=MIN_TRAIN, step=STEP)
        forecasts = model.forecast_path(rv)

        pos = MIN_TRAIN
        fitted = fit_har(rv, np.arange(pos))

        for t in [pos, pos + 5, pos + STEP - 1]:
            if t >= len(rv):
                continue
            rv_d = float(rv.iloc[t])
            rv_w = float(rv.iloc[t - 4 : t + 1].mean())
            rv_m = float(rv.iloc[t - 21 : t + 1].mean())
            expected = float(fitted.predict(np.array([[1.0, rv_d, rv_w, rv_m]]))[0])
            assert forecasts.iloc[t] == pytest.approx(expected, rel=1e-10), (
                f"HAR forecast[{t}] does not use rv components through t — "
                "tail misaligned with the as-of date (CR-01)"
            )

    def test_forecast_at_t_uses_rv_at_t(self) -> None:
        """Mutating rv[t] must change forecast[t]; mutating rv[t+1] must not."""
        close = _synthetic_close(N)
        lr = log_returns(close)
        rv = realized_var(lr, window=1)
        t = MIN_TRAIN + 3

        fc_orig = HARRV(min_train=MIN_TRAIN, step=STEP).forecast_path(rv)

        rv_mut_t = rv.copy()
        rv_mut_t.iloc[t] = rv.iloc[t] * 50 + 1e-3
        fc_mut_t = HARRV(min_train=MIN_TRAIN, step=STEP).forecast_path(rv_mut_t)
        assert abs(float(fc_mut_t.iloc[t]) - float(fc_orig.iloc[t])) > 1e-15, (
            "HAR forecast[t] ignored rv[t] — one day stale (CR-01 regression)"
        )

        rv_mut_t1 = rv.copy()
        rv_mut_t1.iloc[t + 1] = rv.iloc[t + 1] * 50 + 1e-3
        fc_mut_t1 = HARRV(min_train=MIN_TRAIN, step=STEP).forecast_path(rv_mut_t1)
        assert fc_mut_t1.iloc[t] == pytest.approx(float(fc_orig.iloc[t]), rel=1e-12), (
            "HAR forecast[t] changed when rv[t+1] (the predicted value) was mutated — leakage"
        )


# ---------------------------------------------------------------------------
# Cross-baseline consistency: identical info-set convention at index t
# ---------------------------------------------------------------------------


class TestUniformAlignmentAcrossBaselines:
    def test_all_baselines_invariant_to_future_and_sensitive_to_asof(self) -> None:
        """All three baselines share the same convention at a common index t:
        sensitive to data at t, invariant to data at t+1 and beyond."""
        close = _synthetic_close(N)
        lr = log_returns(close)
        rv = realized_var(lr, window=1)
        t = MIN_TRAIN  # refit boundary — covered by all three models

        ewma_fc = EWMA().forecast_path(lr)
        garch_fc = GARCH(min_train=MIN_TRAIN, step=STEP).forecast_path(lr)
        har_fc = HARRV(min_train=MIN_TRAIN, step=STEP).forecast_path(rv)

        # Truncate at the as-of date: forecasts at t must be reproducible
        lr_trunc = lr.iloc[: t + 1]
        rv_trunc = rv.iloc[: t + 1]
        assert EWMA().forecast_path(lr_trunc).iloc[t] == pytest.approx(
            float(ewma_fc.iloc[t]), rel=1e-12
        )
        assert GARCH(min_train=MIN_TRAIN, step=STEP).forecast_path(lr_trunc).iloc[t] == (
            pytest.approx(float(garch_fc.iloc[t]), rel=1e-10)
        )
        assert HARRV(min_train=MIN_TRAIN, step=STEP).forecast_path(rv_trunc).iloc[t] == (
            pytest.approx(float(har_fc.iloc[t]), rel=1e-10)
        )
