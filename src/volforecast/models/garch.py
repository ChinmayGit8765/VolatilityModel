"""GARCH(1,1) baseline variance forecaster.

This module implements the GARCH(1,1) baseline for use in the walk-forward
evaluation harness.  GARCH adds time-varying volatility clustering over EWMA
and is the primary classical benchmark the ML model must clear.

Scaling contract:
    GARCH is fit on 100x-scaled decimal log returns (i.e. percent returns).
    Fitting on raw decimal returns (~0.01) causes DataScaleWarning and optimizer
    convergence to boundary parameters (alpha+beta≈1 — integrated GARCH).
    ``GARCH_SCALE = 100.0`` is the module-level constant; it is never inlined.

Unit contract:
    - Input: decimal log returns (from features/estimators.log_returns).
    - Output: daily decimal VARIANCE (same units as compute_target output).
    - The forecast variance from arch is in (100*return)^2 units.
      Divide by GARCH_SCALE**2 exactly once to return decimal variance.
    - Do NOT pass conditional volatility (std dev) to QLIKE without squaring.

Walk-forward refit contract (Pitfall 5):
    The model refits every ``step`` observations (monthly, default 21).
    After each refit, ``res.forecast(horizon=step, reindex=False)`` is called
    once to produce forecasts for ALL step test days — NOT one fit per day.
    The h.1..h.step variance columns are extracted and de-scaled to decimal.

Fallback contract (Open Question #1):
    If a refit fails convergence (convergence_flag != 0) or stationarity
    (alpha+beta >= 1.0), the model falls back to:
    1. The previous window's fitted params (reuse last ARCHModelResult).
    2. EWMA forecast for that window's test days (if no prior fit exists).
    Each fallback increments ``self.fallback_count`` (report transparency).

Lazy import:
    ``arch`` is imported inside functions/class methods — not at module load
    time.  This allows the feature pipeline (Plan 04) to import volforecast
    without paying the GARCH runtime overhead unless explicitly requested.

References:
    - Bollerslev, T. (1986). Generalized autoregressive conditional
      heteroskedasticity. Journal of Econometrics, 31(3), 307–327.
    - Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with
      estimates of the variance of UK inflation. Econometrica, 50(4), 987–1007.
    - arch 8.0: bashtage.github.io/arch/univariate/
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from volforecast.features.estimators import ewma_variance

if TYPE_CHECKING:
    pass  # type stubs only — arch imported lazily

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Module-level constants
# --------------------------------------------------------------------------- #

#: 100x multiplier applied to decimal log returns before fitting.
#: Ensures the optimizer receives values in the 0.1–10 range expected by arch.
#: Divide forecast variance by GARCH_SCALE**2 to recover decimal variance.
GARCH_SCALE: float = 100.0


# --------------------------------------------------------------------------- #
#  Core fit/forecast helpers
# --------------------------------------------------------------------------- #


def fit_garch(log_returns_decimal: pd.Series):  # -> ARCHModelResult
    """Fit GARCH(1,1) on 100x-scaled decimal log returns.

    Scales the input by GARCH_SCALE before fitting so the optimizer sees
    values in the ~0.1–10 range it expects.  Uses ``rescale=False`` to
    prevent arch from applying a second (hidden) rescaling step.

    Asserts after every fit:
    - ``res.convergence_flag == 0``  (scipy optimizer terminated successfully)
    - ``alpha + beta < 1.0``         (stationarity / finite unconditional var)

    Args:
        log_returns_decimal: pd.Series of decimal log returns.  First element
            may be NaN; NaNs are dropped before passing to arch.

    Returns:
        ARCHModelResult with fit GARCH(1,1) params.

    Raises:
        AssertionError: If convergence or stationarity assertions fail.
    """
    from arch import arch_model  # lazy import

    clean = log_returns_decimal.dropna()
    scaled = GARCH_SCALE * clean

    am = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off")

    assert res.convergence_flag == 0, (
        f"GARCH did not converge (convergence_flag={res.convergence_flag})"
    )
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])
    assert alpha + beta < 1.0, f"GARCH non-stationary: alpha+beta={alpha + beta:.6f} >= 1.0"
    return res


def garch_forecast_variance_decimal(res, horizon: int = 1) -> np.ndarray:
    """Extract next ``horizon`` variance forecasts in DECIMAL log-return units.

    The model was fitted on 100x-scaled returns, so the forecast variance
    from arch is in (100 * return)^2 units.  This function divides by
    GARCH_SCALE**2 exactly once to recover decimal variance.

    Args:
        res: ARCHModelResult from fit_garch.
        horizon: Number of steps to forecast.  Default 1.

    Returns:
        np.ndarray of length ``horizon`` containing decimal variance forecasts.
        The first element is the next-step (h.1) forecast.
    """
    fc = res.forecast(horizon=horizon, reindex=False)
    # fc.variance is a DataFrame with columns h.1 .. h.{horizon}
    # Last row contains the forecasts from the most recent observation.
    last_row = fc.variance.iloc[-1]  # pd.Series with index h.1, h.2, ...
    variance_scaled = last_row.values.astype(float)  # shape (horizon,)
    return variance_scaled / (GARCH_SCALE**2)


# --------------------------------------------------------------------------- #
#  GARCH walk-forward forecaster class
# --------------------------------------------------------------------------- #


class GARCH:
    """Walk-forward GARCH(1,1) variance forecaster.

    Mirrors the ``EWMA.forecast_path`` interface so the baseline runner in
    ``reports/baseline.py`` can call all three baselines identically:

        model = GARCH(min_train=252, step=21)
        forecasts = model.forecast_path(log_returns_series)
        test_forecasts = forecasts.iloc[split.test_idx]

    Walk-forward refit cadence:
        The model refits every ``step`` observations (Pitfall 5 guard).
        After each refit, ``forecast(horizon=step)`` is called once to
        produce all ``step`` daily forecasts for the upcoming test window.
        This amortises the GARCH fitting cost (~seconds) over 21 test days.

    Fallback chain (Open Question #1):
        1. Convergence or stationarity failure → use previous ARCHModelResult.
        2. No previous result (first window failure) → use EWMA variance.
        Each fallback increments ``self.fallback_count``.

    Args:
        min_train: Minimum training window.  Default 252 (≈1 year daily).
        step: Refit cadence and test-window length.  Default 21 (≈1 month).
        ewma_lam: Fallback EWMA decay factor.  Default 0.94 (RiskMetrics).

    Attributes:
        fallback_count: Number of refits that fell back to previous params
            or EWMA.  Populated after ``forecast_path`` completes.
    """

    def __init__(
        self,
        min_train: int = 252,
        step: int = 21,
        ewma_lam: float = 0.94,
    ) -> None:
        self.min_train = min_train
        self.step = step
        self.ewma_lam = ewma_lam
        self.fallback_count: int = 0

    def forecast_path(self, log_returns_series: pd.Series) -> pd.Series:
        """Compute walk-forward GARCH(1,1) variance forecasts for every date.

        The series is processed in expanding windows.  GARCH is refit every
        ``step`` observations starting at position ``min_train``.  Each refit
        at position ``pos`` trains on returns up to AND INCLUDING ``pos``
        (data <= pos) and generates ``step`` forward variance forecasts
        (one per test day): forecast[pos + k] is the (k+1)-step-ahead variance,
        i.e. the forecast of RV[pos + k + 1], matching the target alignment
        ``compute_target[t] = RV[t+1]`` (CR-01).

        Positions before the first refit (i.e. < min_train) receive NaN
        because no forecast can be produced without a minimum training window.

        Args:
            log_returns_series: pd.Series of decimal log returns.  Should
                have a monotonically increasing DatetimeIndex.  The first
                element is expected to be NaN (no prior close price).

        Returns:
            pd.Series of one-step-ahead GARCH variance forecasts, same index
            as input.  Positions < min_train are NaN.  All other positions
            are positive float64 decimal variance values.
        """
        self.fallback_count = 0
        n = len(log_returns_series)
        forecasts = np.full(n, np.nan)

        # Pre-compute EWMA for fallback (cheap; vectorised)
        ewma = ewma_variance(log_returns_series, lam=self.ewma_lam).values

        last_res = None  # last successful ARCHModelResult

        refit_start = self.min_train
        pos = refit_start

        while pos < n:
            # Determine test window for this refit
            test_end = min(pos + self.step, n)
            horizon = test_end - pos  # actual horizon (may be < step at series end)

            if horizon <= 0:
                break

            # Training slice: all returns up to AND INCLUDING pos (data <= pos).
            # The forecast issued at position pos is as-of pos and targets
            # RV[pos+1] (= compute_target[pos]), so r[pos] is legitimately in
            # the information set (CR-01 alignment fix).
            train_slice = log_returns_series.iloc[: pos + 1]

            # Attempt fit
            fitted_res = None
            try:
                fitted_res = fit_garch(train_slice)
                last_res = fitted_res
            except (AssertionError, Exception) as exc:  # noqa: BLE001
                log.warning(
                    "GARCH refit at pos=%d failed (%s: %s) — using fallback",
                    pos,
                    type(exc).__name__,
                    exc,
                )
                self.fallback_count += 1

            # Extract forecasts from this refit
            if fitted_res is not None:
                # Primary: multi-step variance from this fit
                var_decimal = garch_forecast_variance_decimal(fitted_res, horizon=horizon)
            elif last_res is not None:
                # Fallback 1: re-use previous successful fit's forecast
                try:
                    var_decimal = garch_forecast_variance_decimal(last_res, horizon=horizon)
                except Exception as exc2:  # noqa: BLE001
                    log.warning(
                        "Fallback re-forecast also failed (%s) — using EWMA",
                        exc2,
                    )
                    var_decimal = ewma[pos : pos + horizon]
            else:
                # Fallback 2: EWMA for this window (no prior fit at all)
                var_decimal = ewma[pos : pos + horizon]

            # Store forecasts for this test window
            forecasts[pos:test_end] = var_decimal[:horizon]

            pos = test_end

        return pd.Series(forecasts, index=log_returns_series.index, dtype=np.float64)
