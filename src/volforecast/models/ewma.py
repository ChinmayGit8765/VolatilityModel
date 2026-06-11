"""EWMA (Exponentially Weighted Moving Average) baseline variance forecaster.

This module implements the RiskMetrics EWMA baseline for use in the walk-forward
evaluation harness.  EWMA is the simplest deterministic baseline: it requires no
fitting (no optimizer, no convergence check), making it the ideal first baseline
to prove the end-to-end pipeline before GARCH's numerical hazards are introduced.

Interface contract:
    The ``EWMA`` class exposes ``forecast_path(log_returns) -> pd.Series`` which
    returns the *one-step-ahead* variance forecast for every date in the input.
    This uniform interface is designed so that the baseline runner (reports/baseline.py)
    can call all three baselines (EWMA, GARCH, HAR-RV in Plans 02-03) identically:

        forecast = model.forecast_path(log_returns)
        test_forecasts = forecast.iloc[split.test_idx]

    No per-fold refit is required for EWMA (the recursion is stateless given the
    returns), so ``forecast_path`` is called once on the full return series and
    the baseline runner slices by walk-forward test indices.

Alignment contract (CR-01):
    The target is ``compute_target[t] = RV[t+1]`` — defined as-of t, using data
    up to and including t.  ``forecast_path`` therefore returns
    ``ewma_variance(log_returns, lam)`` with NO extra shift:

    At each position t, ``forecast[t] = ewma[t]`` is the EWMA variance computed
    from all log returns up to and including t — the one-step-ahead forecast of
    RV[t+1] issued as-of t.  Under the RiskMetrics recursion the conditional
    variance forecast for t+1 IS the EWMA level at t:
    ``sigma2[t+1|t] = lam * sigma2[t|t-1] + (1 - lam) * r[t]^2 = ewma[t]``.

    No data past t is used (the recursion is strictly causal), and the forecast
    carries exactly the same information set as the Phase-3 ML features
    (as-of t) — so the baseline-vs-ML comparison is fair.

Units:
    - Input: decimal log returns (output of estimators.log_returns).
    - Output: daily decimal VARIANCE (same units as the target from target.py).
    - Do NOT pass percent returns (100× too large) or standard deviation (must
      square first).

References:
    - J.P. Morgan / Reuters (1996). RiskMetrics Technical Document, 4th ed.
    - Patton, A.J. (2011). Volatility forecast comparison using imperfect volatility
      proxies. Journal of Econometrics, 160(1), 246–256.
"""

from __future__ import annotations

import pandas as pd

from volforecast.features.estimators import EWMA_LAMBDA, ewma_variance


class EWMA:
    """One-step-ahead EWMA variance forecaster.

    This is a stateless forecaster: the recursion is fully determined by the
    input log returns and the decay factor, requiring no per-fold refit.

    The ``forecast_path`` method returns the complete one-step-ahead variance
    forecast series for the entire input: the value at index t is the forecast
    of RV[t+1] issued as-of t, using only returns up to and including t
    (no look-ahead past the as-of date).

    Args:
        lam: EWMA decay factor (lambda in the RiskMetrics recursion).
             Default: EWMA_LAMBDA=0.94 (J.P. Morgan daily consensus value).

    Example::

        import pandas as pd
        from volforecast.features.estimators import log_returns
        from volforecast.models.ewma import EWMA

        model = EWMA(lam=0.94)
        forecasts = model.forecast_path(log_returns_series)
        # forecasts.iloc[t] uses only returns_series.iloc[:t+1] (data <= t)
        # and is scored against RV[t+1] (compute_target[t])
    """

    def __init__(self, lam: float = EWMA_LAMBDA) -> None:
        if not (0.0 < lam < 1.0):
            raise ValueError(f"lam must be in (0, 1), got {lam}")
        self.lam = lam

    def forecast_path(self, log_returns_series: pd.Series) -> pd.Series:
        """Compute one-step-ahead EWMA variance forecasts for every date.

        The forecast at position t predicts RV[t+1] using the EWMA variance
        computed from all log returns up to and including position t:

            forecast[t] = ewma_variance(log_returns, lam)[t]

        This matches the target alignment (CR-01): ``compute_target[t] = RV[t+1]``
        is defined as-of t, so the forecast issued at t may use data <= t.
        Under RiskMetrics, the conditional variance forecast for t+1 equals the
        EWMA level at t — no extra shift is applied.

        Args:
            log_returns_series: pd.Series of decimal log returns (e.g., from
                ``volforecast.features.estimators.log_returns``).  Must have a
                monotonically increasing DatetimeIndex.  The first element is
                expected to be NaN (no prior close price).

        Returns:
            pd.Series of one-step-ahead EWMA variance forecasts, same index as
            input.
            - Index 0: NaN (propagated from the NaN first log return)
            - Index t >= 1: EWMA variance using returns r[1]..r[t] (data <= t),
              forecasting RV[t+1]

            Forecasts are positive and finite wherever the input is non-NaN.
            dtype=float64.
        """
        # forecast[t] = ewma[t] uses returns <= t and predicts RV[t+1]
        # (the target compute_target[t] = RV[t+1] is as-of t — same info set)
        return ewma_variance(log_returns_series, lam=self.lam)
