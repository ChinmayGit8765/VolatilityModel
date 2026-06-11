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

No-look-ahead guarantee:
    ``forecast_path`` returns ``ewma_variance(log_returns, lam).shift(1)``.

    At each position t, this gives the EWMA variance computed from *all* log
    returns strictly up to and including t-1 — the one-step-ahead forecast is
    indexed at t but uses only data < t.  The shift(1) makes the guarantee
    explicit and testable: the forecast at date t comes from ewma[t-1].

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
    forecast series for the entire input, shift(1)-aligned so that the value at
    index t uses only returns strictly before t (no look-ahead).

    Args:
        lam: EWMA decay factor (lambda in the RiskMetrics recursion).
             Default: EWMA_LAMBDA=0.94 (J.P. Morgan daily consensus value).

    Example::

        import pandas as pd
        from volforecast.features.estimators import log_returns
        from volforecast.models.ewma import EWMA

        model = EWMA(lam=0.94)
        forecasts = model.forecast_path(log_returns_series)
        # forecasts.iloc[t] uses only returns_series.iloc[:t]
    """

    def __init__(self, lam: float = EWMA_LAMBDA) -> None:
        if not (0.0 < lam < 1.0):
            raise ValueError(f"lam must be in (0, 1), got {lam}")
        self.lam = lam

    def forecast_path(self, log_returns_series: pd.Series) -> pd.Series:
        """Compute one-step-ahead EWMA variance forecasts for every date.

        The forecast at position t is the EWMA variance computed from all log
        returns up to and including position t-1.  This is implemented as:

            ewma_variance(log_returns, lam).shift(1)

        The ``shift(1)`` makes the no-look-ahead property explicit: the value
        at index t comes from the EWMA recursion at t-1, which uses only returns
        r[0], ..., r[t-1].

        Args:
            log_returns_series: pd.Series of decimal log returns (e.g., from
                ``volforecast.features.estimators.log_returns``).  Must have a
                monotonically increasing DatetimeIndex.  The first element is
                expected to be NaN (no prior close price).

        Returns:
            pd.Series of one-step-ahead EWMA variance forecasts, same index as
            input.
            - Index 0: NaN (shift(1) of EWMA[0])
            - Index 1: NaN (shift(1) of EWMA[1] — propagated from NaN input)
            - Index t >= 2: EWMA variance using returns r[0]..r[t-1]

            Forecasts are positive and finite wherever the input is non-NaN.
            dtype=float64.
        """
        ewma_vals = ewma_variance(log_returns_series, lam=self.lam)
        # shift(1): forecast[t] = ewma[t-1] which uses only returns < t
        return ewma_vals.shift(1)
