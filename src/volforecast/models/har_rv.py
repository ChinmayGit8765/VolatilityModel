"""HAR-RV (Heterogeneous Autoregressive Realized Volatility) baseline forecaster.

This module implements the Corsi (2009) HAR-RV model for the walk-forward
evaluation harness.  HAR-RV adds ~30 lines of OLS for outsized quant credibility
by capturing the multi-scale (daily/weekly/monthly) persistence of realized variance.

Model specification (Corsi 2009):
    RV(t) = c + β_d * RV(t-1) + β_w * RV_w(t-1) + β_m * RV_m(t-1) + ε(t)

    where:
    - RV(t-1)   = yesterday's daily realized variance  (rv_d)
    - RV_w(t-1) = mean of RV over the past 5 days      (rv_w, rolling mean not last value)
    - RV_m(t-1) = mean of RV over the past 22 days     (rv_m, rolling mean not last value)

    These are ROLLING MEANS, not the last value — the State of the Art note in
    the research file calls this out explicitly.

Unit contract:
    - Input: daily realized variance in decimal units (e.g. squared log returns).
    - Output: next-day realized variance forecast in the SAME decimal units.
    - No scaling is applied — HAR regresses variance directly on lagged variance.

Walk-forward contract:
    HAR is refitted every ``step`` observations.  OLS is microseconds-fast, so
    refitting every step is fine (Open Question #2) and maintains methodological
    consistency with the walk-forward cadence.

Lazy import:
    ``statsmodels`` is imported inside functions/class methods — not at module
    load time — so the feature pipeline can be imported without statsmodels cost.

References:
    - Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized
      Volatility. Journal of Financial Econometrics, 7(2), 174–196.
    - Andersen, T.G., Bollerslev, T., Diebold, F.X. & Labys, P. (2003).
      Modeling and Forecasting Realized Volatility. Econometrica.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Regressor matrix builder
# --------------------------------------------------------------------------- #


def build_har_features(rv_daily: pd.Series) -> pd.DataFrame:
    """Build the HAR-RV regressor matrix: [const, rv_d, rv_w, rv_m].

    Constructs the Corsi (2009) components from a daily realized-variance series:

    - rv_d = rv_daily.shift(1)               — yesterday's RV
    - rv_w = rv_daily.shift(1).rolling(5).mean()  — mean of t-1..t-5
    - rv_m = rv_daily.shift(1).rolling(22).mean() — mean of t-1..t-22

    The dependent variable is rv_daily itself (next-day RV).  Callers align
    y = rv_daily with X = build_har_features(rv_daily) on the index.

    A ``statsmodels.api.add_constant`` intercept column named ``const`` is
    prepended to the feature DataFrame.

    Rows with any NaN are dropped before returning (the rolling window creates
    NaN in the first 22 rows; the shift adds one more → ~23 rows dropped).

    Args:
        rv_daily: pd.Series of daily realized variance, decimal units.
            May contain an initial NaN (from log_returns.pow(2) first row).

    Returns:
        pd.DataFrame with columns [const, rv_d, rv_w, rv_m] and no NaN rows.
        Index is a subset of rv_daily.index.
    """
    import statsmodels.api as sm  # lazy import

    rv_d = rv_daily.shift(1)  # yesterday's RV
    rv_w = rv_daily.shift(1).rolling(5).mean()  # rolling mean of t-1..t-5
    rv_m = rv_daily.shift(1).rolling(22).mean()  # rolling mean of t-1..t-22

    X = pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m})
    X = X.dropna()
    return sm.add_constant(X, has_constant="add")


# --------------------------------------------------------------------------- #
#  Core fit / forecast helpers
# --------------------------------------------------------------------------- #


def fit_har(rv_daily: pd.Series, train_idx: np.ndarray):  # -> RegressionResults
    """Fit HAR-RV OLS model on the training window.

    Args:
        rv_daily: Full realized-variance series (all observations).
        train_idx: Integer positions of the training window.

    Returns:
        statsmodels RegressionResultsWrapper from sm.OLS.fit().
    """
    import statsmodels.api as sm  # lazy import

    # Build features on the full series then select training rows
    X_all = build_har_features(rv_daily)

    # Only keep rows whose integer position is in train_idx
    # X_all has an index that is a subset of rv_daily.index
    # Convert position-based train_idx to index labels
    train_labels = rv_daily.index[train_idx]
    # Intersect: only rows that are both in X_all (non-NaN) AND in training window
    X_train = X_all.loc[X_all.index.isin(train_labels)]

    # Dependent variable: next-day RV aligned to X_train's index
    y_train = rv_daily.loc[X_train.index]

    model = sm.OLS(y_train, X_train)
    return model.fit()


def har_forecast(fitted, rv_daily_tail: pd.Series) -> float:
    """Compute one-step-ahead HAR-RV forecast.

    Uses the last available rv_d, rv_w (mean of last 5), rv_m (mean of last 22)
    from the tail series to construct the feature vector, then calls
    fitted.predict().

    Args:
        fitted: RegressionResults from fit_har.
        rv_daily_tail: pd.Series of realized variance ending at the most recent
            available observation.  Must have >= 22 observations.

    Returns:
        Scalar float one-step-ahead RV forecast (decimal variance units).
    """
    rv_d_val = float(rv_daily_tail.iloc[-1])
    rv_w_val = float(rv_daily_tail.iloc[-5:].mean())
    rv_m_val = float(rv_daily_tail.iloc[-22:].mean())

    # Must match the column order from build_har_features: [const, rv_d, rv_w, rv_m]
    X_pred = np.array([[1.0, rv_d_val, rv_w_val, rv_m_val]])
    return float(fitted.predict(X_pred)[0])


# --------------------------------------------------------------------------- #
#  HAR-RV walk-forward forecaster class
# --------------------------------------------------------------------------- #


class HARRV:
    """Walk-forward HAR-RV OLS variance forecaster.

    Mirrors the ``EWMA.forecast_path`` interface so the baseline runner in
    ``reports/baseline.py`` can call all three baselines identically:

        model = HARRV(min_train=252, step=21)
        forecasts = model.forecast_path(rv_series)
        test_forecasts = forecasts.iloc[split.test_idx]

    Walk-forward cadence:
        HAR refits every ``step`` observations.  OLS is microseconds-fast,
        so per-step refits are used for methodological consistency.

    Note:
        The input to ``forecast_path`` is the daily realized-variance series
        (not log returns).  In the baseline runner, this is computed from the
        close prices before calling forecast_path.

    Args:
        min_train: Minimum training window.  Default 252 (≈1 year daily).
        step: Refit cadence and test-window length.  Default 21 (≈1 month).

    Attributes:
        n_refits: Number of successful refits.  Populated after forecast_path.
    """

    def __init__(
        self,
        min_train: int = 252,
        step: int = 21,
    ) -> None:
        self.min_train = min_train
        self.step = step
        self.n_refits: int = 0

    def forecast_path(self, rv_daily: pd.Series) -> pd.Series:
        """Compute walk-forward HAR-RV one-step-ahead forecasts for every date.

        The series is processed in expanding windows.  HAR is refit every
        ``step`` observations starting at position ``min_train``.  Each refit
        generates one forecast for the next step days (one per test day).

        Positions before the first refit (i.e. < min_train) receive NaN.

        Args:
            rv_daily: pd.Series of daily realized variance, decimal units.
                Must have a monotonically increasing DatetimeIndex.

        Returns:
            pd.Series of one-step-ahead HAR-RV variance forecasts, same index
            as input.  Positions < min_train are NaN.  All other positions
            are float64 decimal variance values.
        """
        self.n_refits = 0
        n = len(rv_daily)
        forecasts = np.full(n, np.nan)

        pos = self.min_train

        while pos < n:
            test_end = min(pos + self.step, n)

            # Training window: all positions up to (not including) pos
            train_idx = np.arange(pos)

            try:
                fitted = fit_har(rv_daily, train_idx)
                self.n_refits += 1
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "HAR-RV fit failed at pos=%d: %s — skipping window",
                    pos,
                    exc,
                )
                pos = test_end
                continue

            # Generate one-step-ahead forecasts for each test day
            for t in range(pos, test_end):
                # Tail includes all observations up to and including t-1
                tail = rv_daily.iloc[:t]
                if len(tail) < 22:
                    # Not enough history for rv_m — leave NaN
                    forecasts[t] = np.nan
                    continue
                try:
                    forecasts[t] = har_forecast(fitted, tail)
                except Exception as exc2:  # noqa: BLE001
                    log.warning("HAR-RV forecast at t=%d failed: %s", t, exc2)
                    forecasts[t] = np.nan

            pos = test_end

        return pd.Series(forecasts, index=rv_daily.index, dtype=np.float64)
