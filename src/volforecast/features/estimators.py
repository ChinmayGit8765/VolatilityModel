"""Deterministic feature estimators for VolForecast.

This module provides stateless, purely computational helpers for the feature
pipeline and for the EWMA baseline.  All functions are vectorised over pandas
Series / DataFrames and produce outputs with the same DatetimeIndex as the input.

Unit convention (shared with target.py):
    - Inputs: decimal close prices; decimal log returns r[t] = ln(close[t]/close[t-1]).
    - Outputs: decimal log-return VARIANCE (e.g., ~1e-4 for typical daily equity).
    - NO percent-squared, NO annualization anywhere in this module.
    - Annualization (× 252) and vol (sqrt) are display-only at report edges.

Design notes:
    - All rolling windows use integer counts (not duration strings) so that
      the harness sees a predictable NaN prefix of exactly ``window - 1`` rows
      (Pitfall 3: duration-based windows produce variable NaN counts on
      irregular equity calendars — use integer-count rolling only).
    - ewma_variance uses ``adjust=False`` (mandatory).  ``adjust=True`` uses an
      initialisation that diverges on short series and can produce NaN or
      unexpectedly large early values; the RiskMetrics recursion is defined with
      ``adjust=False`` (infinite-history initialisation via the geometric sum).
    - EWMA_LAMBDA = 0.94 is the J.P. Morgan RiskMetrics (1994) consensus value
      for daily returns and is the single authoritative constant across the project.

Garman-Klass equity caveat (Open Question #4):
    The Garman-Klass estimator (garman_klass_var) assumes no overnight gap.  For
    equities, the open price at t is the exchange open price, not the prior close.
    The overnight return (prior close → next open) is therefore NOT captured by
    the GK formula.  This causes a slight underestimate of equity daily variance
    on days with large overnight moves.  Yang-Zhang handles overnight gaps but adds
    implementation complexity; GK is used for all assets in v1 and this caveat is
    documented in the model card.  Crypto is 24/7 so the assumption holds exactly.

References:
    - J.P. Morgan / Reuters (1996). RiskMetrics Technical Document.
    - Parkinson, M. (1980). The extreme value method for estimating the variance
      of the rate of return. Journal of Business, 53(1), 61–65.
    - Garman, M.B. & Klass, M.J. (1980). On the estimation of security price
      volatilities from historical data. Journal of Business, 53(1), 67–78.
    - Patton, A.J. (2011). Volatility forecast comparison using imperfect volatility
      proxies. Journal of Econometrics, 160(1), 246–256.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- #
#  Module-level constants
# --------------------------------------------------------------------- #

#: RiskMetrics (J.P. Morgan 1994) daily decay factor.
#: This constant is the single source of truth across the project:
#:   - features/estimators.py (variance recursion)
#:   - models/ewma.py (baseline forecaster reuses this constant)
#:   - DO NOT duplicate the value — import from here instead.
EWMA_LAMBDA: float = 0.94


# --------------------------------------------------------------------- #
#  Return helpers
# --------------------------------------------------------------------- #


def log_returns(close: pd.Series) -> pd.Series:
    """Compute daily decimal log returns.

    r[t] = ln(close[t] / close[t-1])

    The first row is NaN (no prior close available).  The result is in
    decimal units — not percent.

    Args:
        close: Series of close prices with DatetimeIndex.

    Returns:
        Series of decimal log returns, same index as ``close``.
        dtype=float64.  First element is NaN.
    """
    return np.log(close).diff()


def squared_returns(close: pd.Series) -> pd.Series:
    """Compute squared decimal log returns.

    sr[t] = (ln(close[t] / close[t-1]))^2

    First row is NaN (inherited from log_returns).  This is the instantaneous
    squared-return variance proxy used by the target module.

    Args:
        close: Series of close prices with DatetimeIndex.

    Returns:
        Series of squared decimal log returns, same index as ``close``.
        First element is NaN.
    """
    return log_returns(close) ** 2


# --------------------------------------------------------------------- #
#  Volatility estimators
# --------------------------------------------------------------------- #


def realized_var(log_returns_series: pd.Series, window: int) -> pd.Series:
    """Rolling count-based realized variance (mean of squared log returns).

    rv[t] = mean(r[t-window+1]^2, ..., r[t]^2)

    Uses an integer ``window`` (count-based), not a duration string, so the
    NaN prefix is exactly ``window - 1`` rows regardless of calendar gaps
    (Pitfall 3 guard).

    Args:
        log_returns_series: Series of decimal log returns (e.g., from
            ``log_returns()``).  First row is expected to be NaN.
        window: Look-back count in trading periods.  Must be >= 1.

    Returns:
        Series of realized variance, same index as input.
        First ``window - 1`` rows are NaN (insufficient history).
        dtype=float64.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return log_returns_series.pow(2).rolling(window).mean()


def ewma_variance(log_returns_series: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """EWMA variance recursion (RiskMetrics, J.P. Morgan 1994).

    h[t] = (1 - lambda) * r[t]^2 + lambda * h[t-1]

    Implemented via pandas ``ewm(alpha=1-lam, adjust=False).mean()`` applied to
    squared log returns.  ``adjust=False`` is MANDATORY — it matches the
    infinite-history initialisation of the RiskMetrics recursion.  ``adjust=True``
    uses a finite-history correction that diverges on short series and is
    inconsistent with the standard EWMA definition.

    Args:
        log_returns_series: Series of decimal log returns.
        lam: EWMA decay factor.  Default EWMA_LAMBDA=0.94 (RiskMetrics daily).

    Returns:
        Series of EWMA variance, same index as input.
        Values are defined even at index 0 (initialised from the first
        observation with full weight); no leading NaN from the EWMA itself
        (though a leading NaN in the input propagates through).
        dtype=float64.
    """
    return log_returns_series.pow(2).ewm(alpha=1 - lam, adjust=False).mean()


# --------------------------------------------------------------------- #
#  OHLC-based range estimators
# --------------------------------------------------------------------- #


def parkinson_var(df: pd.DataFrame) -> pd.Series:
    """Parkinson (1980) daily variance from high/low prices.

    Estimator: (ln(H/L))^2 / (4 * ln(2))

    5x more efficient than close-to-close squared returns.  Assumes geometric
    Brownian motion with no drift and no overnight gap.  Row-wise: each output
    row uses only the high and low of the same row (no look-ahead, no rolling).

    Args:
        df: DataFrame with DatetimeIndex and columns ``high``, ``low``.

    Returns:
        Series of daily Parkinson variance in decimal log-return units.
        Same index as ``df``.  All values are strictly positive.
        dtype=float64.
    """
    log_hl = np.log(df["high"] / df["low"])
    return log_hl**2 / (4.0 * np.log(2.0))


def garman_klass_var(df: pd.DataFrame) -> pd.Series:
    """Garman-Klass (1980) daily variance from OHLC prices.

    Estimator: 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2

    7.4x more efficient than close-to-close.  Uses all four OHLC prices.
    Row-wise: each output row uses only the OHLC of the same row.

    Note for equities (Open Question #4): The GK estimator assumes no overnight
    gap.  For equity data the open price is the exchange open, not the prior
    close, so the overnight return component is not captured.  This causes a
    slight underestimate on days with large overnight moves.  Crypto is 24/7 so
    the assumption holds exactly.  See module docstring for full caveat.

    Args:
        df: DataFrame with DatetimeIndex and columns ``high``, ``low``,
            ``close``, ``open``.

    Returns:
        Series of daily Garman-Klass variance in decimal log-return units.
        Same index as ``df``.  Can be slightly negative when close ≈ open and
        the high-low range is narrow (numerically valid but treated as near-zero
        variance; callers may clip to 0 for display).
        dtype=float64.
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    return 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2


# --------------------------------------------------------------------- #
#  Vol-of-vol, rolling skew/kurt
# --------------------------------------------------------------------- #


def vol_of_vol(rv_series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of a realized-variance series (vol-of-vol).

    Computes the second-order variation of realized variance — how much the
    variance level itself fluctuates over the chosen look-back.  Integer
    count-based window (Pitfall 3).

    vol_of_vol[t] = std(rv[t-window+1], ..., rv[t])

    Args:
        rv_series: Series of realized variance values (e.g., from
            ``realized_var()``).  Must not contain NaN (caller should drop NaN
            prefix before passing, or accept NaN propagation in the output).
        window: Integer look-back count >= 2.  First ``window-1`` output rows
            are NaN.

    Returns:
        Series of rolling std of ``rv_series``, same index as input.
        First ``window-1`` rows NaN.  dtype=float64.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2 for std, got {window}")
    return rv_series.rolling(window).std()


def rolling_skew(log_returns_series: pd.Series, window: int) -> pd.Series:
    """Rolling skewness of log returns over an integer window.

    skew[t] = pandas rolling skewness of r[t-window+1..t].

    Args:
        log_returns_series: Series of decimal log returns (first row typically NaN).
        window: Integer look-back count.  Minimum 3 for a valid skewness estimate.

    Returns:
        Series of rolling skewness, same index as input.
        First ``window-1`` rows NaN (plus the initial NaN from log returns).
        dtype=float64.
    """
    return log_returns_series.rolling(window).skew()


def rolling_kurt(log_returns_series: pd.Series, window: int) -> pd.Series:
    """Rolling excess kurtosis of log returns over an integer window.

    kurt[t] = pandas rolling kurtosis (excess) of r[t-window+1..t].
    The pandas implementation returns EXCESS kurtosis (i.e. normal dist → 0).

    Args:
        log_returns_series: Series of decimal log returns (first row typically NaN).
        window: Integer look-back count.  Minimum 4 for a valid kurtosis estimate.

    Returns:
        Series of rolling excess kurtosis, same index as input.
        First ``window-1`` rows NaN (plus initial NaN from log returns).
        dtype=float64.
    """
    return log_returns_series.rolling(window).kurt()


# --------------------------------------------------------------------- #
#  Lagged vol
# --------------------------------------------------------------------- #


def lagged_vol(rv_series: pd.Series, k: int) -> pd.Series:
    """Lag a realized-variance series by k periods.

    lagged_vol[t] = rv[t-k]

    Produces k leading NaN rows.  No future leakage: value at t uses only
    data up to and including t-k.

    Args:
        rv_series: Series of realized variance values.
        k: Number of periods to lag.  Must be >= 1.

    Returns:
        Series of lagged realized variance, same index as input.
        First ``k`` rows NaN.  dtype=float64.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return rv_series.shift(k)


# --------------------------------------------------------------------- #
#  Calendar features
# --------------------------------------------------------------------- #


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from the UTC DatetimeIndex.

    Appends four integer columns to a copy of ``df``:
    - ``day_of_week``: 0 = Monday, 4 = Friday.  For equity data with no weekend
      rows (Phase 1 Pandera gate) this will always be in [0, 4].  For crypto
      (24/7) it spans [0, 6].
    - ``month``: 1–12.
    - ``is_monday``: 1 on Mondays (day_of_week == 0), else 0.
    - ``is_friday``: 1 on Fridays (day_of_week == 4), else 0.

    Windows are row-wise (no rolling) — each row reads only its own index value.
    No look-ahead.

    Args:
        df: DataFrame with a tz-aware UTC DatetimeIndex named ``"date"``.
            All existing columns are preserved unchanged.

    Returns:
        A new DataFrame (copy) with the four additional integer columns.
    """
    df = df.copy()
    idx = df.index
    df["day_of_week"] = idx.dayofweek  # 0=Mon, 6=Sun
    df["month"] = idx.month  # 1-12
    df["is_monday"] = (idx.dayofweek == 0).astype(int)
    df["is_friday"] = (idx.dayofweek == 4).astype(int)
    return df
