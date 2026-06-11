"""Deterministic feature estimators for VolForecast.

This module provides stateless, purely computational helpers for the feature
pipeline and for the EWMA baseline.  All functions are vectorised over pandas
Series and produce Series with the same DatetimeIndex as the input.

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

References:
    - J.P. Morgan / Reuters (1996). RiskMetrics Technical Document.
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
