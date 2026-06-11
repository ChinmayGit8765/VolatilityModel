"""Canonical realized-volatility target definition for VolForecast.

Unit convention (enforced throughout the pipeline):
    - Target is daily VARIANCE of decimal log returns.
    - Log returns use the natural logarithm of the close-to-close ratio.
    - Values are in decimal units (e.g., ~1e-4 for typical daily equity);
      NOT in percent-squared (~1.0) and NOT annualized.
    - Annualization (multiply by ~252) is display-only at report edges.
    - vol (standard deviation) is sqrt(target) at report edges only;
      the pipeline stores and scores variance throughout.

This module is the single source of truth for the target proxy.  Every
later plan (baselines, ML model, serving) imports from here — no duplication,
no skew.

Target proxy justification (Patton 2011):
    The squared daily log return is a noisy but conditionally unbiased proxy
    for daily realized variance.  QLIKE loss (see eval/metrics.py) is robust
    to this measurement noise, making it the preferred scoring metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical forecast horizon: next-day (1 trading period ahead).
# Downstream harness and feature pipeline reference this constant so the
# horizon is defined exactly once.
HORIZON: int = 1


def compute_target(close: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Compute the next-`horizon` realized variance proxy for a close-price series.

    For horizon=1 (default) the proxy is the next-day squared decimal log return:

        target[t] = (ln(close[t+1] / close[t]))**2

    The value is indexed at t (label window starts at t+1, no overlap with
    feature windows that end at t).  The last `horizon` rows are NaN because
    no forward close price exists — callers MUST NOT fill these NaNs with zero
    (Pitfall 4 guard: a zero target is meaningless and corrupts training).

    For horizon > 1, the proxy is the mean of the next `horizon` squared log
    returns (a realized variance estimate over the forward window), also indexed
    at t.

    Args:
        close: Series of close prices with a tz-aware UTC DatetimeIndex named
               "date".  Must conform to the Phase-1 processed-parquet contract:
               float64, no weekends for equity, no NaN price rows.
        horizon: Number of forward periods.  Default HORIZON=1.

    Returns:
        pd.Series with the same index as `close`, dtype float64.
        Values at positions [0 .. n-horizon-1] are the variance proxy.
        Values at positions [n-horizon .. n-1] are NaN.
    """
    log_returns = np.log(close).diff()  # r[t] = ln(close[t]/close[t-1])

    if horizon == 1:
        # Next-day: shift log_return back by 1 so target[t] = r[t+1]^2
        next_return = log_returns.shift(-1)
        target = next_return**2
    else:
        # Multi-step: mean of next `horizon` squared log returns
        sq_returns = log_returns**2
        # Rolling forward sum of length `horizon`, then shift back by horizon
        # rolling(horizon).sum() at position t uses r[t-horizon+1..t]
        # After shift(-horizon) that maps to r[t+1..t+horizon]
        target = sq_returns.rolling(horizon).mean().shift(-horizon)

    target.name = close.name
    return target


def forward_realized_var(close: pd.Series, window: int = 5) -> pd.Series:
    """Compute mean realized variance over the next `window` trading periods.

    This is the secondary stability check target described in the CONTEXT.md:
    "5-day forward realized variance" used alongside the primary next-day proxy.

        fwd_rv[t] = mean( (ln(close[t+k+1]/close[t+k]))**2 for k in 1..window )

    The last `window` rows are NaN (no complete forward window available).
    NaNs must NOT be filled with zero — callers must drop them before training.

    Args:
        close: Series of close prices, same contract as compute_target.
        window: Number of forward squared log returns to average.  Default 5.

    Returns:
        pd.Series with the same index as `close`, dtype float64.
        Non-NaN for positions [0 .. n-window-1]; NaN for the last `window` rows.
    """
    log_returns = np.log(close).diff()  # r[t] = ln(close[t]/close[t-1])
    sq_returns = log_returns**2

    # rolling(window).mean() at position t uses sq_returns[t-window+1..t]
    # shift(-window) maps to sq_returns[t+1..t+window] — the forward window
    fwd_rv = sq_returns.rolling(window).mean().shift(-window)
    fwd_rv.name = close.name
    return fwd_rv
