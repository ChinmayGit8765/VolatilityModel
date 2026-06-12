"""Regime labelling utilities for volatility-regime analysis.

This module provides two pure functions for assigning regime labels to
walk-forward TEST-FOLD observations:

1. ``assign_vol_terciles`` — labels each row of a realized-variance series as
   "low", "mid", or "high" based on the 33rd and 67th percentiles of THAT
   SERIES (i.e. the test-fold realized variance for a single asset).

2. ``assign_calendar_year`` — returns the integer calendar year for each
   position in a DatetimeIndex.

Design invariants
-----------------
- **No lookahead by construction**: ``assign_vol_terciles`` takes only the
  test-fold realized-variance series as input.  It has no access to training
  data or the full-sample series — the tercile boundaries are computed solely
  from the observations passed in.  Callers MUST pass only test-fold data;
  passing training rows into this function is a methodological violation and
  cannot happen by accident (there is no "full series" argument).
- **Constant/degenerate input**: when all values in the input series are equal
  (or the series has length 1), the 33rd and 67th quantiles coincide with the
  single value.  In this case all rows are assigned "mid" to avoid a degenerate
  empty bucket.  No exception is raised.
- **Units**: the function is unit-agnostic.  It is designed for daily decimal
  realized variance (e.g. ~1e-4 for equity), but operates on any numeric series.

References
----------
- Plan 03-03 / Open Question #3 (vol terciles on test-fold realized variance
  only — no lookahead)
- Plan 03-03 / CONTEXT.md (Regimes: per-asset vol terciles low/mid/high +
  calendar year)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_vol_terciles(realized_var: pd.Series) -> pd.Series:
    """Label each observation as "low", "mid", or "high" volatility regime.

    Tercile boundaries are computed as the 1/3 and 2/3 quantiles of
    ``realized_var`` — the realized variance series that is passed in.

    **Lookahead prevention**: this function receives only the test-fold
    realized-variance series.  It has no access to training-fold data and
    no access to the full historical series.  The tercile boundaries are
    therefore computed solely from test-fold observations (no lookahead).
    Callers must not pass training-fold rows into this function.

    Edge cases:

    - **Constant input**: all values equal (e.g. all-zero after a yfinance
      artifact) — both quantile boundaries coincide, so all rows are assigned
      "mid" without raising.
    - **Tied quantile boundaries**: heavy mass at a single value (e.g. floored
      zero-returns) can make q33 equal the series minimum or q67 equal q33.
      Labels are assigned by direct quantile comparison (``np.where``), which
      never raises on non-monotonic boundaries — unlike ``pd.cut`` with
      explicit bins (WR-01).
    - **Single observation**: one-row Series is handled by the same "mid"
      fallback path.

    Args:
        realized_var: pd.Series of daily decimal realized variance for one
            asset over the test-fold evaluation period.  Should have a
            DatetimeIndex or a positional integer index.  Positive floats.

    Returns:
        pd.Series of str labels — each element is one of ``{"low", "mid",
        "high"}``.  Same index as ``realized_var``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rv = pd.Series(np.linspace(0.0001, 0.001, 90))
        >>> labels = assign_vol_terciles(rv)
        >>> set(labels.unique()) <= {"low", "mid", "high"}
        True
    """
    q33 = float(realized_var.quantile(1 / 3))
    q67 = float(realized_var.quantile(2 / 3))

    if q33 == q67:
        # Degenerate / constant input — assign all rows to "mid"
        return pd.Series(
            ["mid"] * len(realized_var),
            index=realized_var.index,
            dtype=str,
        )

    # WR-01: label by direct quantile comparison.  pd.cut with explicit bins
    # raises on non-monotonic boundaries when ties collapse a bin edge onto
    # the series min (mass at floored zero-returns), and a ±1e-30 bin padding
    # is a float no-op at typical variance magnitudes (~1e-4).  np.where has
    # no monotonicity requirement: low = (-inf, q33], mid = (q33, q67],
    # high = (q67, inf) — identical boundary semantics to the previous
    # right-closed bins.
    labels = np.where(
        realized_var <= q33,
        "low",
        np.where(realized_var <= q67, "mid", "high"),
    )
    return pd.Series(labels, index=realized_var.index, dtype=str)


def assign_calendar_year(index: pd.DatetimeIndex) -> pd.Series:
    """Return the integer calendar year for each position in a DatetimeIndex.

    Extracts the ``.year`` attribute from each date in ``index``.  Useful for
    segmenting walk-forward test-fold observations by calendar year when
    building per-regime evaluation breakdowns.

    Args:
        index: pd.DatetimeIndex (e.g. the index of a test-fold DataFrame or
            realized-variance Series).  Must contain valid datetime values.

    Returns:
        pd.Series of int64 calendar years, same index as ``index``.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range("2020-01-01", periods=3, freq="YE")
        >>> assign_calendar_year(idx).tolist()
        [2020, 2021, 2022]
    """
    return pd.Series(index.year, index=index, dtype="int64")
