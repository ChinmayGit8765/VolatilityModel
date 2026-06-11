"""Single versioned feature codepath for VolForecast (FEAT-07).

This module is the ONE place that produces the feature matrix consumed by both
training (Phase 3 LightGBM) and serving (Phase 4 FastAPI).  Training and
serving both call ``build_features`` — there is no second feature implementation
anywhere in the codebase.

Feature set (all in decimal log-return VARIANCE units — no annualization):
    - rv_5, rv_10, rv_22, rv_66: multi-lookback rolling realized variance
    - log_return: current-row decimal log return
    - squared_return: current-row squared log return
    - lagged_vol: rv_22 shifted by 1 period (yesterday's realized variance)
    - ewma_var: EWMA variance (lambda=0.94, RiskMetrics)
    - garch_cond_var: GARCH(1,1) filtered conditional VARIANCE (std dev²) from
          monthly-refit walk-forward filter; never fitted on data past as-of t
    - parkinson_var: row-wise Parkinson range estimator
    - gk_var: row-wise Garman-Klass range estimator
    - vol_of_vol: rolling std of rv_22 over a 22-period window
    - rolling_skew: rolling skewness of log returns over 22 periods
    - rolling_kurt: rolling excess kurtosis of log returns over 22 periods
    - day_of_week, month, is_monday, is_friday: calendar features from UTC index

GARCH-as-feature implementation contract (T-02-11 mitigation):
    The GARCH(1,1) filtered conditional volatility is extracted via a monthly-
    refit walk-forward filter.  For every row at position t, the feature value
    is derived from a GARCH model fitted on data[:refit_end] where refit_end <=
    t.  The conditional volatility series produced by arch's ``filter()`` on the
    training slice uses only in-sample returns, so the filtered value at the last
    in-sample point never leaks future data.  The value at t is therefore the
    filtered in-sample conditional variance for the most recent refit window that
    covers t.  This is consistent across training and serving as long as the same
    refit schedule is applied.

    The minimum training window for GARCH is ``GARCH_MIN_TRAIN = 252`` (one year).
    Rows before the first successful GARCH refit receive NaN.

Lazy import contract:
    The ``arch`` library is imported inside the GARCH branch only.  When
    ``include_garch=False``, the module loads without any arch dependency.

Cross-asset contract (FEAT-05):
    When ``cross_asset_dfs`` is provided, each source DataFrame is as-of-joined
    onto the target using ``cross_asset.as_of_join`` with a 3-day staleness cap.
    Source DataFrames should already contain the desired feature columns.
    Joined columns are suffixed with the source's dict key (``{col}_{asset_name}``,
    e.g. ``rv_22_btc``) so multiple sources sharing a feature name never collide
    and column names are stable regardless of dict order (WR-01 / FEAT-07).

Units:
    ALL output columns are decimal log-return VARIANCE (e.g. ~1e-4 for typical
    daily equity).  NO percent-squared.  NO annualization.  Callers must not
    divide/multiply by 252 or 100 for model inputs.

References:
    - FEAT-07: single codepath invariant
    - T-02-11: feature window / GARCH-as-feature leak mitigation
    - 02-RESEARCH.md: architecture pattern and pitfall list
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from volforecast.features.cross_asset import as_of_join
from volforecast.features.estimators import (
    calendar_features,
    ewma_variance,
    garman_klass_var,
    lagged_vol,
    log_returns,
    parkinson_var,
    realized_var,
    rolling_kurt,
    rolling_skew,
    squared_returns,
    vol_of_vol,
)

if TYPE_CHECKING:
    pass  # arch imported lazily below

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Realized-variance lookback windows (integer count-based, Pitfall 3).
RV_WINDOWS: tuple[int, ...] = (5, 10, 22, 66)

#: Window for vol-of-vol (rolling std of rv_22).
VOL_OF_VOL_WINDOW: int = 22

#: Window for rolling skew and kurtosis.
SKEW_KURT_WINDOW: int = 22

#: Lag for lagged_vol (rv_22 shifted by 1 period).
LAGGED_VOL_K: int = 1

#: Minimum training observations before the first GARCH refit.
GARCH_MIN_TRAIN: int = 252

#: GARCH refit cadence (monthly = 21 trading days).
GARCH_STEP: int = 21

#: GARCH scale factor (100x decimal log returns).
_GARCH_SCALE: float = 100.0


# ---------------------------------------------------------------------------
# GARCH conditional-variance feature (internal helper)
# ---------------------------------------------------------------------------


def _garch_conditional_var_feature(
    log_returns_series: pd.Series,
) -> pd.Series:
    """Compute the GARCH(1,1) conditional VARIANCE as a feature series.

    Uses the monthly-refit walk-forward forecasting pattern from
    ``models.garch.GARCH.forecast_path``.  At each position t, the feature
    value is the one-step-ahead variance forecast made from a GARCH model
    trained on data up to and including the most recent refit boundary <= t.

    Because the forecast at t is produced from a model fitted on data[0..refit_pos)
    where refit_pos <= t, the feature never uses data past as-of t.  Truncation
    invariance: feature[t] is identical whether computed on data[:t+1] or the
    full series (same refit schedule up to t).  This satisfies T-02-11.

    The ``arch`` library is imported lazily inside this function.

    Args:
        log_returns_series: pd.Series of decimal log returns (first row NaN).

    Returns:
        pd.Series of GARCH conditional VARIANCE (one-step-ahead forecast),
        same index as input.  Positions < GARCH_MIN_TRAIN are NaN.
        All other values are positive float64 decimal variance values.
    """
    from volforecast.models.garch import GARCH

    garch = GARCH(min_train=GARCH_MIN_TRAIN, step=GARCH_STEP)
    cond_var = garch.forecast_path(log_returns_series)

    if garch.fallback_count > 0:
        log.info(
            "GARCH feature: %d refit windows used fallback (convergence failure or EWMA)",
            garch.fallback_count,
        )

    return cond_var


# ---------------------------------------------------------------------------
# Public API — single codepath (FEAT-07)
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    cross_asset_dfs: dict[str, pd.DataFrame] | None = None,
    include_garch: bool = True,
) -> pd.DataFrame:
    """Build the complete feature matrix from OHLCV data.

    This is the SINGLE feature codepath imported identically by training
    (Phase 3) and serving (Phase 4).  There is exactly one definition of
    this function in the codebase (FEAT-07).

    Every feature window ends strictly at as-of t (no look-ahead).  The
    no-lookahead truncation-invariance test in ``tests/unit/test_no_lookahead.py``
    proves this: feature[t] is identical whether computed on data[:t+1] or
    the full series.

    Args:
        df: OHLCV DataFrame with a tz-aware UTC DatetimeIndex named "date".
            Must conform to the Phase-1 processed-parquet contract:
            columns open, high, low, close, volume; float64; no NaN price rows.
        cross_asset_dfs: Optional dict mapping asset names to DataFrames of
            pre-computed cross-asset features.  Each DataFrame must have a
            tz-aware UTC DatetimeIndex named "date" and float64 columns.
            The most recent prior value is joined with a 3-day staleness cap
            (NaN beyond).  Each source's columns are suffixed with its dict
            key (``{col}_{asset_name}``) so identical feature names from
            different sources stay distinct and asset-identifying (WR-01).
            Default None (no cross-asset features).
        include_garch: If True (default), compute the GARCH(1,1) filtered
            conditional VARIANCE as a feature (``garch_cond_var``).  Setting
            to False skips the GARCH computation and omits ``garch_cond_var``
            from the output — this also avoids importing the ``arch`` library
            at call time (lazy import contract).

    Returns:
        pd.DataFrame with the same UTC DatetimeIndex as ``df`` and the full
        feature column set (see module docstring).  Columns with insufficient
        history for their rolling window contain leading NaN rows.  All values
        are in decimal log-return VARIANCE units.

    Raises:
        KeyError: If ``df`` is missing required OHLCV columns.
    """
    # ---- 1. Base return features (row-wise, no rolling) --------------------
    lr = log_returns(df["close"])
    sr = squared_returns(df["close"])

    # ---- 2. Multi-lookback realized variance --------------------------------
    rv_series: dict[str, pd.Series] = {}
    for w in RV_WINDOWS:
        rv_series[f"rv_{w}"] = realized_var(lr, window=w)

    # ---- 3. EWMA variance ---------------------------------------------------
    ewma = ewma_variance(lr)

    # ---- 4. Lagged vol (rv_22 shifted by 1 period) -------------------------
    lag_vol = lagged_vol(rv_series["rv_22"], k=LAGGED_VOL_K)

    # ---- 5. Range-based estimators (row-wise) --------------------------------
    park = parkinson_var(df)
    gk = garman_klass_var(df)

    # ---- 6. Vol-of-vol (rolling std of rv_22) --------------------------------
    vov = vol_of_vol(rv_series["rv_22"], window=VOL_OF_VOL_WINDOW)

    # ---- 7. Rolling skew / kurtosis -----------------------------------------
    rskew = rolling_skew(lr, window=SKEW_KURT_WINDOW)
    rkurt = rolling_kurt(lr, window=SKEW_KURT_WINDOW)

    # ---- 8. Assemble intermediate DataFrame ----------------------------------
    # Ensure the index is named "date" for the cross_asset.as_of_join contract.
    out_index = df.index.rename("date")
    out = pd.DataFrame(
        {
            "log_return": lr,
            "squared_return": sr,
            **rv_series,
            "ewma_var": ewma,
            "lagged_vol": lag_vol,
            "parkinson_var": park,
            "gk_var": gk,
            "vol_of_vol": vov,
            "rolling_skew": rskew,
            "rolling_kurt": rkurt,
        },
        index=out_index,
    )

    # ---- 9. Calendar features (row-wise, no rolling) ------------------------
    # Pass a minimal DataFrame with just the index so calendar_features can
    # read the DatetimeIndex and append columns.
    cal_df = calendar_features(pd.DataFrame(index=df.index))
    for col in ["day_of_week", "month", "is_monday", "is_friday"]:
        out[col] = cal_df[col]

    # ---- 10. GARCH conditional variance feature (lazy import) ---------------
    if include_garch:
        out["garch_cond_var"] = _garch_conditional_var_feature(lr)

    # ---- 11. Cross-asset as-of join (optional) --------------------------------
    if cross_asset_dfs:
        for asset_name, source_df in cross_asset_dfs.items():
            feature_cols = [c for c in source_df.columns if c != "date"]
            if not feature_cols:
                continue
            # Suffix per SOURCE asset (WR-01): two sources sharing a feature
            # name (e.g. both providing rv_22) must produce distinct, stable,
            # asset-identifying columns (rv_22_btc, rv_22_eth) — never the
            # dict-order-dependent pandas _x/_y collision names (FEAT-07).
            out = as_of_join(out, source_df, feature_cols=feature_cols, suffix=f"_{asset_name}")

    return out
