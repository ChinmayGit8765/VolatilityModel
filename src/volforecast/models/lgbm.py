"""LightGBM model contracts for VolForecast Phase 3.

This module is the single source of truth for:

1. **Log-variance transforms** — ``to_log_var`` / ``from_log_var``:
   LightGBM is trained on log(target_variance) for scale stability across
   assets (crypto ~5e-4, equity ~1e-4).  Predictions are inverse-transformed
   back to variance scale before QLIKE/RMSE/MAE scoring.

2. **Asset-categorical constants** — ``KNOWN_ASSETS`` / ``ASSET_DTYPE``:
   Identical dtype applied at both training time and serving time.  Importing
   these constants from a single location prevents the training/serving skew
   described in Pitfall 3 of the Phase 3 research.

3. **Pooled fold assembly** — ``assemble_pooled_train``:
   Combines per-asset training rows for a given walk-forward fold index without
   any cross-asset temporal leakage (Pitfall 1).

Design notes
------------
- ``LOG_VAR_EPS`` is intentionally equal to ``eval.metrics.QLIKE_FLOOR``
  (both 1e-10).  The two constants serve different functions (log-transform
  floor vs. division-by-zero floor in QLIKE) but the same numeric value keeps
  the treatment of near-zero variance consistent across the pipeline.
- ``from_log_var`` returns the geometric mean of variance, not the arithmetic
  mean.  Specifically: ``exp(E[log(X)]) != E[X]`` (Jensen's inequality).
  This is a known, intentional bias. No lognormal bias correction
  (``exp(mu + sigma^2/2)``) is applied — it adds a second parameter to
  estimate and does not demonstrably improve QLIKE at daily frequency.
  Callers that need arithmetic-mean-unbiased forecasts must apply the
  correction themselves.

References
----------
- Plan 03-01 / Pattern 1 (pooled fold assembly)
- Plan 03-01 / Pattern 2 (log-variance target)
- Plan 03-01 / Pattern 7 (category dtype)
- Plan 03-01 / Pitfall 1 (cross-asset temporal leakage)
- Plan 03-01 / Pitfall 3 (training/serving dtype skew)
- Plan 03-01 / Pitfall 5 (Jensen's inequality bias)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from volforecast.eval.harness import walk_forward_splits

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Log-variance epsilon floor
# ---------------------------------------------------------------------------

#: Minimum variance value used by the log transform.
#: Intentionally equal to ``eval.metrics.QLIKE_FLOOR`` (both 1e-10) for
#: consistent near-zero variance treatment across the pipeline.
LOG_VAR_EPS: float = 1e-10


# ---------------------------------------------------------------------------
# Log-variance transforms
# ---------------------------------------------------------------------------


def to_log_var(var: float | Sequence | np.ndarray) -> np.ndarray:
    """Transform variance to log scale for LightGBM regression target.

    Applies ``log(max(var, LOG_VAR_EPS))`` element-wise.  Values at or below
    ``LOG_VAR_EPS`` (including zero and negative values) are floored, ensuring
    the result is always finite.

    Args:
        var: Scalar or array-like of variance values in decimal log-return
             variance units (~1e-4 for equity, ~5e-4 for crypto).

    Returns:
        np.ndarray of log-transformed variance values.  Always finite.
    """
    arr = np.asarray(var, dtype=float)
    return np.log(np.maximum(arr, LOG_VAR_EPS))


def from_log_var(log_var: float | Sequence | np.ndarray) -> np.ndarray:
    """Inverse-transform log-variance predictions back to variance scale.

    Applies ``exp(log_var)`` element-wise.  This is the geometric-mean
    inverse — see Jensen's Inequality caveat in the module docstring.

    **Jensen's inequality note:** ``exp(E[log(X)]) != E[X]``.  Predictions
    returned by this function are geometric-mean forecasts of variance, not
    arithmetic-mean.  No lognormal bias correction is applied (see module
    docstring for rationale).

    Args:
        log_var: Scalar or array-like of log-variance values (model outputs).

    Returns:
        np.ndarray of variance-scale predictions.  Always positive.
    """
    arr = np.asarray(log_var, dtype=float)
    return np.exp(arr)


# ---------------------------------------------------------------------------
# Asset-categorical constants
# ---------------------------------------------------------------------------

#: Canonical 5-asset universe for VolForecast.
#: Slugs match ``data/features/{crypto,equity}/{slug}.parquet`` paths.
#: This list is the single source of truth for both training and serving —
#: importing from here (rather than defining inline) prevents the Pitfall 3
#: training/serving skew.
KNOWN_ASSETS: list[str] = ["BTC-USD", "ETH-USD", "SPY", "AAPL", "MSFT"]

#: Pandas CategoricalDtype for the ``asset`` column.
#: Applied identically at training time (in ``assemble_pooled_train``) and
#: at serving time (in the FastAPI inference handler).  An unknown symbol
#: (not in ``KNOWN_ASSETS``) becomes a NaN category — this is documented
#: behaviour and should be caught upstream via input validation (SERVE-01
#: validates the ``symbol`` path parameter against ``KNOWN_ASSETS``).
ASSET_DTYPE: pd.CategoricalDtype = pd.CategoricalDtype(categories=KNOWN_ASSETS, ordered=False)


# ---------------------------------------------------------------------------
# Pooled fold assembly
# ---------------------------------------------------------------------------


def assemble_pooled_train(
    asset_feature_dfs: dict[str, pd.DataFrame],
    asset_target_series: dict[str, pd.Series],
    fold_i: int,
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble a pooled training set for fold ``fold_i`` without leakage.

    For each asset in ``asset_feature_dfs``:

    1. Generate the same ``walk_forward_splits`` sequence used by the baselines.
    2. If ``fold_i >= len(splits)`` for this asset, skip it (contributes zero
       rows — no error).
    3. Select rows at ``split.train_idx`` integer positions in **that asset's
       own DataFrame** (never a date-range slice).
    4. Add an ``"asset"`` column = symbol, drop NaN-target rows (the final
       ``horizon`` rows from ``compute_target``), and append.

    Test-fold rows are structurally excluded because only ``train_idx``
    positions are selected from each asset independently.  The cross-asset
    leakage invariant holds by construction: even if two assets share calendar
    dates in their training windows, the per-asset integer-position selection
    never mixes asset A's test rows into asset B's training context.

    Args:
        asset_feature_dfs:  Mapping from symbol to feature DataFrame.
                            Each DataFrame's integer positions must match the
                            target series (same index length and alignment).
        asset_target_series: Mapping from symbol to target pd.Series.
                             Typically ``to_log_var(compute_target(close))``.
                             The last ``horizon`` rows may be NaN (they are
                             dropped before pooling).
        fold_i:              Zero-based index into the per-asset split sequence.
        min_train:           Minimum purged training window size (default 252).
        step:                Walk-forward step size in observations (default 21).
        horizon:             Label horizon for purging (default 1).

    Returns:
        ``(X, y)`` tuple where:
        - ``X`` is a pooled pd.DataFrame with an ``"asset"`` column cast to
          ``ASSET_DTYPE``.  All other columns are from the original feature
          DataFrames.
        - ``y`` is the aligned pd.Series of log-variance training targets.

    Raises:
        ValueError: If no asset contributes rows for ``fold_i`` (all skipped
                    or all targets NaN), which would produce an empty pool and
                    silently train on nothing.
    """
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []

    for symbol, feat_df in asset_feature_dfs.items():
        splits = list(walk_forward_splits(len(feat_df), min_train, step, horizon))
        if fold_i >= len(splits):
            # Asset has fewer folds than fold_i — skip without error.
            continue

        split = splits[fold_i]
        x_train = feat_df.iloc[split.train_idx].copy()
        x_train["asset"] = symbol
        y_train = asset_target_series[symbol].iloc[split.train_idx]

        # Drop NaN targets (last `horizon` rows from compute_target).
        valid_mask = y_train.notna()
        x_parts.append(x_train.loc[valid_mask])
        y_parts.append(y_train.loc[valid_mask])

    if not x_parts:
        raise ValueError(
            f"fold_i={fold_i} produced an empty pooled training set — all assets "
            "were either skipped (fold_i >= len(splits)) or had all-NaN targets."
        )

    x_pooled = pd.concat(x_parts, ignore_index=True)
    y_pooled = pd.concat(y_parts, ignore_index=True)

    # Cast "asset" column to the canonical category dtype (Pitfall 3 mitigation).
    x_pooled["asset"] = x_pooled["asset"].astype(ASSET_DTYPE)

    return x_pooled, y_pooled
