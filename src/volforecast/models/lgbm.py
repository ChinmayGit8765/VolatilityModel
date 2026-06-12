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

4. **Grid search and training** — ``PARAM_GRID``, ``grid_search``,
   ``train_pooled_model``, ``train_per_fold_models``:
   Hyperparameter selection on inner validation folds only (never test folds —
   Pitfall 2).  One pooled model is trained PER outer walk-forward fold with
   the shared best params (CR-01) — mirroring the baselines' per-step refit
   discipline.  The final fold's model is the registry champion.

5. **Per-asset evaluation** — ``evaluate_per_asset``:
   Walk-forward variance-scale RMSE/MAE/QLIKE on identical folds as Phase 2
   baselines, using the canonical eval/metrics.py functions.  Each test fold
   is scored ONLY by the model trained at that fold's own cutoff — never by a
   model whose training window contains the fold (CR-01).

6. **SHAP explainability** — ``compute_shap_artifacts``:
   TreeExplainer on the native LGBMRegressor object (Pitfall 6 — never
   the pyfunc wrapper).  Saves global bar + beeswarm PNGs; returns per-asset
   top-10 feature importance.

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
- Grid search uses ``eval_set`` (4.6 API — NOT eval_X/eval_y which are 4.7+)
  with ``early_stopping(stopping_rounds=50)`` and ``random_state=42``.
- Inner validation for grid search uses the last 3 folds of each asset's
  train window (Open Question #2 in RESEARCH.md) — strictly inside train
  boundaries, never test folds (Pitfall 2 mitigation).

References
----------
- Plan 03-01 / Pattern 1 (pooled fold assembly)
- Plan 03-01 / Pattern 2 (log-variance target)
- Plan 03-01 / Pattern 7 (category dtype)
- Plan 03-01 / Pitfall 1 (cross-asset temporal leakage)
- Plan 03-01 / Pitfall 2 (grid search on test folds — FORBIDDEN)
- Plan 03-01 / Pitfall 3 (training/serving dtype skew)
- Plan 03-01 / Pitfall 5 (Jensen's inequality bias)
- Plan 03-01 / Pitfall 6 (SHAP on pyfunc wrapper — FORBIDDEN)
- Plan 03-02 / Pattern 4 (SHAP TreeExplainer artifacts)
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import (headless CI / Windows)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from volforecast.eval.harness import walk_forward_splits
from volforecast.eval.metrics import mae, qlike, rmse

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------

#: Small fixed grid producing ~20 combos for inner-validation-only grid search.
#: 3 * 2 * 2 * 2 = 24 combos — within the 12-32 acceptance range.
#: Chosen per CONTEXT.md locked decisions: regularisation-heavy for small
#: pooled dataset (~5000 rows), fixed random_state=42 for reproducibility.
PARAM_GRID: dict[str, list[Any]] = {
    "num_leaves": [15, 20, 31],
    "min_child_samples": [50, 100],
    "learning_rate": [0.05, 0.01],
    "reg_lambda": [1.0, 5.0],
}
# 3 * 2 * 2 * 2 = 24 combos

# ---------------------------------------------------------------------------
# Inner-validation grid search (Pitfall 2 — test folds NEVER referenced)
# ---------------------------------------------------------------------------

#: Number of inner validation folds used for grid search (Open Question #2).
#: Last 3 folds of the training window give ~3 months of signal; using more
#: would encroach on useful training data.
_INNER_VAL_FOLDS: int = 3


def _build_inner_val_set(
    asset_feature_dfs: dict[str, pd.DataFrame],
    asset_target_series: dict[str, pd.Series],
    outer_fold_i: int,
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build an inner validation set from the last 3 folds of each train window.

    For each asset, we look at the training window for ``outer_fold_i`` and
    carve out the last ``_INNER_VAL_FOLDS`` * ``step`` positions as inner val.
    The remaining positions are the inner training set.

    INVARIANT: all inner-val positions come strictly from within the train
    window of ``outer_fold_i`` — no test-fold positions are ever used
    (Pitfall 2 mitigation).

    Returns:
        (X_val, y_val) concatenated across all assets.
    """
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []

    for symbol, feat_df in asset_feature_dfs.items():
        splits = list(walk_forward_splits(len(feat_df), min_train, step, horizon))
        if outer_fold_i >= len(splits):
            continue

        split = splits[outer_fold_i]
        train_idx = split.train_idx  # strictly inside train window

        # Inner val = last (_INNER_VAL_FOLDS * step) positions of train_idx
        n_val = _INNER_VAL_FOLDS * step
        if len(train_idx) <= n_val:
            # Train window too small to carve out inner val — skip
            continue

        inner_val_idx = train_idx[-n_val:]

        x_val = feat_df.iloc[inner_val_idx].copy()
        x_val["asset"] = symbol
        y_val = asset_target_series[symbol].iloc[inner_val_idx]

        valid_mask = y_val.notna()
        x_parts.append(x_val.loc[valid_mask])
        y_parts.append(y_val.loc[valid_mask])

    if not x_parts:
        raise ValueError(f"outer_fold_i={outer_fold_i} produced an empty inner validation set.")

    x_val_pooled = pd.concat(x_parts, ignore_index=True)
    y_val_pooled = pd.concat(y_parts, ignore_index=True)
    x_val_pooled["asset"] = x_val_pooled["asset"].astype(ASSET_DTYPE)
    return x_val_pooled, y_val_pooled


def _build_inner_train_set(
    asset_feature_dfs: dict[str, pd.DataFrame],
    asset_target_series: dict[str, pd.Series],
    outer_fold_i: int,
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the inner training set (train window minus inner val tail)."""
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []

    for symbol, feat_df in asset_feature_dfs.items():
        splits = list(walk_forward_splits(len(feat_df), min_train, step, horizon))
        if outer_fold_i >= len(splits):
            continue

        split = splits[outer_fold_i]
        train_idx = split.train_idx

        n_val = _INNER_VAL_FOLDS * step
        if len(train_idx) <= n_val:
            # Use full train window as inner train (no inner val — handled in val builder)
            inner_train_idx = train_idx
        else:
            inner_train_idx = train_idx[:-n_val]

        x_tr = feat_df.iloc[inner_train_idx].copy()
        x_tr["asset"] = symbol
        y_tr = asset_target_series[symbol].iloc[inner_train_idx]

        valid_mask = y_tr.notna()
        x_parts.append(x_tr.loc[valid_mask])
        y_parts.append(y_tr.loc[valid_mask])

    if not x_parts:
        raise ValueError(f"outer_fold_i={outer_fold_i} produced an empty inner training set.")

    x_tr_pooled = pd.concat(x_parts, ignore_index=True)
    y_tr_pooled = pd.concat(y_parts, ignore_index=True)
    x_tr_pooled["asset"] = x_tr_pooled["asset"].astype(ASSET_DTYPE)
    return x_tr_pooled, y_tr_pooled


def _min_fold_count(
    asset_feature_dfs: dict[str, pd.DataFrame],
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
) -> int:
    """Return the number of walk-forward folds common to ALL assets.

    Raises:
        ValueError: if no asset produces any fold (check min_train/step/n_rows).
    """
    min_folds: int | None = None
    for feat_df in asset_feature_dfs.values():
        n_folds = len(list(walk_forward_splits(len(feat_df), min_train, step, horizon)))
        if min_folds is None or n_folds < min_folds:
            min_folds = n_folds

    if min_folds is None or min_folds == 0:
        raise ValueError("No walk-forward folds found — check min_train/step/n_rows.")
    return min_folds


def grid_search(
    asset_feature_dfs: dict[str, pd.DataFrame],
    asset_target_series: dict[str, pd.Series],
    param_grid: dict[str, list[Any]] | None = None,
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
    n_estimators: int = 500,
    stopping_rounds: int = 50,
    verbose: bool = True,
) -> dict[str, Any]:
    """Select best hyperparameters on inner validation folds only.

    For each combo in ``param_grid``, trains an LGBMRegressor on the inner
    training set of the LAST outer fold and evaluates QLIKE on the inner
    validation set.  The combo with the lowest average inner-val QLIKE wins.

    CRITICAL: test folds are NEVER referenced here (Pitfall 2 mitigation).
    Only the last outer training fold is used; inner val is carved from its
    tail.

    Args:
        asset_feature_dfs: Mapping symbol → feature DataFrame.
        asset_target_series: Mapping symbol → log-variance target Series.
            Pass ``to_log_var(compute_target(close))`` — targets in log space.
        param_grid: Dict of param name → list of values to try.
            Defaults to ``PARAM_GRID``.
        min_train: Walk-forward harness parameter (default 252).
        step: Walk-forward step size (default 21).
        horizon: Label horizon (default 1).
        n_estimators: Max boosting rounds (default 500; early stopping controls
            actual rounds).
        stopping_rounds: Early stopping rounds (default 50).
        verbose: If True, log progress via module logger.

    Returns:
        Dict of best hyperparameters (keys match ``param_grid`` keys).
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    combos = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())

    if verbose:
        log.info("Grid search: %d combos", len(combos))

    # Use the LAST outer fold available across all assets for inner-val selection.
    min_folds = _min_fold_count(asset_feature_dfs, min_train, step, horizon)
    outer_fold_i = min_folds - 1  # last fold available for all assets

    x_tr, y_tr = _build_inner_train_set(
        asset_feature_dfs, asset_target_series, outer_fold_i, min_train, step, horizon
    )
    x_val, y_val = _build_inner_val_set(
        asset_feature_dfs, asset_target_series, outer_fold_i, min_train, step, horizon
    )

    best_qlike = float("inf")
    best_params: dict[str, Any] = {}

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        model = LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            random_state=42,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            **params,
        )
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=stopping_rounds, verbose=False),
                log_evaluation(period=-1),  # silence per-round output
            ],
            categorical_feature="auto",
        )
        val_preds_log = model.predict(x_val)
        val_preds_var = from_log_var(val_preds_log)
        val_true_var = from_log_var(y_val.values)
        try:
            val_qlike = qlike(val_true_var, val_preds_var)
        except ValueError:
            val_qlike = float("inf")

        if verbose:
            log.info(
                "  combo %d/%d params=%s inner_val_qlike=%.4f best=%.4f",
                i + 1,
                len(combos),
                params,
                val_qlike,
                best_qlike,
            )

        if val_qlike < best_qlike:
            best_qlike = val_qlike
            best_params = params

    if verbose:
        log.info("Best params: %s  (inner_val_qlike=%.4f)", best_params, best_qlike)

    return best_params


def train_pooled_model(
    asset_feature_dfs: dict[str, pd.DataFrame],
    asset_target_series: dict[str, pd.Series],
    params: dict[str, Any],
    fold_i: int,
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
    n_estimators: int = 1000,
    stopping_rounds: int = 50,
) -> tuple[LGBMRegressor, int]:
    """Train a pooled LightGBM model on fold ``fold_i`` with given params.

    Assembles the pooled training set via ``assemble_pooled_train``, carves
    an inner validation window from each asset's train tail (last 3 inner
    folds per Open Question #2), and fits LGBMRegressor with early stopping.

    INVARIANT: inner validation rows come from within the train window of
    ``fold_i`` only — test folds are never referenced (Pitfall 2 mitigation).

    Args:
        asset_feature_dfs: Mapping symbol → feature DataFrame.
        asset_target_series: Mapping symbol → log-variance target Series.
        params: Hyperparameter dict (e.g., from ``grid_search``).
        fold_i: Walk-forward fold index for the outer training window.
        min_train: Minimum training window size (default 252).
        step: Walk-forward step size (default 21).
        horizon: Label horizon (default 1).
        n_estimators: Max boosting rounds (default 1000; early stopping
            controls actual rounds).
        stopping_rounds: Early stopping rounds (default 50).

    Returns:
        ``(model, best_iteration)`` where ``model`` is the fitted
        LGBMRegressor and ``best_iteration`` is ``model.best_iteration_``.
    """
    x_train, y_train = _build_inner_train_set(
        asset_feature_dfs, asset_target_series, fold_i, min_train, step, horizon
    )
    x_val, y_val = _build_inner_val_set(
        asset_feature_dfs, asset_target_series, fold_i, min_train, step, horizon
    )

    model = LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        random_state=42,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=stopping_rounds, verbose=False),
            log_evaluation(period=100),
        ],
        categorical_feature="auto",
    )
    return model, model.best_iteration_


def train_per_fold_models(
    asset_feature_dfs: dict[str, pd.DataFrame],
    asset_target_series: dict[str, pd.Series],
    params: dict[str, Any],
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
    n_estimators: int = 1000,
    stopping_rounds: int = 50,
    verbose: bool = True,
) -> dict[int, LGBMRegressor]:
    """Train one pooled LightGBM model per outer walk-forward fold (CR-01).

    Mirrors the baselines' walk-forward refit discipline: the model that
    scores fold ``k``'s test window is trained ONLY on data inside fold
    ``k``'s training window (with early stopping on that fold's own inner-val
    carve).  This is what makes the ML-vs-baseline comparison leak-free — a
    single model trained at the LAST fold's cutoff would contain every earlier
    test window inside its training set (train-on-test contamination).

    Args:
        asset_feature_dfs: Mapping symbol → feature DataFrame.
        asset_target_series: Mapping symbol → log-variance target Series.
        params: Shared hyperparameter dict (e.g., from ``grid_search``).
        min_train: Minimum training window size (default 252).
        step: Walk-forward step size (default 21).
        horizon: Label horizon (default 1).
        n_estimators: Max boosting rounds per model (early stopping controls
            actual rounds).
        stopping_rounds: Early stopping rounds (default 50).
        verbose: If True, log per-fold progress via module logger.

    Returns:
        ``{fold_i: fitted LGBMRegressor}`` for ``fold_i`` in
        ``range(min_fold_count)``, where ``min_fold_count`` is the number of
        folds common to all assets.  The last entry (``min_fold_count - 1``)
        is the registry champion candidate.
    """
    min_folds = _min_fold_count(asset_feature_dfs, min_train, step, horizon)
    fold_models: dict[int, LGBMRegressor] = {}
    for fold_i in range(min_folds):
        model, best_iter = train_pooled_model(
            asset_feature_dfs,
            asset_target_series,
            params=params,
            fold_i=fold_i,
            min_train=min_train,
            step=step,
            horizon=horizon,
            n_estimators=n_estimators,
            stopping_rounds=stopping_rounds,
        )
        fold_models[fold_i] = model
        if verbose:
            log.info(
                "  per-fold model %d/%d trained (best_iteration=%d)",
                fold_i + 1,
                min_folds,
                best_iter,
            )
    return fold_models


def resolve_fold_model(fold_models: Mapping[int, LGBMRegressor], fold_i: int) -> LGBMRegressor:
    """Return the model that may legally score fold ``fold_i``'s test window.

    Folds beyond the pooled-model range (``fold_i > max(fold_models)``) exist
    only for assets with longer histories than the shortest asset.  The final
    fold's model is leak-free for those later windows too: with an expanding
    window, its per-asset training cutoff (``min_train + max_fold*step -
    horizon``) lies strictly before any later fold's test start
    (``min_train + fold_i*step``).

    Raises:
        KeyError: if ``fold_i`` falls INSIDE the trained range but is missing —
            a corrupted mapping must never be silently substituted.
    """
    if fold_i in fold_models:
        return fold_models[fold_i]
    max_fold = max(fold_models)
    if fold_i > max_fold:
        return fold_models[max_fold]
    raise KeyError(
        f"fold_models has no model for fold {fold_i} (available: {sorted(fold_models)})."
    )


# ---------------------------------------------------------------------------
# Per-asset evaluation
# ---------------------------------------------------------------------------


def evaluate_per_asset(
    fold_models: Mapping[int, LGBMRegressor],
    asset_feature_dfs: dict[str, pd.DataFrame],
    asset_target_series: dict[str, pd.Series],
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
) -> dict[str, dict[str, float]]:
    """Evaluate the per-fold models per-asset on walk-forward test folds.

    Uses the identical ``walk_forward_splits`` sequence as the Phase 2
    baselines.  Predictions are inverse-transformed (``from_log_var``) before
    scoring, so all metrics are on the VARIANCE scale — apples-to-apples
    comparison with ``reports/baseline_metrics.csv``.

    Leak-free per-fold scoring (CR-01): each test fold ``k`` is predicted by
    ``fold_models[k]`` — the model trained at that fold's own cutoff (see
    ``train_per_fold_models``).  Folds beyond ``max(fold_models)`` (longer
    assets only) fall back to the final fold's model, which remains strictly
    out-of-sample for those later windows under the expanding window (see
    ``resolve_fold_model``).

    Cross-asset column alignment (CR-02): different assets have different
    cross-asset feature columns (e.g. ``rv_22_eth`` for BTC, ``rv_22_btc``
    for others).  At evaluation time, every test DataFrame is reindexed to
    ``model.feature_name_`` — the EXACT column order the model was trained
    on.  Missing cross-asset columns are NaN-filled (same as in pooled
    training) and ``predict`` is called with ``validate_features=True`` so a
    name/order mismatch raises instead of silently scoring a transposed
    matrix (LightGBM's default ``validate_features=False`` checks only the
    column COUNT).

    Args:
        fold_models: Mapping fold index → fitted LGBMRegressor (native model
            objects, not pyfunc) from ``train_per_fold_models``.
        asset_feature_dfs: Mapping symbol → feature DataFrame.
        asset_target_series: Mapping symbol → log-variance target Series.
        min_train: Walk-forward harness min train parameter (default 252).
        step: Walk-forward step size (default 21).
        horizon: Label horizon (default 1).

    Returns:
        Dict ``{symbol: {"rmse": ..., "mae": ..., "qlike": ...,
        "n_forecasts": ...}}``.
    """
    results: dict[str, dict[str, float]] = {}

    for symbol, feat_df in asset_feature_dfs.items():
        target_series = asset_target_series[symbol]
        all_pred_var: list[np.ndarray] = []
        all_true_var: list[np.ndarray] = []

        for fold_i, split in enumerate(walk_forward_splits(len(feat_df), min_train, step, horizon)):
            # CR-01: score this fold ONLY with the model trained at this
            # fold's own cutoff (or the leak-free final-fold fallback).
            model = resolve_fold_model(fold_models, fold_i)
            # The model's training column order is the single source of truth
            # for the eval feature schema (CR-02).
            feature_order = list(model.feature_name_)

            test_idx = split.test_idx
            x_test = feat_df.iloc[test_idx].copy()
            x_test["asset"] = symbol

            # Reindex to the model's exact training column order (CR-02).
            # Missing cross-asset columns are NaN-filled; extras dropped.
            x_test = x_test.reindex(columns=feature_order)
            # Restore ASSET_DTYPE after reindex (categorical dtype must match
            # training exactly — Pitfall 3).
            x_test["asset"] = x_test["asset"].astype(ASSET_DTYPE)

            # Compute realized variance (variance scale, not log)
            y_test_log = target_series.iloc[test_idx]
            valid_mask = y_test_log.notna()
            if valid_mask.sum() == 0:
                continue

            x_test_valid = x_test.loc[valid_mask]
            y_test_log_valid = y_test_log.loc[valid_mask]

            preds_log = model.predict(x_test_valid, validate_features=True)
            preds_var = from_log_var(preds_log)
            true_var = from_log_var(y_test_log_valid.values)

            # WR-02: drop floored near-zero realized variance.  to_log_var
            # floors zeros at LOG_VAR_EPS, so from_log_var round-trips them to
            # ~LOG_VAR_EPS > 0 — a plain `<= 0` check never fires, and each
            # floored zero injects a massive QLIKE outlier (pred/eps ratio).
            # The (1 + 1e-9) tolerance absorbs exp(log(eps)) float rounding.
            keep = np.isfinite(true_var) & (true_var > LOG_VAR_EPS * (1 + 1e-9))
            if not keep.any():
                continue

            all_pred_var.append(preds_var[keep])
            all_true_var.append(true_var[keep])

        if not all_pred_var:
            log.warning("No test folds for asset %s — skipping evaluation.", symbol)
            results[symbol] = {
                "rmse": float("nan"),
                "mae": float("nan"),
                "qlike": float("nan"),
                "n_forecasts": 0,
            }
            continue

        pred_var_all = np.concatenate(all_pred_var)
        true_var_all = np.concatenate(all_true_var)

        results[symbol] = {
            "rmse": rmse(true_var_all, pred_var_all),
            "mae": mae(true_var_all, pred_var_all),
            "qlike": qlike(true_var_all, pred_var_all),
            "n_forecasts": int(len(pred_var_all)),
        }
        log.info(
            "  %s: n=%d rmse=%.4e mae=%.4e qlike=%.4f",
            symbol,
            results[symbol]["n_forecasts"],
            results[symbol]["rmse"],
            results[symbol]["mae"],
            results[symbol]["qlike"],
        )

    return results


# ---------------------------------------------------------------------------
# Data lineage helper
# ---------------------------------------------------------------------------


def compute_data_hash(asset_feature_dfs: dict[str, pd.DataFrame]) -> str:
    """Compute an MD5 hash over all 5 feature DataFrames for lineage tracking.

    The hash is deterministic for identical data: DataFrames are serialised
    to bytes via ``pd.util.hash_pandas_object`` (row-order stable, dtypes
    included) and fed into a single MD5 digest.

    Args:
        asset_feature_dfs: Mapping symbol → feature DataFrame.

    Returns:
        8-character hex string (first 8 digits of MD5 digest).
    """
    import hashlib

    md5 = hashlib.md5()
    for symbol in sorted(asset_feature_dfs.keys()):
        row_hashes = pd.util.hash_pandas_object(
            asset_feature_dfs[symbol], index=True
        ).values.tobytes()
        md5.update(symbol.encode())
        md5.update(row_hashes)
    return md5.hexdigest()[:8]


# ---------------------------------------------------------------------------
# SHAP explainability artifacts
# ---------------------------------------------------------------------------


def compute_shap_artifacts(
    model: LGBMRegressor,
    x_reference: pd.DataFrame,
    feature_names: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compute SHAP values and save global summary plots as PNGs.

    Uses ``shap.TreeExplainer`` on the NATIVE ``LGBMRegressor`` object
    (Pitfall 6 — never an mlflow pyfunc wrapper).  Saves:
    - ``shap_global_bar.png``: mean |SHAP| bar chart (global feature importance)
    - ``shap_beeswarm.png``: beeswarm plot (direction + magnitude of SHAP values)

    Per-asset top-10: filter ``x_reference`` to one asset's rows, recompute
    mean |SHAP| ranking, return top-10 feature names.

    Args:
        model: Fitted LGBMRegressor (native model, NOT pyfunc wrapper).
        x_reference: Reference sample DataFrame used to compute SHAP values.
            Typically the concatenated test-fold rows.  Must include the
            ``"asset"`` column cast to ASSET_DTYPE.
        feature_names: List of feature column names (excluding ``"asset"``).
        output_dir: Directory path for PNG output files.

    Returns:
        Dict with keys:
        - ``"bar_path"``: absolute path to the bar PNG
        - ``"beeswarm_path"``: absolute path to the beeswarm PNG
        - ``"per_asset_top10"``: ``{symbol: [feat_name, ...]}`` for each asset
          present in ``x_reference``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SHAP TreeExplainer on the native LGBMRegressor (Pitfall 6 mitigation)
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values over the full reference set
    shap_values = explainer.shap_values(x_reference)
    # shap_values shape: (n_samples, n_features)

    # --- Global bar chart (mean |SHAP|) ---
    bar_path = output_dir / "shap_global_bar.png"
    plt.figure()
    shap.summary_plot(shap_values, x_reference, plot_type="bar", show=False)
    plt.savefig(bar_path, bbox_inches="tight", dpi=100)
    plt.close()

    # --- Beeswarm plot ---
    beeswarm_path = output_dir / "shap_beeswarm.png"
    plt.figure()
    shap.summary_plot(shap_values, x_reference, show=False)
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=100)
    plt.close()

    # --- Per-asset top-10 ---
    per_asset_top10: dict[str, list[str]] = {}
    if "asset" in x_reference.columns:
        unique_assets = x_reference["asset"].dropna().unique()
        for asset_val in unique_assets:
            asset_mask = x_reference["asset"] == asset_val
            if asset_mask.sum() == 0:
                continue
            asset_shap = shap_values[asset_mask]
            mean_abs_shap = np.abs(asset_shap).mean(axis=0)
            # feature_names maps to columns of the input; shap_values aligns with columns
            all_cols = list(x_reference.columns)
            sorted_idx = np.argsort(mean_abs_shap)[::-1]
            top10 = [all_cols[j] for j in sorted_idx[:10] if j < len(all_cols)]
            per_asset_top10[str(asset_val)] = top10

    return {
        "bar_path": bar_path,
        "beeswarm_path": beeswarm_path,
        "per_asset_top10": per_asset_top10,
    }
