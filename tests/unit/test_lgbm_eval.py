"""Regression tests for the LightGBM walk-forward evaluation path.

CR-02 regression: eval feature frames must be reindexed to the model's exact
training column order (``model.feature_name_``) before predict.  LightGBM's
default ``predict(validate_features=False)`` checks only the column COUNT, so
a transposed eval frame silently scores a corrupted matrix.  The test below
trains a real (tiny) pooled model, then evaluates twice — once with the
original per-asset column order and once with every frame's columns reversed —
and asserts identical metrics.  Pre-fix, the reversed order changed (or
crashed) every metric.

CR-01 regression: each test fold must be scored ONLY by the model trained at
that fold's own cutoff.  Recording stub models capture exactly which positions
each fold's model is asked to predict, and the test asserts that no predicted
position is at or before that model's per-asset training cutoff (the
no-train-on-test invariant), including for the documented final-fold fallback
on longer assets.

All tests are offline with small synthetic per-asset DataFrames.
No parquet files, no MLflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.eval.harness import walk_forward_splits
from volforecast.models.lgbm import (
    evaluate_per_asset,
    resolve_fold_model,
    train_pooled_model,
)

# min_train=252, step=21 → first valid fold needs n >= 273 + a margin
_N_ROWS = 300

_PARAMS = {
    "num_leaves": 7,
    "min_child_samples": 5,
    "learning_rate": 0.1,
    "reg_lambda": 0.0,
}


def _make_asset(n: int, extra_col: str, rng: np.random.Generator):
    """Return (feature_df, log_var_target) where the target tracks ``f1``.

    ``f1`` is drawn in the plausible log-variance range so the model learns a
    real dependence on a SPECIFIC column — making any silent column transpose
    at eval time change the predictions measurably.
    """
    f1 = rng.uniform(-9.0, -7.0, size=n)
    df = pd.DataFrame(
        {
            "f1": f1,
            "f2": rng.standard_normal(n),
            extra_col: rng.standard_normal(n),
        },
        index=pd.RangeIndex(n),
    )
    target = pd.Series(f1 + rng.normal(0.0, 0.1, size=n), name="target")
    target.iloc[-1] = np.nan  # mimic compute_target's NaN horizon tail
    return df, target


class TestEvalColumnOrder:
    """CR-02: eval metrics must be invariant to input column order."""

    def test_scrambled_eval_columns_give_identical_metrics(self):
        rng = np.random.default_rng(42)
        # Two assets with DIVERGENT cross-asset columns (x_btc vs x_eth) so
        # the pooled training union and per-asset eval frames differ in shape.
        feat_btc, tgt_btc = _make_asset(_N_ROWS, "x_btc", rng)
        feat_eth, tgt_eth = _make_asset(_N_ROWS, "x_eth", rng)

        asset_feature_dfs = {"BTC-USD": feat_btc, "ETH-USD": feat_eth}
        asset_target_series = {"BTC-USD": tgt_btc, "ETH-USD": tgt_eth}

        model, _best_iter = train_pooled_model(
            asset_feature_dfs,
            asset_target_series,
            params=_PARAMS,
            fold_i=0,
        )
        fold_models = {0: model}  # n=300 → exactly one walk-forward fold

        results_original = evaluate_per_asset(fold_models, asset_feature_dfs, asset_target_series)

        # Reverse every asset frame's column order — pre-fix this silently
        # fed category codes / wrong floats into transposed columns.
        scrambled_dfs = {
            sym: df[list(reversed(df.columns))].copy() for sym, df in asset_feature_dfs.items()
        }
        results_scrambled = evaluate_per_asset(fold_models, scrambled_dfs, asset_target_series)

        assert set(results_original) == {"BTC-USD", "ETH-USD"}
        for sym in results_original:
            orig = results_original[sym]
            scram = results_scrambled[sym]
            assert orig["n_forecasts"] > 0, f"{sym}: no forecasts evaluated"
            assert orig["n_forecasts"] == scram["n_forecasts"]
            for metric in ("rmse", "mae", "qlike"):
                assert orig[metric] == scram[metric], (
                    f"{sym} {metric}: column order changed the metric "
                    f"({orig[metric]} != {scram[metric]}) — eval frame was "
                    "not reindexed to model.feature_name_"
                )


# ---------------------------------------------------------------------------
# CR-01: per-fold leak-freedom
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Duck-typed stand-in for LGBMRegressor that records what it scores."""

    def __init__(self, feature_order: list[str]):
        self.feature_name_ = list(feature_order)
        self.calls: list[tuple[str, np.ndarray]] = []

    def predict(self, x: pd.DataFrame, validate_features: bool = False) -> np.ndarray:
        assert list(x.columns) == self.feature_name_, (
            "eval frame columns do not match the model's training order"
        )
        self.calls.append((str(x["asset"].iloc[0]), x["orig_pos"].to_numpy()))
        return np.zeros(len(x), dtype=float)


class TestPerFoldLeakFreedom:
    """CR-01: no test position may precede its scoring model's train cutoff."""

    def _make_panel(self):
        rng = np.random.default_rng(7)
        # Fold counts (min_train=252, step=21): 300→1, 340→3, 380→5.
        lengths = {"BTC-USD": 380, "ETH-USD": 340, "SPY": 300}
        asset_feature_dfs: dict[str, pd.DataFrame] = {}
        asset_target_series: dict[str, pd.Series] = {}
        for sym, n in lengths.items():
            asset_feature_dfs[sym] = pd.DataFrame(
                {
                    "f1": rng.standard_normal(n),
                    "orig_pos": np.arange(n, dtype=float),
                },
                index=pd.RangeIndex(n),
            )
            tgt = pd.Series(rng.uniform(-9.0, -7.0, size=n))
            tgt.iloc[-1] = np.nan
            asset_target_series[sym] = tgt
        return asset_feature_dfs, asset_target_series

    def test_no_test_position_precedes_its_models_train_cutoff(self):
        asset_feature_dfs, asset_target_series = self._make_panel()
        feature_order = ["f1", "orig_pos", "asset"]

        # 3-model mapping: SPY uses fold 0 only; ETH uses folds 0-2; BTC's
        # folds 3-4 exercise the documented final-fold fallback (leak-free
        # under an expanding window).
        fold_models = {k: _RecordingModel(feature_order) for k in range(3)}

        evaluate_per_asset(fold_models, asset_feature_dfs, asset_target_series)

        for k, stub in fold_models.items():
            assert stub.calls, f"fold model {k} was never used"
            for sym, positions in stub.calls:
                splits = list(walk_forward_splits(len(asset_feature_dfs[sym])))
                train_cutoff = int(splits[k].train_idx.max())
                assert positions.min() > train_cutoff, (
                    f"fold model {k} scored {sym} position {int(positions.min())} "
                    f"at-or-before its own train cutoff {train_cutoff} — "
                    "train-on-test leakage"
                )

    def test_each_fold_scored_by_its_own_model(self):
        """Within the mapping range, fold k's test window goes to model k only."""
        asset_feature_dfs, asset_target_series = self._make_panel()
        feature_order = ["f1", "orig_pos", "asset"]
        fold_models = {k: _RecordingModel(feature_order) for k in range(3)}

        evaluate_per_asset(fold_models, asset_feature_dfs, asset_target_series)

        for k, stub in fold_models.items():
            for sym, positions in stub.calls:
                splits = list(walk_forward_splits(len(asset_feature_dfs[sym])))
                allowed: set[int] = set()
                for fold_i, split in enumerate(splits):
                    # Model k legally scores its own fold, plus later folds
                    # ONLY when it is the final model in the mapping.
                    if fold_i == k or (k == max(fold_models) and fold_i > k):
                        allowed.update(split.test_idx.tolist())
                got = {int(p) for p in positions}
                assert got <= allowed, (
                    f"fold model {k} scored {sym} positions outside its "
                    f"legal test windows: {sorted(got - allowed)[:5]}..."
                )

    def test_floored_zero_rv_rows_excluded(self):
        """WR-02: floored-zero realized variance must not enter the metrics.

        to_log_var floors zeros at LOG_VAR_EPS, so the round-trip value is
        ~LOG_VAR_EPS > 0 and a plain `<= 0` check never fires.  Rows whose
        target round-trips to the floor must be dropped from n_forecasts.
        """
        from volforecast.models.lgbm import to_log_var

        rng = np.random.default_rng(11)
        n = 300  # exactly one fold: test window = positions 273..293 (21 rows)
        feat = pd.DataFrame(
            {"f1": rng.standard_normal(n), "orig_pos": np.arange(n, dtype=float)},
            index=pd.RangeIndex(n),
        )
        tgt = pd.Series(to_log_var(rng.uniform(1e-5, 1e-3, size=n)))
        # Five floored zeros inside the test window
        tgt.iloc[273:278] = float(to_log_var(0.0))
        tgt.iloc[-1] = np.nan

        fold_models = {0: _RecordingModel(["f1", "orig_pos", "asset"])}
        results = evaluate_per_asset(fold_models, {"SPY": feat}, {"SPY": tgt})

        assert results["SPY"]["n_forecasts"] == 21 - 5, (
            "floored-zero rows were not excluded from evaluation"
        )

    def test_resolve_fold_model_contract(self):
        """Fallback beyond the range; KeyError on holes inside the range."""
        m0, m2 = object(), object()
        fold_models = {0: m0, 2: m2}
        assert resolve_fold_model(fold_models, 0) is m0
        assert resolve_fold_model(fold_models, 2) is m2
        assert resolve_fold_model(fold_models, 7) is m2  # beyond range → final
        with pytest.raises(KeyError):
            resolve_fold_model(fold_models, 1)  # hole inside range → loud
