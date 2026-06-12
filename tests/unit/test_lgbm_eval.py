"""Regression tests for the LightGBM walk-forward evaluation path.

CR-02 regression: eval feature frames must be reindexed to the model's exact
training column order (``model.feature_name_``) before predict.  LightGBM's
default ``predict(validate_features=False)`` checks only the column COUNT, so
a transposed eval frame silently scores a corrupted matrix.  The test below
trains a real (tiny) pooled model, then evaluates twice — once with the
original per-asset column order and once with every frame's columns reversed —
and asserts identical metrics.  Pre-fix, the reversed order changed (or
crashed) every metric.

All tests are offline with small synthetic per-asset DataFrames.
No parquet files, no MLflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast.models.lgbm import (
    evaluate_per_asset,
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

        results_original = evaluate_per_asset(model, asset_feature_dfs, asset_target_series)

        # Reverse every asset frame's column order — pre-fix this silently
        # fed category codes / wrong floats into transposed columns.
        scrambled_dfs = {
            sym: df[list(reversed(df.columns))].copy() for sym, df in asset_feature_dfs.items()
        }
        results_scrambled = evaluate_per_asset(model, scrambled_dfs, asset_target_series)

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
