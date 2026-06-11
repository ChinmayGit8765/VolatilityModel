"""Unit tests for the leak-free pooled fold assembly (assemble_pooled_train).

Tests the <behavior> contract from Plan 03-01 Task 3:

1. Row count equals sum of train_idx lengths minus NaN-target rows.
2. No row from any asset's test_idx appears in the pooled training set
   (the no-leakage invariant — explicit assertion).
3. The pooled X carries an "asset" column equal to the source symbol; after
   astype(ASSET_DTYPE) there are no NaN categories.
4. When fold_i exceeds an asset's number of splits, that asset contributes
   zero rows (skipped, no error).
5. NaN-target rows (final horizon rows from compute_target) are excluded.

All tests are offline with small synthetic per-asset DataFrames.
No parquet files, no MLflow, no fixtures beyond inline arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.eval.harness import walk_forward_splits
from volforecast.models.lgbm import (
    ASSET_DTYPE,
    assemble_pooled_train,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

# Use small DataFrames that still produce at least 1 split:
# min_train=252, step=21 requires n >= 252 + 21 = 273
_N_ROWS = 300  # just enough for a couple of folds
_SMALL_N = 200  # NOT enough for any folds (fold_i=0 skips this asset)

_RNG = np.random.default_rng(42)


def _make_feature_df(n: int, extra_col: str = "f1") -> pd.DataFrame:
    """Return a minimal feature DataFrame with integer RangeIndex."""
    return pd.DataFrame(
        {extra_col: _RNG.standard_normal(n)},
        index=pd.RangeIndex(n),
    )


def _make_target_series(n: int, nan_tail: int = 1) -> pd.Series:
    """Return a target Series with the last `nan_tail` rows as NaN."""
    vals = _RNG.uniform(1e-5, 1e-2, size=n).astype(float)
    vals[-nan_tail:] = np.nan
    return pd.Series(vals, name="target")


# ---------------------------------------------------------------------------
# Test: basic row count
# ---------------------------------------------------------------------------


class TestRowCount:
    def test_row_count_equals_train_idx_minus_nan(self):
        """Pooled row count = sum of (len(train_idx) - nan_target_rows_in_train)."""
        n_a, n_b = _N_ROWS, _N_ROWS
        nan_tail = 1  # 1 NaN row per asset per fold
        feat_a = _make_feature_df(n_a, "f_a")
        feat_b = _make_feature_df(n_b, "f_b")
        tgt_a = _make_target_series(n_a, nan_tail=nan_tail)
        tgt_b = _make_target_series(n_b, nan_tail=nan_tail)

        fold_i = 0
        min_train, step, horizon = 252, 21, 1

        # Calculate expected count manually
        splits_a = list(walk_forward_splits(n_a, min_train, step, horizon))
        splits_b = list(walk_forward_splits(n_b, min_train, step, horizon))

        assert fold_i < len(splits_a), "fold_i must be valid for asset A"
        assert fold_i < len(splits_b), "fold_i must be valid for asset B"

        train_idx_a = splits_a[fold_i].train_idx
        train_idx_b = splits_b[fold_i].train_idx

        # Count non-NaN targets in each train set
        non_nan_a = int(tgt_a.iloc[train_idx_a].notna().sum())
        non_nan_b = int(tgt_b.iloc[train_idx_b].notna().sum())
        expected_rows = non_nan_a + non_nan_b

        X, y = assemble_pooled_train(
            {"A": feat_a, "B": feat_b},
            {"A": tgt_a, "B": tgt_b},
            fold_i=fold_i,
            min_train=min_train,
            step=step,
            horizon=horizon,
        )

        assert len(X) == expected_rows
        assert len(y) == expected_rows

    def test_x_y_lengths_match(self):
        """X and y must always have the same length."""
        feat = {"A": _make_feature_df(_N_ROWS)}
        tgt = {"A": _make_target_series(_N_ROWS)}
        X, y = assemble_pooled_train(feat, tgt, fold_i=0)
        assert len(X) == len(y)


# ---------------------------------------------------------------------------
# Test: no-leakage invariant (the critical test)
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_no_test_fold_row_in_pooled_train(self):
        """EXPLICIT ASSERTION: no test-fold position from any asset appears in the pool.

        This is the core methodological invariant.  We track original integer
        positions by injecting a 'orig_pos' column into each asset's feature
        DataFrame, then verify that none of the positions in any asset's
        test_idx for fold_i appear in the corresponding rows of the pooled X.
        """
        n_a, n_b = _N_ROWS, _N_ROWS
        min_train, step, horizon = 252, 21, 1
        fold_i = 0

        # Add original-position tracking columns
        feat_a = _make_feature_df(n_a, "f_a")
        feat_a["orig_pos"] = np.arange(n_a)
        feat_b = _make_feature_df(n_b, "f_b")
        feat_b["orig_pos"] = np.arange(n_b)

        tgt_a = _make_target_series(n_a)
        tgt_b = _make_target_series(n_b)

        splits_a = list(walk_forward_splits(n_a, min_train, step, horizon))
        splits_b = list(walk_forward_splits(n_b, min_train, step, horizon))
        test_idx_a = set(splits_a[fold_i].test_idx.tolist())
        test_idx_b = set(splits_b[fold_i].test_idx.tolist())

        X, _y = assemble_pooled_train(
            {"A": feat_a, "B": feat_b},
            {"A": tgt_a, "B": tgt_b},
            fold_i=fold_i,
            min_train=min_train,
            step=step,
            horizon=horizon,
        )

        # Rows for asset A in the pooled X
        rows_a = X.loc[X["asset"] == "A", "orig_pos"].tolist()
        rows_b = X.loc[X["asset"] == "B", "orig_pos"].tolist()

        # THE KEY INVARIANT: no test-fold position from A in A's contribution
        leaked_a = set(rows_a) & test_idx_a
        assert not leaked_a, f"Asset A test-fold positions leaked into pooled train: {leaked_a}"

        # No test-fold position from B in B's contribution
        leaked_b = set(rows_b) & test_idx_b
        assert not leaked_b, f"Asset B test-fold positions leaked into pooled train: {leaked_b}"

    def test_no_future_train_positions(self):
        """All train positions are strictly less than test start position."""
        n = _N_ROWS
        min_train, step, horizon = 252, 21, 1
        fold_i = 0

        feat = {"A": _make_feature_df(n, "f")}
        feat["A"]["orig_pos"] = np.arange(n)
        tgt = {"A": _make_target_series(n)}

        splits = list(walk_forward_splits(n, min_train, step, horizon))
        test_start = splits[fold_i].test_idx.min()

        X, _ = assemble_pooled_train(
            feat, tgt, fold_i=fold_i, min_train=min_train, step=step, horizon=horizon
        )

        max_train_pos = X["orig_pos"].max()
        assert max_train_pos < test_start, (
            f"Training includes positions at or after test start "
            f"(max_train_pos={max_train_pos}, test_start={test_start})"
        )

    def test_cross_asset_independence(self):
        """Asset A's test rows must not appear in asset B's train contribution.

        This tests the cross-asset variant: if A and B have overlapping
        calendar dates in their test windows, using date-range slicing (wrong)
        could mix A's test rows into B's train rows. Position-based selection
        (correct) guarantees independence because positions are per-asset.
        """
        # Two assets with DIFFERENT lengths so their fold structures diverge
        n_a, n_b = _N_ROWS, _N_ROWS + 50
        min_train, step, horizon = 252, 21, 1
        fold_i = 0

        feat_a = _make_feature_df(n_a, "f_a")
        feat_a["orig_pos"] = np.arange(n_a)
        feat_b = _make_feature_df(n_b, "f_b")
        feat_b["orig_pos"] = np.arange(n_b)

        tgt_a = _make_target_series(n_a)
        tgt_b = _make_target_series(n_b)

        splits_a = list(walk_forward_splits(n_a, min_train, step, horizon))
        splits_b = list(walk_forward_splits(n_b, min_train, step, horizon))

        test_idx_a = set(splits_a[fold_i].test_idx.tolist())
        test_idx_b = set(splits_b[fold_i].test_idx.tolist())

        X, _ = assemble_pooled_train(
            {"A": feat_a, "B": feat_b},
            {"A": tgt_a, "B": tgt_b},
            fold_i=fold_i,
            min_train=min_train,
            step=step,
            horizon=horizon,
        )

        rows_a = set(X.loc[X["asset"] == "A", "orig_pos"].tolist())
        rows_b = set(X.loc[X["asset"] == "B", "orig_pos"].tolist())

        assert not (rows_a & test_idx_a), "A's test positions in A's train rows"
        assert not (rows_b & test_idx_b), "B's test positions in B's train rows"


# ---------------------------------------------------------------------------
# Test: asset column and dtype
# ---------------------------------------------------------------------------


class TestAssetColumn:
    def test_asset_column_exists(self):
        """Pooled X must have an 'asset' column."""
        feat = {"BTC-USD": _make_feature_df(_N_ROWS)}
        tgt = {"BTC-USD": _make_target_series(_N_ROWS)}
        X, _ = assemble_pooled_train(feat, tgt, fold_i=0)
        assert "asset" in X.columns

    def test_asset_column_values(self):
        """'asset' column values match the source symbol names."""
        feat = {"BTC-USD": _make_feature_df(_N_ROWS), "SPY": _make_feature_df(_N_ROWS)}
        tgt = {
            "BTC-USD": _make_target_series(_N_ROWS),
            "SPY": _make_target_series(_N_ROWS),
        }
        X, _ = assemble_pooled_train(feat, tgt, fold_i=0)
        assert set(X["asset"].unique()) == {"BTC-USD", "SPY"}

    def test_asset_dtype_is_categorical(self):
        """'asset' column in pooled X must be cast to ASSET_DTYPE."""
        feat = {"BTC-USD": _make_feature_df(_N_ROWS)}
        tgt = {"BTC-USD": _make_target_series(_N_ROWS)}
        X, _ = assemble_pooled_train(feat, tgt, fold_i=0)
        assert X["asset"].dtype == ASSET_DTYPE

    def test_known_asset_no_nan_category(self):
        """Known assets produce no NaN categories in the asset column."""
        feat = {sym: _make_feature_df(_N_ROWS) for sym in ["BTC-USD", "SPY", "AAPL"]}
        tgt = {sym: _make_target_series(_N_ROWS) for sym in ["BTC-USD", "SPY", "AAPL"]}
        X, _ = assemble_pooled_train(feat, tgt, fold_i=0)
        assert X["asset"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Test: fold_i exceeds splits — asset skipped
# ---------------------------------------------------------------------------


class TestFoldSkipping:
    def test_short_asset_skipped_no_error(self):
        """An asset with n < min_train + step contributes zero rows without error."""
        # _SMALL_N=200 < 252+21=273, so no splits exist for this asset
        # Use valid KNOWN_ASSETS symbols to avoid NaN categories in ASSET_DTYPE
        feat = {
            "BTC-USD": _make_feature_df(_SMALL_N),  # too short for any fold
            "SPY": _make_feature_df(_N_ROWS),  # long enough for fold_i=0
        }
        tgt = {
            "BTC-USD": _make_target_series(_SMALL_N),
            "SPY": _make_target_series(_N_ROWS),
        }
        X, y = assemble_pooled_train(feat, tgt, fold_i=0)

        # Only "SPY" asset should contribute rows; "BTC-USD" skipped (no folds)
        assert "BTC-USD" not in set(X["asset"].tolist()), (
            "'BTC-USD' (too short) should not appear in pooled train"
        )
        assert "SPY" in set(X["asset"].tolist())
        assert len(X) > 0, "Should still have rows from 'SPY' asset"

    def test_fold_i_beyond_all_assets_raises(self):
        """fold_i larger than all assets' split counts raises ValueError."""
        # fold_i=9999 will exceed any reasonable number of splits for n=300
        feat = {"A": _make_feature_df(_N_ROWS)}
        tgt = {"A": _make_target_series(_N_ROWS)}
        with pytest.raises(ValueError, match="empty pooled training set"):
            assemble_pooled_train(feat, tgt, fold_i=9999)

    def test_mixed_fold_depths(self):
        """One asset has enough folds, another doesn't — only the long one contributes."""
        n_short, n_long = 280, _N_ROWS  # 280 gives exactly 1 fold; _N_ROWS gives more
        # Use valid KNOWN_ASSETS symbols to avoid NaN categories in ASSET_DTYPE
        sym_short, sym_long = "ETH-USD", "AAPL"
        feat = {
            sym_short: _make_feature_df(n_short),
            sym_long: _make_feature_df(n_long),
        }
        tgt = {
            sym_short: _make_target_series(n_short),
            sym_long: _make_target_series(n_long),
        }

        # fold_i=1 may exceed sym_short's splits (n_short=280, min_train=252, step=21)
        splits_short = list(walk_forward_splits(n_short))
        # Find a fold_i that the short asset doesn't have but the long one does
        fold_i = len(splits_short)  # exactly one beyond short asset's last fold

        splits_long = list(walk_forward_splits(n_long))
        if fold_i >= len(splits_long):
            pytest.skip("Long asset also lacks fold — increase _N_ROWS")

        X, _ = assemble_pooled_train(feat, tgt, fold_i=fold_i)
        assert sym_long in set(X["asset"].tolist())
        # sym_short should NOT appear (fold_i >= len(splits_short))
        assert sym_short not in set(X["asset"].tolist())


# ---------------------------------------------------------------------------
# Test: NaN-target exclusion
# ---------------------------------------------------------------------------


class TestNanTargetExclusion:
    def test_nan_targets_excluded(self):
        """NaN-target rows (final horizon rows) are excluded from the pool."""
        n = _N_ROWS
        nan_tail = 5  # more than 1 to make the exclusion clearly measurable

        feat = {"A": _make_feature_df(n)}
        tgt = {"A": _make_target_series(n, nan_tail=nan_tail)}

        min_train, step, horizon = 252, 21, 1
        splits = list(walk_forward_splits(n, min_train, step, horizon))
        train_idx = splits[0].train_idx

        # Count how many NaN targets are in the train window
        nan_in_train = int(tgt["A"].iloc[train_idx].isna().sum())

        X, y = assemble_pooled_train(
            feat, tgt, fold_i=0, min_train=min_train, step=step, horizon=horizon
        )

        expected_rows = len(train_idx) - nan_in_train
        assert len(X) == expected_rows, (
            f"Expected {expected_rows} rows (excl. {nan_in_train} NaN targets), got {len(X)}"
        )
        assert y.isna().sum() == 0, "y must not contain NaN values"

    def test_y_contains_no_nan(self):
        """y returned by assemble_pooled_train must never contain NaN."""
        feat = {"A": _make_feature_df(_N_ROWS), "B": _make_feature_df(_N_ROWS)}
        tgt = {
            "A": _make_target_series(_N_ROWS, nan_tail=1),
            "B": _make_target_series(_N_ROWS, nan_tail=3),
        }
        _, y = assemble_pooled_train(feat, tgt, fold_i=0)
        assert y.isna().sum() == 0
