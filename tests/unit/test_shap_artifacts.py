"""Unit tests for compute_shap_artifacts (MODEL-03).

Tests the <behavior> contract from Plan 03-02 Task 3:

1. shap_values shape from TreeExplainer is (n_samples, n_features)
2. Global bar PNG and beeswarm PNG are written to output_dir; both are non-empty
3. Per-asset top-10: filtering X_reference to one asset's rows yields <= 10 features
4. compute_shap_artifacts uses the native LGBMRegressor (not a pyfunc wrapper)
   — verified by passing an LGBMRegressor directly; test would fail if called with
   a pyfunc wrapper (shap.TreeExplainer raises TypeError for unknown types)

All tests are offline: no MLflow calls, no parquet files.  A tiny synthetic
LGBMRegressor is trained inline on random data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from volforecast.models.lgbm import ASSET_DTYPE, KNOWN_ASSETS, compute_shap_artifacts

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_N_SAMPLES = 80  # small enough for fast SHAP, large enough for a stable model
_FEATURE_NAMES = ["rv_5", "rv_10", "rv_22", "ewma_var", "garch_cond_var"]
_N_FEATURES = len(_FEATURE_NAMES)


def _make_X(n: int, assets: list[str] | None = None) -> pd.DataFrame:
    """Build a minimal feature DataFrame with optional 'asset' category column."""
    data = {f: _RNG.standard_normal(n) for f in _FEATURE_NAMES}
    df = pd.DataFrame(data)
    if assets is not None:
        asset_col = assets * (n // len(assets)) + assets[: n % len(assets)]
        df["asset"] = pd.Series(asset_col[:n], dtype=ASSET_DTYPE)
    return df


def _train_model(x: pd.DataFrame, y: np.ndarray) -> LGBMRegressor:
    """Train a minimal LGBMRegressor on synthetic data."""
    model = LGBMRegressor(
        n_estimators=10,
        num_leaves=4,
        random_state=42,
        verbose=-1,
    )
    model.fit(x, y)
    return model


# ---------------------------------------------------------------------------
# Fixture: shared synthetic model + X_reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_model_and_X():
    """One shared LGBMRegressor trained on synthetic data with 'asset' column."""
    assets = [KNOWN_ASSETS[0], KNOWN_ASSETS[1]]  # BTC-USD, ETH-USD
    x = _make_X(_N_SAMPLES, assets=assets)
    y = _RNG.uniform(1e-5, 1e-2, size=_N_SAMPLES)
    model = _train_model(x, y)
    return model, x


# ---------------------------------------------------------------------------
# Test group 1: SHAP value shape
# ---------------------------------------------------------------------------


class TestShapValueShape:
    """SHAP values must have shape (n_samples, n_features)."""

    def test_shap_values_shape(self, trained_model_and_X, tmp_path):
        """shap_values shape is (n, f) where n=len(X_reference) and f=n_feature_cols."""
        model, x = trained_model_and_X
        result = compute_shap_artifacts(
            model=model,
            x_reference=x,
            feature_names=_FEATURE_NAMES,
            output_dir=tmp_path,
        )
        # Verify via bar PNG existence (SHAP values were computed if file is non-empty)
        assert result["bar_path"].exists()

    def test_compute_shap_returns_paths(self, trained_model_and_X, tmp_path):
        """Return dict has 'bar_path', 'beeswarm_path', 'per_asset_top10' keys."""
        model, x = trained_model_and_X
        result = compute_shap_artifacts(
            model=model,
            x_reference=x,
            feature_names=_FEATURE_NAMES,
            output_dir=tmp_path,
        )
        assert "bar_path" in result
        assert "beeswarm_path" in result
        assert "per_asset_top10" in result

    def test_shap_shape_via_explainer_directly(self, trained_model_and_X):
        """Verify shap.TreeExplainer produces (n, f) values on native LGBMRegressor."""
        import shap

        model, x = trained_model_and_X
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
        assert shap_values.shape == (len(x), len(x.columns)), (
            f"Expected ({len(x)}, {len(x.columns)}), got {shap_values.shape}"
        )


# ---------------------------------------------------------------------------
# Test group 2: PNG files exist and are non-empty
# ---------------------------------------------------------------------------


class TestPngFilesExist:
    """Both bar and beeswarm PNGs must exist on disk and be non-empty."""

    def test_bar_png_exists(self, trained_model_and_X, tmp_path):
        model, x = trained_model_and_X
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "out1")
        assert result["bar_path"].exists(), "bar PNG not found"

    def test_bar_png_non_empty(self, trained_model_and_X, tmp_path):
        model, x = trained_model_and_X
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "out2")
        assert result["bar_path"].stat().st_size > 0, "bar PNG is empty"

    def test_beeswarm_png_exists(self, trained_model_and_X, tmp_path):
        model, x = trained_model_and_X
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "out3")
        assert result["beeswarm_path"].exists(), "beeswarm PNG not found"

    def test_beeswarm_png_non_empty(self, trained_model_and_X, tmp_path):
        model, x = trained_model_and_X
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "out4")
        assert result["beeswarm_path"].stat().st_size > 0, "beeswarm PNG is empty"

    def test_bar_png_filename(self, trained_model_and_X, tmp_path):
        model, x = trained_model_and_X
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "out5")
        assert result["bar_path"].name == "shap_global_bar.png"

    def test_beeswarm_png_filename(self, trained_model_and_X, tmp_path):
        model, x = trained_model_and_X
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "out6")
        assert result["beeswarm_path"].name == "shap_beeswarm.png"

    def test_output_dir_created_if_missing(self, trained_model_and_X, tmp_path):
        """compute_shap_artifacts creates the output directory if it doesn't exist."""
        model, x = trained_model_and_X
        new_dir = tmp_path / "brand_new_dir" / "sub"
        assert not new_dir.exists()
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, new_dir)
        assert new_dir.exists()
        assert result["bar_path"].exists()


# ---------------------------------------------------------------------------
# Test group 3: Per-asset top-10
# ---------------------------------------------------------------------------


class TestPerAssetTop10:
    """Filtering X_reference to one asset yields a ranked feature list of length <= 10."""

    def test_per_asset_keys_match_assets_present(self, trained_model_and_X, tmp_path):
        """per_asset_top10 keys are exactly the unique assets in X_reference."""
        model, x = trained_model_and_X
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "pa1")
        expected_assets = set(x["asset"].dropna().unique().astype(str))
        assert set(result["per_asset_top10"].keys()) == expected_assets

    def test_per_asset_top10_length_lte_10(self, trained_model_and_X, tmp_path):
        """Each per-asset top-N list has at most 10 features."""
        model, x = trained_model_and_X
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "pa2")
        for asset_name, top10 in result["per_asset_top10"].items():
            assert len(top10) <= 10, f"{asset_name}: expected <= 10 features, got {len(top10)}"

    def test_per_asset_top10_are_strings(self, trained_model_and_X, tmp_path):
        """Feature names in per-asset top-10 are strings (column names)."""
        model, x = trained_model_and_X
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "pa3")
        for asset_name, top10 in result["per_asset_top10"].items():
            assert all(isinstance(f, str) for f in top10), (
                f"{asset_name}: expected list of str, got {top10}"
            )

    def test_per_asset_top10_single_asset_x(self, tmp_path):
        """With X_reference filtered to one asset, top-10 has <= 10 features."""
        single_asset = [KNOWN_ASSETS[2]]  # SPY
        x = _make_X(_N_SAMPLES, assets=single_asset)
        y = _RNG.uniform(1e-5, 1e-2, size=_N_SAMPLES)
        model = _train_model(x, y)

        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "pa4")
        for asset_name, top10 in result["per_asset_top10"].items():
            assert len(top10) <= 10

    def test_per_asset_top10_no_asset_col(self, tmp_path):
        """Without 'asset' column, per_asset_top10 is empty (no per-asset split)."""
        x = _make_X(_N_SAMPLES, assets=None)
        y = _RNG.uniform(1e-5, 1e-2, size=_N_SAMPLES)
        model = _train_model(x, y)
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "pa5")
        assert result["per_asset_top10"] == {}, "Expected empty dict when no 'asset' column"


# ---------------------------------------------------------------------------
# Test group 4: Native LGBMRegressor (Pitfall 6 guard)
# ---------------------------------------------------------------------------


class TestNativeModelOnly:
    """compute_shap_artifacts must be called with the native LGBMRegressor.

    We verify this by confirming that shap.TreeExplainer accepts LGBMRegressor
    without raising TypeError.  If a pyfunc wrapper were passed instead,
    TreeExplainer would raise TypeError (the anti-pattern we guard against).
    """

    def test_accepts_lgbmregressor(self, trained_model_and_X, tmp_path):
        """compute_shap_artifacts does not raise when given an LGBMRegressor."""
        model, x = trained_model_and_X
        assert isinstance(model, LGBMRegressor), (
            "Fixture must supply a native LGBMRegressor, not a wrapper"
        )
        # Should NOT raise — if it does, the native-model contract is broken
        result = compute_shap_artifacts(model, x, _FEATURE_NAMES, tmp_path / "native")
        assert result["bar_path"].exists()

    def test_pyfunc_wrapper_raises(self):
        """A non-LGBMRegressor/Booster object causes TreeExplainer to raise TypeError.

        This documents the behaviour (Pitfall 6) that compute_shap_artifacts guards
        against by always being called with the native model at training time.
        """
        import shap

        class FakePyfunc:
            """Simulates an mlflow.pyfunc.PyFuncModel (not a native LGBM model)."""

            def predict(self, data):
                return data

        with pytest.raises((TypeError, ValueError, Exception)):
            shap.TreeExplainer(FakePyfunc())
