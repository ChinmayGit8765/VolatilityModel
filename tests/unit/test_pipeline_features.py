"""Unit tests for src/volforecast/features/pipeline.py.

All tests are offline-only (no network, no live data).  Series are
constructed inline or loaded from committed fixture parquets.

Coverage:
- build_features() returns expected column set
- cross-asset columns present when cross_asset_dfs provided; NaN beyond 3-day
- garch_cond_var is VARIANCE (std dev squared), positive, from data <= t
- include_garch=False skips GARCH column (lazy import guard)
- Single codepath: exactly one 'def build_features' in pipeline.py; importable
- Units are decimal variance (not percent-squared, not annualized)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volforecast.features.pipeline import build_features

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Expected column set from build_features (without cross-asset)
EXPECTED_BASE_COLS = {
    "rv_5",
    "rv_10",
    "rv_22",
    "rv_66",
    "log_return",
    "squared_return",
    "lagged_vol",
    "ewma_var",
    "parkinson_var",
    "gk_var",
    "vol_of_vol",
    "rolling_skew",
    "rolling_kurt",
    "day_of_week",
    "month",
    "is_monday",
    "is_friday",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlc(n: int = 300, base: float = 100.0, drift: float = 1.001) -> pd.DataFrame:
    """Build a deterministic OHLC DataFrame with UTC DatetimeIndex named 'date'."""
    rng = np.random.default_rng(1234)
    dates = pd.date_range("2020-01-03", periods=n, freq="B", tz="UTC", name="date")
    noise = rng.normal(0, 0.005, n)
    close = base * (drift ** np.arange(n)) * np.exp(np.cumsum(noise))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1.0 + rng.normal(0, 0.003, n))
    # Ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------


class TestColumnPresence:
    def test_expected_base_columns_present(self) -> None:
        """build_features must produce all expected feature columns."""
        df = _ohlc(200)
        result = build_features(df, include_garch=False)
        missing = EXPECTED_BASE_COLS - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_garch_col_present_when_enabled(self) -> None:
        """garch_cond_var column is present when include_garch=True (default)."""
        df = _ohlc(300)
        result = build_features(df, include_garch=True)
        assert "garch_cond_var" in result.columns

    def test_garch_col_absent_when_disabled(self) -> None:
        """garch_cond_var column is absent when include_garch=False (lazy import)."""
        df = _ohlc(200)
        result = build_features(df, include_garch=False)
        assert "garch_cond_var" not in result.columns

    def test_output_is_dataframe(self) -> None:
        df = _ohlc(100)
        result = build_features(df, include_garch=False)
        assert isinstance(result, pd.DataFrame)

    def test_index_matches_input(self) -> None:
        """Output index must equal input DatetimeIndex."""
        df = _ohlc(100)
        result = build_features(df, include_garch=False)
        pd.testing.assert_index_equal(result.index, df.index)


# ---------------------------------------------------------------------------
# GARCH conditional variance
# ---------------------------------------------------------------------------


class TestGarchCondVar:
    def test_garch_cond_var_positive_where_not_nan(self) -> None:
        """Non-NaN garch_cond_var values must be positive (it is a variance)."""
        df = _ohlc(300)
        result = build_features(df, include_garch=True)
        col = result["garch_cond_var"].dropna()
        assert (col > 0).all(), "garch_cond_var must be positive"

    def test_garch_cond_var_in_decimal_variance_range(self) -> None:
        """Daily decimal variance for typical assets should be in [1e-7, 0.05]."""
        df = _ohlc(300)
        result = build_features(df, include_garch=True)
        col = result["garch_cond_var"].dropna()
        # 1e-7 to 0.05 is a wide but plausible range for decimal daily variance
        assert (col > 1e-7).all(), "garch_cond_var values too small"
        assert (col < 0.05).all(), "garch_cond_var values too large (annualized?)"

    def test_garch_cond_var_is_variance_not_vol(self) -> None:
        """garch_cond_var must be VARIANCE (squared), not std dev.

        Std dev values (~0.01) vs variance values (~0.0001) differ by ~100x.
        Decimal variance for typical assets is << 0.001 per day.
        """
        df = _ohlc(300)
        result = build_features(df, include_garch=True)
        col = result["garch_cond_var"].dropna()
        # If accidentally std dev (~0.01), the median would be >> 0.001
        # Variance for our test asset is in the 1e-5 range
        assert col.median() < 0.001, (
            f"garch_cond_var median {col.median():.6f} too high — likely std dev not variance"
        )


# ---------------------------------------------------------------------------
# Cross-asset integration
# ---------------------------------------------------------------------------


class TestCrossAsset:
    def test_cross_asset_columns_present(self) -> None:
        """When cross_asset_dfs is provided, per-source-suffixed columns appear."""
        target = _ohlc(100)
        source = _ohlc(100)
        # Build source with same dates; rv_22 is a known feature
        source_features = build_features(source, include_garch=False)
        result = build_features(
            target,
            cross_asset_dfs={"btc": source_features[["rv_22"]]},
            include_garch=False,
        )
        assert "rv_22_btc" in result.columns

    def test_cross_asset_nan_beyond_3_days(self) -> None:
        """Cross-asset values beyond 3-day staleness must be NaN."""
        # Target: Monday 2022-02-07
        # Source last row: Monday 2022-01-31 (7 days gap → NaN)
        target_dates = pd.date_range("2022-02-07", periods=5, freq="B", tz="UTC")
        source_dates = pd.date_range("2022-01-25", periods=5, freq="B", tz="UTC")
        n_t = len(target_dates)
        n_s = len(source_dates)

        rng = np.random.default_rng(99)

        def _mk(dates: pd.DatetimeIndex, n: int) -> pd.DataFrame:
            c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
            h = c * 1.01
            lw = c * 0.99
            o = c * 0.999
            return pd.DataFrame(
                {"open": o, "high": h, "low": lw, "close": c, "volume": np.ones(n) * 1e6},
                index=dates,
            )

        target_df = _mk(target_dates, n_t)
        source_feat = pd.DataFrame({"rv22": [0.001] * n_s}, index=source_dates)

        result = build_features(
            target_df,
            cross_asset_dfs={"btc": source_feat},
            include_garch=False,
        )
        # All target dates are > 3 days after last source date → all NaN
        # source last date: 2022-01-31 (Mon); target first date: 2022-02-07 (Mon) = 7 days
        assert "rv22_btc" in result.columns, f"joined column missing: {list(result.columns)}"
        assert result["rv22_btc"].isna().all()

    def test_cross_asset_absent_when_not_provided(self) -> None:
        """No cross-asset columns when cross_asset_dfs is None."""
        df = _ohlc(100)
        result = build_features(df, cross_asset_dfs=None, include_garch=False)
        extra = set(result.columns) - EXPECTED_BASE_COLS
        assert not extra, f"Unexpected cross-asset columns: {extra}"

    def test_two_sources_with_same_feature_name_do_not_collide(self) -> None:
        """WR-01 regression: two sources both providing rv_22 must yield
        distinct per-asset-suffixed columns — never pandas _x/_y collision
        names, and never dict-order-dependent names.

        Value-level: each joined column carries its OWN source's values.
        """
        target = _ohlc(60)
        dates = target.index

        btc_source = pd.DataFrame({"rv_22": np.full(len(dates), 1e-4)}, index=dates)
        eth_source = pd.DataFrame({"rv_22": np.full(len(dates), 9e-4)}, index=dates)

        result = build_features(
            target,
            cross_asset_dfs={"btc": btc_source, "eth": eth_source},
            include_garch=False,
        )

        # Asset-identifying, stable column names
        assert "rv_22_btc" in result.columns, f"columns: {list(result.columns)}"
        assert "rv_22_eth" in result.columns, f"columns: {list(result.columns)}"
        # No pandas collision suffixes or generic suffix leakage
        bad = [c for c in result.columns if c.endswith(("_x", "_y")) or "_xasset" in c]
        assert not bad, f"collision/dict-order-dependent columns produced: {bad}"

        # Value-level: each column carries its own source's values
        assert result["rv_22_btc"].dropna().eq(1e-4).all(), "rv_22_btc lost BTC identity"
        assert result["rv_22_eth"].dropna().eq(9e-4).all(), "rv_22_eth lost ETH identity"

        # Dict-order invariance: reversed insertion order yields identical columns
        result_rev = build_features(
            target,
            cross_asset_dfs={"eth": eth_source, "btc": btc_source},
            include_garch=False,
        )
        assert set(result.columns) == set(result_rev.columns), (
            "cross-asset column names depend on dict insertion order"
        )
        assert result_rev["rv_22_btc"].dropna().eq(1e-4).all()
        assert result_rev["rv_22_eth"].dropna().eq(9e-4).all()


# ---------------------------------------------------------------------------
# Importability / single codepath (FEAT-07)
# ---------------------------------------------------------------------------


class TestSingleCodepath:
    def test_importable_from_pipeline_module(self) -> None:
        """build_features is importable from volforecast.features.pipeline (FEAT-07)."""
        from volforecast.features import pipeline  # noqa: F401

        assert hasattr(pipeline, "build_features")
        assert callable(pipeline.build_features)

    def test_exactly_one_definition(self) -> None:
        """There is exactly one 'def build_features' in pipeline.py (FEAT-07)."""
        import importlib
        import inspect

        pipeline = importlib.import_module("volforecast.features.pipeline")
        source = inspect.getsource(pipeline)
        count = source.count("def build_features")
        assert count == 1, f"Expected exactly 1 'def build_features', found {count}"

    def test_build_features_is_callable(self) -> None:
        assert callable(build_features)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


class TestUnits:
    def test_rv_cols_in_decimal_variance_range(self) -> None:
        """rv_5, rv_22 etc. must be in decimal variance units (not percent-squared)."""
        # Use synthetic data with enough rows for rv_22 (needs >= 22 observations)
        df = _ohlc(200)
        result = build_features(df, include_garch=False)
        # Typical daily equity log return: ~0.01. Variance: ~1e-4.
        # Percent-squared would be ~1.0 — 10000x too large.
        for col in ["rv_5", "rv_22"]:
            col_data = result[col].dropna()
            assert len(col_data) > 0, f"{col} has no non-NaN values"
            assert col_data.max() < 0.1, (
                f"{col} max={col_data.max():.6f} — likely percent-squared (should be ~1e-4)"
            )

    def test_ewma_var_in_decimal_variance_range(self) -> None:
        df = _ohlc(100)
        result = build_features(df, include_garch=False)
        ewma = result["ewma_var"].dropna()
        assert len(ewma) > 0
        assert ewma.max() < 0.1, "ewma_var appears to be in percent-squared units"
        assert (ewma > 0).all()

    def test_parkinson_in_decimal_variance_range(self) -> None:
        df = _ohlc(100)
        result = build_features(df, include_garch=False)
        pv = result["parkinson_var"].dropna()
        assert len(pv) > 0
        assert pv.max() < 0.1, "parkinson_var too large (annualized?)"
        assert (pv > 0).all()
