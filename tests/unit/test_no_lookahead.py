"""No-lookahead (truncation invariance) test for build_features().

This test is the FEAT-02 / FEAT-07 no-skew guarantee: for any date t, the
feature row produced by build_features on the full series equals the feature
row produced on data[:t+1] (the truncated series).

This proves that every rolling window ends strictly at as-of t and that
GARCH-as-feature is never fitted on data past as-of t.

All tests are offline-only (no network, no live data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volforecast.features.pipeline import build_features

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlc(n: int = 350, base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic OHLC DataFrame with UTC DatetimeIndex named 'date'."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-02", periods=n, freq="B", tz="UTC", name="date")
    ret = rng.normal(0, 0.01, n)
    close = base * np.exp(np.cumsum(ret))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1.0 + rng.normal(0, 0.003, n))
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1_000_000.0,
        },
        index=dates,
    )


def _features_at_t(
    df: pd.DataFrame,
    t_pos: int,
    include_garch: bool = False,
) -> pd.Series:
    """Compute features on truncated data df[:t_pos+1] and return row at t_pos."""
    truncated = df.iloc[: t_pos + 1]
    feats = build_features(truncated, include_garch=include_garch)
    return feats.iloc[t_pos]


def _features_full(
    df: pd.DataFrame,
    t_pos: int,
    include_garch: bool = False,
) -> pd.Series:
    """Compute features on full data and return row at t_pos."""
    feats = build_features(df, include_garch=include_garch)
    return feats.iloc[t_pos]


# ---------------------------------------------------------------------------
# Truncation invariance for non-GARCH features
# ---------------------------------------------------------------------------


class TestTruncationInvarianceNoGarch:
    """For non-GARCH features, feature[t] must be identical whether computed
    on data[:t+1] or the full series.  This is the no-lookahead guarantee."""

    @pytest.mark.parametrize("t_pos", [100, 150, 200, 250])
    def test_rv_features_invariant(self, t_pos: int) -> None:
        """rv_5, rv_22, rv_66 at t must not change when future data is removed."""
        df = _ohlc(350)
        row_full = _features_full(df, t_pos, include_garch=False)
        row_trunc = _features_at_t(df, t_pos, include_garch=False)

        for col in ["rv_5", "rv_10", "rv_22", "rv_66"]:
            if pd.isna(row_full[col]) and pd.isna(row_trunc[col]):
                continue  # both NaN — consistent
            assert row_full[col] == pytest.approx(row_trunc[col], rel=1e-10), (
                f"Feature '{col}' at t_pos={t_pos}: "
                f"full={row_full[col]}, truncated={row_trunc[col]}"
            )

    @pytest.mark.parametrize("t_pos", [100, 150, 200])
    def test_ewma_var_invariant(self, t_pos: int) -> None:
        """ewma_var at t must equal ewma_var computed on truncated data."""
        df = _ohlc(300)
        row_full = _features_full(df, t_pos, include_garch=False)
        row_trunc = _features_at_t(df, t_pos, include_garch=False)
        if not (pd.isna(row_full["ewma_var"]) or pd.isna(row_trunc["ewma_var"])):
            assert row_full["ewma_var"] == pytest.approx(row_trunc["ewma_var"], rel=1e-10)

    @pytest.mark.parametrize("t_pos", [100, 150, 200])
    def test_parkinson_invariant(self, t_pos: int) -> None:
        """Parkinson variance at t must be row-wise (trivially invariant)."""
        df = _ohlc(300)
        row_full = _features_full(df, t_pos, include_garch=False)
        row_trunc = _features_at_t(df, t_pos, include_garch=False)
        assert row_full["parkinson_var"] == pytest.approx(row_trunc["parkinson_var"], rel=1e-12)

    @pytest.mark.parametrize("t_pos", [100, 150, 200])
    def test_lagged_vol_invariant(self, t_pos: int) -> None:
        """lagged_vol at t is rv shifted by 1 — uses only past data."""
        df = _ohlc(300)
        row_full = _features_full(df, t_pos, include_garch=False)
        row_trunc = _features_at_t(df, t_pos, include_garch=False)
        if not pd.isna(row_full["lagged_vol"]):
            assert row_full["lagged_vol"] == pytest.approx(row_trunc["lagged_vol"], rel=1e-10)

    @pytest.mark.parametrize("t_pos", [110, 150, 200])
    def test_vol_of_vol_invariant(self, t_pos: int) -> None:
        """vol_of_vol at t uses only data up to and including t."""
        df = _ohlc(300)
        row_full = _features_full(df, t_pos, include_garch=False)
        row_trunc = _features_at_t(df, t_pos, include_garch=False)
        if not pd.isna(row_full["vol_of_vol"]):
            assert row_full["vol_of_vol"] == pytest.approx(row_trunc["vol_of_vol"], rel=1e-10)

    @pytest.mark.parametrize("t_pos", [110, 150, 200])
    def test_rolling_skew_invariant(self, t_pos: int) -> None:
        """rolling_skew at t uses only data up to and including t."""
        df = _ohlc(300)
        row_full = _features_full(df, t_pos, include_garch=False)
        row_trunc = _features_at_t(df, t_pos, include_garch=False)
        if not (pd.isna(row_full["rolling_skew"]) or pd.isna(row_trunc["rolling_skew"])):
            assert row_full["rolling_skew"] == pytest.approx(row_trunc["rolling_skew"], rel=1e-8)

    @pytest.mark.parametrize("t_pos", [120, 150, 200])
    def test_calendar_features_invariant(self, t_pos: int) -> None:
        """Calendar features (day_of_week, month) are row-wise; trivially invariant."""
        df = _ohlc(300)
        row_full = _features_full(df, t_pos, include_garch=False)
        row_trunc = _features_at_t(df, t_pos, include_garch=False)
        assert row_full["day_of_week"] == row_trunc["day_of_week"]
        assert row_full["month"] == row_trunc["month"]

    def test_log_return_invariant(self) -> None:
        """log_return at t is ln(close[t]/close[t-1]) — never looks ahead."""
        df = _ohlc(300)
        t_pos = 150
        row_full = _features_full(df, t_pos, include_garch=False)
        row_trunc = _features_at_t(df, t_pos, include_garch=False)
        assert row_full["log_return"] == pytest.approx(row_trunc["log_return"], rel=1e-12)


# ---------------------------------------------------------------------------
# Truncation invariance for GARCH feature
# ---------------------------------------------------------------------------


class TestGarchNoLookahead:
    """GARCH-as-feature must never be fitted on data past as-of t.

    The test verifies that garch_cond_var[t] is identical whether the model
    is computed on data[:t+1] or a longer slice that includes data > t.

    Note: GARCH conditional variance is computed from the FILTERED series
    (in-sample smoothing), not forecasting.  The filter only uses data up to t
    given a model fitted on data up to t, so truncation invariance holds when
    the same refit window covers the same training data.  We test this by
    verifying that adding future data does not change the value at t.
    """

    def test_garch_cond_var_truncation_invariance(self) -> None:
        """garch_cond_var at t must equal garch_cond_var computed on data[:t+1]."""
        df = _ohlc(350)
        # Use a t_pos well into the series so GARCH has enough training data
        # min_train default is 252; pick t_pos = 280 (after first refit window)
        t_pos = 280

        row_full = _features_full(df, t_pos, include_garch=True)
        row_trunc = _features_at_t(df, t_pos, include_garch=True)

        full_val = row_full["garch_cond_var"]
        trunc_val = row_trunc["garch_cond_var"]

        # If both NaN (not enough training data for GARCH), that's acceptable
        if pd.isna(full_val) and pd.isna(trunc_val):
            return

        # Both should be non-NaN and equal at this point
        assert not pd.isna(full_val), "garch_cond_var is NaN on full series at t_pos=280"
        assert not pd.isna(trunc_val), "garch_cond_var is NaN on truncated series"
        assert full_val == pytest.approx(trunc_val, rel=1e-6), (
            f"GARCH-as-feature lookahead detected: full={full_val:.8e}, truncated={trunc_val:.8e}"
        )

    def test_garch_uses_only_past_data(self) -> None:
        """Verify GARCH-as-feature at t is the same as on data[:t+1] for several t."""
        df = _ohlc(350)
        # Test at the boundary of the first GARCH refit window (min_train=252)
        for t_pos in [260, 270, 280]:
            row_full = _features_full(df, t_pos, include_garch=True)
            row_trunc = _features_at_t(df, t_pos, include_garch=True)

            full_val = row_full.get("garch_cond_var")
            trunc_val = row_trunc.get("garch_cond_var")

            if pd.isna(full_val) or pd.isna(trunc_val):
                continue  # GARCH not yet initialized — acceptable

            assert full_val == pytest.approx(trunc_val, rel=1e-6), (
                f"GARCH lookahead at t_pos={t_pos}: full={full_val:.8e}, trunc={trunc_val:.8e}"
            )


# ---------------------------------------------------------------------------
# Smoke tests on real fixture data
# ---------------------------------------------------------------------------


class TestFixtureSmoke:
    def test_no_lookahead_on_equity_fixture(self) -> None:
        """Truncation invariance smoke test on the real equity fixture."""
        df = pd.read_parquet(FIXTURES / "equity_sample.parquet")
        if len(df) < 80:
            pytest.skip("fixture too short for this test")
        t_pos = min(70, len(df) - 1)
        row_full = _features_full(df, t_pos, include_garch=False)
        row_trunc = _features_at_t(df, t_pos, include_garch=False)
        for col in ["rv_5", "rv_22", "parkinson_var", "ewma_var"]:
            if pd.isna(row_full[col]) or pd.isna(row_trunc[col]):
                continue
            assert row_full[col] == pytest.approx(row_trunc[col], rel=1e-10), (
                f"Equity fixture lookahead in '{col}' at t={t_pos}"
            )

    def test_no_lookahead_on_crypto_fixture(self) -> None:
        """Truncation invariance smoke test on the real crypto fixture."""
        df = pd.read_parquet(FIXTURES / "crypto_sample.parquet")
        if len(df) < 80:
            pytest.skip("fixture too short for this test")
        t_pos = min(80, len(df) - 1)
        row_full = _features_full(df, t_pos, include_garch=False)
        row_trunc = _features_at_t(df, t_pos, include_garch=False)
        for col in ["rv_5", "rv_22", "parkinson_var"]:
            if pd.isna(row_full[col]) or pd.isna(row_trunc[col]):
                continue
            assert row_full[col] == pytest.approx(row_trunc[col], rel=1e-10), (
                f"Crypto fixture lookahead in '{col}' at t={t_pos}"
            )
