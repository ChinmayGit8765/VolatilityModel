"""Unit tests for log-variance transforms and asset-categorical constants.

Tests the <behavior> contract from Plan 03-01 Task 2:
- to_log_var / from_log_var round-trip on positive variances within rtol 1e-9
- Epsilon floor: to_log_var(0.0) and to_log_var(negative) return log(LOG_VAR_EPS)
- from_log_var is exactly np.exp; vector round-trip holds
- ASSET_DTYPE is a pd.CategoricalDtype over KNOWN_ASSETS, ordered=False
- Known symbols → no NaN categories; unknown symbol → NaN category
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---- module under test -------------------------------------------------------
from volforecast.models.lgbm import (
    ASSET_DTYPE,
    KNOWN_ASSETS,
    LOG_VAR_EPS,
    from_log_var,
    to_log_var,
)

# ---- constants ---------------------------------------------------------------

REALISTIC_VARIANCES = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
EXPECTED_KNOWN_ASSETS = ["BTC-USD", "ETH-USD", "SPY", "AAPL", "MSFT"]


# ---- LOG_VAR_EPS tests -------------------------------------------------------


class TestLogVarEps:
    def test_equals_qlike_floor(self):
        """LOG_VAR_EPS must equal 1e-10 (same as QLIKE_FLOOR in eval/metrics.py)."""
        assert LOG_VAR_EPS == 1e-10

    def test_is_float(self):
        assert isinstance(LOG_VAR_EPS, float)


# ---- to_log_var tests --------------------------------------------------------


class TestToLogVar:
    def test_positive_scalar_no_inf_nan(self):
        """to_log_var of a positive scalar returns a finite float."""
        result = to_log_var(0.001)
        assert np.isfinite(result)

    def test_zero_returns_log_eps(self):
        """to_log_var(0.0) must floor at LOG_VAR_EPS and return log(LOG_VAR_EPS)."""
        result = to_log_var(0.0)
        expected = np.log(LOG_VAR_EPS)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_negative_returns_log_eps(self):
        """to_log_var(negative) must floor at LOG_VAR_EPS, no inf/nan."""
        result = to_log_var(-1.0)
        expected = np.log(LOG_VAR_EPS)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_value_below_eps_floored(self):
        """Values below LOG_VAR_EPS are floored (not -inf)."""
        result = to_log_var(1e-20)
        # Must equal log(LOG_VAR_EPS) since 1e-20 < 1e-10
        expected = np.log(LOG_VAR_EPS)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_vector_input(self):
        """to_log_var accepts array-like and returns np.ndarray."""
        result = to_log_var(REALISTIC_VARIANCES)
        assert isinstance(result, np.ndarray)
        assert result.shape == REALISTIC_VARIANCES.shape
        assert np.all(np.isfinite(result))

    def test_returns_ndarray(self):
        """Result type is np.ndarray for scalar and vector inputs."""
        assert isinstance(to_log_var(0.001), (float, np.floating, np.ndarray))
        assert isinstance(to_log_var([0.001, 0.002]), np.ndarray)


# ---- from_log_var tests ------------------------------------------------------


class TestFromLogVar:
    def test_is_exp(self):
        """from_log_var is exactly np.exp."""
        log_vals = np.array([-10.0, -5.0, 0.0, 1.0])
        result = from_log_var(log_vals)
        expected = np.exp(log_vals)
        np.testing.assert_array_equal(result, expected)

    def test_positive_output(self):
        """from_log_var always returns positive values."""
        log_vals = np.linspace(-20, 5, 50)
        result = from_log_var(log_vals)
        assert np.all(result > 0)

    def test_returns_ndarray(self):
        """from_log_var returns np.ndarray for array input."""
        result = from_log_var(np.array([-5.0, -4.0]))
        assert isinstance(result, np.ndarray)


# ---- round-trip tests --------------------------------------------------------


class TestRoundTrip:
    def test_scalar_round_trip(self):
        """from_log_var(to_log_var(v)) recovers v within rtol 1e-9 for v > LOG_VAR_EPS."""
        for v in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
            recovered = from_log_var(to_log_var(v))
            np.testing.assert_allclose(
                recovered, v, rtol=1e-9, err_msg=f"Round-trip failed for v={v}"
            )

    def test_vector_round_trip(self):
        """Vector of realistic variances round-trips within rtol 1e-9."""
        recovered = from_log_var(to_log_var(REALISTIC_VARIANCES))
        np.testing.assert_allclose(recovered, REALISTIC_VARIANCES, rtol=1e-9)

    def test_eps_floor_round_trip(self):
        """Values at LOG_VAR_EPS recover to LOG_VAR_EPS exactly."""
        recovered = from_log_var(to_log_var(LOG_VAR_EPS))
        np.testing.assert_allclose(recovered, LOG_VAR_EPS, rtol=1e-9)


# ---- KNOWN_ASSETS tests ------------------------------------------------------


class TestKnownAssets:
    def test_exact_list(self):
        """KNOWN_ASSETS must be exactly the 5-symbol universe in config order."""
        assert KNOWN_ASSETS == EXPECTED_KNOWN_ASSETS

    def test_is_list_of_strings(self):
        assert isinstance(KNOWN_ASSETS, list)
        assert all(isinstance(s, str) for s in KNOWN_ASSETS)

    def test_length(self):
        assert len(KNOWN_ASSETS) == 5


# ---- ASSET_DTYPE tests -------------------------------------------------------


class TestAssetDtype:
    def test_is_categorical_dtype(self):
        """ASSET_DTYPE must be a pd.CategoricalDtype."""
        assert isinstance(ASSET_DTYPE, pd.CategoricalDtype)

    def test_not_ordered(self):
        """ASSET_DTYPE must be ordered=False."""
        assert ASSET_DTYPE.ordered is False

    def test_categories_match_known_assets(self):
        """ASSET_DTYPE categories must exactly match KNOWN_ASSETS."""
        assert list(ASSET_DTYPE.categories) == KNOWN_ASSETS

    def test_known_symbol_no_nan(self):
        """Applying ASSET_DTYPE to a Series of valid symbols yields no NaN categories."""
        s = pd.Series(KNOWN_ASSETS, dtype=ASSET_DTYPE)
        assert s.isna().sum() == 0
        assert list(s.cat.categories) == KNOWN_ASSETS

    def test_unknown_symbol_becomes_nan(self):
        """An unknown symbol becomes a NaN category (documented behavior)."""
        s = pd.Series(["BTC-USD", "UNKNOWN-XYZ"], dtype=ASSET_DTYPE)
        # "UNKNOWN-XYZ" is not in categories → should be NaN
        assert s.isna().sum() == 1
        assert not s.isna().iloc[0]  # BTC-USD is valid
        assert s.isna().iloc[1]  # UNKNOWN-XYZ is NaN

    def test_all_assets_round_trip(self):
        """All 5 assets can be converted to the category dtype and back to string."""
        s = pd.Series(KNOWN_ASSETS, dtype=ASSET_DTYPE)
        recovered = list(s.astype(str))
        assert recovered == KNOWN_ASSETS
