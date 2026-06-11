"""Unit tests for src/volforecast/features/cross_asset.py.

All tests are offline-only (no network, no live data).  Frames are constructed
inline with deliberately mismatched calendars.

Coverage:
- as_of_join attaches the most recent prior source feature (backward direction)
- tolerance=3D: source row > 3 calendar days stale → NaN
- 2-day gap: value present; 4-day gap: NaN (boundary test)
- Unsorted input is sorted before merge_asof (sort-safe)
- Output index is the target's UTC DatetimeIndex; columns carry suffix
- MAX_CROSS_ASSET_STALENESS constant exists and equals pd.Timedelta("3D")
"""

from __future__ import annotations

import pandas as pd
import pytest

from volforecast.features.cross_asset import MAX_CROSS_ASSET_STALENESS, as_of_join

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc(dates: list[str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(dates, utc=True), name="date")


def _left(dates: list[str], vals: list[float] | None = None) -> pd.DataFrame:
    """Build a left (target) DataFrame with a UTC DatetimeIndex."""
    idx = _utc(dates)
    vals = vals if vals is not None else list(range(len(dates)))
    return pd.DataFrame({"target_val": vals}, index=idx)


def _right(dates: list[str], vals: list[float]) -> pd.DataFrame:
    """Build a right (source) DataFrame with a single feature column."""
    idx = _utc(dates)
    return pd.DataFrame({"btc_rv": vals}, index=idx)


# ---------------------------------------------------------------------------
# MAX_CROSS_ASSET_STALENESS constant
# ---------------------------------------------------------------------------


class TestMaxStalenessConstant:
    def test_value_is_3d(self) -> None:
        assert MAX_CROSS_ASSET_STALENESS == pd.Timedelta("3D")

    def test_type(self) -> None:
        assert isinstance(MAX_CROSS_ASSET_STALENESS, pd.Timedelta)


# ---------------------------------------------------------------------------
# Backward match — basic
# ---------------------------------------------------------------------------


class TestBackwardMatch:
    def test_exact_date_match(self) -> None:
        """When left and right have the same date, the value is attached."""
        left = _left(["2022-01-03", "2022-01-04", "2022-01-05"])
        right = _right(["2022-01-03", "2022-01-04", "2022-01-05"], [1.0, 2.0, 3.0])
        result = as_of_join(left, right, ["btc_rv"])
        # Exact match: btc_rv_xasset at 2022-01-03 should be 1.0
        assert result["btc_rv_xasset"].iloc[0] == pytest.approx(1.0)
        assert result["btc_rv_xasset"].iloc[1] == pytest.approx(2.0)
        assert result["btc_rv_xasset"].iloc[2] == pytest.approx(3.0)

    def test_backward_lookup_picks_prior_value(self) -> None:
        """When left date is between two right dates, take the earlier (prior) value."""
        # right: Mon 3rd and Wed 5th; left: Tue 4th → should get Mon's value
        left = _left(["2022-01-04"])
        right = _right(["2022-01-03", "2022-01-05"], [10.0, 20.0])
        result = as_of_join(left, right, ["btc_rv"])
        # direction=backward → picks 2022-01-03 (prior), not 2022-01-05 (future)
        assert result["btc_rv_xasset"].iloc[0] == pytest.approx(10.0)

    def test_output_index_is_left_index(self) -> None:
        """Output index must equal the target's UTC DatetimeIndex."""
        left = _left(["2022-01-03", "2022-01-04", "2022-01-05"])
        right = _right(["2022-01-03", "2022-01-04", "2022-01-05"], [1.0, 2.0, 3.0])
        result = as_of_join(left, right, ["btc_rv"])
        pd.testing.assert_index_equal(result.index, left.index)

    def test_left_columns_preserved(self) -> None:
        """All original left columns must be present in output."""
        left = _left(["2022-01-03", "2022-01-04"])
        right = _right(["2022-01-03", "2022-01-04"], [1.0, 2.0])
        result = as_of_join(left, right, ["btc_rv"])
        assert "target_val" in result.columns


# ---------------------------------------------------------------------------
# 3-day staleness boundary
# ---------------------------------------------------------------------------


class TestStalenessRule:
    def test_2_day_gap_yields_value(self) -> None:
        """A 2-day gap between source and target should return the prior value."""
        # source on Monday 2022-01-03; target on Wednesday 2022-01-05
        # gap = 2 calendar days <= 3D → should attach the value
        left = _left(["2022-01-05"])  # Wednesday
        right = _right(["2022-01-03"], [42.0])  # Monday (2 days before)
        result = as_of_join(left, right, ["btc_rv"])
        assert result["btc_rv_xasset"].iloc[0] == pytest.approx(42.0)

    def test_3_day_gap_yields_value(self) -> None:
        """A 3-day gap (exactly at tolerance boundary) should still return the value."""
        # source on Monday 2022-01-03; target on Thursday 2022-01-06
        # gap = 3 calendar days = 3D tolerance → should attach (tolerance is inclusive)
        left = _left(["2022-01-06"])  # Thursday
        right = _right(["2022-01-03"], [99.0])  # Monday (3 days before)
        result = as_of_join(left, right, ["btc_rv"])
        assert result["btc_rv_xasset"].iloc[0] == pytest.approx(99.0)

    def test_4_day_gap_yields_nan(self) -> None:
        """A 4-day gap exceeds the 3D tolerance and must produce NaN."""
        # source on Monday 2022-01-03; target on Friday 2022-01-07
        # gap = 4 calendar days > 3D → NaN
        left = _left(["2022-01-07"])  # Friday
        right = _right(["2022-01-03"], [77.0])  # Monday (4 days before)
        result = as_of_join(left, right, ["btc_rv"])
        assert pd.isna(result["btc_rv_xasset"].iloc[0])

    def test_5_day_gap_yields_nan(self) -> None:
        """A 5-day gap (e.g. a long holiday) also yields NaN."""
        left = _left(["2022-01-08"])  # Saturday
        right = _right(["2022-01-03"], [55.0])  # Monday (5 days before)
        result = as_of_join(left, right, ["btc_rv"])
        assert pd.isna(result["btc_rv_xasset"].iloc[0])

    def test_mixed_2_and_4_day_gaps(self) -> None:
        """Mixed rows: 2-day gap has a value; 4-day gap is NaN."""
        left = _left(["2022-01-05", "2022-01-07"])  # Wed (2-day gap), Fri (4-day gap)
        right = _right(["2022-01-03"], [88.0])  # Monday only
        result = as_of_join(left, right, ["btc_rv"])
        assert result["btc_rv_xasset"].iloc[0] == pytest.approx(88.0)  # 2-day gap
        assert pd.isna(result["btc_rv_xasset"].iloc[1])  # 4-day gap

    def test_no_prior_source_row_yields_nan(self) -> None:
        """If there is no prior source row at all, result is NaN."""
        left = _left(["2022-01-03"])  # Monday
        right = _right(["2022-01-05"], [10.0])  # Wednesday (future — not prior)
        result = as_of_join(left, right, ["btc_rv"])
        assert pd.isna(result["btc_rv_xasset"].iloc[0])


# ---------------------------------------------------------------------------
# Sort safety
# ---------------------------------------------------------------------------


class TestSortSafety:
    def test_unsorted_left_is_sorted_before_merge(self) -> None:
        """as_of_join must sort an unsorted left frame and return correct result."""
        # Build left in reverse order
        left = _left(["2022-01-05", "2022-01-04", "2022-01-03"])
        right = _right(["2022-01-03", "2022-01-04", "2022-01-05"], [1.0, 2.0, 3.0])
        result = as_of_join(left, right, ["btc_rv"])
        # All three should have correct values regardless of input order
        assert set(result["btc_rv_xasset"].dropna().values) == {1.0, 2.0, 3.0}

    def test_unsorted_right_is_sorted_before_merge(self) -> None:
        """as_of_join must sort an unsorted right frame before merge_asof."""
        left = _left(["2022-01-03", "2022-01-04", "2022-01-05"])
        # Right in reverse order
        right = _right(["2022-01-05", "2022-01-04", "2022-01-03"], [3.0, 2.0, 1.0])
        result = as_of_join(left, right, ["btc_rv"])
        # Jan-03 → 1.0, Jan-04 → 2.0, Jan-05 → 3.0
        assert result["btc_rv_xasset"].iloc[0] == pytest.approx(1.0)
        assert result["btc_rv_xasset"].iloc[1] == pytest.approx(2.0)
        assert result["btc_rv_xasset"].iloc[2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Suffix handling
# ---------------------------------------------------------------------------


class TestSuffixHandling:
    def test_default_suffix_appended(self) -> None:
        """Joined columns must have the '_xasset' suffix by default."""
        left = _left(["2022-01-03"])
        right = _right(["2022-01-03"], [1.0])
        result = as_of_join(left, right, ["btc_rv"])
        assert "btc_rv_xasset" in result.columns
        assert "btc_rv" not in result.columns

    def test_custom_suffix(self) -> None:
        """Custom suffix is applied to joined columns."""
        left = _left(["2022-01-03"])
        right = _right(["2022-01-03"], [1.0])
        result = as_of_join(left, right, ["btc_rv"], suffix="_btc")
        assert "btc_rv_btc" in result.columns

    def test_multiple_feature_cols(self) -> None:
        """Multiple feature columns are all joined with the suffix."""
        left = _left(["2022-01-03", "2022-01-04"])
        idx = _utc(["2022-01-03", "2022-01-04"])
        right = pd.DataFrame({"rv22": [0.001, 0.002], "rv5": [0.0005, 0.0006]}, index=idx)
        result = as_of_join(left, right, ["rv22", "rv5"])
        assert "rv22_xasset" in result.columns
        assert "rv5_xasset" in result.columns


# ---------------------------------------------------------------------------
# Mismatched calendars (crypto 7-day vs equity 5-day)
# ---------------------------------------------------------------------------


class TestMismatchedCalendars:
    def test_crypto_source_equity_target(self) -> None:
        """Crypto (Mon–Sun) source joined to equity (Mon–Fri) target.

        Weekend crypto data can be used as the last prior value for Monday equity.
        The Saturday/Sunday values are within 3 days of Monday, so they attach.
        """
        # Crypto: Mon, Tue, Wed, Thu, Fri, Sat, Sun
        crypto_dates = [
            "2022-01-03",  # Mon
            "2022-01-04",  # Tue
            "2022-01-05",  # Wed
            "2022-01-06",  # Thu
            "2022-01-07",  # Fri
            "2022-01-08",  # Sat
            "2022-01-09",  # Sun
        ]
        crypto_rv = [0.0010, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016]
        right = _right(crypto_dates, crypto_rv)

        # Equity: Mon–Fri only; next Monday is Jan 10
        equity_dates = ["2022-01-10"]  # next Monday
        left = _left(equity_dates)

        result = as_of_join(left, right, ["btc_rv"])
        # Jan 10 (Mon) picks Jan 09 (Sun, 1 day gap) → 0.0016
        assert result["btc_rv_xasset"].iloc[0] == pytest.approx(0.0016)

    def test_holiday_gap_exceeding_3_days(self) -> None:
        """A 4-day or longer holiday gap on either side produces NaN."""
        # Source: Mon Jan 3; target: Sat Jan 8 (5 days later) → NaN
        left = _left(["2022-01-08"])  # Saturday
        right = _right(["2022-01-03"], [50.0])
        result = as_of_join(left, right, ["btc_rv"])
        assert pd.isna(result["btc_rv_xasset"].iloc[0])


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_index_is_utc(self) -> None:
        """Output index must be tz-aware UTC."""
        left = _left(["2022-01-03", "2022-01-04"])
        right = _right(["2022-01-03", "2022-01-04"], [1.0, 2.0])
        result = as_of_join(left, right, ["btc_rv"])
        assert result.index.tzinfo is not None
        assert str(result.index.tz) == "UTC"

    def test_index_name(self) -> None:
        """Output index name must be 'date'."""
        left = _left(["2022-01-03"])
        right = _right(["2022-01-03"], [1.0])
        result = as_of_join(left, right, ["btc_rv"])
        assert result.index.name == "date"

    def test_returns_dataframe(self) -> None:
        left = _left(["2022-01-03"])
        right = _right(["2022-01-03"], [1.0])
        result = as_of_join(left, right, ["btc_rv"])
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_left(self) -> None:
        """Output must have the same number of rows as left."""
        left = _left(["2022-01-03", "2022-01-04", "2022-01-05"])
        right = _right(["2022-01-03"], [1.0])
        result = as_of_join(left, right, ["btc_rv"])
        assert len(result) == len(left)
