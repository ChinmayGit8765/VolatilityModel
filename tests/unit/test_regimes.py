"""Unit tests for volforecast.eval.regimes.

Tests cover:
- assign_vol_terciles: ~1/3 in each bucket for uniform input
- assign_vol_terciles: sorted series gives low/mid/high labels correctly
- assign_vol_terciles: constant-variance edge case (no raise)
- assign_vol_terciles: degenerate series (all same value) handled gracefully
- assign_vol_terciles: only test-fold data used (function is structurally closed)
- assign_calendar_year: returns integer year per row from DatetimeIndex
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast.eval.regimes import assign_calendar_year, assign_vol_terciles


class TestAssignVolTerciles:
    """Tests for assign_vol_terciles(realized_var)."""

    def test_roughly_one_third_per_bucket(self):
        """A uniform series should produce ~33% in each bucket."""
        rng = np.random.default_rng(42)
        n = 300
        rv = pd.Series(rng.uniform(0.0001, 0.001, size=n))
        labels = assign_vol_terciles(rv)
        counts = labels.value_counts()
        # Each bucket should have between 85 and 115 rows for n=300
        for label in ("low", "mid", "high"):
            assert label in counts.index, f"Label '{label}' missing"
            assert 85 <= counts[label] <= 115, (
                f"Bucket '{label}' has {counts[label]} rows — expected ~100"
            )

    def test_sorted_series_assigns_correct_labels(self):
        """A sorted series should assign low/mid/high in order."""
        rv = pd.Series([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006])
        # 6 elements: positions 0-1 → low, 2-3 → mid, 4-5 → high
        labels = assign_vol_terciles(rv)
        assert labels.iloc[0] == "low"
        assert labels.iloc[2] in ("low", "mid")
        assert labels.iloc[5] == "high"

    def test_low_values_are_low(self):
        """Values below the 1/3 quantile must be labelled 'low'."""
        rv = pd.Series(list(range(1, 100)))  # 1..99
        labels = assign_vol_terciles(rv)
        # The first 33 values (1..33) should be 'low'
        assert (labels.iloc[:33] == "low").all(), "First third should all be 'low'"

    def test_high_values_are_high(self):
        """Values above the 2/3 quantile must be labelled 'high'."""
        rv = pd.Series(list(range(1, 100)))  # 1..99
        labels = assign_vol_terciles(rv)
        # The last 33 values (67..99) should be 'high'
        assert (labels.iloc[-33:] == "high").all(), "Last third should all be 'high'"

    def test_constant_variance_no_raise(self):
        """A constant series must not raise; degenerate buckets are acceptable."""
        rv = pd.Series([0.0005] * 50)
        # Should NOT raise — all values may fall into a single bucket
        labels = assign_vol_terciles(rv)
        assert len(labels) == 50
        # All labels should be one of the valid categories
        valid = {"low", "mid", "high"}
        assert set(labels.unique()).issubset(valid)

    def test_single_element_series(self):
        """A length-1 series must not raise."""
        rv = pd.Series([0.0003])
        labels = assign_vol_terciles(rv)
        assert len(labels) == 1
        assert labels.iloc[0] in ("low", "mid", "high")

    def test_returns_string_or_categorical_series(self):
        """Output must be a pd.Series with str or categorical dtype."""
        rv = pd.Series(np.linspace(0.0001, 0.001, 90))
        labels = assign_vol_terciles(rv)
        assert isinstance(labels, pd.Series)
        # Values must be strings (low / mid / high)
        unique_vals = set(labels.unique())
        assert unique_vals <= {"low", "mid", "high"}

    def test_same_index_as_input(self):
        """Output index must match input index."""
        idx = pd.date_range("2020-01-01", periods=60, freq="D")
        rv = pd.Series(np.linspace(0.0001, 0.001, 60), index=idx)
        labels = assign_vol_terciles(rv)
        pd.testing.assert_index_equal(labels.index, rv.index)

    def test_no_lookahead_structurally(self):
        """The function takes only the test-fold series — it cannot access train data."""
        # This is a structural / documentation test:
        # Pass only a sub-series (simulating test-fold realized var) and verify
        # the tercile boundaries are computed on THAT sub-series only.
        full_rv = pd.Series(np.linspace(0.0001, 0.001, 200))
        test_rv = full_rv.iloc[150:]  # last 50 rows only
        labels = assign_vol_terciles(test_rv)
        # Boundaries should be based on test_rv's own 33/67 quantiles
        q33 = test_rv.quantile(1 / 3)
        q67 = test_rv.quantile(2 / 3)
        for idx_val, label in labels.items():
            val = test_rv[idx_val]
            if val <= q33:
                assert label == "low", f"val={val} q33={q33} should be 'low', got '{label}'"
            elif val <= q67:
                assert label in ("mid", "low"), f"val={val} should be 'mid', got '{label}'"
            else:
                assert label == "high", f"val={val} q67={q67} should be 'high', got '{label}'"


class TestAssignCalendarYear:
    """Tests for assign_calendar_year(index)."""

    def test_basic_year_extraction(self):
        """Returns the integer year for each date."""
        idx = pd.DatetimeIndex(["2020-01-15", "2021-06-01", "2022-12-31"])
        years = assign_calendar_year(idx)
        assert list(years) == [2020, 2021, 2022]

    def test_returns_int_dtype(self):
        """Year values should be integers (int64 or Python int)."""
        idx = pd.date_range("2020-01-01", periods=5, freq="ME")
        years = assign_calendar_year(idx)
        assert isinstance(years, pd.Series)
        assert np.issubdtype(years.dtype, np.integer)

    def test_same_index_as_input(self):
        """Output index must match input DatetimeIndex."""
        idx = pd.date_range("2019-07-01", periods=10, freq="D")
        years = assign_calendar_year(idx)
        pd.testing.assert_index_equal(years.index, idx)

    def test_multi_year_coverage(self):
        """Should correctly label rows spanning multiple years."""
        idx = pd.date_range("2018-01-01", periods=1500, freq="D")
        years = assign_calendar_year(idx)
        unique_years = sorted(years.unique())
        assert 2018 in unique_years
        assert 2022 in unique_years

    def test_single_date(self):
        """A length-1 DatetimeIndex must work."""
        idx = pd.DatetimeIndex(["2024-03-15"])
        years = assign_calendar_year(idx)
        assert len(years) == 1
        assert years.iloc[0] == 2024

    def test_year_value_matches_date_year(self):
        """Each row's year value must equal the date's .year attribute."""
        dates = pd.date_range("2015-01-01", periods=365 * 5, freq="D")
        years = assign_calendar_year(dates)
        for date, year in zip(dates, years):
            assert date.year == year, f"Mismatch: {date} -> year={year}"
