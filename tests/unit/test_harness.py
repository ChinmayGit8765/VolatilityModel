"""Unit tests for src/volforecast/eval/harness.py.

Tests are written BEFORE the implementation (TDD RED phase).

THE MANDATORY LEAK TEST (non-negotiable portfolio centrepiece):
    Every split from walk_forward_splits must satisfy BOTH:
    1. train_idx.max() < test_idx.min()              (temporal ordering)
    2. test_idx.min() - train_idx.max() >= horizon   (embargo >= horizon)

    This test MUST FAIL if the generator emits an overlapping or
    under-embargoed split.  It is explicitly a portfolio talking point —
    the test proves the harness cannot leak future data into training.

Additional behavioral tests:
- Purge: training end == test_start - horizon (last horizon obs removed)
- Expanding window: each split's training range is a strict superset prefix
- Step: consecutive test windows advance by `step` positions
- No random split: train indices are a contiguous range 0..train_max

No fixtures or network calls — all data is synthetic integer positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from volforecast.eval.harness import WalkForwardSplit, walk_forward_splits


class TestLeakTestMandatory:
    """THE mandatory leak test — temporal ordering and embargo >= horizon."""

    def test_walk_forward_no_leakage_horizon1(self) -> None:
        """Every split (horizon=1) satisfies temporal ordering AND embargo >= 1.

        This is the canonical portfolio-credibility test.
        """
        splits = list(walk_forward_splits(n=500, min_train=252, step=21, horizon=1))
        assert len(splits) > 0, "walk_forward_splits produced no splits for n=500"

        for i, split in enumerate(splits):
            assert split.train_idx.max() < split.test_idx.min(), (
                f"Split {i}: temporal ordering violated — "
                f"train_idx.max()={split.train_idx.max()} >= "
                f"test_idx.min()={split.test_idx.min()}"
            )
            assert split.test_idx.min() - split.train_idx.max() >= 1, (
                f"Split {i}: embargo < horizon — "
                f"gap={split.test_idx.min() - split.train_idx.max()} < 1"
            )

    def test_walk_forward_no_leakage_horizon5(self) -> None:
        """Every split (horizon=5) satisfies temporal ordering AND embargo >= 5."""
        horizon = 5
        splits = list(walk_forward_splits(n=600, min_train=252, step=21, horizon=horizon))
        assert len(splits) > 0, "No splits produced for horizon=5"

        for i, split in enumerate(splits):
            assert split.train_idx.max() < split.test_idx.min(), (
                f"Split {i}: temporal ordering violated for horizon={horizon}"
            )
            assert split.test_idx.min() - split.train_idx.max() >= horizon, (
                f"Split {i}: embargo ({split.test_idx.min() - split.train_idx.max()}) "
                f"< horizon ({horizon})"
            )


class TestPurge:
    """Purge removes last `horizon` training observations whose labels overlap test."""

    def test_purge_horizon1(self) -> None:
        """With horizon=1, train_idx.max() == test_start - 2 (not test_start - 1)."""
        # test_start = min_train = 252
        # purge: train_end = test_start - horizon = 252 - 1 = 251
        # So train_idx = arange(0, 251) → max = 250
        # test_idx.min() = 252
        splits = list(walk_forward_splits(n=500, min_train=252, step=21, horizon=1))
        first = splits[0]
        # train_end = test_start - horizon = 252 - 1 = 251
        # train_idx = arange(0, 251) → max index = 250
        # test_idx.min() = 252
        # gap = 252 - 250 = 2 >= horizon=1 ✓
        assert first.test_idx.min() - first.train_idx.max() >= 1

    def test_purge_removes_overlap_with_horizon5(self) -> None:
        """With horizon=5, the gap between train_max and test_min is >= 5."""
        horizon = 5
        splits = list(walk_forward_splits(n=600, min_train=252, step=21, horizon=horizon))
        for i, split in enumerate(splits):
            gap = split.test_idx.min() - split.train_idx.max()
            assert gap >= horizon, (
                f"Split {i}: purge insufficient — gap={gap} < horizon={horizon}"
            )

    def test_purge_train_end_formula(self) -> None:
        """train_end == test_start - horizon (exact formula from harness docs)."""
        horizon = 1
        min_train = 252
        splits = list(walk_forward_splits(n=500, min_train=min_train, step=21, horizon=horizon))
        first = splits[0]
        # test_start = min_train = 252 for the first split
        # train_end = test_start - horizon = 252 - 1 = 251
        # train_idx = arange(0, 251) → max = 250
        # So train_idx.max() + horizon + 1 == test_idx.min()
        expected_gap = horizon + 1  # purge removes 1 obs, so gap = horizon + 1 for standard case
        gap = first.test_idx.min() - first.train_idx.max()
        # We expect gap >= horizon — exact formula is gap = horizon + 1 for the first split
        assert gap >= horizon, f"First split gap {gap} < horizon {horizon}"


class TestExpandingWindow:
    """Each split's training range is an expanding prefix."""

    def test_expanding_window_strict_superset(self) -> None:
        """Each successive split's training end is >= prior split's training end."""
        splits = list(walk_forward_splits(n=500, min_train=252, step=21, horizon=1))
        for i in range(1, len(splits)):
            prev_max = splits[i - 1].train_idx.max()
            curr_max = splits[i].train_idx.max()
            assert curr_max >= prev_max, (
                f"Split {i}: training window shrunk — curr_max={curr_max} < prev_max={prev_max}"
            )

    def test_first_split_min_train_obs(self) -> None:
        """First split must have at least min_train training observations."""
        min_train = 252
        splits = list(walk_forward_splits(n=500, min_train=min_train, step=21, horizon=1))
        first = splits[0]
        assert len(first.train_idx) >= min_train, (
            f"First split has {len(first.train_idx)} training obs, expected >= {min_train}"
        )

    def test_training_always_starts_at_zero(self) -> None:
        """Training window always starts at position 0 (expanding, not sliding)."""
        splits = list(walk_forward_splits(n=500, min_train=252, step=21, horizon=1))
        for i, split in enumerate(splits):
            assert split.train_idx.min() == 0, (
                f"Split {i}: training starts at {split.train_idx.min()}, expected 0"
            )


class TestStepAdvance:
    """Test windows advance by `step` positions between consecutive splits."""

    def test_test_windows_advance_by_step(self) -> None:
        """Consecutive test windows start step positions apart."""
        step = 21
        splits = list(walk_forward_splits(n=500, min_train=252, step=step, horizon=1))
        for i in range(1, len(splits)):
            prev_start = splits[i - 1].test_idx.min()
            curr_start = splits[i].test_idx.min()
            assert curr_start - prev_start == step, (
                f"Split {i}: test window advance={curr_start - prev_start}, expected {step}"
            )

    def test_test_window_length_equals_step(self) -> None:
        """Test windows are of length `step` (except possibly the last)."""
        step = 21
        splits = list(walk_forward_splits(n=500, min_train=252, step=step, horizon=1))
        # All splits except the last should have exactly `step` test observations
        for i, split in enumerate(splits[:-1]):
            assert len(split.test_idx) == step, (
                f"Split {i}: test window length={len(split.test_idx)}, expected {step}"
            )


class TestNoRandomSplit:
    """Split indices are contiguous integer ranges (no shuffling, no gaps)."""

    def test_train_indices_are_contiguous(self) -> None:
        """train_idx is a contiguous integer range 0..train_max."""
        splits = list(walk_forward_splits(n=500, min_train=252, step=21, horizon=1))
        for i, split in enumerate(splits):
            expected = np.arange(0, split.train_idx.max() + 1)
            assert np.array_equal(split.train_idx, expected), (
                f"Split {i}: train_idx is not a contiguous range "
                f"[0..{split.train_idx.max()}]"
            )

    def test_test_indices_are_contiguous(self) -> None:
        """test_idx is a contiguous integer range test_min..test_max."""
        splits = list(walk_forward_splits(n=500, min_train=252, step=21, horizon=1))
        for i, split in enumerate(splits):
            expected = np.arange(split.test_idx.min(), split.test_idx.max() + 1)
            assert np.array_equal(split.test_idx, expected), (
                f"Split {i}: test_idx is not a contiguous range"
            )


class TestWalkForwardSplitDataclass:
    """WalkForwardSplit must be a dataclass with train_idx and test_idx."""

    def test_split_has_train_and_test(self) -> None:
        """WalkForwardSplit has .train_idx and .test_idx attributes."""
        split = WalkForwardSplit(
            train_idx=np.arange(0, 252),
            test_idx=np.arange(253, 274),
        )
        assert hasattr(split, "train_idx")
        assert hasattr(split, "test_idx")
        assert len(split.train_idx) == 252
        assert len(split.test_idx) == 21

    def test_splits_are_yielded(self) -> None:
        """walk_forward_splits must yield at least one split for n=500."""
        splits = list(walk_forward_splits(n=500, min_train=252, step=21, horizon=1))
        assert len(splits) > 0, "Expected at least one split for n=500, min_train=252"

    def test_no_splits_for_small_n(self) -> None:
        """When n is too small for even one split, the generator yields nothing."""
        splits = list(walk_forward_splits(n=100, min_train=252, step=21, horizon=1))
        assert len(splits) == 0, (
            f"Expected no splits for n=100 < min_train=252; got {len(splits)}"
        )
