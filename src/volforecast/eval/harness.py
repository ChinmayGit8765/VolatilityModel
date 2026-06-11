"""Reusable purged expanding-window walk-forward harness.

This module is a LIBRARY — it has no I/O, no script entry point, and no
side effects.  It is imported by:
- Phase 2: baseline evaluation (EWMA, GARCH, HAR-RV)
- Phase 3: ML model evaluation (LightGBM)
- Phase 4: champion/challenger promotion gate

INVARIANT ENFORCED BY THIS FUNCTION (and verified externally by test_harness.py):
    max(train_idx) < min(test_idx)
    min(test_idx) - max(train_idx) >= horizon   (embargo >= horizon)

This invariant is the project's methodological credibility centrepiece.
The unit test in tests/unit/test_harness.py is a portfolio talking point:
it FAILS if the harness is mutated to emit any non-temporal or under-embargoed
split — making it impossible to silently introduce data leakage.

Design rationale:
    The walk-forward split with purge and embargo follows the approach described
    in De Prado (2018) "Advances in Financial Machine Learning", adapted for:
    - Multi-asset panel (each asset's time series is indexed by integer position
      within that asset's frame — the harness is asset-agnostic)
    - Label horizon >= 1 (next-day for the primary target, multi-day for stability)
    - Expanding window: training always starts at position 0 to use all available
      history; not a rolling (sliding) window
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np


@dataclass
class WalkForwardSplit:
    """A single train/test split from the walk-forward harness.

    Attributes:
        train_idx: Integer positions (0-based) in the full time series that
                   form the purged training set.  Always a contiguous range
                   starting at 0.
        test_idx:  Integer positions for the test window.  Always a contiguous
                   range starting at least `horizon` positions after
                   train_idx.max().

    Invariant:
        max(train_idx) < min(test_idx)
        min(test_idx) - max(train_idx) >= horizon
    """

    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    n: int,
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
) -> Generator[WalkForwardSplit, None, None]:
    """Generate purged expanding-window walk-forward splits.

    Expanding window: training always starts at position 0.  Each split
    adds more history to the training set (a strict superset prefix of the
    previous split's training range).

    Purging: the last `horizon` observations in the naive training window
    are removed because their labels (computed at position t+horizon) overlap
    with the first test observation.  Specifically:
        train_end = test_start - horizon      # exclusive upper bound
        train_idx = np.arange(0, train_end)   # integer positions 0..train_end-1

    Embargo: the gap from the last training position to the first test position
    is >= horizon by construction:
        gap = test_start - train_end = horizon
        actual gap = test_start - (train_end - 1) = horizon + 1 >= horizon  ✓

    Enforced invariant (verified externally by tests/unit/test_harness.py):
        max(train_idx) < min(test_idx)
        min(test_idx) - max(train_idx) >= horizon

    Args:
        n:         Total number of observations (length of the time series).
        min_train: Minimum required training window size.  Default 252 (1 year
                   of daily data).  Splits are skipped if the purged training
                   set has fewer than min_train observations.
        step:      Number of positions the test window advances between splits.
                   Default 21 (~1 month of daily data).  Also the length of
                   each test window.
        horizon:   Label horizon in observations (default 1 for next-day).
                   Controls how many training observations are purged and the
                   minimum embargo gap.

    Yields:
        WalkForwardSplit instances with integer-position train_idx and test_idx.

    Notes:
        - If n < min_train + step, no splits are yielded.
        - The last split may have a shorter test window if n is not divisible
          by step (it is dropped if the test window would be empty).
        - This function is stateless and deterministic: given the same arguments,
          it always produces the same sequence of splits.

    Example:
        >>> splits = list(walk_forward_splits(n=500, min_train=252, step=21, horizon=1))
        >>> for split in splits:
        ...     assert split.train_idx.max() < split.test_idx.min()
        ...     assert split.test_idx.min() - split.train_idx.max() >= 1
    """
    test_start = min_train

    while test_start + step <= n:
        test_end = min(test_start + step, n)

        # Purge: remove the last `horizon` training observations because their
        # labels (squared return at t+horizon) overlap with the first test obs.
        # train_end is the exclusive upper bound for np.arange.
        train_end = test_start - horizon  # = test_start - horizon

        # purged_train = positions [0, 1, ..., train_end - 1]
        purged_train = np.arange(0, train_end)
        test = np.arange(test_start, test_end)

        # Only yield when the purged training set meets min_train and test is non-empty
        if len(purged_train) >= min_train and len(test) > 0:
            yield WalkForwardSplit(train_idx=purged_train, test_idx=test)

        test_start += step
