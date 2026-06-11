"""Cross-asset feature alignment for VolForecast.

This module handles the calendar mismatch between 24/7 crypto assets (BTC, ETH)
and session-gapped equity assets (SPY, AAPL, MSFT) using a backward as-of join
with a documented staleness cap.

Staleness rule (FEAT-05):
    When a cross-asset source feature (e.g., BTC realized variance) is joined
    onto a target asset (e.g., SPY) the most recent prior source value is used
    (direction="backward").  If the nearest prior source row is more than 3
    calendar days before the target date, the joined feature is NaN — it is
    never forward-filled further than 3 days.

    This rule is intentionally conservative: crypto trades on weekends but
    equity markets do not.  A Monday equity row can use Friday/Saturday/Sunday
    BTC data (gap <= 3 days), but a long holiday closure (> 3 days) yields NaN
    to avoid using stale cross-asset information.

Sort safety (Pitfall 6):
    ``pd.merge_asof`` does NOT sort its inputs.  Passing an unsorted frame
    silently produces wrong results.  This module always sorts both frames
    ascending by "date" before the call and asserts monotonic_increasing
    afterwards.

References:
    - pandas.pydata.org/docs/reference/api/pandas.merge_asof.html
    - Pitfall 6 (02-RESEARCH.md): unsorted merge_asof produces wrong results
"""

from __future__ import annotations

import pandas as pd

#: Maximum staleness for a cross-asset feature.  Source rows older than 3
#: calendar days from the target date become NaN.  This constant is the single
#: source of truth across the project — importable by any feature consumer.
MAX_CROSS_ASSET_STALENESS: pd.Timedelta = pd.Timedelta("3D")


def as_of_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    feature_cols: list[str],
    suffix: str = "_xasset",
) -> pd.DataFrame:
    """Backward as-of join with a 3-calendar-day staleness cap.

    Attaches the most recent prior value of each ``feature_col`` from ``right``
    to every row in ``left``.  If the nearest prior ``right`` row is more than
    ``MAX_CROSS_ASSET_STALENESS`` (3 calendar days) before the ``left`` row's
    date, the joined value is NaN.

    Both input frames are sorted ascending by their DatetimeIndex before the
    merge_asof call (Pitfall 6: merge_asof does not sort).  The original
    sort order of ``left`` is not preserved in the output — the output is always
    in ascending date order (same as a time-series should be).

    Args:
        left: Target-asset DataFrame with a tz-aware UTC DatetimeIndex named
              "date".  Existing columns are preserved in the output.
        right: Source-asset DataFrame with a tz-aware UTC DatetimeIndex named
               "date".  Must contain all columns in ``feature_cols``.
        feature_cols: List of column names in ``right`` to attach to ``left``.
        suffix: String appended to each joined column name to avoid collisions.
                Default ``"_xasset"``.

    Returns:
        A new DataFrame with all columns from ``left`` plus
        ``{col}{suffix}`` columns for each column in ``feature_cols``.
        The index is the UTC DatetimeIndex from ``left`` (ascending).
        Joined columns are NaN when the nearest prior right row is more than 3
        calendar days stale.

    Raises:
        KeyError: If a column in ``feature_cols`` does not exist in ``right``.
    """
    # --- Reset index so merge_asof operates on a "date" column ---
    left_reset = left.reset_index()  # "date" becomes a regular column
    right_reset = right[feature_cols].reset_index()  # keep only feature cols + date

    # --- Sort ascending by "date" (Pitfall 6: merge_asof requires sorted inputs) ---
    left_sorted = left_reset.sort_values("date").reset_index(drop=True)
    right_sorted = right_reset.sort_values("date").reset_index(drop=True)

    # Assertion: both frames are monotonically increasing before merge_asof
    assert left_sorted["date"].is_monotonic_increasing, (
        "left frame 'date' column is not monotonically increasing after sort — "
        "check for duplicate or NaT index values"
    )
    assert right_sorted["date"].is_monotonic_increasing, (
        "right frame 'date' column is not monotonically increasing after sort — "
        "check for duplicate or NaT index values"
    )

    # --- Build column rename map to apply suffix to joined feature cols ---
    # merge_asof suffixes parameter only triggers on column name collisions;
    # we rename manually to always apply the suffix.
    rename_map = {col: f"{col}{suffix}" for col in feature_cols}

    # --- Perform the backward as-of join ---
    merged = pd.merge_asof(
        left_sorted,
        right_sorted[["date"] + feature_cols],
        on="date",
        direction="backward",
        tolerance=MAX_CROSS_ASSET_STALENESS,
    )

    # --- Apply suffix to joined columns ---
    merged = merged.rename(columns=rename_map)

    # --- Restore UTC DatetimeIndex ---
    merged = merged.set_index("date")
    merged.index = pd.DatetimeIndex(merged.index, tz="UTC", name="date")

    return merged
