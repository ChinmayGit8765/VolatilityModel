"""Calendar-aware OHLCV data quality checks.

Provides four check functions that return a structured CheckResult (passed bool +
offending index list) so the dispatcher can aggregate failures into a quarantine report.

Checks:
- crypto_gap_check: 24/7 continuous daily calendar — any missing day is a gap error.
- equity_session_check: XNYS session calendar — extra rows (fabricated weekend/holiday)
  and missing expected sessions are both reported as failures.
- stale_row_check: flags runs of consecutive identical close values (stuck/stale feed).
- ohlc_consistency_check: high >= low/open/close and low <= open/close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import exchange_calendars as xcals
import pandas as pd


@dataclass
class CheckResult:
    """Result of a single data-quality check.

    Attributes:
        passed: True if the check passed with no violations.
        offending_index: Index labels of rows (or missing rows) that violated the check.
            For gap checks this includes the *missing* timestamps; for extra-row checks
            it includes the fabricated row timestamps; for OHLC/stale checks it includes
            the row timestamps that violated the rule.
        reason: Short human-readable description of the violation (empty string if passed).
    """

    passed: bool
    offending_index: list[Any] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# crypto_gap_check
# ---------------------------------------------------------------------------


def crypto_gap_check(df: pd.DataFrame) -> CheckResult:
    """Check that a crypto OHLCV DataFrame has no missing days.

    Crypto trades 24/7; the expected calendar is a continuous daily date_range
    from the first to the last index value.  Any missing day is a gap error.

    Args:
        df: DataFrame with tz-aware UTC DatetimeIndex named "date".

    Returns:
        CheckResult with passed=True if no days are missing, or passed=False
        with the missing timestamps in offending_index.
    """
    if df.empty:
        return CheckResult(passed=True)

    start = df.index.normalize().min()
    end = df.index.normalize().max()
    expected = pd.date_range(start=start, end=end, freq="D")
    # Normalize actual index to midnight UTC for comparison
    actual = df.index.normalize()

    # expected is tz-naive if start/end are tz-naive; align tz
    if actual.tzinfo is not None and expected.tzinfo is None:
        expected = expected.tz_localize("UTC")
    elif actual.tzinfo is None and expected.tzinfo is not None:
        expected = expected.tz_localize(None)

    missing = expected.difference(actual)

    if len(missing) == 0:
        return CheckResult(passed=True)

    return CheckResult(
        passed=False,
        offending_index=missing.tolist(),
        reason=f"Missing {len(missing)} day(s) in crypto 24/7 calendar: {missing.tolist()[:5]}",
    )


# ---------------------------------------------------------------------------
# equity_session_check
# ---------------------------------------------------------------------------


def equity_session_check(df: pd.DataFrame) -> CheckResult:
    """Check that an equity OHLCV DataFrame contains only valid XNYS trading sessions.

    Two failure modes are checked and merged into one result:
    1. Extra rows: index values that are NOT in the XNYS session calendar
       (fabricated weekend/holiday bars — RESEARCH Pitfall 2).
    2. Missing rows: expected XNYS sessions that are absent from the index.

    The expected session range is derived from the min/max of the DataFrame index.

    Args:
        df: DataFrame with tz-aware UTC DatetimeIndex named "date".

    Returns:
        CheckResult with passed=True if the index exactly matches XNYS sessions in
        the min–max range, or passed=False with all offending timestamps.
    """
    if df.empty:
        return CheckResult(passed=True)

    xnys = xcals.get_calendar("XNYS")

    # sessions_in_range requires tz-naive dates (exchange_calendars 4.x);
    # strip timezone from the UTC-aware index before calling.
    start = df.index.normalize().min().tz_localize(None)
    end = df.index.normalize().max().tz_localize(None)

    # sessions_in_range returns a tz-naive DatetimeIndex (exchange_calendars 4.x)
    expected_sessions = xnys.sessions_in_range(start, end)

    # Align timezone: normalize actual index to tz-naive for comparison
    actual_tz_naive = df.index.normalize().tz_localize(None)

    extra_rows = actual_tz_naive.difference(expected_sessions)
    missing_sessions = expected_sessions.difference(actual_tz_naive)

    offending: list[Any] = []
    reasons: list[str] = []

    if len(extra_rows) > 0:
        offending.extend(extra_rows.tolist())
        reasons.append(f"{len(extra_rows)} fabricated row(s) not in XNYS sessions")

    if len(missing_sessions) > 0:
        offending.extend(missing_sessions.tolist())
        reasons.append(f"{len(missing_sessions)} missing XNYS session(s)")

    if offending:
        return CheckResult(
            passed=False,
            offending_index=offending,
            reason="; ".join(reasons),
        )

    return CheckResult(passed=True)


# ---------------------------------------------------------------------------
# stale_row_check
# ---------------------------------------------------------------------------


def stale_row_check(df: pd.DataFrame, max_run: int = 5) -> CheckResult:
    """Flag runs of more than `max_run` CONSECUTIVE identical close values.

    A stale / stuck feed manifests as the same close value repeated on
    consecutive rows.  Non-consecutive duplicate closes are normal for real
    price series — equity closes are quantized to cents, so collisions across
    multi-year history are expected and must NOT be flagged.  (An earlier
    implementation flagged *global* duplicates against a 95 % uniqueness
    threshold, which falsely quarantined clean real-world equity data.)

    Args:
        df: DataFrame with a "close" column.
        max_run: Longest tolerated run of consecutive identical closes
                 (default 5 — a 6th consecutive repeat fails the check).

    Returns:
        CheckResult with passed=True if no run of consecutive identical closes
        exceeds `max_run`, or passed=False with the offending row index values
        (the rows beyond position `max_run` within each too-long run).
    """
    if df.empty or "close" not in df.columns:
        return CheckResult(passed=True)

    # A stale feed produces runs of consecutive identical closes.
    same_as_prev = df["close"].diff() == 0
    run_id = (~same_as_prev).cumsum()
    run_lengths = same_as_prev.groupby(run_id).cumsum() + 1
    offending_mask = run_lengths > max_run

    if not offending_mask.any():
        return CheckResult(passed=True)

    return CheckResult(
        passed=False,
        offending_index=df.index[offending_mask].tolist(),
        reason=f"Stale feed: close repeated for more than {max_run} consecutive rows",
    )


# ---------------------------------------------------------------------------
# ohlc_consistency_check
# ---------------------------------------------------------------------------


def ohlc_consistency_check(df: pd.DataFrame) -> CheckResult:
    """Check OHLC consistency: high >= low/open/close and low <= open/close.

    Any row violating these invariants is recorded in offending_index.

    Checks applied:
    - high >= low
    - high >= open
    - high >= close
    - low <= open
    - low <= close

    Args:
        df: DataFrame with columns open, high, low, close.

    Returns:
        CheckResult with passed=True if all rows satisfy OHLC consistency,
        or passed=False with the offending row index values.
    """
    if df.empty:
        return CheckResult(passed=True)

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        return CheckResult(
            passed=False,
            offending_index=[],
            reason=f"Missing required OHLC columns: {missing}",
        )

    violations = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    if not violations.any():
        return CheckResult(passed=True)

    offending = df.index[violations].tolist()
    violation_details = []
    if (df["high"] < df["low"]).any():
        violation_details.append("high < low")
    if (df["high"] < df["open"]).any():
        violation_details.append("high < open")
    if (df["high"] < df["close"]).any():
        violation_details.append("high < close")
    if (df["low"] > df["open"]).any():
        violation_details.append("low > open")
    if (df["low"] > df["close"]).any():
        violation_details.append("low > close")

    return CheckResult(
        passed=False,
        offending_index=offending,
        reason=f"OHLC violations in {len(offending)} row(s): {', '.join(violation_details)}",
    )
