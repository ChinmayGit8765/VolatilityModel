"""Validate module for VolForecast — Pandera schema gates for OHLCV data.

Public API
----------
validate_asset(df, asset_class, quarantine_dir)
    Single dispatch entry point.  Runs Pandera schema validation + calendar-aware
    gap/session checks + stale-row + OHLC-consistency checks for the given asset class.
    On ANY failure writes a quarantine report to quarantine_dir and raises.
    On success returns the validated DataFrame.

validate_and_quarantine(df, schema, quarantine_path)
    Low-level Pandera schema validator.  On failure writes quarantine CSV and re-raises.

crypto_ohlcv_schema
    Pandera DataFrameSchema for crypto OHLCV data (open/high/low/close > 0, volume >= 0,
    tz-aware UTC DatetimeIndex named "date", strict, OHLC consistency).

equity_ohlcv_schema
    Pandera DataFrameSchema for equity OHLCV data (same column contract as crypto;
    calendar session validation applied separately via equity_session_check).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pandera.errors import SchemaErrors

from volforecast.validate.checks import (
    CheckResult,
    crypto_gap_check,
    equity_session_check,
    ohlc_consistency_check,
    stale_row_check,
)
from volforecast.validate.schemas import (
    crypto_ohlcv_schema,
    equity_ohlcv_schema,
    validate_and_quarantine,
)

__all__ = [
    "validate_asset",
    "validate_and_quarantine",
    "crypto_ohlcv_schema",
    "equity_ohlcv_schema",
]

# Map asset_class strings to (schema, calendar_check) pairs.
_ASSET_CLASS_SCHEMA = {
    "crypto": crypto_ohlcv_schema,
    "equity": equity_ohlcv_schema,
}


class ValidationError(Exception):
    """Raised by validate_asset when one or more data-quality checks fail.

    The quarantine report has already been written to quarantine_dir before this
    is raised, so the pipeline can halt cleanly with a record of every violation.
    """


def validate_asset(
    df: pd.DataFrame,
    asset_class: Literal["crypto", "equity"],
    quarantine_dir: Path | str,
) -> pd.DataFrame:
    """Gate OHLCV data through Pandera + calendar-aware quality checks.

    Runs in order:
    1. Pandera schema validation (dtype, positivity, OHLC consistency, index type).
    2. Calendar-aware gap/session check:
       - crypto → crypto_gap_check (continuous 24/7 date_range)
       - equity → equity_session_check (XNYS sessions_in_range)
    3. stale_row_check (repeated-close detection).
    4. ohlc_consistency_check (redundant safety net for non-schema paths).

    On ANY failure all violations are aggregated into a single quarantine CSV
    written to ``quarantine_dir/{asset_class}_{timestamp}.csv``, then a
    ``ValidationError`` is raised (fails closed — no invalid data reaches downstream).

    On success returns the Pandera-validated DataFrame (unchanged value-wise).

    Args:
        df: DataFrame to validate.  Must have a tz-aware UTC DatetimeIndex named "date"
            and float64 columns open/high/low/close/volume.
        asset_class: ``"crypto"`` or ``"equity"``.
        quarantine_dir: Directory to write the quarantine report on failure.
            Created automatically if it does not exist.

    Returns:
        The validated DataFrame on success.

    Raises:
        ValueError: If asset_class is not recognised.
        ValidationError: On any validation failure (after writing quarantine CSV).
        pandera.errors.SchemaErrors: Propagated from validate_and_quarantine on
            Pandera-level failures (also after writing quarantine CSV).
    """
    if asset_class not in _ASSET_CLASS_SCHEMA:
        raise ValueError(
            f"Unknown asset_class '{asset_class}'. Expected one of: {list(_ASSET_CLASS_SCHEMA)}"
        )

    schema = _ASSET_CLASS_SCHEMA[asset_class]
    quarantine_dir = Path(quarantine_dir)
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine_path = quarantine_dir / f"{asset_class}_{timestamp}.csv"

    # ------------------------------------------------------------------
    # Step 1: Run all non-Pandera checks first (calendar + stale + OHLC).
    # Collect all non-Pandera failures before attempting Pandera validation.
    # ------------------------------------------------------------------
    failures: list[dict] = []  # list of {check, reason, offending_index} dicts

    # Calendar-aware gap / session check
    if asset_class == "crypto":
        cal_result: CheckResult = crypto_gap_check(df)
        cal_check_name = "crypto_gap_check"
    else:
        cal_result = equity_session_check(df)
        cal_check_name = "equity_session_check"

    if not cal_result.passed:
        for idx_val in cal_result.offending_index:
            failures.append(
                {
                    "check": cal_check_name,
                    "reason": cal_result.reason,
                    "offending_index": str(idx_val),
                }
            )

    # Stale-row check
    stale_result = stale_row_check(df)
    if not stale_result.passed:
        for idx_val in stale_result.offending_index:
            failures.append(
                {
                    "check": "stale_row_check",
                    "reason": stale_result.reason,
                    "offending_index": str(idx_val),
                }
            )

    # OHLC-consistency check (belt-and-suspenders on top of Pandera)
    ohlc_result = ohlc_consistency_check(df)
    if not ohlc_result.passed:
        for idx_val in ohlc_result.offending_index:
            failures.append(
                {
                    "check": "ohlc_consistency_check",
                    "reason": ohlc_result.reason,
                    "offending_index": str(idx_val),
                }
            )

    # ------------------------------------------------------------------
    # Step 2: Pandera schema validation.
    # Run even if non-Pandera checks failed so we collect all failures.
    # ------------------------------------------------------------------
    pandera_exc: SchemaErrors | None = None
    validated_df: pd.DataFrame | None = None
    try:
        validated_df = schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        pandera_exc = exc
        # Merge Pandera failure cases into the quarantine report
        fc = exc.failure_cases
        for _, row in fc.iterrows():
            failures.append(
                {
                    "check": f"pandera:{row.get('check', 'unknown')}",
                    "reason": str(row.get("check", "schema_violation")),
                    "offending_index": str(row.get("index", "")),
                }
            )

    # ------------------------------------------------------------------
    # Step 3: If any failures, write quarantine report and raise.
    # ------------------------------------------------------------------
    if failures:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        quarantine_df = pd.DataFrame(failures, columns=["check", "reason", "offending_index"])
        quarantine_df.to_csv(quarantine_path, index=False)

        if pandera_exc is not None:
            # Re-raise the Pandera exception to preserve its type for callers
            # that specifically catch SchemaErrors, but wrap it so callers
            # catching the broader ValidationError also get it.
            raise pandera_exc

        raise ValidationError(
            f"validate_asset({asset_class!r}): {len(failures)} check failure(s). "
            f"Quarantine report: {quarantine_path}"
        )

    # All checks passed — return the Pandera-validated DataFrame.
    assert validated_df is not None  # guaranteed by no-exception path above
    return validated_df
