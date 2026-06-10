"""Pandera validation schemas for OHLCV data.

All schemas use the pandera.pandas namespace (import pandera.pandas as pa).
The top-level `import pandera as pa` is deprecated since 0.29 and must not be used.

Validation is always called with lazy=True to collect all errors before raising,
enabling quarantine reporting (all failure cases written to CSV before re-raise).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

# tz-aware datetime dtype for UTC-indexed OHLCV data
_UTC_DATETIME_DTYPE = pd.DatetimeTZDtype(tz="UTC")

# OHLC consistency dataframe-level checks (per RESEARCH "Canonical OHLCV schema check")
_OHLCV_CONSISTENCY_CHECKS = [
    pa.Check(
        lambda df: (df["high"] >= df["low"]).all(),
        element_wise=False,
        error="OHLC violation: high < low",
    ),
    pa.Check(
        lambda df: (df["high"] >= df["open"]).all(),
        element_wise=False,
        error="OHLC violation: high < open",
    ),
    pa.Check(
        lambda df: (df["high"] >= df["close"]).all(),
        element_wise=False,
        error="OHLC violation: high < close",
    ),
    pa.Check(
        lambda df: (df["low"] <= df["open"]).all(),
        element_wise=False,
        error="OHLC violation: low > open",
    ),
    pa.Check(
        lambda df: (df["low"] <= df["close"]).all(),
        element_wise=False,
        error="OHLC violation: low > close",
    ),
]

# Canonical crypto OHLCV Pandera schema.
# - Index: tz-aware UTC DatetimeIndex named "date"
# - Columns: open, high, low, close (>0, float64, non-null); volume (>=0, float64, non-null)
# - Dataframe-level OHLC consistency checks
# - strict=True: no extra columns permitted
crypto_ohlcv_schema = pa.DataFrameSchema(
    columns={
        "open": pa.Column(float, pa.Check.gt(0), nullable=False),
        "high": pa.Column(float, pa.Check.gt(0), nullable=False),
        "low": pa.Column(float, pa.Check.gt(0), nullable=False),
        "close": pa.Column(float, pa.Check.gt(0), nullable=False),
        "volume": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    index=pa.Index(_UTC_DATETIME_DTYPE, name="date"),
    checks=_OHLCV_CONSISTENCY_CHECKS,
    coerce=False,
    strict=True,
)


def validate_and_quarantine(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    quarantine_path: Path | str,
) -> pd.DataFrame:
    """Validate a DataFrame against a schema; on failure write quarantine CSV and re-raise.

    This is the hard-fail gate: validation errors are never silently swallowed.
    All failure cases are written to quarantine_path as a CSV for inspection, then the
    SchemaErrors exception is re-raised so the caller's pipeline halts.

    Args:
        df: DataFrame to validate.
        schema: Pandera DataFrameSchema to validate against.
        quarantine_path: File path to write the quarantine CSV on failure. Parent directory
                         is created automatically.

    Returns:
        The validated DataFrame (unchanged, returned for chaining) on success.

    Raises:
        pandera.errors.SchemaErrors: On validation failure (after writing quarantine CSV).
    """
    quarantine_path = Path(quarantine_path)

    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        # exc.failure_cases is a DataFrame: check, column, failure_case, index
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        exc.failure_cases.to_csv(quarantine_path, index=False)
        raise
