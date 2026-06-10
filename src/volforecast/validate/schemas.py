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

# tz-aware datetime dtype for UTC-indexed OHLCV data.
# pandas 2.x defaults to microsecond ("us") resolution for DatetimeTZDtype, whereas
# the default pd.DatetimeTZDtype(tz="UTC") produces nanosecond ("ns") resolution.
# parquet files loaded by pandas 2.x have datetime64[us, UTC] dtype, so the schema
# must match "us" precision or Pandera raises a WRONG_DATATYPE error on clean data.
_UTC_DATETIME_DTYPE = pd.DatetimeTZDtype("us", tz="UTC")

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
# - Index: tz-aware UTC DatetimeIndex named "date" (coerce=True normalises ns→us precision)
# - Columns: open, high, low, close (>0, float64, non-null); volume (>=0, float64, non-null)
# - Dataframe-level OHLC consistency checks
# - strict=True: no extra columns permitted
#
# coerce=True is intentional: pandas 2.x parquet reads produce datetime64[us, UTC] while
# in-memory construction from exchange_calendars.sessions_in_range().tz_localize("UTC")
# produces datetime64[ns, UTC].  Without coerce=True, clean DataFrames fail with
# WRONG_DATATYPE on the index precision mismatch.  The canonical precision is "us".
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
    coerce=True,
    strict=True,
)


# Canonical equity OHLCV Pandera schema.
# Schema-level OHLC/dtype/positivity checks mirror crypto_ohlcv_schema.
# Calendar session validation (XNYS trading days) is applied separately via
# equity_session_check in checks.py — not embedded in the schema index check —
# because Pandera index checks run per-value and cannot express set-membership
# against an exchange calendar without a custom element-wise check that loses
# the structured failure-cases output.  The dispatcher (validate_asset in __init__.py)
# runs both the schema and the session check, so the full gate is enforced.
equity_ohlcv_schema = pa.DataFrameSchema(
    columns={
        "open": pa.Column(float, pa.Check.gt(0), nullable=False),
        "high": pa.Column(float, pa.Check.gt(0), nullable=False),
        "low": pa.Column(float, pa.Check.gt(0), nullable=False),
        "close": pa.Column(float, pa.Check.gt(0), nullable=False),
        "volume": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    index=pa.Index(_UTC_DATETIME_DTYPE, name="date"),
    checks=_OHLCV_CONSISTENCY_CHECKS,
    coerce=True,   # normalise ns→us precision (see note above crypto_ohlcv_schema)
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
