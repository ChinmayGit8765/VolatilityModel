"""Atomic parquet append for the VolForecast prediction log.

Phase 4 contract — exact column schema:
    timestamp_utc, asset, horizon, forecast_var, model_version, alias

Every served forecast appends a row to ``data/predictions/predictions.parquet``
(or the path set via ``PREDICTION_LOG_PATH`` env var).

The write is atomic: rows are written to a ``.tmp.parquet`` sidecar then
renamed with ``os.replace()`` (POSIX-atomic; also atomic on Windows NTFS
within the same volume).  A module-level ``threading.Lock`` serialises
concurrent appends within a single process (uvicorn single-replica).

Exports:
    PREDICTION_LOG_SCHEMA  — list of column names in Phase-4 contract order
    append_predictions     — atomic append function
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Phase-4 contract — column names in exact order
# ---------------------------------------------------------------------------

#: Exact Phase-4 monitoring contract columns, in this order.
PREDICTION_LOG_SCHEMA: list[str] = [
    "timestamp_utc",
    "asset",
    "horizon",
    "forecast_var",
    "model_version",
    "alias",
]

# ---------------------------------------------------------------------------
# Configurable log path
# ---------------------------------------------------------------------------

#: Default log path (relative to CWD or overridden via PREDICTION_LOG_PATH).
#: The container sets PREDICTION_LOG_PATH=/data/predictions/predictions.parquet.
_DEFAULT_LOG_PATH = Path("data") / "predictions" / "predictions.parquet"

# Resolved at import time so tests can monkeypatch PREDICTION_LOG_PATH before
# importing this module (or reload after monkeypatching).
_LOG_PATH: Path = Path(os.environ.get("PREDICTION_LOG_PATH", str(_DEFAULT_LOG_PATH)))

# ---------------------------------------------------------------------------
# Module-level lock — guards concurrent appends within one process
# ---------------------------------------------------------------------------

_log_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_predictions(new_rows: pd.DataFrame) -> None:
    """Atomically append ``new_rows`` to the prediction log parquet.

    Creates the file and parent directories on first call.  On subsequent
    calls, reads the existing file, concatenates, and writes atomically via a
    ``.tmp.parquet`` sidecar → ``os.replace()``.

    Args:
        new_rows: DataFrame with columns matching ``PREDICTION_LOG_SCHEMA``
                  (in any column order; columns are reordered to match the
                  schema before writing).

    Raises:
        KeyError: If ``new_rows`` is missing a required column.
    """
    # Reorder to canonical schema (raises KeyError for missing columns)
    ordered = new_rows[PREDICTION_LOG_SCHEMA].copy()

    with _log_lock:
        log_path = _LOG_PATH  # read module-level path inside lock for reload safety
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if log_path.exists():
            existing = pd.read_parquet(log_path)
            combined = pd.concat([existing, ordered], ignore_index=True)
        else:
            combined = ordered.reset_index(drop=True)

        tmp_path = log_path.with_suffix(".tmp.parquet")
        combined.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, log_path)  # atomic rename (POSIX + Windows NTFS)
