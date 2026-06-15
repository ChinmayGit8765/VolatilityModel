"""Pure read-only data loaders for the VolForecast observability dashboard.

All functions return neutral empty values (None or schema-shaped empty DataFrames)
when their data source is missing, unreachable, or malformed.  They NEVER raise
to the Streamlit caller, and NEVER write files, flip MLflow aliases, or trigger
any mutating operation.

Data-root resolution mirrors volforecast.serving.app._data_root():
  1. VOLFORECAST_DATA_ROOT env var (container default /data)
  2. project_root() / "data"  (local dev default)

Paths:
  FVR parquet:          {data_root}/monitoring/forecast_vs_realized.parquet
  prediction log:       {data_root}/predictions/predictions.parquet
  drift JSON files:     {data_root}/monitoring/{date_str}_drift.json

Imports: No streamlit dependency — this module is unit-testable offline.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from volforecast.monitoring.labeller import FVR_SCHEMA
from volforecast.monitoring.promotion import MODEL_NAME
from volforecast.serving.prediction_log import PREDICTION_LOG_SCHEMA

log = logging.getLogger(__name__)

__all__ = [
    "FVR_SCHEMA",
    "PREDICTION_LOG_SCHEMA",
    "data_root",
    "load_fvr",
    "latest_drift_summary",
    "champion_info",
    "prediction_stats",
    "api_health",
]

# ---------------------------------------------------------------------------
# Data-root resolution (mirrors serving.app._data_root)
# ---------------------------------------------------------------------------


def data_root() -> Path:
    """Return the data root directory.

    Resolution order:
    1. VOLFORECAST_DATA_ROOT env var (set by container to /data)
    2. project_root() / "data" (local dev default)
    """
    env_data = os.environ.get("VOLFORECAST_DATA_ROOT")
    if env_data:
        return Path(env_data)
    from volforecast.config import project_root

    return project_root() / "data"


# ---------------------------------------------------------------------------
# load_fvr
# ---------------------------------------------------------------------------


def load_fvr(data_root_path: Path) -> pd.DataFrame:
    """Load the forecast-vs-realized monitoring parquet.

    Args:
        data_root_path: Root data directory (e.g. project_root() / "data").

    Returns:
        DataFrame with FVR_SCHEMA columns.  Empty (with correct columns) when
        the parquet is absent, malformed, or unreadable — NEVER raises.
    """
    fvr_path = data_root_path / "monitoring" / "forecast_vs_realized.parquet"
    empty = pd.DataFrame(columns=FVR_SCHEMA)
    if not fvr_path.exists():
        return empty
    try:
        df = pd.read_parquet(fvr_path)
        # Ensure all schema columns are present
        for col in FVR_SCHEMA:
            if col not in df.columns:
                df[col] = None
        return df[FVR_SCHEMA]
    except Exception:  # noqa: BLE001
        log.exception("load_fvr: failed to read %s", fvr_path)
        return empty


# ---------------------------------------------------------------------------
# latest_drift_summary
# ---------------------------------------------------------------------------


def _find_dataset_drift(obj: Any) -> bool | None:
    """Recursively search for a 'dataset_drift' boolean in a nested dict/list.

    Returns the boolean value if found, else None.
    """
    if isinstance(obj, dict):
        if "dataset_drift" in obj:
            val = obj["dataset_drift"]
            if isinstance(val, bool):
                return val
        for v in obj.values():
            result = _find_dataset_drift(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_dataset_drift(item)
            if result is not None:
                return result
    return None


def _count_drifted_columns(obj: Any) -> int | None:
    """Count drifted columns by traversing nested dict/list.

    Looks for 'drift_detected': True entries and sums them.
    Returns None if no 'drift_detected' keys are found at all.
    """
    count = 0
    found_any = False

    def _recurse(o: Any) -> None:
        nonlocal count, found_any
        if isinstance(o, dict):
            if "drift_detected" in o:
                found_any = True
                if o["drift_detected"] is True:
                    count += 1
            for v in o.values():
                _recurse(v)
        elif isinstance(o, list):
            for item in o:
                _recurse(item)

    _recurse(obj)
    return count if found_any else None


def latest_drift_summary(monitoring_dir: Path) -> dict | None:
    """Load and summarise the latest drift report JSON.

    Searches for files matching ``{date_str}_drift.json`` in monitoring_dir,
    selects the one with the lexicographically latest date_str (ISO date
    strings sort lexicographically = chronologically), parses the Evidently
    result.dict() payload defensively, and returns a summary dict.

    Args:
        monitoring_dir: Path to the monitoring output directory
            (typically {data_root}/monitoring/).

    Returns:
        None when no *_drift.json files exist, the directory does not exist,
        or any read/parse error occurs.  Otherwise returns::

            {
                "date": str,            # e.g. "2026-06-12"
                "dataset_drift": bool | None,
                "n_drifted_columns": int | None,
                "html_path": str | None,  # path to sibling HTML if it exists
            }
    """
    if not monitoring_dir.exists():
        return None

    try:
        drift_files = sorted(monitoring_dir.glob("*_drift.json"))
        if not drift_files:
            return None

        # Latest by lexicographic sort of the filename (ISO date prefix sorts correctly)
        latest_file = drift_files[-1]

        # Extract date string: remove the _drift.json suffix
        date_str = latest_file.name.replace("_drift.json", "")

        try:
            payload = json.loads(latest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("latest_drift_summary: could not parse %s", latest_file)
            return None

        dataset_drift = _find_dataset_drift(payload)
        n_drifted = _count_drifted_columns(payload)

        html_candidate = monitoring_dir / f"{date_str}_drift.html"
        html_path = str(html_candidate) if html_candidate.exists() else None

        return {
            "date": date_str,
            "dataset_drift": dataset_drift,
            "n_drifted_columns": n_drifted,
            "html_path": html_path,
        }

    except Exception:  # noqa: BLE001
        log.exception("latest_drift_summary: unexpected error reading drift files")
        return None


# ---------------------------------------------------------------------------
# champion_info
# ---------------------------------------------------------------------------


def champion_info(tracking_uri: str) -> dict | None:
    """Query MLflow for the @champion model version metadata.

    Args:
        tracking_uri: MLflow tracking server URI (e.g. "http://mlflow-server:5000").

    Returns:
        None when MLflow is unreachable or the champion alias is unset.
        Otherwise::

            {
                "version": str,
                "run_id": str,
                "params": dict,
                "metrics": dict,
            }

        NEVER calls set_registered_model_alias — strictly read-only.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run = client.get_run(mv.run_id)
        return {
            "version": str(mv.version),
            "run_id": str(mv.run_id),
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
        }
    except Exception:  # noqa: BLE001
        log.debug("champion_info: MLflow unavailable or alias unset")
        return None


# ---------------------------------------------------------------------------
# prediction_stats
# ---------------------------------------------------------------------------


def prediction_stats(data_root_path: Path) -> dict:
    """Load prediction log stats.

    Args:
        data_root_path: Root data directory.

    Returns:
        Always returns a dict (never raises)::

            {
                "row_count": int,           # 0 when log absent
                "last_timestamp": pd.Timestamp | None,
                "latest_per_asset": dict,   # asset -> latest forecast_var
            }
    """
    pred_path = data_root_path / "predictions" / "predictions.parquet"
    empty: dict = {"row_count": 0, "last_timestamp": None, "latest_per_asset": {}}

    if not pred_path.exists():
        return empty

    try:
        df = pd.read_parquet(pred_path)
        if df.empty:
            return empty

        row_count = len(df)

        # Latest timestamp
        ts_col = "timestamp_utc"
        last_timestamp = None
        if ts_col in df.columns:
            last_timestamp = pd.to_datetime(df[ts_col]).max()

        # Latest forecast_var per asset (most recent row per asset)
        latest_per_asset: dict = {}
        if "asset" in df.columns and "forecast_var" in df.columns and ts_col in df.columns:
            df_sorted = df.sort_values(ts_col)
            for asset, grp in df_sorted.groupby("asset"):
                last_row = grp.iloc[-1]
                latest_per_asset[str(asset)] = float(last_row["forecast_var"])

        return {
            "row_count": row_count,
            "last_timestamp": last_timestamp,
            "latest_per_asset": latest_per_asset,
        }

    except Exception:  # noqa: BLE001
        log.exception("prediction_stats: failed to read %s", pred_path)
        return empty


# ---------------------------------------------------------------------------
# api_health
# ---------------------------------------------------------------------------


def api_health(api_base_url: str, timeout: float = 5.0) -> dict:
    """Probe the FastAPI /health endpoint.

    Uses stdlib urllib (no extra HTTP dependency).  Measures round-trip latency
    with time.perf_counter.

    Args:
        api_base_url: Base URL of the API service (e.g. "http://api:8000").
        timeout: Request timeout in seconds.

    Returns:
        Always returns a dict (never raises)::

            # On success:
            {"reachable": True, "status": str, "model_version": str,
             "alias": str, "latency_ms": float}

            # On any error:
            {"reachable": False, "error": str}
    """
    url = f"{api_base_url.rstrip('/')}/health"
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read()
            latency_ms = (time.perf_counter() - t0) * 1000.0
        data = json.loads(body)
        return {
            "reachable": True,
            "status": data.get("status", "unknown"),
            "model_version": data.get("model_version", "unknown"),
            "alias": data.get("alias", "unknown"),
            "latency_ms": latency_ms,
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("api_health: %s unreachable: %s", url, exc)
        return {"reachable": False, "error": str(exc)}
