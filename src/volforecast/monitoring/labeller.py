"""Feedback-loop labeller for VolForecast.

Joins arrived realized volatility against the logged prediction-log forecasts
and writes (or appends to) an idempotent ``data/monitoring/forecast_vs_realized.parquet``
table.  The same table also contains GARCH(1,1) live-forecast rows so that the
Wave-2 performance monitor can compare champion vs GARCH rolling QLIKE.

Idempotency contract
--------------------
The labeller is keyed on ``LABEL_KEY = ["asset", "as_of_date", "model_alias"]``.
Re-running the labeller never duplicates rows — ``drop_duplicates(keep="first")``
preserves original rows and discards exact-key re-submissions.

Calendar correctness
--------------------
Equity weekend / holiday rows are simply absent: ``compute_target`` returns NaN
for dates where no t+1 close exists, and those NaN rows are dropped before
writing.  No fabricated equity-weekend entries are ever inserted.

NaN target guard (Pitfall 4 from target.py)
--------------------------------------------
``realized_var`` NaN values are NEVER zero-filled.  Rows with a NaN realized
target are dropped entirely.  A zero realized variance would corrupt the
QLIKE-fed monitoring table.

``arch`` is imported lazily (inside functions) — not at module load time.  This
keeps the feature pipeline free of the GARCH runtime overhead unless the
labeller is explicitly called.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from volforecast.features.estimators import log_returns as compute_log_returns
from volforecast.features.target import compute_target
from volforecast.models.garch import GARCH, GarchFitError
from volforecast.serving.prediction_log import PREDICTION_LOG_SCHEMA

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Column order for the forecast-vs-realized monitoring parquet.
FVR_SCHEMA: list[str] = [
    "as_of_date",  # date the forecast was made for (date, not datetime)
    "asset",  # str slug, e.g. "BTC-USD"
    "horizon",  # int, always 1 in v1
    "model_version",  # str, e.g. "3" (champion) or "garch_1_1" (GARCH sentinel)
    "model_alias",  # str, "champion" or "garch_baseline"
    "forecast_var",  # float, decimal daily variance (~1e-4..1e-3)
    "realized_var",  # float, decimal daily variance (compute_target output)
]

#: Composite key for idempotency.  Allows champion and garch_baseline rows for
#: the same (asset, as_of_date) to coexist as separate rows.
LABEL_KEY: list[str] = ["asset", "as_of_date", "model_alias"]

#: Sentinel model_version for GARCH baseline rows (not an MLflow version number).
GARCH_MODEL_VERSION: str = "garch_1_1"

#: model_alias value written for GARCH baseline rows.
GARCH_ALIAS: str = "garch_baseline"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _realized_var_for_asset(asset_cfg: dict, data_root: Path) -> pd.Series:
    """Load processed data and return the realized-variance Series for an asset.

    Uses ``compute_target`` exclusively (no re-implementation) so the target
    definition stays in a single place.  Dates where ``compute_target`` returns
    NaN (no t+1 close) are dropped — those dates are NOT labelable.

    Args:
        asset_cfg: Asset dict with keys ``symbol``, ``asset_class``.
        data_root: Root of the data directory (e.g. ``project_root() / "data"``).

    Returns:
        pd.Series of decimal realized variance indexed by tz-aware UTC date,
        with NaN rows already dropped.
    """
    from volforecast.config import processed_path

    path = processed_path(asset_cfg, data_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed parquet for {asset_cfg['symbol']} not found at {path}. "
            "Run Phase 1 ingestion + validation first."
        )
    df = pd.read_parquet(path)
    rv = compute_target(df["close"])
    return rv.dropna()


# ---------------------------------------------------------------------------
# Champion forecast labelling
# ---------------------------------------------------------------------------


def label_champion_forecasts(predictions: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Join prediction-log rows to arrived realized variance.

    Derives the forecast's effective ``as_of_date`` from the prediction row.
    The FastAPI serving layer records each forecast as a row in the prediction
    log with ``timestamp_utc`` (the wall-clock serve time) and an ``alias`` of
    ``"champion"``.  The prediction targets realized variance for the *next*
    trading day after the close date that was the latest available at serve
    time.  In practice, for daily forecasts the as-of date is the close date
    the features were built on.

    Derivation (documented here as required by the plan):
        ``as_of_date`` is derived from ``timestamp_utc`` by converting to UTC
        date and subtracting one day.  This matches the convention that a
        forecast served on date D is built from features up to D-1 close and
        targets RV on day D (i.e. ``compute_target[D-1] = RV[D]``).  The
        prediction log ``timestamp_utc`` is the serve wall-clock time (D); so
        ``as_of_date = timestamp_utc.date() - 1 day``.

    Rows whose ``as_of_date`` has no realized label (forward close not yet
    arrived) are silently dropped — they are not yet labelable.

    Args:
        predictions: DataFrame conforming to ``PREDICTION_LOG_SCHEMA``.
            Expected columns: timestamp_utc, asset, horizon, forecast_var,
            model_version, alias.
        data_root: Root of the data directory.

    Returns:
        DataFrame with columns matching ``FVR_SCHEMA`` for all labelable rows.
    """
    from volforecast.config import load_assets, symbol_slug

    # Validate required columns (T-04-01 input validation) against the
    # canonical serving-side contract — never a hand-copied duplicate that
    # can drift out of sync with PREDICTION_LOG_SCHEMA (audit WARNING fix).
    required = set(PREDICTION_LOG_SCHEMA)
    missing_cols = required - set(predictions.columns)
    if missing_cols:
        raise ValueError(f"prediction log is missing columns: {missing_cols}")

    # Assert dtypes (T-04-01)
    if not pd.api.types.is_float_dtype(predictions["forecast_var"]):
        raise TypeError(
            f"forecast_var must be float dtype, got {predictions['forecast_var'].dtype}"
        )

    if predictions.empty:
        return pd.DataFrame(columns=FVR_SCHEMA)

    # Derive as_of_date: serve wall-clock is date D → as_of D-1
    ts = pd.to_datetime(predictions["timestamp_utc"], utc=True)
    as_of_dates = (ts.dt.date - pd.Timedelta(days=1)).apply(lambda d: pd.Timestamp(d, tz="UTC"))
    predictions = predictions.copy()
    predictions["as_of_date"] = as_of_dates

    # Build realized variance map per asset
    assets_cfg = load_assets(data_root.parent / "config" / "assets.yaml")
    asset_slug_map = {symbol_slug(a["symbol"]): a for a in assets_cfg}

    rows: list[dict] = []
    for asset_slug, grp in predictions.groupby("asset"):
        if asset_slug not in asset_slug_map:
            log.warning("asset %r not in assets.yaml — skipping", asset_slug)
            continue
        asset_cfg = asset_slug_map[asset_slug]
        try:
            rv_series = _realized_var_for_asset(asset_cfg, data_root)
        except FileNotFoundError:
            log.warning("processed parquet missing for %s — skipping champion labels", asset_slug)
            continue

        # rv_series is indexed by tz-aware UTC Timestamp
        rv_index_set = set(rv_series.index)

        for _, pred_row in grp.iterrows():
            aod = pred_row["as_of_date"]
            if aod not in rv_index_set:
                # Forward close not yet arrived — skip this row
                continue
            rows.append(
                {
                    "as_of_date": aod,
                    "asset": asset_slug,
                    "horizon": int(pred_row["horizon"]),
                    "model_version": str(pred_row["model_version"]),
                    "model_alias": str(pred_row["alias"]),
                    "forecast_var": float(pred_row["forecast_var"]),
                    "realized_var": float(rv_series.loc[aod]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=FVR_SCHEMA)
    return pd.DataFrame(rows)[FVR_SCHEMA]


# ---------------------------------------------------------------------------
# GARCH baseline live-forecast rows
# ---------------------------------------------------------------------------


def garch_live_forecasts(
    asset_cfg: dict,
    data_root: Path,
    label_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Compute GARCH(1,1) live one-step-ahead variance forecasts restricted to label dates.

    Uses ``GARCH(min_train=252, step=21).forecast_path`` so the alignment
    matches ``compute_target``: ``forecast[pos]`` targets ``RV[pos+1]``
    (identical to ``compute_target[pos] = RV[pos+1]``).

    The GARCH model is fitted on decimal log returns (via
    ``features.estimators.log_returns``), matching the GARCH module's input
    contract.  The output is decimal variance (already de-scaled by
    ``GARCH_SCALE**2`` inside the GARCH class) — no further de-scaling is
    applied here.

    If the GARCH fit raises a ``GarchFitError`` the labeller logs a warning
    and returns an empty Series for this asset.  The labeller must never crash
    on a single bad GARCH window.

    Args:
        asset_cfg: Asset dict with keys ``symbol``, ``asset_class``.
        data_root: Root of the data directory.
        label_dates: DatetimeIndex of labelable dates to restrict output to.

    Returns:
        pd.Series of decimal GARCH variance forecasts indexed by tz-aware UTC
        date, restricted to ``label_dates``.  Empty Series on fit failure.
    """
    from volforecast.config import processed_path

    path = processed_path(asset_cfg, data_root)
    if not path.exists():
        log.warning(
            "processed parquet missing for %s — skipping garch forecasts", asset_cfg["symbol"]
        )
        return pd.Series(dtype=float)

    df = pd.read_parquet(path)
    ret = compute_log_returns(df["close"])

    try:
        garch_model = GARCH(min_train=252, step=21)
        forecast_path = garch_model.forecast_path(ret)
    except (GarchFitError, Exception) as exc:  # noqa: BLE001 — per-window GARCH failure
        log.warning(
            "GARCH forecast_path failed for %s (%s: %s) — emitting no garch rows",
            asset_cfg["symbol"],
            type(exc).__name__,
            exc,
        )
        return pd.Series(dtype=float)

    # Restrict to label dates (drop NaN positions — before min_train)
    forecast_path = forecast_path.dropna()
    available = forecast_path.index.intersection(label_dates)
    return forecast_path.loc[available]


def label_garch_baseline(asset_cfg: dict, data_root: Path) -> pd.DataFrame:
    """Emit FVR rows with model_alias='garch_baseline' for an asset.

    Builds GARCH one-step-ahead forecasts for all labelable dates (dates where
    realized variance is available) and returns them in FVR_SCHEMA format.

    Args:
        asset_cfg: Asset dict with keys ``symbol``, ``asset_class``.
        data_root: Root of the data directory.

    Returns:
        DataFrame with columns matching ``FVR_SCHEMA``.  Empty if no labelable
        dates or if GARCH fit fails.
    """
    from volforecast.config import symbol_slug

    asset_slug = symbol_slug(asset_cfg["symbol"])

    try:
        rv_series = _realized_var_for_asset(asset_cfg, data_root)
    except FileNotFoundError:
        log.warning("processed parquet missing for %s — skipping garch labels", asset_slug)
        return pd.DataFrame(columns=FVR_SCHEMA)

    if rv_series.empty:
        return pd.DataFrame(columns=FVR_SCHEMA)

    label_dates = rv_series.index
    garch_forecasts = garch_live_forecasts(asset_cfg, data_root, label_dates)

    if garch_forecasts.empty:
        return pd.DataFrame(columns=FVR_SCHEMA)

    # Align: only keep dates where both forecast and realized exist
    common_dates = garch_forecasts.index.intersection(rv_series.index)
    if common_dates.empty:
        return pd.DataFrame(columns=FVR_SCHEMA)

    rows = []
    for date in common_dates:
        rows.append(
            {
                "as_of_date": date,
                "asset": asset_slug,
                "horizon": 1,
                "model_version": GARCH_MODEL_VERSION,
                "model_alias": GARCH_ALIAS,
                "forecast_var": float(garch_forecasts.loc[date]),
                "realized_var": float(rv_series.loc[date]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=FVR_SCHEMA)
    return pd.DataFrame(rows)[FVR_SCHEMA]


# ---------------------------------------------------------------------------
# Idempotent append
# ---------------------------------------------------------------------------


def append_labels(new_rows: pd.DataFrame, out_path: Path) -> int:
    """Atomically append ``new_rows`` to the forecast-vs-realized parquet.

    Deduplicates on ``LABEL_KEY`` before writing.  ``keep="first"`` preserves
    the original row on re-run (idempotent).

    The write is atomic: rows are staged to a ``.tmp.parquet`` sidecar and
    renamed with ``os.replace()`` (POSIX-atomic; also atomic on Windows NTFS
    within the same volume).

    Args:
        new_rows: DataFrame with columns matching ``FVR_SCHEMA``.
        out_path: Destination parquet path.

    Returns:
        Number of net-new rows added (0 on a re-run with identical data).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        prev_len = len(existing)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=LABEL_KEY, keep="first")
    else:
        prev_len = 0
        combined = new_rows.copy().reset_index(drop=True)

    tmp_path = out_path.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)

    net_new = len(combined) - prev_len
    log.info(
        "append_labels: wrote %d total rows (%d net-new) to %s", len(combined), net_new, out_path
    )
    return net_new


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def label_forecasts(data_root: Path, fvr_path: Path) -> int:
    """Run the full labelling pipeline: champion + GARCH rows → FVR parquet.

    Reads the prediction log, joins to realized variance for each asset, adds
    GARCH baseline rows for every configured asset, and appends to
    ``fvr_path`` idempotently.

    Args:
        data_root: Root of the data directory (e.g. ``project_root() / "data"``).
        fvr_path: Path for the forecast-vs-realized output parquet.

    Returns:
        Net-new rows appended to ``fvr_path``.

    Raises:
        FileNotFoundError: If ``data/predictions/predictions.parquet`` does not
            exist.  A clear error is raised — NOT a silent skip — per the
            Environment Availability policy in 04-RESEARCH.md.
    """
    from volforecast.config import load_assets

    pred_log_path = data_root / "predictions" / "predictions.parquet"
    if not pred_log_path.exists():
        raise FileNotFoundError(
            f"Prediction log not found at {pred_log_path}. "
            "Run the FastAPI service to produce forecast rows before labelling."
        )

    predictions = pd.read_parquet(pred_log_path)
    log.info("Loaded %d prediction log rows from %s", len(predictions), pred_log_path)

    # --- Champion rows ---
    champion_rows = label_champion_forecasts(predictions, data_root)
    log.info("Built %d champion FVR rows", len(champion_rows))

    # --- GARCH baseline rows (all configured assets) ---
    config_path = data_root.parent / "config" / "assets.yaml"
    assets_cfg = load_assets(config_path)
    garch_frames: list[pd.DataFrame] = []
    for asset_cfg in assets_cfg:
        gf = label_garch_baseline(asset_cfg, data_root)
        if not gf.empty:
            garch_frames.append(gf)
            log.info(
                "Built %d garch_baseline FVR rows for %s",
                len(gf),
                asset_cfg["symbol"],
            )

    garch_rows = (
        pd.concat(garch_frames, ignore_index=True)
        if garch_frames
        else pd.DataFrame(columns=FVR_SCHEMA)
    )
    log.info("Built %d garch_baseline FVR rows total", len(garch_rows))

    # --- Concat and append ---
    all_new = pd.concat([champion_rows, garch_rows], ignore_index=True)
    if all_new.empty:
        log.info("No new labelable rows — nothing to append")
        return 0

    return append_labels(all_new, fvr_path)
