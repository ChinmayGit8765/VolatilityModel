"""VolForecast observability dashboard — Streamlit multi-section app.

Renders four locked panels sourced exclusively from volforecast.dashboard.data_access
loaders (no direct parquet reads, MLflow client calls, or urllib calls here):

  1. Forecast vs realized  — per-asset time series with GARCH baseline overlay
  2. Drift status          — latest Evidently drift report summary (REPORT-ONLY)
  3. Live model version    — volforecast-lgbm@champion metadata from MLflow registry
  4. Service stats         — prediction-log row count + live /health probe

Every panel renders a friendly st.info / st.warning message when its data source
returns the empty/None sentinel value (fresh-clone / first-run path).

This module is READ-ONLY:
  - No file writes, no to_parquet calls, no write-mode opens.
  - No set_registered_model_alias calls.
  - No retraining triggers.
  - Drift status panel explicitly states that drift is REPORT-ONLY (locked anti-pattern).

Configuration via environment variables (with defaults for local dev):
  MLFLOW_TRACKING_URI  — default http://mlflow-server:5000
  API_BASE_URL         — default http://api:8000
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from volforecast.config import load_assets, symbol_slug
from volforecast.dashboard.data_access import (
    api_health,
    champion_info,
    data_root,
    latest_drift_summary,
    load_fvr,
    prediction_stats,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MLFLOW_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
_API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://api:8000")
_DATA_ROOT = data_root()
_MONITORING_DIR = _DATA_ROOT / "monitoring"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VolForecast Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)
st.title("VolForecast — Observability Dashboard")
st.caption(
    "Read-only system view: forecast accuracy, drift status, model registry, and service health."
)

# ---------------------------------------------------------------------------
# Panel 1: Forecast vs Realized
# ---------------------------------------------------------------------------

st.header("1. Forecast vs Realized Volatility")

fvr_df = load_fvr(_DATA_ROOT)

if fvr_df.empty:
    st.info(
        "No forecast-vs-realized data yet — run the daily pipeline to populate "
        "`data/monitoring/forecast_vs_realized.parquet`."
    )
else:
    # Build asset list from FVR data; fall back to config slugs when FVR is empty
    fvr_assets = sorted(fvr_df["asset"].unique().tolist())
    if not fvr_assets:
        try:
            fallback_assets = [symbol_slug(a["symbol"]) for a in load_assets()]
        except Exception:  # noqa: BLE001
            fallback_assets = []
        fvr_assets = fallback_assets or ["(no assets)"]

    selected_asset = st.selectbox("Select asset", fvr_assets)
    asset_df = fvr_df[fvr_df["asset"] == selected_asset].copy()

    if asset_df.empty:
        st.warning(f"No data for {selected_asset}.")
    else:
        # Build chart data: champion forecast, GARCH baseline, realized
        champion_rows = asset_df[asset_df["model_alias"] == "champion"]
        garch_rows = asset_df[asset_df["model_alias"] == "garch_baseline"]

        # Use as_of_date as the time axis; convert to datetime for plotting
        def _to_dt(series: pd.Series) -> pd.Series:
            return pd.to_datetime(series, utc=True)

        chart_frames = []
        if not champion_rows.empty:
            champ_plot = champion_rows[["as_of_date", "forecast_var", "realized_var"]].copy()
            champ_plot["as_of_date"] = _to_dt(champ_plot["as_of_date"])
            champ_plot = champ_plot.sort_values("as_of_date")
            # Wide format for st.line_chart
            champ_wide = champ_plot.set_index("as_of_date")[["forecast_var", "realized_var"]]
            champ_wide.columns = ["Champion forecast_var", "Realized_var"]
            chart_frames.append(champ_wide)

        if not garch_rows.empty:
            garch_plot = garch_rows[["as_of_date", "forecast_var"]].copy()
            garch_plot["as_of_date"] = _to_dt(garch_plot["as_of_date"])
            garch_plot = garch_plot.sort_values("as_of_date")
            garch_wide = garch_plot.set_index("as_of_date")[["forecast_var"]]
            garch_wide.columns = ["GARCH baseline forecast_var"]
            chart_frames.append(garch_wide)

        if chart_frames:
            combined = pd.concat(chart_frames, axis=1).sort_index()
            st.line_chart(combined)
            with st.expander("Raw FVR data"):
                st.dataframe(asset_df.sort_values("as_of_date"))
        else:
            st.warning("No champion or GARCH rows to plot for this asset.")

# ---------------------------------------------------------------------------
# Panel 2: Drift Status
# ---------------------------------------------------------------------------

st.divider()
st.header("2. Distribution Drift Status")

st.caption(
    ":information_source: Drift is **REPORT-ONLY** — a drift flag never implies "
    "automatic promotion or retraining. Only rolling-QLIKE performance degradation "
    "gates promotion (see `monitoring.promotion`)."
)

drift_summary = latest_drift_summary(_MONITORING_DIR)

if drift_summary is None:
    st.info(
        "No drift reports found under `data/monitoring/` — run the daily pipeline "
        "to generate `{date}_drift.json` files."
    )
else:
    col1, col2 = st.columns(2)
    dataset_drift = drift_summary.get("dataset_drift")
    n_drifted = drift_summary.get("n_drifted_columns")
    date_str = drift_summary.get("date", "unknown")

    with col1:
        st.metric("Report date", date_str)
        if dataset_drift is True:
            st.error("Dataset drift: DETECTED")
        elif dataset_drift is False:
            st.success("Dataset drift: not detected")
        else:
            st.warning("Dataset drift: unknown (could not parse report)")

    with col2:
        if n_drifted is not None:
            st.metric("Drifted columns", n_drifted)
        else:
            st.metric("Drifted columns", "n/a")

    html_path = drift_summary.get("html_path")
    if html_path:
        st.info(f"Full HTML report: `{html_path}`")

# ---------------------------------------------------------------------------
# Panel 3: Live Model Version / Alias
# ---------------------------------------------------------------------------

st.divider()
st.header("3. Live Champion Model")

champ = champion_info(_MLFLOW_URI)

if champ is None:
    st.info(
        "MLflow registry is unreachable or `volforecast-lgbm@champion` alias is not set. "
        "Train and register a model first, then run the daily pipeline."
    )
else:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model name", "volforecast-lgbm")
        st.metric("Champion version", champ.get("version", "—"))
    with col2:
        st.metric("Run ID", champ.get("run_id", "—"))

    params = champ.get("params", {})
    metrics = champ.get("metrics", {})

    if params:
        with st.expander("Training parameters"):
            st.json(params)
    if metrics:
        with st.expander("Evaluation metrics"):
            st.json(metrics)

# ---------------------------------------------------------------------------
# Panel 4: Service Stats
# ---------------------------------------------------------------------------

st.divider()
st.header("4. Service Stats")

stats = prediction_stats(_DATA_ROOT)
health = api_health(_API_BASE_URL, timeout=5.0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction log")
    row_count = stats.get("row_count", 0)
    if row_count == 0:
        st.info(
            "Prediction log is empty — call the `/forecast` endpoint to populate "
            "`data/predictions/predictions.parquet`."
        )
    else:
        st.metric("Total forecasts logged", row_count)
        last_ts = stats.get("last_timestamp")
        if last_ts is not None:
            st.metric("Last forecast at", str(last_ts))
        latest_per_asset = stats.get("latest_per_asset", {})
        if latest_per_asset:
            st.write("Latest forecast_var per asset:")
            asset_rows = [{"asset": k, "forecast_var": v} for k, v in latest_per_asset.items()]
            st.dataframe(pd.DataFrame(asset_rows).set_index("asset"))

with col2:
    st.subheader("API /health")
    if not health.get("reachable"):
        st.warning(
            f"API at `{_API_BASE_URL}` is not reachable. "
            "Start the `api` docker-compose service and ensure the model is loaded."
        )
    else:
        st.success("API is reachable")
        st.metric("Status", health.get("status", "—"))
        st.metric("Model version", health.get("model_version", "—"))
        st.metric("Alias", health.get("alias", "—"))
        latency = health.get("latency_ms")
        if latency is not None:
            st.metric("Round-trip latency", f"{latency:.1f} ms")
