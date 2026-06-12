"""FastAPI inference service for VolForecast.

Loads the ``@champion`` MLflow model at startup via the async lifespan hook,
serves next-day realized-variance forecasts for all 5 tracked assets, and
appends every forecast to the prediction log (Phase 4 contract).

Routes:
    GET /health                — service liveness + loaded model version
    GET /forecast              — all-asset next-day vol forecasts
    GET /forecast/{symbol}     — single-asset forecast (validates against KNOWN_ASSETS)

Security mitigations:
    T-03-09: symbol validated against KNOWN_ASSETS before any filesystem access (V5)
    T-03-10: all 4xx/5xx carry generic ``detail`` strings — no stack traces (V7)
    T-03-12: bound to 0.0.0.0:8000 in the container; host-side compose binds 127.0.0.1 only
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from volforecast.config import load_assets, processed_path, project_root, symbol_slug
from volforecast.features.pipeline import build_features
from volforecast.models.lgbm import ASSET_DTYPE, KNOWN_ASSETS, from_log_var
from volforecast.serving.prediction_log import append_predictions
from volforecast.serving.schemas import AssetForecast, ForecastResponse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model name / alias constants
# ---------------------------------------------------------------------------

_MODEL_NAME: str = "volforecast-lgbm"
_MODEL_ALIAS: str = "champion"

# ---------------------------------------------------------------------------
# Module-level state — populated in lifespan, cleared on shutdown
# ---------------------------------------------------------------------------

#: Shared mutable dict holding the loaded model and resolved version.
#: Keys: "model" (pyfunc.PyFuncModel), "version" (str), "alias" (str)
_model_state: dict = {}


# ---------------------------------------------------------------------------
# Model loader — separated so tests can monkeypatch it
# ---------------------------------------------------------------------------


def _load_champion_model() -> tuple:
    """Load the champion model from MLflow and resolve its version string.

    Returns:
        (native_lgb_model, version_str, alias_str, model_columns_list)

        ``native_lgb_model`` is the underlying LGBMRegressor extracted from the
        pyfunc wrapper.  We bypass the pyfunc schema validation layer (which
        can't coerce pandas CategoricalDtype -> string) and predict directly on
        the native model so the asset column can be passed as ASSET_DTYPE
        (required for LightGBM categorical splits).

        ``model_columns_list`` is the full ordered column list from the model's
        registered signature — used to reindex each inference row to the exact
        21-column training schema (including NaN-filled optional cross-asset
        columns for assets that don't have them).

    Raises:
        RuntimeError: if MLFLOW_TRACKING_URI is not set or the model fails to load.
    """
    import mlflow
    import mlflow.pyfunc

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI environment variable is not set.")

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{_MODEL_NAME}@{_MODEL_ALIAS}"
    log.info("Loading champion model from %s", model_uri)
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)

    # Extract native LGBMRegressor from pyfunc wrapper
    # (avoids schema-validation layer that rejects CategoricalDtype)
    native_model = pyfunc_model._model_impl.lgb_model

    # Read the full ordered column list from the registered model signature.
    # This is the 21-column training schema (18 base + garch + rv_22_eth +
    # rv_22_btc + asset) that every inference row must match.
    schema = pyfunc_model.metadata.get_input_schema()
    model_columns = [c.name for c in schema.inputs]
    log.info("Model signature columns: %s", model_columns)

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(_MODEL_NAME, _MODEL_ALIAS)
    version = str(mv.version)
    log.info("Champion model loaded: version=%s alias=%s", version, _MODEL_ALIAS)
    return native_model, version, _MODEL_ALIAS, model_columns


# ---------------------------------------------------------------------------
# Lifespan: load model on startup, clear on shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """FastAPI lifespan: load @champion model at startup, clear on shutdown."""
    model, version, alias, model_columns = _load_champion_model()
    _model_state["model"] = model
    _model_state["version"] = version
    _model_state["alias"] = alias
    _model_state["model_columns"] = model_columns
    log.info("Serving %s@%s version=%s", _MODEL_NAME, alias, version)
    yield
    _model_state.clear()
    log.info("Model state cleared on shutdown.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VolForecast Inference API",
    description="Next-day realized-variance forecasts for 5 assets via @champion model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Feature-building helper — SINGLE codepath (FEAT-07)
# ---------------------------------------------------------------------------


def _build_cross_asset_dfs(
    all_raw: dict[str, pd.DataFrame],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Pre-compute base features for BTC and ETH for cross-asset wiring.

    Cross-asset contract (mirrors scripts/generate_features.py exactly):
      - BTC rv_22 joined onto ETH, SPY, AAPL, MSFT  (key="btc")
      - ETH rv_22 joined onto BTC                    (key="eth")

    Args:
        all_raw: Mapping symbol -> raw OHLCV DataFrame (processed parquet).

    Returns:
        Mapping symbol -> cross_asset_dfs dict for that target symbol.
    """
    btc_sym = "BTC-USD"
    eth_sym = "ETH-USD"

    btc_base: pd.DataFrame | None = None
    eth_base: pd.DataFrame | None = None

    if btc_sym in all_raw:
        btc_base = build_features(all_raw[btc_sym], include_garch=False)
    if eth_sym in all_raw:
        eth_base = build_features(all_raw[eth_sym], include_garch=False)

    cross: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in KNOWN_ASSETS:
        sym_cross: dict[str, pd.DataFrame] = {}
        if btc_base is not None and sym != btc_sym:
            sym_cross["btc"] = btc_base[["rv_22"]]
        if eth_base is not None and sym == btc_sym:
            sym_cross["eth"] = eth_base[["rv_22"]]
        cross[sym] = sym_cross
    return cross


def _data_root() -> Path:
    """Return the data root directory.

    Resolution order:
    1. ``VOLFORECAST_DATA_ROOT`` environment variable (set by the container to /data)
    2. ``{project_root()}/data`` (local dev default)

    The container separates VOLFORECAST_ROOT (code/config at /app) from
    VOLFORECAST_DATA_ROOT (bind-mounted data at /data) so both are accessible.
    """
    env_data = os.environ.get("VOLFORECAST_DATA_ROOT")
    if env_data:
        return Path(env_data)
    return project_root() / "data"


def _forecast_for(symbols: list[str]) -> list[AssetForecast]:
    """Compute next-day variance forecasts for the given symbol list.

    Uses the single build_features codepath (FEAT-07).  Adds the ``asset``
    column and casts to ASSET_DTYPE to match training (Pitfall 3 mitigation).

    Args:
        symbols: List of symbols from KNOWN_ASSETS to forecast.

    Returns:
        List of AssetForecast items.

    Raises:
        HTTPException(500): On any data or model error, with a generic detail.
    """
    model = _model_state["model"]
    model_version: str = _model_state["version"]
    alias: str = _model_state["alias"]
    model_columns: list[str] = _model_state.get("model_columns", [])

    # --- Determine data root ---
    data_root = _data_root()

    # --- Load all raw processed DataFrames (needed for cross-asset base feats) ---
    assets_cfg = load_assets(project_root() / "config" / "assets.yaml")
    # Build a slug -> asset_cfg lookup
    slug_to_cfg = {symbol_slug(a["symbol"]): a for a in assets_cfg}

    # We need BTC + ETH for cross-asset, plus each requested symbol.
    # Load all KNOWN_ASSETS for cross-asset base computation.
    all_raw: dict[str, pd.DataFrame] = {}
    for sym in KNOWN_ASSETS:
        if sym not in slug_to_cfg:
            # Fallback: try direct symbol match
            cfg = next((a for a in assets_cfg if symbol_slug(a["symbol"]) == sym), None)
            if cfg is None:
                log.warning("No config entry found for %s", sym)
                continue
        else:
            cfg = slug_to_cfg[sym]
        p = processed_path(cfg, data_root)
        if not p.exists():
            log.warning("Processed parquet not found: %s", p)
            continue
        df = pd.read_parquet(p)
        if df.index.name != "date":
            df.index.name = "date"
        all_raw[sym] = df

    if not all_raw:
        raise HTTPException(status_code=500, detail="Service unavailable.")

    # --- Build cross-asset dict ---
    cross_by_symbol = _build_cross_asset_dfs(all_raw)

    # --- Compute forecasts for each requested symbol ---
    forecasts: list[AssetForecast] = []
    log_rows: list[dict] = []
    ts_utc = datetime.now(tz=UTC)

    for sym in symbols:
        if sym not in all_raw:
            log.error("Raw data missing for %s", sym)
            raise HTTPException(status_code=500, detail="Service unavailable.")

        raw_df = all_raw[sym]
        cross_dfs = cross_by_symbol.get(sym)

        # Build features via the single codepath (FEAT-07)
        feat_df = build_features(raw_df, cross_asset_dfs=cross_dfs or None, include_garch=True)

        # Drop NaN rows (leading rows from rolling windows)
        feat_df = feat_df.dropna()
        if feat_df.empty:
            log.error("No non-NaN feature rows for %s", sym)
            raise HTTPException(status_code=500, detail="Service unavailable.")

        # Take the last as-of row
        last_row = feat_df.iloc[[-1]].copy()
        as_of_date = last_row.index[-1].date()

        # Add asset column + cast to ASSET_DTYPE (Pitfall 3 mitigation)
        last_row["asset"] = sym
        last_row["asset"] = last_row["asset"].astype(ASSET_DTYPE)

        # Reindex to the full training schema (NaN-fill optional cross-asset cols).
        # This mirrors evaluate_per_asset's reindex logic so the native LightGBM
        # model receives exactly the 21-column schema it was trained on.
        if model_columns:
            last_row = last_row.reindex(columns=model_columns)
            # Re-apply ASSET_DTYPE after reindex (reindex may drop category dtype)
            last_row["asset"] = last_row["asset"].astype(ASSET_DTYPE)

        # Predict via native LGBMRegressor (bypasses pyfunc schema coercion
        # which can't convert CategoricalDtype to string required by pyfunc).
        # CR-02: validate_features=True makes a column name/order mismatch
        # raise loudly instead of silently scoring a transposed matrix.
        # Native predict returns np.ndarray of log-variance predictions.
        log_var_pred = model.predict(last_row, validate_features=True)
        # Ensure we have a 1-element array
        if hasattr(log_var_pred, "__len__"):
            log_var_val = float(log_var_pred[0])
        else:
            log_var_val = float(log_var_pred)

        forecast_var = float(from_log_var(np.array([log_var_val]))[0])
        forecast_vol = math.sqrt(forecast_var)

        forecasts.append(
            AssetForecast(
                asset=sym,
                forecast_var=forecast_var,
                forecast_vol=forecast_vol,
                horizon_days=1,
                as_of_date=as_of_date,
                model_version=model_version,
                alias=alias,
            )
        )

        # Build prediction log row
        log_rows.append(
            {
                "timestamp_utc": ts_utc,
                "asset": sym,
                "horizon": 1,
                "forecast_var": forecast_var,
                "model_version": model_version,
                "alias": alias,
            }
        )

    # --- Append to prediction log (Phase 4 contract) ---
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        try:
            append_predictions(log_df)
        except Exception:
            # Prediction log failure is non-fatal for the forecast response
            log.exception("Failed to append to prediction log.")

    return forecasts


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Return service liveness and loaded model version.

    Returns 200 with::

        {"status": "ok", "model_name": "volforecast-lgbm",
         "model_version": "<N>", "alias": "champion"}

    Returns 503 if the model has not been loaded (startup failure).
    """
    if not _model_state:
        raise HTTPException(status_code=503, detail="Service unavailable.")
    return {
        "status": "ok",
        "model_name": _MODEL_NAME,
        "model_version": _model_state["version"],
        "alias": _model_state["alias"],
    }


@app.get("/forecast", response_model=ForecastResponse)
def forecast_all() -> ForecastResponse:
    """Return next-day vol forecasts for all 5 tracked assets.

    Computes features via the single ``build_features`` codepath (FEAT-07),
    appends rows to the prediction log, and returns a ``ForecastResponse``.
    """
    if not _model_state:
        raise HTTPException(status_code=503, detail="Service unavailable.")
    forecasts = _forecast_for(list(KNOWN_ASSETS))
    return ForecastResponse(
        forecasts=forecasts,
        generated_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@app.get("/forecast/{symbol}", response_model=ForecastResponse)
def forecast_symbol(symbol: str) -> ForecastResponse:
    """Return next-day vol forecast for a single asset.

    Args:
        symbol: Asset symbol, must be in KNOWN_ASSETS.

    Raises:
        HTTPException(404): If ``symbol`` is not in KNOWN_ASSETS.
            Detail is generic — no filesystem paths or stack traces (V5/V7).
    """
    # T-03-09: validate BEFORE any filesystem access
    if symbol not in KNOWN_ASSETS:
        raise HTTPException(status_code=404, detail="Asset not found.")

    if not _model_state:
        raise HTTPException(status_code=503, detail="Service unavailable.")

    forecasts = _forecast_for([symbol])
    return ForecastResponse(
        forecasts=forecasts,
        generated_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
