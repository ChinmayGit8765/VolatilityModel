"""Integration tests for the FastAPI inference service.

All tests run OFFLINE — no MLflow, no compose, no network.
The champion model is replaced with a deterministic stub via the module-level
_model_state dict override and PREDICTION_LOG_PATH pointed at tmp_path.

Test coverage:
  - test_health: /health returns 200 with model name, version, alias
  - test_forecast: /forecast returns 5-asset ForecastResponse with correct schema
  - test_forecast_symbol_known: /forecast/{symbol} returns single-asset forecast
  - test_forecast_symbol_unknown_404: unknown symbol returns 404 (generic detail, no stack trace)
  - test_prediction_log_grows: /forecast call appends row(s) to the parquet log
  - test_forecast_model_version_propagated: model_version from stub appears in response
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from volforecast.models.lgbm import KNOWN_ASSETS

# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------

LOG_VAR_STUB = -8.0  # exp(-8) ≈ 3.35e-4, a plausible variance


class _StubModel:
    """Mimics mlflow.pyfunc.PyFuncModel.predict() — returns fixed log-var array."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), LOG_VAR_STUB)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with a mocked champion model and tmp prediction log."""
    # Point the prediction log at a temp directory
    log_path = tmp_path / "predictions.parquet"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:9999")  # unreachable — OK
    # Point data root at a known location

    monkeypatch.setenv(
        "VOLFORECAST_ROOT",
        str(Path(__file__).parent.parent.parent),
    )

    # Import and inject fake model state BEFORE creating the TestClient
    # so the lifespan never actually calls mlflow.
    import importlib

    import volforecast.serving.app as app_mod

    import volforecast.serving.prediction_log as pl_mod

    importlib.reload(pl_mod)

    # Patch _load_model to return the stub without hitting MLflow
    monkeypatch.setattr(app_mod, "_load_champion_model", lambda: (_StubModel(), "3", "champion"))

    # Build the TestClient — lifespan runs on context-manager entry
    client = TestClient(app_mod.app, raise_server_exceptions=True)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health(api_client: TestClient) -> None:
    """/health returns 200 with expected keys."""
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "volforecast-lgbm"
    assert "model_version" in body
    assert body["alias"] == "champion"


def test_forecast_returns_all_assets(api_client: TestClient) -> None:
    """/forecast returns a ForecastResponse with one entry per KNOWN_ASSETS."""
    resp = api_client.get("/forecast")
    assert resp.status_code == 200
    body = resp.json()
    assert "forecasts" in body
    assert "generated_at" in body
    forecasts = body["forecasts"]
    assert len(forecasts) == len(KNOWN_ASSETS), (
        f"Expected {len(KNOWN_ASSETS)} forecasts, got {len(forecasts)}"
    )

    symbols_returned = {f["asset"] for f in forecasts}
    assert symbols_returned == set(KNOWN_ASSETS)


def test_forecast_schema(api_client: TestClient) -> None:
    """/forecast response items have all required fields with correct types."""
    resp = api_client.get("/forecast")
    assert resp.status_code == 200
    for item in resp.json()["forecasts"]:
        assert "asset" in item
        assert "forecast_var" in item
        assert item["forecast_var"] > 0, "forecast_var must be positive"
        assert "forecast_vol" in item
        # forecast_vol should approximately equal sqrt(forecast_var)
        assert abs(item["forecast_vol"] - math.sqrt(item["forecast_var"])) < 1e-10
        assert "model_version" in item
        assert "alias" in item
        assert "as_of_date" in item
        assert item["alias"] == "champion"


def test_forecast_symbol_known(api_client: TestClient) -> None:
    """/forecast/{symbol} for a known asset returns a single-item ForecastResponse."""
    resp = api_client.get("/forecast/BTC-USD")
    assert resp.status_code == 200
    body = resp.json()
    forecasts = body["forecasts"]
    assert len(forecasts) == 1
    assert forecasts[0]["asset"] == "BTC-USD"


def test_forecast_symbol_unknown_404(api_client: TestClient) -> None:
    """/forecast/{symbol} for an unknown symbol returns 404 with generic detail."""
    resp = api_client.get("/forecast/DOGE")
    assert resp.status_code == 404
    body = resp.json()
    # Must have a "detail" field
    assert "detail" in body
    detail = body["detail"]
    # Generic detail — must NOT leak stack traces or filesystem paths
    assert "Traceback" not in str(detail)
    assert "/data/" not in str(detail)
    assert "Exception" not in str(detail)


def test_prediction_log_grows(api_client: TestClient, tmp_path: Path) -> None:
    """After /forecast, the prediction log parquet gains rows."""
    log_path = tmp_path / "predictions.parquet"

    resp = api_client.get("/forecast")
    assert resp.status_code == 200

    assert log_path.exists(), "predictions.parquet must be created after /forecast"
    df = pd.read_parquet(log_path)
    expected_cols = ["timestamp_utc", "asset", "horizon", "forecast_var", "model_version", "alias"]
    assert list(df.columns) == expected_cols
    assert len(df) == len(KNOWN_ASSETS), (
        f"Expected {len(KNOWN_ASSETS)} log rows after /forecast, got {len(df)}"
    )


def test_prediction_log_grows_single(api_client: TestClient, tmp_path: Path) -> None:
    """After /forecast/{symbol}, the prediction log gains exactly one row."""
    log_path = tmp_path / "predictions.parquet"

    resp = api_client.get("/forecast/SPY")
    assert resp.status_code == 200

    assert log_path.exists()
    df = pd.read_parquet(log_path)
    assert len(df) == 1
    assert df.iloc[0]["asset"] == "SPY"


def test_forecast_model_version_propagated(api_client: TestClient) -> None:
    """model_version from the stub appears in every /forecast item."""
    resp = api_client.get("/forecast")
    assert resp.status_code == 200
    for item in resp.json()["forecasts"]:
        assert item["model_version"] == "3", (
            f"Expected model_version='3' from stub, got {item['model_version']!r}"
        )
