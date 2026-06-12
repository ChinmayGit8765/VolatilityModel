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
    """Mimics native LGBMRegressor.predict() — returns fixed log-var array.

    Accepts **kwargs so the serving layer's ``validate_features=True`` (CR-02)
    passes through without error.
    """

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        return np.full(len(df), LOG_VAR_STUB)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_synthetic_processed(root: Path) -> None:
    """Create a hermetic project root: assets.yaml + synthetic processed parquets.

    CI has no data/ directory (gitignored, DVC-tracked), so the fixture must not
    depend on real ingested data. 320 rows is enough for every feature window
    (rv_66, GARCH_MIN_TRAIN=252) in build_features.
    """
    import shutil

    repo_root = Path(__file__).parent.parent.parent
    (root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(repo_root / "config" / "assets.yaml", root / "config" / "assets.yaml")

    rng = np.random.default_rng(7)
    n = 320
    assets = [
        ("crypto", "BTC-USD"),
        ("crypto", "ETH-USD"),
        ("equity", "SPY"),
        ("equity", "AAPL"),
        ("equity", "MSFT"),
    ]
    for asset_class, slug in assets:
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        log_ret = rng.normal(0.0, 0.015, size=n)
        close = 100.0 * np.exp(np.cumsum(log_ret))
        spread = np.abs(rng.normal(0.0, 0.005, size=n))
        df = pd.DataFrame(
            {
                "open": close * (1 - spread / 2),
                "high": close * (1 + spread),
                "low": close * (1 - spread),
                "close": close,
                "volume": rng.uniform(1e6, 5e6, size=n),
            },
            index=pd.Index(dates, name="date"),
        )
        out = root / "data" / "processed" / asset_class / f"{slug}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return a TestClient with a mocked champion model and tmp prediction log.

    The client is yielded inside a ``with TestClient(...)`` block so the
    lifespan (startup/shutdown) fires correctly.
    """
    # Point the prediction log at a temp directory
    log_path = tmp_path / "predictions.parquet"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:9999")  # unreachable — OK
    # Hermetic project root: synthetic processed data + assets.yaml (CI has no data/)
    _write_synthetic_processed(tmp_path)
    monkeypatch.setenv("VOLFORECAST_ROOT", str(tmp_path))

    # Import and inject fake model state BEFORE creating the TestClient
    # so the lifespan never actually calls mlflow.
    import importlib

    import volforecast.serving.app as app_mod
    import volforecast.serving.prediction_log as pl_mod

    importlib.reload(pl_mod)

    # Patch _load_champion_model to return the stub without hitting MLflow.
    # Returns (native_model, version, alias, model_columns).
    # model_columns=[] means _forecast_for will skip reindex — OK for the stub
    # which accepts any column set.
    monkeypatch.setattr(
        app_mod, "_load_champion_model", lambda: (_StubModel(), "3", "champion", [])
    )

    # Use TestClient as context manager so the ASGI lifespan runs
    with TestClient(app_mod.app, raise_server_exceptions=True) as client:
        yield client


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


def test_forecast_stale_data_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WR-06: a feature row exceeding the staleness bound yields 503, not a
    silently stale forecast.

    Setting VOLFORECAST_MAX_STALENESS_DAYS=-1 makes even a zero-day gap trip
    the guard, exercising the 503 path hermetically.
    """
    log_path = tmp_path / "predictions.parquet"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:9999")  # unreachable — OK
    monkeypatch.setenv("VOLFORECAST_MAX_STALENESS_DAYS", "-1")
    _write_synthetic_processed(tmp_path)
    monkeypatch.setenv("VOLFORECAST_ROOT", str(tmp_path))

    import importlib

    import volforecast.serving.app as app_mod
    import volforecast.serving.prediction_log as pl_mod

    importlib.reload(pl_mod)
    monkeypatch.setattr(
        app_mod, "_load_champion_model", lambda: (_StubModel(), "3", "champion", [])
    )

    with TestClient(app_mod.app, raise_server_exceptions=True) as client:
        resp = client.get("/forecast")

    assert resp.status_code == 503
    detail = str(resp.json().get("detail", ""))
    # Generic detail — must NOT leak dates, paths, or stack traces
    assert "Traceback" not in detail
    assert "/data/" not in detail


def test_forecast_model_version_propagated(api_client: TestClient) -> None:
    """model_version from the stub appears in every /forecast item."""
    resp = api_client.get("/forecast")
    assert resp.status_code == 200
    for item in resp.json()["forecasts"]:
        assert item["model_version"] == "3", (
            f"Expected model_version='3' from stub, got {item['model_version']!r}"
        )
