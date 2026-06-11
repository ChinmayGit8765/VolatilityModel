"""Unit tests for the atomic prediction-log append (Phase 4 contract).

Tests run offline — no MLflow, no compose, no network required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rows(n: int = 2, model_version: str = "3") -> pd.DataFrame:
    """Create n rows matching the PREDICTION_LOG_SCHEMA."""
    from volforecast.serving.prediction_log import PREDICTION_LOG_SCHEMA

    ts = datetime.now(tz=UTC)
    rows = {
        "timestamp_utc": [ts] * n,
        "asset": ["BTC-USD"] * n,
        "horizon": [1] * n,
        "forecast_var": [0.0004] * n,
        "model_version": [model_version] * n,
        "alias": ["champion"] * n,
    }
    return pd.DataFrame(rows)[PREDICTION_LOG_SCHEMA]


# ---------------------------------------------------------------------------
# PREDICTION_LOG_SCHEMA
# ---------------------------------------------------------------------------


def test_prediction_log_schema_order() -> None:
    """PREDICTION_LOG_SCHEMA must be in the exact Phase-4 contract order."""
    from volforecast.serving.prediction_log import PREDICTION_LOG_SCHEMA

    assert PREDICTION_LOG_SCHEMA == [
        "timestamp_utc",
        "asset",
        "horizon",
        "forecast_var",
        "model_version",
        "alias",
    ]


# ---------------------------------------------------------------------------
# First append creates the file
# ---------------------------------------------------------------------------


def test_first_append_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """append_predictions creates the parquet file on the first call."""
    log_path = tmp_path / "predictions" / "predictions.parquet"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))

    # Re-import to pick up the env var (module re-initialises path on import)
    import importlib

    import volforecast.serving.prediction_log as pl_mod

    importlib.reload(pl_mod)

    rows = _make_rows(3)
    pl_mod.append_predictions(rows)

    assert log_path.exists(), "parquet file should be created on first append"
    df = pd.read_parquet(log_path)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# Second append concatenates
# ---------------------------------------------------------------------------


def test_second_append_concatenates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two sequential appends preserve all rows (first_batch + second_batch)."""
    log_path = tmp_path / "predictions.parquet"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))

    import importlib

    import volforecast.serving.prediction_log as pl_mod

    importlib.reload(pl_mod)

    batch1 = _make_rows(2, model_version="3")
    batch2 = _make_rows(3, model_version="4")

    pl_mod.append_predictions(batch1)
    pl_mod.append_predictions(batch2)

    df = pd.read_parquet(log_path)
    assert len(df) == 5, f"Expected 5 rows after two appends, got {len(df)}"
    # First batch rows come first (concat preserves order)
    assert list(df["model_version"].iloc[:2]) == ["3", "3"]
    assert list(df["model_version"].iloc[2:]) == ["4", "4", "4"]


# ---------------------------------------------------------------------------
# Atomic write — no .tmp file observable after completion
# ---------------------------------------------------------------------------


def test_atomic_write_no_tmp_remains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After append_predictions, no .tmp.parquet file is left on disk."""
    log_path = tmp_path / "predictions.parquet"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))

    import importlib

    import volforecast.serving.prediction_log as pl_mod

    importlib.reload(pl_mod)

    pl_mod.append_predictions(_make_rows(1))

    tmp_file = log_path.with_suffix(".tmp.parquet")
    assert not tmp_file.exists(), ".tmp.parquet must not remain after write"


# ---------------------------------------------------------------------------
# Readback types
# ---------------------------------------------------------------------------


def test_readback_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """forecast_var is float64, horizon is int, columns are in schema order."""
    from volforecast.serving.prediction_log import PREDICTION_LOG_SCHEMA

    log_path = tmp_path / "predictions.parquet"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))

    import importlib

    import volforecast.serving.prediction_log as pl_mod

    importlib.reload(pl_mod)

    pl_mod.append_predictions(_make_rows(2))

    df = pd.read_parquet(log_path)
    # Column order must match schema
    assert list(df.columns) == PREDICTION_LOG_SCHEMA

    # Type assertions
    assert pd.api.types.is_float_dtype(df["forecast_var"]), "forecast_var must be float"
    assert pd.api.types.is_integer_dtype(df["horizon"]), "horizon must be int"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


def test_asset_forecast_validates() -> None:
    """AssetForecast accepts all documented fields."""
    import math
    from datetime import date

    from volforecast.serving.schemas import AssetForecast

    fv = 0.0004
    af = AssetForecast(
        asset="BTC-USD",
        forecast_var=fv,
        forecast_vol=math.sqrt(fv),
        horizon_days=1,
        as_of_date=date.today(),
        model_version="3",
        alias="champion",
    )
    assert af.asset == "BTC-USD"
    assert af.forecast_var == fv
    assert af.horizon_days == 1


def test_forecast_response_wraps_list() -> None:
    """ForecastResponse contains a list of AssetForecast and a generated_at string."""
    import math
    from datetime import date

    from volforecast.serving.schemas import AssetForecast, ForecastResponse

    fv = 0.0003
    af = AssetForecast(
        asset="SPY",
        forecast_var=fv,
        forecast_vol=math.sqrt(fv),
        as_of_date=date.today(),
        model_version="3",
        alias="champion",
    )
    resp = ForecastResponse(forecasts=[af], generated_at="2026-06-11T00:00:00Z")
    assert len(resp.forecasts) == 1
    assert resp.generated_at == "2026-06-11T00:00:00Z"


# ---------------------------------------------------------------------------
# Sequential-append all-rows-preserved (concurrency-ish smoke test)
# ---------------------------------------------------------------------------


def test_sequential_appends_preserve_all_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple sequential appends keep all rows; file is always valid parquet."""
    log_path = tmp_path / "multi.parquet"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))

    import importlib

    import volforecast.serving.prediction_log as pl_mod

    importlib.reload(pl_mod)

    total = 0
    for i in range(5):
        batch = _make_rows(i + 1)
        pl_mod.append_predictions(batch)
        total += i + 1
        df = pd.read_parquet(log_path)
        assert len(df) == total, f"After append {i + 1}: expected {total} rows, got {len(df)}"
        assert list(df.columns) == list(_make_rows(1).columns)
