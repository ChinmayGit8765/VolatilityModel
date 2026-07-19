"""Regression tests for the daily flow's forecast_task (audit BLOCKER-1 fix).

Bug history: the daily flow never called the serving API, so no forecasts were
ever generated or logged — label_task became a silent no-op loop (0 champion
rows labelled forever).  These tests pin the two properties that make that
impossible again:

1. forecast_task hits ``GET {VOLFORECAST_API_URL}/forecast`` and RAISES on any
   HTTP/connection error — a dead serving layer fails the flow loudly.
2. daily_flow calls forecast_task AFTER features_task and BEFORE label_task
   (static source-order check, same style as test_cli_entrypoint.py).

All tests are hermetic: urllib is monkeypatched, no network calls, no Prefect
server (``task.fn()`` bypasses the Prefect engine and its retry delays).
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent


class _FakeResponse:
    """Minimal context-manager stand-in for urllib.request.urlopen's response."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_forecast_task_hits_forecast_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """forecast_task must GET {VOLFORECAST_API_URL}/forecast with a 300s timeout."""
    import pipelines.daily_pipeline as dp

    captured: dict = {}
    body = json.dumps(
        {
            "forecasts": [{"asset": f"A{i}", "forecast_var": 1e-4} for i in range(5)],
            "generated_at": "2026-07-19T00:00:00Z",
        }
    ).encode()

    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ANN202
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("VOLFORECAST_API_URL", "http://fake-api:1234")

    result = dp.forecast_task.fn()

    assert result == 5, "forecast_task must return the number of forecasts"
    assert captured["url"] == "http://fake-api:1234/forecast"
    assert captured["timeout"] == 300, "GARCH-as-feature needs the long 300s timeout"


def test_forecast_task_defaults_to_compose_api_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without VOLFORECAST_API_URL, the worker-container view http://api:8000 is used."""
    import pipelines.daily_pipeline as dp

    captured: dict = {}
    body = json.dumps({"forecasts": []}).encode()

    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ANN202
        captured["url"] = req.full_url
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("VOLFORECAST_API_URL", raising=False)

    result = dp.forecast_task.fn()

    assert result == 0
    assert captured["url"] == "http://api:8000/forecast"


def test_forecast_task_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-2xx response must RAISE — a dead serving layer fails the flow loudly."""
    import pipelines.daily_pipeline as dp

    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ANN202
        raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", None, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("VOLFORECAST_API_URL", "http://fake-api:1234")

    with pytest.raises(urllib.error.HTTPError):
        dp.forecast_task.fn()


def test_forecast_task_raises_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection failure must RAISE — never a silent skip."""
    import pipelines.daily_pipeline as dp

    def fake_urlopen(req, timeout=None):  # noqa: ANN001, ANN202
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(urllib.error.URLError):
        dp.forecast_task.fn()


def test_forecast_task_has_retries_configured() -> None:
    """forecast_task must retry transient failures (retries=2, 60s delay)."""
    import pipelines.daily_pipeline as dp

    assert dp.forecast_task.retries == 2
    assert dp.forecast_task.retry_delay_seconds == 60


def test_daily_flow_calls_forecast_between_features_and_label() -> None:
    """daily_flow must call forecast_task AFTER features_task, BEFORE label_task.

    Static source-order check: if forecast_task is dropped from the flow (or
    moved after labelling), the daily loop regresses to the silent no-op the
    audit found — features are built but no forecasts exist to label.
    """
    src = (REPO_ROOT / "pipelines" / "daily_pipeline.py").read_text(encoding="utf-8")

    # Restrict the scan to the flow body so task *definitions* don't match.
    flow_body = src[src.index("def daily_flow") :]

    i_features = flow_body.index("features_task()")
    i_forecast = flow_body.index("forecast_task()")
    i_label = flow_body.index("label_task()")

    assert i_features < i_forecast, "forecast_task must run AFTER features_task"
    assert i_forecast < i_label, "forecast_task must run BEFORE label_task"
