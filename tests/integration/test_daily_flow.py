"""Offline integration tests for the Prefect daily flow.

All tests run hermetically — no live Prefect server, no MLflow, no network calls.
External calls are monkeypatched with lightweight stubs.

A session-scoped ``prefect_test_harness`` fixture spins up an in-memory Prefect
server for the duration of the test session; individual tests use ``task.fn()``
for direct business-logic assertions where a full flow run is unnecessary
(possible because tasks use ``_get_logger()`` which falls back to stdlib
logging when there is no Prefect context).

Test coverage:
  - test_happy_path_no_retrain: full flow with should_retrain=False skips
    retrain and promotion_gate.
  - test_force_retrain_path: force_retrain=True runs retrain→eval→promotion-gate
    exactly once each.
  - test_performance_flag_triggers_retrain: stubbed performance flag=True triggers
    retrain independently of force_retrain.
  - test_promotion_defaults_no_promote: promotion gate task.fn() returns False when
    challenger QLIKE does not beat champion (real QLIKE via frozen-window helper).
  - test_flow_returns_expected_keys: return dict contains all documented keys.
  - test_label_task_fn_direct: task.fn() calls label_forecasts directly (offline).
  - test_performance_check_task_cold_start: FVR file absent → task.fn() returns False.
  - test_drift_check_task_no_reference: reference snapshot absent → task.fn() returns
    str path and does not raise.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Session-scoped Prefect in-memory server (Pattern 7 from 04-RESEARCH.md)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    """Start an in-memory Prefect server for the test session."""
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness():
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_run_script(script_path: object, description: str) -> MagicMock:
    """Stub for _run_script — returns a MagicMock (no subprocess invoked)."""
    return MagicMock(returncode=0, stdout="ok", stderr="")


def _make_fvr_file(fvr_path: Path, n: int = 30, seed: int = 42) -> None:
    """Write a synthetic FVR parquet with both champion and garch_baseline rows."""
    rng = np.random.default_rng(seed)
    today = datetime.date.today()
    rows = []
    for i in range(n):
        d = today - datetime.timedelta(days=i + 1)
        rv = float(rng.uniform(1e-4, 3e-4))
        rows.append(
            {
                "as_of_date": d,
                "asset": "BTC-USD",
                "horizon": 1,
                "model_version": "3",
                "model_alias": "champion",
                "forecast_var": rv * 1.02,
                "realized_var": rv,
            }
        )
        rows.append(
            {
                "as_of_date": d,
                "asset": "BTC-USD",
                "horizon": 1,
                "model_version": "garch_1_1",
                "model_alias": "garch_baseline",
                "forecast_var": rv * 1.04,
                "realized_var": rv,
            }
        )
    fvr_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(fvr_path)


# ---------------------------------------------------------------------------
# Tests — full flow (through prefect_test_harness)
# ---------------------------------------------------------------------------


def test_happy_path_no_retrain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: no performance degradation, force_retrain=False.

    retrain_task must NOT be called; should_retrain=False, promoted=False.
    """
    import subprocess

    import pipelines.daily_pipeline as dp

    monkeypatch.setattr(dp, "_run_script", _stub_run_script)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: MagicMock(returncode=0, stdout="ok", stderr=""),
    )
    monkeypatch.setattr(
        "volforecast.monitoring.labeller.label_forecasts",
        lambda data_root, fvr_path: 5,
    )
    monkeypatch.setattr(
        "volforecast.monitoring.performance.run_performance_monitor",
        lambda fvr_path, **kwargs: False,
    )
    monkeypatch.setattr(dp, "_reference_path", lambda *a, **kw: tmp_path / "nonexistent.parquet")
    monkeypatch.setattr(dp, "_fvr_path", lambda: tmp_path / "fvr.parquet")
    monkeypatch.setattr(dp, "_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(dp, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(dp, "_monitoring_dir", lambda: tmp_path / "monitoring")

    retrain_calls: list[str] = []
    monkeypatch.setattr(dp.retrain_task, "fn", lambda: (retrain_calls.append("r"), "99")[1])

    result = dp.daily_flow(force_retrain=False)

    assert result["should_retrain"] is False
    assert result["promoted"] is False
    assert result["challenger_version"] is None
    assert len(retrain_calls) == 0, "retrain_task must not be called when should_retrain=False"
    assert "rows_labelled" in result
    assert "drift_report_path" in result
    assert "run_date" in result


def test_force_retrain_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """force_retrain=True triggers retrain regardless of performance flag.

    retrain_task, eval_task, and promotion_gate_task each called exactly once.
    """
    import subprocess

    import pipelines.daily_pipeline as dp

    monkeypatch.setattr(dp, "_run_script", _stub_run_script)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: MagicMock(returncode=0, stdout="ok", stderr=""),
    )
    monkeypatch.setattr(
        "volforecast.monitoring.labeller.label_forecasts",
        lambda data_root, fvr_path: 3,
    )
    monkeypatch.setattr(
        "volforecast.monitoring.performance.run_performance_monitor",
        lambda fvr_path, **kwargs: False,  # perf flag OFF — force_retrain overrides
    )
    monkeypatch.setattr(dp, "_reference_path", lambda *a, **kw: tmp_path / "nonexistent.parquet")
    monkeypatch.setattr(dp, "_fvr_path", lambda: tmp_path / "fvr.parquet")
    monkeypatch.setattr(dp, "_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(dp, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(dp, "_monitoring_dir", lambda: tmp_path / "monitoring")

    call_log: list[str] = []

    monkeypatch.setattr(dp.retrain_task, "fn", lambda: (call_log.append("retrain"), "10")[1])
    monkeypatch.setattr(dp.eval_task, "fn", lambda: call_log.append("eval"))
    monkeypatch.setattr(
        dp.promotion_gate_task, "fn", lambda v: (call_log.append("promo"), False)[1]
    )

    result = dp.daily_flow(force_retrain=True)

    assert result["should_retrain"] is True
    assert result["challenger_version"] == "10"
    assert result["promoted"] is False
    assert call_log.count("retrain") == 1, "retrain must run exactly once"
    assert call_log.count("eval") == 1, "eval must run exactly once"
    assert call_log.count("promo") == 1, "promotion gate must run exactly once"


def test_performance_flag_triggers_retrain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Performance degradation flag=True triggers retrain even when force_retrain=False.

    We create a real FVR file so performance_check_task reads it (not cold-start),
    and stub run_performance_monitor to return True.
    """
    import subprocess

    import pipelines.daily_pipeline as dp

    # Create a real FVR file so cold-start check doesn't short-circuit
    fvr_path = tmp_path / "fvr.parquet"
    _make_fvr_file(fvr_path)

    monkeypatch.setattr(dp, "_run_script", _stub_run_script)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: MagicMock(returncode=0, stdout="ok", stderr=""),
    )
    monkeypatch.setattr(
        "volforecast.monitoring.labeller.label_forecasts",
        lambda data_root, fvr_path_: 0,
    )
    # Performance monitor says retrain!
    monkeypatch.setattr(
        "volforecast.monitoring.performance.run_performance_monitor",
        lambda fvr_path_, **kwargs: True,
    )
    monkeypatch.setattr(dp, "_reference_path", lambda *a, **kw: tmp_path / "nonexistent.parquet")
    monkeypatch.setattr(dp, "_fvr_path", lambda: fvr_path)
    monkeypatch.setattr(dp, "_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(dp, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(dp, "_monitoring_dir", lambda: tmp_path / "monitoring")

    retrain_calls: list[str] = []
    monkeypatch.setattr(dp.retrain_task, "fn", lambda: (retrain_calls.append("retrain"), "5")[1])
    monkeypatch.setattr(dp.eval_task, "fn", lambda: None)
    monkeypatch.setattr(dp.promotion_gate_task, "fn", lambda v: False)

    result = dp.daily_flow(force_retrain=False)

    assert result["should_retrain"] is True
    assert len(retrain_calls) == 1, "retrain_task must fire exactly once on perf flag"


def test_flow_returns_expected_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Return dict must contain exactly the documented summary keys."""
    import subprocess

    import pipelines.daily_pipeline as dp

    monkeypatch.setattr(dp, "_run_script", _stub_run_script)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: MagicMock(returncode=0, stdout="ok", stderr=""),
    )
    monkeypatch.setattr(
        "volforecast.monitoring.labeller.label_forecasts",
        lambda data_root, fvr_path: 0,
    )
    monkeypatch.setattr(
        "volforecast.monitoring.performance.run_performance_monitor",
        lambda fvr_path, **kwargs: False,
    )
    monkeypatch.setattr(dp, "_reference_path", lambda *a, **kw: tmp_path / "nonexistent.parquet")
    monkeypatch.setattr(dp, "_fvr_path", lambda: tmp_path / "fvr.parquet")
    monkeypatch.setattr(dp, "_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(dp, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(dp, "_monitoring_dir", lambda: tmp_path / "monitoring")

    result = dp.daily_flow(force_retrain=False)

    required_keys = {
        "rows_labelled",
        "drift_report_path",
        "should_retrain",
        "challenger_version",
        "promoted",
        "run_date",
    }
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"


# ---------------------------------------------------------------------------
# Tests — task.fn() direct business-logic assertions (no Prefect context needed)
# ---------------------------------------------------------------------------


def test_label_task_fn_direct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct task.fn() call verifies label_task delegates to label_forecasts.

    Uses monkeypatching so no file I/O occurs.
    """
    import pipelines.daily_pipeline as dp

    monkeypatch.setattr(dp, "_data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(dp, "_fvr_path", lambda: tmp_path / "fvr.parquet")
    monkeypatch.setattr(
        "volforecast.monitoring.labeller.label_forecasts",
        lambda data_root, fvr_path: 7,
    )

    # Call business logic directly via .fn() — bypasses Prefect state
    result = dp.label_task.fn()
    assert result == 7


def test_performance_check_task_cold_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When FVR file does not exist, performance_check_task.fn() returns False (cold start)."""
    import pipelines.daily_pipeline as dp

    monkeypatch.setattr(dp, "_fvr_path", lambda: tmp_path / "nonexistent_fvr.parquet")

    result = dp.performance_check_task.fn()
    assert result is False, "performance_check_task must return False when FVR file does not exist"


def test_drift_check_task_no_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When reference snapshot is absent, drift_check_task.fn() returns a str and does not raise."""
    import pipelines.daily_pipeline as dp

    monitoring_dir = tmp_path / "monitoring"
    monkeypatch.setattr(
        dp, "_reference_path", lambda *a, **kw: tmp_path / "nonexistent_ref.parquet"
    )
    monkeypatch.setattr(dp, "_monitoring_dir", lambda: monitoring_dir)
    monkeypatch.setattr(dp, "_data_root", lambda: tmp_path / "data")

    result = dp.drift_check_task.fn()
    assert isinstance(result, str), "drift_check_task must return a str path"


def test_promotion_defaults_no_promote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Promotion gate returns False when challenger QLIKE does not beat champion.

    Uses real frozen-window QLIKE via promotion.py; challenger has worse forecasts.
    When challenger_qlike >= champion_qlike, promote_if_better returns False BEFORE
    any MLflow call — so no MLflow mock is needed for this test.
    """
    import pipelines.daily_pipeline as dp
    from volforecast.monitoring.promotion import PROMOTION_COOLDOWN_DAYS

    # Build FVR with both champion and challenger rows; challenger has worse QLIKE
    fvr_path = tmp_path / "fvr.parquet"
    today = datetime.date.today()
    window_days = PROMOTION_COOLDOWN_DAYS * 3
    rng = np.random.default_rng(0)
    rows = []
    for i in range(window_days):
        d = today - datetime.timedelta(days=i + 1)
        rv = float(rng.uniform(1e-4, 3e-4))
        rows.append(
            {
                "as_of_date": d,
                "asset": "BTC-USD",
                "horizon": 1,
                "model_version": "3",
                "model_alias": "champion",
                "forecast_var": rv * 1.01,  # close to realized → good QLIKE (low loss)
                "realized_var": rv,
            }
        )
        rows.append(
            {
                "as_of_date": d,
                "asset": "BTC-USD",
                "horizon": 1,
                "model_version": "4",
                "model_alias": "challenger",
                "forecast_var": rv * 2.5,  # far from realized → bad QLIKE (high loss)
                "realized_var": rv,
            }
        )
    fvr_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(fvr_path)

    monkeypatch.setattr(dp, "_fvr_path", lambda: fvr_path)

    # Stub mlflow.set_tracking_uri so it never tries to reach a server
    import mlflow

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    # Run the promotion gate directly via .fn() — challenger is worse, no-promote expected
    # promote_if_better returns False before calling MlflowClient when challenger loses
    promoted = dp.promotion_gate_task.fn(challenger_version="4")

    assert promoted is False, "Promotion gate must return False when challenger QLIKE is worse"
