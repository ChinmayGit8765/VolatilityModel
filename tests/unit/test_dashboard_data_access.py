"""Offline unit tests for volforecast.dashboard.data_access.

All tests are hermetic: tmp_path + synthetic data only.
No real MLflow server, no real API, no network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from volforecast.monitoring.labeller import FVR_SCHEMA
from volforecast.serving.prediction_log import PREDICTION_LOG_SCHEMA

# ---------------------------------------------------------------------------
# Helpers — synthetic data builders
# ---------------------------------------------------------------------------


def _make_fvr_df(n: int = 5) -> pd.DataFrame:
    """Synthetic FVR rows (champion + garch_baseline)."""
    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for d in dates:
        for alias in ("champion", "garch_baseline"):
            rows.append(
                {
                    "as_of_date": d,
                    "asset": "BTC-USD",
                    "horizon": 1,
                    "model_version": "3" if alias == "champion" else "garch_1_1",
                    "model_alias": alias,
                    "forecast_var": float(rng.uniform(1e-4, 1e-3)),
                    "realized_var": float(rng.uniform(1e-4, 1e-3)),
                }
            )
    return pd.DataFrame(rows)[FVR_SCHEMA]


def _make_prediction_log(n: int = 10) -> pd.DataFrame:
    """Synthetic prediction log rows."""
    import numpy as np

    rng = np.random.default_rng(1)
    ts = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "asset": ["BTC-USD"] * n,
            "horizon": [1] * n,
            "forecast_var": rng.uniform(1e-4, 1e-3, n).tolist(),
            "model_version": ["3"] * n,
            "alias": ["champion"] * n,
        }
    )[PREDICTION_LOG_SCHEMA]


def _write_fvr(tmp_path: Path, df: pd.DataFrame) -> Path:
    """Write FVR parquet to the expected location inside tmp_path (as data root)."""
    p = tmp_path / "monitoring" / "forecast_vs_realized.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def _write_prediction_log(tmp_path: Path, df: pd.DataFrame) -> Path:
    """Write prediction log to the expected location inside tmp_path (as data root)."""
    p = tmp_path / "predictions" / "predictions.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def _write_drift_json(monitoring_dir: Path, date_str: str, payload: dict) -> Path:
    """Write a synthetic drift JSON file to monitoring_dir."""
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    p = monitoring_dir / f"{date_str}_drift.json"
    p.write_text(json.dumps(payload))
    return p


def _evidently_07_payload(count: float, share: float, drift_share: float = 0.5) -> dict:
    """Trimmed mirror of the REAL Evidently 0.7.21 report JSON.

    Copied (and truncated to 2 ValueDrift entries) from the actual pipeline
    output ``data/monitoring/2026-07-19_drift.json``: a top-level ``metrics``
    array holding one DriftedColumnsCount entry (value = {count, share}) and
    one ValueDrift entry per column (value = float).  No ``dataset_drift``
    boolean and no ``drift_detected`` flags exist anywhere in this format.
    """
    return {
        "metrics": [
            {
                "id": "15e89f895b482f9b84ba7274ed18a106",
                "metric_name": f"DriftedColumnsCount(drift_share={drift_share})",
                "config": {
                    "type": "evidently:metric_v2:DriftedColumnsCount",
                    "drift_share": drift_share,
                },
                "value": {"count": count, "share": share},
            },
            {
                "id": "1358ee3ea052bbf5500ada43d67da668",
                "metric_name": (
                    "ValueDrift(column=log_return,"
                    "method=Wasserstein distance (normed),threshold=0.1)"
                ),
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "log_return",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.0033209075502455805,
            },
            {
                "id": "1fae27b9653b49267e0b830cfffc18da",
                "metric_name": (
                    "ValueDrift(column=rv_5,method=Wasserstein distance (normed),threshold=0.1)"
                ),
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "rv_5",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.005700922203894937,
            },
        ],
        "tests": [],
    }


# ---------------------------------------------------------------------------
# load_fvr tests
# ---------------------------------------------------------------------------


class TestLoadFvr:
    """Tests for load_fvr: empty-state + happy path."""

    def test_missing_parquet_returns_empty_dataframe(self, tmp_path: Path) -> None:
        """When the FVR parquet does not exist, return empty DataFrame with schema cols."""
        from volforecast.dashboard.data_access import load_fvr

        result = load_fvr(tmp_path)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        # All FVR_SCHEMA columns must be present
        for col in FVR_SCHEMA:
            assert col in result.columns, f"Missing column: {col}"

    def test_existing_parquet_returns_data(self, tmp_path: Path) -> None:
        """When the FVR parquet exists, return the parsed DataFrame."""
        from volforecast.dashboard.data_access import load_fvr

        df = _make_fvr_df(5)
        _write_fvr(tmp_path, df)

        result = load_fvr(tmp_path)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == len(df)
        for col in FVR_SCHEMA:
            assert col in result.columns

    def test_does_not_raise_on_missing_parquet(self, tmp_path: Path) -> None:
        """load_fvr must never raise when the parquet is absent."""
        from volforecast.dashboard.data_access import load_fvr

        # This must not raise
        try:
            load_fvr(tmp_path)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"load_fvr raised on missing parquet: {exc}")


# ---------------------------------------------------------------------------
# latest_drift_summary tests
# ---------------------------------------------------------------------------


class TestLatestDriftSummary:
    """Tests for latest_drift_summary: empty-state + happy path."""

    def test_no_drift_json_returns_none(self, tmp_path: Path) -> None:
        """When no *_drift.json files exist, return None."""
        from volforecast.dashboard.data_access import latest_drift_summary

        monitoring_dir = tmp_path / "monitoring"
        monitoring_dir.mkdir()

        result = latest_drift_summary(monitoring_dir)
        assert result is None

    def test_missing_monitoring_dir_returns_none(self, tmp_path: Path) -> None:
        """When monitoring dir itself does not exist, return None."""
        from volforecast.dashboard.data_access import latest_drift_summary

        result = latest_drift_summary(tmp_path / "monitoring" / "nonexistent")
        assert result is None

    def test_real_shape_drift_detected(self, tmp_path: Path) -> None:
        """share >= 0.5 in the REAL Evidently 0.7 shape → dataset_drift True (exact)."""
        from volforecast.dashboard.data_access import latest_drift_summary

        monitoring_dir = tmp_path / "monitoring"
        payload = _evidently_07_payload(count=12.0, share=0.6)
        _write_drift_json(monitoring_dir, "2026-06-12", payload)

        result = latest_drift_summary(monitoring_dir)

        assert result is not None
        assert isinstance(result, dict)
        # Required keys
        assert "date" in result
        assert "dataset_drift" in result
        assert "n_drifted_columns" in result
        assert "html_path" in result
        # EXACT assertions — no tautologies
        assert result["date"] == "2026-06-12"
        assert result["dataset_drift"] is True
        assert result["n_drifted_columns"] == 12

    def test_real_shape_share_below_threshold_is_not_drifted(self, tmp_path: Path) -> None:
        """share < 0.5 in the REAL shape → dataset_drift False (never None)."""
        from volforecast.dashboard.data_access import latest_drift_summary

        monitoring_dir = tmp_path / "monitoring"
        payload = _evidently_07_payload(count=2.0, share=0.1)
        _write_drift_json(monitoring_dir, "2026-06-13", payload)

        result = latest_drift_summary(monitoring_dir)

        assert result is not None
        assert result["dataset_drift"] is False
        assert result["n_drifted_columns"] == 2

    def test_real_shape_zero_drift(self, tmp_path: Path) -> None:
        """count=0.0/share=0.0 (the exact values in the real repo file) → False / 0."""
        from volforecast.dashboard.data_access import latest_drift_summary

        monitoring_dir = tmp_path / "monitoring"
        payload = _evidently_07_payload(count=0.0, share=0.0)
        _write_drift_json(monitoring_dir, "2026-07-19", payload)

        result = latest_drift_summary(monitoring_dir)

        assert result is not None
        assert result["dataset_drift"] is False
        assert result["n_drifted_columns"] == 0

    def test_real_shape_respects_configured_drift_share(self, tmp_path: Path) -> None:
        """The threshold comes from config.drift_share when present (not hardcoded 0.5)."""
        from volforecast.dashboard.data_access import latest_drift_summary

        monitoring_dir = tmp_path / "monitoring"
        # share=0.3 would be "no drift" at the default 0.5, but drift_share=0.25
        payload = _evidently_07_payload(count=5.0, share=0.3, drift_share=0.25)
        _write_drift_json(monitoring_dir, "2026-06-14", payload)

        result = latest_drift_summary(monitoring_dir)

        assert result is not None
        assert result["dataset_drift"] is True
        assert result["n_drifted_columns"] == 5

    def test_parses_actual_repo_drift_file_if_present(self, tmp_path: Path) -> None:
        """Integration-style: parse the REAL pipeline output file, skip when absent.

        ``data/monitoring/2026-07-19_drift.json`` is gitignored pipeline
        output — skipping keeps CI hermetic while still exercising the parser
        against genuine Evidently 0.7.21 bytes on dev machines.
        """
        from volforecast.dashboard.data_access import latest_drift_summary

        repo_root = Path(__file__).resolve().parents[2]
        real_file = repo_root / "data" / "monitoring" / "2026-07-19_drift.json"
        if not real_file.exists():
            pytest.skip("real drift JSON not present (fresh clone / CI)")

        # Copy into an isolated monitoring dir so latest-file selection is
        # deterministic regardless of what else lives in data/monitoring/.
        monitoring_dir = tmp_path / "monitoring"
        monitoring_dir.mkdir(parents=True)
        (monitoring_dir / real_file.name).write_bytes(real_file.read_bytes())

        result = latest_drift_summary(monitoring_dir)

        assert result is not None
        assert result["date"] == "2026-07-19"
        # The parser MUST resolve both fields on real output — None means the
        # parser regressed to targeting a schema Evidently never produces.
        assert isinstance(result["dataset_drift"], bool)
        assert isinstance(result["n_drifted_columns"], int)
        assert result["n_drifted_columns"] >= 0

    def test_multiple_drift_jsons_picks_latest(self, tmp_path: Path) -> None:
        """With multiple drift JSONs, the one with the latest date string is selected."""
        from volforecast.dashboard.data_access import latest_drift_summary

        monitoring_dir = tmp_path / "monitoring"
        _write_drift_json(monitoring_dir, "2026-06-10", {"metrics": []})
        _write_drift_json(monitoring_dir, "2026-06-12", {"metrics": [], "latest": True})
        _write_drift_json(monitoring_dir, "2026-06-11", {"metrics": []})

        result = latest_drift_summary(monitoring_dir)

        assert result is not None
        assert result["date"] == "2026-06-12"

    def test_drift_json_with_no_drift_flag(self, tmp_path: Path) -> None:
        """When drift JSON has no recognisable dataset_drift key, dataset_drift is None."""
        from volforecast.dashboard.data_access import latest_drift_summary

        monitoring_dir = tmp_path / "monitoring"
        _write_drift_json(monitoring_dir, "2026-06-12", {"completely_unknown": "structure"})

        result = latest_drift_summary(monitoring_dir)

        assert result is not None
        assert result["dataset_drift"] is None  # defensive — not found
        assert result["n_drifted_columns"] is None

    def test_does_not_raise_on_malformed_json(self, tmp_path: Path) -> None:
        """Malformed JSON should not crash the loader — returns None or partial dict."""
        from volforecast.dashboard.data_access import latest_drift_summary

        monitoring_dir = tmp_path / "monitoring"
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        bad_file = monitoring_dir / "2026-06-12_drift.json"
        bad_file.write_text("{ this is not valid json }")

        try:
            result = latest_drift_summary(monitoring_dir)
            # Should return None or dict, not raise
            assert result is None or isinstance(result, dict)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"latest_drift_summary raised on malformed JSON: {exc}")


# ---------------------------------------------------------------------------
# champion_info tests
# ---------------------------------------------------------------------------


class TestChampionInfo:
    """Tests for champion_info: empty-state + happy path via monkeypatching."""

    def test_returns_none_when_mlflow_unreachable(self) -> None:
        """When MLflow raises any error, return None without re-raising."""
        from volforecast.dashboard.data_access import champion_info

        with patch("volforecast.dashboard.data_access.MlflowClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_model_version_by_alias.side_effect = Exception("Connection refused")

            result = champion_info("http://unreachable:5000")

            assert result is None

    def test_returns_none_when_alias_unset(self) -> None:
        """When the alias raises MlflowException, return None."""
        from volforecast.dashboard.data_access import champion_info

        with patch("volforecast.dashboard.data_access.MlflowClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_model_version_by_alias.side_effect = Exception(
                "RESOURCE_DOES_NOT_EXIST"
            )

            result = champion_info("http://mlflow:5000")
            assert result is None

    def test_returns_dict_with_expected_keys_on_success(self) -> None:
        """When MLflow is reachable, return dict with version, run_id, params, metrics."""
        from volforecast.dashboard.data_access import champion_info

        mock_mv = MagicMock()
        mock_mv.version = "3"
        mock_mv.run_id = "abc123"

        mock_run = MagicMock()
        mock_run.data.params = {"n_estimators": "200", "learning_rate": "0.05"}
        mock_run.data.metrics = {"val_rmse": 0.0012, "val_qlike": -8.5}

        with patch("volforecast.dashboard.data_access.MlflowClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_model_version_by_alias.return_value = mock_mv
            mock_client.get_run.return_value = mock_run

            result = champion_info("http://mlflow:5000")

        assert result is not None
        assert isinstance(result, dict)
        assert result["version"] == "3"
        assert result["run_id"] == "abc123"
        assert isinstance(result["params"], dict)
        assert isinstance(result["metrics"], dict)

    def test_read_only_never_calls_set_alias(self) -> None:
        """champion_info must NEVER call set_registered_model_alias."""
        from volforecast.dashboard.data_access import champion_info

        with patch("volforecast.dashboard.data_access.MlflowClient") as MockClient:
            mock_client = MockClient.return_value
            mock_mv = MagicMock()
            mock_mv.version = "3"
            mock_mv.run_id = "abc123"
            mock_run = MagicMock()
            mock_run.data.params = {}
            mock_run.data.metrics = {}
            mock_client.get_model_version_by_alias.return_value = mock_mv
            mock_client.get_run.return_value = mock_run

            champion_info("http://mlflow:5000")

            mock_client.set_registered_model_alias.assert_not_called()


# ---------------------------------------------------------------------------
# prediction_stats tests
# ---------------------------------------------------------------------------


class TestPredictionStats:
    """Tests for prediction_stats: empty-state + happy path."""

    def test_missing_parquet_returns_empty_stats(self, tmp_path: Path) -> None:
        """When prediction log parquet does not exist, return zeroed-out stats."""
        from volforecast.dashboard.data_access import prediction_stats

        result = prediction_stats(tmp_path)

        assert isinstance(result, dict)
        assert result["row_count"] == 0
        assert result["last_timestamp"] is None
        assert isinstance(result["latest_per_asset"], dict)
        assert len(result["latest_per_asset"]) == 0

    def test_existing_parquet_returns_stats(self, tmp_path: Path) -> None:
        """When prediction log exists, return populated stats."""
        from volforecast.dashboard.data_access import prediction_stats

        df = _make_prediction_log(10)
        _write_prediction_log(tmp_path, df)

        result = prediction_stats(tmp_path)

        assert result["row_count"] == 10
        assert result["last_timestamp"] is not None
        assert isinstance(result["latest_per_asset"], dict)
        assert "BTC-USD" in result["latest_per_asset"]

    def test_does_not_raise_on_missing_parquet(self, tmp_path: Path) -> None:
        """prediction_stats must never raise on missing file."""
        from volforecast.dashboard.data_access import prediction_stats

        try:
            prediction_stats(tmp_path)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"prediction_stats raised on missing parquet: {exc}")


# ---------------------------------------------------------------------------
# api_health tests
# ---------------------------------------------------------------------------


class TestApiHealth:
    """Tests for api_health: unreachable + healthy response."""

    def test_connection_error_returns_not_reachable(self) -> None:
        """On connection error, return dict with reachable=False, no raise."""
        import urllib.error

        from volforecast.dashboard.data_access import api_health

        with patch("volforecast.dashboard.data_access.urllib") as mock_urllib:
            mock_urllib.request.urlopen.side_effect = urllib.error.URLError("refused")

            result = api_health("http://api:8000", timeout=1)

        assert isinstance(result, dict)
        assert result["reachable"] is False

    def test_timeout_returns_not_reachable(self) -> None:
        """On timeout, return dict with reachable=False."""

        from volforecast.dashboard.data_access import api_health

        with patch("volforecast.dashboard.data_access.urllib") as mock_urllib:
            mock_urllib.request.urlopen.side_effect = TimeoutError("timed out")

            result = api_health("http://api:8000", timeout=1)

        assert isinstance(result, dict)
        assert result["reachable"] is False

    def test_healthy_response_returns_populated_dict(self) -> None:
        """On healthy 200 response, return dict with reachable=True + status/version/alias."""
        from volforecast.dashboard.data_access import api_health

        health_json = json.dumps(
            {
                "status": "ok",
                "model_name": "volforecast-lgbm",
                "model_version": "3",
                "alias": "champion",
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = health_json
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("volforecast.dashboard.data_access.urllib") as mock_urllib:
            mock_urllib.request.urlopen.return_value = mock_resp

            result = api_health("http://api:8000", timeout=5)

        assert isinstance(result, dict)
        assert result["reachable"] is True
        assert result["status"] == "ok"
        assert result["model_version"] == "3"
        assert result["alias"] == "champion"
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], (int, float))

    def test_does_not_raise_on_connection_error(self) -> None:
        """api_health must never raise on any connection problem."""
        from volforecast.dashboard.data_access import api_health

        with patch("volforecast.dashboard.data_access.urllib") as mock_urllib:
            mock_urllib.request.urlopen.side_effect = Exception("Something unexpected")

            try:
                result = api_health("http://api:8000", timeout=1)
                assert result["reachable"] is False
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"api_health raised on connection error: {exc}")


# ---------------------------------------------------------------------------
# Import-level checks
# ---------------------------------------------------------------------------


class TestImportGuards:
    """Verify data_access module imports correct schemas and is read-only."""

    def test_data_access_imports_fvr_schema_from_labeller(self) -> None:
        """data_access.py must import FVR_SCHEMA from volforecast.monitoring.labeller."""
        import volforecast.dashboard.data_access as da

        # FVR_SCHEMA exported by data_access should match the canonical source
        from volforecast.monitoring.labeller import FVR_SCHEMA as canonical

        assert da.FVR_SCHEMA == canonical

    def test_data_access_imports_prediction_log_schema_from_serving(self) -> None:
        """data_access.py must import PREDICTION_LOG_SCHEMA from serving.prediction_log."""
        import volforecast.dashboard.data_access as da
        from volforecast.serving.prediction_log import PREDICTION_LOG_SCHEMA as canonical

        assert da.PREDICTION_LOG_SCHEMA == canonical

    def test_data_access_has_no_streamlit_import(self) -> None:
        """data_access.py must not import streamlit (must be unit-testable offline)."""
        import ast
        import importlib.util

        spec = importlib.util.find_spec("volforecast.dashboard.data_access")
        assert spec is not None
        source = Path(spec.origin).read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else ([node.module] if node.module else [])
                )
                for name in names:
                    assert name != "streamlit" and not (name or "").startswith("streamlit."), (
                        f"data_access.py must not import streamlit (found: {name})"
                    )
