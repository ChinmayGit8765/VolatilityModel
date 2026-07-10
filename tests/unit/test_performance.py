"""Offline tests for the rolling-QLIKE champion-vs-GARCH performance monitor.

All tests are hermetic: synthetic forecast_vs_realized DataFrames only, no
real data files, no network calls.  A stub ``alert_fn`` verifies call counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic FVR frame builders
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(99)

#: Column names matching FVR_SCHEMA from labeller.py
FVR_COLS = [
    "as_of_date",
    "asset",
    "horizon",
    "model_version",
    "model_alias",
    "forecast_var",
    "realized_var",
]


def _make_fvr(
    n_days: int,
    champ_forecast_scale: float = 1.0,
    garch_forecast_scale: float = 1.0,
    realized_base: float = 5e-4,
) -> pd.DataFrame:
    """Build a synthetic forecast_vs_realized frame with champion + garch rows.

    Both models produce forecasts for the same ``n_days`` dates.  The
    ``forecast_scale`` parameters control how far the forecasts are from the
    realized values:

    - scale == 1.0: perfect forecast (qlike ≈ 0)
    - scale >> 1.0 (e.g., 100.0): very poor forecast (qlike >> 0)
    - scale << 1.0 (e.g., 0.01): also poor (qlike >> 0, under-prediction)

    Args:
        n_days: Number of trading days (dates).
        champ_forecast_scale: Multiplier applied to realized variance for the
            champion forecast. scale=1.0 → perfect.
        garch_forecast_scale: Multiplier applied to realized variance for the
            GARCH baseline forecast. scale=1.0 → perfect.
        realized_base: Base level of realized variance (daily decimal, ~5e-4).

    Returns:
        DataFrame with FVR_COLS columns and ``2 * n_days`` rows
        (n_days champion + n_days garch_baseline).
    """
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B", tz="UTC")
    realized = RNG.uniform(realized_base * 0.5, realized_base * 2, size=n_days)

    champ_rows = pd.DataFrame(
        {
            "as_of_date": dates,
            "asset": "BTC-USD",
            "horizon": 1,
            "model_version": "3",
            "model_alias": "champion",
            "forecast_var": realized * champ_forecast_scale,
            "realized_var": realized,
        }
    )
    garch_rows = pd.DataFrame(
        {
            "as_of_date": dates,
            "asset": "BTC-USD",
            "horizon": 1,
            "model_version": "garch_1_1",
            "model_alias": "garch_baseline",
            "forecast_var": realized * garch_forecast_scale,
            "realized_var": realized,
        }
    )
    return pd.concat([champ_rows, garch_rows], ignore_index=True)[FVR_COLS]


class _AlertSpy:
    """Stub alert function that records each invocation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, title: str, body: dict, **kwargs: Any) -> None:  # noqa: ANN401
        self.calls.append({"title": title, "body": body, **kwargs})

    @property
    def call_count(self) -> int:
        return len(self.calls)


# ---------------------------------------------------------------------------
# Test 1: cold-start — fewer than PERF_WINDOW//2 rows → (False, cold_start)
# ---------------------------------------------------------------------------


class TestColdStart:
    """Test 1: cold-start gate prevents false alert when history is too short."""

    def test_cold_start_returns_false_no_alert(self) -> None:
        """Frame with < PERF_WINDOW//2 champion or garch rows returns (False, cold_start)."""
        from volforecast.monitoring.performance import PERF_WINDOW, check_performance_drift

        # Build a frame with fewer than PERF_WINDOW//2 rows per alias
        n_short = max(1, PERF_WINDOW // 2 - 1)
        fvr = _make_fvr(n_days=n_short)

        should_retrain, report = check_performance_drift(fvr)

        assert should_retrain is False, "cold_start must return should_retrain=False"
        assert "reason" in report, "cold_start report must include 'reason' key"
        assert report["reason"] == "cold_start", (
            f"Expected reason='cold_start', got {report['reason']!r}"
        )

    def test_cold_start_with_zero_rows(self) -> None:
        """Empty frame must return (False, cold_start) without raising."""
        from volforecast.monitoring.performance import check_performance_drift

        empty = pd.DataFrame(columns=FVR_COLS)
        should_retrain, report = check_performance_drift(empty)
        assert should_retrain is False
        assert report.get("reason") == "cold_start"


# ---------------------------------------------------------------------------
# Test 2: degradation trigger — champion QLIKE > 1.10 * GARCH QLIKE
# ---------------------------------------------------------------------------


class TestDegradationTrigger:
    """Test 2: champion underperforms GARCH by > DEGRADATION_THRESHOLD → retrain=True."""

    def test_degraded_champion_triggers_retrain(self) -> None:
        """Champion forecast far from realized, GARCH near realized → should_retrain=True."""
        from volforecast.monitoring.performance import (
            DEGRADATION_THRESHOLD,
            PERF_WINDOW,
            check_performance_drift,
        )

        # Need at least PERF_WINDOW//2 rows per alias
        n_days = PERF_WINDOW  # full window for reliable signal

        # Champion: forecast = 50x realized → very high QLIKE
        # GARCH:    forecast = 1x realized → near-zero QLIKE
        fvr = _make_fvr(
            n_days=n_days,
            champ_forecast_scale=50.0,  # terrible champion forecast
            garch_forecast_scale=1.0,  # perfect GARCH forecast
        )

        should_retrain, report = check_performance_drift(fvr)

        assert should_retrain is True, (
            "Expected should_retrain=True "
            f"(champion QLIKE >> GARCH QLIKE by >{DEGRADATION_THRESHOLD:.0%})"
        )
        assert "champion_qlike" in report, "report must include champion_qlike"
        assert "garch_qlike" in report, "report must include garch_qlike"
        assert "threshold" in report, "report must include threshold"
        assert report["champion_qlike"] > report["garch_qlike"] * (1 + DEGRADATION_THRESHOLD), (
            "champion_qlike must exceed garch_qlike * (1 + threshold)"
        )

    def test_degraded_triggers_exactly_one_alert(self, tmp_path: Path) -> None:
        """run_performance_monitor fires exactly one alert on degradation."""
        from volforecast.monitoring.performance import PERF_WINDOW, run_performance_monitor

        n_days = PERF_WINDOW
        fvr = _make_fvr(n_days=n_days, champ_forecast_scale=50.0, garch_forecast_scale=1.0)

        # Write synthetic FVR to parquet
        fvr_path = tmp_path / "forecast_vs_realized.parquet"
        fvr.to_parquet(fvr_path, index=False)

        spy = _AlertSpy()
        result = run_performance_monitor(fvr_path, alert_fn=spy)

        assert result is True, "run_performance_monitor must return True on degradation"
        assert spy.call_count == 1, (
            f"Expected exactly 1 alert call on degradation, got {spy.call_count}"
        )


# ---------------------------------------------------------------------------
# Test 3: no trigger — champion comparable to or better than GARCH
# ---------------------------------------------------------------------------


class TestNoTrigger:
    """Test 3: champion comparable/better → should_retrain=False, zero alerts."""

    def test_comparable_champion_no_trigger(self) -> None:
        """Champion and GARCH with equal quality → should_retrain=False."""
        from volforecast.monitoring.performance import PERF_WINDOW, check_performance_drift

        n_days = PERF_WINDOW
        # Both models near-perfect → QlIKE close to 0 for both
        fvr = _make_fvr(n_days=n_days, champ_forecast_scale=1.0, garch_forecast_scale=1.0)

        should_retrain, report = check_performance_drift(fvr)

        assert should_retrain is False, "Equal-quality models must NOT trigger retrain"
        # Must include QLIKE values in report
        assert "champion_qlike" in report
        assert "garch_qlike" in report

    def test_better_champion_no_alert(self, tmp_path: Path) -> None:
        """Champion better than GARCH → run_performance_monitor fires zero alerts."""
        from volforecast.monitoring.performance import PERF_WINDOW, run_performance_monitor

        n_days = PERF_WINDOW
        # Champion: 1x (perfect). GARCH: 50x (terrible).
        fvr = _make_fvr(n_days=n_days, champ_forecast_scale=1.0, garch_forecast_scale=50.0)
        fvr_path = tmp_path / "fvr.parquet"
        fvr.to_parquet(fvr_path, index=False)

        spy = _AlertSpy()
        result = run_performance_monitor(fvr_path, alert_fn=spy)

        assert result is False, "run_performance_monitor must return False when champion wins"
        assert spy.call_count == 0, (
            f"Expected 0 alert calls when champion wins, got {spy.call_count}"
        )


# ---------------------------------------------------------------------------
# Test 4: canonical metric reuse — qlike() from eval/metrics is used
# ---------------------------------------------------------------------------


class TestCanonicalMetricReuse:
    """Test 4: monitor uses volforecast.eval.metrics.qlike, not a re-implementation."""

    def test_qlike_canonical_values_match(self) -> None:
        """Known-input assertion: monitor QLIKE values match direct qlike() output."""
        from volforecast.eval.metrics import qlike
        from volforecast.monitoring.performance import PERF_WINDOW, check_performance_drift

        n_days = PERF_WINDOW
        # Use perfectly-controlled synthetic arrays (no randomness in this test)
        realized = np.full(n_days, 5e-4)  # constant realized variance
        champ_forecast = realized * 3.0  # champion over-forecasts 3x → some QLIKE
        garch_forecast = realized * 1.0  # GARCH perfect → QLIKE ≈ 0

        fvr = pd.DataFrame(
            {
                "as_of_date": pd.date_range("2025-01-01", periods=n_days, freq="B", tz="UTC"),
                "asset": "SPY",
                "horizon": 1,
                "model_version": "3",
                "model_alias": "champion",
                "forecast_var": champ_forecast,
                "realized_var": realized,
            }
        )
        fvr_garch = pd.DataFrame(
            {
                "as_of_date": pd.date_range("2025-01-01", periods=n_days, freq="B", tz="UTC"),
                "asset": "SPY",
                "horizon": 1,
                "model_version": "garch_1_1",
                "model_alias": "garch_baseline",
                "forecast_var": garch_forecast,
                "realized_var": realized,
            }
        )
        fvr_full = pd.concat([fvr, fvr_garch], ignore_index=True)

        _, report = check_performance_drift(fvr_full)

        # Compute expected values using canonical qlike directly
        expected_champ_qlike = qlike(realized, champ_forecast)
        expected_garch_qlike = qlike(realized, garch_forecast)

        assert report.get("champion_qlike") == pytest.approx(expected_champ_qlike, rel=1e-6), (
            f"monitor champion_qlike {report.get('champion_qlike')} != "
            f"canonical qlike {expected_champ_qlike}"
        )
        assert report.get("garch_qlike") == pytest.approx(expected_garch_qlike, abs=1e-10), (
            f"monitor garch_qlike {report.get('garch_qlike')} != "
            f"canonical qlike {expected_garch_qlike}"
        )
