"""Unit tests for src/volforecast/eval/metrics.py.

Tests are written BEFORE the implementation (TDD RED phase).
All tests verify:
- qlike(x, x) == 0 (THE mandatory unit test — Patton variance form)
- QLIKE asymmetry: qlike(a, b) != qlike(b, a) for a != b
- QLIKE non-negativity and no NaN/Inf on extreme inputs
- QLIKE floor clips near-zero forecast_var to prevent log(0)/div-by-zero
- rmse and mae match numpy reference implementations

No fixtures or network calls — all data is constructed inline.
"""

from __future__ import annotations

import numpy as np

from volforecast.eval.metrics import QLIKE_FLOOR, mae, qlike, rmse


class TestQlikePerfectForecast:
    """THE mandatory test: qlike(x, x) must equal 0 for any positive x."""

    def test_qlike_perfect_forecast_small(self) -> None:
        """qlike(x, x) == 0 for typical daily equity variance values."""
        x = np.array([0.001, 0.002, 0.0005])
        result = qlike(x, x)
        assert abs(result) < 1e-12, (
            f"qlike(x, x) must equal 0 within 1e-12; got {result}"
        )

    def test_qlike_perfect_forecast_single(self) -> None:
        """qlike on a single element must also equal 0 at perfect forecast."""
        x = np.array([0.0001])
        assert abs(qlike(x, x)) < 1e-12

    def test_qlike_perfect_forecast_varying_scales(self) -> None:
        """qlike(x, x) == 0 must hold across multiple scales (no unit error)."""
        for scale in [1e-6, 1e-4, 1e-2, 1.0]:
            x = np.array([scale, scale * 2, scale * 0.5])
            result = qlike(x, x)
            assert abs(result) < 1e-12, (
                f"qlike(x, x) failed at scale {scale}: got {result}"
            )


class TestQlikeAsymmetry:
    """QLIKE is asymmetric: qlike(a, b) != qlike(b, a) for a != b."""

    def test_qlike_is_asymmetric(self) -> None:
        """Confirming the directional property of QLIKE."""
        a = np.array([0.001, 0.002, 0.0015])
        b = np.array([0.0008, 0.0025, 0.001])
        q_ab = qlike(a, b)
        q_ba = qlike(b, a)
        assert q_ab != q_ba, (
            f"qlike(a, b) == qlike(b, a) = {q_ab:.6f}; QLIKE must be asymmetric"
        )


class TestQlikeNonNegativity:
    """QLIKE must never return negative or NaN/Inf values."""

    def test_qlike_non_negative(self) -> None:
        """For any reasonable input, QLIKE >= 0."""
        rv = np.array([0.001, 0.002, 0.0005, 0.003])
        forecast = np.array([0.0008, 0.0025, 0.0006, 0.002])
        result = qlike(rv, forecast)
        assert result >= 0.0, f"QLIKE must be non-negative; got {result}"

    def test_qlike_no_nan_on_extreme_forecast(self) -> None:
        """Near-zero forecast_var must be clipped to QLIKE_FLOOR, not produce NaN."""
        rv = np.array([0.001, 0.002])
        forecast_near_zero = np.array([1e-15, 1e-20])  # below QLIKE_FLOOR
        result = qlike(rv, forecast_near_zero)
        assert not np.isnan(result), "QLIKE returned NaN on near-zero forecast"
        assert not np.isinf(result), "QLIKE returned Inf on near-zero forecast"
        assert result >= 0.0, f"QLIKE must be non-negative; got {result}"

    def test_qlike_no_inf_on_zero_forecast(self) -> None:
        """Exactly-zero forecast_var must be clipped, not produce Inf."""
        rv = np.array([0.001])
        forecast_zero = np.array([0.0])
        result = qlike(rv, forecast_zero)
        assert not np.isinf(result), "QLIKE returned Inf for zero forecast_var"
        assert not np.isnan(result), "QLIKE returned NaN for zero forecast_var"


class TestQlikeFloor:
    """QLIKE_FLOOR must be defined and used to clip forecast_var."""

    def test_qlike_floor_exists(self) -> None:
        """QLIKE_FLOOR constant must be importable and equal 1e-10."""
        assert QLIKE_FLOOR == 1e-10, f"QLIKE_FLOOR must be 1e-10; got {QLIKE_FLOOR}"

    def test_qlike_floor_applied(self) -> None:
        """Clipping forecast to QLIKE_FLOOR must produce same result as clipping manually."""
        rv = np.array([0.001, 0.002])
        tiny_forecast = np.array([1e-15, 1e-20])
        floor_forecast = np.full_like(tiny_forecast, QLIKE_FLOOR)

        result_tiny = qlike(rv, tiny_forecast)
        result_floor = qlike(rv, floor_forecast)
        assert abs(result_tiny - result_floor) < 1e-12, (
            "QLIKE(rv, tiny) and QLIKE(rv, QLIKE_FLOOR) should match after clipping"
        )


class TestRmse:
    """rmse must match the standard numpy reference."""

    def test_rmse_zero_on_perfect_forecast(self) -> None:
        """rmse(x, x) == 0."""
        x = np.array([1.0, 2.0, 3.0])
        assert abs(rmse(x, x)) < 1e-14

    def test_rmse_matches_numpy(self) -> None:
        """rmse result must match np.sqrt(np.mean((y_true - y_pred)**2))."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.7])
        expected = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        assert abs(rmse(y_true, y_pred) - expected) < 1e-12

    def test_rmse_scalar_output(self) -> None:
        """rmse must return a scalar float."""
        result = rmse(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
        assert isinstance(result, float), f"rmse must return float; got {type(result)}"


class TestMae:
    """mae must match the standard numpy reference."""

    def test_mae_zero_on_perfect_forecast(self) -> None:
        """mae(x, x) == 0."""
        x = np.array([1.0, 2.0, 3.0])
        assert abs(mae(x, x)) < 1e-14

    def test_mae_matches_numpy(self) -> None:
        """mae result must match np.mean(np.abs(y_true - y_pred))."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.7])
        expected = float(np.mean(np.abs(y_true - y_pred)))
        assert abs(mae(y_true, y_pred) - expected) < 1e-12

    def test_mae_scalar_output(self) -> None:
        """mae must return a scalar float."""
        result = mae(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
        assert isinstance(result, float), f"mae must return float; got {type(result)}"
