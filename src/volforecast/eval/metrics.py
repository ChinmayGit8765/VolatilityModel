"""Canonical evaluation metrics for realized volatility forecasting.

This module is the single source of truth for all evaluation metrics used in
VolForecast.  It is imported by:
- Phase 2: baseline evaluation (EWMA, GARCH, HAR-RV)
- Phase 3: ML model evaluation (LightGBM)
- Phase 4: champion/challenger promotion gate

UNIT CONTRACT: All metrics expect inputs in the same unit — daily decimal
variance (e.g., ~1e-4 for typical daily equity).  Do NOT pass:
- Percent variance (100x too large)
- Standard deviation (must square to variance first)
- Annualized values

QLIKE argument order: qlike(realized, forecast) — the realized variance is the
first argument, matching the statistical convention that loss is measured with
respect to the realization.  QLIKE is asymmetric: qlike(a, b) != qlike(b, a).

IMPORTANT — arch API note:
    arch's `res.conditional_volatility` returns STANDARD DEVIATION, not variance.
    You MUST square it before passing to qlike:
        forecast_var = res.conditional_volatility.iloc[-1] ** 2

QLIKE formula (Patton 2011, variance form):
    QLIKE(σ², h) = mean(σ²/h - ln(σ²/h) - 1)

    where σ² = realized variance (rv_var) and h = forecast variance (forecast_var).

    Verification: at perfect forecast h = σ²:
        σ²/σ² - ln(σ²/σ²) - 1 = 1 - 0 - 1 = 0  ✓

    This form satisfies qlike(x, x) == 0 for all positive x.

    The alternative form log(h) + σ²/h does NOT satisfy qlike(x, x) == 0
    (it equals 0 only when σ² = e^{-1} ≈ 0.368) — do not use it.
"""

from __future__ import annotations

import numpy as np

# Minimum forecast variance value — clips near-zero or zero forecast_var to
# prevent log(0) or division-by-zero in QLIKE.
# Value chosen to be well below any realistic realized daily variance
# (typical daily equity ~1e-4, crypto ~5e-4).
QLIKE_FLOOR: float = 1e-10


def qlike(rv_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """Canonical QLIKE loss in the Patton (2011) variance form.

    QLIKE = mean(ratio - ln(ratio) - 1)  where ratio = rv_var / max(forecast_var, QLIKE_FLOOR)

    Properties:
    - qlike(x, x) == 0 for all positive x  (mandatory identity)
    - qlike(rv, h) >= 0 for all positive inputs  (non-negativity)
    - qlike(rv, h) != qlike(h, rv) in general  (asymmetric)
    - Robust to noisy variance proxies (Patton 2011)

    Args:
        rv_var: Realized variance (proxy). Array-like of positive floats.
                Units: daily decimal variance (e.g., ~1e-4 for equity).
        forecast_var: Forecast variance. Array-like of positive floats.
                      Clipped to QLIKE_FLOOR to prevent log(0)/div-by-zero.
                      Units must match rv_var.

    Returns:
        Scalar mean QLIKE loss. Always >= 0.

    Example:
        >>> import numpy as np
        >>> x = np.array([0.001, 0.002, 0.0005])
        >>> abs(qlike(x, x)) < 1e-12
        True
    """
    rv = np.asarray(rv_var, dtype=float)
    h = np.maximum(np.asarray(forecast_var, dtype=float), QLIKE_FLOOR)
    ratio = rv / h
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Standard RMSE: sqrt(mean((y_true - y_pred)^2)).

    Args:
        y_true: Realized values. Array-like of floats.
        y_pred: Predicted values. Array-like of floats.

    Returns:
        Scalar RMSE. Always >= 0.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Standard MAE: mean(|y_true - y_pred|).

    Args:
        y_true: Realized values. Array-like of floats.
        y_pred: Predicted values. Array-like of floats.

    Returns:
        Scalar MAE. Always >= 0.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))
