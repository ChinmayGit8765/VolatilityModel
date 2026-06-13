"""Rolling-QLIKE champion-vs-GARCH performance monitor (MON-03).

Computes rolling 21-day QLIKE for the champion model and for the GARCH(1,1)
baseline over the ``forecast_vs_realized.parquet`` table produced by
``volforecast.monitoring.labeller``.  When the champion underperforms GARCH
by the documented relative threshold, a retrain flag is returned and an alert
is fired via the configurable alert channel (MON-04).

This is the **authoritative** retrain trigger — stronger than distribution
drift.  Distribution drift is logged/reported only; performance degradation
here is the signal that feeds the Prefect daily flow's conditional retrain
branch (Plan 05).

Constants and rationale
-----------------------
``PERF_WINDOW = 21``
    Rolling window size in trading days.  21 days ≈ one calendar month of
    daily data.  Large enough to distinguish signal from noise on QLIKE;
    small enough to detect degradation within a month of it starting.
    Choice: Claude's discretion per 04-CONTEXT.md.

``DEGRADATION_THRESHOLD = 0.10``
    Champion QLIKE must exceed GARCH QLIKE by more than 10% (relative) to
    trigger.  I.e., degraded = (champ_qlike > garch_qlike * 1.10).

    Rationale: 10% relative QLIKE underperformance over 21 observations is
    clearly meaningful — random QLIKE noise over a 21-day window rarely
    produces a *persistent* 10% gap.  Values below 5% would create retrain
    storms during normal variance; values above 25% would mask real regime
    shifts.  10% is the midpoint of the credible range and is explicitly
    documented so operators can tune it.  This is Claude's discretion per
    04-CONTEXT.md ("Exact threshold for performance degradation — document
    the choice").

Cold-start policy (Pitfall 3 from 04-RESEARCH.md)
--------------------------------------------------
The monitor requires at least ``PERF_WINDOW // 2`` rows for *both* champion
and garch_baseline before firing.  On day 1 of monitoring the parquet may
have only 1 row per alias; comparing them would produce a meaningless single-
point QLIKE.  Until the minimum sample size is met, the monitor returns
``(False, {"reason": "cold_start", ...})`` and fires no alert.

Key links
---------
- ``volforecast.eval.metrics.qlike``: canonical QLIKE (Patton 2011 variance
  form); floored at 1e-10, raises on non-finite inputs.  NEVER re-implemented
  here — imported and reused exactly (T-04-09).
- ``volforecast.monitoring.labeller.FVR_SCHEMA``: forecast_vs_realized schema;
  model_alias values ``"champion"`` and ``"garch_baseline"`` are the split keys.
- ``volforecast.monitoring.alerts.send_alert``: alert delivery function;
  injected as ``alert_fn`` for testability.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from volforecast.eval.metrics import qlike
from volforecast.monitoring.alerts import send_alert

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants — documented Claude's discretion (04-CONTEXT.md)
# ---------------------------------------------------------------------------

#: Rolling window in trading days (≈ one calendar month).
#: Rationale: long enough to distinguish signal from noise; short enough to
#: detect degradation within a month.  Claude's discretion per CONTEXT.md.
PERF_WINDOW: int = 21

#: Relative QLIKE degradation threshold.
#: Trigger condition: champ_qlike > garch_qlike * (1 + DEGRADATION_THRESHOLD).
#: 10% relative underperformance over 21 days is clearly meaningful
#: (random QLIKE noise rarely produces a persistent >10% gap).
#: Too low → retrain storms; too high → slow detection.  10% is the
#: documented midpoint of the credible [5%, 25%] range.
#: Claude's discretion per CONTEXT.md; calibrate with more data post-launch.
DEGRADATION_THRESHOLD: float = 0.10

#: Minimum rows per alias required before firing the monitor.
#: = PERF_WINDOW // 2 (10 rows).  Prevents cold-start false alerts.
_MIN_ROWS: int = PERF_WINDOW // 2

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_performance_drift(fvr: pd.DataFrame) -> tuple[bool, dict]:
    """Compare rolling champion vs GARCH QLIKE and return a retrain flag.

    Takes the most recent ``PERF_WINDOW`` rows of ``fvr`` (sorted by
    ``as_of_date``), splits into champion and garch_baseline slices, and
    computes QLIKE for each using the canonical
    ``volforecast.eval.metrics.qlike`` function (argument order:
    ``qlike(realized_var, forecast_var)``).

    Cold-start policy: If either alias has fewer than ``_MIN_ROWS``
    (= ``PERF_WINDOW // 2``) rows in the window, return
    ``(False, {"reason": "cold_start", ...})`` immediately — no alert.

    Degradation policy: ``degraded = champ_qlike > garch_qlike * (1 + DEGRADATION_THRESHOLD)``.

    Args:
        fvr: ``forecast_vs_realized`` DataFrame with columns
            ``["as_of_date", "asset", "horizon", "model_version",
             "model_alias", "forecast_var", "realized_var"]``.
            Must include rows with ``model_alias in {"champion", "garch_baseline"}``.

    Returns:
        ``(should_retrain: bool, report: dict)`` where ``report`` contains:

        On cold-start:
            ``{"reason": "cold_start", "n_champion": int, "n_garch": int}``

        On full evaluation:
            ``{"champion_qlike": float, "garch_qlike": float,
               "threshold": float, "n_champion": int, "n_garch": int,
               "degraded": bool}``
    """
    # Sort by as_of_date and take the most recent PERF_WINDOW rows per alias
    if fvr.empty:
        log.debug("check_performance_drift: empty fvr — cold start")
        return False, {"reason": "cold_start", "n_champion": 0, "n_garch": 0}

    # Sort all rows by date, take the tail window across both aliases combined.
    # This matches the "most recent 21 days" semantic — entries for different
    # assets may interleave, but the QLIKE is computed per-alias over the slice.
    fvr_sorted = fvr.sort_values("as_of_date")
    recent = fvr_sorted.tail(PERF_WINDOW * 2)  # upper bound; split below

    champion_rows = recent[recent["model_alias"] == "champion"]
    garch_rows = recent[recent["model_alias"] == "garch_baseline"]

    n_champion = len(champion_rows)
    n_garch = len(garch_rows)

    # Cold-start gate (Pitfall 3)
    if n_champion < _MIN_ROWS or n_garch < _MIN_ROWS:
        log.info(
            "check_performance_drift: cold_start (n_champion=%d, n_garch=%d, min=%d)",
            n_champion,
            n_garch,
            _MIN_ROWS,
        )
        return False, {
            "reason": "cold_start",
            "n_champion": n_champion,
            "n_garch": n_garch,
        }

    # Compute QLIKE using the canonical metric (T-04-09 — no re-implementation)
    champ_qlike = qlike(
        champion_rows["realized_var"].to_numpy(),
        champion_rows["forecast_var"].to_numpy(),
    )
    garch_qlike = qlike(
        garch_rows["realized_var"].to_numpy(),
        garch_rows["forecast_var"].to_numpy(),
    )

    degraded: bool = champ_qlike > garch_qlike * (1 + DEGRADATION_THRESHOLD)

    log.info(
        "check_performance_drift: champ_qlike=%.6f garch_qlike=%.6f degraded=%s",
        champ_qlike,
        garch_qlike,
        degraded,
    )

    return degraded, {
        "champion_qlike": champ_qlike,
        "garch_qlike": garch_qlike,
        "threshold": DEGRADATION_THRESHOLD,
        "n_champion": n_champion,
        "n_garch": n_garch,
        "degraded": degraded,
    }


def run_performance_monitor(
    fvr_path: Path,
    *,
    alert_fn: Callable[..., None] = send_alert,
) -> bool:
    """Read the forecast_vs_realized parquet and run the performance monitor.

    Convenience entry point for the Prefect daily flow (Plan 05).  Reads the
    parquet at ``fvr_path``, calls ``check_performance_drift``, and on
    degradation fires the alert via ``alert_fn``.

    The ``alert_fn`` is injected to allow test stubs (e.g., ``_AlertSpy``)
    without monkeypatching the real ``send_alert`` import.

    Args:
        fvr_path: Path to ``forecast_vs_realized.parquet``.
        alert_fn: Alert delivery callable with the same signature as
            ``volforecast.monitoring.alerts.send_alert``.  Defaults to the
            real ``send_alert``.

    Returns:
        ``True`` if performance degradation was detected (should retrain),
        ``False`` otherwise.

    Raises:
        FileNotFoundError: If ``fvr_path`` does not exist.
    """
    if not fvr_path.exists():
        raise FileNotFoundError(
            f"forecast_vs_realized parquet not found at {fvr_path}. "
            "Run the labeller (Plan 04-01) before the performance monitor."
        )

    fvr = pd.read_parquet(fvr_path)
    should_retrain, report = check_performance_drift(fvr)

    if should_retrain:
        alert_fn(
            title="Champion underperforms GARCH baseline — retraining recommended",
            body={
                "champion_qlike": report["champion_qlike"],
                "garch_qlike": report["garch_qlike"],
                "threshold": report["threshold"],
                "n_champion": report["n_champion"],
                "n_garch": report["n_garch"],
                "perf_window": PERF_WINDOW,
            },
        )
        log.warning(
            "run_performance_monitor: DEGRADATION DETECTED — champion_qlike=%.6f "
            "garch_qlike=%.6f (threshold=%.2f%%)",
            report["champion_qlike"],
            report["garch_qlike"],
            DEGRADATION_THRESHOLD * 100,
        )
    else:
        log.info(
            "run_performance_monitor: no degradation (reason=%s)",
            report.get("reason", "ok"),
        )

    return should_retrain
