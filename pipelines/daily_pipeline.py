"""Prefect 3 daily orchestration flow for VolForecast.

Wires ingest → validate → features → forecast → label → drift-check →
(conditional retrain → eval → register-challenger) → promotion-gate as thin
@task wrappers over existing scripts and the Plan 01-04 monitoring functions.

Task order is locked (per 04-CONTEXT.md, forecast_task added by the milestone
audit fix — the daily flow previously never generated forecasts, so label_task
was a silent no-op loop):
    ingest_validate_task
    → features_task
    → forecast_task
    → label_task
    → drift_check_task
    → performance_check_task  (retrain trigger — MON-03)
    → [conditional] retrain_task + eval_task
    → [conditional] promotion_gate_task

Retraining trigger (ORCH-02, Pitfall 5 from 04-RESEARCH.md):
    ``force_retrain`` is an EPHEMERAL flow parameter — it does not persist between
    runs.  The performance flag returned by ``performance_check_task`` is also
    computed fresh per run.  No flag file is ever left behind.

Challenger registration (T-04-15 / ORCH-03):
    ``retrain_task`` runs ``train_lgbm.py`` then IMMEDIATELY reassigns the freshly
    registered version from @champion to @challenger via MlflowClient, so the
    auto-champion set by train_lgbm.py is overridden before any other system
    touches the registry.  The @champion alias is only moved back by
    ``promotion_gate_task`` via ``promote_if_better`` (the single authorised path
    per Plan 04).

Drift check (locked anti-pattern):
    ``drift_check_task`` is REPORT-ONLY.  It writes dated HTML + JSON to
    ``data/monitoring/`` but returns no promotion signal and never sets a retrain
    flag.  Only ``performance_check_task`` returns the retrain flag.

All paths are resolved via ``volforecast.config.project_root()`` so the flow
works both on the host (VOLFORECAST_ROOT or cwd) and in the Prefect worker
container (VOLFORECAST_ROOT set in docker-compose.yml env).

Usage (offline test):
    from prefect.testing.utilities import prefect_test_harness
    with prefect_test_harness():
        result = daily_flow(force_retrain=False)

Usage (manual run via compose):
    PREFECT_API_URL=http://localhost:4200/api uv run prefect deployment run \\
        'volforecast-daily/volforecast-daily'
"""

from __future__ import annotations

import datetime
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.exceptions import MissingContextError


def _get_logger() -> logging.Logger:
    """Return a Prefect run logger if inside a Prefect context, else a stdlib logger.

    This allows task.fn() calls in tests (which have no Prefect context) to work
    without raising MissingContextError.
    """
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the project root: VOLFORECAST_ROOT env var or parent of pipelines/."""
    import os

    env_root = os.environ.get("VOLFORECAST_ROOT")
    if env_root:
        return Path(env_root)
    # __file__ is <repo>/pipelines/daily_pipeline.py -> parent.parent = repo root
    return Path(__file__).parent.parent


def _data_root() -> Path:
    return _repo_root() / "data"


def _fvr_path() -> Path:
    return _data_root() / "monitoring" / "forecast_vs_realized.parquet"


def _reference_path(champion_version: str) -> Path:
    """Return the frozen Evidently reference snapshot for the given champion version."""
    return _data_root() / "monitoring" / "reference" / f"{champion_version}_reference.parquet"


def _resolve_reference_path() -> Path | None:
    """Resolve the drift reference snapshot for the CURRENT champion version.

    Audit WARNING fix: the reference was previously hardcoded to version "3",
    so it went stale (and the drift check silently compared against the wrong
    distribution) whenever the champion moved.

    Resolution order:
      1. Ask MLflow for the current champion via
         ``MlflowClient.get_model_version_by_alias("volforecast-lgbm", "champion")``
         (``MLFLOW_TRACKING_URI`` env, default ``http://localhost:5000``) and
         use ``{version}_reference.parquet`` if that snapshot exists.
      2. On ANY exception (MLflow down, alias unset) — or when the resolved
         snapshot file is missing — fall back to the HIGHEST-versioned
         ``*_reference.parquet`` present in ``data/monitoring/reference/``.

    Logs which reference was chosen.  Returns None when no snapshot exists.
    """
    import os

    logger = _get_logger()

    try:
        import mlflow
        from mlflow import MlflowClient

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        mv = client.get_model_version_by_alias("volforecast-lgbm", "champion")
        champion_version = str(mv.version)
        candidate = _reference_path(champion_version)
        if candidate.exists():
            logger.info(
                "Drift reference: champion v%s (from MLflow) -> %s",
                champion_version,
                candidate,
            )
            return candidate
        logger.warning(
            "Champion is v%s but reference snapshot %s does not exist — "
            "falling back to the highest-versioned snapshot. "
            "Run scripts/create_reference_snapshot.py to freeze one for v%s.",
            champion_version,
            candidate,
            champion_version,
        )
    except Exception as exc:  # noqa: BLE001 — MLflow unreachable / alias unset
        logger.warning(
            "Could not resolve champion version from MLflow (%s) — "
            "falling back to the highest-versioned reference snapshot.",
            exc,
        )

    ref_dir = _data_root() / "monitoring" / "reference"
    candidates: list[tuple[int, Path]] = []
    if ref_dir.exists():
        for path in ref_dir.glob("*_reference.parquet"):
            stem = path.name.removesuffix("_reference.parquet")
            try:
                candidates.append((int(stem), path))
            except ValueError:
                continue
    if not candidates:
        logger.warning("No reference snapshots found in %s", ref_dir)
        return None

    version, path = max(candidates)
    logger.info("Drift reference: fallback highest-versioned snapshot v%d -> %s", version, path)
    return path


def _monitoring_dir() -> Path:
    return _data_root() / "monitoring"


# ---------------------------------------------------------------------------
# Helper: run a subprocess script, raise on non-zero exit
# ---------------------------------------------------------------------------


def _run_script(script_path: Path, description: str) -> subprocess.CompletedProcess:
    """Run a Python script via the current interpreter; raise on non-zero exit."""
    logger = _get_logger()
    logger.info("Running %s (%s)", description, script_path)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(
            "%s failed (exit %d):\nstdout=%s\nstderr=%s",
            description,
            result.returncode,
            result.stdout[-2000:],
            result.stderr[-2000:],
        )
        raise RuntimeError(
            f"{description} exited with code {result.returncode}. stderr: {result.stderr[-500:]}"
        )
    logger.info("%s OK: %s", description, result.stdout.strip()[-200:])
    return result


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(name="ingest-validate", retries=2, retry_delay_seconds=30)
def ingest_validate_task() -> None:
    """Invoke the volforecast ingest CLI (cache-first validated ingest of all assets).

    Validation is built into the ingest gate, so this single task covers
    ingest + validate.  Non-zero exit raises so Prefect surfaces a data-quality
    hard-fail (locked taxonomy: data-quality failure → pipeline hard-fail).
    """
    logger = _get_logger()
    logger.info("Starting ingest+validate via 'volforecast ingest' CLI")

    result = subprocess.run(
        [sys.executable, "-m", "volforecast.cli", "ingest"],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(
            "volforecast ingest failed (exit %d):\nstdout=%s\nstderr=%s",
            result.returncode,
            result.stdout[-2000:],
            result.stderr[-2000:],
        )
        raise RuntimeError(
            f"Ingest+validate failed with exit code {result.returncode}. "
            f"stderr: {result.stderr[-500:]}"
        )
    logger.info("Ingest+validate complete: %s", result.stdout.strip()[-200:])


@task(name="generate-features")
def features_task() -> None:
    """Run scripts/generate_features.py — writes per-asset feature parquets."""
    _run_script(_repo_root() / "scripts" / "generate_features.py", "generate_features")


@task(name="generate-forecasts", retries=2, retry_delay_seconds=60)
def forecast_task() -> int:
    """Call the serving API ``GET /forecast`` so live forecasts are generated + logged.

    This exercises the REAL serving path end-to-end: champion model load, the
    single ``build_features`` codepath, and the prediction-log append with
    model-version metadata.  The API is reached over the compose network via
    ``VOLFORECAST_API_URL`` (default ``http://api:8000`` — the prefect-worker
    container's view of the ``api`` service).

    A dead serving layer must fail the flow LOUDLY: any HTTP error or
    connection failure raises (after retries=2 with 60s delay) — it is never
    silently skipped.  The milestone audit's core finding was a silent no-op
    loop where label_task had nothing to label because nothing ever called
    ``/forecast``.

    Timeout is 300s: GARCH-as-a-feature makes ``/forecast`` take ~40-60s
    per call.

    Returns:
        Number of forecasts returned by the API.
    """
    import json
    import os
    import urllib.request

    logger = _get_logger()
    api_base = os.environ.get("VOLFORECAST_API_URL", "http://api:8000")
    url = f"{api_base.rstrip('/')}/forecast"
    logger.info("Requesting live forecasts: GET %s (timeout=300s)", url)

    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310 — internal compose URL
        body = resp.read()

    payload = json.loads(body)
    forecasts = payload.get("forecasts", []) if isinstance(payload, dict) else []
    n_forecasts = len(forecasts)
    logger.info(
        "Serving API returned %d forecasts (appended to the prediction log "
        "with model-version metadata)",
        n_forecasts,
    )
    return n_forecasts


@task(name="label-forecasts")
def label_task() -> int:
    """Call label_forecasts(data_root, fvr_path) — returns net-new row count.

    Reads predictions.parquet + processed data, appends new rows to
    forecast_vs_realized.parquet (idempotent via LABEL_KEY dedup).
    """
    logger = _get_logger()
    from volforecast.monitoring.labeller import label_forecasts

    data_root = _data_root()
    fvr_path = _fvr_path()
    logger.info("Labelling forecasts: data_root=%s fvr_path=%s", data_root, fvr_path)
    net_new = label_forecasts(data_root, fvr_path)
    logger.info("Label task complete: net_new_rows=%d", net_new)
    return net_new


@task(name="drift-check")
def drift_check_task() -> str:
    """Run Evidently distribution drift report — REPORT-ONLY, no retrain signal.

    Reads the frozen reference snapshot for the current champion version and
    compares it against recent feature distributions.  Writes dated HTML + JSON
    to data/monitoring/.  Returns the output directory path as a string for
    observability logging.

    This task returns NO promotion signal and does NOT trigger retraining.
    Only ``performance_check_task`` provides the retrain flag (locked anti-pattern
    per CONTEXT.md / T-04-04).
    """
    import pandas as pd

    from volforecast.monitoring.drift import run_distribution_drift, select_numerical_columns

    logger = _get_logger()
    date_str = datetime.date.today().isoformat()
    monitoring_dir = _monitoring_dir()

    # --- Load reference snapshot (frozen, never updated automatically) ---
    # Resolved against the CURRENT champion version (audit WARNING fix: was
    # hardcoded to v3), with fallback to the highest-versioned snapshot.
    ref_path = _resolve_reference_path()
    if ref_path is None or not ref_path.exists():
        logger.warning(
            "No usable reference snapshot found under %s — skipping drift check. "
            "Run scripts/create_reference_snapshot.py to create one.",
            _data_root() / "monitoring" / "reference",
        )
        return str(monitoring_dir)

    reference_df = pd.read_parquet(ref_path)
    numerical_columns = select_numerical_columns(reference_df)

    if not numerical_columns:
        logger.warning("No numerical columns in reference snapshot — skipping drift check.")
        return str(monitoring_dir)

    # --- Load current feature window ---
    feat_dfs: list[pd.DataFrame] = []
    features_root = _data_root() / "features"
    for class_dir in ("crypto", "equity"):
        class_path = features_root / class_dir
        if not class_path.exists():
            continue
        for parquet_file in sorted(class_path.glob("*.parquet")):
            try:
                df = pd.read_parquet(parquet_file)
                feat_dfs.append(df)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", parquet_file, exc)

    if not feat_dfs:
        logger.warning("No feature parquets found — skipping drift check.")
        return str(monitoring_dir)

    current_df = pd.concat(feat_dfs, ignore_index=True)

    # Keep only columns present in both frames
    common_cols = [c for c in numerical_columns if c in current_df.columns]
    if not common_cols:
        logger.warning(
            "No common numerical columns between reference and current features — "
            "skipping drift check."
        )
        return str(monitoring_dir)

    logger.info(
        "Running Evidently drift check: %d columns, date=%s, output=%s",
        len(common_cols),
        date_str,
        monitoring_dir,
    )
    try:
        run_distribution_drift(
            reference_df=reference_df,
            current_df=current_df,
            numerical_columns=common_cols,
            output_dir=monitoring_dir,
            date_str=date_str,
        )
        logger.info("Drift report written: %s/%s_drift.{html,json}", monitoring_dir, date_str)
    except Exception as exc:
        # Drift check failure is non-fatal — log and continue
        logger.error("Drift check failed (non-fatal): %s", exc)

    return str(monitoring_dir)


@task(name="performance-check")
def performance_check_task() -> bool:
    """Call run_performance_monitor(fvr_path) — returns the should_retrain flag.

    This is the authoritative retrain trigger (MON-03).  Returns True when the
    champion's rolling QLIKE degrades beyond DEGRADATION_THRESHOLD relative to
    GARCH baseline over PERF_WINDOW days.  False when cold-start or no degradation.
    """
    from volforecast.monitoring.performance import run_performance_monitor

    logger = _get_logger()
    fvr_path = _fvr_path()

    if not fvr_path.exists():
        logger.info(
            "forecast_vs_realized.parquet not found at %s — "
            "cold-start: skipping performance check, retrain=False",
            fvr_path,
        )
        return False

    logger.info("Running performance monitor against %s", fvr_path)
    should_retrain = run_performance_monitor(fvr_path)
    logger.info("Performance check complete: should_retrain=%s", should_retrain)
    return should_retrain


@task(name="retrain")
def retrain_task() -> str:
    """Run train_lgbm.py then reassign the freshly registered version to @challenger.

    train_lgbm.py sets @champion on every run (Phase 3 behaviour).  Immediately
    after it completes, this task uses MlflowClient to:
      1. Read the current @champion version (just registered by train_lgbm.py).
      2. Assign that version the @challenger alias.
      3. Attempt to restore @champion to the previous version (from tag).

    This ensures the promotion gate is the ONLY path to @champion during the
    feedback loop (T-04-15).  Returns the new challenger version string.
    """
    import os

    import mlflow
    from mlflow import MlflowClient

    logger = _get_logger()

    # --- Run train_lgbm.py ---
    _run_script(_repo_root() / "scripts" / "train_lgbm.py", "train_lgbm")

    # --- Reassign newly registered version from @champion to @challenger ---
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_name = "volforecast-lgbm"

    try:
        new_mv = client.get_model_version_by_alias(model_name, "champion")
        new_version = new_mv.version
        logger.info(
            "train_lgbm.py registered version %s as @champion — "
            "reassigning to @challenger (T-04-15)",
            new_version,
        )
        # Assign @challenger to the new version
        client.set_registered_model_alias(model_name, "challenger", new_version)
        logger.info("Set @challenger -> version %s", new_version)

        # Restore @champion to the previous champion version (if tagged)
        try:
            prev_version = new_mv.tags.get("previous_champion_version")
        except Exception:
            prev_version = None

        if prev_version:
            client.set_registered_model_alias(model_name, "champion", prev_version)
            logger.info("Restored @champion -> version %s (previous champion)", prev_version)
        else:
            logger.warning(
                "No previous_champion_version tag found on v%s — "
                "@champion remains on v%s temporarily. "
                "Promotion gate controls the final assignment.",
                new_version,
                new_version,
            )
    except Exception as exc:
        logger.error(
            "Failed to reassign @champion -> @challenger after retrain: %s. "
            "Manual intervention may be needed to restore the @champion alias.",
            exc,
        )
        raise

    return str(new_version)


@task(name="eval")
def eval_task() -> None:
    """Run scripts/eval_lgbm.py — generates ml_vs_baselines comparison report."""
    _run_script(_repo_root() / "scripts" / "eval_lgbm.py", "eval_lgbm")


@task(name="promotion-gate")
def promotion_gate_task(challenger_version: str) -> bool:
    """Compare champion vs challenger QLIKE on a frozen window; promote if better.

    Reads forecast_vs_realized.parquet, computes QLIKE for both @champion and
    @challenger over the last (PROMOTION_COOLDOWN_DAYS * 3) days, then calls
    ``promote_if_better``.  Default outcome is no-promote (T-04-15).

    Returns True if the challenger was promoted to @champion, False otherwise.
    """
    import os

    import mlflow
    import pandas as pd
    from mlflow import MlflowClient

    from volforecast.monitoring.promotion import (
        PROMOTION_COOLDOWN_DAYS,
        frozen_window_qlike,
        promote_if_better,
    )

    logger = _get_logger()
    fvr_path = _fvr_path()
    today = datetime.date.today()

    if not fvr_path.exists():
        logger.info(
            "forecast_vs_realized.parquet not found — skipping promotion gate (no-promote)."
        )
        return False

    fvr = pd.read_parquet(fvr_path)

    # Frozen window: last PROMOTION_COOLDOWN_DAYS * 3 calendar days
    window_days = PROMOTION_COOLDOWN_DAYS * 3
    window_end = today - datetime.timedelta(days=1)
    window_start = window_end - datetime.timedelta(days=window_days)

    try:
        champ_qlike, champ_prov = frozen_window_qlike(
            fvr, "champion", str(window_start), str(window_end)
        )
        logger.info(
            "Champion QLIKE=%.6f over [%s, %s], n_rows=%d",
            champ_qlike,
            window_start,
            window_end,
            champ_prov["n_rows"],
        )
    except ValueError as exc:
        logger.warning("Cannot compute champion QLIKE: %s — no-promote (default).", exc)
        return False

    try:
        chall_qlike, chall_prov = frozen_window_qlike(
            fvr, "challenger", str(window_start), str(window_end)
        )
        logger.info(
            "Challenger QLIKE=%.6f over [%s, %s], n_rows=%d",
            chall_qlike,
            window_start,
            window_end,
            chall_prov["n_rows"],
        )
    except ValueError as exc:
        logger.warning("Cannot compute challenger QLIKE: %s — no-promote (default).", exc)
        return False

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    promoted = promote_if_better(
        challenger_qlike=chall_qlike,
        champion_qlike=champ_qlike,
        last_promotion_date=None,
        today=today,
        challenger_version=int(challenger_version),
        client=client,
    )
    if promoted:
        logger.info(
            "PROMOTED: @champion -> challenger version %s (QLIKE %.6f < champion %.6f)",
            challenger_version,
            chall_qlike,
            champ_qlike,
        )
    else:
        logger.info(
            "No promotion: challenger QLIKE %.6f >= champion QLIKE %.6f or cooldown "
            "not elapsed — default no-promote.",
            chall_qlike,
            champ_qlike,
        )
    return promoted


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@flow(name="volforecast-daily")
def daily_flow(force_retrain: bool = False) -> dict[str, Any]:
    """Daily VolForecast orchestration flow (ORCH-01, ORCH-02).

    Chains ingest → features → forecast → label → drift-check →
    performance-check → (conditional retrain → eval → promotion-gate).

    Args:
        force_retrain: Ephemeral flow parameter — if True, runs retrain regardless
            of the performance monitor flag.  NOT stored between runs (Pitfall 5).
            Set via the Prefect UI "Quick run" or CLI `--param force_retrain=true`.

    Returns:
        Summary dict with keys: n_forecasts, rows_labelled, drift_report_path,
        should_retrain, challenger_version, promoted, run_date.  Logged for
        observability.
    """
    logger = _get_logger()
    logger.info("daily_flow starting — force_retrain=%s", force_retrain)

    # 1. Ingest + validate (retries=2, retry_delay_seconds=30)
    ingest_validate_task()

    # 2. Generate features
    features_task()

    # 3. Generate live forecasts via the serving API (audit BLOCKER-1 fix):
    #    exercises champion load + prediction-log append so label_task always
    #    has fresh rows to label.  A dead serving layer fails the flow loudly.
    n_forecasts = forecast_task()

    # 4. Label forecasts — returns net-new row count
    rows_labelled = label_task()

    # 5. Drift check (report-only — returns output dir path, no retrain signal)
    drift_report_path = drift_check_task()

    # 6. Performance check — authoritative retrain trigger (MON-03)
    should_retrain_flag = performance_check_task()
    should_retrain = should_retrain_flag or force_retrain

    # 7. Conditional retrain path
    challenger_version: str | None = None
    promoted = False

    if should_retrain:
        logger.info(
            "Retrain triggered (perf_flag=%s, force_retrain=%s) — "
            "running retrain + eval + promotion-gate",
            should_retrain_flag,
            force_retrain,
        )
        # 7a. Retrain + register as @challenger (never auto-champion)
        challenger_version = retrain_task()

        # 7b. Eval report
        eval_task()

        # 7c. Promotion gate (only called when retraining happened — default no-promote)
        promoted = promotion_gate_task(challenger_version)
    else:
        logger.info(
            "No retrain needed (perf_flag=%s, force_retrain=%s) — "
            "skipping retrain + promotion gate",
            should_retrain_flag,
            force_retrain,
        )

    summary: dict[str, Any] = {
        "n_forecasts": n_forecasts,
        "rows_labelled": rows_labelled,
        "drift_report_path": drift_report_path,
        "should_retrain": should_retrain,
        "challenger_version": challenger_version,
        "promoted": promoted,
        "run_date": datetime.date.today().isoformat(),
    }
    logger.info("daily_flow complete: %s", summary)
    return summary
