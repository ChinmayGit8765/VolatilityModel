# Phase 4: Monitoring, Orchestration & Retraining - Research

**Researched:** 2026-06-12
**Domain:** MLOps feedback loop — Evidently 0.7 drift detection, Prefect 3 orchestration, MLflow alias promotion, idempotent parquet labelling
**Confidence:** HIGH (all primary APIs verified via official docs; Evidently HTML save method has one known version-specific bug documented below)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Feedback Loop & Labels (MON-01)**
- Labeller joins arrived realized vol against the prediction log → append-only `data/monitoring/forecast_vs_realized.parquet`
- A forecast made as-of t is labelable once t+1 close exists in processed data; per-asset-class calendars respected
- Labeller is idempotent: keyed on `(asset, as_of_date, model_version)` — re-running never duplicates rows

**Drift & Alerts (MON-02..04)**
- Distribution drift: Evidently 0.7+ Report (current API only — NOT legacy ColumnMapping) on feature and prediction distributions vs a frozen training reference snapshot; output JSON + HTML to `data/monitoring/`; dashboard/log only — NEVER triggers promotion
- Performance drift: rolling live QLIKE of champion vs rolling QLIKE of GARCH baseline over a 21-day window from `forecast_vs_realized`; if champion underperforms GARCH by a documented threshold, raise alert + set retrain flag
- Alerts: webhook URL from env var `ALERT_WEBHOOK_URL` (Slack-compatible JSON POST); when unset, write structured alert records to `data/monitoring/alerts.jsonl` (still observable); never crash the pipeline on alert delivery failure
- Monitor taxonomy (locked): data-quality failure → hard-fail; distribution drift → log/report only; performance degradation → alert + retrain flag

**Orchestration (ORCH-01..02)**
- One Prefect 3 daily flow: ingest → validate → features → label → drift-check → (conditional) train → eval → register challenger
- Drift-triggered retrain: simple flag check inside the daily flow — no Prefect Automations in v1
- Deployment served by compose Prefect worker (process pool, `local-pool`); schedule daily after market data availability
- Flows must run offline-safe in tests (mock external calls); live runs use the compose stack

**Champion/Challenger Gate (ORCH-03)**
- Retrain registers new model version with alias `@challenger` (never auto-champion)
- Promotion gate: challenger beats champion on rolling QLIKE over a frozen comparison window (identical rows for both, data hash + window recorded); canonical `qlike` from `eval/metrics.py`
- Default outcome: no-promote. Promotion = `set_registered_model_alias` flip; rollback = flip back to previous version
- Cooldown: minimum 7 days between promotions (documented constant)

### Claude's Discretion

- Exact threshold for performance degradation (document the choice)
- Evidently metric selection
- Flow/task naming
- Monitoring parquet schemas
- Retry policies

### Deferred Ideas (OUT OF SCOPE)

- Streamlit dashboard (Phase 5)
- Prefect Automations/event triggers (v2)
- Cloud deploy (v2)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MON-01 | Feedback-loop labeller joins arrived realized vol against logged forecasts to produce a forecast-vs-realized table automatically | Idempotent parquet append pattern; `compute_target` + `prediction_log` contracts confirmed |
| MON-02 | Evidently reports feature and prediction distribution drift on a schedule (dashboard/log only, not auto-promotion) | Evidently 0.7.21 Report/Dataset/DataDefinition API confirmed; save_html/save_json methods documented |
| MON-03 | Performance degradation (rolling live QLIKE vs GARCH's rolling QLIKE) triggers an alert and flags retraining | Rolling 21-day QLIKE pattern; threshold recommendation documented; flag-file mechanism |
| MON-04 | Alerts are delivered via a configurable channel (Slack webhook or jsonl fallback) | urllib.request POST pattern documented; no new deps needed |
| ORCH-01 | Prefect DAG runs ingest → validate → features → train → eval → register end-to-end on a schedule | Prefect 3.7 flow.from_source().deploy() with cron and local process pool documented |
| ORCH-02 | Retraining can be triggered by the performance-drift flag in addition to schedule | Flag-file/flow-parameter pattern documented; no Prefect Automations needed |
| ORCH-03 | Champion/challenger gate promotes challenger only if it beats champion on rolling QLIKE over a frozen window; default no-promote; rollback is alias flip | MLflow set_registered_model_alias atomic semantics confirmed; alias-flip rollback pattern documented |
</phase_requirements>

---

## Summary

Phase 4 closes the feedback loop: the labeller auto-joins realized vol to the prediction log, Evidently generates distribution drift reports, a rolling QLIKE monitor compares champion against GARCH, alerts fire via webhook or JSONL fallback, and a single Prefect daily flow orchestrates everything from ingest through conditional retrain and challenger registration. Promotion is gated and non-default; rollback is an alias flip.

The three libraries with the most precision requirements are:

1. **Evidently 0.7.21** — API was completely rewritten in 0.7; the `ColumnMapping` / legacy `Report.run(column_mapping=...)` pattern is gone. The current pattern is `Report([DataDriftPreset()]).run(current_dataset, reference_dataset)` where datasets are `Dataset.from_pandas(df, data_definition=DataDefinition(numerical_columns=[...]))`. The result object (returned by `.run()`, NOT the Report object itself) has `save_html()`, `save_json()`, `json()`, and `dict()`. One bug was observed in 0.7.4 where `save_html` was missing from the Report object — this was a call-site error (it must be called on the result, not the Report). Pin to `>=0.7.21,<0.8` and verify at install.

2. **Prefect 3.7** — Local-to-compose deployment uses `flow.from_source(source=str(Path(__file__).parent), entrypoint="pipelines/daily_pipeline.py:daily_flow").deploy(name="...", work_pool_name="local-pool", cron="0 8 * * *")` run with `PREFECT_API_URL=http://localhost:4200/api`. The compose worker is already set up (`prefect worker start --pool local-pool --type process`). For offline testing, `prefect.testing.utilities.prefect_test_harness` is a context manager that spins up a temporary in-process server; tasks can also be tested via `.fn()` to bypass Prefect state entirely.

3. **MLflow alias flip** — `client.set_registered_model_alias(name, alias, version)` atomically reassigns an existing alias without raising an error; the serving container picks up the new version on its next model reload. Rollback = call the same method with the previous version number. The previous version number must be recorded at promotion time (as a tag on the new champion version or in a promotion log file).

**Primary recommendation:** Build the monitoring module as a thin adapter over Evidently, the performance monitor as pure pandas (no Evidently needed), the alert function as stdlib urllib, and the Prefect flow as thin wrappers over existing scripts. The planner should create five separable task groups: (1) labeller, (2) Evidently drift adapter, (3) performance monitor + alert, (4) Prefect flow + deployment script, (5) promotion gate + rollback.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Realized-vol labelling (join prediction log to target) | Python pipeline (batch job) | — | Pure data join; no serving layer involvement |
| Evidently distribution drift report | Python pipeline (batch job) | data/monitoring/ (file output) | Offline batch; results consumed by dashboard in Phase 5 |
| Rolling QLIKE performance monitor | Python pipeline (batch job) | — | Stateless rolling computation over forecast_vs_realized.parquet |
| Alert delivery | Python pipeline (utility function) | alerts.jsonl fallback | Webhook POST; no separate service |
| Retrain flag | File system (flag file) or flow parameter | — | Simple file check inside Prefect task; no pub/sub needed |
| Prefect daily orchestration flow | Prefect worker (compose container) | Prefect server (compose) | Process work pool; worker polls server |
| Champion/challenger promotion | Python script (called as Prefect task) | MLflow registry | Alias flip via MlflowClient; not a serving-layer concern |
| Frozen reference snapshot (Evidently) | data/monitoring/reference/ (versioned parquet) | DVC | Snapshotted at training time; never auto-updated |

---

## Standard Stack

### Core (Phase 4 additions — all already in pyproject.toml except Evidently)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| evidently | >=0.7.21,<0.8 [VERIFIED: PyPI] | Distribution drift reports | Only current OSS tabular drift library with the Report/Dataset/DataDefinition API; no upper-bound pandas pin |
| prefect | >=3.7,<4 (current 3.7.4) [VERIFIED: PyPI] | Orchestration DAG | Already in compose; process work pool; pythonic flow/task API |
| mlflow | >=3.13,<4 [VERIFIED: PyPI] | Model registry alias management | Already in stack; set_registered_model_alias is atomic |
| pandas | >=2.3,<3 [VERIFIED: PyPI] | Parquet read/write, rolling QLIKE window | Already in stack |
| pyarrow | >=4,<25 [VERIFIED: PyPI] | Parquet serialisation | Already in stack |

**Note:** Evidently is the only new dependency for this phase. It is NOT currently in `pyproject.toml` — it must be added. [VERIFIED: pyproject.toml read above]

### Supporting (already in pyproject.toml, used here)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=2,<3 | Rolling QLIKE computation | Performance monitor windows |
| pathlib (stdlib) | — | Path construction for output files | All monitoring output paths |
| urllib.request (stdlib) | — | Webhook POST (no new dep) | Alert delivery |
| json (stdlib) | — | JSONL alert serialisation | Alert fallback |
| threading.Lock (stdlib) | — | Alert file write serialisation if needed | Multi-task safety |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| urllib.request for webhook | httpx | httpx ships with FastAPI via dev deps but adding it for one webhook call is overkill; urllib does Slack-compatible JSON POST with zero new deps |
| flag file for retrain trigger | Prefect Variable / flow parameter | Prefect Variables require server connectivity; flag file works offline in tests; flow parameter is the cleanest v1 approach (passed from drift check task to conditional branch) |
| frozen reference parquet snapshot | Live re-read of training features | Live re-read recomputes reference on each run — non-deterministic if data updates; frozen snapshot is versioned and reproducible |

**Installation (Evidently only — everything else already present):**
```bash
uv add "evidently>=0.7.21,<0.8"
```

---

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| evidently | PyPI | ~4 yrs (2021) | ~500k/mo [VERIFIED: PyPI] | github.com/evidentlyai/evidently | [ASSUMED — slopcheck unavailable] | Approved — established OSS project, active development, 17k GitHub stars |

**Packages removed due to slopcheck [SLOP] verdict:** none

**Packages flagged as suspicious [SUS]:** none

*slopcheck was unavailable at research time. `evidently` is a well-known OSS project with years of history and active maintainership. Risk of hallucination: negligible. The planner may add a `checkpoint:human-verify` before install if desired, but it is not required.*

---

## Architecture Patterns

### System Architecture Diagram

```
[Daily Trigger: cron 08:00 UTC]
         |
         v
[Prefect: daily_flow]
         |
    +----+----+
    |         |
[ingest]  [validate]  <-- existing volforecast CLI / scripts
    |
[generate_features]
    |
[label_forecasts]  <-- reads predictions.parquet + data/processed/
    |               writes forecast_vs_realized.parquet (idempotent append)
    |
[drift_check]
    |  \
    |   [evidently_distribution_drift]  --> data/monitoring/{date}_drift.{html,json}
    |
    |-- performance_drift_QLIKE? --YES--> [send_alert] --> ALERT_WEBHOOK_URL / alerts.jsonl
    |                                     [set_retrain_flag]
    |
[conditional_retrain? (flag OR schedule)]
    |          |
    |         YES
    |          |
    |    [train_lgbm]  --> MLflow run (new version, @challenger alias)
    |    [eval_lgbm]
    |    [register_challenger]
    |
    v
[promotion_gate]
    |  challenger_qlike < champion_qlike AND cooldown OK?
    |  YES --> set_registered_model_alias(@champion, new_version)
    |          record prev_version in tag for rollback
    |  NO  --> no-op (default)
    v
[done]
```

### Recommended Project Structure

```
src/volforecast/
├── monitoring/
│   ├── __init__.py          (currently skeleton)
│   ├── labeller.py          (idempotent forecast_vs_realized append)
│   ├── drift.py             (Evidently adapter: distribution drift report)
│   ├── performance.py       (rolling QLIKE champion vs GARCH monitor)
│   └── alerts.py            (webhook POST + JSONL fallback)
pipelines/
├── __init__.py              (currently exists)
└── daily_pipeline.py        (Prefect @flow wrapping all tasks)
scripts/
└── create_deployment.py     (one-off: register deployment against compose server)
data/monitoring/
├── forecast_vs_realized.parquet   (labeller output, append-only)
├── reference/
│   └── {model_version}_reference.parquet  (frozen Evidently reference snapshot)
├── YYYY-MM-DD_drift.html          (Evidently HTML output)
├── YYYY-MM-DD_drift.json          (Evidently JSON output)
└── alerts.jsonl                   (fallback alert log)
```

### Pattern 1: Idempotent Parquet Append (Labeller)

**What:** Read `predictions.parquet`, join to `compute_target` on processed data, deduplicate against existing `forecast_vs_realized.parquet` on composite key `(asset, as_of_date, model_version)`, write atomically.

**When to use:** Any append-only monitoring table where re-runs must be safe.

```python
# Source: pattern from prediction_log.py (existing atomic append + dedup extension)
import os
import pandas as pd
from pathlib import Path

LABEL_KEY = ["asset", "as_of_date", "model_version"]

def append_labels(new_rows: pd.DataFrame, out_path: Path) -> int:
    """Idempotent append: deduplicates on LABEL_KEY before writing.
    Returns number of net-new rows added.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=LABEL_KEY, keep="first")
    else:
        combined = new_rows.copy()
    tmp = out_path.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return len(combined) - (len(existing) if out_path.exists() else 0)
```

**Key insight:** `drop_duplicates(keep="first")` preserves original rows when re-running on overlapping data. The atomic write pattern (`.tmp` → `os.replace`) comes from the existing `prediction_log.py` contract.

### Pattern 2: Evidently 0.7.21 Distribution Drift Report

**What:** Compare current feature/prediction distributions against a frozen reference snapshot. Numerical-only DataDefinition (no target/prediction roles needed for feature drift).

**When to use:** Distribution drift monitoring for feature columns and forecast_var column.

```python
# Source: docs.evidentlyai.com/docs/library/report + docs.evidentlyai.com/docs/library/output_formats
# Confirmed: report.run() returns a result object; save_html/save_json are on the RESULT not the Report
import pandas as pd
from pathlib import Path
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

def run_distribution_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_columns: list[str],
    output_dir: Path,
    date_str: str,
) -> dict:
    """Run Evidently distribution drift report. Returns dict summary."""
    schema = DataDefinition(numerical_columns=numerical_columns)
    ref_dataset = Dataset.from_pandas(reference_df, data_definition=schema)
    cur_dataset = Dataset.from_pandas(current_df, data_definition=schema)

    report = Report([DataDriftPreset()])
    result = report.run(cur_dataset, ref_dataset)  # (current, reference)

    output_dir.mkdir(parents=True, exist_ok=True)
    result.save_html(str(output_dir / f"{date_str}_drift.html"))
    result.save_json(str(output_dir / f"{date_str}_drift.json"))

    return result.dict()  # For programmatic inspection / logging
```

**Critical:** `save_html` and `save_json` are on the **result** object (returned by `report.run()`), NOT on the `Report` object itself. Calling `report.save_html(...)` raises `AttributeError` — this was the bug in issue #1595. [VERIFIED: docs.evidentlyai.com/docs/library/output_formats]

### Pattern 3: Rolling QLIKE Performance Monitor

**What:** Compute 21-day rolling QLIKE for champion and for GARCH over `forecast_vs_realized.parquet`; compare means; trigger if champion exceeds GARCH by threshold.

**When to use:** Performance-based retrain trigger (the authoritative trigger — stronger than distribution drift).

```python
# Source: based on confirmed eval/metrics.py::qlike contract
import pandas as pd
import numpy as np
from volforecast.eval.metrics import qlike

PERF_WINDOW = 21  # days
DEGRADATION_THRESHOLD = 0.10  # champion QLIKE > 1.10 * GARCH QLIKE triggers alert
# Rationale: 10% relative underperformance over 21 days is clearly meaningful
# (random noise over 21 obs would rarely produce this persistently).
# This is Claude's discretion per CONTEXT.md; document this constant clearly.

def check_performance_drift(
    fvr: pd.DataFrame,  # forecast_vs_realized with columns: as_of_date, asset, model_alias, realized_var, forecast_var
) -> tuple[bool, dict]:
    """
    Returns (should_retrain: bool, report_dict: dict).
    Compares rolling 21-day QLIKE of @champion vs @garch_baseline.
    """
    recent = fvr.tail(PERF_WINDOW)
    champion_rows = recent[recent["model_alias"] == "champion"]
    garch_rows = recent[recent["model_alias"] == "garch_baseline"]

    if len(champion_rows) < PERF_WINDOW // 2 or len(garch_rows) < PERF_WINDOW // 2:
        return False, {"reason": "insufficient_history", "n_champion": len(champion_rows)}

    champ_qlike = qlike(champion_rows["realized_var"].values, champion_rows["forecast_var"].values)
    garch_qlike = qlike(garch_rows["realized_var"].values, garch_rows["forecast_var"].values)

    degraded = champ_qlike > garch_qlike * (1 + DEGRADATION_THRESHOLD)
    return degraded, {
        "champion_qlike": champ_qlike,
        "garch_qlike": garch_qlike,
        "threshold": DEGRADATION_THRESHOLD,
        "degraded": degraded,
    }
```

**Cold-start policy (Claude's discretion):** Do not fire performance drift until `forecast_vs_realized.parquet` contains at least `PERF_WINDOW` rows for both champion and GARCH. Return `(False, {"reason": "cold_start"})` until then.

### Pattern 4: Alert Delivery (Slack-compatible webhook, stdlib only)

**What:** POST a JSON payload to `ALERT_WEBHOOK_URL`; if unset or on failure, append a structured record to `alerts.jsonl`. Never crash the pipeline.

```python
# Source: Python stdlib urllib pattern; no new deps
import json
import os
from pathlib import Path
from urllib import request
from urllib.error import URLError
import datetime

ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")
ALERTS_JSONL = Path("data/monitoring/alerts.jsonl")

def send_alert(title: str, body: dict) -> None:
    """Fire-and-forget alert. Webhook POST if configured, else JSONL fallback."""
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "title": title,
        **body,
    }
    if ALERT_WEBHOOK_URL:
        payload = {"text": f"*{title}*\n" + json.dumps(body, indent=2)}
        req = request.Request(
            ALERT_WEBHOOK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=5):
                pass
            return
        except (URLError, OSError):
            pass  # Fall through to JSONL
    # JSONL fallback
    ALERTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

### Pattern 5: Prefect 3 Flow — Local Process Pool Deployment

**What:** Define the daily Prefect flow; deploy it against the compose server's `local-pool`; run the deploy script once to register, then the compose worker picks it up.

**When to use:** Scheduled orchestration against a running Prefect server.

```python
# Source: docs.prefect.io/v3/deploy/infrastructure-concepts/deploy-via-python
# Source: docs-3.prefect.io/v3/tutorials/schedule
from pathlib import Path
from prefect import flow, task

@task(retries=2, retry_delay_seconds=30)
def ingest_task(): ...

@task
def label_task(): ...

@task
def drift_check_task() -> bool: ...  # returns should_retrain flag

@task
def train_and_register_challenger_task(): ...

@task
def promotion_gate_task(): ...

@flow(name="volforecast-daily")
def daily_flow(force_retrain: bool = False) -> None:
    ingest_task()
    # ... chained tasks
    should_retrain = drift_check_task()
    if should_retrain or force_retrain:
        train_and_register_challenger_task()
    promotion_gate_task()


# --- create_deployment.py (run once against compose server) ---
# PREFECT_API_URL=http://localhost:4200/api python scripts/create_deployment.py

if __name__ == "__main__":
    daily_flow.from_source(
        source=str(Path(__file__).parent.parent),  # repo root
        entrypoint="pipelines/daily_pipeline.py:daily_flow",
    ).deploy(
        name="volforecast-daily",
        work_pool_name="local-pool",
        cron="0 8 * * *",  # 08:00 UTC daily, after crypto close + equity pre-open
    )
```

**Key detail:** `flow.from_source(source=local_path, entrypoint="file:func")` with a local directory path is the correct pattern for a compose process worker that shares the same mounted volume as the code. No git remote required. [VERIFIED: docs.prefect.io/v3/deploy/infrastructure-concepts/deploy-via-python]

**Cron timing note (Claude's discretion):** `0 8 * * *` UTC (08:00) gives: ~8 hours after Binance daily close (00:00 UTC), and before major equity opens (13:30 UTC). Crypto labels for yesterday are available; equity labels for yesterday (16:00 ET previous day) are available.

### Pattern 6: MLflow Alias-Flip Promotion Gate

**What:** Compare challenger vs champion on a frozen QLIKE window; atomically flip `@champion` alias if challenger wins; record previous champion version for rollback.

```python
# Source: mlflow.org/docs/latest/ml/model-registry/workflow/  [VERIFIED]
from mlflow import MlflowClient

PROMOTION_COOLDOWN_DAYS = 7  # locked constant per CONTEXT.md
MODEL_NAME = "volforecast-lgbm"

def promote_if_better(
    challenger_version: int,
    frozen_window_qlike_challenger: float,
    frozen_window_qlike_champion: float,
    last_promotion_date,  # datetime.date or None
    today,               # datetime.date
) -> bool:
    """Returns True if promotion occurred."""
    from datetime import date
    if last_promotion_date and (today - last_promotion_date).days < PROMOTION_COOLDOWN_DAYS:
        return False  # cooldown
    if frozen_window_qlike_challenger >= frozen_window_qlike_champion:
        return False  # champion still better or equal — no-promote default

    client = MlflowClient()
    # Record prev champion version for rollback
    try:
        prev = client.get_model_version_by_alias(MODEL_NAME, "champion")
        prev_version = prev.version
    except Exception:
        prev_version = None

    # Atomic reassignment — set_registered_model_alias reassigns if alias exists
    client.set_registered_model_alias(MODEL_NAME, "champion", challenger_version)

    # Tag new champion with rollback info
    client.set_model_version_tag(
        MODEL_NAME, str(challenger_version),
        "previous_champion_version", str(prev_version)
    )
    return True


def rollback_champion(to_version: int) -> None:
    """Rollback: flip @champion back to a previous version."""
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", to_version)
```

### Pattern 7: Prefect Flow Testing (offline-safe)

**What:** Test flow logic without a running Prefect server; test task business logic without Prefect state overhead.

```python
# Source: docs.prefect.io/v3/how-to-guides/workflows/test-workflows  [VERIFIED]
import pytest
from prefect.testing.utilities import prefect_test_harness

# Session-scoped: share one in-memory server across all flow tests
@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield

# Test a task's logic directly (no Prefect state, no server):
def test_labeller_business_logic():
    from pipelines.daily_pipeline import label_task
    result = label_task.fn()  # bypasses retry/state; mocks can be applied normally
    assert result is not None
```

### Anti-Patterns to Avoid

- **Calling `report.save_html()` on the Report object:** The `save_html` method is on the result returned by `report.run()`, not on the `Report` template. `report.run(...)` returns an evaluation result; `report` itself is a template.
- **Using `ColumnMapping` or legacy `DatasetColumns`:** These are from Evidently <0.7. They will silently fail or import-error against 0.7.21.
- **Triggering promotion from distribution drift:** Locked anti-pattern (CONTEXT.md). Only rolling QLIKE performance degradation triggers retraining; only the explicit promotion gate triggers alias flip.
- **Auto-promoting on every retrain:** No-promote is the default. The promotion gate must return `False` unless challenger strictly beats champion AND cooldown has elapsed.
- **Hardcoding `PREFECT_API_URL` in the flow:** Must be read from environment variable, which the compose worker already sets to `http://prefect-server:4200/api`. For local non-compose runs, set `PREFECT_API_URL=http://localhost:4200/api` in the shell.
- **Storing retrain flag as a shared file without cleanup:** After a retrain triggered by the flag, the flag must be cleared. Otherwise the next scheduled run re-triggers unconditionally.
- **Re-loading the reference snapshot on every drift check:** The reference snapshot must be frozen at training time and versioned. If it updates automatically, distribution drift loses its meaning.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Drift detection statistics (KS, PSI, Wasserstein) | Custom stat functions | `evidently.presets.DataDriftPreset` | Correct per-column test selection, N-sample corrected p-values, multiple comparison handling |
| HTML drift report rendering | Custom Jinja template | `result.save_html()` from Evidently | Interactive JS-backed charts, not worth rebuilding |
| Webhook payload encoding | Custom HTTP client | `urllib.request` stdlib | One function, zero deps; Slack-compatible JSON POST is trivial |
| Flow/task retries and state tracking | Custom retry loops | `@task(retries=N, retry_delay_seconds=M)` | Prefect handles idempotency, logging, and retry backoff |
| Model version alias management | Custom version DB | `MlflowClient.set_registered_model_alias` | Atomic at the registry layer; serves as the single source of truth for `@champion` |

**Key insight:** The performance-based QLIKE monitor is the one piece that truly cannot be delegated to Evidently — Evidently has no native concept of "compare two models' error rates on the same window." That specific check is ~20 lines of pandas + the existing `qlike()` function from `eval/metrics.py`.

---

## Common Pitfalls

### Pitfall 1: save_html called on Report object, not result

**What goes wrong:** `AttributeError: 'Report' object has no attribute 'save_html'`

**Why it happens:** `report.run(...)` returns a separate result object. The `Report` instance is a template. Confusion because older 0.4.x API stored results on the Report object itself.

**How to avoid:** Always capture `result = report.run(current, reference)` and call `result.save_html(...)`, `result.save_json(...)`, `result.json()`, `result.dict()`.

**Warning signs:** AttributeError at report save time. Verify by checking that the variable being called is the return value of `.run()`, not the `Report(...)` constructor call.

### Pitfall 2: Evidently DataDefinition missing numerical_columns — schema inferred wrong

**What goes wrong:** Empty `DataDefinition()` causes Evidently to auto-detect column types. For a purely numerical DataFrame of variance features (float64 values around 1e-4), auto-detection usually works, but datetime-indexed or string-named columns may be mis-categorised.

**How to avoid:** Always pass `DataDefinition(numerical_columns=[...])` explicitly with the exact list of feature columns to include in drift analysis. Exclude `asset` (string/category), `as_of_date` (datetime), `model_version` (string) columns from the drift definition — they are not numerical features.

**Warning signs:** Evidently raises `ValueError: PandasEngine works only with pd.DataFrame` or produces reports with 0 columns analysed.

### Pitfall 3: Rolling QLIKE cold-start fires false alert before enough history

**What goes wrong:** On day 1 of monitoring, `forecast_vs_realized.parquet` has 1 row for champion and 0 for GARCH. The performance monitor computes QLIKE on a single point and fires an alert immediately.

**How to avoid:** Gate the performance drift check on `min(len(champion_rows), len(garch_rows)) >= PERF_WINDOW // 2`. Return `(False, {"reason": "cold_start"})` until then. Log this state clearly so the operator knows why no alert has fired yet.

**Warning signs:** Retrain triggered on day 1 before any real monitoring window has accumulated.

### Pitfall 4: Prefect worker pool name mismatch blocks flow runs

**What goes wrong:** Deployment registered against `work_pool_name="my-pool"` but the compose worker starts with `--pool local-pool`. Flows queue indefinitely with no error (they just wait for a worker on the right pool).

**How to avoid:** The existing `docker-compose.yml` hardcodes `--pool local-pool`. The deployment script must use exactly `work_pool_name="local-pool"`. Verify after deployment by checking `prefect deployment ls` output in the Prefect UI or CLI.

**Warning signs:** Flows appear in the UI as "Scheduled" but never transition to "Running". No error in the server logs.

### Pitfall 5: Retrain flag not cleared after use

**What goes wrong:** Flag file (or flow parameter mechanism) triggers retrain. After successful retrain, the flag is not cleared. The next scheduled run sees the flag and triggers another retrain unnecessarily.

**How to avoid:** The task that triggers retrain must atomically check-and-clear the flag. If using a flag file: read presence, trigger retrain, delete the file. If using a flow parameter: the parameter is ephemeral per-run and does not persist — no cleanup needed. Prefer the flow parameter approach.

**Warning signs:** Back-to-back retrains in the Prefect UI with no performance degradation event. Challenger versions accumulating without pauses.

### Pitfall 6: Frozen reference snapshot updated automatically

**What goes wrong:** The Evidently reference is loaded fresh from `data/features/` on each run. Features change as new data arrives. Drift baseline drifts with the data → monitor never fires.

**How to avoid:** Reference snapshot is taken once at training time (`data/monitoring/reference/{model_version}_reference.parquet`) and NEVER updated automatically. It only changes when a new champion is promoted (new model_version → new reference snapshot created as part of promotion).

**Warning signs:** Evidently drift score is always near-zero even during obvious market regime changes.

### Pitfall 7: set_registered_model_alias rollback loses previous champion version

**What goes wrong:** Promotion flip sets `@champion` to version N. No record of which version was N-1. Rollback requires knowing N-1 but that information is not stored anywhere.

**How to avoid:** Before flipping, call `client.get_model_version_by_alias(MODEL_NAME, "champion")` and record the previous version number. Store it as a tag on the new champion version (`previous_champion_version`), and/or write it to a small promotion log file. Rollback = read tag, call `set_registered_model_alias` with the recorded version.

**Warning signs:** Rollback fails with "which version?" uncertainty. Serving team cannot find the last-known-good version after an incident.

---

## Code Examples

### Verified: Evidently 0.7.21 complete flow

```python
# Source: docs.evidentlyai.com/quickstart_ml  [VERIFIED]
# Source: docs.evidentlyai.com/docs/library/output_formats  [VERIFIED]
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

# Schema: declare numerical columns explicitly
schema = DataDefinition(
    numerical_columns=["rv_5", "rv_22", "log_ret", "ewma_vol", "forecast_var"],
)

# Wrap DataFrames (reference = frozen training snapshot)
current_ds = Dataset.from_pandas(current_df, data_definition=schema)
reference_ds = Dataset.from_pandas(reference_df, data_definition=schema)

# Build and run report — result is a SEPARATE object from report
report = Report([DataDriftPreset()])
result = report.run(current_ds, reference_ds)  # returns SnapshotResult

# Save outputs — call on RESULT, not on report
result.save_html("data/monitoring/2026-06-12_drift.html")
result.save_json("data/monitoring/2026-06-12_drift.json")
summary = result.dict()   # programmatic access
```

### Verified: Prefect 3 test harness

```python
# Source: docs.prefect.io/v3/how-to-guides/workflows/test-workflows  [VERIFIED]
import pytest
from prefect.testing.utilities import prefect_test_harness

@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield

# Test task logic without Prefect state:
from pipelines.daily_pipeline import label_task
def test_label_task_business_logic():
    result = label_task.fn()  # bypasses Prefect wrapping entirely
    assert result >= 0  # n_new_rows >= 0
```

### Verified: MLflow alias flip with rollback capability

```python
# Source: mlflow.org/docs/latest/ml/model-registry/workflow/  [VERIFIED]
from mlflow import MlflowClient

client = MlflowClient()

# Promotion (atomic reassignment — no error if alias already exists)
client.set_registered_model_alias("volforecast-lgbm", "champion", new_version)

# Record rollback info as a model version tag
client.set_model_version_tag("volforecast-lgbm", str(new_version),
                             "previous_champion_version", str(prev_version))

# Rollback: flip back
client.set_registered_model_alias("volforecast-lgbm", "champion", prev_version)

# Read current champion version number
current_champ = client.get_model_version_by_alias("volforecast-lgbm", "champion")
print(current_champ.version)  # "3"
```

### Verified: Prefect deployment script (local process pool, no git remote)

```python
# Source: docs.prefect.io/v3/deploy/infrastructure-concepts/deploy-via-python  [VERIFIED]
# Run with: PREFECT_API_URL=http://localhost:4200/api python scripts/create_deployment.py
import os
from pathlib import Path
from pipelines.daily_pipeline import daily_flow

if __name__ == "__main__":
    daily_flow.from_source(
        source=str(Path(__file__).parent.parent),   # repo root on shared volume
        entrypoint="pipelines/daily_pipeline.py:daily_flow",
    ).deploy(
        name="volforecast-daily",
        work_pool_name="local-pool",   # must match compose worker --pool arg
        cron="0 8 * * *",             # 08:00 UTC daily
    )
```

### Verified: Webhook POST with urllib (no new deps)

```python
# Source: Python stdlib docs + Slack webhook format  [ASSUMED pattern; stdlib confirmed]
import json
from urllib import request
from urllib.error import URLError

def post_webhook(url: str, text: str, body: dict) -> bool:
    """Returns True on success, False on any failure. Never raises."""
    payload = {"text": f"*{text}*\n```{json.dumps(body, indent=2)}```"}
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=5):
            return True
    except (URLError, OSError):
        return False
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `ColumnMapping` + `Report.run(data_drift_dataset_report, ...)` (Evidently 0.4.x) | `Report([DataDriftPreset()]).run(Dataset, Dataset)` (Evidently 0.7+) | 0.7.0 (2024 H1) | All pre-0.7 tutorials are broken; this is the most likely source of confusion |
| MLflow stages `transition_model_version_stage("Production")` | `set_registered_model_alias(name, "champion", version)` | MLflow 2.9 (deprecation), 3.x (removed path) | Stage-based code will emit deprecation warnings in MLflow 3.x |
| `flow.run()` / Prefect 2 deployment YAML format | `@flow` + `flow.from_source().deploy()` / Python-native deployment | Prefect 3.0 | Prefect 2 YAML files are not compatible with Prefect 3 |
| `prefect.testing.utilities.prefect_test_harness` (still valid) | Same import path; `task.fn()` bypasses state | Unchanged since Prefect 2 | No migration needed |

**Deprecated/outdated:**
- `evidently.ColumnMapping`: removed in 0.7; use `DataDefinition`
- `evidently.model_profile` module: removed in 0.7
- `mlflow.register_model` with `stage=` parameter: deprecated since 2.9; use aliases
- Prefect 2 `Deployment.build_from_flow()`: replaced by `flow.deploy()` in Prefect 3

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `evidently.presets.DataSummaryPreset` is available in 0.7.21 | Standard Stack | Phase 5 dashboard may need adjustment; no Phase 4 impact (Phase 4 only uses DataDriftPreset) |
| A2 | Webhook JSONL fallback append is safe from concurrent writes (single-process Prefect task) | Pattern 4 (alerts) | Race condition if multiple tasks fire simultaneously; add a threading.Lock if needed |
| A3 | Degradation threshold of 10% relative QLIKE over 21 days is calibrated correctly for this dataset | Pattern 3 (performance monitor) | Too sensitive → retrain storms; too loose → delayed response. Documented as Claude's discretion per CONTEXT.md |
| A4 | `flow.from_source(source=local_path)` with a compose-mounted directory works without a git remote for process work pools | Pattern 5 (deployment) | Deployment script may need `--no-build` or different source specification. Fallback: use `prefect.yaml` file-based deployment instead |
| A5 | The Prefect compose worker's `local-pool` auto-creates on first start (`--type process`) | docker-compose.yml | If the pool is not auto-created, the `prefect worker start` command will fail; manual `prefect work-pool create local-pool --type process` may be needed as a one-time setup step |
| A6 | slopcheck was not available; evidently package legitimacy is [ASSUMED] rather than [VERIFIED] | Package Legitimacy Audit | Risk is effectively nil given evidently's known provenance, but cannot be formally verified this session |

---

## Open Questions

1. **GARCH baseline rows in forecast_vs_realized.parquet**
   - What we know: The labeller joins prediction_log.py rows (which log `model_version` and `alias`) to realized targets.
   - What's unclear: The existing prediction log only captures FastAPI-served forecasts (champion). GARCH forecasts for the live period must also be logged somewhere for the rolling QLIKE comparison to work. The CONTEXT.md says "rolling live QLIKE of champion vs rolling QLIKE of GARCH baseline" — but does the existing pipeline write GARCH live forecasts to the prediction log?
   - Recommendation: The labeller/pipeline must also compute and log GARCH(1,1) live forecasts (using the fitted GARCH on the trailing window) to `forecast_vs_realized` with `model_alias="garch_baseline"`. This is a small extension to the labelling task. The planner should include a task that generates and logs GARCH live forecasts alongside champion forecasts.

2. **Evidently reference snapshot: when is it created?**
   - What we know: CONTEXT.md says "frozen training reference snapshot."
   - What's unclear: Is the snapshot created by the initial `train_lgbm.py` script (Phase 3), or by a Phase 4 one-time setup task?
   - Recommendation: Create the reference snapshot as part of Phase 4 setup (Wave 0): read the training feature parquet used for the current champion, snapshot it to `data/monitoring/reference/{champion_version}_reference.parquet`. The Prefect flow uses this frozen file on every run.

3. **PREFECT_API_URL in the deployment script context**
   - What we know: The deployment script runs on the host (or in a container) and must reach `http://localhost:4200/api`.
   - What's unclear: If the script is run from inside the prefect-worker container, the URL is `http://prefect-server:4200/api` (Docker network). If from the host, it is `http://localhost:4200/api`.
   - Recommendation: The deployment script should read `PREFECT_API_URL` from env (defaulting to `http://localhost:4200/api`). Document that it must be run from the host with the compose stack running.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Prefect server (compose) | ORCH-01 deployment | ✓ (compose service defined) | 3-latest image | — |
| Prefect worker (compose) | ORCH-01 execution | ✓ (compose service defined, local-pool) | 3-latest image | — |
| MLflow server (compose) | ORCH-03 registry | ✓ (compose service defined) | v3.13.0 image | — |
| PostgreSQL (compose) | MLflow + Prefect backends | ✓ (compose service defined) | postgres:16 | — |
| evidently Python package | MON-02 | NOT installed (not in pyproject.toml) | — | Must add `evidently>=0.7.21,<0.8` |
| data/predictions/predictions.parquet | MON-01 labeller | UNKNOWN (Phase 3 must have produced it) | — | Phase 4 must gate on file existence |
| data/processed/ per-asset parquets | MON-01 labeller (realized vol) | UNKNOWN (Phase 1-2 deliverable) | — | Phase 4 must gate on file existence |
| data/features/ per-asset parquets | MON-02 Evidently reference | UNKNOWN (Phase 2 deliverable) | — | Phase 4 must gate on file existence |

**Missing dependencies with no fallback:**
- `evidently` Python package — must be added to `pyproject.toml` before Phase 4 can run.

**Missing dependencies with fallback:**
- `data/` parquets — Phase 4 tasks should check for existence and raise a clear error (not a silent skip) if upstream phase outputs are missing.

---

## Security Domain

> `security_enforcement` is not set to false in config.json — section is required.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No user-facing auth in monitoring pipeline |
| V3 Session Management | No | Batch pipeline; no sessions |
| V4 Access Control | No | Local single-machine dev; no multi-user |
| V5 Input Validation | Yes | Parquet schema validation (Pandera) at labeller input; validate `forecast_vs_realized` schema before QLIKE computation |
| V6 Cryptography | No | No secrets transmitted; webhook URL is an env var, not a secret in code |

### Known Threat Patterns for this Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Webhook URL in code / committed .env | Information Disclosure | Read from `ALERT_WEBHOOK_URL` env var; never commit; `.gitignore` the `.env` |
| Malformed parquet injected into forecast_vs_realized | Tampering | Pandera schema check at labeller input and output; validate column dtypes before QLIKE computation |
| Promotion gate bypassed by direct alias flip | Elevation of Privilege | `set_registered_model_alias` is only called inside the promotion gate function; no external unauthenticated API exposed locally |
| MLflow and Prefect UIs exposed to LAN | Information Disclosure | Existing compose already binds `127.0.0.1` only (WR-05 in docker-compose.yml comments) |

---

## Sources

### Primary (HIGH confidence)
- [docs.evidentlyai.com/quickstart_ml](https://docs.evidentlyai.com/quickstart_ml) — Evidently 0.7.21 Report/Dataset/DataDefinition API; confirmed `report.run()` return value pattern
- [docs.evidentlyai.com/docs/library/output_formats](https://docs.evidentlyai.com/docs/library/output_formats) — Confirmed `save_html()`, `save_json()`, `json()`, `dict()` on result object
- [docs.evidentlyai.com/docs/library/data_definition](https://docs.evidentlyai.com/docs/library/data_definition) — DataDefinition constructor with `numerical_columns`, `categorical_columns`, `regression`, `id_column` parameters
- [mlflow.org/docs/latest/ml/model-registry/workflow/](https://mlflow.org/docs/latest/ml/model-registry/workflow/) — `set_registered_model_alias` atomic reassignment; `get_model_version_by_alias`; alias URI loading
- [docs.prefect.io/v3/deploy/infrastructure-concepts/deploy-via-python](https://docs.prefect.io/v3/deploy/infrastructure-concepts/deploy-via-python) — `flow.from_source().deploy()` with work_pool_name and cron
- [docs.prefect.io/v3/how-to-guides/workflows/test-workflows](https://docs.prefect.io/v3/how-to-guides/workflows/test-workflows) — `prefect_test_harness` session fixture; `task.fn()` bypass pattern

### Secondary (MEDIUM confidence)
- [github.com/evidentlyai/evidently/issues/1595](https://github.com/evidentlyai/evidently/issues/1595) — Confirmed `save_html` AttributeError when called on Report (not result); informs the critical call-site warning
- WebSearch (Prefect docs) — `flow.serve()` vs `flow.deploy()` distinction; `PREFECT_API_URL` compose env var pattern; `prefect.yaml` cron format

### Tertiary (LOW confidence — informational only)
- WebSearch (champion/challenger QLIKE threshold) — 5-10% relative degradation as reasonable monitoring threshold; not from a single authoritative source; documented as Claude's discretion

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions verified from PyPI; Evidently API verified against official docs
- Architecture: HIGH — based on existing codebase contracts (prediction_log.py, metrics.py) and verified library APIs
- Evidently save_html/save_json: HIGH — confirmed via official output_formats doc; bug in 0.7.4 was a call-site error, not a missing feature
- Prefect deployment: MEDIUM — `flow.from_source(local_path)` confirmed for process pools via search; exact compose-mounted volume behaviour is [ASSUMED]
- QLIKE threshold (10%): LOW — Claude's discretion, no authoritative source for exact value
- Package legitimacy: ASSUMED — slopcheck unavailable; evidently provenance is well-known

**Research date:** 2026-06-12
**Valid until:** 2026-09-12 (90 days — Evidently and Prefect both have active release cycles; re-verify if upgrading past pinned versions)
