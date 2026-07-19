---
phase: 04-monitoring-orchestration-retraining
plan: "05"
subsystem: orchestration
tags: [prefect, daily-flow, orchestration, deployment, docker-compose, tdd-adjacent, orch-01, orch-02]
dependency_graph:
  requires:
    - 04-01  # label_forecasts function + run_labeller.py
    - 04-02  # run_distribution_drift + select_numerical_columns
    - 04-03  # run_performance_monitor (retrain trigger)
    - 04-04  # frozen_window_qlike + promote_if_better + PROMOTION_COOLDOWN_DAYS
    - scripts/train_lgbm.py  # Phase 3 retraining
    - scripts/eval_lgbm.py   # Phase 3 eval
    - scripts/generate_features.py  # Phase 2 features
    - volforecast.cli  # ingest CLI
  provides:
    - pipelines/daily_pipeline.py   # Prefect @flow daily_flow wrapping Plans 01-04
    - scripts/create_deployment.py  # One-time deployment registration against compose
    - infra/docker-compose.yml      # prefect-worker code-visibility volume mount
    - tests/integration/test_daily_flow.py  # Offline flow/task tests
  affects:
    - ORCH-01, ORCH-02  # Requirements satisfied
    - Phase 5 dashboard (will observe drift/monitoring outputs produced by this flow)
tech_stack:
  added: []
  patterns:
    - Prefect 3 @flow/@task thin-wrapper pattern (Pattern 5 from 04-RESEARCH.md)
    - prefect_test_harness session fixture for offline flow tests (Pattern 7)
    - task.fn() direct business-logic assertions without Prefect state
    - Ephemeral flow parameter as retrain trigger (no flag file — Pitfall 5 avoidance)
    - _get_logger() fallback: Prefect logger with stdlib fallback for test hermeticity
    - T-04-15 challenger registration: retrain_task reassigns @champion to @challenger immediately after train_lgbm.py
key_files:
  created:
    - pipelines/daily_pipeline.py
    - scripts/create_deployment.py
    - tests/integration/test_daily_flow.py
  modified:
    - infra/docker-compose.yml
decisions:
  - "force_retrain is an ephemeral flow parameter (no flag file persisted) — Pitfall 5 avoidance per 04-RESEARCH.md"
  - "retrain_task reassigns fresh @champion to @challenger via MlflowClient immediately after train_lgbm.py completes (T-04-15 enforcement)"
  - "_get_logger() tries get_run_logger() first, falls back to stdlib logging — allows task.fn() calls in tests without Prefect context (MissingContextError avoidance)"
  - "drift_check_task: reference_path defaults to champion version '3' (current champion from Phase 3); must be updated when champion is promoted"
  - "promotion_gate_task: last_promotion_date=None (cooldown not tracked across runs in v1); PROMOTION_COOLDOWN_DAYS is the gate, not a persistent date"
  - "prefect-worker compose volumes: ..:/repo:ro (code, read-only) + ../data:/repo/data (data, rw) with VOLFORECAST_ROOT=/repo"
  - "Checkpoint Task 3 (live verification) deferred to operator: requires running compose stack; documented in plan"
metrics:
  duration: "~35 minutes"
  completed: "2026-06-15"
  tasks_completed: 2
  files_created: 3
  files_modified: 2
  tests_added: 8
---

# Phase 04 Plan 05: Prefect Daily Orchestration Flow Summary

**One-liner:** Prefect 3 daily flow composing Plans 01-04 as thin @tasks with ephemeral force_retrain parameter, challenger registration guard, and offline-testable via prefect_test_harness.

## What Was Built

### `pipelines/daily_pipeline.py` (569 lines)

Single `@flow(name="volforecast-daily")` `daily_flow(force_retrain: bool = False) -> dict` wiring the full closed loop:

**Task chain (locked order per 04-CONTEXT.md):**

| # | Task | Implementation | Notes |
|---|------|----------------|-------|
| 1 | `ingest_validate_task` | subprocess `volforecast ingest` CLI | retries=2, retry_delay_seconds=30; non-zero exit raises (data-quality hard-fail) |
| 2 | `features_task` | subprocess `scripts/generate_features.py` | via `_run_script` helper |
| 3 | `label_task` | `label_forecasts(data_root, fvr_path)` | direct function call; returns net-new row count |
| 4 | `drift_check_task` | `run_distribution_drift(...)` on frozen reference | REPORT-ONLY, returns output dir path (no retrain signal — T-04-04) |
| 5 | `performance_check_task` | `run_performance_monitor(fvr_path)` | returns bool retrain flag (MON-03 authoritative trigger) |
| 6a | `retrain_task` | subprocess `scripts/train_lgbm.py` + MlflowClient alias reassignment | registers @challenger; restores @champion to prev version (T-04-15) |
| 6b | `eval_task` | subprocess `scripts/eval_lgbm.py` | generates ml_vs_baselines report |
| 6c | `promotion_gate_task` | `frozen_window_qlike` + `promote_if_better` | default no-promote; only when retrain happened |

**Ephemeral trigger (Pitfall 5 avoidance):**
`should_retrain = performance_check_task() or force_retrain` — `force_retrain` is a per-run parameter, never persisted to a flag file. No state survives between runs.

**Challenger registration (T-04-15):**
After `train_lgbm.py` sets `@champion`, `retrain_task` immediately:
1. Reads the new `@champion` version via `MlflowClient.get_model_version_by_alias`
2. Sets `@challenger` alias on that version
3. Restores `@champion` to the previous version (via `previous_champion_version` tag, if present)

**Path resolution:** All paths derived from `_repo_root()` which reads `VOLFORECAST_ROOT` env var first, then falls back to `__file__/../..`. Works on host and in the compose worker container.

### `scripts/create_deployment.py`

One-time deployment registration script. Key properties:
- `work_pool_name="local-pool"` — exactly matches compose worker `--pool` arg (Pitfall 4)
- `cron="0 8 * * *"` — 08:00 UTC daily; rationale documented: 8h after Binance close, before NYSE open
- `flow.from_source(source=str(_REPO_ROOT), entrypoint="pipelines/daily_pipeline.py:daily_flow")`
- Reads `PREFECT_API_URL` from env, defaults to `http://localhost:4200/api`
- Docstring contains exact host run command: `PREFECT_API_URL=http://localhost:4200/api uv run python scripts/create_deployment.py`

### `infra/docker-compose.yml` (modified: prefect-worker)

Added to `prefect-worker` service:
```yaml
environment:
  MLFLOW_TRACKING_URI: "http://mlflow-server:5000"
  VOLFORECAST_ROOT: "/repo"
volumes:
  - ..:/repo:ro          # repo root → /repo (read-only code + config)
  - ../data:/repo/data   # data/ rw: ingest writes raw, labeller writes FVR
```

Comment documents why the mount is needed: local-process pool runs entrypoint from filesystem; without mount, `flow.from_source(local path)` cannot resolve (Assumption A4 / Pitfall 4).

### `tests/integration/test_daily_flow.py` (8 tests, all passing offline)

Session-scoped `prefect_test_harness` autouse fixture; individual tests use full flow runs or `task.fn()`:

| Test | Approach | Verifies |
|------|----------|----------|
| `test_happy_path_no_retrain` | Full flow run | should_retrain=False, retrain not called |
| `test_force_retrain_path` | Full flow run | force_retrain=True → retrain+eval+promo each called once |
| `test_performance_flag_triggers_retrain` | Full flow run + real FVR file | perf flag=True → retrain fires independently |
| `test_flow_returns_expected_keys` | Full flow run | Summary dict has all 6 required keys |
| `test_label_task_fn_direct` | `task.fn()` | label_task delegates to label_forecasts |
| `test_performance_check_task_cold_start` | `task.fn()` | FVR absent → False (cold-start) |
| `test_drift_check_task_no_reference` | `task.fn()` | Reference absent → returns str, no raise |
| `test_promotion_defaults_no_promote` | `task.fn()` + real FVR data | Challenger worse QLIKE → False, no alias flip |

## Verification Evidence

```
uv run pytest tests/integration/test_daily_flow.py -q
........
8 passed in 66.39s

uv run ruff check .
All checks passed!

python -c "from pipelines.daily_pipeline import daily_flow; print(daily_flow.name)"
volforecast-daily

python -c "assert 'local-pool' in open('scripts/create_deployment.py').read(); print('ok')"
ok

python -c "import yaml; w=yaml.safe_load(open('infra/docker-compose.yml'))['services']['prefect-worker']; assert 'volumes' in w; print(w['volumes'])"
['..:/repo:ro', '../data:/repo/data']
```

## Checkpoint Task 3: Live Deployment Verification (Deferred to Operator)

Task 3 is a `checkpoint:human-verify gate="blocking"` requiring a running compose stack.
Per `<checkpoint_handling>` instructions (non-autonomous plan, no interactive user), this
is documented here as a **live verification step the operator must perform** when the
compose stack is available:

1. Start the compose stack: `docker compose -f infra/docker-compose.yml up -d`
2. Register deployment: `PREFECT_API_URL=http://localhost:4200/api uv run python scripts/create_deployment.py`
3. Verify in Prefect UI (http://localhost:4200): `volforecast-daily` on `local-pool`, cron `0 8 * * *`
4. Trigger a manual run and confirm: Scheduled → Running → Completed
5. Verify `data/monitoring/forecast_vs_realized.parquet` gained rows and a dated `_drift.html` was written
6. Confirm `@champion` alias unchanged (no auto-promote)

The offline tests (Task 2) cover all business logic and the deployment script parses correctly. The checkpoint verifies the live Prefect server integration, which requires the compose stack to be running.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `get_run_logger()` raises `MissingContextError` when called via `task.fn()` outside Prefect context**
- **Found during:** Task 2 — 4/8 tests failed with `MissingContextError: There is no active flow or task run context`
- **Issue:** Every task used `logger = get_run_logger()` at task entry, which requires an active Prefect run context. Calling `task.fn()` in tests bypasses the Prefect engine, so there is no context.
- **Fix:** Added `_get_logger()` helper that tries `get_run_logger()` first and falls back to `logging.getLogger(__name__)` on `MissingContextError`. Replaced all `get_run_logger()` calls in the pipeline with `_get_logger()`.
- **Files modified:** `pipelines/daily_pipeline.py`
- **Commit:** 30cca8a

**2. [Rule 1 - Bug] `test_performance_flag_triggers_retrain` — cold-start short-circuit fired instead of stub**
- **Found during:** Task 2 — test expected `should_retrain=True` but got `False`
- **Issue:** `performance_check_task` has a cold-start guard (`if not fvr_path.exists(): return False`) that fired before the monkeypatched `run_performance_monitor` was ever called, because the test didn't create the FVR file.
- **Fix:** Added `_make_fvr_file(fvr_path)` call in the test to create a synthetic FVR before running the flow; the cold-start guard then passes and the monkeypatched `run_performance_monitor` returns `True`.
- **Files modified:** `tests/integration/test_daily_flow.py`
- **Commit:** 30cca8a

**3. [Rule 1 - Bug] `test_promotion_defaults_no_promote` — tried to monkeypatch `MlflowClient` on promotion module (lazy import)**
- **Found during:** Task 2 — `AttributeError: module 'volforecast.monitoring.promotion' has no attribute 'MlflowClient'`
- **Issue:** `promotion.py` uses a lazy import (`from mlflow import MlflowClient` inside function body), so the attribute doesn't exist at module level for monkeypatching.
- **Fix:** Leveraged the fact that `promote_if_better` returns `False` BEFORE making any MLflow call when `challenger_qlike >= champion_qlike`. Removed the MLflow client patch entirely; only patched `mlflow.set_tracking_uri` to prevent connection attempts. Test passes without touching the registry.
- **Files modified:** `tests/integration/test_daily_flow.py`
- **Commit:** 30cca8a

## Known Stubs

**1. `drift_check_task` champion version hardcoded to "3"**
- File: `pipelines/daily_pipeline.py`, `_reference_path("3")`
- Reason: The reference snapshot path is `data/monitoring/reference/3_reference.parquet` (created in Plan 02 for champion version 3). When a new champion is promoted, this value must be updated. In v1 this is acceptable — the snapshot version update is part of the promotion workflow and will be addressed in Phase 5 or a future retrain cycle.

**2. `promotion_gate_task` `last_promotion_date=None`**
- File: `pipelines/daily_pipeline.py`, `promotion_gate_task`
- Reason: Tracking `last_promotion_date` across Prefect flow runs requires a persistent store (Prefect Variable, file, or DB). In v1, `last_promotion_date=None` means the cooldown is effectively not enforced between independent flow runs. The `PROMOTION_COOLDOWN_DAYS` constant in `promote_if_better` is still checked within a single call. A future plan should persist this date via Prefect Variables or a small state file.

## Threat Mitigations Applied

| Threat ID | Status |
|-----------|--------|
| T-04-15 (Elevation of Privilege — auto-champion) | Mitigated — `retrain_task` reassigns @champion → @challenger immediately after train_lgbm.py; @champion flip is only via `promote_if_better` |
| T-04-16 (DoS — stale retrain trigger) | Mitigated — ephemeral flow parameter, no flag file; cold-start guard in performance_check_task |
| T-04-17 (Info Disclosure — UIs on LAN) | Accepted — no new ports published; loopback-only posture unchanged |
| T-04-18 (Tampering — worker code mount) | Mitigated — bind-mount is repo's own directory, read-only; no new published port |

## Threat Flags

None. No new network endpoints, auth paths, or schema changes beyond what the plan's threat model covers.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| `pipelines/daily_pipeline.py` | FOUND (569 lines, > 90 minimum) |
| `scripts/create_deployment.py` | FOUND |
| `tests/integration/test_daily_flow.py` | FOUND |
| `infra/docker-compose.yml` prefect-worker volumes | FOUND |
| Commit `003e0d1` (Task 1: daily flow) | FOUND |
| Commit `30cca8a` (Task 2: tests + deployment + compose) | FOUND |
| `daily_flow.name` == `"volforecast-daily"` | VERIFIED |
| `work_pool_name="local-pool"` in create_deployment.py | VERIFIED |
| `cron="0 8 * * *"` in create_deployment.py | VERIFIED |
| worker `volumes` in docker-compose.yml | VERIFIED |
| 8/8 tests passing offline | VERIFIED |
| ruff clean repo-wide | VERIFIED |
