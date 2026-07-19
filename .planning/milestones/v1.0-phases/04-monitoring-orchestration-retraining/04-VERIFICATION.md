---
phase: 04-monitoring-orchestration-retraining
verified: 2026-06-15T00:00:00Z
status: passed
score: 5/5 must-haves verified
human_verification_resolved: >
  2026-07-19 — live deployment check executed by the orchestrator on the running compose
  stack. Deployment volforecast-daily registered from INSIDE the prefect-worker container
  (source=/repo; registering from the Windows host recorded a host path the Linux worker
  cannot chdir into — fixed operational procedure). Manual run 'daffodil-condor'
  (0ba41ea4) went Scheduled -> Running -> COMPLETED on work pool local-pool (cron 0 8 * * *).
  Evidence: ingest refreshed all 5 assets through 2026-07-17; labeller net_new_rows=166
  (forecast_vs_realized.parquet 5313 -> 5479 rows, including 10 champion rows as_of
  2026-06-10..11 alongside garch_baseline); 2026-07-19_drift.{html,json} written;
  performance check ran (should_retrain=False); @champion alias unchanged at v4 (no
  auto-promote). A latent defect was found and fixed during this verification: cli.py
  lacked a __main__ guard so the flow's `python -m volforecast.cli` ingest was a silent
  no-op (exit 0, nothing fetched); fixed with the guard + explicit `ingest` subcommand
  + regression tests (tests/unit/test_cli_entrypoint.py).
overrides_applied: 0
human_verification:
  - test: "Register volforecast-daily deployment against running compose Prefect server and trigger a manual run end-to-end"
    expected: "Deployment appears in Prefect UI under work pool local-pool with cron 0 8 * * *; manual run completes Scheduled -> Running -> Completed; data/monitoring/forecast_vs_realized.parquet gains rows; a dated _drift.html is written to data/monitoring/; @champion alias is unchanged (no auto-promote)"
    why_human: "Requires a running compose stack (prefect-server, prefect-worker, mlflow-server, postgres). All business logic is code-verified and offline-tested; only the live Prefect server integration with the worker code-visibility mount (vol ..:/repo:ro) can be confirmed by a human with docker-compose up."
---

# Phase 04: Monitoring + Orchestration + Retraining Verification Report

**Phase Goal:** The system closes its feedback loop — auto-arriving realized vol labels populate the forecast-vs-realized table, drift and performance degradation raise alerts and trigger retraining, and promotion is gated on rolling QLIKE with rollback as a single alias flip.

**Verified:** 2026-06-15
**Status:** human_needed
**Re-verification:** No — initial verification

All five success criteria pass automated checks. One human verification item remains (live Prefect deployment + end-to-end run), which was explicitly deferred by the plan as an operator step and documented in 04-05-SUMMARY.md. All code-level deliverables are substantive, wired, and data-flowing.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Forecast-vs-realized table populates automatically as realized vol arrives — no manual labelling step (MON-01) | VERIFIED | `label_forecasts(data_root, fvr_path) -> int` in `labeller.py:374` reads `predictions.parquet`, joins `compute_target`, appends atomically via `.tmp.parquet` + `os.replace`. GARCH baseline rows added for all configured assets. Run by `scripts/run_labeller.py` (thin CLI) and `label_task` in the Prefect flow. |
| 2 | Evidently reports feature and prediction distribution drift on a schedule; drift informs dashboards/logs and can flag retraining but NEVER auto-promotes a model (MON-02) | VERIFIED | `drift.py` module docstring explicitly states the locked anti-pattern. No `set_registered_model_alias`, no retrain flag in `drift.py`. `drift_check_task()` returns a `str` (output dir path), not a bool retrain signal. `DataDefinition(numerical_columns=[...])` always explicit (Pitfall 2). `result.save_html/save_json` on RESULT not Report (Pitfall 1). `evidently>=0.7.21,<0.8` in `pyproject.toml:29`. Frozen reference snapshot exists at `data/monitoring/reference/3_reference.parquet`. |
| 3 | When the live model's rolling QLIKE degrades relative to GARCH's rolling QLIKE, an alert is delivered via the configured channel (webhook or JSONL fallback) and retraining is flagged (MON-03, MON-04) | VERIFIED | `performance.py`: `PERF_WINDOW=21`, `DEGRADATION_THRESHOLD=0.10`, cold-start gate at `PERF_WINDOW//2=10` rows. Uses canonical `qlike()` from `eval.metrics` (not reimplemented). `alerts.py`: stdlib-only `urllib` POST to `ALERT_WEBHOOK_URL` env with `timeout=5`; on `(URLError, OSError)` falls through to `alerts.jsonl` append; function guaranteed never to raise. `run_performance_monitor` reads FVR parquet, fires `alert_fn` on degradation, returns `bool`. |
| 4 | The Prefect DAG (ingest → validate → features → label → drift-check → conditional retrain → eval → register → promotion-gate) completes end-to-end both on schedule and when triggered by the performance-drift flag (ORCH-01, ORCH-02) | VERIFIED (code-level) | `daily_pipeline.py` (569 lines) implements `@flow(name="volforecast-daily")` with locked task order. `should_retrain = performance_check_task() or force_retrain` — `force_retrain` is an ephemeral flow parameter (no flag file). `drift_check_task` returns `str`, never bool retrain signal. `scripts/create_deployment.py` uses `work_pool_name="local-pool"`, `cron="0 8 * * *"`. `infra/docker-compose.yml` prefect-worker has `volumes: [..:/repo:ro, ../data:/repo/data]` + `VOLFORECAST_ROOT: /repo`. Live deployment registration (Task 3 of 04-05) explicitly deferred to operator per plan checkpoint — treated as human verification item per phase instructions. |
| 5 | A challenger is promoted only if it beats the champion on rolling QLIKE over a frozen comparison window; default outcome is no-promote; rollback is demonstrated as a single alias flip (ORCH-03) | VERIFIED | `promotion.py`: `PROMOTION_COOLDOWN_DAYS=7` documented constant. `select_frozen_window()` is the single shared source before alias split — identical row set for both candidates. `promote_if_better()` returns False on tie or loss (strict-win: `challenger_qlike < champion_qlike`). `rollback_champion(to_version)` = single `set_registered_model_alias` call. Previous champion version recorded as `previous_champion_version` tag at promotion time. 16 offline unit tests pass with `FakeMlflowClient` stub. |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/volforecast/monitoring/labeller.py` | Idempotent labeller + GARCH live-forecast logging | VERIFIED | 436 lines (>80 min). Contains `def label_forecasts`, `FVR_SCHEMA`, `LABEL_KEY`, `append_labels`, `label_champion_forecasts`, `garch_live_forecasts`, `label_garch_baseline`. |
| `scripts/run_labeller.py` | CLI entry that runs the labeller against data/ paths | VERIFIED | Thin CLI: resolves `data_root = project_root()/"data"`, calls `label_forecasts`, exits 1 on `FileNotFoundError`. |
| `tests/unit/test_labeller.py` | Offline idempotency + join-correctness + GARCH-row tests | VERIFIED | 6 tests: idempotency, champion+garch coexist, no-forward-close, realized join correctness, decimal units guard, missing prediction log raises. |
| `src/volforecast/monitoring/drift.py` | Evidently 0.7 distribution-drift adapter | VERIFIED | Contains `DataDriftPreset`, `run_distribution_drift`, `select_numerical_columns`. Report-only, no promotion path. |
| `scripts/create_reference_snapshot.py` | One-time frozen reference snapshot creation | VERIFIED | Version-keyed (`{version}_reference.parquet`), refuses to overwrite same-version file, raises `FileNotFoundError` on missing feature parquets. |
| `tests/unit/test_drift.py` | Offline drift adapter tests | VERIFIED | 4 tests: api shape, numerical_columns honored, result-object call site, drift detected on shifted data. |
| `data/monitoring/reference/3_reference.parquet` | Frozen reference snapshot for champion v3 | VERIFIED | File exists (6,578 rows x 20 numerical feature columns). |
| `src/volforecast/monitoring/performance.py` | Rolling-QLIKE champion-vs-GARCH performance monitor | VERIFIED | 248 lines (>50 min). Contains `def check_performance_drift`, `PERF_WINDOW=21`, `DEGRADATION_THRESHOLD=0.10`. |
| `src/volforecast/monitoring/alerts.py` | Slack-compatible webhook POST with JSONL fallback | VERIFIED | Contains `def send_alert`. Stdlib only. Never raises. |
| `tests/unit/test_performance.py` | Cold-start, degradation-trigger, and no-trigger tests | VERIFIED | 7 tests (see test coverage details in 04-03-SUMMARY.md). |
| `tests/unit/test_alerts.py` | Webhook-success, webhook-failure-fallback, and env-fallback tests | VERIFIED | 4 tests. |
| `src/volforecast/monitoring/promotion.py` | QLIKE-gated MLflow alias-flip promotion + rollback | VERIFIED | Contains `set_registered_model_alias` (lines 263, 304). `PROMOTION_COOLDOWN_DAYS=7`. `select_frozen_window` shared source. |
| `tests/unit/test_promotion.py` | No-promote-default, promote-on-win, cooldown, and rollback tests | VERIFIED | 16 tests with `FakeMlflowClient` stub. |
| `pipelines/daily_pipeline.py` | Prefect @flow daily_flow wrapping Plans 01-04 | VERIFIED | 569 lines (>90 min). Contains `@flow`. Task order matches locked sequence. `force_retrain` ephemeral. |
| `scripts/create_deployment.py` | One-time deployment registration (local-pool, cron) | VERIFIED | Contains `work_pool_name="local-pool"`, `cron="0 8 * * *"`, reads `PREFECT_API_URL` from env. |
| `tests/integration/test_daily_flow.py` | Offline flow/task tests via prefect_test_harness + .fn() | VERIFIED | 8 tests. Session-scoped `prefect_test_harness` autouse fixture. |
| `infra/docker-compose.yml` | prefect-worker code-visibility volume mount | VERIFIED | `volumes: [..:/repo:ro, ../data:/repo/data]` + `VOLFORECAST_ROOT: /repo` + `MLFLOW_TRACKING_URI: http://mlflow-server:5000` in prefect-worker service. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `labeller.py` | `data/predictions/predictions.parquet` | `pd.read_parquet` of prediction log | WIRED | `label_forecasts:395` reads `pred_log_path = data_root/"predictions"/"predictions.parquet"` |
| `labeller.py` | `volforecast.features.target.compute_target` | realized-variance label from processed close | WIRED | `_realized_var_for_asset:99` calls `compute_target(df["close"])` |
| `labeller.py` | `volforecast.models.garch` | GARCH live forecast for the as-of window | WIRED | `garch_live_forecasts:251-252` imports `GARCH, GarchFitError` and calls `GARCH(min_train=252, step=21).forecast_path(ret)` |
| `drift.py` | `evidently Report/Dataset/DataDefinition` | `Report([DataDriftPreset()]).run(current_ds, reference_ds)` | WIRED | `drift.py:117-147` imports and uses all four symbols; `result.save_html/save_json` on RESULT object |
| `drift.py` | `data/monitoring/reference/{version}_reference.parquet` | frozen reference read (never written) | WIRED | `drift_check_task:218` reads `_reference_path("3")` as `reference_df`; drift.py itself reads but never writes the reference |
| `performance.py` | `volforecast.eval.metrics.qlike` | canonical QLIKE on champion and garch_baseline rows | WIRED | `performance.py:62` imports `qlike`; used at lines 158-165 |
| `performance.py` | `data/monitoring/forecast_vs_realized.parquet` | rolling window read of champion + garch_baseline rows | WIRED | `run_performance_monitor:219` reads `pd.read_parquet(fvr_path)` |
| `alerts.py` | `ALERT_WEBHOOK_URL` env / `data/monitoring/alerts.jsonl` | urllib POST with jsonl fallback | WIRED | `send_alert:102` reads env; `_post_webhook:169` uses `urllib.request.urlopen`; `_append_jsonl` writes fallback |
| `promotion.py` | `volforecast.eval.metrics.qlike` | frozen-window QLIKE for challenger vs champion | WIRED | `promotion.py:54` imports `qlike`; used at line 141 |
| `promotion.py` | `MlflowClient.set_registered_model_alias` | atomic @champion alias reassignment + rollback | WIRED | Lines 263 (`promote_if_better`) and 304 (`rollback_champion`) |
| `daily_pipeline.py` | `volforecast.monitoring.{labeller,drift,performance,promotion}` | thin @task wrappers over Plans 01-04 functions | WIRED | `label_task:186`, `drift_check_task:211`, `performance_check_task:295`, `promotion_gate_task:407-410` all import from monitoring modules |
| `scripts/create_deployment.py` | `prefect-server (compose) via PREFECT_API_URL` | `flow.from_source(repo_root).deploy(work_pool_name='local-pool', cron=...)` | WIRED (code-level) | Lines 79-91 implement `daily_flow.from_source(...).deploy(...)`. Live server registration deferred to operator. |
| `infra/docker-compose.yml` | prefect-worker code visibility | bind-mount the repo so local-process flow runs find pipelines/ | WIRED | `volumes: ..:/repo:ro` at compose level + `VOLFORECAST_ROOT: /repo` env var |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `labeller.py::append_labels` | `combined` (FVR rows) | `compute_target(df["close"])` from processed parquet + `GARCH.forecast_path(ret)` | Yes — DB/file reads of real processed OHLCV + GARCH computation | FLOWING |
| `performance.py::check_performance_drift` | `fvr` DataFrame | `pd.read_parquet(fvr_path)` — forecast_vs_realized.parquet | Yes — reads real monitoring parquet written by labeller | FLOWING |
| `promotion.py::frozen_window_qlike` | `alias_rows` | `select_frozen_window(fvr, ...)` then alias filter | Yes — reads from FVR parquet passed in; computes real QLIKE | FLOWING |
| `drift.py::run_distribution_drift` | `result` | `report.run(cur_dataset, ref_dataset)` — Evidently 0.7 Report | Yes — real Evidently computation against feature parquets | FLOWING |
| `daily_pipeline.py::drift_check_task` | `reference_df`, `current_df` | `pd.read_parquet(ref_path)` + per-asset feature parquets | Yes — reads real frozen reference + live feature parquets | FLOWING |

---

### Behavioral Spot-Checks

Code-level behavioral checks (no running server required):

| Behavior | Verification | Result | Status |
|----------|-------------|--------|--------|
| `labeller.py` imports without side effects | `FVR_SCHEMA`, `LABEL_KEY` defined; `LABEL_KEY == ["asset","as_of_date","model_alias"]` | Module-level constants confirmed in source at lines 50-62 | PASS |
| `drift.py` contains no promotion path | Grep for `set_registered_model_alias` in `drift.py` | Only string in docstring comment (`"  - flips an MLflow alias (set_registered_model_alias)"`) at line 11 — no call site | PASS |
| `promotion.py` `set_registered_model_alias("champion")` call sites | Grep across all *.py for `set_registered_model_alias` | In `promotion.py`: lines 263 (`promote_if_better`) and 304 (`rollback_champion`). In `daily_pipeline.py`: line 362 (restoring prev champion in `retrain_task` — documented per T-04-15). No other sites. | PASS (see WARNING-1 below) |
| `daily_pipeline.py` flow name | `daily_flow.name == "volforecast-daily"` | `@flow(name="volforecast-daily")` at line 498 | PASS |
| `scripts/create_deployment.py` work pool | `"local-pool"` present | Confirmed at line 90 | PASS |
| `infra/docker-compose.yml` worker volumes | `volumes` key in prefect-worker | `..:/repo:ro` and `../data:/repo/data` confirmed at lines 129-130 | PASS |
| `last_promotion_date=None` in flow | Cooldown not enforced across flow runs | `promotion_gate_task:471` passes `last_promotion_date=None` unconditionally | PASS (see WARNING-2 below) |
| No TBD/FIXME/XXX markers in phase files | Grep for debt markers | None found in `src/volforecast/monitoring/`, `pipelines/daily_pipeline.py`, `scripts/` phase files | PASS |

---

### Probe Execution

Step 7c: SKIPPED — no phase-declared probe scripts found; no `scripts/*/tests/probe-*.sh` pattern in project.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MON-01 | 04-01 | Feedback-loop labeller joins arrived realized vol against logged forecasts to produce a forecast-vs-realized table automatically | SATISFIED | `label_forecasts()` in `labeller.py`; `run_labeller.py` CLI; 6 unit tests |
| MON-02 | 04-02 | Evidently reports feature and prediction distribution drift on a schedule (dashboard/log only, not auto-promotion) | SATISFIED | `drift.py` with Evidently 0.7.21; `DataDriftPreset`; report-only docstring; reference snapshot at `data/monitoring/reference/3_reference.parquet` |
| MON-03 | 04-03 | Performance degradation (rolling live QLIKE vs GARCH's rolling QLIKE) triggers an alert and flags retraining | SATISFIED | `performance.py::check_performance_drift` + `run_performance_monitor`; PERF_WINDOW=21, DEGRADATION_THRESHOLD=0.10; wired to `performance_check_task` in daily flow |
| MON-04 | 04-03 | Alerts are delivered via a configurable channel (e.g., Slack webhook or email) | SATISFIED | `alerts.py::send_alert`; ALERT_WEBHOOK_URL env + JSONL fallback; never raises; stdlib only |
| ORCH-01 | 04-05 | Prefect DAG runs ingest → validate → features → train → eval → register end-to-end on a schedule | SATISFIED (code-level) | `daily_pipeline.py` @flow with locked task order; `create_deployment.py` with `cron="0 8 * * *"`; live deploy deferred to operator |
| ORCH-02 | 04-05 | Retraining can be triggered by the performance-drift flag in addition to schedule | SATISFIED | `should_retrain = performance_check_task() or force_retrain` at line 531; `force_retrain` is ephemeral flow parameter |
| ORCH-03 | 04-04 | Champion/challenger gate promotes the challenger only if it beats the champion on rolling QLIKE over a frozen comparison window; default outcome is no-promote; rollback is an alias flip | SATISFIED | `promotion.py::promote_if_better` (strict-win + cooldown) + `rollback_champion` (single alias flip); 16 unit tests with mocked client |

No orphaned requirements for Phase 4 found. All 7 declared requirement IDs (MON-01 through MON-04, ORCH-01 through ORCH-03) are mapped to plans and have implementation evidence.

---

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `pipelines/daily_pipeline.py:362` | `set_registered_model_alias(model_name, "champion", prev_version)` inside `retrain_task` | WARNING | This is a third @champion alias-flip call site outside `promote_if_better`/`rollback_champion`. Documented in SUMMARY as intentional T-04-15 mitigation (restoring the prior champion after `train_lgbm.py` auto-promotes). Functionally correct — it is a restoration, not a forward promotion. The `promotion.py` module docstring's "grep-verifiable single code path" claim is technically inaccurate. Does not compromise the promotion gate. |
| `pipelines/daily_pipeline.py:471` | `last_promotion_date=None` passed unconditionally to `promote_if_better` | WARNING | The 7-day cooldown between promotions is enforced within a single flow run (the constant is checked) but the date is not persisted across runs. Documented in 04-05-SUMMARY.md as a known stub — "a future plan should persist this date via Prefect Variables or a small state file." Effective cooldown across runs = 0 days. |
| `pipelines/daily_pipeline.py:218` | `_reference_path("3")` — champion version hardcoded to "3" | INFO | Drift check always loads the v3 snapshot regardless of promoted champion. Documented in SUMMARY as a v1 acceptable limitation. Would require updating when champion is promoted. Non-blocking for current state (champion is still v3). |

No TBD/FIXME/XXX debt markers found in any phase-modified file.

---

### Human Verification Required

#### 1. Live Prefect Deployment Registration and End-to-End Flow Run

**Test:** Start the compose stack (`docker compose -f infra/docker-compose.yml up -d`). From the HOST shell, run `PREFECT_API_URL=http://localhost:4200/api uv run python scripts/create_deployment.py`. Open the Prefect UI at http://localhost:4200. Then trigger a manual run via the UI "Run → Quick run" or `PREFECT_API_URL=http://localhost:4200/api uv run prefect deployment run 'volforecast-daily/volforecast-daily'`.

**Expected:**
1. `create_deployment.py` prints "Deployment registered successfully" with no error.
2. Prefect UI shows `volforecast-daily` under work pool `local-pool` with schedule `0 8 * * *`.
3. Manual run transitions Scheduled → Running → Completed (NOT stuck Scheduled — that indicates Pitfall 4 work-pool-name mismatch or worker cannot find the code via the `..:/repo:ro` mount).
4. `data/monitoring/forecast_vs_realized.parquet` gains rows.
5. A dated `data/monitoring/{date}_drift.html` is written.
6. The `@champion` alias in MLflow is unchanged (no auto-promote — default no-promote confirmed live).

**Why human:** Requires a running compose stack with postgres, prefect-server, prefect-worker, and mlflow-server. The worker bind-mount (`..:/repo:ro`) resolution on Windows + WSL2 Docker Desktop backend must be confirmed in the actual environment. The offline tests (8 tests via `prefect_test_harness`) cover all business logic; this step verifies the infrastructure wiring only.

---

### Gaps Summary

No gaps blocking goal achievement. All must-have truths are VERIFIED at the code level.

Two warnings are documented but do not block the phase goal:

**WARNING-1 (Cooldown not persisted across flow runs):** `last_promotion_date=None` in `promotion_gate_task` means the 7-day cooldown is not enforced between independent Prefect flow runs. Within a single run it functions correctly. The SUMMARY acknowledges this as a v1 limitation requiring a future Prefect Variable or state file. The gate's strict-win QLIKE requirement still protects against spurious promotions. This is acceptable for v1 but should be addressed in a future phase.

**WARNING-2 (Third @champion alias-flip site in retrain_task):** The `promotion.py` module docstring's "grep-verifiable single code path" claim is technically inaccurate because `daily_pipeline.py:362` also sets @champion. This is an intentional design choice (T-04-15) to restore the previous champion after `train_lgbm.py` auto-promotes, documented in the SUMMARY. The promotion gate is not bypassed by this call — it restores the prior champion, the gate then decides whether to promote the challenger. The docstring comment should be updated to acknowledge this third site.

---

_Verified: 2026-06-15_
_Verifier: Claude (gsd-verifier)_
