---
phase: "04"
plan: "03"
subsystem: monitoring
tags: [monitoring, alerts, performance-monitor, qlike, tdd, mlo-ops]
dependency_graph:
  requires:
    - "04-01"  # labeller/forecast_vs_realized contract (FVR_SCHEMA, model_alias values)
    - "02-*"   # eval/metrics.py canonical QLIKE function
  provides:
    - "volforecast.monitoring.alerts.send_alert"
    - "volforecast.monitoring.performance.check_performance_drift"
    - "volforecast.monitoring.performance.run_performance_monitor"
  affects:
    - "04-05"  # Prefect daily flow (Plan 05) consumes run_performance_monitor return value
tech_stack:
  added: []
  patterns:
    - "TDD RED/GREEN with monkeypatch for network isolation"
    - "Fire-and-forget alert with stdlib urllib (no new deps)"
    - "Canonical QLIKE reuse: import from eval.metrics, no re-implementation"
    - "Dependency injection via alert_fn parameter for test hermeticity"
    - "Cold-start gate: min(n_champion, n_garch) >= PERF_WINDOW // 2"
key_files:
  created:
    - src/volforecast/monitoring/alerts.py
    - src/volforecast/monitoring/performance.py
    - tests/unit/test_alerts.py
    - tests/unit/test_performance.py
  modified: []
decisions:
  - "PERF_WINDOW=21: one calendar month of trading days; long enough to distinguish signal from noise, short enough to detect degradation within a month (Claude's discretion, documented)"
  - "DEGRADATION_THRESHOLD=0.10: 10% relative QLIKE underperformance; midpoint of credible [5%, 25%] range; avoids retrain storms while detecting real regime shifts (Claude's discretion, documented)"
  - "_MIN_ROWS=PERF_WINDOW//2=10: minimum rows per alias to avoid cold-start false alerts (Pitfall 3)"
  - "QLIKE computed via canonical volforecast.eval.metrics.qlike only — no re-implementation (T-04-09)"
  - "send_alert: injectable webhook_url and jsonl_path args for test hermeticity; env read at call time not module load"
  - "alert_fn injected into run_performance_monitor for test stub isolation without monkeypatching imports"
metrics:
  duration: "~25 minutes"
  completed: "2026-06-14"
  tasks_completed: 2
  tests_added: 11
  lines_added: 952
---

# Phase 04 Plan 03: Performance Monitor + Alert Delivery Summary

Rolling-QLIKE champion-vs-GARCH performance monitor with Slack-compatible webhook alert + JSONL fallback (MON-03, MON-04), built TDD with full cold-start safety and canonical QLIKE reuse.

## What Was Built

### Task 1: Alert delivery (MON-04) — `src/volforecast/monitoring/alerts.py`

`send_alert(title, body, *, webhook_url=None, jsonl_path=None)` — fire-and-forget alert delivery:

- Resolves webhook URL from injected arg then `ALERT_WEBHOOK_URL` env var (never hardcoded)
- POSTs Slack-compatible JSON `{"text": "*title*\n{body}"}` via stdlib `urllib.request` with `timeout=5`
- On `(URLError, OSError)` catches, logs, and falls back to JSONL append
- JSONL fallback: `mkdir -p` + append one JSON line `{timestamp, title, **body}` to `alerts.jsonl`
- On JSONL write failure: catches `OSError`, logs, swallows — pipeline never crashes
- Payload contains only title + caller-supplied body dict (no env values, no tokens — T-04-07)
- Stdlib only (`json`, `os`, `urllib`, `datetime`) — no new dependencies

### Task 2: Performance monitor (MON-03) — `src/volforecast/monitoring/performance.py`

`check_performance_drift(fvr: pd.DataFrame) -> tuple[bool, dict]`:

- Sorts by `as_of_date`, takes tail `PERF_WINDOW * 2` rows, splits by `model_alias`
- Cold-start gate: if `min(n_champion, n_garch) < PERF_WINDOW // 2` returns `(False, {"reason": "cold_start", ...})`
- Computes QLIKE via `volforecast.eval.metrics.qlike` (canonical, no re-implementation — T-04-09)
- Degradation: `champ_qlike > garch_qlike * (1 + DEGRADATION_THRESHOLD)` → returns `(True, report_dict)`
- Report includes: `champion_qlike`, `garch_qlike`, `threshold`, `n_champion`, `n_garch`, `degraded`

`run_performance_monitor(fvr_path, *, alert_fn=send_alert) -> bool`:

- Reads parquet from path, calls `check_performance_drift`, fires `alert_fn` on degradation
- Returns bool for Prefect daily flow (Plan 05) conditional retrain branch
- `alert_fn` injected (default: real `send_alert`) for test hermeticity

### Constants (both documented with rationale in module docstring)

| Constant | Value | Rationale |
|----------|-------|-----------|
| `PERF_WINDOW` | 21 | 1 calendar month of trading days; signal-to-noise balance |
| `DEGRADATION_THRESHOLD` | 0.10 | 10% relative QLIKE gap; midpoint of credible [5%, 25%] range |
| `_MIN_ROWS` | 10 | `PERF_WINDOW // 2`; cold-start false alert prevention |

## Test Coverage (11 tests, all hermetic)

### test_alerts.py (4 tests)
| Test | Verifies |
|------|----------|
| webhook success | urlopen called once; no jsonl written |
| webhook failure fallback | URLError swallowed; 1 jsonl record appended; no raise |
| env unset fallback | urlopen NOT called; jsonl written |
| no secret leakage | payload has "text" key + title; SOME_SECRET and webhook URL token not in body |

### test_performance.py (7 tests)
| Test | Verifies |
|------|----------|
| cold_start short window | < PERF_WINDOW//2 rows → (False, cold_start) |
| cold_start empty frame | empty DataFrame → (False, cold_start), no raise |
| degraded champion triggers | champ_scale=50x, garch_scale=1x → should_retrain=True, qlike values in report |
| degraded triggers 1 alert | run_performance_monitor fires injected alert_fn exactly once |
| comparable no trigger | equal-quality models → should_retrain=False |
| better champion no alert | champion better → run_performance_monitor fires 0 alerts |
| canonical metric reuse | known-input QLIKE matches eval.metrics.qlike output exactly |

## Verification Results

```
uv run pytest tests/unit/test_performance.py tests/unit/test_alerts.py -q
...........
11 passed in 0.37s

uv run ruff check .
All checks passed!
```

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 2deaea1 | test (RED) | Add failing alert delivery tests |
| 1a46876 | feat (GREEN) | Implement alert delivery with webhook + JSONL fallback |
| e639008 | test (RED) | Add failing performance monitor tests |
| 5130222 | feat (GREEN) | Implement rolling-QLIKE champion-vs-GARCH performance monitor |

## Deviations from Plan

None — plan executed exactly as written.

- Both TDD RED/GREEN cycles completed correctly (tests failed before implementation, all passed after)
- All 4 plan STRIDE threats mitigated: T-04-07 (no secrets in payload), T-04-08 (never raises), T-04-09 (canonical QLIKE), T-04-10 (monitor returns flag only, never flips alias)
- Ruff UP017 (`datetime.UTC` alias) and UP035 (`collections.abc.Callable`) auto-fixed by `ruff --fix` — clean after fix

## Known Stubs

None. Both `alerts.py` and `performance.py` are fully wired. `send_alert` has injectable defaults (`webhook_url=None`, `jsonl_path=None`) but these are intentional API design for testability, not stubs — the defaults resolve to the real env var and `data/monitoring/alerts.jsonl`.

## Threat Surface Scan

No new network endpoints, auth paths, or file access patterns beyond what is declared in the plan's threat model. All four declared threats (T-04-07 through T-04-10) are mitigated by the implementation. No additional threat flags.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| src/volforecast/monitoring/alerts.py | FOUND |
| src/volforecast/monitoring/performance.py | FOUND |
| tests/unit/test_alerts.py | FOUND |
| tests/unit/test_performance.py | FOUND |
| commit 2deaea1 (RED alerts) | FOUND |
| commit 1a46876 (GREEN alerts) | FOUND |
| commit e639008 (RED performance) | FOUND |
| commit 5130222 (GREEN performance) | FOUND |
