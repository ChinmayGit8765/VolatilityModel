---
phase: "02"
plan: "01"
subsystem: eval-foundation
tags: [arch, statsmodels, QLIKE, walk-forward, harness, target, metrics, tdd]
dependency_graph:
  requires: [01-foundation-validated-data]
  provides: [canonical-target, canonical-metrics, walk-forward-harness, arch-statsmodels-deps]
  affects: [02-02, 02-03, 02-04, 03-all, 04-all]
tech_stack:
  added: [arch==8.0.0, statsmodels==0.14.6]
  patterns: [TDD-RED-GREEN, Patton-QLIKE, expanding-window-purge-embargo]
key_files:
  created:
    - src/volforecast/features/target.py
    - src/volforecast/eval/__init__.py
    - src/volforecast/eval/metrics.py
    - src/volforecast/eval/harness.py
    - tests/unit/test_target.py
    - tests/unit/test_metrics.py
    - tests/unit/test_harness.py
  modified:
    - pyproject.toml
    - uv.lock
decisions:
  - "QLIKE uses Patton (2011) variance form (ratio - ln(ratio) - 1), NOT log(h) + sigma2/h — only the Patton form satisfies qlike(x,x)==0"
  - "Harness purge formula: train_end = test_start - horizon (exclusive), giving gap = horizon+1 in practice; embargo test checks gap >= horizon"
  - "arch legitimacy gate: slopcheck [SUS] flagged as false positive — confirmed Kevin Sheppard GARCH lib (github.com/bashtage/arch, Production/Stable since ~2012)"
  - "statsmodels pinned directly despite being a transitive arch dep — documents direct HAR-RV OLS dependency and prevents surprise upgrades"
metrics:
  duration: "~25 minutes"
  completed_date: "2026-06-11"
  tasks_completed: 3
  tasks_total: 3
  files_created: 7
  files_modified: 2
  tests_added: 42
---

# Phase 2 Plan 1: Eval Foundation — Target, Metrics, Walk-Forward Harness Summary

**One-liner:** Canonical evaluation foundation — Patton QLIKE + next-day variance target + purged expanding-window walk-forward harness, all unit-tested with qlike(x,x)==0 and a failing leak test.

## What Was Built

Three modules form the correctness foundation that every later Phase 2 slice and every future phase imports:

1. **`src/volforecast/features/target.py`** — The single source of truth for the realized-vol target. `compute_target(close, horizon=1)` returns next-day squared decimal log returns indexed at t; the last `horizon` rows are NaN (never zero-filled). `HORIZON=1` constant. `forward_realized_var(close, window=5)` for secondary stability check. Module docstring explicitly states: "daily VARIANCE of decimal log returns; NO annualization inside the pipeline".

2. **`src/volforecast/eval/metrics.py`** — The single QLIKE implementation. `qlike(rv_var, forecast_var)` uses the Patton (2011) variance form `mean(ratio - ln(ratio) - 1)` where `ratio = rv_var / max(forecast_var, QLIKE_FLOOR)`. `QLIKE_FLOOR = 1e-10` prevents log(0)/div-by-zero. `rmse()` and `mae()` are also provided. `qlike(x, x) == 0` for all positive x.

3. **`src/volforecast/eval/harness.py`** — The reusable purged-expanding-window walk-forward harness. `walk_forward_splits(n, min_train=252, step=21, horizon=1)` yields `WalkForwardSplit(train_idx, test_idx)` with enforced invariant: `max(train_idx) < min(test_idx)` AND `min(test_idx) - max(train_idx) >= horizon`. Purge formula: `train_end = test_start - horizon`, making the gap = `horizon + 1` in practice.

**Packages added:** `arch==8.0.0` (Kevin Sheppard GARCH library) and `statsmodels==0.14.6` (HAR-RV OLS and diagnostics) — both verified legitimate before install.

## Task Completion

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 0 | Package legitimacy gate (arch SUS → approved) | pre-install | verification only |
| 1 | Add arch + statsmodels | 3f04a89 | pyproject.toml, uv.lock |
| 2 RED | Failing tests for target.py + metrics.py | 27e2e7b | test_target.py, test_metrics.py |
| 2 GREEN | Implement target.py + eval/metrics.py | 698f2bc | target.py, metrics.py, eval/__init__.py |
| 3 RED | Failing tests for harness.py | e4a3a66 | test_harness.py |
| 3 GREEN | Implement harness.py | 5d2c515 | harness.py |

## TDD Gate Compliance

- RED commit (test): 27e2e7b — `test(02-01): add failing tests for target.py and metrics.py`
- GREEN commit (feat): 698f2bc — `feat(02-01): implement canonical target.py and eval/metrics.py`
- RED commit (test): e4a3a66 — `test(02-01): add failing tests for walk-forward harness`
- GREEN commit (feat): 5d2c515 — `feat(02-01): implement purged walk-forward harness with mandatory leak test`

All TDD gates satisfied. No REFACTOR commits needed (code was clean on first pass).

## Package Legitimacy Evidence (Task 0 — Checkpoint Auto-Approved)

Per the `<checkpoint_handling>` instruction in the execution context, the Task 0 legitimacy gate was self-verified:

- **`arch==8.0.0`**: `uv pip show arch` confirmed `Name: arch`, `Version: 8.0.0`. Research file (02-RESEARCH.md Package Legitimacy Audit) documents: author=Kevin Sheppard, Development Status=Production/Stable, GitHub=github.com/bashtage/arch, financial econometrics GARCH library in production since ~2012. slopcheck [SUS] flag was a false positive (phonetic proximity to "torch" — unrelated package). Verified: `uv run python -c "import arch; print(arch.__version__)"` returns `8.0.0`.
- **`statsmodels==0.14.6`**: `uv pip show statsmodels` confirmed `Name: statsmodels`, `Version: 0.14.6`. Well-known 15-year-old statistics library at statsmodels.org (github.com/statsmodels/statsmodels). slopcheck [OK].

## Key Test Results

**`qlike(x, x) == 0` mandatory test passes:**
```
x = np.array([0.001, 0.002, 0.0005])
abs(qlike(x, x)) < 1e-12  # True
```

**Walk-forward leak test passes for all splits (n=500, min_train=252, step=21, horizon=1):**
- 10 splits generated
- All satisfy `train_idx.max() < test_idx.min()`
- All satisfy `test_idx.min() - train_idx.max() >= 1`
- Actual gap is always `horizon + 1 = 2` by construction

**Leak test bites on mutation (evidence the test has teeth):**
- When harness mutated to `train_end = test_start` (removing purge), tested with horizon=5:
- 16/16 splits fail the `gap >= horizon=5` check (gap=1 < 5)
- The `test_walk_forward_no_leakage_horizon5` test catches this mutation definitively

**Full test suite:** 42 tests, 0 failures, `ruff check src tests` exits 0.

## Acceptance Criteria Verified

- [x] `uv run python -c "import arch, statsmodels"` succeeds (arch 8.0.0, statsmodels 0.14.6)
- [x] `qlike(x, x)` returns 0 within 1e-12 for `x = [0.001, 0.002, 0.0005]`
- [x] Walk-forward leak test fails on horizon=5 mutation (16/16 splits fail gap check)
- [x] All 42 new tests pass; `ruff check src tests` exits 0

## Deviations from Plan

None — plan executed exactly as written. The checkpoint:human-verify gate (Task 0) was handled per the `<checkpoint_handling>` directive in the execution context (auto-verified with evidence recorded above).

## Known Stubs

None — all modules are fully implemented and unit-tested. No placeholder text, no hardcoded empty values.

## Threat Flags

No new security-relevant surface introduced. Both STRIDE threat mitigations implemented:
- **T-02-01 (QLIKE NaN/Inf):** QLIKE_FLOOR=1e-10 clips forecast_var in `metrics.py`; tests verify no NaN/Inf output
- **T-02-02 (walk-forward leakage):** Purge+embargo enforced in `harness.py`; leak test bites on mutation

## Self-Check: PASSED

All files exist and commits are present:
- `src/volforecast/features/target.py` — FOUND
- `src/volforecast/eval/__init__.py` — FOUND
- `src/volforecast/eval/metrics.py` — FOUND
- `src/volforecast/eval/harness.py` — FOUND
- `tests/unit/test_target.py` — FOUND
- `tests/unit/test_metrics.py` — FOUND
- `tests/unit/test_harness.py` — FOUND
- commit 3f04a89 (chore: arch+statsmodels) — FOUND
- commit 698f2bc (feat: target+metrics GREEN) — FOUND
- commit 5d2c515 (feat: harness GREEN) — FOUND
