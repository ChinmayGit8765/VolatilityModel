---
phase: 04-monitoring-orchestration-retraining
plan: "04"
subsystem: monitoring/promotion
tags: [mlops, champion-challenger, qlike, mlflow, promotion-gate, rollback, tdd]
dependency_graph:
  requires:
    - 04-01  # labeller FVR parquet + FVR_SCHEMA
    - eval/metrics.py  # canonical qlike()
  provides:
    - src/volforecast/monitoring/promotion.py  # QLIKE-gated alias-flip gate
  affects:
    - MLflow registry @champion alias (gated; never direct)
tech_stack:
  added: []
  patterns:
    - Frozen-window QLIKE comparison via shared select_frozen_window (identical rows)
    - Injectable MLflow client pattern for offline unit tests
    - Strict-win + cooldown gate (default no-promote)
    - Rollback via single set_registered_model_alias call + previous_champion_version tag
key_files:
  created:
    - src/volforecast/monitoring/promotion.py
    - tests/unit/test_promotion.py
  modified: []
decisions:
  - select_frozen_window used as single shared source before alias filter — guarantees identical row coverage for both candidates (T-04-12)
  - client injectable kwarg — allows hermetic offline testing with FakeMlflowClient stub without mocking mlflow at import time
  - Strict-win threshold — challenger_qlike < champion_qlike (not <=); tie is no-promote (documented in module docstring)
  - Cooldown boundary is (today - last_promotion_date).days >= PROMOTION_COOLDOWN_DAYS (exclusive: exactly 7 days elapsed allows promotion)
  - rollback_champion sets no tags — it is a pure alias flip; tag is set at promotion time on the new champion, not at rollback time
metrics:
  duration_minutes: 9
  completed_date: "2026-06-14"
  tasks_completed: 1
  files_created: 2
  files_modified: 0
---

# Phase 04 Plan 04: Champion/Challenger Promotion Gate Summary

**One-liner:** QLIKE-gated MLflow alias-flip promotion gate with frozen identical-row window, 7-day cooldown, and single-call rollback via injectable client.

## What Was Built

`src/volforecast/monitoring/promotion.py` implements the ORCH-03 promotion gate:

- `select_frozen_window(fvr, window_start, window_end)` — shared date-range filter used by both candidates, ensuring identical row coverage before alias split (T-04-12 mitigation)
- `frozen_window_qlike(fvr, model_alias, window_start, window_end)` — computes canonical QLIKE (from `volforecast.eval.metrics.qlike`) for a single alias on the frozen window; returns score + provenance dict (n_rows, window bounds, key-pair hash)
- `promote_if_better(challenger_qlike, champion_qlike, last_promotion_date, today, *, challenger_version, client)` — strict-win + 7-day cooldown gate; records previous champion version as model tag before flipping; default outcome is no-promote
- `rollback_champion(to_version, *, client)` — single `set_registered_model_alias` call; no tag writes

`tests/unit/test_promotion.py` — 16 hermetic offline tests using `FakeMlflowClient` stub:
- Test 1 (no-promote default): tie and losing challenger both return False with no alias flip
- Test 2 (promote on strict win): alias flipped exactly once; `previous_champion_version` tag recorded on new champion version
- Test 3 (cooldown blocks): 3 days, 6 days elapsed — blocked; exactly 7 days — allowed
- Test 4 (identical-rows guard): both candidates produce same n_rows and window bounds from the single `select_frozen_window` source
- Test 5 (rollback): exactly one alias flip, no tags, uses MODEL_NAME constant

## TDD Gate Compliance

- RED commit `2cec453`: `test(04-04): add failing promotion gate tests (RED)` — all 16 tests fail (ModuleNotFoundError — promotion module not yet created)
- GREEN commit `1bac06e`: `feat(04-04): implement QLIKE-gated champion/challenger promotion gate (GREEN)` — all 16 tests pass

## Acceptance Criteria Verification

| Criterion | Status |
|-----------|--------|
| All 5 promotion tests pass with mocked MLflow client | PASS — 16 tests, 0 failures |
| Default path is no-promote: alias NOT called on tie/loss | PASS — TestNoPromoteDefault |
| Strict win + elapsed cooldown: alias flipped once, prev version tagged | PASS — TestPromoteOnStrictWin |
| Champion and challenger QLIKE on identical frozen-window rows | PASS — TestIdenticalRowsGuard via select_frozen_window |
| QLIKE uses volforecast.eval.metrics.qlike only | PASS — single import in promotion.py |
| PROMOTION_COOLDOWN_DAYS=7 documented constant | PASS — TestConstants |
| set_registered_model_alias "champion" only in promote_if_better/rollback_champion | PASS — grep verified (lines 263 and 304 only) |
| uv run ruff check clean | PASS — project-wide |

## Deviations from Plan

None — plan executed exactly as written.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced beyond what the plan's threat model covers. The `set_registered_model_alias` single-code-path constraint (T-04-11) is grep-verifiable and enforced via module docstring.

## Self-Check

Files exist:
- src/volforecast/monitoring/promotion.py — FOUND
- tests/unit/test_promotion.py — FOUND

Commits exist:
- 2cec453 (RED) — FOUND
- 1bac06e (GREEN) — FOUND

## Self-Check: PASSED
