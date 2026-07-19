---
phase: 05-dashboard-honest-documentation
plan: "02"
subsystem: docs
tags: [model-card, honest-benchmarking, evaluation, lightgbm, qlike, walk-forward]
dependency_graph:
  requires: []
  provides: [MODEL_CARD.md]
  affects: [README.md]
tech_stack:
  added: []
  patterns: [honest-benchmarking, regime-segmented-evaluation, verbatim-metric-sourcing]
key_files:
  created:
    - MODEL_CARD.md
  modified: []
decisions:
  - "All metrics copied verbatim from reports/ml_vs_baselines.{md,csv} and reports/baseline_eval.md; zero invented numbers"
  - "Section 8 (Honest Findings) reproduces the Section 4 losses list in full with tabular format for clarity"
  - "Per-calendar-year view limited to two representative assets (BTC-USD + SPY) to control card length while retaining the inter-year QLIKE pattern"
  - "High-vol QLIKE catastrophic losses (BTC 11.50 vs GARCH 0.68; ETH 13.88 vs HAR 1.10) are given their own bolded subsection as the most important finding"
metrics:
  duration_minutes: ~12
  completed_date: "2026-06-15"
  tasks_completed: 1
  tasks_total: 1
  files_changed: 1
---

# Phase 05 Plan 02: MODEL_CARD.md — Honest Regime-Segmented Benchmarking Summary

## One-liner

Honest MODEL_CARD.md for `volforecast-lgbm` with verbatim per-asset and per-regime QLIKE/RMSE/MAE tables, a named-losses section that explicitly calls out every high-vol QLIKE blowup (BTC 11.50 vs GARCH 0.68; ETH 13.88 vs HAR 1.10), QLIKE methodology (Patton 2011), walk-forward protocol (min_train=252, step=21), and all v1 known limitations including the promotion-cooldown-not-persisted issue.

## What Was Built

`MODEL_CARD.md` (409 lines) at the repo root covers:

1. **Model details** — `volforecast-lgbm`, LightGBM 4.6 regressor, MLflow registry with `@champion`/`@challenger` aliases (not deprecated stages)
2. **Intended use** — volatility-forecasting + MLOps DEMO; explicitly NOT a trading strategy, NOT price-direction prediction, NOT for live risk decisions
3. **Training data and window** — BTC-USD, ETH-USD, SPY, AAPL, MSFT daily OHLCV ~2022–2026; ccxt/Binance + yfinance; DVC-versioned
4. **Evaluation methodology** — purged walk-forward expanding window (min_train=252, step=21), identical folds for LightGBM and all three baselines
5. **Metric definitions** — QLIKE (Patton 2011 variance form, =0 at perfect forecast), RMSE, MAE; all in daily decimal variance units
6. **Per-asset overall results table** — LightGBM vs EWMA/GARCH/HAR for all 5 assets (20 rows, copied verbatim from Section 1 of `reports/ml_vs_baselines.md`)
7. **Per-vol-tercile breakdown** — 5 × 12 = 60 rows covering low/mid/high tercile for each asset (Section 2 verbatim)
8. **Per-calendar-year view** — BTC-USD and SPY QLIKE by year 2022–2026 (representative; from Section 3)
9. **Honest findings** — Section 8 explicitly names all overall losses and every high-vol regime QLIKE blowup with deltas; explains the mechanism (QLIKE penalises under-forecasting; tree model under-predicts vol spikes)
10. **Assumptions** — daily horizon, decimal-variance unit convention, RV proxy, GARCH scaling, EWMA lambda, HAR lags, walk-forward guarantee
11. **Known limitations** — QLIKE high-vol weakness, promotion-cooldown-not-persisted (v1 limitation from Phase 4), no true HF RV, single-exchange crypto, free-data quality
12. **Machine-readable provenance** — citation of `reports/ml_vs_baselines.csv` and generation scripts

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Author MODEL_CARD.md from the report files | b40c76b | MODEL_CARD.md (created, 409 lines) |

## Verification Results

Automated content checks (from plan `<verify>` block) — all passed:
- `QLIKE` present
- `walk-forward` present (case-insensitive)
- `not a trading strategy` present (case-insensitive)
- `lightgbm` appears >= 5 times
- `GARCH`, `EWMA`, `HAR` all present
- High-vol QLIKE loss number (`11.499663`) present

Manual spot checks — all passed:
- `11.499663` — BTC-USD high-vol LightGBM QLIKE (matches `reports/ml_vs_baselines.md` line 69)
- `13.878358` — ETH-USD high-vol LightGBM QLIKE (matches line 86)
- `3.625649` — AAPL overall LightGBM QLIKE (matches line 15)
- `1.634448` — SPY EWMA overall QLIKE (matches line 33)
- `0.680276` — BTC-USD GARCH high-vol QLIKE (matches line 71)
- `1.100224` — ETH-USD HAR high-vol QLIKE (matches line 89)
- `min_train=252` and `step=21` present in methodology table
- `Patton` (2011 QLIKE reference) present
- Promotion-cooldown-not-persisted documented in Section 10
- `ml_vs_baselines` cited as provenance

Line count: 409 (requirement: >= 90)

## Deviations from Plan

None — plan executed exactly as written.

The per-calendar-year section uses two representative assets (BTC-USD and SPY) rather than all
five to keep the card readable, which is within the plan's "at minimum a representative
per-calendar-year view" specification.

## Threat Flag Review

No new security-relevant surface introduced. MODEL_CARD.md contains only published evaluation
metrics from committed report files plus methodology prose — no credentials, API keys, or
infrastructure secrets.

## Self-Check

- [x] `MODEL_CARD.md` exists: FOUND
- [x] Commit `b40c76b` exists: FOUND
- [x] No unexpected file deletions in commit

## Self-Check: PASSED
