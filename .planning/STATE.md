# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-10)

**Core value:** Honestly benchmark an ML volatility model against the correct classical baselines (GARCH(1,1)/EWMA/HAR-RV) under leak-free walk-forward evaluation, inside a genuine end-to-end MLOps lifecycle.
**Current focus:** Phase 1 — Foundation & Validated Data

## Current Position

Phase: 1 of 5 (Foundation & Validated Data)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-06-10 — Roadmap created (5 phases, 35/35 v1 requirements mapped)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: HAR-RV added as third classical baseline (research: ~30 lines, outsized quant credibility)
- [Roadmap]: MLflow aliases (`@champion`/`@challenger`) replace deprecated registry stages
- [Roadmap]: pandas-only for v1 (MLflow 3.x pins `pandas<3`); Python 3.12 forced by SHAP 0.52
- [Roadmap]: Eval harness built BEFORE the ML model (Phase 2 before Phase 3) — prevents motivated reasoning
- [Roadmap]: Drift triggers retraining + evaluation only, never auto-promotion

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 2 planning]: Purged walk-forward with embargo for multi-asset panels has no off-the-shelf implementation — highest methodological risk; needs deeper research during phase planning
- [Phase 4 planning]: Evidently 0.7 API is a rewrite (older examples broken); Prefect 3 drift-trigger wiring is the least-documented integration in the stack
- [Contingency]: ML may genuinely not beat GARCH at daily horizon — Phases 3 and 5 bake in regime-segmented honest reporting so the project succeeds either way
- [Data]: Binance may be geo-blocked from CI/cloud runners — fixtures mandatory in CI (Phase 1); verify ccxt incomplete-candle and yfinance `auto_adjust` behavior against pinned versions on first run

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-10
Stopped at: Roadmap and state initialized; ready for `/gsd:plan-phase 1`
Resume file: None
