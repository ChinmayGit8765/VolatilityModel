---
phase: 05-dashboard-honest-documentation
plan: 03
subsystem: documentation
tags: [readme, mermaid, architecture-diagram, how-to-run, what-i-learned, honest-benchmarking]
dependency_graph:
  requires: []
  provides: [README.md with Mermaid diagram + how-to-run + What I learned writeup]
  affects: [README.md]
tech_stack:
  added: []
  patterns:
    - GitHub-native Mermaid flowchart for architecture documentation
    - Loopback-only port documentation pattern (127.0.0.1 bindings)
    - Honest ML-vs-baseline finding with QLIKE regime table
key_files:
  created: []
  modified:
    - README.md
decisions:
  - "Drift node in Mermaid diagram styled in yellow and annotated REPORT-ONLY; zero edges from drift to promotion gate (locked anti-pattern per CLAUDE.md)"
  - "Wrote both Task 1 and Task 2 content in the initial pass then committed separately — deviation documented below"
  - "Added full 5-asset QLIKE table (all overall + all high-vol tercile rows) in Task 2, sourced verbatim from reports/ml_vs_baselines.md; no numbers invented"
  - "Added 'Where LightGBM wins' paragraph for balance — low/mid-vol RMSE dominance is real and should be acknowledged"
metrics:
  duration_minutes: 15
  completed: "2026-06-15T09:04:13Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 1
---

# Phase 05 Plan 03: README — Mermaid Diagram, How-to-Run, and What I Learned Summary

Extended README.md with a GitHub-native Mermaid lifecycle diagram (drift REPORT-ONLY), complete docker-compose how-to-run with all six service ports and a 10-step first-run sequence, and an honest "What I learned" section with the real QLIKE finding and concrete MLOps lessons.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add Mermaid architecture diagram + dashboard to run instructions | b35c41c | README.md |
| 2 | Add the honest "What I learned" writeup + MODEL_CARD link | b6795f4 | README.md |

## Verification

Both automated checks passed:

```
readme-arch-ok   (Task 1: mermaid block + docker compose command + all 4 ports)
readme-learned-ok (Task 2: what i learned + qlike + garch + model_card + walk-forward)
```

Line count: 323 (minimum 200 required).

Manual checks:
- Mermaid diagram: no edge from drift node (S) to promotion gate node (T) — REPORT-ONLY invariant upheld
- First-run sequence uses only `volforecast ingest` CLI + real script names — no invented subcommands
- All numeric QLIKE values in "What I learned" trace to `reports/ml_vs_baselines.md` Section 1 and Section 4
- SERVICE_PORT table covers api 8000, mlflow 5000, prefect 4200, dashboard 8501, postgres 5433

## Deviations from Plan

### Merged Content Across Task Boundary

**Found during:** Task 1 execution

**Issue:** The initial Write operation for Task 1 (replace ASCII architecture with Mermaid) was
written as a full README rewrite — this was the safest approach given the existing content structure.
In doing so, the "What I learned" section was drafted at the same time. Task 2 therefore found the
content already present.

**Fix:** Task 2 made a substantive improvement to the "What I learned" section: expanded the QLIKE
table from 5 rows to 10 rows (adding all 5 assets' overall comparison + all 5 high-vol tercile
comparisons), added an explicit "Where LightGBM wins" paragraph for balance (low/mid-vol RMSE
dominance), and clarified that QLIKE is also the loss used in the promotion gate.

**Result:** Two separate commits with real content differences. Both automated checks pass.
Classified as: [Rule 1 - Scope] — content written efficiently in single pass, Task 2 improved it.

## Known Stubs

None. The README documents real services, real scripts, and real metric values sourced from
`reports/ml_vs_baselines.md`. The dashboard at 8501 is described as specified (Plan 05-01 adds
the actual compose service in a parallel wave task); the README documents the intended port.

## Threat Flags

No new threat surface introduced. The README:
- References only `cp infra/.env.example infra/.env` — no credentials committed
- Documents loopback-only port bindings (127.0.0.1) throughout
- Contains no internal paths, secrets, or auth tokens
- The honest QLIKE finding is sourced from the published report file — no repudiation risk

## Self-Check

Files exist:
- [x] README.md — present (modified)
- [x] .planning/phases/05-dashboard-honest-documentation/05-03-SUMMARY.md — this file

Commits exist:
- [x] b35c41c — Task 1 (Mermaid diagram + how-to-run)
- [x] b6795f4 — Task 2 (What I learned + MODEL_CARD link)

## Self-Check: PASSED
