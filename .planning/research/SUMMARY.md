# Project Research Summary

**Project:** VolForecast — volatility forecasting MLOps platform
**Domain:** Financial time-series ML (crypto + equities realized-vol forecasting) wrapped in a full MLOps lifecycle (portfolio/skill-demo project)
**Researched:** 2026-06-10
**Confidence:** HIGH

## Executive Summary

VolForecast is a batch-oriented forecasting platform with an API on top — not a streaming system. Because the forecast horizon is daily, experts build this as a canonical MLOps loop (ingest → validate → features → train → walk-forward eval → register → serve → monitor → retrain) with a thin FastAPI serving layer and a closed feedback loop. The project's structural advantage is that volatility labels arrive automatically (tomorrow's realized vol IS the ground truth), making the monitoring/retraining loop genuine rather than simulated — almost no portfolio project has this. The differentiation is the *combination*: correct quant methodology (GARCH/EWMA/HAR-RV baselines, QLIKE, purged walk-forward) plus a complete MLOps lifecycle (MLflow alias-based registry, Evidently drift, QLIKE-gated champion/challenger promotion, drift-triggered retraining). Research found no existing project occupying that intersection.

The recommended stack is Python 3.12 (forced by SHAP 0.52), pandas 2.3.x (MLflow 3.13 hard-pins `pandas<3` — do NOT use pandas 3.0), arch/LightGBM for models, MLflow 3 + Prefect 3 + Evidently 0.7 + Pandera + DVC for the lifecycle, FastAPI + Streamlit for serving/observability, all on docker-compose with a Postgres backing store (never SQLite on a Windows bind mount). Two PROJECT.md requirements need rewording: MLflow registry "stages" are deprecated — use `@champion`/`@challenger` aliases — and "pandas/Polars" should resolve to pandas-only for v1.

The dominant risks are methodological, not infrastructural: (1) lookahead leakage via overlapping multi-day RV label windows — the eval harness must use purged walk-forward with an embargo ≥ label horizon and be built BEFORE any ML model; (2) a misfit GARCH baseline (scale/convergence failures) turning the headline comparison into a strawman; (3) a buggy QLIKE metric silently corrupting champion/challenger promotion; (4) drift detection that cries wolf on every vol regime shift — drift must trigger retraining+evaluation only, never auto-promotion. A real secondary risk: ML may genuinely not beat GARCH at daily horizon (the literature says HAR/GARCH are hard to beat). Mitigation is to reframe success now — the deliverable is the honest benchmark + the platform, with regime-segmented results ("ML wins in high-vol regimes, GARCH holds in calm regimes") as a stronger senior signal than a fragile headline win.

## Key Findings

### Recommended Stack

The binding ecosystem constraint is MLflow 3.13's `pandas<3, numpy<3, scikit-learn<2` pins combined with SHAP 0.52's `Python>=3.12, numpy>=2` floor. Everything else slots in cleanly around Python 3.12 + pandas 2.3 + numpy 2.x. Manage with uv; develop in WSL2; run all stateful services (MLflow, Prefect, Postgres) in docker-compose from day one. Full details: `STACK.md`.

**Core technologies:**
- Python 3.12 + pandas >=2.3,<3 + numpy >=2,<3 — the verified compatibility envelope (pandas 3.0 is blocked by MLflow's pin)
- ccxt 4.5.x + yfinance >=1.4 (`nospam` extra) — crypto/equity OHLCV; both treated as flaky dependencies behind a cache-first, DVC-versioned store
- arch 8.0 (GARCH/EWMA) + LightGBM 4.6 + SHAP 0.52 — classical baselines, ML challenger, explainability (TreeExplainer is exact for LightGBM)
- MLflow 3.x — tracking + registry via **aliases** (`@champion`/`@challenger`), not deprecated stages
- Prefect 3.x — orchestration (lighter than Airflow; server + worker containerized to avoid Windows quirks)
- Evidently 0.7.x — drift detection; the 0.7 API is a complete rewrite, so use only official docs (all older tutorials are broken)
- Pandera 0.31 (`import pandera.pandas as pa`) — in-pipeline validation gates
- DVC 3.x (local remote) + FastAPI 0.136 + Streamlit 1.58 + PostgreSQL in compose

**Key avoid list:** pandas 3.0, MLflow stages API, Evidently <0.7 examples, SQLite on Windows bind mounts, Python 3.11/3.14, DVC Google Drive remote, Feast (v1), bare `TimeSeriesSplit` as the eval story.

### Expected Features

The "users" are hiring managers and quant-literate reviewers. Table stakes = lifecycle completeness AND methodological correctness; missing either makes the project read as a tutorial clone or as naive. Full details: `FEATURES.md`.

**Must have (table stakes):**
- Multi-source OHLCV ingestion (ccxt + yfinance) with Pandera validation gates
- Explicit, documented realized-vol target definition (the single most-scrutinized choice)
- EWMA + GARCH(1,1) baselines, walk-forward/expanding-window evaluation ONLY, QLIKE alongside RMSE/MAE
- LightGBM with a versioned feature pipeline shared between training and serving
- MLflow tracking + registry; FastAPI + Docker serving; Evidently drift detection; Prefect DAG; GitHub Actions CI; honest README with architecture diagram

**Should have (differentiators):**
- **HAR-RV as a third baseline** — ~30 lines, the academic-standard RV benchmark, outsized quant credibility (recommended addition to PROJECT.md's baseline set)
- Closed feedback loop (forecast log joined with auto-arriving realized labels) — the project's structural advantage
- Champion/challenger promotion gated on rolling QLIKE; drift-*triggered* (not just scheduled) retraining
- Range-based estimators (Parkinson, Garman-Klass) as near-free features; Diebold-Mariano significance tests
- MODEL_CARD.md with honest, regime-segmented results; Streamlit observability dashboard; SHAP; DVC; rollback drill

**Defer (v2+):** crypto intraday true-RV target, multi-horizon forecasts, cloud deploy, DL challenger, Feast.

**Anti-features (explicitly not building):** trading strategy/backtest PnL, price-direction prediction, deep learning in v1, Kafka/streaming, Kubernetes, random K-fold CV anywhere, 50+ ticker universe.

### Architecture Approach

A batch pipeline loop feeding a thin online serving layer, with a monitoring/feedback plane closing back into retraining. The non-negotiable structural decision: `src/volforecast/` is an installable package so that training, serving, pipelines, and the label materializer all import the SAME feature code (training/serving skew is the #1 silent killer). Prefect flows live in `pipelines/` as logic-free one-line wrappers — the whole pipeline must run as plain Python. Data moves through a trust gradient (`raw/ → validated/ → features/`) gated by Pandera, all DVC-versioned. The prediction log (append-only parquet: ts, asset, horizon, forecast, model_version, latency) is the contract between serving and monitoring; monitoring never calls the API. Full details: `ARCHITECTURE.md`.

**Major components:**
1. Data plane (`ingest/`, `validate/`, `features/`) — dual-calendar adapters (crypto 24/7 UTC vs exchange sessions) behind one canonical OHLCV schema; single feature codepath with labels materialized alongside
2. Model plane (`models/`) — EWMA/GARCH/LightGBM behind one `fit/predict` protocol; one shared walk-forward harness producing RMSE/MAE/QLIKE on identical folds; MLflow registry with alias-based promotion
3. Serving plane (`serving/`) — FastAPI loading `models:/vol_forecaster@champion` at startup (+ reload hook), appending every forecast to the prediction log
4. Monitoring/feedback plane (`monitoring/`) — label materializer, Evidently drift reports, champion/challenger gate on rolling QLIKE; drift triggers retraining, NEVER promotion
5. Orchestration (`pipelines/`, Prefect 3) + read-only Streamlit dashboard + DVC + GitHub Actions

### Critical Pitfalls

Top 5 of 14 documented (full list with recovery strategies: `PITFALLS.md`):

1. **Lookahead leakage via overlapping RV label windows** — multi-day labels are intervals, not points; use purged walk-forward with embargo ≥ horizon H; unit-test timestamp boundaries; run the shifted-label leak smoke test. Build the harness before any ML model.
2. **GARCH misfit (scale/convergence)** — fit on 100× percent returns; assert convergence and alpha+beta<1 per refit; centralize variance-unit conversion. A strawman GARCH destroys the project's core credibility claim.
3. **QLIKE implemented wrong** — one canonical `qlike(rv_variance, forecast_variance)` function (Patton 2011 form) used by ALL evaluation and the promotion gate; test `qlike(x,x)==0`; clip forecast floor.
4. **Drift detection crying wolf on regime shifts** — separate data-quality / distribution-drift / performance-drift monitors with different consequences; retrain trigger = rolling QLIKE degradation *relative to GARCH* + cooldown, not feature drift.
5. **Calendar/unit chaos** — per-asset calendars and annualization factors as config attributes; as-of joins for cross-asset features; one canonical internal unit (daily variance of decimal log returns) with conversions only at the display layer.

Cross-cutting: Windows 11 friction (develop in WSL2 filesystem, `.gitattributes` with `eol=lf`, Postgres not bind-mounted SQLite) must be settled in the foundation phase — retrofit is expensive.

## Implications for Roadmap

Research converges on a dependency-driven build order (ARCHITECTURE.md's build order, FEATURES.md's dependency graph, and PITFALLS.md's phase mapping all agree). Suggested structure:

### Phase 1: Foundation & Infrastructure
**Rationale:** CI, packaging, and Windows/Docker decisions are cheap on day one and expensive to retrofit; every later phase imports the package and runs in compose.
**Delivers:** Repo scaffold, installable `src/volforecast` package (uv + pyproject), `.gitattributes` (eol=lf), docker-compose skeleton (MLflow, Prefect, Postgres), GitHub Actions (ruff + pytest + docker build).
**Addresses:** CI/CD table stake, README skeleton.
**Avoids:** Pitfall 12 (Windows/Docker/CRLF friction), SQLite-on-bind-mount corruption.

### Phase 2: Ingestion & Validation
**Rationale:** Everything downstream consumes validated data; the dual-calendar decision made here ripples through the entire system.
**Delivers:** ccxt + yfinance adapters behind one canonical OHLCV schema, cache-first DVC-versioned store, Pandera gates (calendar-aware gaps, unclosed-candle rejection, historical-revision detection), recorded fixtures for CI.
**Addresses:** Multi-source ingestion + validation table stakes.
**Avoids:** Pitfalls 4 (calendar mismatch), 5 (yfinance fragility), 6 (ccxt incomplete candles/geo-block).

### Phase 3: Feature Pipeline & Target Definition
**Rationale:** Features feed baselines, ML, and serving alike; leakage discipline (all windows end strictly at `asof`) and the canonical units module are established here.
**Delivers:** Documented RV target, single feature codepath (lagged RV, returns, EWMA, range estimators, calendar features), label materialization, units module with property tests.
**Avoids:** Pitfalls 11 (unit confusion), 4 (cross-asset as-of joins), leakage at the feature level.

### Phase 4: Baselines & Evaluation Harness
**Rationale:** The bar must exist before anything tries to clear it — harness before ML prevents motivated reasoning. This phase is the credibility centerpiece; end of this phase = shippable project per PROJECT.md.
**Delivers:** EWMA + GARCH(1,1) + HAR-RV behind a common protocol; purged walk-forward harness with embargo; canonical QLIKE/RMSE/MAE metrics module (tested); final holdout period reserved; honest results table.
**Avoids:** Pitfalls 1 (leakage), 2 (GARCH misfit), 3 (QLIKE bugs), 14 (holdout reserved before tuning exists).

### Phase 5: ML Model & Experiment Tracking
**Rationale:** First registered model versions; evaluated on the identical folds as baselines.
**Delivers:** Regularized LightGBM with tuning inside walk-forward folds only; MLflow tracking with lineage tags (git SHA, DVC hash, train window) on every run; registry with `@champion`/`@challenger` aliases; SHAP global importance.
**Uses:** LightGBM, MLflow 3 aliases, SHAP.
**Avoids:** Pitfalls 8 (stages deprecation), 14 (eval overfitting).

### Phase 6: Serving
**Rationale:** The prediction log schema defined here is the contract monitoring builds on; serving must exist before monitoring has anything to consume.
**Delivers:** FastAPI (`/forecast`, `/health`, `/model-info`, `/reload`) loading the champion alias at startup, importing the shared feature package; Dockerized; append-only prediction log; container smoke test comparing API forecast to offline forecast.
**Avoids:** Training/serving skew (anti-pattern #1), per-request model resolution.

### Phase 7: Monitoring & Feedback Loop
**Rationale:** The feedback loop becomes real once predictions are logged; the monitor taxonomy must be decided before Evidently is wired.
**Delivers:** Label materializer joining forecasts with realized vol; Evidently 0.7 behind an adapter seam; three-class monitor taxonomy (quality → page, drift → log, performance/rolling-QLIKE-vs-GARCH → retrain trigger); champion/challenger gate scoring both models on an identical frozen window with no-promote default.
**Avoids:** Pitfalls 7 (drift false positives), 9 (apples-to-oranges promotion), 13 (Evidently legacy API).

### Phase 8: Orchestration & Automated Retraining
**Rationale:** Orchestrate last — Prefect wraps already-working functions; orchestrating first means debugging logic through a scheduler (the most common MLOps build mistake).
**Delivers:** Daily scoring flow + retrain flow as thin Prefect wrappers; cron schedules + drift-triggered retrain with cooldown; ingestion failure = skip-with-alert; scripted and rehearsed alias-flip rollback.
**Avoids:** Pitfalls 10 (silent degradation), drift-triggered auto-promotion (anti-pattern #4).

### Phase 9: Dashboard & Polish
**Rationale:** Pure consumers of stores that now exist; SHAP/model card need final, stable numbers.
**Delivers:** Read-only Streamlit dashboard (forecast-vs-realized with GARCH overlay, rolling QLIKE, drift status, model version, latency); MODEL_CARD.md with regime-segmented results; Diebold-Mariano tests; README architecture diagram; optional cloud deploy.

### Phase Ordering Rationale

- **Harness before ML (4 before 5):** evaluating LightGBM without the GARCH/HAR-RV bar in place invites motivated reasoning — all three research files independently flag this.
- **Serving before monitoring (6 before 7):** monitoring consumes the prediction log, which doesn't exist until serving writes it (a backfill script can simulate history to test monitoring sooner).
- **Orchestration near-last (8):** thin wrappers over proven functions; the inverse ordering is the most common failure mode in MLOps projects.
- **Infrastructure decisions in Phase 1:** WSL2/line-endings/compose/Postgres choices are architectural — every pitfall in this category is cheap to prevent and expensive to recover from.
- Each phase boundary maps directly onto a data/artifact contract (validated parquet, feature parquet, registry alias, prediction log), so phases are independently verifiable.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (Baselines & Eval Harness):** purged walk-forward with embargo for multi-asset panels has no off-the-shelf implementation; GARCH refit cadence and QLIKE form need careful spec — highest methodological risk in the project.
- **Phase 7 (Monitoring & Feedback):** Evidently 0.7 API is newly rewritten with sparse correct examples; the monitor taxonomy and drift-threshold tuning on historical regime shifts need design work.
- **Phase 8 (Orchestration):** Prefect 3 drift-trigger wiring (event-driven deployments) is the least-documented integration in the stack.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** uv/ruff/pytest/compose/Actions are thoroughly documented; STACK.md already covers the Windows specifics.
- **Phase 5 (LightGBM + MLflow):** well-trodden; the alias pattern is documented in STACK/ARCHITECTURE research.
- **Phase 6 (FastAPI serving):** standard pattern; the key constraints (shared feature package, startup model load) are already specified.
- **Phase 9 (Dashboard):** Streamlit read-only consumer; trivial once stores exist.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified against PyPI metadata 2026-06-10; key pins (MLflow `pandas<3`, SHAP `>=3.12`) verified from `requires_dist` |
| Features | HIGH | MLOps lifecycle features verified against ecosystem projects and vendor docs; vol methodology verified against academic literature; complexity estimates MEDIUM (judgment-based) |
| Architecture | HIGH | Canonical MLOps shape; specifics verified against current MLflow/Prefect/Evidently official docs |
| Pitfalls | HIGH | Quant pitfalls verified against literature (Patton 2011, arch docs); tooling pitfalls against GitHub issues and official migration guides; some Windows/ccxt items MEDIUM |

**Overall confidence:** HIGH

### Gaps to Address

- **yfinance behavior at v1.x:** "no breaking changes" claim is MEDIUM confidence; verify config deprecation warnings and `auto_adjust` semantics against the pinned version on first run in Phase 2.
- **ccxt incomplete-candle behavior:** verify against the pinned ccxt version's docs in Phase 2 (currently training-data knowledge, MEDIUM).
- **Binance geo-blocking from CI/cloud:** GitHub Actions runners are US-based — fixtures mandatory in CI; re-verify exchange reachability when the cloud-deploy phase is planned (fallback: Kraken/Coinbase via ccxt config swap).
- **Prefect drift-trigger mechanics:** event-driven retrain wiring in Prefect 3 needs validation during Phase 8 planning (LOW-MEDIUM documented).
- **"ML never beats GARCH" outcome:** not a research gap but a planning contingency — Phase 4/9 plans should bake in the regime-segmented honest-reporting framing so the project succeeds either way.
- **PROJECT.md wording updates needed:** "MLflow stages" → aliases; "pandas/Polars" → pandas-only v1; HAR-RV added to the baseline set.

## Sources

### Primary (HIGH confidence)
- PyPI JSON metadata (mlflow, prefect, evidently, pandera, lightgbm, arch, yfinance, ccxt, shap, dvc, pandas, fastapi, streamlit, numpy) — versions + dependency constraints
- MLflow official docs + RFC #10336 — alias-based registry, stages deprecation, MLflow 3 breaking changes
- pandas v3.0.0 whatsnew — breaking changes confirming the <3 pin
- docs.evidentlyai.com — 0.7 API + migration guide
- pandera.readthedocs.io — `pandera.pandas` namespace change
- arch library docs — data scaling, convergence behavior
- Patton (2011), ScienceDirect — QLIKE robustness; Journal of Financial Econometrics — ML vol forecasting standards
- Prefect 3 official docs — deployments, Docker workers

### Secondary (MEDIUM confidence)
- yfinance GitHub issues #2422/#2411/#2567 + release notes — rate-limit waves, 1.0 changes, `auto_adjust` default flip
- Databricks MLOps workflow docs — champion/challenger via aliases
- Finance Research Letters — range estimators improving RV forecasts
- AWS time-series MLOps reference architecture; representative GitHub MLOps/vol-forecasting repos — competitor gap analysis

### Tertiary (LOW confidence)
- Prefect-on-native-Windows quirks (community reports) — sidestepped by containerized workers regardless
- GitHub topic scans for competitor analysis — coverage not exhaustive

---
*Research completed: 2026-06-10*
*Ready for roadmap: yes*
