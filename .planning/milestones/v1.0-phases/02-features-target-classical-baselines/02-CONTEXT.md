# Phase 2: Features, Target & Classical Baselines - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning
**Mode:** Autonomous smart discuss (recommended answers auto-accepted per user directive)

<domain>
## Phase Boundary

A leak-free purged walk-forward harness scores honest classical baselines (EWMA, GARCH(1,1), HAR-RV) on a canonically defined realized-vol target. Delivers: shared target/units module, single versioned feature pipeline (imported identically by training and future serving), three classical baselines producing walk-forward forecasts per asset, canonical unit-tested metrics (QLIKE/RMSE/MAE), and a committed per-asset evaluation report — the bar Phase 3's ML must clear. No ML model, no MLflow logging, no serving — those are Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Target & Units
- Canonical units: daily VARIANCE of decimal log returns; vol is sqrt at the edges; one shared module (`src/volforecast/features/target.py` or equivalent) documents and owns this
- Label: next-day squared log return as the variance proxy (QLIKE robust to noisy proxies, Patton 2011); 5-day forward realized variance reported as a secondary stability check
- No annualization anywhere inside the pipeline — display-only at report edges

### Walk-Forward Harness
- Expanding window, minimum 252 training observations, step 21 observations
- Purging of overlapping label windows at split boundaries; embargo gap >= label horizon; a unit test FAILS if any split is non-temporal or embargo < horizon
- Harness is a reusable library (Phase 4 promotion gate reuses it); horizon-parameterized
- GARCH(1,1): fit with `arch` on 100x scaled decimal log returns; assert convergence flag and alpha+beta<1 on every refit; refit every 21 observations (monthly), daily forecasts from the most recent fit
- HAR-RV: OLS on daily/weekly(5)/monthly(22) lagged RV components
- EWMA: RiskMetrics lambda=0.94 (documented), variance recursion

### Features & Cross-Asset Staleness
- Every feature window ends strictly at as-of date t; label window starts at t+1 — no overlap
- Feature set: realized vol over 5/10/22/66 lookbacks, log returns, squared returns, lagged vol, EWMA vol, GARCH(1,1) conditional vol-as-feature, Parkinson, Garman-Klass, vol-of-vol, rolling skew/kurtosis, calendar features (day-of-week, month; session/overnight flags for equities)
- Cross-asset features (e.g., BTC RV as ETH/equity input): backward as-of join with max staleness 3 calendar days; beyond that the feature is NaN (documented rule)
- GARCH-as-feature uses the same monthly-refit filtered conditional vol — never fitted on data past as-of
- Single codepath: training and (future) serving import the identical feature module (FEAT-07)

### Evaluation Report
- Committed artifacts: `reports/baseline_eval.md` + per-asset metrics CSV — "the published bar" for Phase 3
- Canonical metrics module: QLIKE in Patton variance form with unit test `qlike(x,x)==0`, plus RMSE and MAE; shared by all future evaluation and the Phase 4 promotion gate
- MLflow logging deferred to Phase 3

### Claude's Discretion
- Module layout inside src/volforecast/features/ and src/volforecast/models/, exact HAR-RV estimation details, parquet schemas for feature matrices, report formatting

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/volforecast/ingest/base.py` — OHLCV contract (`OHLCV_COLUMNS`, `merge_bars`), tz-aware UTC microsecond timestamps
- `src/volforecast/validate/` — `validate_asset` dispatcher; processed data in `data/processed/{crypto,equity}/*.parquet` is the trusted input
- `src/volforecast/config.py` — `project_root()` resolution (VOLFORECAST_ROOT env var), `load_assets` for the 5-asset universe
- Test conventions: fixtures in tests/fixtures/ (parquet), offline-only, `VOLFORECAST_NO_LIVE_API` guard

### Established Patterns
- `import pandera.pandas as pa`; pandas 2.3 pinned; ruff + pytest; uv-managed Python 3.12.13 (uv NOT on PATH — needs `$env:LOCALAPPDATA\Programs\Python\Python313\Scripts`)
- TDD style used in Phase 1 (RED/GREEN commits) for clear-I/O modules

### Integration Points
- Input: `data/processed/` parquet (BTC-USD, ETH-USD, SPY, AAPL, MSFT)
- Output consumed by Phase 3: feature matrix builder + walk-forward harness + metrics module + published baseline bar
- arch (8.x) and statsmodels available in the pinned dependency matrix (verify arch present in pyproject; add if missing)

</code_context>

<specifics>
## Specific Ideas

- "Beating, or honestly not beating, GARCH is the eval-rigour centrepiece" — the report must be honest per asset, even where baselines are strong
- Walk-forward only; a failing unit test on non-temporal splits is itself a portfolio talking point

</specifics>

<deferred>
## Deferred Ideas

- LightGBM + MLflow tracking (Phase 3), promotion gate use of the harness (Phase 4)
- Intraday true realized vol, multi-horizon forecasts (v2)

</deferred>
