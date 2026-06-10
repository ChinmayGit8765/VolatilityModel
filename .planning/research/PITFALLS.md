# Pitfalls Research

**Domain:** Financial volatility forecasting + MLOps platform (crypto + equities, GARCH/LightGBM, MLflow/Prefect/Evidently stack)
**Researched:** 2026-06-10
**Confidence:** HIGH (core quant-finance pitfalls verified against literature and official docs; tooling pitfalls verified against GitHub issues and official migration guides)

## Critical Pitfalls

### Pitfall 1: Lookahead leakage via overlapping realized-vol windows (not just random splits)

**What goes wrong:**
Everyone knows random train/test splits leak on time series. The subtler killer: the *target itself* spans a future window. If the label at day `t` is realized vol over days `t+1..t+22`, then a training sample at day `t` and a "test" sample at day `t+10` share 12 days of overlapping future returns. A naive walk-forward split that only separates *feature* timestamps still leaks the target. Result: ML model "beats GARCH by 30%" in backtest, falls apart live.

**Why it happens:**
Standard `TimeSeriesSplit` separates rows by index, not by the time span the *label* covers. Multi-day RV targets are intervals, not points. Also tempting: using `rolling().std()` centered windows, or computing features and labels in one DataFrame where a rolling op accidentally includes day `t`'s return in both feature and label.

**How to avoid:**
- Define label timestamps explicitly: label for day `t` = RV computed from returns strictly in `(t, t+H]`. Features use data in `(-inf, t]` only.
- Use **purged walk-forward CV with an embargo gap** of at least the label horizon H between train end and test start (de Prado's purging/embargo concept).
- Write a unit test that asserts: `max(feature_data_timestamp) <= t < min(label_return_timestamp)` for every sample.
- Smoke test: train the model with a feature equal to the label shifted by +1 day. If test metrics are suspiciously perfect, your harness leaks.

**Warning signs:**
ML beats GARCH by implausible margins (>15–20% QLIKE improvement on daily horizon is suspicious — literature says HAR/GARCH are hard to beat). Test metrics improve as forecast horizon grows. Performance collapses on truly out-of-sample (post-development) data.

**Phase to address:**
Evaluation-harness phase (build the purged walk-forward harness *before* any ML model phase; baselines and ML must share the identical harness).

---

### Pitfall 2: GARCH fitted on raw log returns — convergence failures and garbage parameters

**What goes wrong:**
`arch` fitted on raw daily log returns (scale ~0.01) emits `DataScaleWarning: y is poorly scaled` and the optimizer can converge to a boundary (alpha+beta ≈ 1 or alpha ≈ 0), producing flat or explosive conditional vol. Forecasts are then silently wrong, and your "GARCH baseline" is a strawman — which destroys the project's core credibility claim (honest benchmarking vs the *correct* classical model).

**Why it happens:**
The arch optimizer works best when the scale of y is between 1 and 1000; raw log returns are ~0.005–0.05. Developers also forget that `forecast()` returns *variance* in the rescaled units, then mix scales when comparing to RV computed from raw returns.

**How to avoid:**
- Fit on **percent returns** (`100 * log_returns`) explicitly rather than relying on `rescale=True` (auto-rescale changes parameter units and is easy to mis-invert).
- After every refit in the walk-forward loop, assert: convergence flag is success, `alpha + beta < 1` (stationarity), and conditional vol is within sane bounds (e.g., annualized 5%–300%).
- Convert forecast variance back to the return scale once, in one tested function (`variance / 100**2`), never inline.
- In walk-forward refits, wrap `fit()` in a try/except with fallback to previous window's parameters; log every failed fit as a metric.

**Warning signs:**
`DataScaleWarning` in logs. GARCH forecast looks like a flat line. `alpha + beta > 0.999`. GARCH QLIKE wildly worse than EWMA (EWMA should be close to GARCH(1,1), not 10x better).

**Phase to address:**
Classical-baselines phase. Add convergence assertions as pytest tests, not just notebook checks.

---

### Pitfall 3: QLIKE implemented wrong (variance vs vol, ratio direction, zero handling)

**What goes wrong:**
QLIKE has multiple forms in the literature and three classic implementation bugs: (1) feeding standard deviations where variances are required, (2) inverting the ratio (forecast/actual instead of actual/forecast), (3) blowing up when forecast variance ≈ 0. A wrong QLIKE silently reorders model rankings — the entire champion/challenger promotion logic then promotes worse models.

**Why it happens:**
QLIKE = mean( RV²/h − ln(RV²/h) − 1 ) where RV² is realized *variance* and h is forecast *variance* (Patton 2011 robust form). People pass annualized vol into one argument and daily variance into the other, or copy a formula written in terms of log(h) + RV²/h without realizing it's the same thing up to a constant — then mix forms across models.

**How to avoid:**
- One canonical `qlike(rv_variance, forecast_variance)` function in a metrics module with docstring citing Patton (2011), used by baseline eval, ML eval, and the champion/challenger gate. No reimplementation anywhere.
- Unit tests: `qlike(x, x) == 0` for the (σ²/h − ln −1) form; QLIKE penalizes under-forecasts more than over-forecasts of equal ratio; raises/clips on non-positive forecasts.
- Clip forecast variance at a small floor (e.g., 1e-8 in daily-variance units) and log when clipping fires.
- Note in MODEL_CARD: QLIKE is robust to noisy vol proxies (Patton 2011) — this is *why* it's the headline metric over MAE, which can rank inferior forecasts higher under noisy proxies.

**Warning signs:**
QLIKE values negative for the (σ²/h − ln(σ²/h) − 1) form (impossible — it's ≥ 0). Model rankings flip between QLIKE and RMSE-on-variance in ways MAE/RMSE-on-vol don't explain. NaN/inf in metric output.

**Phase to address:**
Evaluation-harness phase (metrics module written and tested before any model is evaluated).

---

### Pitfall 4: Crypto 24/7 vs equity session calendars treated as one timeline

**What goes wrong:**
BTC has 365 daily bars/year; SPY has ~252. Naive joins on calendar date create NaNs or forward-fills that fabricate stale equity data on weekends; cross-asset features (e.g., "BTC vol as SPY feature") silently use Friday's SPY value on Sunday with no flag; annualization with a single √252 overstates crypto vol comparisons or √365 understates equity. Also: "daily" bar boundaries differ — Binance daily candles close at 00:00 UTC, equity bars at 16:00 ET — so "same day" rows are actually 19–21 hours apart, which leaks intraday information into cross-asset features if joined naively.

**Why it happens:**
pandas makes `df_crypto.join(df_equity)` trivially easy and silently fills/aligns. Developers test on crypto-only or equity-only slices where the bug can't appear.

**How to avoid:**
- Keep per-asset-class calendars explicit: crypto pipeline on 365-day UTC calendar, equities on the exchange calendar (use `pandas_market_calendars`/`exchange_calendars`, which handle holidays and half-days).
- Annualization factor is an *asset attribute* (365 for crypto, 252 for equities) stored in config, never a literal in code. Better: report and compare daily vol primarily; annualize only at the display layer.
- For cross-asset features, define alignment as "last observation strictly before the target asset's bar close" (an as-of/merge_asof join with explicit timestamps in UTC), and add a staleness feature (hours since the source bar closed) instead of pretending alignment is exact.
- Pandera schemas per asset class: crypto expects no gaps >1 day; equities expects gaps only on exchange holidays/weekends (validate against the official calendar, don't just count NaNs).

**Warning signs:**
SPY rows on Saturdays. Crypto vol "spikes" every Monday (weekend returns lumped). Cross-asset feature importance suspiciously dominant (often a leak via timezone misalignment). Annualized crypto and equity vols compared in the dashboard with one factor.

**Phase to address:**
Data-ingestion + validation phase (calendars), feature-pipeline phase (as-of joins, per-asset annualization).

---

### Pitfall 5: yfinance treated as a reliable production API

**What goes wrong:**
yfinance is a scraper of Yahoo's private endpoints. Documented, recurring failure modes: `YFRateLimitError: Too Many Requests` waves that hit even low-volume users and persist across IPs (GitHub issues #2422, #2411, #2567 — recurring through 2025); silent schema/endpoint breakage requiring emergency version bumps; and the `auto_adjust` default changing to `True` (≈ v0.2.51, early 2025), which changed `Close` from raw to split/dividend-adjusted — anyone computing returns off `Close` got silently different numbers. A scheduled Prefect ingestion flow that assumes yfinance always works will fail intermittently and, worse, can trigger spurious drift alarms when partial data slips through.

**Why it happens:**
It looks like an API but has no SLA, no auth, no versioned contract. Yahoo changes things server-side without notice.

**How to avoid:**
- Ingestion layer with retries + exponential backoff + jitter, a hard per-run request budget, and batched `yf.download(tickers_list)` rather than per-ticker loops.
- **Cache-first architecture**: persist raw pulls (DVC-versioned parquet); pipeline reads from the local store, fetch only appends new bars. A fetch failure means "no new data today," never "pipeline broken."
- Pin yfinance version; set `auto_adjust` explicitly (recommend `auto_adjust=True` and document that returns are adjusted — or pull both and store `Close` + `Adj Close`); add a Pandera check that today's pull of historical bars matches yesterday's stored values (detects silent backfill/adjustment changes).
- Design the DAG so ingestion failure is a *skip with alert*, not a crash that blocks serving (serve from last good features).

**Warning signs:**
Intermittent CI failures only in the ingestion test job. Historical bars changing values between pulls (corporate actions/adjustment drift). Empty DataFrames returned without exception (older yfinance versions fail silently).

**Phase to address:**
Data-ingestion phase (cache-first design is architectural — retrofit is expensive). CI phase: never call live yfinance in unit tests; use recorded fixtures.

---

### Pitfall 6: ccxt/Binance gotchas — geo-blocking, rate limits, and incomplete candles

**What goes wrong:**
Three distinct traps: (1) Binance blocks US IPs (HTTP 451) — a pipeline that works locally dies when deployed to a US cloud region or GitHub Actions runner; (2) hammering `fetch_ohlcv` in a loop without `enableRateLimit` earns IP bans; (3) the most recent candle returned is the *currently forming* (incomplete) candle — ingesting it as final gives you a partial day's return, corrupting the freshest label and the drift monitor simultaneously.

**Why it happens:**
ccxt abstracts exchanges but not their policies. The incomplete-candle behavior is documented but routinely missed because it only corrupts the last row.

**How to avoid:**
- `exchange = ccxt.binance({'enableRateLimit': True})` always; paginate with `since` + limit, respecting the 1000-candle cap per call.
- Drop the last candle unless `candle_close_time < now_utc` — make "bar is closed" an explicit Pandera check (timestamp of newest bar must be ≤ today 00:00 UTC).
- Decide deploy region early; fall back to another exchange via ccxt's uniform interface (Kraken, Coinbase) or Binance Vision public data dumps for history. GitHub Actions runners are US-based — CI must use fixtures, never live Binance calls.

**Warning signs:**
HTTP 451/403 in cloud logs but not locally. Today's BTC "realized vol" always lower than yesterday's (partial candle). Bans appearing after backfill runs.

**Phase to address:**
Data-ingestion phase. Cloud-deploy phase must re-verify exchange reachability from the target region.

---

### Pitfall 7: Drift detection that cries wolf on every volatility regime shift

**What goes wrong:**
Volatility *clusters and regime-shifts by nature* — that is the signal, not drift. Evidently's default data-drift presets (KS/PSI per column) will fire on every vol spike: a VIX-style event shifts the distribution of every vol feature simultaneously. If drift alerts auto-trigger retraining, you get retrain storms exactly when markets are stressed — retraining on a brief panic regime, then degrading when it mean-reverts. After a few false alarms, alerts get ignored (alarm fatigue), defeating the monitoring story.

**Why it happens:**
Default drift detectors assume stationary feature distributions. Financial features are heteroskedastic by construction. Tutorials demo drift on i.i.d. tabular data and the defaults transfer badly.

**How to avoid:**
- Separate three monitor classes with different actions: (a) **data-quality drift** (schema, missingness, stale values) → page immediately; (b) **feature/prediction drift** (distribution shift) → dashboard + log, *never* auto-retrain alone; (c) **performance drift** (rolling QLIKE of live forecasts vs realized labels, vs the GARCH baseline's rolling QLIKE) → this is the retrain trigger. Because labels arrive automatically (tomorrow's RV), you can monitor *actual error*, which is far stronger than proxy drift — make this the centerpiece.
- Gate retrains: trigger only if model's rolling QLIKE degrades *relative to* GARCH's rolling QLIKE over a minimum window (e.g., 20 obs) AND a cooldown since last retrain has passed. Comparing against GARCH controls for "the market got harder for everyone."
- Tune drift tests on historical regime shifts (e.g., known vol spikes in your backtest period) and document the chosen thresholds — this is itself a great portfolio talking point.

**Warning signs:**
Drift alerts >1/week in calm markets. Retrains clustering during vol spikes. Post-retrain QLIKE worse than pre-retrain.

**Phase to address:**
Monitoring phase (monitor taxonomy), retraining phase (gating logic). Decide the taxonomy *before* wiring Evidently, or you'll wire the wrong triggers.

---

### Pitfall 8: MLflow stages-based registry workflow — deprecated API as the backbone

**What goes wrong:**
PROJECT.md specifies "staging→prod promotion" and "MLflow model stages for rollback." Model registry **stages are deprecated since MLflow 2.9** (`transition_model_version_stage` will be removed in a future major release). Building the promotion/rollback machinery on stages means churn mid-project and — for a portfolio meant to demonstrate current MLOps practice — signals stale knowledge to reviewers.

**Why it happens:**
Most MLflow tutorials and blog posts (and LLM training data) predate the deprecation and still show `transition_model_version_stage("Production")`.

**How to avoid:**
- Use **model version aliases**: `@champion` and `@challenger` aliases on the registered model; serving loads `models:/volforecast@champion`; promotion = reassigning the alias; rollback = pointing `@champion` back at the prior version. This maps 1:1 onto the champion/challenger requirement and is the documented migration path.
- Store evaluation evidence (rolling QLIKE vs baseline) as model-version tags so promotion decisions are auditable in the registry UI.
- Serving must not hot-resolve the alias on every request: resolve at startup + on an explicit reload endpoint/interval, and log the resolved version with every prediction (needed for the feedback loop anyway).

**Warning signs:**
`FutureWarning`/deprecation warnings from MLflow client calls. Tutorials you're following reference MLflow ≤2.8 UI screenshots.

**Phase to address:**
Experiment-tracking/registry phase. Write the promotion helper around aliases from day one.

---

### Pitfall 9: Champion/challenger evaluation that isn't apples-to-apples

**What goes wrong:**
Challenger evaluated on different dates, different label version, or different feature snapshot than champion — promotion gate passes a model that's worse. Classic variants: challenger trained on data through today but evaluated on a window the champion never saw; comparing challenger's *backtest* QLIKE to champion's *live* QLIKE (live is always harder); evaluating both on a window so short the difference is noise.

**Why it happens:**
The comparison logic is glue code written last, under time pressure, without the rigor applied to the main eval harness.

**How to avoid:**
- Promotion gate = both models scored on the **identical** frozen evaluation window (same dates, same labels, same feature parquet hash — DVC makes this checkable).
- Require a minimum evaluation window (e.g., ≥60 daily obs) and a margin (challenger QLIKE must beat champion by >X% or via a Diebold-Mariano test at p<0.10 — DM test is a cheap, high-credibility addition).
- GARCH is a permanent silent challenger: if ML champion stops beating GARCH on rolling QLIKE, the dashboard says so. That honesty is a stated project feature.

**Warning signs:**
Frequent promotions with tiny metric deltas. Promoted model immediately underperforms live. Promotion logs lack the eval window dates/data hash.

**Phase to address:**
Retraining/champion-challenger phase, reusing the evaluation-harness phase's code verbatim.

---

### Pitfall 10: Retraining loop that silently degrades (no gate, no lineage, no rollback drill)

**What goes wrong:**
Scheduled retrain runs, produces a model, auto-promotes it. Months later you discover: a data-quality regression poisoned training features (e.g., a yfinance adjustment change rewrote history), each retrain "passed" because it was compared against nothing or against a stale benchmark, and you can't reproduce any deployed model because training data wasn't versioned per run. The platform is now *worse* than a frozen model and nobody noticed.

**Why it happens:**
Automation removes the human eyeball. Without an explicit "do nothing" path, every retrain becomes a deploy. Lineage feels optional until the first incident.

**How to avoid:**
- Retrain ≠ deploy. The Prefect DAG's terminal step is "register challenger + run promotion gate"; promotion is a separate gated step where *no-promote is the default outcome*.
- Every training run logs: git SHA, DVC data hash, feature config hash, train window dates → as MLflow run tags. Reproducibility = re-running from those refs.
- Pandera validation must gate the *training* data path, not just ingestion (a feature-pipeline bug can corrupt features after valid ingestion). Add a "golden sample" test: a frozen input slice with known expected feature values, run in CI.
- Actually rehearse rollback once (flip `@champion` to previous version, verify serving picks it up) and script it — untested rollback is not rollback.

**Warning signs:**
Promotion rate ≈ 100%. No MLflow run can answer "what data trained the current prod model?" Monitoring dashboard shows slow QLIKE creep across model versions.

**Phase to address:**
Orchestration/retraining phase; lineage tags belong in the first MLflow phase so they exist before automation.

---

### Pitfall 11: Annualization and vol-unit confusion smeared across the codebase

**What goes wrong:**
Variance vs vol (σ² vs σ), daily vs annualized, percent vs decimal, log vs simple returns — with 4 binary unit choices there are 16 representations, and any module pair disagreeing produces errors that are wrong by √252 (~15.9x), 100x, or 10000x — or worse, by a *subtle* factor like √(365/252) ≈ 1.2 that looks plausible on a chart. GARCH forecasts variance of percent returns; RV is computed from decimal log returns; QLIKE needs variances; the dashboard wants annualized vol in percent; LightGBM trains on whatever the feature pipeline emits. One un-tracked conversion and the model card metrics are fiction.

**Why it happens:**
Each library has its own native convention and conversions happen inline at call sites.

**How to avoid:**
- Pick **one canonical internal representation** — recommend *daily variance of decimal log returns* — and document it in the repo. All models forecast it; all metrics consume it; conversions to display units (annualized vol %) happen only in the dashboard/API serialization layer via named functions (`daily_var_to_ann_vol(var, periods_per_year)`).
- Consider training LightGBM on `log(RV)` for better-behaved residuals — but then the inverse transform (and its bias) is one more conversion to centralize and test.
- Property tests: round-trip conversions are identity; annualized SPY vol over a known calm period lands in 5–25%; BTC in 20–100%.

**Warning signs:**
Dashboard vol numbers that don't pass the sniff test (SPY at 180% or 0.8%). RMSE values differing by orders of magnitude between baseline and ML eval scripts. `**2` and `np.sqrt` scattered through call sites.

**Phase to address:**
Feature-pipeline phase (canonical units module + tests), enforced in every later phase's review.

---

### Pitfall 12: Windows-native dev vs Linux containers friction (Docker, Prefect, paths)

**What goes wrong:**
On Windows 11: bind-mounted volumes from NTFS into Linux containers have ~10x slower I/O and broken file-watching; line-ending (CRLF) corruption breaks shell entrypoints inside images (`exec format error` / `\r: command not found`); `os.path` vs POSIX paths diverge between local runs and containerized runs; MLflow tracking URIs with Windows paths (`file:///C:/...`) break when the same code runs in a container; Prefect on native Windows has signal-handling and `multiprocessing` quirks. Each is small; together they burn entire weekends.

**Why it happens:**
Hybrid workflow — editing on Windows, executing in Linux containers — crosses the filesystem/line-ending/path boundary constantly.

**How to avoid:**
- Develop inside **WSL2** with the repo cloned in the WSL filesystem (`~/...`, not `/mnt/c/...`) — this single decision eliminates the volume-performance and most path problems. Docker Desktop with the WSL2 backend.
- `.gitattributes` forcing `* text=auto eol=lf` for `.sh`, Dockerfiles, and entrypoints, committed in repo-setup.
- All service URIs (MLflow tracking, Prefect API, model store) come from env vars in docker-compose, never hardcoded paths; use `pathlib` exclusively.
- Run MLflow server + Prefect server as compose services from the start, so "local" and "deployed" are the same topology.

**Warning signs:**
Container works on `docker run` from a teammate's Mac guide but not yours. `^M` in error output. Tests pass natively, fail in the container (or vice versa). Hot-reload not triggering in mounted volumes.

**Phase to address:**
Repo/infra setup phase (phase 0) — `.gitattributes`, WSL2 decision, and compose skeleton before any feature code.

---

### Pitfall 13: Evidently version churn — building on the legacy API

**What goes wrong:**
Evidently 0.7+ rewrote its core API (Report/Metric structure, unified Tests into Reports, new result objects) on the path to 1.0. Most tutorials, blog posts, and LLM-generated code use the legacy `Report(metrics=[DataDriftPreset()])` patterns from 0.4.x. Copy-pasting them against a current install fails or quietly uses deprecated shims that will be removed; mixing old and new API styles in one codebase is a mess to untangle.

**Why it happens:**
The library moved fast (ML monitoring → LLM observability focus) and the training-data/tutorial corpus lags.

**How to avoid:**
- Pin the Evidently version explicitly; read the official migration guide (docs.evidentlyai.com/faq/migration) before writing monitoring code; verify every snippet against the pinned version's docs, not blog posts.
- Isolate Evidently behind a thin internal interface (`compute_drift_report(reference_df, current_df) -> DriftResult`) so an API change touches one module. You need this seam anyway for the custom QLIKE-degradation monitor, which Evidently doesn't provide out of the box.

**Warning signs:**
ImportErrors on `evidently.report` vs `evidently.Report` paths. Tutorial code requiring a downgrade to install.

**Phase to address:**
Monitoring phase — version pinning + adapter module first.

---

### Pitfall 14: Modest-data ML that overfits the eval, or never beats GARCH and the project has no story

**What goes wrong:**
Daily data gives ~2,500 samples per asset over 10 years (and crypto's pre-2020 regime is arguably a different distribution). Two failure modes: (a) hyperparameter-tuning LightGBM against the walk-forward test windows until it "beats" GARCH — multiple-comparison overfitting that evaporates live; (b) discovering at the end that ML genuinely doesn't beat GARCH at the daily horizon (a very real possibility — the literature finds HAR/GARCH hard to beat on daily single-asset RV) with no plan for what the project then demonstrates.

**Why it happens:**
The eval set gets reused for every tuning iteration; and the project's implicit success criterion ("ML beats GARCH") is partly outside the builder's control.

**How to avoid:**
- Reserve a final untouched holdout period (e.g., the most recent 6–12 months) scored once, at the end. Tune only on inner walk-forward folds.
- Pool assets into one model with asset-ID/asset-class features (more samples, and "cross-asset learning" is a better story than per-asset models).
- Keep LightGBM heavily regularized (shallow trees, high min_child_samples) — defaults overfit 2.5k rows badly.
- **Reframe success now**: the deliverable is the *honest benchmark + the platform*, not the win. PROJECT.md already says this — make the model card and README structure reflect it from the start ("where ML helps: high-vol regimes / crypto; where GARCH holds: calm equity regimes" is a stronger portfolio story than a fragile 5% win).

**Warning signs:**
Dozens of MLflow runs with different hyperparameters all evaluated on the same final windows. QLIKE improvement that shrinks every time the eval window moves forward. Feature count approaching sample count / 10.

**Phase to address:**
ML-model phase (tuning protocol), defined in the evaluation-harness phase (holdout reserved before tuning begins).

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Calling yfinance/ccxt live inside tests and CI | No fixture setup | Flaky CI, rate-limit bans, geo-block failures on Actions runners | Never — record fixtures in phase 1 |
| Inline unit conversions (`*np.sqrt(252)` at call sites) | Fast to write | √252/100x bugs smeared everywhere; un-auditable model card | Never — central units module from feature phase |
| Skipping the embargo gap in walk-forward CV ("only 22 days") | Simpler splitter | Systematic optimism in every reported metric | Never — it's the project's credibility core |
| One shared "daily" calendar for crypto + equities | One DataFrame, easy joins | Fabricated weekend equity data; misaligned cross-asset features | Only in a single-asset-class spike notebook |
| Auto-promote on retrain (skip the gate) initially | Demos "automation" sooner | Silent degradation; the exact anti-pattern interviewers probe for | Acceptable behind a `--dry-run` flag only |
| Hardcoding MLflow stage strings (`"Production"`) | Matches old tutorials | Deprecated API removal; stale-skills signal | Never — aliases cost the same effort |
| SQLite MLflow backend + local artifact dir | Zero setup | Fine for this project's scale | Acceptable throughout (single-user); just keep URIs in env vars for cloud swap |
| Training/serving feature code duplicated (notebook vs API) | Quick first deploy | Training/serving skew — forecasts differ from backtest silently | Never — one feature package imported by both |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| yfinance | Treating it as a stable API; per-ticker download loops; relying on `auto_adjust` default | Cache-first store, batched downloads, retries/backoff, pin version, set `auto_adjust` explicitly, re-validate stored history against fresh pulls |
| ccxt/Binance | Ingesting the still-forming last candle; no `enableRateLimit`; deploying to US cloud/CI (HTTP 451 geo-block) | Drop unclosed candles (Pandera check on bar close time), enable rate limiting, plan exchange fallback, fixtures in CI |
| arch (GARCH) | Fitting raw decimal returns; ignoring convergence flag; mis-inverting auto-rescale | Fit on 100x returns, assert convergence + alpha+beta<1 per refit, single tested variance-unit conversion |
| MLflow | Stage-based promotion (deprecated 2.9+); resolving model URI per-request; no data lineage tags | `@champion`/`@challenger` aliases, resolve at startup + reload hook, tag runs with git SHA + DVC hash |
| Evidently | Copy-pasting 0.4.x legacy-API tutorials; using data-drift presets as retrain triggers | Pin version, follow official migration guide, wrap in adapter, performance-based (QLIKE) retrain trigger |
| Prefect | Native-Windows agent quirks; flows that crash the whole DAG on a transient fetch failure | Run worker in WSL2/container; task-level retries; ingestion failure = skip-with-alert, serving unaffected |
| DVC | Versioning only raw data, not feature parquets; Windows/WSL path mix in `.dvc` files | Version raw + features; keep repo and cache in one filesystem (WSL); record data hash in MLflow tags |
| FastAPI serving | Loading model from registry on every request; no model-version field in responses | Load at startup, expose `/reload`; every response carries model version + feature timestamp for the feedback loop |
| GitHub Actions | Live market-data calls in CI; building Linux images with CRLF entrypoints from Windows checkout | Fixture-based tests; `.gitattributes` eol=lf; build/test the Docker image in CI, not just the package |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Refitting GARCH every day for every asset in the walk-forward backtest | Backtest takes hours; iteration grinds | Refit weekly/monthly with daily `forecast(reindex=...)` updates; cache fitted params per window | >3 assets × >5yr daily backtest |
| Recomputing the full feature matrix on every pipeline run | Pipeline minutes-long for a 1-row update | Incremental append for daily updates; full rebuild only on feature-config change (hash-keyed) | Daily schedule + multi-asset |
| Evidently full-history reference window | Drift reports slow; reference includes old regimes, muting recent drift | Rolling reference window (e.g., trailing 90–180 days), regenerated on each (re)deploy | ~2 years of accumulated predictions |
| SHAP on every API request | p99 latency blows up | Precompute SHAP at training time (global) + batch job for recent predictions; serve cached values | First load test |
| Pandas-everything feature pipeline | Fine at this scale; slow only if intraday added | Don't pre-optimize; Polars swap is the documented stretch path | Only if intraday (stretch goal) lands |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Exchange API keys committed in docker-compose/env files (even read-only ccxt keys) | Key leakage; account abuse | Public OHLCV endpoints need no keys — use keyless ccxt; if keys ever added, `.env` + secrets manager, never committed |
| MLflow + Prefect UIs exposed without auth on cloud deploy | Anyone can delete runs, repoint `@champion`, poison the registry | Local: fine. Cloud phase: reverse proxy with basic auth at minimum; never bind 0.0.0.0 on public IP |
| Inference API unauthenticated + unrate-limited on public cloud | Free-tier resource exhaustion; scraping | API key header + simple rate limit middleware in the cloud-deploy phase |
| Pickled model artifacts loaded from unpinned sources | Arbitrary code execution via pickle | Load only from your own registry; pin model URI by version/alias; verify run lineage tags |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Dashboard shows forecasts without units/horizon ("vol = 0.18") | Reviewer can't tell daily vs annualized, % vs decimal — credibility hit | Label everything: "1-day-ahead realized vol, annualized, %"; consistent units across all charts |
| Forecast-vs-realized chart without the GARCH baseline overlaid | The project's core claim (honest benchmarking) is invisible | Every performance chart shows ML vs GARCH vs EWMA; rolling QLIKE ratio panel |
| Drift page that's a wall of red Evidently defaults | Looks broken/noisy; signals untuned monitoring | Curated panel: data-quality status, performance trend, last retrain + reason, current champion version |
| Model card with only headline metrics | Reads as marketing, not engineering honesty | Include regime breakdown (calm vs stressed), where GARCH wins, known limitations — per PROJECT.md this *is* the feature |
| README without a "why volatility, why these baselines" section | Reviewers assume it's another price-prediction toy | Lead with the eval-rigor story: walk-forward + purging, QLIKE, GARCH benchmark |

## "Looks Done But Isn't" Checklist

- [ ] **Walk-forward harness:** Often missing the embargo/purge gap for multi-day RV labels — verify with the shifted-label leak smoke test and a unit test on timestamp boundaries.
- [ ] **GARCH baseline:** Often "fits" but with failed convergence or boundary parameters — verify per-window convergence flags are logged and asserted, not just the final forecast plotted.
- [ ] **QLIKE metric:** Often correct-looking but fed vols instead of variances — verify `qlike(x, x) == 0` test exists and both eval scripts import the same function.
- [ ] **Ingestion:** Often works for backfill but not for the daily incremental path (incomplete candles, weekend gaps, retry path untested) — verify a simulated "yesterday's run" replays cleanly.
- [ ] **Feedback loop:** Often logs predictions but never joins them to realized labels — verify a table exists with (timestamp, model_version, forecast, realized_RV, QLIKE_contribution) populating automatically.
- [ ] **Champion/challenger:** Often promotes on backtest-vs-live comparisons — verify the gate scores both models on the identical frozen window and logs window dates + data hash.
- [ ] **Rollback:** Often "supported by MLflow" but never executed — verify a scripted alias-flip rollback has been run once end-to-end against the live serving container.
- [ ] **Docker:** Often runs via `uvicorn` locally but the image is broken (CRLF entrypoint, missing model URI env) — verify CI builds and smoke-tests the container, including a real `/predict` call.
- [ ] **Drift monitoring:** Often produces reports nobody acts on — verify each monitor class has a defined consequence (page / log / retrain-gate) written down.
- [ ] **Reproducibility:** Often MLflow logs metrics but not lineage — verify any registered version can answer: git SHA, DVC data hash, feature config, train window.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Leakage discovered after results reported | HIGH | Fix splitter, add leak tests, rerun *all* backtests (baselines + ML), rewrite model card numbers; treat old numbers as void |
| GARCH baseline was misfit (strawman) | MEDIUM | Refit with scaling + convergence checks; rerun comparisons; if ML "advantage" shrinks, update narrative — honesty is the feature |
| QLIKE bug after champion promotions | HIGH | Fix metric, replay promotion decisions against logged eval windows, demote if needed via alias flip; add metric unit tests |
| yfinance silently rewrote history (adjustment change) | MEDIUM | DVC lets you diff stored vs fresh pulls; pin the affected range to the DVC version; document in data README |
| Built on MLflow stages, now migrating | LOW-MEDIUM | Mechanical: map Production→`@champion`, Staging→`@challenger`, swap `transition_model_version_stage` for `set_registered_model_alias`, update serving URI |
| Retrain storm degraded prod model | LOW | Alias-flip rollback to last good version; add cooldown + relative-to-GARCH gating before re-enabling auto-retrain |
| Built monitoring on legacy Evidently API | MEDIUM | Adapter seam limits blast radius to one module; follow official migration guide; re-pin version |
| Windows/Docker path-and-CRLF mess mid-project | MEDIUM | Move repo into WSL2 filesystem, add `.gitattributes`, normalize line endings in one commit, rebuild images |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Windows/Docker/WSL2 friction (12) | Phase 0 — repo & infra setup | Container builds + runs from clean clone; `.gitattributes` present; compose brings up MLflow+Prefect |
| yfinance fragility (5), ccxt gotchas (6) | Phase 1 — ingestion & validation | Simulated daily run with fetch failure → skip-with-alert; Pandera rejects unclosed candles; CI uses fixtures |
| Calendar mismatch (4) | Phase 1 (calendars) + Phase 2 (joins) | No equity rows on non-trading days; cross-asset features use as-of joins; per-asset annualization in config |
| Unit confusion (11) | Phase 2 — feature pipeline | Units module with round-trip property tests; sniff-test bounds on annualized vols |
| GARCH misfit (2) | Phase 2/3 — classical baselines | Convergence + stationarity assertions in pytest; GARCH vs EWMA QLIKE sanity ratio |
| QLIKE bugs (3), lookahead leakage (1) | Phase 3 — evaluation harness (before ML) | `qlike(x,x)==0`; shifted-label leak smoke test; embargo ≥ label horizon enforced by splitter tests |
| Eval overfitting / no-win story (14) | Phase 3 (holdout reserved) + Phase 4 (tuning protocol) | Final holdout scored once; tuning runs only on inner folds; model card has regime breakdown |
| MLflow stages deprecation (8) | Phase 4 — tracking & registry | Promotion helper uses aliases; zero deprecation warnings; lineage tags on every run |
| Training/serving skew (debt table) | Phase 5 — serving | API imports the same feature package as training; container smoke test compares API forecast to offline forecast |
| Drift false positives (7), Evidently churn (13) | Phase 6 — monitoring | Monitor taxonomy doc; drift replayed over a historical vol spike without retrain-trigger firing; Evidently pinned + wrapped |
| Champion/challenger validity (9), silent degradation (10) | Phase 7 — retraining & orchestration | Gate logs frozen window + data hash; no-promote default; scripted rollback executed once; cooldown enforced |

## Sources

- Patton (2011), "Volatility forecast comparison using imperfect volatility proxies" — QLIKE/MSE robustness, why MAE misranks under noisy proxies ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S030440761000076X)); supporting discussion in [arXiv:2506.07928](https://arxiv.org/html/2506.07928v1) (HAR/benchmark hard to beat; QLIKE forms and asymmetry) — HIGH confidence
- arch library docs — DataScaleWarning, rescale behavior, "scale of y between 1 and 1000" ([arch docs](https://bashtage.github.io/arch/univariate/introduction.html)) — HIGH confidence
- yfinance rate-limit issue threads — recurring `YFRateLimitError` waves incl. across IPs ([#2422](https://github.com/ranaroussi/yfinance/issues/2422), [#2411](https://github.com/ranaroussi/yfinance/issues/2411), [#2567](https://github.com/ranaroussi/yfinance/issues/2567)) — HIGH confidence; `auto_adjust` default change to True (~v0.2.51) — MEDIUM confidence (verify against pinned version's changelog in Phase 1)
- MLflow registry stages deprecation (2.9+) and alias migration ([Model Registry workflow docs](https://mlflow.org/docs/latest/ml/model-registry/workflow/), [RFC #10336](https://github.com/mlflow/mlflow/issues/10336)) — HIGH confidence
- Evidently 0.7 API redesign and migration guide ([docs.evidentlyai.com/faq/migration](https://docs.evidentlyai.com/faq/migration), [API change blog](https://www.evidentlyai.com/blog/evidently-api-change)) — HIGH confidence
- ccxt Binance behavior (rate limits, incomplete last candle, HTTP 451 US geo-block), de Prado purged CV/embargo, Docker-on-Windows WSL2/CRLF issues — training-data knowledge consistent with widely documented community experience — MEDIUM confidence (verify ccxt candle behavior against pinned version docs in Phase 1)

---
*Pitfalls research for: crypto + equity volatility forecasting MLOps platform (VolForecast)*
*Researched: 2026-06-10*
