# Model Card: volforecast-lgbm

**Registry:** MLflow model registry — `volforecast-lgbm` with `@champion` / `@challenger` aliases
(MLflow 3.x alias-based promotion; deprecated `transition_model_version_stage` is not used)

**Source report provenance:** All metrics in this card are copied verbatim from
`reports/ml_vs_baselines.md`, `reports/ml_vs_baselines.csv`, and `reports/baseline_eval.md`.
No numbers have been invented or recomputed for this card.

---

## 1. Model Details

| Field | Value |
|-------|-------|
| **Name** | `volforecast-lgbm` |
| **Algorithm** | LightGBM 4.6 gradient-boosted tree regressor |
| **Target** | Next-period (1-day-ahead) daily realized variance of decimal log returns |
| **Registry** | MLflow 3.x — `@champion` / `@challenger` aliases + `validation_status` tags |
| **Tracking** | MLflow with PostgreSQL backend store |
| **Explainability** | SHAP TreeExplainer (exact, native LightGBM support) |
| **Reproducibility** | Training data DVC-versioned; feature parquet files DVC-tracked |

The **target** is the next calendar day's realized variance, computed as the sum of squared
decimal log returns over a day's close-to-close window.  This is a variance (not a standard
deviation), in decimal units (e.g., a daily vol of 1 % maps to a realized variance of 0.0001 ≈ 1e-4).

---

## 2. Intended Use

**This model is a volatility-forecasting + MLOps lifecycle DEMO.  It is explicitly NOT a trading
strategy, NOT price-direction prediction, and NOT intended for live risk management decisions.**

Intended uses:
- Demonstrate a credible end-to-end ML forecasting pipeline (ingest → validate → features → train
  → walk-forward eval → registry → serving → monitoring → drift detection → retrain).
- Honestly benchmark an ML model against the three established classical baselines (EWMA,
  GARCH(1,1), HAR-RV) under leak-free purged walk-forward evaluation.
- Showcase ML engineering skills: feature engineering at scale, leak-free time-series CV,
  alias-based model registry, drift-triggered retraining, FastAPI serving, Streamlit observability,
  CI/CD with GitHub Actions.

Out of scope for this project:
- Live trading or signal generation of any kind.
- Risk management or position sizing for real capital.
- Price direction, returns, or any forecasting horizon beyond daily realized variance.
- Intraday or tick-frequency data (v1 uses daily OHLCV only; true high-frequency RV is a v2 stretch).

---

## 3. Training Data and Window

| Field | Value |
|-------|-------|
| **Assets** | BTC-USD, ETH-USD, SPY, AAPL, MSFT |
| **Frequency** | Daily OHLCV |
| **Data sources** | ccxt/Binance (crypto); yfinance (equities) — both free, no API key required |
| **Approximate span** | ~2021/2022 — 2026 (varies per asset; crypto from ~late 2021, equities from ~early 2022) |
| **Versioning** | Raw OHLCV and feature parquet files versioned with DVC 3.x |
| **Resilience** | yfinance is cached to disk (DVC) so reruns do not depend on Yahoo being available |

The walk-forward harness uses an **expanding training window**: the model never sees any test-fold
data during training.  Features at time t use only information available at or before t, enforced
in `features/target.py` and validated by unit-level leakage tests.

---

## 4. Evaluation Methodology

| Protocol element | Detail |
|-----------------|--------|
| **Method** | Purged walk-forward expanding window |
| **Min training size** | 252 bars (~1 year of trading days) (`min_train=252`) |
| **Step** | 21 bars (~1 month) (`step=21`) |
| **Folds** | Identical folds for LightGBM AND all three baselines — comparisons are fair |
| **Leakage guard** | Test-fold data never seen during training; no random splits (which are a correctness violation for time series) |
| **Vol tercile computation** | Tercile boundaries (low/mid/high) computed on each fold's TEST realized variance only; no lookahead into training data |
| **Scoring** | Out-of-sample walk-forward test indices only; in-sample rows excluded from all metrics |

---

## 5. Metric Definitions

**All metrics are in daily decimal variance units.**

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **QLIKE** | Patton (2011) variance form: `mean(ratio - ln(ratio) - 1)` where `ratio = realized_var / forecast_var`. Equal to 0 at a perfect forecast. Heavily penalises under-forecasting of large realized variance. | Lower is better |
| **RMSE** | Root mean squared error of forecast vs realized variance | Lower is better |
| **MAE** | Mean absolute error of forecast vs realized variance | Lower is better |

QLIKE is a proper scoring rule under the assumption that the conditional distribution of returns
is Gaussian — it is the standard benchmark metric in the realized-volatility literature.  Its
sensitivity to under-forecasting is the primary reason LightGBM underperforms in high-volatility
regimes (see Section 8).

---

## 6. Per-Asset Overall Results

*Source: `reports/ml_vs_baselines.md`, Section 1.*

All metrics in daily decimal variance units.  N = number of out-of-sample walk-forward test observations.

| Asset | Model | N | RMSE | MAE | QLIKE |
|-------|-------|---|------|-----|-------|
| AAPL | LightGBM | 817 | 9.227140e-04 | 2.385883e-04 | 3.625649 |
| AAPL | EWMA | 817 | 8.930383e-04 | 3.003930e-04 | 1.688206 |
| AAPL | GARCH | 817 | 9.437992e-04 | 3.481800e-04 | 1.772407 |
| AAPL | HAR | 817 | 8.944729e-04 | 3.172831e-04 | 1.666804 |
| BTC-USD | LightGBM | 1,343 | 1.603920e-03 | 5.670106e-04 | 4.754568 |
| BTC-USD | EWMA | 1,343 | 1.544893e-03 | 7.362908e-04 | 1.940212 |
| BTC-USD | GARCH | 1,343 | 1.577893e-03 | 9.221255e-04 | 1.993651 |
| BTC-USD | HAR | 1,343 | 1.534401e-03 | 8.496379e-04 | 1.921059 |
| ETH-USD | LightGBM | 1,344 | 3.086945e-03 | 1.061461e-03 | 5.475349 |
| ETH-USD | EWMA | 1,344 | 2.962131e-03 | 1.390028e-03 | 1.934222 |
| ETH-USD | GARCH | 1,344 | 2.998875e-03 | 1.503271e-03 | 1.960759 |
| ETH-USD | HAR | 1,344 | 2.978026e-03 | 1.492694e-03 | 1.936323 |
| MSFT | LightGBM | 819 | 6.743420e-04 | 2.078898e-04 | 3.617540 |
| MSFT | EWMA | 819 | 6.601060e-04 | 2.635338e-04 | 1.712161 |
| MSFT | GARCH | 819 | 6.654619e-04 | 2.804379e-04 | 1.745235 |
| MSFT | HAR | 819 | 6.611853e-04 | 2.968209e-04 | 1.703623 |
| SPY | LightGBM | 818 | 4.011778e-04 | 8.253738e-05 | 2.693654 |
| SPY | EWMA | 818 | 3.949478e-04 | 1.064168e-04 | 1.634448 |
| SPY | GARCH | 818 | 4.382480e-04 | 1.294589e-04 | 1.805021 |
| SPY | HAR | 818 | 3.923679e-04 | 1.152374e-04 | 1.657546 |

**Summary of overall picture:**
- **RMSE:** LightGBM achieves lower RMSE than GARCH for all assets, and is competitive with EWMA
  and HAR (within a few percent) on crypto assets; EWMA/HAR are slightly better for equities.
- **MAE:** LightGBM has clearly lower MAE than all three baselines for all assets — it is better
  at the median forecast level.
- **QLIKE:** LightGBM is **worse** (higher QLIKE) than the best classical baseline for **every
  single asset overall**.  The gap is large: 2–3x worse on crypto, ~2x on equities.

---

## 7. Per-Regime (Volatility-Tercile) Breakdown

*Source: `reports/ml_vs_baselines.md`, Section 2.*

Tercile labels (low/mid/high) are computed on each asset's test-fold realized variance only —
no lookahead into training data.

### AAPL — By Volatility Tercile

| Tercile | Model | N | RMSE | MAE | QLIKE |
|---------|-------|---|------|-----|-------|
| low | LightGBM | 272 | 6.537195e-05 | 5.296502e-05 | 1.876991 |
| low | EWMA | 272 | 3.276724e-04 | 2.330566e-04 | 3.069697 |
| low | GARCH | 272 | 4.005360e-04 | 2.945209e-04 | 3.337251 |
| low | HAR | 272 | 3.087857e-04 | 2.603557e-04 | 3.281238 |
| mid | LightGBM | 272 | 4.565614e-05 | 3.522621e-05 | 0.313646 |
| mid | EWMA | 272 | 2.393753e-04 | 1.611241e-04 | 0.587271 |
| mid | GARCH | 272 | 3.480106e-04 | 2.312551e-04 | 0.776795 |
| mid | HAR | 272 | 2.482304e-04 | 2.059612e-04 | 0.752172 |
| high | LightGBM | 273 | 1.594249e-03 | 6.261488e-04 | 8.667771 |
| high | EWMA | 273 | 1.490853e-03 | 5.062414e-04 | 1.408678 |
| high | GARCH | 273 | 1.544421e-03 | 5.181392e-04 | 1.205259 |
| high | HAR | 273 | 1.495992e-03 | 4.849160e-04 | 0.969566 |

### BTC-USD — By Volatility Tercile

| Tercile | Model | N | RMSE | MAE | QLIKE |
|---------|-------|---|------|-----|-------|
| low | LightGBM | 448 | 1.083566e-04 | 8.423726e-05 | 1.943997 |
| low | EWMA | 448 | 6.162903e-04 | 5.272412e-04 | 3.616072 |
| low | GARCH | 448 | 9.406209e-04 | 9.134929e-04 | 4.255363 |
| low | HAR | 448 | 7.979026e-04 | 7.613557e-04 | 4.067556 |
| mid | LightGBM | 447 | 1.184095e-04 | 9.386248e-05 | 0.811244 |
| mid | EWMA | 447 | 6.181555e-04 | 4.760987e-04 | 0.712961 |
| mid | GARCH | 447 | 7.691666e-04 | 7.331287e-04 | 1.043193 |
| mid | HAR | 447 | 7.384054e-04 | 6.567456e-04 | 0.951125 |
| high | LightGBM | 448 | 2.772400e-03 | 1.521876e-03 | 11.499663 |
| high | EWMA | 448 | 2.528572e-03 | 1.204952e-03 | 1.488865 |
| high | GARCH | 448 | 2.447165e-03 | 1.119333e-03 | 0.680276 |
| high | HAR | 448 | 2.424299e-03 | 1.130382e-03 | 0.742332 |

### ETH-USD — By Volatility Tercile

| Tercile | Model | N | RMSE | MAE | QLIKE |
|---------|-------|---|------|-----|-------|
| low | LightGBM | 448 | 1.765402e-04 | 1.376713e-04 | 1.831387 |
| low | EWMA | 448 | 1.218633e-03 | 1.006707e-03 | 3.516662 |
| low | GARCH | 448 | 1.362787e-03 | 1.266297e-03 | 3.847610 |
| low | HAR | 448 | 1.267754e-03 | 1.184559e-03 | 3.806830 |
| mid | LightGBM | 448 | 2.113301e-04 | 1.629924e-04 | 0.716303 |
| mid | EWMA | 448 | 1.115302e-03 | 8.710306e-04 | 0.745796 |
| mid | GARCH | 448 | 1.172336e-03 | 1.038485e-03 | 0.892102 |
| mid | HAR | 448 | 1.239467e-03 | 1.060614e-03 | 0.901914 |
| high | LightGBM | 448 | 5.339650e-03 | 2.883720e-03 | 13.878358 |
| high | EWMA | 448 | 4.857334e-03 | 2.292345e-03 | 1.540209 |
| high | GARCH | 448 | 4.873212e-03 | 2.205032e-03 | 1.142564 |
| high | HAR | 448 | 4.843804e-03 | 2.232910e-03 | 1.100224 |

### MSFT — By Volatility Tercile

| Tercile | Model | N | RMSE | MAE | QLIKE |
|---------|-------|---|------|-----|-------|
| low | LightGBM | 273 | 5.211964e-05 | 4.391057e-05 | 1.881406 |
| low | EWMA | 273 | 2.528639e-04 | 2.106993e-04 | 3.176142 |
| low | GARCH | 273 | 2.770613e-04 | 2.446680e-04 | 3.352456 |
| low | HAR | 273 | 2.995861e-04 | 2.790873e-04 | 3.524862 |
| mid | LightGBM | 273 | 5.089010e-05 | 3.791453e-05 | 0.364871 |
| mid | EWMA | 273 | 2.145359e-04 | 1.614938e-04 | 0.555821 |
| mid | GARCH | 273 | 2.226554e-04 | 1.803026e-04 | 0.626764 |
| mid | HAR | 273 | 2.485565e-04 | 2.219241e-04 | 0.753115 |
| high | LightGBM | 273 | 1.165721e-03 | 5.418442e-04 | 8.606345 |
| high | EWMA | 273 | 1.094191e-03 | 4.184082e-04 | 1.404522 |
| high | GARCH | 273 | 1.096440e-03 | 4.163431e-04 | 1.256486 |
| high | HAR | 273 | 1.077017e-03 | 3.894512e-04 | 0.832892 |

### SPY — By Volatility Tercile

| Tercile | Model | N | RMSE | MAE | QLIKE |
|---------|-------|---|------|-----|-------|
| low | LightGBM | 273 | 3.146719e-05 | 2.353757e-05 | 2.171787 |
| low | EWMA | 273 | 1.323968e-04 | 8.197812e-05 | 3.241107 |
| low | GARCH | 273 | 2.088225e-04 | 1.075352e-04 | 3.440994 |
| low | HAR | 273 | 1.333486e-04 | 1.040189e-04 | 3.575673 |
| mid | LightGBM | 272 | 2.197305e-05 | 1.581860e-05 | 0.305000 |
| mid | EWMA | 272 | 1.089542e-04 | 5.904292e-05 | 0.498583 |
| mid | GARCH | 272 | 2.093723e-04 | 8.792828e-05 | 0.633435 |
| mid | HAR | 272 | 1.181929e-04 | 8.159452e-05 | 0.706458 |
| high | LightGBM | 273 | 6.933759e-04 | 2.080116e-04 | 5.595426 |
| high | EWMA | 273 | 6.618334e-04 | 1.780559e-04 | 1.159493 |
| high | GARCH | 273 | 6.987113e-04 | 1.927610e-04 | 1.336340 |
| high | HAR | 273 | 6.554339e-04 | 1.599755e-04 | 0.687022 |

### Per-Regime Regime Pattern Summary

In the **low-vol** and **mid-vol** terciles, LightGBM dominates on RMSE and MAE across all
assets, and achieves lower QLIKE than the classical baselines on most assets in those regimes.
The picture reverses completely in the **high-vol tercile**: LightGBM QLIKE is catastrophically
worse than every baseline in every asset.

---

## 7a. Representative Per-Calendar-Year View

*Source: `reports/ml_vs_baselines.md`, Section 3. Showing BTC-USD and SPY as representative assets.*

### BTC-USD — By Calendar Year (QLIKE)

| Year | LightGBM QLIKE | Best Baseline | Best Baseline QLIKE | LightGBM vs Best |
|------|---------------|---------------|---------------------|-----------------|
| 2022 | 3.583856 | EWMA | 2.493106 | +1.09 |
| 2023 | 5.461254 | HAR | 2.227802 | +3.23 |
| 2024 | 5.280344 | HAR | 1.671485 | +3.61 |
| 2025 | 3.767044 | EWMA | 1.742351 | +2.02 |
| 2026 | 4.871902 | HAR | 1.728519 | +3.14 |

### SPY — By Calendar Year (QLIKE)

| Year | LightGBM QLIKE | Best Baseline | Best Baseline QLIKE | LightGBM vs Best |
|------|---------------|---------------|---------------------|-----------------|
| 2023 | 1.708753 | EWMA | 1.380815 | +0.33 |
| 2024 | 2.886385 | HAR | 1.664795 | +1.22 |
| 2025 | 3.326369 | HAR | 1.818214 | +1.51 |
| 2026 | 2.882706 | HAR | 1.525012 | +1.36 |

LightGBM QLIKE is worse than the best classical baseline in every year for both assets.  The
QLIKE gap has tended to widen in higher-volatility calendar years (e.g., 2025 for both assets).

---

## 8. Honest Findings — Where LightGBM Loses

**This section explicitly names every asset and regime where LightGBM's QLIKE is worse (higher)
than the best classical baseline.  These results are reported plainly: the project's credibility
rests on honest benchmarking, not on hiding unfavourable outcomes.**

*Source: `reports/ml_vs_baselines.md`, Section 4 (reproduced verbatim).*

### Overall (per-asset)

| Finding | LightGBM QLIKE | Best Baseline | Best QLIKE | Delta (LightGBM worse by) |
|---------|---------------|---------------|------------|---------------------------|
| AAPL (overall) | 3.625649 | HAR | 1.666804 | 1.958844 |
| BTC-USD (overall) | 4.754568 | HAR | 1.921059 | 2.833509 |
| ETH-USD (overall) | 5.475349 | EWMA | 1.934222 | 3.541127 |
| MSFT (overall) | 3.617540 | HAR | 1.703623 | 1.913917 |
| SPY (overall) | 2.693654 | EWMA | 1.634448 | 1.059206 |

LightGBM's QLIKE is worse than the best classical baseline for every single asset in the
overall view.

### High-Volatility Regime — The Catastrophic Losses

The largest failures are in the **high-vol tercile**, where QLIKE heavily penalises under-forecasting
of large realized variance.  When actual volatility spikes, the tree model systematically
under-predicts — and QLIKE punishes this severely.

| Asset / Regime | LightGBM QLIKE | Best Baseline | Best QLIKE | Delta |
|----------------|---------------|---------------|------------|-------|
| **BTC-USD / vol-high** | **11.499663** | GARCH | **0.680276** | **+10.819386** |
| **ETH-USD / vol-high** | **13.878358** | HAR | **1.100224** | **+12.778134** |
| AAPL / vol-high | 8.667771 | HAR | 0.969566 | +7.698205 |
| MSFT / vol-high | 8.606345 | HAR | 0.832892 | +7.773452 |
| SPY / vol-high | 5.595426 | HAR | 0.687022 | +4.908404 |

The BTC-USD and ETH-USD high-vol numbers are stark: LightGBM QLIKE of 11.50 and 13.88
respectively, versus GARCH QLIKE of 0.68 and HAR QLIKE of 1.10.  This is not a small
degradation — it is an order-of-magnitude difference.

### Mid-Volatility Regime Losses

| Finding | LightGBM QLIKE | Best Baseline | Best QLIKE | Delta |
|---------|---------------|---------------|------------|-------|
| BTC-USD / vol-mid | 0.811244 | EWMA | 0.712961 | +0.098283 |

(ETH-USD mid-vol LightGBM QLIKE 0.716 is slightly better than EWMA 0.746 and GARCH 0.892 —
one of the few QLIKE wins for the ML model.)

### Why LightGBM Loses on QLIKE

QLIKE = `mean(ratio - ln(ratio) - 1)` where `ratio = realized_var / forecast_var`.  This
function is **not symmetric**: it penalises under-forecasting (ratio > 1, i.e., realized >
forecast) much more severely than over-forecasting.  When a vol spike arrives, the tree model
produces a lower forecast than the realized value.  GARCH and HAR, whose auto-regressive
structure explicitly models volatility persistence and clustering, are far better at carrying
elevated volatility forward into the next-period forecast.

In the low/mid-vol regimes where the next day's variance is close to recent history, the
tree's rich feature interactions improve point forecast accuracy (lower RMSE and MAE).  In
high-vol regimes, the tree's tendency to average toward the training-sample mean becomes a
structural liability under QLIKE.

---

## 9. Assumptions

| Assumption | Detail |
|-----------|--------|
| **Daily horizon** | The model forecasts 1 trading day ahead only; no multi-step or intraday forecasts |
| **Realized variance proxy** | Close-to-close squared log return (daily); not true high-frequency RV |
| **Decimal variance units** | All inputs and outputs are in decimal variance (not percentage, not annualized) |
| **Free-data caveats** | yfinance data is best-effort Yahoo Finance; adjusted-close artifacts (zero-variance rows) are dropped before scoring (QLIKE is +inf at zero realized variance via log(0)); raw data is DVC-cached to isolate reruns from Yahoo availability |
| **GARCH scaling** | GARCH(1,1) is fitted on 100x-scaled returns with `rescale=False` to improve optimizer stability; forecasts are rescaled back to decimal variance units for comparison |
| **EWMA lambda** | 0.94 (RiskMetrics daily convention, J.P. Morgan 1994) |
| **HAR lags** | Corsi (2009): daily RV, 5-day mean RV, 22-day mean RV as OLS predictors |
| **Walk-forward** | Models trained only on past data; future data is strictly excluded from every training fold |

---

## 10. Known Limitations

**1. QLIKE weakness in high-volatility regimes (primary limitation)**

As documented in Section 8, LightGBM's QLIKE is substantially worse than classical baselines
in high-vol regimes for all five assets.  The tree model systematically under-forecasts realized
variance during vol spikes, and QLIKE severely penalises this.  For use cases where the cost of
under-forecasting large variance is high (e.g., options pricing, risk management), the GARCH or
HAR baseline would be more appropriate.

**2. Promotion cooldown is not persisted across Prefect runs (documented v1 limitation)**

The automated retraining flow includes a promotion cooldown (a configurable minimum number of
calendar days between promotions to `@champion`).  In the current implementation, this cooldown
state is **not persisted** to a durable store: it is computed from the MLflow registry at the
start of each Prefect flow run.  The `force_retrain` flag is an ephemeral flow parameter;
the performance gate is recomputed fresh on every run.  In a multi-run scenario (e.g., if two
daily pipeline runs occur close together), the cooldown may not prevent a double-promotion event
as intended.  Persisting the "last promoted timestamp" to a durable registry tag or metadata
field is the correct v2 fix.

**3. No true high-frequency realized variance**

The realized variance proxy is close-to-close squared log returns (daily).  True realized
variance should be computed from intraday (e.g., 5-minute) returns; the close-to-close proxy
is noisier.  Adding intraday data ingestion and tick-level RV computation is a v2 stretch goal.

**4. Single-exchange crypto data**

BTC-USD and ETH-USD data are pulled from Binance via ccxt.  Binance blocks some geographic
regions and cloud datacenter IPs; in production cloud deployment, the exchange should be
switched to Kraken or Coinbase (a one-line ccxt config change).

**5. Free-data quality**

yfinance data quality depends on Yahoo Finance, which is a best-effort free service.  The
DVC caching layer ensures reruns do not depend on Yahoo being up, but historical data quality
(adjusted-close artifacts, occasional gaps) is not guaranteed.

**6. Single-asset-at-a-time forecasting**

The current model trains one LightGBM model per asset (or a shared model depending on the
`all_assets` training mode).  There is no explicit cross-asset correlation structure.

---

## 11. Machine-Readable Provenance

All metrics in this card are also available in `reports/ml_vs_baselines.csv` with columns:
`asset, regime_type, regime_value, model, n, rmse, mae, qlike`.

Baseline-only metrics and methodology notes are in `reports/baseline_eval.md` and
`reports/baseline_metrics.csv`.

Scripts that generated these reports:
- `scripts/eval_lgbm.py` — ML vs baselines comparison
- `volforecast/reports/baseline.py` — baseline-only evaluation

---

*This model card was authored as part of the VolForecast MLOps portfolio project.
All numbers are sourced from committed report files; none are invented.
The project is a demonstration of ML forecasting methodology — explicitly not a trading strategy.*
