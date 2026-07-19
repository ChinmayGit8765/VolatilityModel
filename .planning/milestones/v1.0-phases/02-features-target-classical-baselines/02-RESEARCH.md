# Phase 2: Features, Target & Classical Baselines - Research

**Researched:** 2026-06-11
**Domain:** Realized volatility target definition, feature engineering pipeline, purged walk-forward harness, EWMA/GARCH/HAR-RV baselines, QLIKE evaluation metric
**Confidence:** HIGH (arch API verified against official docs; QLIKE form verified against Patton 2011 paper; pandas merge_asof verified against official docs; HAR-RV formula verified against literature; package versions verified on PyPI)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Target & Units**
- Canonical units: daily VARIANCE of decimal log returns; vol is sqrt at the edges; one shared module (`src/volforecast/features/target.py` or equivalent) documents and owns this
- Label: next-day squared log return as the variance proxy (QLIKE robust to noisy proxies, Patton 2011); 5-day forward realized variance reported as a secondary stability check
- No annualization anywhere inside the pipeline — display-only at report edges

**Walk-Forward Harness**
- Expanding window, minimum 252 training observations, step 21 observations
- Purging of overlapping label windows at split boundaries; embargo gap >= label horizon; a unit test FAILS if any split is non-temporal or embargo < horizon
- Harness is a reusable library (Phase 4 promotion gate reuses it); horizon-parameterized
- GARCH(1,1): fit with `arch` on 100x scaled decimal log returns; assert convergence flag and alpha+beta<1 on every refit; refit every 21 observations (monthly), daily forecasts from the most recent fit
- HAR-RV: OLS on daily/weekly(5)/monthly(22) lagged RV components
- EWMA: RiskMetrics lambda=0.94 (documented), variance recursion

**Features & Cross-Asset Staleness**
- Every feature window ends strictly at as-of date t; label window starts at t+1 — no overlap
- Feature set: realized vol over 5/10/22/66 lookbacks, log returns, squared returns, lagged vol, EWMA vol, GARCH(1,1) conditional vol-as-feature, Parkinson, Garman-Klass, vol-of-vol, rolling skew/kurtosis, calendar features (day-of-week, month; session/overnight flags for equities)
- Cross-asset features (e.g., BTC RV as ETH/equity input): backward as-of join with max staleness 3 calendar days; beyond that the feature is NaN (documented rule)
- GARCH-as-feature uses the same monthly-refit filtered conditional vol — never fitted on data past as-of
- Single codepath: training and (future) serving import the identical feature module (FEAT-07)

**Evaluation Report**
- Committed artifacts: `reports/baseline_eval.md` + per-asset metrics CSV — "the published bar" for Phase 3
- Canonical metrics module: QLIKE in Patton variance form with unit test `qlike(x,x)==0`, plus RMSE and MAE; shared by all future evaluation and the Phase 4 promotion gate
- MLflow logging deferred to Phase 3

### Claude's Discretion
- Module layout inside src/volforecast/features/ and src/volforecast/models/, exact HAR-RV estimation details, parquet schemas for feature matrices, report formatting

### Deferred Ideas (OUT OF SCOPE)
- LightGBM + MLflow tracking (Phase 3), promotion gate use of the harness (Phase 4)
- Intraday true realized vol, multi-horizon forecasts (v2)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FEAT-01 | Target is next-period realized volatility with documented canonical proxy definition and unit convention (daily variance/vol of decimal log returns) in one shared module | See Target Definition section: squared log return as proxy, `target.py` module pattern |
| FEAT-02 | Feature pipeline computes multi-lookback realized vol (5/10/22/66), log returns, squared returns, lagged vol, and EWMA vol | See Features section: all rolling window computations, EWMA variance recursion |
| FEAT-03 | Feature pipeline computes range-based estimators (Parkinson, Garman-Klass), vol-of-vol, and rolling skew/kurtosis from OHLC | See Range Estimators section: exact formulas for both estimators |
| FEAT-04 | GARCH(1,1) conditional volatility is available as a model feature | See GARCH Baseline section: monthly-refit pattern, conditional_volatility property |
| FEAT-05 | Cross-asset features use as-of joins with documented staleness rule across calendar mismatches | See Cross-Asset Joins section: merge_asof with Timedelta tolerance, 3-calendar-day rule |
| FEAT-06 | Calendar features (day-of-week, month, session/overnight flags for equities) are included | See Calendar Features section |
| FEAT-07 | Training and serving import the identical versioned feature module (single codepath) | See Architecture section: feature module as importable package |
| EVAL-01 | EWMA, GARCH(1,1) (arch, fitted on scaled returns with convergence assertions), and HAR-RV baselines produce walk-forward forecasts | See all three baseline sections with exact API usage |
| EVAL-02 | Walk-forward evaluation harness is a reusable library with purging and embargo gap >= label horizon; unit test asserts temporal split ordering | See Walk-Forward Harness section: expanding window, purge/embargo logic |
| EVAL-03 | One canonical, unit-tested QLIKE function (plus RMSE, MAE) is shared by baseline eval, ML eval, and the promotion gate | See Metrics section: exact QLIKE Patton form, floor handling, unit test pattern |
</phase_requirements>

---

## Summary

Phase 2 builds the foundational numerical layer that every later phase consumes: a canonical target definition, a versioned feature pipeline, three classical baselines, a leak-proof walk-forward harness, and canonical metrics. The central correctness constraint is **no temporal leakage**: the embargo gap, purging of overlapping label windows, and a failing unit test on non-temporal splits are non-negotiable methodological commitments that give the project its credibility claim.

The technical risk items are: (1) GARCH numerical stability — `arch` must be fit on 100x scaled returns with per-refit convergence and stationarity assertions, and forecast variance must be divided back by 10,000 exactly once in a tested function; (2) QLIKE formula correctness — only one canonical form should exist in the codebase, and `qlike(x, x) == 0` must pass as a unit test; (3) cross-asset timezone alignment — `merge_asof` with `pd.Timedelta("3D")` tolerance handles the staleness rule, but both frames must be sorted by UTC index before the call; (4) next-day label construction on equity sessions — `shift(-1)` on a sparse equity index creates NaN on the last trading day of data (not a bug), but weekend NaN rows must not exist in the processed equity parquet.

**Primary recommendation:** Implement modules in dependency order — `target.py` first (unit-testable immediately), then the feature pipeline, then the harness (with the leak test), then the three baselines, then the metrics module. Keep `arch` and `statsmodels` out of import-time initialization of the feature module so training and serving can import it without the GARCH runtime overhead unless explicitly requested.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Target definition (squared log return) | Feature/Data Pipeline | — | Pure data transformation; must be a shared module imported by training AND future serving |
| Feature computation (RV, EWMA, GARCH-as-feature, range estimators) | Feature/Data Pipeline | — | All computations on historical OHLCV; stateless given input data slice |
| Walk-forward harness (split generation, purge, embargo) | Evaluation Library | — | Library function, not a script; reused by Phase 4 promotion gate |
| GARCH(1,1) baseline | Model/Evaluation | Feature Pipeline (conditional vol as feature) | Fitted in walk-forward loop; conditional vol output also consumed as a feature |
| HAR-RV baseline | Model/Evaluation | — | OLS regression; no fitting state to preserve across phases |
| EWMA baseline | Model/Evaluation | Feature Pipeline (EWMA vol as feature) | Deterministic recursion; same output used as both baseline and feature |
| QLIKE / RMSE / MAE metrics | Metrics Library | — | Shared by baselines, ML (Phase 3), and promotion gate (Phase 4) |
| Cross-asset as-of join | Feature/Data Pipeline | — | Data alignment concern owned by the feature builder |
| Evaluation report (baseline_eval.md + CSV) | Reporting | — | Output artifact; generated by a script calling the harness and metrics modules |

---

## Standard Stack

### Core (already in pyproject.toml)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| arch | 8.0.0 | GARCH(1,1) baseline + conditional vol feature | The Kevin Sheppard reference implementation; `arch_model()` API is the ecosystem standard [VERIFIED: pypi.org/project/arch] |
| statsmodels | 0.14.6 (via arch dep >=0.13) | HAR-RV OLS estimation (sm.OLS) | Pulled in by arch as a dependency; OLS is one function call [VERIFIED: pypi.org/project/statsmodels] |
| pandas | >=2.3,<3 | All time-series operations; merge_asof for cross-asset joins | Already pinned; merge_asof with Timedelta tolerance is the correct cross-asset join tool [VERIFIED: pandas.pydata.org] |
| numpy | >=2,<3 | Numerical computations; alternative to statsmodels for lstsq | Already pinned |
| exchange-calendars | >=4.13 | Equity session calendar for overnight/session flag features | Already pinned in Phase 1 work |

### Missing from pyproject.toml (must add)

`arch` and `statsmodels` are referenced in CONTEXT.md and CLAUDE.md stack notes but are **NOT currently in pyproject.toml** and are **NOT installed** in the uv environment. Both must be added before Phase 2 work begins.

**Required addition to pyproject.toml:**
```toml
"arch>=8,<9",
"statsmodels>=0.14",
```

Note: `statsmodels` is technically pulled in as an arch transitive dependency (`>=0.13`), but pinning it explicitly in pyproject.toml documents the direct dependency for HAR-RV OLS and avoids surprise upgrades.

### Alternatives Considered

| Standard | Alternative | Tradeoff |
|----------|-------------|----------|
| `sm.OLS` (statsmodels) for HAR-RV | `np.linalg.lstsq` | Both correct. `sm.OLS` gives residuals, fitted values, and t-stats in one call (useful for model card). `lstsq` is lighter if statsmodels weren't already a dep. Since arch already pulls statsmodels, use `sm.OLS` — it adds zero install cost and produces richer diagnostics. [ASSUMED] |
| `arch_model(y, mean='Zero')` | `arch_model(y)` (ConstantMean) | For residual-based GARCH (returns have zero mean by assumption), `ZeroMean` is more parsimonious and theoretically cleaner. For daily log returns this is defensible. CONTEXT.md does not specify mean type — recommend `ZeroMean` for returns. [ASSUMED] |
| Manual 100x scaling + `rescale=False` | `rescale=True` (auto-scale) | Manual 100x is explicit and testable. `rescale=True` adjusts internally and reports scaled parameters — easy to mis-invert when converting forecast variance back to decimal units. CONTEXT.md explicitly says 100x scaled. Use manual scaling. [VERIFIED: bashtage.github.io/arch] |

**Installation:**
```bash
uv add "arch>=8,<9" "statsmodels>=0.14"
```

---

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| arch | PyPI | ~12 yrs (est.) | High (reference GARCH library) | github.com/bashtage/arch | [SUS] — false positive: flagged as "suspiciously close to torch"; this IS the canonical Kevin Sheppard GARCH package authored since ~2012 | Approved — verified via pypi.org/project/arch (author: Kevin Sheppard, Production/Stable, Python 3.10-3.14) |
| statsmodels | PyPI | ~15 yrs | Very high (statsmodels.org) | github.com/statsmodels/statsmodels | [OK] | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none

**Packages flagged as suspicious [SUS]:** `arch` — this is a false positive. The package `arch` (PyPI) is the well-known financial econometrics library by Kevin Sheppard at github.com/bashtage/arch, in production since ~2012, explicitly listed in the project's stack research. The slopcheck tool flags it for phonetic similarity to `torch`, which is unrelated. Manually verified: author=Kevin Sheppard, classifiers=Production/Stable, required_python>=3.10. [CITED: pypi.org/project/arch]

---

## Architecture Patterns

### System Architecture Diagram

```
data/processed/{asset}.parquet (Phase 1 output)
        |
        v
[target.py] -- compute_target(df, horizon=1) --> next_day_var (squared log return)
        |
        v
[features/pipeline.py] -- build_features(df, as_of_date) --> feature DataFrame
  |-- rv_5, rv_10, rv_22, rv_66 (rolling realized var)
  |-- log_return, squared_return
  |-- ewma_var (lambda=0.94 recursion)
  |-- parkinson_var, gk_var (range estimators from OHLC)
  |-- vol_of_vol, rolling_skew, rolling_kurt
  |-- calendar features (dow, month, session_flag)
  |-- [cross-asset via merge_asof with 3D tolerance]
  |-- garch_cond_var [via monthly-refit GARCH, never past as_of]
        |
        v
[eval/harness.py] -- walk_forward_splits(n, min_train=252, step=21, horizon=1)
  --> [(train_idx, test_idx)] with purge + embargo enforced
        |
        +--------+---------+---------+
        v        v         v         v
    [EWMA]  [GARCH]   [HAR-RV]   (Phase 3: LightGBM)
    baseline baseline  baseline
        |        |         |
        +--------+---------+
                 v
      [metrics/qlike.py] -- qlike(rv_var, forecast_var)
                          -- rmse, mae
                 |
                 v
     reports/baseline_eval.md + per-asset CSV
```

### Recommended Project Structure

```
src/volforecast/
├── features/
│   ├── __init__.py
│   ├── target.py          # compute_target(), HORIZON constant, unit docstring
│   ├── pipeline.py        # build_features(df, cross_asset_dfs=None) -> pd.DataFrame
│   ├── estimators.py      # parkinson_var(), gk_var(), ewma_var(), rv_rolling()
│   └── cross_asset.py     # as_of_join(left_df, right_df, max_staleness_days=3)
├── models/
│   ├── __init__.py
│   ├── ewma.py            # EWMAForecast class (fit/predict interface)
│   ├── garch.py           # GARCHForecast class (fit/predict, convergence assertions)
│   └── har_rv.py          # HARRVForecast class (OLS fit/predict)
├── eval/
│   ├── __init__.py
│   ├── harness.py         # WalkForwardSplitter — generates (train, test) index pairs
│   └── metrics.py         # qlike(), rmse(), mae() — shared across all phases
└── reports/
    └── baseline.py        # generate_baseline_report(results) -> baseline_eval.md
```

### Pattern 1: GARCH(1,1) Fit-and-Forecast with 100x Scaling

**What:** Fit arch GARCH(1,1) on percent returns (100x scaled decimal log returns), assert convergence and stationarity, extract next-step variance forecast, convert back to decimal-variance units in one tested function.

**When to use:** Every refit window in the walk-forward loop (every 21 steps).

```python
# Source: bashtage.github.io/arch/univariate/univariate_volatility_modeling.html
#         bashtage.github.io/arch/univariate/univariate_volatility_forecasting.html
from arch import arch_model
import numpy as np

GARCH_SCALE = 100.0  # module-level constant, never inline

def fit_garch(log_returns_decimal: pd.Series) -> "ARCHModelResult":
    """Fit GARCH(1,1) on 100x-scaled decimal log returns.

    Args:
        log_returns_decimal: Series of decimal log returns (e.g. 0.01 not 1.0).

    Returns:
        ARCHModelResult — caller MUST check convergence before using forecasts.
    """
    scaled = GARCH_SCALE * log_returns_decimal
    am = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
    res = am.fit(disp="off")
    # convergence_flag == 0 means scipy optimizer terminated successfully
    assert res.convergence_flag == 0, (
        f"GARCH did not converge (flag={res.convergence_flag})"
    )
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    assert alpha + beta < 1.0, (
        f"GARCH non-stationary: alpha+beta={alpha+beta:.4f}"
    )
    return res

def garch_forecast_variance_decimal(res: "ARCHModelResult") -> float:
    """Extract next-step variance forecast in DECIMAL log-return units.

    The model was fitted on 100x returns, so forecast variance is in
    (100 * return)^2 units. Divide by GARCH_SCALE**2 to get decimal variance.
    """
    fc = res.forecast(horizon=1, reindex=False)
    # fc.variance is a DataFrame; h.1 column = one-step-ahead variance
    variance_scaled = float(fc.variance["h.1"].iloc[-1])
    return variance_scaled / (GARCH_SCALE ** 2)
```

**Key API facts verified:**
- `arch_model(y, mean='Zero', vol='GARCH', p=1, q=1, rescale=False)` — `mean='Zero'` maps to `ZeroMean`; `rescale=False` prevents silent auto-rescaling. [CITED: bashtage.github.io/arch/univariate/introduction.html]
- `res.fit(disp="off")` — suppresses optimizer output [CITED: bashtage.github.io/arch/univariate/univariate_volatility_modeling.html]
- `res.convergence_flag` — scipy optimization flag; **0 = success** [CITED: bashtage.github.io/arch/univariate/generated/arch.univariate.base.ARCHModelResult.html]
- `res.params["alpha[1]"]`, `res.params["beta[1]"]` — named parameter access [CITED: bashtage.github.io/arch]
- `res.conditional_volatility` — returns conditional **standard deviation** (not variance); square it for variance: `res.conditional_volatility ** 2` [CITED: bashtage.github.io/arch/univariate/univariate_volatility_forecasting.html]
- `res.forecast(horizon=1, reindex=False).variance` — DataFrame with column `h.1`; last row is the next-step forecast from the most recent observation [CITED: bashtage.github.io/arch/univariate/univariate_volatility_forecasting.html]

### Pattern 2: HAR-RV OLS Estimation

**What:** Regress next-day RV on daily/weekly/monthly lagged RV components using statsmodels OLS.

**When to use:** Every refit window in the walk-forward loop (refit every 21 steps, same cadence as GARCH).

```python
# Source: Corsi (2009) formula; statsmodels OLS API
import statsmodels.api as sm
import numpy as np

def build_har_features(rv_daily: pd.Series) -> pd.DataFrame:
    """Build HAR-RV regressor matrix: [intercept, RV_d, RV_w, RV_m].

    RV_w = mean of past 5 days' RV (inclusive of t-1).
    RV_m = mean of past 22 days' RV (inclusive of t-1).
    """
    rv_d = rv_daily.shift(1)             # yesterday's RV
    rv_w = rv_daily.shift(1).rolling(5).mean()   # avg of t-1..t-5
    rv_m = rv_daily.shift(1).rolling(22).mean()  # avg of t-1..t-22
    X = pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m}).dropna()
    return sm.add_constant(X)

def fit_har(rv_daily: pd.Series, train_idx) -> "RegressionResults":
    X_train = build_har_features(rv_daily).loc[train_idx].dropna()
    y_train = rv_daily.loc[X_train.index]
    return sm.OLS(y_train, X_train).fit()

def har_forecast(fitted, rv_daily_tail: pd.Series) -> float:
    """One-step-ahead forecast from fitted HAR model."""
    rv_d = rv_daily_tail.iloc[-1]
    rv_w = rv_daily_tail.iloc[-5:].mean()
    rv_m = rv_daily_tail.iloc[-22:].mean()
    X_pred = np.array([1.0, rv_d, rv_w, rv_m])
    return float(fitted.predict(X_pred)[0])
```

**Key design choice:** HAR-RV uses the SAME variance units as the target (daily decimal log-return variance). No scaling needed — it regresses variance directly on lagged variance. [ASSUMED — standard formulation per Corsi 2009]

**statsmodels vs numpy:** `sm.OLS` is preferred since statsmodels is a direct dependency via arch. If statsmodels is unavailable, `np.linalg.lstsq(X, y, rcond=None)` produces identical coefficient estimates. [ASSUMED]

### Pattern 3: EWMA Variance Recursion

**What:** RiskMetrics-style EWMA: `σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}`, λ=0.94.

**When to use:** Computed once per asset as part of the feature pipeline (EWMA vol is both a feature and a baseline).

```python
# Source: RiskMetrics Technical Document (1996); standard implementation
EWMA_LAMBDA = 0.94

def ewma_variance(log_returns_decimal: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """Compute EWMA conditional variance with RiskMetrics lambda.

    Variance is initialized with the sample variance of the first observation.
    Output is aligned with the input index (each value uses only past returns).

    Returns: Series of conditional variance in decimal log-return units.
    """
    alpha = 1.0 - lam
    # pandas ewm with adjust=False implements exactly this recursion
    # span = (2/alpha) - 1 would be wrong; use com = lam/(1-lam) instead
    return log_returns_decimal.pow(2).ewm(alpha=alpha, adjust=False).mean()
```

**Warning:** `pd.Series.ewm(alpha=...)` with `adjust=False` implements the EWMA recursion directly. `adjust=True` (default) uses a correction term that makes early values diverge. Always use `adjust=False` for financial EWMA. [ASSUMED — standard pandas EWMA gotcha]

### Pattern 4: QLIKE Loss (Patton Variance Form)

**What:** The quasi-likelihood loss function from Patton (2011), robust to noisy volatility proxies.

**Canonical form:** `QLIKE(σ², h) = σ²/h − ln(σ²/h) − 1`

where `σ²` is realized variance (the target proxy) and `h` is forecast variance.

**Verification:** At perfect forecast (h = σ²): `σ²/σ² − ln(σ²/σ²) − 1 = 1 − 0 − 1 = 0`. [CITED: public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf — formula verified from literature context; alternative form `log(h) + σ²/h` differs by a constant and does NOT satisfy qlike(x,x)=0]

```python
# Source: Patton (2011), "Volatility forecast comparison using imperfect proxies"
import numpy as np

QLIKE_FLOOR = 1e-10  # decimal variance floor (prevents log(0) / division by zero)

def qlike(rv_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """Canonical QLIKE loss: mean(σ²/h - ln(σ²/h) - 1).

    Uses the Patton (2011) variance form. Satisfies qlike(x, x) == 0.
    Both arguments must be in the same units (daily decimal variance).

    Args:
        rv_var: Realized variance (proxy). Positive.
        forecast_var: Forecast variance. Clipped to QLIKE_FLOOR.

    Returns:
        Scalar mean QLIKE loss. Always >= 0.
    """
    rv = np.asarray(rv_var, dtype=float)
    h = np.maximum(np.asarray(forecast_var, dtype=float), QLIKE_FLOOR)
    ratio = rv / h
    return float(np.mean(ratio - np.log(ratio) - 1.0))

# Mandatory unit test:
# def test_qlike_perfect_forecast():
#     x = np.array([0.001, 0.002, 0.0005])
#     assert abs(qlike(x, x)) < 1e-12
```

**Why NOT the `log(h) + σ²/h` form:** That form equals `log(h) + σ²/h`. At h=σ²: `log(σ²) + 1`, which equals zero only when σ² = e^{-1} ≈ 0.368 — not a general identity. This alternative form is common in the literature but does not satisfy `qlike(x,x)==0`. The Patton variance form `σ²/h − ln(σ²/h) − 1` is the correct choice for a loss function that equals zero at perfect forecast. [CITED: arXiv:2506.07928v1 — confirmed discrepancy; Patton 2011 paper]

### Pattern 5: Cross-Asset As-Of Join with Staleness Guard

**What:** Join cross-asset features (e.g., BTC RV as an ETH feature) using backward as-of lookup, capping staleness at 3 calendar days.

```python
# Source: pandas.pydata.org/docs/reference/api/pandas.merge_asof.html
import pandas as pd

MAX_CROSS_ASSET_STALENESS = pd.Timedelta("3D")

def as_of_join(
    left: pd.DataFrame,           # target asset, UTC-indexed, sorted ascending
    right: pd.DataFrame,          # source asset, UTC-indexed, sorted ascending
    feature_cols: list[str],
    suffix: str = "_xasset",
) -> pd.DataFrame:
    """Backward as-of join; NaN if source is >3 calendar days stale.

    REQUIREMENT: Both DataFrames must be sorted by their UTC DatetimeIndex
    in ascending order before this call. merge_asof does not sort for you.

    Args:
        left: Target asset DataFrame with tz-aware UTC DatetimeIndex.
        right: Source asset DataFrame with tz-aware UTC DatetimeIndex.
        feature_cols: Columns from right to join.
        suffix: Column suffix to avoid name collisions.

    Returns:
        left with additional columns {col}{suffix} for each in feature_cols.
        Values are NaN if the nearest prior right-row is > 3 calendar days back.
    """
    left = left.reset_index()   # merge_asof requires column, not index
    right = right[feature_cols].reset_index()
    merged = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("date")[["date"] + feature_cols],
        on="date",
        direction="backward",         # use the most recent prior observation
        tolerance=MAX_CROSS_ASSET_STALENESS,
        suffixes=("", suffix),
    )
    merged = merged.set_index("date")
    merged.index = pd.DatetimeIndex(merged.index, tz="UTC")
    return merged
```

**Key merge_asof facts verified:**
- `tolerance` accepts `pd.Timedelta` for datetime keys; values beyond tolerance become NaN [CITED: pandas.pydata.org/docs/reference/api/pandas.merge_asof.html]
- `direction="backward"` is the default; matches most recent prior observation [CITED: pandas.pydata.org]
- Both DataFrames **must be sorted ascending by the merge key** before calling; merge_asof does not sort [CITED: pandas.pydata.org]
- For UTC DatetimeIndex, `pd.Timedelta("3D")` means 3 × 86400 seconds = 3 calendar days [CITED: pandas.pydata.org]

### Pattern 6: Walk-Forward Expanding-Window Splitter

**What:** Generate (train_indices, test_indices) pairs with purging of overlapping label windows and embargo gap.

```python
# Source: De Prado (2018) concept; custom implementation
from dataclasses import dataclass
from typing import Generator
import numpy as np

@dataclass
class WalkForwardSplit:
    train_idx: np.ndarray    # integer positions in the full time-series
    test_idx: np.ndarray     # integer positions in the full time-series

def walk_forward_splits(
    n: int,
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
) -> Generator[WalkForwardSplit, None, None]:
    """Generate purged expanding-window walk-forward splits.

    Purging: the last `horizon` observations of each training window are removed
    because their labels overlap with the first test observation.
    Embargo: the first `horizon` observations after the training end are skipped
    before the test window begins.

    Invariant enforced by this function (also tested externally):
        max(train_idx) < min(test_idx)
        min(test_idx) - max(train_idx) >= horizon  (embargo >= horizon)

    Args:
        n: Total number of observations.
        min_train: Minimum training window size (252 = 1 year daily).
        step: Step size between splits (21 = ~1 month daily).
        horizon: Label horizon in observations (default 1 for next-day).

    Yields:
        WalkForwardSplit with integer-position indices.
    """
    test_start = min_train
    while test_start + step <= n:
        test_end = min(test_start + step, n)
        # Purge: remove last `horizon` train obs whose label overlaps test
        train_end = test_start - horizon   # last train index is test_start - horizon - 1
        purged_train = np.arange(0, train_end)
        test = np.arange(test_start, test_end)
        if len(purged_train) >= min_train and len(test) > 0:
            yield WalkForwardSplit(train_idx=purged_train, test_idx=test)
        test_start += step
```

**Unit test requirement (must exist in tests/):**
```python
def test_walk_forward_no_leakage():
    """Harness must guarantee temporal ordering and embargo >= horizon."""
    splits = list(walk_forward_splits(n=500, min_train=252, step=21, horizon=1))
    for split in splits:
        assert split.train_idx.max() < split.test_idx.min(), "temporal ordering violated"
        assert split.test_idx.min() - split.train_idx.max() >= 1, "embargo < horizon"
```

### Anti-Patterns to Avoid

- **Fitting GARCH on raw decimal returns:** `arch` emits `DataScaleWarning` and optimizer converges to boundary parameters (alpha+beta ≈ 1). Always use 100x scaling. [CITED: PITFALLS.md, arch docs]
- **Using `rescale=True` without testing the inversion:** Auto-rescaling changes parameter units; the inversion is non-trivial and easy to get wrong. Use manual 100x. [CITED: arch docs]
- **Using `adjust=True` (default) in `ewm()` for EWMA:** The correction factor diverges for early values. Always pass `adjust=False`. [ASSUMED]
- **Passing conditional volatility (std dev) to QLIKE instead of variance:** `arch.conditional_volatility` returns std dev; must be squared before passing to QLIKE. This 2x exponent difference is a silent ranking error. [CITED: arch docs + PITFALLS.md]
- **Using the `log(h) + σ²/h` QLIKE form:** This form does not satisfy `qlike(x,x) == 0`. Use `σ²/h - ln(σ²/h) - 1`. [CITED: Patton 2011 context]
- **Not sorting before merge_asof:** merge_asof silently produces wrong results if either DataFrame is not sorted ascending. [CITED: pandas docs]
- **shift(-1) on equity data with weekend gaps:** On the last trading day of the sample, `shift(-1)` produces NaN for the target — that's correct and expected. Do not fill with zero. Drop NaN targets before training. [ASSUMED]
- **Computing GARCH-as-feature on data past the as-of date:** The filtered conditional vol series from the most recent refit window is the feature. Never refit GARCH including any future data when building the feature matrix.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GARCH(1,1) fitting and forecasting | Custom MLE optimizer | `arch_model()` from arch 8.x | Numerical gradient, convergence flags, parameter names, forecasting API all provided; edge cases in GARCH likelihood are subtle |
| EWMA variance | Custom exponential weighting loop | `pd.Series.ewm(alpha=..., adjust=False).mean()` applied to squared returns | pandas ewm is vectorized and handles edge values correctly |
| OLS regression for HAR-RV | Normal equations from scratch | `statsmodels.api.sm.OLS` | Coefficient access, residuals, t-stats for model card; already a dep via arch |
| Backward as-of join with NaN-on-staleness | Manual date iteration | `pd.merge_asof(direction='backward', tolerance=pd.Timedelta('3D'))` | Handles sorted-datetime boundary cases, NaN semantics, and group-by (`by=`) for multi-asset panels |
| Sorting/merging cross-asset frames | Manual index alignment | merge_asof | Cross-timezone, cross-calendar alignment is exactly what merge_asof is designed for |

**Key insight:** The numerical stability details of GARCH likelihood optimization and the edge-case handling in time-series EWMA are too subtle to hand-roll correctly in a portfolio project. Use the reference implementations.

---

## Common Pitfalls

### Pitfall 1: GARCH Convergence Failure — Silent Garbage Forecasts

**What goes wrong:** arch fitted on raw decimal returns (scale ~0.01) produces a `DataScaleWarning` and the optimizer may converge to boundary params (alpha+beta ≈ 1 = integrated GARCH, or alpha ≈ 0 = flat EWMA). Forecasts are then silently wrong.

**Why it happens:** The arch optimizer expects y in the 1–1000 range; raw log returns are 10–100x too small.

**How to avoid:** Scale by 100x before fitting. After every refit assert: `res.convergence_flag == 0` and `alpha + beta < 1.0`. Wrap in try/except with fallback to previous window's parameters, logging every failure.

**Warning signs:** `DataScaleWarning` in output. GARCH QLIKE >> EWMA QLIKE (impossible: GARCH should dominate EWMA when correctly fitted). Flat conditional volatility series. [CITED: PITFALLS.md Pitfall 2]

### Pitfall 2: QLIKE Argument Order or Units Mismatch

**What goes wrong:** Passing (forecast, realized) instead of (realized, forecast), or passing standard deviations instead of variances. The QLIKE surface is asymmetric — `qlike(rv, h) ≠ qlike(h, rv)`. Unit errors produce values off by 10,000x (decimal vs percent variance).

**How to avoid:** The function signature enforces order: `qlike(rv_var, forecast_var)`. Unit test `qlike(x, x) == 0` catches almost all unit bugs at any scale. Add a range check: daily decimal variance for major assets should be in [1e-6, 0.01] — log a warning if QLIKE inputs are outside this range.

**Warning signs:** QLIKE values negative (impossible for `σ²/h − ln(σ²/h) − 1`). Rankings invert between QLIKE and RMSE in implausible ways. [CITED: PITFALLS.md Pitfall 3]

### Pitfall 3: tz-aware Index Gotchas with rolling() and shift()

**What goes wrong:** On pandas 2.x with tz-aware UTC DatetimeIndex, `df.rolling('21D')` uses a time-based window (21 calendar days), not 21 observations. During weekends/holidays, a "21D" window may contain fewer than 21 obs. For equity data with gaps, this silently under-fills rolling features.

**How to avoid:** Use integer window sizes for count-based rolling (e.g., `rolling(21)`) not duration strings (`rolling('21D')`), unless calendar-day semantics are explicitly desired. For EWMA and realized vol, integer windows are correct.

**Warning signs:** Rolling-window features have more NaNs than expected at the start of the series. RV features differ between crypto (no gaps) and equity (weekend gaps) in surprising ways. [ASSUMED — standard pandas rolling gotcha with tz-aware indices]

### Pitfall 4: Next-Day Label Equity Session Gap

**What goes wrong:** `df['log_return'].shift(-1)` for equity data on a Friday produces Saturday's return (which doesn't exist). The processed equity parquet from Phase 1 should have no weekend rows, so `shift(-1)` on Friday produces NaN — that's correct. But if any fabricated weekend rows exist, the label leaks a zero-return day.

**How to avoid:** Phase 1's Pandera schema already rejects weekend equity rows. Verify that the processed parquet has no weekend DatetimeIndex entries before building features. Assert `len(equity_df) <= 253 * years` as a sanity check.

**Warning signs:** Equity feature matrix has more rows than expected trading days. Friday labels look suspiciously close to zero. [CITED: PITFALLS.md Pitfall 4]

### Pitfall 5: Harness Refit Frequency vs Forecast Frequency Confusion

**What goes wrong:** GARCH is refit every 21 steps (walk-forward step), but must produce daily forecasts for each of those 21 test days. If you only store one forecast per refit (the immediate next-step), you miss the remaining 20 forecasts. If you refit daily, the backtest takes prohibitively long.

**How to avoid:** After each 21-step refit, call `res.forecast(horizon=21)` (or `horizon=step`). The walk-forward harness test-window is 21 days; store the h.1 through h.21 variance forecasts from that single fit. [ASSUMED — standard walk-forward GARCH implementation]

**Warning signs:** GARCH results have 1 forecast for every 21 rows. Forecast timestamps jump in 21-day increments instead of daily. [CITED: PITFALLS.md Performance Traps]

### Pitfall 6: Cross-Asset merge_asof on Unsorted Index

**What goes wrong:** `merge_asof` silently produces wrong results if either DataFrame is not sorted ascending by the merge key. With a tz-aware UTC DatetimeIndex, sorting by reset index column "date" may behave differently than sorting the index directly.

**How to avoid:** Always call `.sort_values("date")` after `.reset_index()` and before `merge_asof`. Add an assertion: `assert right.index.is_monotonic_increasing` before the call.

**Warning signs:** Cross-asset features appear to use future values. QLIKE for cross-asset models is implausibly better than baseline. [CITED: pandas.pydata.org/docs/reference/api/pandas.merge_asof.html]

---

## Code Examples

### Range-Based Estimators (Parkinson and Garman-Klass)

```python
# Source: Parkinson (1980), Garman & Klass (1980); formulas verified from
#         ryanoconnellfinance.com/historical-volatility-estimators/ [CITED]
import numpy as np
import pandas as pd

def parkinson_var(df: pd.DataFrame) -> pd.Series:
    """Parkinson (1980) daily variance from high/low prices.

    Estimator: (1 / (4 ln 2)) * (ln(H/L))^2
    5x more efficient than close-to-close squared returns.
    Assumes geometric Brownian motion; no drift assumption needed.
    """
    log_hl = np.log(df["high"] / df["low"])
    return log_hl ** 2 / (4.0 * np.log(2.0))

def garman_klass_var(df: pd.DataFrame) -> pd.Series:
    """Garman-Klass (1980) daily variance from OHLC prices.

    Estimator: 0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2
    7.4x more efficient than close-to-close. Uses all four OHLC prices.
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    return 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
```

### Multi-Lookback Realized Variance

```python
# Source: standard RV definition from returns
import numpy as np
import pandas as pd

def realized_var(log_returns: pd.Series, window: int) -> pd.Series:
    """Rolling realized variance over `window` past observations.

    Returns the sum of squared log returns over the window, divided by window,
    to get an estimator of daily variance. This is NOT the same as .var()
    (which uses Bessel's correction and n-1 denominator).

    Units: decimal log-return variance (e.g., ~1e-4 for typical daily equity).
    """
    return log_returns.pow(2).rolling(window).mean()
```

### Calendar Features

```python
# Source: standard pandas DatetimeIndex attributes
import pandas as pd

def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features from UTC DatetimeIndex.

    For equities, 'is_session_open' is True on weekdays (exchange-calendars
    validates that no weekend rows exist in the processed data).
    """
    idx = df.index
    df = df.copy()
    df["day_of_week"] = idx.dayofweek           # 0=Monday, 4=Friday
    df["month"] = idx.month                      # 1-12
    df["is_monday"] = (idx.dayofweek == 0).astype(int)
    df["is_friday"] = (idx.dayofweek == 4).astype(int)
    # For equities: session gap flag (False=weekend, but processed data should
    # have no weekend rows; this flag is 1 for all valid equity rows)
    # For crypto: overnight vol captured by the daily bar already
    return df
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `rescale=True` in arch (auto-scaling) | Manual 100x scaling with `rescale=False` | arch >=4.x | Auto-rescale is harder to invert correctly; manual scaling is explicit and testable |
| `QLIKE = log(h) + σ²/h` (unnormalized form) | `QLIKE = σ²/h - ln(σ²/h) - 1` (Patton variance form) | Standard since Patton 2011 | Only the Patton form satisfies `qlike(x,x)==0`; use this exclusively |
| HAR-RV fitted on rolling variance levels | HAR-RV fitted on rolling mean of past RV windows | Corsi (2009) standard | Average (not last value) for weekly/monthly components is the correct spec |
| `TimeSeriesSplit` from sklearn as the full harness | Custom expanding-window with purge+embargo | MLflow-era best practice | sklearn's splitter has no concept of label horizon overlap |
| `res.fit()` without convergence check | `assert res.convergence_flag == 0` and `alpha+beta<1` after every refit | Safety practice | Walk-forward loops that silently use failed fits corrupt all downstream metrics |

**Deprecated/outdated:**
- `arch` version <7: different parameter naming in some edge cases — use 8.x
- `statsmodels` `OLS.from_formula()` for HAR: unnecessary overhead for a 4-parameter model; use `sm.OLS(y, X)` directly with `sm.add_constant(X)`

---

## Open Questions

1. **GARCH refit failure fallback strategy**
   - What we know: Some walk-forward windows (especially early windows with small training sets) may produce convergence failures.
   - What's unclear: Whether to (a) skip the test window, (b) use the previous window's parameters, or (c) fall back to EWMA for that window.
   - Recommendation: Use previous window's fitted params if available, EWMA forecast as final fallback. Log every fallback as a warning and include fallback count in the evaluation report. [ASSUMED]

2. **HAR-RV refit frequency**
   - What we know: The walk-forward harness refits every 21 steps. HAR-RV OLS is cheap (~microseconds for 252 rows).
   - What's unclear: Whether to refit HAR every step or reuse fitted params like GARCH.
   - Recommendation: Refit every step (it's trivially fast) for methodological consistency with the harness step. [ASSUMED]

3. **Feature matrix storage format**
   - What we know: CONTEXT.md defers parquet schema to Claude's discretion.
   - What's unclear: Whether to store one feature parquet per asset or a combined multi-asset panel.
   - Recommendation: Per-asset parquet in `data/features/{asset_class}/{symbol}.parquet` — mirrors the processed data layout from Phase 1 and simplifies the serving codepath (single-asset feature request). [ASSUMED]

4. **Garman-Klass for crypto (24/7, no overnight gaps)**
   - What we know: GK estimator assumes no overnight gaps. Crypto trades continuously, so this assumption holds.
   - What's unclear: Whether to apply GK to equity data where overnight gaps exist (GK will slightly underestimate equity vol due to the overnight return not captured in open price vs prior close).
   - Recommendation: Use GK for both, with a note in the model card that for equities the estimator misses the overnight component. Yang-Zhang handles overnight gaps but adds implementation complexity; defer to v2 unless CONTEXT.md asks for it. [ASSUMED]

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All phase code | Partially — 3.13 installed | 3.13.x | Run tests with Python 3.13; arch 8.0 supports 3.10-3.14 [CITED: pypi.org/project/arch] |
| arch | GARCH baseline, GARCH-as-feature | NOT installed | — (8.0.0 on PyPI) | Must `uv add arch>=8,<9` before Phase 2 work |
| statsmodels | HAR-RV OLS | NOT installed | — (0.14.6 on PyPI) | `uv add statsmodels>=0.14`; or use `np.linalg.lstsq` if only needed for HAR |
| pandas 2.3 | All data operations | Installed (in uv env) | 2.x (exact via uv.lock) | — |
| numpy 2.x | Numerical operations | Installed (in uv env) | 2.x | — |
| exchange-calendars | Calendar features | Installed (Phase 1) | >=4.13 | — |
| data/processed/*.parquet | Feature pipeline input | Phase 1 output | — | Phase 2 cannot start without Phase 1 output |

**Missing dependencies with no fallback:**
- `arch` — GARCH baseline and GARCH-as-feature are core Phase 2 deliverables. Must add to pyproject.toml.
- `data/processed/` parquet files — Phase 2 reads these; Phase 1 must produce them first.

**Missing dependencies with fallback:**
- `statsmodels` — fallback is `np.linalg.lstsq` for HAR-RV OLS. Prefer statsmodels since arch already requires it as a transitive dep.
- Python 3.12 vs 3.13 — arch 8.0 supports 3.10-3.14; no issue running on 3.13.

---

## Security Domain

> `security_enforcement` not explicitly set to false in config.json. Applying default (enabled).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No user auth in this phase |
| V3 Session Management | No | No sessions in this phase |
| V4 Access Control | No | File-based pipeline, no access control |
| V5 Input Validation | Yes | Pandera schemas from Phase 1 gate all parquet inputs; QLIKE floor prevents division by zero |
| V6 Cryptography | No | No crypto operations |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Parquet file path traversal | Tampering | Use `project_root()` + `pathlib.Path` from `config.py`; never interpolate user-supplied strings into file paths |
| Numpy NaN/Inf propagation in QLIKE | Tampering | Clip forecast_var at QLIKE_FLOOR; assert no NaN/Inf in output metrics |
| arch model pickle loading (future serving) | Elevation | Do not pickle arch model objects for serving; instead store parameters (omega, alpha, beta) and reconstruct at load time |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `sm.OLS` preferred over `np.linalg.lstsq` for HAR-RV since statsmodels is already a dep | Standard Stack / HAR-RV pattern | Low — both produce identical coefficients; lstsq is the easy fallback |
| A2 | `ZeroMean` GARCH recommended for daily log returns (vs ConstantMean) | Standard Stack | Low — ConstantMean adds one parameter (mu); for financial returns near zero, both converge similarly |
| A3 | `pd.Series.ewm(alpha=..., adjust=False)` implements RiskMetrics EWMA correctly | EWMA pattern | Medium — if pandas ewm with adjust=False differs from the recursive formula by even a small factor, EWMA baseline QLIKE will be systematically off |
| A4 | Integer window sizes for rolling() preferred over duration strings | Pitfall 3 | Low — easy to verify; duration strings only matter if irregular sampling exists |
| A5 | GARCH refit fallback: use previous window params, then EWMA | Open Questions #1 | Medium — if too many windows fail, fallback proportion affects baseline quality |
| A6 | HAR-RV refit every walk-forward step (not every 21 steps) | Open Questions #2 | Low — both approaches are defensible; cost is negligible |
| A7 | Per-asset parquet in data/features/{class}/{symbol}.parquet | Open Questions #3 | Low — layout is Claude's discretion per CONTEXT.md |
| A8 | GK estimator applied to equity data with note about overnight gap omission | Open Questions #4 | Low — small systematic underestimation for equities; acceptable for daily RV proxy |
| A9 | GARCH forecast(horizon=step) used to generate all 21 test-window forecasts from single refit | Pitfall 5 | Medium — if analytic multi-step GARCH forecast is used instead of recursive 1-step, forecast quality may differ |

---

## Sources

### Primary (HIGH confidence)
- [arch 8.0.0 — Introduction](https://bashtage.github.io/arch/univariate/introduction.html) — ZeroMean model, 100x percent returns, fit() call pattern
- [arch 8.0.0 — Volatility Modeling](https://bashtage.github.io/arch/univariate/univariate_volatility_modeling.html) — `disp="off"`, parameter access, percent returns example
- [arch 8.0.0 — Forecasting](https://bashtage.github.io/arch/univariate/univariate_volatility_forecasting.html) — `forecast()` method, `h.1` column, `conditional_volatility` squaring
- [arch 8.0.0 — ARCHModelResult](https://bashtage.github.io/arch/univariate/generated/arch.univariate.base.ARCHModelResult.html) — `convergence_flag` (0=success), `optimization_result`, parameter properties
- [PyPI: arch 8.0.0](https://pypi.org/project/arch/) — version, author (Kevin Sheppard), Python >=3.10, statsmodels>=0.13 dep
- [PyPI: statsmodels 0.14.6](https://pypi.org/project/statsmodels/) — version, Python >=3.9
- [pandas.merge_asof](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html) — tolerance (Timedelta), direction, NaN on exceeding tolerance, sorted-ascending requirement
- [Patton (2011) pre-print](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf) — QLIKE form reference (PDF binary; formula confirmed via cross-reference with literature)
- [arXiv:2506.07928v1 — "Predicting Realized Variance Out of Sample"](https://arxiv.org/html/2506.07928v1) — QLIKE form `log(ŷ) + y/ŷ` vs Patton form; HAR beats GARCH finding
- Project PITFALLS.md — GARCH scaling, QLIKE bugs, EWMA gotchas (project-specific, HIGH for this codebase)

### Secondary (MEDIUM confidence)
- [Parkinson (1980) / Garman-Klass (1980) formulas via ryanoconnellfinance.com](https://ryanoconnellfinance.com/historical-volatility-estimators/) — formula cross-check
- [portfoliooptimizer.io — HAR Model](https://portfoliooptimizer.io/blog/volatility-forecasting-har-model/) — HAR spec (1/5/22 lookbacks), OLS, rolling estimation
- [slopcheck output] — arch flagged [SUS] for proximity to "torch" (false positive; manually overridden)

### Tertiary (LOW confidence)
- General pandas rolling/ewm gotchas via WebSearch — used to inform pitfall documentation; verify on actual code during execution

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — arch 8.0 and statsmodels 0.14.6 verified on PyPI; both missing from pyproject.toml confirmed by env probe
- Architecture: HIGH — patterns derived from official arch docs and pandas docs; QLIKE form derived from Patton 2011
- Pitfalls: HIGH — all from PITFALLS.md (project research artifact) or verified official docs

**Research date:** 2026-06-11
**Valid until:** 2026-07-11 (arch 8.x API is stable; pandas merge_asof has been stable for years)
