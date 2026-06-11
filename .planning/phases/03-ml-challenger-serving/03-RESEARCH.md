# Phase 3: ML Challenger & Serving — Research

**Researched:** 2026-06-11
**Domain:** LightGBM training + MLflow 3.x registry (aliases) + SHAP explainability + FastAPI serving in docker-compose
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**ML Training Design**
- One pooled LightGBM across all 5 assets with asset as a categorical feature; per-asset metrics still reported
- Identical walk-forward folds as Phase 2 baselines (reuse eval/harness.py splits per asset); pooled training assembles train rows from each asset's train indices only — no cross-asset temporal leakage (train cutoff respected per asset)
- Regression on log(target variance) for scale stability; predictions inverse-transformed (exp) back to variance before QLIKE/RMSE/MAE; metrics reported on variance scale with the canonical eval/metrics.py
- Modest fixed hyperparameter grid (~20 combos) selected on validation folds only (last fold(s) of train window); test folds never used for tuning
- Reproducibility: fixed seed, params + data hash logged to MLflow

**MLflow Tracking & Registry**
- Tracking server: the compose-stack MLflow at http://localhost:5000 (Postgres-backed) — set MLFLOW_TRACKING_URI
- Registry: model name `volforecast-lgbm`; alias `@champion` assigned to the initial best run; `@challenger` is Phase 4's concern
- No deprecated stages — aliases + tags only
- Regimes for the report: per-asset vol terciles (low/mid/high realized vol) and calendar year

**Serving**
- FastAPI endpoints: GET /health, GET /forecast (all assets), GET /forecast/{symbol}; responses include forecast variance + vol, model_version, alias, as_of date
- Features computed via the same `build_features()` import on local processed data — single codepath (FEAT-07)
- Model loaded from registry by alias `models:/volforecast-lgbm@champion` at startup, with manual refresh endpoint or lazy reload allowed
- Prediction log: append-only parquet at `data/predictions/predictions.parquet` with columns (timestamp_utc, asset, horizon, forecast_var, model_version, alias) — the Phase 4 monitoring contract
- Service runs in docker-compose alongside MLflow stack; Dockerfile in src/volforecast/serving or infra/

**SHAP & Report**
- TreeExplainer; global summary bar + beeswarm PNGs; per-asset top-10 feature importances; logged as MLflow artifacts under the registered run
- `reports/ml_vs_baselines.md` + CSV: LightGBM vs EWMA/GARCH/HAR-RV per asset and per regime; losses stated plainly

### Claude's Discretion
- Exact LightGBM param grid, log-variance epsilon handling, FastAPI module layout, Dockerfile base image (python:3.12-slim + libgomp1 per research), how compose mounts model/data

### Deferred Ideas (OUT OF SCOPE)
- Champion/challenger promotion gate, drift monitoring, scheduled/triggered retraining (Phase 4)
- Streamlit dashboard (Phase 5); LSTM/TFT challenger (v2); cloud deploy (v2)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MODEL-01 | LightGBM regression model trains on the feature set and is evaluated on identical walk-forward folds as the baselines | Pooled fold-assembly pattern, log-variance target, per-asset metrics using existing harness.py + metrics.py |
| MODEL-02 | All runs (params, metrics, artifacts) tracked in MLflow; models registered with version aliases (@champion), not deprecated stages | MLflow 3.13 client alias APIs verified; `mlflow.lightgbm.log_model` + `MlflowClient.set_registered_model_alias` |
| MODEL-03 | SHAP explainability artifacts produced for the registered model | SHAP 0.52 TreeExplainer with LightGBM; numpy 2 compatibility confirmed; summary_plot save pattern |
| EVAL-04 | Evaluation reports ML-vs-baseline comparison per asset and per regime, reported honestly even where ML loses | Regime definition (vol terciles + calendar year), comparison against baseline_metrics.csv baseline bar |
| SERVE-01 | FastAPI service serves next-day vol forecasts for all tracked assets from the champion model alias | FastAPI 0.136.3 + uvicorn 0.49.0; lifespan pattern; pyfunc.load_model at startup |
| SERVE-02 | Service runs in Docker via docker-compose with a health endpoint and model-version metadata in responses | Dockerfile pattern (python:3.12-slim + libgomp1); compose artifact volume mount requirement; network topology |
| SERVE-03 | Every served forecast is appended to a prediction log (timestamp, asset, horizon, forecast, model version) — the contract for monitoring | Parquet append-via-concat atomic write pattern; file lock for concurrent safety |
</phase_requirements>

---

## Summary

Phase 3 is a three-part build: **(1) pooled LightGBM training** using the Phase 2 evaluation harness and feature pipeline as-is, **(2) MLflow 3.x tracking + alias-based registry**, and **(3) a FastAPI serving container** loading the champion model from the registry. All three are tightly coupled through existing contracts — the planner must not invent new abstractions where existing ones already satisfy the requirement.

The central architectural challenge is the **artifact store access topology**: the MLflow server's entrypoint uses `--default-artifact-root /mlflow/artifacts` (NOT `--artifacts-destination`), which means clients (including the serving container) require **direct filesystem access** to the `mlflow_artifacts` named volume, not just HTTP access to the tracking server. The serving container must mount this volume read-only. This is the highest-risk infrastructure detail in the phase.

The second major technical challenge is **pooled fold assembly without leakage**: the harness operates per-asset, but pooled training must respect each asset's train-cutoff independently — rows from asset A's test fold must never appear in the combined training set even if asset B's train window encompasses those dates.

**Primary recommendation:** Build in wave order — training script first (host-side), MLflow tracking + registry second (verify alias APIs against the live compose stack), SHAP third, then the FastAPI serving container last so all model artifacts already exist when the serving Dockerfile is tested.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Feature computation | Python library (`volforecast.features.pipeline`) | — | Already implemented; FEAT-07 single-codepath invariant means training and serving both import `build_features`. No new tier. |
| Walk-forward evaluation | Python library (`volforecast.eval.harness`) | — | Already implemented and tested. LightGBM training uses the identical `walk_forward_splits` calls as baselines. |
| Model training + grid search | Python script (`scripts/train_lgbm.py`) | — | Runs on the host (or in a one-shot container). Does NOT belong in the serving container. |
| Experiment tracking + artifact storage | `mlflow-server` container (existing compose service) | Postgres container (backend) | Already running. Phase 3 adds runs to it, does not change the service. |
| Model registry + aliases | `mlflow-server` (MLflow Model Registry) | — | `MlflowClient.set_registered_model_alias` writes to the Postgres-backed registry. No new service needed. |
| SHAP artifact production | Training script (offline, at training time) | Logged to MLflow artifact store | Precomputed during training; served as static artifacts. Never computed per-request. |
| Evaluation report generation | Python script (`scripts/eval_lgbm.py` or integrated into training) | — | Generates `reports/ml_vs_baselines.md` + CSV. Runs offline. |
| Forecast serving (REST API) | `api` service container (FastAPI + uvicorn) | — | New compose service. Loads model from `models:/volforecast-lgbm@champion` at startup. |
| Prediction log persistence | `api` container writing to bind-mounted data path | — | Parquet append on the host-side `data/predictions/` path. Must be a bind mount (host needs access for Phase 4 monitoring). |

---

## Standard Stack

### Core (new additions to pyproject.toml — not yet present)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lightgbm | 4.6.0 | Pooled gradient boosted trees for vol regression | Already in STACK.md pinned matrix; fastest CPU tabular learner; native SHAP TreeExplainer support; libgomp1 needed in Docker slim [VERIFIED: pypi.org/pypi/lightgbm] |
| shap | 0.52.0 | SHAP explainability via TreeExplainer | Forces Python >=3.12, numpy>=2 — already satisfied by this stack; 0.52 resolved prior numpy-2 ImportError (issue #3700) [VERIFIED: pypi.org/pypi/shap] |
| fastapi | 0.136.3 | REST inference service | Standard Python serving layer; requires pydantic>=2.9 (stack already uses pydantic v2) [VERIFIED: pypi.org/pypi/fastapi] |
| uvicorn | 0.49.0 | ASGI server for FastAPI | Standard pairing; `uvicorn[standard]` adds httptools + uvloop performance extras [VERIFIED: pypi.org/pypi/uvicorn] |

**Packages already in pyproject.toml (confirm importable before adding):**
- `mlflow>=3.13,<4` — present
- `pandas>=2.3,<3` — present
- `numpy>=2,<3` — present
- `pyarrow>=4,<25` — present (needed for parquet prediction log)

### Already-present, phase-critical contracts

| Module | Location | What Phase 3 Uses |
|--------|----------|--------------------|
| `walk_forward_splits` | `src/volforecast/eval/harness.py` | Identical fold generation for LightGBM eval |
| `qlike`, `rmse`, `mae` | `src/volforecast/eval/metrics.py` | Scoring LightGBM forecasts on variance scale |
| `build_features` | `src/volforecast/features/pipeline.py` | Feature computation for training AND serving |
| `compute_target` | `src/volforecast/features/target.py` | Log-variance target derivation from close prices |
| `baseline_metrics.csv` | `reports/baseline_metrics.csv` | The bar LightGBM must be compared against |

**Installation (add to pyproject.toml):**
```bash
# Add to [project].dependencies in pyproject.toml:
# "lightgbm==4.6.0",
# "shap==0.52.0",
# "fastapi==0.136.3",
# "uvicorn[standard]==0.49.0",
uv sync
```

**Version verification performed:**
- lightgbm 4.6.0 — confirmed current at pypi.org/pypi/lightgbm/json [VERIFIED: PyPI registry]
- shap 0.52.0 — confirmed current at pypi.org/pypi/shap/json [VERIFIED: PyPI registry]
- fastapi 0.136.3 — confirmed current at pypi.org/pypi/fastapi/json [VERIFIED: PyPI registry]
- uvicorn 0.49.0 — confirmed current at pypi.org/pypi/uvicorn/json [VERIFIED: PyPI registry]

---

## Package Legitimacy Audit

> slopcheck was not available in this environment — all packages tagged [ASSUMED] for provenance, but registry existence and source repos verified individually.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| lightgbm | PyPI | 8+ yrs (Microsoft) | Very high | github.com/microsoft/LightGBM | [ASSUMED] | Approved — canonical Microsoft OSS project |
| shap | PyPI | 7+ yrs (lundberg/shap) | Very high | github.com/shap/shap | [ASSUMED] | Approved — canonical explainability library |
| fastapi | PyPI | 6+ yrs (tiangolo) | Very high | github.com/fastapi/fastapi | [ASSUMED] | Approved — industry standard Python web framework |
| uvicorn | PyPI | 6+ yrs (Kludex/encode) | Very high | github.com/encode/uvicorn | [ASSUMED] | Approved — standard ASGI server |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*slopcheck was unavailable. All four packages are well-established with 6+ year histories, multiple millions of weekly downloads, and official GitHub repositories from known maintainers. Risk is negligible.*

---

## Architecture Patterns

### System Architecture Diagram

```
[data/features/*.parquet]  [data/processed/*.parquet]
        |                          |
        v                          v
[scripts/train_lgbm.py (host)] ----build_features()----> [Feature matrix]
        |                                                        |
        |  walk_forward_splits() x 5 assets                     |
        |  pooled fold assembly (per-asset cutoff respected)     |
        |  log(target_var + eps) regression target               |
        |                                                        |
        |----> [~20-combo grid on val folds only]                |
        |           |                                            |
        |    best params selected                                |
        |           v                                            v
        |     [Final LightGBM train on all train folds]  <------/
        |
        |--mlflow.set_tracking_uri("http://localhost:5000")
        |--mlflow.lightgbm.log_model(lgb_model, "model", ...)
        |--MlflowClient.create_registered_model("volforecast-lgbm")
        |--MlflowClient.set_registered_model_alias(..., "champion", version)
        |
        |--[SHAP TreeExplainer] --> summary_plot PNGs
        |--mlflow.log_artifact(shap_png)
        |
        v
[MLflow tracking server :5000] <-- Postgres backend
[mlflow_artifacts named volume] <-- artifact files stored here
        |
        | (served via tracking-server artifact proxy --serve-artifacts; no api volume mount)
        v
[api container: FastAPI + uvicorn]
  startup:
    mlflow.set_tracking_uri("http://mlflow-server:5000")  # compose internal DNS
    model = mlflow.pyfunc.load_model("models:/volforecast-lgbm@champion")
    # downloads artifacts from mlflow_artifacts volume (direct FS access)
        |
  GET /health --> {status: ok, model_version: N, alias: champion}
  GET /forecast --> build_features() on latest data --> model.predict() --> exp(log_var)
  GET /forecast/{symbol} --> same, filtered to symbol
        |
        v
[data/predictions/predictions.parquet] <-- bind-mounted host path
  append: read existing + concat new rows + atomic write to .tmp then rename
```

### Recommended Project Structure (new files only)

```
src/volforecast/
├── models/
│   └── lgbm.py              # LightGBM training: fold assembly, grid search, train, eval
├── serving/
│   ├── __init__.py          # (already exists, empty)
│   ├── app.py               # FastAPI app, lifespan, routes
│   ├── schemas.py           # Pydantic v2 request/response models
│   └── prediction_log.py    # Parquet append logic (atomic write)
scripts/
└── train_lgbm.py            # Entry point: calls lgbm.py, logs to MLflow, registers
infra/
└── api/
    └── Dockerfile           # python:3.12-slim + libgomp1 + uv sync
reports/
└── ml_vs_baselines.md       # Generated by train_lgbm.py (EVAL-04)
data/
└── predictions/             # Created at runtime; prediction_log.parquet lives here
```

### Pattern 1: Pooled Fold Assembly Without Cross-Asset Leakage

**What:** Combine per-asset training rows into a single DataFrame for pooled LightGBM, without mixing any asset's test-fold rows into another asset's training context.

**When to use:** Every training fold in the walk-forward loop.

**The key invariant:** Each asset's train indices from `walk_forward_splits(n=len(asset_df), ...)` are integer positions within that asset's own DataFrame. "Asset A's test fold" means positions in A's DataFrame — those rows must be excluded from A's contribution to the pooled train set, regardless of whether the calendar dates overlap with B's train window.

```python
# Source: derived from harness.py contract + CONTEXT.md locked decision
# For a given walk-forward split index (fold_i):
#   - Per asset, generate the same split sequence used by baselines
#   - Take train_idx for that fold from each asset's split sequence
#   - Pool them; never use test_idx rows in training

import numpy as np
import pandas as pd
from volforecast.eval.harness import walk_forward_splits

def assemble_pooled_train(
    asset_feature_dfs: dict[str, pd.DataFrame],  # {symbol: features_df}
    asset_target_series: dict[str, pd.Series],   # {symbol: log_target}
    fold_i: int,
    min_train: int = 252,
    step: int = 21,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble pooled train set for fold_i without cross-asset leakage."""
    X_parts, y_parts = [], []
    for symbol, feat_df in asset_feature_dfs.items():
        splits = list(walk_forward_splits(len(feat_df), min_train, step, horizon))
        if fold_i >= len(splits):
            continue
        split = splits[fold_i]
        X_train = feat_df.iloc[split.train_idx].copy()
        X_train["asset"] = symbol  # categorical feature
        y_train = asset_target_series[symbol].iloc[split.train_idx]
        # Drop NaN targets (final horizon rows) before adding to pool
        valid = y_train.notna()
        X_parts.append(X_train[valid])
        y_parts.append(y_train[valid])
    return pd.concat(X_parts), pd.concat(y_parts)
```

**Critical detail:** The number of folds per asset varies (crypto has ~1344 test obs vs equity ~818). Loop over `min(len(splits_per_asset))` folds, or iterate on the asset with the fewest folds and skip others when they have no split at `fold_i`.

### Pattern 2: Log-Variance Target with Epsilon Floor

**What:** Transform raw variance target to log scale for training; inverse-transform predictions before scoring.

**Why log-variance:** Variance is right-skewed and spans many orders of magnitude (crypto ~5e-4, equity ~1e-4); log transform produces a near-Gaussian target. LightGBM MSE loss on log(var) ≈ a relative loss on the original scale.

**Epsilon choice:** The `QLIKE_FLOOR = 1e-10` in `metrics.py` is already the floor for zero-variance handling. For the log-transform, use `eps = 1e-10` consistently. Document the choice.

```python
# Source: derived from CONTEXT.md locked decision + eval/metrics.py contract
import numpy as np

LOG_VAR_EPS: float = 1e-10  # same as QLIKE_FLOOR in metrics.py for consistency

def to_log_var(var: np.ndarray) -> np.ndarray:
    """Transform variance to log scale for LightGBM regression target."""
    return np.log(np.maximum(var, LOG_VAR_EPS))

def from_log_var(log_var: np.ndarray) -> np.ndarray:
    """Inverse-transform log-variance predictions back to variance scale."""
    return np.exp(log_var)
    # Note: exp(E[log(X)]) != E[X] — this is the geometric mean, not arithmetic.
    # For QLIKE/RMSE reported on variance scale, this is acceptable bias.
    # Do NOT apply a lognormal bias correction (adds complexity without
    # demonstrable benefit at this dataset size).
```

**Important:** All scoring happens on the variance scale. Predictions from `from_log_var()` are passed directly to `qlike(rv_var, forecast_var)`, `rmse()`, `mae()` — same functions used by baselines. This is the EVAL-04 apples-to-apples requirement.

### Pattern 3: MLflow 3.x Tracking and Alias Assignment

**What:** Log a training run, register the model, and assign the `@champion` alias.

**Exact API (MLflow 3.13):** [VERIFIED: mlflow.org/docs/latest/ml/model-registry/workflow/]

```python
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient

MODEL_NAME = "volforecast-lgbm"
CHAMPION_ALIAS = "champion"

mlflow.set_tracking_uri("http://localhost:5000")  # or env var MLFLOW_TRACKING_URI

with mlflow.start_run(run_name="lgbm-pooled-v1") as run:
    # Log hyperparameters
    mlflow.log_params(best_params)

    # Log per-fold, per-asset metrics (prefix with asset name)
    for asset, metrics in per_asset_metrics.items():
        mlflow.log_metrics({
            f"{asset}_qlike": metrics["qlike"],
            f"{asset}_rmse": metrics["rmse"],
        })

    # Log data hash for lineage (from DVC or hash of feature parquet)
    mlflow.set_tags({
        "data_hash": feature_data_hash,
        "git_sha": git_sha,
        "train_window_end": train_end_date,
    })

    # Log the model — supports both native Booster and LGBMRegressor
    # Use registered_model_name to auto-register
    model_info = mlflow.lightgbm.log_model(
        lgb_model=trained_model,
        name="model",  # artifact path within the run
        registered_model_name=MODEL_NAME,
        signature=signature,  # infer_signature(X_val, predictions)
        input_example=X_val.head(3),
    )

# Assign @champion alias to the newly registered version
client = MlflowClient()
# model_info.registered_model_version contains the version number
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias=CHAMPION_ALIAS,
    version=model_info.registered_model_version,
)

# Verify round-trip
mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
print(f"Champion: version={mv.version}, run_id={mv.run_id}")
```

**Tag validation_status on the version (not on the registered model):**
```python
client.set_model_version_tag(MODEL_NAME, mv.version, "validation_status", "passed")
```

### Pattern 4: SHAP TreeExplainer Artifacts

**What:** Compute global SHAP values on a reference sample and log PNGs as MLflow artifacts.

**SHAP 0.52 + LightGBM 4.6 compatibility:** SHAP 0.52 explicitly resolved the `numpy.core.multiarray failed to import` ImportError that affected earlier versions with numpy 2 (GitHub issue #3700). The combination is safe. [VERIFIED: shap.readthedocs.io release notes + PyPI metadata]

```python
# Source: SHAP docs (shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
import shap
import matplotlib.pyplot as plt

def compute_shap_artifacts(
    model,           # trained LGBMRegressor or Booster
    X_reference: pd.DataFrame,  # a representative sample (e.g. all test-fold rows)
    feature_names: list[str],
    output_dir: str,
) -> dict[str, str]:
    """Compute SHAP values and save summary plots; return file paths."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_reference)
    # shap_values shape: (n_samples, n_features) for regression

    # Global bar chart (mean |SHAP|)
    plt.figure()
    shap.summary_plot(shap_values, X_reference, plot_type="bar", show=False)
    bar_path = f"{output_dir}/shap_global_bar.png"
    plt.savefig(bar_path, bbox_inches="tight", dpi=100)
    plt.close()

    # Beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values, X_reference, show=False)
    beeswarm_path = f"{output_dir}/shap_beeswarm.png"
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=100)
    plt.close()

    return {"bar": bar_path, "beeswarm": beeswarm_path}

# Inside mlflow.start_run():
shap_paths = compute_shap_artifacts(model, X_test_all, feature_names, tmpdir)
mlflow.log_artifact(shap_paths["bar"], artifact_path="shap")
mlflow.log_artifact(shap_paths["beeswarm"], artifact_path="shap")
```

**Note on per-asset SHAP:** For per-asset top-10, filter `X_reference` to rows for that asset before calling `shap_values()`. The explainer object can be reused (it wraps the same model).

**Memory note:** On ~5000 rows x 19 features, SHAP is fast (sub-second for TreeExplainer). No batching needed.

### Pattern 5: FastAPI Service with lifespan and MLflow pyfunc

**What:** Load the champion model at startup; expose /health, /forecast, /forecast/{symbol}.

**FastAPI lifespan pattern (0.136.x):** [VERIFIED: fastapi.tiangolo.com/advanced/events/]

```python
# Source: fastapi.tiangolo.com/advanced/events/
from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow.pyfunc
import mlflow

# Module-level state (populated during lifespan startup)
_model_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = "models:/volforecast-lgbm@champion"
    model = mlflow.pyfunc.load_model(model_uri)
    # Resolve the alias to get the concrete version number for response metadata
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias("volforecast-lgbm", "champion")
    _model_state["model"] = model
    _model_state["version"] = mv.version
    _model_state["alias"] = "champion"
    yield
    # --- Shutdown ---
    _model_state.clear()

app = FastAPI(lifespan=lifespan)
```

**predict() output:** `mlflow.pyfunc.load_model` wrapping a LightGBM model returns `pd.DataFrame` or `np.ndarray` from `.predict()` depending on input type. Pass a `pd.DataFrame` and expect `np.ndarray` back (one float per row). Apply `np.exp()` for the inverse log-variance transform.

**Thread safety of model.predict():** LightGBM's native Booster (and LGBMRegressor) `predict()` is thread-safe (read-only). Multiple FastAPI workers can call it concurrently without locking. [ASSUMED — widely documented behavior, not from official LightGBM docs re: Python thread safety]

### Pattern 6: Prediction Log — Atomic Parquet Append

**What:** Append each forecast batch to a parquet file without corrupting concurrent reads.

**Why atomic write:** Parquet does not support partial writes; writing directly to the target file while another process reads it risks a torn file. The pattern: write to `.tmp` then `os.replace()` (atomic on Linux/POSIX; also atomic on Windows NTFS within the same volume).

```python
# Source: derived from pyarrow parquet docs + atomic rename pattern
import pandas as pd
import os
import threading
from pathlib import Path

_log_lock = threading.Lock()  # module-level; guards against concurrent API requests

PREDICTION_LOG_PATH = Path("data/predictions/predictions.parquet")
PREDICTION_LOG_SCHEMA = ["timestamp_utc", "asset", "horizon", "forecast_var",
                          "model_version", "alias"]

def append_predictions(new_rows: pd.DataFrame) -> None:
    """Atomically append new_rows to the prediction log parquet."""
    with _log_lock:
        if PREDICTION_LOG_PATH.exists():
            existing = pd.read_parquet(PREDICTION_LOG_PATH)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            combined = new_rows
        tmp_path = PREDICTION_LOG_PATH.with_suffix(".tmp.parquet")
        combined.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, PREDICTION_LOG_PATH)  # atomic rename
```

**Concurrency note:** FastAPI with uvicorn uses async event loop + optional thread pool. The `_log_lock` guards against concurrent appends within the same process. For Phase 3 (single replica), this is sufficient. Phase 4 monitoring reads the parquet file but does not write to it.

### Pattern 7: LightGBM Categorical Feature — pandas category dtype

**What:** Mark the `asset` column as a pandas `category` dtype so LightGBM uses its native categorical split algorithm (Fisher 1958 ordering, no OHE).

**Exact API (LightGBM 4.6):** [VERIFIED: lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html]

```python
# pandas category dtype approach (preferred — no need to pass categorical_feature=)
X_train["asset"] = X_train["asset"].astype("category")
X_val["asset"] = X_val["asset"].astype("category")

model = LGBMRegressor(
    objective="regression",
    n_estimators=1000,          # large; early stopping controls actual rounds
    learning_rate=0.05,
    num_leaves=20,
    min_child_samples=50,       # critical for ~2500 rows/asset
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_lambda=1.0,
    random_state=42,
)

model.fit(
    X_train, y_train,           # y_train is log(var) — log-transformed
    eval_set=[(X_val, y_val)],  # note: eval_set still works in 4.6 (eval_X/eval_y new in 4.7)
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100),
    ],
    categorical_feature="auto",  # auto-detect pandas category dtype
)
best_iter = model.best_iteration_
```

**IMPORTANT — eval_set deprecation note:** The LightGBM docs mention `eval_X`/`eval_y` as preferred over `eval_set` starting in 4.7.0, but since the pinned version is 4.6.0, `eval_set` is the correct API to use. Do not use `eval_X`/`eval_y` until the version pin is bumped.

**Category dtype must be consistent between train and inference:** When serving, the `asset` column must be cast to `category` using the same categories. Use `pd.CategoricalDtype(categories=KNOWN_ASSETS, ordered=False)` to enforce a fixed category set.

```python
KNOWN_ASSETS = ["BTC-USD", "ETH-USD", "SPY", "AAPL", "MSFT"]
ASSET_DTYPE = pd.CategoricalDtype(categories=KNOWN_ASSETS, ordered=False)
# Applied at training time AND serving time
X["asset"] = X["asset"].astype(ASSET_DTYPE)
```

### Pattern 8: Serving Dockerfile

**What:** Minimal Dockerfile for the FastAPI serving container.

**Key requirement:** `libgomp1` must be installed before LightGBM is imported. This is the #1 failure mode for LightGBM on Debian/Ubuntu slim images — `libgomp.so.1: cannot open shared object file`. [VERIFIED: lightgbm PyPI README + STACK.md]

```dockerfile
FROM python:3.12-slim

# OpenMP runtime required by LightGBM on slim images (no OpenMP in slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv then sync dependencies (matches local dev environment exactly)
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --no-dev --frozen

COPY src/ ./src/
COPY data/features/ ./data/features/   # or mount as bind volume

ENV PYTHONPATH=/app/src
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

CMD ["uv", "run", "uvicorn", "volforecast.serving.app:app",
     "--host", "0.0.0.0", "--port", "8000"]
```

**Line ending note:** Dockerfile must use LF line endings. `.gitattributes` should already enforce this (`* text=auto eol=lf` for Dockerfiles).

### Pattern 9: docker-compose Service for the API

**What:** Add the `api` service to `infra/docker-compose.yml`.

**Critical infrastructure decision — artifact store access:**

The existing MLflow entrypoint uses `--default-artifact-root /mlflow/artifacts` WITHOUT `--serve-artifacts` or `--artifacts-destination`. This means `pyfunc.load_model()` in the serving container accesses artifacts **directly from the filesystem**, not via the tracking server HTTP proxy. [VERIFIED: mlflow.org tracking server docs — `--default-artifact-root` without `--artifacts-destination` = client needs direct FS access]

Therefore the `api` service MUST mount the `mlflow_artifacts` named volume read-only:

```yaml
api:
  build:
    context: ..
    dockerfile: infra/api/Dockerfile
  depends_on:
    mlflow-server:
      condition: service_healthy   # wait for MLflow to be ready
  ports:
    - "127.0.0.1:8000:8000"       # loopback-only; matches security posture of rest of stack
  environment:
    MLFLOW_TRACKING_URI: "http://mlflow-server:5000"
    PREDICTION_LOG_PATH: "/data/predictions/predictions.parquet"
  volumes:
    - ../data:/data                            # bind mount for features + prediction log
    # NOTE: no mlflow_artifacts mount — Plan 03-01 enables --serve-artifacts proxy mode
  restart: unless-stopped
```

**Internal network:** All compose services share the default bridge network. The `api` container reaches MLflow by hostname `mlflow-server` (Docker compose DNS). From the host, MLflow is at `localhost:5000`; the API is at `localhost:8000`.

**MLFLOW_TRACKING_URI differences:**
- Training script (host side): `http://localhost:5000`
- API container (inside compose network): `http://mlflow-server:5000`
- Both must be set via environment variable, never hardcoded in source.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Walk-forward splits | New splitter logic | `walk_forward_splits` from `eval/harness.py` | Already written, tested, and proven correct. New splitter risks subtle leakage. |
| QLIKE / RMSE / MAE | New metric implementations | `qlike`, `rmse`, `mae` from `eval/metrics.py` | Already unit-tested (`qlike(x,x)==0` verified). New implementations break apples-to-apples comparison with baselines. |
| Feature computation | Duplicate `build_features` | `build_features` from `features/pipeline.py` | FEAT-07 invariant — one codepath. Duplication is the training/serving skew pitfall. |
| Variance ↔ log-variance transforms | Inline `np.log` / `np.exp` at call sites | Centralized `to_log_var` / `from_log_var` in `models/lgbm.py` | Inline transforms scattered across files reproduce the unit-confusion pitfall (PITFALLS.md #11). |
| SHAP visualization | Custom matplotlib feature importance | `shap.summary_plot(... plot_type="bar")` and `shap.summary_plot(...)` for beeswarm | TreeExplainer handles LightGBM's internal representation correctly; hand-rolled importance misses interaction effects. |
| Model loading at runtime | Direct pickle/joblib load | `mlflow.pyfunc.load_model("models:/volforecast-lgbm@champion")` | Alias-based load decouples serving code from version numbers; required for Phase 4 alias-flip rollback. |
| HTTP server scaffolding | Custom WSGI/ASGI | FastAPI + uvicorn | Error handling, OpenAPI schema, Pydantic validation, async support are all free. |
| Parquet append | SQLite or JSON log | Pandas concat + atomic rename to `.parquet` | Parquet is the Phase 4 monitoring contract; SQLite on a bind mount has the Windows NTFS locking pitfall. |
| Hyperparameter search | Custom nested-loop CV | Simple Python dict grid over `itertools.product` | ~20 combos fit in memory; sklearn `GridSearchCV` would use random splits (FORBIDDEN). Use the walk-forward harness explicitly. |

**Key insight:** This phase's value is the MLOps wiring and honest evaluation, not framework-building. Every component that already exists (harness, metrics, features) must be reused verbatim.

---

## Common Pitfalls

### Pitfall 1: Cross-Asset Temporal Leakage in Pooled Fold Assembly

**What goes wrong:** Asset A has 1,596 rows (crypto). Asset B has 1,071 rows (equity). Fold `i` for Asset A has train_end = 800; fold `i` for Asset B has train_end = 800 too. But if the training script iterates naively over calendar dates rather than per-asset positions, rows from the equity test window can slip into the pooled training set via a date-range JOIN.

**Why it happens:** Developers index the pooled DataFrame by date and use a single cutoff date for all assets. This is wrong because crypto has rows on weekends that equity doesn't, and the fold counts diverge between assets.

**How to avoid:** Assemble per-asset train indices from their own `walk_forward_splits(n=len(asset_df))` calls. Concatenate row selections, never date-range slices. No join across assets during fold assembly.

**Warning signs:** LightGBM beats GARCH by >15% QLIKE improvement, especially on equity assets. Test performance suspiciously better than expected.

### Pitfall 2: Test Folds Used for Hyperparameter Selection

**What goes wrong:** For each combo in the grid, the developer evaluates on the final walk-forward test folds (the same ones used to report model performance), then selects the best combo. This is multiple-comparison overfitting against the holdout.

**Why it happens:** It feels natural to use the "full evaluation" for grid search. But "validation" and "test" are different.

**How to avoid:** The CONTEXT.md locked decision specifies "validation folds only (last fold(s) of train window)." This means: for each training fold in the walk-forward sequence, hold out a sub-window from the train split as the inner validation set. Concretely: for a training set ending at position T, the inner validation set = positions [T-step:T]. Grid search selects params that minimize average QLIKE on these inner validation windows. Test folds (positions [T+horizon:T+step]) are NEVER touched during grid search.

**Warning signs:** The best-performing grid combo also has the highest variance across test folds. QLIKE improvement shrinks when run on a later time slice.

### Pitfall 3: LightGBM category dtype Inconsistency Between Training and Serving

**What goes wrong:** Training uses `pd.CategoricalDtype(categories=KNOWN_ASSETS)` with 5 categories. At serving time, `build_features()` returns a DataFrame where `asset` is a plain string column. LightGBM raises `ValueError: Categorical feature must use integer encoding` or silently produces wrong predictions.

**Why it happens:** `build_features()` does not add an `asset` column — the training script adds it post-hoc. If the serving code doesn't replicate the exact dtype setup, the model gets a different input schema.

**How to avoid:** Define `KNOWN_ASSETS` and `ASSET_DTYPE = pd.CategoricalDtype(categories=KNOWN_ASSETS, ordered=False)` as constants in `models/lgbm.py`. Import and apply them in both the training script and the serving app. The model's `MLflow` signature should capture this schema.

**Warning signs:** Serving predictions all identical or NaN. LightGBM warning: "Unknown categories will be set to NaN."

### Pitfall 4: Artifact Access Failure in the Serving Container

**What goes wrong:** The serving container calls `mlflow.pyfunc.load_model("models:/volforecast-lgbm@champion")`. The tracking server resolves the URI and returns the artifact path as `/mlflow/artifacts/...` (a local filesystem path). The serving container does NOT have the `mlflow_artifacts` volume mounted, so the path doesn't exist. Error: `FileNotFoundError: /mlflow/artifacts/...`

**Why it happens:** Developers assume the tracking server proxies artifacts (it would if started with `--artifacts-destination`, but the current entrypoint uses `--default-artifact-root`). The current setup requires direct FS access.

**How to avoid (RESOLVED by Plan 03-01):** The MLflow entrypoint is migrated to `--artifacts-destination /mlflow/artifacts --serve-artifacts` (proxy mode). Clients download artifacts via HTTP through the tracking server; the `api` service does NOT mount the `mlflow_artifacts` volume. Do NOT add `mlflow_artifacts:/mlflow/artifacts:ro` to the `api` service — Plan 03-04 Task 3 is authoritative.

**Warning signs:** `mlflow.pyfunc.load_model` raising `FileNotFoundError` or `PermissionError` on startup. Container crashes immediately after printing the model URI.

### Pitfall 5: log-variance Jensen's Inequality Bias in Reported Metrics

**What goes wrong:** Training on `log(var)` and predicting `exp(model_output)` introduces a downward bias relative to the true mean (Jensen's inequality: `E[exp(X)] > exp(E[X])`). On small datasets this bias is measurable and could cause the model's QLIKE to be artificially worse than expected, not due to model quality but due to the transform.

**Why it happens:** The inverse transform `exp()` of an unbiased log-prediction is a biased variance prediction.

**How to avoid (at this project scale):** Do NOT apply a lognormal bias correction (`exp(mu + sigma^2/2)`) — it adds a second parameter to estimate and rarely improves things at daily frequency. Instead: document in the model card that predictions are geometric-mean forecasts of variance, not arithmetic-mean. The QLIKE loss is already robust to this level of calibration error. If QLIKE is substantially worse than HAR-RV despite identical features, suspect the bias and check calibration.

**Warning signs:** QLIKE degrades monotonically with no sign of improvement, even with strong regularization and large `min_child_samples`. The bias manifests as a consistent under-forecast of variance spikes.

### Pitfall 6: SHAP TreeExplainer Called on Wrong Model Object

**What goes wrong:** `shap.TreeExplainer(model)` is called on the `mlflow.pyfunc.PyFuncModel` wrapper (the object returned by `load_model`) rather than the underlying LightGBM model. TreeExplainer does not recognize pyfunc wrappers and raises `TypeError` or produces meaningless values.

**Why it happens:** If SHAP artifacts are computed after the model is loaded from MLflow (not at training time), the developer passes the pyfunc wrapper.

**How to avoid:** Always compute SHAP artifacts at training time, passing the native `LGBMRegressor` (or `Booster`) object directly — before `log_model()` is called. The CONTEXT.md decision confirms this: "logged as MLflow artifacts under the registered run" (i.e., during training, not serving).

To recover the native model from a loaded pyfunc (if needed for post-hoc analysis):
```python
native_model = loaded_pyfunc_model._model_impl.lgb_model  # internal, version-dependent
```
Avoid this pattern — compute SHAP during training.

### Pitfall 7: Prediction Log Path Not Writable from the Container

**What goes wrong:** The serving container writes to `/data/predictions/predictions.parquet` but the bind mount is either not configured (path doesn't exist in the container) or mounted read-only.

**Why it happens:** The `data/` directory is gitignored and DVC-tracked. Developers forget to add the bind mount for `data/` to the compose service.

**How to avoid:** Add `- ../data:/data` (relative to `infra/`) to the `api` service volumes. Ensure `data/predictions/` exists on the host before first run (or create it in the `append_predictions` function with `mkdir(parents=True, exist_ok=True)`).

**Warning signs:** `FileNotFoundError` or `PermissionError` on the first `/forecast` call. Predictions log never grows.

---

## Code Examples

### Hyperparameter Grid (~20 combos)

```python
# Source: derived from CONTEXT.md locked decision + PITFALLS.md #14 regularization guidance
import itertools

PARAM_GRID = {
    "num_leaves": [15, 20],
    "min_child_samples": [50, 100],
    "learning_rate": [0.05, 0.01],
    "reg_lambda": [1.0, 5.0],
    "feature_fraction": [0.8, 1.0],
}
# 2*2*2*2*2 = 32 combos; apply step=2 for leaf to keep at ~20:
# Or use: 2*2*2*2 = 16 with one pair removed, close to the ~20 target
# The exact combos are Claude's discretion per CONTEXT.md

all_combos = list(itertools.product(*PARAM_GRID.values()))
# Each combo is a tuple; zip with keys to get a dict:
param_dicts = [dict(zip(PARAM_GRID.keys(), combo)) for combo in all_combos]
```

### MLflow Model Signature Inference

```python
# Source: mlflow.org/docs/latest/python_api/mlflow.lightgbm.html
from mlflow.models import infer_signature

signature = infer_signature(
    X_val_sample,           # input DataFrame (with asset as category)
    model.predict(X_val_sample),  # predicted log-variances (float array)
)
```

### Pydantic v2 Response Schema (FastAPI)

```python
# Source: fastapi.tiangolo.com + pydantic v2 docs
from pydantic import BaseModel
from datetime import date

class AssetForecast(BaseModel):
    asset: str
    forecast_var: float          # daily decimal variance
    forecast_vol: float          # sqrt(forecast_var)
    horizon_days: int = 1
    as_of_date: date
    model_version: str
    alias: str

class ForecastResponse(BaseModel):
    forecasts: list[AssetForecast]
    generated_at: str            # ISO 8601 UTC timestamp
```

### Health Endpoint

```python
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_name": "volforecast-lgbm",
        "model_version": _model_state.get("version"),
        "alias": _model_state.get("alias"),
    }
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MLflow stages (`transition_model_version_stage`) | Model aliases (`set_registered_model_alias`) | MLflow 2.9 (deprecated) | Stages will be removed; aliases are the production API |
| `@app.on_event("startup")` | `@asynccontextmanager async def lifespan(app)` | FastAPI 0.93+ | Deprecated startup events; lifespan is the current pattern |
| `eval_set` in LGBMRegressor.fit() | `eval_X` / `eval_y` (starting 4.7) | LightGBM 4.7 (not yet released as of 4.6.0 pin) | `eval_set` still valid in 4.6.0; do NOT switch until pin bumps |
| `shap.summary_plot()` | `shap.plots.beeswarm()` + `shap.plots.bar()` | SHAP 0.40+ | Both APIs coexist in 0.52; `summary_plot` is the simpler path for PNGs |
| MLflow `mlflow models serve` CLI | Custom FastAPI service | N/A (project choice) | Custom service is deliberate portfolio surface area (STACK.md) |

**Deprecated/outdated:**
- `mlflow.lightgbm.autolog()`: Logs everything automatically but produces noisy runs. For this project, prefer explicit `log_params` + `log_metrics` + `log_model` for clarity and portfolio reviewability.
- `lgb.train()` (native API): LGBMRegressor (sklearn API) is preferred here because it integrates cleanly with MLflow's lightgbm flavor and produces a pyfunc-loadable model.

---

## Runtime State Inventory

> Phase 3 is not a rename/refactor phase. Omitted per instructions.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| MLflow server | Training + serving | ✓ (compose up) | 3.13.0 | — (blocking) |
| Postgres | MLflow backend | ✓ (compose up) | 16 | — (blocking) |
| mlflow_artifacts volume | Training artifact write + serving pyfunc load | ✓ (compose volume) | — | — (blocking; see Pitfall 4) |
| docker-compose | Running API service | ✓ (Windows 11 + Docker Desktop) | current | — |
| data/features/*.parquet | Training input | ✓ (Phase 2 output) | — | Re-run Phase 2 scripts |
| lightgbm (Python) | Training + serving | ✗ (not in pyproject.toml) | 4.6.0 target | Add to dependencies |
| shap (Python) | Training (SHAP artifacts) | ✗ (not in pyproject.toml) | 0.52.0 target | Add to dependencies |
| fastapi (Python) | Serving | ✗ (not in pyproject.toml) | 0.136.3 target | Add to dependencies |
| uvicorn (Python) | Serving | ✗ (not in pyproject.toml) | 0.49.0 target | Add to dependencies |
| libgomp1 (Linux) | LightGBM in Docker container | ✗ (not in base image) | apt package | Add RUN apt-get install in Dockerfile |

**Missing dependencies with no fallback:**
- lightgbm, shap, fastapi, uvicorn — add to `pyproject.toml` before any training or serving code runs
- libgomp1 — add to serving Dockerfile before LightGBM import

**Missing dependencies with fallback:**
- data/features/*.parquet — should exist from Phase 2; if not, `scripts/generate_features.py` regenerates them

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8 (already in `[dependency-groups].dev`) |
| Config file | none — uses pytest defaults |
| Quick run command | `uv run pytest tests/unit/ -x -q` |
| Full suite command | `uv run pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODEL-01 | Pooled fold assembly respects per-asset train cutoff | unit | `uv run pytest tests/unit/test_lgbm_folds.py -x` | ❌ Wave 0 |
| MODEL-01 | Log-variance transform round-trip (to_log_var / from_log_var) | unit | `uv run pytest tests/unit/test_lgbm_transforms.py -x` | ❌ Wave 0 |
| MODEL-02 | MLflow run contains required tags (data_hash, git_sha) | integration (skippable in CI with `--no-mlflow`) | manual / compose up | ❌ |
| MODEL-03 | SHAP shap_values shape matches (n_samples, n_features) | unit | `uv run pytest tests/unit/test_shap_artifacts.py -x` | ❌ Wave 0 |
| EVAL-04 | ml_vs_baselines report contains all 5 assets | smoke | `uv run pytest tests/smoke/test_report.py -x` | ❌ Wave 0 |
| SERVE-01 | /health returns 200 with model_version | integration | `uv run pytest tests/integration/test_api.py::test_health -x` | ❌ Wave 0 |
| SERVE-01 | /forecast returns correct schema (AssetForecast list) | integration | `uv run pytest tests/integration/test_api.py::test_forecast -x` | ❌ Wave 0 |
| SERVE-03 | Prediction log grows after each /forecast call | integration | `uv run pytest tests/integration/test_api.py::test_prediction_log -x` | ❌ Wave 0 |
| SERVE-03 | Atomic write: no corrupt parquet after concurrent appends | unit | `uv run pytest tests/unit/test_prediction_log.py -x` | ❌ Wave 0 |

**Note on integration tests:** Tests that call the FastAPI app should use FastAPI's `TestClient` (from `starlette.testclient`) with a fixture that injects a mock model via lifespan override — never start the full compose stack in CI. MLflow calls should be mocked with `unittest.mock.patch`.

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_lgbm_folds.py` — covers MODEL-01 fold assembly no-leakage invariant
- [ ] `tests/unit/test_lgbm_transforms.py` — covers MODEL-01 log-variance round-trip
- [ ] `tests/unit/test_shap_artifacts.py` — covers MODEL-03 shape check
- [ ] `tests/unit/test_prediction_log.py` — covers SERVE-03 concurrent append safety
- [ ] `tests/smoke/test_report.py` — covers EVAL-04 report completeness
- [ ] `tests/integration/test_api.py` — covers SERVE-01, SERVE-02, SERVE-03 via TestClient

---

## Security Domain

> `security_enforcement` not explicitly set to false in config — treating as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | API is localhost-only (loopback bind); no external exposure in Phase 3 |
| V3 Session Management | no | Stateless REST; no sessions |
| V4 Access Control | no | Single-tenant, local dev only |
| V5 Input Validation | yes | FastAPI + Pydantic v2 — path parameter `symbol` validated against known asset list; reject unknown symbols with 404 |
| V6 Cryptography | no | No encryption needed for local serving |
| V7 Error Handling | yes | Never expose stack traces in API responses; `detail` field in 4xx/5xx must be generic |

### Known Threat Patterns for this Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `symbol` parameter | Tampering | Validate `symbol` is in `KNOWN_ASSETS` list before any FS access |
| Pickle deserialization via `pyfunc.load_model` | Tampering | Load only from own registry; lineage tags on registered versions verify provenance |
| Prediction log disclosure | Information Disclosure | `data/predictions/` is bind-mounted inside compose; not exposed via HTTP; acceptable for Phase 3 |
| MLflow UI unauthenticated | Elevation of privilege | Port `127.0.0.1:5000` only — never `0.0.0.0:5000`; enforced in compose.yml |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | LightGBM `predict()` in LGBMRegressor is thread-safe for concurrent reads | Pattern 5 | Serving would need a threading lock around `model.predict()` calls |
| A2 | SHAP 0.52 works with LGBMRegressor (sklearn API) without needing the native Booster | Pattern 4 | Would need to call `model.booster_` to get the native Booster before passing to TreeExplainer |
| A3 | fastapi 0.136.3 / uvicorn 0.49.0 are not yet pinned in pyproject.toml; adding them will not introduce transient dependency conflicts with existing pinned packages | Standard Stack | Would need version range adjustment; unlikely given the loose pins of starlette/anyio |
| A4 | The `mlflow_artifacts` named volume path inside the training container (when run host-side) is accessible at `/mlflow/artifacts` via the compose volume mapping | Architecture Patterns | Training script run on the host needs `MLFLOW_ARTIFACT_ROOT` env var or the mlflow-server must handle artifact upload via HTTP |

**Note on A4:** The training script runs on the **host** (outside compose), sets `MLFLOW_TRACKING_URI=http://localhost:5000`, and calls `mlflow.lightgbm.log_model()`. MLflow will attempt to write artifacts. Since the artifact store is a named volume (`mlflow_artifacts`) inside the `mlflow-server` container, and `--default-artifact-root /mlflow/artifacts` is set, the client will try to write directly to `/mlflow/artifacts/` — which does NOT exist on the host. This is a concrete risk: **the training script may fail to log artifacts when run from the host.** Options:
1. Run the training script inside a container that also mounts `mlflow_artifacts`.
2. Switch the entrypoint to `--artifacts-destination /mlflow/artifacts --serve-artifacts` — then clients upload via HTTP and the volume mount is only needed by the server, not clients. (Recommended for Phase 3.)
3. Use a host-side artifact path (`MLFLOW_ARTIFACT_ROOT=./mlruns/artifacts`) and accept that artifacts are on the host, then mount that path into the `api` container.

Option 2 is the clean production pattern and eliminates the serving container's direct volume dependency. The planner should include a task to update `mlflow-entrypoint.sh` to add `--serve-artifacts` flag.

---

## Open Questions (RESOLVED)

1. **Training script execution context (artifact access — see A4)** — RESOLVED: Plan 03-01 adds `--artifacts-destination /mlflow/artifacts --serve-artifacts` to `infra/mlflow-entrypoint.sh`; clients access artifacts via the tracking-server HTTP proxy. No volume mount in `api`.
   - What we know: `--default-artifact-root` = client needs direct FS access; named volume is only accessible inside containers
   - What's unclear: Does running the training script inside a one-shot container (with the volume mounted) vs. adding `--serve-artifacts` to the MLflow entrypoint better fit this project's workflow?
   - Recommendation: Add `--serve-artifacts` to the MLflow entrypoint (update `infra/mlflow-entrypoint.sh`). This is the documented MLflow proxy mode, requires no volume mount in the `api` service, and is a clean Phase 4-ready pattern. One-line entrypoint change. Keep in scope for Phase 3 Wave 0.

2. **Number of walk-forward folds for grid search validation** — RESOLVED: last 3 inner folds (implemented in Plan 03-02 Task 1).
   - What we know: Step=21 means crypto has ~63 folds (1344/21), equity ~39 folds (818/21). The "last fold(s) of the train window" is ambiguous about how many inner validation folds to use.
   - What's unclear: Using only the last fold may be insufficient signal (noisy QLIKE estimate); using the last 5 gives a less biased estimate.
   - Recommendation: Use last 3 inner folds for grid search (each fold = 21 steps = ~1 month). 3 months of validation is enough signal; more would encroach on useful training data.

3. **Regime definition for EVAL-04 report** — RESOLVED: test-fold-only terciles, labelled at scoring time (implemented in Plan 03-03 Task 1).
   - What we know: CONTEXT.md specifies "per-asset vol terciles (low/mid/high realized vol) and calendar year"
   - What's unclear: Tercile cutoffs computed on the full evaluation window or just the test folds? Test-fold terciles are cleaner (no lookahead).
   - Recommendation: Compute terciles on the realized variance distribution of the test-fold data only. Label each test row with its regime at scoring time.

---

## Sources

### Primary (HIGH confidence)
- [mlflow.org/docs/latest/ml/model-registry/workflow/](https://mlflow.org/docs/latest/ml/model-registry/workflow/) — exact alias API (`set_registered_model_alias`, `get_model_version_by_alias`), loading via `models:/name@alias`
- [lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) — categorical_feature API, early stopping callbacks, eval_set
- [mlflow.org/docs/latest/python_api/mlflow.lightgbm.html](https://mlflow.org/docs/latest/python_api/mlflow.lightgbm.html) — `log_model` signature, LGBMRegressor support, pyfunc flavor
- [mlflow.org/docs/latest/python_api/mlflow.pyfunc.html](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) — `load_model` signature (no `env_manager`), predict output types
- [fastapi.tiangolo.com/advanced/events/](https://fastapi.tiangolo.com/advanced/events/) — lifespan pattern for model loading
- [shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) — constructor params, LightGBM support, shap_values return shape
- [mlflow.org/docs/latest/self-hosting/architecture/tracking-server/](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) — artifact proxy modes (`--default-artifact-root` vs `--artifacts-destination`)
- PyPI JSON metadata for lightgbm 4.6.0, shap 0.52.0, fastapi 0.136.3, uvicorn 0.49.0 — version confirmation
- `infra/mlflow-entrypoint.sh` (this repo) — confirmed `--default-artifact-root` without `--serve-artifacts`
- `src/volforecast/eval/harness.py`, `eval/metrics.py`, `features/pipeline.py`, `features/target.py` (this repo) — existing contracts

### Secondary (MEDIUM confidence)
- shap GitHub issue #3700 — numpy 2 compatibility resolved in 0.52
- Community pattern for `libgomp1` on Debian slim — apt install confirmed from multiple sources
- MLflow tracking documentation on artifact proxy scenarios — client direct-access requirement for `--default-artifact-root` mode

### Tertiary (LOW confidence)
- LightGBM thread safety for `predict()` — widely cited in community but not in official docs
- lognormal bias behavior at daily frequency — training-knowledge reasoning

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions verified from PyPI; existing contracts read directly from source
- Architecture: HIGH — artifact access topology verified from official MLflow docs + entrypoint script in repo
- Pitfalls: HIGH — all critical pitfalls derived from existing PITFALLS.md (HIGH confidence) + new phase-specific pitfalls from official docs
- Serving patterns: HIGH — FastAPI/uvicorn lifespan pattern from official docs; Docker libgomp1 from well-documented community pattern

**Research date:** 2026-06-11
**Valid until:** 2026-07-11 (stable stack; MLflow/FastAPI/LightGBM release cadence is slow)
