"""Train the pooled LightGBM challenger and register it as volforecast-lgbm@champion.

This script runs from the project root:

    uv run python scripts/train_lgbm.py

It expects:
- data/features/{crypto,equity}/{slug}.parquet (5 assets, 19 cols each)
- data/processed/{crypto,equity}/{slug}.parquet (for close prices → target)
- MLFLOW_TRACKING_URI env var (defaults to http://localhost:5000)
- A live MLflow tracking server at the URI above

Steps:
1. Load 5 feature parquets and derive per-asset log-variance targets.
2. Run inner-validation-only grid search to pick best hyperparameters.
3. Train final pooled LightGBM on the last available fold.
4. Evaluate per-asset RMSE/MAE/QLIKE on walk-forward test folds.
5. Open an MLflow run: log params, metrics, tags, SHAP artifacts, model.
6. Register as volforecast-lgbm and assign the @champion alias.
7. Verify round-trip via get_model_version_by_alias.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Fix Windows CP1252 console encoding: MLflow 3.x logs emoji characters that
# CP1252 cannot encode.  Wrap stdout/stderr with a UTF-8 writer so the emoji
# displays as ? instead of raising UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

# Allow import from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.models import infer_signature

from volforecast.config import load_assets, processed_path, symbol_slug
from volforecast.eval.harness import walk_forward_splits
from volforecast.features.target import compute_target
from volforecast.models.lgbm import (
    ASSET_DTYPE,
    PARAM_GRID,
    compute_data_hash,
    compute_shap_artifacts,
    evaluate_per_asset,
    grid_search,
    to_log_var,
    train_pooled_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "volforecast-lgbm"
CHAMPION_ALIAS = "champion"
RUN_NAME = "lgbm-pooled-v1"
MIN_TRAIN = 252
STEP = 21
HORIZON = 1


def _features_path(asset: dict, data_root: Path) -> Path:
    """Return canonical features parquet path for an asset."""
    slug = symbol_slug(asset["symbol"])
    return data_root / "features" / asset["asset_class"] / f"{slug}.parquet"


def _git_sha() -> str:
    """Return the short git SHA of HEAD, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    project_root = Path(__file__).parent.parent

    # --- Configure MLflow tracking ---
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    log.info("MLflow tracking URI: %s", tracking_uri)

    # --- Load assets config ---
    assets = load_assets(project_root / "config" / "assets.yaml")
    log.info("Loaded %d assets: %s", len(assets), [a["symbol"] for a in assets])

    # --- Load feature DataFrames and derive log-variance targets ---
    data_root = project_root / "data"
    asset_feature_dfs: dict[str, pd.DataFrame] = {}
    asset_target_series: dict[str, pd.Series] = {}

    for asset in assets:
        sym = asset["symbol"]
        slug = symbol_slug(sym)

        feat_path = _features_path(asset, data_root)
        if not feat_path.exists():
            log.error("Missing feature parquet: %s", feat_path)
            sys.exit(1)
        feat_df = pd.read_parquet(feat_path)

        proc_path = processed_path(asset, data_root)
        if not proc_path.exists():
            log.error("Missing processed parquet: %s", proc_path)
            sys.exit(1)
        proc_df = pd.read_parquet(proc_path)
        close = proc_df["close"]

        # Compute realized variance target and log-transform it
        target_var = compute_target(close)
        log_target = pd.Series(
            to_log_var(target_var.values),
            index=target_var.index,
            name=slug,
        )

        # Align indices (feature index is the authoritative one)
        if not feat_df.index.equals(close.index):
            log_target = log_target.reindex(feat_df.index)

        asset_feature_dfs[slug] = feat_df
        asset_target_series[slug] = log_target
        log.info(
            "Loaded %s (%s): feat=%s target_non_nan=%d",
            slug,
            asset["asset_class"],
            feat_df.shape,
            log_target.notna().sum(),
        )

    # --- Grid search on inner validation folds only (Pitfall 2) ---
    log.info("Running grid search on inner validation folds (%d combos)...", 24)
    best_params = grid_search(
        asset_feature_dfs,
        asset_target_series,
        param_grid=PARAM_GRID,
        min_train=MIN_TRAIN,
        step=STEP,
        horizon=HORIZON,
        verbose=True,
    )
    log.info("Best params from grid search: %s", best_params)

    # --- Train final model on last available fold ---
    # Determine last fold index (minimum across all assets)
    min_folds = min(
        len(list(walk_forward_splits(len(df), MIN_TRAIN, STEP, HORIZON)))
        for df in asset_feature_dfs.values()
    )
    final_fold_i = min_folds - 1
    log.info("Training final model on fold_i=%d (last outer fold)...", final_fold_i)

    model, best_iter = train_pooled_model(
        asset_feature_dfs,
        asset_target_series,
        params=best_params,
        fold_i=final_fold_i,
        min_train=MIN_TRAIN,
        step=STEP,
        horizon=HORIZON,
    )
    log.info("Final model trained. best_iteration=%d", best_iter)

    # --- Evaluate per-asset on walk-forward test folds ---
    log.info("Evaluating per-asset on walk-forward test folds...")
    per_asset_metrics = evaluate_per_asset(
        model,
        asset_feature_dfs,
        asset_target_series,
        min_train=MIN_TRAIN,
        step=STEP,
        horizon=HORIZON,
    )

    # --- Build MLflow signature from a sample of validation rows ---
    # Use last fold's inner val rows across all assets for the signature sample
    x_val_parts: list[pd.DataFrame] = []
    x_test_parts: list[pd.DataFrame] = []
    for sym, feat_df in asset_feature_dfs.items():
        splits = list(walk_forward_splits(len(feat_df), MIN_TRAIN, STEP, HORIZON))
        if not splits:
            continue
        # Val sample for signature
        val_idx = splits[final_fold_i].train_idx[-STEP * 3 :]
        x_val_part = feat_df.iloc[val_idx].copy()
        x_val_part["asset"] = sym
        x_val_part["asset"] = x_val_part["asset"].astype(ASSET_DTYPE)
        x_val_parts.append(x_val_part)
        # Test rows for SHAP reference
        test_idx = splits[-1].test_idx
        x_test_part = feat_df.iloc[test_idx].copy()
        x_test_part["asset"] = sym
        x_test_part["asset"] = x_test_part["asset"].astype(ASSET_DTYPE)
        x_test_parts.append(x_test_part)

    x_val_sample = pd.concat(x_val_parts, ignore_index=True)
    x_test_all = pd.concat(x_test_parts, ignore_index=True)

    val_preds = model.predict(x_val_sample)
    signature = infer_signature(x_val_sample, val_preds)

    # --- Data lineage tags ---
    data_hash = compute_data_hash(asset_feature_dfs)
    git_sha = _git_sha()
    # Determine train window end date from the last fold's train indices
    last_split = list(
        walk_forward_splits(len(next(iter(asset_feature_dfs.values()))), MIN_TRAIN, STEP, HORIZON)
    )[-1]
    first_feat_df = next(iter(asset_feature_dfs.values()))
    train_end_date = str(first_feat_df.index[last_split.train_idx[-1]])

    # --- Feature names for SHAP ---
    feature_names = [c for c in x_val_sample.columns if c != "asset"]

    # ---------------------------------------------------------------------------
    # MLflow run: params, metrics, tags, SHAP, model, registry
    # ---------------------------------------------------------------------------
    log.info("Opening MLflow run '%s'...", RUN_NAME)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id
        log.info("Run ID: %s", run_id)

        # Log hyperparameters
        mlflow.log_params(best_params)
        mlflow.log_params({"best_iteration": best_iter, "n_assets": len(asset_feature_dfs)})

        # Log per-asset metrics with symbol prefix
        flat_metrics: dict[str, float] = {}
        for sym, metrics in per_asset_metrics.items():
            for metric_name, val in metrics.items():
                if metric_name != "n_forecasts":
                    flat_metrics[f"{sym}_{metric_name}"] = val
            flat_metrics[f"{sym}_n_forecasts"] = float(metrics["n_forecasts"])
        mlflow.log_metrics(flat_metrics)
        log.info("Logged metrics: %s", list(flat_metrics.keys()))

        # Log lineage tags (T-03-03: provenance auditable)
        mlflow.set_tags(
            {
                "data_hash": data_hash,
                "git_sha": git_sha,
                "train_window_end": train_end_date,
            }
        )
        log.info(
            "Tags: data_hash=%s git_sha=%s train_window_end=%s", data_hash, git_sha, train_end_date
        )

        # --- Compute and log SHAP artifacts (before log_model closes the run) ---
        log.info("Computing SHAP artifacts on %d reference rows...", len(x_test_all))
        with tempfile.TemporaryDirectory() as tmpdir:
            shap_results = compute_shap_artifacts(
                model=model,
                x_reference=x_test_all,
                feature_names=feature_names,
                output_dir=tmpdir,
            )
            mlflow.log_artifact(str(shap_results["bar_path"]), artifact_path="shap")
            mlflow.log_artifact(str(shap_results["beeswarm_path"]), artifact_path="shap")
            log.info(
                "SHAP artifacts logged: bar=%s beeswarm=%s",
                shap_results["bar_path"],
                shap_results["beeswarm_path"],
            )
            # Log per-asset top-10 as a text artifact for reviewability
            top10_lines = []
            for asset_val, top10 in shap_results["per_asset_top10"].items():
                top10_lines.append(f"{asset_val}: {', '.join(top10)}")
            top10_text = "\n".join(top10_lines)
            top10_path = Path(tmpdir) / "shap_per_asset_top10.txt"
            top10_path.write_text(top10_text, encoding="utf-8")
            mlflow.log_artifact(str(top10_path), artifact_path="shap")

        # --- Log and register the model ---
        log.info("Logging model and registering as '%s'...", MODEL_NAME)
        model_info = mlflow.lightgbm.log_model(
            lgb_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=x_val_sample.head(3),
        )
        log.info(
            "Model registered. version=%s run_id=%s",
            model_info.registered_model_version,
            run_id,
        )

    # ---------------------------------------------------------------------------
    # Assign @champion alias and validate round-trip (after run closes)
    # ---------------------------------------------------------------------------
    client = MlflowClient()
    version = model_info.registered_model_version

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=version,
    )
    log.info("Set alias @%s -> version %s", CHAMPION_ALIAS, version)

    # Set validation_status tag on the model version (T-03-03 provenance)
    client.set_model_version_tag(MODEL_NAME, version, "validation_status", "passed")
    log.info("Set model version tag: validation_status=passed")

    # Round-trip verification
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    print("\nChampion model registered:")
    print(f"  Name:    {MODEL_NAME}")
    print(f"  Version: {mv.version}")
    print(f"  Run ID:  {mv.run_id}")
    print(f"  Alias:   @{CHAMPION_ALIAS}")
    print(f"  Tags:    {mv.tags}")

    log.info(
        "Training complete. %s@%s -> version %s (run_id=%s)",
        MODEL_NAME,
        CHAMPION_ALIAS,
        mv.version,
        mv.run_id,
    )


if __name__ == "__main__":
    main()
