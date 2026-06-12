"""Generate ML-vs-baseline comparison report: per-asset and per-regime.

This script produces:
  - reports/ml_vs_baselines.csv  (machine-readable per-asset / per-regime metrics)
  - reports/ml_vs_baselines.md   (human-readable comparison with honest findings)

Run from the project root:

    uv run python scripts/eval_lgbm.py

Requirements:
  - data/features/{crypto,equity}/{slug}.parquet  (5 assets)
  - data/processed/{crypto,equity}/{slug}.parquet (for close prices → target + baselines)
  - MLFLOW_TRACKING_URI env var (defaults to http://localhost:5000)
  - A live MLflow tracking server with the champion model registered at
    models:/volforecast-lgbm@champion

Design:
  - The champion model is loaded as a native LGBMRegressor via mlflow.lightgbm.load_model
    for registry/lineage validation; its logged hyperparameters are then used to
    deterministically retrain ONE pooled model PER outer walk-forward fold (CR-01).
    Each fold's test window is scored only by that fold's own model — the registry
    stores only the final-fold champion, so the leak-free comparison retrains the
    earlier folds locally (random_state=42 makes the final-fold retrain reproduce
    the champion's training procedure exactly).
  - Baseline forecasts (EWMA, GARCH, HAR-RV) are recomputed on the SAME walk-forward
    test indices as LightGBM using the existing models/{ewma,garch,har_rv}.py — no new
    metric implementations.
  - The markdown rendering function ``render_report`` is a pure function (no I/O) so
    it can be tested offline without MLflow (see tests/smoke/test_report.py).

Walk-forward regime labelling:
  - For each asset, all test-fold rows are collected with their as-of date.
  - ``assign_vol_terciles`` is called on the FULL test-fold realized variance series
    for that asset (no lookahead: only test-fold data is passed).
  - ``assign_calendar_year`` labels by calendar year.

Threat model compliance:
  - T-03-07: The report contains only metrics and dates (no secrets, no credentials).
  - T-03-08: LightGBM is scored with the same eval/metrics.py + identical folds as
    baselines — no parallel metric implementation that could diverge.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Fix Windows CP1252 console encoding
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
import numpy as np
from mlflow import MlflowClient

from volforecast.config import load_assets, processed_path, symbol_slug
from volforecast.eval.harness import walk_forward_splits
from volforecast.eval.metrics import mae, qlike, rmse
from volforecast.eval.regimes import assign_calendar_year, assign_vol_terciles
from volforecast.features.estimators import log_returns, realized_var
from volforecast.features.target import HORIZON, compute_target
from volforecast.models.ewma import EWMA
from volforecast.models.garch import GARCH
from volforecast.models.har_rv import HARRV
from volforecast.models.lgbm import (
    ASSET_DTYPE,
    LOG_VAR_EPS,
    PARAM_GRID,
    from_log_var,
    resolve_fold_model,
    to_log_var,
    train_per_fold_models,
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
MIN_TRAIN = 252
STEP = 21
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _features_path(asset: dict, data_root: Path) -> Path:
    """Return canonical features parquet path for an asset."""
    slug = symbol_slug(asset["symbol"])
    return data_root / "features" / asset["asset_class"] / f"{slug}.parquet"


def _parse_param(raw: str) -> int | float:
    """Parse an MLflow-logged hyperparameter string back to int/float."""
    try:
        return int(raw)
    except ValueError:
        return float(raw)


def _collect_per_fold_rows(
    fold_models: dict,
    asset_feature_dfs: dict[str, pd.DataFrame],
    asset_target_series: dict[str, pd.Series],
    asset_processed_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Collect per-test-row predictions and baseline forecasts for all assets.

    Returns a DataFrame with columns:
      asset, date, realized_var, lgbm_var, ewma_var, garch_var, har_var

    Each row corresponds to one test-fold observation.  Realized var and
    all forecast columns are in decimal daily variance units.

    The walk-forward folds are IDENTICAL for LightGBM and all three baselines
    (T-03-08 compliance): splits are computed once per asset and shared.

    Leak-free per-fold scoring (CR-01): fold ``k``'s test window is predicted
    by ``fold_models[k]`` — the model trained at that fold's own cutoff —
    mirroring the baselines' walk-forward refit discipline.
    """
    rows: list[dict] = []

    for symbol, feat_df in asset_feature_dfs.items():
        proc_df = asset_processed_dfs[symbol]
        close = proc_df["close"]
        lr = log_returns(close)
        rv_daily = realized_var(lr, window=1)

        # Pre-compute baseline forecast paths (stateless for EWMA; walk-forward for GARCH/HAR)
        ewma_model = EWMA()
        ewma_forecasts = ewma_model.forecast_path(lr)

        garch_model = GARCH(min_train=MIN_TRAIN, step=STEP)
        log.info("  Running GARCH walk-forward for %s...", symbol)
        garch_forecasts = garch_model.forecast_path(lr)

        har_model = HARRV(min_train=MIN_TRAIN, step=STEP)
        log.info("  Running HAR-RV walk-forward for %s...", symbol)
        har_forecasts = har_model.forecast_path(rv_daily)

        # Walk forward: LightGBM predictions on test folds
        log_target = asset_target_series[symbol]

        for fold_i, split in enumerate(walk_forward_splits(len(feat_df), MIN_TRAIN, STEP, HORIZON)):
            # CR-01: score this fold ONLY with its own fold's model.
            model = resolve_fold_model(fold_models, fold_i)
            # CR-02: the model's training column order is the single source of
            # truth for the eval feature schema.
            feature_order = list(model.feature_name_)

            test_idx = split.test_idx
            test_dates = feat_df.index[test_idx]

            # LightGBM prediction (same as evaluate_per_asset).
            x_test = feat_df.iloc[test_idx].copy()
            x_test["asset"] = symbol
            x_test = x_test.reindex(columns=feature_order)
            x_test["asset"] = x_test["asset"].astype(ASSET_DTYPE)

            y_test_log = log_target.iloc[test_idx]
            valid_mask = y_test_log.notna()

            x_test_valid = x_test.loc[valid_mask]
            y_test_log_valid = y_test_log.loc[valid_mask]
            test_dates_valid = test_dates[valid_mask.values]

            if len(x_test_valid) == 0:
                continue

            preds_log = model.predict(x_test_valid, validate_features=True)
            lgbm_var_arr = from_log_var(preds_log)
            true_var_arr = from_log_var(y_test_log_valid.values)

            # Baseline forecasts aligned to same test positions
            ewma_arr = ewma_forecasts.iloc[test_idx][valid_mask].values
            garch_arr = garch_forecasts.iloc[test_idx][valid_mask].values
            har_arr = har_forecasts.iloc[test_idx][valid_mask].values

            for i in range(len(test_dates_valid)):
                rv_val = float(true_var_arr[i])
                # WR-02: drop floored near-zero realized variance.  to_log_var
                # floors zeros at LOG_VAR_EPS, so the round-trip value is
                # ~LOG_VAR_EPS > 0 — a plain `<= 0` check never fires and each
                # floored zero injects a massive QLIKE outlier.  Same filter
                # as evaluate_per_asset.
                if rv_val <= LOG_VAR_EPS * (1 + 1e-9) or not np.isfinite(rv_val):
                    continue
                rows.append(
                    {
                        "asset": symbol,
                        "date": test_dates_valid[i],
                        "realized_var": rv_val,
                        "lgbm_var": float(lgbm_var_arr[i]),
                        "ewma_var": float(ewma_arr[i]),
                        "garch_var": float(garch_arr[i]),
                        "har_var": float(har_arr[i]),
                    }
                )

    return pd.DataFrame(rows)


def _compute_metrics_for_group(group: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute QLIKE/RMSE/MAE for each model on a group of rows.

    Args:
        group: DataFrame with columns: realized_var, lgbm_var, ewma_var,
               garch_var, har_var.  All rows must be valid (no NaN, rv > 0).

    Returns:
        Dict mapping model name -> {rmse, mae, qlike, n}
    """
    rv = group["realized_var"].values

    models = {
        "LightGBM": group["lgbm_var"].values,
        "EWMA": group["ewma_var"].values,
        "GARCH": group["garch_var"].values,
        "HAR": group["har_var"].values,
    }

    results: dict[str, dict[str, float]] = {}
    for model_name, fc in models.items():
        valid = np.isfinite(fc) & (fc > 0)
        if valid.sum() == 0:
            results[model_name] = {
                "rmse": float("nan"),
                "mae": float("nan"),
                "qlike": float("nan"),
                "n": 0,
            }
            continue
        rv_v = rv[valid]
        fc_v = fc[valid]
        try:
            q = qlike(rv_v, fc_v)
        except ValueError:
            q = float("nan")
        results[model_name] = {
            "rmse": rmse(rv_v, fc_v),
            "mae": mae(rv_v, fc_v),
            "qlike": q,
            "n": int(valid.sum()),
        }
    return results


def _build_comparison_frames(all_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the per-asset and per-regime comparison DataFrames.

    Returns:
        (per_asset_df, per_regime_df)

        per_asset_df columns: asset, model, n, rmse, mae, qlike
        per_regime_df columns: asset, regime_type, regime_value, model, n, rmse, mae, qlike
    """
    per_asset_records = []
    per_regime_records = []

    for asset, asset_group in all_rows.groupby("asset"):
        # Add regime labels
        asset_group = asset_group.copy()
        asset_group["tercile"] = assign_vol_terciles(asset_group["realized_var"]).values
        asset_group["year"] = assign_calendar_year(pd.DatetimeIndex(asset_group["date"])).values

        # --- Per-asset overall ---
        metrics = _compute_metrics_for_group(asset_group)
        for model_name, m in metrics.items():
            per_asset_records.append(
                {
                    "asset": asset,
                    "model": model_name,
                    "n": m["n"],
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "qlike": m["qlike"],
                }
            )

        # --- Per tercile ---
        for tercile_val in ["low", "mid", "high"]:
            subset = asset_group[asset_group["tercile"] == tercile_val]
            if len(subset) == 0:
                continue
            metrics = _compute_metrics_for_group(subset)
            for model_name, m in metrics.items():
                per_regime_records.append(
                    {
                        "asset": asset,
                        "regime_type": "tercile",
                        "regime_value": tercile_val,
                        "model": model_name,
                        "n": m["n"],
                        "rmse": m["rmse"],
                        "mae": m["mae"],
                        "qlike": m["qlike"],
                    }
                )

        # --- Per calendar year ---
        for year_val in sorted(asset_group["year"].unique()):
            subset = asset_group[asset_group["year"] == year_val]
            if len(subset) == 0:
                continue
            metrics = _compute_metrics_for_group(subset)
            for model_name, m in metrics.items():
                per_regime_records.append(
                    {
                        "asset": asset,
                        "regime_type": "year",
                        "regime_value": str(int(year_val)),
                        "model": model_name,
                        "n": m["n"],
                        "rmse": m["rmse"],
                        "mae": m["mae"],
                        "qlike": m["qlike"],
                    }
                )

    per_asset_df = pd.DataFrame(per_asset_records)
    per_regime_df = pd.DataFrame(per_regime_records)
    return per_asset_df, per_regime_df


def render_report(per_asset_df: pd.DataFrame, per_regime_df: pd.DataFrame) -> str:
    """Render the ML-vs-baseline comparison as a Markdown string.

    This is a PURE FUNCTION — it takes already-computed metric DataFrames and
    returns the markdown string.  It has no side effects and no I/O.
    It is tested offline in tests/smoke/test_report.py without MLflow.

    Args:
        per_asset_df: Columns: asset, model, n, rmse, mae, qlike.
        per_regime_df: Columns: asset, regime_type, regime_value, model,
                       n, rmse, mae, qlike.

    Returns:
        Markdown string.
    """
    lines = []

    lines.append("# ML vs Baselines Evaluation Report\n\n")
    lines.append("**Generated by:** `scripts/eval_lgbm.py`\n\n")
    lines.append(
        "**Unit convention:** All metrics in daily decimal variance units "
        "(e.g., ~1e-4 for typical daily equity).  QLIKE = 0 at a perfect forecast "
        "(Patton 2011 variance form).  Lower is better for all metrics.\n\n"
    )
    lines.append(
        "**Evaluation protocol:** Walk-forward expanding window (min_train=252, step=21).  "
        "LightGBM and all three baselines scored on **identical** folds per asset.  "
        "Vol terciles (low/mid/high) computed on test-fold realized variance only "
        "(no lookahead — boundaries derived from the same fold's realizations).\n\n"
    )
    lines.append("---\n\n")

    # -----------------------------------------------------------------------
    # Section 1: Per-asset overall
    # -----------------------------------------------------------------------
    lines.append("## Section 1: Per-Asset Overall Comparison\n\n")
    lines.append(
        "| Asset | Model | N | RMSE | MAE | QLIKE |\n|-------|-------|---|------|-----|-------|\n"
    )
    for asset in sorted(per_asset_df["asset"].unique()):
        asset_rows = per_asset_df[per_asset_df["asset"] == asset]
        for _, row in asset_rows.iterrows():
            lines.append(
                f"| {row['asset']} | {row['model']} | {row['n']:,} "
                f"| {row['rmse']:.6e} | {row['mae']:.6e} | {row['qlike']:.6f} |\n"
            )
    lines.append("\n")

    # -----------------------------------------------------------------------
    # Section 2: Per vol-tercile
    # -----------------------------------------------------------------------
    lines.append("## Section 2: Per-Vol-Tercile Breakdown\n\n")
    lines.append(
        "_Tercile labels (low/mid/high) are computed on each asset's TEST-FOLD "
        "realized variance only — no lookahead into training data._\n\n"
    )

    tercile_df = per_regime_df[per_regime_df["regime_type"] == "tercile"]
    for asset in sorted(tercile_df["asset"].unique()):
        lines.append(f"### {asset} — By Volatility Tercile\n\n")
        lines.append(
            "| Tercile | Model | N | RMSE | MAE | QLIKE |\n"
            "|---------|-------|---|------|-----|-------|\n"
        )
        for tercile_val in ["low", "mid", "high"]:
            subset = tercile_df[
                (tercile_df["asset"] == asset) & (tercile_df["regime_value"] == tercile_val)
            ]
            for _, row in subset.iterrows():
                lines.append(
                    f"| {tercile_val} | {row['model']} | {row['n']:,} "
                    f"| {row['rmse']:.6e} | {row['mae']:.6e} | {row['qlike']:.6f} |\n"
                )
        lines.append("\n")

    # -----------------------------------------------------------------------
    # Section 3: Per calendar year
    # -----------------------------------------------------------------------
    lines.append("## Section 3: Per-Calendar-Year Breakdown\n\n")
    year_df = per_regime_df[per_regime_df["regime_type"] == "year"]
    for asset in sorted(year_df["asset"].unique()):
        lines.append(f"### {asset} — By Calendar Year\n\n")
        lines.append(
            "| Year | Model | N | RMSE | MAE | QLIKE |\n|------|-------|---|------|-----|-------|\n"
        )
        asset_year_df = year_df[year_df["asset"] == asset]
        for year_val in sorted(asset_year_df["regime_value"].unique()):
            subset = asset_year_df[asset_year_df["regime_value"] == year_val]
            for _, row in subset.iterrows():
                lines.append(
                    f"| {year_val} | {row['model']} | {row['n']:,} "
                    f"| {row['rmse']:.6e} | {row['mae']:.6e} | {row['qlike']:.6f} |\n"
                )
        lines.append("\n")

    # -----------------------------------------------------------------------
    # Section 4: Honest findings — where LightGBM loses
    # -----------------------------------------------------------------------
    lines.append("## Section 4: Honest Findings — Where LightGBM Underperforms\n\n")
    lines.append(
        "This section explicitly names every asset and regime where LightGBM's "
        "QLIKE is **worse** (higher) than the best classical baseline.  "
        "These results are reported plainly: the project's credibility rests on "
        "honest benchmarking, not on hiding unfavourable outcomes.\n\n"
    )

    losing_lines = []

    # Overall losses
    for asset in sorted(per_asset_df["asset"].unique()):
        asset_rows = per_asset_df[per_asset_df["asset"] == asset].copy()
        lgbm_row = asset_rows[asset_rows["model"] == "LightGBM"]
        baseline_rows = asset_rows[asset_rows["model"].isin(["EWMA", "GARCH", "HAR"])]
        if lgbm_row.empty or baseline_rows.empty:
            continue
        lgbm_q = float(lgbm_row["qlike"].iloc[0])
        best_baseline_q = float(baseline_rows["qlike"].min())
        best_baseline_name = baseline_rows.loc[baseline_rows["qlike"].idxmin(), "model"]
        if lgbm_q > best_baseline_q:
            losing_lines.append(
                f"- **{asset} (overall):** LightGBM QLIKE = {lgbm_q:.6f} vs "
                f"{best_baseline_name} QLIKE = {best_baseline_q:.6f} "
                f"(LightGBM worse by {lgbm_q - best_baseline_q:.6f})\n"
            )

    # Regime losses (tercile)
    for asset in sorted(tercile_df["asset"].unique()):
        for tercile_val in ["low", "mid", "high"]:
            subset = tercile_df[
                (tercile_df["asset"] == asset) & (tercile_df["regime_value"] == tercile_val)
            ]
            lgbm_row = subset[subset["model"] == "LightGBM"]
            baseline_rows = subset[subset["model"].isin(["EWMA", "GARCH", "HAR"])]
            if lgbm_row.empty or baseline_rows.empty:
                continue
            lgbm_q = float(lgbm_row["qlike"].iloc[0])
            best_q = float(baseline_rows["qlike"].min())
            best_name = baseline_rows.loc[baseline_rows["qlike"].idxmin(), "model"]
            if lgbm_q > best_q:
                losing_lines.append(
                    f"- **{asset} / vol-{tercile_val} regime:** LightGBM QLIKE = {lgbm_q:.6f} vs "
                    f"{best_name} QLIKE = {best_q:.6f} "
                    f"(LightGBM worse by {lgbm_q - best_q:.6f})\n"
                )

    # Year losses
    for asset in sorted(year_df["asset"].unique()):
        asset_year_df = year_df[year_df["asset"] == asset]
        for year_val in sorted(asset_year_df["regime_value"].unique()):
            subset = asset_year_df[asset_year_df["regime_value"] == year_val]
            lgbm_row = subset[subset["model"] == "LightGBM"]
            baseline_rows = subset[subset["model"].isin(["EWMA", "GARCH", "HAR"])]
            if lgbm_row.empty or baseline_rows.empty:
                continue
            lgbm_q = float(lgbm_row["qlike"].iloc[0])
            best_q = float(baseline_rows["qlike"].min())
            best_name = baseline_rows.loc[baseline_rows["qlike"].idxmin(), "model"]
            if lgbm_q > best_q:
                losing_lines.append(
                    f"- **{asset} / {year_val}:** LightGBM QLIKE = {lgbm_q:.6f} vs "
                    f"{best_name} QLIKE = {best_q:.6f} "
                    f"(LightGBM worse by {lgbm_q - best_q:.6f})\n"
                )

    if losing_lines:
        for line in losing_lines:
            lines.append(line)
    else:
        lines.append(
            "_LightGBM outperforms or matches the best baseline on QLIKE in every "
            "asset and regime evaluated._\n"
        )

    lines.append("\n")
    lines.append("---\n\n")
    lines.append(
        "*Metrics: daily decimal variance units.  "
        "QLIKE = 0 at perfect forecast.  Lower is better for all three metrics.*\n"
    )

    return "".join(lines)


def _write_csv(per_asset_df: pd.DataFrame, per_regime_df: pd.DataFrame, path: Path) -> None:
    """Write unified CSV with both per-asset and per-regime rows."""
    # per-asset rows: regime_type='overall', regime_value='all'
    overall = per_asset_df.copy()
    overall["regime_type"] = "overall"
    overall["regime_value"] = "all"

    combined = pd.concat([overall, per_regime_df], ignore_index=True)
    combined = combined[
        ["asset", "regime_type", "regime_value", "model", "n", "rmse", "mae", "qlike"]
    ]

    tmp = path.with_suffix(".tmp")
    combined.to_csv(tmp, index=False, float_format="%.10e")
    tmp.replace(path)


def main() -> None:
    """Main entry point: load champion, collect per-fold rows, build and write report."""
    project_root = Path(__file__).parent.parent

    # --- Configure MLflow ---
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    log.info("MLflow tracking URI: %s", tracking_uri)

    # --- Load champion model as native LGBMRegressor ---
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    log.info("Champion: version=%s run_id=%s", mv.version, mv.run_id)

    # Load as native lightgbm (not pyfunc) to avoid schema-enforcement issues
    # with categorical columns — the native model accepts ASSET_DTYPE directly
    model = mlflow.lightgbm.load_model(f"runs:/{mv.run_id}/model")
    log.info("Champion LGBMRegressor loaded (native).")

    # --- Load assets ---
    assets = load_assets(project_root / "config" / "assets.yaml")
    data_root = project_root / "data"

    asset_feature_dfs: dict[str, pd.DataFrame] = {}
    asset_target_series: dict[str, pd.Series] = {}
    asset_processed_dfs: dict[str, pd.DataFrame] = {}

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

        # Log-variance target (aligned to feature index)
        target_var = compute_target(close)
        log_target = pd.Series(
            to_log_var(target_var.values),
            index=target_var.index,
            name=slug,
        )
        if not feat_df.index.equals(close.index):
            log_target = log_target.reindex(feat_df.index)

        asset_feature_dfs[slug] = feat_df
        asset_target_series[slug] = log_target
        asset_processed_dfs[slug] = proc_df
        log.info("Loaded %s: feat=%s", slug, feat_df.shape)

    # --- Retrain one pooled model per outer fold (CR-01 — leak-free) ---
    # The registry stores only the final-fold champion.  A leak-free
    # walk-forward comparison needs one model per fold, so we retrain them
    # deterministically (random_state=42) with the champion run's logged
    # hyperparameters — the final-fold retrain reproduces the champion's
    # training procedure exactly.
    run = client.get_run(mv.run_id)
    best_params = {k: _parse_param(run.data.params[k]) for k in PARAM_GRID}
    log.info("Champion hyperparameters from run %s: %s", mv.run_id, best_params)

    log.info("Retraining per-fold pooled models for leak-free comparison...")
    fold_models = train_per_fold_models(
        asset_feature_dfs,
        asset_target_series,
        params=best_params,
        min_train=MIN_TRAIN,
        step=STEP,
        horizon=HORIZON,
    )

    # Sanity: the retrained models must share the champion's feature schema.
    final_model = fold_models[max(fold_models)]
    if list(final_model.feature_name_) != list(model.feature_name_):
        log.warning(
            "Retrained final-fold feature order differs from registry champion "
            "— possible data or params drift since training. Champion: %s / Retrained: %s",
            list(model.feature_name_),
            list(final_model.feature_name_),
        )

    # --- Collect per-fold rows across all assets ---
    log.info("Collecting per-fold predictions and baseline forecasts...")
    all_rows = _collect_per_fold_rows(
        fold_models,
        asset_feature_dfs,
        asset_target_series,
        asset_processed_dfs,
    )
    log.info(
        "Collected %d total test-fold rows across %d assets.", len(all_rows), len(asset_feature_dfs)
    )

    # --- Build comparison frames ---
    log.info("Building per-asset and per-regime comparison frames...")
    per_asset_df, per_regime_df = _build_comparison_frames(all_rows)

    # --- Log summary ---
    for asset in sorted(per_asset_df["asset"].unique()):
        asset_rows = per_asset_df[per_asset_df["asset"] == asset]
        for _, row in asset_rows.iterrows():
            log.info(
                "  %s %s: n=%d rmse=%.4e mae=%.4e qlike=%.6f",
                row["asset"],
                row["model"],
                row["n"],
                row["rmse"],
                row["mae"],
                row["qlike"],
            )

    # --- Write reports ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    md_path = REPORTS_DIR / "ml_vs_baselines.md"
    md_content = render_report(per_asset_df, per_regime_df)
    tmp_md = md_path.with_suffix(".tmp")
    tmp_md.write_text(md_content, encoding="utf-8")
    tmp_md.replace(md_path)
    log.info("Written: %s", md_path)

    csv_path = REPORTS_DIR / "ml_vs_baselines.csv"
    _write_csv(per_asset_df, per_regime_df, csv_path)
    log.info("Written: %s", csv_path)

    print(f"\nReport written to:\n  {md_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
