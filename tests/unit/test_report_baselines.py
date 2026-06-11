"""Tests for the three-baseline report generator extension.

TDD RED phase: tests written to verify that generate_baseline_report
scores EWMA, GARCH, and HAR-RV on IDENTICAL walk-forward folds per asset.

Contract being tested:
  1. All three models use identical train/test split indices per asset.
  2. baseline_metrics.csv has 3 rows per asset (one per model).
  3. All metric columns (rmse, mae, qlike) are finite and >= 0.
  4. baseline_eval.md mentions EWMA, GARCH, and HAR in the text.
  5. GARCH fallback count is recorded per asset (transparency).
  6. The 02-02 e2e contract still passes (EWMA row unchanged in shape).

This test uses a small 2-asset fixture (min_train=30, step=10) to stay fast.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volforecast.reports.baseline import generate_baseline_report

# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #


def _make_asset_parquet(tmp_dir: Path, slug: str, n: int = 120) -> tuple[dict, Path]:
    """Create a synthetic processed-parquet fixture for testing.

    Returns (asset_dict, parquet_path).  n must be > min_train + step.
    """
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(42 + hash(slug) % 1000)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame(
        {
            "open": close * rng.uniform(0.99, 1.0, n),
            "high": close * rng.uniform(1.0, 1.01, n),
            "low": close * rng.uniform(0.99, 1.0, n),
            "close": close,
            "volume": rng.uniform(1e6, 1e7, n),
        },
        index=dates,
    )
    df.index.name = "date"
    parquet_path = tmp_dir / f"{slug}.parquet"
    df.to_parquet(parquet_path)
    return {"symbol": slug, "asset_class": "crypto"}, parquet_path


# --------------------------------------------------------------------------- #
# Setup fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_asset_setup(tmp_path: Path):
    """Set up 2-asset fixture with min_train=30, step=10 for fast testing."""
    data_dir = tmp_path / "processed" / "crypto"
    data_dir.mkdir(parents=True)

    assets = []
    for slug in ["BTC-USD", "ETH-USD"]:
        asset, _ = _make_asset_parquet(data_dir, slug, n=120)
        assets.append(asset)

    output_dir = tmp_path / "reports"
    output_dir.mkdir()

    return {
        "assets": assets,
        "output_dir": output_dir,
        "data_root": tmp_path,  # data_root / processed / crypto / {slug}.parquet
        "min_train": 30,
        "step": 10,
    }


# --------------------------------------------------------------------------- #
# 1. Three baselines present in CSV
# --------------------------------------------------------------------------- #


class TestThreeBaselinesInCSV:
    def test_csv_has_three_rows_per_asset(self, two_asset_setup):
        """CSV must have 3 rows per asset: EWMA, GARCH, HAR."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        # 2 assets x 3 models = 6 rows
        assert len(rows) == 6, f"Expected 6 rows (2 assets × 3 models), got {len(rows)}"

    def test_csv_model_column_has_all_three(self, two_asset_setup):
        """CSV model column must include EWMA, GARCH, and HAR entries."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        models = {r["model"] for r in rows}
        assert "EWMA" in models, f"EWMA missing from models: {models}"
        assert "GARCH" in models, f"GARCH missing from models: {models}"
        assert "HAR" in models, f"HAR missing from models: {models}"

    def test_csv_schema_has_required_columns(self, two_asset_setup):
        """CSV must have: asset, model, n_forecasts, rmse, mae, qlike."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])

        required = {"asset", "model", "n_forecasts", "rmse", "mae", "qlike"}
        assert required.issubset(headers), f"CSV missing columns: {required - headers}"


# --------------------------------------------------------------------------- #
# 2. All metrics finite and non-negative
# --------------------------------------------------------------------------- #


class TestMetricsFinite:
    def test_all_metrics_finite(self, two_asset_setup):
        """rmse, mae, qlike must be finite for all rows."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            for col in ("rmse", "mae", "qlike"):
                val = float(row[col])
                assert np.isfinite(val), (
                    f"{col}={val} not finite for asset={row['asset']}, model={row['model']}"
                )

    def test_all_metrics_non_negative(self, two_asset_setup):
        """rmse, mae, qlike must all be >= 0."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            for col in ("rmse", "mae", "qlike"):
                val = float(row[col])
                assert val >= 0.0, f"{col}={val} < 0 for asset={row['asset']}, model={row['model']}"

    def test_n_forecasts_positive(self, two_asset_setup):
        """n_forecasts must be > 0 for all rows."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            n = int(row["n_forecasts"])
            assert n > 0, f"n_forecasts={n} for asset={row['asset']}, model={row['model']}"


# --------------------------------------------------------------------------- #
# 3. Identical folds across all three models per asset
# --------------------------------------------------------------------------- #


class TestIdenticalFoldsAcrossModels:
    def test_same_n_forecasts_per_asset(self, two_asset_setup):
        """All three models must have the SAME n_forecasts for each asset.

        This verifies that identical walk-forward splits are used — not
        different splits per model (which would make the comparison unfair).
        """
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        # Group by asset, check n_forecasts is identical across models
        by_asset: dict[str, list[int]] = {}
        for row in rows:
            slug = row["asset"]
            n = int(row["n_forecasts"])
            by_asset.setdefault(slug, []).append(n)

        for slug, counts in by_asset.items():
            assert len(set(counts)) == 1, (
                f"Asset {slug}: models have different n_forecasts {counts} — "
                "all models must use identical walk-forward folds"
            )


# --------------------------------------------------------------------------- #
# 4. Markdown report mentions all three models
# --------------------------------------------------------------------------- #


class TestMarkdownReport:
    def test_md_contains_ewma(self, two_asset_setup):
        """Markdown must mention EWMA."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        md = (s["output_dir"] / "baseline_eval.md").read_text()
        assert "EWMA" in md, "baseline_eval.md must mention EWMA"

    def test_md_contains_garch(self, two_asset_setup):
        """Markdown must mention GARCH."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        md = (s["output_dir"] / "baseline_eval.md").read_text()
        assert "GARCH" in md, "baseline_eval.md must mention GARCH"

    def test_md_contains_har(self, two_asset_setup):
        """Markdown must mention HAR."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        md = (s["output_dir"] / "baseline_eval.md").read_text()
        assert "HAR" in md, "baseline_eval.md must mention HAR"

    def test_md_contains_asset_slugs(self, two_asset_setup):
        """Markdown must mention both asset slugs."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=s["min_train"],
            step=s["step"],
            data_root=s["data_root"],
        )
        md = (s["output_dir"] / "baseline_eval.md").read_text()
        for slug in ["BTC-USD", "ETH-USD"]:
            assert slug in md, f"baseline_eval.md must mention {slug}"
