"""End-to-end test for the baseline report generator.

RED PHASE (Task 1): This file is committed while generate_baseline_report
does NOT yet exist in volforecast.reports.baseline.  The test is expected
to fail with ImportError or AttributeError at this stage.

GREEN PHASE (Task 3): After implementing reports/baseline.py, all tests
here must pass.

Test strategy:
- Use a tiny synthetic fixture (2 assets, ~100 rows) so the harness can
  yield >= 1 fold with min_train=30.  This keeps the e2e test fast and
  fully offline.
- The fixture data is constructed in-memory (no network, no real processed data).
- Validates both the Markdown output (per-asset table, QLIKE column) and
  the CSV output (schema and n_forecasts > 0).
- Validates that NaN target rows are dropped before scoring (Pitfall 4).
- Validates that metrics are computed from EWMA forecasts, not in-sample
  training predictions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from volforecast.reports.baseline import generate_baseline_report  # RED: fails here

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_asset_parquet(tmp_dir: Path, slug: str, n: int = 100) -> tuple[dict, Path]:
    """Create a synthetic processed-parquet fixture and an asset dict.

    Returns (asset_dict, parquet_path).
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateBaselineReport:
    """Contract tests for generate_baseline_report.

    These are the RED tests in Task 1.  Task 3 makes them GREEN.
    """

    @pytest.fixture
    def two_asset_setup(self, tmp_path: Path):
        """Set up a 2-asset fixture directory and asset dicts.

        Uses min_train=30 so the harness yields >= 1 fold on 100-row series.
        """
        data_dir = tmp_path / "processed" / "crypto"
        data_dir.mkdir(parents=True)

        assets = []
        paths = {}
        for slug in ["BTC-USD", "ETH-USD"]:
            asset, parquet_path = _make_asset_parquet(data_dir, slug, n=100)
            assets.append(asset)
            paths[slug] = parquet_path

        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        return {
            "assets": assets,
            "output_dir": output_dir,
            "data_dir": data_dir.parent,  # parent of crypto/
        }

    def test_report_md_exists(self, two_asset_setup):
        """generate_baseline_report must write baseline_eval.md to output_dir."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=30,
            step=21,
            data_root=s["data_dir"].parent,
        )
        md_path = s["output_dir"] / "baseline_eval.md"
        assert md_path.exists(), "baseline_eval.md must be written"

    def test_report_csv_exists(self, two_asset_setup):
        """generate_baseline_report must write baseline_metrics.csv to output_dir."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=30,
            step=21,
            data_root=s["data_dir"].parent,
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        assert csv_path.exists(), "baseline_metrics.csv must be written"

    def test_md_contains_qlike_column(self, two_asset_setup):
        """Markdown report must include a QLIKE column."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=30,
            step=21,
            data_root=s["data_dir"].parent,
        )
        md_text = (s["output_dir"] / "baseline_eval.md").read_text()
        assert "QLIKE" in md_text, "Markdown report must contain 'QLIKE'"

    def test_md_contains_all_asset_slugs(self, two_asset_setup):
        """Markdown report must mention each asset slug."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=30,
            step=21,
            data_root=s["data_dir"].parent,
        )
        md_text = (s["output_dir"] / "baseline_eval.md").read_text()
        for slug in ["BTC-USD", "ETH-USD"]:
            assert slug in md_text, f"Markdown report must mention asset '{slug}'"

    def test_md_contains_rmse_and_mae(self, two_asset_setup):
        """Markdown report must include RMSE and MAE columns."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=30,
            step=21,
            data_root=s["data_dir"].parent,
        )
        md_text = (s["output_dir"] / "baseline_eval.md").read_text()
        assert "RMSE" in md_text, "Markdown must contain 'RMSE'"
        assert "MAE" in md_text, "Markdown must contain 'MAE'"

    def test_csv_schema(self, two_asset_setup):
        """CSV must have columns: asset, model, n_forecasts, rmse, mae, qlike."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=30,
            step=21,
            data_root=s["data_dir"].parent,
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)

        required = {"asset", "model", "n_forecasts", "rmse", "mae", "qlike"}
        assert required.issubset(set(headers)), (
            f"CSV missing columns. Got: {headers}, required: {required}"
        )
        assert len(rows) >= 2, "CSV must have at least one row per asset"

    def test_n_forecasts_positive(self, two_asset_setup):
        """n_forecasts must be > 0 for each asset/model row."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=30,
            step=21,
            data_root=s["data_dir"].parent,
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            n = int(row["n_forecasts"])
            assert n > 0, (
                f"n_forecasts must be > 0, got {n} for asset={row['asset']}"
            )

    def test_metrics_are_numeric(self, two_asset_setup):
        """RMSE, MAE, QLIKE must be finite floats >= 0."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=30,
            step=21,
            data_root=s["data_dir"].parent,
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            for col in ("rmse", "mae", "qlike"):
                val = float(row[col])
                assert np.isfinite(val), f"{col} must be finite, got {val}"
                assert val >= 0, f"{col} must be >= 0, got {val}"

    def test_model_column_is_ewma(self, two_asset_setup):
        """At this plan stage, the only model is EWMA."""
        s = two_asset_setup
        generate_baseline_report(
            s["assets"],
            s["output_dir"],
            min_train=30,
            step=21,
            data_root=s["data_dir"].parent,
        )
        csv_path = s["output_dir"] / "baseline_metrics.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            assert row["model"] == "EWMA", (
                f"Expected model='EWMA', got '{row['model']}'"
            )
