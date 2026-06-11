"""Smoke tests for report completeness (offline, no MLflow / compose stack).

Tests verify that ``render_report`` produces a well-formed ML-vs-baseline
comparison markdown with:

- All 5 asset names present
- All 4 model names present (LightGBM, EWMA, GARCH, HAR)
- Tercile labels (low/mid/high) present
- At least one calendar year present
- The honest-losses section explicitly names losing assets/regimes

These tests are deliberately offline: they feed a synthetic comparison frame
into the pure ``render_report`` function and never require MLflow, the compose
stack, or the actual model artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow import from scripts/ (which contains eval_lgbm.py)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from eval_lgbm import render_report  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: synthetic comparison frames
# ---------------------------------------------------------------------------

ASSETS = ["BTC-USD", "ETH-USD", "SPY", "AAPL", "MSFT"]
MODELS = ["LightGBM", "EWMA", "GARCH", "HAR"]
TERCILES = ["low", "mid", "high"]
YEARS = ["2022", "2023", "2024"]

# Synthetic per-asset metrics: LightGBM has high QLIKE (losing to baselines)
# This makes the honest-findings section non-empty and assertable.
_LGBM_QLIKE = 3.5
_BEST_BASELINE_QLIKE = 1.8  # e.g. HAR wins


def _make_per_asset_df() -> pd.DataFrame:
    """Synthetic per-asset metrics frame.

    LightGBM QLIKE is intentionally higher than the best baseline (HAR)
    for all 5 assets, so the honest-losses section is non-empty.
    """
    records = []
    for asset in ASSETS:
        # LightGBM: high QLIKE (loses)
        records.append(
            {
                "asset": asset,
                "model": "LightGBM",
                "n": 500,
                "rmse": 1.5e-3,
                "mae": 7.0e-4,
                "qlike": _LGBM_QLIKE,
            }
        )
        # EWMA
        records.append(
            {
                "asset": asset,
                "model": "EWMA",
                "n": 500,
                "rmse": 1.4e-3,
                "mae": 6.5e-4,
                "qlike": 2.0,
            }
        )
        # GARCH
        records.append(
            {
                "asset": asset,
                "model": "GARCH",
                "n": 500,
                "rmse": 1.6e-3,
                "mae": 9.0e-4,
                "qlike": 2.1,
            }
        )
        # HAR (best baseline)
        records.append(
            {
                "asset": asset,
                "model": "HAR",
                "n": 500,
                "rmse": 1.4e-3,
                "mae": 8.0e-4,
                "qlike": _BEST_BASELINE_QLIKE,
            }
        )
    return pd.DataFrame(records)


def _make_per_regime_df() -> pd.DataFrame:
    """Synthetic per-regime metrics frame.

    Includes both 'tercile' and 'year' regime types.
    LightGBM has high QLIKE in the 'high' vol tercile and in year 2022,
    so the honest-losses section can reference these specific losing regimes.
    """
    records = []
    for asset in ASSETS:
        # --- Tercile regimes ---
        for tercile in TERCILES:
            # LightGBM is especially bad in the 'high' vol regime
            lgbm_q = 5.0 if tercile == "high" else 1.5
            for model_name, model_qlike in [
                ("LightGBM", lgbm_q),
                ("EWMA", 2.0),
                ("GARCH", 2.1),
                ("HAR", 1.7),
            ]:
                records.append(
                    {
                        "asset": asset,
                        "regime_type": "tercile",
                        "regime_value": tercile,
                        "model": model_name,
                        "n": 150,
                        "rmse": 1.5e-3,
                        "mae": 7.0e-4,
                        "qlike": model_qlike,
                    }
                )

        # --- Calendar year regimes ---
        for year in YEARS:
            for model_name, model_qlike in [
                ("LightGBM", 4.0),
                ("EWMA", 1.9),
                ("GARCH", 2.0),
                ("HAR", 1.8),
            ]:
                records.append(
                    {
                        "asset": asset,
                        "regime_type": "year",
                        "regime_value": year,
                        "model": model_name,
                        "n": 180,
                        "rmse": 1.5e-3,
                        "mae": 7.0e-4,
                        "qlike": model_qlike,
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def per_asset_df() -> pd.DataFrame:
    return _make_per_asset_df()


@pytest.fixture(scope="module")
def per_regime_df() -> pd.DataFrame:
    return _make_per_regime_df()


@pytest.fixture(scope="module")
def rendered_md(per_asset_df, per_regime_df) -> str:
    """Rendered markdown string from synthetic frames.  Computed once per module."""
    return render_report(per_asset_df, per_regime_df)


class TestRenderReportStructure:
    """Structural checks: sections, tables, keywords."""

    def test_contains_qlike_keyword(self, rendered_md):
        assert "QLIKE" in rendered_md, "Report must mention 'QLIKE'"

    def test_contains_all_5_assets(self, rendered_md):
        for asset in ASSETS:
            assert asset in rendered_md, f"Asset '{asset}' missing from report"

    def test_contains_all_4_models(self, rendered_md):
        for model in MODELS:
            assert model in rendered_md, f"Model '{model}' missing from report"

    def test_contains_all_tercile_labels(self, rendered_md):
        for tercile in TERCILES:
            assert tercile in rendered_md, f"Tercile label '{tercile}' missing from report"

    def test_contains_calendar_year(self, rendered_md):
        found = any(year in rendered_md for year in YEARS)
        assert found, f"No calendar year from {YEARS} found in report"

    def test_report_is_nonempty_string(self, rendered_md):
        assert isinstance(rendered_md, str)
        assert len(rendered_md) > 500, "Report seems too short to be valid"

    def test_has_section_headers(self, rendered_md):
        assert "Section 1" in rendered_md, "Missing 'Section 1' header"
        assert "Section 2" in rendered_md, "Missing 'Section 2' header"
        assert "Section 3" in rendered_md, "Missing 'Section 3' header"
        assert "Section 4" in rendered_md, "Missing 'Section 4' (honest findings) header"


class TestHonestFindings:
    """Honest-losses section: LightGBM underperformance is named explicitly."""

    def test_honest_findings_section_exists(self, rendered_md):
        assert "Honest Findings" in rendered_md or "Underperforms" in rendered_md, (
            "Report must contain an honest-findings section"
        )

    def test_lgbm_losing_asset_named(self, rendered_md):
        """With synthetic data where LightGBM QLIKE=3.5 > HAR QLIKE=1.8,
        at least one asset should be named in the losses section."""
        # Any of the 5 assets should appear with explicit loss statement
        found_loss = any(f"{asset} (overall)" in rendered_md for asset in ASSETS)
        assert found_loss, (
            "The honest-findings section must name at least one (asset, overall) "
            f"where LightGBM loses. Report excerpt:\n"
            f"{rendered_md[-2000:]}"
        )

    def test_lgbm_qlike_value_appears_in_losses(self, rendered_md):
        """The LightGBM QLIKE value should appear in the honest findings."""
        # The QLIKE=3.5 should be formatted as e.g. '3.500000'
        assert "3.500000" in rendered_md, (
            "LightGBM QLIKE=3.5 should appear in the honest findings section"
        )

    def test_high_vol_regime_loss_named(self, rendered_md):
        """In the synthetic data, LightGBM QLIKE=5.0 in vol-high; should be named."""
        assert "vol-high" in rendered_md or "high" in rendered_md, (
            "High volatility regime must appear in report"
        )
        # The 5.0 QLIKE for 'high' tercile should also appear
        assert "5.000000" in rendered_md, (
            "LightGBM QLIKE=5.0 for high-vol regime should appear in findings"
        )


class TestOfflineOnly:
    """Verify the test is offline: no MLflow, no network calls required."""

    def test_render_report_does_not_import_mlflow(self, per_asset_df, per_regime_df):
        """render_report is a pure function with no MLflow import or network calls."""
        # This test passes as long as render_report completes without raising.
        # If MLflow/network were required, the test would fail in CI without a
        # running MLflow server (the entire point of making render_report pure).
        md = render_report(per_asset_df, per_regime_df)
        assert len(md) > 0
