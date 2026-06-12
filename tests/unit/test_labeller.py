"""Unit tests for volforecast.monitoring.labeller.

Hermetic, offline, no MLflow/network.  Synthetic processed parquets and
prediction logs are written under tmp_path (VOLFORECAST_ROOT is set via
monkeypatch so config helpers resolve correctly).

Five behaviour tests:
  1. Idempotency — calling label twice adds 0 net-new rows on second call.
  2. Champion + GARCH coexist — both model_alias values present for same key.
  3. No-forward-close — last processed date produces no labelled row.
  4. Realized join correctness — joined realized_var matches compute_target.
  5. GARCH value units — garch_baseline forecast_var is decimal (~1e-4..1e-3).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volforecast.features.target import compute_target
from volforecast.monitoring.labeller import (
    GARCH_ALIAS,
    label_champion_forecasts,
    label_forecasts,
    label_garch_baseline,
)

# ---------------------------------------------------------------------------
# Synthetic-data fixture helpers (replicate test_api.py pattern)
# ---------------------------------------------------------------------------

_RNG_SEED = 42
_N_ROWS = 290  # > 252 (GARCH min_train) + buffer for labels


def _make_close_series(n: int = _N_ROWS, seed: int = _RNG_SEED) -> pd.Series:
    """Synthetic close-price series with tz-aware UTC DatetimeIndex."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 0.015, size=n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(close, index=pd.Index(dates, name="date"), name="close")


def _write_processed(root: Path, asset_class: str, slug: str, close: pd.Series) -> Path:
    """Write a minimal processed parquet (close column only) under root/data/processed/."""
    out = root / "data" / "processed" / asset_class / f"{slug}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"close": close.values}, index=close.index)
    df.to_parquet(out)
    return out


def _write_assets_yaml(root: Path, assets: list[dict]) -> Path:
    """Write a minimal config/assets.yaml."""
    import yaml

    cfg_path = root / "config" / "assets.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    content = {"assets": assets}
    with open(cfg_path, "w") as f:
        yaml.dump(content, f)
    return cfg_path


def _write_predictions(
    root: Path,
    asset: str,
    as_of_dates: list[pd.Timestamp],
    forecast_var: float = 3e-4,
    model_version: str = "3",
    alias: str = "champion",
) -> Path:
    """Write a synthetic prediction log parquet under root/data/predictions/."""
    out = root / "data" / "predictions" / "predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    # timestamp_utc is one day AFTER as_of_date (label_champion_forecasts subtracts 1 day)
    timestamps = [d + pd.Timedelta(days=1) for d in as_of_dates]
    df = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "asset": [asset] * len(as_of_dates),
            "horizon": [1] * len(as_of_dates),
            "forecast_var": [forecast_var] * len(as_of_dates),
            "model_version": [model_version] * len(as_of_dates),
            "alias": [alias] * len(as_of_dates),
        }
    )
    # Merge if file already exists (multiple assets)
    if out.exists():
        existing = pd.read_parquet(out)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Shared fixture: synthetic one-asset project root
# ---------------------------------------------------------------------------


@pytest.fixture()
def synth_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a hermetic project root with one crypto asset (BTC-USD)."""
    monkeypatch.setenv("VOLFORECAST_ROOT", str(tmp_path))

    assets = [{"symbol": "BTC/USDT", "asset_class": "crypto", "exchange": "binance"}]
    _write_assets_yaml(tmp_path, assets)

    close = _make_close_series()
    _write_processed(tmp_path, "crypto", "BTC-USD", close)

    # Labelable dates = all dates where compute_target is not NaN (all but last)
    rv = compute_target(close)
    labelable_dates = rv.dropna().index

    # Use dates AFTER the GARCH min_train window (252 rows) so GARCH also has forecasts
    # for the same dates — required for the coexistence test.
    pred_dates = list(labelable_dates[255:265])
    _write_predictions(tmp_path, "BTC-USD", pred_dates)

    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: Idempotency
# ---------------------------------------------------------------------------


def test_idempotency(synth_root: Path) -> None:
    """Calling label_forecasts twice yields 0 net-new rows on the second call."""
    data_root = synth_root / "data"
    fvr_path = data_root / "monitoring" / "forecast_vs_realized.parquet"

    # First run — should add rows
    n1 = label_forecasts(data_root, fvr_path)
    assert n1 > 0, f"Expected >0 net-new rows on first run, got {n1}"

    row_count_after_first = len(pd.read_parquet(fvr_path))

    # Second run — idempotent, no new rows
    n2 = label_forecasts(data_root, fvr_path)
    assert n2 == 0, f"Expected 0 net-new rows on second run (idempotent), got {n2}"

    row_count_after_second = len(pd.read_parquet(fvr_path))
    assert row_count_after_second == row_count_after_first, (
        "Row count changed on second run — idempotency violated"
    )


# ---------------------------------------------------------------------------
# Test 2: Champion + GARCH coexist in output
# ---------------------------------------------------------------------------


def test_champion_and_garch_coexist(synth_root: Path) -> None:
    """Output contains both champion and garch_baseline rows for the same (asset, as_of_date)."""
    data_root = synth_root / "data"
    fvr_path = data_root / "monitoring" / "forecast_vs_realized.parquet"

    label_forecasts(data_root, fvr_path)

    df = pd.read_parquet(fvr_path)
    aliases = set(df["model_alias"].unique())

    assert "champion" in aliases, f"Expected 'champion' alias in output, got: {aliases}"
    assert GARCH_ALIAS in aliases, f"Expected '{GARCH_ALIAS}' alias in output, got: {aliases}"

    # Verify coexistence: at least one (asset, as_of_date) pair has both aliases
    pivot = df.groupby(["asset", "as_of_date"])["model_alias"].apply(set)
    coexisting = pivot[pivot.apply(lambda s: "champion" in s and GARCH_ALIAS in s)]
    assert len(coexisting) > 0, (
        "No (asset, as_of_date) pair has both champion and garch_baseline rows"
    )


# ---------------------------------------------------------------------------
# Test 3: No-forward-close → no labelled row
# ---------------------------------------------------------------------------


def test_no_forward_close_produces_no_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A prediction whose as_of_date is the last processed date produces no FVR row.

    compute_target returns NaN for the last row (no t+1 close exists), so
    there can be no realized_var label — that prediction must be dropped.
    """
    monkeypatch.setenv("VOLFORECAST_ROOT", str(tmp_path))

    assets = [{"symbol": "BTC/USDT", "asset_class": "crypto", "exchange": "binance"}]
    _write_assets_yaml(tmp_path, assets)

    close = _make_close_series(n=280)
    _write_processed(tmp_path, "crypto", "BTC-USD", close)

    # Use the LAST date of the processed series as as_of_date
    # compute_target at the last position is NaN (no t+1 close)
    last_date = close.index[-1]

    _write_predictions(tmp_path, "BTC-USD", [last_date])

    data_root = tmp_path / "data"

    # Champion labelling for last date should yield 0 champion rows (no realized label)
    predictions = pd.read_parquet(data_root / "predictions" / "predictions.parquet")
    champion_df = label_champion_forecasts(predictions, data_root)

    # Filter to the last_date as_of_date rows (timestamp_utc = last_date + 1 day)
    # so as_of_date derived = last_date
    champion_for_last = (
        champion_df[champion_df["as_of_date"] == last_date]
        if not champion_df.empty
        else pd.DataFrame()
    )

    assert len(champion_for_last) == 0, (
        f"Expected 0 rows for last date (no forward close), got {len(champion_for_last)}"
    )


# ---------------------------------------------------------------------------
# Test 4: Realized join correctness
# ---------------------------------------------------------------------------


def test_realized_join_correctness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Joined realized_var equals compute_target at the as_of_date (within float tolerance)."""
    monkeypatch.setenv("VOLFORECAST_ROOT", str(tmp_path))

    assets = [{"symbol": "BTC/USDT", "asset_class": "crypto", "exchange": "binance"}]
    _write_assets_yaml(tmp_path, assets)

    close = _make_close_series(n=280)
    _write_processed(tmp_path, "crypto", "BTC-USD", close)

    # Use 5th labelable date as our test date
    rv = compute_target(close)
    labelable = rv.dropna()
    test_date = labelable.index[5]
    expected_rv = float(labelable.iloc[5])

    _write_predictions(tmp_path, "BTC-USD", [test_date])

    data_root = tmp_path / "data"
    predictions = pd.read_parquet(data_root / "predictions" / "predictions.parquet")
    champion_df = label_champion_forecasts(predictions, data_root)

    assert not champion_df.empty, "Expected at least one champion row"
    matched = champion_df[champion_df["as_of_date"] == test_date]
    assert len(matched) == 1, f"Expected exactly 1 row for test_date, got {len(matched)}"

    actual_rv = float(matched.iloc[0]["realized_var"])
    assert math.isclose(actual_rv, expected_rv, rel_tol=1e-9), (
        f"realized_var mismatch: expected {expected_rv}, got {actual_rv}"
    )


# ---------------------------------------------------------------------------
# Test 5: GARCH value units (decimal variance, not 100x-scaled)
# ---------------------------------------------------------------------------


def test_garch_forecast_var_decimal_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GARCH baseline forecast_var is decimal variance (~1e-4..1e-3), not 100x-scaled.

    A 100x-scaled value would be ~1.0..10.0, which would be clearly wrong and
    would corrupt the QLIKE comparison.  This test asserts the correct magnitude,
    confirming exactly one /GARCH_SCALE**2 de-scale (no double-scale bug).
    """
    monkeypatch.setenv("VOLFORECAST_ROOT", str(tmp_path))

    assets = [{"symbol": "BTC/USDT", "asset_class": "crypto", "exchange": "binance"}]
    _write_assets_yaml(tmp_path, assets)

    close = _make_close_series(n=_N_ROWS)
    _write_processed(tmp_path, "crypto", "BTC-USD", close)

    asset_cfg = {"symbol": "BTC/USDT", "asset_class": "crypto", "exchange": "binance"}
    data_root = tmp_path / "data"

    garch_df = label_garch_baseline(asset_cfg, data_root)

    assert not garch_df.empty, "Expected non-empty garch_baseline DataFrame"
    assert (garch_df["model_alias"] == GARCH_ALIAS).all()

    # All forecast_var values must be in the decimal range [1e-6, 1e-1]
    # (not 100x-scaled which would be ~1.0..10.0)
    forecast_vars = garch_df["forecast_var"].values
    assert (forecast_vars > 1e-6).all(), (
        f"Some GARCH forecast_var values suspiciously small: {forecast_vars.min()}"
    )
    assert (forecast_vars < 1e-1).all(), (
        f"Some GARCH forecast_var values suspiciously large (100x-scaled?): {forecast_vars.max()}"
    )

    # Tighter check: typical daily crypto variance ~1e-4..1e-2
    # At least 90% of values should be in [1e-5, 1e-2]
    in_range = ((forecast_vars >= 1e-5) & (forecast_vars <= 1e-2)).sum()
    frac = in_range / len(forecast_vars)
    assert frac >= 0.9, (
        f"Expected >=90% of GARCH forecast_var in [1e-5, 1e-2], got {frac:.1%}. "
        f"Min={forecast_vars.min():.2e}, max={forecast_vars.max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test: FileNotFoundError when prediction log is absent
# ---------------------------------------------------------------------------


def test_missing_prediction_log_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """label_forecasts raises a clear FileNotFoundError when predictions.parquet is absent."""
    monkeypatch.setenv("VOLFORECAST_ROOT", str(tmp_path))

    assets = [{"symbol": "BTC/USDT", "asset_class": "crypto", "exchange": "binance"}]
    _write_assets_yaml(tmp_path, assets)

    close = _make_close_series()
    _write_processed(tmp_path, "crypto", "BTC-USD", close)

    data_root = tmp_path / "data"
    fvr_path = data_root / "monitoring" / "forecast_vs_realized.parquet"

    # predictions.parquet does NOT exist
    with pytest.raises(FileNotFoundError, match="Prediction log not found"):
        label_forecasts(data_root, fvr_path)
