"""Unit tests for volforecast.validate — equity schema + validate_asset dispatcher.

Tests cover:
- equity_ohlcv_schema: pandera schema for equity OHLCV data
- validate_asset: dispatcher that fails closed with quarantine report on any failure
- validate_and_quarantine: schema-level quarantine helper (from Plan 01)

All tests are fully offline (committed fixtures only). The tmp_path fixture from
pytest provides a temporary quarantine directory.

TDD RED: this file is written before the implementation in schemas.py / __init__.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _load_fixture(name: str) -> pd.DataFrame:
    """Load a parquet fixture from tests/fixtures/."""
    return pd.read_parquet(FIXTURES_DIR / name)


# ---------------------------------------------------------------------------
# equity_ohlcv_schema: basic structural tests
# ---------------------------------------------------------------------------


class TestEquityOhlcvSchema:
    """equity_ohlcv_schema must accept valid data and reject schema violations."""

    def test_equity_schema_accepts_valid_data(self) -> None:
        """A well-formed equity OHLCV DataFrame validates without error."""
        import exchange_calendars as xcals

        from volforecast.validate import equity_ohlcv_schema

        xnys = xcals.get_calendar("XNYS")
        sessions = xnys.sessions_in_range("2022-01-03", "2022-01-14")
        idx = sessions.tz_localize("UTC")
        n = len(idx)
        # Vary close values so stale_row_check (used by validate_asset) does not fire.
        # The schema itself does not run stale checks, but this fixture is also reused
        # by validate_asset tests so we make it realistic.
        closes = [150.0 + i * 0.5 for i in range(n)]
        df = pd.DataFrame(
            {
                "open": [150.0 + i * 0.3 for i in range(n)],
                "high": [155.0 + i * 0.3 for i in range(n)],
                "low": [148.0 + i * 0.3 for i in range(n)],
                "close": closes,
                "volume": [1_000_000.0 + i * 1000 for i in range(n)],
            },
            index=idx,
        )
        df.index.name = "date"
        validated = equity_ohlcv_schema.validate(df, lazy=True)
        assert validated is not None
        assert list(validated.columns) == ["open", "high", "low", "close", "volume"]

    def test_equity_schema_rejects_non_positive_price(self) -> None:
        """A row with a zero or negative price value must fail equity_ohlcv_schema."""
        import pandera.errors

        from volforecast.validate import equity_ohlcv_schema

        dates = pd.date_range("2022-01-03", periods=3, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [-1.0, 150.0, 150.0],  # negative price on row 0
                "high": [155.0, 155.0, 155.0],
                "low": [-2.0, 148.0, 148.0],  # negative price on row 0
                "close": [152.0, 152.0, 152.0],
                "volume": [1_000_000.0] * 3,
            },
            index=dates,
        )
        df.index.name = "date"
        with pytest.raises((pandera.errors.SchemaErrors, pandera.errors.SchemaError)):
            equity_ohlcv_schema.validate(df, lazy=True)

    def test_equity_schema_in_validate_import(self) -> None:
        """equity_ohlcv_schema must be importable from volforecast.validate."""
        from volforecast.validate import equity_ohlcv_schema  # noqa: F401


# ---------------------------------------------------------------------------
# validate_asset: weekend-bad fixture raises + writes quarantine
# ---------------------------------------------------------------------------


class TestValidateAssetEquityRejectsWeekend:
    """validate_asset with 'equity' and the weekend fixture must fail closed."""

    def test_validate_asset_equity_rejects_weekend(self, tmp_path: Path) -> None:
        """Weekend-bad fixture through validate_asset raises and writes a quarantine CSV."""
        from volforecast.validate import validate_asset

        df = _load_fixture("equity_bad_weekend.parquet")
        quarantine_dir = tmp_path / "quarantine"

        with pytest.raises(Exception):  # ValidationError or SchemaErrors
            validate_asset(df, asset_class="equity", quarantine_dir=quarantine_dir)

        # A quarantine file must have been written
        quarantine_files = list(quarantine_dir.glob("equity_*.csv"))
        assert len(quarantine_files) == 1, (
            f"Expected exactly 1 quarantine CSV for equity, found: {quarantine_files}"
        )
        quarantine_df = pd.read_csv(quarantine_files[0])
        assert len(quarantine_df) > 0, "Quarantine CSV must contain at least one row"
        assert "check" in quarantine_df.columns or "reason" in quarantine_df.columns, (
            "Quarantine CSV must have at least a 'check' or 'reason' column"
        )


# ---------------------------------------------------------------------------
# validate_asset: OHLC-violation fixture raises + writes quarantine
# ---------------------------------------------------------------------------


class TestValidateAssetCryptoRejectsOhlcViolation:
    """validate_asset with 'crypto' and an OHLC-violation row must fail closed."""

    def test_validate_asset_crypto_rejects_ohlc_violation(self, tmp_path: Path) -> None:
        """A high<low row in a crypto DataFrame raises and writes a quarantine CSV."""
        from volforecast.validate import validate_asset

        dates = pd.date_range("2022-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [40000.0] * 5,
                "high": [41000.0, 41000.0, 39000.0, 41000.0, 41000.0],
                "low": [39000.0, 39000.0, 40000.0, 39000.0, 39000.0],  # row 2: high<low
                "close": [40500.0] * 5,
                "volume": [5000.0] * 5,
            },
            index=dates,
        )
        df.index.name = "date"
        quarantine_dir = tmp_path / "quarantine"

        with pytest.raises(Exception):  # ValidationError or SchemaErrors
            validate_asset(df, asset_class="crypto", quarantine_dir=quarantine_dir)

        quarantine_files = list(quarantine_dir.glob("crypto_*.csv"))
        assert len(quarantine_files) == 1, (
            f"Expected exactly 1 quarantine CSV for crypto, found: {quarantine_files}"
        )
        quarantine_df = pd.read_csv(quarantine_files[0])
        assert len(quarantine_df) > 0, "Quarantine CSV must contain at least one row"


# ---------------------------------------------------------------------------
# validate_asset: clean fixtures return df and write NO quarantine
# ---------------------------------------------------------------------------


class TestValidateAssetCleanPasses:
    """validate_asset on clean fixtures returns the validated df without writing quarantine."""

    def test_validate_asset_clean_crypto_passes(self, tmp_path: Path) -> None:
        """crypto_sample.parquet (clean) returns the validated df; no quarantine written."""
        from volforecast.validate import validate_asset

        df = _load_fixture("crypto_sample.parquet")
        quarantine_dir = tmp_path / "quarantine"

        result = validate_asset(df, asset_class="crypto", quarantine_dir=quarantine_dir)

        assert result is not None
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        # No quarantine file should have been written
        quarantine_files = list(quarantine_dir.glob("*.csv")) if quarantine_dir.exists() else []
        assert len(quarantine_files) == 0, (
            f"No quarantine file expected for clean data, but found: {quarantine_files}"
        )

    def test_validate_asset_clean_equity_passes(self, tmp_path: Path) -> None:
        """A clean equity DataFrame passes validate_asset without writing a quarantine file."""
        import exchange_calendars as xcals

        from volforecast.validate import validate_asset

        xnys = xcals.get_calendar("XNYS")
        sessions = xnys.sessions_in_range("2022-01-03", "2022-01-14")
        idx = sessions.tz_localize("UTC")
        n = len(idx)
        # Vary close values to avoid triggering stale_row_check (threshold 95% unique).
        df = pd.DataFrame(
            {
                "open": [150.0 + i * 0.3 for i in range(n)],
                "high": [155.0 + i * 0.3 for i in range(n)],
                "low": [148.0 + i * 0.3 for i in range(n)],
                "close": [152.0 + i * 0.5 for i in range(n)],
                "volume": [1_000_000.0 + i * 1000 for i in range(n)],
            },
            index=idx,
        )
        df.index.name = "date"
        quarantine_dir = tmp_path / "quarantine"

        result = validate_asset(df, asset_class="equity", quarantine_dir=quarantine_dir)

        assert result is not None
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        quarantine_files = list(quarantine_dir.glob("*.csv")) if quarantine_dir.exists() else []
        assert len(quarantine_files) == 0, (
            f"No quarantine file expected for clean equity data, but found: {quarantine_files}"
        )


# ---------------------------------------------------------------------------
# validate_and_quarantine: low-level schema helper (from Plan 01)
# ---------------------------------------------------------------------------


class TestValidateAndQuarantine:
    """validate_and_quarantine writes a quarantine CSV and re-raises on schema failure."""

    def test_validate_and_quarantine_writes_csv_on_failure(self, tmp_path: Path) -> None:
        """A schema-violating DataFrame writes a CSV and re-raises SchemaErrors."""
        import pandera.errors

        from volforecast.validate import crypto_ohlcv_schema, validate_and_quarantine

        dates = pd.date_range("2022-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [-1.0, 40000.0, 40000.0],  # negative price on row 0
                "high": [41000.0, 41000.0, 41000.0],
                "low": [39000.0, 39000.0, 39000.0],
                "close": [40500.0, 40500.0, 40500.0],
                "volume": [5000.0, 5000.0, 5000.0],
            },
            index=dates,
        )
        df.index.name = "date"
        q_path = tmp_path / "quarantine.csv"

        with pytest.raises(pandera.errors.SchemaErrors):
            validate_and_quarantine(df, crypto_ohlcv_schema, q_path)

        assert q_path.exists(), "Quarantine CSV must be written on schema failure"
        q_df = pd.read_csv(q_path)
        assert len(q_df) > 0

    def test_validate_and_quarantine_returns_df_on_success(self, tmp_path: Path) -> None:
        """A valid DataFrame is returned; no quarantine file is written."""
        from volforecast.validate import crypto_ohlcv_schema, validate_and_quarantine

        df = _load_fixture("crypto_sample.parquet")
        q_path = tmp_path / "quarantine.csv"

        result = validate_and_quarantine(df, crypto_ohlcv_schema, q_path)

        assert result is not None
        assert not q_path.exists(), "Quarantine CSV must NOT be written for valid data"
