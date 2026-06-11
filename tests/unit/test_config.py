"""Unit tests for volforecast.config path resolution (WR-07).

Config (config/assets.yaml) and data (data/) must resolve from a SINGLE
project root — VOLFORECAST_ROOT when set, otherwise the current working
directory — so the CLI never reads config from one tree while writing data
into another.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _write_assets_yaml(root: Path) -> Path:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "assets.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            {"assets": [{"symbol": "BTC/USDT", "asset_class": "crypto", "exchange": "binance"}]},
            f,
        )
    return config_path


class TestProjectRoot:
    """project_root() resolves from VOLFORECAST_ROOT, falling back to cwd."""

    def test_project_root_defaults_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from volforecast.config import project_root

        monkeypatch.delenv("VOLFORECAST_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)
        assert project_root() == Path.cwd()

    def test_project_root_respects_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from volforecast.config import project_root

        monkeypatch.setenv("VOLFORECAST_ROOT", str(tmp_path))
        assert project_root() == tmp_path


class TestLoadAssetsRootResolution:
    """load_assets resolves the default config path against project_root()."""

    def test_load_assets_missing_config_raises_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no config/assets.yaml under the root, a FileNotFoundError with
        the expected location and remediation hints must be raised (not a bare
        open() failure or a silent read from the installed package's tree)."""
        from volforecast.config import load_assets

        monkeypatch.delenv("VOLFORECAST_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)  # empty directory: no config/ here

        with pytest.raises(FileNotFoundError) as excinfo:
            load_assets()

        message = str(excinfo.value)
        assert str(tmp_path / "config" / "assets.yaml") in message
        assert "VOLFORECAST_ROOT" in message

    def test_load_assets_uses_volforecast_root_from_any_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With VOLFORECAST_ROOT set, config resolves under it regardless of cwd."""
        from volforecast.config import load_assets

        project = tmp_path / "project"
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        _write_assets_yaml(project)

        monkeypatch.setenv("VOLFORECAST_ROOT", str(project))
        monkeypatch.chdir(elsewhere)

        assets = load_assets()
        assert len(assets) == 1
        assert assets[0]["symbol"] == "BTC/USDT"


class TestDataPathsShareRoot:
    """raw_path/processed_path defaults resolve under the SAME root as config."""

    def test_default_data_paths_resolve_under_project_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from volforecast.config import processed_path, raw_path

        monkeypatch.setenv("VOLFORECAST_ROOT", str(tmp_path))
        asset = {"symbol": "BTC/USDT", "asset_class": "crypto"}

        assert raw_path(asset) == tmp_path / "data" / "raw" / "crypto" / "BTC-USD.parquet"
        assert (
            processed_path(asset) == tmp_path / "data" / "processed" / "crypto" / "BTC-USD.parquet"
        )
