"""Generate per-asset feature parquets for all 5 assets.

Reads: data/processed/{asset_class}/{slug}.parquet
Writes: data/features/{asset_class}/{slug}.parquet

Cross-asset wiring (per CONTEXT.md):
- BTC RV_22 joined as a cross-asset feature onto ETH, SPY, AAPL, MSFT
- ETH RV_22 joined as a cross-asset feature onto BTC (mutual crypto cross-asset)

Run from project root:
    uv run python scripts/generate_features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Allow import from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volforecast.config import load_assets, processed_path, symbol_slug
from volforecast.features.pipeline import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def features_path(asset: dict, data_root: Path | None = None) -> Path:
    """Return canonical features parquet path for an asset."""
    root = data_root or (Path(__file__).parent.parent / "data")
    slug = symbol_slug(asset["symbol"])
    return root / "features" / asset["asset_class"] / f"{slug}.parquet"


def main() -> None:
    project_root = Path(__file__).parent.parent
    assets = load_assets(project_root / "config" / "assets.yaml")
    log.info("Loaded %d assets: %s", len(assets), [a["symbol"] for a in assets])

    # --- Step 1: Load all processed DataFrames ---
    raw: dict[str, pd.DataFrame] = {}
    for asset in assets:
        path = processed_path(asset, project_root / "data")
        if not path.exists():
            log.error("Missing processed parquet: %s", path)
            sys.exit(1)
        df = pd.read_parquet(path)
        # Ensure index is named "date" (Phase 1 contract)
        if df.index.name != "date":
            df.index.name = "date"
        raw[asset["symbol"]] = df
        log.info("Loaded %s: %d rows from %s", asset["symbol"], len(df), path)

    # --- Step 2: Build base features (without GARCH) for cross-asset source ---
    # We build BTC features first (no GARCH to avoid slow startup),
    # then use BTC RV as a cross-asset source for other assets.
    log.info("Building base features for cross-asset sources...")

    btc_sym = next(
        (a["symbol"] for a in assets if "BTC" in a["symbol"].upper()), None
    )
    eth_sym = next(
        (a["symbol"] for a in assets if "ETH" in a["symbol"].upper()), None
    )

    btc_base_feats: pd.DataFrame | None = None
    eth_base_feats: pd.DataFrame | None = None

    if btc_sym and btc_sym in raw:
        log.info("Building BTC base features (for cross-asset source)...")
        btc_base_feats = build_features(raw[btc_sym], include_garch=False)
        log.info("BTC base features: %d rows x %d cols", *btc_base_feats.shape)

    if eth_sym and eth_sym in raw:
        log.info("Building ETH base features (for cross-asset source)...")
        eth_base_feats = build_features(raw[eth_sym], include_garch=False)
        log.info("ETH base features: %d rows x %d cols", *eth_base_feats.shape)

    # --- Step 3: Build full features for each asset with GARCH + cross-asset ---
    for asset in assets:
        sym = asset["symbol"]
        slug = symbol_slug(sym)
        asset_class = asset["asset_class"]

        log.info("Building full features for %s (%s)...", sym, asset_class)

        # Build cross_asset_dfs for this target asset
        cross_asset_dfs: dict[str, pd.DataFrame] = {}

        if btc_base_feats is not None and sym != btc_sym:
            # Attach BTC RV_22 as a cross-asset feature for all non-BTC assets
            cross_asset_dfs["BTC"] = btc_base_feats[["rv_22"]].rename(
                columns={"rv_22": "btc_rv22"}
            )

        if eth_base_feats is not None and sym == btc_sym:
            # Attach ETH RV_22 as a cross-asset feature for BTC
            cross_asset_dfs["ETH"] = eth_base_feats[["rv_22"]].rename(
                columns={"rv_22": "eth_rv22"}
            )

        df = raw[sym]
        result = build_features(
            df,
            cross_asset_dfs=cross_asset_dfs if cross_asset_dfs else None,
            include_garch=True,
        )

        out_path = features_path(asset, project_root / "data")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path)
        log.info(
            "Saved %s: %d rows x %d cols -> %s",
            sym,
            *result.shape,
            out_path,
        )

        # Quick sanity check
        expected_cols = {"rv_5", "rv_22", "rv_66", "ewma_var", "garch_cond_var", "parkinson_var", "gk_var"}
        missing = expected_cols - set(result.columns)
        if missing:
            log.error("Missing expected columns for %s: %s", sym, missing)
            sys.exit(1)
        log.info("  Column check passed. Total non-NaN garch_cond_var: %d", result["garch_cond_var"].notna().sum())

    log.info("All 5 feature parquets generated successfully.")


if __name__ == "__main__":
    main()
