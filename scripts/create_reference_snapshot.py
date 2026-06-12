"""Create a frozen Evidently reference snapshot from training feature data.

FREEZE CONTRACT
---------------
This script is a one-time (or one-per-champion-version) setup tool.  Once a
reference snapshot exists for a given champion model version it is NEVER
overwritten automatically — doing so would silently move the drift baseline and
make future drift scores meaningless (RESEARCH Pitfall 6).

The snapshot file is named  ``data/monitoring/reference/{version}_reference.parquet``
where ``{version}`` is the integer version number of the current ``@champion``
model in the MLflow registry (e.g.  ``3_reference.parquet``).

When a new champion is promoted (Plan 04) a *new* snapshot file is created
(new version → new filename).  The old file is kept for audit.

Run once from the project root after Phase 3 training completes:
    uv run python scripts/create_reference_snapshot.py

Requires:
    - MLflow server reachable (MLFLOW_TRACKING_URI or default localhost:5000)
    - data/features/{crypto,equity}/{slug}.parquet  (Phase 2 output)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Allow import from src/ when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volforecast.config import load_assets, project_root, symbol_slug

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Non-feature columns to EXCLUDE from the drift reference                     #
# --------------------------------------------------------------------------- #
_EXCLUDE_COLS = frozenset(
    {
        "asset",
        "symbol",
        "date",
        "as_of_date",
        "model_version",
        "model_alias",
        "alias",
    }
)


def _resolve_champion_version() -> str:
    """Return the version string of the current @champion model from MLflow.

    Raises:
        RuntimeError: if MLflow registry is unreachable or alias not found.
    """
    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        mv = client.get_model_version_by_alias("volforecast-lgbm", "champion")
        return mv.version
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not resolve @champion version from MLflow registry. "
            "Ensure the MLflow server is running and the champion alias is set. "
            f"Original error: {exc}"
        ) from exc


def _features_path(asset: dict, data_root: Path) -> Path:
    slug = symbol_slug(asset["symbol"])
    return data_root / "features" / asset["asset_class"] / f"{slug}.parquet"


def _select_numerical_columns(df: pd.DataFrame) -> list[str]:
    """Return float/int columns, excluding known non-feature columns."""
    return [col for col in df.select_dtypes(include=["number"]).columns if col not in _EXCLUDE_COLS]


def create_reference_snapshot(
    data_root: Path | None = None,
    config_path: Path | None = None,
    champion_version: str | None = None,
) -> Path:
    """Concatenate all training feature parquets into a frozen reference snapshot.

    Args:
        data_root:        Root of the data directory.  Defaults to <project_root>/data.
        config_path:      Path to assets.yaml.  Defaults to <project_root>/config/assets.yaml.
        champion_version: Model version string to embed in the snapshot filename.
                          If ``None`` the current @champion alias is resolved from
                          the MLflow registry.

    Returns:
        Path of the written snapshot file (or the pre-existing file if already frozen).

    Raises:
        FileNotFoundError: if any required feature parquet is missing.
        RuntimeError:      if MLflow alias resolution fails and champion_version is None.
    """
    root = project_root()
    data_root = data_root or (root / "data")
    config_path = config_path or (root / "config" / "assets.yaml")

    # --- Resolve champion version ---
    if champion_version is None:
        log.info("Resolving @champion version from MLflow registry...")
        champion_version = _resolve_champion_version()
    log.info("Champion version: %s", champion_version)

    # --- Check for existing snapshot (idempotent freeze) ---
    reference_dir = data_root / "monitoring" / "reference"
    snapshot_path = reference_dir / f"{champion_version}_reference.parquet"

    if snapshot_path.exists():
        log.info(
            "Snapshot already exists for version %s at %s — NOT overwriting "
            "(frozen reference contract).",
            champion_version,
            snapshot_path,
        )
        return snapshot_path

    # --- Load all feature parquets ---
    assets = load_assets(config_path)
    log.info("Loaded %d assets: %s", len(assets), [a["symbol"] for a in assets])

    frames: list[pd.DataFrame] = []
    for asset in assets:
        feat_path = _features_path(asset, data_root)
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Feature parquet not found: {feat_path}\n"
                "Ensure Phase 2 feature generation has run successfully before "
                "creating the reference snapshot."
            )
        df = pd.read_parquet(feat_path)
        log.info("Loaded %s: %d rows x %d cols", asset["symbol"], *df.shape)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined: %d total rows before column selection", len(combined))

    # --- Keep only numerical feature columns ---
    numerical_cols = _select_numerical_columns(combined)
    if not numerical_cols:
        raise RuntimeError(
            "No numerical feature columns found after excluding non-feature columns. "
            f"Available columns: {list(combined.columns)}"
        )
    reference_df = combined[numerical_cols]
    reference_df = reference_df.dropna(how="all")
    log.info(
        "Reference snapshot: %d rows x %d numerical cols  (NaN-all rows dropped)",
        *reference_df.shape,
    )
    log.info("Columns: %s", numerical_cols)

    # --- Write snapshot ---
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_df.to_parquet(snapshot_path, index=False)
    log.info("Frozen reference snapshot written to: %s", snapshot_path)
    return snapshot_path


def main() -> None:
    try:
        path = create_reference_snapshot()
        print(f"Reference snapshot: {path}")
    except FileNotFoundError as exc:
        log.error("Missing input data: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        log.error("Setup error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
