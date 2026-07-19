"""Regression tests for the `python -m volforecast.cli` execution path.

Bug history: cli.py had no ``if __name__ == "__main__"`` guard, so the daily
flow's ingest task (`python -m volforecast.cli`) imported the module, did
nothing, and exited 0 — a silent no-op ingest that let the flow "complete"
on stale data. These tests pin the two properties that make that impossible:

1. ``-m volforecast.cli`` with NO arguments must exit non-zero (argparse
   requires a subcommand) — a bare invocation can never look like success.
2. The daily flow's ingest task must pass the ``ingest`` subcommand.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent


def test_cli_module_without_args_exits_nonzero() -> None:
    """`python -m volforecast.cli` (no subcommand) must fail loudly, not no-op."""
    result = subprocess.run(
        [sys.executable, "-m", "volforecast.cli"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0, (
        "bare `-m volforecast.cli` exited 0 — the __main__ guard or required "
        "subcommand is missing, so the daily flow's ingest could silently no-op"
    )
    combined = result.stdout + result.stderr
    assert "usage" in combined.lower() or "required" in combined.lower()


def test_cli_module_has_main_guard() -> None:
    """cli.py must invoke main() under `if __name__ == "__main__"`."""
    src = (REPO_ROOT / "src" / "volforecast" / "cli.py").read_text(encoding="utf-8")
    assert '__name__ == "__main__"' in src


def test_daily_flow_ingest_task_passes_ingest_subcommand() -> None:
    """The flow's ingest task must invoke the CLI WITH the `ingest` subcommand."""
    src = (REPO_ROOT / "pipelines" / "daily_pipeline.py").read_text(encoding="utf-8")
    assert '"-m", "volforecast.cli", "ingest"' in src, (
        "ingest_validate_task no longer passes the `ingest` subcommand — "
        "the daily flow would silently skip ingestion"
    )
