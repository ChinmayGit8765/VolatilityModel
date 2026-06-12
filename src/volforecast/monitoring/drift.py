"""Evidently 0.7 distribution-drift adapter.

PURPOSE
-------
Compare live feature + prediction distributions against a frozen training
reference snapshot and write dated JSON + HTML reports to ``data/monitoring/``.

LOCKED ANTI-PATTERN
-------------------
This module is **report/log only**.  It contains NO call that:
  - flips an MLflow alias (``set_registered_model_alias``)
  - sets or clears a retrain flag
  - triggers any downstream promotion action

Distribution drift is an observability signal, not an action trigger.
Only rolling-QLIKE performance degradation (``monitoring.performance``) is
wired to the retrain flag.

FROZEN REFERENCE CONTRACT
--------------------------
The reference DataFrame is read from the frozen snapshot created by
``scripts/create_reference_snapshot.py``.  It is passed in by the caller
and is NEVER written or modified here.  The snapshot changes only when a
new champion is promoted; the drift check always reads the existing frozen
file (see RESEARCH Pitfall 6).

EVIDENTLY API
-------------
Uses Evidently 0.7.21 current API only:
  - ``Report([DataDriftPreset()])``  (NOT legacy ColumnMapping / DatasetColumns)
  - ``report.run(current_dataset, reference_dataset)`` → returns a result object
  - ``result.save_html(path)``, ``result.save_json(path)``, ``result.dict()``
    called on the RESULT, NOT on the Report template (Pitfall 1).
  - ``DataDefinition(numerical_columns=[...])`` passed explicitly so Evidently
    never mis-categorises string/datetime columns (Pitfall 2).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Non-feature columns to exclude from drift analysis
# ---------------------------------------------------------------------------
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


def select_numerical_columns(df: pd.DataFrame) -> list[str]:
    """Return numerical-dtype column names, excluding known non-feature columns.

    Use this helper to build the ``numerical_columns`` argument for
    :func:`run_distribution_drift` so that string/datetime/categorical columns
    are never passed to ``DataDefinition`` (Evidently Pitfall 2).

    Args:
        df: Input DataFrame that may contain a mix of feature and metadata columns.

    Returns:
        Ordered list of column names with a numerical dtype that are NOT in the
        known non-feature exclusion list.
    """
    return [col for col in df.select_dtypes(include=["number"]).columns if col not in _EXCLUDE_COLS]


def run_distribution_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_columns: list[str],
    output_dir: Path,
    date_str: str,
) -> dict:
    """Run an Evidently distribution-drift report and write dated output files.

    Follows RESEARCH Pattern 2 exactly:
      - ``DataDefinition(numerical_columns=numerical_columns)`` — explicit schema.
      - ``Dataset.from_pandas(df, data_definition=schema)`` — wraps both frames.
      - ``report.run(current_ds, reference_ds)`` — current first, reference second.
      - ``result.save_html(...)`` / ``result.save_json(...)`` — on the RESULT object,
        NOT on the Report template (Pitfall 1).

    Args:
        reference_df:       Frozen training reference snapshot.  Pass only the
                            numerical feature columns (use :func:`select_numerical_columns`
                            or slice before calling).  Never modified here.
        current_df:         Live observation window to compare against the reference.
                            Must contain at least the columns in ``numerical_columns``.
        numerical_columns:  Explicit list of columns to include in the drift analysis.
                            Non-numeric / non-feature columns must be excluded by
                            the caller (see :func:`select_numerical_columns`).
        output_dir:         Directory where the dated HTML and JSON files are written.
                            Created automatically if it does not exist.
        date_str:           Date string used as the filename prefix (e.g. ``"2026-06-12"``).
                            Output files: ``{date_str}_drift.html`` and
                            ``{date_str}_drift.json``.

    Returns:
        ``result.dict()`` — a nested dictionary containing per-column drift statistics,
        p-values, drift shares, and the overall ``dataset_drift`` boolean flag.
        Suitable for programmatic inspection, logging, and downstream alerting.

    Raises:
        ImportError: if ``evidently>=0.7.21`` is not installed.
        ValueError:  if ``numerical_columns`` is empty.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    if not numerical_columns:
        raise ValueError(
            "numerical_columns must not be empty. "
            "Use select_numerical_columns() to derive the list from your DataFrame."
        )

    # Build schema: always explicit — Evidently Pitfall 2
    schema = DataDefinition(numerical_columns=numerical_columns)

    # Wrap DataFrames (only pass the columns we declared in the schema)
    ref_df_sliced = reference_df[numerical_columns].copy()
    cur_df_sliced = current_df[numerical_columns].copy()

    ref_dataset = Dataset.from_pandas(ref_df_sliced, data_definition=schema)
    cur_dataset = Dataset.from_pandas(cur_df_sliced, data_definition=schema)

    # Build and run — result is a SEPARATE object from the Report template
    report = Report([DataDriftPreset()])
    result = report.run(cur_dataset, ref_dataset)  # (current, reference)

    # Write outputs — called on RESULT, NOT on report (Evidently Pitfall 1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save_html(str(output_dir / f"{date_str}_drift.html"))
    result.save_json(str(output_dir / f"{date_str}_drift.json"))

    # Return programmatic summary for logging / downstream inspection
    return result.dict()
