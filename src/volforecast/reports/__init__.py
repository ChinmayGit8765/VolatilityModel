"""VolForecast reports package.

Provides report generators that wire the evaluation pipeline (target,
harness, baselines, metrics) into committed human-readable artifacts.

Phase 02 baseline report:
    - baseline.py: generate_baseline_report() — per-asset EWMA metrics table
      (extended in Plan 02-03 to add GARCH and HAR-RV columns)
"""
