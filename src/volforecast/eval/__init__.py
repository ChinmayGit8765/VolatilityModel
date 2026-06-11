"""Evaluation library for VolForecast.

This package provides the canonical evaluation utilities shared by all phases:
- metrics.py: QLIKE, RMSE, MAE in a single importable module
- harness.py: Walk-forward splitter with purge and embargo

Import directly from sub-modules:
    from volforecast.eval.metrics import qlike, rmse, mae
    from volforecast.eval.harness import walk_forward_splits, WalkForwardSplit
"""
