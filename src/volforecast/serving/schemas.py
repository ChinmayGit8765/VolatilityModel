"""Pydantic v2 request/response schemas for the VolForecast serving API.

Exports:
    AssetForecast  — per-asset forecast with metadata
    ForecastResponse — wraps a list of AssetForecast
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class AssetForecast(BaseModel):
    """Single-asset next-day volatility forecast.

    Fields
    ------
    asset : str
        Asset symbol (e.g. "BTC-USD", "SPY").
    forecast_var : float
        Predicted next-day realized VARIANCE (decimal log-return units, ~1e-4 for equity).
    forecast_vol : float
        Predicted next-day realized VOLATILITY = sqrt(forecast_var).
        The caller is responsible for computing this correctly.
    horizon_days : int
        Forecast horizon in calendar days (default 1 — next-day forecast).
    as_of_date : date
        The as-of date for the most recent feature row used in the prediction.
    model_version : str
        Concrete MLflow model registry version number (e.g. "3").
    alias : str
        MLflow model alias used at load time (e.g. "champion").
    """

    asset: str
    forecast_var: float
    forecast_vol: float
    horizon_days: int = 1
    as_of_date: date
    model_version: str
    alias: str


class ForecastResponse(BaseModel):
    """Response envelope for /forecast and /forecast/{symbol} endpoints.

    Fields
    ------
    forecasts : list[AssetForecast]
        One entry per requested asset (5 entries for /forecast, 1 for /forecast/{symbol}).
    generated_at : str
        ISO-8601 UTC timestamp when the response was generated (e.g. "2026-06-11T10:00:00Z").
    """

    forecasts: list[AssetForecast]
    generated_at: str
