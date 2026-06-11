"""VolForecast models package.

Exports the baseline forecasters used in the walk-forward evaluation harness.

Phase 02 baselines:
    - EWMA: RiskMetrics EWMA(lambda=0.94) variance forecaster (Plan 02-02)
    - GARCH: GARCH(1,1) via arch library (Plan 02-03)
    - HARV: HAR-RV via OLS (Plan 02-03)

Phase 03 ML models:
    - LightGBM regressor (Plan 03)
"""

from volforecast.models.ewma import EWMA
from volforecast.models.garch import GARCH
from volforecast.models.har_rv import HARRV

__all__ = ["EWMA", "GARCH", "HARRV"]
