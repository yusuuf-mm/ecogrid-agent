"""services/ml/inference package.

Exposes the runtime forecaster (``SolarForecaster``) and the
contract-bound bridge function (``forecast_solar_generation``) that
the LangChain agent will wrap as a tool in Phase 3.
"""
from __future__ import annotations

from services.ml.inference import predictor
from services.ml.inference_module import SolarForecaster

__all__ = ["SolarForecaster", "predictor", "forecast_solar_generation"]


def forecast_solar_generation(features):  # type: ignore[no-untyped-def]
    """Thin wrapper used by the agent layer. Delegates to the contract
    bridge in ``predictor.py`` so the contract surface stays separate
    from the runtime class.
    """
    return predictor.forecast_solar_generation(features)
