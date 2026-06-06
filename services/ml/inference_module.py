"""
services/ml/inference_module.py

Runtime forecaster. ``SolarForecaster`` loads the trained joblib
artifact once and exposes ``forecast(date, conditions) -> list[float]``.

This module is named ``inference_module`` (not ``inference``) to avoid
shadowing the ``services.ml.inference`` package that holds the
contract-bound bridge for the agent layer.

If ``conditions`` is not provided, a typical seasonal profile is
generated for the given date so the rest of the pipeline (the LP
solver) always gets 24 values back, even when the upstream weather
service is unreachable.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from loguru import logger
from sklearn.pipeline import Pipeline

from services.ml.model import add_cyclical_features, predict_day

DEFAULT_ARTIFACT_PATH = "data/models/solar_forecast.joblib"

SUMMER_IRRADIANCE_PROFILE = [
    0, 0, 0, 0, 0, 30, 120, 280, 480, 680, 830, 920, 950, 930, 850, 720, 540, 320, 120, 20, 0, 0, 0, 0
]
WINTER_IRRADIANCE_PROFILE = [
    0, 0, 0, 0, 0, 0, 40, 140, 280, 410, 520, 580, 600, 570, 490, 370, 220, 70, 0, 0, 0, 0, 0, 0
]
SUMMER_TEMP_C = 28.0
WINTER_TEMP_C = 4.0
TYPICAL_CLOUD_COVER_PCT = 35.0


def _seasonal_blend(month: int) -> float:
    """0.0 = midwinter, 1.0 = midsummer. Linear across the year."""
    return float(0.5 - 0.5 * np.cos(2 * np.pi * (month - 1) / 12.0))


def _seasonal_irradiance_profile(month: int) -> list[float]:
    """Linearly interpolate between the summer and winter hourly profiles
    based on month. Endpoints: Dec/Jan=Feb=Winter, Jun/Jul/Aug=Summer.
    """
    blend = _seasonal_blend(month)
    return [
        (1 - blend) * w + blend * s
        for w, s in zip(WINTER_IRRADIANCE_PROFILE, SUMMER_IRRADIANCE_PROFILE)
    ]


def _seasonal_temp_c(month: int) -> float:
    blend = _seasonal_blend(month)
    return (1 - blend) * WINTER_TEMP_C + blend * SUMMER_TEMP_C


def _typical_conditions(date_str: str) -> list[dict]:
    """Generate a typical 24-hour weather profile for the given date.

    Used when no explicit conditions are supplied. Hourly temperature
    gets a small diurnal bump; irradiance follows the seasonal envelope.
    """
    dt = datetime.fromisoformat(date_str)
    month = dt.month
    base_irradiance = _seasonal_irradiance_profile(month)
    base_temp = _seasonal_temp_c(month)

    conditions = []
    for hour in range(24):
        diurnal_temp = 4.0 * np.sin(np.pi * (hour - 6) / 12.0) if 6 <= hour <= 18 else 0.0
        conditions.append(
            {
                "hour": hour,
                "month": month,
                "temperature_c": float(base_temp + diurnal_temp),
                "cloud_cover_pct": TYPICAL_CLOUD_COVER_PCT,
                "solar_irradiance": float(base_irradiance[hour]),
            }
        )
    return conditions


class SolarForecaster:
    """Loads the trained pipeline once and serves hourly forecasts."""

    def __init__(self, artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> None:
        self._artifact_path = Path(artifact_path)
        if not self._artifact_path.exists():
            raise FileNotFoundError(
                f"Solar forecast artifact not found at {self._artifact_path}. "
                f"Train the model first: python -m services.ml.train"
            )
        self._pipeline: Pipeline = joblib.load(self._artifact_path)
        self._model_version = self._artifact_path.stem
        logger.info(
            f"SolarForecaster loaded from {self._artifact_path} "
            f"(version={self._model_version})"
        )

    @property
    def model_version(self) -> str:
        return self._model_version

    def forecast(
        self,
        date: str,
        conditions: Optional[list[dict]] = None,
    ) -> list[float]:
        """Return 24 hourly solar generation values (kW) for ``date``.

        If ``conditions`` is ``None``, typical seasonal averages are used.
        Otherwise ``conditions`` must be a list of 24 dicts with keys
        ``temperature_c``, ``cloud_cover_pct``, ``solar_irradiance`` and
        optionally ``hour`` and ``month``.
        """
        if conditions is None:
            conditions = _typical_conditions(date)
        return predict_day(self._pipeline, conditions, date=date)
