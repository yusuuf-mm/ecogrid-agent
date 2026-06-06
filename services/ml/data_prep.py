"""
services/ml/data_prep.py

Data loading and synthetic data generation for the solar forecast model.

Real source: NREL NSRDB CSVs at ``data/raw/nrel_solar/*.csv`` (see
``data/raw/README.md``). Until those are wired in, training and tests
use ``generate_synthetic_data`` so the pipeline is end-to-end runnable.

Column schema (must match between real and synthetic data):
    timestamp           str    ISO datetime, e.g. "2022-07-15 13:00:00"
    hour                int    0-23
    month               int    1-12
    temperature_c       float  degrees Celsius
    cloud_cover_pct     float  0-100
    solar_irradiance    float  W/m^2 (GHI)
    solar_generation_kw float  target, derived from irradiance + capacity
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "hour",
    "month",
    "temperature_c",
    "cloud_cover_pct",
    "solar_irradiance",
    "solar_generation_kw",
]

SYSTEM_CAPACITY_KW = 500.0
PERFORMANCE_RATIO = 0.18
STC_IRRADIANCE_WM2 = 1000.0


def generate_synthetic_data(n_days: int = 365) -> pd.DataFrame:
    """Generate a deterministic synthetic hourly solar generation dataset.

    The signal is a combination of:
    - a diurnal envelope (zero outside 6am-7pm, peak at solar noon)
    - a seasonal envelope (summer ~1.6x winter)
    - multiplicative cloud noise (0.5-1.0)
    - per-day random amplitude jitter

    Generation is computed from irradiance via a simple linear model
    (kW = irradiance / STC * capacity * performance_ratio) plus Gaussian
    sensor noise. The relationship is exactly the one the model will be
    asked to learn in production, so a well-fit model should achieve
    R^2 > 0.85 on a held-out split.
    """
    rng = np.random.default_rng(seed=42)
    n_hours = n_days * 24
    timestamps = pd.date_range("2023-01-01", periods=n_hours, freq="h")

    hour = timestamps.hour.to_numpy()
    month = timestamps.month.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    diurnal = np.cos(2 * np.pi * (hour - 13) / 24)
    diurnal = np.clip(diurnal, 0.0, None)
    diurnal[hour < 6] = 0.0
    diurnal[hour >= 19] = 0.0
    diurnal /= diurnal.max()

    seasonal = 0.55 + 0.45 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)

    day_idx = np.repeat(np.arange(n_days), 24)
    daily_amplitude = rng.uniform(0.85, 1.15, size=n_days)
    amplitude = daily_amplitude[day_idx]

    cloud_cover_pct = np.clip(
        rng.normal(loc=40.0, scale=25.0, size=n_hours),
        0.0,
        100.0,
    )
    cloud_factor = 1.0 - 0.75 * (cloud_cover_pct / 100.0)

    clear_sky_irradiance = 950.0 * diurnal * seasonal
    solar_irradiance = clear_sky_irradiance * cloud_factor * amplitude
    solar_irradiance = np.clip(solar_irradiance, 0.0, 1100.0)

    temperature_c = (
        12.0
        + 14.0 * seasonal
        + 6.0 * diurnal
        + rng.normal(0.0, 2.0, size=n_hours)
        - 0.1 * cloud_cover_pct
    )

    clean_generation = (
        solar_irradiance
        / STC_IRRADIANCE_WM2
        * SYSTEM_CAPACITY_KW
        * PERFORMANCE_RATIO
    )
    sensor_noise = rng.normal(0.0, 4.0, size=n_hours)
    solar_generation_kw = np.clip(clean_generation + sensor_noise, 0.0, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "month": month,
            "temperature_c": np.round(temperature_c, 2),
            "cloud_cover_pct": np.round(cloud_cover_pct, 2),
            "solar_irradiance": np.round(solar_irradiance, 2),
            "solar_generation_kw": np.round(solar_generation_kw, 3),
        }
    )
    return df


def load_and_prepare(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load training data from a CSV or fall back to synthetic data.

    Args:
        csv_path: path to a CSV with the columns listed in this module's
            docstring. If ``None``, ``generate_synthetic_data(365)`` is
            used instead.
    """
    if csv_path is None:
        return generate_synthetic_data(n_days=365)

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data CSV not found: {path}")

    df = pd.read_csv(path)
    _validate_columns(df)

    if "solar_generation_kw" not in df.columns:
        df = _derive_generation_target(df)
    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["timestamp"]).dt.month

    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Training data is missing required columns: {missing}. "
            f"See services/ml/data_prep.py for the schema."
        )


def _derive_generation_target(df: pd.DataFrame) -> pd.DataFrame:
    """If raw NSRDB data is loaded without a generation target, derive it
    from irradiance using the standard linear model. Useful for the
    real-data path until we have co-located meter logs."""
    df = df.copy()
    df["solar_generation_kw"] = (
        df["solar_irradiance"]
        / STC_IRRADIANCE_WM2
        * SYSTEM_CAPACITY_KW
        * PERFORMANCE_RATIO
    )
    return df
