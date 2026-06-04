"""
services/ml/model.py

Model definition, training, evaluation, and prediction for the solar
forecasting service. The trained artifact is a scikit-learn Pipeline
(``StandardScaler`` -> ``XGBRegressor``) saved with joblib.

The pipeline owns the feature ordering, so callers must hand it a
DataFrame whose columns match ``FEATURE_COLUMNS`` in the same order.
``predict_day`` and ``inference.SolarForecaster`` are responsible for
constructing that DataFrame from a list of hourly condition dicts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    "hour",
    "month",
    "temperature_c",
    "cloud_cover_pct",
    "solar_irradiance",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]
TARGET_COLUMN = "solar_generation_kw"

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "n_jobs": 4,
    "random_state": 42,
}


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos cyclical encodings for hour and month.

    Tree models do not natively learn that hour 23 is adjacent to hour 0,
    so we add the sin/cos pair. Identical transforms are applied at
    training and inference time — see ``inference.SolarForecaster``.
    """
    df = df.copy()
    hour_rad = 2 * np.pi * df["hour"] / 24.0
    month_rad = 2 * np.pi * (df["month"] - 1) / 12.0
    df["hour_sin"] = np.sin(hour_rad)
    df["hour_cos"] = np.cos(hour_rad)
    df["month_sin"] = np.sin(month_rad)
    df["month_cos"] = np.cos(month_rad)
    return df


def build_pipeline() -> Pipeline:
    """Construct an unfitted Pipeline with the project's XGBoost config."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(**XGB_PARAMS)),
        ]
    )


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Training DataFrame is missing target column '{TARGET_COLUMN}'."
        )
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Training DataFrame is missing feature columns: {missing}. "
            f"Did you call add_cyclical_features() first?"
        )
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return X, y


def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Pipeline:
    """Train a fresh pipeline on a DataFrame.

    The DataFrame must already include the cyclical features — call
    ``add_cyclical_features`` first. Train/test split is a chronological
    80/20 because solar data is time-series and a random split would
    leak future information into the training set.
    """
    df = add_cyclical_features(df)
    split = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    X_train, y_train = _split_features_target(train_df)
    X_test, y_test = _split_features_target(test_df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, test_df)
    logger.info(
        f"Trained solar forecast model. "
        f"train_rows={len(train_df)} test_rows={len(test_df)} "
        f"mae={metrics['mae']:.3f} rmse={metrics['rmse']:.3f} r2={metrics['r2']:.4f}"
    )
    return pipeline


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict:
    """Score a fitted pipeline against a labeled DataFrame."""
    df = add_cyclical_features(df)
    X, y = _split_features_target(df)
    preds = model.predict(X)
    preds = np.clip(preds, 0.0, None)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "r2": float(r2_score(y, preds)),
        "n": int(len(y)),
    }


def predict_day(
    model: Pipeline,
    conditions: list[dict],
    date: Optional[str] = None,
) -> list[float]:
    """Predict 24 hourly solar generation values for a single day.

    Args:
        model: a fitted Pipeline (the artifact loaded from disk).
        conditions: list of 24 dicts, one per hour 0..23. Each dict
            must contain ``temperature_c``, ``cloud_cover_pct``,
            ``solar_irradiance``. May also contain ``hour``. If
            ``hour`` is missing, the list index is used.
        date: optional ISO date. If provided and ``hour`` is missing
            from a condition dict, hour-of-day is derived from the
            date's seasonal profile. Currently unused beyond logging.

    Returns:
        list of 24 floats (kW per hour). Non-negative.
    """
    if len(conditions) != 24:
        raise ValueError(
            f"predict_day expects exactly 24 hourly conditions, got {len(conditions)}."
        )

    rows = []
    for idx, cond in enumerate(conditions):
        if "hour" in cond:
            hour = int(cond["hour"])
        else:
            hour = idx
        if "month" in cond:
            month = int(cond["month"])
        else:
            month = 7
        if "temperature_c" not in cond:
            raise ValueError(
                f"conditions[{idx}] is missing required key 'temperature_c'."
            )
        if "cloud_cover_pct" not in cond:
            raise ValueError(
                f"conditions[{idx}] is missing required key 'cloud_cover_pct'."
            )
        if "solar_irradiance" not in cond:
            raise ValueError(
                f"conditions[{idx}] is missing required key 'solar_irradiance'."
            )
        rows.append(
            {
                "hour": hour,
                "month": month,
                "temperature_c": float(cond["temperature_c"]),
                "cloud_cover_pct": float(cond["cloud_cover_pct"]),
                "solar_irradiance": float(cond["solar_irradiance"]),
            }
        )

    df = pd.DataFrame(rows)
    df = add_cyclical_features(df)
    preds = model.predict(df[FEATURE_COLUMNS])
    preds = np.clip(preds, 0.0, None)

    if date is not None:
        logger.debug(f"predict_day: date={date} peak_kw={float(preds.max()):.1f}")

    return [float(v) for v in preds]
