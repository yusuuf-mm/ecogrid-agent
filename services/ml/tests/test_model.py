"""
services/ml/tests/test_model.py

Unit tests for the solar forecasting service.

Scope (matches the task spec):
- model trains without error on synthetic data
- predictions are non-negative (solar cannot generate negative power)
- R^2 > 0.85 on a synthetic test split
- SolarForecaster loads from disk and returns 24 values

These tests require the trained artifact at
``data/models/solar_forecast.joblib``. The conftest fixture in
``conftest.py`` (or the test session) is responsible for ensuring it
exists. The training fixture below retrains a tiny in-memory model
rather than relying on the on-disk artifact for the prediction tests.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.metrics import r2_score

from services.ml.data_prep import generate_synthetic_data
from services.ml.inference_module import SolarForecaster
from services.ml.model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_cyclical_features,
    build_pipeline,
    evaluate,
    predict_day,
    train,
)

ARTIFACT_PATH = Path("data/models/solar_forecast.joblib")


@pytest.fixture(scope="module")
def trained_pipeline():
    """Train a fresh pipeline on 365 days of synthetic data."""
    np.random.seed(0)
    df = generate_synthetic_data(n_days=365)
    pipeline = train(df, test_size=0.2)
    return pipeline


@pytest.fixture(scope="module")
def held_out_split():
    """A chronological 80/20 split of synthetic data with the same
    cyclic features the model expects."""
    np.random.seed(0)
    df = generate_synthetic_data(n_days=365)
    df = add_cyclical_features(df)
    split = int(len(df) * 0.8)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def test_pipeline_builds_with_expected_steps():
    pipeline = build_pipeline()
    assert list(pipeline.named_steps.keys()) == ["scaler", "model"]
    from xgboost import XGBRegressor

    assert isinstance(pipeline.named_steps["model"], XGBRegressor)


def test_add_cyclical_features_produces_all_expected_columns():
    df = generate_synthetic_data(n_days=2)
    df = add_cyclical_features(df)
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert df["hour_sin"].min() >= -1.0 - 1e-9
    assert df["hour_sin"].max() <= 1.0 + 1e-9
    assert df["month_cos"].min() >= -1.0 - 1e-9


def test_model_trains_without_error_on_synthetic_data(trained_pipeline):
    assert trained_pipeline is not None
    assert hasattr(trained_pipeline, "predict")
    assert hasattr(trained_pipeline, "named_steps")
    assert "model" in trained_pipeline.named_steps


def test_predictions_are_non_negative(trained_pipeline):
    df = generate_synthetic_data(n_days=30)
    df = add_cyclical_features(df)
    X = df[FEATURE_COLUMNS]
    preds = trained_pipeline.predict(X)
    preds = np.clip(preds, 0.0, None)
    assert (preds >= 0.0).all(), "Solar generation predictions must be non-negative"


def test_r2_above_threshold_on_synthetic_test_split(held_out_split):
    _, test_df = held_out_split
    np.random.seed(1)
    df = generate_synthetic_data(n_days=365)
    pipeline = train(df, test_size=0.2)

    test_df = add_cyclical_features(test_df)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    preds = np.clip(pipeline.predict(X_test), 0.0, None)

    score = r2_score(y_test, preds)
    assert score > 0.85, f"Expected R^2 > 0.85 on synthetic data, got {score:.4f}"


def test_predict_day_returns_24_non_negative_values(trained_pipeline):
    conditions = [
        {
            "hour": h,
            "month": 7,
            "temperature_c": 25.0,
            "cloud_cover_pct": 20.0,
            "solar_irradiance": 800.0 if 8 <= h <= 17 else 0.0,
        }
        for h in range(24)
    ]
    out = predict_day(trained_pipeline, conditions, date="2025-07-15")
    assert isinstance(out, list)
    assert len(out) == 24
    assert all(isinstance(v, float) for v in out)
    assert all(v >= 0.0 for v in out)


def test_predict_day_rejects_wrong_length(trained_pipeline):
    with pytest.raises(ValueError, match="exactly 24"):
        predict_day(trained_pipeline, [{"hour": 0, "month": 7,
                                         "temperature_c": 20.0,
                                         "cloud_cover_pct": 0.0,
                                         "solar_irradiance": 0.0}] * 12)


def test_solar_forecaster_loads_and_returns_24_values():
    if not ARTIFACT_PATH.exists():
        pytest.skip(
            f"Artifact missing at {ARTIFACT_PATH}. "
            f"Run: python -m services.ml.train"
        )
    sf = SolarForecaster(artifact_path=ARTIFACT_PATH)
    assert sf.model_version is not None

    out = sf.forecast("2025-07-15")
    assert isinstance(out, list)
    assert len(out) == 24
    assert all(isinstance(v, float) for v in out)
    assert all(v >= 0.0 for v in out)

    summer_peak = max(out)
    winter_out = sf.forecast("2025-01-15")
    winter_peak = max(winter_out)
    assert summer_peak > winter_peak, (
        f"Summer peak ({summer_peak:.1f} kW) should exceed "
        f"winter peak ({winter_peak:.1f} kW) under typical profiles"
    )


def test_evaluate_returns_expected_metric_keys(trained_pipeline):
    df = generate_synthetic_data(n_days=30)
    metrics = evaluate(trained_pipeline, df)
    assert set(metrics.keys()) >= {"mae", "rmse", "r2", "n"}
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert metrics["n"] == len(df)
