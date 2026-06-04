# CLAUDE.md — services/ml

You are working on the `feat/ml-forecasting` branch of EcoGrid-Agent.

Your scope is `services/ml/` and `data/` (raw data, processed data, model artifacts).
Do not touch any other service directory.

Read `CONTEXT.md` and `shared/contracts.py` before writing any code.
The `SolarForecastResult` type you return is defined in `shared.contracts`.

This branch is independent. You do not depend on the solver, the API, or the
RAG pipeline. Build and test it in complete isolation.

---

## What This Service Is

A trained XGBoost model that predicts solar generation (kW) for each of the
24 hours in a day, given date and weather conditions. It runs inside the Celery
worker at inference time — one prediction per optimization request.

The model artifact (`solar_forecast.joblib`) is committed to the repo so that
other branches don't need to retrain. Train it once, commit it, done.

---

## What to Build

### `services/ml/config.py`
Settings via `pydantic-settings`:
- `MODEL_PATH: str = "data/models/solar_forecast.joblib"`
- `BATTERY_CAPACITY_KWH: float = 1000.0` (used to cap predictions)

### `services/ml/data_prep.py`

**`generate_synthetic_data(n_days: int = 365) -> pd.DataFrame`**

Generates realistic synthetic solar data. Columns:
- `timestamp` — hourly datetime
- `hour` — int 0-23
- `month` — int 1-12
- `day_of_year` — int 1-365
- `temperature_c` — realistic seasonal range, add Gaussian noise
- `cloud_cover_pct` — 0-100, heavier in winter months, random spikes
- `solar_irradiance` — proportional to `sin(π * (hour-6)/12)` for hours 6-18,
  zero otherwise; scaled by `(1 - cloud_cover_pct/100)` and seasonal factor
- `solar_generation_kw` — target variable; proportional to `solar_irradiance`,
  capped at 450 kW, zero at night, add small noise

The synthetic data must be realistic enough that a model trained on it achieves
R² > 0.85 on a held-out split. If it doesn't, adjust the noise level.

**`load_and_prepare(csv_path: str | None = None) -> pd.DataFrame`**

If `csv_path` is provided, load from disk and standardise column names to match
the synthetic schema. If `csv_path` is None, call `generate_synthetic_data()`.

### `services/ml/model.py`

**Features** (engineer these from the raw columns):
- `hour`, `month`, `day_of_year`
- `hour_sin = sin(2π * hour / 24)`, `hour_cos = cos(2π * hour / 24)` — cyclical
- `month_sin = sin(2π * month / 12)`, `month_cos = cos(2π * month / 12)`
- `temperature_c`, `cloud_cover_pct`, `solar_irradiance`

**Pipeline**: `StandardScaler` → `XGBRegressor`

XGBRegressor params:
```python
n_estimators=300,
max_depth=6,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8,
objective="reg:squarederror",
random_state=42,
```

**`train(df: pd.DataFrame) -> Pipeline`**
80/20 train/test split (shuffle=False — respect time order).
Print MAE, RMSE, R² on the test split before returning the pipeline.

**`evaluate(model: Pipeline, df: pd.DataFrame) -> dict`**
Returns `{"mae": float, "rmse": float, "r2": float}`.

**`predict_day(model: Pipeline, conditions: list[dict]) -> list[float]`**
Input: 24 dicts, each with keys matching the feature columns.
Output: 24 floats (predicted `solar_generation_kw`), clipped to `[0, 450]`.

### `services/ml/train.py`
CLI entry point:
```
python -m services.ml.train [--data-path PATH] [--output-path PATH]
```
- Default output: `data/models/solar_forecast.joblib`
- Saves the fitted Pipeline with `joblib.dump`
- Prints evaluation metrics to stdout

### `services/ml/inference.py`

**Class: `SolarForecaster`**

```python
class SolarForecaster:
    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.model = joblib.load(model_path)  # load once at init

    def forecast(
        self,
        date: str,
        conditions: list[dict] | None = None,
    ) -> SolarForecastResult:
        ...
```

If `conditions` is None, generate typical conditions for that date using
seasonal averages (month from the date, typical temperature, 20% cloud cover).

Returns `SolarForecastResult` from `shared.contracts`.

### `services/ml/tests/test_model.py`
Required tests:
1. `generate_synthetic_data()` returns a DataFrame with the correct columns and
   `len >= 365 * 24` rows.
2. `train()` completes without error and returns a scikit-learn `Pipeline`.
3. All predictions from `predict_day()` are non-negative (solar can't go below 0).
4. R² on a fresh train/test split of synthetic data is > 0.85.
5. `SolarForecaster` loads the saved model and returns a `SolarForecastResult`
   with exactly 24 values in `hourly_forecast_kw`.

---

## Deliverables

- Trained model artifact at `data/models/solar_forecast.joblib` committed to the branch
- Test run showing R² > 0.85 included in the PR description
- `poetry run python -m services.ml.train` works from the repo root without arguments
