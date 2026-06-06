# services/ml — Solar Forecasting Service

Branch: `feat/ml-forecasting`
Owns: `services/ml/`, `data/models/solar_forecast.joblib`, `data/raw/nrel_solar/`

## Role

Take hourly weather conditions for a target day and produce 24 hourly solar
generation predictions (kW). The output feeds the LP solver as
`SolverConstraints.solar_forecast_kw`.

## Module Layout

```
services/ml/
├── data_prep.py        # synthetic data generator + CSV loader
├── model.py            # Pipeline definition, train/evaluate/predict
├── train.py            # CLI: train and save artifact
├── inference.py        # SolarForecaster — loaded once at worker startup
├── training/           # reserved for future training pipeline
└── inference/
    └── predictor.py    # thin wrapper exposing forecast_solar_generation()
```

Training and inference live in separate modules so the model artifact carries
all fitted preprocessing transforms. Inference never re-implements feature
engineering — it loads the pipeline and calls `predict`.

## Feature Specification

The training DataFrame and the inference input dict must produce identical
feature columns. The pipeline handles scaling; the model module handles
deriving derived columns (cyclical encodings).

| Feature             | Type    | Source                                  |
|---------------------|---------|-----------------------------------------|
| `hour`              | int 0-23| timestamp.hour                          |
| `month`             | int 1-12| timestamp.month                         |
| `temperature_c`     | float   | weather input                           |
| `cloud_cover_pct`   | float   | weather input, 0-100                    |
| `solar_irradiance`  | float   | weather input, W/m²                     |
| `hour_sin`          | float   | sin(2π · hour / 24)                     |
| `hour_cos`          | float   | cos(2π · hour / 24)                     |
| `month_sin`         | float   | sin(2π · month / 12)                    |
| `month_cos`         | float   | cos(2π · month / 12)                    |

Cyclical encodings are required: tree models do not natively learn that
hour 23 is adjacent to hour 0. Sine/cosine pairs close the loop.

## Model

scikit-learn `Pipeline`:
- `StandardScaler` — features are on different scales (irradiance ~1000,
  hour ~0-23, cloud_cover ~0-100).
- `XGBRegressor` — `n_estimators=200`, `max_depth=6`, `learning_rate=0.05`,
  `objective="reg:squarederror"`.

Artifact: `data/models/solar_forecast.joblib` (joblib dump of the fitted
Pipeline). The artifact travels with the repo, so a fresh checkout can run
inference without retraining. Retraining is a separate CLI: `python -m
services.ml.train`.

## Synthetic Data

Until NREL NSRDB data is wired in, training and tests use
`generate_synthetic_data(n_days=365)`. The generator is deterministic given
the same numpy seed; tests must call `np.random.seed(...)` if they need
reproducibility.

Generation physics baked into the synthetic signal:
- Diurnal envelope: zero generation outside 6am-7pm, peak at solar noon.
- Seasonal envelope: summer ~1.6× winter.
- Cloud noise: multiplicative on irradiance, 0.5-1.0.
- Temperature loosely correlated with irradiance (clear days are hotter).
- System capacity cap: 500 kW peak (assumed site size).

The synthetic R² on a held-out split should exceed 0.85 with the default
hyperparameters. If it doesn't, the synthetic generator is broken, not the
model.

## Inference Contract

`SolarForecaster.forecast(date, conditions=None)`:

- If `conditions` is `None`, generate typical seasonal averages for the date:
  - summer (Jun-Aug): hotter, less cloud, irradiance 0-950
  - winter (Dec-Feb): colder, more cloud, irradiance 0-650
  - shoulder seasons: linear interpolation between summer and winter
- If `conditions` is a list of 24 dicts, each dict is mapped to a feature
  row. Required keys: `temperature_c`, `cloud_cover_pct`, `solar_irradiance`.
- Always returns exactly 24 floats. Non-negative. (Pipeline can produce
  small negatives at night when features are near zero — `inference.py`
  clamps them to 0 at the boundary.)

## What Lives Here vs. Upstream

This service produces numbers. It does not:
- Fetch weather data (caller's job — passes conditions in).
- Interpret policy documents (RAG service's job).
- Schedule battery charge/discharge (solver's job).

## Out of Scope for This Branch

- Live weather API integration.
- Multi-site (fleet) forecasting.
- Probabilistic / quantile forecasts.
- Feature store integration.
