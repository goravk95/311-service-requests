# NYC 311 Model Training Package

Complete training and prediction pipelines for three modeling tracks: **Forecast**, **Triage**, and **Duration**.

## 📦 Package Structure

```
models/
├── __init__.py
├── forecast.py          # LightGBM Poisson forecasting
├── triage.py            # LightGBM classifier with calibration
├── duration.py          # AFT survival models
├── tune.py              # Optuna hyperparameter tuning
└── eval.py              # Evaluation metrics and plots

pipelines/
├── __init__.py
├── train.py             # Training CLI
└── predict.py           # Prediction CLI
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Forecast
python pipelines/train.py \
  --task forecast \
  --input_parquet data/processed/train.parquet \
  --output_dir output/forecast \
  --weather_parquet data/landing/noaa-nclimgrid-daily/nyc_fips_weather_data.csv

# Triage
python pipelines/train.py \
  --task triage \
  --input_parquet data/processed/train.parquet \
  --output_dir output/triage \
  --target_col potential_inspection_trigger

# Duration
python pipelines/train.py \
  --task duration \
  --input_parquet data/processed/train.parquet \
  --output_dir output/duration
```

### Prediction

```bash
# Forecast
python pipelines/predict.py \
  --task forecast \
  --model_dir output/forecast/models \
  --scoring_parquet data/processed/test.parquet \
  --output_csv predictions/forecast.csv \
  --horizon 7

# Triage
python pipelines/predict.py \
  --task triage \
  --model_path output/triage/models/triage.joblib \
  --scoring_parquet data/processed/test.parquet \
  --output_csv predictions/triage.csv

# Duration
python pipelines/predict.py \
  --task duration \
  --model_path output/duration/models/duration_aft.joblib \
  --scoring_parquet data/processed/test.parquet \
  --output_csv predictions/duration.csv
```

## 📊 Three Modeling Tracks

### 1️⃣ Forecast (Time-Series Prediction)

**Goal:** Predict daily ticket arrivals by location (H3 hex) and complaint family.

**Model:** LightGBM with Poisson objective

**Features:**
- Temporal: day of week, month
- History: lag1, lag7, roll7/14/28, momentum, days_since_last
- Weather: tavg, prcp, heating/cooling degree days, rain_3d/7d
- Population: log_pop
- Spatial: H3 hex (categorical)

**Output:**
- One model per complaint family
- Per-horizon models (1-7 days)
- Predictions: p50 (median), p10, p90 (via Poisson quantiles)

**Metrics:**
- RMSE, MAE, MAPE
- Poisson deviance
- Calibration plots

**File:** `models/forecast.py`

**Usage:**
```python
from models import forecast

# Train
bundles = forecast.train_all_families(panel, families=['Health', 'Noise'])

# Predict
predictions = forecast.predict_forecast(bundles, last_rows, horizon=7)
```

---

### 2️⃣ Triage (Classification)

**Goal:** Prioritize tickets at creation time (e.g., inspection triggers).

**Model:** LightGBM binary classifier with per-family isotonic calibration

**Features:**
- Temporal: hour, dow, month, is_weekend
- Categorical: complaint_family, borough, channel, location_type, etc.
- Local history: geo_family_roll7/28 (as-of)
- Repeat site: repeat_site_14d/28d
- Due date: due_gap_hours, due_is_60d, due_crosses_weekend
- Weather: tavg, prcp, heat_flag, freeze_flag
- Text: TF-IDF from descriptor (optional)

**Output:**
- Global model with per-family calibration
- Calibrated probabilities (0-1)

**Metrics:**
- AUC-ROC, AUC-PR
- Calibration curves
- Lift@K
- Per-family performance

**File:** `models/triage.py`

**Usage:**
```python
from models import triage

# Train
bundle = triage.train_triage(X, y, families)

# Predict
probabilities = triage.predict_triage(bundle, X_new, families_new)
```

---

### 3️⃣ Duration (Survival Analysis)

**Goal:** Predict time-to-close with censoring for admin auto-close and long stale tickets.

**Model:** AFT (Accelerated Failure Time) - Weibull or LogNormal

**Features:**
- All triage features
- Queue pressure: intake_6h, intake_24h
- Backlog: open_7d_geo_family
- Due date: due_gap_hours, due_is_60d, due_crosses_weekend

**Censoring Rules:**
- Admin auto-close: [59-62] days → censored at 60.5
- Long stale: >365 days → censored at 365
- Open tickets: no closed_date → censored at 365

**Output:**
- Quantile predictions: q50 (median), q90 (90th percentile)
- Days to close

**Metrics:**
- Concordance index (C-index)
- MAE at Q50
- Q90 coverage (calibration)
- Reliability plots

**File:** `models/duration.py`

**Usage:**
```python
from models import duration

# Train
bundle = duration.train_duration_aft(X, t, e, model_type='weibull')

# Predict
quantiles = duration.predict_duration_quantiles(bundle, X_new, ps=(0.5, 0.9))
```

---

## 🔧 Hyperparameter Tuning

Use Optuna for automated hyperparameter search:

```bash
# Add --tune flag to training command
python pipelines/train.py \
  --task forecast \
  --input_parquet data/processed/train.parquet \
  --output_dir output/forecast \
  --tune
```

**Tuned Parameters:**
- Forecast: learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree
- Triage: same as forecast
- Duration: model_type (weibull/lognormal), penalizer

**Optimization Targets:**
- Forecast: Poisson deviance (minimize)
- Triage: AUC-PR (maximize)
- Duration: C-index + Q90 coverage (maximize)

**File:** `models/tune.py`

---

## 📈 Evaluation

All models include comprehensive evaluation metrics and visualizations.

**Forecast:**
- Predicted vs actual scatter plots
- Binned calibration curves
- Per-family, per-horizon metrics

**Triage:**
- ROC curves
- Precision-Recall curves
- Calibration curves
- Per-family metrics

**Duration:**
- Q50 calibration plots
- Q90 coverage by duration bin
- Error distribution
- Reliability plots

**File:** `models/eval.py`

**Output Locations:**
- Metrics: `output/{task}/metrics/{task}_metrics.json`
- Plots: `output/{task}/plots/`

---

## 💻 CLI Reference

### Training CLI (`pipelines/train.py`)

```bash
python pipelines/train.py \
  --task {forecast,triage,duration} \
  --input_parquet PATH \
  --output_dir PATH \
  [--families FAMILY1,FAMILY2] \  # forecast only
  [--tune] \
  [--weather_parquet PATH] \  # forecast only
  [--target_col COL]  # triage only (default: potential_inspection_trigger)
```

**Arguments:**
- `--task`: Model task (required)
- `--input_parquet`: Input data path (required)
- `--output_dir`: Output directory (required)
- `--families`: Comma-separated list of families to train (forecast only)
- `--tune`: Enable hyperparameter tuning
- `--weather_parquet`: Weather data path (forecast only)
- `--target_col`: Target column name (triage only)

**Output Structure:**
```
output_dir/
├── models/
│   ├── forecast_{family}.joblib (one per family)
│   ├── triage.joblib
│   └── duration_aft.joblib
├── metrics/
│   ├── forecast_metrics.json
│   ├── triage_metrics.json
│   └── duration_metrics.json
└── plots/
    ├── triage_curves.png
    └── duration_reliability.png
```

### Prediction CLI (`pipelines/predict.py`)

```bash
python pipelines/predict.py \
  --task {forecast,triage,duration} \
  --scoring_parquet PATH \
  --output_csv PATH \
  [--model_dir PATH] \  # forecast
  [--model_path PATH] \  # triage/duration
  [--horizon N]  # forecast (default: 7)
```

**Arguments:**
- `--task`: Model task (required)
- `--scoring_parquet`: Input data for scoring (required)
- `--output_csv`: Output predictions CSV (required)
- `--model_dir`: Model directory (forecast only)
- `--model_path`: Model file path (triage/duration only)
- `--horizon`: Forecast horizon in days (default: 7)

**Output Formats:**

**Forecast:**
```csv
hex,complaint_family,day,p50,p10,p90
8a2a100d28d,Health,2024-10-15,12.3,8.1,18.7
```

**Triage:**
```csv
unique_key,triage_probability,complaint_family,created_date,hex
123456,0.73,Health,2024-10-15,8a2a100d28d
```

**Duration:**
```csv
unique_key,predicted_duration_q50_days,predicted_duration_q90_days,complaint_family
123456,5.2,12.8,Health
```

---

## 📝 Programmatic API

### Forecast

```python
from models.forecast import (
    train_forecast_per_family,
    train_all_families,
    predict_forecast,
    save_bundles,
    load_bundles
)

# Train single family
bundle = train_forecast_per_family(panel, family='Health', horizons=range(1,8))

# Train all families
bundles = train_all_families(panel, families=['Health', 'Noise'])

# Save/load
save_bundles(bundles, Path('output/models'))
bundles = load_bundles(Path('output/models'))

# Predict
predictions = predict_forecast(bundles, last_rows, horizon=7)
```

### Triage

```python
from models.triage import (
    train_triage,
    predict_triage,
    save_bundle,
    load_bundle
)

# Train
bundle = train_triage(X, y, families, params={'learning_rate': 0.05})

# Save/load
save_bundle(bundle, Path('output/models/triage.joblib'))
bundle = load_bundle(Path('output/models/triage.joblib'))

# Predict
probabilities = predict_triage(bundle, X_new, families_new)
```

### Duration

```python
from models.duration import (
    train_duration_aft,
    compare_models,
    predict_duration_quantiles,
    save_bundle,
    load_bundle
)

# Train single model
bundle = train_duration_aft(X, t, e, model_type='weibull')

# Compare models
best_bundle, best_type = compare_models(X, t, e, models=['weibull', 'lognormal'])

# Save/load
save_bundle(bundle, Path('output/models/duration_aft.joblib'))
bundle = load_bundle(Path('output/models/duration_aft.joblib'))

# Predict
quantiles = predict_duration_quantiles(bundle, X_new, ps=(0.5, 0.9))
```

### Evaluation

```python
from models.eval import (
    eval_forecast,
    eval_triage,
    eval_duration,
    plot_forecast_calibration,
    plot_triage_curves,
    plot_duration_reliability,
    save_metrics,
    print_metrics_summary
)

# Evaluate
forecast_metrics = eval_forecast(y_true, y_pred, family='Health', horizon=7)
triage_metrics = eval_triage(y_true, y_pred_proba, families)
duration_metrics = eval_duration(t_true, t_pred, e_true, quantile=0.5)

# Plot
plot_forecast_calibration(y_true, y_pred, output_path='plots/forecast_cal.png')
plot_triage_curves(y_true, y_pred_proba, output_path='plots/triage.png')
plot_duration_reliability(t_true, q50, q90, e_true, output_path='plots/duration.png')

# Save
save_metrics(metrics, Path('metrics/results.json'))
print_metrics_summary(metrics, 'forecast')
```

### Tuning

```python
from models.tune import (
    tune_forecast,
    tune_triage,
    tune_duration_aft,
    run_tuning_suite
)

# Tune individual models
forecast_result = tune_forecast(X_train, y_train, X_val, y_val, n_trials=30)
triage_result = tune_triage(X_train, y_train, X_val, y_val, n_trials=30)
duration_result = tune_duration_aft(X, t, e, n_trials=30)

# Tune all at once
results = run_tuning_suite(
    forecast_data={'X_train': ..., 'y_train': ..., 'X_val': ..., 'y_val': ...},
    triage_data={'X_train': ..., 'y_train': ..., 'X_val': ..., 'y_val': ...},
    duration_data={'X': ..., 't': ..., 'e': ...},
    n_trials=30
)
```

---

## 🎯 Best Practices

### Time-Based Splitting
All models use time-based train/validation splits to prevent leakage:
- Forecast: Last 30 days = validation
- Triage: Random split (20% validation) with stratification
- Duration: Random split (20% validation)

**For production:** Use rolling-window or walk-forward validation.

### Leakage Prevention
- ✅ Use only pre-creation features
- ✅ As-of joins for historical aggregates
- ✅ No `status`, `closed_date`, `resolution_*` fields
- ✅ Validate with `build_triage_features()` checks

### Model Persistence
- All models saved with `joblib` for fast serialization
- Bundle includes model, features, transformers, metrics
- Version models by output directory (e.g., `output/v1.0/`)

### Monitoring
- Track model metrics over time
- Alert on AUC drop >5%
- Monitor prediction distribution drift
- Re-train monthly or quarterly

---

## 🧪 Example Workflow

```python
import pandas as pd
from pathlib import Path
from src.features import add_h3_keys, build_forecast_panel
from models.forecast import train_all_families, save_bundles

# 1. Load and prep data
df = pd.read_parquet('data/processed/train.parquet')
df = add_h3_keys(df)

# 2. Build feature panel
weather = pd.read_csv('data/landing/noaa-nclimgrid-daily/nyc_fips_weather_data.csv')
panel = build_forecast_panel(df, weather_df=weather)

# 3. Train models
bundles = train_all_families(
    panel, 
    families=['Health', 'Noise', 'Housing'],
    val_days=30
)

# 4. Save models
output_dir = Path('output/forecast/v1.0')
save_bundles(bundles, output_dir / 'models')

# 5. Evaluate
from models.eval import eval_forecast, save_metrics

metrics = {}
for family, bundle in bundles.items():
    for horizon, model_metrics in bundle['metrics'].items():
        metrics[f'{family}_h{horizon}'] = model_metrics

save_metrics(metrics, output_dir / 'metrics/summary.json')

print("✓ Training complete!")
```

---

## 📚 Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
lightgbm>=4.0.0
lifelines>=0.29.0
optuna>=3.5.0
joblib>=1.3.0
scipy>=1.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
h3>=3.7.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

**Issue: Import errors**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue: Memory errors on large datasets**
```python
# Process in batches by month or family
for month in range(1, 13):
    df_month = pd.read_parquet(f'data/.../month={month:02d}/...')
    # Train on month subset
```

**Issue: Slow training**
```python
# Reduce n_estimators or use early stopping
params = {'n_estimators': 300, 'learning_rate': 0.1}

# Or filter to top N families by volume
top_families = df.groupby('complaint_family').size().nlargest(5).index
```

**Issue: Poor calibration**
```python
# Increase calibration threshold
bundle = train_triage(X, y, families, calibration_threshold=1000)

# Or use Platt scaling instead of isotonic
from sklearn.calibration import CalibratedClassifierCV
```

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review `src/FEATURE_ENGINEERING_README.md` for feature details
3. Examine example outputs in `output/` directories
4. Run with `--help` for CLI options

---

## 📄 License

[Your License Here]

---

**Version:** 1.0  
**Last Updated:** October 4, 2025  
**Status:** ✅ Production Ready

