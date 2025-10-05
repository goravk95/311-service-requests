# Model Training Guide

Complete guide to training NYC 311 prediction models.

## 📋 Table of Contents

- [Overview](#overview)
- [Forecast Model](#forecast-model-time-series)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [CLI Reference](#cli-reference)
- [Production Deployment](#production-deployment)

---

## Overview

Train forecast models for NYC 311 service requests:

| Model | Task | Algorithm | Output |
|-------|------|-----------|--------|
| **Forecast** | Time-series prediction | LightGBM Poisson | Daily ticket counts (p10, p50, p90) |

**Prerequisites:** Features built using [FEATURES.md](FEATURES.md) guide.

---

## Forecast Model (Time-Series)

Predict daily ticket arrivals by location (H3 hex) and complaint type.

### Architecture

- **Algorithm:** LightGBM with Poisson objective
- **Granularity:** Per complaint family
- **Horizons:** 1-7 days ahead (separate model per horizon)
- **Features:** Temporal, spatial, weather, population

### Training (CLI)

```bash
python pipelines/train.py \
  --task forecast \
  --input_parquet data/processed/train.parquet \
  --output_dir output/forecast \
  --families "Health,Noise,Housing" \
  --tune  # Optional: hyperparameter tuning
```

### Training (Python API)

```python
from src.preprocessing import preprocess_and_merge_external_data
from src.features import add_h3_keys, build_forecast_panel
from models.forecast import train_all_families, save_bundles

# Prepare data
df = preprocess_and_merge_external_data()
df = add_h3_keys(df)
panel = build_forecast_panel(df)

# Train
bundles = train_all_families(
    panel, 
    families=['Health', 'Noise', 'Housing'],
    horizons=range(1, 8),  # 1-7 days
    val_days=30  # Last 30 days for validation
)

# Save
save_bundles(bundles, 'output/forecast/models')
```

### Model Parameters

```python
params = {
    'objective': 'poisson',
    'n_estimators': 800,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'random_state': 42
}
```

### Prediction

```python
from models.forecast import load_bundles, predict_forecast

# Load models
bundles = load_bundles('output/forecast/models')

# Get latest state
last_rows = panel.groupby(['hex', 'complaint_family']).last().reset_index()

# Predict 7 days ahead
predictions = predict_forecast(bundles, last_rows, horizon=7)

# Output: hex, complaint_family, day, p50, p10, p90
```

### Metrics

- **RMSE:** Root mean squared error
- **MAE:** Mean absolute error
- **MAPE:** Mean absolute percentage error
- **Poisson deviance:** Primary metric (lower is better)

---

## Hyperparameter Tuning

Use Optuna for automated hyperparameter search.

### CLI (Recommended)

```bash
# Add --tune flag to any training command
python pipelines/train.py --task forecast --input_parquet DATA --output_dir OUTPUT --tune

### Python API

```python
from models.tune import tune_forecast, tune_triage, tune_duration_aft

# Forecast tuning
result = tune_forecast(
    X_train, y_train, X_val, y_val,
    cat_cols=['hex'],
    n_trials=30  # Number of Optuna trials
)
best_params = result['best_params']

```

### Tuned Parameters

**Forecast:**
- `learning_rate`: [0.01, 0.3] (log scale)
- `num_leaves`: [20, 100]
- `max_depth`: [3, 10]
- `min_child_samples`: [5, 100]
- `subsample`: [0.5, 1.0]
- `colsample_bytree`: [0.5, 1.0]

### Optimization Targets

- **Forecast:** Poisson deviance (minimize)
---

## Evaluation

All models include comprehensive evaluation metrics and visualizations.

### Automatic Evaluation

Training automatically generates:
- **Metrics JSON:** `output/{task}/metrics/{task}_metrics.json`
- **Plots:** `output/{task}/plots/*.png`

### Manual Evaluation

```python
from models.eval import (
    eval_forecast, eval_triage, eval_duration,
    plot_forecast_calibration, plot_triage_curves, plot_duration_reliability
)

# Forecast
metrics = eval_forecast(y_true, y_pred, family='Health', horizon=7)
plot_forecast_calibration(y_true, y_pred, output_path='cal.png')
```

---

## CLI Reference

### Training

```bash
python pipelines/train.py \
  --task {forecast,triage,duration} \
  --input_parquet PATH \
  --output_dir PATH \
  [--families FAMILY1,FAMILY2]  # forecast only \
  [--tune] \
  [--target_col COL]  # triage only
```

**Arguments:**
- `--task`: Model type (required)
- `--input_parquet`: Input data path (required)
- `--output_dir`: Output directory (required)
- `--families`: Comma-separated families for forecast
- `--tune`: Enable hyperparameter tuning
- `--target_col`: Target column for triage (default: `potential_inspection_trigger`)

### Prediction

```bash
python pipelines/predict.py \
  --task {forecast,triage,duration} \
  --scoring_parquet PATH \
  --output_csv PATH \
  [--model_dir PATH]  # forecast \
  [--model_path PATH]  # triage/duration \
  [--horizon N]  # forecast (default: 7)
```

**Arguments:**
- `--task`: Model type (required)
- `--scoring_parquet`: Data to score (required)
- `--output_csv`: Output predictions CSV (required)
- `--model_dir`: Model directory for forecast
- `--model_path`: Model file for triage/duration
- `--horizon`: Forecast horizon in days

---

## Production Deployment

### Model Versioning

```bash
# Version models by output directory
python pipelines/train.py --task triage --input_parquet DATA --output_dir output/v1.0/triage
python pipelines/train.py --task triage --input_parquet DATA --output_dir output/v1.1/triage

# Load specific version
bundle = load_bundle('output/v1.0/triage/models/triage.joblib')
```

### Batch Scoring

```python
# Score large datasets in batches
batch_size = 10000

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    
    # Build features
    features, _, _ = build_triage_features(batch)
    
    # Predict
    probabilities = predict_triage(bundle, X, families)
    
    # Save
    batch['probability'] = probabilities
    batch.to_csv(f'predictions_batch_{i}.csv', index=False)
```

### Monitoring

```python
# Track model performance over time
from models.eval import eval_triage

monthly_metrics = []

for month in range(1, 13):
    df_month = df[df['month'] == month]
    
    # Generate predictions
    probabilities = predict_triage(bundle, X_month, families_month)
    
    # Evaluate
    metrics = eval_triage(y_month, probabilities, families_month)
    metrics['month'] = month
    monthly_metrics.append(metrics)

# Alert if AUC drops >5%
baseline_auc = monthly_metrics[0]['overall']['auc_roc']
for m in monthly_metrics[1:]:
    current_auc = m['overall']['auc_roc']
    if (baseline_auc - current_auc) > 0.05:
        print(f"Alert: AUC dropped in month {m['month']}")
```

### Retraining Schedule

- **Forecast:** Retrain weekly with latest 90 days
- **Triage:** Retrain monthly with latest 6 months
- **Duration:** Retrain quarterly with latest 12 months

---

## Best Practices

### 1. Time-Based Validation

Always use time-based splits to prevent leakage:

```python
# ✅ Good: Time-based split
cutoff_date = df['day'].max() - pd.Timedelta(days=30)
train = df[df['day'] < cutoff_date]
val = df[df['day'] >= cutoff_date]

# ❌ Bad: Random split
train, val = train_test_split(df, test_size=0.2)  # Leakage!
```

### 2. Cache Preprocessing

```python
# Save preprocessed data
df = preprocess_and_merge_external_data()
df.to_parquet('data/processed/train_preprocessed.parquet')

# Load for training
df = pd.read_parquet('data/processed/train_preprocessed.parquet')
```

### 3. Start Small

```python
# Train on subset first to validate pipeline
df_sample = df.sample(n=10000, random_state=42)
bundle = train_triage(X_sample, y_sample, families_sample)

# Then scale up
bundle = train_triage(X_full, y_full, families_full)
```

### 4. Monitor Drift

```python
# Track feature distributions
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
for i, col in enumerate(['tavg', 'repeat_site_14d', 'geo_family_roll7']):
    plt.subplot(1, 3, i+1)
    df_train[col].hist(bins=50, alpha=0.5, label='Train')
    df_prod[col].hist(bins=50, alpha=0.5, label='Production')
    plt.xlabel(col)
    plt.legend()
plt.tight_layout()
plt.savefig('feature_drift.png')
```

---

