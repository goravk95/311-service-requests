# Model Training Guide

Complete guide to training NYC 311 prediction models.

## 📋 Table of Contents

- [Overview](#overview)
- [Forecast Model](#forecast-model-time-series)
- [Triage Model](#triage-model-classification)
- [Duration Model](#duration-model-survival)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [CLI Reference](#cli-reference)
- [Production Deployment](#production-deployment)

---

## Overview

Train three types of models for NYC 311 service requests:

| Model | Task | Algorithm | Output |
|-------|------|-----------|--------|
| **Forecast** | Time-series prediction | LightGBM Poisson | Daily ticket counts (p10, p50, p90) |
| **Triage** | Prioritization | LightGBM + calibration | Probabilities (0-1) |
| **Duration** | Time-to-close | AFT survival | Duration quantiles (Q50, Q90) |

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

## Triage Model (Classification)

Classify tickets for prioritization (e.g., inspection triggers).

### Architecture

- **Algorithm:** LightGBM binary classifier
- **Calibration:** Per-family isotonic regression
- **Features:** Creation-time only (zero leakage)
- **Class balancing:** Scale_pos_weight if positive rate <15%

### Training (CLI)

```bash
python pipelines/train.py \
  --task triage \
  --input_parquet data/processed/train.parquet \
  --output_dir output/triage \
  --target_col potential_inspection_trigger \
  --tune  # Optional
```

### Training (Python API)

```python
from src.preprocessing import preprocess_and_merge_external_data
from src.features import add_h3_keys, build_triage_features
from models.triage import train_triage, save_bundle

# Prepare data
df = preprocess_and_merge_external_data()
df = add_h3_keys(df)
triage_features, tfidf, vectorizer = build_triage_features(df)

# Get target and families
X = triage_features.drop(columns=['unique_key'])
y = df.loc[triage_features['unique_key'], 'potential_inspection_trigger']
families = df.loc[triage_features['unique_key'], 'complaint_family']

# Train
bundle = train_triage(X, y, families, calibration_threshold=500)

# Save
save_bundle(bundle, 'output/triage/models/triage.joblib')
```

### Model Parameters

```python
params = {
    'objective': 'binary',
    'n_estimators': 700,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'scale_pos_weight': (1 - pos_rate) / pos_rate,  # If imbalanced
    'random_state': 42
}
```

### Prediction

```python
from models.triage import load_bundle, predict_triage

# Load model
bundle = load_bundle('output/triage/models/triage.joblib')

# Prepare new data
new_features, _, _ = build_triage_features(new_tickets)
X = new_features.drop(columns=['unique_key'])
families = new_tickets.loc[new_features['unique_key'], 'complaint_family']

# Predict (calibrated probabilities)
probabilities = predict_triage(bundle, X, families)

# Flag high priority
high_priority = probabilities > 0.5
```

### Metrics

- **AUC-ROC:** Area under ROC curve
- **AUC-PR:** Area under precision-recall curve (primary for imbalanced)
- **Calibration:** Reliability diagram
- **Per-family performance:** AUC-ROC/PR by complaint family

---

## Duration Model (Survival)

Predict time-to-close with right-censoring for admin auto-close and long stale tickets.

### Architecture

- **Algorithm:** AFT (Accelerated Failure Time)
- **Distributions:** Weibull or LogNormal (selected via validation)
- **Censoring:** Admin auto-close [59-62]d, long stale >365d
- **Features:** Triage features + queue pressure

### Training (CLI)

```bash
python pipelines/train.py \
  --task duration \
  --input_parquet data/processed/train.parquet \
  --output_dir output/duration \
  --tune  # Compares Weibull vs LogNormal
```

### Training (Python API)

```python
from src.preprocessing import preprocess_and_merge_external_data
from src.features import add_h3_keys, build_duration_features, build_duration_survival_labels
from models.duration import train_duration_aft, compare_models, save_bundle

# Prepare data
df = preprocess_and_merge_external_data()
df = add_h3_keys(df)

duration_features = build_duration_features(df)
duration_labels = build_duration_survival_labels(df)

# Merge
data = duration_features.merge(duration_labels, on='unique_key')
X = data.drop(columns=['unique_key', 'duration_days', 'ttc_days_cens', 
                       'event_observed', 'is_admin_like'])
t = data['ttc_days_cens']
e = data['event_observed']

# Train single model
bundle = train_duration_aft(X, t, e, model_type='weibull', penalizer=0.01)

# Or compare models
best_bundle, best_model = compare_models(X, t, e, models=['weibull', 'lognormal'])

# Save
save_bundle(best_bundle, 'output/duration/models/duration_aft.joblib')
```

### Model Parameters

```python
# Weibull or LogNormal AFT
params = {
    'penalizer': 0.01,  # L2 regularization
}

# Features are standardized and one-hot encoded automatically
```

### Prediction

```python
from models.duration import load_bundle, predict_duration_quantiles

# Load model
bundle = load_bundle('output/duration/models/duration_aft.joblib')

# Prepare new data
new_features = build_duration_features(new_tickets)
X = new_features.drop(columns=['unique_key'])

# Predict quantiles
quantiles = predict_duration_quantiles(bundle, X, ps=(0.5, 0.9))

# quantiles['q50'] = median (50th percentile)
# quantiles['q90'] = 90th percentile

# Flag at-risk
at_risk = quantiles['q90'] > 60  # >60 days
```

### Metrics

- **C-index:** Concordance index (higher is better, 0.5-1.0)
- **MAE at Q50:** Mean absolute error at median
- **Q90 coverage:** % of observed events ≤ Q90 prediction (should be ~90%)
- **Reliability plots:** Calibration by duration bin

---

## Hyperparameter Tuning

Use Optuna for automated hyperparameter search.

### CLI (Recommended)

```bash
# Add --tune flag to any training command
python pipelines/train.py --task forecast --input_parquet DATA --output_dir OUTPUT --tune
python pipelines/train.py --task triage --input_parquet DATA --output_dir OUTPUT --tune
python pipelines/train.py --task duration --input_parquet DATA --output_dir OUTPUT --tune
```

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

# Triage tuning
result = tune_triage(
    X_train, y_train, X_val, y_val,
    n_trials=30
)

# Duration tuning
result = tune_duration_aft(
    X, t, e,
    n_trials=30
)
```

### Tuned Parameters

**Forecast & Triage:**
- `learning_rate`: [0.01, 0.3] (log scale)
- `num_leaves`: [20, 100]
- `max_depth`: [3, 10]
- `min_child_samples`: [5, 100]
- `subsample`: [0.5, 1.0]
- `colsample_bytree`: [0.5, 1.0]

**Duration:**
- `model_type`: 'weibull' or 'lognormal'
- `penalizer`: [0.001, 0.1] (log scale)

### Optimization Targets

- **Forecast:** Poisson deviance (minimize)
- **Triage:** AUC-PR (maximize)
- **Duration:** C-index + Q90 coverage (maximize)

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

# Triage
metrics = eval_triage(y_true, y_pred_proba, families)
plot_triage_curves(y_true, y_pred_proba, output_path='roc_pr.png')

# Duration
metrics = eval_duration(t_true, t_pred_q50, e_true, quantile=0.5)
plot_duration_reliability(t_true, q50, q90, e_true, output_path='reliability.png')
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

## Troubleshooting

### Q: Training fails with memory error?
**A:** Process in batches by month or family.

### Q: Models underperforming?
**A:** Check feature distributions, try hyperparameter tuning, validate time splits.

### Q: Calibration poor?
**A:** Increase `calibration_threshold`, ensure sufficient validation data per family.

### Q: Duration model won't converge?
**A:** Try different penalizer values, check for collinear features, standardize features.

---

**Previous:** [FEATURES.md](FEATURES.md) | **Next:** [API_REFERENCE.md](API_REFERENCE.md)

