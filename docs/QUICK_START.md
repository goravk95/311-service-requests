# Quick Start Guide

Get started with NYC 311 feature engineering and model training in 5 minutes.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 End-to-End Example

### Step 1: Preprocess Data (with Weather)

```python
from src.preprocessing import preprocess_and_merge_external_data

# Load, clean, and merge all data (weather features computed here)
df = preprocess_and_merge_external_data()

# Weather features now included:
# - tavg, tmax, tmin, prcp (base)
# - heating_degree, cooling_degree (derived)
# - heat_flag, freeze_flag (extremes)
# - rain_3d, rain_7d (rolling)

print(df.shape)
print(df.columns)
```

### Step 2: Build Features

```python
from src.features import (
    add_h3_keys,
    build_forecast_panel,
    build_triage_features,
    build_duration_features,
    build_duration_survival_labels
)

# Add H3 spatial keys
df = add_h3_keys(df, lat='latitude', lon='longitude', res=8)

# Option A: Forecast features (time-series)
forecast_panel = build_forecast_panel(df)  # Weather already in df

# Option B: Triage features (classification)
triage_features, tfidf_matrix, vectorizer = build_triage_features(df)

# Option C: Duration features (survival)
duration_labels = build_duration_survival_labels(df)
duration_features = build_duration_features(df)
```

### Step 3: Train Models

#### CLI Method (Recommended)

```bash
# Forecast
python pipelines/train.py \
  --task forecast \
  --input_parquet data/processed/train.parquet \
  --output_dir output/forecast

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

#### Python API Method

```python
from models.forecast import train_all_families
from models.triage import train_triage
from models.duration import train_duration_aft

# Forecast
forecast_bundles = train_all_families(
    forecast_panel, 
    families=['Health', 'Noise']
)

# Triage
X = triage_features.drop(columns=['unique_key'])
y = df.loc[triage_features['unique_key'], 'potential_inspection_trigger']
families = df.loc[triage_features['unique_key'], 'complaint_family']
triage_bundle = train_triage(X, y, families)

# Duration
data = duration_features.merge(duration_labels, on='unique_key')
X = data.drop(columns=['unique_key', 'duration_days', 'ttc_days_cens', 
                       'event_observed', 'is_admin_like'])
t = data['ttc_days_cens']
e = data['event_observed']
duration_bundle = train_duration_aft(X, t, e, model_type='weibull')
```

### Step 4: Generate Predictions

```bash
# Forecast (7-day ahead)
python pipelines/predict.py \
  --task forecast \
  --model_dir output/forecast/models \
  --scoring_parquet data/test.parquet \
  --output_csv predictions/forecast.csv

# Triage (prioritization)
python pipelines/predict.py \
  --task triage \
  --model_path output/triage/models/triage.joblib \
  --scoring_parquet data/test.parquet \
  --output_csv predictions/triage.csv

# Duration (time-to-close)
python pipelines/predict.py \
  --task duration \
  --model_path output/duration/models/duration_aft.joblib \
  --scoring_parquet data/test.parquet \
  --output_csv predictions/duration.csv
```

## 📊 Output Examples

### Forecast Predictions
```csv
hex,complaint_family,day,p50,p10,p90
8a2a100d28d,Health,2024-10-15,12.3,8.1,18.7
8a2a100d28d,Noise,2024-10-15,5.2,3.1,8.9
```

### Triage Predictions
```csv
unique_key,triage_probability,complaint_family,created_date
123456,0.73,Health,2024-10-15
123457,0.12,Noise,2024-10-15
```

### Duration Predictions
```csv
unique_key,predicted_duration_q50_days,predicted_duration_q90_days
123456,5.2,12.8
123457,2.1,6.3
```

## 🎯 Common Use Cases

### Use Case 1: Daily Forecasting Pipeline

```python
from src.preprocessing import preprocess_and_merge_external_data
from src.features import add_h3_keys, build_forecast_panel
from models.forecast import load_bundles, predict_forecast

# Load preprocessed data (weather included)
df = preprocess_and_merge_external_data()
df = add_h3_keys(df)

# Build panel
panel = build_forecast_panel(df)

# Load trained models
bundles = load_bundles('output/forecast/models')

# Get latest state
last_rows = panel.groupby(['hex', 'complaint_family']).last().reset_index()

# Forecast 7 days ahead
predictions = predict_forecast(bundles, last_rows, horizon=7)

# Save
predictions.to_csv('forecast_7day.csv', index=False)
```

### Use Case 2: Real-Time Triage Scoring

```python
from src.features import add_h3_keys, build_triage_features
from models.triage import load_bundle, predict_triage

# New tickets
new_tickets = pd.read_parquet('new_tickets_today.parquet')
new_tickets = add_h3_keys(new_tickets)

# Build features
triage_features, _, _ = build_triage_features(new_tickets)

# Load model
bundle = load_bundle('output/triage/models/triage.joblib')

# Predict
X = triage_features.drop(columns=['unique_key'])
families = new_tickets.loc[triage_features['unique_key'], 'complaint_family']
probabilities = predict_triage(bundle, X, families)

# Flag high priority
high_priority = probabilities > 0.5
print(f"High priority tickets: {high_priority.sum()}")
```

### Use Case 3: Duration Estimation for SLA Monitoring

```python
from src.features import add_h3_keys, build_duration_features
from models.duration import load_bundle, predict_duration_quantiles

# Open tickets
open_tickets = pd.read_parquet('open_tickets.parquet')
open_tickets = add_h3_keys(open_tickets)

# Build features
duration_features = build_duration_features(open_tickets)

# Load model
bundle = load_bundle('output/duration/models/duration_aft.joblib')

# Predict
X = duration_features.drop(columns=['unique_key'])
quantiles = predict_duration_quantiles(bundle, X, ps=(0.5, 0.9))

# Flag at-risk of SLA breach (Q90 > 60 days)
at_risk = quantiles['q90'] > 60
print(f"At-risk of SLA breach: {at_risk.sum()}")
```

## 🔧 Troubleshooting

### Issue: Import errors
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Missing weather features
```python
# Check if weather features exist
print('heating_degree' in df.columns)  # Should be True after preprocessing

# If False, run preprocessing again
df = preprocess_and_merge_external_data()
```

### Issue: H3 module not found
```bash
pip install h3
```

### Issue: Model files not found
```bash
# Check output directory structure
ls -R output/*/models/

# Expected structure:
# output/
#   forecast/models/*.joblib
#   triage/models/triage.joblib
#   duration/models/duration_aft.joblib
```

## ⏭️ Next Steps

1. **Understand features:** Read [FEATURES.md](FEATURES.md)
2. **Deep dive into training:** Read [TRAINING.md](TRAINING.md)
3. **API reference:** See [API_REFERENCE.md](API_REFERENCE.md)
4. **Hyperparameter tuning:** Add `--tune` flag to training
5. **Production deployment:** Review model persistence and serving patterns

## 💡 Pro Tips

1. **Weather features are pre-computed** in `preprocess_and_merge_external_data()` - no need to pass weather separately
2. **Use parquet format** for 10x faster I/O vs CSV
3. **Train on subsets** first to validate pipeline (filter to 1 month of data)
4. **Cache preprocessing** results to avoid recomputing
5. **Version your models** by output directory (e.g., `output/v1.0/`, `output/v1.1/`)

---

**Questions?** See [FEATURES.md](FEATURES.md) and [TRAINING.md](TRAINING.md) for detailed documentation.

