# Feature Engineering Quick Start

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
import pandas as pd
from src.features import (
    add_h3_keys,
    build_forecast_panel,
    build_triage_features,
    build_duration_survival_labels,
    build_duration_features
)

# Load your data
df = pd.read_parquet('data/landing/311-service-requests/...')

# Step 1: Add H3 keys
df = add_h3_keys(df)

# Step 2: Build features for your model type

## For Forecasting (time-series)
forecast_panel = build_forecast_panel(df, weather_df=weather)

## For Triage (prioritization)
triage_features, tfidf_matrix, vectorizer = build_triage_features(df)

## For Duration (survival analysis)
duration_labels = build_duration_survival_labels(df)
duration_features = build_duration_features(df)
```

## Model-Specific Workflows

### Forecast Model
```python
# Get features and target
X = forecast_panel[['dow', 'month', 'lag7', 'roll7', 'roll28', 
                   'momentum', 'tavg', 'log_pop']]
y = forecast_panel['y']

# Train model
from lightgbm import LGBMRegressor
model = LGBMRegressor()
model.fit(X, y)
```

### Triage Model
```python
# Get features
X = triage_features.drop(columns=['unique_key'])
y = df.loc[triage_features['unique_key'], 'potential_inspection_trigger']

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
```

### Duration Model
```python
# Merge features and labels
data = duration_features.merge(duration_labels, on='unique_key')

# Prepare for survival analysis
X = data.drop(columns=['unique_key', 'duration_days', 'ttc_days_cens', 
                       'event_observed', 'is_admin_like'])
y = np.array(list(zip(data['event_observed'].astype(bool), 
                      data['ttc_days_cens'])),
             dtype=[('event', bool), ('time', float)])

# Train survival model
from sksurv.ensemble import RandomSurvivalForest
model = RandomSurvivalForest()
model.fit(X, y)
```

## Key Features by Model

### Forecast Panel
- **Target:** `y` (ticket count)
- **History:** `lag1`, `lag7`, `roll7`, `roll14`, `roll28`, `momentum`
- **Calendar:** `dow`, `month`
- **Weather:** `tavg`, `prcp`, `heating_degree`, `cooling_degree`
- **Spatial:** `hex` (H3 cell), `log_pop`

### Triage Features
- **Temporal:** `hour`, `dow`, `month`, `is_weekend`
- **Location History:** `geo_family_roll7`, `geo_family_roll28`
- **Repeat Site:** `repeat_site_14d`, `repeat_site_28d`
- **Due Date:** `due_gap_hours`, `due_is_60d`, `due_crosses_weekend`
- **Weather:** `tavg`, `prcp`, `heat_flag`, `freeze_flag`
- **Text:** TF-IDF from `descriptor_clean` (separate matrix)

### Duration Features
All triage features plus:
- **Queue Pressure:** `intake_6h`, `intake_24h`
- **Backlog:** `open_7d_geo_family`

### Duration Labels
- `ttc_days_cens` - Censored time-to-close
- `event_observed` - 1 if true closure, 0 if censored
- `is_admin_like` - Flag for admin auto-close

## Leakage Prevention

✅ **Safe to use:**
- `created_date`, `due_date`
- Location fields (`latitude`, `longitude`, `borough`)
- Complaint fields (`complaint_family`, `descriptor_clean`)
- Historical aggregates (via as-of joins)

❌ **Never use (leakage):**
- `status`, `closed_date`
- `resolution_description`, `resolution_outcome`
- `time_to_resolution*` fields

## Common Issues

### Issue: Missing H3 keys
**Solution:** Check lat/lon validity. Invalid coordinates return `hex=NaN`.

```python
df['has_valid_location'] = df['hex'].notna()
print(f"Valid locations: {df['has_valid_location'].mean():.1%}")
```

### Issue: Panel too large
**Solution:** The panel is sparse by design. If memory is an issue:
- Filter to specific date ranges
- Filter to specific complaint families
- Increase H3 resolution (smaller areas, fewer groups)

### Issue: TF-IDF fails
**Solution:** Check for sufficient text data.

```python
# Ensure descriptor_clean exists and has content
df['descriptor_clean'] = df['descriptor_clean'].fillna('').astype(str)
print(f"Non-empty descriptors: {(df['descriptor_clean'] != '').mean():.1%}")
```

## Performance Tips

1. **Use parquet format** for fast I/O
2. **Filter early** to reduce data size before feature engineering
3. **Process in batches** for large datasets (e.g., by month)
4. **Cache intermediate results** (e.g., forecast panel)

```python
# Example: batch processing
for year in range(2020, 2025):
    for month in range(1, 13):
        df_month = pd.read_parquet(f'data/.../year={year}/month={month:02d}/...')
        df_month = add_h3_keys(df_month)
        panel_month = build_forecast_panel(df_month)
        panel_month.to_parquet(f'data/features/panel_{year}_{month:02d}.parquet')
```

## Further Reading

- **Full documentation:** `src/FEATURE_ENGINEERING_README.md`
- **Example notebook:** `notebooks/Step 5 - Feature Engineering.ipynb`
- **Example script:** `src/feature_engineering_example.py`
- **Utils reference:** `src/utils.py`
- **Main module:** `src/features.py`

## Support

For issues or questions:
1. Check the full documentation (`FEATURE_ENGINEERING_README.md`)
2. Run the example script (`python src/feature_engineering_example.py`)
3. Review the validation notebook (`Step 5 - Feature Engineering.ipynb`)

## License

[Your License Here]

