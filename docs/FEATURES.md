# Feature Engineering Guide

Complete guide to feature engineering for NYC 311 service requests.

## 📋 Table of Contents

- [Overview](#overview)
- [Preprocessing (Weather Features)](#preprocessing-weather-features)
- [H3 Spatial Keys](#h3-spatial-keys)
- [Forecast Features](#forecast-features-time-series)
- [Triage Features](#triage-features-classification)
- [Duration Features](#duration-features-survival)
- [API Reference](#api-reference)

---

## Overview

The feature engineering pipeline transforms raw 311 data into model-ready features for three tasks:

1. **Forecast:** Time-series prediction of ticket arrivals
2. **Triage:** Classification for ticket prioritization
3. **Duration:** Survival analysis for time-to-close

**Key Principle:** Zero leakage - only use information available at prediction time.

---

## Preprocessing (Weather Features)

**⚠️ IMPORTANT:** Weather features are now computed during preprocessing, not feature engineering.

### In `src/preprocessing.py`

The `preprocess_and_merge_external_data()` function automatically:

1. **Converts weather units to US standard:**
   - Temperature: Celsius → Fahrenheit
   - Precipitation: Tenths of mm → Inches
2. Merges base weather data (tavg, tmax, tmin in °F; prcp in inches)
3. Computes **derived weather features:**
   - `heating_degree` = max(0, 65 - tavg) in °F
   - `cooling_degree` = max(0, tavg - 65) in °F
   - `heat_flag` = 1 if tmax ≥ 90°F
   - `freeze_flag` = 1 if tmin ≤ 32°F
4. Computes **rolling precipitation:**
   - `rain_3d` = 3-day rolling sum (inches)
   - `rain_7d` = 7-day rolling sum (inches)

```python
from src.preprocessing import preprocess_and_merge_external_data

# All weather features computed here
df = preprocess_and_merge_external_data()

# Check weather features
weather_cols = ['tavg', 'tmax', 'tmin', 'prcp',
               'heating_degree', 'cooling_degree', 
               'heat_flag', 'freeze_flag',
               'rain_3d', 'rain_7d']
print(df[weather_cols].head())
```

**Why This Change?**
- Avoids duplicate computation
- Weather features available to all downstream tasks
- Consistent feature definitions across models
- Simpler pipeline

---

## H3 Spatial Keys

Add hexagonal spatial keys (H3 resolution 8 ≈ 0.7 km² area).

### Function: `add_h3_keys()`

```python
from src.features import add_h3_keys

df = add_h3_keys(df, lat='latitude', lon='longitude', res=8)

# Adds columns:
# - hex: H3 cell ID
# - day: Date floored to day
# - dow: Day of week (0=Monday)
# - hour: Hour of day
# - month: Month of year
```

**Validation:**
- Checks lat ∈ [-90, 90], lon ∈ [-180, 180]
- Sets `hex=NaN` for invalid coordinates
- Handles missing lat/lon gracefully

---

## Forecast Features (Time-Series)

Build a sparse panel for predicting daily ticket arrivals by location and complaint type.

### Grain
One row per `(hex, complaint_family, day)` with target `y = count(tickets)`.

### Function: `build_forecast_panel()`

```python
from src.features import build_forecast_panel

# Weather features already in df from preprocessing
panel = build_forecast_panel(df, use_population_offset=True)

# Returns sparse panel with columns:
# - hex, complaint_family, day, y
# - dow, month
# - lag1, lag7, roll7, roll14, roll28, momentum, days_since_last
# - nbr_roll7, nbr_roll28 (neighbor aggregates)
# - tavg, prcp, heating_degree, cooling_degree, rain_3d, rain_7d
# - pop_hex, log_pop
```

### Features

| Category | Features | Description |
|----------|----------|-------------|
| **Target** | `y` | Daily ticket count |
| **Calendar** | `dow`, `month` | Day of week, month |
| **Lag** | `lag1`, `lag7` | Values 1 and 7 days ago |
| **Rolling** | `roll7`, `roll14`, `roll28` | Rolling sums |
| **Momentum** | `momentum` | `roll7 / (roll28 + ε)` |
| **Recency** | `days_since_last` | Days since last ticket |
| **Neighbors** | `nbr_roll7`, `nbr_roll28` | H3 k=1 ring sums |
| **Weather** | `tavg`, `prcp`, `heating_degree`, etc. | From preprocessing |
| **Population** | `log_pop` | Log population density |

### Sparsity

The panel is **sparse** - only creates rows for days that exist in data.

```python
# Typical sparsity: >90%
unique_groups = panel.groupby(['hex', 'complaint_family']).ngroups
total_days = (panel['day'].max() - panel['day'].min()).days + 1
max_dense = unique_groups * total_days
actual = len(panel)
sparsity = 1 - (actual / max_dense)

print(f"Sparsity: {sparsity:.1%}")  # Usually >90%
```

### Usage

```python
# For modeling
X = panel[['dow', 'month', 'lag7', 'roll7', 'roll28', 
          'momentum', 'tavg', 'heating_degree', 'log_pop']]
y = panel['y']

from lightgbm import LGBMRegressor
model = LGBMRegressor(objective='poisson')
model.fit(X, y)
```

---

## Triage Features (Classification)

Build ticket-level features for prioritization at creation time.

### Grain
One row per ticket with features available at `created_date`.

### Function: `build_triage_features()`

```python
from src.features import build_triage_features

# Weather features already in df from preprocessing
triage_features, tfidf_matrix, vectorizer = build_triage_features(df)

# Returns:
# 1. Feature DataFrame (numeric + one-hots)
# 2. TF-IDF sparse matrix (from descriptor_clean)
# 3. Fitted TfidfVectorizer
```

### Features

| Category | Features | Description |
|----------|----------|-------------|
| **Temporal** | `hour`, `dow`, `month`, `is_weekend` | When created |
| **Categorical** | One-hot encoded | `complaint_family`, `borough`, `channel`, etc. |
| **Local History** | `geo_family_roll7/28` | As-of rolling counts |
| **Recency** | `days_since_last_geo_family` | Time since last in area |
| **Repeat Site** | `repeat_site_14d/28d` | Prior tickets at location |
| **Due Date** | `due_gap_hours`, `due_is_60d` | SLA features |
| **Weather** | `tavg`, `prcp`, `heat_flag`, `freeze_flag` | From preprocessing |
| **Text** | TF-IDF matrix | Descriptor n-grams |

### Zero Leakage

**✅ Safe to use:**
- `created_date`, `due_date`
- Location (lat, lon, borough, hex)
- Complaint type/descriptor
- Historical aggregates (as-of joins)

**❌ Never use (leakage):**
- `status`, `closed_date`
- `resolution_description`, `resolution_outcome`
- `time_to_resolution*` fields

### As-Of Joins

Local history features use **backward-looking joins**:

```python
# For each ticket at time T, aggregate prior tickets in same hex+family
# Only uses data from before T (no future information)
history = tickets[tickets['created_date'] < T].groupby(['hex', 'complaint_family']).size()
```

### Usage

```python
# For modeling
X = triage_features.drop(columns=['unique_key'])
y = df.loc[triage_features['unique_key'], 'potential_inspection_trigger']

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

# With TF-IDF
from scipy.sparse import hstack
X_combined = hstack([X.values, tfidf_matrix])
model.fit(X_combined, y)
```

---

## Duration Features (Survival)

Build features for predicting time-to-close with censoring.

### Labels: `build_duration_survival_labels()`

```python
from src.features import build_duration_survival_labels

labels = build_duration_survival_labels(df)

# Returns:
# - unique_key: Ticket identifier
# - duration_days: Actual duration (may be NaN)
# - ttc_days_cens: Censored duration
# - event_observed: 1 if true closure, 0 if censored
# - is_admin_like: 1 if admin auto-close
```

**Censoring Rules:**
1. **Admin auto-close:** [59-62] days → censored at 60.5
2. **Long stale:** >365 days → censored at 365
3. **Open tickets:** No closed_date → censored at 365

### Features: `build_duration_features()`

```python
from src.features import build_duration_features

duration_features = build_duration_features(df)

# Returns all triage features plus:
# - intake_6h, intake_24h: Recent intake volume
# - open_7d_geo_family: Backlog proxy
```

### Queue Pressure Features

| Feature | Description |
|---------|-------------|
| `intake_6h` | Tickets in same region+family in last 6 hours |
| `intake_24h` | Tickets in same region+family in last 24 hours |
| `open_7d_geo_family` | Proxy for open queue (tickets in last 7 days) |

### Usage

```python
# Combine features and labels
data = duration_features.merge(labels, on='unique_key')

# Prepare for survival model
X = data.drop(columns=['unique_key', 'duration_days', 'ttc_days_cens', 
                       'event_observed', 'is_admin_like'])
t = data['ttc_days_cens']  # Time
e = data['event_observed']  # Event indicator

from lifelines import WeibullAFTFitter
model = WeibullAFTFitter()
model.fit(X, duration_col=t, event_col=e)
```

---

## API Reference

### Main Functions

```python
# Spatial keys
add_h3_keys(df, lat='latitude', lon='longitude', res=8) -> pd.DataFrame

# Forecast
build_forecast_panel(df, use_population_offset=True) -> pd.DataFrame

# Triage
build_triage_features(df) -> Tuple[pd.DataFrame, csr_matrix, TfidfVectorizer]

# Duration
build_duration_survival_labels(df) -> pd.DataFrame
build_duration_features(df, panel=None) -> pd.DataFrame
```

### Utility Functions (src/utils.py)

These are mostly used internally but available if needed:

```python
# H3 operations
lat_lon_to_h3(lat, lon, resolution=8) -> str
get_h3_neighbors(hex_id, k=1) -> List[str]
expand_k_ring(df_panel, k=1, agg_cols=['roll7', 'roll28']) -> pd.DataFrame

# Rolling features
compute_rolling_as_of(group_df, date_col='day', value_col='y', windows=[7,14,28]) -> pd.DataFrame
compute_lag_features(group_df, date_col='day', value_col='y', lags=[1,7]) -> pd.DataFrame
compute_days_since_last(group_df, date_col='day') -> pd.DataFrame

# Text
make_descriptor_tfidf(df, col='descriptor_clean', min_df=5) -> Tuple[csr_matrix, TfidfVectorizer]

# Time-based rolling
compute_time_based_rolling_counts(df, timestamp_col, group_cols, window_hours) -> pd.DataFrame
```

**Note:** Weather feature functions (`add_weather_derived_features`, `compute_rain_rolling`) are now in `src/preprocessing.py`.

---

## Best Practices

### 1. Use Preprocessing for Weather

```python
# ✅ Good: Weather computed once in preprocessing
df = preprocess_and_merge_external_data()  # Weather features added here
panel = build_forecast_panel(df)  # Uses pre-computed features

# ❌ Bad: Not using preprocessing (weather features missing)
panel = build_forecast_panel(df)  # Error: weather features not found!
```

### 2. Cache Intermediate Results

```python
# Save preprocessed data
df.to_parquet('data/processed/train_with_weather.parquet')

# Load for feature engineering
df = pd.read_parquet('data/processed/train_with_weather.parquet')
```

### 3. Validate Features

```python
# Check for leakage
leakage_cols = ['status', 'closed_date', 'resolution_description']
assert not any(col in triage_features.columns for col in leakage_cols)

# Check weather features
assert 'heating_degree' in df.columns, "Run preprocessing first"

# Check sparsity
sparsity = 1 - (len(panel) / (panel.groupby(['hex', 'complaint_family']).ngroups * total_days))
assert sparsity > 0.9, "Panel should be sparse"
```

### 4. Handle Missing H3

```python
# Filter valid hexes
valid_hex = df['hex'].notna()
print(f"Valid locations: {valid_hex.mean():.1%}")

df_valid = df[valid_hex]
```

---

## Troubleshooting

### Q: Weather features missing?
**A:** Run preprocessing first: `df = preprocess_and_merge_external_data()`

### Q: Panel too large?
**A:** It's sparse by design. Filter to specific date ranges or families if needed.

### Q: TF-IDF fails?
**A:** Ensure `descriptor_clean` exists and has text: `df['descriptor_clean'].notna().mean()`

### Q: H3 import error?
**A:** Install: `pip install h3`

---

**Next:** See [TRAINING.md](TRAINING.md) for model training guide.

