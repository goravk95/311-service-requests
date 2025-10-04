# NYC 311 Feature Engineering

Comprehensive, leakage-safe feature engineering for three modeling tracks: **Forecast**, **Triage**, and **Duration**.

## Overview

This module provides fast, vectorized feature builders that emphasize:
- **H3-based spatial grouping** (resolution 8)
- **Sparse time-series panels** (no dense calendar expansion)
- **As-of joins** for historical features (zero leakage)
- **Queue pressure metrics** for duration modeling
- **TF-IDF text features** for descriptors

## Core Modules

### `utils.py`
Helper functions for:
- H3 spatial operations and neighbor expansion
- As-of rolling aggregations (lag/rolling features)
- Time-based rolling counts
- TF-IDF text vectorization
- Weather feature engineering

### `features.py`
Main feature engineering functions:
- `add_h3_keys()` - Add spatial and temporal keys
- `build_forecast_panel()` - Time-series panel for forecasting
- `build_triage_features()` - Ticket-level features for prioritization
- `build_duration_survival_labels()` - Labels with censoring
- `build_duration_features()` - Duration model features

## Quick Start

```python
import pandas as pd
from features import (
    add_h3_keys,
    build_forecast_panel,
    build_triage_features,
    build_duration_survival_labels,
    build_duration_features
)

# Load your data
df = pd.read_parquet('data/landing/311-service-requests/...')

# Step 1: Add H3 spatial keys
df = add_h3_keys(df, lat='latitude', lon='longitude', res=8)

# Step 2: Build forecast panel (time-series)
forecast_panel = build_forecast_panel(df, weather_df=weather, use_population_offset=True)

# Step 3: Build triage features (ticket-level)
triage_features, tfidf_matrix, vectorizer = build_triage_features(df)

# Step 4: Build duration labels
duration_labels = build_duration_survival_labels(df)

# Step 5: Build duration features
duration_features = build_duration_features(df, panel=forecast_panel)
```

## Function Reference

### 1. `add_h3_keys()`

Adds H3 spatial keys and temporal features.

**Arguments:**
- `df`: Input DataFrame
- `lat`: Latitude column name (default: `'latitude'`)
- `lon`: Longitude column name (default: `'longitude'`)
- `res`: H3 resolution (default: `8`)

**Returns:**
DataFrame with added columns:
- `hex`: H3 cell ID
- `day`: Date floored to day
- `dow`: Day of week (0=Monday)
- `hour`: Hour of day
- `month`: Month of year

**Example:**
```python
df = add_h3_keys(df, lat='latitude', lon='longitude', res=8)
print(df[['hex', 'day', 'dow', 'hour', 'month']].head())
```

---

### 2. `build_forecast_panel()`

Builds a sparse time-series panel for forecasting ticket arrivals.

**Grain:** One row per `(hex, complaint_family, day)` combination.

**Arguments:**
- `df`: DataFrame with H3 keys added
- `weather_df`: Optional weather data with columns `[day, fips, tmax, tmin, tavg, prcp]`
- `use_population_offset`: Include population features (default: `True`)

**Returns:**
Sparse panel DataFrame with columns:
- **Identifiers:** `hex`, `complaint_family`, `day`
- **Target:** `y` (count of tickets)
- **Calendar:** `dow`, `month`
- **History:** `lag1`, `lag7`, `roll7`, `roll14`, `roll28`, `momentum`, `days_since_last`
- **Neighbors:** `nbr_roll7`, `nbr_roll28` (k=1 ring aggregates)
- **Weather:** `tavg`, `prcp`, `heating_degree`, `cooling_degree`, `rain_3d`, `rain_7d`
- **Population:** `pop_hex`, `log_pop`

**Key Features:**
- **Sparse panels:** Only creates rows for days that exist in data (no full calendars)
- **Per-group history:** Lag and rolling features computed separately for each `(hex, complaint_family)` group
- **Neighbor aggregates:** Sums `roll7` and `roll28` over H3 k=1 ring
- **Weather joins:** County-level weather matched by `fips` and `day`

**Example:**
```python
panel = build_forecast_panel(df, weather_df=weather, use_population_offset=True)
print(f"Panel shape: {panel.shape}")
print(f"Unique groups: {panel.groupby(['hex', 'complaint_family']).ngroups}")
```

---

### 3. `build_triage_features()`

Builds ticket-level features for triage/prioritization (NO LEAKAGE).

**Grain:** One row per ticket at creation time.

**Arguments:**
- `df`: DataFrame with H3 keys and temporal features

**Returns:**
Tuple of:
1. **Feature DataFrame** with columns:
   - **Temporal:** `hour`, `dow`, `month`, `is_created_at_midnight`, `is_weekend`
   - **Categorical one-hots:** `complaint_family_*`, `borough_*`, `location_type_*`, etc.
   - **Due date:** `due_gap_hours`, `due_is_60d`, `due_crosses_weekend`
   - **Weather:** `tavg`, `prcp`, `heat_flag`, `freeze_flag`
   - **Local history (as-of):** `geo_family_roll7`, `geo_family_roll28`, `days_since_last_geo_family`
   - **Repeat-site:** `repeat_site_14d`, `repeat_site_28d`
2. **TF-IDF sparse matrix** (from `descriptor_clean`)
3. **Fitted TfidfVectorizer**

**Key Features:**
- **Zero leakage:** Uses only creation-time fields (no `status`, `resolution_*`, `closed_date`)
- **As-of history:** Merges historical aggregates backward in time (no future information)
- **Repeat-site tracking:** Counts prior tickets at same location (using `bbl` or address hash)
- **Text features:** TF-IDF with 1-2 grams, min_df=5, max 500 features

**Example:**
```python
triage_features, tfidf_matrix, vectorizer = build_triage_features(df)
print(f"Features shape: {triage_features.shape}")
print(f"TF-IDF shape: {tfidf_matrix.shape}")

# Check for leakage
leakage_cols = ['status', 'closed_date', 'resolution_description']
assert not any(col in triage_features.columns for col in leakage_cols)
```

---

### 4. `build_duration_survival_labels()`

Builds labels for duration/survival modeling with right-censoring.

**Grain:** One row per ticket.

**Arguments:**
- `df`: DataFrame with `created_date` and `closed_date`

**Returns:**
DataFrame with columns:
- `unique_key`: Ticket identifier
- `duration_days`: Actual duration (may be NaN for open tickets)
- `ttc_days_cens`: Censored duration (capped at 60.5 or 365)
- `event_observed`: 1 if true closure, 0 if censored
- `is_admin_like`: 1 if duration ∈ [59, 62] days (admin auto-close)

**Censoring Rules:**
1. **Admin auto-close:** Duration ∈ [59, 62] days → censored at 60.5
2. **Long stale:** Duration > 365 days → censored at 365
3. **Open tickets:** No `closed_date` → censored at 365

**Example:**
```python
labels = build_duration_survival_labels(df)
print(f"Event observed: {labels['event_observed'].sum()}")
print(f"Censored: {(labels['event_observed'] == 0).sum()}")
print(f"Admin-like: {labels['is_admin_like'].sum()}")
```

---

### 5. `build_duration_features()`

Builds features for duration prediction (combines triage + queue pressure).

**Grain:** One row per ticket at creation time.

**Arguments:**
- `df`: DataFrame with H3 keys and temporal features
- `panel`: Optional forecast panel for queue metrics

**Returns:**
Feature DataFrame combining:
- **All triage features** (from `build_triage_features()`)
- **Intake pressure:** `intake_6h`, `intake_24h` (rolling ticket counts in same region+family)
- **Queue backlog:** `open_7d_geo_family` (proxy for open items)
- **Due date features:** `due_gap_hours`, `due_is_60d`, `due_crosses_weekend`

**Key Features:**
- **Queue pressure:** Measures system load at creation time
- **Intake surge detection:** 6h and 24h rolling counts in same `fips×complaint_family`
- **Backlog proxy:** Count of recent tickets in same `hex×family` (approximates open queue)

**Example:**
```python
duration_features = build_duration_features(df, panel=forecast_panel)
print(f"Features shape: {duration_features.shape}")
print(duration_features[['intake_6h', 'intake_24h', 'open_7d_geo_family']].describe())
```

## Modeling Workflows

### Forecast Model (Time-Series)
```python
# Build panel
panel = build_forecast_panel(df, weather_df=weather)

# Train/test split (by time)
train_panel = panel[panel['day'] < '2024-01-01']
test_panel = panel[panel['day'] >= '2024-01-01']

# Features and target
feature_cols = ['dow', 'month', 'lag7', 'roll7', 'roll28', 'momentum',
                'tavg', 'heating_degree', 'log_pop']
X_train = train_panel[feature_cols]
y_train = train_panel['y']

# Train model (e.g., LightGBM, Prophet, etc.)
```

### Triage Model (Classification)
```python
# Build features
triage_features, tfidf_matrix, vectorizer = build_triage_features(df)

# Create target (e.g., high priority if inspection trigger)
y = df.loc[triage_features['unique_key'], 'potential_inspection_trigger']

# Combine dense and sparse features
from scipy.sparse import hstack
X_dense = triage_features.drop(columns=['unique_key']).values
X_combined = hstack([X_dense, tfidf_matrix])

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_combined, y)
```

### Duration Model (Survival)
```python
# Build features and labels
duration_features = build_duration_features(df)
duration_labels = build_duration_survival_labels(df)

# Merge
model_data = duration_features.merge(duration_labels, on='unique_key')

# Train survival model
from sksurv.ensemble import RandomSurvivalForest
X = model_data.drop(columns=['unique_key', 'duration_days', 'ttc_days_cens', 
                             'event_observed', 'is_admin_like'])
y = np.array(list(zip(model_data['event_observed'].astype(bool), 
                      model_data['ttc_days_cens'])),
             dtype=[('event', bool), ('time', float)])

model = RandomSurvivalForest()
model.fit(X, y)
```

## Performance Considerations

### Sparse Panels
The forecast panel is **sparse** - it only creates rows for `(hex, complaint_family, day)` combinations that exist in the data. This avoids exploding memory for unused combinations.

**Good:**
```python
# Only creates ~100K rows for active groups
panel = build_forecast_panel(df)  
```

**Bad (don't do this):**
```python
# Would create millions of rows for all possible combinations
all_combos = pd.MultiIndex.from_product([
    df['hex'].unique(), 
    df['complaint_family'].unique(),
    pd.date_range('2010-01-01', '2025-09-30')
])
```

### As-Of Joins
Historical features use `merge_asof` with `direction='backward'` to ensure no future leakage.

### Vectorized Operations
All feature computations are vectorized (no Python loops over rows):
- Lag features: Dict lookups via `pd.Series.map()`
- Rolling features: Cumulative sums + as-of indexing
- Time-based rolling: `groupby().rolling()` with time windows

### Memory Management
- Text features: Sparse TF-IDF matrix (not dense)
- Categorical features: Top-N encoding (limits explosion)
- Missing values: Fill with 0 or "_missing" category

## Data Requirements

### Required Columns
```python
required = [
    'created_date',      # Datetime
    'latitude',          # Float
    'longitude',         # Float
    'complaint_family',  # String
]
```

### Recommended Columns
```python
recommended = [
    'closed_date',       # Datetime (for labels)
    'due_date',          # Datetime
    'descriptor_clean',  # String (for TF-IDF)
    'borough',           # String
    'fips',              # String (for weather join)
    'bbl',               # String (for repeat-site)
    'open_data_channel_type',  # String
    'location_type',     # String
    'facility_type',     # String
    'address_type',      # String
]
```

### Optional Columns
```python
optional = [
    'GEOID',            # String (for population)
    'population',       # Float
    'tavg', 'tmax', 'tmin', 'prcp',  # Weather (or pass separate weather_df)
]
```

## Validation & Testing

### Leakage Checks
```python
# Ensure no post-creation fields in features
leakage_cols = [
    'status', 'closed_date', 'resolution_description',
    'resolution_outcome', 'resolution_action_updated_date',
    'time_to_resolution', 'time_to_resolution_hours',
    'time_to_resolution_days'
]

triage_features, _, _ = build_triage_features(df)
leakage_found = [c for c in leakage_cols if c in triage_features.columns]
assert len(leakage_found) == 0, f"Leakage detected: {leakage_found}"
```

### Temporal Ordering
```python
# Ensure as-of joins use only past data
df_sorted = df.sort_values('created_date')
triage_features, _, _ = build_triage_features(df_sorted)

# Check that history features only use prior data
for idx, row in df_sorted.head(100).iterrows():
    ticket_date = row['created_date']
    # Verify history features only aggregate tickets before ticket_date
```

### Sparsity Check
```python
# Forecast panel should be sparse
panel = build_forecast_panel(df)
unique_groups = panel.groupby(['hex', 'complaint_family']).ngroups
total_days = (panel['day'].max() - panel['day'].min()).days + 1
max_dense_rows = unique_groups * total_days
actual_rows = len(panel)

sparsity = 1 - (actual_rows / max_dense_rows)
print(f"Panel sparsity: {sparsity:.1%}")  # Should be >90%
```

## Example: Full Pipeline

See `feature_engineering_example.py` for a complete end-to-end example:

```bash
python src/feature_engineering_example.py
```

This will:
1. Load sample data (or generate synthetic)
2. Run all feature builders
3. Perform quality checks
4. Display sample outputs

## Dependencies

```txt
pandas>=1.5.0
numpy>=1.23.0
h3>=3.7.0
scikit-learn>=1.2.0
scipy>=1.10.0
```

Install with:
```bash
pip install pandas numpy h3 scikit-learn scipy
```

## FAQ

**Q: Can I use a different H3 resolution?**  
A: Yes, pass `res=` to `add_h3_keys()`. Note that smaller resolutions (e.g., 7) create larger areas, while larger (e.g., 9) create smaller areas.

**Q: What if I don't have weather data?**  
A: The functions will fill weather features with 0. Pass `weather_df=None` to skip weather joins.

**Q: How do I handle missing lat/lon?**  
A: The functions validate lat/lon and set `hex=NaN` for invalid coordinates. These rows are filtered out in `build_forecast_panel()`.

**Q: Can I customize TF-IDF parameters?**  
A: Yes, edit the `make_descriptor_tfidf()` call in `build_triage_features()` or call it separately with custom parameters.

**Q: How do I add new features?**  
A: Extend the functions or create new ones following the same leakage-safe patterns (use only pre-creation data, as-of joins, etc.).

**Q: What about multi-county weather?**  
A: The weather join uses `fips` to match county-level weather. Ensure your `weather_df` has all relevant FIPS codes.

## Citation

If you use this feature engineering code, please cite:

```
NYC 311 Service Requests Feature Engineering
Author: [Your Team]
Year: 2024
GitHub: [your-repo]
```

## License

[Your License Here]

