# API Reference

Comprehensive API documentation for NYC 311 feature engineering and model training.

## 📚 Module Reference

### Preprocessing (`src/preprocessing.py`)

Core data preprocessing and weather feature engineering.

```python
# Main preprocessing function (weather features computed here)
preprocess_and_merge_external_data() -> pd.DataFrame

# Component functions
create_date_features(df: pd.DataFrame) -> pd.DataFrame
map_freetext_columns(df: pd.DataFrame, mappings_path: str) -> pd.DataFrame
filter_and_clean(df: pd.DataFrame) -> pd.DataFrame
merge_census_data(df: pd.DataFrame, census_path: str, shapefile_path: str) -> pd.DataFrame

# Weather feature engineering (called automatically by merge_weather_data)
add_weather_derived_features(df: pd.DataFrame) -> pd.DataFrame
compute_rain_rolling(df: pd.DataFrame, windows: list = [3, 7]) -> pd.DataFrame
merge_weather_data(df: pd.DataFrame, weather_data_path: str) -> pd.DataFrame
```

---

### Feature Engineering (`src/features.py`)

Build model-ready features for forecast, triage, and duration tasks.

```python
# H3 spatial keys
add_h3_keys(
    df: pd.DataFrame, 
    lat: str = 'latitude', 
    lon: str = 'longitude', 
    res: int = 8
) -> pd.DataFrame

# Forecast features (time-series panel)
build_forecast_panel(
    df: pd.DataFrame,
    use_population_offset: bool = True
) -> pd.DataFrame
# Note: Weather features must already be present in df from preprocessing

# Triage features (classification)
build_triage_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, csr_matrix, TfidfVectorizer]

# Duration labels (survival analysis)
build_duration_survival_labels(
    df: pd.DataFrame
) -> pd.DataFrame

# Duration features
build_duration_features(
    df: pd.DataFrame,
    panel: Optional[pd.DataFrame] = None
) -> pd.DataFrame
```

---

### Forecast Model (`models/forecast.py`)

LightGBM Poisson regression for time-series forecasting.

```python
# Train single family
train_forecast_per_family(
    panel: pd.DataFrame,
    family: str,
    horizons: List[int] = list(range(1, 8)),
    feature_cols: Optional[List[str]] = None,
    val_days: int = 30,
    params: Optional[Dict] = None
) -> Dict

# Train all families
train_all_families(
    panel: pd.DataFrame,
    families: Optional[List[str]] = None,
    horizons: List[int] = list(range(1, 8)),
    val_days: int = 30,
    params: Optional[Dict] = None
) -> Dict[str, Dict]

# Predict
predict_forecast(
    bundles: Dict[str, Dict],
    last_rows: pd.DataFrame,
    horizon: int = 7
) -> pd.DataFrame

# Persistence
save_bundles(bundles: Dict[str, Dict], output_dir: Path) -> None
load_bundles(input_dir: Path) -> Dict[str, Dict]
```

---

### Triage Model (`models/triage.py`)

LightGBM classifier with per-family isotonic calibration.

```python
# Train
train_triage(
    dfX: pd.DataFrame,
    y: pd.Series,
    families: pd.Series,
    params: Optional[Dict] = None,
    calibration_threshold: int = 500,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict

# Predict
predict_triage(
    bundle: Dict,
    dfX_new: pd.DataFrame,
    families_new: pd.Series
) -> np.ndarray

# Persistence
save_bundle(bundle: Dict, output_path: Path) -> None
load_bundle(input_path: Path) -> Dict
```

---

### Duration Model (`models/duration.py`)

AFT survival models (Weibull, LogNormal).

```python
# Train single model
train_duration_aft(
    X: pd.DataFrame,
    t: pd.Series,
    e: pd.Series,
    model_type: str = 'weibull',
    test_size: float = 0.2,
    random_state: int = 42,
    penalizer: float = 0.01
) -> Dict

# Compare models
compare_models(
    X: pd.DataFrame,
    t: pd.Series,
    e: pd.Series,
    models: List[str] = ['weibull', 'lognormal'],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict, str]

# Predict
predict_duration_quantiles(
    bundle: Dict,
    X_new: pd.DataFrame,
    ps: Tuple[float, ...] = (0.5, 0.9)
) -> pd.DataFrame

# Persistence
save_bundle(bundle: Dict, output_path: Path) -> None
load_bundle(input_path: Path) -> Dict
```

---

### Hyperparameter Tuning (`models/tune.py`)

Optuna-based hyperparameter optimization.

```python
# Tune forecast model
tune_forecast(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: Optional[list] = None,
    n_trials: int = 30,
    random_state: int = 42
) -> Dict

# Tune triage model
tune_triage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
    random_state: int = 42
) -> Dict

# Tune duration model
tune_duration_aft(
    X: pd.DataFrame,
    t: pd.Series,
    e: pd.Series,
    test_size: float = 0.2,
    n_trials: int = 30,
    random_state: int = 42
) -> Dict

# Tune all models
run_tuning_suite(
    forecast_data: Optional[Dict] = None,
    triage_data: Optional[Dict] = None,
    duration_data: Optional[Dict] = None,
    n_trials: int = 30,
    random_state: int = 42
) -> Dict
```

---

### Evaluation (`models/eval.py`)

Comprehensive metrics and visualizations.

```python
# Evaluate forecast
eval_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    family: str = "Unknown",
    horizon: int = 1
) -> Dict

# Evaluate triage
eval_triage(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    families: Optional[pd.Series] = None
) -> Dict

# Evaluate duration
eval_duration(
    t_true: pd.Series,
    t_pred: pd.Series,
    e_true: pd.Series,
    quantile: float = 0.5
) -> Dict

# Visualization
plot_forecast_calibration(y_true, y_pred, title, output_path) -> None
plot_triage_curves(y_true, y_pred_proba, title, output_path) -> None
plot_duration_reliability(t_true, t_pred_q50, t_pred_q90, e_true, title, output_path) -> None

# Utilities
save_metrics(metrics: Dict, output_path: Path) -> None
print_metrics_summary(metrics: Dict, model_type: str) -> None
```

---

### Utilities (`src/utils.py`)

Helper functions for feature engineering.

```python
# H3 operations
validate_lat_lon(lat: pd.Series, lon: pd.Series) -> pd.Series
lat_lon_to_h3(lat: float, lon: float, resolution: int = 8) -> Optional[str]
get_h3_neighbors(hex_id: str, k: int = 1) -> List[str]
expand_k_ring(df_panel, k=1, hex_col='hex', agg_cols=['roll7', 'roll28']) -> pd.DataFrame

# Rolling features
compute_rolling_as_of(group_df, date_col='day', value_col='y', windows=[7,14,28]) -> pd.DataFrame
compute_lag_features(group_df, date_col='day', value_col='y', lags=[1,7]) -> pd.DataFrame
compute_days_since_last(group_df, date_col='day') -> pd.DataFrame

# Text features
make_descriptor_tfidf(df, col='descriptor_clean', min_df=5, ngram_range=(1,2)) -> Tuple[csr_matrix, TfidfVectorizer]

# As-of joins
merge_asof_by_group(df_tickets, df_panel, by_cols, left_on='day', right_on='day') -> pd.DataFrame

# Time-based rolling
compute_time_based_rolling_counts(df, timestamp_col, group_cols, window_hours) -> pd.DataFrame
```

**Note:** Weather utility functions (`add_weather_derived_features`, `compute_rain_rolling`) have been moved to `src/preprocessing.py`.

---

## 🔄 Typical Workflow

```python
# 1. Preprocessing (weather features computed here)
from src.preprocessing import preprocess_and_merge_external_data
df = preprocess_and_merge_external_data()

# 2. Add H3 keys
from src.features import add_h3_keys
df = add_h3_keys(df)

# 3A. Forecast workflow
from src.features import build_forecast_panel
from models.forecast import train_all_families, predict_forecast

panel = build_forecast_panel(df)
bundles = train_all_families(panel)
predictions = predict_forecast(bundles, last_rows, horizon=7)

# 3B. Triage workflow
from src.features import build_triage_features
from models.triage import train_triage, predict_triage

features, tfidf, vectorizer = build_triage_features(df)
X = features.drop(columns=['unique_key'])
y = df.loc[features['unique_key'], 'target']
families = df.loc[features['unique_key'], 'complaint_family']

bundle = train_triage(X, y, families)
probabilities = predict_triage(bundle, X_new, families_new)

# 3C. Duration workflow
from src.features import build_duration_features, build_duration_survival_labels
from models.duration import train_duration_aft, predict_duration_quantiles

features = build_duration_features(df)
labels = build_duration_survival_labels(df)

data = features.merge(labels, on='unique_key')
X = data.drop(columns=['unique_key', 'duration_days', 'ttc_days_cens', 'event_observed', 'is_admin_like'])
t = data['ttc_days_cens']
e = data['event_observed']

bundle = train_duration_aft(X, t, e, model_type='weibull')
quantiles = predict_duration_quantiles(bundle, X_new, ps=(0.5, 0.9))
```

---

## 📋 Data Types

### Feature DataFrames

```python
# Forecast panel
pd.DataFrame with columns:
    hex: str
    complaint_family: str
    day: datetime
    y: float (target)
    dow, month: int
    lag1, lag7, roll7, roll14, roll28, momentum: float
    days_since_last: int
    tavg, prcp, heating_degree, cooling_degree, rain_3d, rain_7d: float
    log_pop: float

# Triage features
pd.DataFrame with columns:
    unique_key: int/str
    hour, dow, month, is_weekend: int
    <categorical_feature>_<category>: int (one-hot)
    geo_family_roll7, geo_family_roll28, days_since_last_geo_family: float
    repeat_site_14d, repeat_site_28d: int
    due_gap_hours: float
    due_is_60d, due_crosses_weekend: int
    tavg, prcp, heat_flag, freeze_flag: float/int

# Duration labels
pd.DataFrame with columns:
    unique_key: int/str
    duration_days: float
    ttc_days_cens: float
    event_observed: int (0/1)
    is_admin_like: int (0/1)
```

### Model Bundles

```python
# Forecast bundle (per family)
{
    'family': str,
    'feature_cols': List[str],
    'cat_cols': List[str],
    'models': {horizon: LGBMRegressor},
    'metrics': {horizon: dict},
    'horizons': List[int]
}

# Triage bundle
{
    'model': LGBMClassifier,
    'feature_cols': List[str],
    'calibrators': {family: IsotonicRegression},
    'metrics': dict
}

# Duration bundle
{
    'model': WeibullAFTFitter or LogNormalAFTFitter,
    'transformer': ColumnTransformer,
    'metrics': dict,
    'model_type': str,
    'feature_cols': List[str]
}
```

---

**Previous:** [TRAINING.md](TRAINING.md) | **Home:** [README](README.md)

