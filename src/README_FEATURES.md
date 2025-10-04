# NYC 311 Feature Engineering Module

Complete feature engineering for NYC DOHMH 311 tickets with three modeling tracks: **Forecast**, **Triage**, and **Duration**.

## 📚 Documentation Index

| Document | Purpose | Audience |
|----------|---------|----------|
| **[QUICK_START.md](QUICK_START.md)** | Get started in 5 minutes | Developers |
| **[FEATURE_ENGINEERING_README.md](FEATURE_ENGINEERING_README.md)** | Complete reference guide | All users |
| **[../FEATURE_ENGINEERING_SUMMARY.md](../FEATURE_ENGINEERING_SUMMARY.md)** | Implementation details | Architects/Reviewers |

## 🚀 Quick Start

```python
from src.features import add_h3_keys, build_forecast_panel, build_triage_features

# Load data
df = pd.read_parquet('data/...')

# Add spatial keys
df = add_h3_keys(df)

# Build features
forecast_panel = build_forecast_panel(df, weather_df=weather)
triage_features, tfidf, vectorizer = build_triage_features(df)
```

## 📦 What's Included

### Core Modules
- **`utils.py`** - Helper functions (H3, rolling, as-of joins, TF-IDF)
- **`features.py`** - Main feature builders (forecast, triage, duration)

### Examples
- **`feature_engineering_example.py`** - Runnable demo with validation
- **`../notebooks/Step 5 - Feature Engineering.ipynb`** - Interactive walkthrough

### Documentation
- **`QUICK_START.md`** - Quick reference (5 min read)
- **`FEATURE_ENGINEERING_README.md`** - Full docs (30 min read)
- **`../FEATURE_ENGINEERING_SUMMARY.md`** - Technical summary (15 min read)

## 🎯 Three Modeling Tracks

### 1️⃣ Forecast (Time-Series)
**Goal:** Predict daily ticket arrivals by location and complaint type

**Function:** `build_forecast_panel()`

**Features:**
- Lag features (1, 7 days)
- Rolling features (7, 14, 28 days)
- Momentum (trend)
- Weather (temp, precip, degree days)
- Calendar (day of week, month)
- Population offset

**Output:** Sparse panel with one row per (hex, complaint_family, day)

### 2️⃣ Triage (Classification)
**Goal:** Prioritize tickets at creation time (e.g., inspection triggers)

**Function:** `build_triage_features()`

**Features:**
- Temporal (hour, day of week, month)
- Location history (geo-family rolling counts)
- Repeat site (14d, 28d counts)
- Due date (gap, flags)
- Weather at creation
- Text (TF-IDF from descriptor)
- Categorical one-hots

**Output:** Feature DataFrame + TF-IDF sparse matrix

### 3️⃣ Duration (Survival)
**Goal:** Predict time-to-close with censoring

**Functions:** 
- `build_duration_survival_labels()` - Labels with censoring
- `build_duration_features()` - Triage features + queue pressure

**Features:**
- All triage features
- Queue pressure (intake 6h, 24h)
- Open backlog proxy

**Output:** Feature DataFrame + labels with censoring

## ✅ Key Features

- ✓ **H3 spatial grouping** (resolution 8, ~0.7 km²)
- ✓ **Sparse panels** (>90% sparsity, no dense calendars)
- ✓ **Zero leakage** (as-of joins, no post-creation data)
- ✓ **Vectorized** (fast pandas operations, no Python loops)
- ✓ **Production-ready** (type hints, docstrings, validation)

## 📊 Example Output

### Forecast Panel
```
   hex           complaint_family  day         y  lag7  roll7  roll28  momentum  tavg  log_pop
   8a2a100d28d  Health           2024-01-15  12    8     45     120     0.375    32.1  8.52
   8a2a100d28d  Noise            2024-01-15   5    6     28      85     0.329    32.1  8.52
   ...
```

### Triage Features
```
   unique_key  hour  dow  month  geo_family_roll7  repeat_site_14d  due_gap_hours  tavg  ...
   123456        14    2      1                32               2          1440.0  32.1  ...
   123457         9    3      1                18               0          1440.0  28.5  ...
   ...
```

### Duration Labels
```
   unique_key  duration_days  ttc_days_cens  event_observed  is_admin_like
   123456                3.2            3.2               1              0
   123457               61.5           60.5               0              1  # Admin auto-close
   123458              400.0          365.0               0              0  # Long stale
   ...
```

## 🏃 Running Examples

### Terminal
```bash
# Install dependencies
pip install -r requirements.txt

# Run example script
python src/feature_engineering_example.py
```

### Jupyter
```bash
# Launch notebook
jupyter notebook notebooks/Step\ 5\ -\ Feature\ Engineering.ipynb
```

## 🔍 Validation

All features include built-in validation:

```python
from src.feature_engineering_example import example_pipeline

results = example_pipeline(df, weather_df)

# Automatic checks:
# ✓ No leakage columns
# ✓ Panel sparsity >90%
# ✓ All critical features present
# ✓ Reasonable censoring rates
```

## 📈 Model Workflows

### Forecast
```python
from lightgbm import LGBMRegressor

X = panel[['dow', 'month', 'lag7', 'roll7', 'roll28', 'momentum', 'tavg', 'log_pop']]
y = panel['y']

model = LGBMRegressor()
model.fit(X, y)
```

### Triage
```python
from sklearn.ensemble import RandomForestClassifier

X = triage_features.drop(columns=['unique_key'])
y = df.loc[triage_features['unique_key'], 'potential_inspection_trigger']

model = RandomForestClassifier()
model.fit(X, y)
```

### Duration
```python
from sksurv.ensemble import RandomSurvivalForest

data = duration_features.merge(duration_labels, on='unique_key')
X = data.drop(columns=['unique_key', 'duration_days', 'ttc_days_cens', 
                       'event_observed', 'is_admin_like'])
y = np.array([(bool(e), t) for e, t in zip(data['event_observed'], data['ttc_days_cens'])],
             dtype=[('event', bool), ('time', float)])

model = RandomSurvivalForest()
model.fit(X, y)
```

## 🛠️ Requirements

```
pandas>=1.5.0
numpy>=1.23.0
h3>=3.7.0
scikit-learn>=1.2.0
scipy>=1.10.0
```

## 💡 Tips

1. **Start with QUICK_START.md** for immediate usage
2. **Read FEATURE_ENGINEERING_README.md** for deep dive
3. **Run feature_engineering_example.py** to test
4. **Open Step 5 notebook** for interactive exploration
5. **Check FEATURE_ENGINEERING_SUMMARY.md** for architecture details

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing H3 keys | Check lat/lon validity with `df['hex'].notna().mean()` |
| Panel too large | Filter to date range or complaint families first |
| TF-IDF fails | Ensure `descriptor_clean` has non-empty text |
| Import errors | Run `pip install -r requirements.txt` |

## 📞 Support

1. Check documentation (see index above)
2. Run examples (script or notebook)
3. Review validation output

## 🎓 Learn More

- **H3 Hexagons:** https://h3geo.org/
- **Survival Analysis:** https://scikit-survival.readthedocs.io/
- **Leakage Prevention:** See "Leakage Prevention" section in main README

## 📄 License

[Your License Here]

---

**Version:** 1.0  
**Last Updated:** October 4, 2025  
**Status:** ✅ Production Ready

