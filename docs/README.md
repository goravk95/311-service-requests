# NYC 311 Service Requests - Documentation

Complete documentation for feature engineering and model training.

## 📚 Documentation Structure

| Document | Description | Read Time |
|----------|-------------|-----------|
| **[QUICK_START.md](QUICK_START.md)** | Get started in 5 minutes | 5 min |
| **[FEATURES.md](FEATURES.md)** | Feature engineering guide | 20 min |
| **[TRAINING.md](TRAINING.md)** | Model training guide | 20 min |
| **[API_REFERENCE.md](API_REFERENCE.md)** | Complete API reference | 30 min |

## 🚀 Quick Links

### For Data Scientists
- **New to the project?** Start with [QUICK_START.md](QUICK_START.md)
- **Building features?** See [FEATURES.md](FEATURES.md)
- **Training models?** See [TRAINING.md](TRAINING.md)

### For Engineers
- **API reference:** [API_REFERENCE.md](API_REFERENCE.md)
- **Module code:** `src/features.py`, `models/*.py`, `pipelines/*.py`

## 🎯 What's Included

### Feature Engineering (`src/features.py`)
- ✅ H3 spatial grouping (resolution 8)
- ✅ Time-series features (lag, rolling, momentum)
- ✅ Weather features (pre-computed in preprocessing)
- ✅ Zero-leakage as-of joins
- ✅ Sparse panel construction

### Model Training (`models/`)
- ✅ **Forecast:** LightGBM Poisson (multi-horizon)
- ✅ **Triage:** LightGBM + isotonic calibration
- ✅ **Duration:** AFT survival models
- ✅ Hyperparameter tuning (Optuna)
- ✅ Comprehensive evaluation

### Pipelines (`pipelines/`)
- ✅ Training CLI (`train.py`)
- ✅ Prediction CLI (`predict.py`)
- ✅ Batch processing
- ✅ Model persistence

## 📊 Workflow

```
Raw Data
   ↓
Preprocessing (src/preprocessing.py)
   ├── Clean & map freetext
   ├── Merge census data
   └── Merge weather + derived features ← WEATHER COMPUTED HERE
   ↓
Feature Engineering (src/features.py)
   ├── Add H3 spatial keys
   ├── Build forecast panel
   ├── Build triage features
   └── Build duration features
   ↓
Model Training (models/*.py)
   ├── Forecast (LightGBM Poisson)
   ├── Triage (LightGBM + calibration)
   └── Duration (AFT survival)
   ↓
Prediction (pipelines/predict.py)
   └── Generate forecasts/probabilities/durations
```

## 💡 Key Changes

### Weather Feature Engineering
**Weather features are now computed in preprocessing** (`src/preprocessing.py`):
- `heating_degree`, `cooling_degree` - Degree days (base 65°F)
- `heat_flag`, `freeze_flag` - Temperature extremes
- `rain_3d`, `rain_7d` - Rolling precipitation

This happens automatically when you use `preprocess_and_merge_external_data()`.

The feature engineering code (`src/features.py`) provides backward compatibility but prioritizes pre-computed features.

## 🔗 Related Files

- **Main README:** `../README.md` - Project overview
- **Preprocessing:** `../src/preprocessing.py` - Data cleaning & weather features
- **Feature code:** `../src/features.py` - Feature builders
- **Training code:** `../models/*.py` - Model training
- **Pipelines:** `../pipelines/*.py` - CLI scripts

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🎓 Learning Path

**Beginner → Advanced:**

1. **Start here:** [QUICK_START.md](QUICK_START.md) (5 min)
2. **Understand features:** [FEATURES.md](FEATURES.md) (20 min)
3. **Train your first model:** [TRAINING.md](TRAINING.md) (20 min)
4. **Deep dive:** [API_REFERENCE.md](API_REFERENCE.md) (30 min)

---

**Version:** 2.0 (Consolidated)  
**Last Updated:** October 4, 2025  
**Status:** ✅ Production Ready

