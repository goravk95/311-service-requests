# NYC 311 Models Package

Production-ready training and prediction for NYC 311 service requests.

## 🚀 Quick Start

```python
# Forecast
from models.forecast import train_all_families, predict_forecast
bundles = train_all_families(panel, families=['Health', 'Noise'])
predictions = predict_forecast(bundles, last_rows, horizon=7)

# Triage
from models.triage import train_triage, predict_triage
bundle = train_triage(X, y, families)
probabilities = predict_triage(bundle, X_new, families_new)

# Duration
from models.duration import train_duration_aft, predict_duration_quantiles
bundle = train_duration_aft(X, t, e, model_type='weibull')
quantiles = predict_duration_quantiles(bundle, X_new, ps=(0.5, 0.9))
```

## 📦 Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **forecast.py** | Time-series forecasting | `train_forecast_per_family()`, `predict_forecast()` |
| **triage.py** | Ticket prioritization | `train_triage()`, `predict_triage()` |
| **duration.py** | Survival analysis | `train_duration_aft()`, `predict_duration_quantiles()` |
| **tune.py** | Hyperparameter tuning | `tune_forecast()`, `tune_triage()`, `tune_duration_aft()` |
| **eval.py** | Evaluation & plots | `eval_forecast()`, `eval_triage()`, `eval_duration()` |

## 💻 CLI Usage

```bash
# Training
python pipelines/train.py --task {forecast,triage,duration} --input_parquet DATA --output_dir OUTPUT

# Prediction
python pipelines/predict.py --task {forecast,triage,duration} --model_dir MODELS --scoring_parquet DATA --output_csv OUTPUT
```

## 📚 Documentation

- **[TRAINING_README.md](../TRAINING_README.md)** - Complete reference (714 lines)
- **[TRAINING_SUMMARY.md](../TRAINING_SUMMARY.md)** - Implementation summary
- **Module docstrings** - Inline API documentation

## 🎯 Model Details

### Forecast
- **Algorithm:** LightGBM Poisson regression
- **Granularity:** (hex, complaint_family, day)
- **Horizons:** 1-7 days
- **Output:** p10, p50, p90 predictions

### Triage
- **Algorithm:** LightGBM binary classifier + isotonic calibration
- **Features:** Creation-time only (zero leakage)
- **Output:** Calibrated probabilities (0-1)

### Duration
- **Algorithm:** AFT survival (Weibull or LogNormal)
- **Censoring:** Admin auto-close, long stale, open tickets
- **Output:** Q50, Q90 duration quantiles

## 📊 Metrics

- **Forecast:** RMSE, MAE, MAPE, Poisson deviance
- **Triage:** AUC-ROC, AUC-PR, calibration
- **Duration:** C-index, Q50 MAE, Q90 coverage

## 🔧 Dependencies

```
pandas, numpy, scikit-learn
lightgbm, lifelines, optuna, joblib
matplotlib, seaborn, scipy, h3
```

## ✅ Status

**Version:** 1.0  
**Status:** ✅ Production Ready  
**Last Updated:** October 4, 2025

