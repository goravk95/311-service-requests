"""
Forecast Model: Multi-horizon time-series prediction using LightGBM with Poisson objective.
Trains separate models per complaint_family and per horizon (1-4 weeks).
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
# import optuna
# from optuna.samplers import TPESampler
from typing import Dict, Optional
import warnings
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


def shift_by_date(group, target_col, time_delta):
    """
    Shift y backward by horizon weeks (look forward in time)
    
    Args:
        group: DataFrame with columns [hex, complaint_family, week, y]
        target_col: Target column name
        time_delta: Time delta (in weeks)
    
    Returns:
        DataFrame with added target column
    """
    group = group.set_index('week').sort_index()
    shifted = group['y'].shift(freq=-time_delta)
    group[f'{target_col}'] = shifted
    return group.reset_index()
    

def create_horizon_targets(panel: pd.DataFrame, horizons: List[int] = list(range(1, 5))) -> pd.DataFrame:
    """
    Create per-horizon target variables y_h1, y_h2, y_h3, y_h4.
    Uses sparse construction (only for weeks that exist in panel).
    
    Args:
        panel: DataFrame with columns [hex, complaint_family, week, y]
        horizons: List of forecast horizons in weeks
    
    Returns:
        DataFrame with added target columns y_h1, y_h2, ..., y_hN
    """
    result = panel.copy()
    result = result.sort_values(['hex8', 'complaint_family', 'week'])
    
    for horizon in horizons:
        target_col = f'y_h{horizon}'
        time_delta = pd.Timedelta(weeks=horizon)
        result = (
            result.groupby(['hex8', 'complaint_family'], group_keys=False)
            .apply(lambda group: shift_by_date(group, target_col, time_delta))
        )
        result[target_col] = result[target_col].fillna(0)

    return result


def train_forecast_per_family(
    panel: pd.DataFrame,
    family: str,
    horizons: List[int] = list(range(1, 5)),
    feature_cols: Optional[List[str]] = None,
    val_weeks: int = 8,
    params: Optional[Dict] = None
) -> Dict:
    """
    Train multi-horizon forecast models for a single complaint family.
    
    Args:
        panel: DataFrame from build_forecast_panel with all families
        family: Complaint family to train on
        horizons: List of forecast horizons (weeks)
        feature_cols: Feature columns to use (auto-detected if None)
        val_weeks: Number of weeks to use for validation
        params: LightGBM parameters (uses defaults if None)
    
    Returns:
        Bundle dict with 'family', 'feature_cols', 'cat_cols', 'models', 'metrics'
    """
    # Filter to family
    df = panel[panel['complaint_family'] == family].copy()
    
    if len(df) == 0:
        raise ValueError(f"No data for family: {family}")
    
    print(f"Training forecast for {family}: {len(df)} rows")
    
    # Create horizon targets
    df = create_horizon_targets(df, horizons)
    
    # Default feature columns
    if feature_cols is None:
        feature_cols = [
            'week_of_year', 'month', 'quarter',
            'lag1', 'lag4', 'roll4', 'roll12',
            'momentum', 'weeks_since_last',
            'tavg', 'prcp', 'heating_degree', 'cooling_degree',
            'rain_3d', 'rain_7d', 'log_pop', 'nbr_roll4', 'nbr_roll12'
        ]
        # Add hex as categorical
        # if 'hex8' in df.columns:
        #     feature_cols = ['hex8'] + feature_cols
        
        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    cat_cols = ['hex8'] if 'hex8' in feature_cols else []
    
    # Time-based split
    cutoff_date = df['week'].max() - pd.Timedelta(weeks=val_weeks)
    train_mask = df['week'] < cutoff_date
    val_mask = df['week'] >= cutoff_date
    
    X_train = df.loc[train_mask, feature_cols]
    X_val = df.loc[val_mask, feature_cols]
    
    # Default LightGBM params
    if params is None:
        params = {
            'objective': 'poisson',
            'n_estimators': 800,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'random_state': 42,
            'verbose': -1
        }
    
    # Train one model per horizon
    models = {}
    metrics = {}
    
    for horizon in horizons:
        target_col = f'y_h{horizon}'
        
        # Get targets (drop NaN from shifting)
        y_train = df.loc[train_mask, target_col].dropna()
        y_val = df.loc[val_mask, target_col].dropna()
        
        # Align features
        X_train_h = X_train.loc[y_train.index]
        X_val_h = X_val.loc[y_val.index]
        
        if len(y_train) < 50:
            print(f"  Skipping h={horizon}: insufficient training data ({len(y_train)} rows)")
            continue
        
        # Train model
        model = lgb.LGBMRegressor(**params)
        
        # Fit with categorical features
        if cat_cols:
            model.fit(
                X_train_h, y_train,
                eval_set=[(X_val_h, y_val)] if len(y_val) > 0 else None,
                categorical_feature=cat_cols,
                eval_metric='poisson'
            )
        else:
            model.fit(
                X_train_h, y_train,
                eval_set=[(X_val_h, y_val)] if len(y_val) > 0 else None,
                eval_metric='poisson'
            )
        
        models[horizon] = model
        
        # Compute validation metrics
        if len(y_val) > 0:
            y_pred = model.predict(X_val_h)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            mae = np.mean(np.abs(y_val - y_pred))
            
            # Poisson deviance
            epsilon = 1e-10
            poisson_dev = 2 * np.mean(
                y_val * np.log((y_val + epsilon) / (y_pred + epsilon)) - (y_val - y_pred)
            )
            
            metrics[horizon] = {
                'rmse': float(rmse),
                'mae': float(mae),
                'poisson_deviance': float(poisson_dev),
                'n_train': len(y_train),
                'n_val': len(y_val)
            }
            
            print(f"  h={horizon}: RMSE={rmse:.3f}, MAE={mae:.3f}, Poisson Dev={poisson_dev:.3f}")
    
    bundle = {
        'family': family,
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'models': models,
        'metrics': metrics,
        'horizons': list(models.keys())
    }
    
    return bundle


def train_all_families(
    panel: pd.DataFrame,
    families: Optional[List[str]] = None,
    horizons: List[int] = list(range(1, 5)),
    val_weeks: int = 8,
    params: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    Train forecast models for all (or specified) complaint families.
    
    Args:
        panel: Full forecast panel
        families: List of families to train (all if None)
        horizons: Forecast horizons (weeks)
        val_weeks: Validation weeks
        params: LightGBM parameters
    
    Returns:
        Dict mapping family -> bundle
    """
    if families is None:
        families = panel['complaint_family'].unique()
    
    bundles = {}
    
    for family in families:
        bundle = train_forecast_per_family(
            panel, family, horizons, val_weeks=val_weeks, params=params
        )
        bundles[family] = bundle
    
    print(f"\n✓ Trained {len(bundles)} family models")
    return bundles


def predict_forecast(
    bundles: Dict[str, Dict],
    last_rows: pd.DataFrame,
    horizon: int = 4
) -> pd.DataFrame:
    """
    Generate forecasts from the latest state.
    
    Args:
        bundles: Dict of trained model bundles per family
        last_rows: Latest row per [hex, complaint_family] with features
        horizon: Forecast horizon (1-4 weeks)
    
    Returns:
        DataFrame with [hex, complaint_family, week, p50, p10, p90]
    """
    predictions = []
    
    for family, bundle in bundles.items():
        if horizon not in bundle['models']:
            continue
        
        # Filter to this family
        df_fam = last_rows[last_rows['complaint_family'] == family].copy()
        
        if len(df_fam) == 0:
            continue
        
        model = bundle['models'][horizon]
        feature_cols = bundle['feature_cols']
        
        # Check features exist
        missing = [c for c in feature_cols if c not in df_fam.columns]
        if missing:
            print(f"Warning: {family} missing features: {missing}")
            continue
        
        X = df_fam[feature_cols]
        
        # Predict
        y_pred = model.predict(X)
        
        # For Poisson, use variance = mean for uncertainty
        # p50 = mean, p10/p90 via Poisson quantiles
        from scipy.stats import poisson
        
        p50 = y_pred
        p10 = np.array([poisson.ppf(0.1, lam) if lam > 0 else 0 for lam in y_pred])
        p90 = np.array([poisson.ppf(0.9, lam) if lam > 0 else 0 for lam in y_pred])
        
        # Forecast week
        forecast_week = df_fam['week'] + pd.Timedelta(weeks=horizon)
        
        pred_df = pd.DataFrame({
            'hex8': df_fam['hex8'].values,
            'complaint_family': family,
            'week': forecast_week,
            'p50': p50,
            'p10': p10,
            'p90': p90
        })
        
        predictions.append(pred_df)
    
    if not predictions:
        return pd.DataFrame(columns=['hex8', 'complaint_family', 'week', 'p50', 'p10', 'p90'])
    
    result = pd.concat(predictions, ignore_index=True)
    return result


def save_bundles(bundles: Dict[str, Dict], output_dir: Path) -> None:
    """
    Save forecast bundles to disk.
    
    Args:
        bundles: Dict of model bundles
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for family, bundle in bundles.items():
        # Sanitize filename
        safe_name = family.replace('/', '_').replace(' ', '_')
        path = output_dir / f"forecast_{safe_name}.joblib"
        joblib.dump(bundle, path)
        print(f"Saved {family} -> {path}")


def load_bundles(input_dir: Path) -> Dict[str, Dict]:
    """
    Load forecast bundles from disk.
    
    Args:
        input_dir: Directory with .joblib files
    
    Returns:
        Dict of model bundles
    """
    input_dir = Path(input_dir)
    bundles = {}
    
    for path in input_dir.glob("forecast_*.joblib"):
        bundle = joblib.load(path)
        family = bundle['family']
        bundles[family] = bundle
        print(f"Loaded {family} <- {path}")
    
    return bundles





# def tune(
#     X_train: pd.DataFrame,
#     y_train: pd.Series,
#     X_val: pd.DataFrame,
#     y_val: pd.Series,
#     cat_cols: Optional[list] = None,
#     n_trials: int = 30,
#     random_state: int = 42
# ) -> Dict:
#     """
#     Tune LightGBM forecast model using Optuna.
#     Optimizes Poisson deviance on validation set.
    
#     Args:
#         X_train: Training features
#         y_train: Training targets
#         X_val: Validation features
#         y_val: Validation targets
#         cat_cols: Categorical column names
#         n_trials: Number of Optuna trials
#         random_state: Random seed
    
#     Returns:
#         Dict with best_params and best_score
#     """
#     print(f"Tuning forecast model with {n_trials} trials...")
    
#     def objective(trial):
#         params = {
#             'objective': 'poisson',
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#             'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#             'max_depth': trial.suggest_int('max_depth', 3, 10),
#             'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'n_estimators': 500,
#             'random_state': random_state,
#             'verbose': -1
#         }
        
#         model = lgb.LGBMRegressor(**params)
        
#         if cat_cols:
#             model.fit(X_train, y_train, categorical_feature=cat_cols)
#         else:
#             model.fit(X_train, y_train)
        
#         y_pred = model.predict(X_val)
        
#         # Poisson deviance (lower is better)
#         epsilon = 1e-10
#         poisson_dev = 2 * np.mean(
#             y_val * np.log((y_val + epsilon) / (y_pred + epsilon)) - (y_val - y_pred)
#         )
        
#         return poisson_dev
    
#     study = optuna.create_study(
#         direction='minimize',
#         sampler=TPESampler(seed=random_state)
#     )
    
#     study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
#     print(f"✓ Best Poisson deviance: {study.best_value:.4f}")
#     print(f"✓ Best params: {study.best_params}")
    
#     return {
#         'best_params': study.best_params,
#         'best_score': study.best_value,
#         # 'study': study
#     }


def eval_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    family: str = "Unknown",
    horizon: int = 1
) -> Dict:
    """
    Evaluate forecast predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        family: Complaint family name
        horizon: Forecast horizon
    
    Returns:
        Dict with RMSE, MAE, Poisson deviance, MAPE
    """
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Poisson deviance
    epsilon = 1e-10
    poisson_dev = 2 * np.mean(
        y_true * np.log((y_true + epsilon) / (y_pred + epsilon)) - (y_true - y_pred)
    )
    
    metrics = {
        'family': family,
        'horizon': horizon,
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'poisson_deviance': float(poisson_dev),
        'n_samples': len(y_true)
    }
    
    return metrics


def plot_forecast_calibration(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Forecast Calibration",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot predicted vs actual for forecast calibration.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        output_path: Path to save figure (None = display only)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    axes[0].scatter(y_pred, y_true, alpha=0.3, s=10)
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', label='Perfect calibration')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'{title} - Scatter')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Binned calibration
    n_bins = 10
    bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y_pred, bins[:-1]) - 1
    
    bin_means_pred = []
    bin_means_true = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means_pred.append(y_pred[mask].mean())
            bin_means_true.append(y_true[mask].mean())
    
    axes[1].plot(bin_means_pred, bin_means_true, 'o-', label='Binned mean')
    axes[1].plot([0, max(bin_means_pred)], [0, max(bin_means_pred)], 'r--', label='Perfect')
    axes[1].set_xlabel('Mean Predicted (binned)')
    axes[1].set_ylabel('Mean Actual')
    axes[1].set_title(f'{title} - Calibration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration plot -> {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_metrics(metrics: Dict, output_path: Path) -> None:
    """
    Save metrics dict to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Output JSON path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved metrics -> {output_path}")


def print_metrics_summary(metrics: Dict, model_type: str) -> None:
    """
    Print formatted summary of metrics.
    
    Args:
        metrics: Metrics dictionary
        model_type: 'forecast', 'triage', or 'duration'
    """
    print(f"\n{'='*60}")
    print(f"{model_type.upper()} MODEL EVALUATION")
    print(f"{'='*60}")
    
    if model_type == 'forecast':
        print(f"RMSE: {metrics.get('rmse', 'N/A'):.3f}")
        print(f"MAE: {metrics.get('mae', 'N/A'):.3f}")
        print(f"MAPE: {metrics.get('mape', 'N/A'):.1f}%")
        print(f"Poisson Deviance: {metrics.get('poisson_deviance', 'N/A'):.3f}")
    
    elif model_type == 'triage':
        overall = metrics.get('overall', {})
        print(f"AUC-ROC: {overall.get('auc_roc', 'N/A'):.3f}")
        print(f"AUC-PR: {overall.get('auc_pr', 'N/A'):.3f}")
        print(f"Positive Rate: {overall.get('positive_rate', 'N/A'):.1%}")
        
        if 'per_family' in metrics:
            print(f"\nPer-family metrics available for {len(metrics['per_family'])} families")
    
    elif model_type == 'duration':
        print(f"C-Index: {metrics.get('c_index', 'N/A'):.3f}")
        print(f"MAE (Q50): {metrics.get('mae', 'N/A'):.1f} days")
        if metrics.get('coverage') is not None:
            print(f"Q90 Coverage: {metrics['coverage']:.1%}")
        print(f"Event Rate: {metrics.get('event_rate', 'N/A'):.1%}")
    
    print(f"{'='*60}\n")


