"""
Hyperparameter tuning using Optuna for Forecast, Triage, and Duration models.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from lifelines import WeibullAFTFitter, LogNormalAFTFitter
from lifelines.utils import concordance_index
from typing import Dict, Callable, Optional
import warnings

warnings.filterwarnings('ignore')


def tune_forecast(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: Optional[list] = None,
    n_trials: int = 30,
    random_state: int = 42
) -> Dict:
    """
    Tune LightGBM forecast model using Optuna.
    Optimizes Poisson deviance on validation set.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        cat_cols: Categorical column names
        n_trials: Number of Optuna trials
        random_state: Random seed
    
    Returns:
        Dict with best_params and best_score
    """
    print(f"Tuning forecast model with {n_trials} trials...")
    
    def objective(trial):
        params = {
            'objective': 'poisson',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': 500,
            'random_state': random_state,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        
        if cat_cols:
            model.fit(X_train, y_train, categorical_feature=cat_cols)
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        # Poisson deviance (lower is better)
        epsilon = 1e-10
        poisson_dev = 2 * np.mean(
            y_val * np.log((y_val + epsilon) / (y_pred + epsilon)) - (y_val - y_pred)
        )
        
        return poisson_dev
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=random_state)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"✓ Best Poisson deviance: {study.best_value:.4f}")
    print(f"✓ Best params: {study.best_params}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }


def tune_triage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
    random_state: int = 42
) -> Dict:
    """
    Tune LightGBM triage classifier using Optuna.
    Optimizes AUC-PR on validation set.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of Optuna trials
        random_state: Random seed
    
    Returns:
        Dict with best_params and best_score
    """
    print(f"Tuning triage model with {n_trials} trials...")
    
    from sklearn.metrics import average_precision_score
    
    # Compute class weight
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate < 0.5 else 1.0
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': scale_pos_weight,
            'n_estimators': 500,
            'random_state': random_state,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_pr = average_precision_score(y_val, y_pred_proba)
        
        return auc_pr
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"✓ Best AUC-PR: {study.best_value:.4f}")
    print(f"✓ Best params: {study.best_params}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }


def tune_duration_aft(
    X: pd.DataFrame,
    t: pd.Series,
    e: pd.Series,
    test_size: float = 0.2,
    n_trials: int = 30,
    random_state: int = 42
) -> Dict:
    """
    Tune AFT duration model using Optuna.
    Optimizes concordance index + q90 coverage.
    
    Args:
        X: Features
        t: Time-to-event
        e: Event indicator
        test_size: Validation split
        n_trials: Number of Optuna trials
        random_state: Random seed
    
    Returns:
        Dict with best_params, best_score, best_model_type
    """
    print(f"Tuning AFT duration model with {n_trials} trials...")
    
    from models.duration import prepare_features
    
    # Split data
    X_train, X_val, t_train, t_val, e_train, e_val = train_test_split(
        X, t, e, test_size=test_size, random_state=random_state
    )
    
    # Prepare features once
    X_train_prep, transformer = prepare_features(X_train)
    X_val_prep, _ = prepare_features(X_val, transformer=transformer)
    
    def objective(trial):
        # Choose model type
        model_type = trial.suggest_categorical('model_type', ['weibull', 'lognormal'])
        penalizer = trial.suggest_float('penalizer', 0.001, 0.1, log=True)
        
        # Create training dataframe
        df_train = X_train_prep.copy()
        df_train['duration'] = t_train.values
        df_train['event'] = e_train.values
        
        df_val = X_val_prep.copy()
        df_val['duration'] = t_val.values
        df_val['event'] = e_val.values
        
        try:
            # Train model
            if model_type == 'weibull':
                model = WeibullAFTFitter(penalizer=penalizer)
            else:
                model = LogNormalAFTFitter(penalizer=penalizer)
            
            model.fit(df_train, duration_col='duration', event_col='event')
            
            # Evaluate
            c_index = concordance_index(t_val, -model.predict_median(df_val), e_val)
            
            # Q90 coverage (on observed events only)
            observed_mask = e_val == 1
            if observed_mask.sum() > 10:
                t_obs = t_val[observed_mask]
                df_obs = df_val[observed_mask]
                q90_pred = model.predict_percentile(df_obs, p=0.9)
                q90_coverage = (t_obs.values <= q90_pred.values).mean()
            else:
                q90_coverage = 0.9  # Default
            
            # Combined score (c-index primary, q90 coverage secondary)
            score = c_index + 0.1 * q90_coverage
            
            return score
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0  # Return poor score for failed trials
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"✓ Best score: {study.best_value:.4f}")
    print(f"✓ Best params: {study.best_params}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'best_model_type': study.best_params['model_type'],
        'study': study
    }


def run_tuning_suite(
    forecast_data: Optional[Dict] = None,
    triage_data: Optional[Dict] = None,
    duration_data: Optional[Dict] = None,
    n_trials: int = 30,
    random_state: int = 42
) -> Dict:
    """
    Run full tuning suite for all models.
    
    Args:
        forecast_data: Dict with X_train, y_train, X_val, y_val, cat_cols
        triage_data: Dict with X_train, y_train, X_val, y_val
        duration_data: Dict with X, t, e
        n_trials: Number of trials per model
        random_state: Random seed
    
    Returns:
        Dict with tuning results for each model
    """
    results = {}
    
    if forecast_data:
        print("\n" + "="*60)
        print("TUNING FORECAST MODEL")
        print("="*60)
        results['forecast'] = tune_forecast(
            forecast_data['X_train'],
            forecast_data['y_train'],
            forecast_data['X_val'],
            forecast_data['y_val'],
            cat_cols=forecast_data.get('cat_cols'),
            n_trials=n_trials,
            random_state=random_state
        )
    
    if triage_data:
        print("\n" + "="*60)
        print("TUNING TRIAGE MODEL")
        print("="*60)
        results['triage'] = tune_triage(
            triage_data['X_train'],
            triage_data['y_train'],
            triage_data['X_val'],
            triage_data['y_val'],
            n_trials=n_trials,
            random_state=random_state
        )
    
    if duration_data:
        print("\n" + "="*60)
        print("TUNING DURATION MODEL")
        print("="*60)
        results['duration'] = tune_duration_aft(
            duration_data['X'],
            duration_data['t'],
            duration_data['e'],
            n_trials=n_trials,
            random_state=random_state
        )
    
    print("\n" + "="*60)
    print("✓ TUNING COMPLETE")
    print("="*60)
    
    return results

