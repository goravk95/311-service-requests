"""
Forecast Model: Multi-horizon time-series prediction using LightGBM with Poisson objective.
Trains separate models per complaint_family and per horizon (1-7 days).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path


def create_horizon_targets(panel: pd.DataFrame, horizons: List[int] = list(range(1, 8))) -> pd.DataFrame:
    """
    Create per-horizon target variables y_h1, y_h2, ..., y_h7.
    Uses sparse construction (only for days that exist in panel).
    
    Args:
        panel: DataFrame with columns [hex, complaint_family, day, y]
        horizons: List of forecast horizons in days
    
    Returns:
        DataFrame with added target columns y_h1, y_h2, ..., y_hN
    """
    result = panel.copy()
    result = result.sort_values(['hex', 'complaint_family', 'day'])
    
    for horizon in horizons:
        target_col = f'y_h{horizon}'
        
        # Shift y backward by horizon days (look forward in time)
        result[target_col] = result.groupby(['hex', 'complaint_family'])['y'].shift(-horizon)
    
    return result


def train_forecast_per_family(
    panel: pd.DataFrame,
    family: str,
    horizons: List[int] = list(range(1, 8)),
    feature_cols: Optional[List[str]] = None,
    val_days: int = 30,
    params: Optional[Dict] = None
) -> Dict:
    """
    Train multi-horizon forecast models for a single complaint family.
    
    Args:
        panel: DataFrame from build_forecast_panel with all families
        family: Complaint family to train on
        horizons: List of forecast horizons (days)
        feature_cols: Feature columns to use (auto-detected if None)
        val_days: Number of days to use for validation
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
            'dow', 'month',
            'lag1', 'lag7', 'roll7', 'roll14', 'roll28',
            'momentum', 'days_since_last',
            'tavg', 'prcp', 'heating_degree', 'cooling_degree',
            'rain_3d', 'rain_7d', 'log_pop'
        ]
        # Add hex as categorical
        if 'hex' in df.columns:
            feature_cols = ['hex'] + feature_cols
        
        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    cat_cols = ['hex'] if 'hex' in feature_cols else []
    
    # Time-based split
    cutoff_date = df['day'].max() - pd.Timedelta(days=val_days)
    train_mask = df['day'] < cutoff_date
    val_mask = df['day'] >= cutoff_date
    
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
    horizons: List[int] = list(range(1, 8)),
    val_days: int = 30,
    params: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    Train forecast models for all (or specified) complaint families.
    
    Args:
        panel: Full forecast panel
        families: List of families to train (all if None)
        horizons: Forecast horizons
        val_days: Validation days
        params: LightGBM parameters
    
    Returns:
        Dict mapping family -> bundle
    """
    if families is None:
        families = panel['complaint_family'].unique()
    
    bundles = {}
    
    for family in families:
        try:
            bundle = train_forecast_per_family(
                panel, family, horizons, val_days=val_days, params=params
            )
            bundles[family] = bundle
        except Exception as e:
            print(f"Failed to train {family}: {e}")
    
    print(f"\n✓ Trained {len(bundles)} family models")
    return bundles


def predict_forecast(
    bundles: Dict[str, Dict],
    last_rows: pd.DataFrame,
    horizon: int = 7
) -> pd.DataFrame:
    """
    Generate forecasts from the latest state.
    
    Args:
        bundles: Dict of trained model bundles per family
        last_rows: Latest row per [hex, complaint_family] with features
        horizon: Forecast horizon (1-7)
    
    Returns:
        DataFrame with [hex, complaint_family, day, p50, p10, p90]
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
        
        # Forecast day
        forecast_day = df_fam['day'] + pd.Timedelta(days=horizon)
        
        pred_df = pd.DataFrame({
            'hex': df_fam['hex'].values,
            'complaint_family': family,
            'day': forecast_day,
            'p50': p50,
            'p10': p10,
            'p90': p90
        })
        
        predictions.append(pred_df)
    
    if not predictions:
        return pd.DataFrame(columns=['hex', 'complaint_family', 'day', 'p50', 'p10', 'p90'])
    
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

