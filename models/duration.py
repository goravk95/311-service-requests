"""
Duration Model: AFT survival models with right-censoring for time-to-close prediction.
"""

import numpy as np
import pandas as pd
from lifelines import LogNormalAFTFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path


def prepare_features(
    X: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    transformer: Optional[ColumnTransformer] = None
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Prepare features: standardize numerics, one-hot encode categoricals.
    
    Args:
        X: Feature DataFrame
        numeric_cols: Numeric column names (auto-detect if None)
        categorical_cols: Categorical column names (auto-detect if None)
        transformer: Pre-fitted transformer (fit new one if None)
    
    Returns:
        Tuple of (transformed DataFrame, transformer)
    """
    if numeric_cols is None:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove any non-existent columns
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    
    if transformer is None:
        # Create transformer
        transformers = []
        
        if numeric_cols:
            transformers.append(('num', StandardScaler(), numeric_cols))
        
        if categorical_cols:
            transformers.append((
                'cat',
                OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                categorical_cols
            ))
        
        transformer = ColumnTransformer(transformers, remainder='drop')
        X_transformed = transformer.fit_transform(X)
    else:
        X_transformed = transformer.transform(X)
    
    # Get feature names
    feature_names = []
    if numeric_cols:
        feature_names.extend(numeric_cols)
    if categorical_cols and hasattr(transformer.named_transformers_['cat'], 'get_feature_names_out'):
        cat_names = transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        feature_names.extend(cat_names)
    
    X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    
    return X_df, transformer


def train_duration_aft(
    X: pd.DataFrame,
    t: pd.Series,
    e: pd.Series,
    model_type: str = 'weibull',
    test_size: float = 0.2,
    random_state: int = 42,
    penalizer: float = 0.01
) -> Dict:
    """
    Train AFT survival model for duration prediction.
    
    Args:
        X: Feature DataFrame (from build_duration_features)
        t: Time-to-event (ttc_days_cens)
        e: Event indicator (1=observed, 0=censored)
        model_type: 'weibull' or 'lognormal'
        test_size: Validation split fraction
        random_state: Random seed
        penalizer: L2 regularization strength
    
    Returns:
        Bundle dict with 'model', 'transformer', 'metrics', 'model_type'
    """
    print(f"Training AFT duration model ({model_type}): {len(X)} samples, {e.mean():.1%} observed")
    
    # Ensure alignment
    assert len(X) == len(t) == len(e), "Length mismatch"
    
    # Remove any NaN in target
    valid_mask = t.notna() & e.notna()
    X = X[valid_mask]
    t = t[valid_mask]
    e = e[valid_mask]
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, t_train, t_val, e_train, e_val = train_test_split(
        X, t, e,
        test_size=test_size,
        random_state=random_state
    )
    
    # Prepare features (standardize + one-hot)
    X_train_prep, transformer = prepare_features(X_train)
    X_val_prep, _ = prepare_features(X_val, transformer=transformer)
    
    # Create training dataframe for lifelines
    df_train = X_train_prep.copy()
    df_train['duration'] = t_train.values
    df_train['event'] = e_train.values
    
    # Train model
    if model_type == 'weibull':
        model = WeibullAFTFitter(penalizer=penalizer)
    elif model_type == 'lognormal':
        model = LogNormalAFTFitter(penalizer=penalizer)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.fit(df_train, duration_col='duration', event_col='event')
    
    print(f"✓ Trained {model_type} AFT model")
    
    # Evaluate on validation
    df_val = X_val_prep.copy()
    df_val['duration'] = t_val.values
    df_val['event'] = e_val.values
    
    # Concordance index
    c_index_train = model.concordance_index_
    c_index_val = concordance_index(t_val, -model.predict_median(df_val), e_val)
    
    # Quantile predictions for observed events
    observed_mask = e_val == 1
    if observed_mask.sum() > 0:
        t_obs = t_val[observed_mask]
        df_obs = df_val[observed_mask]
        
        q50_pred = model.predict_median(df_obs)
        q90_pred = model.predict_percentile(df_obs, p=0.9)
        
        # Coverage: what % of observed events fall below q90?
        q90_coverage = (t_obs.values <= q90_pred.values).mean()
        
        # MAE at median
        mae_q50 = np.mean(np.abs(t_obs.values - q50_pred.values))
    else:
        q90_coverage = np.nan
        mae_q50 = np.nan
    
    metrics = {
        'c_index_train': float(c_index_train),
        'c_index_val': float(c_index_val),
        'q90_coverage': float(q90_coverage) if not np.isnan(q90_coverage) else None,
        'mae_q50': float(mae_q50) if not np.isnan(mae_q50) else None,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'event_rate_train': float(e_train.mean()),
        'event_rate_val': float(e_val.mean())
    }
    
    print(f"  C-index train: {c_index_train:.3f}, val: {c_index_val:.3f}")
    if not np.isnan(q90_coverage):
        print(f"  Q90 coverage: {q90_coverage:.3f}, MAE Q50: {mae_q50:.1f} days")
    
    bundle = {
        'model': model,
        'transformer': transformer,
        'metrics': metrics,
        'model_type': model_type,
        'feature_cols': list(X.columns)
    }
    
    return bundle


def predict_duration_quantiles(
    bundle: Dict,
    X_new: pd.DataFrame,
    ps: Tuple[float, ...] = (0.5, 0.9)
) -> pd.DataFrame:
    """
    Predict duration quantiles (e.g., median, 90th percentile).
    
    Args:
        bundle: Trained AFT model bundle
        X_new: Feature DataFrame for prediction
        ps: Quantiles to predict (e.g., 0.5=median, 0.9=90th percentile)
    
    Returns:
        DataFrame with columns q50, q90, etc.
    """
    model = bundle['model']
    transformer = bundle['transformer']
    feature_cols = bundle['feature_cols']
    
    # Check features
    missing = [c for c in feature_cols if c not in X_new.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    # Prepare features
    X_prep, _ = prepare_features(X_new[feature_cols], transformer=transformer)
    
    # Predict quantiles
    result = pd.DataFrame(index=X_new.index)
    
    for p in ps:
        col_name = f'q{int(p*100)}'
        
        if p == 0.5:
            result[col_name] = model.predict_median(X_prep).values
        else:
            result[col_name] = model.predict_percentile(X_prep, p=p).values
    
    return result


def compare_models(
    X: pd.DataFrame,
    t: pd.Series,
    e: pd.Series,
    models: List[str] = ['weibull', 'lognormal'],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict, str]:
    """
    Compare multiple AFT models and return the best one.
    
    Args:
        X: Features
        t: Time-to-event
        e: Event indicator
        models: List of model types to try
        test_size: Validation split
        random_state: Random seed
    
    Returns:
        Tuple of (best_bundle, best_model_type)
    """
    print("Comparing AFT models...")
    
    bundles = {}
    scores = {}
    
    for model_type in models:
        try:
            bundle = train_duration_aft(
                X, t, e,
                model_type=model_type,
                test_size=test_size,
                random_state=random_state
            )
            bundles[model_type] = bundle
            
            # Score = c-index + (q90_coverage if available)
            c_val = bundle['metrics']['c_index_val']
            q90 = bundle['metrics'].get('q90_coverage', 0.9)
            score = c_val + (0.1 * q90 if q90 is not None else 0)
            scores[model_type] = score
            
        except Exception as e:
            print(f"Failed to train {model_type}: {e}")
    
    if not bundles:
        raise ValueError("No models successfully trained")
    
    # Select best
    best_model = max(scores, key=scores.get)
    best_bundle = bundles[best_model]
    
    print(f"\n✓ Best model: {best_model} (score={scores[best_model]:.3f})")
    
    return best_bundle, best_model


def save_bundle(bundle: Dict, output_path: Path) -> None:
    """Save duration bundle to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"Saved duration model -> {output_path}")


def load_bundle(input_path: Path) -> Dict:
    """Load duration bundle from disk."""
    bundle = joblib.load(input_path)
    print(f"Loaded duration model <- {input_path}")
    return bundle

