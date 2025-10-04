"""
Triage Model: Global classifier with per-family calibration for ticket prioritization.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Tuple
import joblib
from pathlib import Path


def train_triage(
    dfX: pd.DataFrame,
    y: pd.Series,
    families: pd.Series,
    params: Optional[Dict] = None,
    calibration_threshold: int = 500,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Train global triage classifier with per-family calibration.
    
    Args:
        dfX: Feature DataFrame (from build_triage_features)
        y: Target labels (binary: 0/1)
        families: complaint_family for each row
        params: LightGBM parameters (uses defaults if None)
        calibration_threshold: Min samples per family for calibration
        test_size: Validation split fraction
        random_state: Random seed
    
    Returns:
        Bundle dict with 'model', 'feature_cols', 'calibrators', 'metrics'
    """
    print(f"Training triage model: {len(dfX)} samples, {y.mean():.1%} positive rate")
    
    # Ensure alignment
    assert len(dfX) == len(y) == len(families), "Length mismatch"
    
    # Store feature columns
    feature_cols = [c for c in dfX.columns if c != 'unique_key']
    
    # Split data (time-based would be better but using random for simplicity)
    X_train, X_val, y_train, y_val, fam_train, fam_val = train_test_split(
        dfX[feature_cols], y, families,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(y.unique()) > 1 else None
    )
    
    # Handle class imbalance
    pos_rate = y_train.mean()
    use_class_weight = pos_rate < 0.15
    
    # Default LightGBM params
    if params is None:
        params = {
            'objective': 'binary',
            'n_estimators': 700,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'random_state': random_state,
            'verbose': -1
        }
        
        if use_class_weight:
            params['scale_pos_weight'] = (1 - pos_rate) / pos_rate
            print(f"Using scale_pos_weight={params['scale_pos_weight']:.2f}")
    
    # Train model
    model = lgb.LGBMClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc'
    )
    
    # Get validation predictions
    y_pred_proba_val = model.predict_proba(X_val)[:, 1]
    
    # Train per-family calibrators
    calibrators = {}
    val_df = pd.DataFrame({
        'family': fam_val.values,
        'y_true': y_val.values,
        'y_pred': y_pred_proba_val
    })
    
    for family in val_df['family'].unique():
        fam_mask = val_df['family'] == family
        fam_data = val_df[fam_mask]
        
        if len(fam_data) >= calibration_threshold:
            try:
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(fam_data['y_pred'], fam_data['y_true'])
                calibrators[family] = calibrator
                print(f"  Calibrated {family}: {len(fam_data)} samples")
            except Exception as e:
                print(f"  Failed to calibrate {family}: {e}")
    
    print(f"✓ Trained global model + {len(calibrators)} family calibrators")
    
    # Compute metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
    
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    
    metrics = {
        'train_auc_roc': float(roc_auc_score(y_train, y_pred_proba_train)),
        'train_auc_pr': float(average_precision_score(y_train, y_pred_proba_train)),
        'val_auc_roc': float(roc_auc_score(y_val, y_pred_proba_val)),
        'val_auc_pr': float(average_precision_score(y_val, y_pred_proba_val)),
        'positive_rate': float(pos_rate),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_calibrators': len(calibrators)
    }
    
    print(f"  Train AUC-ROC: {metrics['train_auc_roc']:.3f}, AUC-PR: {metrics['train_auc_pr']:.3f}")
    print(f"  Val AUC-ROC: {metrics['val_auc_roc']:.3f}, AUC-PR: {metrics['val_auc_pr']:.3f}")
    
    bundle = {
        'model': model,
        'feature_cols': feature_cols,
        'calibrators': calibrators,
        'metrics': metrics
    }
    
    return bundle


def predict_triage(
    bundle: Dict,
    dfX_new: pd.DataFrame,
    families_new: pd.Series
) -> np.ndarray:
    """
    Predict triage probabilities with per-family calibration.
    
    Args:
        bundle: Trained model bundle
        dfX_new: Feature DataFrame for prediction
        families_new: complaint_family for each row
    
    Returns:
        Array of calibrated probabilities
    """
    model = bundle['model']
    feature_cols = bundle['feature_cols']
    calibrators = bundle['calibrators']
    
    # Check features
    missing = [c for c in feature_cols if c not in dfX_new.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    X = dfX_new[feature_cols]
    
    # Get base predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Apply per-family calibration
    calibrated = np.zeros_like(y_pred_proba)
    
    for i, (prob, family) in enumerate(zip(y_pred_proba, families_new)):
        if family in calibrators:
            calibrated[i] = calibrators[family].predict([prob])[0]
        else:
            calibrated[i] = prob  # No calibration available
    
    return calibrated


def save_bundle(bundle: Dict, output_path: Path) -> None:
    """Save triage bundle to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"Saved triage model -> {output_path}")


def load_bundle(input_path: Path) -> Dict:
    """Load triage bundle from disk."""
    bundle = joblib.load(input_path)
    print(f"Loaded triage model <- {input_path}")
    return bundle

