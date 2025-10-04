"""
Training Pipeline CLI for Forecast, Triage, and Duration models.
"""

import sys
sys.path.append('.')

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.features import (
    add_h3_keys,
    build_forecast_panel,
    build_triage_features,
    build_duration_survival_labels,
    build_duration_features
)

from models import forecast, triage, duration, eval as model_eval, tune


def train_forecast_task(
    df: pd.DataFrame,
    output_dir: Path,
    families: list = None,
    do_tune: bool = False,
    weather_df: pd.DataFrame = None
):
    """
    Train forecast models.
    
    Args:
        df: Input DataFrame
        output_dir: Output directory for models
        families: List of families to train (all if None)
        do_tune: Whether to run hyperparameter tuning
        weather_df: Weather DataFrame
    """
    print("\n" + "="*60)
    print("TRAINING FORECAST MODELS")
    print("="*60)
    
    # Build forecast panel
    print("\nBuilding forecast panel...")
    panel = build_forecast_panel(df, weather_df=weather_df, use_population_offset=True)
    
    if len(panel) == 0:
        print("Error: Empty forecast panel")
        return
    
    print(f"✓ Panel shape: {panel.shape}")
    
    # Filter families if specified
    if families:
        panel = panel[panel['complaint_family'].isin(families)]
        families_to_train = families
    else:
        families_to_train = panel['complaint_family'].unique()
    
    print(f"Training {len(families_to_train)} families...")
    
    # Hyperparameter tuning (optional)
    params = None
    if do_tune:
        print("\nRunning hyperparameter tuning...")
        
        # Take first family for tuning
        sample_family = families_to_train[0]
        df_sample = panel[panel['complaint_family'] == sample_family].copy()
        
        # Create targets
        df_sample = forecast.create_horizon_targets(df_sample, horizons=[7])
        
        # Features
        feature_cols = [
            'dow', 'month', 'lag1', 'lag7', 'roll7', 'roll14', 'roll28',
            'momentum', 'days_since_last', 'tavg', 'prcp',
            'heating_degree', 'cooling_degree', 'rain_3d', 'rain_7d', 'log_pop'
        ]
        if 'hex' in df_sample.columns:
            feature_cols = ['hex'] + feature_cols
        feature_cols = [c for c in feature_cols if c in df_sample.columns]
        
        cat_cols = ['hex'] if 'hex' in feature_cols else []
        
        # Time split
        cutoff_date = df_sample['day'].max() - pd.Timedelta(days=30)
        train_mask = df_sample['day'] < cutoff_date
        val_mask = df_sample['day'] >= cutoff_date
        
        X_train = df_sample.loc[train_mask, feature_cols].dropna()
        y_train = df_sample.loc[train_mask, 'y_h7'].dropna()
        X_val = df_sample.loc[val_mask, feature_cols].dropna()
        y_val = df_sample.loc[val_mask, 'y_h7'].dropna()
        
        # Align
        X_train = X_train.loc[y_train.index]
        X_val = X_val.loc[y_val.index]
        
        tune_result = tune.tune_forecast(
            X_train, y_train, X_val, y_val,
            cat_cols=cat_cols, n_trials=20
        )
        
        # Apply best params
        params = tune_result['best_params']
        params['objective'] = 'poisson'
        params['n_estimators'] = 800
        params['random_state'] = 42
        params['verbose'] = -1
        
        print(f"✓ Tuning complete, using params: {params}")
    
    # Train models
    bundles = forecast.train_all_families(
        panel, families=families_to_train, params=params
    )
    
    # Save models
    models_dir = output_dir / "models"
    forecast.save_bundles(bundles, models_dir)
    
    # Evaluate and save metrics
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {}
    for family, bundle in bundles.items():
        all_metrics[family] = bundle['metrics']
    
    model_eval.save_metrics(all_metrics, metrics_dir / "forecast_metrics.json")
    
    print(f"\n✓ Forecast training complete: {len(bundles)} families trained")


def train_triage_task(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str = 'potential_inspection_trigger',
    do_tune: bool = False
):
    """
    Train triage model.
    
    Args:
        df: Input DataFrame
        output_dir: Output directory
        target_col: Target column name
        do_tune: Whether to run hyperparameter tuning
    """
    print("\n" + "="*60)
    print("TRAINING TRIAGE MODEL")
    print("="*60)
    
    # Build triage features
    print("\nBuilding triage features...")
    triage_features, tfidf_matrix, vectorizer = build_triage_features(df)
    
    if len(triage_features) == 0:
        print("Error: Empty triage features")
        return
    
    print(f"✓ Features shape: {triage_features.shape}")
    
    # Get target
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found")
        return
    
    # Align with features
    y = df.loc[triage_features['unique_key'], target_col]
    families = df.loc[triage_features['unique_key'], 'complaint_family']
    
    # Remove unique_key from features
    X = triage_features.drop(columns=['unique_key'])
    
    print(f"Target: {y.mean():.1%} positive rate")
    
    # Hyperparameter tuning (optional)
    params = None
    if do_tune:
        print("\nRunning hyperparameter tuning...")
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
        
        tune_result = tune.tune_triage(
            X_train, y_train, X_val, y_val, n_trials=20
        )
        
        params = tune_result['best_params']
        params['objective'] = 'binary'
        params['n_estimators'] = 700
        params['random_state'] = 42
        params['verbose'] = -1
        
        print(f"✓ Tuning complete, using params: {params}")
    
    # Train model
    bundle = triage.train_triage(X, y, families, params=params)
    
    # Save model
    models_dir = output_dir / "models"
    triage.save_bundle(bundle, models_dir / "triage.joblib")
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred_proba = triage.predict_triage(bundle, X, families)
    
    metrics = model_eval.eval_triage(y, y_pred_proba, families)
    model_eval.print_metrics_summary(metrics, 'triage')
    
    # Save metrics
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_eval.save_metrics(metrics, metrics_dir / "triage_metrics.json")
    
    # Plot curves
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_eval.plot_triage_curves(
        y, y_pred_proba,
        title="Triage Model",
        output_path=plots_dir / "triage_curves.png"
    )
    
    print(f"\n✓ Triage training complete")


def train_duration_task(
    df: pd.DataFrame,
    output_dir: Path,
    do_tune: bool = False
):
    """
    Train duration/survival model.
    
    Args:
        df: Input DataFrame
        output_dir: Output directory
        do_tune: Whether to run hyperparameter tuning
    """
    print("\n" + "="*60)
    print("TRAINING DURATION MODEL")
    print("="*60)
    
    # Build duration features and labels
    print("\nBuilding duration features and labels...")
    duration_features = build_duration_features(df)
    duration_labels = build_duration_survival_labels(df)
    
    if len(duration_features) == 0 or len(duration_labels) == 0:
        print("Error: Empty duration data")
        return
    
    # Merge
    data = duration_features.merge(duration_labels, on='unique_key', how='inner')
    
    print(f"✓ Data shape: {data.shape}")
    print(f"✓ Event rate: {data['event_observed'].mean():.1%}")
    
    # Prepare X, t, e
    X = data.drop(columns=['unique_key', 'duration_days', 'ttc_days_cens', 
                           'event_observed', 'is_admin_like'])
    t = data['ttc_days_cens']
    e = data['event_observed']
    
    # Model selection and tuning
    if do_tune:
        print("\nRunning hyperparameter tuning and model selection...")
        best_bundle, best_model = duration.compare_models(
            X, t, e, models=['weibull', 'lognormal']
        )
    else:
        print("\nTraining Weibull AFT model (no tuning)...")
        best_bundle = duration.train_duration_aft(X, t, e, model_type='weibull')
    
    # Save model
    models_dir = output_dir / "models"
    duration.save_bundle(best_bundle, models_dir / "duration_aft.joblib")
    
    # Evaluate
    print("\nEvaluating model...")
    quantiles = duration.predict_duration_quantiles(best_bundle, X, ps=(0.5, 0.9))
    
    metrics = model_eval.eval_duration(t, quantiles['q50'], e, quantile=0.5)
    metrics_q90 = model_eval.eval_duration(t, quantiles['q90'], e, quantile=0.9)
    
    combined_metrics = {
        'q50': metrics,
        'q90': metrics_q90,
        'model_type': best_bundle['model_type']
    }
    
    model_eval.print_metrics_summary(metrics, 'duration')
    
    # Save metrics
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_eval.save_metrics(combined_metrics, metrics_dir / "duration_metrics.json")
    
    # Plot reliability
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_eval.plot_duration_reliability(
        t, quantiles['q50'], quantiles['q90'], e,
        title="Duration Model",
        output_path=plots_dir / "duration_reliability.png"
    )
    
    print(f"\n✓ Duration training complete")


def main():
    parser = argparse.ArgumentParser(description="Train NYC 311 models")
    parser.add_argument('--task', required=True, choices=['forecast', 'triage', 'duration'],
                       help='Model task to train')
    parser.add_argument('--input_parquet', required=True, type=str,
                       help='Path to input parquet file')
    parser.add_argument('--output_dir', required=True, type=str,
                       help='Output directory for models and metrics')
    parser.add_argument('--families', type=str, default=None,
                       help='Comma-separated list of complaint families (forecast only)')
    parser.add_argument('--tune', action='store_true',
                       help='Run hyperparameter tuning')
    parser.add_argument('--weather_parquet', type=str, default=None,
                       help='Path to weather parquet/csv (forecast only)')
    parser.add_argument('--target_col', type=str, default='potential_inspection_trigger',
                       help='Target column name (triage only)')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("NYC 311 MODEL TRAINING")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Input: {args.input_parquet}")
    print(f"Output: {args.output_dir}")
    print(f"Tuning: {args.tune}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(args.input_parquet)
    print(f"✓ Loaded {len(df):,} records")
    
    # Add H3 keys if not present
    if 'hex' not in df.columns:
        print("Adding H3 keys...")
        df = add_h3_keys(df)
    
    # Load weather if provided
    weather_df = None
    if args.weather_parquet:
        print(f"Loading weather data from {args.weather_parquet}...")
        if args.weather_parquet.endswith('.csv'):
            weather_df = pd.read_csv(args.weather_parquet)
        else:
            weather_df = pd.read_parquet(args.weather_parquet)
        print(f"✓ Loaded {len(weather_df):,} weather records")
    
    # Parse families
    families = None
    if args.families:
        families = [f.strip() for f in args.families.split(',')]
        print(f"Training families: {families}")
    
    # Train
    if args.task == 'forecast':
        train_forecast_task(df, output_dir, families=families, 
                          do_tune=args.tune, weather_df=weather_df)
    
    elif args.task == 'triage':
        train_triage_task(df, output_dir, target_col=args.target_col, 
                         do_tune=args.tune)
    
    elif args.task == 'duration':
        train_duration_task(df, output_dir, do_tune=args.tune)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE")
    print("="*60)
    print(f"Models saved to: {output_dir / 'models'}")
    print(f"Metrics saved to: {output_dir / 'metrics'}")
    if (output_dir / 'plots').exists():
        print(f"Plots saved to: {output_dir / 'plots'}")


if __name__ == '__main__':
    main()

