"""
Prediction Pipeline CLI for Forecast, Triage, and Duration models.
"""

import sys
sys.path.append('.')

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.features import (
    add_h3_keys,
    build_triage_features,
    build_duration_features
)

from models import forecast, triage, duration


def predict_forecast_task(
    df: pd.DataFrame,
    model_dir: Path,
    output_csv: Path,
    horizon: int = 7
):
    """
    Generate forecast predictions.
    
    Args:
        df: Input DataFrame
        model_dir: Directory with trained models
        output_csv: Output CSV path
        horizon: Forecast horizon (1-7)
    """
    print("\n" + "="*60)
    print("FORECAST PREDICTION")
    print("="*60)
    
    # Load models
    print(f"\nLoading models from {model_dir}...")
    bundles = forecast.load_bundles(model_dir)
    
    if not bundles:
        print("Error: No forecast models found")
        return
    
    print(f"✓ Loaded {len(bundles)} family models")
    
    # Get latest row per [hex, family]
    print("\nExtracting latest state per location/family...")
    
    df_sorted = df.sort_values('day')
    last_rows = df_sorted.groupby(['hex', 'complaint_family']).last().reset_index()
    
    print(f"✓ Found {len(last_rows)} location/family combinations")
    
    # Predict
    print(f"\nGenerating {horizon}-day forecasts...")
    predictions = forecast.predict_forecast(bundles, last_rows, horizon=horizon)
    
    if len(predictions) == 0:
        print("Warning: No predictions generated")
        return
    
    print(f"✓ Generated {len(predictions):,} predictions")
    
    # Save
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    
    print(f"✓ Saved predictions -> {output_csv}")
    
    # Summary
    print("\nPrediction summary:")
    print(predictions[['p50', 'p10', 'p90']].describe())


def predict_triage_task(
    df: pd.DataFrame,
    model_path: Path,
    output_csv: Path
):
    """
    Generate triage predictions.
    
    Args:
        df: Input DataFrame
        model_path: Path to trained model
        output_csv: Output CSV path
    """
    print("\n" + "="*60)
    print("TRIAGE PREDICTION")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    bundle = triage.load_bundle(model_path)
    
    # Build features
    print("\nBuilding triage features...")
    triage_features, tfidf_matrix, vectorizer = build_triage_features(df)
    
    if len(triage_features) == 0:
        print("Error: Empty triage features")
        return
    
    print(f"✓ Features shape: {triage_features.shape}")
    
    # Get families
    families = df.loc[triage_features['unique_key'], 'complaint_family']
    
    # Remove unique_key from features
    X = triage_features.drop(columns=['unique_key'])
    
    # Predict
    print("\nGenerating predictions...")
    probabilities = triage.predict_triage(bundle, X, families)
    
    # Create output
    results = pd.DataFrame({
        'unique_key': triage_features['unique_key'],
        'triage_probability': probabilities
    })
    
    # Merge with original data for context
    if 'unique_key' in df.columns:
        results = results.merge(
            df[['unique_key', 'complaint_family', 'created_date', 'hex']],
            on='unique_key',
            how='left'
        )
    
    print(f"✓ Generated {len(results):,} predictions")
    
    # Save
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    
    print(f"✓ Saved predictions -> {output_csv}")
    
    # Summary
    print("\nPrediction summary:")
    print(results['triage_probability'].describe())
    print(f"High priority (>0.5): {(results['triage_probability'] > 0.5).sum():,} "
          f"({(results['triage_probability'] > 0.5).mean():.1%})")


def predict_duration_task(
    df: pd.DataFrame,
    model_path: Path,
    output_csv: Path
):
    """
    Generate duration predictions.
    
    Args:
        df: Input DataFrame
        model_path: Path to trained model
        output_csv: Output CSV path
    """
    print("\n" + "="*60)
    print("DURATION PREDICTION")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    bundle = duration.load_bundle(model_path)
    
    # Build features
    print("\nBuilding duration features...")
    duration_features = build_duration_features(df)
    
    if len(duration_features) == 0:
        print("Error: Empty duration features")
        return
    
    print(f"✓ Features shape: {duration_features.shape}")
    
    # Prepare X
    X = duration_features.drop(columns=['unique_key'])
    
    # Predict quantiles
    print("\nGenerating predictions...")
    quantiles = duration.predict_duration_quantiles(bundle, X, ps=(0.5, 0.9))
    
    # Create output
    results = pd.DataFrame({
        'unique_key': duration_features['unique_key'],
        'predicted_duration_q50_days': quantiles['q50'],
        'predicted_duration_q90_days': quantiles['q90']
    })
    
    # Merge with original data for context
    if 'unique_key' in df.columns:
        results = results.merge(
            df[['unique_key', 'complaint_family', 'created_date', 'hex']],
            on='unique_key',
            how='left'
        )
    
    print(f"✓ Generated {len(results):,} predictions")
    
    # Save
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    
    print(f"✓ Saved predictions -> {output_csv}")
    
    # Summary
    print("\nPrediction summary:")
    print(results[['predicted_duration_q50_days', 'predicted_duration_q90_days']].describe())


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for NYC 311 models")
    parser.add_argument('--task', required=True, choices=['forecast', 'triage', 'duration'],
                       help='Model task to predict')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Directory with trained models (for forecast)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model file (for triage/duration)')
    parser.add_argument('--scoring_parquet', required=True, type=str,
                       help='Path to input parquet file for scoring')
    parser.add_argument('--output_csv', required=True, type=str,
                       help='Output CSV path for predictions')
    parser.add_argument('--horizon', type=int, default=7,
                       help='Forecast horizon in days (forecast only)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("NYC 311 MODEL PREDICTION")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Input: {args.scoring_parquet}")
    print(f"Output: {args.output_csv}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(args.scoring_parquet)
    print(f"✓ Loaded {len(df):,} records")
    
    # Add H3 keys if not present
    if 'hex' not in df.columns:
        print("Adding H3 keys...")
        df = add_h3_keys(df)
    
    # Add unique_key if not present
    if 'unique_key' not in df.columns:
        df['unique_key'] = df.index
    
    # Predict
    if args.task == 'forecast':
        if not args.model_dir:
            print("Error: --model_dir required for forecast task")
            return
        
        predict_forecast_task(df, Path(args.model_dir), Path(args.output_csv), 
                            horizon=args.horizon)
    
    elif args.task == 'triage':
        if not args.model_path:
            print("Error: --model_path required for triage task")
            return
        
        predict_triage_task(df, Path(args.model_path), Path(args.output_csv))
    
    elif args.task == 'duration':
        if not args.model_path:
            print("Error: --model_path required for duration task")
            return
        
        predict_duration_task(df, Path(args.model_path), Path(args.output_csv))
    
    print("\n" + "="*60)
    print("✓ PREDICTION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

