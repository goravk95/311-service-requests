"""
Example usage of feature engineering functions for NYC 311 service requests.
Demonstrates the complete pipeline for all three modeling tracks.
"""

import pandas as pd
import numpy as np
from features import (
    add_h3_keys,
    build_forecast_panel,
    build_triage_features,
    build_duration_survival_labels,
    build_duration_features
)


def example_pipeline(df: pd.DataFrame, 
                    weather_df: pd.DataFrame = None) -> dict:
    """
    Run complete feature engineering pipeline.
    
    Args:
        df: Raw 311 service request DataFrame
        weather_df: Optional weather data
    
    Returns:
        Dictionary with all feature sets and labels
    """
    print("Starting feature engineering pipeline...")
    
    # Step 1: Add H3 spatial keys and temporal features
    print("\n1. Adding H3 keys and temporal features...")
    df_with_keys = add_h3_keys(df, lat='latitude', lon='longitude', res=8)
    print(f"   Added H3 keys to {len(df_with_keys)} rows")
    print(f"   Valid hex cells: {df_with_keys['hex'].notna().sum()}")
    
    # Step 2: Build forecast panel for time-series modeling
    print("\n2. Building forecast panel...")
    forecast_panel = build_forecast_panel(
        df_with_keys, 
        weather_df=weather_df,
        use_population_offset=True
    )
    print(f"   Panel shape: {forecast_panel.shape}")
    print(f"   Unique hexes: {forecast_panel['hex'].nunique()}")
    print(f"   Date range: {forecast_panel['day'].min()} to {forecast_panel['day'].max()}")
    
    # Step 3: Build triage features for prioritization
    print("\n3. Building triage features...")
    triage_features, tfidf_matrix, vectorizer = build_triage_features(df_with_keys)
    print(f"   Triage features shape: {triage_features.shape}")
    if tfidf_matrix is not None:
        print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Step 4: Build duration labels with censoring
    print("\n4. Building duration labels...")
    duration_labels = build_duration_survival_labels(df_with_keys)
    print(f"   Labels shape: {duration_labels.shape}")
    print(f"   Event observed: {duration_labels['event_observed'].sum()}")
    print(f"   Censored: {(~duration_labels['event_observed'].astype(bool)).sum()}")
    print(f"   Admin-like censoring: {duration_labels['is_admin_like'].sum()}")
    
    # Step 5: Build duration features (triage + queue pressure)
    print("\n5. Building duration features...")
    duration_features = build_duration_features(df_with_keys, panel=forecast_panel)
    print(f"   Duration features shape: {duration_features.shape}")
    
    # Step 6: Quality checks
    print("\n6. Quality checks...")
    
    # Check for leakage in triage features
    leakage_cols = ['status', 'resolution_description', 'resolution_outcome', 
                   'resolution_action_updated_date', 'closed_date', 
                   'time_to_resolution', 'time_to_resolution_hours', 
                   'time_to_resolution_days']
    leakage_found = [col for col in leakage_cols if col in triage_features.columns]
    if leakage_found:
        print(f"   WARNING: Potential leakage columns found: {leakage_found}")
    else:
        print("   ✓ No leakage columns in triage features")
    
    # Check forecast panel sparsity
    if len(forecast_panel) > 0:
        unique_groups = forecast_panel.groupby(['hex', 'complaint_family']).size()
        avg_days_per_group = unique_groups.mean()
        print(f"   ✓ Forecast panel is sparse: avg {avg_days_per_group:.1f} days per group")
    
    # Check for missing critical features
    critical_features = ['lag7', 'roll7', 'roll28', 'momentum']
    missing_critical = [f for f in critical_features if f not in forecast_panel.columns]
    if missing_critical:
        print(f"   WARNING: Missing critical forecast features: {missing_critical}")
    else:
        print("   ✓ All critical forecast features present")
    
    print("\n✓ Pipeline complete!")
    
    return {
        'df_with_keys': df_with_keys,
        'forecast_panel': forecast_panel,
        'triage_features': triage_features,
        'tfidf_matrix': tfidf_matrix,
        'tfidf_vectorizer': vectorizer,
        'duration_labels': duration_labels,
        'duration_features': duration_features
    }


def load_sample_data(parquet_path: str = None) -> pd.DataFrame:
    """
    Load sample NYC 311 data for testing.
    
    Args:
        parquet_path: Path to parquet file (optional)
    
    Returns:
        Sample DataFrame
    """
    if parquet_path:
        return pd.read_parquet(parquet_path)
    else:
        # Create minimal synthetic sample for testing
        np.random.seed(42)
        n_samples = 1000
        
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
        
        df = pd.DataFrame({
            'created_date': dates,
            'latitude': np.random.uniform(40.5, 40.9, n_samples),
            'longitude': np.random.uniform(-74.2, -73.7, n_samples),
            'complaint_family': np.random.choice(['Health', 'Noise', 'Housing', 'Street'], n_samples),
            'complaint_type': np.random.choice(['Food Establishment', 'Residential', 'Commercial'], n_samples),
            'descriptor_clean': np.random.choice(['dirty', 'loud', 'broken', 'unsafe'], n_samples),
            'open_data_channel_type': np.random.choice(['PHONE', 'ONLINE', 'MOBILE'], n_samples),
            'borough': np.random.choice(['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX'], n_samples),
            'location_type': np.random.choice(['Residential', 'Commercial', 'Street'], n_samples),
            'facility_type': np.random.choice(['Building', 'Restaurant', 'Store'], n_samples),
            'address_type': np.random.choice(['ADDRESS', 'INTERSECTION'], n_samples),
            'incident_address': [f'{np.random.randint(1,999)} Main St' for _ in range(n_samples)],
            'street_name': 'Main St',
            'incident_zip': np.random.choice(['10001', '10002', '11201'], n_samples),
            'bbl': [f'{np.random.randint(1000000, 9999999)}' for _ in range(n_samples)],
            'fips': np.random.choice(['36061', '36047', '36081'], n_samples),
            'GEOID': [f'360{np.random.randint(100000, 999999)}' for _ in range(n_samples)],
            'population': np.random.randint(500, 5000, n_samples),
            'is_created_at_midnight': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'potential_inspection_trigger': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        })
        
        # Add closed dates (some missing)
        df['closed_date'] = df['created_date'] + pd.to_timedelta(
            np.random.exponential(5, n_samples), unit='D'
        )
        # Make 10% still open (no closed date)
        df.loc[np.random.choice(df.index, size=int(n_samples*0.1)), 'closed_date'] = pd.NaT
        
        # Add due dates
        df['due_date'] = df['created_date'] + pd.Timedelta(days=60)
        
        return df


if __name__ == '__main__':
    print("NYC 311 Feature Engineering - Example Usage\n")
    print("=" * 60)
    
    # Load sample data
    print("Loading sample data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} sample records")
    
    # Create sample weather data
    weather_dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    weather_df = pd.DataFrame({
        'day': weather_dates,
        'fips': '36061',
        'tavg': np.random.normal(60, 15, len(weather_dates)),
        'tmax': np.random.normal(70, 15, len(weather_dates)),
        'tmin': np.random.normal(50, 15, len(weather_dates)),
        'prcp': np.random.exponential(0.1, len(weather_dates))
    })
    
    # Add more FIPS
    weather_df = pd.concat([
        weather_df,
        weather_df.assign(fips='36047'),
        weather_df.assign(fips='36081')
    ], ignore_index=True)
    
    # Run pipeline
    results = example_pipeline(df, weather_df)
    
    # Display sample results
    print("\n" + "=" * 60)
    print("\nSample Forecast Panel (first 5 rows):")
    print(results['forecast_panel'].head())
    
    print("\nSample Triage Features (first 5 rows, first 10 cols):")
    print(results['triage_features'].iloc[:5, :10])
    
    print("\nSample Duration Labels (first 5 rows):")
    print(results['duration_labels'].head())
    
    print("\n" + "=" * 60)
    print("\n✓ Example pipeline completed successfully!")

