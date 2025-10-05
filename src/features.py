"""
Feature engineering for NYC 311 service requests.
Implements leakage-safe feature builders for Forecast, Triage, and Duration models.
"""

import numpy as np
import pandas as pd
import h3
from typing import Optional, Tuple, Dict, List
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import (
    aggregate_on_parent,
    add_history_features,
    make_descriptor_tfidf,
    compute_time_based_rolling_counts
)


def add_h3_keys(df: pd.DataFrame, 
                lat: str = 'latitude', 
                lon: str = 'longitude', 
                res: int = 8) -> pd.DataFrame:
    """
    Add H3 hex keys and temporal features to DataFrame.
    
    Args:
        df: Input DataFrame with location and timestamp data
        lat: Latitude column name
        lon: Longitude column name
        res: H3 resolution (default 8)
    
    Returns:
        DataFrame with added columns: hex, day, dow, hour, month
    """
    result = df.copy()

    result['hex'] = result.apply(
        lambda row: h3.latlng_to_cell(row[lat], row[lon], res),
        axis=1
    )
    result['day'] = result['created_date'].dt.floor('D')
    result['dow'] = result['created_date'].dt.dayofweek
    result['hour'] = result['created_date'].dt.hour
    result['month'] = result['created_date'].dt.month
    result['is_weekend'] = (result['dow'].isin([5, 6])).astype(int)
    
    return result


def build_forecast_panel(df: pd.DataFrame, 
                        use_population_offset: bool = True) -> pd.DataFrame:
    """
    Build sparse forecast panel with time-series features.
    
    Grain: one row per (hex, complaint_family, day) with target y = count(tickets).
    Includes lag/rolling features, neighbor aggregates, weather, and population.
    
    Weather features (tavg, tmax, tmin, prcp, heating_degree, cooling_degree,
    rain_3d, rain_7d, heat_flag, freeze_flag) should already be present in df
    from preprocessing (via preprocess_and_merge_external_data).
    
    Args:
        df: Input DataFrame with H3 keys and weather features already added
        use_population_offset: Whether to include population features
    
    Returns:
        Sparse panel DataFrame with forecast features
    """
    # Filter to valid hex and complaint_family
    df_valid = df[df['hex'].notna() & df['complaint_family'].notna()].copy()
    
    if len(df_valid) == 0:
        return pd.DataFrame()
    
    # Create base panel: aggregate by hex, complaint_family, day
    panel = df_valid.groupby(['hex', 'complaint_family', 'day']).agg(
        y=('hex', 'size')  # Count tickets
    ).reset_index()
    
    panel = panel.sort_values('day')

    panel = panel.groupby(['hex', 'complaint_family'], group_keys=False).apply(
        lambda group: add_history_features(group, date_col='day', value_col='y', lags=[1, 7], windows=[7, 28])
    )
    panel['momentum'] = panel['roll7'] / (panel['roll28'] + 1e-6)
    panel['dow'] = panel['day'].dt.dayofweek
    panel['month'] = panel['day'].dt.month

    panel = aggregate_on_parent(panel, res=7, hex_col='hex', agg_cols=['roll7', 'roll28'])
    
    weather_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'heating_degree', 
                   'cooling_degree', 'rain_3d', 'rain_7d']
    
    # Only aggregate weather columns that exist
    weather_agg = {}
    for col in weather_cols:
        # Use mean for most weather features, sum for precipitation
        weather_agg[col] = 'sum' if col == 'prcp' else 'mean'
    
    weather_from_df = df_valid.groupby(['hex', 'complaint_family', 'day']).agg(weather_agg).reset_index()
    
    # Merge with panel
    panel = panel.merge(
        weather_from_df,
        on=['hex', 'complaint_family', 'day'],
        how='left'
    )
    
    hex_pop_map = df_valid.groupby(['hex', 'GEOID'])['population'].first().reset_index()
    hex_pop_map = hex_pop_map.groupby('hex')['population'].sum().reset_index()
    hex_pop_map.columns = ['hex', 'pop_hex']
    
    panel = panel.merge(hex_pop_map, on='hex', how='left')
    panel['pop_hex'] = panel['pop_hex'].fillna(0)
    
    panel['log_pop'] = np.log(np.maximum(panel['pop_hex'], 1))
    
    # Ensure all expected columns exist
    expected_cols = ['hex', 'complaint_family', 'day', 'y', 'dow', 'month',
                    'lag1', 'lag7', 'roll7', 'roll28', 'momentum',
                    'days_since_last', 'tavg', 'prcp', 'heating_degree',
                    'cooling_degree', 'rain_3d', 'rain_7d', 'log_pop', 'nbr_roll7', 'nbr_roll28']
    
    return panel[expected_cols].reset_index(drop=True)


def build_triage_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, csr_matrix, TfidfVectorizer]:
    """
    Build triage features for ticket-level classification.
    
    Grain: one row per ticket at creation time (no leakage).
    Includes categorical one-hots, temporal features, local history, and TF-IDF text features.
    
    Weather features should already be present in df from preprocessing
    (via preprocess_and_merge_external_data).
    
    Args:
        df: Input DataFrame with H3 keys, temporal features, and weather features added
    
    Returns:
        Tuple of (feature DataFrame, TF-IDF sparse matrix, fitted vectorizer)
    """
    result = df.copy()
    
    result = result[result['created_date'].notna()].copy()
    
    categorical_cols = ['complaint_family', 'open_data_channel_type', 'location_type',
                       'borough', 'facility_type']
    
    for col in categorical_cols:
        result[col] = result[col].fillna('_missing').astype(str)
        
        dummies = pd.get_dummies(result[col], prefix=col, prefix_sep='_')
        
        if len(dummies.columns) > 10:
            top_cols = dummies.sum().nlargest(10).index
            dummies = dummies[top_cols]
        
        result = pd.concat([result, dummies], axis=1)
    
    result['due_date'] = pd.to_datetime(result['due_date'], errors='coerce')
    result['due_gap_hours'] = (
        (result['due_date'] - result['created_date']).dt.total_seconds() / 3600
    ).fillna(0)
    
    # Due crosses weekend
    result['due_crosses_weekend'] = (
        (result['dow'] <= 4) &  # Created on weekday
        ((result['due_gap_hours'] / 24 + result['dow']) >= 5)  # Extends into weekend
    ).astype(int)
    
    history_panel = result.groupby(['hex', 'complaint_family', 'day']).size().reset_index(name='daily_count')
    
    def compute_history(group):
        group = group.sort_values('day')
        
        # 7-day rolling
        group['geo_family_roll7'] = group['daily_count'].rolling(window=7, min_periods=1).sum()
        
        # 28-day rolling
        group['geo_family_roll28'] = group['daily_count'].rolling(window=28, min_periods=1).sum()
        
        # Days since last
        group['days_since_last_geo_family'] = group['day'].diff().dt.days.fillna(999)
        
        return group
    
    history_panel = history_panel.groupby(['hex', 'complaint_family'], group_keys=False).apply(compute_history)
    
    # Merge back to tickets using as-of join
    result = result.sort_values(['day', 'hex', 'complaint_family'])
    history_panel = history_panel.sort_values(['day', 'hex', 'complaint_family'])
    result = pd.merge_asof(
        result,
        history_panel[['hex', 'complaint_family', 'day', 'geo_family_roll7', 
                        'geo_family_roll28', 'days_since_last_geo_family']],
        on='day',
        by=['hex', 'complaint_family'],
        direction='backward'
    )
    
    for col in ['geo_family_roll7', 'geo_family_roll28', 'days_since_last_geo_family']:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0.0)
    
    result['site_key'] = result['bbl'].fillna('unknown')

    site_panel = result.groupby(['site_key', 'day']).size().reset_index(name='daily_site_count')
    
    # Compute rolling features per site
    def compute_site_history(group):
        group = group.sort_values('day')
        
        # 14-day rolling count
        group['repeat_site_14d'] = group['daily_site_count'].rolling(window=14, min_periods=1).sum()
        
        # 28-day rolling count
        group['repeat_site_28d'] = group['daily_site_count'].rolling(window=28, min_periods=1).sum()
        
        return group
    
    site_panel = site_panel.groupby('site_key', group_keys=False).apply(compute_site_history)
    
    # Merge back to tickets using as-of join
    result = result.sort_values(['day', 'site_key'])
    site_panel = site_panel.sort_values(['day', 'site_key'])
    result = pd.merge_asof(
        result,
        site_panel[['site_key', 'day', 'repeat_site_14d', 'repeat_site_28d']],
        on='day',
        by='site_key',
        direction='backward'
    )
    
    # Subtract 1 to exclude current ticket from the count (we want prior tickets only)
    result['repeat_site_14d'] = (result['repeat_site_14d'] - 1).clip(lower=0)
    result['repeat_site_28d'] = (result['repeat_site_28d'] - 1).clip(lower=0)
    
    # Fill any missing values
    result['repeat_site_14d'] = result['repeat_site_14d'].fillna(0.0)
    result['repeat_site_28d'] = result['repeat_site_28d'].fillna(0.0)
    
    # === TEXT FEATURES (TF-IDF) ===
    tfidf_matrix = None
    vectorizer = None
    
    try:
        tfidf_matrix, vectorizer = make_descriptor_tfidf(
            result, 
            col='descriptor_clean',
            min_df=5,
            ngram_range=(1, 2),
            max_features=500
        )
    except Exception as e:
        print(f"Warning: TF-IDF failed: {e}")
        tfidf_matrix = None
        vectorizer = None
    
    # === SELECT FEATURE COLUMNS ===
    numeric_features = [
        'hour', 'dow', 'month', 'is_created_at_midnight', 'is_weekend',
        'due_gap_hours', 'due_is_60d', 'due_crosses_weekend',
        'tavg', 'prcp', 'heat_flag', 'freeze_flag',
        'geo_family_roll7', 'geo_family_roll28', 'days_since_last_geo_family',
        'repeat_site_14d', 'repeat_site_28d'
    ]
    
    # Get one-hot columns
    onehot_cols = [c for c in result.columns if any(
        c.startswith(f'{cat}_') for cat in categorical_cols
    )]
    
    # Combine all feature columns
    feature_cols = ['unique_key'] + [c for c in numeric_features if c in result.columns] + onehot_cols
    
    # Fill missing numeric features
    for col in numeric_features:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)
    
    features_df = result[feature_cols].copy()
    
    return features_df, tfidf_matrix, vectorizer


def build_duration_survival_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build labels for duration/survival modeling with censoring.
    
    Args:
        df: Input DataFrame with created_date and closed_date
    
    Returns:
        DataFrame with duration labels and censoring indicators
    """
    result = df.copy()
    
    result['duration_days'] = (
        (result['closed_date'] - result['created_date']).dt.total_seconds() / 86400
    )
    
    # Initialize censoring
    result['event_observed'] = 0
    result['ttc_days_cens'] = result['duration_days']
    result['is_admin_like'] = 0
    
    # Mark admin auto-close (59-62 days)
    admin_mask = (result['duration_days'] >= 59) & (result['duration_days'] <= 62)
    result.loc[admin_mask, 'is_admin_like'] = 1
    result.loc[admin_mask, 'ttc_days_cens'] = 60.5
    result.loc[admin_mask, 'event_observed'] = 0  # Censored
    
    # Mark long stale (>365 days)
    stale_mask = (result['duration_days'] > 365)
    result.loc[stale_mask, 'ttc_days_cens'] = 365
    result.loc[stale_mask, 'event_observed'] = 0  # Censored
    
    # Mark true closures (not censored)
    true_close_mask = (
        result['closed_date'].notna() & 
        ~admin_mask & 
        ~stale_mask
    )
    result.loc[true_close_mask, 'event_observed'] = 1
    
    # Handle missing closed_date (open tickets)
    missing_close_mask = result['closed_date'].isna()
    result.loc[missing_close_mask, 'ttc_days_cens'] = 365  # Censor at 1 year
    result.loc[missing_close_mask, 'event_observed'] = 0
    
    # Select output columns
    output_cols = [
        'unique_key',
        'duration_days',
        'ttc_days_cens',
        'event_observed',
        'is_admin_like'
    ]
    
    return result[output_cols].copy()


def build_duration_features(df: pd.DataFrame, 
                           triage_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build features for duration prediction (combines triage + queue pressure).
    
    Args:
        df: Input DataFrame with H3 keys and temporal features
        panel: Optional forecast panel for queue metrics
    
    Returns:
        DataFrame with duration features keyed by unique_key
    """
    # Start with triage features
    if triage_features is None:
        triage_features, _, _ = build_triage_features(df)
    
    result = df.copy()
    
    # Create unique key if not present
    if 'unique_key' not in result.columns:
        result['unique_key'] = result.index
    
    # === INTAKE PRESSURE FEATURES ===
    # Compute rolling intake counts in same region+family
    if 'fips' in result.columns and 'complaint_family' in result.columns:
        result = compute_time_based_rolling_counts(
            result,
            timestamp_col='created_date',
            group_cols=['fips', 'complaint_family'],
            window_hours=[6, 24]
        )
    else:
        result['intake_6h'] = 0
        result['intake_24h'] = 0
    
    # === OPEN BACKLOG PROXY ===
    # Approximate open items in same hex+family
    if 'hex' in result.columns and 'complaint_family' in result.columns:
        # Count tickets created in last 7 days (proxy for open queue)
        result = result.sort_values(['hex', 'complaint_family', 'created_date'])
        
        def compute_open_proxy(group):
            open_7d = []
            
            for idx, row in group.iterrows():
                current_time = row['created_date']
                cutoff_time = current_time - pd.Timedelta(days=7)
                
                # Count tickets in window
                mask = (group['created_date'] >= cutoff_time) & \
                       (group['created_date'] < current_time)
                open_7d.append(mask.sum())
            
            group['open_7d_geo_family'] = open_7d
            return group
        
        result = result.groupby(['hex', 'complaint_family'], group_keys=False).apply(compute_open_proxy)
    else:
        result['open_7d_geo_family'] = 0
    
    # === DUE DATE FEATURES (already in triage) ===
    # Ensure these exist
    for col in ['due_gap_hours', 'due_is_60d', 'due_crosses_weekend']:
        if col not in result.columns:
            result[col] = 0
    
    # === COMBINE WITH TRIAGE FEATURES ===
    queue_features = result[['unique_key', 'intake_6h', 'intake_24h', 'open_7d_geo_family']].copy()
    
    # Merge with triage features
    final_features = triage_features.merge(queue_features, on='unique_key', how='left')
    
    # Fill missing
    for col in ['intake_6h', 'intake_24h', 'open_7d_geo_family']:
        if col in final_features.columns:
            final_features[col] = final_features[col].fillna(0)
    
    return final_features

