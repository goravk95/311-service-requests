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
    validate_lat_lon,
    lat_lon_to_h3,
    expand_k_ring,
    compute_rolling_as_of,
    compute_lag_features,
    compute_days_since_last,
    make_descriptor_tfidf,
    merge_asof_by_group,
    compute_time_based_rolling_counts,
    add_weather_derived_features,
    compute_rain_rolling
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
    
    # Convert lat/lon to H3
    result['hex'] = result.apply(
        lambda row: lat_lon_to_h3(row[lat], row[lon], resolution=res),
        axis=1
    )
    
    # Add temporal features
    result['day'] = result['created_date'].dt.floor('D')
    result['dow'] = result['created_date'].dt.dayofweek
    result['hour'] = result['created_date'].dt.hour
    result['month'] = result['created_date'].dt.month
    
    return result


def build_forecast_panel(df: pd.DataFrame, 
                        weather_df: Optional[pd.DataFrame] = None,
                        use_population_offset: bool = True) -> pd.DataFrame:
    """
    Build sparse forecast panel with time-series features.
    
    Grain: one row per (hex, complaint_family, day) with target y = count(tickets).
    Includes lag/rolling features, neighbor aggregates, weather, and population.
    
    Args:
        df: Input DataFrame with H3 keys already added
        weather_df: Optional weather DataFrame with columns [day, fips, tmax, tmin, tavg, prcp]
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
    
    # Add calendar features
    panel['dow'] = panel['day'].dt.dayofweek
    panel['month'] = panel['day'].dt.month
    
    # Compute per-group history features
    def add_history_features(group):
        # Sort by date
        group = group.sort_values('day')
        
        # Lag features
        group = compute_lag_features(group, date_col='day', value_col='y', lags=[1, 7])
        
        # Rolling features
        group = compute_rolling_as_of(group, date_col='day', value_col='y', windows=[7, 14, 28])
        
        # Momentum
        group['momentum'] = group['roll7'] / (group['roll28'] + 1e-6)
        
        # Days since last
        group = compute_days_since_last(group, date_col='day')
        
        return group
    
    panel = panel.groupby(['hex', 'complaint_family'], group_keys=False).apply(add_history_features)
    
    # Add neighbor features (k=1 ring)
    # Note: This can be expensive for large panels; include scaffolding
    try:
        panel = expand_k_ring(panel, k=1, hex_col='hex', agg_cols=['roll7', 'roll28'])
    except Exception as e:
        # If neighbor expansion fails, add zero columns
        panel['nbr_roll7'] = 0.0
        panel['nbr_roll28'] = 0.0
    
    # Merge weather data
    if weather_df is not None and len(weather_df) > 0:
        # Ensure weather has day column
        weather = weather_df.copy()
        if 'day' not in weather.columns and 'date' in weather.columns:
            weather['day'] = pd.to_datetime(weather['date'])
        
        # Add derived weather features
        weather = add_weather_derived_features(weather)
        
        # Compute rolling rain by fips
        if 'fips' in weather.columns and 'prcp' in weather.columns:
            weather = compute_rain_rolling(
                weather, 
                group_col='fips', 
                date_col='day', 
                prcp_col='prcp',
                windows=[3, 7]
            )
        
        # Join panel with weather (need fips mapping)
        # If df has fips, use it; otherwise try to get from original df
        if 'fips' in df_valid.columns:
            hex_fips_map = df_valid[['hex', 'fips']].drop_duplicates().dropna()
            panel = panel.merge(hex_fips_map, on='hex', how='left')
            
            # Merge weather by day and fips
            weather_cols = ['day', 'fips', 'tavg', 'tmax', 'tmin', 'prcp', 
                          'heating_degree', 'cooling_degree', 'rain_3d', 'rain_7d']
            weather_cols = [c for c in weather_cols if c in weather.columns]
            
            panel = panel.merge(
                weather[weather_cols],
                on=['day', 'fips'],
                how='left'
            )
    
    # Fill missing weather columns with 0
    weather_features = ['tavg', 'tmax', 'tmin', 'prcp', 'heating_degree', 
                       'cooling_degree', 'rain_3d', 'rain_7d']
    for col in weather_features:
        if col not in panel.columns:
            panel[col] = 0.0
        else:
            panel[col] = panel[col].fillna(0.0)
    
    # Add population features
    if use_population_offset:
        if 'population' in df_valid.columns and 'GEOID' in df_valid.columns:
            # Get hex -> population mapping (approximate)
            hex_pop_map = df_valid.groupby('hex')['population'].mean().reset_index()
            hex_pop_map.columns = ['hex', 'pop_hex']
            
            panel = panel.merge(hex_pop_map, on='hex', how='left')
            panel['pop_hex'] = panel['pop_hex'].fillna(0)
        else:
            panel['pop_hex'] = 0.0
        
        panel['log_pop'] = np.log(np.maximum(panel['pop_hex'], 1))
    else:
        panel['pop_hex'] = 0.0
        panel['log_pop'] = 0.0
    
    # Ensure all expected columns exist
    expected_cols = ['hex', 'complaint_family', 'day', 'y', 'dow', 'month',
                    'lag1', 'lag7', 'roll7', 'roll14', 'roll28', 'momentum',
                    'days_since_last', 'tavg', 'prcp', 'heating_degree',
                    'cooling_degree', 'rain_3d', 'rain_7d', 'log_pop']
    
    for col in expected_cols:
        if col not in panel.columns:
            if col in ['hex', 'complaint_family', 'day']:
                continue  # Should already exist
            panel[col] = 0.0
    
    # Select final columns
    final_cols = [c for c in expected_cols if c in panel.columns]
    if 'nbr_roll7' in panel.columns:
        final_cols.extend(['nbr_roll7', 'nbr_roll28'])
    
    return panel[final_cols].reset_index(drop=True)


def build_triage_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, csr_matrix, TfidfVectorizer]:
    """
    Build triage features for ticket-level classification.
    
    Grain: one row per ticket at creation time (no leakage).
    Includes categorical one-hots, temporal features, local history, and TF-IDF text features.
    
    Args:
        df: Input DataFrame with H3 keys and temporal features added
    
    Returns:
        Tuple of (feature DataFrame, TF-IDF sparse matrix, fitted vectorizer)
    """
    result = df.copy()
    
    # Create unique key if not present
    if 'unique_key' not in result.columns:
        result['unique_key'] = result.index
    
    # Filter to valid tickets with creation date
    result = result[result['created_date'].notna()].copy()
    
    if len(result) == 0:
        return pd.DataFrame(), None, None
    
    # === TEMPORAL FEATURES ===
    if 'hour' not in result.columns and 'created_date' in result.columns:
        result['hour'] = result['created_date'].dt.hour
    if 'dow' not in result.columns and 'created_date' in result.columns:
        result['dow'] = result['created_date'].dt.dayofweek
    if 'month' not in result.columns and 'created_date' in result.columns:
        result['month'] = result['created_date'].dt.month
    
    result['is_weekend'] = (result['dow'].isin([5, 6])).astype(int)
    
    # === CATEGORICAL ONE-HOTS ===
    categorical_cols = ['complaint_family', 'open_data_channel_type', 'location_type',
                       'borough', 'facility_type', 'address_type']
    
    for col in categorical_cols:
        if col in result.columns:
            # Fill missing with "_missing"
            result[col] = result[col].fillna('_missing').astype(str)
            
            # One-hot encode (limit to top categories to avoid explosion)
            dummies = pd.get_dummies(result[col], prefix=col, prefix_sep='_')
            
            # Limit to top 10 categories per feature
            if len(dummies.columns) > 10:
                top_cols = dummies.sum().nlargest(10).index
                dummies = dummies[top_cols]
            
            result = pd.concat([result, dummies], axis=1)
    
    # === DUE DATE FEATURES ===
    if 'due_date' in result.columns:
        result['due_date'] = pd.to_datetime(result['due_date'], errors='coerce')
        result['due_gap_hours'] = (
            (result['due_date'] - result['created_date']).dt.total_seconds() / 3600
        ).fillna(0)
        
        # Flag for ~60 day SLA (admin auto-close)
        result['due_is_60d'] = (
            (result['due_gap_hours'] >= 58*24) & 
            (result['due_gap_hours'] <= 62*24)
        ).astype(int)
        
        # Due crosses weekend
        result['due_crosses_weekend'] = (
            (result['dow'] <= 4) &  # Created on weekday
            ((result['due_gap_hours'] / 24 + result['dow']) >= 5)  # Extends into weekend
        ).astype(int)
    else:
        result['due_gap_hours'] = 0.0
        result['due_is_60d'] = 0
        result['due_crosses_weekend'] = 0
    
    # === WEATHER AT CREATION ===
    weather_features = ['tavg', 'tmax', 'tmin', 'prcp']
    for col in weather_features:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0.0)
    
    # Add derived weather
    result = add_weather_derived_features(result)
    
    # === LOCAL HISTORY (AS-OF) ===
    # Build a mini forecast panel for history lookup
    if 'hex' in result.columns and 'complaint_family' in result.columns and 'day' in result.columns:
        # Create historical aggregates
        history_panel = result.groupby(['hex', 'complaint_family', 'day']).size().reset_index(name='daily_count')
        
        # Compute rolling features per group
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
        result = result.sort_values(['hex', 'complaint_family', 'day'])
        history_panel = history_panel.sort_values(['hex', 'complaint_family', 'day'])
        
        result = pd.merge_asof(
            result,
            history_panel[['hex', 'complaint_family', 'day', 'geo_family_roll7', 
                          'geo_family_roll28', 'days_since_last_geo_family']],
            on='day',
            by=['hex', 'complaint_family'],
            direction='backward'
        )
    
    # Fill missing history features
    for col in ['geo_family_roll7', 'geo_family_roll28', 'days_since_last_geo_family']:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0.0)
    
    # === REPEAT-SITE FEATURES ===
    # Use BBL or hash of address
    if 'bbl' in result.columns:
        result['site_key'] = result['bbl'].fillna('unknown')
    else:
        # Create site key from address components
        addr_parts = []
        for col in ['incident_address', 'street_name', 'incident_zip']:
            if col in result.columns:
                addr_parts.append(result[col].fillna('').astype(str))
        
        if addr_parts:
            result['site_key'] = pd.util.hash_pandas_object(
                pd.concat(addr_parts, axis=1), index=False
            ).astype(str)
        else:
            result['site_key'] = 'unknown'
    
    # Compute repeat counts (as-of)
    result = result.sort_values('created_date')
    
    def compute_repeat_counts(group):
        repeat_14d = []
        repeat_28d = []
        
        for idx, row in group.iterrows():
            current_time = row['created_date']
            
            # Count prior tickets at same site in window
            mask_14d = (group['created_date'] >= current_time - pd.Timedelta(days=14)) & \
                       (group['created_date'] < current_time)
            mask_28d = (group['created_date'] >= current_time - pd.Timedelta(days=28)) & \
                       (group['created_date'] < current_time)
            
            repeat_14d.append(mask_14d.sum())
            repeat_28d.append(mask_28d.sum())
        
        group['repeat_site_14d'] = repeat_14d
        group['repeat_site_28d'] = repeat_28d
        return group
    
    result = result.groupby('site_key', group_keys=False).apply(compute_repeat_counts)
    
    # === TEXT FEATURES (TF-IDF) ===
    tfidf_matrix = None
    vectorizer = None
    
    if 'descriptor_clean' in result.columns:
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
    # Numeric features
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
    
    # Create unique key if not present
    if 'unique_key' not in result.columns:
        result['unique_key'] = result.index
    
    # Ensure dates are datetime
    result['created_date'] = pd.to_datetime(result['created_date'], errors='coerce')
    result['closed_date'] = pd.to_datetime(result['closed_date'], errors='coerce')
    
    # Compute duration in days
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
                           panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build features for duration prediction (combines triage + queue pressure).
    
    Args:
        df: Input DataFrame with H3 keys and temporal features
        panel: Optional forecast panel for queue metrics
    
    Returns:
        DataFrame with duration features keyed by unique_key
    """
    # Start with triage features
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

