"""
Feature engineering for NYC 311 service requests.
Implements leakage-safe feature builders for Forecast, Triage, and Duration models.
"""

import numpy as np
import pandas as pd
import h3
from typing import List


def aggregate_on_parent(df_panel: pd.DataFrame, res: int = 7, 
                  hex_col: str = 'hex', 
                  agg_cols: List[str] = ['roll7', 'roll28']) -> pd.DataFrame:
    """
    Expand a panel to include neighbor aggregations.
    For each hex, compute neighbor sums of specified columns.
    
    Optimized vectorized implementation - avoids O(N²) row-by-row iteration.
    
    Args:
        df_panel: DataFrame with hex column and metrics
        k: Ring distance
        hex_col: Name of hex column
        agg_cols: Columns to aggregate from neighbors
    
    Returns:
        DataFrame with original data plus neighbor aggregates (prefixed with 'nbr_')
    """
    df_panel[f'hex_{res}'] = df_panel[hex_col].apply(lambda x: h3.cell_to_parent(x, res))
    for col in agg_cols:
        df_panel[f'nbr_{col}'] = df_panel.groupby(['day', 'complaint_family', f'hex_{res}'])[col].transform('sum')
    return df_panel



def add_history_features(group_df: pd.DataFrame,
                            date_col: str = 'day',
                            value_col: str = 'y',
                            lags: List[int] = [1, 7],
                            windows: List[int] = [7, 28]) -> pd.DataFrame:
    """
    Calculate group history features.
    """
    # Find min and max dates in group_df
    min_date = group_df[date_col].min()
    max_date = group_df[date_col].max()

    # Create complete date range
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    # Get unique combinations of hex and complaint_family from group_df
    hex_val = group_df['hex'].iloc[0]
    complaint_family_val = group_df['complaint_family'].iloc[0]

    # Create complete panel with all dates
    complete_panel = pd.DataFrame({
        'hex': hex_val,
        'complaint_family': complaint_family_val,
        date_col: date_range
    })

    # Merge with original data to fill in y values
    group_df = complete_panel.merge(
        group_df[['hex', 'complaint_family', date_col, value_col]], 
        on=['hex', 'complaint_family', date_col], 
        how='left'
    ).fillna({'y': 0})

    for lag in lags:
        col_name = f'lag{lag}'
        group_df[col_name] = group_df[value_col].shift(lag)

    for window in windows:
        col_name = f'roll{window}'
        group_df[col_name] = group_df[value_col].rolling(window).sum()

    group_df = group_df[group_df[value_col] != 0]
    group_df['days_since_last'] = group_df[date_col].diff().dt.days

    return group_df


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


