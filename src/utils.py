"""
Utility functions for feature engineering on NYC 311 service requests.
Provides helpers for H3 spatial operations, as-of joins, neighbor expansion, and text processing.
"""

import numpy as np
import pandas as pd
import h3
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def validate_lat_lon(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """
    Returns a boolean mask for valid latitude/longitude pairs.
    
    Args:
        lat: Latitude series
        lon: Longitude series
    
    Returns:
        Boolean series indicating valid coordinates
    """
    valid_lat = pd.to_numeric(lat, errors='coerce').notna() & \
                (pd.to_numeric(lat, errors='coerce') >= -90) & \
                (pd.to_numeric(lat, errors='coerce') <= 90)
    valid_lon = pd.to_numeric(lon, errors='coerce').notna() & \
                (pd.to_numeric(lon, errors='coerce') >= -180) & \
                (pd.to_numeric(lon, errors='coerce') <= 180)
    return valid_lat & valid_lon


def lat_lon_to_h3(lat: float, lon: float, resolution: int = 8) -> Optional[str]:
    """
    Convert a single lat/lon pair to H3 hex ID.
    
    Args:
        lat: Latitude
        lon: Longitude
        resolution: H3 resolution (default 8)
    
    Returns:
        H3 hex ID or None if invalid
    """
    try:
        if pd.isna(lat) or pd.isna(lon):
            return None
        lat_f = float(lat)
        lon_f = float(lon)
        if lat_f < -90 or lat_f > 90 or lon_f < -180 or lon_f > 180:
            return None
        return h3.geo_to_h3(lat_f, lon_f, resolution)
    except (ValueError, TypeError):
        return None


def get_h3_neighbors(hex_id: str, k: int = 1) -> List[str]:
    """
    Get k-ring neighbors of an H3 hex.
    
    Args:
        hex_id: H3 hex ID
        k: Ring distance (default 1)
    
    Returns:
        List of neighbor hex IDs (excluding center)
    """
    try:
        neighbors = h3.k_ring(hex_id, k)
        return [h for h in neighbors if h != hex_id]
    except:
        return []


def expand_k_ring(df_panel: pd.DataFrame, k: int = 1, 
                  hex_col: str = 'hex', 
                  agg_cols: List[str] = ['roll7', 'roll28']) -> pd.DataFrame:
    """
    Expand a panel to include neighbor aggregations.
    For each hex, compute neighbor sums of specified columns.
    
    Args:
        df_panel: DataFrame with hex column and metrics
        k: Ring distance
        hex_col: Name of hex column
        agg_cols: Columns to aggregate from neighbors
    
    Returns:
        DataFrame with original data plus neighbor aggregates (prefixed with 'nbr_')
    """
    # Get unique hexes
    unique_hexes = df_panel[hex_col].dropna().unique()
    
    # Build neighbor mapping
    neighbor_map = {}
    for hex_id in unique_hexes:
        neighbors = get_h3_neighbors(hex_id, k=k)
        neighbor_map[hex_id] = neighbors
    
    # For each row, compute neighbor aggregates
    neighbor_features = {}
    for col in agg_cols:
        neighbor_features[f'nbr_{col}'] = []
    
    for idx, row in df_panel.iterrows():
        hex_id = row[hex_col]
        if pd.isna(hex_id) or hex_id not in neighbor_map:
            for col in agg_cols:
                neighbor_features[f'nbr_{col}'].append(0.0)
            continue
        
        neighbors = neighbor_map[hex_id]
        
        # Filter panel to same day, complaint_family, and neighbor hexes
        filter_mask = (df_panel[hex_col].isin(neighbors))
        if 'day' in df_panel.columns:
            filter_mask &= (df_panel['day'] == row['day'])
        if 'complaint_family' in df_panel.columns:
            filter_mask &= (df_panel['complaint_family'] == row['complaint_family'])
        
        neighbor_data = df_panel[filter_mask]
        
        for col in agg_cols:
            if len(neighbor_data) > 0:
                neighbor_features[f'nbr_{col}'].append(neighbor_data[col].sum())
            else:
                neighbor_features[f'nbr_{col}'].append(0.0)
    
    # Add neighbor features to original dataframe
    result = df_panel.copy()
    for col, values in neighbor_features.items():
        result[col] = values
    
    return result


def compute_rolling_as_of(group_df: pd.DataFrame, 
                          date_col: str = 'day',
                          value_col: str = 'y',
                          windows: List[int] = [7, 14, 28]) -> pd.DataFrame:
    """
    Compute rolling sums for multiple windows using cumulative approach.
    
    Args:
        group_df: DataFrame for a single group (sorted by date)
        date_col: Date column name
        value_col: Value column to aggregate
        windows: List of window sizes in days
    
    Returns:
        DataFrame with rolling features added
    """
    df = group_df.sort_values(date_col).copy()
    
    # Compute cumulative sum
    df['_cumsum'] = df[value_col].cumsum()
    
    # For each window, find the cutoff date and subtract cumsum at that point
    for window in windows:
        col_name = f'roll{window}'
        
        # Shift dates back by window days
        cutoff_dates = df[date_col] - pd.Timedelta(days=window)
        
        # For each row, find cumsum at cutoff date
        roll_values = []
        for i, (idx, row) in enumerate(df.iterrows()):
            cutoff = cutoff_dates.iloc[i]
            # Find last row before cutoff
            before_mask = df[date_col] < cutoff
            if before_mask.any():
                cumsum_before = df.loc[before_mask, '_cumsum'].iloc[-1]
            else:
                cumsum_before = 0
            
            # Rolling sum = current cumsum - cumsum before window
            roll_values.append(row['_cumsum'] - cumsum_before)
        
        df[col_name] = roll_values
    
    df.drop(columns=['_cumsum'], inplace=True)
    return df


def compute_lag_features(group_df: pd.DataFrame,
                        date_col: str = 'day',
                        value_col: str = 'y',
                        lags: List[int] = [1, 7]) -> pd.DataFrame:
    """
    Compute lag features using dict lookups (sparse-safe).
    
    Args:
        group_df: DataFrame for a single group (sorted by date)
        date_col: Date column name
        value_col: Value column to aggregate
        lags: List of lag periods in days
    
    Returns:
        DataFrame with lag features added
    """
    df = group_df.sort_values(date_col).copy()
    
    # Build date -> value lookup
    date_value_map = dict(zip(df[date_col], df[value_col]))
    
    for lag in lags:
        col_name = f'lag{lag}'
        lag_dates = df[date_col] - pd.Timedelta(days=lag)
        df[col_name] = lag_dates.map(date_value_map).fillna(0)
    
    return df


def compute_days_since_last(group_df: pd.DataFrame,
                            date_col: str = 'day') -> pd.DataFrame:
    """
    Compute days since last occurrence in the group.
    
    Args:
        group_df: DataFrame for a single group (sorted by date)
        date_col: Date column name
    
    Returns:
        DataFrame with days_since_last feature added
    """
    df = group_df.sort_values(date_col).copy()
    
    days_since = []
    for i, date in enumerate(df[date_col]):
        if i == 0:
            days_since.append(999)  # Large value for first occurrence
        else:
            prev_date = df[date_col].iloc[i-1]
            days_since.append((date - prev_date).days)
    
    df['days_since_last'] = days_since
    return df


def make_descriptor_tfidf(df: pd.DataFrame, 
                          col: str = 'descriptor_clean',
                          min_df: int = 5,
                          ngram_range: Tuple[int, int] = (1, 2),
                          max_features: int = 500) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Create TF-IDF features from text column.
    
    Args:
        df: DataFrame with text column
        col: Column name containing text
        min_df: Minimum document frequency
        ngram_range: N-gram range for tokenization
        max_features: Maximum number of features
    
    Returns:
        Tuple of (sparse feature matrix, fitted vectorizer)
    """
    # Fill missing values with empty string
    text_data = df[col].fillna('').astype(str)
    
    # Initialize and fit vectorizer
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        ngram_range=ngram_range,
        max_features=max_features,
        strip_accents='unicode',
        lowercase=True,
        token_pattern=r'\b\w+\b'
    )
    
    X_sparse = vectorizer.fit_transform(text_data)
    
    return X_sparse, vectorizer


def merge_asof_by_group(df_tickets: pd.DataFrame,
                       df_panel: pd.DataFrame,
                       by_cols: List[str],
                       left_on: str = 'day',
                       right_on: str = 'day',
                       direction: str = 'backward',
                       suffixes: Tuple[str, str] = ('', '_panel')) -> pd.DataFrame:
    """
    Perform merge_asof for each group separately (for sparse data).
    
    Args:
        df_tickets: Left DataFrame (tickets)
        df_panel: Right DataFrame (panel data)
        by_cols: Columns to group by
        left_on: Date column in left DataFrame
        right_on: Date column in right DataFrame
        direction: Direction for merge_asof
        suffixes: Suffixes for overlapping columns
    
    Returns:
        Merged DataFrame
    """
    # Ensure data is sorted
    df_tickets = df_tickets.sort_values(by_cols + [left_on])
    df_panel = df_panel.sort_values(by_cols + [right_on])
    
    # Get groups
    ticket_groups = df_tickets.groupby(by_cols)
    panel_groups = df_panel.groupby(by_cols)
    
    merged_groups = []
    
    for name, ticket_group in ticket_groups:
        if name in panel_groups.groups:
            panel_group = panel_groups.get_group(name)
            
            merged = pd.merge_asof(
                ticket_group,
                panel_group,
                left_on=left_on,
                right_on=right_on,
                direction=direction,
                suffixes=suffixes
            )
            merged_groups.append(merged)
        else:
            # No panel data for this group - keep tickets with NaN panel features
            merged_groups.append(ticket_group)
    
    result = pd.concat(merged_groups, ignore_index=True)
    return result


def compute_time_based_rolling_counts(df: pd.DataFrame,
                                      timestamp_col: str,
                                      group_cols: List[str],
                                      window_hours: List[int]) -> pd.DataFrame:
    """
    Compute rolling counts based on time windows (for intake pressure features).
    
    Args:
        df: DataFrame with timestamps
        timestamp_col: Timestamp column name
        group_cols: Columns to group by
        window_hours: List of window sizes in hours
    
    Returns:
        DataFrame with rolling count features added
    """
    df = df.sort_values(group_cols + [timestamp_col]).copy()
    
    for window in window_hours:
        col_name = f'intake_{window}h'
        
        # For each group, compute rolling count
        def rolling_count(group):
            counts = []
            for idx, row in group.iterrows():
                current_time = row[timestamp_col]
                cutoff_time = current_time - pd.Timedelta(hours=window)
                
                # Count rows in window
                mask = (group[timestamp_col] >= cutoff_time) & \
                       (group[timestamp_col] < current_time)
                counts.append(mask.sum())
            
            group[col_name] = counts
            return group
        
        df = df.groupby(group_cols, group_keys=False).apply(rolling_count)
    
    return df


def add_weather_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived weather features.
    
    Args:
        df: DataFrame with base weather columns (tavg, tmax, tmin, prcp)
    
    Returns:
        DataFrame with derived weather features added
    """
    result = df.copy()
    
    # Heating and cooling degree days
    if 'tavg' in result.columns:
        result['heating_degree'] = np.maximum(0, 65 - result['tavg'].fillna(65))
        result['cooling_degree'] = np.maximum(0, result['tavg'].fillna(65) - 65)
    else:
        result['heating_degree'] = 0
        result['cooling_degree'] = 0
    
    # Temperature extremes
    if 'tmax' in result.columns:
        result['heat_flag'] = (result['tmax'].fillna(0) >= 90).astype(int)
    else:
        result['heat_flag'] = 0
    
    if 'tmin' in result.columns:
        result['freeze_flag'] = (result['tmin'].fillna(40) <= 32).astype(int)
    else:
        result['freeze_flag'] = 0
    
    return result


def compute_rain_rolling(df: pd.DataFrame,
                        group_col: str = 'fips',
                        date_col: str = 'day',
                        prcp_col: str = 'prcp',
                        windows: List[int] = [3, 7]) -> pd.DataFrame:
    """
    Compute rolling precipitation sums per group.
    
    Args:
        df: DataFrame with precipitation data
        group_col: Column to group by (e.g., 'fips')
        date_col: Date column
        prcp_col: Precipitation column
        windows: List of window sizes in days
    
    Returns:
        DataFrame with rain rolling features added
    """
    df = df.sort_values([group_col, date_col]).copy()
    
    for window in windows:
        col_name = f'rain_{window}d'
        
        # Use pandas rolling with time-based window
        df[col_name] = df.groupby(group_col)[prcp_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )
    
    return df

