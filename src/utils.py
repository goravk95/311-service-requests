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
