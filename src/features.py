"""
Feature engineering for NYC 311 service requests.
Implements leakage-safe feature builders for Forecast, Triage, and Duration models.
"""

import numpy as np
import pandas as pd
import h3
from typing import List, Optional
from datetime import datetime
from . import config

def aggregate_on_parent(
    df_panel: pd.DataFrame,
    res: int = 7,
    hex_col: str = "hex8",
    agg_cols: List[str] = ["roll4", "roll12"],
    time_col: str = "week",
) -> pd.DataFrame:
    """
    Expand a panel to include neighbor aggregations.
    For each hex, compute neighbor sums of specified columns.

    Optimized vectorized implementation - avoids O(N²) row-by-row iteration.

    Args:
        df_panel: DataFrame with hex column and metrics
        res: Resolution for parent hex
        hex_col: Name of hex column
        agg_cols: Columns to aggregate from neighbors
        time_col: Name of time column (week or day)

    Returns:
        DataFrame with original data plus neighbor aggregates (prefixed with 'nbr_')
    """
    df_panel[f"hex{res}"] = df_panel[hex_col].apply(lambda x: h3.cell_to_parent(x, res))
    for col in agg_cols:
        df_panel[f"nbr_{col}"] = df_panel.groupby([time_col, "complaint_family", f"hex{res}"])[
            col
        ].transform("sum")
    return df_panel


def add_history_features(
    group_df: pd.DataFrame,
    date_col: str = "week",
    value_col: str = "y",
    lags: List[int] = [1, 4],
    windows: List[int] = [4, 12],
) -> pd.DataFrame:
    """
    Calculate group history features.
    """
    hex_val = group_df["hex8"].iloc[0]
    complaint_family_val = group_df["complaint_family"].iloc[0]

    min_period = pd.Period(group_df[date_col].min(), freq="W-MON")
    max_period = pd.Period(group_df[date_col].max(), freq="W-MON")

    # Generate all periods between min and max
    period_range = pd.period_range(start=min_period, end=max_period, freq="W-MON")
    date_range = period_range.to_timestamp()

    # Create complete panel with all dates
    complete_panel = pd.DataFrame(
        {"hex8": hex_val, "complaint_family": complaint_family_val, date_col: date_range}
    )

    # Merge with original data to fill in y values
    group_df = complete_panel.merge(
        group_df[["hex8", "complaint_family", date_col, value_col]],
        on=["hex8", "complaint_family", date_col],
        how="left",
    ).fillna({"y": 0})

    for lag in lags:
        col_name = f"lag{lag}"
        group_df[col_name] = group_df[value_col].shift(lag)

    for window in windows:
        col_name = f"roll{window}"
        group_df[col_name] = group_df[value_col].rolling(window).sum()

    group_df = group_df[group_df[value_col] != 0]
    group_df["weeks_since_last"] = group_df[date_col].diff().dt.days / 7

    return group_df


def build_forecast_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build sparse forecast panel with time-series features.

    Grain: one row per (hex, complaint_family, week) with target y = count(tickets).
    Includes lag/rolling features, neighbor aggregates, weather, and population.

    Weather features (tavg, tmax, tmin, prcp, heating_degree, cooling_degree,
    rain_3d, rain_7d, heat_flag, freeze_flag) should already be present in df
    from preprocessing (via preprocess_and_merge_external_data).

    Args:
        df: Input DataFrame with H3 keys and weather features already added

    Returns:
        Sparse panel DataFrame with forecast features
    """
    # Filter to valid hex and complaint_family
    df_valid = df.copy()
    df_valid["hex8"] = df_valid.apply(
        lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], 8), axis=1
    )
    # Aggregate to weekly level (Monday start)
    df_valid["week"] = df_valid["created_date"].dt.to_period("W-MON").dt.to_timestamp()
    df_valid = df_valid[df_valid["hex8"].notna() & df_valid["complaint_family"].notna()].copy()

    # Create base panel: aggregate by hex, complaint_family, week
    panel = (
        df_valid.groupby(["hex8", "complaint_family", "week"])
        .agg(y=("hex8", "size"))  # Count tickets
        .reset_index()
    )

    panel = panel.sort_values("week")

    panel = panel.groupby(["hex8", "complaint_family"], group_keys=False).apply(
        lambda group: add_history_features(
            group, date_col="week", value_col="y", lags=[1, 4], windows=[4, 12]
        )
    )
    panel = panel.fillna(0)
    panel["momentum"] = panel["roll4"] / (panel["roll12"] + 1e-6)

    # Week-based temporal features
    panel["week_of_year"] = panel["week"].dt.isocalendar().week
    panel["month"] = panel["week"].dt.month
    panel["quarter"] = panel["week"].dt.quarter
    panel["hex6"] = panel["hex8"].apply(lambda x: h3.cell_to_parent(x, 6))

    panel = aggregate_on_parent(panel, res=7, hex_col="hex8", agg_cols=["roll4", "roll12"])

    weather_cols = [
        "tavg",
        "tmax",
        "tmin",
        "prcp",
        "heating_degree",
        "cooling_degree",
        "rain_3d",
        "rain_7d",
        "heat_flag",
        "freeze_flag",
    ]

    # Only aggregate weather columns that exist (average over the week)
    weather_agg = {}
    for col in weather_cols:
        weather_agg[col] = "mean"

    weather_from_df = (
        df_valid.groupby(["hex8", "complaint_family", "week"]).agg(weather_agg).reset_index()
    )

    # Merge with panel
    panel = panel.merge(weather_from_df, on=["hex8", "complaint_family", "week"], how="left")

    hex_pop_map = df_valid.groupby(["hex8", "GEOID"])["population"].first().reset_index()
    hex_pop_map = hex_pop_map.groupby("hex8")["population"].sum().reset_index()
    hex_pop_map.columns = ["hex8", "pop_hex"]

    panel = panel.merge(hex_pop_map, on="hex8", how="left")
    panel["pop_hex"] = panel["pop_hex"].fillna(0)

    panel["log_pop"] = np.log(np.maximum(panel["pop_hex"], 1))

    # Ensure all expected columns exist
    expected_cols = [
        "hex8",
        "hex6",
        "complaint_family",
        "week",
        "y",
        "week_of_year",
        "month",
        "quarter",
        "lag1",
        "lag4",
        "roll4",
        "roll12",
        "momentum",
        "weeks_since_last",
        "tmin",
        "tmax",
        "tavg",
        "prcp",
        "heating_degree",
        "cooling_degree",
        "heat_flag",
        "freeze_flag",
        "rain_3d",
        "rain_7d",
        "log_pop",
        "nbr_roll4",
        "nbr_roll12",
    ]

    return panel[expected_cols].reset_index(drop=True)

def save_forecast_panel_data(df: pd.DataFrame) -> None:
    """
    Save the preprocessed data to S3.
    """
    output_path = config.PRESENTATION_DATA_PATH + '/model_fitting_data.parquet'
    df.to_parquet(output_path, index=False, compression='snappy')

    output_path = config.PRESENTATION_DATA_PATH + '/streamlit_data.parquet'
    cutoff = pd.Timestamp('2025-01-01')
    mask_test = pd.to_datetime(df["week"]) >= cutoff
    df = df[mask_test]
    df.to_parquet(output_path, index=False, compression='snappy')

