"""
Example usage of the preprocessing module for NYC 311 DOHMH data.

This script demonstrates how to use the preprocessing functions
to clean and merge external data.
"""

import os
import sys
import polars as pl
import pandas as pd
from pathlib import Path

# Add src to path
INGESTION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, INGESTION_PATH)

from ingestion import config
from model import preprocessing


def example_basic_preprocessing():
    """
    Example 1: Basic preprocessing without external data merging.
    """
    print("Example 1: Basic Preprocessing")
    print("-" * 80)
    
    # Load data from parquet files
    data_path = Path("../../data/landing/311-service-requests")
    lf = pl.scan_parquet(
        str(data_path / "**/*.parquet"),
        hive_partitioning=True,
    )
    
    # Filter for DOHMH agency and select relevant columns
    dohmh_data = lf.filter(pl.col("agency") == "DOHMH").select([
        "due_date",
        "facility_type",
        "cross_street_2",
        "cross_street_1",
        "bbl",
        "location_type",
        "street_name",
        "incident_address",
        "address_type",
        "longitude",
        "latitude",
        "city",
        "incident_zip",
        "resolution_description",
        "closed_date",
        "resolution_action_updated_date",
        "descriptor",
        "community_board",
        "borough",
        "created_date",
        "agency",
        "complaint_type",
        "status",
        "open_data_channel_type",
        "year",
        "month"
    ]).collect()
    
    df = dohmh_data.to_pandas()
    
    # Apply preprocessing pipeline
    df_clean = preprocessing.preprocess_dohmh_data(df)
    
    print(f"\nFinal dataset shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")
    
    return df_clean


def example_full_pipeline():
    """
    Example 2: Full pipeline with external data merging using default config paths.
    """
    print("\n\nExample 2: Full Pipeline with External Data")
    print("-" * 80)
    
    # Load data from parquet files
    data_path = Path("../../data/landing/311-service-requests")
    lf = pl.scan_parquet(
        str(data_path / "**/*.parquet"),
        hive_partitioning=True,
    )
    
    # Filter for DOHMH agency
    dohmh_data = lf.filter(pl.col("agency") == "DOHMH").select([
        "due_date",
        "facility_type",
        "cross_street_2",
        "cross_street_1",
        "bbl",
        "location_type",
        "street_name",
        "incident_address",
        "address_type",
        "longitude",
        "latitude",
        "city",
        "incident_zip",
        "resolution_description",
        "closed_date",
        "resolution_action_updated_date",
        "descriptor",
        "community_board",
        "borough",
        "created_date",
        "agency",
        "complaint_type",
        "status",
        "open_data_channel_type",
        "year",
        "month"
    ]).collect()
    
    df = dohmh_data.to_pandas()
    
    # Apply full pipeline - paths from config are used by default
    df_complete = preprocessing.preprocess_and_merge_external_data(df)
    
    print(f"\nFinal dataset shape: {df_complete.shape}")
    print(f"\nSample of merged columns:")
    print(f"  - GEOID: {df_complete['GEOID'].notna().sum():,} non-null values")
    print(f"  - Temperature (tmax): {df_complete['tmax'].notna().sum():,} non-null values")
    print(f"  - Precipitation: {df_complete['prcp'].notna().sum():,} non-null values")
    
    return df_complete


def example_step_by_step():
    """
    Example 3: Step-by-step processing for more control.
    """
    print("\n\nExample 3: Step-by-Step Processing")
    print("-" * 80)
    
    # Load data
    data_path = Path("../../data/landing/311-service-requests")
    lf = pl.scan_parquet(
        str(data_path / "**/*.parquet"),
        hive_partitioning=True,
    )
    
    dohmh_data = lf.filter(pl.col("agency") == "DOHMH").select([
        "due_date",
        "facility_type",
        "cross_street_2",
        "cross_street_1",
        "bbl",
        "location_type",
        "street_name",
        "incident_address",
        "address_type",
        "longitude",
        "latitude",
        "city",
        "incident_zip",
        "resolution_description",
        "closed_date",
        "resolution_action_updated_date",
        "descriptor",
        "community_board",
        "borough",
        "created_date",
        "agency",
        "complaint_type",
        "status",
        "open_data_channel_type",
        "year",
        "month"
    ]).collect()
    
    df = dohmh_data.to_pandas()
    
    # Step 1: Create date features
    df = preprocessing.create_date_features(df)
    
    # Step 2: Map freetext columns
    df = preprocessing.map_freetext_columns(df)
    
    # Step 3: Inspect complaint families before filtering
    print("\nComplaint family distribution:")
    print(df['complaint_family'].value_counts())
    
    # Step 4: Filter and clean (removes duplicates, invalid dates, unwanted families)
    df = preprocessing.filter_and_clean(df)
    
    print(f"\nFinal dataset shape: {df.shape}")
    
    return df


def example_custom_paths():
    """
    Example 4: Full pipeline with custom paths (override defaults).
    """
    print("\n\nExample 4: Full Pipeline with Custom Paths")
    print("-" * 80)
    
    # Load data
    data_path = Path("../../data/landing/311-service-requests")
    lf = pl.scan_parquet(
        str(data_path / "**/*.parquet"),
        hive_partitioning=True,
    )
    
    dohmh_data = lf.filter(pl.col("agency") == "DOHMH").select([
        "due_date",
        "facility_type",
        "cross_street_2",
        "cross_street_1",
        "bbl",
        "location_type",
        "street_name",
        "incident_address",
        "address_type",
        "longitude",
        "latitude",
        "city",
        "incident_zip",
        "resolution_description",
        "closed_date",
        "resolution_action_updated_date",
        "descriptor",
        "community_board",
        "borough",
        "created_date",
        "agency",
        "complaint_type",
        "status",
        "open_data_channel_type",
        "year",
        "month"
    ]).collect()
    
    df = dohmh_data.to_pandas()
    
    # Define custom paths (override defaults)
    custom_census_path = "path/to/custom/census_data.csv"
    custom_shapefile_path = "path/to/custom/shapefile"
    custom_weather_path = "path/to/custom/weather_data.csv"
    
    # Apply full pipeline with custom paths
    # df_complete = preprocessing.preprocess_and_merge_external_data(
    #     df,
    #     census_data_path=custom_census_path,
    #     shapefile_path=custom_shapefile_path,
    #     weather_data_path=custom_weather_path
    # )
    
    print("This example shows how to override default config paths")
    print("Uncomment the lines above to use custom paths")
    
    return None


if __name__ == "__main__":
    # Run example 1 (basic preprocessing)
    df_clean = example_basic_preprocessing()
    
    # Uncomment to run other examples:
    # df_complete = example_full_pipeline()
    # df_step = example_step_by_step()
    # example_custom_paths()

