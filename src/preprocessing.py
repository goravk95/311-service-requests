"""
Data Preprocessing Module for NYC 311 Service Requests (DOHMH)

This module contains functions for cleaning and preprocessing NYC 311 service request data,
including deduplication, feature engineering, freetext mapping, and external data merging.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from typing import Optional, Tuple, List

from . import config


def load_dohmh_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load DOHMH service request data from S3/local parquet files.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the parquet data directory. If None, uses config.SERVICE_REQUESTS_DATA_PATH
        
    Returns
    -------
    pd.DataFrame
        DataFrame with DOHMH service request data

    """
    if data_path is None:
        data_path = config.SERVICE_REQUESTS_DATA_PATH
    
    df = pd.read_parquet(data_path)
    
    return df


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived date and time-based features.
    
    Creates the following features:
    
    Date/Time Components:
    - created_date_date: Date portion of created_date (no time component)
    - created_date_hour: Created date rounded to hour (for deduplication)
    
    Duration Features (Timedelta):
    - time_to_resolution: Duration between created_date and closed_date
    - time_closed_to_resolution_update: Duration between closed_date and resolution_action_updated_date
    
    Duration Features (Hours as float):
    - time_to_resolution_hours: Resolution time in hours
    - time_closed_to_resolution_update_hours: Update time in hours
    
    Duration Features (Days as float):
    - time_to_resolution_days: Resolution time in days (computed from hours/24)
    - time_closed_to_resolution_update_days: Update time in days (computed from hours/24)
    
    Validation Flags (Boolean):
    - is_closed_before_created: True if closed_date < created_date (invalid)
    - is_identical_created_closed: True if created_date == closed_date (invalid)
    - is_created_at_midnight: True if created_date time is 00:00:00
    - is_closed_at_midnight: True if closed_date time is 00:00:00
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with created_date, closed_date, and resolution_action_updated_date columns
        
    Returns
    -------
    pd.DataFrame
        Dataframe with new date features added (12 total features)
    """
    df = df.copy()

    df['created_date_date'] = df['created_date'].dt.date
    df['created_date_hour'] = df['created_date'].dt.floor('h')
    
    df['time_to_resolution'] = df['closed_date'] - df['created_date']
    df['time_closed_to_resolution_update'] = df['resolution_action_updated_date'] - df['closed_date']
    
    df['time_to_resolution_hours'] = df['time_to_resolution'].dt.total_seconds() / 3600
    df['time_to_resolution_days'] = df['time_to_resolution_hours'] / 24
    df['time_closed_to_resolution_update_hours'] = df['time_closed_to_resolution_update'].dt.total_seconds() / 3600
    df['time_closed_to_resolution_update_days'] = df['time_closed_to_resolution_update_hours'] / 24
    
    df['is_closed_before_created'] = df['time_to_resolution'] < pd.Timedelta(0)
    df['is_identical_created_closed'] = df['time_to_resolution'] == pd.Timedelta(0)
    df['is_created_at_midnight'] = df['created_date'].dt.time == pd.Timestamp('00:00:00').time()
    df['is_closed_at_midnight'] = df['closed_date'].dt.time == pd.Timestamp('00:00:00').time()
    
    return df


def map_freetext_columns(df: pd.DataFrame, mappings_path: Optional[str] = None) -> pd.DataFrame:
    """
    Map freetext columns to standardized categories using Excel mappings.
    
    Maps the following columns:
    - complaint_type -> complaint_family (and other derived fields)
    - descriptor -> standardized descriptor categories
    - resolution_description -> resolution_outcome
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with freetext columns
    mappings_path : str, optional
        Path to the Excel file with mappings. If None, uses config.MAPPINGS_PATH
        
    Returns
    -------
    pd.DataFrame
        Dataframe with mapped columns added
    """
    df = df.copy()
    
    # Use default mappings path from config if not provided
    if mappings_path is None:
        mappings_path = str(config.MAPPINGS_PATH)
    
    # Load all sheets from the Excel file
    dict_mappings = pd.read_excel(mappings_path, sheet_name=None)
    
    df_complaint_type = dict_mappings['complaint_type']
    df_descriptor = dict_mappings['descriptor']
    df_resolution_description = dict_mappings['resolution_description']
    
    # Merge mappings
    df = df.merge(df_complaint_type, on='complaint_type', how='left')
    df = df.merge(df_descriptor, on='descriptor', how='left')
    df = df.merge(df_resolution_description, on='resolution_description', how='left')
    
    # Fix specific resolution outcome edge case
    broken_string = 'this case was an isolated incident'
    df['resolution_outcome'] = np.where(
        df['resolution_description'].str.contains(broken_string, case=False, na=False),
        "inspection",
        df['resolution_outcome']
    )
    
    return df


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and clean the dataset by removing invalid and unwanted records.
    
    Performs the following operations:
    1. Remove exact duplicate records
    2. Remove duplicates based on key fields (created_date_hour, complaint_type, 
       incident_address, borough, descriptor, resolution_description)
    3. Filter out records with is_closed_before_created dates
    4. Filter out records with is_identical_created_closed timestamps
    5. Filter to specified complaint families (vector_control, food_safety, 
       air_smoke_mold, animal_control)
    6. Remove records marked as duplicate_of_previous resolutions
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with service request data. Must have date features already created
        and freetext columns already mapped.
        
    Returns
    -------
    pd.DataFrame
        Filtered and cleaned dataframe
    """
    df = df.drop_duplicates(keep='first')
    
    df = df.drop_duplicates(
        subset=[
            "created_date_hour",
            "complaint_type",
            "incident_address",
            "borough",
            "descriptor",
            "resolution_description"
        ]
    )
 
    df = df[df['resolution_outcome'] != 'duplicate_of_previous']
 
    df = df[df['complaint_family'].isin(config.COMPLAINT_FAMILIES)]
    
    df = df[df['is_closed_before_created'] == False]
    df = df[df['is_identical_created_closed'] == False]
    
    return df


def merge_census_data(
    df: pd.DataFrame,
    census_data_path: str,
    shapefile_path: str
) -> pd.DataFrame:
    """
    Perform spatial join with census block groups and merge population data.
    
    Uses latitude/longitude to spatially join service requests with census
    block groups, then merges in population data by GEOID and year.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with latitude, longitude, and year columns
    census_data_path : str
        Path to CSV file with population data (must have GEOID and year columns)
    shapefile_path : str
        Path to directory containing census block group shapefiles
        
    Returns
    -------
    pd.DataFrame
        Dataframe with census data merged (includes GEOID and population fields)
    """
    # Load population data
    df_pop = pd.read_csv(census_data_path)
    df_pop['GEOID'] = df_pop['GEOID'].astype(str)
    
    # Load census block group shapefile
    gdf_bg = gpd.read_file(shapefile_path)
    gdf_bg = gdf_bg.to_crs("EPSG:4326")  # Ensure consistent CRS
    
    # Convert service requests to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf_orig = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Spatial join with block groups
    gdf_orig_bg = gpd.sjoin(
        gdf_orig,
        gdf_bg[['GEOID', 'geometry']],
        how="left"
    )
    
    # Merge population data
    df_merged = gdf_orig_bg.merge(
        df_pop,
        on=['GEOID', 'year'],
        how='left'
    )
    
    return df_merged


def merge_weather_data(
    df: pd.DataFrame,
    weather_data_path: str
) -> pd.DataFrame:
    """
    Merge weather data by FIPS code and date.
    
    Extracts FIPS code from GEOID (first 5 characters) and merges weather data
    (temperature and precipitation) by FIPS and date.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with GEOID and date columns
    weather_data_path : str
        Path to CSV file with weather data (must have fips, date, tmax, tmin, tavg, prcp columns)
        
    Returns
    -------
    pd.DataFrame
        Dataframe with weather data merged
    """
    df_weather = pd.read_csv(weather_data_path)
    df_weather['fips'] = df_weather['fips'].astype(str)
    df_weather['date'] = pd.to_datetime(df_weather['date']).dt.date
    
    # Extract FIPS from GEOID (first 5 characters)
    df['fips'] = df['GEOID'].apply(lambda x: str(x)[:5] if pd.notna(x) else None)
    
    # Merge weather data
    df_merged = df.merge(
        df_weather[['fips', 'date', 'tmax', 'tmin', 'tavg', 'prcp']],
        left_on=['fips', 'created_date_date'],
        right_on=['fips', 'date'],
        how='left'
    )
    
    return df_merged


def preprocess_dohmh_data(
    df: pd.DataFrame,
    mappings_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Main preprocessing pipeline for DOHMH service request data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw DOHMH service request data
    mappings_path : str, optional
        Path to Excel file with freetext mappings
        
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataframe
    """
    df = create_date_features(df)
    df = map_freetext_columns(df, mappings_path)
    df = filter_and_clean(df)
    
    return df


def preprocess_and_merge_external_data(
) -> pd.DataFrame:
    """
    Full pipeline: load, preprocess, and merge external datasets.
    
    Loads DOHMH data from config.SERVICE_REQUESTS_DATA_PATH, preprocesses it,
    and merges census and weather data.
    
    Parameters
    ----------
        
    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataframe with external data merged
    """
    print("Loading DOHMH data...")
    df = load_dohmh_data()
    print("Data Shape:", df.shape)

    census_data_path = str(config.CENSUS_DATA_PATH)
    shapefile_path = str(config.SHAPEFILE_PATH)
    weather_data_path = str(config.WEATHER_DATA_PATH)
    mappings_path = str(config.MAPPINGS_PATH)

    print("Preprocessing DOHMH data...")
    df = preprocess_dohmh_data(df, mappings_path=mappings_path)
    print("Data Shape:", df.shape)
    
    print("Merging census data...")
    df = merge_census_data(df, census_data_path, shapefile_path)
    print("Data Shape:", df.shape)

    print("Merging weather data...")
    df = merge_weather_data(df, weather_data_path)
    print("Data Shape:", df.shape)

    print("Final Data Shape:", df.shape)

    return df

