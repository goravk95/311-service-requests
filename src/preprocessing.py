"""
Data Preprocessing Module for NYC 311 Service Requests (DOHMH)

This module contains functions for cleaning and preprocessing NYC 311 service request data,
including deduplication, feature engineering, freetext mapping, and external data merging.
"""

from datetime import date
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Optional

from . import config


def load_dohmh_data() -> pd.DataFrame:
    """
    Load DOHMH service request data from S3/local parquet files.


    Returns
    -------
    pd.DataFrame
        DataFrame with DOHMH service request data

    """
    df = pd.read_parquet(config.LANDING_DATA_PATH)

    return df


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived date and time-based features.

    Creates the following features:

    Date/Time Components:
    - day: Date portion of created_date (no time component)
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

    df["day"] = df["created_date"].dt.date
    df["created_date_hour"] = df["created_date"].dt.floor("h")

    df["time_to_resolution"] = df["closed_date"] - df["created_date"]
    df["time_closed_to_resolution_update"] = (
        df["resolution_action_updated_date"] - df["closed_date"]
    )

    df["time_to_resolution_hours"] = df["time_to_resolution"].dt.total_seconds() / 3600
    df["time_to_resolution_days"] = df["time_to_resolution_hours"] / 24
    df["time_closed_to_resolution_update_hours"] = (
        df["time_closed_to_resolution_update"].dt.total_seconds() / 3600
    )
    df["time_closed_to_resolution_update_days"] = df["time_closed_to_resolution_update_hours"] / 24

    df["is_closed_before_created"] = df["time_to_resolution"] < pd.Timedelta(0)
    df["is_identical_created_closed"] = df["time_to_resolution"] == pd.Timedelta(0)
    df["is_created_at_midnight"] = df["created_date"].dt.time == pd.Timestamp("00:00:00").time()
    df["is_closed_at_midnight"] = df["closed_date"].dt.time == pd.Timestamp("00:00:00").time()

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

    df_complaint_type = dict_mappings["complaint_type"]
    df_descriptor = dict_mappings["descriptor"]
    df_resolution_description = dict_mappings["resolution_description"]

    # Merge mappings
    df = df.merge(df_complaint_type, on="complaint_type", how="left")
    df = df.merge(df_descriptor, on="descriptor", how="left")
    df = df.merge(df_resolution_description, on="resolution_description", how="left")

    # Fix specific resolution outcome edge case
    broken_string = "this case was an isolated incident"
    df["resolution_outcome"] = np.where(
        df["resolution_description"].str.contains(broken_string, case=False, na=False),
        "inspection",
        df["resolution_outcome"],
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
    df = df.drop_duplicates(keep="first")

    df = df.drop_duplicates(
        subset=[
            "created_date_hour",
            "complaint_type",
            "incident_address",
            "borough",
            "descriptor",
            "resolution_description",
        ]
    )

    # df = df[df["resolution_outcome"] != "duplicate_of_previous"]

    # df = df[df['complaint_family'].isin(config.COMPLAINT_FAMILIES)]

    df = df[df["is_closed_before_created"] == False]
    # df = df[df["is_identical_created_closed"] == False]

    df = df[df['day'] >= date(2010, 1, 1)]

    return df


def extend_population_data(df_pop: pd.DataFrame) -> pd.DataFrame:
    """
    Extend population data by forward filling 2024-2025 and backfilling 2010-2012.

    Forward fill (2024-2025):
    - Calculates annual population change rates based on 2018-2023 data
    - Projects population for 2024 and 2025 using these rates

    Backfill (2010-2012):
    - Calculates annual population change rates based on 2013-2018 data
    - Projects population backwards for 2010, 2011, and 2012 using these rates

    For GEOIDs with missing or invalid growth rates, uses the overall
    population growth rate as fallback.

    Parameters
    ----------
    df_pop : pd.DataFrame
        Population dataframe with GEOID, year, and population columns

    Returns
    -------
    pd.DataFrame
        Population dataframe with 2010-2012 and 2024-2025 data appended
    """
    df_pop = df_pop.copy()

    # ========== FORWARD FILL: 2024 and 2025 ==========
    # Get the most recent population data (2023 and 2018) for each GEOID
    df_2023 = df_pop[df_pop["year"] == 2023][["GEOID", "population"]].rename(
        columns={"population": "pop_2023"}
    )
    df_2018 = df_pop[df_pop["year"] == 2018][["GEOID", "population"]].rename(
        columns={"population": "pop_2018"}
    )

    # Merge to calculate percent change (forward)
    df_change_forward = df_2018.merge(df_2023, on="GEOID", how="inner")
    df_change_forward["pct_change_5yr"] = (
        df_change_forward["pop_2023"] - df_change_forward["pop_2018"]
    ) / df_change_forward["pop_2018"]
    df_change_forward["annual_change"] = df_change_forward["pct_change_5yr"] / 5

    # Calculate overall population change rate for fallback (forward)
    overall_pop_2018 = df_change_forward["pop_2018"].sum()
    overall_pop_2023 = df_change_forward["pop_2023"].sum()
    overall_pct_change_5yr_forward = (overall_pop_2023 - overall_pop_2018) / overall_pop_2018
    overall_annual_change_forward = overall_pct_change_5yr_forward / 5

    # For values where annual_change is inf or nan, use the overall population change rate
    df_change_forward["annual_change"] = df_change_forward["annual_change"].fillna(
        overall_annual_change_forward
    )
    df_change_forward["annual_change"] = df_change_forward["annual_change"].replace(
        [float("inf"), float("-inf")], overall_annual_change_forward
    )

    # Calculate 2024 and 2025 populations
    df_change_forward["pop_2024"] = df_change_forward["pop_2023"] * (
        1 + df_change_forward["annual_change"] * 1
    )
    df_change_forward["pop_2025"] = df_change_forward["pop_2023"] * (
        1 + df_change_forward["annual_change"] * 2
    )

    # Round to integers
    df_change_forward["pop_2024"] = df_change_forward["pop_2024"].round()
    df_change_forward["pop_2025"] = df_change_forward["pop_2025"].round()

    # Create new rows for 2024 and 2025
    df_2024 = df_change_forward[["GEOID", "pop_2024"]].rename(columns={"pop_2024": "population"})
    df_2024["year"] = 2024

    df_2025 = df_change_forward[["GEOID", "pop_2025"]].rename(columns={"pop_2025": "population"})
    df_2025["year"] = 2025

    # ========== BACKFILL: 2010, 2011, and 2012 ==========
    # Get population data for 2013 and 2018 for each GEOID
    df_2013 = df_pop[df_pop["year"] == 2013][["GEOID", "population"]].rename(
        columns={"population": "pop_2013"}
    )
    # Reuse df_2018 from above

    # Merge to calculate percent change (backward)
    df_change_backward = df_2013.merge(df_2018, on="GEOID", how="inner")
    df_change_backward["pct_change_5yr"] = (
        df_change_backward["pop_2018"] - df_change_backward["pop_2013"]
    ) / df_change_backward["pop_2013"]
    df_change_backward["annual_change"] = df_change_backward["pct_change_5yr"] / 5

    # Calculate overall population change rate for fallback (backward)
    overall_pop_2013 = df_change_backward["pop_2013"].sum()
    overall_pop_2018_back = df_change_backward["pop_2018"].sum()
    overall_pct_change_5yr_backward = (overall_pop_2018_back - overall_pop_2013) / overall_pop_2013
    overall_annual_change_backward = overall_pct_change_5yr_backward / 5

    # For values where annual_change is inf or nan, use the overall population change rate
    df_change_backward["annual_change"] = df_change_backward["annual_change"].fillna(
        overall_annual_change_backward
    )
    df_change_backward["annual_change"] = df_change_backward["annual_change"].replace(
        [float("inf"), float("-inf")], overall_annual_change_backward
    )

    # Calculate 2010, 2011, and 2012 populations (working backwards from 2013)
    df_change_backward["pop_2012"] = df_change_backward["pop_2013"] * (
        1 + df_change_backward["annual_change"] * (-1)
    )
    df_change_backward["pop_2011"] = df_change_backward["pop_2013"] * (
        1 + df_change_backward["annual_change"] * (-2)
    )
    df_change_backward["pop_2010"] = df_change_backward["pop_2013"] * (
        1 + df_change_backward["annual_change"] * (-3)
    )

    # Round to integers
    df_change_backward["pop_2012"] = df_change_backward["pop_2012"].round()
    df_change_backward["pop_2011"] = df_change_backward["pop_2011"].round()
    df_change_backward["pop_2010"] = df_change_backward["pop_2010"].round()

    # Create new rows for 2010, 2011, and 2012
    df_2012 = df_change_backward[["GEOID", "pop_2012"]].rename(columns={"pop_2012": "population"})
    df_2012["year"] = 2012

    df_2011 = df_change_backward[["GEOID", "pop_2011"]].rename(columns={"pop_2011": "population"})
    df_2011["year"] = 2011

    df_2010 = df_change_backward[["GEOID", "pop_2010"]].rename(columns={"pop_2010": "population"})
    df_2010["year"] = 2010

    # Concatenate with original data
    df_extended = pd.concat(
        [df_pop, df_2010, df_2011, df_2012, df_2024, df_2025], ignore_index=True
    )

    return df_extended


def merge_census_data(df: pd.DataFrame, census_data_path: str, shapefile_path: str) -> pd.DataFrame:
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
    df_pop["GEOID"] = df_pop["GEOID"].astype(str)

    # Extend population data for 2010-2012 and 2024-2025
    df_pop = extend_population_data(df_pop)

    # Load census block group shapefile
    gdf_bg = gpd.read_file(shapefile_path)
    gdf_bg = gdf_bg.to_crs("EPSG:4326")  # Ensure consistent CRS

    # Convert service requests to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf_orig = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Spatial join with block groups
    gdf_orig_bg = gpd.sjoin(gdf_orig, gdf_bg[["GEOID", "geometry"]], how="left")

    # Merge population data
    df_merged = gdf_orig_bg.merge(df_pop, on=["GEOID", "year"], how="left")

    return df_merged


def convert_weather_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert weather data from metric to US units.

    Conversions:
    - Temperature: Celsius to Fahrenheit (°F = °C × 9/5 + 32)
    - Precipitation: mm to inches (inches = mm / 25.4)

    NOAA nClimGrid data comes in:
    - Temperature: Celsius
    - Precipitation: Tenths of millimeters

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with weather columns in metric units (tavg, tmax, tmin, prcp)

    Returns
    -------
    pd.DataFrame
        DataFrame with weather in US units (Fahrenheit and inches)
    """
    result = df.copy()

    # Convert temperatures from Celsius to Fahrenheit
    for temp_col in ["tavg", "tmax", "tmin"]:
        if temp_col in result.columns:
            result[temp_col] = (result[temp_col] * 9 / 5) + 32

    # Convert precipitation from tenths of mm to inches
    if "prcp" in result.columns:
        result["prcp"] = result["prcp"] / 25.4

    return result


def add_weather_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived weather features from base weather columns.

    Assumes temperatures are already in Fahrenheit and precipitation in inches.

    Calculates:
    - Heating/cooling degree days (base 65°F)
    - Temperature extreme flags (heat ≥90°F, freeze ≤32°F)
    - Rolling precipitation (3-day, 7-day in inches)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with base weather columns in US units (tavg, tmax, tmin in °F, prcp in inches)

    Returns
    -------
    pd.DataFrame
        DataFrame with all derived weather features added
    """
    result = df.copy()

    result["heating_degree"] = np.maximum(0, 65 - result["tavg"].fillna(65))
    result["cooling_degree"] = np.maximum(0, result["tavg"].fillna(65) - 65)
    result["heat_flag"] = (result["tmax"].fillna(0) >= 90).astype(int)
    result["freeze_flag"] = (result["tmin"].fillna(40) <= 32).astype(int)

    # Rolling precipitation (only if fips and prcp exist)
    if "fips" in result.columns and "prcp" in result.columns:
        # Ensure proper sorting (handle both 'date' and 'day')
        date_col = "date" if "date" in result.columns else "day"
        if date_col in result.columns:
            result = result.sort_values(["fips", date_col])

            # Compute rolling sums per FIPS
            for window in [3, 7]:
                col_name = f"rain_{window}d"
                result[col_name] = result.groupby("fips")["prcp"].transform(
                    lambda x: x.rolling(window=window, min_periods=1).sum()
                )

    return result


def merge_weather_data(df: pd.DataFrame, weather_data_path: str) -> pd.DataFrame:
    """
    Merge weather data with all derived features pre-computed.

    Loads weather data, converts units to US standard, computes all derived features
    on the weather DataFrame (which is much smaller), then merges everything at once:
    - Unit conversion: Celsius to Fahrenheit, tenths of mm to inches
    - Base weather: tavg, tmax, tmin (°F), prcp (inches)
    - Derived: heating_degree, cooling_degree, heat_flag, freeze_flag
    - Rolling: rain_3d, rain_7d (inches)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with GEOID and day columns
    weather_data_path : str
        Path to CSV file with weather data in metric units
        (must have fips, date, tmax, tmin, tavg in Celsius, prcp in tenths of mm)

    Returns
    -------
    pd.DataFrame
        Dataframe with all weather features merged in US units
    """
    # Load and prepare weather data
    df_weather = pd.read_csv(weather_data_path)
    df_weather["fips"] = df_weather["fips"].astype(str)
    df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.date

    # Convert units from metric to US (Celsius to Fahrenheit, tenths of mm to inches)
    df_weather = convert_weather_units(df_weather)

    # Derived features (degree days, extreme flags, rolling precipitation)
    df_weather = add_weather_derived_features(df_weather)

    # Extract FIPS from GEOID (first 5 characters)
    df["fips"] = df["GEOID"].apply(lambda x: str(x)[:5] if pd.notna(x) else None)

    # Merge all weather features at once
    weather_cols = [
        "fips",
        "date",
        "tmax",
        "tmin",
        "tavg",
        "prcp",
        "heating_degree",
        "cooling_degree",
        "heat_flag",
        "freeze_flag",
        "rain_3d",
        "rain_7d",
    ]

    df_merged = df.merge(
        df_weather[weather_cols],
        left_on=["fips", "day"],
        right_on=["fips", "date"],
        how="inner",
    )
    df_merged = df_merged.drop(columns=["date"])

    return df_merged


def preprocess_dohmh_data(df: pd.DataFrame, mappings_path: Optional[str] = None) -> pd.DataFrame:
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


def preprocess_and_merge_external_data() -> pd.DataFrame:
    """
    Full pipeline: load, preprocess, and merge external datasets.

    Loads DOHMH data from config.LANDING_DATA_PATH, preprocesses it,
    and merges census and weather data.

    Weather data is automatically converted from metric to US units:
    - Temperature: Celsius to Fahrenheit
    - Precipitation: Tenths of millimeters to inches

    Parameters
    ----------

    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataframe with external data merged
        Weather features are in US units (°F and inches)
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

    print()
    print("Final Data Shape:", df.shape)

    return df


def save_preprocessed_data(df: pd.DataFrame) -> None:
    """
    Save the preprocessed data to S3, partitioned by year.
    """
    partition_column = "year"

    for year_value in df[partition_column].unique():
        partition_df = df[df[partition_column] == year_value]
        partition_df.to_parquet(config.CURATION_DATA_PATH+ f"/{partition_column}={year_value}/part-0000.parquet", index=False, compression='snappy')
        