"""
NYC 311 Service Requests Analysis Package

This package provides tools for fetching, preprocessing, and analyzing NYC 311 service request data,
with a focus on Department of Health and Mental Hygiene (DOHMH) complaints.

Modules:
--------
ingestion_config : Configuration for data fetching
model_config : Configuration for preprocessing and modeling
fetch : Functions to fetch data from Socrata API and external sources
preprocessing : Data cleaning and feature engineering
plotting : Visualization functions
"""

# Import config
from . import config

# Import fetch functions
from .fetch import (
    fetch_all_service_requests,
    fetch_current_month_service_requests,
    fetch_acs_census_population_data,
    fetch_noaa_weather_data,
)

# Import preprocessing functions
from .preprocessing import (
    load_dohmh_data,
    create_date_features,
    map_freetext_columns,
    filter_and_clean,
    merge_census_data,
    merge_weather_data,
    preprocess_dohmh_data,
    preprocess_and_merge_external_data,
)

# Import plotting functions
from .plotting import create_hexbin_density_map

__all__ = [
    # Config
    'config',
    # Fetch functions
    'fetch_all_service_requests',
    'fetch_current_month_service_requests',
    'fetch_acs_census_population_data',
    'fetch_noaa_weather_data',
    # Preprocessing functions
    'load_dohmh_data',
    'create_date_features',
    'map_freetext_columns',
    'filter_and_clean',
    'merge_census_data',
    'merge_weather_data',
    'preprocess_dohmh_data',
    'preprocess_and_merge_external_data',
    # Plotting functions
    'create_hexbin_density_map',
]

__version__ = '0.1.0'

