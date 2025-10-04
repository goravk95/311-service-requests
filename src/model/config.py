"""
Configuration settings for the NYC 311 DOHMH preprocessing and modeling.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "landing"
RESOURCES_DIR = Path(__file__).parent.parent / "resources"

# Input data paths
PARQUET_DATA_PATH = DATA_DIR / "311-service-requests"
CENSUS_DATA_PATH = DATA_DIR / "acs-population" / "combined_population_data.csv"
WEATHER_DATA_PATH = DATA_DIR / "noaa-nclimgrid-daily" / "nyc_fips_weather_data.csv"
SHAPEFILE_PATH = RESOURCES_DIR / "tl_2022_36_bg"
MAPPINGS_PATH = RESOURCES_DIR / "mappings" / "freetext_column_mappings.xlsx"

# DOHMH columns to select from the 311 data
DOHDMH_COLUMNS = [
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
    "month",
]

# Complaint families to include in preprocessing
# These represent the top 4 families by volume, covering ~94% of all DOHMH complaints
COMPLAINT_FAMILIES = [
    'vector_control',    # 483,382 records (52.07%)
    'food_safety',       # 198,853 records (21.42%)
    'air_smoke_mold',    # 132,134 records (14.23%)
    'animal_control',    #  55,146 records (5.94%)
]

