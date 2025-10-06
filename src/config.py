"""Configuration settings for NYC 311 Service Requests analysis.

This module contains all configuration parameters for data fetching, preprocessing,
and modeling of NYC 311 service request data.
"""

import os
import pyarrow as pa
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "landing"
RESOURCES_DIR = Path(__file__).parent / "resources"

API_ENDPOINT = "data.cityofnewyork.us"
DATASET_ID = "erm2-nwe9"
DATE_COLUMN = "created_date"

APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN")
API_KEY_ID = os.getenv("SOCRATA_API_KEY_ID")
API_KEY_SECRET = os.getenv("SOCRATA_API_KEY_SECRET")

DATA_START_YEAR = 2010
DATA_END_YEAR = 2026

MAX_CONCURRENT_REQUESTS = 20
LANDING_DATA_PATH = "s3://hbc-technical-assessment-gk/landing/DOHMH"
CURATION_DATA_PATH = "s3://hbc-technical-assessment-gk/curation/DOHMH"
PRESENTATION_DATA_PATH = "s3://hbc-technical-assessment-gk/presentation/DOHMH"

CENSUS_DATA_PATH = DATA_DIR / "acs-population" / "combined_population_data.csv"
WEATHER_DATA_PATH = DATA_DIR / "noaa-nclimgrid-daily" / "nyc_fips_weather_data.csv"

SHAPEFILE_PATH = RESOURCES_DIR / "tl_2022_36_bg"
MAPPINGS_PATH = RESOURCES_DIR / "mappings" / "freetext_column_mappings.xlsx"

AGENCY = "DOHMH"
AGENCY_FILTER = f"agency = '{AGENCY}'"
DOHMH_COLUMNS = [
    "unique_key",
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
]

COMPLAINT_FAMILIES = [
    "vector_control",
    "food_safety",
    "air_smoke_mold",
    "animal_control",
]

SCHEMA = pa.schema(
    [
        ("unique_key", pa.string()),
        ("created_date", pa.timestamp("ms")),
        ("closed_date", pa.timestamp("ms")),
        ("agency", pa.string()),
        ("agency_name", pa.string()),
        ("complaint_type", pa.string()),
        ("descriptor", pa.string()),
        ("location_type", pa.string()),
        ("incident_zip", pa.string()),
        ("incident_address", pa.string()),
        ("street_name", pa.string()),
        ("cross_street_1", pa.string()),
        ("cross_street_2", pa.string()),
        ("intersection_street_1", pa.string()),
        ("intersection_street_2", pa.string()),
        ("address_type", pa.string()),
        ("city", pa.string()),
        ("landmark", pa.string()),
        ("facility_type", pa.string()),
        ("status", pa.string()),
        ("due_date", pa.timestamp("ms")),
        ("resolution_description", pa.string()),
        ("resolution_action_updated_date", pa.timestamp("ms")),
        ("community_board", pa.string()),
        ("bbl", pa.string()),
        ("borough", pa.string()),
        ("x_coordinate_state_plane", pa.float64()),
        ("y_coordinate_state_plane", pa.float64()),
        ("open_data_channel_type", pa.string()),
        ("park_facility_name", pa.string()),
        ("park_borough", pa.string()),
        ("vehicle_type", pa.string()),
        ("taxi_company_borough", pa.string()),
        ("taxi_pick_up_location", pa.string()),
        ("bridge_highway_name", pa.string()),
        ("bridge_highway_direction", pa.string()),
        ("road_ramp", pa.string()),
        ("bridge_highway_segment", pa.string()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("location", pa.string()),
    ]
)
NUMERICAL_COLUMNS = [
    'lag1', 'lag4', 'roll4', 'roll12',
    'momentum', 'weeks_since_last',
    'tavg', 'prcp', 'heating_degree', 'cooling_degree',
    'rain_3d', 'rain_7d', 'log_pop', 'nbr_roll4', 'nbr_roll12'
]

CATEGORICAL_COLUMNS = [
    'month', 'heat_flag', 'freeze_flag', 'hex6', 'complaint_family', 'covid_flag'
]

MODEL_TIMESTAMP = '20251006_015425'
