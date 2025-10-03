"""Configuration settings for NYC 311 Service Requests data loader.

This module contains all configuration parameters including API credentials,
data source settings, and processing options.
"""

import os
import pyarrow as pa

# Socrata API Configuration
API_ENDPOINT = "data.cityofnewyork.us"
DATASET_ID = "erm2-nwe9"  # NYC 311 Service Requests dataset
DATE_COLUMN = "created_date"

# API Credentials (loaded from environment variables)
APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN")
API_KEY_ID = os.getenv("SOCRATA_API_KEY_ID")
API_KEY_SECRET = os.getenv("SOCRATA_API_KEY_SECRET")

# Data Processing Configuration
DATA_START_YEAR = 2010
DATA_END_YEAR = 2026
LOCAL_OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "data", "landing"))
S3_OUTPUT_DIR = "s3://hbc-technical-assessment-gk/landing/311-service-requests"

# Performance Settings
MAX_CONCURRENT_REQUESTS = 20

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
        ("incident_zip", pa.string()),  # keep as text to preserve leading zeros
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
        ("bbl", pa.string()),  # large IDs as text
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
