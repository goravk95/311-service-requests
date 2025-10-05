"""
Streamlit App for NYC 311 Service Requests Analysis
"""

import os
import sys
import streamlit as st
import pandas as pd

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import forecast
from src import config

def load_bundles():
    bundle_mean = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_mean.pkl')
    bundle_90 = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_90.pkl')
    bundle_50 = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_50.pkl')
    bundle_10 = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_10.pkl')

    return bundle_mean, bundle_90, bundle_50, bundle_10

def load_streamlit_data():
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "resources", "streamlit_data.parquet")
    )
    df = pd.read_parquet(file_path)
    return df

df = load_streamlit_data()
bundle_mean, bundle_90, bundle_50, bundle_10 = load_bundles()

def app():
    st.title("NYC 311 Service Requests Analysis")

if __name__ == "__main__":
    # export PYTHONPATH="$PYTHONPATH:/Users/gorav_kumar/Documents/GitHub/nyc-311-service-requests/"
    # streamlit run ./streamlit_app/streamlit_app.py
    app()







