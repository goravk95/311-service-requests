"""
Streamlit App for NYC 311 Service Requests Analysis
"""

import streamlit as st
import pandas as pd

from . import forecast
from . import config


def load_bundles():
    bundle_mean = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_mean.pkl')
    bundle_90 = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_90.pkl')
    bundle_50 = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_50.pkl')
    bundle_10 = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_10.pkl')

    return bundle_mean, bundle_90, bundle_50, bundle_10

def load_forecast_panel():
    df = pd.read_parquet(config.PRESENTATION_DATA_PATH)
    return df

df = load_forecast_panel()
bundle_mean, bundle_90, bundle_50, bundle_10 = load_bundles()

def app():
    st.title("NYC 311 Service Requests Analysis")
    st.write(df)
    st.write(bundle_mean)
    st.write(bundle_90)
    st.write(bundle_50)
    st.write(bundle_10)

if __name__ == "__main__":
    # export PYTHONPATH="$PYTHONPATH:/Users/gorav_kumar/Documents/GitHub/nyc-311-service-requests/"
    # streamlit run ./src/streamlit_app.py
    app()







