"""
Streamlit App for NYC 311 Service Requests Analysis

This app allows users to explore model performance and understand how future weeks 
may look by comparing to historical ones.
"""

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import forecast
from src import config
from src import plotting

# Page configuration
st.set_page_config(
    page_title="NYC 311 Service Requests Analysis",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_bundles():
    """Load all model bundles (cached for performance)"""
    bundle_mean = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_mean.pkl')
    bundle_90 = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_90.pkl')
    bundle_50 = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_50.pkl')
    bundle_10 = forecast.load_bundle(config.MODEL_TIMESTAMP, 'lgb_10.pkl')
    return bundle_mean, bundle_90, bundle_50, bundle_10

@st.cache_data
def load_streamlit_data():
    """Load the preprocessed data (cached for performance)"""
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "resources", "streamlit_data.parquet")
    )
    df = pd.read_parquet(file_path)
    return df.fillna(0)

def main():
    """Main Streamlit app"""
    
    # Load data and models
    df = load_streamlit_data()
    bundle_mean, bundle_90, bundle_50, bundle_10 = load_bundles()
    
    # Title and description
    st.title("🗽 NYC 311 Service Requests Analysis - DOHMH")
    st.markdown("""
    Explore model performance and understand how future weeks may look by comparing to historical data.
    Select parameters in the sidebar to visualize predictions and actual service request patterns across NYC.
    """)
    
    # Sidebar for user inputs
    st.sidebar.header("Configuration")
    
    # User Options
    week_list = sorted(df['week'].unique())[-52:][::-1]
    complaint_list = sorted(df['complaint_family'].unique())
    value_col_list = ['y', 'pred_mean', 'pred_50', 'pred_10', 'pred_90']
    
    # Value column descriptions
    value_col_descriptions = {
        'y': 'Actual Values',
        'pred_mean': 'Mean Prediction',
        'pred_50': '50th Percentile (Median)',
        'pred_10': '10th Percentile',
        'pred_90': '90th Percentile'
    }
    
    # Sidebar inputs
    st.sidebar.subheader("Time Period")
    week = st.sidebar.selectbox(
        "Select Week",
        options=week_list,
        index=0,  # Default to most recent week
        format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'),
        help="Choose a week to analyze (last 52 weeks available)"
    )
    
    st.sidebar.subheader("Visualization Settings")
    
    # Complaint family descriptions
    complaint_descriptions = {
        'food_safety': 'Issues related to food handling, safety, and labeling',
        'vector_control': 'Infestations or hazards from rodents, mosquitoes, or pigeons',
        'housing_health': 'Indoor or building conditions affecting health and safety',
        'animal_control': 'Animal care violations, unlicensed or dangerous animals',
        'air_smoke_mold': 'Indoor air quality issues including smoke and mold',
        'hazmat_lead_asbestos': 'Exposure to hazardous materials like lead or asbestos',
        'childcare_recreation': 'Safety or compliance issues at recreational or childcare facilities',
        'misc_other': 'Miscellaneous public complaints or administrative requests',
        'water_quality': 'Unsafe or contaminated drinking or bottled water',
        'covid': 'Violations of vaccine mandates or face covering requirements'
    }
    
    complaint_family = st.sidebar.selectbox(
        "Complaint Family",
        options=complaint_list,
        index=complaint_list.index('food_safety') if 'food_safety' in complaint_list else 0,
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Select the complaint family to visualize on the map"
    )
    
    # Show description for selected complaint family
    if complaint_family in complaint_descriptions:
        st.sidebar.caption(complaint_descriptions[complaint_family])
    
    value_col = st.sidebar.radio(
        "Value Type to Display",
        options=value_col_list,
        index=value_col_list.index('pred_mean'),
        format_func=lambda x: value_col_descriptions[x],
        help="Choose between actual values or different prediction quantiles"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Model Timestamp:** {config.MODEL_TIMESTAMP}")
    
    # Main content area
    with st.spinner('Generating predictions...'):
        # Filter data by selected week
        df_filtered = df[df['week'] == week].copy()
        
        # Generate predictions for all quantiles
        df_filtered['pred_mean'] = bundle_mean[1].predict(
            df_filtered[config.NUMERICAL_COLUMNS + config.CATEGORICAL_COLUMNS]
        ).round(0)
        df_filtered['pred_90'] = bundle_90[1].predict(
            df_filtered[config.NUMERICAL_COLUMNS + config.CATEGORICAL_COLUMNS]
        ).round(0)
        df_filtered['pred_50'] = bundle_50[1].predict(
            df_filtered[config.NUMERICAL_COLUMNS + config.CATEGORICAL_COLUMNS]
        ).round(0)
        df_filtered['pred_10'] = bundle_10[1].predict(
            df_filtered[config.NUMERICAL_COLUMNS + config.CATEGORICAL_COLUMNS]
        ).round(0)
    
    # Summary section
    st.header("📊 Weekly Summary by Complaint Family")
    
    df_complaint_summary = df_filtered.groupby('complaint_family')[value_col_list].sum().reset_index()
    df_complaint_summary = df_complaint_summary.sort_values('y', ascending=False)
    
    # Display metrics for selected complaint family
    col1, col2, col3, col4 = st.columns(4)
    
    selected_summary = df_complaint_summary[df_complaint_summary['complaint_family'] == complaint_family]
    if len(selected_summary) > 0:
        actual = selected_summary['y'].values[0]
        pred_mean = selected_summary['pred_mean'].values[0]
        pred_10 = selected_summary['pred_10'].values[0]
        pred_90 = selected_summary['pred_90'].values[0]
    else:
        actual = 0
        pred_mean = 0
        pred_10 = 0
        pred_90 = 0
    
    col1.metric("Actual Requests", f"{int(actual):,}")
    col2.metric("Mean Prediction", f"{int(pred_mean):,}")
    col3.metric("10th Percentile", f"{int(pred_10):,}")
    col4.metric("90th Percentile", f"{int(pred_90):,}")
    
    # Display summary table
    st.subheader("All Complaint Families")
    
    # Ensure we have data to display
    if len(df_complaint_summary) > 0:
        # Rename columns for display
        display_summary = df_complaint_summary.copy()
        display_summary.columns = ['Complaint Family', 'Actual', 'Mean Pred', 
                                    '50th %ile', '10th %ile', '90th %ile']
        
        # Format numeric columns as integers with thousands separator
        for col in ['Actual', 'Mean Pred', '50th %ile', '10th %ile', '90th %ile']:
            display_summary[col] = display_summary[col].apply(lambda x: f"{int(x):,}")
        
        st.dataframe(display_summary, use_container_width=True, height=200)
    else:
        # Show empty table with 0 values
        empty_df = pd.DataFrame({
            'Complaint Family': ['No data'],
            'Actual': [0],
            'Mean Pred': [0],
            '50th %ile': [0],
            '10th %ile': [0],
            '90th %ile': [0]
        })
        st.dataframe(empty_df, use_container_width=True, height=200)
    
    # Geographic visualization
    st.header(f"🗺️ Geographic Distribution: {complaint_family.replace('_', ' ').title()}")
    st.markdown(f"""
    Showing **{value_col_descriptions[value_col]}** for week starting 
    **{pd.to_datetime(week).strftime('%B %d, %Y')}**
    """)
    
    # Filter for selected complaint family
    df_complaint = df_filtered[df_filtered['complaint_family'] == complaint_family].copy()
    
    if len(df_complaint) > 0:
        with st.spinner('Generating map...'):
            # Create the plot
            fig, ax, counts = plotting.plot_h3_counts_for_week(
                df_complaint,
                week,
                complaint_family,
                value_col=value_col,
                title='',
                cmap="Reds",
                figsize=(6, 6),  # Smaller map size
            )
            
            # Display the plot (use_container_width=False to respect figsize)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            
            # Show hexagon statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Hexagons", len(counts))
            col2.metric("Total Requests", f"{int(counts['count'].sum()):,}")
            col3.metric("Avg per Hexagon", f"{counts['count'].mean():.1f}")
    else:
        st.warning(f"No data available for {complaint_family} in the selected week.")
    
    # Footer
    st.markdown("---")
    st.caption(f"Data source: NYC 311 Service Requests | Model: LightGBM ({config.MODEL_TIMESTAMP})")


if __name__ == "__main__":
    # To run: streamlit run ./streamlit_app/app.py
    main()







