"""Model training pipeline for NYC 311 service requests.

This module orchestrates the end-to-end training pipeline including data fetching,
preprocessing, feature engineering, and model training for multiple quantile forecasts.
"""

from datetime import datetime
from . import fetch
from . import preprocessing
from . import features
from . import forecast
from . import config


def train_models(run_fetch: bool = False, save_data: bool = False, save_models: bool = False, save_models_s3: bool = False) -> None:
    """Train forecast models for NYC 311 service requests.

    Args:
        run_fetch: Whether to fetch fresh data from external sources.
        save_data: Whether to save preprocessed data.
        save_models: Whether to save trained models.
    """
    print("Starting training pipeline...")
    if run_fetch:
        print("Fetching data...")
        fetch.fetch_all_service_requests(save = save_data)
        df_pop = fetch.fetch_acs_census_population_data(start_year=2013, end_year=2023, save=save_data)
        df_weather = fetch.fetch_noaa_weather_data(start_year=2010, end_year=2025, save=save_data)
    
    print("Preprocessing data...")
    df = preprocessing.preprocess_and_merge_external_data()
    if save_data:
        preprocessing.save_preprocessed_data(df)

    print("Building forecast panel...")
    forecast_panel = features.build_forecast_panel(df)
    if save_data:
        features.save_forecast_panel_data(forecast_panel)

    horizons = [1]

    print("Training models...")
    print("Training mean model...")
    bundle_mean = forecast.train_models(
        forecast_panel,
        config.NUMERICAL_COLUMNS,
        config.CATEGORICAL_COLUMNS,
        horizons,
        model_type='mean'
    )

    print("Training 90th percentile model...")
    bundle_90 = forecast.train_models(
        forecast_panel,
        config.NUMERICAL_COLUMNS,
        config.CATEGORICAL_COLUMNS,
        horizons,
        model_type='90'
    )

    print("Training 50th percentile model...")
    bundle_50 = forecast.train_models(
        forecast_panel,
        config.NUMERICAL_COLUMNS,
        config.CATEGORICAL_COLUMNS,
        horizons,
        model_type='50'
    )

    print("Training 10th percentile model...")
    bundle_10 = forecast.train_models(
        forecast_panel,
        config.NUMERICAL_COLUMNS,
        config.CATEGORICAL_COLUMNS,
        horizons,
        model_type='10'
    )

    print("All models training complete!")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if save_models_s3:
        print("Saving models to S3...")
        forecast.save_bundle_s3(bundle_mean,  timestamp, 'lgb_mean.pkl')
        forecast.save_bundle_s3(bundle_90,  timestamp, 'lgb_90.pkl')
        forecast.save_bundle_s3(bundle_50,  timestamp, 'lgb_50.pkl')
        forecast.save_bundle_s3(bundle_10,  timestamp, 'lgb_10.pkl')
        
    if save_models:
        print("Saving models...")
        forecast.save_bundle(bundle_mean,  timestamp, 'lgb_mean.pkl')
        forecast.save_bundle(bundle_90,  timestamp, 'lgb_90.pkl')
        forecast.save_bundle(bundle_50,  timestamp, 'lgb_50.pkl')
        forecast.save_bundle(bundle_10,  timestamp, 'lgb_10.pkl')

    print('Done!!!')

if __name__ == "__main__":
    train_models(run_fetch = False, save_data = False, save_models = True)