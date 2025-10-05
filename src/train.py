
from datetime import datetime
from . import fetch
from . import preprocessing
from . import features
from . import forecast

def train_models(run_fetch = False, save_data = False, save_models = False):
    if run_fetch:
        fetch.fetch_all_service_requests(save = save_data)
        df_pop = fetch.fetch_acs_census_population_data(start_year=2013, end_year=2023, save=save_data)
        df_weather = fetch.fetch_noaa_weather_data(start_year=2010, end_year=2025, save=save_data)
    
    df = preprocessing.preprocess_and_merge_external_data()
    if save_data:
        preprocessing.save_preprocessed_data(df)
    
    forecast_panel = features.build_forecast_panel(df)
    if save_data:
        features.save_forecast_panel_data(forecast_panel)


    numerical_columns = [
            'lag1', 'lag4', 'roll4', 'roll12',
            'momentum', 'weeks_since_last',
            'tavg', 'prcp', 'heating_degree', 'cooling_degree',
            'rain_3d', 'rain_7d', 'log_pop', 'nbr_roll4', 'nbr_roll12'
        ]

    categorical_columns = ['week_of_year', 'month', 'heat_flag', 'freeze_flag', 'hex6', 'complaint_family']
    horizons = [1]

    poisson_params = {
                'objective': 'poisson',
                'n_estimators': 800,
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'random_state': 42,
                'verbose': -1
            }

    bundle_mean = forecast.train_models(
        forecast_panel,
        numerical_columns,
        categorical_columns,
        horizons,
        poisson_params
    )

    poisson_params = {
                'objective': 'quantile',
                'alpha': 0.9,
                'n_estimators': 800,
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'random_state': 42,
                'verbose': -1
            }

    bundle_90 = forecast.train_models(
        forecast_panel,
        numerical_columns,
        categorical_columns,
        horizons,
        poisson_params
    )

    poisson_params = {
                'objective': 'quantile',
                'alpha': 0.5,
                'n_estimators': 800,
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'random_state': 42,
                'verbose': -1
            }

    bundle_50 = forecast.train_models(
        forecast_panel,
        numerical_columns,
        categorical_columns,
        horizons,
        poisson_params
    )


    poisson_params = {
                'objective': 'quantile',
                'alpha': 0.1,
                'n_estimators': 800,
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'random_state': 42,
                'verbose': -1
            }

    bundle_10 = forecast.train_models(
        forecast_panel,
        numerical_columns,
        categorical_columns,
        horizons,
        poisson_params
    )


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_models:
        forecast.save_bundle(bundle_mean,  timestamp, 'lgb_mean.pkl')
        forecast.save_bundle(bundle_90,  timestamp, 'lgb_90.pkl')
        forecast.save_bundle(bundle_50,  timestamp, 'lgb_50.pkl')
        forecast.save_bundle(bundle_10,  timestamp, 'lgb_10.pkl')

if __name__ == "__main__":
    train_model(run_fetch = False, save_data = False, save_models = True)