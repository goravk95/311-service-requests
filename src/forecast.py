"""Forecast Model: Multi-horizon time-series prediction using LightGBM.

This module implements multi-horizon forecasting models with both mean (Poisson) and
quantile regression objectives. Includes model training, hyperparameter tuning, evaluation,
and persistence utilities.
"""

import json
from pathlib import Path
import warnings
from io import BytesIO
from typing import Dict, Optional
import cloudpickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, BaseCrossValidator
from lightgbm import LGBMRegressor
from scipy import stats
import optuna
from optuna.samplers import TPESampler
import boto3

from . import config

warnings.filterwarnings("ignore")


def shift_by_date(group: pd.DataFrame, target_col: str, time_delta: pd.Timedelta) -> pd.DataFrame:
    """Shift target variable backward by specified time delta.

    Args:
        group: DataFrame with time series data.
        target_col: Target column name.
        time_delta: Time delta to shift.

    Returns:
        DataFrame with shifted target column added.
    """
    group = group.set_index("week").sort_index()
    shifted = group["y"].shift(freq=-time_delta)
    group[f"{target_col}"] = shifted
    return group.reset_index()


def create_horizon_targets(panel: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """Create per-horizon target variables.

    Args:
        panel: DataFrame with time series panel data.
        horizons: List of forecast horizons in weeks.

    Returns:
        DataFrame with added target columns for each horizon.
    """
    if horizons is None:
        horizons = [1]
    result = panel.copy()
    result = result.sort_values(["hex8", "complaint_family", "week"])

    for horizon in horizons:
        target_col = f"y_h{horizon}"
        time_delta = pd.Timedelta(weeks=horizon)
        result = result.groupby(["hex8", "complaint_family"], group_keys=False).apply(
            lambda group: shift_by_date(group, target_col, time_delta)
        )
        result[target_col] = result[target_col].fillna(0)

    return result


def select_features(X: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Select subset of columns from DataFrame.

    Args:
        X: DataFrame of features.
        feature_list: List of features to select.

    Returns:
        DataFrame with selected columns.
    """
    X = X[feature_list].copy()
    return X


def filter_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Filter X and y based on null values in X.

    Args:
        X: DataFrame of features.
        y: Series of target values.

    Returns:
        Tuple of filtered (X, y).
    """
    nan_mask = pd.isnull(X).any(axis=1)
    X_transformed = X[~nan_mask]
    y_transformed = y[~nan_mask]
    print("X shape post-filtering:", X_transformed.shape)
    return X_transformed, y_transformed


def split_train_test_by_cutoff(
    X: pd.DataFrame, y: pd.Series, date_column: str = "day", cutoff: str = "2024-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split X and y by a date cutoff.

    Args:
        X: Features DataFrame.
        y: Target Series.
        date_column: Name of date column.
        cutoff: Date cutoff string.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    cutoff = pd.Timestamp(cutoff)
    mask_test = pd.to_datetime(X[date_column]) >= cutoff
    X_train, X_test = X[~mask_test].copy(), X[mask_test].copy()
    y_train, y_test = y.loc[X_train.index].copy(), y.loc[X_test.index].copy()
    return X_train, X_test, y_train, y_test


class YearTimeSeriesSplit(BaseCrossValidator):
    """Time series splitter that splits data based on years.

    Args:
        year_column: Array containing the year for each sample.
    """

    def __init__(self, year_column):
        self.year_column = np.array(year_column)
        self.unique_years = np.unique(self.year_column)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.unique_years) - 1

    def split(self, X, y=None, groups=None):
        X = np.array(X)
        for i in range(1, len(self.unique_years)):
            train_years = self.unique_years[:i]
            test_years = [self.unique_years[i]]

            train_idx = np.where(np.isin(self.year_column, train_years))[0]
            test_idx = np.where(np.isin(self.year_column, test_years))[0]

            yield train_idx, test_idx


def fit_pipeline(
    df_input,
    regressor,
    target_column,
    input_columns,
    numerical_columns,
    categorical_columns,
    date_column="week",
    test_cutoff="2024-01-01",
    cv_scoring=None,
):
    """
    Fit pipeline with a fixed train/test split where test is 2024-01-01 onwards.
    Performs time-based (expanding window) cross-validation on the TRAIN ONLY.

    Parameters:
        df_input: input DataFrame
        regressor: sklearn-compatible regressor
        target_column: str, target name
        input_columns: list of columns to use as model inputs
        numerical_columns: list of numeric columns (passed through)
        categorical_columns: list of categorical columns (one-hot)
        date_column: str, name of date column (default 'week')
        test_cutoff: str | timestamp, boundary where test >= cutoff (default '2024-01-01')
        cv_splits: int, number of time-based CV splits on train (default 5)
        cv_scoring: str | callable, sklearn scoring for CV (optional)
        cv_max_train_size: int | None, optional cap for train size in each fold

    Returns:
        pipeline: fitted sklearn pipeline
        X_train, X_test, y_train, y_test: the split datasets
        cv_scores: list | None of CV scores (if scoring provided)
        cv_splitter: the TimeSeriesSplit instance used
    """
    # Sort by date ascending
    df = df_input.sort_values(by=date_column).reset_index(drop=True)

    # Build features/target
    feature_list = numerical_columns + categorical_columns
    X = df[input_columns].copy()
    y = df[target_column].copy()

    print("X shape pre-filtering:", X.shape)
    X, y = filter_data(X, y)

    # Enforce datetime
    X[date_column] = pd.to_datetime(X[date_column])

    # Fixed cutoff split: test is 2024+
    X_train, X_test, y_train, y_test = split_train_test_by_cutoff(
        X, y, date_column=date_column, cutoff=test_cutoff
    )

    print(
        f"Train dates [{X_train[date_column].min()} to {X_train[date_column].max()}], "
        f"Test dates [{X_test[date_column].min()} to {X_test[date_column].max()}]"
    )
    print("X training shape:", X_train.shape)
    print("X test shape:", X_test.shape)

    # Preprocessor: OHE for categoricals; pass-through numerics
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_columns,
            )
        ],
        remainder="passthrough",  # keep numeric columns
    )

    pipeline = Pipeline(
        steps=[
            (
                "select_features",
                FunctionTransformer(select_features, kw_args={"feature_list": feature_list}),
            ),
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    # Extract year from date column for YearTimeSeriesSplit
    year_column = pd.to_datetime(X_train[date_column]).dt.year.values
    cv_splitter = YearTimeSeriesSplit(year_column=year_column)

    cv_scores = None
    if cv_scoring is not None:
        # Evaluate with CV on the training set only
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv_splitter, scoring=cv_scoring, n_jobs=None
        )
        print(f"CV ({cv_scoring}) scores:", cv_scores)
        print("CV mean:", cv_scores.mean())

    # Fit final model on all TRAIN data
    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test, cv_scores, cv_splitter


def train_models(
    df: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
    horizons: list[int] | None = None,
    model_type: str = "mean",
    test_cutoff: str = "2024-01-01",
) -> dict:
    """Train multi-horizon forecast models.

    Args:
        df: DataFrame with forecast panel data.
        numerical_columns: List of numerical feature columns.
        categorical_columns: List of categorical feature columns.
        horizons: List of forecast horizons (weeks).
        model_type: Type of model ('mean', '90', '50', '10').
        test_cutoff: Date cutoff for train/test split.

    Returns:
        Dictionary with trained models, metrics, and metadata.
    """
    if horizons is None:
        horizons = [1]
    json_path = config.RESOURCES_DIR / "model_optimal_params.json"
    with open(json_path, "r") as f:
        optimal_params = json.load(f)

    if model_type not in optimal_params:
        raise ValueError(
            f"model_type must be one of {list(optimal_params.keys())}, got {model_type}"
        )

    # Get the optimal params for this model type
    tuned_params = optimal_params[model_type]

    # Set up base parameters
    params = {
        "n_estimators": 800,
        "random_state": 42,
        "verbose": -1,
        **tuned_params,  # Add the tuned hyperparameters
    }

    # Set objective based on model type
    if model_type == "mean":
        params["objective"] = "poisson"
    else:
        params["objective"] = "quantile"
        params["alpha"] = float(model_type) / 100  # Convert '90' to 0.9, etc.

    df = create_horizon_targets(df, horizons)
    input_columns = df.columns.tolist()

    regressor = LGBMRegressor(**params)
    models = {}
    metrics = {}
    full_fits = {}
    for horizon in horizons:
        print(f"Training model for horizon {horizon}")
        target_col = f"y_h{horizon}"
        pipeline, X_train, X_test, y_train, y_test, cv_scores, cv_splitter = fit_pipeline(
            df,
            regressor,
            target_col,
            input_columns,
            numerical_columns,
            categorical_columns,
            test_cutoff=test_cutoff,
            cv_scoring="neg_mean_absolute_error",
        )
        full_fits[horizon] = {
            "pipeline": pipeline,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "cv_scores": cv_scores,
            "cv_splitter": cv_splitter,
        }

        models[horizon] = pipeline
        y_pred_test = pipeline.predict(X_test)
        y_pred_train = pipeline.predict(X_train)

        metrics[horizon] = {
            "train": eval_forecast(y_train, y_pred_train, horizon),
            "test": eval_forecast(y_test, y_pred_test, horizon),
        }
        print("train metrics")
        print(
            f"  h={horizon}: RMSE={metrics[horizon]['train']['rmse']:.3f}, MAE={metrics[horizon]['train']['mae']:.3f}, Poisson Dev={metrics[horizon]['train']['poisson_deviance']:.3f}"
        )
        print("test metrics")
        print(
            f"  h={horizon}: RMSE={metrics[horizon]['test']['rmse']:.3f}, MAE={metrics[horizon]['test']['mae']:.3f}, Poisson Dev={metrics[horizon]['test']['poisson_deviance']:.3f}"
        )
        print()

    bundle = {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "models": models,
        "metrics": metrics,
        "horizons": horizons,
        "full_fits": full_fits,
    }

    return bundle


def predict_forecast(
    bundles: Dict[str, Dict], last_rows: pd.DataFrame, horizon: int = 4
) -> pd.DataFrame:
    """
    Generate forecasts from the latest state.

    Args:
        bundles: Dict of trained model bundles per family
        last_rows: Latest row per [hex, complaint_family] with features
        horizon: Forecast horizon (1-4 weeks)

    Returns:
        DataFrame with [hex, complaint_family, week, p50, p10, p90]
    """
    predictions = []

    for family, bundle in bundles.items():
        if horizon not in bundle["models"]:
            continue

        # Filter to this family
        df_fam = last_rows[last_rows["complaint_family"] == family].copy()

        if len(df_fam) == 0:
            continue

        model = bundle["models"][horizon]
        feature_cols = bundle["feature_cols"]

        # Check features exist
        missing = [c for c in feature_cols if c not in df_fam.columns]
        if missing:
            print(f"Warning: {family} missing features: {missing}")
            continue

        X = df_fam[feature_cols]

        # Predict
        y_pred = model.predict(X)

        # For Poisson, use variance = mean for uncertainty
        # p50 = mean, p10/p90 via Poisson quantiles
        from scipy.stats import poisson

        p50 = y_pred
        p10 = np.array([poisson.ppf(0.1, lam) if lam > 0 else 0 for lam in y_pred])
        p90 = np.array([poisson.ppf(0.9, lam) if lam > 0 else 0 for lam in y_pred])

        # Forecast week
        forecast_week = df_fam["week"] + pd.Timedelta(weeks=horizon)

        pred_df = pd.DataFrame(
            {
                "hex8": df_fam["hex8"].values,
                "complaint_family": family,
                "week": forecast_week,
                "p50": p50,
                "p10": p10,
                "p90": p90,
            }
        )

        predictions.append(pred_df)

    if not predictions:
        return pd.DataFrame(columns=["hex8", "complaint_family", "week", "p50", "p10", "p90"])

    result = pd.concat(predictions, ignore_index=True)
    return result


def save_bundle(bundle: Dict[str, Dict], timestamp: str, filename: str) -> None:
    """
    Save a dictionary of sklearn objects (models, transformers, etc.)
    as one bundled pickle file in models/<timestamp>/.

    Args:
        model_dict (dict): Dictionary of sklearn objects.
        output_dir (str): Base directory to save the models.
        filename (str): Name of the file to save the models. Ending with .pkl

    Returns:
        str: Path to the saved pickle file.
    """
    output_dir = config.PROJECT_ROOT / "models" / timestamp / "full_bundle"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    with open(file_path, "wb") as f:
        cloudpickle.dump(bundle, f)

    output_dir = config.PROJECT_ROOT / "models" / timestamp / "just_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    with open(file_path, "wb") as f:
        cloudpickle.dump(bundle["models"], f)

    print(f"Model bundle saved to: {file_path}")


def load_bundle(timestamp: Path, filename: str, folder: str = "just_model") -> Dict[str, Dict]:
    """
    Load forecast bundles from disk.

    Args:
        input_dir: Directory with .pkl files

    Returns:
        Dict of model bundles
    """
    file_path = config.PROJECT_ROOT / "models" / timestamp / folder / filename

    with open(file_path, "rb") as f:
        bundle = cloudpickle.load(f)
    print(f"Model bundle loaded from: {file_path}")

    return bundle


def save_bundle_s3(bundle: Dict[str, Dict], timestamp: str, filename: str) -> None:
    """
    Save a dictionary of sklearn objects (models, transformers, etc.)
    as one bundled pickle file to S3 in models/<timestamp>/.

    Args:
        bundle (dict): Dictionary containing models and other sklearn objects.
        timestamp (str): Timestamp string for versioning (e.g., '20251006_154213').
        filename (str): Name of the file to save the models. Should end with .pkl

    Returns:
        None
    """
    s3_client = boto3.client("s3")

    bucket_name = config.BUCKET_NAME
    prefix = "models"
    full_bundle_key = f"{prefix}/{timestamp}/full_bundle/{filename}"
    buffer = BytesIO()
    cloudpickle.dump(bundle, buffer)
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, bucket_name, full_bundle_key)
    print(f"Full bundle saved to: s3://{bucket_name}/{full_bundle_key}")

    just_model_key = f"{prefix}/{timestamp}/just_model/{filename}"
    buffer = BytesIO()
    cloudpickle.dump(bundle["models"], buffer)
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, bucket_name, just_model_key)
    print(f"Model bundle saved to: s3://{bucket_name}/{just_model_key}")


def load_bundle_s3(timestamp: str, filename: str, folder: str = "just_model") -> Dict[str, Dict]:
    """
    Load forecast bundles from S3.

    Args:
        timestamp (str): Timestamp string for versioning (e.g., '20251006_154213').
        filename (str): Name of the file to load. Should end with .pkl
        folder (str): Subfolder name ('just_model' or 'full_bundle'). Default: 'just_model'

    Returns:
        Dict of model bundles
    """
    s3_client = boto3.client("s3")

    bucket_name = config.BUCKET_NAME
    prefix = "models"
    s3_key = f"{prefix}/{timestamp}/{folder}/{filename}"

    buffer = BytesIO()
    s3_client.download_fileobj(bucket_name, s3_key, buffer)
    buffer.seek(0)
    bundle = cloudpickle.load(buffer)
    print(f"Model bundle loaded from: s3://{bucket_name}/{s3_key}")

    return bundle


def tune(
    df_input: pd.DataFrame,
    horizon: int,
    input_columns: list,
    numerical_columns: list,
    categorical_columns: list,
    date_column: str = "week",
    test_cutoff: str = "2024-01-01",
    n_trials: int = 30,
    random_state: int = 42,
    alpha: Optional[float] = None,
) -> Dict:
    """
    Tune LightGBM forecast model using Optuna.
    Supports both Poisson (mean) and quantile regression.
    Uses same pipeline structure as fit_pipeline with OHE.

    Args:
        df_input: Input DataFrame
        horizon: Forecast horizon (weeks)
        input_columns: List of columns to use as model inputs
        numerical_columns: List of numeric columns (passed through)
        categorical_columns: List of categorical columns (one-hot encoded)
        date_column: Name of date column (default 'week')
        test_cutoff: Boundary where test >= cutoff (default '2024-01-01')
        n_trials: Number of Optuna trials
        random_state: Random seed
        alpha: Quantile to predict (0.1 for 10th percentile, 0.5 for median, 0.9 for 90th).
               If None, uses Poisson objective for mean prediction.

    Returns:
        Dict with best_params, best_score, and alpha
    """
    objective_name = f"quantile (α={alpha})" if alpha is not None else "poisson (mean)"
    print(f"Tuning forecast model for {objective_name} with {n_trials} trials...")

    df = create_horizon_targets(df_input, [horizon])
    df = df.sort_values(by=date_column).reset_index(drop=True)
    target_col = f"y_h{horizon}"

    # Build features/target
    feature_list = numerical_columns + categorical_columns
    X = df[input_columns].copy()
    y = df[target_col].copy()

    print("X shape pre-filtering:", X.shape)
    X, y = filter_data(X, y)

    # Enforce datetime
    X[date_column] = pd.to_datetime(X[date_column])

    # Fixed cutoff split: test is test_cutoff onwards
    X_train, X_test, y_train, y_test = split_train_test_by_cutoff(
        X, y, date_column=date_column, cutoff=test_cutoff
    )

    print(
        f"Train dates [{X_train[date_column].min()} to {X_train[date_column].max()}], "
        f"Test dates [{X_test[date_column].min()} to {X_test[date_column].max()}]"
    )
    print("X training shape:", X_train.shape)
    print("X test shape:", X_test.shape)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 48, 112),
            "max_depth": trial.suggest_int("max_depth", 6, 9),
            "min_child_samples": trial.suggest_int("min_child_samples", 24, 60),
            "subsample": trial.suggest_float("subsample", 0.70, 0.95),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),  # activates subsample
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.95),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 3.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
            "n_estimators": trial.suggest_int("n_estimators", 350, 900),
            "random_state": random_state,
            "verbose": -1,
        }

        # Set objective based on alpha
        if alpha is not None:
            params["objective"] = "quantile"
            params["alpha"] = alpha
        else:
            params["objective"] = "poisson"

        regressor = LGBMRegressor(**params)

        # Preprocessor: OHE for categoricals; pass-through numerics (fresh for each trial)
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "one_hot_encoder",
                    OneHotEncoder(handle_unknown="ignore", drop="first"),
                    categorical_columns,
                )
            ],
            remainder="passthrough",  # keep numeric columns
        )

        # Build pipeline
        pipeline = Pipeline(
            steps=[
                (
                    "select_features",
                    FunctionTransformer(select_features, kw_args={"feature_list": feature_list}),
                ),
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )

        # Fit on train
        pipeline.fit(X_train, y_train)

        # Predict on test
        y_pred = pipeline.predict(X_test)

        # Choose evaluation metric based on objective
        if alpha is not None:
            # Pinball loss (quantile loss) - lower is better
            errors = y_test - y_pred
            loss = np.mean(np.maximum(alpha * errors, (alpha - 1) * errors))
        else:
            # Poisson deviance (lower is better)
            epsilon = 1e-10
            loss = 2 * np.mean(
                y_test * np.log((y_test + epsilon) / (y_pred + epsilon)) - (y_test - y_pred)
            )

        return loss

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=random_state))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    metric_name = "Pinball loss" if alpha is not None else "Poisson deviance"
    print(f"✓ Best {metric_name}: {study.best_value:.4f}")
    print(f"✓ Best params: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "alpha": alpha,
        # 'study': study
    }


def eval_forecast(
    y_true: pd.Series, y_pred: pd.Series, family: str = "Unknown", horizon: int = 1
) -> Dict:
    """
    Evaluate forecast predictions.

    Args:
        y_true: True values
        y_pred: Predicted values
        family: Complaint family name
        horizon: Forecast horizon

    Returns:
        Dict with RMSE, MAE, Poisson deviance, MAPE
    """
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    # Poisson deviance
    epsilon = 1e-10
    poisson_dev = 2 * np.mean(
        y_true * np.log((y_true + epsilon) / (y_pred + epsilon)) - (y_true - y_pred)
    )

    metrics = {
        "horizon": horizon,
        "rmse": float(rmse),
        "mae": float(mae),
        "poisson_deviance": float(poisson_dev),
        "n_samples": len(y_true),
    }

    return metrics


def plot_forecast_calibration(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Forecast Calibration",
    output_path: Optional[Path] = None,
    model_type: Optional[str] = None,
    n_bins: int = 10,
) -> None:
    """
    Plot predicted vs actual for forecast calibration.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        output_path: Path to save figure (None = display only)
        model_type: Model type - "mean" for Poisson or percentile as string ("10", "50", "90")
        n_bins: Number of bins for calibration plot (default: 10)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Add model type to title if provided
    plot_title = title
    if model_type is not None:
        if model_type.lower() == "mean":
            plot_title = f"{title} (Mean)"
        else:
            # Handle percentile as string ("10", "50", "90")
            plot_title = f"{title} (P{model_type})"

    # Scatter plot
    axes[0].scatter(y_pred, y_true, alpha=0.3, s=10)
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], "r--", label="Perfect calibration")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"{plot_title} - Scatter")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Binned calibration
    bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y_pred, bins[:-1]) - 1

    bin_values_pred = []
    bin_values_true = []

    # Determine whether to use mean or percentile for calibration
    if model_type is not None and model_type.lower() != "mean":
        # Use percentile for quantile models
        percentile_value = float(model_type)
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_values_pred.append(np.percentile(y_pred[mask], 50))  # Median of predicted
                bin_values_true.append(np.percentile(y_true[mask], percentile_value))

        axes[1].plot(bin_values_pred, bin_values_true, "o-", label=f"Binned P{model_type}")
        axes[1].plot([0, max(bin_values_pred)], [0, max(bin_values_pred)], "r--", label="Perfect")
        axes[1].set_xlabel(f"P{model_type} Predicted (binned)")
        axes[1].set_ylabel(f"P{model_type} Actual")
    else:
        # Use mean for mean models
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_values_pred.append(y_pred[mask].mean())
                bin_values_true.append(y_true[mask].mean())

        axes[1].plot(bin_values_pred, bin_values_true, "o-", label="Binned mean")
        axes[1].plot([0, max(bin_values_pred)], [0, max(bin_values_pred)], "r--", label="Perfect")
        axes[1].set_xlabel("Mean Predicted (binned)")
        axes[1].set_ylabel("Mean Actual")

    axes[1].set_title(f"{plot_title} - Calibration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved calibration plot -> {output_path}")
    else:
        plt.show()

    plt.close()


def evaluate_models(bundle: Dict, model_type: Optional[str] = None, n_bins: int = 10) -> Dict:
    """
    Evaluate models with cross-validation scores, calibration plots, and residual analysis.
    Includes normality tests and QQ plots for residuals.

    Args:
        bundle: Model bundle containing trained models and evaluation data
        model_type: Model type - "mean" for Poisson or percentile as string ("10", "50", "90")
        n_bins: Number of bins for calibration plot (default: 10)
    """
    for horizon in bundle["horizons"]:
        # CV Scores Plot
        plt.plot(bundle["full_fits"][horizon]["cv_scores"])
        plt.axhline(np.mean(bundle["full_fits"][horizon]["cv_scores"]), linestyle="--", c="#000000")
        plt.title(f"CV Scores for Horizon {horizon}")
        plt.xlabel("Fold")
        plt.ylabel("CV Score")
        plt.show()

        # Get predictions
        y_true_test = bundle["full_fits"][horizon]["y_test"]
        y_pred_test = bundle["full_fits"][horizon]["pipeline"].predict(
            bundle["full_fits"][horizon]["X_test"]
        )

        # Forecast Calibration Plot
        plot_forecast_calibration(
            y_true_test,
            y_pred_test,
            title=f"Forecast Calibration for Horizon {horizon}",
            model_type=model_type,
            n_bins=n_bins,
        )

        # Residual Analysis
        residuals = y_true_test - y_pred_test

        # Create figure with 3 subplots for residual analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Histogram of residuals with normal distribution overlay
        axes[0].hist(
            residuals, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black"
        )

        # Overlay normal distribution
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0].plot(
            x,
            stats.norm.pdf(x, mu, std),
            "r-",
            linewidth=2,
            label=f"Normal(μ={mu:.2f}, σ={std:.2f})",
        )
        axes[0].axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        axes[0].set_xlabel("Residuals")
        axes[0].set_ylabel("Density")
        axes[0].set_title(f"Residual Distribution (Horizon {horizon})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. QQ Plot
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f"Q-Q Plot (Horizon {horizon})")
        axes[1].grid(True, alpha=0.3)

        # 3. Residuals vs Fitted Values
        axes[2].scatter(y_pred_test, residuals, alpha=0.3, s=10)
        axes[2].axhline(0, color="red", linestyle="--", linewidth=2)
        axes[2].set_xlabel("Fitted Values")
        axes[2].set_ylabel("Residuals")
        axes[2].set_title(f"Residuals vs Fitted (Horizon {horizon})")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Statistical tests for normality
        print(f"\n{'='*60}")
        print(f"Horizon {horizon} - Residual Analysis")
        print(f"{'='*60}")
        print(f"Residual Statistics:")
        print(f"  Mean: {mu:.4f}")
        print(f"  Std Dev: {std:.4f}")
        print(f"  Median: {np.median(residuals):.4f}")
        print(f"  Skewness: {stats.skew(residuals):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(residuals):.4f}")

        # Shapiro-Wilk test for normality (sample up to 5000 points for efficiency)
        sample_size = min(5000, len(residuals))
        residuals_sample = (
            np.random.choice(residuals, size=sample_size, replace=False)
            if len(residuals) > sample_size
            else residuals
        )
        shapiro_stat, shapiro_p = stats.shapiro(residuals_sample)
        print(f"\nShapiro-Wilk Normality Test (n={len(residuals_sample)}):")
        print(f"  Statistic: {shapiro_stat:.4f}")
        print(f"  P-value: {shapiro_p:.4e}")
        print(
            f"  Interpretation: {'Residuals are normally distributed (α=0.05)' if shapiro_p > 0.05 else 'Residuals are NOT normally distributed (α=0.05)'}"
        )

        # Anderson-Darling test
        anderson_result = stats.anderson(residuals, dist="norm")
        print(f"\nAnderson-Darling Normality Test:")
        print(f"  Statistic: {anderson_result.statistic:.4f}")
        print(f"  Critical Values: {anderson_result.critical_values}")
        print(f"  Significance Levels: {anderson_result.significance_level}%")

        print(f"\n{'-'*60}")
        print("Model Performance Metrics:")
        print(bundle["metrics"][horizon])
        print(f"{'='*60}\n")
