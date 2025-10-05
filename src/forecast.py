"""
Forecast Model: Multi-horizon time-series prediction using LightGBM with Poisson objective.
Trains separate models per horizon (1-4 weeks).
"""

import json
import cloudpickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from typing import Dict, Optional
import warnings
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, BaseCrossValidator
from lightgbm import LGBMRegressor
from scipy import stats

from . import config

warnings.filterwarnings("ignore")


def shift_by_date(group, target_col, time_delta):
    """
    Shift y backward by horizon weeks (look forward in time)

    Args:
        group: DataFrame with columns [hex, complaint_family, week, y]
        target_col: Target column name
        time_delta: Time delta (in weeks)

    Returns:
        DataFrame with added target column
    """
    group = group.set_index("week").sort_index()
    shifted = group["y"].shift(freq=-time_delta)
    group[f"{target_col}"] = shifted
    return group.reset_index()


def create_horizon_targets(
    panel: pd.DataFrame, horizons: List[int] = [1]
) -> pd.DataFrame:
    """
    Create per-horizon target variables y_h1, y_h2, y_h3, y_h4.
    Uses sparse construction (only for weeks that exist in panel).

    Args:
        panel: DataFrame with columns [hex, complaint_family, week, y]
        horizons: List of forecast horizons in weeks

    Returns:
        DataFrame with added target columns y_h1, y_h2, ..., y_hN
    """
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


def select_features(X, feature_list):
    """
    Select columns from X based off of list

        Parameters:
            X: DataFrame of features transformation
            feature_list: list of features
        Returns:
            X: DataFrame of subsetted columns
    """
    X = X[feature_list].copy()
    return X


def filter_data(X, y):
    """
    Filter X and y based on null values in X

        Parameters:
            X: DataFrame of features
            y: Series of target values
        Returns:
            X_transformed: subsetted DataFrame
            y_transformed: subsetted Series
    """
    nan_mask = pd.isnull(X).any(axis=1)
    X_transformed = X[~nan_mask]
    y_transformed = y[~nan_mask]
    print("X shape post-filtering:", X_transformed.shape)
    return X_transformed, y_transformed


def split_train_test_by_cutoff(X, y, date_column="day", cutoff="2024-01-01"):
    """Split X and y by a date cutoff where test >= cutoff.
    Returns X_train, X_test, y_train, y_test
    """
    cutoff = pd.Timestamp(cutoff)
    mask_test = pd.to_datetime(X[date_column]) >= cutoff
    X_train, X_test = X[~mask_test].copy(), X[mask_test].copy()
    y_train, y_test = y.loc[X_train.index].copy(), y.loc[X_test.index].copy()
    return X_train, X_test, y_train, y_test


class YearTimeSeriesSplit(BaseCrossValidator):
    """
    Time series splitter that splits data based on years.
    
    Parameters
    ----------
    year_column : array-like
        Array containing the year for each sample.
    """
    def __init__(self, year_column):
        self.year_column = np.array(year_column)
        self.unique_years = np.unique(self.year_column)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        # Number of splits is number of years minus 1
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
            ),
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
    horizons: List[int] = [1],
    params: Optional[Dict] = None,
    test_cutoff: str = "2025-01-01"
) -> Dict:
    """
    Train multi-horizon forecast models for a single complaint family.

    Args:
        panel: DataFrame from build_forecast_panel with all families
        horizons: List of forecast horizons (weeks)
        feature_cols: Feature columns to use (auto-detected if None)
        params: LightGBM parameters (uses defaults if None)

    Returns:
        Bundle dict with 'family', 'feature_cols', 'cat_cols', 'models', 'metrics'
    """
    df = create_horizon_targets(df, horizons)
    input_columns = df.columns.tolist()
    if params is None:
        params = {
            "objective": "poisson",
            "n_estimators": 800,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "random_state": 42,
            "verbose": -1,
        }

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
            test_cutoff = test_cutoff,
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
        print('train metrics')
        print(
            f"  h={horizon}: RMSE={metrics[horizon]['train']['rmse']:.3f}, MAE={metrics[horizon]['train']['mae']:.3f}, Poisson Dev={metrics[horizon]['train']['poisson_deviance']:.3f}"
        )
        print('test metrics')
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
        input_dir: Directory with .joblib files

    Returns:
        Dict of model bundles
    """
    file_path = config.PROJECT_ROOT / "models" / timestamp / folder / filename

    with open(file_path, "rb") as f:
        bundle = cloudpickle.load(f)
    print(f"Model bundle loaded from: {file_path}")

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
    random_state: int = 42
) -> Dict:
    """
    Tune LightGBM forecast model using Optuna.
    Optimizes Poisson deviance on test set.
    Uses same pipeline structure as fit_pipeline with OHE.

    Args:
        df_input: Input DataFrame
        target_column: Target column name
        input_columns: List of columns to use as model inputs
        numerical_columns: List of numeric columns (passed through)
        categorical_columns: List of categorical columns (one-hot encoded)
        date_column: Name of date column (default 'week')
        test_cutoff: Boundary where test >= cutoff (default '2024-01-01')
        n_trials: Number of Optuna trials
        random_state: Random seed

    Returns:
        Dict with best_params and best_score
    """
    print(f"Tuning forecast model with {n_trials} trials...")
    
    # Sort by date ascending
    df = df_input.sort_values(by=date_column).reset_index(drop=True)
    df = create_horizon_targets(df, [horizon])
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

    # Preprocessor: OHE for categoricals; pass-through numerics
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_columns,
            ),
        ],
        remainder="passthrough",  # keep numeric columns
    )

    def objective(trial):
        params = {
            'objective': 'poisson',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 50, 80),
            'max_depth': trial.suggest_int('max_depth', 6, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'n_estimators': 500,
            'random_state': random_state,
            'verbose': -1
        }

        regressor = LGBMRegressor(**params)
        
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

        # Poisson deviance (lower is better)
        epsilon = 1e-10
        poisson_dev = 2 * np.mean(
            y_test * np.log((y_test + epsilon) / (y_pred + epsilon)) - (y_test - y_pred)
        )

        return poisson_dev

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=random_state)
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"✓ Best Poisson deviance: {study.best_value:.4f}")
    print(f"✓ Best params: {study.best_params}")

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
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
) -> None:
    """
    Plot predicted vs actual for forecast calibration.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        output_path: Path to save figure (None = display only)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    axes[0].scatter(y_pred, y_true, alpha=0.3, s=10)
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], "r--", label="Perfect calibration")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"{title} - Scatter")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Binned calibration
    n_bins = 10
    bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y_pred, bins[:-1]) - 1

    bin_means_pred = []
    bin_means_true = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means_pred.append(y_pred[mask].mean())
            bin_means_true.append(y_true[mask].mean())

    axes[1].plot(bin_means_pred, bin_means_true, "o-", label="Binned mean")
    axes[1].plot([0, max(bin_means_pred)], [0, max(bin_means_pred)], "r--", label="Perfect")
    axes[1].set_xlabel("Mean Predicted (binned)")
    axes[1].set_ylabel("Mean Actual")
    axes[1].set_title(f"{title} - Calibration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved calibration plot -> {output_path}")
    else:
        plt.show()

    plt.close()


def evaluate_models(
    bundle: Dict,
) -> Dict:
    """
    Evaluate models with cross-validation scores, calibration plots, and residual analysis.
    Includes normality tests and QQ plots for residuals.
    """
    for horizon in bundle["horizons"]:
        # CV Scores Plot
        plt.plot(bundle["full_fits"][horizon]["cv_scores"])
        plt.axhline(np.mean(bundle["full_fits"][horizon]["cv_scores"]), linestyle = '--', c = '#000000')
        plt.title(f"CV Scores for Horizon {horizon}")
        plt.xlabel("Fold")
        plt.ylabel("CV Score")
        plt.show()

        # Get predictions
        y_true_test = bundle["full_fits"][horizon]["y_test"]
        y_pred_test = bundle["full_fits"][horizon]["pipeline"].predict(bundle["full_fits"][horizon]["X_test"])
        
        # Forecast Calibration Plot
        plot_forecast_calibration(y_true_test, y_pred_test, title=f"Forecast Calibration for Horizon {horizon}")

        # Residual Analysis
        residuals = y_true_test - y_pred_test
        
        # Create figure with 3 subplots for residual analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Histogram of residuals with normal distribution overlay
        axes[0].hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Overlay normal distribution
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
        axes[0].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].set_xlabel('Residuals')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Residual Distribution (Horizon {horizon})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. QQ Plot
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f'Q-Q Plot (Horizon {horizon})')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Residuals vs Fitted Values
        axes[2].scatter(y_pred_test, residuals, alpha=0.3, s=10)
        axes[2].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Fitted Values')
        axes[2].set_ylabel('Residuals')
        axes[2].set_title(f'Residuals vs Fitted (Horizon {horizon})')
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
        residuals_sample = np.random.choice(residuals, size=sample_size, replace=False) if len(residuals) > sample_size else residuals
        shapiro_stat, shapiro_p = stats.shapiro(residuals_sample)
        print(f"\nShapiro-Wilk Normality Test (n={len(residuals_sample)}):")
        print(f"  Statistic: {shapiro_stat:.4f}")
        print(f"  P-value: {shapiro_p:.4e}")
        print(f"  Interpretation: {'Residuals are normally distributed (α=0.05)' if shapiro_p > 0.05 else 'Residuals are NOT normally distributed (α=0.05)'}")
        
        # Anderson-Darling test
        anderson_result = stats.anderson(residuals, dist='norm')
        print(f"\nAnderson-Darling Normality Test:")
        print(f"  Statistic: {anderson_result.statistic:.4f}")
        print(f"  Critical Values: {anderson_result.critical_values}")
        print(f"  Significance Levels: {anderson_result.significance_level}%")
        
        print(f"\n{'-'*60}")
        print("Model Performance Metrics:")
        print(bundle["metrics"][horizon])
        print(f"{'='*60}\n")
