"""
Misc functions that were attempted, but had to be dropped.
"""

import pandas as pd
from typing import Tuple, List
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def make_descriptor_tfidf(
    df: pd.DataFrame,
    col: str = "descriptor_clean",
    min_df: int = 5,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 500,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Create TF-IDF features from text column.

    Args:
        df: DataFrame with text column
        col: Column name containing text
        min_df: Minimum document frequency
        ngram_range: N-gram range for tokenization
        max_features: Maximum number of features

    Returns:
        Tuple of (sparse feature matrix, fitted vectorizer)
    """
    # Fill missing values with empty string
    text_data = df[col].fillna("").astype(str)

    # Initialize and fit vectorizer
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        ngram_range=ngram_range,
        max_features=max_features,
        strip_accents="unicode",
        lowercase=True,
        token_pattern=r"\b\w+\b",
    )

    X_sparse = vectorizer.fit_transform(text_data)

    return X_sparse, vectorizer


def compute_time_based_rolling_counts(
    df: pd.DataFrame, timestamp_col: str, group_cols: List[str], window_hours: List[int]
) -> pd.DataFrame:
    """
    Compute rolling counts based on time windows (for intake pressure features).
    Vectorized using searchsorted for O(n log n) complexity per group.

    Args:
        df: DataFrame with timestamps
        timestamp_col: Timestamp column name
        group_cols: Columns to group by
        window_hours: List of window sizes in hours

    Returns:
        DataFrame with rolling count features added
    """
    df = df.sort_values(group_cols + [timestamp_col]).copy()

    for window in window_hours:
        col_name = f"intake_{window}h"

        # Vectorized rolling count using searchsorted
        def rolling_count(group):
            timestamps = group[timestamp_col].values
            timestamps_int = timestamps.astype("datetime64[ns]").astype(np.int64)

            window_ns = pd.Timedelta(hours=window).value  # in nanoseconds
            counts = np.zeros(len(timestamps), dtype=np.int32)

            for i in range(len(timestamps)):
                current_time = timestamps_int[i]
                # Find first index >= (current_time - window)
                start_idx = np.searchsorted(
                    timestamps_int[:i], current_time - window_ns, side="left"
                )
                counts[i] = i - start_idx

            group[col_name] = counts
            return group

        df = df.groupby(group_cols, group_keys=False).apply(rolling_count)

    return df


def build_triage_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, csr_matrix, TfidfVectorizer]:
    """
    Build triage features for ticket-level classification.

    Grain: one row per ticket at creation time (no leakage).
    Includes categorical one-hots, temporal features, local history, and TF-IDF text features.

    Weather features should already be present in df from preprocessing
    (via preprocess_and_merge_external_data).

    Args:
        df: Input DataFrame with H3 keys, temporal features, and weather features added

    Returns:
        Tuple of (feature DataFrame, TF-IDF sparse matrix, fitted vectorizer)
    """
    result = df.copy()

    result = result[result["created_date"].notna()].copy()

    categorical_cols = [
        "complaint_family",
        "open_data_channel_type",
        "location_type",
        "borough",
        "facility_type",
    ]

    for col in categorical_cols:
        result[col] = result[col].fillna("_missing").astype(str)

        dummies = pd.get_dummies(result[col], prefix=col, prefix_sep="_")

        if len(dummies.columns) > 10:
            top_cols = dummies.sum().nlargest(10).index
            dummies = dummies[top_cols]

        result = pd.concat([result, dummies], axis=1)

    result["due_date"] = pd.to_datetime(result["due_date"], errors="coerce")
    result["due_gap_hours"] = (
        (result["due_date"] - result["created_date"]).dt.total_seconds() / 3600
    ).fillna(0)

    # Due crosses weekend
    result["due_crosses_weekend"] = (
        (result["dow"] <= 4)  # Created on weekday
        & ((result["due_gap_hours"] / 24 + result["dow"]) >= 5)  # Extends into weekend
    ).astype(int)

    history_panel = (
        result.groupby(["hex", "complaint_family", "day"]).size().reset_index(name="daily_count")
    )

    def compute_history(group):
        group = group.sort_values("day")

        # 7-day rolling
        group["geo_family_roll7"] = group["daily_count"].rolling(window=7, min_periods=1).sum()

        # 28-day rolling
        group["geo_family_roll28"] = group["daily_count"].rolling(window=28, min_periods=1).sum()

        # Days since last
        group["days_since_last_geo_family"] = group["day"].diff().dt.days.fillna(999)

        return group

    history_panel = history_panel.groupby(["hex", "complaint_family"], group_keys=False).apply(
        compute_history
    )

    # Merge back to tickets using as-of join
    result = result.sort_values(["day", "hex", "complaint_family"])
    history_panel = history_panel.sort_values(["day", "hex", "complaint_family"])
    result = pd.merge_asof(
        result,
        history_panel[
            [
                "hex",
                "complaint_family",
                "day",
                "geo_family_roll7",
                "geo_family_roll28",
                "days_since_last_geo_family",
            ]
        ],
        on="day",
        by=["hex", "complaint_family"],
        direction="backward",
    )

    for col in ["geo_family_roll7", "geo_family_roll28", "days_since_last_geo_family"]:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0.0)

    result["site_key"] = result["bbl"].fillna("unknown")

    site_panel = result.groupby(["site_key", "day"]).size().reset_index(name="daily_site_count")

    # Compute rolling features per site
    def compute_site_history(group):
        group = group.sort_values("day")

        # 14-day rolling count
        group["repeat_site_14d"] = group["daily_site_count"].rolling(window=14, min_periods=1).sum()

        # 28-day rolling count
        group["repeat_site_28d"] = group["daily_site_count"].rolling(window=28, min_periods=1).sum()

        return group

    site_panel = site_panel.groupby("site_key", group_keys=False).apply(compute_site_history)

    # Merge back to tickets using as-of join
    result = result.sort_values(["day", "site_key"])
    site_panel = site_panel.sort_values(["day", "site_key"])
    result = pd.merge_asof(
        result,
        site_panel[["site_key", "day", "repeat_site_14d", "repeat_site_28d"]],
        on="day",
        by="site_key",
        direction="backward",
    )

    # Subtract 1 to exclude current ticket from the count (we want prior tickets only)
    result["repeat_site_14d"] = (result["repeat_site_14d"] - 1).clip(lower=0)
    result["repeat_site_28d"] = (result["repeat_site_28d"] - 1).clip(lower=0)

    # Fill any missing values
    result["repeat_site_14d"] = result["repeat_site_14d"].fillna(0.0)
    result["repeat_site_28d"] = result["repeat_site_28d"].fillna(0.0)

    # === TEXT FEATURES (TF-IDF) ===
    tfidf_matrix = None
    vectorizer = None

    try:
        tfidf_matrix, vectorizer = make_descriptor_tfidf(
            result, col="descriptor_clean", min_df=5, ngram_range=(1, 2), max_features=500
        )
    except Exception as e:
        print(f"Warning: TF-IDF failed: {e}")
        tfidf_matrix = None
        vectorizer = None

    # === SELECT FEATURE COLUMNS ===
    numeric_features = [
        "hour",
        "dow",
        "month",
        "is_created_at_midnight",
        "is_weekend",
        "due_gap_hours",
        "due_is_60d",
        "due_crosses_weekend",
        "tavg",
        "prcp",
        "heat_flag",
        "freeze_flag",
        "geo_family_roll7",
        "geo_family_roll28",
        "days_since_last_geo_family",
        "repeat_site_14d",
        "repeat_site_28d",
    ]

    # Get one-hot columns
    onehot_cols = [
        c for c in result.columns if any(c.startswith(f"{cat}_") for cat in categorical_cols)
    ]

    # Combine all feature columns
    feature_cols = (
        ["unique_key"] + [c for c in numeric_features if c in result.columns] + onehot_cols
    )

    # Fill missing numeric features
    for col in numeric_features:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    features_df = result[feature_cols].copy()

    return features_df, tfidf_matrix, vectorizer


def build_duration_survival_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build labels for duration/survival modeling with censoring.

    Args:
        df: Input DataFrame with created_date and closed_date

    Returns:
        DataFrame with duration labels and censoring indicators
    """
    result = df.copy()

    result["duration_days"] = (
        result["closed_date"] - result["created_date"]
    ).dt.total_seconds() / 86400

    # Initialize censoring
    result["event_observed"] = 0
    result["ttc_days_cens"] = result["duration_days"]
    result["is_admin_like"] = 0

    # Mark admin auto-close (59-62 days)
    admin_mask = (result["duration_days"] >= 59) & (result["duration_days"] <= 62)
    result.loc[admin_mask, "is_admin_like"] = 1
    result.loc[admin_mask, "ttc_days_cens"] = 60.5
    result.loc[admin_mask, "event_observed"] = 0  # Censored

    # Mark long stale (>365 days)
    stale_mask = result["duration_days"] > 365
    result.loc[stale_mask, "ttc_days_cens"] = 365
    result.loc[stale_mask, "event_observed"] = 0  # Censored

    # Mark true closures (not censored)
    true_close_mask = result["closed_date"].notna() & ~admin_mask & ~stale_mask
    result.loc[true_close_mask, "event_observed"] = 1

    # Handle missing closed_date (open tickets)
    missing_close_mask = result["closed_date"].isna()
    result.loc[missing_close_mask, "ttc_days_cens"] = 365  # Censor at 1 year
    result.loc[missing_close_mask, "event_observed"] = 0

    # Select output columns
    output_cols = [
        "unique_key",
        "duration_days",
        "ttc_days_cens",
        "event_observed",
        "is_admin_like",
    ]

    return result[output_cols].copy()


def build_duration_features(df: pd.DataFrame, triage_features: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for duration prediction (combines triage + queue pressure).

    Args:
        df: Input DataFrame with H3 keys and temporal features
        panel: Optional forecast panel for queue metrics

    Returns:
        DataFrame with duration features keyed by unique_key
    """
    result = df.copy()

    # === INTAKE PRESSURE FEATURES ===
    # Compute rolling intake counts in same region+family
    result = compute_time_based_rolling_counts(
        result,
        timestamp_col="created_date",
        group_cols=["fips", "complaint_family"],
        window_hours=[6, 24],
    )

    # === OPEN BACKLOG PROXY ===
    # Approximate open items in same hex+family
    # Reuse the geo_family_roll7 we already computed, subtract 1 to exclude current ticket
    result["open_7d_geo_family"] = (result["geo_family_roll7"] - 1).clip(lower=0)

    queue_features = result[["unique_key", "intake_6h", "intake_24h", "open_7d_geo_family"]].copy()

    final_features = triage_features.merge(queue_features, on="unique_key", how="left")

    return final_features
