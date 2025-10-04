"""
Evaluation metrics and visualization for Forecast, Triage, and Duration models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, 
    precision_recall_curve, confusion_matrix
)
from lifelines.utils import concordance_index
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


def eval_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    family: str = "Unknown",
    horizon: int = 1
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
    # Remove NaN
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {}
    
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
        'family': family,
        'horizon': horizon,
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'poisson_deviance': float(poisson_dev),
        'n_samples': len(y_true)
    }
    
    return metrics


def plot_forecast_calibration(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Forecast Calibration",
    output_path: Optional[Path] = None
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
    axes[0].plot([0, max_val], [0, max_val], 'r--', label='Perfect calibration')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'{title} - Scatter')
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
    
    axes[1].plot(bin_means_pred, bin_means_true, 'o-', label='Binned mean')
    axes[1].plot([0, max(bin_means_pred)], [0, max(bin_means_pred)], 'r--', label='Perfect')
    axes[1].set_xlabel('Mean Predicted (binned)')
    axes[1].set_ylabel('Mean Actual')
    axes[1].set_title(f'{title} - Calibration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration plot -> {output_path}")
    else:
        plt.show()
    
    plt.close()


def eval_triage(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    families: Optional[pd.Series] = None
) -> Dict:
    """
    Evaluate triage model predictions.
    
    Args:
        y_true: True labels (binary)
        y_pred_proba: Predicted probabilities
        families: Complaint families for per-family metrics
    
    Returns:
        Dict with AUC-ROC, AUC-PR, and optionally per-family metrics
    """
    metrics = {}
    
    # Overall metrics
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    
    metrics['overall'] = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'n_samples': len(y_true),
        'positive_rate': float(y_true.mean())
    }
    
    # Per-family metrics
    if families is not None:
        metrics['per_family'] = {}
        
        for family in families.unique():
            mask = families == family
            if mask.sum() < 30 or y_true[mask].sum() < 5:
                continue
            
            try:
                fam_auc_roc = roc_auc_score(y_true[mask], y_pred_proba[mask])
                fam_auc_pr = average_precision_score(y_true[mask], y_pred_proba[mask])
                
                metrics['per_family'][family] = {
                    'auc_roc': float(fam_auc_roc),
                    'auc_pr': float(fam_auc_pr),
                    'n_samples': int(mask.sum()),
                    'positive_rate': float(y_true[mask].mean())
                }
            except Exception as e:
                print(f"Failed to compute metrics for {family}: {e}")
    
    return metrics


def plot_triage_curves(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    title: str = "Triage Model",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot ROC and PR curves for triage model.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    axes[0].plot(fpr, tpr, label=f'AUC = {auc_roc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'{title} - ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    
    axes[1].plot(recall, precision, label=f'AUC = {auc_pr:.3f}')
    axes[1].axhline(y_true.mean(), color='k', linestyle='--', label='Random')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'{title} - PR Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Calibration curve
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins[:-1]) - 1
    
    bin_means_pred = []
    bin_means_true = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means_pred.append(y_pred_proba[mask].mean())
            bin_means_true.append(y_true[mask].mean())
    
    axes[2].plot(bin_means_pred, bin_means_true, 'o-', label='Model')
    axes[2].plot([0, 1], [0, 1], 'k--', label='Perfect')
    axes[2].set_xlabel('Mean Predicted Probability')
    axes[2].set_ylabel('Fraction of Positives')
    axes[2].set_title(f'{title} - Calibration')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved triage curves -> {output_path}")
    else:
        plt.show()
    
    plt.close()


def eval_duration(
    t_true: pd.Series,
    t_pred: pd.Series,
    e_true: pd.Series,
    quantile: float = 0.5
) -> Dict:
    """
    Evaluate duration model predictions.
    
    Args:
        t_true: True durations
        t_pred: Predicted durations (e.g., median or q90)
        e_true: Event indicators (1=observed, 0=censored)
        quantile: Which quantile was predicted (0.5=median, 0.9=q90)
    
    Returns:
        Dict with concordance index, MAE, coverage
    """
    # C-index (on all data)
    c_index = concordance_index(t_true, t_pred, e_true)
    
    # MAE on observed events only
    observed_mask = e_true == 1
    if observed_mask.sum() > 0:
        mae = np.mean(np.abs(t_true[observed_mask] - t_pred[observed_mask]))
        
        # Coverage: for q90, what % of observed events fall below prediction?
        if quantile == 0.9:
            coverage = (t_true[observed_mask] <= t_pred[observed_mask]).mean()
        else:
            coverage = None
    else:
        mae = None
        coverage = None
    
    metrics = {
        'c_index': float(c_index),
        'mae': float(mae) if mae is not None else None,
        'coverage': float(coverage) if coverage is not None else None,
        'quantile': quantile,
        'n_samples': len(t_true),
        'n_observed': int(observed_mask.sum()),
        'event_rate': float(e_true.mean())
    }
    
    return metrics


def plot_duration_reliability(
    t_true: pd.Series,
    t_pred_q50: pd.Series,
    t_pred_q90: pd.Series,
    e_true: pd.Series,
    title: str = "Duration Model",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot reliability plots for duration predictions.
    
    Args:
        t_true: True durations
        t_pred_q50: Predicted median durations
        t_pred_q90: Predicted 90th percentile durations
        e_true: Event indicators
        title: Plot title
        output_path: Path to save figure
    """
    observed_mask = e_true == 1
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Q50 calibration (observed only)
    if observed_mask.sum() > 0:
        axes[0].scatter(t_pred_q50[observed_mask], t_true[observed_mask], alpha=0.3, s=10)
        max_val = max(t_true[observed_mask].max(), t_pred_q50[observed_mask].max())
        axes[0].plot([0, max_val], [0, max_val], 'r--', label='Perfect')
        axes[0].set_xlabel('Predicted Q50 (days)')
        axes[0].set_ylabel('Actual (days)')
        axes[0].set_title(f'{title} - Q50 Calibration')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Q90 coverage
    if observed_mask.sum() > 0:
        bins = np.linspace(0, t_true[observed_mask].max(), 20)
        coverage = []
        bin_centers = []
        
        for i in range(len(bins) - 1):
            mask = (t_true[observed_mask] >= bins[i]) & (t_true[observed_mask] < bins[i+1])
            if mask.sum() > 5:
                coverage.append((t_true[observed_mask][mask] <= t_pred_q90[observed_mask][mask]).mean())
                bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        axes[1].plot(bin_centers, coverage, 'o-', label='Actual coverage')
        axes[1].axhline(0.9, color='r', linestyle='--', label='Expected (90%)')
        axes[1].set_xlabel('Duration Bin (days)')
        axes[1].set_ylabel('Coverage')
        axes[1].set_title(f'{title} - Q90 Coverage')
        axes[1].set_ylim([0, 1])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Error distribution
    if observed_mask.sum() > 0:
        errors = t_true[observed_mask] - t_pred_q50[observed_mask]
        axes[2].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[2].axvline(0, color='r', linestyle='--', label='Perfect')
        axes[2].set_xlabel('Error (Actual - Predicted, days)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'{title} - Error Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved duration reliability plot -> {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_metrics(metrics: Dict, output_path: Path) -> None:
    """
    Save metrics dict to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Output JSON path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved metrics -> {output_path}")


def print_metrics_summary(metrics: Dict, model_type: str) -> None:
    """
    Print formatted summary of metrics.
    
    Args:
        metrics: Metrics dictionary
        model_type: 'forecast', 'triage', or 'duration'
    """
    print(f"\n{'='*60}")
    print(f"{model_type.upper()} MODEL EVALUATION")
    print(f"{'='*60}")
    
    if model_type == 'forecast':
        print(f"RMSE: {metrics.get('rmse', 'N/A'):.3f}")
        print(f"MAE: {metrics.get('mae', 'N/A'):.3f}")
        print(f"MAPE: {metrics.get('mape', 'N/A'):.1f}%")
        print(f"Poisson Deviance: {metrics.get('poisson_deviance', 'N/A'):.3f}")
    
    elif model_type == 'triage':
        overall = metrics.get('overall', {})
        print(f"AUC-ROC: {overall.get('auc_roc', 'N/A'):.3f}")
        print(f"AUC-PR: {overall.get('auc_pr', 'N/A'):.3f}")
        print(f"Positive Rate: {overall.get('positive_rate', 'N/A'):.1%}")
        
        if 'per_family' in metrics:
            print(f"\nPer-family metrics available for {len(metrics['per_family'])} families")
    
    elif model_type == 'duration':
        print(f"C-Index: {metrics.get('c_index', 'N/A'):.3f}")
        print(f"MAE (Q50): {metrics.get('mae', 'N/A'):.1f} days")
        if metrics.get('coverage') is not None:
            print(f"Q90 Coverage: {metrics['coverage']:.1%}")
        print(f"Event Rate: {metrics.get('event_rate', 'N/A'):.1%}")
    
    print(f"{'='*60}\n")

