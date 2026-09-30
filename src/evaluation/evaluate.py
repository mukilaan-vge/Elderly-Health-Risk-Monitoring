"""Model evaluation metrics and diagnostic visualization generator."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated pipelines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger("evaluate")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification evaluation metrics.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        y_prob: Predicted class probabilities (optional, for ROC-AUC).

    Returns:
        Dict of computed scalar metrics and detailed per-class report.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics: Dict[str, Any] = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_macro": round(recall_macro, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }

    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            metrics["roc_auc_macro_ovr"] = round(roc_auc, 4)
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics["roc_auc_macro_ovr"] = None

    # Detailed per-class classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> Path:
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Confusion Matrix - Elderly Health Risk Classification", fontsize=12, pad=12)
    plt.xlabel("Predicted Risk Tier", fontsize=11)
    plt.ylabel("Actual Risk Tier", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Confusion matrix plot saved: {output_path}")
    return output_path


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    output_path: Path,
    top_n: int = 12,
) -> Optional[Path]:
    """Plot and save feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(9, 5))
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    sns.barplot(x=top_scores, y=top_features, hue=top_features, palette="viridis", legend=False)
    plt.title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=12, pad=12)
    plt.xlabel("Gini Importance Score", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Feature importance plot saved: {output_path}")
    return output_path


def plot_model_comparison(
    models_metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> Path:
    """Plot benchmark comparison bar chart between models."""
    model_names = list(models_metrics.keys())
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    x = np.arange(len(metric_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(model_names):
        values = [models_metrics[name].get(k, 0.0) for k in metric_keys]
        offset = (i - len(model_names) / 2 + 0.5) * width
        rects = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(rects, padding=3, fmt="%.2f", fontsize=8)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Architecture Comparison - Elderly Health Risk", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Model comparison plot saved: {output_path}")
    return output_path

