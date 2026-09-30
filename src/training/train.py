"""Model training, evaluation, MLflow experiment tracking, and model registry module.

Primary Model: Random Forest Classifier
Baselines: Logistic Regression, Decision Tree
Experiment Tracking: MLflow (local SQLite tracking uri)
Model Registry: MLflow Model Registry + Local Model Version Metadata
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
)
from src.preprocessing.preprocessor import TARGET_NAMES, run_preprocessing_pipeline
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("training")


def setup_mlflow(config: Dict[str, Any]) -> str:
    """Initialize local MLflow tracking server configuration."""
    tracking_uri = config["mlflow"]["tracking_uri"]
    exp_name = config["mlflow"]["experiment_name"]

    # If tracking URI is relative sqlite, resolve it
    if tracking_uri.startswith("sqlite:///"):
        rel_db = tracking_uri.replace("sqlite:///", "")
        abs_db = resolve_path(rel_db)
        abs_db.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{abs_db.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    logger.info(f"MLflow Tracking URI: {tracking_uri} | Experiment: {exp_name}")
    return tracking_uri


def load_processed_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """Load preprocessed training and test datasets."""
    train_path = resolve_path(config["data"]["processed_train_path"])
    test_path = resolve_path(config["data"]["processed_test_path"])

    if not train_path.exists() or not test_path.exists():
        logger.info("Processed data not found. Running preprocessing pipeline...")
        run_preprocessing_pipeline(config)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = config["data"]["target_column"]
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, y_train, X_test, y_test, feature_cols


def train_single_run(
    model_name: str,
    model: Any,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    register_model: bool = False,
    registered_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a single model, evaluate, log everything to MLflow, and optionally register."""
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        logger.info(f"Starting MLflow Run [{run_id}] for model '{model_name}'...")

        # 1. Fit Model
        model.fit(X_train, y_train)

        # 2. Predict on Test Set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # 3. Compute Metrics
        metrics = compute_classification_metrics(y_test.values, y_pred, y_prob)
        logger.info(
            f"Results for '{model_name}': Accuracy={metrics['accuracy']:.4f}, "
            f"F1 Macro={metrics['f1_macro']:.4f}, F1 Weighted={metrics['f1_weighted']:.4f}"
        )

        # 4. Log Parameters and Metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_param("model_family", model.__class__.__name__)
        mlflow.log_param("num_features", len(feature_names))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        for k in ["accuracy", "precision_macro", "precision_weighted", "recall_macro", "recall_weighted", "f1_macro", "f1_weighted"]:
            mlflow.log_metric(k, metrics[k])

        if metrics.get("roc_auc_macro_ovr") is not None:
            mlflow.log_metric("roc_auc_macro_ovr", metrics["roc_auc_macro_ovr"])

        # 5. Log Model to MLflow Artifacts
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name=registered_name if register_model else None,
        )

        metrics["run_id"] = run_id
        metrics["model_name"] = model_name
        metrics["params"] = params

        return metrics


def train_and_evaluate_all_models(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train primary Random Forest, tune parameters, compare baselines, log runs, and register champion."""
    if config is None:
        config = load_config()

    setup_mlflow(config)
    X_train, y_train, X_test, y_test, feature_names = load_processed_data(config)
    class_names = [TARGET_NAMES[i] for i in sorted(y_test.unique())]

    all_results: Dict[str, Dict[str, Any]] = {}

    # Run 1: Baseline - Logistic Regression
    lr_params = {"max_iter": 1000, "random_state": 42, "class_weight": "balanced"}
    lr_model = LogisticRegression(**lr_params)
    all_results["Logistic_Regression"] = train_single_run(
        "Baseline_LogisticRegression",
        lr_model,
        lr_params,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
    )

    # Run 2: Baseline - Decision Tree
    dt_params = {"max_depth": 4, "random_state": 42, "class_weight": "balanced"}
    dt_model = DecisionTreeClassifier(**dt_params)
    all_results["Decision_Tree"] = train_single_run(
        "Baseline_DecisionTree",
        dt_model,
        dt_params,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
    )

    # Run 3: Random Forest (Candidate 1 - Shallow)
    rf_shallow_params = {
        "n_estimators": 50,
        "max_depth": 4,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42,
    }
    rf_shallow = RandomForestClassifier(**rf_shallow_params)
    all_results["RandomForest_v1_Shallow"] = train_single_run(
        "RandomForest_v1_Shallow",
        rf_shallow,
        rf_shallow_params,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
    )

    # Run 4: Random Forest (Champion Model - Tuned)
    reg_name = config["mlflow"]["registered_model_name"]
    rf_champ_params = {
        "n_estimators": 120,
        "max_depth": 6,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
    }
    rf_champ = RandomForestClassifier(**rf_champ_params)
    all_results["RandomForest_v2_Champion"] = train_single_run(
        "RandomForest_v2_Champion",
        rf_champ,
        rf_champ_params,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        register_model=True,
        registered_name=reg_name,
    )

    # Fit final champion model on full training set and persist artifacts
    final_model_path = resolve_path(config["training"]["model_path"])
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_champ, final_model_path)
    logger.info(f"Champion Random Forest model serialized to: {final_model_path}")

    # Generate Visualizations
    cm_path = resolve_path(config["training"]["confusion_matrix_path"])
    y_test_pred = rf_champ.predict(X_test)
    plot_confusion_matrix(y_test.values, y_test_pred, class_names, cm_path)

    fi_path = resolve_path(config["training"]["feature_importance_path"])
    plot_feature_importance(rf_champ, feature_names, fi_path)

    comp_path = resolve_path("models/model_comparison.png")
    plot_model_comparison(all_results, comp_path)

    # Save metrics JSON
    metrics_path = resolve_path(config["training"]["metrics_path"])
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Evaluation metrics saved to: {metrics_path}")

    # Save lightweight Model Registry Metadata JSON
    registry_path = resolve_path("models/model_registry.json")
    registry_meta = {
        "registered_model_name": reg_name,
        "current_active_version": "v2",
        "champion_model_type": "RandomForestClassifier",
        "champion_run_id": all_results["RandomForest_v2_Champion"]["run_id"],
        "status": "Production",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "performance": {
            "accuracy": all_results["RandomForest_v2_Champion"]["accuracy"],
            "f1_macro": all_results["RandomForest_v2_Champion"]["f1_macro"],
            "f1_weighted": all_results["RandomForest_v2_Champion"]["f1_weighted"],
            "roc_auc": all_results["RandomForest_v2_Champion"].get("roc_auc_macro_ovr"),
        },
        "hyperparameters": rf_champ_params,
        "feature_names": feature_names,
        "target_classes": TARGET_NAMES,
        "version_history": [
            {
                "version": "v1",
                "model_name": "RandomForest_v1_Shallow",
                "accuracy": all_results["RandomForest_v1_Shallow"]["accuracy"],
                "f1_macro": all_results["RandomForest_v1_Shallow"]["f1_macro"],
                "status": "Archived",
            },
            {
                "version": "v2",
                "model_name": "RandomForest_v2_Champion",
                "accuracy": all_results["RandomForest_v2_Champion"]["accuracy"],
                "f1_macro": all_results["RandomForest_v2_Champion"]["f1_macro"],
                "status": "Production",
            },
        ],
    }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry_meta, f, indent=4)
    logger.info(f"Model Registry metadata saved: {registry_path}")

    return {
        "results": all_results,
        "registry": registry_meta,
    }


def load_champion_model(config: Optional[Dict[str, Any]] = None) -> Any:
    """Load active production champion model from disk."""
    if config is None:
        config = load_config()
    model_path = resolve_path(config["training"]["model_path"])
    if not model_path.exists():
        train_and_evaluate_all_models(config)
    return joblib.load(model_path)


if __name__ == "__main__":
    out = train_and_evaluate_all_models()
    print("=" * 60)
    print("PHASES 3, 4, 5 TRAINING & EXPERIMENT SUMMARY")
    print("=" * 60)
    for model_name, res in out["results"].items():
        print(
            f"{model_name:25s} | Accuracy: {res['accuracy']:.4f} | "
            f"F1 Macro: {res['f1_macro']:.4f} | F1 Weighted: {res['f1_weighted']:.4f}"
        )
    print("\nModel Registry Active Champion:")
    print(f"Model Name: {out['registry']['registered_model_name']}")
    print(f"Version:    {out['registry']['current_active_version']}")
    print(f"Status:     {out['registry']['status']}")


def train_models(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compatibility wrapper used by the main pipeline.

    Calls the complete training, evaluation, MLflow tracking,
    and model registry workflow.
    """
    return train_and_evaluate_all_models(config)