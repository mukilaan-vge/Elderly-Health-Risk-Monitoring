"""
Main MLOps pipeline orchestrator.

Pipeline stages:
1. Data validation
2. Data preprocessing
3. Model training
4. Model evaluation
5. Model artifact verification
6. Pipeline completion summary

This script connects the existing project modules together.
"""

import sys
import traceback
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.utils.config import load_config
from src.utils.logger import get_logger

from src.preprocessing.validator import validate_dataset
from src.preprocessing.preprocessor import (
    run_preprocessing_pipeline,
)

from src.training.train import train_models


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(
    "pipeline"
)


# ============================================================
# PIPELINE HEADER
# ============================================================

def print_header() -> None:
    """
    Display the pipeline header.
    """

    print()
    print("=" * 75)
    print(
        "ELDERLY HEALTH RISK MONITORING"
    )
    print(
        "PRODUCTION-GRADE MLOps PIPELINE"
    )
    print("=" * 75)
    print()


# ============================================================
# STAGE 1 — DATA VALIDATION
# ============================================================

def run_validation(
    config
) -> None:
    """
    Validate the project dataset.
    """

    print()
    print("-" * 75)
    print("STAGE 1 — DATA VALIDATION")
    print("-" * 75)

    logger.info(
        "Starting data validation..."
    )

    raw_data_path = config["data"]["raw_csv_path"]

    try:

        validation_result = validate_dataset(
            raw_data_path
        )

    except TypeError:

        # Some versions of the validator may expect
        # the complete configuration rather than only
        # the dataset path.
        validation_result = validate_dataset(
            config
        )


    if validation_result is False:

        raise RuntimeError(
            "Dataset validation failed."
        )


    logger.info(
        "Data validation completed."
    )

    print(
        "✓ Data validation completed"
    )


# ============================================================
# STAGE 2 — PREPROCESSING
# ============================================================

def run_preprocessing(
    config
):
    """
    Run the preprocessing pipeline.
    """

    print()
    print("-" * 75)
    print("STAGE 2 — DATA PREPROCESSING")
    print("-" * 75)

    logger.info(
        "Starting preprocessing..."
    )

    result = run_preprocessing_pipeline(
        config
    )

    logger.info(
        "Preprocessing completed."
    )

    print(
        "✓ Data preprocessing completed"
    )

    return result


# ============================================================
# STAGE 3 — MODEL TRAINING
# ============================================================

def run_training(
    config
):
    """
    Train and evaluate the configured models.
    """

    print()
    print("-" * 75)
    print("STAGE 3 — MODEL TRAINING")
    print("-" * 75)

    logger.info(
        "Starting model training..."
    )

    result = train_models(
        config
    )

    logger.info(
        "Model training completed."
    )

    print(
        "✓ Model training completed"
    )

    return result


# ============================================================
# ARTIFACT VERIFICATION
# ============================================================

def verify_artifacts(
    config
) -> None:
    """
    Verify that important pipeline artifacts exist.
    """

    print()
    print("-" * 75)
    print("VERIFYING PIPELINE ARTIFACTS")
    print("-" * 75)


    artifacts = {
        "Processed training data":
            config["data"]["processed_train_path"],

        "Processed testing data":
            config["data"]["processed_test_path"],

        "Preprocessor":
            config["data"]["preprocessor_path"],

        "Trained model":
            config["training"]["model_path"],
    }


    missing_artifacts = []


    for name, artifact_path in artifacts.items():

        path = Path(
            artifact_path
        )


        if not path.is_absolute():

            path = (
                PROJECT_ROOT
                / path
            )


        if path.exists():

            print(
                f"✓ {name}: {path}"
            )

        else:

            print(
                f"✗ {name}: NOT FOUND"
            )

            missing_artifacts.append(
                name
            )


    if missing_artifacts:

        raise RuntimeError(
            "Missing pipeline artifacts: "
            + ", ".join(
                missing_artifacts
            )
        )


# ============================================================
# PIPELINE SUMMARY
# ============================================================

def print_summary() -> None:
    """
    Display final pipeline status.
    """

    print()
    print("=" * 75)
    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )
    print("=" * 75)

    print()
    print(
        "Completed stages:"
    )

    print(
        "  ✓ Data validation"
    )

    print(
        "  ✓ Data preprocessing"
    )

    print(
        "  ✓ Model training"
    )

    print(
        "  ✓ Artifact verification"
    )

    print()

    print(
        "Next MLOps stages:"
    )

    print(
        "  → Experiment tracking with MLflow"
    )

    print(
        "  → Model registry"
    )

    print(
        "  → FastAPI deployment"
    )

    print(
        "  → Monitoring and drift detection"
    )

    print(
        "  → CI/CD"
    )

    print()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> int:
    """
    Execute the complete MLOps pipeline.
    """

    print_header()


    try:

        # ----------------------------------------------------
        # Load configuration
        # ----------------------------------------------------

        config = load_config()


        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        run_validation(
            config
        )


        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        run_preprocessing(
            config
        )


        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        run_training(
            config
        )


        # ----------------------------------------------------
        # Verify generated artifacts
        # ----------------------------------------------------

        verify_artifacts(
            config
        )


        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        print_summary()


        return 0


    except Exception as exc:

        print()
        print("=" * 75)
        print(
            "PIPELINE FAILED"
        )
        print("=" * 75)

        print()
        print(
            f"Error: {exc}"
        )

        print()
        print(
            "Full traceback:"
        )

        traceback.print_exc()


        logger.error(
            f"Pipeline failed: {exc}"
        )


        return 1


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    exit_code = main()

    sys.exit(
        exit_code
    )