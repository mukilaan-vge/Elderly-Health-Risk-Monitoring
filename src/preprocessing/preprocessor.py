"""
Preprocessing pipeline for Elderly Health Risk Monitoring.

This module:
1. Loads the cleaned/interim NPHA dataset.
2. Uses Health_Risk_Level as the target.
3. Performs feature engineering.
4. Splits the data into train/test sets.
5. Fits preprocessing ONLY on the training data.
6. Saves processed train/test datasets.
7. Saves the fitted preprocessing object for API inference.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger


logger = get_logger("preprocessor")


# ============================================================
# TARGET CLASSES
# ============================================================

TARGET_NAMES = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
}


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

SLEEP_COLUMNS = [
    "Stress_Keeps_Patient_from_Sleeping",
    "Medication_Keeps_Patient_from_Sleeping",
    "Pain_Keeps_Patient_from_Sleeping",
    "Bathroom_Needs_Keeps_Patient_from_Sleeping",
    "Uknown_Keeps_Patient_from_Sleeping",
    "Trouble_Sleeping",
    "Prescription_Sleep_Medication",
]

HEALTH_COLUMNS = [
    "Physical_Health",
    "Mental_Health",
    "Dental_Health",
]

PAIN_COLUMN = "Pain_Keeps_Patient_from_Sleeping"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered health-related features.

    Engineered features:
        Sleep_Disturbance_Score
        Health_Deficit_Score
        High_Pain_Flag
    """

    df = df.copy()

    # --------------------------------------------------------
    # Required source columns
    # --------------------------------------------------------

    missing_sleep_columns = [
        col
        for col in SLEEP_COLUMNS
        if col not in df.columns
    ]

    if missing_sleep_columns:
        raise ValueError(
            "Missing columns required for "
            "Sleep_Disturbance_Score: "
            f"{missing_sleep_columns}"
        )

    missing_health_columns = [
        col
        for col in HEALTH_COLUMNS
        if col not in df.columns
    ]

    if missing_health_columns:
        raise ValueError(
            "Missing columns required for "
            "Health_Deficit_Score: "
            f"{missing_health_columns}"
        )

    if PAIN_COLUMN not in df.columns:
        raise ValueError(
            f"Required pain column '{PAIN_COLUMN}' "
            "was not found."
        )

    # --------------------------------------------------------
    # Safely clean numeric survey responses
    # --------------------------------------------------------

    def clean_numeric_values(
        series: pd.Series
    ) -> pd.Series:

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        numeric_series = numeric_series.mask(
            numeric_series.isin([-1, -2])
        )

        return numeric_series

    # --------------------------------------------------------
    # Sleep Disturbance Score
    # --------------------------------------------------------

    sleep_data = df[SLEEP_COLUMNS].apply(
        clean_numeric_values
    )

    df["Sleep_Disturbance_Score"] = (
        sleep_data.sum(
            axis=1,
            skipna=True
        )
    )

    # --------------------------------------------------------
    # Health Deficit Score
    # --------------------------------------------------------

    health_data = df[HEALTH_COLUMNS].apply(
        clean_numeric_values
    )

    df["Health_Deficit_Score"] = (
        health_data.sum(
            axis=1,
            skipna=True
        )
    )

    # --------------------------------------------------------
    # High Pain Flag
    # --------------------------------------------------------

    pain_values = clean_numeric_values(
        df[PAIN_COLUMN]
    )

    df["High_Pain_Flag"] = (
        pain_values
        .fillna(0)
        .astype(int)
    )

    return df

    # --------------------------------------------------------
    # Handle survey refusal codes
    # --------------------------------------------------------

    # The NPHA dataset can use negative values such as -1 and -2
    # for unavailable/refused responses.
    #
    # Do NOT call DataFrame.replace() on the complete mixed-type
    # dataframe because this can cause compatibility issues with
    # newer Pandas versions.
    #
    # Instead, handle the values only after converting the relevant
    # health columns to numeric values.


    def clean_numeric_values(series: pd.Series) -> pd.Series:
        """
        Convert a column to numeric and replace invalid survey
        response codes with NaN.
        """

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        numeric_series = numeric_series.mask(
            numeric_series.isin([-1, -2])
        )

        return numeric_series
    # --------------------------------------------------------
    # Sleep Disturbance Score
    # --------------------------------------------------------

    sleep_data = df[SLEEP_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df["Sleep_Disturbance_Score"] = (
        sleep_data.sum(
            axis=1,
            skipna=True
        )
    )


    # --------------------------------------------------------
    # Health Deficit Score
    # --------------------------------------------------------

    health_data = df[HEALTH_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df["Health_Deficit_Score"] = (
        health_data.sum(
            axis=1,
            skipna=True
        )
    )


    # --------------------------------------------------------
    # High Pain Flag
    # --------------------------------------------------------

    pain_values = pd.to_numeric(
        df[PAIN_COLUMN],
        errors="coerce"
    )

    df["High_Pain_Flag"] = (
        pain_values
        .fillna(0)
        .astype(int)
    )


    return df


# ============================================================
# TARGET PREPARATION
# ============================================================

def prepare_target(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Prepare Health_Risk_Level as the model target.

    The interim dataset already contains Health_Risk_Level.

    If the interim dataset instead contains the original
    Number_of_Doctors_Visited target, convert it into:
        0 = Low Risk
        1 = Moderate Risk
        2 = High Risk
    """

    df = df.copy()

    target_column = config["data"]["target_column"]
    raw_target_column = config["data"]["raw_target_column"]


    # --------------------------------------------------------
    # Case 1: Target already exists
    # --------------------------------------------------------

    if target_column in df.columns:

        logger.info(
            f"Using existing target column: {target_column}"
        )

        df[target_column] = pd.to_numeric(
            df[target_column],
            errors="coerce"
        )

        df = df.dropna(
            subset=[target_column]
        )

        df[target_column] = (
            df[target_column]
            .astype(int)
        )

        return df


    # --------------------------------------------------------
    # Case 2: Raw target still exists
    # --------------------------------------------------------

    if raw_target_column in df.columns:

        logger.info(
            f"Creating '{target_column}' from "
            f"'{raw_target_column}'."
        )

        raw_target = pd.to_numeric(
            df[raw_target_column],
            errors="coerce"
        )

        df[target_column] = pd.cut(
            raw_target,
            bins=[
                -np.inf,
                1,
                3,
                np.inf
            ],
            labels=[
                0,
                1,
                2
            ]
        )

        df[target_column] = (
            df[target_column]
            .astype("Int64")
        )

        df = df.dropna(
            subset=[target_column]
        )

        df[target_column] = (
            df[target_column]
            .astype(int)
        )

        return df


    raise ValueError(
        "Unable to find target column. Expected either "
        f"'{target_column}' or '{raw_target_column}'. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(
    X: pd.DataFrame
) -> ColumnTransformer:
    """
    Build the sklearn preprocessing transformer.

    Numerical features:
        Median imputation
        StandardScaler

    Categorical features:
        Most-frequent imputation
        OneHotEncoder
    """

    numeric_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )


    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )


    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )


    transformers = []


    if numeric_features:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )


    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )


    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def run_preprocessing_pipeline(
    config: Optional[Dict[str, Any]] = None
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    ColumnTransformer
]:
    """
    Execute the complete preprocessing pipeline.

    Returns:
        X_train_processed
        X_test_processed
        y_train
        y_test
        fitted_preprocessor
    """

    if config is None:
        config = load_config()


    logger.info("=" * 70)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 70)


    # ========================================================
    # PATHS
    # ========================================================

    interim_path = resolve_path(
        config["data"]["interim_csv_path"]
    )

    raw_path = resolve_path(
        config["data"]["raw_csv_path"]
    )

    train_path = resolve_path(
        config["data"]["processed_train_path"]
    )

    test_path = resolve_path(
        config["data"]["processed_test_path"]
    )

    preprocessor_path = resolve_path(
        config["data"]["preprocessor_path"]
    )


    train_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    preprocessor_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    if interim_path.exists():

        logger.info(
            f"Loading cleaned interim dataset: {interim_path}"
        )

        df = pd.read_csv(
            interim_path
        )

    elif raw_path.exists():

        logger.info(
            "Interim dataset not found."
        )

        logger.info(
            f"Loading raw dataset: {raw_path}"
        )

        df = pd.read_csv(
            raw_path
        )

    else:

        raise FileNotFoundError(
            "Dataset could not be found.\n"
            f"Expected:\n"
            f"  {interim_path}\n"
            f"or\n"
            f"  {raw_path}"
        )


    logger.info(
        f"Dataset shape before preprocessing: {df.shape}"
    )

    logger.info(
        f"Dataset columns: {list(df.columns)}"
    )


    # ========================================================
    # PREPARE TARGET
    # ========================================================

    df = prepare_target(
        df,
        config
    )


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = engineer_features(
        df
    )


    # ========================================================
    # SEPARATE FEATURES AND TARGET
    # ========================================================

    target_column = config["data"]["target_column"]

    y = df[target_column].copy()


    X = df.drop(
        columns=[
            target_column,
            config["data"]["raw_target_column"],
        ],
        errors="ignore"
    )


    # ========================================================
    # REMOVE IDENTIFIER COLUMNS
    # ========================================================

    identifier_columns = [
        "id",
        "ID",
        "patient_id",
        "Patient_ID",
        "record_id",
        "Record_ID",
        "Unnamed: 0",
    ]

    existing_identifier_columns = [
        col
        for col in identifier_columns
        if col in X.columns
    ]

    if existing_identifier_columns:

        X = X.drop(
            columns=existing_identifier_columns
        )

        logger.info(
            "Removed identifier columns: "
            f"{existing_identifier_columns}"
        )


    logger.info(
        f"Final feature matrix shape: {X.shape}"
    )

    logger.info(
        f"Target distribution:\n{y.value_counts().sort_index()}"
    )


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    test_size = config["data"].get(
        "test_size",
        0.20
    )

    random_state = config["data"].get(
        "random_state",
        42
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


    logger.info(
        f"Training samples: {len(X_train)}"
    )

    logger.info(
        f"Testing samples: {len(X_test)}"
    )


    # ========================================================
    # BUILD PREPROCESSOR
    # ========================================================

    preprocessor = build_preprocessor(
        X_train
    )


    # ========================================================
    # FIT ONLY ON TRAINING DATA
    # ========================================================

    logger.info(
        "Fitting preprocessing transformer on training data..."
    )

    X_train_processed_array = (
        preprocessor.fit_transform(
            X_train
        )
    )


    logger.info(
        "Transforming test data..."
    )

    X_test_processed_array = (
        preprocessor.transform(
            X_test
        )
    )


    # ========================================================
    # GET FEATURE NAMES
    # ========================================================

    try:

        feature_names = (
            preprocessor.get_feature_names_out()
        )

    except Exception:

        feature_names = [
            f"feature_{i}"
            for i in range(
                X_train_processed_array.shape[1]
            )
        ]


    # ========================================================
    # CONVERT TO DATAFRAMES
    # ========================================================

    X_train_processed = pd.DataFrame(
        X_train_processed_array,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed_array,
        columns=feature_names,
        index=X_test.index,
    )


    # ========================================================
    # ADD TARGET COLUMN
    # ========================================================

    train_processed = X_train_processed.copy()

    train_processed[target_column] = (
        y_train.values
    )


    test_processed = X_test_processed.copy()

    test_processed[target_column] = (
        y_test.values
    )


    # ========================================================
    # SAVE PROCESSED DATA
    # ========================================================

    train_processed.to_csv(
        train_path,
        index=False
    )

    test_processed.to_csv(
        test_path,
        index=False
    )


    logger.info(
        f"Saved processed training data: {train_path}"
    )

    logger.info(
        f"Saved processed testing data: {test_path}"
    )


    # ========================================================
    # SAVE PREPROCESSOR
    # ========================================================

    joblib.dump(
        preprocessor,
        preprocessor_path
    )


    logger.info(
        f"Saved fitted preprocessor: "
        f"{preprocessor_path}"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    logger.info(
        f"Processed feature count: "
        f"{len(feature_names)}"
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY"
    )

    logger.info(
        "=" * 70
    )


    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )


# ============================================================
# COMMAND-LINE EXECUTION
# ============================================================

if __name__ == "__main__":

    run_preprocessing_pipeline()