import pandas as pd
from pathlib import Path

from src.preprocessing.preprocessor import engineer_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DATA = PROJECT_ROOT / "data" / "interim" / "cleaned_health_risk.csv"
PROCESSED_TRAIN = PROJECT_ROOT / "data" / "processed" / "train.csv"


def test_engineer_features_creates_expected_features():
    """Verify that feature engineering creates the required derived features."""

    df = pd.read_csv(INTERIM_DATA)

    result = engineer_features(df.copy())

    expected_features = [
        "Sleep_Disturbance_Score",
        "Health_Deficit_Score",
        "High_Pain_Flag",
    ]

    for feature in expected_features:
        assert feature in result.columns


def test_engineered_features_contain_no_nan_values():
    """Verify that engineered features do not contain missing values."""

    df = pd.read_csv(INTERIM_DATA)
    result = engineer_features(df.copy())

    engineered_features = [
        "Sleep_Disturbance_Score",
        "Health_Deficit_Score",
        "High_Pain_Flag",
    ]

    assert result[engineered_features].isna().sum().sum() == 0


def test_processed_training_data_exists():
    """Verify that the preprocessing pipeline produced training data."""

    assert PROCESSED_TRAIN.exists()


def test_processed_training_data_contains_target():
    """Verify that the processed training dataset contains the target."""

    df = pd.read_csv(PROCESSED_TRAIN)

    assert "Health_Risk_Level" in df.columns


def test_processed_training_data_contains_engineered_features():
    """Verify that processed data contains all engineered features."""

    df = pd.read_csv(PROCESSED_TRAIN)

    expected_features = [
        "numeric__Sleep_Disturbance_Score",
        "numeric__Health_Deficit_Score",
        "numeric__High_Pain_Flag",
    ]

    for feature in expected_features:
        assert feature in df.columns