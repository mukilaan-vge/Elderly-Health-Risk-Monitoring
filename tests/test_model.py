import joblib
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "health_risk_model.joblib"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test.csv"


def test_model_file_exists():
    """Verify that the trained model artifact exists."""

    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    """Verify that the trained model can be loaded successfully."""

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_produces_valid_predictions():
    """Verify that the model produces valid health-risk predictions."""

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_DATA)

    X = df.drop(columns=["Health_Risk_Level"])
    y_pred = model.predict(X)

    assert len(y_pred) == len(df)

    # Expected classes:
    # 0 = Low Risk
    # 1 = Moderate Risk
    # 2 = High Risk
    assert set(y_pred).issubset({0, 1, 2})


def test_model_probability_output():
    """Verify that the model produces valid class probabilities."""

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_DATA)

    X = df.drop(columns=["Health_Risk_Level"])

    probabilities = model.predict_proba(X)

    assert probabilities.shape[0] == len(df)

    # Three health-risk classes
    assert probabilities.shape[1] == 3

    # Each row's probabilities should sum to approximately 1
    probability_sums = probabilities.sum(axis=1)

    assert all(abs(total - 1.0) < 1e-6 for total in probability_sums)