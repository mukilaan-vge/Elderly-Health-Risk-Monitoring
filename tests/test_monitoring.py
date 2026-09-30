import json

import numpy as np

from src.monitoring.drift_detector import (
    calculate_psi,
    classify_drift,
)
from src.monitoring.logger import (
    get_prediction_count,
    log_prediction,
    read_prediction_logs,
)


def test_psi_no_drift():
    """Identical distributions should produce no meaningful drift."""

    reference = np.array(
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    )

    current = np.array(
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    )

    psi = calculate_psi(reference, current)

    assert np.isfinite(psi)
    assert psi >= 0
    assert psi < 0.10


def test_psi_detects_distribution_change():
    """A changed distribution should produce a positive PSI."""

    reference = np.array(
        [1, 1, 1, 1, 1,
         2, 2, 2, 2, 2,
         3, 3, 3, 3, 3,
         4, 4, 4, 4, 4,
         5, 5, 5, 5, 5]
    )

    current = np.array(
        [4, 4, 4, 4, 4,
         5, 5, 5, 5, 5,
         5, 5, 5, 5, 5,
         5, 5, 5, 5, 5,
         5, 5, 5, 5, 5]
    )

    psi = calculate_psi(reference, current)

    assert np.isfinite(psi)
    assert psi > 0


def test_drift_classification():
    """Verify PSI threshold classification."""

    assert classify_drift(0.05) == "No Significant Drift"
    assert classify_drift(0.15) == "Moderate Drift"
    assert classify_drift(0.30) == "Significant Drift"


def test_prediction_logger(tmp_path):
    """Verify that predictions are written to the monitoring log."""

    log_path = tmp_path / "test_predictions.jsonl"

    input_features = {
        "Age": 65,
        "Physical_Health": 2,
        "Mental_Health": 2,
    }

    log_prediction(
        input_data=input_features,
        prediction=2,
        probabilities={
            "Low Risk": 0.2,
            "Moderate Risk": 0.3,
            "High Risk": 0.5,
        },
        model_version="v2",
        response_time_ms=20.5,
        log_path=log_path,
    )

    assert log_path.exists()

    with open(log_path, "r", encoding="utf-8") as file:
        record = json.loads(file.readline())

    # Match the monitoring schema used by the project.
    assert "input_features" in record
    assert record["input_features"]["Age"] == 65

    assert record["prediction"] == 2
    assert record["model_version"] == "v2"


def test_read_prediction_logs(tmp_path):
    """Verify that prediction logs can be read back."""

    log_path = tmp_path / "test_predictions.jsonl"

    for prediction in [0, 1, 2]:

        log_prediction(
            input_data={"Age": 65},
            prediction=prediction,
            probabilities={
                "Low Risk": 0.3,
                "Moderate Risk": 0.3,
                "High Risk": 0.4,
            },
            model_version="v2",
            log_path=log_path,
        )

    logs = read_prediction_logs(log_path)

    assert len(logs) == 3
    assert logs[0]["prediction"] == 0
    assert logs[1]["prediction"] == 1
    assert logs[2]["prediction"] == 2


def test_prediction_count(tmp_path):
    """Verify prediction count from monitoring logs."""

    log_path = tmp_path / "test_predictions.jsonl"

    for _ in range(5):

        log_prediction(
            input_data={"Age": 65},
            prediction=1,
            model_version="v2",
            log_path=log_path,
        )

    count = get_prediction_count(log_path)

    assert count == 5


def test_monitoring_log_contains_required_fields(tmp_path):
    """Verify important fields in a monitoring record."""

    log_path = tmp_path / "test_predictions.jsonl"

    log_prediction(
        input_data={"Age": 70},
        prediction=1,
        probabilities={
            "Low Risk": 0.2,
            "Moderate Risk": 0.6,
            "High Risk": 0.2,
        },
        model_version="v2",
        response_time_ms=15.2,
        log_path=log_path,
    )

    logs = read_prediction_logs(log_path)

    assert len(logs) == 1

    record = logs[0]

    assert "timestamp_utc" in record
    assert "input_features" in record
    assert "prediction" in record
    assert "risk_tier" in record
    assert "model_version" in record