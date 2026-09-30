import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient and run FastAPI startup/shutdown events."""
    with TestClient(app) as test_client:
        yield test_client


def get_valid_payload():
    """Return a valid request payload for the prediction API."""

    return {
        "Age": 65,
        "Physical_Health": 2,
        "Mental_Health": 2,
        "Dental_Health": 2,
        "Employment": 3,
        "Stress_Keeps_Patient_from_Sleeping": 0,
        "Medication_Keeps_Patient_from_Sleeping": 0,
        "Pain_Keeps_Patient_from_Sleeping": 0,
        "Bathroom_Needs_Keeps_Patient_from_Sleeping": 0,
        "Uknown_Keeps_Patient_from_Sleeping": 0,
        "Trouble_Sleeping": 2,
        "Prescription_Sleep_Medication": 3,
        "Race": 1,
        "Gender": 1,
    }


def test_root_endpoint(client):
    """Verify that the root endpoint is available."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint(client):
    """Verify API health and model status."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "v2"


def test_prediction_endpoint(client):
    """Verify that the prediction endpoint returns a valid response."""

    response = client.post(
        "/predict",
        json=get_valid_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "risk_level" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "model_version" in data


def test_prediction_class_is_valid(client):
    """Verify that the predicted health-risk class is valid."""

    response = client.post(
        "/predict",
        json=get_valid_payload(),
    )

    data = response.json()

    assert data["prediction"] in [0, 1, 2]

    assert data["risk_level"] in [
        "Low Risk",
        "Moderate Risk",
        "High Risk",
    ]


def test_prediction_probabilities_are_valid(client):
    """Verify that prediction probabilities are valid."""

    response = client.post(
        "/predict",
        json=get_valid_payload(),
    )

    data = response.json()

    probabilities = data["probabilities"]

    assert "Low Risk" in probabilities
    assert "Moderate Risk" in probabilities
    assert "High Risk" in probabilities

    total_probability = sum(probabilities.values())

    # API returns probabilities rounded to 4 decimal places.
    assert abs(total_probability - 1.0) < 0.001


def test_invalid_prediction_request(client):
    """Verify that invalid input is rejected by the API."""

    invalid_payload = get_valid_payload()

    # Age cannot be greater than 120.
    invalid_payload["Age"] = 150

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422