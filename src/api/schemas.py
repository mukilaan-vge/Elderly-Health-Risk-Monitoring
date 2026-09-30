"""
Pydantic schemas for the Elderly Health Risk Monitoring API.

These schemas define:
- API input validation
- Prediction response format
- Health-check response
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class HealthRiskRequest(BaseModel):
    """
    Input data required to generate an elderly health-risk prediction.
    """

    Age: float = Field(
        ...,
        ge=0,
        le=120,
        description="Age of the elderly person."
    )

    Physical_Health: float = Field(
        ...,
        description="Physical health score."
    )

    Mental_Health: float = Field(
        ...,
        description="Mental health score."
    )

    Dental_Health: float = Field(
        ...,
        description="Dental health score."
    )

    Employment: float = Field(
        ...,
        description="Employment-related health survey value."
    )

    Stress_Keeps_Patient_from_Sleeping: float = Field(
        ...,
        description="Whether stress affects sleep."
    )

    Medication_Keeps_Patient_from_Sleeping: float = Field(
        ...,
        description="Whether medication affects sleep."
    )

    Pain_Keeps_Patient_from_Sleeping: float = Field(
        ...,
        description="Whether pain affects sleep."
    )

    Bathroom_Needs_Keeps_Patient_from_Sleeping: float = Field(
        ...,
        description="Whether bathroom needs affect sleep."
    )

    Uknown_Keeps_Patient_from_Sleeping: float = Field(
        ...,
        description="Unknown reason affecting sleep."
    )

    Trouble_Sleeping: float = Field(
        ...,
        description="Trouble sleeping indicator."
    )

    Prescription_Sleep_Medication: float = Field(
        ...,
        description="Prescription sleep medication indicator."
    )

    Race: float = Field(
        ...,
        description="Race category encoded as a numerical value."
    )

    Gender: float = Field(
        ...,
        description="Gender category encoded as a numerical value."
    )


class PredictionResponse(BaseModel):
    """
    Response returned by the prediction endpoint.
    """

    prediction: int = Field(
        ...,
        description="Predicted health-risk class."
    )

    risk_level: str = Field(
        ...,
        description="Human-readable health-risk category."
    )

    confidence: Optional[float] = Field(
        None,
        description="Prediction confidence."
    )

    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Probability for each health-risk class."
    )

    model_version: str = Field(
        ...,
        description="Version of the model used for prediction."
    )


class HealthResponse(BaseModel):
    """
    Response returned by the API health endpoint.
    """

    status: str

    model_loaded: bool

    model_version: str