"""
FastAPI application for Elderly Health Risk Monitoring.

Endpoints:
    GET  /
    GET  /health
    POST /predict
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    HealthRiskRequest,
    PredictionResponse,
    HealthResponse,
)
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger


# ============================================================
# CONFIGURATION
# ============================================================

config = load_config()

logger = get_logger("api")


# ============================================================
# PROJECT PATHS
# ============================================================

MODEL_PATH = resolve_path(
    config["training"]["model_path"]
)

PREPROCESSOR_PATH = resolve_path(
    config["data"]["preprocessor_path"]
)

REGISTRY_PATH = resolve_path(
    "models/model_registry.json"
)

API_LOG_PATH = resolve_path(
    config["api"]["log_path"]
)


# ============================================================
# RISK CLASS NAMES
# ============================================================

TARGET_NAMES = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
}


# ============================================================
# GLOBAL MODEL OBJECTS
# ============================================================

model = None
preprocessor = None

model_version = "unknown"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model_artifacts() -> None:
    """
    Load the trained model, preprocessing pipeline,
    and model version information.
    """

    global model
    global preprocessor
    global model_version


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )


    # --------------------------------------------------------
    # Check preprocessor
    # --------------------------------------------------------

    if not PREPROCESSOR_PATH.exists():

        raise FileNotFoundError(
            f"Preprocessor not found at: "
            f"{PREPROCESSOR_PATH}"
        )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    logger.info(
        f"Loading model from: {MODEL_PATH}"
    )

    model = joblib.load(
        MODEL_PATH
    )


    # --------------------------------------------------------
    # Load preprocessor
    # --------------------------------------------------------

    logger.info(
        f"Loading preprocessor from: "
        f"{PREPROCESSOR_PATH}"
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )


    # --------------------------------------------------------
    # Load model registry information
    # --------------------------------------------------------

    if REGISTRY_PATH.exists():

        try:

            with open(
                REGISTRY_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                registry = json.load(file)

            model_version = registry.get(
                "current_active_version",
                "unknown"
            )

        except Exception as exc:

            logger.warning(
                f"Could not read model registry: {exc}"
            )

            model_version = "unknown"

    else:

        model_version = "unknown"


    logger.info(
        f"Active model version: {model_version}"
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=config["api"]["title"],
    version=config["api"]["version"],
    description=(
        "REST API for predicting elderly health risk "
        "using the production Random Forest model."
    ),
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event() -> None:
    """
    Load model artifacts when the API starts.
    """

    try:

        load_model_artifacts()

        logger.info(
            "API startup completed successfully."
        )

    except Exception as exc:

        logger.error(
            f"API startup failed: {exc}"
        )

        # We do not crash the server here.
        #
        # /health will report model_loaded=False and
        # /predict will return an appropriate error.
        #
        # This makes debugging easier when running locally.


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root() -> Dict[str, Any]:
    """
    Basic API information.
    """

    return {
        "service": "Elderly Health Risk Prediction API",
        "status": "running",
        "version": config["api"]["version"],
        "model_version": model_version,
        "endpoints": {
            "health": "/health",
            "prediction": "/predict",
            "documentation": "/docs",
        },
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check() -> HealthResponse:
    """
    Check whether the API and model are ready.
    """

    return HealthResponse(
        status=(
            "healthy"
            if model is not None and preprocessor is not None
            else "degraded"
        ),
        model_loaded=(
            model is not None
            and preprocessor is not None
        ),
        model_version=model_version,
    )


# ============================================================
# LOG API REQUEST
# ============================================================

def log_prediction_request(
    input_data: Dict[str, Any],
    prediction: int,
    probabilities: Dict[str, float],
    response_time_ms: float,
) -> None:
    """
    Store prediction information in JSON Lines format.

    One prediction = one JSON object per line.
    """

    API_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    log_entry = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "input": input_data,

        "prediction": prediction,

        "risk_level": TARGET_NAMES.get(
            prediction,
            "Unknown"
        ),

        "probabilities": probabilities,

        "model_version": model_version,

        "response_time_ms": round(
            response_time_ms,
            3
        ),
    }


    with open(
        API_LOG_PATH,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                log_entry
            ) + "\n"
        )


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: HealthRiskRequest
) -> PredictionResponse:
    """
    Generate an elderly health-risk prediction.
    """

    global model
    global preprocessor


    # --------------------------------------------------------
    # Check model availability
    # --------------------------------------------------------

    if model is None or preprocessor is None:

        try:

            load_model_artifacts()

        except Exception as exc:

            logger.error(
                f"Model loading failed: {exc}"
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Prediction service is unavailable "
                    "because the model or preprocessor "
                    "could not be loaded."
                ),
            )


    start_time = time.perf_counter()


    try:

        # ----------------------------------------------------
        # Convert Pydantic request to dictionary
        # ----------------------------------------------------

        input_data = request.model_dump()


        # ----------------------------------------------------
        # Convert input into DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )


        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------
        #
        # The model was trained using three engineered features:
        #
        # Sleep_Disturbance_Score
        # Health_Deficit_Score
        # High_Pain_Flag
        #
        # We must recreate these features during inference.

        sleep_columns = [
            "Stress_Keeps_Patient_from_Sleeping",
            "Medication_Keeps_Patient_from_Sleeping",
            "Pain_Keeps_Patient_from_Sleeping",
            "Bathroom_Needs_Keeps_Patient_from_Sleeping",
            "Uknown_Keeps_Patient_from_Sleeping",
            "Trouble_Sleeping",
            "Prescription_Sleep_Medication",
        ]


        health_columns = [
            "Physical_Health",
            "Mental_Health",
            "Dental_Health",
        ]


        # ----------------------------------------------------
        # Sleep disturbance score
        # ----------------------------------------------------

        input_df["Sleep_Disturbance_Score"] = (
            input_df[sleep_columns]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .sum(
                axis=1,
                skipna=True
            )
        )


        # ----------------------------------------------------
        # Health deficit score
        # ----------------------------------------------------

        input_df["Health_Deficit_Score"] = (
            input_df[health_columns]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .sum(
                axis=1,
                skipna=True
            )
        )


        # ----------------------------------------------------
        # High pain flag
        # ----------------------------------------------------

        pain_values = pd.to_numeric(
            input_df[
                "Pain_Keeps_Patient_from_Sleeping"
            ],
            errors="coerce"
        )


        input_df["High_Pain_Flag"] = (
            pain_values
            .fillna(0)
            .astype(int)
        )


        # ----------------------------------------------------
        # Apply fitted preprocessing
        # ----------------------------------------------------

        processed_input = (
            preprocessor.transform(
                input_df
            )
        )


        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        prediction_array = model.predict(
            processed_input
        )

        prediction = int(
            prediction_array[0]
        )


        # ----------------------------------------------------
        # Generate probabilities
        # ----------------------------------------------------

        probabilities = {}


        if hasattr(
            model,
            "predict_proba"
        ):

            probability_array = (
                model.predict_proba(
                    processed_input
                )[0]
            )


            classes = getattr(
                model,
                "classes_",
                range(
                    len(probability_array)
                )
            )


            for class_value, probability in zip(
                classes,
                probability_array
            ):

                class_id = int(
                    class_value
                )

                probabilities[
                    TARGET_NAMES.get(
                        class_id,
                        str(class_id)
                    )
                ] = round(
                    float(probability),
                    4
                )


        # ----------------------------------------------------
        # Calculate confidence
        # ----------------------------------------------------

        confidence = None

        if probabilities:

            confidence = max(
                probabilities.values()
            )


        # ----------------------------------------------------
        # Response time
        # ----------------------------------------------------

        response_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        # ----------------------------------------------------
        # Log prediction
        # ----------------------------------------------------

        log_prediction_request(
            input_data=input_data,
            prediction=prediction,
            probabilities=probabilities,
            response_time_ms=response_time_ms,
        )


        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return PredictionResponse(
            prediction=prediction,

            risk_level=TARGET_NAMES.get(
                prediction,
                "Unknown"
            ),

            confidence=confidence,

            probabilities=probabilities,

            model_version=model_version,
        )


    except Exception as exc:

        logger.exception(
            f"Prediction failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. "
                f"Error: {str(exc)}"
            ),
        )
FRONTEND_PATH = resolve_path("frontend")

if FRONTEND_PATH.exists():
    app.mount(
        "/dashboard",
        StaticFiles(directory=FRONTEND_PATH, html=True),
        name="dashboard",
    )