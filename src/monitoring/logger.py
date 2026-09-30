"""
Production prediction logger for the Elderly Health Risk Monitoring system.

Logs every API prediction in JSON Lines format so that the logs
can later be used for monitoring and drift detection.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Any

import json


# Default monitoring log location
DEFAULT_LOG_PATH = Path(
    "data/monitoring/api_requests.jsonl"
)


TARGET_NAMES = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
}


def ensure_log_directory(log_path: Path) -> None:
    """
    Create the monitoring directory if it does not exist.
    """

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def make_json_safe(value: Any):
    """
    Convert NumPy/Pandas values into JSON-compatible values.
    """

    if value is None:
        return None

    # NumPy scalar values
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    # Dictionaries
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    # Lists / tuples
    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value


def log_prediction(
    input_data: Dict[str, Any],
    prediction: int,
    probabilities: Optional[Dict[str, float]] = None,
    model_version: str = "unknown",
    response_time_ms: Optional[float] = None,
    log_path: Optional[Path] = None,
) -> None:
    """
    Write a prediction event to the monitoring log.

    Schema:

    {
        "timestamp_utc": "...",
        "status": "SUCCESS",
        "input_features": {...},
        "predicted_class": 2,
        "risk_tier": "High Risk",
        "confidence": 0.85,
        "probabilities": {...},
        "model_version": "v2",
        "response_time_ms": 25.4,
        "error_message": null
    }
    """

    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    log_path = Path(log_path)

    ensure_log_directory(log_path)

    prediction = int(prediction)

    risk_tier = TARGET_NAMES.get(
        prediction,
        "Unknown",
    )

    confidence = None

    if probabilities:
        try:
            confidence = float(
                max(probabilities.values())
            )
        except (ValueError, TypeError):
            confidence = None

    record = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "status": "SUCCESS",

        "input_features": make_json_safe(
            input_data
        ),

        "predicted_class": prediction,

        "risk_tier": risk_tier,

        "confidence": confidence,

        "probabilities": make_json_safe(
            probabilities
        ),

        "model_version": str(
            model_version
        ),

        "response_time_ms": (
            float(response_time_ms)
            if response_time_ms is not None
            else None
        ),

        "error_message": None,
    }

    with open(
        log_path,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def read_prediction_logs(
    log_path: Optional[Path] = None,
):
    """
    Read all prediction records from the JSONL log.
    """

    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    log_path = Path(log_path)

    if not log_path.exists():
        return []

    records = []

    with open(
        log_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue

    return records


def get_prediction_count(
    log_path: Optional[Path] = None,
) -> int:
    """
    Return the number of valid prediction records.
    """

    return len(
        read_prediction_logs(log_path)
    )


def clear_prediction_logs(
    log_path: Optional[Path] = None,
) -> None:
    """
    Delete the monitoring log.
    """

    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    log_path = Path(log_path)

    if log_path.exists():
        log_path.unlink()