"""
Drift detection for the Elderly Health Risk Monitoring system.

Uses Population Stability Index (PSI) to compare
reference training data with production prediction data.
"""

from pathlib import Path
from typing import Dict, Any

import json
import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger


logger = get_logger("drift_detector")


# PSI thresholds
NO_DRIFT_THRESHOLD = 0.10
MODERATE_DRIFT_THRESHOLD = 0.25


def calculate_psi(
    reference,
    current,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    Accepts both:
        - Pandas Series
        - NumPy arrays
        - Python lists

    PSI interpretation:
        < 0.10       No significant drift
        0.10 - 0.25  Moderate drift
        >= 0.25      Significant drift
    """

    # Convert everything to NumPy arrays.
    reference = np.asarray(reference)
    current = np.asarray(current)

    # Convert values to numeric.
    reference = pd.to_numeric(
        pd.Series(reference),
        errors="coerce",
    ).dropna().to_numpy()

    current = pd.to_numeric(
        pd.Series(current),
        errors="coerce",
    ).dropna().to_numpy()

    # If either dataset has no valid values, PSI cannot be calculated.
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Identical distributions have zero drift.
    if np.array_equal(reference, current):
        return 0.0

    # If all reference values are identical, create a small range.
    ref_min = float(np.min(reference))
    ref_max = float(np.max(reference))

    if ref_min == ref_max:
        ref_max = ref_min + 1.0

    # Create bins based on the reference distribution.
    bin_edges = np.linspace(ref_min, ref_max, bins + 1)

    # Make sure current values outside the reference range
    # are included in the first/last bins.
    reference_clipped = np.clip(
        reference,
        bin_edges[0],
        bin_edges[-1],
    )

    current_clipped = np.clip(
        current,
        bin_edges[0],
        bin_edges[-1],
    )

    # Calculate frequency distributions.
    reference_counts, _ = np.histogram(
        reference_clipped,
        bins=bin_edges,
    )

    current_counts, _ = np.histogram(
        current_clipped,
        bins=bin_edges,
    )

    # Convert counts into proportions.
    reference_dist = reference_counts.astype(float) / len(reference)
    current_dist = current_counts.astype(float) / len(current)

    # Avoid division by zero.
    epsilon = 1e-10

    reference_dist = np.maximum(reference_dist, epsilon)
    current_dist = np.maximum(current_dist, epsilon)

    # PSI formula:
    #
    # PSI = (Current% - Reference%) *
    #       ln(Current% / Reference%)
    #
    psi = np.sum(
        (current_dist - reference_dist)
        * np.log(current_dist / reference_dist)
    )

    return float(psi)


def classify_drift(psi: float) -> str:
    """
    Convert PSI value into a human-readable drift category.
    """

    if psi < NO_DRIFT_THRESHOLD:
        return "No Significant Drift"

    if psi < MODERATE_DRIFT_THRESHOLD:
        return "Moderate Drift"

    return "Significant Drift"


def load_monitoring_inputs(
    reference_path: Path,
    monitoring_log_path: Path,
):
    """
    Load reference training data and current production data.
    """

    reference_df = pd.read_csv(reference_path)

    if not monitoring_log_path.exists():
        return reference_df, pd.DataFrame()

    records = []

    with open(
        monitoring_log_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping invalid monitoring log entry."
                )

    if not records:
        return reference_df, pd.DataFrame()

    current_df = pd.DataFrame(
        [
            record.get(
                "input_features",
                record.get("input", {}),
            )
            for record in records
        ]
    )

    return reference_df, current_df


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate PSI for all common numeric features.
    """

    if reference_df.empty or current_df.empty:
        return {}

    common_features = [
        column
        for column in reference_df.columns
        if column in current_df.columns
    ]

    if not common_features:
        raise ValueError(
            "No common features exist between reference and current datasets."
        )

    results = {}

    for feature in common_features:

        reference_values = pd.to_numeric(
            reference_df[feature],
            errors="coerce",
        ).dropna()

        current_values = pd.to_numeric(
            current_df[feature],
            errors="coerce",
        ).dropna()

        # Ignore columns that cannot be converted into useful numeric data.
        if len(reference_values) == 0 or len(current_values) == 0:
            continue

        psi = calculate_psi(
            reference_values,
            current_values,
        )

        results[feature] = {
            "psi": round(psi, 4),
            "classification": classify_drift(psi),
        }

    return results


def save_drift_report(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save drift results as JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "features": results,
        "summary": {
            "total_features": len(results),
            "no_significant_drift": sum(
                1
                for result in results.values()
                if result["classification"] == "No Significant Drift"
            ),
            "moderate_drift": sum(
                1
                for result in results.values()
                if result["classification"] == "Moderate Drift"
            ),
            "significant_drift": sum(
                1
                for result in results.values()
                if result["classification"] == "Significant Drift"
            ),
        },
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )


def run_drift_detection(config=None):
    """
    Run the complete drift detection process.
    """

    if config is None:
        config = load_config()

    reference_path = Path(
        config["monitoring"]["reference_data_path"]
    )

    monitoring_log_path = Path(
        config["monitoring"]["log_path"]
    )

    report_path = Path(
        config["monitoring"]["drift_report_path"]
    )

    logger.info(
        "Loading reference data from %s",
        reference_path,
    )

    reference_df, current_df = load_monitoring_inputs(
        reference_path,
        monitoring_log_path,
    )

    if current_df.empty:
        logger.warning(
            "No production monitoring data available."
        )

        return {}

    results = detect_drift(
        reference_df,
        current_df,
    )

    save_drift_report(
        results,
        report_path,
    )

    return results


if __name__ == "__main__":

    print("=" * 60)
    print("ELDERLY HEALTH RISK - DRIFT DETECTION")
    print("=" * 60)

    try:

        results = run_drift_detection()

        if not results:
            print(
                "No drift results available."
            )

        else:

            print("\nFeature results:")

            for feature, result in results.items():

                print(
                    f"{feature}: "
                    f"PSI={result['psi']:.4f} | "
                    f"{result['classification']}"
                )

            print("\nDrift report saved successfully.")

    except Exception as error:

        print(
            f"Drift detection failed: {error}"
        )