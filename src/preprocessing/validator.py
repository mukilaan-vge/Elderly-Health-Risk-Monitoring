"""Data validation module for Elderly Health Risk Monitoring pipeline.

Performs schema validation, column presence verification, data type checks,
missing-value and duplicate detection, and domain range validation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("validator")

# Expected feature value domain constraints based on NPHA codebook
FEATURE_CONSTRAINTS = {
    "Age": {"allowed": [1, 2], "description": "1: 50-64, 2: 65-80"},
    "Physical_Health": {"min": 1, "max": 5, "refusal_codes": [-1], "description": "1: Excellent to 5: Poor"},
    "Mental_Health": {"min": 1, "max": 5, "refusal_codes": [-1], "description": "1: Excellent to 5: Poor"},
    "Dental_Health": {"min": 1, "max": 6, "refusal_codes": [-1], "description": "1: Excellent to 5: Poor, 6: Dentures"},
    "Employment": {"allowed": [1, 2, 3, 4], "refusal_codes": [-1], "description": "1: FT, 2: PT, 3: Retired, 4: Not working"},
    "Stress_Keeps_Patient_from_Sleeping": {"allowed": [0, 1], "description": "0: No, 1: Yes"},
    "Medication_Keeps_Patient_from_Sleeping": {"allowed": [0, 1], "description": "0: No, 1: Yes"},
    "Pain_Keeps_Patient_from_Sleeping": {"allowed": [0, 1], "description": "0: No, 1: Yes"},
    "Bathroom_Needs_Keeps_Patient_from_Sleeping": {"allowed": [0, 1], "description": "0: No, 1: Yes"},
    "Uknown_Keeps_Patient_from_Sleeping": {"allowed": [0, 1], "description": "0: No, 1: Yes"},
    "Trouble_Sleeping": {"allowed": [0, 1, 2, 3], "refusal_codes": [-1], "description": "Trouble sleeping frequency/indicator"},
    "Prescription_Sleep_Medication": {"allowed": [1, 2, 3], "refusal_codes": [-1], "description": "1: Regular, 2: Occasional, 3: None"},
    "Race": {"allowed": [1, 2, 3, 4, 5], "refusal_codes": [-1, -2], "description": "Race/ethnicity category"},
    "Gender": {"allowed": [1, 2], "refusal_codes": [-1, -2], "description": "1: Male, 2: Female"},
}

TARGET_CONSTRAINTS = {
    "Number_of_Doctors_Visited": {"allowed": [1, 2, 3], "description": "1: Low (0-1), 2: Moderate (2-3), 3: High (4+)"},
    "Health_Risk_Level": {"allowed": [0, 1, 2], "description": "0: Low, 1: Moderate, 2: High"},
}


class DataValidationError(Exception):
    """Custom exception raised when critical data validation fails."""
    pass


def validate_dataframe(df: pd.DataFrame, is_training: bool = True, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Perform comprehensive schema and data validation on a DataFrame.

    Args:
        df: Input DataFrame to validate.
        is_training: If True, also validates presence and distribution of target column.
        config: Optional configuration dictionary.

    Returns:
        Dict summarizing validation results, errors, warnings, and statistics.
    """
    if config is None:
        config = load_config()

    expected_features = config["data"]["raw_features"]
    raw_target = config["data"]["raw_target_column"]

    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {
        "num_rows": int(len(df)),
        "num_cols": int(df.shape[1]),
        "null_count": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "refusal_codes_detected": {},
    }

    # 1. Check for empty dataframe
    if df.empty:
        errors.append("DataFrame is empty (0 rows).")
        return {"is_valid": False, "errors": errors, "warnings": warnings, "stats": stats}

    # 2. Column presence validation
    missing_cols = [c for c in expected_features if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required feature columns: {missing_cols}")

    if is_training and (raw_target not in df.columns and "Health_Risk_Level" not in df.columns):
        errors.append(f"Missing required target column: '{raw_target}' or 'Health_Risk_Level'")

    # 3. Duplicate row check
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        warnings.append(f"Detected {dup_count} duplicate rows in dataset (common in categorical survey cohorts).")

    # 4. Feature constraints and refusal code detection
    for feature in expected_features:
        if feature not in df.columns:
            continue

        col_series = df[feature]
        constraint = FEATURE_CONSTRAINTS.get(feature, {})
        refusal_codes = constraint.get("refusal_codes", [])

        # Check refusal codes (negative values in survey)
        refusal_mask = col_series.isin(refusal_codes)
        refusal_count = int(refusal_mask.sum())
        if refusal_count > 0:
            stats["refusal_codes_detected"][feature] = refusal_count
            warnings.append(
                f"Feature '{feature}' contains {refusal_count} survey refusal codes {refusal_codes} "
                "requiring statistical imputation."
            )

        # Check valid domain
        valid_series = col_series[~refusal_mask].dropna()
        if "allowed" in constraint:
            invalid_vals = valid_series[~valid_series.isin(constraint["allowed"])].unique().tolist()
            if invalid_vals:
                errors.append(f"Feature '{feature}' contains out-of-domain values: {invalid_vals}")
        elif "min" in constraint and "max" in constraint:
            out_of_bounds = valid_series[(valid_series < constraint["min"]) | (valid_series > constraint["max"])]
            if not out_of_bounds.empty:
                errors.append(
                    f"Feature '{feature}' has values outside [{constraint['min']}, {constraint['max']}]: "
                    f"{out_of_bounds.unique().tolist()}"
                )

    # 5. Target column validation if present
    target_to_check = raw_target if raw_target in df.columns else ("Health_Risk_Level" if "Health_Risk_Level" in df.columns else None)
    if is_training and target_to_check:
        t_series = df[target_to_check]
        allowed_targets = TARGET_CONSTRAINTS[target_to_check]["allowed"]
        invalid_targets = t_series[~t_series.isin(allowed_targets)].unique().tolist()
        if invalid_targets:
            errors.append(f"Target '{target_to_check}' contains invalid target values: {invalid_targets}")
        stats["target_distribution"] = {str(k): int(v) for k, v in t_series.value_counts().to_dict().items()}

    is_valid = len(errors) == 0
    if not is_valid:
        logger.error(f"Validation FAILED with {len(errors)} error(s): {errors}")
    else:
        logger.info(f"Validation PASSED with {len(warnings)} warning(s).")

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def validate_single_input(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single incoming inference record for FastAPI.

    Args:
        payload: Dictionary of input features.

    Returns:
        Tuple of (is_valid: bool, errors: List[str]).
    """
    errors: List[str] = []

    for feature, constraint in FEATURE_CONSTRAINTS.items():
        if feature not in payload:
            errors.append(f"Missing required field: '{feature}'")
            continue

        val = payload[feature]
        if not isinstance(val, (int, float)) or np.isnan(val):
            errors.append(f"Field '{feature}' must be a numeric value, got '{val}' ({type(val).__name__})")
            continue

        val = int(val)
        refusal_codes = constraint.get("refusal_codes", [])
        if val in refusal_codes:
            # Allow refusal codes as they will be handled by preprocessor imputer
            continue

        if "allowed" in constraint:
            if val not in constraint["allowed"]:
                errors.append(
                    f"Invalid value {val} for '{feature}'. Allowed values: {constraint['allowed']}"
                )
        elif "min" in constraint and "max" in constraint:
            if val < constraint["min"] or val > constraint["max"]:
                errors.append(
                    f"Value {val} for '{feature}' out of bounds [{constraint['min']}, {constraint['max']}]"
                )

    return len(errors) == 0, errors


if __name__ == "__main__":
    from src.data.data_loader import load_raw_data

    df = load_raw_data()
    result = validate_dataframe(df, is_training=True)
    print("=" * 60)
    print("PHASE 2: DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"Is Valid:   {result['is_valid']}")
    print(f"Errors:     {result['errors']}")
    print(f"Warnings:   {len(result['warnings'])} warnings found")
    for w in result["warnings"]:
        print(f"  - {w}")
    print(f"Stats:      {result['stats']}")

