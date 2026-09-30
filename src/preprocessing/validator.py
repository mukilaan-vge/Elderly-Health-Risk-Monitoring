"""
Dataset validation module for the Elderly Health Risk Monitoring project.
"""

from pathlib import Path
from typing import Union
import pandas as pd


# Expected columns in the raw elderly health dataset
REQUIRED_COLUMNS = [
    "Age",
    "Physical_Health",
    "Mental_Health",
    "Dental_Health",
    "Employment",
    "Stress_Keeps_Patient_from_Sleeping",
    "Medication_Keeps_Patient_from_Sleeping",
    "Pain_Keeps_Patient_from_Sleeping",
    "Bathroom_Needs_Keeps_Patient_from_Sleeping",
    "Uknown_Keeps_Patient_from_Sleeping",
    "Trouble_Sleeping",
    "Prescription_Sleep_Medication",
    "Race",
    "Gender",
    "Number_of_Doctors_Visited",
]


def validate_dataset(
    data_source: Union[str, Path, dict]
) -> bool:
    """
    Validate the raw elderly health dataset.

    Parameters
    ----------
    data_source:
        Can be:
        - Path to the CSV file
        - String path to the CSV file
        - Configuration dictionary containing data paths

    Returns
    -------
    bool
        True if validation succeeds.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found.
    ValueError
        If the dataset fails validation.
    """

    # ---------------------------------------------------------
    # Resolve dataset path
    # ---------------------------------------------------------

    if isinstance(data_source, dict):
        data_path = data_source["data"]["raw_csv_path"]
    else:
        data_path = data_source

    data_path = Path(data_path)

    # If a relative path is supplied, interpret it relative
    # to the project root.
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path

    # ---------------------------------------------------------
    # Check file existence
    # ---------------------------------------------------------

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    if data_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, but received: {data_path}"
        )

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    print(f"Dataset: {data_path}")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        raise ValueError(
            f"Unable to read dataset: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    if df.empty:
        raise ValueError("Dataset is empty.")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # Check required columns
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing_columns)
        )

    print("\nRequired columns: OK")

    # ---------------------------------------------------------
    # Check duplicate rows
    # ---------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")

    # Duplicates are reported but do not automatically fail
    # validation because preprocessing can handle them later.

    # ---------------------------------------------------------
    # Check missing values
    # ---------------------------------------------------------

    missing_values = df.isnull().sum()
    total_missing = int(missing_values.sum())

    print(f"Missing values: {total_missing}")

    if total_missing > 0:
        print("\nColumns containing missing values:")

        for column, count in missing_values.items():
            if count > 0:
                print(f"  - {column}: {count}")

        print(
            "\nMissing values detected. "
            "They will be handled during preprocessing."
        )

    # ---------------------------------------------------------
    # Check numeric columns
    # ---------------------------------------------------------

    numeric_columns = [
        "Age",
        "Physical_Health",
        "Mental_Health",
        "Dental_Health",
        "Employment",
        "Stress_Keeps_Patient_from_Sleeping",
        "Medication_Keeps_Patient_from_Sleeping",
        "Pain_Keeps_Patient_from_Sleeping",
        "Bathroom_Needs_Keeps_Patient_from_Sleeping",
        "Uknown_Keeps_Patient_from_Sleeping",
        "Trouble_Sleeping",
        "Prescription_Sleep_Medication",
        "Race",
        "Gender",
        "Number_of_Doctors_Visited",
    ]

    invalid_numeric_columns = []

    for column in numeric_columns:
        if column not in df.columns:
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_count = (
            converted.isna() & df[column].notna()
        ).sum()

        if invalid_count > 0:
            invalid_numeric_columns.append(
                (column, int(invalid_count))
            )

    if invalid_numeric_columns:
        print("\nNon-numeric values detected:")

        for column, count in invalid_numeric_columns:
            print(f"  - {column}: {count}")

        print(
            "\nThese values will need to be handled "
            "during preprocessing."
        )
    else:
        print("Numeric columns: OK")

    # ---------------------------------------------------------
    # Check target column
    # ---------------------------------------------------------

    target_column = "Number_of_Doctors_Visited"

    if target_column in df.columns:
        target_values = df[target_column].dropna()

        print(
            f"\nTarget column: {target_column}"
        )

        print(
            f"Unique target values: "
            f"{target_values.nunique()}"
        )

        print(
            f"Target range: "
            f"{target_values.min()} - {target_values.max()}"
        )

    # ---------------------------------------------------------
    # Final validation result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET VALIDATION PASSED")
    print("=" * 60)

    return True


# -------------------------------------------------------------
# Allow direct execution
# -------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Default location used by this project
        default_path = (
            Path("data")
            / "raw"
            / "npha_elderly_health_risk.csv"
        )

        validate_dataset(default_path)

    except Exception as exc:
        print("\nVALIDATION FAILED")
        print(f"Error: {exc}")
        raise