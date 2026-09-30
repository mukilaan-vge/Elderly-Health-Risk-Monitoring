"""Dataset collection, download, metadata extraction, and raw data management.

Dataset: National Poll on Healthy Aging (NPHA)
Source: UCI Machine Learning Repository (ID: 936) / University of Michigan / AARP
DOI: 10.3886/ICPSR37305.v1
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import requests

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("data_loader")

DIRECT_CSV_URL = "https://archive.ics.uci.edu/static/public/936/data.csv"


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash of a file for data versioning."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def fetch_and_save_raw_data(config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch the NPHA dataset from UCI, save raw CSV, and generate versioned metadata.

    Args:
        config: Optional configuration dictionary.

    Returns:
        Tuple of (raw DataFrame, metadata dictionary).
    """
    if config is None:
        config = load_config()

    raw_csv_path = resolve_path(config["data"]["raw_csv_path"])
    raw_metadata_path = resolve_path(config["data"]["raw_metadata_path"])
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    raw_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Attempting to fetch NPHA dataset from UCI repository...")
    df = None
    fetch_method = "ucimlrepo"

    try:
        from ucimlrepo import fetch_ucirepo
        npha = fetch_ucirepo(id=config["data"]["uci_id"])
        features = npha.data.features
        targets = npha.data.targets
        df = pd.concat([features, targets], axis=1)
        logger.info(f"Successfully fetched dataset via ucimlrepo. Shape: {df.shape}")
    except Exception as e:
        logger.warning(f"ucimlrepo fetch failed ({e}). Falling back to direct HTTP download from {DIRECT_CSV_URL}...")
        try:
            response = requests.get(DIRECT_CSV_URL, timeout=15)
            response.raise_for_status()
            with open(raw_csv_path, "wb") as f:
                f.write(response.content)
            df = pd.read_csv(raw_csv_path)
            fetch_method = "direct_http"
            logger.info(f"Successfully downloaded raw CSV via HTTP. Shape: {df.shape}")
        except Exception as http_err:
            logger.error(f"HTTP download also failed: {http_err}")
            raise RuntimeError(
                "Unable to download NPHA dataset from UCI. Please check your internet connection "
                f"or manually download the CSV from {DIRECT_CSV_URL} and place it at {raw_csv_path}."
            ) from http_err

    # Save raw CSV
    df.to_csv(raw_csv_path, index=False)
    file_hash = calculate_sha256(raw_csv_path)
    file_size_bytes = raw_csv_path.stat().st_size

    # Inspect target distribution
    target_col = config["data"]["raw_target_column"]
    target_counts = df[target_col].value_counts().to_dict() if target_col in df.columns else {}

    # Build rich metadata for data lineage and viva review
    metadata = {
        "dataset_name": "National Poll on Healthy Aging (NPHA)",
        "source": "UCI Machine Learning Repository (University of Michigan / AARP / Michigan Medicine)",
        "uci_id": config["data"]["uci_id"],
        "dataset_doi": "https://doi.org/10.3886/ICPSR37305.v1",
        "fetch_method": fetch_method,
        "fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "file_path": str(raw_csv_path),
        "file_size_bytes": file_size_bytes,
        "sha256_checksum": file_hash,
        "num_instances": int(df.shape[0]),
        "num_features": int(df.shape[1] - 1),
        "total_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "raw_target_column": target_col,
        "target_class_counts": {str(k): int(v) for k, v in target_counts.items()},
        "target_class_mapping": {
            "1": "Low Risk (0-1 doctors visited)",
            "2": "Moderate Risk (2-3 doctors visited)",
            "3": "High Risk (4+ doctors visited)",
        },
        "missing_values_count": int(df.isnull().sum().sum()),
        "suitability_rationale": (
            "Authentic geriatric survey data focused on seniors aged 50-80+. "
            "Evaluates multi-domain health indicators (physical, mental, dental, sleep disturbances, "
            "prescription sleep aids, and socioeconomic factors) to classify elderly clinical risk and health utilization."
        ),
    }

    with open(raw_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Raw dataset stored at: {raw_csv_path}")
    logger.info(f"Dataset metadata stored at: {raw_metadata_path}")
    logger.info(f"Dataset SHA256 version hash: {file_hash}")

    return df, metadata


def load_raw_data(config: Dict[str, Any] = None) -> pd.DataFrame:
    """Load raw dataset from disk, or fetch if not present.

    Args:
        config: Optional configuration dictionary.

    Returns:
        pd.DataFrame of raw elderly health survey data.
    """
    if config is None:
        config = load_config()

    raw_csv_path = resolve_path(config["data"]["raw_csv_path"])
    if not raw_csv_path.exists():
        logger.info(f"Raw data not found at {raw_csv_path}. Initiating fetch...")
        df, _ = fetch_and_save_raw_data(config)
        return df

    return pd.read_csv(raw_csv_path)


def load_dataset_metadata(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load dataset metadata JSON from disk."""
    if config is None:
        config = load_config()
    raw_metadata_path = resolve_path(config["data"]["raw_metadata_path"])
    if not raw_metadata_path.exists():
        _, meta = fetch_and_save_raw_data(config)
        return meta
    with open(raw_metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    df, meta = fetch_and_save_raw_data()
    print("=" * 60)
    print("PHASE 1: DATASET INSPECTION & METADATA SUMMARY")
    print("=" * 60)
    print(f"Dataset Name:    {meta['dataset_name']}")
    print(f"Source:          {meta['source']}")
    print(f"DOI:             {meta['dataset_doi']}")
    print(f"Instances:       {meta['num_instances']}")
    print(f"Features:        {meta['num_features']}")
    print(f"SHA256 Checksum: {meta['sha256_checksum']}")
    print(f"Target Variable: {meta['raw_target_column']}")
    print(f"Class Counts:    {meta['target_class_counts']}")
    print("\nFirst 3 records:")
    print(df.head(3))

