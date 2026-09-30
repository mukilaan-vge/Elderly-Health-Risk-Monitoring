"""Unit tests for Phase 1: Data Collection & Management."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.data.data_loader import load_dataset_metadata, load_raw_data
from src.utils.config import load_config, resolve_path


def test_raw_dataset_exists():
    """Verify raw CSV file exists on disk."""
    config = load_config()
    raw_path = resolve_path(config["data"]["raw_csv_path"])
    assert raw_path.exists(), f"Raw dataset not found at {raw_path}"
    assert raw_path.stat().st_size > 0, "Raw dataset file is empty"


def test_metadata_structure():
    """Verify metadata JSON structure and integrity."""
    metadata = load_dataset_metadata()
    required_keys = [
        "dataset_name",
        "source",
        "num_instances",
        "num_features",
        "sha256_checksum",
        "target_class_counts",
    ]
    for key in required_keys:
        assert key in metadata, f"Metadata missing required key: {key}"

    assert metadata["num_instances"] == 714
    assert metadata["num_features"] == 14
    assert len(metadata["sha256_checksum"]) == 64


def test_load_raw_data_shape_and_columns():
    """Verify raw data loads into a DataFrame with correct shape and target."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (714, 15)

    config = load_config()
    target_col = config["data"]["raw_target_column"]
    assert target_col in df.columns
    unique_targets = sorted(df[target_col].unique())
    assert unique_targets == [1, 2, 3]


def test_no_null_in_raw_dataset():
    """Verify raw dataset has zero null entries as expected from UCI curation."""
    df = load_raw_data()
    assert df.isnull().sum().sum() == 0

