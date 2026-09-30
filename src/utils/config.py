"""Configuration loader and path manager for the elderly health risk pipeline."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

# Resolve the project root (elderly-health-risk-mlops directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file with relative paths resolved.

    Args:
        config_path: Optional relative or absolute path to config.yaml.

    Returns:
        Dict of configuration parameters.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def resolve_path(rel_path: str) -> Path:
    """Resolve a relative project path to an absolute Path object.

    Args:
        rel_path: Path relative to project root.

    Returns:
        Absolute Path object.
    """
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p

