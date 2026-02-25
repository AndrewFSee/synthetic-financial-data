"""YAML configuration loader and merger."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary with configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.debug("Loaded config from %s", path)
    return config or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries.

    Values in `override` take precedence over `base`. Nested dicts are merged
    recursively rather than replaced wholesale.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_and_merge(
    model_config_path: Union[str, Path],
    default_config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load a model config and merge with optional defaults.

    Args:
        model_config_path: Path to model-specific config file.
        default_config_path: Path to default config (optional base).

    Returns:
        Merged configuration dictionary.
    """
    model_cfg = load_config(model_config_path)

    if default_config_path is not None:
        default_cfg = load_config(default_config_path)
        return merge_configs(default_cfg, model_cfg)

    return model_cfg
