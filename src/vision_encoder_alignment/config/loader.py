"""Safe YAML loading entry points."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .validation import (
    ConfigValidationError,
    validate_experiment_config,
    validate_model_registry,
)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 YAML file with PyYAML's safe loader."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigValidationError(f"{config_path}: unable to load YAML: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ConfigValidationError(f"{config_path}: expected a YAML mapping")
    return dict(loaded)


def load_model_registry(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a model registry."""

    registry = load_yaml_mapping(path)
    validate_model_registry(registry)
    return registry


def load_experiment_config(
    path: str | Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Load and structurally validate an experiment configuration."""

    config = load_yaml_mapping(path)
    validate_experiment_config(config, registry)
    return config
