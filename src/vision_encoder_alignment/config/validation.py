"""Structural validation for repository configuration files."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MODEL_REGISTRY_SCHEMA_VERSION = 1
EXPERIMENT_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_ID = "vea.feature_artifact.v0"
SUPPORTED_REVISION_STATUSES = frozenset({"pinned", "unresolved"})
SUPPORTED_PHASES = frozenset({"P0", "P1"})
SUPPORTED_DTYPES = frozenset({"float16", "float32"})


class ConfigValidationError(ValueError):
    """Raised when a configuration field violates the structural contract."""


def _fail(path: str, message: str) -> None:
    raise ConfigValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected a mapping")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "expected a non-empty string")
    return value


def _positive_int(value: Any, path: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(path, "expected a positive integer")
    return value


def _integer(value: Any, path: str) -> int:
    if type(value) is not int:
        _fail(path, "expected an integer")
    return value


def validate_model_registry(registry: Mapping[str, Any]) -> None:
    """Validate the Phase 0 model registry without testing runtime access."""

    root = _mapping(registry, "registry")
    if root.get("schema_version") != MODEL_REGISTRY_SCHEMA_VERSION:
        _fail("schema_version", f"expected {MODEL_REGISTRY_SCHEMA_VERSION}")

    models = _mapping(root.get("models"), "models")
    if not models:
        _fail("models", "expected at least one model")

    for model_key, raw_model in models.items():
        if not isinstance(model_key, str) or not model_key.strip():
            _fail("models", "model keys must be non-empty strings")
        model_path = f"models.{model_key}"
        model = _mapping(raw_model, model_path)

        _nonempty_string(model.get("model_id"), f"{model_path}.model_id")
        revision_status = _nonempty_string(
            model.get("revision_status"), f"{model_path}.revision_status"
        )
        if revision_status not in SUPPORTED_REVISION_STATUSES:
            _fail(
                f"{model_path}.revision_status",
                f"unsupported value {revision_status!r}",
            )

        revision = model.get("revision")
        if revision_status == "unresolved":
            if revision is not None:
                _fail(
                    f"{model_path}.revision",
                    "expected null when revision_status is unresolved",
                )
        elif not isinstance(revision, str) or not revision.strip():
            _fail(
                f"{model_path}.revision",
                "expected a non-empty exact revision when revision_status is pinned",
            )

        canonical_input = _mapping(
            model.get("canonical_input"), f"{model_path}.canonical_input"
        )
        _positive_int(
            canonical_input.get("height"), f"{model_path}.canonical_input.height"
        )
        _positive_int(
            canonical_input.get("width"), f"{model_path}.canonical_input.width"
        )
        _positive_int(
            canonical_input.get("patch_size_hint"),
            f"{model_path}.canonical_input.patch_size_hint",
        )

        grid_hint = canonical_input.get("grid_hint")
        if (
            not isinstance(grid_hint, list)
            or len(grid_hint) != 2
            or any(type(item) is not int or item <= 0 for item in grid_hint)
        ):
            _fail(
                f"{model_path}.canonical_input.grid_hint",
                "expected two positive integers",
            )
        patch_count = _positive_int(
            canonical_input.get("patch_count_hint"),
            f"{model_path}.canonical_input.patch_count_hint",
        )
        if grid_hint[0] * grid_hint[1] != patch_count:
            _fail(
                f"{model_path}.canonical_input.patch_count_hint",
                "expected grid_hint product to equal patch_count_hint",
            )

        _positive_int(
            model.get("hidden_size_hint"), f"{model_path}.hidden_size_hint"
        )
        _positive_int(
            model.get("layer_count_hint"), f"{model_path}.layer_count_hint"
        )
        loader_policy = _mapping(
            model.get("loader_policy"), f"{model_path}.loader_policy"
        )
        if loader_policy.get("trust_remote_code") is not False:
            _fail(
                f"{model_path}.loader_policy.trust_remote_code",
                "must be false under the Phase 0 research contract",
            )


def validate_experiment_config(
    config: Mapping[str, Any], registry: Mapping[str, Any]
) -> None:
    """Validate an experiment config and its model registry references."""

    root = _mapping(config, "config")
    validate_model_registry(registry)

    if root.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        _fail("schema_version", f"expected {EXPERIMENT_SCHEMA_VERSION}")
    _nonempty_string(root.get("experiment_id"), "experiment_id")

    phase = _nonempty_string(root.get("phase"), "phase")
    if phase not in SUPPORTED_PHASES:
        _fail("phase", f"unsupported value {phase!r}")

    model_keys = root.get("models")
    if not isinstance(model_keys, list):
        _fail("models", "expected a list")
    if not model_keys:
        _fail("models", "expected at least one model key")
    registry_models = _mapping(registry.get("models"), "registry.models")
    seen_model_keys: set[str] = set()
    for index, model_key in enumerate(model_keys):
        path = f"models[{index}]"
        _nonempty_string(model_key, path)
        if model_key in seen_model_keys:
            _fail(path, f"duplicate model key {model_key!r}")
        seen_model_keys.add(model_key)
        if model_key not in registry_models:
            _fail(path, f"unknown model key {model_key!r}")

    dataset = _mapping(root.get("dataset"), "dataset")
    _nonempty_string(dataset.get("manifest"), "dataset.manifest")
    _nonempty_string(dataset.get("root_env"), "dataset.root_env")
    _positive_int(dataset.get("expected_image_count"), "dataset.expected_image_count")

    preprocessing = _mapping(root.get("preprocessing"), "preprocessing")
    _positive_int(
        preprocessing.get("resize_shorter_side"),
        "preprocessing.resize_shorter_side",
    )
    _positive_int(preprocessing.get("crop_height"), "preprocessing.crop_height")
    _positive_int(preprocessing.get("crop_width"), "preprocessing.crop_width")
    if preprocessing.get("random_augmentation") is not False:
        _fail(
            "preprocessing.random_augmentation",
            "must be false for the baseline experiment",
        )

    extraction = _mapping(root.get("extraction"), "extraction")
    _positive_int(extraction.get("batch_size"), "extraction.batch_size")
    for field in ("parameter_dtype", "compute_dtype", "storage_dtype"):
        dtype = _nonempty_string(extraction.get(field), f"extraction.{field}")
        if dtype not in SUPPORTED_DTYPES:
            _fail(f"extraction.{field}", f"unsupported dtype {dtype!r}")
    if extraction.get("gradients") != "disabled":
        _fail("extraction.gradients", "expected 'disabled'")
    if extraction.get("model_mode") != "eval":
        _fail("extraction.model_mode", "expected 'eval'")

    artifacts = _mapping(root.get("artifacts"), "artifacts")
    if artifacts.get("schema_id") != ARTIFACT_SCHEMA_ID:
        _fail("artifacts.schema_id", f"expected {ARTIFACT_SCHEMA_ID!r}")
    _positive_int(artifacts.get("shard_size"), "artifacts.shard_size")
    if artifacts.get("checksum") != "sha256":
        _fail("artifacts.checksum", "expected 'sha256'")

    randomness = _mapping(root.get("randomness"), "randomness")
    _integer(randomness.get("seed"), "randomness.seed")
