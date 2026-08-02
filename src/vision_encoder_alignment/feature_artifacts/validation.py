"""Canonical metadata and tensor validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np

from .schema import ArtifactValidationError, FeatureArtifactMetadata

ARTIFACT_SCHEMA_ID = "vea.feature_artifact.v0"
CANONICAL_TAP_NAMESPACE = "backbone"
SUPPORTED_RUN_STATUSES = frozenset({"complete", "incomplete", "invalid"})
SUPPORTED_DTYPES = frozenset({"float16", "float32"})
CHECKSUM_ALGORITHM = "sha256"
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_TAP_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOLERANCES = {
    "float16": {"rtol": 1e-3, "atol": 1e-3},
    "float32": {"rtol": 1e-5, "atol": 1e-6},
}


def _fail(path: str, message: str) -> None:
    raise ArtifactValidationError(f"{path}: {message}")


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "expected a non-empty string")
    return value


def _positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(path, "expected a positive integer")
    return value


def _safe_identifier(value: object, path: str) -> str:
    identifier = _nonempty_string(value, path)
    if _IDENTIFIER_PATTERN.fullmatch(identifier) is None:
        _fail(path, "expected letters, digits, dot, underscore, or hyphen only")
    return identifier


def _safe_file_name(value: object, path: str, suffix: str) -> str:
    file_name = _nonempty_string(value, path)
    if (
        PurePosixPath(file_name).name != file_name
        or PureWindowsPath(file_name).name != file_name
        or not file_name.endswith(suffix)
    ):
        _fail(path, f"expected a basename ending in {suffix!r}")
    return file_name


def canonical_tensor_keys(tap_name: str) -> tuple[str, str]:
    """Return canonical patch and global-mean keys for one tap."""

    if not isinstance(tap_name, str) or _TAP_PATTERN.fullmatch(tap_name) is None:
        _fail("tap_name", "expected a non-empty canonical tap name")
    return (
        f"{CANONICAL_TAP_NAMESPACE}.{tap_name}.patch",
        f"{CANONICAL_TAP_NAMESPACE}.{tap_name}.global_mean",
    )


def validate_metadata(
    metadata: FeatureArtifactMetadata, *, require_complete: bool = False
) -> None:
    """Validate all typed sidecar metadata fields."""

    if not isinstance(metadata, FeatureArtifactMetadata):
        _fail("metadata", "expected FeatureArtifactMetadata")
    if metadata.schema_id != ARTIFACT_SCHEMA_ID:
        _fail("schema_id", f"expected {ARTIFACT_SCHEMA_ID!r}")
    _safe_identifier(metadata.run_id, "run_id")
    _safe_identifier(metadata.model_key, "model_key")
    if metadata.tap_namespace != CANONICAL_TAP_NAMESPACE:
        _fail("tap_namespace", f"expected {CANONICAL_TAP_NAMESPACE!r}")

    expected_patch_key, expected_global_key = canonical_tensor_keys(metadata.tap_name)
    if metadata.patch_tensor_key != expected_patch_key:
        _fail("patch_tensor_key", f"expected {expected_patch_key!r}")
    if metadata.global_mean_tensor_key != expected_global_key:
        _fail("global_mean_tensor_key", f"expected {expected_global_key!r}")

    if not isinstance(metadata.image_ids, list):
        _fail("image_ids", "expected a list")
    seen: set[str] = set()
    for index, image_id in enumerate(metadata.image_ids):
        path = f"image_ids[{index}]"
        value = _nonempty_string(image_id, path)
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            _fail(path, "absolute paths are not valid image IDs")
        if value in seen:
            _fail(path, f"duplicate image ID {value!r}")
        seen.add(value)

    grid_height = _positive_int(metadata.grid_height, "grid_height")
    grid_width = _positive_int(metadata.grid_width, "grid_width")
    patch_count = _positive_int(metadata.patch_count, "patch_count")
    if grid_height * grid_width != patch_count:
        _fail("patch_count", "expected grid_height * grid_width")
    _positive_int(metadata.hidden_dimension, "hidden_dimension")
    if type(metadata.shard_index) is not int or metadata.shard_index < 0:
        _fail("shard_index", "expected a non-negative integer")
    if metadata.dtype not in SUPPORTED_DTYPES:
        _fail("dtype", f"unsupported dtype {metadata.dtype!r}")

    _safe_file_name(
        metadata.safetensors_file_name,
        "safetensors_file_name",
        ".safetensors",
    )
    _safe_file_name(metadata.sidecar_file_name, "sidecar_file_name", ".json")
    if metadata.checksum_algorithm != CHECKSUM_ALGORITHM:
        _fail("checksum_algorithm", f"expected {CHECKSUM_ALGORITHM!r}")

    if metadata.run_status not in SUPPORTED_RUN_STATUSES:
        _fail("run_status", f"unsupported status {metadata.run_status!r}")
    if require_complete and metadata.run_status != "complete":
        _fail("run_status", "expected 'complete'")
    if metadata.checksum_value:
        if _SHA256_PATTERN.fullmatch(metadata.checksum_value) is None:
            _fail("checksum_value", "expected a lowercase SHA-256 hex digest")
    elif metadata.run_status == "complete":
        _fail("checksum_value", "complete artifacts require a SHA-256 digest")

    timestamp = _nonempty_string(metadata.created_at_utc, "created_at_utc")
    if not timestamp.endswith("Z"):
        _fail("created_at_utc", "expected an ISO-8601 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise ArtifactValidationError(
            "created_at_utc: expected a valid ISO-8601 UTC timestamp"
        ) from exc


def validate_feature_tensors(
    metadata: FeatureArtifactMetadata,
    patch: np.ndarray,
    global_mean: np.ndarray,
) -> None:
    """Validate canonical tensor shapes, values, dtype, and patch means."""

    validate_metadata(metadata)
    if not isinstance(patch, np.ndarray):
        _fail(metadata.patch_tensor_key, "expected a NumPy array")
    if not isinstance(global_mean, np.ndarray):
        _fail(metadata.global_mean_tensor_key, "expected a NumPy array")
    if patch.ndim != 3:
        _fail(metadata.patch_tensor_key, "expected rank 3 [B, P, D]")
    if global_mean.ndim != 2:
        _fail(metadata.global_mean_tensor_key, "expected rank 2 [B, D]")
    if patch.shape[0] != global_mean.shape[0]:
        _fail("tensors.batch_dimension", "patch and global_mean batch sizes differ")
    if patch.shape[2] != global_mean.shape[1]:
        _fail("tensors.hidden_dimension", "patch and global_mean dimensions differ")
    if patch.shape[1] != metadata.patch_count:
        _fail("patch_count", "metadata does not match the patch tensor dimension")
    if patch.shape[2] != metadata.hidden_dimension:
        _fail("hidden_dimension", "metadata does not match the tensor dimension")
    if len(metadata.image_ids) != patch.shape[0]:
        _fail("image_ids", "count does not match the tensor batch dimension")

    patch_dtype = patch.dtype.name
    global_dtype = global_mean.dtype.name
    if patch_dtype not in SUPPORTED_DTYPES:
        _fail(metadata.patch_tensor_key, f"unsupported dtype {patch_dtype!r}")
    if global_dtype not in SUPPORTED_DTYPES:
        _fail(metadata.global_mean_tensor_key, f"unsupported dtype {global_dtype!r}")
    if patch_dtype != global_dtype:
        _fail("tensors.dtype", "patch and global_mean dtypes differ")
    if patch_dtype != metadata.dtype:
        _fail("dtype", "metadata does not match tensor dtype")

    if not np.isfinite(patch).all():
        _fail(metadata.patch_tensor_key, "contains NaN or Inf")
    if not np.isfinite(global_mean).all():
        _fail(metadata.global_mean_tensor_key, "contains NaN or Inf")

    calculated_mean = patch.mean(axis=1, dtype=np.float32)
    tolerance = _TOLERANCES[patch_dtype]
    if not np.allclose(
        global_mean.astype(np.float32, copy=False),
        calculated_mean,
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
    ):
        _fail(
            metadata.global_mean_tensor_key,
            "does not match the patch-axis mean within dtype tolerance",
        )


def validate_tensor_map(
    metadata: FeatureArtifactMetadata, tensors: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate exact tensor keys and return tensors in canonical order."""

    expected = {metadata.patch_tensor_key, metadata.global_mean_tensor_key}
    actual = set(tensors)
    if actual != expected:
        _fail(
            "tensor_keys",
            f"expected {sorted(expected)!r}, found {sorted(actual)!r}",
        )
    patch = tensors[metadata.patch_tensor_key]
    global_mean = tensors[metadata.global_mean_tensor_key]
    validate_feature_tensors(metadata, patch, global_mean)
    return patch, global_mean
