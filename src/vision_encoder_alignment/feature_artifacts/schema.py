"""Typed feature artifact metadata structures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np


class ArtifactValidationError(ValueError):
    """Raised when an artifact violates the canonical contract."""


@dataclass(frozen=True, slots=True)
class FeatureArtifactMetadata:
    """Metadata serialized into one feature shard JSON sidecar."""

    schema_id: str
    run_id: str
    model_key: str
    tap_namespace: str
    tap_name: str
    patch_tensor_key: str
    global_mean_tensor_key: str
    image_ids: list[str]
    grid_height: int
    grid_width: int
    patch_count: int
    hidden_dimension: int
    dtype: str
    shard_index: int
    safetensors_file_name: str
    sidecar_file_name: str
    checksum_algorithm: str
    checksum_value: str
    created_at_utc: str
    run_status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary without extra fields."""

        return {
            "schema_id": self.schema_id,
            "run_id": self.run_id,
            "model_key": self.model_key,
            "tap_namespace": self.tap_namespace,
            "tap_name": self.tap_name,
            "patch_tensor_key": self.patch_tensor_key,
            "global_mean_tensor_key": self.global_mean_tensor_key,
            "image_ids": list(self.image_ids),
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "patch_count": self.patch_count,
            "hidden_dimension": self.hidden_dimension,
            "dtype": self.dtype,
            "shard_index": self.shard_index,
            "safetensors_file_name": self.safetensors_file_name,
            "sidecar_file_name": self.sidecar_file_name,
            "checksum_algorithm": self.checksum_algorithm,
            "checksum_value": self.checksum_value,
            "created_at_utc": self.created_at_utc,
            "run_status": self.run_status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureArtifactMetadata":
        """Construct metadata while rejecting missing or unknown sidecar fields."""

        if not isinstance(value, Mapping):
            raise ArtifactValidationError("sidecar: expected a JSON object")
        expected = {field.name for field in fields(cls)}
        actual = set(value)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing:
            raise ArtifactValidationError(
                f"sidecar: missing required fields: {', '.join(missing)}"
            )
        if unknown:
            raise ArtifactValidationError(
                f"sidecar: unknown fields are not allowed: {', '.join(unknown)}"
            )
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True, slots=True)
class LoadedFeatureArtifact:
    """A validated complete feature shard returned by the reader."""

    metadata: FeatureArtifactMetadata
    patch: np.ndarray
    global_mean: np.ndarray
