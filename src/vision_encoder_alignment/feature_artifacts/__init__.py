"""Canonical feature artifact metadata, validation, and I/O."""

from .checksum import sha256_file, verify_sha256
from .io import (
    read_feature_artifact,
    read_sidecar,
    write_feature_artifact,
    write_sidecar,
)
from .schema import (
    ArtifactValidationError,
    FeatureArtifactMetadata,
    LoadedFeatureArtifact,
)
from .validation import (
    ARTIFACT_SCHEMA_ID,
    canonical_tensor_keys,
    validate_feature_tensors,
    validate_metadata,
    validate_tensor_map,
)

__all__ = [
    "ARTIFACT_SCHEMA_ID",
    "ArtifactValidationError",
    "FeatureArtifactMetadata",
    "LoadedFeatureArtifact",
    "canonical_tensor_keys",
    "read_feature_artifact",
    "read_sidecar",
    "sha256_file",
    "validate_feature_tensors",
    "validate_metadata",
    "validate_tensor_map",
    "verify_sha256",
    "write_feature_artifact",
    "write_sidecar",
]
