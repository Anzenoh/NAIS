"""Atomic Safetensors and JSON sidecar I/O."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from .checksum import sha256_file, verify_sha256
from .schema import (
    ArtifactValidationError,
    FeatureArtifactMetadata,
    LoadedFeatureArtifact,
)
from .validation import validate_feature_tensors, validate_metadata, validate_tensor_map


def _atomic_replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _temporary_path(directory: Path, prefix: str) -> Path:
    descriptor, path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    os.close(descriptor)
    return Path(path)


def _remove_if_present(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def write_sidecar(path: str | Path, metadata: FeatureArtifactMetadata) -> None:
    """Write deterministic UTF-8 JSON metadata."""

    validate_metadata(metadata)
    serialized = json.dumps(
        metadata.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    sidecar_path = Path(path)
    with sidecar_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def read_sidecar(path: str | Path) -> FeatureArtifactMetadata:
    """Read and validate one strict JSON sidecar."""

    sidecar_path = Path(path)
    try:
        with sidecar_path.open("r", encoding="utf-8") as stream:
            raw: Any = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"sidecar: unable to read {sidecar_path}: {exc}") from exc
    metadata = FeatureArtifactMetadata.from_dict(raw)
    validate_metadata(metadata)
    return metadata


def write_feature_artifact(
    output_directory: str | Path,
    metadata: FeatureArtifactMetadata,
    patch: np.ndarray,
    global_mean: np.ndarray,
) -> FeatureArtifactMetadata:
    """Write a validated shard and commit the complete sidecar last."""

    validate_metadata(metadata)
    if metadata.run_status != "incomplete":
        raise ArtifactValidationError("run_status: writer input must be 'incomplete'")
    if metadata.checksum_value:
        raise ArtifactValidationError("checksum_value: writer input must be empty")
    validate_feature_tensors(metadata, patch, global_mean)

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensor_path = directory / metadata.safetensors_file_name
    sidecar_path = directory / metadata.sidecar_file_name
    if tensor_path.exists() or sidecar_path.exists():
        raise FileExistsError("artifact output paths must not already exist")

    temporary_tensor: Path | None = None
    temporary_sidecar: Path | None = None
    tensor_committed = False
    sidecar_committed = False
    try:
        temporary_tensor = _temporary_path(directory, ".tensor-")
        tensors = {
            metadata.patch_tensor_key: np.ascontiguousarray(patch),
            metadata.global_mean_tensor_key: np.ascontiguousarray(global_mean),
        }
        save_file(tensors, str(temporary_tensor))
        _atomic_replace(temporary_tensor, tensor_path)
        temporary_tensor = None
        tensor_committed = True

        checksum = sha256_file(tensor_path)
        completed_metadata = replace(
            metadata,
            checksum_value=checksum,
            run_status="complete",
        )
        validate_metadata(completed_metadata, require_complete=True)

        temporary_sidecar = _temporary_path(directory, ".sidecar-")
        write_sidecar(temporary_sidecar, completed_metadata)
        _atomic_replace(temporary_sidecar, sidecar_path)
        temporary_sidecar = None
        sidecar_committed = True
        return completed_metadata
    except Exception:
        if sidecar_committed:
            _remove_if_present(sidecar_path)
        if tensor_committed:
            _remove_if_present(tensor_path)
        raise
    finally:
        _remove_if_present(temporary_tensor)
        _remove_if_present(temporary_sidecar)


def read_feature_artifact(sidecar: str | Path) -> LoadedFeatureArtifact:
    """Read one complete shard after checksum and contract validation."""

    sidecar_path = Path(sidecar)
    metadata = read_sidecar(sidecar_path)
    validate_metadata(metadata, require_complete=True)
    if sidecar_path.name != metadata.sidecar_file_name:
        raise ArtifactValidationError(
            "sidecar_file_name: metadata does not match the opened sidecar"
        )

    tensor_path = sidecar_path.parent / metadata.safetensors_file_name
    if not tensor_path.is_file():
        raise ArtifactValidationError(
            f"safetensors_file_name: file does not exist: {metadata.safetensors_file_name}"
        )
    verify_sha256(tensor_path, metadata.checksum_value)
    try:
        tensors = load_file(str(tensor_path))
    except Exception as exc:
        raise ArtifactValidationError(f"safetensors: unable to load tensor file: {exc}") from exc
    patch, global_mean = validate_tensor_map(metadata, tensors)
    return LoadedFeatureArtifact(
        metadata=metadata,
        patch=patch,
        global_mean=global_mean,
    )
