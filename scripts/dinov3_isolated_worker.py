"""Offline worker for the persistent DINOv3 process/artifact boundary.

The module intentionally imports only the Python standard library at import time.
Heavy runtime imports happen only after the socket guard and offline environment
have been installed by :func:`run_worker`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import random
import re
import socket
import sys
from typing import Any


REQUEST_SCHEMA_ID = "vea.dinov3_isolated_worker_request.v0"
RESPONSE_SCHEMA_ID = "vea.dinov3_isolated_worker_response.v0"
PROFILE_SCHEMA_ID = "vea.dinov3_isolated_runtime_profile.v0"
RUNTIME_MANIFEST_SCHEMA_ID = "vea.dinov3_persistent_runtime.v0"
ALLOWED_OPERATIONS = frozenset(
    {"artifact_boundary_self_test", "runtime_self_test"}
)
PREPROCESSING_PROFILE_ID = "dinov3-manual-fp32-imagenet-v1"

EXPECTED_INPUT_SHA256 = (
    "b0fa2f6e3b2fbca50cde7e4a2dc3a65a0f788a184aec382365e520d54b468411"
)
EXPECTED_NORMALIZED_SHA256 = (
    "c0ebe2af6db2e61d052dc58272379beef749a4c5392d4e2204a11862cf172526"
)
EXPECTED_CANDIDATE_FREEZE_SHA256 = (
    "995687ed934c0a94b52a040af048f87bc8de4e5089475f70565ad37f1f33ecd9"
)

IMAGE_IDS = ("image-b", "image-a")
TAP_NAMES = ("embedding_output", "block_06")
PATCH_SHAPE = (2, 4, 3)
GLOBAL_SHAPE = (2, 3)

_REQUEST_FIELDS = frozenset(
    {
        "schema_id",
        "operation",
        "run_id",
        "repository_head",
        "runtime_profile_sha256",
        "workspace_relative_inbound_artifact_path",
        "workspace_relative_outbound_artifact_path",
        "deterministic_seed",
        "expected_preprocessing_profile_id",
    }
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_MAX_JSON_BYTES = 1024 * 1024
_MAX_REQUEST_BYTES = 64 * 1024
_RUNTIME_MANIFEST_FIELDS = frozenset(
    {
        "schema_id",
        "runtime_id",
        "repository_profile_sha256",
        "provisioned_at_utc",
        "python_version",
        "shared_torch",
        "candidate_local_packages",
        "candidate_freeze",
        "metadata_files",
        "wheel_files",
        "installed_sources",
        "current_base_freeze_sha256",
        "dependency_validation",
        "forbidden_candidate_package_count",
        "torchvision",
        "manual_preprocessing_profile_id",
        "model_weight_present",
        "network_independent_verification_status",
    }
)

_INBOUND_VALUE_HASHES = (
    (
        "7eb4ca01077988fa5de9ffff007bd66097cc410ee7b919afecb87a2554a85f90",
        "3f19a90ad27c0b4dee2ef959aa840a55530e34f4b4742ca1fbed21eda7d991c9",
    ),
    (
        "eb897d1b2e4dafb832512bffc8e38e037df31e61262833142062b7329f8ebe1c",
        "e6b72e4fc33d85501da61775976d1601ab74ff8227b26646b61fceccd1ec6ccb",
    ),
)
_INBOUND_FILE_HASHES = (
    "31c2f1c7aab1bfddd9aa025a406c8425418155ff7b57339365b5139fe1ea69df",
    "ac88cb5d2a8a8a28779ec6444270db0e8352953ff2edd8a7d634d15c60268865",
)
_OUTBOUND_VALUE_HASHES = (
    (
        "45a99655901702d55ab6284a18aed6a5e16677181d16c7a7517b68c2ae2c0c7a",
        "a8ed788140a07f4db935b3205c81a27c38be1aa885bf659220704528a591e010",
    ),
    (
        "1b089fcdd28fee09782f771ed7ac440a0f171296f270978c79d0bd979f615367",
        "45cab7d97e645faad748045c80472b7c176d441272669c941a108dfc9294f014",
    ),
)
_OUTBOUND_FILE_HASHES = (
    "19621cacf073aa0433e5e502472bdb968c467427e2be390f363caa475746fdfa",
    "c9115946aab2a02d1d2cc6081aaad01589c7809efb1ba27b31f08e6981b8b533",
)


class WorkerValidationError(ValueError):
    """Raised when a request or runtime violates the worker contract."""


class _SocketGuard:
    """Deny outbound sockets while retaining an auditable attempt count."""

    def __init__(self) -> None:
        self.attempt_count = 0
        self._socket_class = socket.socket
        self._create_connection = socket.create_connection

    def install(self) -> None:
        guard = self
        original = self._socket_class

        class GuardedSocket(original):  # type: ignore[misc, valid-type]
            def connect(self, address: object) -> None:
                guard.attempt_count += 1
                raise RuntimeError("network access is disabled for the isolated worker")

            def connect_ex(self, address: object) -> int:
                guard.attempt_count += 1
                raise RuntimeError("network access is disabled for the isolated worker")

        def guarded_create_connection(*args: object, **kwargs: object) -> None:
            guard.attempt_count += 1
            raise RuntimeError("network access is disabled for the isolated worker")

        socket.socket = GuardedSocket
        socket.create_connection = guarded_create_connection  # type: ignore[assignment]

    def restore(self) -> None:
        socket.socket = self._socket_class
        socket.create_connection = self._create_connection


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WorkerValidationError("duplicate JSON field")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise WorkerValidationError(f"non-finite JSON number is forbidden: {value}")


def _read_json(path: Path, *, maximum_bytes: int = _MAX_JSON_BYTES) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        if len(payload) > maximum_bytes:
            raise WorkerValidationError("JSON document exceeds the size limit")
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except WorkerValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerValidationError("unable to read required JSON") from exc
    if not isinstance(value, dict):
        raise WorkerValidationError("JSON document must be an object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise WorkerValidationError(f"{field}: expected lowercase SHA-256")
    return value


def _require_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise WorkerValidationError(f"{field}: invalid identifier")
    return value


def _require_git_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise WorkerValidationError(f"{field}: expected lowercase 40-hex Git SHA")
    return value


def _resolve_relative(
    workspace: Path, value: object, field: str, *, must_exist: bool
) -> Path:
    if not isinstance(value, str) or not value:
        raise WorkerValidationError(f"{field}: expected a relative path")
    relative = Path(value)
    windows = PureWindowsPath(value)
    if relative.is_absolute() or windows.is_absolute() or windows.drive:
        raise WorkerValidationError(f"{field}: absolute paths are forbidden")
    if any(part in {"", ".", ".."} for part in windows.parts):
        raise WorkerValidationError(f"{field}: path traversal is forbidden")
    root = workspace.resolve(strict=True)
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkerValidationError(f"{field}: workspace escape is forbidden") from exc
    if must_exist and not candidate.exists():
        raise WorkerValidationError(f"{field}: required path does not exist")
    return candidate


def _parse_request(path: Path) -> dict[str, Any]:
    request = _read_json(path, maximum_bytes=_MAX_REQUEST_BYTES)
    actual = frozenset(request)
    if actual != _REQUEST_FIELDS:
        missing = sorted(_REQUEST_FIELDS - actual)
        unknown = sorted(actual - _REQUEST_FIELDS)
        raise WorkerValidationError(
            f"request fields mismatch; missing={missing!r}, unknown={unknown!r}"
        )
    if request["schema_id"] != REQUEST_SCHEMA_ID:
        raise WorkerValidationError("request schema_id is not supported")
    if request["operation"] not in ALLOWED_OPERATIONS:
        raise WorkerValidationError("request operation is not supported")
    _require_identifier(request["run_id"], "run_id")
    _require_git_sha(request["repository_head"], "repository_head")
    _require_sha256(request["runtime_profile_sha256"], "runtime_profile_sha256")
    seed = request["deterministic_seed"]
    if type(seed) is not int or seed != 0:
        raise WorkerValidationError("deterministic_seed: expected integer 0")
    if request["expected_preprocessing_profile_id"] != PREPROCESSING_PROFILE_ID:
        raise WorkerValidationError("preprocessing profile ID mismatch")
    return request


def _validate_profile_and_manifest(
    profile_path: Path, runtime_root: Path, request: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    profile = _read_json(profile_path)
    if profile.get("schema_id") != PROFILE_SCHEMA_ID:
        raise WorkerValidationError("runtime profile schema mismatch")
    profile_sha256 = _sha256_file(profile_path)
    if profile_sha256 != request["runtime_profile_sha256"]:
        raise WorkerValidationError("runtime profile SHA-256 mismatch")
    runtime_id = _require_identifier(profile.get("runtime_id"), "runtime_id")
    if profile.get("allowed_worker_operations") != [
        "artifact_boundary_self_test",
        "runtime_self_test",
    ]:
        raise WorkerValidationError("runtime profile worker operation mismatch")
    protocol = profile.get("worker_protocol")
    if not isinstance(protocol, dict) or protocol != {
        "request_schema_id": REQUEST_SCHEMA_ID,
        "response_schema_id": RESPONSE_SCHEMA_ID,
    }:
        raise WorkerValidationError("runtime profile worker protocol mismatch")
    preprocessing = profile.get("manual_preprocessing")
    if not isinstance(preprocessing, dict):
        raise WorkerValidationError("runtime profile preprocessing is missing")
    preprocessing_checks = {
        "profile_id": PREPROCESSING_PROFILE_ID,
        "expected_input_sha256": EXPECTED_INPUT_SHA256,
        "expected_output_sha256": EXPECTED_NORMALIZED_SHA256,
        "input_shape": [2, 224, 224, 3],
        "output_shape": [2, 3, 224, 224],
        "output_dtype": "float32",
        "rescale_factor_dtype": "float32",
    }
    if any(preprocessing.get(key) != value for key, value in preprocessing_checks.items()):
        raise WorkerValidationError("runtime profile preprocessing identity mismatch")
    candidate = profile.get("candidate")
    if not isinstance(candidate, dict):
        raise WorkerValidationError("runtime profile candidate identity is missing")
    expected_freeze = candidate.get("expected_freeze")
    if not isinstance(expected_freeze, dict) or expected_freeze.get("sha256") != (
        EXPECTED_CANDIDATE_FREEZE_SHA256
    ):
        raise WorkerValidationError("runtime profile candidate freeze mismatch")

    manifest_path = runtime_root / "runtime-manifest.json"
    manifest = _read_json(manifest_path)
    if frozenset(manifest) != _RUNTIME_MANIFEST_FIELDS:
        raise WorkerValidationError("runtime manifest fields mismatch")
    if manifest.get("schema_id") != RUNTIME_MANIFEST_SCHEMA_ID:
        raise WorkerValidationError("runtime manifest schema mismatch")
    if manifest.get("runtime_id") != runtime_id:
        raise WorkerValidationError("runtime manifest runtime ID mismatch")
    if manifest.get("repository_profile_sha256") != profile_sha256:
        raise WorkerValidationError("runtime manifest profile SHA-256 mismatch")
    candidate_freeze = manifest.get("candidate_freeze")
    if not isinstance(candidate_freeze, dict) or frozenset(candidate_freeze) != {
        "bytes",
        "line_count",
        "sha256",
    }:
        raise WorkerValidationError("runtime manifest candidate freeze fields mismatch")
    if candidate_freeze.get("bytes") != 5206 or candidate_freeze.get("line_count") != 275:
        raise WorkerValidationError("runtime manifest candidate freeze size mismatch")
    freeze_sha = _require_sha256(
        candidate_freeze.get("sha256"), "candidate_freeze.sha256"
    )
    if freeze_sha != EXPECTED_CANDIDATE_FREEZE_SHA256:
        raise WorkerValidationError("candidate freeze SHA-256 mismatch")
    if manifest.get("manual_preprocessing_profile_id") != PREPROCESSING_PROFILE_ID:
        raise WorkerValidationError("runtime manifest preprocessing profile mismatch")
    if manifest.get("model_weight_present") is not False:
        raise WorkerValidationError("runtime manifest reports model weights")
    if manifest.get("network_independent_verification_status") != "VERIFIED":
        raise WorkerValidationError("runtime manifest is not offline-verified")
    torchvision = manifest.get("torchvision")
    if torchvision != {"distribution_count": 0, "module_count": 0}:
        raise WorkerValidationError("runtime manifest reports forbidden vision use")

    config_path = runtime_root / "metadata" / "config.json"
    processor_path = runtime_root / "metadata" / "preprocessor_config.json"
    if not config_path.is_file() or not processor_path.is_file():
        raise WorkerValidationError("runtime metadata layout is incomplete")
    return profile, manifest, _sha256_file(manifest_path)


def generate_deterministic_images() -> Any:
    """Recreate the exact two-image T007A-1R uint8 input batch."""

    import numpy as np

    y, x = np.indices((224, 224), dtype=np.uint16)
    gradient = np.stack(
        (x % 256, y % 256, (x + y) % 256), axis=-1
    ).astype(np.uint8)
    checker_base = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8)
    checker = np.stack(
        (
            checker_base,
            (3 * x + 5 * y) % 256,
            np.where(checker_base == 0, (7 * x) % 256, (11 * y) % 256),
        ),
        axis=-1,
    ).astype(np.uint8)
    images = np.ascontiguousarray(np.stack((gradient, checker)), dtype=np.uint8)
    if images.shape != (2, 224, 224, 3) or images.dtype != np.dtype(np.uint8):
        raise WorkerValidationError("deterministic input geometry mismatch")
    if _sha256_bytes(images.tobytes(order="C")) != EXPECTED_INPUT_SHA256:
        raise WorkerValidationError("deterministic input SHA-256 mismatch")
    return images


def manual_preprocess(images: Any) -> Any:
    """Apply the accepted metadata-driven DINOv3 FP32 numeric sequence."""

    import numpy as np

    if not isinstance(images, np.ndarray):
        raise WorkerValidationError("manual input must be a NumPy array")
    if images.shape != (2, 224, 224, 3) or images.dtype != np.dtype(np.uint8):
        raise WorkerValidationError("manual input contract mismatch")
    normalized = images.astype(np.float32, copy=True)
    normalized *= np.float32(1.0 / 255.0)
    normalized -= np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    normalized /= np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = normalized.transpose(0, 3, 1, 2)
    normalized = np.ascontiguousarray(normalized, dtype=np.float32)
    if normalized.shape != (2, 3, 224, 224):
        raise WorkerValidationError("manual output shape mismatch")
    if normalized.dtype != np.dtype(np.float32) or not np.isfinite(normalized).all():
        raise WorkerValidationError("manual output dtype or finiteness mismatch")
    if _sha256_bytes(normalized.tobytes(order="C")) != EXPECTED_NORMALIZED_SHA256:
        raise WorkerValidationError("manual preprocessing SHA-256 mismatch")
    return normalized


def _synthetic_tensors(offset: int) -> tuple[Any, Any]:
    import numpy as np

    base = np.arange(24, dtype=np.float32).reshape(PATCH_SHAPE)
    patch = np.ascontiguousarray(base + np.float32(offset), dtype=np.float32)
    global_mean = np.ascontiguousarray(
        patch.mean(axis=1, dtype=np.float32), dtype=np.float32
    )
    return patch, global_mean


def _artifact_entry(loaded: Any, sidecar_name: str, tensor_path: Path) -> dict[str, Any]:
    import numpy as np

    metadata = loaded.metadata
    verify_result = _artifact_api()[5](tensor_path, metadata.checksum_value)
    if verify_result is not None:
        raise WorkerValidationError("verify_sha256 returned a non-None success value")
    if _artifact_api()[4](tensor_path) != metadata.checksum_value:
        raise WorkerValidationError("artifact checksum mismatch")
    return {
        "checksum": metadata.checksum_value,
        "dtype": loaded.patch.dtype.name,
        "global_shape": list(loaded.global_mean.shape),
        "global_values_sha256": _sha256_bytes(
            np.ascontiguousarray(loaded.global_mean).tobytes(order="C")
        ),
        "image_ids": list(metadata.image_ids),
        "patch_shape": list(loaded.patch.shape),
        "patch_values_sha256": _sha256_bytes(
            np.ascontiguousarray(loaded.patch).tobytes(order="C")
        ),
        "sidecar": sidecar_name,
        "tap_name": metadata.tap_name,
        "tensor_keys": [metadata.patch_tensor_key, metadata.global_mean_tensor_key],
    }


def _artifact_api() -> tuple[Any, ...]:
    from vision_encoder_alignment.feature_artifacts import (
        ARTIFACT_SCHEMA_ID,
        FeatureArtifactMetadata,
        canonical_tensor_keys,
        read_feature_artifact,
        sha256_file,
        verify_sha256,
        write_feature_artifact,
    )

    return (
        ARTIFACT_SCHEMA_ID,
        FeatureArtifactMetadata,
        canonical_tensor_keys,
        read_feature_artifact,
        sha256_file,
        verify_sha256,
        write_feature_artifact,
    )


def _validate_artifact_directory(
    directory: Path, offsets: tuple[int, int], run_id: str
) -> dict[str, Any]:
    import numpy as np

    _, _, canonical_tensor_keys, read_feature_artifact, _, _, _ = _artifact_api()
    entries: list[dict[str, Any]] = []
    for shard_index, (tap_name, offset) in enumerate(zip(TAP_NAMES, offsets, strict=True)):
        sidecar_name = f"shard-{shard_index:05d}.json"
        sidecar_path = directory / sidecar_name
        loaded = read_feature_artifact(sidecar_path)
        patch, global_mean = _synthetic_tensors(offset)
        patch_key, global_key = canonical_tensor_keys(tap_name)
        metadata = loaded.metadata
        expected_metadata = {
            "schema_id": "vea.feature_artifact.v0",
            "run_id": run_id,
            "model_key": "dinov3_vit_b16",
            "tap_namespace": "backbone",
            "tap_name": tap_name,
            "patch_tensor_key": patch_key,
            "global_mean_tensor_key": global_key,
            "image_ids": list(IMAGE_IDS),
            "grid_height": 2,
            "grid_width": 2,
            "patch_count": 4,
            "hidden_dimension": 3,
            "dtype": "float32",
            "shard_index": shard_index,
            "safetensors_file_name": f"shard-{shard_index:05d}.safetensors",
            "sidecar_file_name": sidecar_name,
            "checksum_algorithm": "sha256",
            "created_at_utc": "2026-07-20T00:00:00Z",
            "run_status": "complete",
        }
        actual_metadata = metadata.to_dict()
        expected_metadata["checksum_value"] = actual_metadata["checksum_value"]
        if actual_metadata != expected_metadata:
            raise WorkerValidationError("artifact metadata mismatch")
        np.testing.assert_array_equal(loaded.patch, patch)
        np.testing.assert_array_equal(loaded.global_mean, global_mean)
        np.testing.assert_array_equal(
            loaded.global_mean, loaded.patch.mean(axis=1, dtype=np.float32)
        )
        tensor_path = directory / metadata.safetensors_file_name
        entry = _artifact_entry(loaded, sidecar_name, tensor_path)
        if entry["checksum"] != _INBOUND_FILE_HASHES[shard_index]:
            raise WorkerValidationError("inbound Safetensors checksum mismatch")
        if entry["patch_values_sha256"] != _INBOUND_VALUE_HASHES[shard_index][0]:
            raise WorkerValidationError("inbound patch-value SHA-256 mismatch")
        if entry["global_values_sha256"] != _INBOUND_VALUE_HASHES[shard_index][1]:
            raise WorkerValidationError("inbound global-value SHA-256 mismatch")
        entries.append(entry)
    return {
        "validated": True,
        "image_ids": list(IMAGE_IDS),
        "tap_order": list(TAP_NAMES),
        "patch_shape": list(PATCH_SHAPE),
        "global_shape": list(GLOBAL_SHAPE),
        "dtype": "float32",
        "entries": entries,
    }


def _write_artifact_directory(directory: Path, run_id: str) -> dict[str, Any]:
    (
        artifact_schema_id,
        metadata_class,
        canonical_tensor_keys,
        read_feature_artifact,
        _,
        _,
        write_feature_artifact,
    ) = _artifact_api()
    entries: list[dict[str, Any]] = []
    for shard_index, (tap_name, offset) in enumerate(
        zip(TAP_NAMES, (0, 100), strict=True)
    ):
        patch, global_mean = _synthetic_tensors(offset)
        patch_key, global_key = canonical_tensor_keys(tap_name)
        sidecar_name = f"shard-{shard_index:05d}.json"
        tensor_name = f"shard-{shard_index:05d}.safetensors"
        metadata = metadata_class(
            schema_id=artifact_schema_id,
            run_id=run_id,
            model_key="dinov3_vit_b16",
            tap_namespace="backbone",
            tap_name=tap_name,
            patch_tensor_key=patch_key,
            global_mean_tensor_key=global_key,
            image_ids=list(IMAGE_IDS),
            grid_height=2,
            grid_width=2,
            patch_count=4,
            hidden_dimension=3,
            dtype="float32",
            shard_index=shard_index,
            safetensors_file_name=tensor_name,
            sidecar_file_name=sidecar_name,
            checksum_algorithm="sha256",
            checksum_value="",
            created_at_utc="2026-07-20T00:00:00Z",
            run_status="incomplete",
        )
        completed = write_feature_artifact(directory, metadata, patch, global_mean)
        loaded = read_feature_artifact(directory / completed.sidecar_file_name)
        entry = _artifact_entry(
            loaded,
            completed.sidecar_file_name,
            directory / completed.safetensors_file_name,
        )
        if entry["checksum"] != _OUTBOUND_FILE_HASHES[shard_index]:
            raise WorkerValidationError("outbound Safetensors checksum mismatch")
        if entry["patch_values_sha256"] != _OUTBOUND_VALUE_HASHES[shard_index][0]:
            raise WorkerValidationError("outbound patch-value SHA-256 mismatch")
        if entry["global_values_sha256"] != _OUTBOUND_VALUE_HASHES[shard_index][1]:
            raise WorkerValidationError("outbound global-value SHA-256 mismatch")
        entries.append(entry)
    return {
        "validated": True,
        "image_ids": list(IMAGE_IDS),
        "tap_order": list(TAP_NAMES),
        "patch_shape": list(PATCH_SHAPE),
        "global_shape": list(GLOBAL_SHAPE),
        "dtype": "float32",
        "entries": entries,
    }


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    import numpy as np

    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _runtime_self_test(
    seed: int, runtime_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch
    from transformers.models.dinov3_vit.configuration_dinov3_vit import (
        DINOv3ViTConfig,
    )
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()

    images = generate_deterministic_images()
    input_before = _sha256_bytes(images.tobytes(order="C"))
    normalized = manual_preprocess(images)
    input_after = _sha256_bytes(images.tobytes(order="C"))
    if input_before != input_after:
        raise WorkerValidationError("manual preprocessing mutated its caller input")

    config_path = runtime_root / "metadata" / "config.json"
    config_dict = _read_json(config_path)
    config = DINOv3ViTConfig.from_dict(config_dict)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        model = DINOv3ViTModel(config)
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise WorkerValidationError("random model is not frozen")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_tensor_count = sum(1 for _ in model.parameters())
    if parameter_count != 85_660_416 or parameter_tensor_count != 211:
        raise WorkerValidationError("random model parameter identity mismatch")
    state_before = _state_sha256(model)

    pixel_values = torch.from_numpy(normalized)
    with torch.inference_mode():
        first = model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        second = model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
    expected_shape = (2, 201, 768)
    if tuple(first.last_hidden_state.shape) != expected_shape:
        raise WorkerValidationError("random model last-hidden shape mismatch")
    if first.hidden_states is None or len(first.hidden_states) != 13:
        raise WorkerValidationError("random model hidden-state count mismatch")
    if any(tuple(value.shape) != expected_shape for value in first.hidden_states):
        raise WorkerValidationError("random model hidden-state shape mismatch")
    if first.pooler_output is None or tuple(first.pooler_output.shape) != (2, 768):
        raise WorkerValidationError("random model pooler shape mismatch")
    values = list(first.hidden_states) + [first.last_hidden_state, first.pooler_output]
    if not all(torch.isfinite(value).all().item() for value in values):
        raise WorkerValidationError("random model output contains NaN or Inf")
    if not torch.equal(first.last_hidden_state, second.last_hidden_state):
        raise WorkerValidationError("random model repeat last-hidden mismatch")
    if not all(
        torch.equal(left, right)
        for left, right in zip(first.hidden_states, second.hidden_states, strict=True)
    ):
        raise WorkerValidationError("random model repeat hidden-state mismatch")
    if not torch.equal(first.pooler_output, second.pooler_output):
        raise WorkerValidationError("random model repeat pooler mismatch")
    state_after = _state_sha256(model)
    if state_before != state_after:
        raise WorkerValidationError("random model state changed during inference")

    python_rng_after = random.getstate()
    numpy_rng_after = np.random.get_state()
    torch_rng_after = torch.random.get_rng_state().clone()
    rng = {
        "python_unchanged": python_rng_before == python_rng_after,
        "numpy_unchanged": _numpy_rng_equal(numpy_rng_before, numpy_rng_after),
        "torch_unchanged": torch.equal(torch_rng_before, torch_rng_after),
    }
    if not all(rng.values()):
        raise WorkerValidationError("runtime self-test changed caller RNG state")

    last_hidden = first.last_hidden_state.detach().cpu().contiguous().numpy()
    preprocessing = {
        "executed": True,
        "profile_id": PREPROCESSING_PROFILE_ID,
        "input_shape": [2, 224, 224, 3],
        "input_dtype": "uint8",
        "input_sha256": input_before,
        "input_unchanged": True,
        "output_shape": [2, 3, 224, 224],
        "output_dtype": "float32",
        "output_c_contiguous": bool(normalized.flags.c_contiguous),
        "output_finite": True,
        "normalized_sha256": _sha256_bytes(normalized.tobytes(order="C")),
        "resize_count": 0,
        "crop_count": 0,
        "pil_conversion_count": 0,
        "image_processor_construction_count": 0,
    }
    random_model = {
        "executed": True,
        "class": "DINOv3ViTModel",
        "random_initialization_seed": seed,
        "checkpoint_weight_bytes": 0,
        "device": "cpu",
        "dtype": "float32",
        "evaluation_mode": True,
        "frozen": True,
        "parameter_count": parameter_count,
        "parameter_tensor_count": parameter_tensor_count,
        "total_tokens": 201,
        "hidden_state_count": 13,
        "hidden_state_shapes": [list(value.shape) for value in first.hidden_states],
        "last_hidden_shape": list(first.last_hidden_state.shape),
        "pooler_shape": list(first.pooler_output.shape),
        "finite": True,
        "repeat_exact": True,
        "last_hidden_sha256": _sha256_bytes(last_hidden.tobytes(order="C")),
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": True,
        "rng": rng,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if random_model["cuda_available"] or random_model["cuda_device_count"] != 0:
        raise WorkerValidationError("runtime self-test is not CPU-only")
    return preprocessing, random_model


def _torchvision_presence() -> tuple[int, int]:
    distribution_count = 0
    try:
        importlib.metadata.distribution("torchvision")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        distribution_count = 1
    module_spec_count = int(importlib.util.find_spec("torchvision") is not None)
    module_count = sum(
        1 for name in sys.modules if name == "torchvision" or name.startswith("torchvision.")
    )
    return distribution_count + module_spec_count, module_count


def run_worker(
    *,
    profile_path: Path,
    runtime_root: Path,
    workspace: Path,
    request_path: Path,
    response_path: Path,
) -> dict[str, Any]:
    """Execute one strict request and return the response dictionary."""

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(workspace / ".worker-cache" / "huggingface"),
            "XDG_CACHE_HOME": str(workspace / ".worker-cache"),
        }
    )
    guard = _SocketGuard()
    guard.install()
    try:
        request = _parse_request(request_path)
        profile, manifest, manifest_sha256 = _validate_profile_and_manifest(
            profile_path, runtime_root, request
        )
        inbound = _resolve_relative(
            workspace,
            request["workspace_relative_inbound_artifact_path"],
            "workspace_relative_inbound_artifact_path",
            must_exist=True,
        )
        outbound = _resolve_relative(
            workspace,
            request["workspace_relative_outbound_artifact_path"],
            "workspace_relative_outbound_artifact_path",
            must_exist=False,
        )
        if inbound == outbound or outbound.exists():
            raise WorkerValidationError("outbound artifact path must be fresh")

        presence_count, imported_count_before = _torchvision_presence()
        if presence_count != 0 or imported_count_before != 0:
            raise WorkerValidationError("forbidden vision runtime is present")

        inbound_result = _validate_artifact_directory(
            inbound, (1000, 1100), request["run_id"]
        )
        outbound_result = _write_artifact_directory(outbound, request["run_id"])
        if request["operation"] == "runtime_self_test":
            preprocessing, random_model = _runtime_self_test(
                request["deterministic_seed"], runtime_root
            )
        else:
            preprocessing = {
                "executed": False,
                "profile_id": PREPROCESSING_PROFILE_ID,
                "normalized_sha256": None,
            }
            random_model = {"executed": False, "shapes": []}

        presence_after, imported_count_after = _torchvision_presence()
        if presence_after != 0 or imported_count_after != 0:
            raise WorkerValidationError("forbidden vision runtime was loaded")
        if guard.attempt_count != 0:
            raise WorkerValidationError("isolated runtime attempted network access")

        runtime_id = _require_identifier(profile.get("runtime_id"), "runtime_id")
        candidate_freeze = manifest["candidate_freeze"]
        freeze_sha = _require_sha256(
            candidate_freeze.get("sha256"), "candidate_freeze.sha256"
        )
        markers = [
            "DINOV3_PERSISTENT_RUNTIME_MANIFEST_OK",
            "DINOV3_PERSISTENT_INBOUND_ARTIFACT_OK",
            "DINOV3_PERSISTENT_OUTBOUND_ARTIFACT_OK",
            "DINOV3_PERSISTENT_RUNTIME_NETWORK_GUARD_OK",
        ]
        if request["operation"] == "runtime_self_test":
            markers.extend(
                [
                    "DINOV3_PERSISTENT_MANUAL_PREPROCESSING_OK",
                    "DINOV3_PERSISTENT_RANDOM_FORWARD_OK",
                ]
            )
        response = {
            "schema_id": RESPONSE_SCHEMA_ID,
            "operation": request["operation"],
            "status": "success",
            "run_id": request["run_id"],
            "runtime_id": runtime_id,
            "runtime_manifest_sha256": manifest_sha256,
            "candidate_freeze_sha256": freeze_sha,
            "manual_preprocessing_hash": preprocessing["normalized_sha256"],
            "manual_preprocessing": preprocessing,
            "random_model_shapes": {
                "last_hidden": random_model.get("last_hidden_shape"),
                "hidden_states": random_model.get("hidden_state_shapes", []),
                "pooler": random_model.get("pooler_shape"),
            },
            "random_model": random_model,
            "artifact_validation_result": {
                "inbound": inbound_result,
                "outbound": outbound_result,
            },
            "inbound_checksums": [
                entry["checksum"] for entry in inbound_result["entries"]
            ],
            "outbound_checksums": [
                entry["checksum"] for entry in outbound_result["entries"]
            ],
            "socket_attempt_count": guard.attempt_count,
            "model_weight_byte_count": 0,
            "torchvision_use_count": presence_after + imported_count_after,
            "warnings": [],
            "failure_classification": None,
            "markers": markers,
        }
        _write_json(response_path, response)
        return response
    finally:
        guard.restore()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument(
        "--workspace-root", "--workspace", dest="workspace_root", required=True, type=Path
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--inbound-artifact", type=Path)
    parser.add_argument("--outbound-artifact", type=Path)
    return parser


def _resolve_cli_workspace_path(
    workspace: Path, value: Path, field: str, *, must_exist: bool
) -> Path:
    root = workspace.resolve(strict=True)
    candidate = value if value.is_absolute() else root / value
    candidate = candidate.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkerValidationError(f"{field}: workspace escape is forbidden") from exc
    if must_exist and not candidate.exists():
        raise WorkerValidationError(f"{field}: required path does not exist")
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        workspace = args.workspace_root.resolve(strict=True)
        request_path = _resolve_cli_workspace_path(
            workspace, args.request, "request", must_exist=True
        )
        response_path = _resolve_cli_workspace_path(
            workspace, args.response, "response", must_exist=False
        )
        if response_path.exists():
            raise WorkerValidationError("response path must be fresh")
        request = _parse_request(request_path)
        inbound = _resolve_relative(
            workspace,
            request["workspace_relative_inbound_artifact_path"],
            "workspace_relative_inbound_artifact_path",
            must_exist=True,
        )
        outbound = _resolve_relative(
            workspace,
            request["workspace_relative_outbound_artifact_path"],
            "workspace_relative_outbound_artifact_path",
            must_exist=False,
        )
        if args.inbound_artifact is not None and _resolve_cli_workspace_path(
            workspace, args.inbound_artifact, "inbound_artifact", must_exist=True
        ) != inbound:
            raise WorkerValidationError("inbound CLI path does not match request")
        if args.outbound_artifact is not None and _resolve_cli_workspace_path(
            workspace, args.outbound_artifact, "outbound_artifact", must_exist=False
        ) != outbound:
            raise WorkerValidationError("outbound CLI path does not match request")
        response = run_worker(
            profile_path=args.profile.resolve(strict=True),
            runtime_root=args.runtime_root.resolve(strict=True),
            workspace=workspace,
            request_path=request_path,
            response_path=response_path,
        )
    except Exception as exc:
        classification = type(exc).__name__
        print(f"DINOV3_ISOLATED_WORKER_FAILED:{classification}", file=sys.stderr)
        return 1
    for marker in response["markers"]:
        print(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
