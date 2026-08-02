"""Probe the pinned DINOv3 checkpoint inside its persistent CPU runtime.

This entry point deliberately has a standard-library-only import surface.
Offline policy, path confinement, and the socket guard are established before
NumPy, PyTorch, Transformers, Safetensors, or the canonical feature APIs are
imported.  The response contains semantic evidence and hashes only; model
tensors are never serialized into the response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import re
import socket
import stat
import sys
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA_ID = "vea.dinov3_pretrained_semantic_probe_request.v0"
RESPONSE_SCHEMA_ID = "vea.dinov3_pretrained_semantic_probe_response.v0"
SEMANTIC_MARKER = "DINOV3_PRETRAINED_RUNTIME_SEMANTICS_OK"

PROBE_MARKERS = (
    "DINOV3_SEMANTIC_PROFILE_OK",
    "DINOV3_PRETRAINED_MODEL_LOAD_OK",
    "DINOV3_PRETRAINED_EMBEDDING_OUTPUT_OK",
    "DINOV3_PRETRAINED_PREFIX_TOKEN_LAYOUT_OK",
    "DINOV3_PRETRAINED_PATCH_ORDER_OK",
    "DINOV3_PRETRAINED_BLOCK_MAPPING_OK",
    "DINOV3_PRETRAINED_FINAL_NORM_OK",
    "DINOV3_PRETRAINED_POOLER_OK",
    "DINOV3_PRETRAINED_CANONICAL_RESULT_OK",
    "DINOV3_PRETRAINED_ARTIFACT_ROUNDTRIP_OK",
    "DINOV3_PRETRAINED_REPEAT_DETERMINISM_OK",
    "DINOV3_PRETRAINED_RNG_STATE_OK",
    "DINOV3_PRETRAINED_NETWORK_GUARD_OK",
    SEMANTIC_MARKER,
)

MODEL_KEY = "dinov3_vit_b16"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
MODEL_STATE_SHA256 = "c382afd66ac658255f9f1406bad1e5e5fcf410af40cbc5c306b770792a4f30fd"
INPUT_SHA256 = "b0fa2f6e3b2fbca50cde7e4a2dc3a65a0f788a184aec382365e520d54b468411"
NORMALIZED_SHA256 = "c0ebe2af6db2e61d052dc58272379beef749a4c5392d4e2204a11862cf172526"
IMAGE_IDS = ("image-b", "image-a")
SELECTED_ARTIFACT_TAPS = ("embedding_output", "block_06", "final_norm")
SPECIAL_TOKEN_POLICY = "dinov3_cls1_register4_prefix_verified_runtime_v1"

EXPECTED_PARAMETER_COUNT = 85_660_416
EXPECTED_PARAMETER_TENSOR_COUNT = 211
EXPECTED_LAYER_COUNT = 12
EXPECTED_HIDDEN_DIMENSION = 768
EXPECTED_GRID = (14, 14)
EXPECTED_PATCH_COUNT = 196
EXPECTED_TOTAL_TOKENS = 201
EXPECTED_HIDDEN_STATE_COUNT = 13
EXPECTED_CHECKPOINT_FILES = {
    "config.json": "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec",
    "preprocessor_config.json": "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
    "model.safetensors": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
}
EXPECTED_INSTALLED_SOURCES = {
    "transformers/image_transforms.py": "163556a472733191afe8cd86e08348d599bd88eb2370a3747ca687b1c14a6691",
    "transformers/models/dinov3_vit/configuration_dinov3_vit.py": (
        "1ac7cb889e2314e8cadf9b0bed43d42cfb05dce6d730f719be4380c8a10a8a46"
    ),
    "transformers/models/dinov3_vit/image_processing_dinov3_vit_fast.py": (
        "5a84a6a7283b77e0a0b000851150dd9d04dc54a11baa3d2cd3e2603d4669ec80"
    ),
    "transformers/models/dinov3_vit/modeling_dinov3_vit.py": (
        "00652d90ff8c7ae32b047ebe2acd2271f97e03f768cf570ac051045a4e0d5cf5"
    ),
}

_MAX_REQUEST_BYTES = 64 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_SECRET_ENV_FRAGMENTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
    "API_KEY",
    "PRIVATE_KEY",
)


class SemanticProbeError(RuntimeError):
    """Raised when the persistent semantic-probe contract is violated."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def _fail(classification: str, message: str) -> None:
    raise SemanticProbeError(classification, message)


class _SocketGuard:
    """Deny network operations while retaining a count of attempted calls."""

    def __init__(self) -> None:
        self.attempt_count = 0
        self._socket_class = socket.socket
        self._create_connection = socket.create_connection
        self._getaddrinfo = socket.getaddrinfo

    def install(self) -> None:
        guard = self
        original = self._socket_class

        class GuardedSocket(original):  # type: ignore[misc, valid-type]
            def connect(self, address: object) -> None:
                guard.attempt_count += 1
                raise RuntimeError("network access is disabled for semantic probe")

            def connect_ex(self, address: object) -> int:
                guard.attempt_count += 1
                raise RuntimeError("network access is disabled for semantic probe")

        def guarded_create_connection(*args: object, **kwargs: object) -> None:
            guard.attempt_count += 1
            raise RuntimeError("network access is disabled for semantic probe")

        def guarded_getaddrinfo(*args: object, **kwargs: object) -> None:
            guard.attempt_count += 1
            raise RuntimeError("network access is disabled for semantic probe")

        socket.socket = GuardedSocket
        socket.create_connection = guarded_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]

    def restore(self) -> None:
        socket.socket = self._socket_class
        socket.create_connection = self._create_connection
        socket.getaddrinfo = self._getaddrinfo


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("request_invalid", "request contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    _fail("request_invalid", "request contains a non-finite JSON number")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SemanticProbeError("identity_validation_failed", "unable to hash a required file") from exc
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_reparse_point(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return bool(flag and attributes & flag)
    except OSError as exc:
        raise SemanticProbeError("path_validation_failed", "unable to inspect a path") from exc


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolved_directory(path: Path, field: str) -> Path:
    if not path.is_absolute() or _is_reparse_point(path):
        _fail("unsafe_path", f"{field} must be an absolute non-link directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SemanticProbeError("path_validation_failed", f"{field} does not exist") from exc
    if not resolved.is_dir():
        _fail("unsafe_path", f"{field} is not a directory")
    return resolved


def _workspace_path(
    workspace: Path,
    value: str | Path,
    field: str,
    *,
    must_exist: bool,
) -> Path:
    candidate_value = Path(value)
    candidate = candidate_value if candidate_value.is_absolute() else workspace / candidate_value
    try:
        normalized = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise SemanticProbeError("path_validation_failed", f"{field} is unavailable") from exc
    if not _contains(workspace, normalized):
        _fail("unsafe_path", f"{field} escapes the task workspace")
    current = workspace
    for part in normalized.relative_to(workspace).parts:
        current = current / part
        if current.exists() and _is_reparse_point(current):
            _fail("unsafe_path", f"{field} crosses a link or reparse point")
    if must_exist and not normalized.is_file():
        _fail("unsafe_path", f"{field} must be an existing regular file")
    return normalized


def _read_request(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_REQUEST_BYTES:
            _fail("request_invalid", "request is absent or too large")
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except SemanticProbeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticProbeError("request_invalid", "unable to parse request") from exc
    if not isinstance(value, dict):
        _fail("request_invalid", "request must be an object")
    expected = {
        "schema_id",
        "run_id",
        "semantic_profile_sha256",
        "workspace_relative_artifact_directory",
    }
    if set(value) != expected:
        _fail("request_invalid", "request fields do not match the exact schema")
    if value["schema_id"] != REQUEST_SCHEMA_ID:
        _fail("request_invalid", "request schema ID does not match")
    run_id = value["run_id"]
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        _fail("request_invalid", "run ID is invalid")
    profile_sha = value["semantic_profile_sha256"]
    if not isinstance(profile_sha, str) or _SHA256_PATTERN.fullmatch(profile_sha) is None:
        _fail("request_invalid", "semantic profile SHA-256 is invalid")
    relative = value["workspace_relative_artifact_directory"]
    if not isinstance(relative, str) or not relative:
        _fail("request_invalid", "artifact directory is invalid")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
        or "\\" in relative
    ):
        _fail("request_invalid", "artifact directory must be a safe workspace-relative path")
    if relative != "actual-output":
        _fail("request_invalid", "artifact directory must be the approved actual-output path")
    return value


def _validate_offline_environment(workspace: Path) -> None:
    required = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, expected in required.items():
        if os.environ.get(key) != expected:
            _fail("offline_environment_invalid", f"required offline variable {key} is invalid")
    for key in os.environ:
        upper = key.upper()
        if key not in {"HF_HUB_DISABLE_IMPLICIT_TOKEN", "TOKENIZERS_PARALLELISM"} and any(
            fragment in upper for fragment in _SECRET_ENV_FRAGMENTS
        ):
            _fail("offline_environment_invalid", "a secret-like variable survived sanitization")
    for key in ("HF_HOME", "XDG_CACHE_HOME", "TEMP", "TMP"):
        value = os.environ.get(key)
        if not isinstance(value, str) or not value:
            _fail("offline_environment_invalid", f"scratch variable {key} is absent")
        candidate = Path(value)
        if not candidate.is_absolute():
            _fail("offline_environment_invalid", f"scratch variable {key} is not absolute")
        try:
            normalized = candidate.resolve(strict=False)
        except OSError as exc:
            raise SemanticProbeError(
                "offline_environment_invalid", "unable to resolve scratch storage"
            ) from exc
        if not _contains(workspace, normalized):
            _fail("offline_environment_invalid", "scratch storage escapes the task workspace")


def _assert_import_light() -> None:
    forbidden = ("numpy", "torch", "transformers", "PIL", "safetensors", "torchvision")
    loaded = [name for name in sys.modules if name.split(".", 1)[0] in forbidden]
    if loaded:
        _fail("startup_import_violation", "a heavy runtime module was imported before the socket guard")


def _expect_equal(actual: object, expected: object, classification: str, field: str) -> None:
    if actual != expected:
        _fail(classification, f"{field} does not match the semantic contract")


def _profile_mapping(profile: Any) -> dict[str, Any]:
    to_dict = getattr(profile, "to_dict", None)
    if not callable(to_dict):
        _fail("semantic_profile_invalid", "semantic profile has no serialization API")
    value = to_dict()
    if not isinstance(value, dict):
        _fail("semantic_profile_invalid", "semantic profile serialization is invalid")
    return value


def _nested(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            _fail("semantic_profile_invalid", f"semantic profile is missing {'.'.join(path)}")
        current = current[part]
    return current


def _validate_declared_file_identities(
    declared: object,
    *,
    root: Path,
    name_field: str,
    expected_hashes: Mapping[str, str],
    classification: str,
) -> dict[str, str]:
    if not isinstance(declared, (list, tuple)):
        _fail("semantic_profile_invalid", "declared file identities must be a sequence")
    observed: dict[str, str] = {}
    for item in declared:
        if not isinstance(item, Mapping):
            _fail("semantic_profile_invalid", "declared file identity must be an object")
        name = item.get(name_field)
        expected_sha = item.get("sha256")
        if not isinstance(name, str) or not isinstance(expected_sha, str):
            _fail("semantic_profile_invalid", "declared file identity is malformed")
        if name not in expected_hashes or expected_sha != expected_hashes[name]:
            _fail("semantic_profile_invalid", "declared file identity is not approved")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            _fail("semantic_profile_invalid", "declared file path is unsafe")
        candidate = root.joinpath(*relative.parts)
        if not candidate.is_file() or _is_reparse_point(candidate):
            _fail(classification, "a required identity file is unavailable")
        actual_sha = _sha256_file(candidate)
        if actual_sha != expected_sha:
            _fail(classification, "a required identity file SHA-256 does not match")
        size = item.get("bytes")
        if size is not None and (type(size) is not int or candidate.stat().st_size != size):
            _fail(classification, "a required identity file byte count does not match")
        observed[name] = actual_sha
    if observed != dict(expected_hashes):
        _fail("semantic_profile_invalid", "declared file identity set is incomplete")
    return observed


def _validate_profile_contract(profile_value: Mapping[str, Any]) -> None:
    model = _nested(profile_value, "model")
    geometry = _nested(profile_value, "geometry")
    canonical_input = _nested(profile_value, "canonical_input")
    normalized = _nested(profile_value, "canonical_input", "normalized")
    taps = _nested(profile_value, "taps")
    expected_model = {
        "key": MODEL_KEY,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_class": "DINOv3ViTModel",
        "config_class": "DINOv3ViTConfig",
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "expected_parameter_tensor_count": EXPECTED_PARAMETER_TENSOR_COUNT,
        "state_sha256": MODEL_STATE_SHA256,
    }
    for key, expected in expected_model.items():
        _expect_equal(model.get(key), expected, "semantic_profile_invalid", f"model.{key}")
    expected_geometry = {
        "input_height": 224,
        "input_width": 224,
        "patch_size": 16,
        "grid_height": 14,
        "grid_width": 14,
        "patch_count": EXPECTED_PATCH_COUNT,
        "hidden_size": EXPECTED_HIDDEN_DIMENSION,
        "layer_count": EXPECTED_LAYER_COUNT,
        "total_token_count": EXPECTED_TOTAL_TOKENS,
        "candidate_cls_indices": [0],
        "candidate_register_indices": [1, 2, 3, 4],
        "candidate_patch_start_index": 5,
        "candidate_patch_end_index": 200,
        "special_token_policy": SPECIAL_TOKEN_POLICY,
    }
    for key, expected in expected_geometry.items():
        _expect_equal(geometry.get(key), expected, "semantic_profile_invalid", f"geometry.{key}")
    expected_input = {
        "shape": [2, 224, 224, 3],
        "dtype": "uint8",
        "image_ids": list(IMAGE_IDS),
        "sha256": INPUT_SHA256,
    }
    for key, expected in expected_input.items():
        _expect_equal(
            canonical_input.get(key), expected, "semantic_profile_invalid", f"canonical_input.{key}"
        )
    expected_normalized = {
        "shape": [2, 3, 224, 224],
        "dtype": "float32",
        "sha256": NORMALIZED_SHA256,
        "c_contiguous": True,
        "finite": True,
    }
    for key, expected in expected_normalized.items():
        _expect_equal(
            normalized.get(key), expected, "semantic_profile_invalid", f"normalized.{key}"
        )
    expected_taps = ["embedding_output"] + [f"block_{index:02d}" for index in range(1, 13)] + [
        "final_norm"
    ]
    _expect_equal(taps.get("catalog"), expected_taps, "semantic_profile_invalid", "taps.catalog")
    _expect_equal(
        taps.get("artifact_selected"),
        list(SELECTED_ARTIFACT_TAPS),
        "semantic_profile_invalid",
        "taps.artifact_selected",
    )
    tolerances = _nested(profile_value, "tolerances")
    _expect_equal(tolerances.get("rtol"), 1e-6, "semantic_profile_invalid", "tolerances.rtol")
    _expect_equal(tolerances.get("atol"), 1e-6, "semantic_profile_invalid", "tolerances.atol")


def _startup_identity(
    *,
    semantic_profile_path: Path,
    runtime_profile_path: Path,
    snapshot_profile_path: Path,
    runtime_root: Path,
    snapshot_root: Path,
    repository_root: Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every static identity before importing the model runtime."""

    from vision_encoder_alignment.dinov3_semantic_boundary import load_semantic_profile
    from vision_encoder_alignment.runtime_boundary import validate_runtime_root, load_runtime_profile
    from vision_encoder_alignment.snapshot_boundary import (
        load_snapshot_profile,
        snapshot_files_directory,
        validate_snapshot_root,
    )

    semantic_profile = load_semantic_profile(semantic_profile_path)
    semantic_profile_sha256 = _sha256_file(semantic_profile_path)
    if request["semantic_profile_sha256"] != semantic_profile_sha256:
        _fail("semantic_profile_identity_mismatch", "request semantic profile SHA-256 does not match")
    profile_value = _profile_mapping(semantic_profile)
    _validate_profile_contract(profile_value)

    runtime_profile = load_runtime_profile(runtime_profile_path)
    snapshot_profile = load_snapshot_profile(snapshot_profile_path)
    runtime_manifest = validate_runtime_root(
        runtime_root, runtime_profile, repository_root=repository_root
    )
    snapshot_manifest = validate_snapshot_root(
        snapshot_root,
        snapshot_profile,
        repository_root=repository_root,
        persistent_runtime_root=runtime_root,
    )
    identities = _nested(profile_value, "identities")
    actual_identities = {
        "registry_sha256": _sha256_file(repository_root / "configs" / "models.yaml"),
        "runtime_profile_sha256": runtime_profile.source_sha256,
        "snapshot_profile_sha256": snapshot_profile.source_sha256,
        "persistent_runtime_manifest_sha256": runtime_manifest.source_sha256,
        "external_snapshot_manifest_sha256": snapshot_manifest.source_sha256,
        "persistent_freeze_sha256": runtime_manifest.candidate_freeze_sha256,
    }
    for key, actual in actual_identities.items():
        expected = identities.get(key)
        if not isinstance(expected, str) or expected != actual:
            _fail("identity_graph_mismatch", f"identity {key} does not match")

    snapshot_files = snapshot_files_directory(snapshot_root)
    checkpoint_hashes = _validate_declared_file_identities(
        _nested(profile_value, "checkpoint_files"),
        root=snapshot_files,
        name_field="filename",
        expected_hashes=EXPECTED_CHECKPOINT_FILES,
        classification="checkpoint_identity_mismatch",
    )
    site_packages = runtime_root / "venv" / "Lib" / "site-packages"
    installed_source_hashes = _validate_declared_file_identities(
        _nested(profile_value, "installed_sources"),
        root=site_packages,
        name_field="path",
        expected_hashes=EXPECTED_INSTALLED_SOURCES,
        classification="installed_source_identity_mismatch",
    )
    return {
        "semantic_profile": semantic_profile,
        "semantic_profile_value": profile_value,
        "semantic_profile_sha256": semantic_profile_sha256,
        "runtime_profile": runtime_profile,
        "runtime_manifest": runtime_manifest,
        "snapshot_profile": snapshot_profile,
        "snapshot_manifest": snapshot_manifest,
        "snapshot_files": snapshot_files,
        "identities": actual_identities,
        "checkpoint_hashes": checkpoint_hashes,
        "installed_source_hashes": installed_source_hashes,
    }


def generate_deterministic_images() -> Any:
    """Return the task's exact two-image uint8 batch without mutating RNG."""

    import numpy as np

    y, x = np.indices((224, 224), dtype=np.uint16)
    gradient = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    checker_base = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8)
    checker = np.stack(
        (
            checker_base,
            (3 * x + 5 * y) % 256,
            np.where(checker_base == 0, (7 * x) % 256, (11 * y) % 256),
        ),
        axis=-1,
    ).astype(np.uint8)
    images = np.stack((gradient, checker), axis=0)
    if images.shape != (2, 224, 224, 3) or images.dtype != np.dtype(np.uint8):
        _fail("deterministic_input_mismatch", "deterministic input geometry is invalid")
    if _sha256_bytes(images.tobytes(order="C")) != INPUT_SHA256:
        _fail("deterministic_input_mismatch", "deterministic input SHA-256 does not match")
    return images


def manual_preprocess(images: Any) -> Any:
    """Apply the exact torchvision-free FP32 ImageNet normalization formula."""

    import numpy as np

    if not isinstance(images, np.ndarray) or images.shape != (2, 224, 224, 3):
        _fail("manual_preprocessing_failed", "input batch shape is invalid")
    if images.dtype != np.dtype(np.uint8):
        _fail("manual_preprocessing_failed", "input batch dtype is invalid")
    normalized = images.astype(np.float32, copy=True)
    normalized *= np.float32(1.0 / 255.0)
    normalized -= np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    normalized /= np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = np.ascontiguousarray(normalized.transpose(0, 3, 1, 2), dtype=np.float32)
    if normalized.shape != (2, 3, 224, 224) or not normalized.flags.c_contiguous:
        _fail("manual_preprocessing_failed", "normalized tensor geometry is invalid")
    if normalized.dtype != np.dtype(np.float32) or not np.isfinite(normalized).all():
        _fail("manual_preprocessing_failed", "normalized tensor values are invalid")
    if _sha256_bytes(normalized.tobytes(order="C")) != NORMALIZED_SHA256:
        _fail("manual_preprocessing_failed", "normalized tensor SHA-256 does not match")
    return normalized


def _numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    import numpy as np

    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _model_state_sha256(model: Any) -> str:
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


def _array_sha256(array: Any) -> str:
    import numpy as np

    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        _fail("canonical_result_invalid", "feature hash input must be a C-contiguous array")
    return _sha256_bytes(array.tobytes(order="C"))


def _max_abs_difference(left: Any, right: Any) -> float:
    import torch

    if tuple(left.shape) != tuple(right.shape):
        _fail("tensor_shape_mismatch", "compared tensor shapes differ")
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left - right)).item())


def _mean_abs_difference(left: Any, right: Any) -> float:
    import torch

    if tuple(left.shape) != tuple(right.shape):
        _fail("tensor_shape_mismatch", "compared tensor shapes differ")
    if left.numel() == 0:
        return 0.0
    return float(torch.mean(torch.abs(left - right)).item())


def _loading_count(loading_info: Mapping[str, Any], key: str) -> int:
    value = loading_info.get(key, ())
    if not isinstance(value, (list, tuple)):
        _fail("loading_info_invalid", f"loading information {key} is not a sequence")
    return len(value)


def _load_pretrained_model(snapshot_files: Path) -> tuple[Any, dict[str, int]]:
    import torch
    from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

    with torch.random.fork_rng(devices=[], enabled=True):
        config = DINOv3ViTConfig.from_pretrained(
            str(snapshot_files),
            local_files_only=True,
            trust_remote_code=False,
            token=False,
        )
        loaded = DINOv3ViTModel.from_pretrained(
            str(snapshot_files),
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            token=False,
            use_safetensors=True,
            dtype=torch.float32,
            output_loading_info=True,
        )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        _fail("model_load_failed", "model loader did not return loading information")
    model, loading_info = loaded
    if not isinstance(loading_info, Mapping):
        _fail("loading_info_invalid", "model loading information is not an object")
    loading_counts = {
        "error_message_count": _loading_count(loading_info, "error_msgs"),
        "mismatched_key_count": _loading_count(loading_info, "mismatched_keys"),
        "missing_key_count": _loading_count(loading_info, "missing_keys"),
        "unexpected_key_count": _loading_count(loading_info, "unexpected_keys"),
    }
    if any(loading_counts.values()):
        _fail("loading_info_invalid", "checkpoint loading information is not clean")
    if type(config).__name__ != "DINOv3ViTConfig" or type(model).__name__ != "DINOv3ViTModel":
        _fail("model_identity_mismatch", "loaded model/config classes do not match")
    model.eval()
    model.requires_grad_(False)
    return model, loading_counts


def _audit_model(model: Any, loading_counts: Mapping[str, int]) -> dict[str, Any]:
    import torch

    embeddings = getattr(model, "embeddings", None)
    patch_embeddings = getattr(embeddings, "patch_embeddings", None)
    cls_token = getattr(embeddings, "cls_token", None)
    register_tokens = getattr(embeddings, "register_tokens", None)
    layers = getattr(model, "layer", None)
    final_norm = getattr(model, "norm", None)
    if any(item is None for item in (embeddings, patch_embeddings, cls_token, register_tokens, layers, final_norm)):
        _fail("component_path_unresolved", "a required DINOv3 component is unavailable")
    parameters = tuple(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    parameter_tensor_count = len(parameters)
    trainable_count = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    devices = {parameter.device.type for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    config = model.config
    checks = {
        "parameter_count": (parameter_count, EXPECTED_PARAMETER_COUNT),
        "parameter_tensor_count": (parameter_tensor_count, EXPECTED_PARAMETER_TENSOR_COUNT),
        "layer_count": (len(layers), EXPECTED_LAYER_COUNT),
        "hidden_size": (getattr(config, "hidden_size", None), EXPECTED_HIDDEN_DIMENSION),
        "patch_size": (getattr(config, "patch_size", None), 16),
        "image_size": (getattr(config, "image_size", None), 224),
        "register_count": (getattr(config, "num_register_tokens", None), 4),
        "cls_shape": (tuple(cls_token.shape), (1, 1, 768)),
        "register_shape": (tuple(register_tokens.shape), (1, 4, 768)),
        "patch_weight_shape": (tuple(patch_embeddings.weight.shape), (768, 3, 16, 16)),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            _fail("model_identity_mismatch", f"model {field} does not match")
    if devices != {"cpu"} or dtypes != {torch.float32}:
        _fail("model_runtime_mismatch", "model is not uniformly CPU FP32")
    if model.training or trainable_count != 0:
        _fail("model_runtime_mismatch", "model is not frozen in evaluation mode")
    forbidden_head_modules = [
        name
        for name, _ in model.named_modules()
        if name and (name == "pooler" or name.endswith(".pooler") or name == "head" or name.endswith(".head"))
    ]
    if forbidden_head_modules:
        _fail("pooler_semantics_unresolved", "an unexpected pooler/head module is present")
    return {
        "model_class": type(model).__name__,
        "config_class": type(config).__name__,
        "parameter_count": parameter_count,
        "parameter_tensor_count": parameter_tensor_count,
        "trainable_parameter_count": trainable_count,
        "device": "cpu",
        "dtype": "float32",
        "training": bool(model.training),
        "evaluation_mode": not bool(model.training),
        "frozen": trainable_count == 0,
        "loading_info": dict(loading_counts),
        "component_paths": [
            "embeddings",
            "embeddings.patch_embeddings",
            "embeddings.cls_token",
            "embeddings.register_tokens",
            "layer",
            "norm",
        ],
        "encoder_block_count": len(layers),
        "cls_parameter_shape": list(cls_token.shape),
        "register_parameter_shape": list(register_tokens.shape),
        "patch_projection_weight_shape": list(patch_embeddings.weight.shape),
        "separate_pooler_or_head_module": False,
    }


def _capture_forward(model: Any, pixel_values: Any) -> tuple[Any, dict[str, Any]]:
    """Capture embedding, projection, and every raw block output for one forward."""

    import torch

    captures: dict[str, list[Any]] = {"embedding": [], "patch_projection": []}
    captures.update({f"block_{index:02d}": [] for index in range(1, 13)})
    handles: list[Any] = []

    def capture(name: str):
        def hook(module: Any, arguments: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                _fail("hook_output_invalid", f"hook {name} did not return a tensor")
            captures[name].append(output.detach().cpu().clone())

        return hook

    try:
        handles.append(model.embeddings.register_forward_hook(capture("embedding")))
        handles.append(
            model.embeddings.patch_embeddings.register_forward_hook(capture("patch_projection"))
        )
        for index, layer in enumerate(model.layer, start=1):
            handles.append(layer.register_forward_hook(capture(f"block_{index:02d}")))
        with torch.inference_mode():
            outputs = model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    flattened: dict[str, Any] = {}
    for name, values in captures.items():
        if len(values) != 1:
            _fail("hook_capture_invalid", f"hook {name} did not execute exactly once")
        flattened[name] = values[0]
    return outputs, flattened


def _validate_forward_semantics(model: Any, outputs: Any, capture: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    hidden_states = getattr(outputs, "hidden_states", None)
    last_hidden = getattr(outputs, "last_hidden_state", None)
    pooler = getattr(outputs, "pooler_output", None)
    if not isinstance(hidden_states, tuple) or len(hidden_states) != EXPECTED_HIDDEN_STATE_COUNT:
        _fail("hidden_state_mapping_mismatch", "hidden-state tuple length does not match")
    expected_sequence_shape = (2, EXPECTED_TOTAL_TOKENS, EXPECTED_HIDDEN_DIMENSION)
    if any(tuple(value.shape) != expected_sequence_shape for value in hidden_states):
        _fail("hidden_state_mapping_mismatch", "hidden-state tuple shape does not match")
    if tuple(last_hidden.shape) != expected_sequence_shape:
        _fail("final_norm_mismatch", "last hidden state shape does not match")
    if pooler is None or tuple(pooler.shape) != (2, EXPECTED_HIDDEN_DIMENSION):
        _fail("pooler_semantics_unresolved", "pooler output shape does not match")
    embedding = capture["embedding"]
    patch_projection = capture["patch_projection"]
    if tuple(embedding.shape) != expected_sequence_shape:
        _fail("embedding_semantics_mismatch", "embedding sequence shape does not match")
    if tuple(patch_projection.shape) != (2, 768, 14, 14):
        _fail("patch_order_unresolved", "patch projection shape does not match")

    reconstructed_patches = patch_projection.flatten(2).transpose(1, 2).contiguous()
    cls_expanded = model.embeddings.cls_token.detach().cpu().expand(2, -1, -1)
    register_expanded = model.embeddings.register_tokens.detach().cpu().expand(2, -1, -1)
    reconstructed_embedding = torch.cat(
        (cls_expanded, register_expanded, reconstructed_patches), dim=1
    )
    prefix_exact = torch.equal(embedding, reconstructed_embedding)
    cls_exact = torch.equal(embedding[:, 0:1, :], cls_expanded)
    register_exact = torch.equal(embedding[:, 1:5, :], register_expanded)
    patch_exact = torch.equal(embedding[:, 5:, :], reconstructed_patches)
    if not (prefix_exact and cls_exact and register_exact and patch_exact):
        _fail("prefix_token_mapping_mismatch", "runtime prefix/patch concatenation does not match")
    patch_max_diff = _max_abs_difference(embedding[:, 5:, :], reconstructed_patches)
    if not torch.allclose(
        embedding[:, 5:, :], reconstructed_patches, rtol=1e-6, atol=1e-6
    ):
        _fail("dinov3_patch_order_unresolved", "row-major patch reconstruction does not match")

    embedding_diff = _max_abs_difference(hidden_states[0], embedding)
    if not torch.equal(hidden_states[0], embedding):
        _fail("hidden_state_mapping_mismatch", "hidden_states[0] is not embedding output")
    block_entries: list[dict[str, Any]] = []
    for index in range(1, 13):
        tap_name = f"block_{index:02d}"
        hook_value = capture[tap_name]
        library_value = hidden_states[index]
        matches = torch.equal(hook_value, library_value)
        max_abs_diff = _max_abs_difference(hook_value, library_value)
        if index <= 11 and not matches:
            _fail("hidden_state_mapping_mismatch", f"{tap_name} does not match hidden state {index}")
        if index == 12 and matches:
            _fail("final_norm_mismatch", "hidden_states[12] unexpectedly equals raw block_12")
        block_entries.append(
            {
                "tap_name": tap_name,
                "block_index": index,
                "hidden_state_index": index,
                "hook_shape": list(hook_value.shape),
                "hidden_state_shape": list(library_value.shape),
                "hidden_state_matches_hook": matches,
                "max_abs_diff": max_abs_diff,
            }
        )

    raw_final = capture["block_12"]
    with torch.inference_mode():
        explicit_final = model.norm(raw_final)
    explicit_last_diff = _max_abs_difference(explicit_final, last_hidden)
    hidden_last_diff = _max_abs_difference(hidden_states[12], last_hidden)
    if not torch.equal(explicit_final, last_hidden):
        _fail("final_norm_mismatch", "explicit final norm does not equal last_hidden_state")
    if not torch.equal(hidden_states[12], last_hidden):
        _fail("final_norm_mismatch", "hidden_states[12] is not the normalized final output")
    if not torch.equal(pooler, last_hidden[:, 0, :]):
        _fail("pooler_semantics_unresolved", "pooler output is not final CLS")

    tensors_to_check = list(hidden_states) + [
        last_hidden,
        pooler,
        embedding,
        patch_projection,
        *[capture[f"block_{index:02d}"] for index in range(1, 13)],
    ]
    if any(value.dtype != torch.float32 or value.device.type != "cpu" for value in tensors_to_check):
        _fail("model_runtime_mismatch", "captured output is not CPU FP32")
    if not all(bool(torch.isfinite(value).all().item()) for value in tensors_to_check):
        _fail("nonfinite_model_output", "captured output contains NaN or Inf")

    full_sequences = {"embedding_output": embedding}
    full_sequences.update({f"block_{index:02d}": capture[f"block_{index:02d}"] for index in range(1, 13)})
    full_sequences["final_norm"] = explicit_final
    patch_mean = explicit_final[:, 5:, :].mean(dim=1)
    return {
        "full_sequences": full_sequences,
        "embedding": {
            "patch_embedding_shape": list(reconstructed_patches.shape),
            "patch_embedding_dtype": "float32",
            "patch_embedding_hook_matches": patch_exact,
            "patch_embedding_hook_max_abs_diff": patch_max_diff,
            "sequence_shape": list(embedding.shape),
            "sequence_dtype": "float32",
            "hidden_state_0_shape": list(hidden_states[0].shape),
            "hidden_state_0_matches": True,
            "hidden_state_0_max_abs_diff": embedding_diff,
        },
        "prefix_tokens": {
            "cls_indices": [0],
            "register_indices": [1, 2, 3, 4],
            "special_indices": [0, 1, 2, 3, 4],
            "patch_indices": list(range(5, 201)),
            "sequence_order": "cls_register_patch",
            "overlap_count": 0,
            "partition_complete": True,
        },
        "patch_order": {
            "source_relation": "conv2d_output_flatten_spatial_then_transpose",
            "conv2d_hook_shape": list(patch_projection.shape),
            "reconstructed_shape": list(reconstructed_patches.shape),
            "embedding_patch_slice_shape": list(embedding[:, 5:, :].shape),
            "row_major": True,
            "matches": patch_exact,
            "max_abs_diff": patch_max_diff,
            "rtol": 1e-6,
            "atol": 1e-6,
        },
        "hidden_states": {
            "count": len(hidden_states),
            "shapes": [list(value.shape) for value in hidden_states],
            "embedding_hidden_state_index": 0,
            "embedding_matches": True,
            "embedding_max_abs_diff": embedding_diff,
            "blocks": block_entries,
        },
        "final_norm": {
            "raw_final_shape": list(raw_final.shape),
            "explicit_norm_shape": list(explicit_final.shape),
            "last_hidden_shape": list(last_hidden.shape),
            "explicit_norm_matches_last_hidden": True,
            "explicit_norm_max_abs_diff": explicit_last_diff,
            "hidden_state_12_matches_last_hidden": True,
            "hidden_state_12_max_abs_diff": hidden_last_diff,
            "raw_final_differs_from_last_hidden": not torch.equal(raw_final, last_hidden),
            "raw_final_vs_last_hidden_max_abs_diff": _max_abs_difference(raw_final, last_hidden),
            "canonical_block_12_source": "raw_final_block_hook",
            "canonical_final_norm_source": "explicit_model_norm",
        },
        "pooler": {
            "pooler_shape": list(pooler.shape),
            "patch_mean_shape": list(patch_mean.shape),
            "source_relation": "last_hidden_state_cls",
            "separate_head_used": False,
            "cls_matches": True,
            "cls_max_abs_diff": _max_abs_difference(pooler, last_hidden[:, 0, :]),
            "canonical_global_source": "patch_axis_mean",
            "excluded_from_canonical_global": True,
            "max_abs_difference_from_patch_mean": _max_abs_difference(pooler, patch_mean),
            "mean_abs_difference_from_patch_mean": _mean_abs_difference(pooler, patch_mean),
        },
        "repeat_raw": {
            "embedding": embedding,
            "patch_projection": patch_projection,
            "blocks": tuple(capture[f"block_{index:02d}"] for index in range(1, 13)),
            "hidden_states": hidden_states,
            "last_hidden_state": last_hidden.detach().cpu().clone(),
            "pooler_output": pooler.detach().cpu().clone(),
        },
    }


def _build_canonical_result(full_sequences: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np

    from vision_encoder_alignment.extractors.base import (
        CanonicalTapFeatures,
        ExtractionResult,
        ExtractorDescriptor,
        NumericNormalizationSpec,
        TokenLayout,
        build_vit_tap_specs,
        select_patch_tokens,
    )

    taps = build_vit_tap_specs(12)
    expected_names = ("embedding_output",) + tuple(
        f"block_{index:02d}" for index in range(1, 13)
    ) + ("final_norm",)
    if tuple(tap.name for tap in taps) != expected_names:
        _fail("canonical_result_invalid", "canonical tap catalog does not match")
    token_layout = TokenLayout(
        grid_height=14,
        grid_width=14,
        total_token_count=201,
        patch_token_indices=tuple(range(5, 201)),
        special_token_indices=(0, 1, 2, 3, 4),
        special_token_policy=SPECIAL_TOKEN_POLICY,
    )
    normalization = NumericNormalizationSpec(
        scale=float(np.float32(1.0 / 255.0)),
        mean=(0.485, 0.456, 0.406),
        standard_deviation=(0.229, 0.224, 0.225),
        channel_order="RGB",
    )
    descriptor = ExtractorDescriptor(
        model_key=MODEL_KEY,
        model_id=MODEL_ID,
        revision=REVISION,
        family="dinov3",
        implementation_class="DINOv3PretrainedSemanticProbe",
        vision_component_path="self",
        input_height=224,
        input_width=224,
        grid_height=14,
        grid_width=14,
        hidden_dimension=768,
        layer_count=12,
        parameter_dtype="float32",
        compute_dtype="float32",
        output_dtype="float32",
        evaluation_mode=True,
        gradients_enabled=False,
        normalization=normalization,
        token_layout=token_layout,
        taps=taps,
    )
    features: list[Any] = []
    evidence: list[dict[str, Any]] = []
    for tap in taps:
        sequence_tensor = full_sequences.get(tap.name)
        if sequence_tensor is None:
            _fail("canonical_result_invalid", f"full sequence for {tap.name} is unavailable")
        full = np.array(
            sequence_tensor.detach().cpu().contiguous().numpy(),
            dtype=np.float32,
            copy=True,
            order="C",
        )
        if full.shape != (2, 201, 768) or not full.flags.c_contiguous:
            _fail("canonical_result_invalid", f"full sequence for {tap.name} has wrong shape")
        patch = select_patch_tokens(full, token_layout)
        feature = CanonicalTapFeatures.from_patch(tap, patch)
        if feature.patch.shape != (2, 196, 768) or feature.global_mean.shape != (2, 768):
            _fail("canonical_result_invalid", f"canonical feature shape for {tap.name} is wrong")
        if feature.patch.dtype != np.dtype(np.float32) or feature.global_mean.dtype != np.dtype(np.float32):
            _fail("canonical_result_invalid", f"canonical feature dtype for {tap.name} is wrong")
        if feature.patch.flags.writeable or feature.global_mean.flags.writeable:
            _fail("canonical_result_invalid", f"canonical feature for {tap.name} is mutable")
        expected_global = feature.patch.mean(axis=1, dtype=np.float32)
        if not np.array_equal(feature.global_mean, expected_global):
            _fail("canonical_result_invalid", f"Global Mean for {tap.name} is not exact patch mean")
        features.append(feature)
        evidence.append(
            {
                "name": tap.name,
                "full_sequence_shape": list(full.shape),
                "patch_shape": list(feature.patch.shape),
                "global_shape": list(feature.global_mean.shape),
                "dtype": "float32",
                "finite": bool(
                    np.isfinite(feature.patch).all() and np.isfinite(feature.global_mean).all()
                ),
                "patch_sha256": _array_sha256(np.ascontiguousarray(feature.patch)),
                "global_sha256": _array_sha256(np.ascontiguousarray(feature.global_mean)),
                "global_matches_patch_mean": True,
            }
        )
    result = ExtractionResult(
        descriptor=descriptor,
        image_ids=IMAGE_IDS,
        tap_features=tuple(features),
    )
    if result.image_ids != IMAGE_IDS or result.tap_names != expected_names:
        _fail("canonical_result_invalid", "canonical result ordering is wrong")
    for tap_name in expected_names:
        if result.get_tap(tap_name).tap.name != tap_name:
            _fail("canonical_result_invalid", "canonical tap lookup failed")
    try:
        result.get_tap("unknown_tap")
    except Exception:
        unknown_lookup_rejected = True
    else:
        unknown_lookup_rejected = False
    if not unknown_lookup_rejected:
        _fail("canonical_result_invalid", "unknown canonical tap lookup succeeded")
    return result, evidence


def _compare_repeat(
    first_raw: Mapping[str, Any],
    second_raw: Mapping[str, Any],
    first_evidence: Sequence[Mapping[str, Any]],
    second_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import torch

    embedding_exact = torch.equal(first_raw["embedding"], second_raw["embedding"])
    patch_projection_exact = torch.equal(
        first_raw["patch_projection"], second_raw["patch_projection"]
    )
    blocks_exact = [
        torch.equal(left, right)
        for left, right in zip(first_raw["blocks"], second_raw["blocks"], strict=True)
    ]
    hidden_exact = [
        torch.equal(left, right)
        for left, right in zip(
            first_raw["hidden_states"], second_raw["hidden_states"], strict=True
        )
    ]
    last_hidden_exact = torch.equal(
        first_raw["last_hidden_state"], second_raw["last_hidden_state"]
    )
    pooler_exact = torch.equal(first_raw["pooler_output"], second_raw["pooler_output"])
    first_hashes = [
        (item["patch_sha256"], item["global_sha256"]) for item in first_evidence
    ]
    second_hashes = [
        (item["patch_sha256"], item["global_sha256"]) for item in second_evidence
    ]
    hashes_exact = first_hashes == second_hashes
    exact = all(
        (
            embedding_exact,
            patch_projection_exact,
            all(blocks_exact),
            all(hidden_exact),
            last_hidden_exact,
            pooler_exact,
            hashes_exact,
        )
    )
    if not exact:
        _fail("repeat_nondeterminism", "same-process repeat output is not exact")
    return {
        "embedding_exact": embedding_exact,
        "hidden_states_exact": all(hidden_exact) and all(blocks_exact) and patch_projection_exact,
        "last_hidden_exact": last_hidden_exact,
        "pooler_exact": pooler_exact,
        "patch_hashes_exact": hashes_exact,
        "global_hashes_exact": hashes_exact,
    }


def _write_actual_artifacts(
    *,
    artifact_directory: Path,
    artifact_relative_directory: str,
    run_id: str,
    result: Any,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    import numpy as np

    from vision_encoder_alignment.feature_artifacts import (
        ARTIFACT_SCHEMA_ID,
        FeatureArtifactMetadata,
        canonical_tensor_keys,
        read_feature_artifact,
        write_feature_artifact,
    )

    if artifact_directory.exists():
        _fail("artifact_path_collision", "actual-output artifact directory must be fresh")
    entries: list[dict[str, Any]] = []
    for shard_index, tap_name in enumerate(SELECTED_ARTIFACT_TAPS):
        feature = result.get_tap(tap_name)
        patch_key, global_key = canonical_tensor_keys(tap_name)
        tensor_name = f"{tap_name}.safetensors"
        sidecar_name = f"{tap_name}.json"
        metadata = FeatureArtifactMetadata(
            schema_id=ARTIFACT_SCHEMA_ID,
            run_id=run_id,
            model_key=MODEL_KEY,
            tap_namespace="backbone",
            tap_name=tap_name,
            patch_tensor_key=patch_key,
            global_mean_tensor_key=global_key,
            image_ids=list(IMAGE_IDS),
            grid_height=14,
            grid_width=14,
            patch_count=196,
            hidden_dimension=768,
            dtype="float32",
            shard_index=shard_index,
            safetensors_file_name=tensor_name,
            sidecar_file_name=sidecar_name,
            checksum_algorithm="sha256",
            checksum_value="",
            created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            run_status="incomplete",
        )
        completed = write_feature_artifact(
            artifact_directory, metadata, feature.patch, feature.global_mean
        )
        sidecar_path = artifact_directory / sidecar_name
        tensor_path = artifact_directory / tensor_name
        loaded = read_feature_artifact(sidecar_path)
        patch_exact = np.array_equal(loaded.patch, feature.patch)
        global_exact = np.array_equal(loaded.global_mean, feature.global_mean)
        if not patch_exact or not global_exact:
            _fail("artifact_roundtrip_mismatch", "actual-output artifact values changed")
        if completed.checksum_value != _sha256_file(tensor_path):
            _fail("artifact_roundtrip_mismatch", "actual-output artifact checksum changed")
        prefix = PurePosixPath(artifact_relative_directory)
        entries.append(
            {
                "tap_name": tap_name,
                "sidecar_relative_path": str(prefix / sidecar_name),
                "safetensors_relative_path": str(prefix / tensor_name),
                "checksum_sha256": completed.checksum_value,
                "patch_sha256": _array_sha256(np.ascontiguousarray(loaded.patch)),
                "global_sha256": _array_sha256(np.ascontiguousarray(loaded.global_mean)),
            }
        )
    return {
        "workspace_relative_directory": artifact_relative_directory,
        "selected_taps": list(SELECTED_ARTIFACT_TAPS),
        "roundtrip_validated": True,
        "entries": entries,
    }


def _build_response(
    *,
    identity: Mapping[str, Any],
    request: Mapping[str, Any],
    model_facts: Mapping[str, Any],
    state_before: str,
    state_after: str,
    input_facts: Mapping[str, Any],
    semantic_facts: Mapping[str, Any],
    tap_evidence: Sequence[Mapping[str, Any]],
    repeat: Mapping[str, Any],
    artifact: Mapping[str, Any],
    invariants: Mapping[str, Any],
    socket_attempts: int,
) -> dict[str, Any]:
    static_identities = identity["identities"]
    installed_sources = _nested(identity["semantic_profile_value"], "installed_sources")
    source_hashes = [
        {
            "path": item["path"],
            "sha256": identity["installed_source_hashes"][item["path"]],
        }
        for item in installed_sources
    ]
    loading = model_facts["loading_info"]
    response_model = {
        "compute_dtype": "float32",
        "config_class": model_facts["config_class"],
        "device": "cpu",
        "evaluation_mode": model_facts["evaluation_mode"],
        "gradients_enabled": False,
        "loading_error_count": loading["error_message_count"],
        "mismatched_key_count": loading["mismatched_key_count"],
        "missing_key_count": loading["missing_key_count"],
        "model_class": model_facts["model_class"],
        "output_dtype": "float32",
        "parameter_count": model_facts["parameter_count"],
        "parameter_dtype": "float32",
        "parameter_tensor_count": model_facts["parameter_tensor_count"],
        "parameters_frozen": model_facts["frozen"],
        "state_sha256": state_after,
        "unexpected_key_count": loading["unexpected_key_count"],
    }
    components = {
        "model": "self",
        "embeddings": "embeddings",
        "patch_embeddings": "embeddings.patch_embeddings",
        "cls_token": "embeddings.cls_token",
        "register_tokens": "embeddings.register_tokens",
        "encoder_layers": "layer",
        "norm": "norm",
        "pooler": "pooler_output",
    }
    token_layout = {
        "grid_height": 14,
        "grid_width": 14,
        "total_token_count": 201,
        "patch_token_indices": list(range(5, 201)),
        "special_token_indices": [0, 1, 2, 3, 4],
        "special_token_policy": SPECIAL_TOKEN_POLICY,
        "partition_complete": True,
    }
    workspace_paths = [
        relative
        for entry in artifact["entries"]
        for relative in (
            entry["sidecar_relative_path"],
            entry["safetensors_relative_path"],
        )
    ]
    return {
        "schema_id": RESPONSE_SCHEMA_ID,
        "run_id": request["run_id"],
        "status": "success",
        "semantic_marker": SEMANTIC_MARKER,
        "identities": {
            "semantic_profile_sha256": identity["semantic_profile_sha256"],
            **dict(static_identities),
            "source_hashes": source_hashes,
        },
        "model": response_model,
        "input": dict(input_facts),
        "components": components,
        "embedding": semantic_facts["embedding"],
        "prefix_tokens": semantic_facts["prefix_tokens"],
        "patch_order": semantic_facts["patch_order"],
        "hidden_states": semantic_facts["hidden_states"],
        "final_norm": semantic_facts["final_norm"],
        "pooler": semantic_facts["pooler"],
        "token_layout": token_layout,
        "taps": [dict(item) for item in tap_evidence],
        "artifact": dict(artifact),
        "repeat": dict(repeat),
        "invariants": dict(invariants),
        "socket_attempts": socket_attempts,
        "forward_count": 2,
        "markers": list(PROBE_MARKERS),
        "workspace_paths": workspace_paths,
    }


def _write_response(path: Path, response: Mapping[str, Any]) -> None:
    if path.exists():
        _fail("response_path_collision", "response path must be fresh")
    try:
        serialized = json.dumps(
            response,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TypeError, ValueError) as exc:
        raise SemanticProbeError("response_write_failed", "unable to write probe response") from exc


def run_probe(
    *,
    semantic_profile_path: Path,
    runtime_profile_path: Path,
    snapshot_profile_path: Path,
    runtime_root: Path,
    snapshot_root: Path,
    workspace: Path,
    request_path: Path,
    response_path: Path,
) -> dict[str, Any]:
    """Execute one bounded two-forward pretrained semantic probe."""

    workspace = _resolved_directory(workspace, "workspace_root")
    repository_root = Path(__file__).resolve().parents[1]
    repository_root = _resolved_directory(repository_root, "repository_root")
    request_path = _workspace_path(workspace, request_path, "request", must_exist=True)
    response_path = _workspace_path(workspace, response_path, "response", must_exist=False)
    if response_path.exists():
        _fail("response_path_collision", "response path must be fresh")
    _validate_offline_environment(workspace)
    _assert_import_light()
    guard = _SocketGuard()
    guard.install()
    try:
        request = _read_request(request_path)
        from vision_encoder_alignment.dinov3_semantic_boundary import read_semantic_request

        boundary_request = read_semantic_request(request_path)
        if boundary_request.to_dict() != request:
            _fail("request_invalid", "boundary request does not match parsed request")
        artifact_relative = request["workspace_relative_artifact_directory"]
        artifact_directory = _workspace_path(
            workspace, artifact_relative, "artifact_directory", must_exist=False
        )
        if artifact_directory.exists():
            _fail("artifact_path_collision", "artifact directory must be fresh")
        identity = _startup_identity(
            semantic_profile_path=semantic_profile_path,
            runtime_profile_path=runtime_profile_path,
            snapshot_profile_path=snapshot_profile_path,
            runtime_root=runtime_root,
            snapshot_root=snapshot_root,
            repository_root=repository_root,
            request=request,
        )

        import numpy as np
        import torch

        if any(
            name == "torchvision" or name.startswith("torchvision.") for name in sys.modules
        ):
            _fail("torchvision_imported", "torchvision was imported into the probe process")
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()
        torch_rng_before = torch.random.get_rng_state().clone()

        images = generate_deterministic_images()
        input_before = _sha256_bytes(images.tobytes(order="C"))
        normalized = manual_preprocess(images)
        input_after = _sha256_bytes(images.tobytes(order="C"))
        if input_before != input_after:
            _fail("input_mutated", "manual preprocessing mutated the canonical input")
        pixel_values = torch.from_numpy(normalized)

        model, loading_counts = _load_pretrained_model(identity["snapshot_files"])
        model_facts = _audit_model(model, loading_counts)
        state_before = _model_state_sha256(model)
        if state_before != MODEL_STATE_SHA256:
            _fail("model_state_mismatch", "loaded model state SHA-256 does not match")

        first_outputs, first_capture = _capture_forward(model, pixel_values)
        first_semantics = _validate_forward_semantics(model, first_outputs, first_capture)
        first_result, first_taps = _build_canonical_result(first_semantics["full_sequences"])

        second_outputs, second_capture = _capture_forward(model, pixel_values)
        second_semantics = _validate_forward_semantics(model, second_outputs, second_capture)
        _, second_taps = _build_canonical_result(second_semantics["full_sequences"])
        repeat = _compare_repeat(
            first_semantics["repeat_raw"],
            second_semantics["repeat_raw"],
            first_taps,
            second_taps,
        )

        state_after = _model_state_sha256(model)
        if state_after != state_before or state_after != MODEL_STATE_SHA256:
            _fail("model_state_mutated", "model state changed during semantic inference")
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            _fail("model_state_mutated", "model training/frozen state changed during inference")

        artifact = _write_actual_artifacts(
            artifact_directory=artifact_directory,
            artifact_relative_directory=artifact_relative,
            run_id=request["run_id"],
            result=first_result,
        )
        python_rng_after = random.getstate()
        numpy_rng_after = np.random.get_state()
        torch_rng_after = torch.random.get_rng_state().clone()
        rng = {
            "python_unchanged": python_rng_before == python_rng_after,
            "numpy_unchanged": _numpy_rng_equal(numpy_rng_before, numpy_rng_after),
            "torch_cpu_unchanged": bool(torch.equal(torch_rng_before, torch_rng_after)),
        }
        if not all(rng.values()):
            _fail("rng_state_mutated", "probe changed caller RNG state")
        if guard.attempt_count != 0:
            _fail("network_attempted", "probe attempted a network operation")
        torchvision_import_count = sum(
            1
            for name in sys.modules
            if name == "torchvision" or name.startswith("torchvision.")
        )
        if torchvision_import_count != 0:
            _fail("torchvision_imported", "torchvision was imported into the probe process")

        input_facts = {
            "image_ids": list(IMAGE_IDS),
            "input_shape": list(images.shape),
            "input_dtype": str(images.dtype),
            "input_sha256": input_before,
            "input_unchanged": input_before == input_after,
            "normalized_shape": list(normalized.shape),
            "normalized_dtype": str(normalized.dtype),
            "normalized_sha256": _sha256_bytes(normalized.tobytes(order="C")),
            "normalized_c_contiguous": bool(normalized.flags.c_contiguous),
            "normalized_finite": bool(np.isfinite(normalized).all()),
            "resize_count": 0,
            "crop_count": 0,
            "pil_conversion_count": 0,
            "image_processor_construction_count": 0,
            "torchvision_import_count": torchvision_import_count,
        }
        invariants = {
            "model_state_unchanged": state_before == state_after,
            "numpy_rng_unchanged": rng["numpy_unchanged"],
            "python_rng_unchanged": rng["python_unchanged"],
            "torch_rng_unchanged": rng["torch_cpu_unchanged"],
            "training_false": not bool(model.training),
        }
        response = _build_response(
            identity=identity,
            request=request,
            model_facts=model_facts,
            state_before=state_before,
            state_after=state_after,
            input_facts=input_facts,
            semantic_facts=first_semantics,
            tap_evidence=first_taps,
            repeat=repeat,
            artifact=artifact,
            invariants=invariants,
            socket_attempts=guard.attempt_count,
        )
        from vision_encoder_alignment.dinov3_semantic_boundary import validate_semantic_response

        validate_semantic_response(response, identity["semantic_profile"])
        _write_response(response_path, response)
        return response
    finally:
        guard.restore()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-profile", required=True, type=Path)
    parser.add_argument("--runtime-profile", required=True, type=Path)
    parser.add_argument("--snapshot-profile", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        response = run_probe(
            semantic_profile_path=args.semantic_profile.resolve(strict=True),
            runtime_profile_path=args.runtime_profile.resolve(strict=True),
            snapshot_profile_path=args.snapshot_profile.resolve(strict=True),
            runtime_root=args.runtime_root.resolve(strict=True),
            snapshot_root=args.snapshot_root.resolve(strict=True),
            workspace=args.workspace_root.resolve(strict=True),
            request_path=args.request,
            response_path=args.response,
        )
    except Exception as exc:
        classification = getattr(exc, "classification", type(exc).__name__)
        if not isinstance(classification, str) or _RUN_ID_PATTERN.fullmatch(classification) is None:
            classification = type(exc).__name__
        print(
            f"DINOV3_PRETRAINED_SEMANTIC_PROBE_FAILED:{classification}",
            file=sys.stderr,
        )
        return 1
    if response.get("markers") != list(PROBE_MARKERS):
        print("DINOV3_PRETRAINED_SEMANTIC_PROBE_FAILED:marker_mismatch", file=sys.stderr)
        return 1
    for marker in PROBE_MARKERS:
        print(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
