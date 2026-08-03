"""Import-light process boundary for real DINOv3 feature extraction.

Only finite JSON control data crosses this module.  Pixel arrays and feature
tensors are transported as task-owned temporary files and are validated by the
caller and worker independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

from .dinov3_production_identity import (
    APPLE_EXTRACTOR_PROFILE_SCHEMA_ID,
    EXTRACTOR_PROFILE_SCHEMA_ID,
    load_production_source_identities,
    validate_production_sources,
)
from .runtime_boundary import (
    build_sanitized_worker_environment,
    load_runtime_profile,
    runtime_python_path,
    validate_runtime_root,
)
from .production_runtime_contract import (
    DINOV3_LANE_ID,
    DINOV3_SELECTOR,
    ProductionRuntimeProfile,
    load_production_runtime_profile,
    validate_profile_source_bindings,
)
from .snapshot_boundary import (
    APPLE_HF_SNAPSHOT_LAYOUT_ID,
    load_snapshot_profile,
    validate_apple_hf_snapshot,
    validate_snapshot_root,
)


EXTRACTION_REQUEST_SCHEMA_ID = "vea.dinov3_extraction_request.v0"
EXTRACTION_RESPONSE_SCHEMA_ID = "vea.dinov3_extraction_response.v0"
EXTRACTION_OPERATION = "extract_canonical_features"
MODEL_KEY = "dinov3_vit_b16"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
MODEL_STATE_SHA256 = "c382afd66ac658255f9f1406bad1e5e5fcf410af40cbc5c306b770792a4f30fd"
PROFILE_ID = "dinov3-vitb16-r5931719-isolated-extractor-v0"
APPLE_PROFILE_ID = "dinov3-vitb16-r5931719-apple-cuda-fp32-v1"
LEGACY_EXECUTION_SELECTOR = "legacy_windows_cpu"
TAP_CATALOG = (
    "embedding_output",
    "block_01",
    "block_02",
    "block_03",
    "block_04",
    "block_05",
    "block_06",
    "block_07",
    "block_08",
    "block_09",
    "block_10",
    "block_11",
    "block_12",
    "final_norm",
)
REQUIRED_MARKERS = (
    "DINOV3_REAL_EXTRACTOR_PROFILE_OK",
    "DINOV3_REAL_EXTRACTOR_DESCRIPTOR_OK",
    "DINOV3_REAL_EXTRACTOR_ALL_TAPS_OK",
    "DINOV3_REAL_EXTRACTOR_ARTIFACT_BOUNDARY_OK",
    "DINOV3_REAL_EXTRACTOR_RNG_STATE_OK",
    "DINOV3_REAL_EXTRACTOR_NETWORK_GUARD_OK",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SECRET_KEY = re.compile(
    r"(?i)(?:^|_)(?:access_token|auth_token|api_key|secret|password|passwd|credential)(?:$|_)"
)
_SECRET_VALUE = re.compile(
    r"(?i)(?:bearer\s+\S+|hf_[A-Za-z0-9]{10,}|(?:api[_-]?key|password|secret)\s*[:=])"
)
_RAW_KEYS = frozenset(
    {"raw_tensor", "tensor_values", "values", "checkpoint_content", "weight_content"}
)
_MAX_PROFILE_BYTES = 512 * 1024
_MAX_REQUEST_BYTES = 128 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class DINOv3ExtractionBoundaryError(RuntimeError):
    """Raised when the isolated extraction boundary fails closed."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


class SanitizedProcessFailure(RuntimeError):
    """Path- and payload-free cause retained below the public boundary error."""

    __slots__ = ("classification", "return_code", "timed_out")

    def __init__(
        self,
        classification: str,
        *,
        return_code: int | None = None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(classification)
        self.classification = classification
        self.return_code = return_code
        self.timed_out = timed_out

    def __repr__(self) -> str:
        return (
            "SanitizedProcessFailure("
            f"classification={self.classification!r}, "
            f"return_code={self.return_code!r}, timed_out={self.timed_out!r})"
        )


def _fail(classification: str, path: str, message: str) -> None:
    raise DINOv3ExtractionBoundaryError(classification, f"{path}: {message}")


class FrozenJSONMap(Mapping[str, Any]):
    """Recursively immutable JSON mapping with attribute lookup."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(value)))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FrozenJSONMap is immutable")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenJSONMap({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DINOv3ExtractionBoundaryError("io_error", "unable to hash file") from exc
    return digest.hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("invalid_json", "json", f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail("invalid_json", "json", f"non-finite value {value!r}")


def _parse_json(payload: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except DINOv3ExtractionBoundaryError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DINOv3ExtractionBoundaryError(
            "invalid_json", f"{context}: invalid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        _fail("invalid_json", context, "expected an object")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DINOv3ExtractionBoundaryError(
            "invalid_json", "document is not finite JSON"
        ) from exc


def _is_reparse(path: Path) -> bool:
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
        raise DINOv3ExtractionBoundaryError("unsafe_path", "unable to inspect path") from exc


def _read_regular(path: str | Path, maximum: int, context: str) -> tuple[Path, bytes]:
    candidate = Path(path)
    if not candidate.is_absolute() or _is_reparse(candidate):
        _fail("unsafe_path", context, "expected an absolute non-link file")
    try:
        resolved = candidate.resolve(strict=True)
        info = resolved.stat()
        payload = resolved.read_bytes()
    except OSError as exc:
        raise DINOv3ExtractionBoundaryError("io_error", f"{context} is unreadable") from exc
    if not stat.S_ISREG(info.st_mode) or len(payload) > maximum:
        _fail("unsafe_path", context, "file type or size is invalid")
    return resolved, payload


def _resolved_directory(path: str | Path, context: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or _is_reparse(candidate):
        _fail("unsafe_path", context, "expected an absolute non-link directory")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise DINOv3ExtractionBoundaryError("io_error", f"{context} is unavailable") from exc
    if not resolved.is_dir():
        _fail("unsafe_path", context, "expected a directory")
    return resolved


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _relative_path(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail("schema_validation_failed", path, "expected a POSIX relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        _fail("unsafe_path", path, "absolute paths are forbidden")
    if any(part in {"", ".", ".."} or ":" in part for part in value.split("/")):
        _fail("unsafe_path", path, "unsafe path component")
    return value


def resolve_workspace_relative_path(
    workspace_root: str | Path,
    relative_path: str,
    *,
    must_exist: bool,
    expect_directory: bool = False,
) -> Path:
    root = _resolved_directory(workspace_root, "workspace_root")
    relative = _relative_path(relative_path, "workspace_relative_path")
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.exists() or current.is_symlink():
            if _is_reparse(current):
                _fail("workspace_escape", "workspace_relative_path", "reparse component")
    try:
        resolved = current.resolve(strict=must_exist)
    except OSError as exc:
        raise DINOv3ExtractionBoundaryError(
            "io_error", "workspace path is unavailable"
        ) from exc
    if not _contains(root, resolved):
        _fail("workspace_escape", "workspace_relative_path", "path escapes workspace")
    if must_exist:
        if expect_directory and not resolved.is_dir():
            _fail("unsafe_path", "workspace_relative_path", "expected a directory")
        if not expect_directory and not resolved.is_file():
            _fail("unsafe_path", "workspace_relative_path", "expected a regular file")
    elif resolved.exists() or resolved.is_symlink():
        _fail("path_collision", "workspace_relative_path", "output already exists")
    return resolved


def _exact_mapping(value: object, fields: frozenset[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("schema_validation_failed", path, "expected an object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        _fail(
            "schema_validation_failed",
            path,
            f"field mismatch; missing={missing!r}, unknown={unknown!r}",
        )
    return value


def _identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        _fail("schema_validation_failed", path, "expected a safe identifier")
    return value


def _sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail("schema_validation_failed", path, "expected lowercase SHA-256")
    return value


def _string_tuple(value: object, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        _fail("schema_validation_failed", path, "expected an array")
    result = tuple(_identifier(item, f"{path}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not result:
        _fail("schema_validation_failed", path, "expected a non-empty array")
    if len(set(result)) != len(result):
        _fail("schema_validation_failed", path, "duplicates are forbidden")
    return result


def _scan_sensitive(value: object, path: str = "document") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SECRET_KEY.search(key) or key in _RAW_KEYS:
                _fail("sensitive_data_rejected", f"{path}.{key}", "forbidden field")
            _scan_sensitive(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_sensitive(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if _SECRET_VALUE.search(value):
            _fail("sensitive_data_rejected", path, "secret-like value")
        windows = PureWindowsPath(value)
        if PurePosixPath(value).is_absolute() or windows.is_absolute() or windows.drive:
            _fail("sensitive_data_rejected", path, "absolute path value")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail("invalid_json", path, "non-finite value")


@dataclass(frozen=True, slots=True)
class DINOv3ExtractorProfile:
    """Validated immutable extractor profile."""

    document: FrozenJSONMap
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.document)

    @property
    def source_sha256(self) -> str:
        return self.sha256

    @property
    def sources(self) -> FrozenJSONMap:
        return self.document.sources

    @property
    def identities(self) -> FrozenJSONMap:
        return self.document.identities

    @property
    def model(self) -> FrozenJSONMap:
        return self.document.model

    @property
    def tap_catalog(self) -> tuple[str, ...]:
        return tuple(self.document.tap_catalog)

    @property
    def production_sources(self) -> tuple[Any, ...]:
        return tuple(self.document.production_sources)

    @property
    def checkpoint_files(self) -> tuple[Any, ...]:
        return tuple(self.document.checkpoint_files)


@dataclass(frozen=True, slots=True)
class DINOv3AppleInvocationPlan:
    """Shell-free metadata-only plan for the explicit Apple worker lane."""

    command: tuple[str, ...]
    environment: Mapping[str, str]
    interpreter: Path
    selector: str
    device: str
    dtype: str


def build_dinov3_apple_invocation_plan(
    *,
    runtime_profile: ProductionRuntimeProfile,
    runtime_profile_path: str | Path,
    runtime_root: str | Path,
    repository_root: str | Path,
    workspace_root: str | Path,
    worker_script: str | Path,
    extractor_profile_path: str | Path,
    snapshot_profile_path: str | Path,
    snapshot_root: str | Path,
    request_relative_path: str,
    response_relative_path: str,
    environ: Mapping[str, str] | None = None,
) -> DINOv3AppleInvocationPlan:
    """Build, but never execute, the exact Linux isolated CUDA invocation."""

    if (
        not isinstance(runtime_profile, ProductionRuntimeProfile)
        or runtime_profile.lane_id != DINOV3_LANE_ID
        or runtime_profile.selector != DINOV3_SELECTOR
        or runtime_profile.system_site_packages
        or runtime_profile.interpreter_layout != "linux_bin"
        or runtime_profile.device != "cuda:0"
        or runtime_profile.dtype != "float32"
    ):
        _fail("profile_identity_mismatch", "runtime_profile", "not the exact Apple DINO lane")
    repository = _resolved_directory(repository_root, "repository_root")
    workspace = _resolved_directory(workspace_root, "workspace_root")
    runtime = _resolved_directory(runtime_root, "runtime_root")
    snapshot = _resolved_directory(snapshot_root, "snapshot_root")
    worker, _ = _read_regular(worker_script, 2 * 1024 * 1024, "worker_script")
    extractor_profile, _ = _read_regular(
        extractor_profile_path, _MAX_PROFILE_BYTES, "extractor_profile"
    )
    runtime_profile_file, runtime_profile_payload = _read_regular(
        runtime_profile_path, _MAX_PROFILE_BYTES, "runtime_profile"
    )
    snapshot_profile, _ = _read_regular(
        snapshot_profile_path, _MAX_PROFILE_BYTES, "snapshot_profile"
    )
    for source in (worker, extractor_profile, runtime_profile_file, snapshot_profile):
        try:
            source.relative_to(repository)
        except ValueError:
            _fail("unsafe_path", "repository_source", "Apple invocation source escapes repository")
    if hashlib.sha256(runtime_profile_payload).hexdigest() != runtime_profile.source_sha256:
        _fail("profile_identity_mismatch", "runtime_profile", "runtime profile SHA mismatch")
    request_relative = _relative_path(request_relative_path, "request_relative_path")
    response_relative = _relative_path(response_relative_path, "response_relative_path")
    interpreter = runtime_python_path(runtime, interpreter_layout="linux_bin")
    environment = build_sanitized_worker_environment(
        runtime_python=interpreter,
        repository_root=repository,
        workspace_root=workspace,
        environ=environ,
        cuda_visible_devices="0",
    )
    environment.update(
        {
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "VEA_DINOV3_EXECUTION_SELECTOR": DINOV3_SELECTOR,
        }
    )
    command = (
        str(interpreter),
        "-B",
        str(worker),
        "--execution-selector",
        DINOV3_SELECTOR,
        "--extractor-profile",
        str(extractor_profile),
        "--runtime-profile",
        str(runtime_profile_file),
        "--snapshot-profile",
        str(snapshot_profile),
        "--runtime-root",
        str(runtime),
        "--snapshot-root",
        str(snapshot),
        "--workspace-root",
        str(workspace),
        "--request",
        request_relative,
        "--response",
        response_relative,
    )
    return DINOv3AppleInvocationPlan(
        command=command,
        environment=MappingProxyType(dict(environment)),
        interpreter=interpreter,
        selector=DINOV3_SELECTOR,
        device="cuda:0",
        dtype="float32",
    )


_PROFILE_FIELDS = frozenset(
    {
        "checkpoint_files",
        "descriptor",
        "forbidden_operations",
        "forbidden_packages",
        "identities",
        "input_transport",
        "manual_preprocessing",
        "model",
        "output_artifact_policy",
        "policies",
        "profile_id",
        "production_sources",
        "protocol",
        "reference_feature_hash_count",
        "schema_id",
        "sources",
        "tap_catalog",
        "tap_mapping",
        "temporary_output_policy",
        "token_layout",
    }
)
_APPLE_PROFILE_FIELDS = _PROFILE_FIELDS | frozenset({"runtime_profile", "snapshot_layout"})


def _validate_profile(
    value: Mapping[str, Any],
    *,
    execution_selector: str = LEGACY_EXECUTION_SELECTOR,
) -> None:
    if execution_selector not in {LEGACY_EXECUTION_SELECTOR, DINOV3_SELECTOR}:
        _fail("profile_identity_mismatch", "execution_selector", "unknown execution selector")
    is_apple = execution_selector == DINOV3_SELECTOR
    _exact_mapping(value, _APPLE_PROFILE_FIELDS if is_apple else _PROFILE_FIELDS, "profile")
    _scan_sensitive(value, "profile")
    expected = {
        "schema_id": APPLE_EXTRACTOR_PROFILE_SCHEMA_ID if is_apple else EXTRACTOR_PROFILE_SCHEMA_ID,
        "profile_id": APPLE_PROFILE_ID if is_apple else PROFILE_ID,
        "reference_feature_hash_count": 28,
    }
    for key, required in expected.items():
        if value[key] != required:
            _fail("profile_identity_mismatch", f"profile.{key}", f"expected {required!r}")
    if is_apple:
        layout = _exact_mapping(
            value["snapshot_layout"],
            frozenset({"layout_id", "model_id", "revision"}),
            "profile.snapshot_layout",
        )
        if dict(layout) != {
            "layout_id": APPLE_HF_SNAPSHOT_LAYOUT_ID,
            "model_id": MODEL_ID,
            "revision": REVISION,
        }:
            _fail("profile_identity_mismatch", "profile.snapshot_layout", "layout mismatch")
    load_production_source_identities(value)
    model = _exact_mapping(
        value["model"],
        frozenset(
            {
                "config_class",
                "expected_parameter_count",
                "expected_parameter_tensor_count",
                "key",
                "model_class",
                "model_id",
                "revision",
                "state_sha256",
            }
        ),
        "profile.model",
    )
    exact_model = {
        "config_class": "DINOv3ViTConfig",
        "expected_parameter_count": 85_660_416,
        "expected_parameter_tensor_count": 211,
        "key": MODEL_KEY,
        "model_class": "DINOv3ViTModel",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "state_sha256": MODEL_STATE_SHA256,
    }
    if dict(model) != exact_model:
        _fail("profile_identity_mismatch", "profile.model", "model identity is not exact")
    identities = value["identities"]
    if not isinstance(identities, Mapping) or len(identities) != 9:
        _fail("profile_identity_mismatch", "profile.identities", "identity graph is incomplete")
    for key, digest in identities.items():
        _sha256(digest, f"profile.identities.{key}")
    if tuple(value["tap_catalog"]) != TAP_CATALOG:
        _fail("profile_identity_mismatch", "profile.tap_catalog", "tap catalog mismatch")
    mapping = value["tap_mapping"]
    if not isinstance(mapping, list) or len(mapping) != len(TAP_CATALOG):
        _fail("profile_identity_mismatch", "profile.tap_mapping", "mapping count mismatch")
    names = tuple(item.get("name") if isinstance(item, Mapping) else None for item in mapping)
    if names != TAP_CATALOG:
        _fail("profile_identity_mismatch", "profile.tap_mapping", "mapping order mismatch")
    expected_sources = {
        "runtime_profile_path",
        "semantic_evidence_path",
        "semantic_profile_path",
        "snapshot_profile_path",
    }
    sources = _exact_mapping(value["sources"], frozenset(expected_sources), "profile.sources")
    for key, source in sources.items():
        _relative_path(source, f"profile.sources.{key}")
    protocol = value["protocol"]
    if protocol != {
        "allowed_operation": EXTRACTION_OPERATION,
        "request_schema_id": EXTRACTION_REQUEST_SCHEMA_ID,
        "response_schema_id": EXTRACTION_RESPONSE_SCHEMA_ID,
    }:
        _fail("profile_identity_mismatch", "profile.protocol", "protocol mismatch")
    token = value["token_layout"]
    if not isinstance(token, Mapping) or (
        token.get("total_token_count") != 201
        or token.get("patch_token_indices") != list(range(5, 201))
        or token.get("special_token_indices") != [0, 1, 2, 3, 4]
    ):
        _fail("profile_identity_mismatch", "profile.token_layout", "token layout mismatch")
    policies = value["policies"]
    if is_apple:
        expected_policies = {
            "cpu_only": False,
            "cuda_required": True,
            "device": "cuda:0",
            "dtype": "float32",
            "evaluation_mode": True,
            "fp32_only": True,
            "frozen_parameters": True,
            "inference_mode": True,
            "local_files_only": True,
            "network_allowed": False,
            "processor_allowed": False,
            "ready_for_p1_extraction": False,
            "token_allowed": False,
            "torchvision_allowed": False,
            "trust_remote_code": False,
            "use_safetensors": True,
        }
        runtime_link = _exact_mapping(
            value["runtime_profile"],
            frozenset({"path", "profile_id", "selector", "sha256"}),
            "profile.runtime_profile",
        )
        if dict(runtime_link) != {
            "path": "configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json",
            "profile_id": "dinov3-apple-cuda-fp32-v1",
            "selector": DINOV3_SELECTOR,
            "sha256": value["identities"]["runtime_profile_sha256"],
        }:
            _fail("profile_identity_mismatch", "profile.runtime_profile", "Apple runtime link mismatch")
    else:
        expected_policies = {
            "cpu_only": True,
            "evaluation_mode": True,
            "fp32_only": True,
            "frozen_parameters": True,
            "inference_mode": True,
            "local_files_only": True,
            "network_allowed": False,
            "processor_allowed": False,
            "token_allowed": False,
            "torchvision_allowed": False,
            "trust_remote_code": False,
            "use_safetensors": True,
        }
    if not isinstance(policies, Mapping) or dict(policies) != expected_policies:
        _fail("profile_identity_mismatch", "profile.policies", "worker policy mismatch")


def load_dinov3_extractor_profile(
    path: str | Path,
    *,
    execution_selector: str = LEGACY_EXECUTION_SELECTOR,
) -> DINOv3ExtractorProfile:
    """Load one strict canonical extractor profile."""

    _, payload = _read_regular(path, _MAX_PROFILE_BYTES, "extractor_profile")
    value = _parse_json(payload, "extractor_profile")
    _validate_profile(value, execution_selector=execution_selector)
    return DINOv3ExtractorProfile(_freeze(value), hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True, slots=True)
class DINOv3ExtractionRequest:
    """Strict metadata-only request sent to the persistent worker."""

    run_id: str
    extractor_profile_sha256: str
    model_key: str
    model_id: str
    revision: str
    image_ids: tuple[str, ...]
    requested_tap_names: tuple[str, ...]
    input_shape: tuple[int, int, int, int]
    input_dtype: str
    raw_input_sha256: str
    input_file_sha256: str
    workspace_relative_input_path: str
    workspace_relative_output_directory: str
    expected_worker_policy: FrozenJSONMap

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_worker_policy": _thaw(self.expected_worker_policy),
            "extractor_profile_sha256": self.extractor_profile_sha256,
            "image_ids": list(self.image_ids),
            "input_dtype": self.input_dtype,
            "input_file_sha256": self.input_file_sha256,
            "input_shape": list(self.input_shape),
            "model_id": self.model_id,
            "model_key": self.model_key,
            "operation": EXTRACTION_OPERATION,
            "raw_input_sha256": self.raw_input_sha256,
            "requested_tap_names": list(self.requested_tap_names),
            "revision": self.revision,
            "run_id": self.run_id,
            "schema_id": EXTRACTION_REQUEST_SCHEMA_ID,
            "workspace_relative_input_path": self.workspace_relative_input_path,
            "workspace_relative_output_directory": self.workspace_relative_output_directory,
        }


_REQUEST_FIELDS = frozenset(
    {
        "expected_worker_policy",
        "extractor_profile_sha256",
        "image_ids",
        "input_dtype",
        "input_file_sha256",
        "input_shape",
        "model_id",
        "model_key",
        "operation",
        "raw_input_sha256",
        "requested_tap_names",
        "revision",
        "run_id",
        "schema_id",
        "workspace_relative_input_path",
        "workspace_relative_output_directory",
    }
)


_EXPECTED_WORKER_POLICY = {
    "cpu_only": True,
    "fp32_only": True,
    "local_files_only": True,
    "network_allowed": False,
    "processor_allowed": False,
    "torchvision_allowed": False,
}


_APPLE_EXPECTED_WORKER_POLICY = {
    "cpu_fallback": False,
    "cpu_only": False,
    "cuda_required": True,
    "device": "cuda:0",
    "dtype": "float32",
    "fp32_only": True,
    "local_files_only": True,
    "mixed_precision": False,
    "network_allowed": False,
    "processor_allowed": False,
    "token_allowed": False,
    "torchvision_allowed": False,
}


def validate_dinov3_extraction_request(
    value: Mapping[str, Any],
    *,
    execution_selector: str = LEGACY_EXECUTION_SELECTOR,
) -> DINOv3ExtractionRequest:
    """Validate request schema, identities, geometry, and ordered taps."""

    item = _exact_mapping(value, _REQUEST_FIELDS, "request")
    _scan_sensitive(item, "request")
    if item["schema_id"] != EXTRACTION_REQUEST_SCHEMA_ID or item["operation"] != EXTRACTION_OPERATION:
        _fail("request_invalid", "request.protocol", "unsupported schema or operation")
    run_id = _identifier(item["run_id"], "request.run_id")
    image_ids = _string_tuple(item["image_ids"], "request.image_ids")
    taps = _string_tuple(item["requested_tap_names"], "request.requested_tap_names")
    if any(tap not in TAP_CATALOG for tap in taps):
        _fail("request_invalid", "request.requested_tap_names", "unknown tap")
    shape_value = item["input_shape"]
    if not isinstance(shape_value, list) or len(shape_value) != 4 or any(type(x) is not int for x in shape_value):
        _fail("request_invalid", "request.input_shape", "expected four integers")
    shape = tuple(shape_value)
    if shape != (len(image_ids), 224, 224, 3):
        _fail("request_invalid", "request.input_shape", "canonical geometry mismatch")
    identities = (item["model_key"], item["model_id"], item["revision"])
    if identities != (MODEL_KEY, MODEL_ID, REVISION):
        _fail("request_invalid", "request.model", "model identity mismatch")
    if execution_selector not in {LEGACY_EXECUTION_SELECTOR, DINOV3_SELECTOR}:
        _fail("request_invalid", "execution_selector", "unknown execution selector")
    policy = item["expected_worker_policy"]
    expected_policy = (
        _APPLE_EXPECTED_WORKER_POLICY
        if execution_selector == DINOV3_SELECTOR
        else _EXPECTED_WORKER_POLICY
    )
    if policy != expected_policy:
        _fail("request_invalid", "request.expected_worker_policy", "policy mismatch")
    return DINOv3ExtractionRequest(
        run_id=run_id,
        extractor_profile_sha256=_sha256(item["extractor_profile_sha256"], "request.extractor_profile_sha256"),
        model_key=MODEL_KEY,
        model_id=MODEL_ID,
        revision=REVISION,
        image_ids=image_ids,
        requested_tap_names=taps,
        input_shape=shape,  # type: ignore[arg-type]
        input_dtype=(
            item["input_dtype"]
            if item["input_dtype"] == "uint8"
            else _fail("request_invalid", "request.input_dtype", "expected uint8")
        ),
        raw_input_sha256=_sha256(item["raw_input_sha256"], "request.raw_input_sha256"),
        input_file_sha256=_sha256(item["input_file_sha256"], "request.input_file_sha256"),
        workspace_relative_input_path=_relative_path(item["workspace_relative_input_path"], "request.workspace_relative_input_path"),
        workspace_relative_output_directory=_relative_path(item["workspace_relative_output_directory"], "request.workspace_relative_output_directory"),
        expected_worker_policy=_freeze(policy),
    )


def write_dinov3_extraction_request(
    path: str | Path,
    request: DINOv3ExtractionRequest,
    *,
    workspace_root: str | Path,
    execution_selector: str = LEGACY_EXECUTION_SELECTOR,
) -> Path:
    """Exclusively publish canonical request JSON within the workspace."""

    if not isinstance(request, DINOv3ExtractionRequest):
        _fail("request_invalid", "request", "unexpected object type")
    validated = validate_dinov3_extraction_request(
        request.to_dict(), execution_selector=execution_selector
    )
    root = _resolved_directory(workspace_root, "workspace_root")
    candidate = Path(path)
    if not candidate.is_absolute():
        _fail("unsafe_path", "request_path", "expected an absolute path")
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        _fail("workspace_escape", "request_path", "path escapes workspace")
    destination = resolve_workspace_relative_path(root, relative, must_exist=False)
    payload = _canonical_json(validated.to_dict())
    if len(payload) > _MAX_REQUEST_BYTES:
        _fail("request_invalid", "request", "serialized request is too large")
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise DINOv3ExtractionBoundaryError("io_error", "unable to write request") from exc
    return destination


def read_dinov3_extraction_request(
    path: str | Path,
    *,
    execution_selector: str = LEGACY_EXECUTION_SELECTOR,
) -> DINOv3ExtractionRequest:
    """Read a bounded request document with strict JSON semantics."""

    _, payload = _read_regular(path, _MAX_REQUEST_BYTES, "request")
    return validate_dinov3_extraction_request(
        _parse_json(payload, "request"), execution_selector=execution_selector
    )


@dataclass(frozen=True, slots=True)
class DINOv3ExtractionResponse:
    """Validated response containing identities and artifact metadata only."""

    document: FrozenJSONMap

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.document)


_RESPONSE_FIELDS = frozenset(
    {
        "artifacts",
        "canonical_mapping",
        "descriptor",
        "failure_classification",
        "forward_count",
        "identities",
        "image_ids",
        "input",
        "invariants",
        "markers",
        "model_state_sha256",
        "operation",
        "processor_construction_count",
        "requested_tap_names",
        "run_id",
        "schema_id",
        "socket_attempt_count",
        "status",
        "torchvision_import_count",
        "warnings",
        "workspace_paths",
    }
)


_ARTIFACT_FIELDS = frozenset(
    {
        "checksum_sha256",
        "dtype",
        "global_sha256",
        "global_shape",
        "patch_sha256",
        "patch_shape",
        "safetensors_relative_path",
        "sidecar_relative_path",
        "tap_name",
    }
)


def validate_dinov3_extraction_response(
    value: Mapping[str, Any],
    *,
    request: DINOv3ExtractionRequest,
    profile: DINOv3ExtractorProfile,
) -> DINOv3ExtractionResponse:
    """Validate a metadata-only response against the exact invocation."""

    item = _exact_mapping(value, _RESPONSE_FIELDS, "response")
    _scan_sensitive(item, "response")
    exact = {
        "schema_id": EXTRACTION_RESPONSE_SCHEMA_ID,
        "operation": EXTRACTION_OPERATION,
        "status": "success",
        "run_id": request.run_id,
        "model_state_sha256": MODEL_STATE_SHA256,
        "forward_count": 1,
        "processor_construction_count": 0,
        "torchvision_import_count": 0,
        "socket_attempt_count": 0,
        "failure_classification": None,
    }
    for key, expected in exact.items():
        if item[key] != expected:
            _fail("worker_response_mismatch", f"response.{key}", f"expected {expected!r}")
    if tuple(item["image_ids"]) != request.image_ids or tuple(item["requested_tap_names"]) != request.requested_tap_names:
        _fail("worker_response_mismatch", "response.order", "image or tap order mismatch")
    identities = item["identities"]
    expected_identities = {
        "external_snapshot_manifest_sha256": profile.identities.external_snapshot_manifest_sha256,
        "extractor_profile_sha256": profile.sha256,
        "persistent_freeze_sha256": profile.identities.persistent_freeze_sha256,
        "persistent_runtime_manifest_sha256": profile.identities.persistent_runtime_manifest_sha256,
        "runtime_profile_sha256": profile.identities.runtime_profile_sha256,
        "snapshot_profile_sha256": profile.identities.snapshot_profile_sha256,
    }
    if identities != expected_identities:
        _fail("worker_response_mismatch", "response.identities", "identity graph mismatch")
    input_facts = item["input"]
    if not isinstance(input_facts, Mapping) or (
        input_facts.get("raw_input_sha256") != request.raw_input_sha256
        or input_facts.get("input_file_sha256") != request.input_file_sha256
        or input_facts.get("input_unchanged") is not True
        or input_facts.get("normalized_dtype") != "float32"
        or input_facts.get("normalized_shape") != [len(request.image_ids), 3, 224, 224]
        or _SHA256.fullmatch(str(input_facts.get("normalized_sha256"))) is None
    ):
        _fail("worker_response_mismatch", "response.input", "input facts mismatch")
    descriptor = item["descriptor"]
    if descriptor != {
        "grid_height": 14,
        "grid_width": 14,
        "hidden_dimension": 768,
        "layer_count": 12,
        "output_dtype": "float32",
        "patch_count": 196,
        "total_token_count": 201,
    }:
        _fail("worker_response_mismatch", "response.descriptor", "descriptor mismatch")
    mapping = item["canonical_mapping"]
    expected_mapping = {
        "block_12_source": "raw_final_block_hook",
        "final_norm_source": "last_hidden_state",
        "global_source": "patch_axis_mean",
        "hidden_state_12_relation": "final_norm",
        "native_pooler_excluded": True,
        "patch_token_indices": list(range(5, 201)),
        "raw_block_12_differs_from_final_norm": True,
        "special_tokens_excluded": True,
    }
    if mapping != expected_mapping:
        _fail("semantic_mapping_mismatch", "response.canonical_mapping", "mapping mismatch")
    invariants = item["invariants"]
    expected_invariants = {
        "model_state_unchanged": True,
        "numpy_rng_unchanged": True,
        "parameters_frozen": True,
        "python_rng_unchanged": True,
        "torch_rng_unchanged": True,
        "training_false": True,
    }
    if invariants != expected_invariants:
        _fail("worker_response_mismatch", "response.invariants", "invariant failure")
    artifacts = item["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(request.requested_tap_names):
        _fail("artifact_boundary_failed", "response.artifacts", "artifact count mismatch")
    expected_paths: list[str] = []
    for index, raw in enumerate(artifacts):
        artifact = _exact_mapping(raw, _ARTIFACT_FIELDS, f"response.artifacts[{index}]")
        tap = request.requested_tap_names[index]
        if artifact["tap_name"] != tap:
            _fail("artifact_boundary_failed", f"response.artifacts[{index}].tap_name", "order mismatch")
        sidecar = _relative_path(artifact["sidecar_relative_path"], f"response.artifacts[{index}].sidecar_relative_path")
        tensors = _relative_path(artifact["safetensors_relative_path"], f"response.artifacts[{index}].safetensors_relative_path")
        prefix = f"output/{index:02d}-{tap}"
        if sidecar != f"{prefix}/shard-{index:05d}.json" or tensors != f"{prefix}/shard-{index:05d}.safetensors":
            _fail("artifact_boundary_failed", f"response.artifacts[{index}]", "artifact path mismatch")
        if artifact["patch_shape"] != [len(request.image_ids), 196, 768] or artifact["global_shape"] != [len(request.image_ids), 768] or artifact["dtype"] != "float32":
            _fail("artifact_boundary_failed", f"response.artifacts[{index}]", "shape or dtype mismatch")
        for field in ("checksum_sha256", "patch_sha256", "global_sha256"):
            _sha256(artifact[field], f"response.artifacts[{index}].{field}")
        expected_paths.extend((sidecar, tensors))
    if item["workspace_paths"] != expected_paths:
        _fail("artifact_boundary_failed", "response.workspace_paths", "path inventory mismatch")
    if item["warnings"] != [] or tuple(item["markers"]) != REQUIRED_MARKERS:
        _fail("worker_response_mismatch", "response.markers", "markers or warnings mismatch")
    return DINOv3ExtractionResponse(_freeze(dict(item)))


def read_dinov3_extraction_response(
    path: str | Path,
    *,
    workspace_root: str | Path,
    request: DINOv3ExtractionRequest,
    profile: DINOv3ExtractorProfile,
) -> DINOv3ExtractionResponse:
    """Read a strict response and verify every referenced workspace file."""

    root = _resolved_directory(workspace_root, "workspace_root")
    candidate = Path(path)
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        _fail("workspace_escape", "response_path", "path escapes workspace")
    resolved = resolve_workspace_relative_path(root, relative, must_exist=True)
    _, payload = _read_regular(resolved, _MAX_RESPONSE_BYTES, "response")
    response = validate_dinov3_extraction_response(
        _parse_json(payload, "response"), request=request, profile=profile
    )
    for relative_path in response.document.workspace_paths:
        resolve_workspace_relative_path(root, relative_path, must_exist=True)
    return response


def invoke_dinov3_extraction_worker(
    *,
    runtime_root: str | Path,
    snapshot_root: str | Path,
    repository_root: str | Path,
    extractor_profile_path: str | Path,
    worker_script: str | Path,
    workspace_root: str | Path,
    request: DINOv3ExtractionRequest,
    timeout_seconds: float,
    execution_selector: str = "legacy_windows_cpu",
) -> DINOv3ExtractionResponse:
    """Launch exact persistent Python with ``shell=False`` and validate output."""

    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        _fail("schema_validation_failed", "timeout_seconds", "expected positive finite value")
    if execution_selector not in {LEGACY_EXECUTION_SELECTOR, DINOV3_SELECTOR}:
        _fail("profile_identity_mismatch", "execution_selector", "unknown execution selector")
    is_apple = execution_selector == DINOV3_SELECTOR
    repository = _resolved_directory(repository_root, "repository_root")
    runtime = _resolved_directory(runtime_root, "runtime_root")
    snapshot = _resolved_directory(snapshot_root, "snapshot_root")
    workspace = _resolved_directory(workspace_root, "workspace_root")
    roots = (repository, runtime, snapshot, workspace)
    if any(
        _contains(left, right) or _contains(right, left)
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        _fail("unsafe_path", "roots", "repository, runtime, snapshot, and workspace must be disjoint")
    profile_file, _ = _read_regular(extractor_profile_path, _MAX_PROFILE_BYTES, "extractor_profile")
    worker_file, _ = _read_regular(worker_script, 2 * 1024 * 1024, "worker_script")
    try:
        profile_file.relative_to(repository)
        worker_file.relative_to(repository)
    except ValueError:
        _fail("unsafe_path", "repository_source", "profile and worker must be repository files")
    profile = (
        load_dinov3_extractor_profile(
            profile_file, execution_selector=DINOV3_SELECTOR
        )
        if is_apple
        else load_dinov3_extractor_profile(profile_file)
    )
    if request.extractor_profile_sha256 != profile.sha256:
        _fail("request_profile_mismatch", "request.extractor_profile_sha256", "profile mismatch")
    validate_production_sources(repository, profile_file)
    snapshot_profile_path = repository / str(profile.sources.snapshot_profile_path)
    snapshot_profile = load_snapshot_profile(snapshot_profile_path)
    snapshot_manifest = (
        validate_apple_hf_snapshot(
            snapshot,
            cache_root=snapshot.parents[2],
            model_id=MODEL_ID,
            revision=REVISION,
            files=profile.checkpoint_files,
        )
        if is_apple
        else validate_snapshot_root(
            snapshot,
            snapshot_profile,
            repository_root=repository,
            persistent_runtime_root=runtime,
        )
    )
    runtime_profile_path = repository / str(profile.sources.runtime_profile_path)
    if is_apple:
        runtime_profile = load_production_runtime_profile(runtime_profile_path)
        validate_profile_source_bindings(runtime_profile, repository)
        if (
            not runtime_profile.package_metadata_resolved
            or runtime_profile.selector != DINOV3_SELECTOR
            or runtime_profile.lane_id != DINOV3_LANE_ID
            or runtime_profile.runtime_root != runtime.as_posix()
            or runtime_profile.interpreter_layout != "linux_bin"
            or runtime_profile.system_site_packages
            or runtime_profile.device != "cuda:0"
            or runtime_profile.dtype != "float32"
            or runtime_profile.cpu_fallback
            or runtime_profile.mixed_precision
            or runtime_profile.network_allowed
            or runtime_profile.ready_for_p1_extraction
        ):
            _fail("profile_identity_mismatch", "runtime_profile", "Apple runtime contract mismatch")
        actual_identities = {
            "runtime_profile_sha256": runtime_profile.source_sha256,
            "snapshot_profile_sha256": snapshot_profile.source_sha256,
            "external_snapshot_manifest_sha256": snapshot_manifest.source_sha256,
            "registry_sha256": sha256_file(repository / "configs/models.yaml"),
            "semantic_profile_sha256": sha256_file(
                repository / str(profile.sources.semantic_profile_path)
            ),
            "semantic_evidence_sha256": sha256_file(
                repository / str(profile.sources.semantic_evidence_path)
            ),
        }
    else:
        runtime_profile = load_runtime_profile(runtime_profile_path)
        runtime_manifest = validate_runtime_root(
            runtime, runtime_profile, repository_root=repository
        )
        actual_identities = {
            "runtime_profile_sha256": runtime_profile.source_sha256,
            "snapshot_profile_sha256": snapshot_profile.source_sha256,
            "persistent_runtime_manifest_sha256": runtime_manifest.source_sha256,
            "external_snapshot_manifest_sha256": snapshot_manifest.source_sha256,
            "persistent_freeze_sha256": runtime_manifest.candidate_freeze_sha256,
        }
    for key, actual in actual_identities.items():
        if profile.identities[key] != actual:
            _fail("profile_identity_mismatch", f"profile.identities.{key}", "live identity mismatch")
    resolve_workspace_relative_path(workspace, request.workspace_relative_input_path, must_exist=True)
    resolve_workspace_relative_path(workspace, request.workspace_relative_output_directory, must_exist=False)
    request_path = (
        write_dinov3_extraction_request(
            workspace / "request.json",
            request,
            workspace_root=workspace,
            execution_selector=DINOV3_SELECTOR,
        )
        if is_apple
        else write_dinov3_extraction_request(
            workspace / "request.json", request, workspace_root=workspace
        )
    )
    response_path = workspace / "response.json"
    if is_apple:
        assert isinstance(runtime_profile, ProductionRuntimeProfile)
        plan = build_dinov3_apple_invocation_plan(
            runtime_profile=runtime_profile,
            runtime_profile_path=runtime_profile_path,
            runtime_root=runtime,
            repository_root=repository,
            workspace_root=workspace,
            worker_script=worker_file,
            extractor_profile_path=profile_file,
            snapshot_profile_path=snapshot_profile_path,
            snapshot_root=snapshot,
            request_relative_path=request_path.relative_to(workspace).as_posix(),
            response_relative_path=response_path.relative_to(workspace).as_posix(),
        )
        command = list(plan.command)
        environment = dict(plan.environment)
    else:
        python = runtime_python_path(runtime)
        environment = build_sanitized_worker_environment(
            runtime_python=python,
            repository_root=repository,
            workspace_root=workspace,
            cuda_visible_devices="",
        )
        environment.update(
            {
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "DO_NOT_TRACK": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        command = [
            str(python),
            "-B",
            str(worker_file),
            "--extractor-profile",
            str(profile_file),
            "--runtime-profile",
            str(runtime_profile_path),
            "--snapshot-profile",
            str(snapshot_profile_path),
            "--runtime-root",
            str(runtime),
            "--snapshot-root",
            str(snapshot),
            "--workspace-root",
            str(workspace),
            "--request",
            request_path.relative_to(workspace).as_posix(),
            "--response",
            response_path.relative_to(workspace).as_posix(),
        ]
    process_failure: SanitizedProcessFailure | None = None
    completed: subprocess.CompletedProcess[bytes] | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            shell=False,
            check=False,
            capture_output=True,
            text=False,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired:
        process_failure = SanitizedProcessFailure("worker_timeout", timed_out=True)
    except OSError:
        process_failure = SanitizedProcessFailure("worker_launch_failed")
    if process_failure is not None:
        raise DINOv3ExtractionBoundaryError(
            process_failure.classification, "isolated extraction process failed"
        ) from process_failure
    assert completed is not None
    if completed.returncode != 0:
        process_failure = SanitizedProcessFailure(
            "worker_nonzero_exit", return_code=completed.returncode
        )
        raise DINOv3ExtractionBoundaryError(
            "worker_nonzero_exit", f"isolated extraction worker exited {completed.returncode}"
        ) from process_failure
    return read_dinov3_extraction_response(
        response_path,
        workspace_root=workspace,
        request=request,
        profile=profile,
    )


__all__ = [
    "DINOv3ExtractionBoundaryError",
    "SanitizedProcessFailure",
    "DINOv3ExtractionRequest",
    "DINOv3ExtractionResponse",
    "DINOv3AppleInvocationPlan",
    "DINOv3ExtractorProfile",
    "EXTRACTION_OPERATION",
    "EXTRACTION_REQUEST_SCHEMA_ID",
    "EXTRACTION_RESPONSE_SCHEMA_ID",
    "EXTRACTOR_PROFILE_SCHEMA_ID",
    "MODEL_ID",
    "MODEL_KEY",
    "MODEL_STATE_SHA256",
    "REQUIRED_MARKERS",
    "REVISION",
    "TAP_CATALOG",
    "build_dinov3_apple_invocation_plan",
    "load_dinov3_extractor_profile",
    "invoke_dinov3_extraction_worker",
    "read_dinov3_extraction_response",
    "resolve_workspace_relative_path",
    "sha256_file",
    "validate_dinov3_extraction_request",
    "validate_dinov3_extraction_response",
    "write_dinov3_extraction_request",
]
