"""Strict, import-light boundary for the external DINOv3 snapshot.

The module validates repository snapshot profiles and external snapshot
manifests using only the Python standard library.  Filesystem verification is
read-only and rejects links, reparse points, root overlap, partial layouts, and
content that does not match the exact size/SHA-256 inventory.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any


SNAPSHOT_PROFILE_SCHEMA_ID = "vea.model_snapshot_profile.v0"
SNAPSHOT_MANIFEST_SCHEMA_ID = "vea.external_model_snapshot.v0"

SNAPSHOT_ID = "dinov3-vitb16-r5931719-safetensors-v1"
REGISTRY_KEY = "dinov3_vit_b16"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
ACCESS_CLASSIFICATION = "gated_manual_existing_user_authorization"
WEIGHT_FORMAT = "single_file_safetensors"
PERSISTENT_RUNTIME_ID = (
    "dinov3-vitb16-r5931719-py312-torch230-tf4560-manual-v1"
)
PERSISTENT_RUNTIME_PROFILE_SHA256 = '52fc589cc47127c4c764f9631f17777f31f9784f7d4b92426f5b31ac4e3a2163'
REQUIRED_PERSISTENT_FREEZE_SHA256 = (
    "995687ed934c0a94b52a040af048f87bc8de4e5089475f70565ad37f1f33ecd9"
)
CONFIG_CLASS = "DINOv3ViTConfig"
MODEL_CLASS = "DINOv3ViTModel"
EXPECTED_PARAMETER_COUNT = 85_660_416
EXPECTED_PARAMETER_TENSOR_COUNT = 211
SELECTED_TOTAL_BYTES = 342_663_521

_EXPECTED_FILE_ROWS = (
    (
        "config.json",
        744,
        "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec",
    ),
    (
        "preprocessor_config.json",
        585,
        "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
    ),
    (
        "model.safetensors",
        342_662_192,
        "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
    ),
)
_EXPECTED_EXTERNAL_ROOT = {
    "default_template": f"%LOCALAPPDATA%/VEA/snapshots/{SNAPSHOT_ID}",
    "environment_variable": "LOCALAPPDATA",
    "persistent_runtime_relation": "outside_persistent_runtime",
    "relative_parts": ["VEA", "snapshots", SNAPSHOT_ID],
    "repository_relation": "outside_repository",
    "staging_name_prefix": f".{SNAPSHOT_ID}.staging-",
}
_EXPECTED_LOAD_POLICY = {
    "cpu_only": True,
    "dtype": "float32",
    "forward_allowed": False,
    "local_files_only": True,
    "processor_object_allowed": False,
    "registry_mutation_allowed": False,
    "trust_remote_code": False,
    "use_safetensors": True,
    "user_hub_cache_allowed": False,
}

_MAX_PROFILE_BYTES = 256 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_EMAIL_PATTERN = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
)
_TOKEN_VALUE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+\S+|hf_[A-Za-z0-9]{10,}|"
    r"(?:token|secret|password|passwd|api[_-]?key)\s*[:=])"
)


class SnapshotBoundaryError(RuntimeError):
    """Raised when the model-snapshot boundary contract is violated."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def _fail(classification: str, path: str, message: str) -> None:
    raise SnapshotBoundaryError(classification, f"{path}: {message}")


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("invalid_json", "json", f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    _fail("invalid_json", "json", f"non-finite numeric value {value!r}")


def _parse_json_bytes(payload: bytes, *, context: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except SnapshotBoundaryError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotBoundaryError(
            "invalid_json", f"{context}: invalid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        _fail("invalid_json", context, "expected a JSON object")
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize finite JSON deterministically with sorted keys and LF ending."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise SnapshotBoundaryError(
            "invalid_json", "json: value is not finite JSON data"
        ) from exc
    return text.encode("utf-8")


def _is_reparse_point(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return bool(reparse_flag and attributes & reparse_flag)
    except OSError as exc:
        raise SnapshotBoundaryError(
            "path_validation_failed", "unable to inspect a filesystem path"
        ) from exc


def _read_bounded_regular_file(
    path: str | Path,
    *,
    maximum_bytes: int,
    context: str,
) -> tuple[Path, bytes]:
    candidate = Path(path)
    if not candidate.is_absolute():
        _fail("unsafe_path", context, "expected an absolute path")
    if _is_reparse_point(candidate):
        _fail("unsafe_path", context, "symbolic links and reparse points are forbidden")
    try:
        resolved = candidate.resolve(strict=True)
        file_stat = resolved.stat()
    except OSError as exc:
        raise SnapshotBoundaryError(
            "path_validation_failed", f"{context}: file is not readable"
        ) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        _fail("unsafe_path", context, "expected a regular file")
    if file_stat.st_size > maximum_bytes:
        _fail("invalid_json", context, f"file exceeds {maximum_bytes} bytes")
    try:
        return resolved, resolved.read_bytes()
    except OSError as exc:
        raise SnapshotBoundaryError(
            "path_validation_failed", f"{context}: file is not readable"
        ) from exc


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one non-link regular file."""

    candidate = Path(path)
    if _is_reparse_point(candidate):
        _fail("unsafe_path", "file", "symbolic links and reparse points are forbidden")
    digest = hashlib.sha256()
    try:
        file_stat = candidate.stat()
        if not stat.S_ISREG(file_stat.st_mode):
            _fail("unsafe_path", "file", "expected a regular file")
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except SnapshotBoundaryError:
        raise
    except OSError as exc:
        raise SnapshotBoundaryError(
            "path_validation_failed", "file: unable to hash required content"
        ) from exc
    return digest.hexdigest()


def _expect_exact_fields(
    value: object,
    expected: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("schema_validation_failed", path, "expected an object")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _fail(
            "schema_validation_failed",
            path,
            f"missing required fields: {', '.join(missing)}",
        )
    if unknown:
        _fail(
            "schema_validation_failed",
            path,
            f"unknown fields are not allowed: {', '.join(unknown)}",
        )
    return value


def _expect_string(value: object, path: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        _fail("schema_validation_failed", path, "expected a non-empty string")
    if len(value) > maximum:
        _fail("schema_validation_failed", path, "string is too long")
    if any(ord(character) < 32 for character in value):
        _fail("schema_validation_failed", path, "control characters are forbidden")
    return value


def _expect_identifier(value: object, path: str) -> str:
    identifier = _expect_string(value, path, maximum=128)
    if _IDENTIFIER_PATTERN.fullmatch(identifier) is None:
        _fail("schema_validation_failed", path, "expected a safe identifier")
    return identifier


def _expect_sha256(value: object, path: str) -> str:
    digest = _expect_string(value, path, maximum=64)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        _fail("schema_validation_failed", path, "expected a lowercase SHA-256 digest")
    return digest


def _expect_revision(value: object, path: str) -> str:
    revision = _expect_string(value, path, maximum=40)
    if _GIT_SHA_PATTERN.fullmatch(revision) is None:
        _fail("schema_validation_failed", path, "expected an immutable 40-hex revision")
    return revision


def _expect_int(
    value: object,
    path: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        _fail(
            "schema_validation_failed", path, f"expected an integer >= {minimum}"
        )
    if maximum is not None and value > maximum:
        _fail(
            "schema_validation_failed", path, f"expected an integer <= {maximum}"
        )
    return value


def _expect_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail("schema_validation_failed", path, "expected a boolean")
    return value


def _expect_sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail("schema_validation_failed", path, "expected an array")
    return value


def _expect_basename(value: object, path: str) -> str:
    name = _expect_string(value, path, maximum=255)
    if (
        PurePosixPath(name).name != name
        or PureWindowsPath(name).name != name
        or PureWindowsPath(name).drive
        or name in {".", ".."}
    ):
        _fail("unsafe_path", path, "expected a basename")
    return name


def _expect_exact(actual: object, expected: object, path: str) -> None:
    if actual != expected:
        _fail("snapshot_identity_mismatch", path, f"expected {expected!r}")


def _reject_sensitive_string(value: str, path: str, *, allow_template: bool = False) -> None:
    if _EMAIL_PATTERN.search(value) is not None:
        _fail("sensitive_data_rejected", path, "email-like values are forbidden")
    if _TOKEN_VALUE_PATTERN.search(value) is not None:
        _fail("sensitive_data_rejected", path, "secret-like values are forbidden")
    if allow_template and value.startswith("%LOCALAPPDATA%/"):
        return
    windows = PureWindowsPath(value)
    if PurePosixPath(value).is_absolute() or windows.is_absolute() or windows.drive:
        _fail("sensitive_data_rejected", path, "absolute paths are forbidden")


def _scan_sensitive_strings(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _scan_sensitive_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_sensitive_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        _reject_sensitive_string(
            value,
            path,
            allow_template=path.endswith("external_root.default_template"),
        )
    elif isinstance(value, float) and not math.isfinite(value):
        _fail("schema_validation_failed", path, "non-finite values are forbidden")


class FrozenMapping(Mapping[str, Any]):
    """Recursively immutable mapping used for nested policy records."""

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
        raise AttributeError("FrozenMapping is immutable")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMapping({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SnapshotFileIdentity:
    """Immutable identity for one selected snapshot file."""

    filename: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, value: object, path: str) -> "SnapshotFileIdentity":
        mapping = _expect_exact_fields(
            value,
            frozenset({"bytes", "filename", "sha256"}),
            path,
        )
        return cls(
            filename=_expect_basename(mapping["filename"], f"{path}.filename"),
            size_bytes=_expect_int(mapping["bytes"], f"{path}.bytes", minimum=1),
            sha256=_expect_sha256(mapping["sha256"], f"{path}.sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes": self.size_bytes,
            "filename": self.filename,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SnapshotProfile:
    """Validated exact repository profile for the selected DINOv3 snapshot."""

    schema_id: str
    snapshot_id: str
    access: str
    weight_format: str
    files: tuple[SnapshotFileIdentity, ...]
    selected_total_bytes: int
    registry_key: str
    model_id: str
    revision: str
    config_class: str
    model_class: str
    expected_parameter_count: int
    expected_parameter_tensor_count: int
    external_root: Mapping[str, Any]
    runtime_id: str
    runtime_profile_sha256: str
    required_persistent_freeze_sha256: str
    load_policy: Mapping[str, Any]
    source_sha256: str = field(default="", repr=False, compare=False)

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        source_sha256: str = "",
    ) -> "SnapshotProfile":
        expected = frozenset(
            {
                "access",
                "external_root",
                "files",
                "load_policy",
                "model",
                "persistent_runtime",
                "schema_id",
                "selected_total_bytes",
                "snapshot_id",
                "weight_format",
            }
        )
        mapping = _expect_exact_fields(value, expected, "profile")
        _scan_sensitive_strings(mapping, "profile")

        schema_id = _expect_identifier(mapping["schema_id"], "profile.schema_id")
        snapshot_id = _expect_identifier(mapping["snapshot_id"], "profile.snapshot_id")
        access = _expect_identifier(mapping["access"], "profile.access")
        weight_format = _expect_identifier(
            mapping["weight_format"], "profile.weight_format"
        )
        _expect_exact(schema_id, SNAPSHOT_PROFILE_SCHEMA_ID, "profile.schema_id")
        _expect_exact(snapshot_id, SNAPSHOT_ID, "profile.snapshot_id")
        _expect_exact(access, ACCESS_CLASSIFICATION, "profile.access")
        _expect_exact(weight_format, WEIGHT_FORMAT, "profile.weight_format")

        file_values = _expect_sequence(mapping["files"], "profile.files")
        files = tuple(
            SnapshotFileIdentity.from_dict(item, f"profile.files[{index}]")
            for index, item in enumerate(file_values)
        )
        actual_file_rows = tuple(
            (item.filename, item.size_bytes, item.sha256) for item in files
        )
        _expect_exact(actual_file_rows, _EXPECTED_FILE_ROWS, "profile.files")
        if len({item.filename for item in files}) != len(files):
            _fail("schema_validation_failed", "profile.files", "duplicate filenames")

        selected_total_bytes = _expect_int(
            mapping["selected_total_bytes"],
            "profile.selected_total_bytes",
            minimum=1,
        )
        _expect_exact(
            selected_total_bytes,
            SELECTED_TOTAL_BYTES,
            "profile.selected_total_bytes",
        )
        if selected_total_bytes != sum(item.size_bytes for item in files):
            _fail(
                "snapshot_identity_mismatch",
                "profile.selected_total_bytes",
                "does not equal the selected file inventory",
            )

        model = _expect_exact_fields(
            mapping["model"],
            frozenset(
                {
                    "config_class",
                    "expected_parameter_count",
                    "expected_parameter_tensor_count",
                    "model_class",
                    "model_id",
                    "registry_key",
                    "revision",
                }
            ),
            "profile.model",
        )
        config_class = _expect_identifier(
            model["config_class"], "profile.model.config_class"
        )
        model_class = _expect_identifier(
            model["model_class"], "profile.model.model_class"
        )
        model_id = _expect_string(model["model_id"], "profile.model.model_id")
        registry_key = _expect_identifier(
            model["registry_key"], "profile.model.registry_key"
        )
        revision = _expect_revision(model["revision"], "profile.model.revision")
        parameter_count = _expect_int(
            model["expected_parameter_count"],
            "profile.model.expected_parameter_count",
            minimum=1,
        )
        parameter_tensor_count = _expect_int(
            model["expected_parameter_tensor_count"],
            "profile.model.expected_parameter_tensor_count",
            minimum=1,
        )
        for actual, required, path in (
            (config_class, CONFIG_CLASS, "profile.model.config_class"),
            (model_class, MODEL_CLASS, "profile.model.model_class"),
            (model_id, MODEL_ID, "profile.model.model_id"),
            (registry_key, REGISTRY_KEY, "profile.model.registry_key"),
            (revision, REVISION, "profile.model.revision"),
            (
                parameter_count,
                EXPECTED_PARAMETER_COUNT,
                "profile.model.expected_parameter_count",
            ),
            (
                parameter_tensor_count,
                EXPECTED_PARAMETER_TENSOR_COUNT,
                "profile.model.expected_parameter_tensor_count",
            ),
        ):
            _expect_exact(actual, required, path)

        external_root = _validate_external_root(mapping["external_root"])
        persistent = _validate_persistent_runtime(mapping["persistent_runtime"])
        load_policy = _validate_load_policy(mapping["load_policy"])
        if source_sha256:
            _expect_sha256(source_sha256, "profile.source_sha256")

        return cls(
            schema_id=schema_id,
            snapshot_id=snapshot_id,
            access=access,
            weight_format=weight_format,
            files=files,
            selected_total_bytes=selected_total_bytes,
            registry_key=registry_key,
            model_id=model_id,
            revision=revision,
            config_class=config_class,
            model_class=model_class,
            expected_parameter_count=parameter_count,
            expected_parameter_tensor_count=parameter_tensor_count,
            external_root=_freeze_json(external_root),
            runtime_id=persistent["runtime_id"],
            runtime_profile_sha256=persistent["profile_sha256"],
            required_persistent_freeze_sha256=persistent[
                "required_freeze_sha256"
            ],
            load_policy=_freeze_json(load_policy),
            source_sha256=source_sha256,
        )

    @property
    def environment_variable(self) -> str:
        return self.external_root["environment_variable"]

    @property
    def relative_root_parts(self) -> tuple[str, ...]:
        return tuple(self.external_root["relative_parts"])

    @property
    def staging_name_prefix(self) -> str:
        return self.external_root["staging_name_prefix"]

    @property
    def config_file(self) -> SnapshotFileIdentity:
        return self.file_by_name("config.json")

    @property
    def preprocessor_file(self) -> SnapshotFileIdentity:
        return self.file_by_name("preprocessor_config.json")

    @property
    def weight_file(self) -> SnapshotFileIdentity:
        return self.file_by_name("model.safetensors")

    def file_by_name(self, filename: str) -> SnapshotFileIdentity:
        for identity in self.files:
            if identity.filename == filename:
                return identity
        _fail("snapshot_identity_mismatch", "profile.files", "required file is absent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "access": self.access,
            "external_root": _thaw_json(self.external_root),
            "files": [item.to_dict() for item in self.files],
            "load_policy": _thaw_json(self.load_policy),
            "model": {
                "config_class": self.config_class,
                "expected_parameter_count": self.expected_parameter_count,
                "expected_parameter_tensor_count": self.expected_parameter_tensor_count,
                "model_class": self.model_class,
                "model_id": self.model_id,
                "registry_key": self.registry_key,
                "revision": self.revision,
            },
            "persistent_runtime": {
                "profile_sha256": self.runtime_profile_sha256,
                "required_freeze_sha256": self.required_persistent_freeze_sha256,
                "runtime_id": self.runtime_id,
            },
            "schema_id": self.schema_id,
            "selected_total_bytes": self.selected_total_bytes,
            "snapshot_id": self.snapshot_id,
            "weight_format": self.weight_format,
        }


def _validate_external_root(value: object) -> dict[str, Any]:
    path = "profile.external_root"
    mapping = _expect_exact_fields(value, frozenset(_EXPECTED_EXTERNAL_ROOT), path)
    result = {
        "default_template": _expect_string(
            mapping["default_template"], f"{path}.default_template"
        ),
        "environment_variable": _expect_identifier(
            mapping["environment_variable"], f"{path}.environment_variable"
        ),
        "persistent_runtime_relation": _expect_identifier(
            mapping["persistent_runtime_relation"],
            f"{path}.persistent_runtime_relation",
        ),
        "relative_parts": [
            _expect_basename(item, f"{path}.relative_parts[{index}]")
            for index, item in enumerate(
                _expect_sequence(mapping["relative_parts"], f"{path}.relative_parts")
            )
        ],
        "repository_relation": _expect_identifier(
            mapping["repository_relation"], f"{path}.repository_relation"
        ),
        "staging_name_prefix": _expect_basename(
            mapping["staging_name_prefix"], f"{path}.staging_name_prefix"
        ),
    }
    _expect_exact(result, _EXPECTED_EXTERNAL_ROOT, path)
    return result


def _validate_persistent_runtime(value: object) -> dict[str, str]:
    path = "profile.persistent_runtime"
    mapping = _expect_exact_fields(
        value,
        frozenset({"profile_sha256", "required_freeze_sha256", "runtime_id"}),
        path,
    )
    result = {
        "profile_sha256": _expect_sha256(
            mapping["profile_sha256"], f"{path}.profile_sha256"
        ),
        "required_freeze_sha256": _expect_sha256(
            mapping["required_freeze_sha256"], f"{path}.required_freeze_sha256"
        ),
        "runtime_id": _expect_identifier(mapping["runtime_id"], f"{path}.runtime_id"),
    }
    expected = {
        "profile_sha256": PERSISTENT_RUNTIME_PROFILE_SHA256,
        "required_freeze_sha256": REQUIRED_PERSISTENT_FREEZE_SHA256,
        "runtime_id": PERSISTENT_RUNTIME_ID,
    }
    _expect_exact(result, expected, path)
    return result


def _validate_load_policy(value: object) -> dict[str, Any]:
    path = "profile.load_policy"
    mapping = _expect_exact_fields(value, frozenset(_EXPECTED_LOAD_POLICY), path)
    result: dict[str, Any] = {}
    for key, expected in _EXPECTED_LOAD_POLICY.items():
        item = mapping[key]
        if isinstance(expected, bool):
            result[key] = _expect_bool(item, f"{path}.{key}")
        else:
            result[key] = _expect_identifier(item, f"{path}.{key}")
    _expect_exact(result, _EXPECTED_LOAD_POLICY, path)
    return result


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Validated external snapshot manifest without machine-specific paths."""

    schema_id: str
    snapshot_id: str
    repository_profile_sha256: str
    model_id: str
    revision: str
    weight_format: str
    files: tuple[SnapshotFileIdentity, ...]
    total_bytes: int
    provisioned_at_utc: str
    credential_backed_logical_request_count: int
    model_weight_transferred_bytes: int
    integrity_status: str
    network_independent_verification_status: str
    source_sha256: str = field(default="", repr=False, compare=False)

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        source_sha256: str = "",
    ) -> "SnapshotManifest":
        expected = frozenset(
            {
                "credential_backed_logical_request_count",
                "files",
                "integrity_status",
                "model_id",
                "model_weight_transferred_bytes",
                "network_independent_verification_status",
                "provisioned_at_utc",
                "repository_profile_sha256",
                "revision",
                "schema_id",
                "snapshot_id",
                "total_bytes",
                "weight_format",
            }
        )
        mapping = _expect_exact_fields(value, expected, "manifest")
        _scan_sensitive_strings(mapping, "manifest")

        schema_id = _expect_identifier(mapping["schema_id"], "manifest.schema_id")
        snapshot_id = _expect_identifier(
            mapping["snapshot_id"], "manifest.snapshot_id"
        )
        model_id = _expect_string(mapping["model_id"], "manifest.model_id")
        revision = _expect_revision(mapping["revision"], "manifest.revision")
        weight_format = _expect_identifier(
            mapping["weight_format"], "manifest.weight_format"
        )
        for actual, required, path in (
            (schema_id, SNAPSHOT_MANIFEST_SCHEMA_ID, "manifest.schema_id"),
            (snapshot_id, SNAPSHOT_ID, "manifest.snapshot_id"),
            (model_id, MODEL_ID, "manifest.model_id"),
            (revision, REVISION, "manifest.revision"),
            (weight_format, WEIGHT_FORMAT, "manifest.weight_format"),
        ):
            _expect_exact(actual, required, path)

        files = tuple(
            SnapshotFileIdentity.from_dict(item, f"manifest.files[{index}]")
            for index, item in enumerate(
                _expect_sequence(mapping["files"], "manifest.files")
            )
        )
        names = tuple(item.filename for item in files)
        expected_names = tuple(row[0] for row in _EXPECTED_FILE_ROWS)
        _expect_exact(names, expected_names, "manifest.files")
        if len(set(names)) != len(names):
            _fail("schema_validation_failed", "manifest.files", "duplicate filenames")

        total_bytes = _expect_int(
            mapping["total_bytes"], "manifest.total_bytes", minimum=1
        )
        if total_bytes != sum(item.size_bytes for item in files):
            _fail(
                "snapshot_identity_mismatch",
                "manifest.total_bytes",
                "does not equal the file inventory",
            )
        provisioned_at_utc = _expect_utc_timestamp(
            mapping["provisioned_at_utc"], "manifest.provisioned_at_utc"
        )
        request_count = _expect_int(
            mapping["credential_backed_logical_request_count"],
            "manifest.credential_backed_logical_request_count",
            maximum=16,
        )
        transferred_bytes = _expect_int(
            mapping["model_weight_transferred_bytes"],
            "manifest.model_weight_transferred_bytes",
            maximum=400 * 1024 * 1024,
        )
        integrity_status = _expect_identifier(
            mapping["integrity_status"], "manifest.integrity_status"
        )
        verification_status = _expect_identifier(
            mapping["network_independent_verification_status"],
            "manifest.network_independent_verification_status",
        )
        _expect_exact(integrity_status, "VERIFIED", "manifest.integrity_status")
        _expect_exact(
            verification_status,
            "VERIFIED",
            "manifest.network_independent_verification_status",
        )
        repository_profile_sha256 = _expect_sha256(
            mapping["repository_profile_sha256"],
            "manifest.repository_profile_sha256",
        )
        if source_sha256:
            _expect_sha256(source_sha256, "manifest.source_sha256")

        return cls(
            schema_id=schema_id,
            snapshot_id=snapshot_id,
            repository_profile_sha256=repository_profile_sha256,
            model_id=model_id,
            revision=revision,
            weight_format=weight_format,
            files=files,
            total_bytes=total_bytes,
            provisioned_at_utc=provisioned_at_utc,
            credential_backed_logical_request_count=request_count,
            model_weight_transferred_bytes=transferred_bytes,
            integrity_status=integrity_status,
            network_independent_verification_status=verification_status,
            source_sha256=source_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "credential_backed_logical_request_count": self.credential_backed_logical_request_count,
            "files": [item.to_dict() for item in self.files],
            "integrity_status": self.integrity_status,
            "model_id": self.model_id,
            "model_weight_transferred_bytes": self.model_weight_transferred_bytes,
            "network_independent_verification_status": self.network_independent_verification_status,
            "provisioned_at_utc": self.provisioned_at_utc,
            "repository_profile_sha256": self.repository_profile_sha256,
            "revision": self.revision,
            "schema_id": self.schema_id,
            "snapshot_id": self.snapshot_id,
            "total_bytes": self.total_bytes,
            "weight_format": self.weight_format,
        }


def _expect_utc_timestamp(value: object, path: str) -> str:
    timestamp = _expect_string(value, path, maximum=64)
    if not timestamp.endswith("Z"):
        _fail("schema_validation_failed", path, "expected a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise SnapshotBoundaryError(
            "schema_validation_failed", f"{path}: invalid ISO-8601 timestamp"
        ) from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        _fail("schema_validation_failed", path, "expected UTC")
    return timestamp


def load_snapshot_profile(path: str | Path) -> SnapshotProfile:
    """Load and strictly validate the exact repository snapshot profile."""

    _, payload = _read_bounded_regular_file(
        path,
        maximum_bytes=_MAX_PROFILE_BYTES,
        context="profile",
    )
    digest = hashlib.sha256(payload).hexdigest()
    value = _parse_json_bytes(payload, context="profile")
    return SnapshotProfile.from_dict(value, source_sha256=digest)


def load_snapshot_manifest(path: str | Path) -> SnapshotManifest:
    """Load and strictly validate one external snapshot manifest."""

    _, payload = _read_bounded_regular_file(
        path,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        context="manifest",
    )
    digest = hashlib.sha256(payload).hexdigest()
    value = _parse_json_bytes(payload, context="manifest")
    return SnapshotManifest.from_dict(value, source_sha256=digest)


def validate_snapshot_manifest(
    manifest: SnapshotManifest,
    profile: SnapshotProfile,
) -> None:
    """Cross-check an external manifest against its repository profile."""

    if not isinstance(manifest, SnapshotManifest) or not isinstance(
        profile, SnapshotProfile
    ):
        _fail("schema_validation_failed", "manifest", "unexpected object type")
    checks = (
        (manifest.snapshot_id, profile.snapshot_id, "snapshot_id"),
        (
            manifest.repository_profile_sha256,
            profile.source_sha256,
            "repository_profile_sha256",
        ),
        (manifest.model_id, profile.model_id, "model_id"),
        (manifest.revision, profile.revision, "revision"),
        (manifest.weight_format, profile.weight_format, "weight_format"),
        (manifest.files, profile.files, "files"),
        (manifest.total_bytes, profile.selected_total_bytes, "total_bytes"),
        (
            manifest.model_weight_transferred_bytes,
            profile.weight_file.size_bytes,
            "model_weight_transferred_bytes",
        ),
        (
            manifest.credential_backed_logical_request_count,
            2,
            "credential_backed_logical_request_count",
        ),
    )
    for actual, expected, field_name in checks:
        if actual != expected:
            _fail(
                "snapshot_identity_mismatch",
                f"manifest.{field_name}",
                "does not match the repository profile",
            )


def resolve_default_snapshot_root(
    profile: SnapshotProfile,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the fixed LOCALAPPDATA snapshot root without a home fallback."""

    if not isinstance(profile, SnapshotProfile):
        _fail("schema_validation_failed", "profile", "expected SnapshotProfile")
    environment = os.environ if environ is None else environ
    local_app_data = environment.get(profile.environment_variable)
    if not isinstance(local_app_data, str) or not local_app_data:
        _fail("snapshot_root_unavailable", profile.environment_variable, "is not set")
    base = Path(local_app_data)
    if not base.is_absolute():
        _fail("unsafe_path", profile.environment_variable, "expected an absolute path")
    root = base.joinpath(*profile.relative_root_parts)
    if root.name != profile.snapshot_id:
        _fail("unsafe_snapshot_root", "snapshot_root", "unexpected root basename")
    return root


def _resolved_directory(path: str | Path, context: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        _fail("unsafe_path", context, "expected an absolute path")
    if _is_reparse_point(candidate):
        _fail("unsafe_path", context, "symbolic links and reparse points are forbidden")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SnapshotBoundaryError(
            "path_validation_failed", f"{context}: directory does not exist"
        ) from exc
    if not resolved.is_dir():
        _fail("unsafe_path", context, "expected a directory")
    return resolved


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_overlapping_roots(
    snapshot_root: Path,
    repository_root: Path,
    persistent_runtime_root: Path,
) -> None:
    roots = (
        ("snapshot_root", snapshot_root),
        ("repository_root", repository_root),
        ("persistent_runtime_root", persistent_runtime_root),
    )
    for index, (left_name, left) in enumerate(roots):
        for right_name, right in roots[index + 1 :]:
            if _contains(left, right) or _contains(right, left):
                _fail(
                    "unsafe_snapshot_root",
                    "roots",
                    f"{left_name} and {right_name} must not overlap",
                )


def _safe_existing_child(
    root: Path,
    relative_path: str,
    *,
    context: str,
    expect_directory: bool = False,
) -> Path:
    posix = PurePosixPath(relative_path)
    windows = PureWindowsPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in relative_path.split("/"))
    ):
        _fail("unsafe_path", context, "expected a safe relative path")
    current = root
    for part in relative_path.split("/"):
        current = current / part
        if not current.exists() and not current.is_symlink():
            _fail("path_validation_failed", context, "required path does not exist")
        if _is_reparse_point(current):
            _fail(
                "unsafe_path",
                context,
                "symbolic links and reparse points are forbidden",
            )
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise SnapshotBoundaryError(
            "path_validation_failed", f"{context}: unable to resolve path"
        ) from exc
    if not _contains(root, resolved):
        _fail("unsafe_path", context, "resolved path escapes the snapshot root")
    if expect_directory:
        if not resolved.is_dir():
            _fail("unsafe_path", context, "expected a directory")
    elif not resolved.is_file():
        _fail("unsafe_path", context, "expected a regular file")
    return resolved


def _validate_exact_directory_entries(
    directory: Path,
    expected: set[str],
    context: str,
) -> None:
    try:
        entries = tuple(directory.iterdir())
    except OSError as exc:
        raise SnapshotBoundaryError(
            "path_validation_failed", f"{context}: unable to list directory"
        ) from exc
    for entry in entries:
        if _is_reparse_point(entry):
            _fail(
                "unsafe_path",
                context,
                "symbolic links and reparse points are forbidden",
            )
    actual = {entry.name for entry in entries}
    if actual != expected:
        _fail(
            "snapshot_layout_mismatch",
            context,
            "directory entries do not match the exact snapshot layout",
        )


def _verify_file_identity(
    path: Path,
    identity: SnapshotFileIdentity,
    context: str,
) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SnapshotBoundaryError(
            "snapshot_integrity_failed", f"{context}: unable to inspect file"
        ) from exc
    if size != identity.size_bytes:
        _fail(
            "snapshot_integrity_failed",
            context,
            "file size does not match the snapshot profile",
        )
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, identity.sha256):
        _fail(
            "snapshot_integrity_failed",
            context,
            "SHA-256 does not match the snapshot profile",
        )


def snapshot_files_directory(snapshot_root: str | Path) -> Path:
    """Return an existing, non-link ``files`` directory without modifying it."""

    root = _resolved_directory(snapshot_root, "snapshot_root")
    return _safe_existing_child(
        root,
        "files",
        context="snapshot.files",
        expect_directory=True,
    )


def validate_snapshot_root(
    snapshot_root: str | Path,
    profile: SnapshotProfile,
    *,
    repository_root: str | Path,
    persistent_runtime_root: str | Path,
    _allow_owned_staging: bool = False,
) -> SnapshotManifest:
    """Read-only verification of exact snapshot layout, manifest, and bytes."""

    if not isinstance(profile, SnapshotProfile):
        _fail("schema_validation_failed", "profile", "expected SnapshotProfile")
    root = _resolved_directory(snapshot_root, "snapshot_root")
    repository = _resolved_directory(repository_root, "repository_root")
    persistent_runtime = _resolved_directory(
        persistent_runtime_root, "persistent_runtime_root"
    )
    _reject_overlapping_roots(root, repository, persistent_runtime)

    if _allow_owned_staging:
        suffix = root.name.removeprefix(profile.staging_name_prefix)
        if (
            not root.name.startswith(profile.staging_name_prefix)
            or not suffix
            or _IDENTIFIER_PATTERN.fullmatch(suffix) is None
        ):
            _fail(
                "snapshot_identity_mismatch",
                "snapshot_root",
                "basename is not a recognized task-owned staging root",
            )
    elif root.name != profile.snapshot_id:
        _fail(
            "snapshot_identity_mismatch",
            "snapshot_root",
            "basename does not match the snapshot ID",
        )

    _validate_exact_directory_entries(
        root,
        {"files", "snapshot-manifest.json"},
        "snapshot_root",
    )
    files_directory = _safe_existing_child(
        root,
        "files",
        context="snapshot.files",
        expect_directory=True,
    )
    _validate_exact_directory_entries(
        files_directory,
        {item.filename for item in profile.files},
        "snapshot.files",
    )
    manifest_path = _safe_existing_child(
        root,
        "snapshot-manifest.json",
        context="snapshot.manifest",
    )
    manifest = load_snapshot_manifest(manifest_path)
    validate_snapshot_manifest(manifest, profile)
    for identity in profile.files:
        file_path = _safe_existing_child(
            root,
            f"files/{identity.filename}",
            context=f"snapshot.files.{identity.filename}",
        )
        _verify_file_identity(
            file_path,
            identity,
            f"snapshot.files.{identity.filename}",
        )
    if not profile.weight_file.filename.endswith(".safetensors"):
        _fail(
            "snapshot_identity_mismatch",
            "profile.files",
            "model weight must be a Safetensors file",
        )
    return manifest


__all__ = [
    "ACCESS_CLASSIFICATION",
    "CONFIG_CLASS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_PARAMETER_TENSOR_COUNT",
    "MODEL_CLASS",
    "MODEL_ID",
    "PERSISTENT_RUNTIME_ID",
    "PERSISTENT_RUNTIME_PROFILE_SHA256",
    "REGISTRY_KEY",
    "REQUIRED_PERSISTENT_FREEZE_SHA256",
    "REVISION",
    "SELECTED_TOTAL_BYTES",
    "SNAPSHOT_ID",
    "SNAPSHOT_MANIFEST_SCHEMA_ID",
    "SNAPSHOT_PROFILE_SCHEMA_ID",
    "WEIGHT_FORMAT",
    "SnapshotBoundaryError",
    "SnapshotFileIdentity",
    "SnapshotManifest",
    "SnapshotProfile",
    "canonical_json_bytes",
    "load_snapshot_manifest",
    "load_snapshot_profile",
    "resolve_default_snapshot_root",
    "sha256_file",
    "snapshot_files_directory",
    "validate_snapshot_manifest",
    "validate_snapshot_root",
]
