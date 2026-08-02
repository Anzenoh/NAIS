"""Standard-library process boundary for isolated encoder runtimes.

This module intentionally stays import-light.  It validates the versioned JSON
contracts and filesystem boundary used to launch an isolated worker, but it does
not import any numeric or model runtime package.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any


RUNTIME_PROFILE_SCHEMA_ID = "vea.dinov3_isolated_runtime_profile.v0"
RUNTIME_MANIFEST_SCHEMA_ID = "vea.dinov3_persistent_runtime.v0"
WORKER_REQUEST_SCHEMA_ID = "vea.dinov3_isolated_worker_request.v0"
WORKER_RESPONSE_SCHEMA_ID = "vea.dinov3_isolated_worker_response.v0"

ALLOWED_WORKER_OPERATIONS = frozenset(
    {"artifact_boundary_self_test", "runtime_self_test"}
)
ALLOWED_RESPONSE_STATUSES = frozenset({"success", "error"})
_ARTIFACT_MARKERS = (
    "DINOV3_PERSISTENT_RUNTIME_MANIFEST_OK",
    "DINOV3_PERSISTENT_INBOUND_ARTIFACT_OK",
    "DINOV3_PERSISTENT_OUTBOUND_ARTIFACT_OK",
    "DINOV3_PERSISTENT_RUNTIME_NETWORK_GUARD_OK",
)
_RUNTIME_MARKERS = _ARTIFACT_MARKERS + (
    "DINOV3_PERSISTENT_MANUAL_PREPROCESSING_OK",
    "DINOV3_PERSISTENT_RANDOM_FORWARD_OK",
)

_MAX_PROFILE_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_PACKAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_EXACT_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9]+(?:[A-Za-z0-9.+-]*[A-Za-z0-9])?$")
_MUTABLE_VERSION_PATTERN = re.compile(r"[<>=!~,*\s]")
_EMAIL_PATTERN = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
)
_TOKEN_VALUE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+\S+|hf_[A-Za-z0-9]{10,}|"
    r"(?:token|secret|password|passwd|credential|api[_-]?key)\s*[:=])"
)
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
_REQUIRED_FORBIDDEN_PACKAGES = frozenset(
    {
        "directml",
        "nvidia-*",
        "opencv-python",
        "torch",
        "torchaudio",
        "torchvision",
        "triton",
    }
)
_REQUIRED_RUNTIME_LAYOUT = frozenset(
    {
        "runtime-manifest.json",
        "metadata/config.json",
        "metadata/preprocessor_config.json",
        "wheelhouse",
        "venv/Scripts/python.exe",
    }
)


class RuntimeBoundaryError(RuntimeError):
    """Raised when an isolated-runtime boundary contract is violated."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def _fail(classification: str, path: str, message: str) -> None:
    raise RuntimeBoundaryError(classification, f"{path}: {message}")


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
    except RuntimeBoundaryError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeBoundaryError(
            "invalid_json", f"{context}: invalid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        _fail("invalid_json", context, "expected a JSON object")
    return value


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
        raise RuntimeBoundaryError(
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
        raise RuntimeBoundaryError(
            "path_validation_failed", f"{context}: file is not readable"
        ) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        _fail("unsafe_path", context, "expected a regular file")
    if file_stat.st_size > maximum_bytes:
        _fail("invalid_json", context, f"file exceeds {maximum_bytes} bytes")
    try:
        return resolved, resolved.read_bytes()
    except OSError as exc:
        raise RuntimeBoundaryError(
            "path_validation_failed", f"{context}: file is not readable"
        ) from exc


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise RuntimeBoundaryError(
            "invalid_json", "json: value is not finite JSON data"
        ) from exc
    return text.encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeBoundaryError(
            "path_validation_failed", "unable to hash a required file"
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


def _expect_git_sha(value: object, path: str) -> str:
    revision = _expect_string(value, path, maximum=40)
    if _GIT_SHA_PATTERN.fullmatch(revision) is None:
        _fail("schema_validation_failed", path, "expected a lowercase 40-hex revision")
    return revision


def _expect_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail("schema_validation_failed", path, "expected a boolean")
    return value


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


def _expect_float(value: object, path: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail("schema_validation_failed", path, "expected a finite number")
    return float(value)


def _expect_sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail("schema_validation_failed", path, "expected an array")
    return value


def _expect_exact_version(value: object, path: str) -> str:
    version = _expect_string(value, path, maximum=64)
    if (
        _MUTABLE_VERSION_PATTERN.search(version) is not None
        or _EXACT_VERSION_PATTERN.fullmatch(version) is None
    ):
        _fail("schema_validation_failed", path, "expected one exact package version")
    return version


def _expect_basename(value: object, path: str) -> str:
    name = _expect_string(value, path, maximum=255)
    if (
        PurePosixPath(name).name != name
        or PureWindowsPath(name).name != name
        or PureWindowsPath(name).drive
    ):
        _fail("unsafe_path", path, "expected a basename")
    return name


def _expect_relative_path(value: object, path: str) -> str:
    relative = _expect_string(value, path, maximum=512)
    if "\\" in relative:
        _fail("unsafe_path", path, "backslashes are not allowed in protocol paths")
    windows = PureWindowsPath(relative)
    posix = PurePosixPath(relative)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        _fail("unsafe_path", path, "absolute paths are forbidden")
    raw_parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        _fail("unsafe_path", path, "path traversal or empty components are forbidden")
    if any(":" in part or "\x00" in part for part in raw_parts):
        _fail("unsafe_path", path, "unsafe path components are forbidden")
    return relative


def _expect_string_tuple(
    value: object,
    path: str,
    *,
    identifiers: bool = False,
) -> tuple[str, ...]:
    sequence = _expect_sequence(value, path)
    converted = tuple(
        (_expect_identifier(item, f"{path}[{index}]") if identifiers else _expect_string(item, f"{path}[{index}]"))
        for index, item in enumerate(sequence)
    )
    if len(set(converted)) != len(converted):
        _fail("schema_validation_failed", path, "duplicate values are forbidden")
    return converted


def _reject_sensitive_string(value: str, path: str) -> None:
    if _EMAIL_PATTERN.search(value) is not None:
        _fail("sensitive_data_rejected", path, "email-like values are forbidden")
    if _TOKEN_VALUE_PATTERN.search(value) is not None:
        _fail("sensitive_data_rejected", path, "secret-like values are forbidden")
    windows = PureWindowsPath(value)
    if PurePosixPath(value).is_absolute() or windows.is_absolute() or windows.drive:
        _fail("sensitive_data_rejected", path, "absolute paths are forbidden")


def _scan_response_strings(value: object, path: str = "response") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _scan_response_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_response_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        _reject_sensitive_string(value, path)


class FrozenMapping(Mapping[str, Any]):
    """Recursively immutable mapping with read-only attribute access."""

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
class FileIdentity:
    """Immutable identity for one exact retained file."""

    filename: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, value: object, path: str) -> "FileIdentity":
        mapping = _expect_exact_fields(
            value, frozenset({"filename", "size_bytes", "sha256"}), path
        )
        return cls(
            filename=_expect_basename(mapping["filename"], f"{path}.filename"),
            size_bytes=_expect_int(
                mapping["size_bytes"], f"{path}.size_bytes", minimum=1
            ),
            sha256=_expect_sha256(mapping["sha256"], f"{path}.sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Immutable identity for one installed source file."""

    relative_path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: object, path: str) -> "SourceIdentity":
        mapping = _expect_exact_fields(
            value, frozenset({"relative_path", "sha256"}), path
        )
        return cls(
            relative_path=_expect_relative_path(
                mapping["relative_path"], f"{path}.relative_path"
            ),
            sha256=_expect_sha256(mapping["sha256"], f"{path}.sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"relative_path": self.relative_path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class InstalledSourceIdentity:
    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, value: object, path: str) -> "InstalledSourceIdentity":
        mapping = _expect_exact_fields(
            value, frozenset({"bytes", "path", "sha256"}), path
        )
        return cls(
            path=_expect_relative_path(mapping["path"], f"{path}.path"),
            size_bytes=_expect_int(mapping["bytes"], f"{path}.bytes", minimum=1),
            sha256=_expect_sha256(mapping["sha256"], f"{path}.sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"bytes": self.size_bytes, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class WheelIdentity:
    distribution: str
    version: str
    filename: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, value: object, path: str) -> "WheelIdentity":
        mapping = _expect_exact_fields(
            value,
            frozenset({"bytes", "distribution", "filename", "sha256", "version"}),
            path,
        )
        distribution = _expect_string(
            mapping["distribution"], f"{path}.distribution", maximum=128
        )
        if _PACKAGE_NAME_PATTERN.fullmatch(distribution) is None:
            _fail(
                "schema_validation_failed",
                f"{path}.distribution",
                "invalid package name",
            )
        return cls(
            distribution=distribution,
            version=_expect_exact_version(mapping["version"], f"{path}.version"),
            filename=_expect_basename(mapping["filename"], f"{path}.filename"),
            size_bytes=_expect_int(mapping["bytes"], f"{path}.bytes", minimum=1),
            sha256=_expect_sha256(mapping["sha256"], f"{path}.sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes": self.size_bytes,
            "distribution": self.distribution,
            "filename": self.filename,
            "sha256": self.sha256,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class PackageIdentity:
    """Immutable exact package name and version."""

    name: str
    version: str

    @classmethod
    def from_dict(cls, value: object, path: str) -> "PackageIdentity":
        mapping = _expect_exact_fields(value, frozenset({"name", "version"}), path)
        name = _expect_string(mapping["name"], f"{path}.name", maximum=128)
        if _PACKAGE_NAME_PATTERN.fullmatch(name) is None:
            _fail("schema_validation_failed", f"{path}.name", "invalid package name")
        return cls(
            name=name,
            version=_expect_exact_version(mapping["version"], f"{path}.version"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class PythonProfile:
    major: int
    minor: int

    @classmethod
    def from_dict(cls, value: object, path: str) -> "PythonProfile":
        mapping = _expect_exact_fields(value, frozenset({"major", "minor"}), path)
        return cls(
            major=_expect_int(mapping["major"], f"{path}.major", minimum=3),
            minor=_expect_int(mapping["minor"], f"{path}.minor", minimum=0),
        )

    def to_dict(self) -> dict[str, int]:
        return {"major": self.major, "minor": self.minor}


@dataclass(frozen=True, slots=True)
class SharedTorchProfile:
    metadata_version: str
    runtime_version: str

    @classmethod
    def from_dict(cls, value: object, path: str) -> "SharedTorchProfile":
        mapping = _expect_exact_fields(
            value, frozenset({"metadata_version", "runtime_version"}), path
        )
        return cls(
            metadata_version=_expect_exact_version(
                mapping["metadata_version"], f"{path}.metadata_version"
            ),
            runtime_version=_expect_exact_version(
                mapping["runtime_version"], f"{path}.runtime_version"
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "metadata_version": self.metadata_version,
            "runtime_version": self.runtime_version,
        }


@dataclass(frozen=True, slots=True)
class CanonicalInputProfile:
    color_mode: str
    dtype: str
    layout: str
    height: int
    width: int
    channels: int

    @classmethod
    def from_dict(cls, value: object, path: str) -> "CanonicalInputProfile":
        expected = frozenset(
            {"color_mode", "dtype", "layout", "height", "width", "channels"}
        )
        mapping = _expect_exact_fields(value, expected, path)
        profile = cls(
            color_mode=_expect_string(mapping["color_mode"], f"{path}.color_mode"),
            dtype=_expect_string(mapping["dtype"], f"{path}.dtype"),
            layout=_expect_string(mapping["layout"], f"{path}.layout"),
            height=_expect_int(mapping["height"], f"{path}.height", minimum=1),
            width=_expect_int(mapping["width"], f"{path}.width", minimum=1),
            channels=_expect_int(mapping["channels"], f"{path}.channels", minimum=1),
        )
        if profile != cls("RGB", "uint8", "NHWC", 224, 224, 3):
            _fail(
                "schema_validation_failed",
                path,
                "expected canonical RGB uint8 NHWC 224x224x3 input",
            )
        return profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "color_mode": self.color_mode,
            "dtype": self.dtype,
            "height": self.height,
            "layout": self.layout,
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class ManualPreprocessingProfile:
    profile_id: str
    operation_order: tuple[str, ...]
    rescale_factor: float
    rescale_factor_dtype: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    output_layout: str
    output_dtype: str
    resize: bool
    crop: bool
    pil_conversion: bool

    @classmethod
    def from_dict(cls, value: object, path: str) -> "ManualPreprocessingProfile":
        expected = frozenset(
            {
                "profile_id",
                "operation_order",
                "rescale_factor",
                "rescale_factor_dtype",
                "mean",
                "std",
                "output_layout",
                "output_dtype",
                "resize",
                "crop",
                "pil_conversion",
            }
        )
        mapping = _expect_exact_fields(value, expected, path)
        operations = _expect_string_tuple(
            mapping["operation_order"], f"{path}.operation_order", identifiers=True
        )
        if not operations:
            _fail("schema_validation_failed", f"{path}.operation_order", "cannot be empty")
        mean_raw = _expect_sequence(mapping["mean"], f"{path}.mean")
        std_raw = _expect_sequence(mapping["std"], f"{path}.std")
        if len(mean_raw) != 3 or len(std_raw) != 3:
            _fail("schema_validation_failed", path, "mean and std must have three values")
        mean = tuple(
            _expect_float(item, f"{path}.mean[{index}]")
            for index, item in enumerate(mean_raw)
        )
        std = tuple(
            _expect_float(item, f"{path}.std[{index}]")
            for index, item in enumerate(std_raw)
        )
        if any(item <= 0.0 for item in std):
            _fail("schema_validation_failed", f"{path}.std", "values must be positive")
        profile = cls(
            profile_id=_expect_identifier(mapping["profile_id"], f"{path}.profile_id"),
            operation_order=operations,
            rescale_factor=_expect_float(
                mapping["rescale_factor"], f"{path}.rescale_factor"
            ),
            rescale_factor_dtype=_expect_string(
                mapping["rescale_factor_dtype"], f"{path}.rescale_factor_dtype"
            ),
            mean=(mean[0], mean[1], mean[2]),
            std=(std[0], std[1], std[2]),
            output_layout=_expect_string(
                mapping["output_layout"], f"{path}.output_layout"
            ),
            output_dtype=_expect_string(
                mapping["output_dtype"], f"{path}.output_dtype"
            ),
            resize=_expect_bool(mapping["resize"], f"{path}.resize"),
            crop=_expect_bool(mapping["crop"], f"{path}.crop"),
            pil_conversion=_expect_bool(
                mapping["pil_conversion"], f"{path}.pil_conversion"
            ),
        )
        if not math.isclose(profile.rescale_factor, 1.0 / 255.0, rel_tol=0.0, abs_tol=1e-18):
            _fail("schema_validation_failed", f"{path}.rescale_factor", "expected 1/255")
        if profile.rescale_factor_dtype != "float32":
            _fail("schema_validation_failed", f"{path}.rescale_factor_dtype", "expected float32")
        if profile.mean != (0.485, 0.456, 0.406):
            _fail("schema_validation_failed", f"{path}.mean", "unexpected ImageNet mean")
        if profile.std != (0.229, 0.224, 0.225):
            _fail("schema_validation_failed", f"{path}.std", "unexpected ImageNet std")
        if profile.output_layout != "NCHW" or profile.output_dtype != "float32":
            _fail("schema_validation_failed", path, "expected NCHW float32 output")
        if profile.resize or profile.crop or profile.pil_conversion:
            _fail("schema_validation_failed", path, "geometry and PIL operations must be disabled")
        return profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop": self.crop,
            "mean": list(self.mean),
            "operation_order": list(self.operation_order),
            "output_dtype": self.output_dtype,
            "output_layout": self.output_layout,
            "pil_conversion": self.pil_conversion,
            "profile_id": self.profile_id,
            "rescale_factor": self.rescale_factor,
            "rescale_factor_dtype": self.rescale_factor_dtype,
            "resize": self.resize,
            "std": list(self.std),
        }


@dataclass(frozen=True, slots=True)
class WorkerProtocolProfile:
    request_schema_id: str
    response_schema_id: str
    allowed_operations: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object, path: str) -> "WorkerProtocolProfile":
        expected = frozenset(
            {"request_schema_id", "response_schema_id", "allowed_operations"}
        )
        mapping = _expect_exact_fields(value, expected, path)
        operations = _expect_string_tuple(
            mapping["allowed_operations"], f"{path}.allowed_operations", identifiers=True
        )
        profile = cls(
            request_schema_id=_expect_string(
                mapping["request_schema_id"], f"{path}.request_schema_id"
            ),
            response_schema_id=_expect_string(
                mapping["response_schema_id"], f"{path}.response_schema_id"
            ),
            allowed_operations=operations,
        )
        if profile.request_schema_id != WORKER_REQUEST_SCHEMA_ID:
            _fail("schema_validation_failed", f"{path}.request_schema_id", "unexpected schema")
        if profile.response_schema_id != WORKER_RESPONSE_SCHEMA_ID:
            _fail("schema_validation_failed", f"{path}.response_schema_id", "unexpected schema")
        if frozenset(profile.allowed_operations) != ALLOWED_WORKER_OPERATIONS:
            _fail("schema_validation_failed", f"{path}.allowed_operations", "unexpected operations")
        return profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_operations": list(self.allowed_operations),
            "request_schema_id": self.request_schema_id,
            "response_schema_id": self.response_schema_id,
        }


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Fully validated immutable repository runtime profile."""

    schema_id: str
    runtime_id: str
    allowed_worker_operations: tuple[str, ...]
    candidate: Mapping[str, Any]
    canonical_input: Mapping[str, Any]
    current_base: Mapping[str, Any]
    dependency_validation: Mapping[str, Any]
    external_layout: Mapping[str, Any]
    forbidden_packages: tuple[str, ...]
    installed_sources: tuple[InstalledSourceIdentity, ...]
    manual_preprocessing: Mapping[str, Any]
    metadata_files: tuple[FileIdentity, ...]
    model_weights: Mapping[str, Any]
    policies: Mapping[str, Any]
    runtime_root: Mapping[str, Any]
    target: Mapping[str, Any]
    wheel_files: tuple[WheelIdentity, ...]
    worker_protocol: Mapping[str, Any]
    source_sha256: str = field(default="", repr=False, compare=False)

    @classmethod
    def from_dict(
        cls, value: object, *, source_sha256: str = ""
    ) -> "RuntimeProfile":
        expected = frozenset(
            {
                "allowed_worker_operations",
                "candidate",
                "canonical_input",
                "current_base",
                "dependency_validation",
                "external_layout",
                "forbidden_packages",
                "installed_sources",
                "manual_preprocessing",
                "metadata_files",
                "model_weights",
                "policies",
                "runtime_id",
                "runtime_root",
                "schema_id",
                "target",
                "wheel_files",
                "worker_protocol",
            }
        )
        mapping = _expect_exact_fields(value, expected, "profile")
        schema_id = _expect_string(mapping["schema_id"], "profile.schema_id")
        if schema_id != RUNTIME_PROFILE_SCHEMA_ID:
            _fail("schema_validation_failed", "profile.schema_id", "unexpected schema")
        runtime_id = _expect_identifier(mapping["runtime_id"], "profile.runtime_id")
        operations = _expect_string_tuple(
            mapping["allowed_worker_operations"],
            "profile.allowed_worker_operations",
            identifiers=True,
        )
        if frozenset(operations) != ALLOWED_WORKER_OPERATIONS:
            _fail(
                "schema_validation_failed",
                "profile.allowed_worker_operations",
                "unexpected operations",
            )
        cls._validate_candidate(mapping["candidate"])
        cls._validate_canonical_input(mapping["canonical_input"])
        cls._validate_current_base(mapping["current_base"])
        cls._validate_dependency(mapping["dependency_validation"])
        cls._validate_external_layout(mapping["external_layout"])
        cls._validate_manual_preprocessing(mapping["manual_preprocessing"])
        cls._validate_model_weights(mapping["model_weights"])
        cls._validate_policies(mapping["policies"])
        cls._validate_runtime_root_contract(mapping["runtime_root"], runtime_id)
        cls._validate_target(mapping["target"])
        cls._validate_worker_protocol(mapping["worker_protocol"])
        metadata = tuple(
            cls._metadata_identity(item, f"profile.metadata_files[{index}]")
            for index, item in enumerate(
                _expect_sequence(mapping["metadata_files"], "profile.metadata_files")
            )
        )
        wheels = tuple(
            WheelIdentity.from_dict(item, f"profile.wheel_files[{index}]")
            for index, item in enumerate(
                _expect_sequence(mapping["wheel_files"], "profile.wheel_files")
            )
        )
        sources = tuple(
            InstalledSourceIdentity.from_dict(
                item, f"profile.installed_sources[{index}]"
            )
            for index, item in enumerate(
                _expect_sequence(mapping["installed_sources"], "profile.installed_sources")
            )
        )
        forbidden = _expect_string_tuple(
            mapping["forbidden_packages"], "profile.forbidden_packages"
        )
        profile = cls(
            schema_id=schema_id,
            runtime_id=runtime_id,
            allowed_worker_operations=operations,
            candidate=_freeze_json(mapping["candidate"]),
            canonical_input=_freeze_json(mapping["canonical_input"]),
            current_base=_freeze_json(mapping["current_base"]),
            dependency_validation=_freeze_json(mapping["dependency_validation"]),
            external_layout=_freeze_json(mapping["external_layout"]),
            forbidden_packages=forbidden,
            installed_sources=sources,
            manual_preprocessing=_freeze_json(mapping["manual_preprocessing"]),
            metadata_files=metadata,
            model_weights=_freeze_json(mapping["model_weights"]),
            policies=_freeze_json(mapping["policies"]),
            runtime_root=_freeze_json(mapping["runtime_root"]),
            target=_freeze_json(mapping["target"]),
            wheel_files=wheels,
            worker_protocol=_freeze_json(mapping["worker_protocol"]),
            source_sha256=(
                _expect_sha256(source_sha256, "profile.source_sha256")
                if source_sha256
                else ""
            ),
        )
        profile._validate_cross_fields()
        return profile

    @staticmethod
    def _metadata_identity(value: object, path: str) -> FileIdentity:
        mapping = _expect_exact_fields(
            value, frozenset({"bytes", "filename", "sha256"}), path
        )
        return FileIdentity(
            filename=_expect_basename(mapping["filename"], f"{path}.filename"),
            size_bytes=_expect_int(mapping["bytes"], f"{path}.bytes", minimum=1),
            sha256=_expect_sha256(mapping["sha256"], f"{path}.sha256"),
        )

    @staticmethod
    def _validate_freeze(value: object, path: str) -> None:
        mapping = _expect_exact_fields(
            value, frozenset({"bytes", "line_count", "sha256"}), path
        )
        _expect_int(mapping["bytes"], f"{path}.bytes", minimum=1)
        _expect_int(mapping["line_count"], f"{path}.line_count", minimum=1)
        _expect_sha256(mapping["sha256"], f"{path}.sha256")

    @classmethod
    def _validate_candidate(cls, value: object) -> None:
        path = "profile.candidate"
        mapping = _expect_exact_fields(
            value,
            frozenset(
                {"expected_freeze", "local_packages", "python", "system_site_packages"}
            ),
            path,
        )
        cls._validate_freeze(mapping["expected_freeze"], f"{path}.expected_freeze")
        packages = _expect_exact_fields(
            mapping["local_packages"],
            frozenset({"huggingface-hub", "pip", "tokenizers", "transformers"}),
            f"{path}.local_packages",
        )
        for name, version in packages.items():
            _expect_exact_version(version, f"{path}.local_packages.{name}")
        python = _expect_exact_fields(
            mapping["python"], frozenset({"major", "minor", "version"}), f"{path}.python"
        )
        if (
            _expect_int(python["major"], f"{path}.python.major") != 3
            or _expect_int(python["minor"], f"{path}.python.minor") != 12
            or _expect_exact_version(python["version"], f"{path}.python.version") != "3.12.0"
        ):
            _fail("schema_validation_failed", f"{path}.python", "expected Python 3.12.0")
        if not _expect_bool(mapping["system_site_packages"], f"{path}.system_site_packages"):
            _fail("schema_validation_failed", f"{path}.system_site_packages", "must be true")

    @staticmethod
    def _validate_canonical_input(value: object) -> None:
        path = "profile.canonical_input"
        mapping = _expect_exact_fields(
            value,
            frozenset(
                {"channel_order", "dtype", "height", "layout", "random_augmentation", "width"}
            ),
            path,
        )
        actual = (
            _expect_string(mapping["channel_order"], f"{path}.channel_order"),
            _expect_string(mapping["dtype"], f"{path}.dtype"),
            _expect_int(mapping["height"], f"{path}.height", minimum=1),
            _expect_string(mapping["layout"], f"{path}.layout"),
            _expect_bool(mapping["random_augmentation"], f"{path}.random_augmentation"),
            _expect_int(mapping["width"], f"{path}.width", minimum=1),
        )
        if actual != ("RGB", "uint8", 224, "NHWC", False, 224):
            _fail("schema_validation_failed", path, "unexpected canonical geometry")

    @classmethod
    def _validate_current_base(cls, value: object) -> None:
        path = "profile.current_base"
        mapping = _expect_exact_fields(
            value,
            frozenset({"expected_freeze", "packages", "python", "source_files", "torch"}),
            path,
        )
        cls._validate_freeze(mapping["expected_freeze"], f"{path}.expected_freeze")
        packages = _expect_exact_fields(
            mapping["packages"],
            frozenset(
                {"Pillow", "huggingface-hub", "numpy", "pytest", "safetensors", "tokenizers", "torch", "transformers"}
            ),
            f"{path}.packages",
        )
        for name, version in packages.items():
            _expect_exact_version(version, f"{path}.packages.{name}")
        if _expect_exact_version(mapping["python"], f"{path}.python") != "3.12.0":
            _fail("schema_validation_failed", f"{path}.python", "unexpected version")
        source_rows = _expect_sequence(mapping["source_files"], f"{path}.source_files")
        source_names: set[str] = set()
        for index, item in enumerate(source_rows):
            item_path = f"{path}.source_files[{index}]"
            row = _expect_exact_fields(item, frozenset({"path", "sha256"}), item_path)
            source_name = _expect_relative_path(row["path"], f"{item_path}.path")
            _expect_sha256(row["sha256"], f"{item_path}.sha256")
            if source_name in source_names:
                _fail("schema_validation_failed", f"{path}.source_files", "duplicate path")
            source_names.add(source_name)
        torch = _expect_exact_fields(
            mapping["torch"],
            frozenset(
                {"candidate_local_distribution", "cuda_available", "cuda_device_count", "metadata_version", "runtime_version"}
            ),
            f"{path}.torch",
        )
        if _expect_bool(torch["candidate_local_distribution"], f"{path}.torch.candidate_local_distribution"):
            _fail("schema_validation_failed", f"{path}.torch", "candidate-local torch is forbidden")
        if _expect_bool(torch["cuda_available"], f"{path}.torch.cuda_available"):
            _fail("schema_validation_failed", f"{path}.torch", "CUDA must be unavailable")
        if _expect_int(torch["cuda_device_count"], f"{path}.torch.cuda_device_count") != 0:
            _fail("schema_validation_failed", f"{path}.torch", "CUDA device count must be zero")
        if (
            _expect_exact_version(torch["metadata_version"], f"{path}.torch.metadata_version") != "2.3.0"
            or _expect_exact_version(torch["runtime_version"], f"{path}.torch.runtime_version") != "2.3.0+cpu"
        ):
            _fail("schema_validation_failed", f"{path}.torch", "unexpected torch identity")

    @staticmethod
    def _validate_dependency(value: object) -> None:
        path = "profile.dependency_validation"
        mapping = _expect_exact_fields(
            value,
            frozenset(
                {
                    "active_requirement_count",
                    "expected_inherited_conflict_count",
                    "expected_inherited_conflict_sha256",
                    "expected_new_conflict_count",
                    "expected_unsatisfied_active_requirement_count",
                    "inherited_conflict_hash_encoding",
                    "inherited_conflict_hash_join",
                    "requirement_owners",
                }
            ),
            path,
        )
        _expect_int(mapping["active_requirement_count"], f"{path}.active_requirement_count")
        _expect_int(mapping["expected_inherited_conflict_count"], f"{path}.expected_inherited_conflict_count")
        _expect_sha256(mapping["expected_inherited_conflict_sha256"], f"{path}.expected_inherited_conflict_sha256")
        if _expect_int(mapping["expected_new_conflict_count"], f"{path}.expected_new_conflict_count") != 0:
            _fail("schema_validation_failed", f"{path}.expected_new_conflict_count", "must be zero")
        if _expect_int(mapping["expected_unsatisfied_active_requirement_count"], f"{path}.expected_unsatisfied_active_requirement_count") != 0:
            _fail("schema_validation_failed", f"{path}.expected_unsatisfied_active_requirement_count", "must be zero")
        _expect_string(mapping["inherited_conflict_hash_encoding"], f"{path}.inherited_conflict_hash_encoding")
        _expect_string(mapping["inherited_conflict_hash_join"], f"{path}.inherited_conflict_hash_join")
        owners = _expect_string_tuple(mapping["requirement_owners"], f"{path}.requirement_owners")
        if set(owners) != {"huggingface-hub", "tokenizers", "transformers"}:
            _fail("schema_validation_failed", f"{path}.requirement_owners", "unexpected owners")

    @staticmethod
    def _validate_external_layout(value: object) -> None:
        path = "profile.external_layout"
        mapping = _expect_exact_fields(
            value,
            frozenset({"manifest", "metadata_directory", "venv_directory", "venv_python", "wheelhouse_directory"}),
            path,
        )
        for key, item in mapping.items():
            _expect_relative_path(item, f"{path}.{key}")

    @staticmethod
    def _validate_manual_preprocessing(value: object) -> None:
        path = "profile.manual_preprocessing"
        mapping = _expect_exact_fields(
            value,
            frozenset(
                {
                    "expected_input_sha256",
                    "expected_output_sha256",
                    "formula",
                    "image_generation",
                    "image_mean",
                    "image_std",
                    "input_shape",
                    "output_dtype",
                    "output_shape",
                    "profile_id",
                    "rescale_factor_dtype",
                    "rescale_factor_fp32",
                    "rescale_factor_json",
                }
            ),
            path,
        )
        _expect_sha256(mapping["expected_input_sha256"], f"{path}.expected_input_sha256")
        _expect_sha256(mapping["expected_output_sha256"], f"{path}.expected_output_sha256")
        formula = _expect_string_tuple(mapping["formula"], f"{path}.formula")
        if len(formula) != 7:
            _fail("schema_validation_failed", f"{path}.formula", "expected seven operations")
        generation = _expect_exact_fields(
            mapping["image_generation"],
            frozenset(
                {
                    "checker_blue", "checker_green", "checker_red", "checker_tile_size",
                    "gradient_blue", "gradient_green", "gradient_red", "indices_dtype", "stack_order"
                }
            ),
            f"{path}.image_generation",
        )
        for key in (
            "checker_blue", "checker_green", "checker_red", "gradient_blue",
            "gradient_green", "gradient_red", "indices_dtype"
        ):
            _expect_string(generation[key], f"{path}.image_generation.{key}")
        _expect_int(generation["checker_tile_size"], f"{path}.image_generation.checker_tile_size", minimum=1)
        if _expect_string_tuple(generation["stack_order"], f"{path}.image_generation.stack_order") != ("gradient", "checker"):
            _fail("schema_validation_failed", f"{path}.image_generation.stack_order", "unexpected order")
        mean = tuple(_expect_float(item, f"{path}.image_mean[{index}]") for index, item in enumerate(_expect_sequence(mapping["image_mean"], f"{path}.image_mean")))
        std = tuple(_expect_float(item, f"{path}.image_std[{index}]") for index, item in enumerate(_expect_sequence(mapping["image_std"], f"{path}.image_std")))
        if mean != (0.485, 0.456, 0.406) or std != (0.229, 0.224, 0.225):
            _fail("schema_validation_failed", path, "unexpected ImageNet constants")
        input_shape = tuple(_expect_int(item, f"{path}.input_shape[{index}]", minimum=1) for index, item in enumerate(_expect_sequence(mapping["input_shape"], f"{path}.input_shape")))
        output_shape = tuple(_expect_int(item, f"{path}.output_shape[{index}]", minimum=1) for index, item in enumerate(_expect_sequence(mapping["output_shape"], f"{path}.output_shape")))
        if input_shape != (2, 224, 224, 3) or output_shape != (2, 3, 224, 224):
            _fail("schema_validation_failed", path, "unexpected preprocessing shapes")
        if _expect_string(mapping["output_dtype"], f"{path}.output_dtype") != "float32":
            _fail("schema_validation_failed", f"{path}.output_dtype", "expected float32")
        if _expect_identifier(mapping["profile_id"], f"{path}.profile_id") != "dinov3-manual-fp32-imagenet-v1":
            _fail("schema_validation_failed", f"{path}.profile_id", "unexpected profile ID")
        if _expect_string(mapping["rescale_factor_dtype"], f"{path}.rescale_factor_dtype") != "float32":
            _fail("schema_validation_failed", f"{path}.rescale_factor_dtype", "expected float32")
        if not math.isclose(_expect_float(mapping["rescale_factor_json"], f"{path}.rescale_factor_json"), 1.0 / 255.0, rel_tol=0.0, abs_tol=1e-18):
            _fail("schema_validation_failed", f"{path}.rescale_factor_json", "expected 1/255")
        if not math.isclose(_expect_float(mapping["rescale_factor_fp32"], f"{path}.rescale_factor_fp32"), 0.003921568859368563, rel_tol=0.0, abs_tol=1e-18):
            _fail("schema_validation_failed", f"{path}.rescale_factor_fp32", "unexpected FP32 factor")

    @staticmethod
    def _validate_model_weights(value: object) -> None:
        path = "profile.model_weights"
        mapping = _expect_exact_fields(
            value, frozenset({"expected_present", "forbidden_filenames", "transferred_bytes"}), path
        )
        if _expect_bool(mapping["expected_present"], f"{path}.expected_present"):
            _fail("schema_validation_failed", f"{path}.expected_present", "must be false")
        for index, name in enumerate(_expect_sequence(mapping["forbidden_filenames"], f"{path}.forbidden_filenames")):
            _expect_basename(name, f"{path}.forbidden_filenames[{index}]")
        if _expect_int(mapping["transferred_bytes"], f"{path}.transferred_bytes") != 0:
            _fail("schema_validation_failed", f"{path}.transferred_bytes", "must be zero")

    @staticmethod
    def _validate_policies(value: object) -> None:
        path = "profile.policies"
        mapping = _expect_exact_fields(
            value,
            frozenset(
                {
                    "cpu_only", "image_processor_runtime_object_allowed", "local_metadata_only_after_provision",
                    "model_checkpoint_allowed", "network_allowed_during_verify", "system_site_packages", "torchvision_allowed"
                }
            ),
            path,
        )
        expected = {
            "cpu_only": True,
            "image_processor_runtime_object_allowed": False,
            "local_metadata_only_after_provision": True,
            "model_checkpoint_allowed": False,
            "network_allowed_during_verify": False,
            "system_site_packages": True,
            "torchvision_allowed": False,
        }
        for key, expected_value in expected.items():
            if _expect_bool(mapping[key], f"{path}.{key}") is not expected_value:
                _fail("schema_validation_failed", f"{path}.{key}", "unexpected policy")

    @staticmethod
    def _validate_runtime_root_contract(value: object, runtime_id: str) -> None:
        path = "profile.runtime_root"
        mapping = _expect_exact_fields(
            value,
            frozenset({"default_template", "environment_variable", "relative_parts", "repository_relation", "staging_name_prefix"}),
            path,
        )
        template = _expect_string(mapping["default_template"], f"{path}.default_template")
        if template != f"%LOCALAPPDATA%/VEA/runtimes/{runtime_id}":
            _fail("schema_validation_failed", f"{path}.default_template", "unexpected template")
        if _expect_identifier(mapping["environment_variable"], f"{path}.environment_variable") != "LOCALAPPDATA":
            _fail("schema_validation_failed", f"{path}.environment_variable", "unexpected variable")
        parts = _expect_string_tuple(mapping["relative_parts"], f"{path}.relative_parts")
        if parts != ("VEA", "runtimes", runtime_id):
            _fail("schema_validation_failed", f"{path}.relative_parts", "unexpected path")
        if _expect_identifier(mapping["repository_relation"], f"{path}.repository_relation") != "outside_repository":
            _fail("schema_validation_failed", f"{path}.repository_relation", "unexpected relation")
        prefix = _expect_string(mapping["staging_name_prefix"], f"{path}.staging_name_prefix")
        if prefix != f".{runtime_id}.staging-":
            _fail("schema_validation_failed", f"{path}.staging_name_prefix", "unexpected prefix")

    @staticmethod
    def _validate_target(value: object) -> None:
        path = "profile.target"
        mapping = _expect_exact_fields(
            value, frozenset({"evidence_revision", "model_id", "registry_key"}), path
        )
        _expect_git_sha(mapping["evidence_revision"], f"{path}.evidence_revision")
        _expect_string(mapping["model_id"], f"{path}.model_id")
        _expect_identifier(mapping["registry_key"], f"{path}.registry_key")

    @staticmethod
    def _validate_worker_protocol(value: object) -> None:
        path = "profile.worker_protocol"
        mapping = _expect_exact_fields(
            value, frozenset({"request_schema_id", "response_schema_id"}), path
        )
        if _expect_string(mapping["request_schema_id"], f"{path}.request_schema_id") != WORKER_REQUEST_SCHEMA_ID:
            _fail("schema_validation_failed", f"{path}.request_schema_id", "unexpected schema")
        if _expect_string(mapping["response_schema_id"], f"{path}.response_schema_id") != WORKER_RESPONSE_SCHEMA_ID:
            _fail("schema_validation_failed", f"{path}.response_schema_id", "unexpected schema")

    def _validate_cross_fields(self) -> None:
        if len({item.filename for item in self.wheel_files}) != len(self.wheel_files) or len(self.wheel_files) != 3:
            _fail("schema_validation_failed", "profile.wheel_files", "expected three unique wheels")
        if {item.distribution for item in self.wheel_files} != {
            "huggingface-hub", "tokenizers", "transformers"
        }:
            _fail("schema_validation_failed", "profile.wheel_files", "unexpected distributions")
        local_packages = self.candidate["local_packages"]
        for wheel in self.wheel_files:
            if local_packages[wheel.distribution] != wheel.version:
                _fail("schema_validation_failed", "profile.wheel_files", "version mismatch")
        if {item.filename for item in self.metadata_files} != {
            "config.json",
            "preprocessor_config.json",
        }:
            _fail("schema_validation_failed", "profile.metadata_files", "unexpected metadata files")
        if len({item.path for item in self.installed_sources}) != len(self.installed_sources):
            _fail("schema_validation_failed", "profile.installed_sources", "duplicate source path")
        lowered_forbidden = {item.lower() for item in self.forbidden_packages}
        if not _REQUIRED_FORBIDDEN_PACKAGES.issubset(lowered_forbidden):
            _fail("schema_validation_failed", "profile.forbidden_packages", "required packages are missing")

    @property
    def target_model_id(self) -> str:
        return str(self.target["model_id"])

    @property
    def evidence_revision(self) -> str:
        return str(self.target["evidence_revision"])

    @property
    def current_base_freeze_sha256(self) -> str:
        return str(self.current_base["expected_freeze"]["sha256"])

    @property
    def expected_candidate_freeze_sha256(self) -> str:
        return str(self.candidate["expected_freeze"]["sha256"])

    @property
    def shared_torch(self) -> SharedTorchProfile:
        torch = self.current_base["torch"]
        return SharedTorchProfile(
            str(torch["metadata_version"]), str(torch["runtime_version"])
        )

    @property
    def candidate_packages(self) -> tuple[PackageIdentity, ...]:
        return tuple(
            PackageIdentity(str(name), str(version))
            for name, version in self.candidate["local_packages"].items()
        )

    @property
    def metadata(self) -> tuple[FileIdentity, ...]:
        return self.metadata_files

    @property
    def wheels(self) -> tuple[FileIdentity, ...]:
        return tuple(
            FileIdentity(item.filename, item.size_bytes, item.sha256)
            for item in self.wheel_files
        )

    @property
    def manual_preprocessing_profile_id(self) -> str:
        return str(self.manual_preprocessing["profile_id"])

    @property
    def default_runtime_root_template(self) -> str:
        return str(self.runtime_root["default_template"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_worker_operations": list(self.allowed_worker_operations),
            "candidate": _thaw_json(self.candidate),
            "canonical_input": _thaw_json(self.canonical_input),
            "current_base": _thaw_json(self.current_base),
            "dependency_validation": _thaw_json(self.dependency_validation),
            "external_layout": _thaw_json(self.external_layout),
            "forbidden_packages": list(self.forbidden_packages),
            "installed_sources": [item.to_dict() for item in self.installed_sources],
            "manual_preprocessing": _thaw_json(self.manual_preprocessing),
            "metadata_files": [
                {"bytes": item.size_bytes, "filename": item.filename, "sha256": item.sha256}
                for item in self.metadata_files
            ],
            "model_weights": _thaw_json(self.model_weights),
            "policies": _thaw_json(self.policies),
            "runtime_id": self.runtime_id,
            "runtime_root": _thaw_json(self.runtime_root),
            "schema_id": self.schema_id,
            "target": _thaw_json(self.target),
            "wheel_files": [item.to_dict() for item in self.wheel_files],
            "worker_protocol": _thaw_json(self.worker_protocol),
        }


@dataclass(frozen=True, slots=True)
class DependencyClosure:
    active_requirement_count: int
    unsatisfied_count: int

    @classmethod
    def from_dict(cls, value: object, path: str) -> "DependencyClosure":
        mapping = _expect_exact_fields(
            value,
            frozenset({"active_requirement_count", "unsatisfied_count"}),
            path,
        )
        result = cls(
            active_requirement_count=_expect_int(
                mapping["active_requirement_count"],
                f"{path}.active_requirement_count",
            ),
            unsatisfied_count=_expect_int(
                mapping["unsatisfied_count"], f"{path}.unsatisfied_count"
            ),
        )
        if result.unsatisfied_count != 0:
            _fail("schema_validation_failed", f"{path}.unsatisfied_count", "expected zero")
        return result

    def to_dict(self) -> dict[str, int]:
        return {
            "active_requirement_count": self.active_requirement_count,
            "unsatisfied_count": self.unsatisfied_count,
        }


@dataclass(frozen=True, slots=True)
class ConflictSummary:
    count: int
    sha256: str

    @classmethod
    def from_dict(cls, value: object, path: str) -> "ConflictSummary":
        mapping = _expect_exact_fields(value, frozenset({"count", "sha256"}), path)
        return cls(
            count=_expect_int(mapping["count"], f"{path}.count"),
            sha256=_expect_sha256(mapping["sha256"], f"{path}.sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class TorchvisionSummary:
    distribution_count: int
    module_count: int

    @classmethod
    def from_dict(cls, value: object, path: str) -> "TorchvisionSummary":
        mapping = _expect_exact_fields(
            value, frozenset({"distribution_count", "module_count"}), path
        )
        result = cls(
            distribution_count=_expect_int(
                mapping["distribution_count"], f"{path}.distribution_count"
            ),
            module_count=_expect_int(mapping["module_count"], f"{path}.module_count"),
        )
        if result.distribution_count or result.module_count:
            _fail("schema_validation_failed", path, "torchvision counts must be zero")
        return result

    def to_dict(self) -> dict[str, int]:
        return {
            "distribution_count": self.distribution_count,
            "module_count": self.module_count,
        }


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    """Fully validated immutable external runtime manifest."""

    schema_id: str
    runtime_id: str
    repository_profile_sha256: str
    provisioned_at_utc: str
    python_version: str
    candidate_freeze: Mapping[str, Any]
    candidate_local_packages: Mapping[str, Any]
    current_base_freeze_sha256: str
    dependency_validation: Mapping[str, Any]
    forbidden_candidate_package_count: int
    installed_sources: tuple[InstalledSourceIdentity, ...]
    manual_preprocessing_profile_id: str
    metadata_files: tuple[FileIdentity, ...]
    model_weight_present: bool
    network_independent_verification_status: str
    shared_torch: Mapping[str, Any]
    torchvision: Mapping[str, Any]
    wheel_files: tuple[WheelIdentity, ...]
    source_sha256: str = field(default="", repr=False, compare=False)

    @classmethod
    def from_dict(
        cls, value: object, *, source_sha256: str = ""
    ) -> "RuntimeManifest":
        expected = frozenset(
            {
                "schema_id",
                "runtime_id",
                "repository_profile_sha256",
                "provisioned_at_utc",
                "python_version",
                "shared_torch",
                "candidate_freeze",
                "candidate_local_packages",
                "current_base_freeze_sha256",
                "dependency_validation",
                "forbidden_candidate_package_count",
                "installed_sources",
                "manual_preprocessing_profile_id",
                "metadata_files",
                "model_weight_present",
                "network_independent_verification_status",
                "shared_torch",
                "torchvision",
                "wheel_files",
            }
        )
        mapping = _expect_exact_fields(value, expected, "manifest")
        RuntimeProfile._validate_freeze(
            mapping["candidate_freeze"], "manifest.candidate_freeze"
        )
        local_packages = _expect_exact_fields(
            mapping["candidate_local_packages"],
            frozenset({"huggingface-hub", "pip", "tokenizers", "transformers"}),
            "manifest.candidate_local_packages",
        )
        for name, version in local_packages.items():
            _expect_exact_version(version, f"manifest.candidate_local_packages.{name}")
        dependency = _expect_exact_fields(
            mapping["dependency_validation"],
            frozenset(
                {
                    "active_requirement_count",
                    "inherited_conflict_count",
                    "inherited_conflict_sha256",
                    "new_conflict_count",
                    "unsatisfied_active_requirement_count",
                }
            ),
            "manifest.dependency_validation",
        )
        _expect_int(
            dependency["active_requirement_count"],
            "manifest.dependency_validation.active_requirement_count",
        )
        _expect_int(
            dependency["inherited_conflict_count"],
            "manifest.dependency_validation.inherited_conflict_count",
        )
        _expect_sha256(
            dependency["inherited_conflict_sha256"],
            "manifest.dependency_validation.inherited_conflict_sha256",
        )
        for key in ("new_conflict_count", "unsatisfied_active_requirement_count"):
            if _expect_int(dependency[key], f"manifest.dependency_validation.{key}") != 0:
                _fail(
                    "schema_validation_failed",
                    f"manifest.dependency_validation.{key}",
                    "expected zero",
                )
        metadata = tuple(
            RuntimeProfile._metadata_identity(
                item, f"manifest.metadata_files[{index}]"
            )
            for index, item in enumerate(
                _expect_sequence(mapping["metadata_files"], "manifest.metadata_files")
            )
        )
        wheels = tuple(
            WheelIdentity.from_dict(item, f"manifest.wheel_files[{index}]")
            for index, item in enumerate(
                _expect_sequence(mapping["wheel_files"], "manifest.wheel_files")
            )
        )
        sources = tuple(
            InstalledSourceIdentity.from_dict(
                item, f"manifest.installed_sources[{index}]"
            )
            for index, item in enumerate(
                _expect_sequence(mapping["installed_sources"], "manifest.installed_sources")
            )
        )
        shared_torch = _expect_exact_fields(
            mapping["shared_torch"],
            frozenset(
                {
                    "candidate_local_distribution",
                    "cuda_available",
                    "cuda_device_count",
                    "metadata_version",
                    "runtime_version",
                }
            ),
            "manifest.shared_torch",
        )
        if _expect_bool(
            shared_torch["candidate_local_distribution"],
            "manifest.shared_torch.candidate_local_distribution",
        ):
            _fail(
                "schema_validation_failed",
                "manifest.shared_torch.candidate_local_distribution",
                "must be false",
            )
        if _expect_bool(
            shared_torch["cuda_available"], "manifest.shared_torch.cuda_available"
        ):
            _fail(
                "schema_validation_failed",
                "manifest.shared_torch.cuda_available",
                "must be false",
            )
        if _expect_int(
            shared_torch["cuda_device_count"],
            "manifest.shared_torch.cuda_device_count",
        ) != 0:
            _fail(
                "schema_validation_failed",
                "manifest.shared_torch.cuda_device_count",
                "must be zero",
            )
        _expect_exact_version(
            shared_torch["metadata_version"],
            "manifest.shared_torch.metadata_version",
        )
        _expect_exact_version(
            shared_torch["runtime_version"],
            "manifest.shared_torch.runtime_version",
        )
        torchvision = _expect_exact_fields(
            mapping["torchvision"],
            frozenset({"distribution_count", "module_count"}),
            "manifest.torchvision",
        )
        if any(
            _expect_int(torchvision[key], f"manifest.torchvision.{key}") != 0
            for key in ("distribution_count", "module_count")
        ):
            _fail(
                "schema_validation_failed",
                "manifest.torchvision",
                "counts must be zero",
            )
        result = cls(
            schema_id=_expect_string(mapping["schema_id"], "manifest.schema_id"),
            runtime_id=_expect_identifier(mapping["runtime_id"], "manifest.runtime_id"),
            repository_profile_sha256=_expect_sha256(
                mapping["repository_profile_sha256"],
                "manifest.repository_profile_sha256",
            ),
            provisioned_at_utc=_expect_string(
                mapping["provisioned_at_utc"], "manifest.provisioned_at_utc"
            ),
            python_version=_expect_exact_version(
                mapping["python_version"], "manifest.python_version"
            ),
            candidate_freeze=_freeze_json(mapping["candidate_freeze"]),
            candidate_local_packages=_freeze_json(
                mapping["candidate_local_packages"]
            ),
            current_base_freeze_sha256=_expect_sha256(
                mapping["current_base_freeze_sha256"],
                "manifest.current_base_freeze_sha256",
            ),
            dependency_validation=_freeze_json(mapping["dependency_validation"]),
            forbidden_candidate_package_count=_expect_int(
                mapping["forbidden_candidate_package_count"],
                "manifest.forbidden_candidate_package_count",
            ),
            installed_sources=sources,
            manual_preprocessing_profile_id=_expect_identifier(
                mapping["manual_preprocessing_profile_id"],
                "manifest.manual_preprocessing_profile_id",
            ),
            metadata_files=metadata,
            model_weight_present=_expect_bool(
                mapping["model_weight_present"], "manifest.model_weight_present"
            ),
            network_independent_verification_status=_expect_identifier(
                mapping["network_independent_verification_status"],
                "manifest.network_independent_verification_status",
            ),
            shared_torch=_freeze_json(mapping["shared_torch"]),
            torchvision=_freeze_json(mapping["torchvision"]),
            wheel_files=wheels,
            source_sha256=(
                _expect_sha256(source_sha256, "manifest.source_sha256")
                if source_sha256
                else ""
            ),
        )
        if result.schema_id != RUNTIME_MANIFEST_SCHEMA_ID:
            _fail("schema_validation_failed", "manifest.schema_id", "unexpected schema")
        if not result.provisioned_at_utc.endswith("Z"):
            _fail("schema_validation_failed", "manifest.provisioned_at_utc", "expected UTC Z timestamp")
        try:
            datetime.fromisoformat(result.provisioned_at_utc[:-1] + "+00:00")
        except ValueError as exc:
            raise RuntimeBoundaryError(
                "schema_validation_failed",
                "manifest.provisioned_at_utc: invalid ISO-8601 timestamp",
            ) from exc
        if result.model_weight_present:
            _fail("schema_validation_failed", "manifest.model_weight_present", "must be false")
        if result.network_independent_verification_status != "VERIFIED":
            _fail(
                "schema_validation_failed",
                "manifest.network_independent_verification_status",
                "expected VERIFIED",
            )
        if result.forbidden_candidate_package_count != 0:
            _fail(
                "schema_validation_failed",
                "manifest.forbidden_candidate_package_count",
                "expected zero",
            )
        _scan_response_strings(result.to_dict(), "manifest")
        return result

    @property
    def candidate_freeze_sha256(self) -> str:
        return str(self.candidate_freeze["sha256"])

    @property
    def candidate_packages(self) -> tuple[PackageIdentity, ...]:
        return tuple(
            PackageIdentity(str(name), str(version))
            for name, version in self.candidate_local_packages.items()
        )

    @property
    def metadata(self) -> tuple[FileIdentity, ...]:
        return self.metadata_files

    @property
    def wheels(self) -> tuple[FileIdentity, ...]:
        return tuple(
            FileIdentity(item.filename, item.size_bytes, item.sha256)
            for item in self.wheel_files
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_freeze": _thaw_json(self.candidate_freeze),
            "candidate_local_packages": _thaw_json(self.candidate_local_packages),
            "current_base_freeze_sha256": self.current_base_freeze_sha256,
            "dependency_validation": _thaw_json(self.dependency_validation),
            "forbidden_candidate_package_count": self.forbidden_candidate_package_count,
            "installed_sources": [item.to_dict() for item in self.installed_sources],
            "manual_preprocessing_profile_id": self.manual_preprocessing_profile_id,
            "metadata_files": [
                {"bytes": item.size_bytes, "filename": item.filename, "sha256": item.sha256}
                for item in self.metadata_files
            ],
            "model_weight_present": self.model_weight_present,
            "network_independent_verification_status": self.network_independent_verification_status,
            "provisioned_at_utc": self.provisioned_at_utc,
            "python_version": self.python_version,
            "repository_profile_sha256": self.repository_profile_sha256,
            "runtime_id": self.runtime_id,
            "schema_id": self.schema_id,
            "shared_torch": _thaw_json(self.shared_torch),
            "torchvision": _thaw_json(self.torchvision),
            "wheel_files": [item.to_dict() for item in self.wheel_files],
        }


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    """Immutable isolated-worker request contract."""

    schema_id: str
    operation: str
    run_id: str
    repository_head: str
    runtime_profile_sha256: str
    workspace_relative_inbound_artifact_path: str
    workspace_relative_outbound_artifact_path: str
    deterministic_seed: int
    expected_preprocessing_profile_id: str

    @classmethod
    def from_dict(cls, value: object) -> "WorkerRequest":
        expected = frozenset(
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
        mapping = _expect_exact_fields(value, expected, "request")
        result = cls(
            schema_id=_expect_string(mapping["schema_id"], "request.schema_id"),
            operation=_expect_identifier(mapping["operation"], "request.operation"),
            run_id=_expect_identifier(mapping["run_id"], "request.run_id"),
            repository_head=_expect_git_sha(
                mapping["repository_head"], "request.repository_head"
            ),
            runtime_profile_sha256=_expect_sha256(
                mapping["runtime_profile_sha256"], "request.runtime_profile_sha256"
            ),
            workspace_relative_inbound_artifact_path=_expect_relative_path(
                mapping["workspace_relative_inbound_artifact_path"],
                "request.workspace_relative_inbound_artifact_path",
            ),
            workspace_relative_outbound_artifact_path=_expect_relative_path(
                mapping["workspace_relative_outbound_artifact_path"],
                "request.workspace_relative_outbound_artifact_path",
            ),
            deterministic_seed=_expect_int(
                mapping["deterministic_seed"],
                "request.deterministic_seed",
                maximum=2**32 - 1,
            ),
            expected_preprocessing_profile_id=_expect_identifier(
                mapping["expected_preprocessing_profile_id"],
                "request.expected_preprocessing_profile_id",
            ),
        )
        if result.schema_id != WORKER_REQUEST_SCHEMA_ID:
            _fail("schema_validation_failed", "request.schema_id", "unexpected schema")
        if result.operation not in ALLOWED_WORKER_OPERATIONS:
            _fail("unknown_worker_operation", "request.operation", "operation is not allowed")
        if result.deterministic_seed != 0:
            _fail(
                "schema_validation_failed",
                "request.deterministic_seed",
                "expected the accepted deterministic seed 0",
            )
        if (
            result.workspace_relative_inbound_artifact_path
            == result.workspace_relative_outbound_artifact_path
        ):
            _fail("unsafe_path", "request", "inbound and outbound paths must differ")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "deterministic_seed": self.deterministic_seed,
            "expected_preprocessing_profile_id": self.expected_preprocessing_profile_id,
            "workspace_relative_inbound_artifact_path": (
                self.workspace_relative_inbound_artifact_path
            ),
            "operation": self.operation,
            "workspace_relative_outbound_artifact_path": (
                self.workspace_relative_outbound_artifact_path
            ),
            "repository_head": self.repository_head,
            "run_id": self.run_id,
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "schema_id": self.schema_id,
        }


def _shape_tuple(value: object, path: str, rank: int) -> tuple[int, ...]:
    sequence = _expect_sequence(value, path)
    if len(sequence) != rank:
        _fail("schema_validation_failed", path, f"expected rank {rank}")
    return tuple(
        _expect_int(item, f"{path}[{index}]", minimum=1)
        for index, item in enumerate(sequence)
    )


def _validate_artifact_entry(value: object, path: str) -> None:
    mapping = _expect_exact_fields(
        value,
        frozenset(
            {
                "checksum",
                "dtype",
                "global_shape",
                "global_values_sha256",
                "image_ids",
                "patch_shape",
                "patch_values_sha256",
                "sidecar",
                "tap_name",
                "tensor_keys",
            }
        ),
        path,
    )
    _expect_sha256(mapping["checksum"], f"{path}.checksum")
    if _expect_string(mapping["dtype"], f"{path}.dtype") != "float32":
        _fail("schema_validation_failed", f"{path}.dtype", "expected float32")
    if _shape_tuple(mapping["patch_shape"], f"{path}.patch_shape", 3) != (2, 4, 3):
        _fail("schema_validation_failed", f"{path}.patch_shape", "unexpected shape")
    if _shape_tuple(mapping["global_shape"], f"{path}.global_shape", 2) != (2, 3):
        _fail("schema_validation_failed", f"{path}.global_shape", "unexpected shape")
    _expect_sha256(mapping["global_values_sha256"], f"{path}.global_values_sha256")
    _expect_sha256(mapping["patch_values_sha256"], f"{path}.patch_values_sha256")
    if _expect_string_tuple(mapping["image_ids"], f"{path}.image_ids") != (
        "image-b",
        "image-a",
    ):
        _fail("schema_validation_failed", f"{path}.image_ids", "unexpected order")
    _expect_basename(mapping["sidecar"], f"{path}.sidecar")
    tap = _expect_identifier(mapping["tap_name"], f"{path}.tap_name")
    tensor_keys = _expect_string_tuple(mapping["tensor_keys"], f"{path}.tensor_keys")
    if tensor_keys != (f"backbone.{tap}.patch", f"backbone.{tap}.global_mean"):
        _fail("schema_validation_failed", f"{path}.tensor_keys", "unexpected keys")


def _validate_artifact_summary(value: object, path: str) -> tuple[str, str]:
    mapping = _expect_exact_fields(
        value,
        frozenset(
            {"validated", "image_ids", "tap_order", "patch_shape", "global_shape", "dtype", "entries"}
        ),
        path,
    )
    if not _expect_bool(mapping["validated"], f"{path}.validated"):
        _fail("schema_validation_failed", f"{path}.validated", "expected true")
    if _expect_string_tuple(mapping["image_ids"], f"{path}.image_ids") != (
        "image-b",
        "image-a",
    ):
        _fail("schema_validation_failed", f"{path}.image_ids", "unexpected order")
    taps = _expect_string_tuple(mapping["tap_order"], f"{path}.tap_order")
    if taps != ("embedding_output", "block_06"):
        _fail("schema_validation_failed", f"{path}.tap_order", "unexpected order")
    if _shape_tuple(mapping["patch_shape"], f"{path}.patch_shape", 3) != (2, 4, 3):
        _fail("schema_validation_failed", f"{path}.patch_shape", "unexpected shape")
    if _shape_tuple(mapping["global_shape"], f"{path}.global_shape", 2) != (2, 3):
        _fail("schema_validation_failed", f"{path}.global_shape", "unexpected shape")
    if _expect_string(mapping["dtype"], f"{path}.dtype") != "float32":
        _fail("schema_validation_failed", f"{path}.dtype", "unexpected dtype")
    entries = _expect_sequence(mapping["entries"], f"{path}.entries")
    if len(entries) != 2:
        _fail("schema_validation_failed", f"{path}.entries", "expected two entries")
    checksums: list[str] = []
    for index, entry in enumerate(entries):
        entry_path = f"{path}.entries[{index}]"
        _validate_artifact_entry(entry, entry_path)
        entry_map = entry if isinstance(entry, Mapping) else {}
        if entry_map.get("tap_name") != taps[index]:
            _fail("schema_validation_failed", f"{entry_path}.tap_name", "order mismatch")
        checksums.append(str(entry_map["checksum"]))
    return checksums[0], checksums[1]


def _validate_manual_preprocessing_response(
    value: object,
    *,
    operation: str,
) -> Mapping[str, Any]:
    path = "response.manual_preprocessing"
    if operation == "artifact_boundary_self_test":
        mapping = _expect_exact_fields(
            value, frozenset({"executed", "profile_id", "normalized_sha256"}), path
        )
        if _expect_bool(mapping["executed"], f"{path}.executed"):
            _fail("schema_validation_failed", f"{path}.executed", "expected false")
        if mapping["normalized_sha256"] is not None:
            _fail("schema_validation_failed", f"{path}.normalized_sha256", "expected null")
    else:
        mapping = _expect_exact_fields(
            value,
            frozenset(
                {
                    "executed", "profile_id", "input_shape", "input_dtype", "input_sha256",
                    "input_unchanged", "output_shape", "output_dtype", "output_c_contiguous",
                    "output_finite", "normalized_sha256", "resize_count", "crop_count",
                    "pil_conversion_count", "image_processor_construction_count"
                }
            ),
            path,
        )
        for key in ("executed", "input_unchanged", "output_c_contiguous", "output_finite"):
            if not _expect_bool(mapping[key], f"{path}.{key}"):
                _fail("schema_validation_failed", f"{path}.{key}", "expected true")
        if _shape_tuple(mapping["input_shape"], f"{path}.input_shape", 4) != (2, 224, 224, 3):
            _fail("schema_validation_failed", f"{path}.input_shape", "unexpected shape")
        if _shape_tuple(mapping["output_shape"], f"{path}.output_shape", 4) != (2, 3, 224, 224):
            _fail("schema_validation_failed", f"{path}.output_shape", "unexpected shape")
        if _expect_string(mapping["input_dtype"], f"{path}.input_dtype") != "uint8":
            _fail("schema_validation_failed", f"{path}.input_dtype", "unexpected dtype")
        if _expect_string(mapping["output_dtype"], f"{path}.output_dtype") != "float32":
            _fail("schema_validation_failed", f"{path}.output_dtype", "unexpected dtype")
        _expect_sha256(mapping["input_sha256"], f"{path}.input_sha256")
        _expect_sha256(mapping["normalized_sha256"], f"{path}.normalized_sha256")
        for key in (
            "resize_count", "crop_count", "pil_conversion_count", "image_processor_construction_count"
        ):
            if _expect_int(mapping[key], f"{path}.{key}") != 0:
                _fail("schema_validation_failed", f"{path}.{key}", "expected zero")
    if _expect_identifier(mapping["profile_id"], f"{path}.profile_id") != "dinov3-manual-fp32-imagenet-v1":
        _fail("schema_validation_failed", f"{path}.profile_id", "unexpected profile")
    return mapping


def _validate_random_model_response(value: object, *, operation: str) -> Mapping[str, Any]:
    path = "response.random_model"
    if operation == "artifact_boundary_self_test":
        mapping = _expect_exact_fields(value, frozenset({"executed", "shapes"}), path)
        if _expect_bool(mapping["executed"], f"{path}.executed"):
            _fail("schema_validation_failed", f"{path}.executed", "expected false")
        if _expect_sequence(mapping["shapes"], f"{path}.shapes"):
            _fail("schema_validation_failed", f"{path}.shapes", "expected empty array")
        return mapping
    mapping = _expect_exact_fields(
        value,
        frozenset(
            {
                "executed", "class", "random_initialization_seed", "checkpoint_weight_bytes",
                "device", "dtype", "evaluation_mode", "frozen", "parameter_count",
                "parameter_tensor_count", "total_tokens", "hidden_state_count",
                "hidden_state_shapes", "last_hidden_shape", "pooler_shape", "finite",
                "repeat_exact", "last_hidden_sha256", "model_state_sha256_before",
                "model_state_sha256_after", "model_state_unchanged", "rng",
                "cuda_available", "cuda_device_count"
            }
        ),
        path,
    )
    true_fields = ("executed", "evaluation_mode", "frozen", "finite", "repeat_exact", "model_state_unchanged")
    for key in true_fields:
        if not _expect_bool(mapping[key], f"{path}.{key}"):
            _fail("schema_validation_failed", f"{path}.{key}", "expected true")
    if _expect_string(mapping["class"], f"{path}.class") != "DINOv3ViTModel":
        _fail("schema_validation_failed", f"{path}.class", "unexpected model")
    _expect_int(mapping["random_initialization_seed"], f"{path}.random_initialization_seed", maximum=2**32 - 1)
    expected_ints = {
        "checkpoint_weight_bytes": 0,
        "parameter_count": 85_660_416,
        "parameter_tensor_count": 211,
        "total_tokens": 201,
        "hidden_state_count": 13,
        "cuda_device_count": 0,
    }
    for key, expected in expected_ints.items():
        if _expect_int(mapping[key], f"{path}.{key}") != expected:
            _fail("schema_validation_failed", f"{path}.{key}", "unexpected value")
    if _expect_string(mapping["device"], f"{path}.device") != "cpu" or _expect_string(mapping["dtype"], f"{path}.dtype") != "float32":
        _fail("schema_validation_failed", path, "expected CPU float32")
    if _expect_bool(mapping["cuda_available"], f"{path}.cuda_available"):
        _fail("schema_validation_failed", f"{path}.cuda_available", "expected false")
    hidden = _expect_sequence(mapping["hidden_state_shapes"], f"{path}.hidden_state_shapes")
    if len(hidden) != 13 or any(_shape_tuple(item, f"{path}.hidden_state_shapes[{index}]", 3) != (2, 201, 768) for index, item in enumerate(hidden)):
        _fail("schema_validation_failed", f"{path}.hidden_state_shapes", "unexpected shapes")
    if _shape_tuple(mapping["last_hidden_shape"], f"{path}.last_hidden_shape", 3) != (2, 201, 768):
        _fail("schema_validation_failed", f"{path}.last_hidden_shape", "unexpected shape")
    if _shape_tuple(mapping["pooler_shape"], f"{path}.pooler_shape", 2) != (2, 768):
        _fail("schema_validation_failed", f"{path}.pooler_shape", "unexpected shape")
    for key in ("last_hidden_sha256", "model_state_sha256_before", "model_state_sha256_after"):
        _expect_sha256(mapping[key], f"{path}.{key}")
    if mapping["model_state_sha256_before"] != mapping["model_state_sha256_after"]:
        _fail("schema_validation_failed", path, "model state changed")
    rng = _expect_exact_fields(
        mapping["rng"],
        frozenset({"python_unchanged", "numpy_unchanged", "torch_unchanged"}),
        f"{path}.rng",
    )
    for key, item in rng.items():
        if not _expect_bool(item, f"{path}.rng.{key}"):
            _fail("schema_validation_failed", f"{path}.rng.{key}", "expected true")
    return mapping


def _validate_random_shapes(value: object, *, operation: str) -> Mapping[str, Any]:
    path = "response.random_model_shapes"
    mapping = _expect_exact_fields(
        value, frozenset({"last_hidden", "hidden_states", "pooler"}), path
    )
    hidden = _expect_sequence(mapping["hidden_states"], f"{path}.hidden_states")
    if operation == "artifact_boundary_self_test":
        if mapping["last_hidden"] is not None or mapping["pooler"] is not None or hidden:
            _fail("schema_validation_failed", path, "artifact-only shapes must be empty")
    else:
        if _shape_tuple(mapping["last_hidden"], f"{path}.last_hidden", 3) != (2, 201, 768):
            _fail("schema_validation_failed", f"{path}.last_hidden", "unexpected shape")
        if len(hidden) != 13 or any(_shape_tuple(item, f"{path}.hidden_states[{index}]", 3) != (2, 201, 768) for index, item in enumerate(hidden)):
            _fail("schema_validation_failed", f"{path}.hidden_states", "unexpected shapes")
        if _shape_tuple(mapping["pooler"], f"{path}.pooler", 2) != (2, 768):
            _fail("schema_validation_failed", f"{path}.pooler", "unexpected shape")
    return mapping


@dataclass(frozen=True, slots=True)
class WorkerResponse:
    """Immutable isolated-worker response contract."""

    schema_id: str
    operation: str
    status: str
    run_id: str
    runtime_id: str
    runtime_manifest_sha256: str | None
    candidate_freeze_sha256: str | None
    manual_preprocessing_hash: str | None
    manual_preprocessing: Mapping[str, Any]
    random_model_shapes: Mapping[str, Any]
    random_model: Mapping[str, Any]
    artifact_validation_result: Mapping[str, Any]
    inbound_checksums: tuple[str, ...]
    outbound_checksums: tuple[str, ...]
    socket_attempt_count: int
    model_weight_byte_count: int
    torchvision_use_count: int
    warnings: tuple[str, ...]
    failure_classification: str | None
    markers: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> "WorkerResponse":
        expected = frozenset(
            {
                "schema_id", "operation", "status", "run_id", "runtime_id",
                "runtime_manifest_sha256", "candidate_freeze_sha256",
                "manual_preprocessing_hash", "manual_preprocessing", "random_model_shapes",
                "random_model", "artifact_validation_result", "inbound_checksums",
                "outbound_checksums", "socket_attempt_count", "model_weight_byte_count",
                "torchvision_use_count", "warnings", "failure_classification", "markers"
            }
        )
        mapping = _expect_exact_fields(value, expected, "response")
        schema_id = _expect_string(mapping["schema_id"], "response.schema_id")
        operation = _expect_identifier(mapping["operation"], "response.operation")
        status = _expect_string(mapping["status"], "response.status")
        if schema_id != WORKER_RESPONSE_SCHEMA_ID:
            _fail("schema_validation_failed", "response.schema_id", "unexpected schema")
        if operation not in ALLOWED_WORKER_OPERATIONS:
            _fail("unknown_worker_operation", "response.operation", "operation is not allowed")
        if status not in ALLOWED_RESPONSE_STATUSES:
            _fail("schema_validation_failed", "response.status", "unexpected status")

        def optional_sha(field_name: str) -> str | None:
            raw = mapping[field_name]
            return None if raw is None else _expect_sha256(raw, f"response.{field_name}")

        manual = _validate_manual_preprocessing_response(
            mapping["manual_preprocessing"], operation=operation
        )
        random_shapes = _validate_random_shapes(
            mapping["random_model_shapes"], operation=operation
        )
        random_model = _validate_random_model_response(
            mapping["random_model"], operation=operation
        )
        artifact = _expect_exact_fields(
            mapping["artifact_validation_result"],
            frozenset({"inbound", "outbound"}),
            "response.artifact_validation_result",
        )
        inbound_from_artifact = _validate_artifact_summary(
            artifact["inbound"], "response.artifact_validation_result.inbound"
        )
        outbound_from_artifact = _validate_artifact_summary(
            artifact["outbound"], "response.artifact_validation_result.outbound"
        )
        inbound = tuple(
            _expect_sha256(item, f"response.inbound_checksums[{index}]")
            for index, item in enumerate(
                _expect_sequence(mapping["inbound_checksums"], "response.inbound_checksums")
            )
        )
        outbound = tuple(
            _expect_sha256(item, f"response.outbound_checksums[{index}]")
            for index, item in enumerate(
                _expect_sequence(mapping["outbound_checksums"], "response.outbound_checksums")
            )
        )
        if inbound != inbound_from_artifact or outbound != outbound_from_artifact:
            _fail("schema_validation_failed", "response", "artifact checksum lists disagree")
        warnings = _expect_string_tuple(mapping["warnings"], "response.warnings")
        markers = _expect_string_tuple(mapping["markers"], "response.markers")
        failure_raw = mapping["failure_classification"]
        result = cls(
            schema_id=schema_id,
            operation=operation,
            status=status,
            run_id=_expect_identifier(mapping["run_id"], "response.run_id"),
            runtime_id=_expect_identifier(mapping["runtime_id"], "response.runtime_id"),
            runtime_manifest_sha256=optional_sha("runtime_manifest_sha256"),
            candidate_freeze_sha256=optional_sha("candidate_freeze_sha256"),
            manual_preprocessing_hash=optional_sha("manual_preprocessing_hash"),
            manual_preprocessing=_freeze_json(manual),
            random_model_shapes=_freeze_json(random_shapes),
            random_model=_freeze_json(random_model),
            artifact_validation_result=_freeze_json(artifact),
            inbound_checksums=inbound,
            outbound_checksums=outbound,
            socket_attempt_count=_expect_int(mapping["socket_attempt_count"], "response.socket_attempt_count"),
            model_weight_byte_count=_expect_int(mapping["model_weight_byte_count"], "response.model_weight_byte_count"),
            torchvision_use_count=_expect_int(mapping["torchvision_use_count"], "response.torchvision_use_count"),
            warnings=warnings,
            failure_classification=(
                None if failure_raw is None else _expect_identifier(failure_raw, "response.failure_classification")
            ),
            markers=markers,
        )
        result._validate_cross_fields()
        _scan_response_strings(result.to_dict())
        return result

    def _validate_cross_fields(self) -> None:
        if self.status == "success":
            if self.runtime_manifest_sha256 is None or self.candidate_freeze_sha256 is None:
                _fail("schema_validation_failed", "response", "success requires runtime hashes")
            expected_manual_hash = self.manual_preprocessing["normalized_sha256"]
            if self.manual_preprocessing_hash != expected_manual_hash:
                _fail("schema_validation_failed", "response.manual_preprocessing_hash", "does not match detail")
            if self.failure_classification is not None or self.warnings:
                _fail("schema_validation_failed", "response", "success requires no failure or warnings")
            if self.socket_attempt_count or self.model_weight_byte_count or self.torchvision_use_count:
                _fail("schema_validation_failed", "response", "prohibited activity was reported")
            expected_markers = (
                _RUNTIME_MARKERS
                if self.operation == "runtime_self_test"
                else _ARTIFACT_MARKERS
            )
            if self.markers != expected_markers:
                _fail("schema_validation_failed", "response.markers", "unexpected marker order")
        elif self.failure_classification is None:
            _fail("schema_validation_failed", "response.failure_classification", "required on error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_validation_result": _thaw_json(self.artifact_validation_result),
            "candidate_freeze_sha256": self.candidate_freeze_sha256,
            "failure_classification": self.failure_classification,
            "inbound_checksums": list(self.inbound_checksums),
            "manual_preprocessing": _thaw_json(self.manual_preprocessing),
            "manual_preprocessing_hash": self.manual_preprocessing_hash,
            "markers": list(self.markers),
            "model_weight_byte_count": self.model_weight_byte_count,
            "operation": self.operation,
            "outbound_checksums": list(self.outbound_checksums),
            "random_model": _thaw_json(self.random_model),
            "random_model_shapes": _thaw_json(self.random_model_shapes),
            "run_id": self.run_id,
            "runtime_id": self.runtime_id,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "schema_id": self.schema_id,
            "socket_attempt_count": self.socket_attempt_count,
            "status": self.status,
            "torchvision_use_count": self.torchvision_use_count,
            "warnings": list(self.warnings),
        }


def load_runtime_profile(path: str | Path) -> RuntimeProfile:
    """Load and strictly validate one repository runtime profile."""

    _, payload = _read_bounded_regular_file(
        path, maximum_bytes=_MAX_PROFILE_BYTES, context="profile"
    )
    digest = hashlib.sha256(payload).hexdigest()
    value = _parse_json_bytes(payload, context="profile")
    return RuntimeProfile.from_dict(value, source_sha256=digest)


def load_runtime_manifest(path: str | Path) -> RuntimeManifest:
    """Load and strictly validate one external runtime manifest."""

    _, payload = _read_bounded_regular_file(
        path, maximum_bytes=_MAX_MANIFEST_BYTES, context="manifest"
    )
    digest = hashlib.sha256(payload).hexdigest()
    value = _parse_json_bytes(payload, context="manifest")
    return RuntimeManifest.from_dict(value, source_sha256=digest)


def validate_runtime_manifest(
    manifest: RuntimeManifest,
    profile: RuntimeProfile,
) -> None:
    """Cross-check an external manifest against its repository profile."""

    if not isinstance(manifest, RuntimeManifest) or not isinstance(profile, RuntimeProfile):
        _fail("schema_validation_failed", "manifest", "unexpected object type")
    dependency = profile.dependency_validation
    expected_dependency = {
        "active_requirement_count": dependency["active_requirement_count"],
        "inherited_conflict_count": dependency["expected_inherited_conflict_count"],
        "inherited_conflict_sha256": dependency[
            "expected_inherited_conflict_sha256"
        ],
        "new_conflict_count": dependency["expected_new_conflict_count"],
        "unsatisfied_active_requirement_count": dependency[
            "expected_unsatisfied_active_requirement_count"
        ],
    }
    checks = (
        (manifest.runtime_id, profile.runtime_id, "runtime_id"),
        (manifest.repository_profile_sha256, profile.source_sha256, "repository_profile_sha256"),
        (dict(manifest.shared_torch), dict(profile.current_base["torch"]), "shared_torch"),
        (
            dict(manifest.candidate_local_packages),
            dict(profile.candidate["local_packages"]),
            "candidate_local_packages",
        ),
        (
            dict(manifest.candidate_freeze),
            dict(profile.candidate["expected_freeze"]),
            "candidate_freeze",
        ),
        (manifest.candidate_freeze_sha256, profile.expected_candidate_freeze_sha256, "candidate_freeze_sha256"),
        (manifest.metadata_files, profile.metadata_files, "metadata_files"),
        (manifest.wheel_files, profile.wheel_files, "wheel_files"),
        (manifest.installed_sources, profile.installed_sources, "installed_sources"),
        (manifest.current_base_freeze_sha256, profile.current_base_freeze_sha256, "current_base_freeze_sha256"),
        (dict(manifest.dependency_validation), expected_dependency, "dependency_validation"),
        (
            manifest.manual_preprocessing_profile_id,
            profile.manual_preprocessing_profile_id,
            "manual_preprocessing_profile_id",
        ),
        (manifest.python_version, profile.candidate["python"]["version"], "python_version"),
        (manifest.forbidden_candidate_package_count, 0, "forbidden_candidate_package_count"),
    )
    for actual, expected, field_name in checks:
        if actual != expected:
            _fail(
                "runtime_identity_mismatch",
                f"manifest.{field_name}",
                "does not match the repository profile",
            )


def resolve_default_runtime_root(
    profile: RuntimeProfile,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the fixed LOCALAPPDATA runtime root without a home fallback."""

    if not isinstance(profile, RuntimeProfile):
        _fail("schema_validation_failed", "profile", "expected RuntimeProfile")
    environment = os.environ if environ is None else environ
    local_app_data = environment.get("LOCALAPPDATA")
    if not isinstance(local_app_data, str) or not local_app_data:
        _fail("runtime_root_unavailable", "LOCALAPPDATA", "is not set")
    base = Path(local_app_data)
    if not base.is_absolute():
        _fail("unsafe_path", "LOCALAPPDATA", "expected an absolute path")
    return base / "VEA" / "runtimes" / profile.runtime_id


def _resolved_directory(path: str | Path, context: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        _fail("unsafe_path", context, "expected an absolute path")
    if _is_reparse_point(candidate):
        _fail("unsafe_path", context, "symbolic links and reparse points are forbidden")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeBoundaryError(
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
    runtime_root: Path,
    repository_root: Path,
) -> None:
    if _contains(repository_root, runtime_root) or _contains(runtime_root, repository_root):
        _fail(
            "unsafe_runtime_root",
            "runtime_root",
            "runtime and repository roots must not overlap",
        )


def _safe_child(
    root: Path,
    relative_path: str,
    *,
    context: str,
    must_exist: bool,
    expect_directory: bool = False,
) -> Path:
    relative = _expect_relative_path(relative_path, context)
    current = root
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        is_leaf = index == len(parts) - 1
        if current.exists() or current.is_symlink():
            if _is_reparse_point(current):
                _fail("workspace_escape", context, "symbolic-link escape is forbidden")
        elif not is_leaf or must_exist:
            _fail("path_validation_failed", context, "required path does not exist")
    try:
        if must_exist:
            resolved = current.resolve(strict=True)
        else:
            parent = current.parent.resolve(strict=True)
            resolved = parent / current.name
    except OSError as exc:
        raise RuntimeBoundaryError(
            "path_validation_failed", f"{context}: unable to resolve path"
        ) from exc
    if not _contains(root, resolved):
        _fail("workspace_escape", context, "resolved path escapes its root")
    if must_exist:
        if expect_directory and not resolved.is_dir():
            _fail("unsafe_path", context, "expected a directory")
        if not expect_directory and not resolved.is_file():
            _fail("unsafe_path", context, "expected a regular file")
    elif resolved.exists() or resolved.is_symlink():
        _fail("unsafe_path", context, "output path already exists")
    return resolved


def resolve_workspace_relative_path(
    workspace_root: str | Path,
    relative_path: str,
    *,
    must_exist: bool,
    expect_directory: bool = False,
) -> Path:
    """Resolve one protocol path while rejecting traversal and symlink escape."""

    root = _resolved_directory(workspace_root, "workspace_root")
    return _safe_child(
        root,
        relative_path,
        context="workspace_path",
        must_exist=must_exist,
        expect_directory=expect_directory,
    )


def runtime_python_path(
    runtime_root: str | Path,
    *,
    interpreter_layout: str = "windows_scripts",
) -> Path:
    """Return the selected versioned venv Python path inside a runtime root."""

    root = _resolved_directory(runtime_root, "runtime_root")
    layouts = {
        "windows_scripts": "venv/Scripts/python.exe",
        "linux_bin": "venv/bin/python",
    }
    if interpreter_layout not in layouts:
        _fail(
            "schema_validation_failed",
            "interpreter_layout",
            "unknown interpreter layout",
        )
    return _safe_child(
        root,
        layouts[interpreter_layout],
        context="runtime_python",
        must_exist=True,
    )


def _verify_file_identity(path: Path, identity: FileIdentity, context: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeBoundaryError(
            "runtime_identity_mismatch", f"{context}: unable to inspect file"
        ) from exc
    if size != identity.size_bytes:
        _fail("runtime_identity_mismatch", context, "file size does not match profile")
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, identity.sha256):
        _fail("runtime_identity_mismatch", context, "SHA-256 does not match profile")


def _validate_exact_directory_entries(
    directory: Path,
    expected: set[str],
    context: str,
) -> None:
    try:
        actual = {item.name for item in directory.iterdir()}
    except OSError as exc:
        raise RuntimeBoundaryError(
            "path_validation_failed", f"{context}: unable to list directory"
        ) from exc
    if actual != expected:
        _fail(
            "runtime_layout_mismatch",
            context,
            "directory entries do not match the exact runtime layout",
        )


def validate_runtime_root(
    runtime_root: str | Path,
    profile: RuntimeProfile,
    *,
    repository_root: str | Path,
) -> RuntimeManifest:
    """Read-only validation of the external runtime's static identity and layout."""

    root = _resolved_directory(runtime_root, "runtime_root")
    repository = _resolved_directory(repository_root, "repository_root")
    _reject_overlapping_roots(root, repository)
    if root.name != profile.runtime_id:
        _fail("runtime_identity_mismatch", "runtime_root", "basename does not match runtime ID")
    _validate_exact_directory_entries(
        root,
        {"runtime-manifest.json", "metadata", "wheelhouse", "venv"},
        "runtime_root",
    )
    metadata_directory = _safe_child(
        root, "metadata", context="runtime.metadata", must_exist=True, expect_directory=True
    )
    wheelhouse_directory = _safe_child(
        root, "wheelhouse", context="runtime.wheelhouse", must_exist=True, expect_directory=True
    )
    _validate_exact_directory_entries(
        metadata_directory,
        {identity.filename for identity in profile.metadata},
        "runtime.metadata",
    )
    _validate_exact_directory_entries(
        wheelhouse_directory,
        {identity.filename for identity in profile.wheels},
        "runtime.wheelhouse",
    )
    for identity in profile.metadata:
        path = _safe_child(
            root,
            f"metadata/{identity.filename}",
            context=f"runtime.metadata.{identity.filename}",
            must_exist=True,
        )
        _verify_file_identity(path, identity, f"runtime.metadata.{identity.filename}")
    for identity in profile.wheels:
        path = _safe_child(
            root,
            f"wheelhouse/{identity.filename}",
            context=f"runtime.wheelhouse.{identity.filename}",
            must_exist=True,
        )
        _verify_file_identity(path, identity, f"runtime.wheelhouse.{identity.filename}")
    runtime_python_path(root)
    pyvenv = _safe_child(
        root, "venv/pyvenv.cfg", context="runtime.pyvenv_cfg", must_exist=True
    )
    try:
        config_text = pyvenv.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeBoundaryError(
            "runtime_identity_mismatch", "runtime.pyvenv_cfg: unable to read file"
        ) from exc
    normalized = "".join(config_text.lower().split())
    if "include-system-site-packages=true" not in normalized:
        _fail(
            "runtime_identity_mismatch",
            "runtime.pyvenv_cfg",
            "system-site-packages is not enabled",
        )
    manifest_path = _safe_child(
        root,
        "runtime-manifest.json",
        context="runtime.manifest",
        must_exist=True,
    )
    manifest = load_runtime_manifest(manifest_path)
    validate_runtime_manifest(manifest, profile)
    return manifest


def _validate_request_against_profile(
    request: WorkerRequest,
    profile: RuntimeProfile,
) -> None:
    if request.runtime_profile_sha256 != profile.source_sha256:
        _fail(
            "request_profile_mismatch",
            "request.runtime_profile_sha256",
            "does not match loaded profile",
        )
    if (
        request.expected_preprocessing_profile_id
        != profile.manual_preprocessing_profile_id
    ):
        _fail(
            "request_profile_mismatch",
            "request.expected_preprocessing_profile_id",
            "does not match loaded profile",
        )


def write_worker_request(
    path: str | Path,
    request: WorkerRequest,
    *,
    workspace_root: str | Path,
) -> Path:
    """Exclusively write canonical request JSON inside a worker workspace."""

    if not isinstance(request, WorkerRequest):
        _fail("schema_validation_failed", "request", "expected WorkerRequest")
    validated = WorkerRequest.from_dict(request.to_dict())
    root = _resolved_directory(workspace_root, "workspace_root")
    candidate = Path(path)
    if not candidate.is_absolute():
        _fail("unsafe_path", "request_path", "expected an absolute path")
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        _fail("workspace_escape", "request_path", "must be inside workspace")
    destination = _safe_child(
        root,
        relative,
        context="request_path",
        must_exist=False,
    )
    payload = _canonical_json_bytes(validated.to_dict())
    if len(payload) > _MAX_REQUEST_BYTES:
        _fail("invalid_json", "request", "serialized request is too large")
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise RuntimeBoundaryError(
            "request_write_failed", "request: unable to write request file"
        ) from exc
    return destination


def read_worker_response(
    path: str | Path,
    *,
    workspace_root: str | Path,
) -> WorkerResponse:
    """Read strict response JSON from inside the supplied worker workspace."""

    root = _resolved_directory(workspace_root, "workspace_root")
    candidate = Path(path)
    if not candidate.is_absolute():
        _fail("unsafe_path", "response_path", "expected an absolute path")
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        _fail("workspace_escape", "response_path", "must be inside workspace")
    resolved = _safe_child(
        root,
        relative,
        context="response_path",
        must_exist=True,
    )
    _, payload = _read_bounded_regular_file(
        resolved,
        maximum_bytes=_MAX_RESPONSE_BYTES,
        context="response",
    )
    return WorkerResponse.from_dict(_parse_json_bytes(payload, context="response"))


def build_sanitized_worker_environment(
    *,
    runtime_python: str | Path,
    repository_root: str | Path,
    workspace_root: str | Path,
    environ: Mapping[str, str] | None = None,
    cuda_visible_devices: str = "",
) -> dict[str, str]:
    """Build a minimal offline environment without inherited secret variables."""

    if cuda_visible_devices not in {"", "0"}:
        _fail(
            "environment_sanitization_failed",
            "cuda_visible_devices",
            "only the hidden legacy device view or selected device zero is allowed",
        )
    python_path = Path(runtime_python)
    repository = _resolved_directory(repository_root, "repository_root")
    workspace = _resolved_directory(workspace_root, "workspace_root")
    base = os.environ if environ is None else environ
    result: dict[str, str] = {}
    for key in ("SystemRoot", "WINDIR", "COMSPEC", "PATHEXT"):
        value = base.get(key)
        if isinstance(value, str) and value and not any(
            fragment in key.upper() for fragment in _SECRET_ENV_FRAGMENTS
        ):
            result[key] = value
    system_root = result.get("SystemRoot") or result.get("WINDIR")
    safe_path_parts = [str(python_path.parent)]
    if system_root:
        safe_path_parts.extend([str(Path(system_root) / "System32"), system_root])
    scratch = workspace / ".runtime-scratch"
    temporary = scratch / "tmp"
    cache = scratch / "cache"
    try:
        temporary.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeBoundaryError(
            "workspace_setup_failed", "workspace: unable to create runtime scratch"
        ) from exc
    result.update(
        {
            "PATH": os.pathsep.join(safe_path_parts),
            "PYTHONPATH": str(repository / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(cache / "huggingface"),
            "XDG_CACHE_HOME": str(cache),
            "TEMP": str(temporary),
            "TMP": str(temporary),
        }
    )
    if any(
        key != "TOKENIZERS_PARALLELISM"
        and any(fragment in key.upper() for fragment in _SECRET_ENV_FRAGMENTS)
        for key in result
    ):
        _fail("environment_sanitization_failed", "environment", "secret-like key survived filtering")
    return result


def _resolved_repository_file(
    path: str | Path,
    repository_root: Path,
    context: str,
) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        _fail("unsafe_path", context, "expected an absolute path")
    try:
        relative = candidate.relative_to(repository_root).as_posix()
    except ValueError:
        _fail("unsafe_path", context, "must be inside repository")
    return _safe_child(
        repository_root,
        relative,
        context=context,
        must_exist=True,
    )


def _validate_response_correlation(
    response: WorkerResponse,
    request: WorkerRequest,
    profile: RuntimeProfile,
    manifest: RuntimeManifest,
) -> None:
    checks = (
        (response.operation, request.operation, "operation"),
        (response.run_id, request.run_id, "run_id"),
        (response.runtime_id, profile.runtime_id, "runtime_id"),
        (response.runtime_manifest_sha256, manifest.source_sha256, "runtime_manifest_sha256"),
        (response.candidate_freeze_sha256, profile.expected_candidate_freeze_sha256, "candidate_freeze_sha256"),
    )
    for actual, expected, field_name in checks:
        if actual != expected:
            _fail(
                "worker_response_mismatch",
                f"response.{field_name}",
                "does not match the invocation contract",
            )
    if (
        response.manual_preprocessing["profile_id"]
        != request.expected_preprocessing_profile_id
    ):
        _fail(
            "worker_response_mismatch",
            "response.manual_preprocessing.profile_id",
            "does not match the request",
        )
    if request.operation == "runtime_self_test":
        runtime_checks = (
            (
                response.manual_preprocessing_hash,
                profile.manual_preprocessing["expected_output_sha256"],
                "manual_preprocessing_hash",
            ),
            (
                response.manual_preprocessing["input_sha256"],
                profile.manual_preprocessing["expected_input_sha256"],
                "manual_preprocessing.input_sha256",
            ),
            (
                response.random_model["random_initialization_seed"],
                request.deterministic_seed,
                "random_model.random_initialization_seed",
            ),
        )
        for actual, expected, field_name in runtime_checks:
            if actual != expected:
                _fail(
                    "worker_response_mismatch",
                    f"response.{field_name}",
                    "does not match the invocation contract",
                )
    if response.status != "success":
        _fail(
            "worker_reported_failure",
            "response.status",
            response.failure_classification or "worker returned an error",
        )


def invoke_isolated_worker(
    *,
    runtime_root: str | Path,
    repository_root: str | Path,
    profile_path: str | Path,
    worker_script: str | Path,
    workspace_root: str | Path,
    request: WorkerRequest,
    timeout_seconds: float = 600.0,
    environ: Mapping[str, str] | None = None,
) -> WorkerResponse:
    """Invoke the isolated worker without a shell and validate its response."""

    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        _fail("schema_validation_failed", "timeout_seconds", "expected a positive finite timeout")
    repository = _resolved_directory(repository_root, "repository_root")
    workspace = _resolved_directory(workspace_root, "workspace_root")
    runtime = _resolved_directory(runtime_root, "runtime_root")
    _reject_overlapping_roots(runtime, repository)
    if (
        _contains(repository, workspace)
        or _contains(workspace, repository)
        or _contains(runtime, workspace)
        or _contains(workspace, runtime)
    ):
        _fail("unsafe_workspace", "workspace_root", "workspace must be outside repository and runtime")
    profile_file = _resolved_repository_file(profile_path, repository, "profile_path")
    worker_file = _resolved_repository_file(worker_script, repository, "worker_script")
    profile = load_runtime_profile(profile_file)
    _validate_request_against_profile(request, profile)
    manifest = validate_runtime_root(runtime, profile, repository_root=repository)
    resolve_workspace_relative_path(
        workspace,
        request.workspace_relative_inbound_artifact_path,
        must_exist=True,
        expect_directory=True,
    )
    resolve_workspace_relative_path(
        workspace,
        request.workspace_relative_outbound_artifact_path,
        must_exist=False,
    )
    request_file = workspace / "worker-request.json"
    response_file = workspace / "worker-response.json"
    if request_file.exists() or response_file.exists():
        _fail("unsafe_path", "workspace", "protocol files already exist")
    write_worker_request(request_file, request, workspace_root=workspace)
    python = runtime_python_path(runtime)
    environment = build_sanitized_worker_environment(
        runtime_python=python,
        repository_root=repository,
        workspace_root=workspace,
        environ=environ,
    )
    command = [
        str(python),
        "-B",
        str(worker_file),
        "--profile",
        str(profile_file),
        "--runtime-root",
        str(runtime),
        "--workspace",
        str(workspace),
        "--request",
        request_file.relative_to(workspace).as_posix(),
        "--response",
        response_file.relative_to(workspace).as_posix(),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            shell=False,
            check=False,
            capture_output=True,
            text=False,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeBoundaryError(
            "worker_timeout", "isolated worker exceeded its timeout"
        ) from exc
    except OSError as exc:
        raise RuntimeBoundaryError(
            "worker_launch_failed", "isolated worker could not be launched"
        ) from exc
    if result.returncode != 0:
        cause = subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
        raise RuntimeBoundaryError(
            "worker_nonzero_exit",
            f"isolated worker exited with code {result.returncode}",
        ) from cause
    try:
        response = read_worker_response(response_file, workspace_root=workspace)
    except RuntimeBoundaryError as exc:
        raise RuntimeBoundaryError(
            "worker_response_invalid", "isolated worker response is invalid"
        ) from exc
    _validate_response_correlation(response, request, profile, manifest)
    return response


__all__ = [
    "ALLOWED_WORKER_OPERATIONS",
    "RUNTIME_MANIFEST_SCHEMA_ID",
    "RUNTIME_PROFILE_SCHEMA_ID",
    "WORKER_REQUEST_SCHEMA_ID",
    "WORKER_RESPONSE_SCHEMA_ID",
    "FileIdentity",
    "FrozenMapping",
    "InstalledSourceIdentity",
    "PackageIdentity",
    "RuntimeBoundaryError",
    "RuntimeManifest",
    "RuntimeProfile",
    "WheelIdentity",
    "WorkerRequest",
    "WorkerResponse",
    "build_sanitized_worker_environment",
    "invoke_isolated_worker",
    "load_runtime_manifest",
    "load_runtime_profile",
    "read_worker_response",
    "resolve_default_runtime_root",
    "resolve_workspace_relative_path",
    "runtime_python_path",
    "sha256_file",
    "validate_runtime_manifest",
    "validate_runtime_root",
    "write_worker_request",
]
