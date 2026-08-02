"""Import-light contracts for the bounded DINOv3 semantic probe.

The current process uses this module to validate the immutable semantic profile,
to exchange small JSON control messages with the persistent runtime, and to
validate tensor *metadata*.  Numeric/model packages are deliberately absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any


SEMANTIC_PROFILE_SCHEMA_ID = "vea.dinov3_semantic_probe_profile.v0"
SEMANTIC_REQUEST_SCHEMA_ID = (
    "vea.dinov3_pretrained_semantic_probe_request.v0"
)
SEMANTIC_RESPONSE_SCHEMA_ID = (
    "vea.dinov3_pretrained_semantic_probe_response.v0"
)
SEMANTIC_PROFILE_ID = "dinov3-vitb16-r5931719-pretrained-semantics-v0"
SEMANTIC_SUCCESS_MARKER = "DINOV3_PRETRAINED_RUNTIME_SEMANTICS_OK"

MODEL_KEY = "dinov3_vit_b16"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
INPUT_SHA256 = "b0fa2f6e3b2fbca50cde7e4a2dc3a65a0f788a184aec382365e520d54b468411"
NORMALIZED_SHA256 = (
    "c0ebe2af6db2e61d052dc58272379beef749a4c5392d4e2204a11862cf172526"
)
MODEL_STATE_SHA256 = (
    "c382afd66ac658255f9f1406bad1e5e5fcf410af40cbc5c306b770792a4f30fd"
)
SPECIAL_TOKEN_POLICY = "dinov3_cls1_register4_prefix_verified_runtime_v1"

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
SELECTED_ARTIFACT_TAPS = ("embedding_output", "block_06", "final_norm")
REQUIRED_SEMANTIC_MARKERS = (
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
    SEMANTIC_SUCCESS_MARKER,
)

_EXPECTED_IDENTITIES = {
    "external_snapshot_manifest_sha256": (
        "d5de6075023ca0b7a5d2f4fb1fa04add8f489f9d9613671c9cb475b3045caa25"
    ),
    "persistent_freeze_sha256": (
        "995687ed934c0a94b52a040af048f87bc8de4e5089475f70565ad37f1f33ecd9"
    ),
    "persistent_runtime_manifest_sha256": (
        "3b17f1158a9d9db4b6213904825528b0a593839be44fe05cc6336514a154d80a"
    ),
    "registry_sha256": (
        "2c75cc26a27febc633b2bad479779327f35abbf36160df2805f43a4e33f48244"
    ),
    "runtime_profile_sha256": (
        "1fe6ba3ef691e97cfb1cb8f3ffc7af7a4590f0bf75de0686c3536df4828900c4"
    ),
    "snapshot_profile_sha256": (
        "e7181b80924f3a95d64a83a3ec2fee88a872de70330656ea6c88a0fe2ad11308"
    ),
}
_EXPECTED_CHECKPOINT_FILES = (
    ("config.json", 744, "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec"),
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
_EXPECTED_INSTALLED_SOURCES = (
    (
        "transformers/models/dinov3_vit/configuration_dinov3_vit.py",
        "1ac7cb889e2314e8cadf9b0bed43d42cfb05dce6d730f719be4380c8a10a8a46",
    ),
    (
        "transformers/models/dinov3_vit/modeling_dinov3_vit.py",
        "00652d90ff8c7ae32b047ebe2acd2271f97e03f768cf570ac051045a4e0d5cf5",
    ),
    (
        "transformers/models/dinov3_vit/image_processing_dinov3_vit_fast.py",
        "5a84a6a7283b77e0a0b000851150dd9d04dc54a11baa3d2cd3e2603d4669ec80",
    ),
    (
        "transformers/image_transforms.py",
        "163556a472733191afe8cd86e08348d599bd88eb2370a3747ca687b1c14a6691",
    ),
)
_EXPECTED_INPUT_CONSTRUCTION = (
    "y, x = np.indices((224, 224), dtype=np.uint16)",
    "gradient = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)",
    "checker_base = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8); checker = np.stack((checker_base, (3 * x + 5 * y) % 256, np.where(checker_base == 0, (7 * x) % 256, (11 * y) % 256)), axis=-1).astype(np.uint8)",
)
_EXPECTED_FORMULA_STEPS = (
    "copy uint8 NHWC input to float32",
    "multiply by np.float32(1.0 / 255.0)",
    "subtract ImageNet mean as float32",
    "divide by ImageNet standard deviation as float32",
    "transpose NHWC to NCHW",
    "materialize C-contiguous float32 output",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAX_PROFILE_BYTES = 256 * 1024
_MAX_REQUEST_BYTES = 16 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_SECRET_KEY_PATTERN = re.compile(
    r"(?i)(?:^|_)(?:access_token|auth_token|api_key|secret|password|passwd|credential)(?:$|_)"
)
_RAW_VALUE_KEYS = frozenset(
    {"raw_tensor", "tensor_values", "values", "weight_content", "checkpoint_content"}
)
_TOKEN_VALUE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+\S+|hf_[A-Za-z0-9]{10,}|(?:api[_-]?key|password|secret)\s*[:=])"
)


class DINOv3SemanticBoundaryError(RuntimeError):
    """Raised when the semantic-probe control boundary is violated."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def _fail(classification: str, path: str, message: str) -> None:
    raise DINOv3SemanticBoundaryError(classification, f"{path}: {message}")


class FrozenJSONMap(Mapping[str, Any]):
    """A recursively immutable mapping with convenient attribute access."""

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


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DINOv3SemanticBoundaryError(
            "invalid_json", "document is not finite JSON data"
        ) from exc


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("invalid_json", "json", f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail("invalid_json", "json", f"non-finite numeric value {value!r}")


def _parse_json(payload: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except DINOv3SemanticBoundaryError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DINOv3SemanticBoundaryError(
            "invalid_json", f"{context}: invalid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        _fail("invalid_json", context, "expected a JSON object")
    return value


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
        raise DINOv3SemanticBoundaryError(
            "unsafe_path", "unable to inspect path"
        ) from exc


def _read_regular_file(
    path: str | Path, *, maximum_bytes: int, context: str
) -> tuple[Path, bytes]:
    candidate = Path(path)
    if not candidate.is_absolute():
        _fail("unsafe_path", context, "expected an absolute path")
    if _is_reparse(candidate):
        _fail("unsafe_path", context, "symlinks and reparse points are forbidden")
    try:
        resolved = candidate.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise DINOv3SemanticBoundaryError(
            "io_error", f"{context}: file is not readable"
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        _fail("unsafe_path", context, "expected a regular file")
    if info.st_size > maximum_bytes:
        _fail("invalid_json", context, f"file exceeds {maximum_bytes} bytes")
    try:
        return resolved, resolved.read_bytes()
    except OSError as exc:
        raise DINOv3SemanticBoundaryError(
            "io_error", f"{context}: file is not readable"
        ) from exc


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without importing a runtime package."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DINOv3SemanticBoundaryError("io_error", "unable to hash file") from exc
    return digest.hexdigest()


def _expect_mapping(
    value: object, expected: frozenset[str], path: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("schema_validation_failed", path, "expected an object")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _fail("schema_validation_failed", path, f"missing fields: {', '.join(missing)}")
    if unknown:
        _fail("schema_validation_failed", path, f"unknown fields: {', '.join(unknown)}")
    return value


def _expect_sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        _fail("schema_validation_failed", path, "expected an array")
    return value


def _expect_tuple(value: object, path: str) -> tuple[Any, ...]:
    return tuple(_expect_sequence(value, path))


def _expect_string(value: object, path: str, *, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        _fail("schema_validation_failed", path, "expected a non-empty bounded string")
    return value


def _expect_identifier(value: object, path: str) -> str:
    text = _expect_string(value, path, maximum=128)
    if _IDENTIFIER_PATTERN.fullmatch(text) is None:
        _fail("schema_validation_failed", path, "expected a safe identifier")
    return text


def _expect_sha256(value: object, path: str) -> str:
    text = _expect_string(value, path, maximum=64)
    if _SHA256_PATTERN.fullmatch(text) is None:
        _fail("schema_validation_failed", path, "expected lowercase SHA-256")
    return text


def _expect_int(value: object, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail("schema_validation_failed", path, f"expected integer >= {minimum}")
    return value


def _expect_float(value: object, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("schema_validation_failed", path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        _fail("schema_validation_failed", path, "expected a finite non-negative number")
    return result


def _expect_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail("schema_validation_failed", path, "expected a boolean")
    return value


def _expect_exact(value: object, expected: object, path: str) -> None:
    if value != expected:
        _fail("schema_validation_failed", path, f"expected {expected!r}")


def _expect_shape(value: object, expected: tuple[int, ...], path: str) -> None:
    sequence = _expect_sequence(value, path)
    shape = tuple(_expect_int(item, f"{path}[{index}]", minimum=1) for index, item in enumerate(sequence))
    if shape != expected:
        _fail("semantic_validation_failed", path, f"expected shape {list(expected)}")


def _expect_relative_path(value: object, path: str) -> str:
    text = _expect_string(value, path, maximum=512)
    if "\\" in text:
        _fail("unsafe_path", path, "backslashes are forbidden")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        _fail("unsafe_path", path, "absolute paths are forbidden")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail("unsafe_path", path, "path traversal is forbidden")
    if any(":" in part or "\x00" in part for part in parts):
        _fail("unsafe_path", path, "unsafe path component")
    return text


def _scan_sensitive(value: object, path: str = "document") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SECRET_KEY_PATTERN.search(key) is not None:
                _fail("sensitive_data_rejected", f"{path}.{key}", "secret-like field")
            if key in _RAW_VALUE_KEYS:
                _fail("sensitive_data_rejected", f"{path}.{key}", "raw values are forbidden")
            _scan_sensitive(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_sensitive(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if _TOKEN_VALUE_PATTERN.search(value) is not None:
            _fail("sensitive_data_rejected", path, "secret-like value")
        windows = PureWindowsPath(value)
        if PurePosixPath(value).is_absolute() or windows.is_absolute() or windows.drive:
            _fail("sensitive_data_rejected", path, "absolute path value")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail("invalid_json", path, "non-finite value")


def resolve_workspace_relative_path(
    workspace_root: str | Path,
    relative_path: str,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a protocol path without allowing traversal or reparse escape."""

    root = Path(workspace_root)
    if not root.is_absolute():
        _fail("unsafe_path", "workspace_root", "expected an absolute path")
    if _is_reparse(root):
        _fail("unsafe_path", "workspace_root", "reparse workspace is forbidden")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise DINOv3SemanticBoundaryError(
            "unsafe_path", "workspace_root is not accessible"
        ) from exc
    relative = _expect_relative_path(relative_path, "workspace_relative_path")
    candidate = resolved_root.joinpath(*relative.split("/"))
    current = resolved_root
    for part in relative.split("/"):
        current = current / part
        if current.exists() or current.is_symlink():
            if _is_reparse(current):
                _fail("unsafe_path", "workspace_relative_path", "reparse component")
    if must_exist and not candidate.exists():
        _fail("unsafe_path", "workspace_relative_path", "path does not exist")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise DINOv3SemanticBoundaryError(
            "unsafe_path", "workspace_relative_path is not accessible"
        ) from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        _fail("unsafe_path", "workspace_relative_path", "path escapes workspace")
    return resolved


@dataclass(frozen=True, slots=True)
class DINOv3SemanticProfile:
    """Validated, immutable semantic-probe profile."""

    document: FrozenJSONMap
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.document)

    @property
    def source_sha256(self) -> str:
        """Compatibility name used by the existing runtime boundaries."""

        return self.sha256

    @property
    def model(self) -> FrozenJSONMap:
        return self.document.model

    @property
    def identities(self) -> FrozenJSONMap:
        return self.document.identities

    @property
    def canonical_input(self) -> FrozenJSONMap:
        return self.document.canonical_input

    @property
    def geometry(self) -> FrozenJSONMap:
        return self.document.geometry

    @property
    def taps(self) -> FrozenJSONMap:
        return self.document.taps

    @property
    def tolerances(self) -> FrozenJSONMap:
        return self.document.tolerances

    @property
    def tap_catalog(self) -> tuple[str, ...]:
        return tuple(self.taps.catalog)

    @property
    def selected_artifact_taps(self) -> tuple[str, ...]:
        return tuple(self.taps.artifact_selected)

    @property
    def patch_token_indices(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.geometry.candidate_patch_start_index,
                self.geometry.candidate_patch_end_index + 1,
            )
        )

    @property
    def special_token_indices(self) -> tuple[int, ...]:
        return tuple(self.geometry.candidate_cls_indices) + tuple(
            self.geometry.candidate_register_indices
        )


@dataclass(frozen=True, slots=True)
class DINOv3SemanticRequest:
    """Validated immutable persistent-probe request."""

    run_id: str
    semantic_profile_sha256: str
    workspace_relative_artifact_directory: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schema_id": SEMANTIC_REQUEST_SCHEMA_ID,
            "semantic_profile_sha256": self.semantic_profile_sha256,
            "workspace_relative_artifact_directory": (
                self.workspace_relative_artifact_directory
            ),
        }


@dataclass(frozen=True, slots=True)
class DINOv3SemanticResponse:
    """Validated immutable semantic response containing metadata only."""

    document: FrozenJSONMap

    @property
    def run_id(self) -> str:
        return self.document.run_id

    @property
    def status(self) -> str:
        return self.document.status

    @property
    def taps(self) -> tuple[FrozenJSONMap, ...]:
        return tuple(self.document.taps)

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.document)

    def semantic_projection(self) -> dict[str, Any]:
        result = self.to_dict()
        del result["run_id"]
        return result


def _validate_profile(value: Mapping[str, Any]) -> None:
    profile = _expect_mapping(
        value,
        frozenset(
            {
                "authorization",
                "canonical_input",
                "checkpoint_files",
                "geometry",
                "identities",
                "installed_sources",
                "manual_preprocessing",
                "model",
                "policies",
                "profile_id",
                "protocol",
                "schema_id",
                "taps",
                "tolerances",
            }
        ),
        "profile",
    )
    _expect_exact(profile["schema_id"], SEMANTIC_PROFILE_SCHEMA_ID, "profile.schema_id")
    _expect_exact(profile["profile_id"], SEMANTIC_PROFILE_ID, "profile.profile_id")

    model = _expect_mapping(
        profile["model"],
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
    expected_model = {
        "config_class": "DINOv3ViTConfig",
        "expected_parameter_count": 85_660_416,
        "expected_parameter_tensor_count": 211,
        "key": MODEL_KEY,
        "model_class": "DINOv3ViTModel",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "state_sha256": MODEL_STATE_SHA256,
    }
    for key, expected in expected_model.items():
        if key.endswith("sha256"):
            _expect_sha256(model[key], f"profile.model.{key}")
        _expect_exact(model[key], expected, f"profile.model.{key}")
    if _GIT_SHA_PATTERN.fullmatch(str(model["revision"])) is None:
        _fail("schema_validation_failed", "profile.model.revision", "expected exact revision")

    identities = _expect_mapping(
        profile["identities"], frozenset(_EXPECTED_IDENTITIES), "profile.identities"
    )
    for key, expected in _EXPECTED_IDENTITIES.items():
        _expect_sha256(identities[key], f"profile.identities.{key}")
        _expect_exact(identities[key], expected, f"profile.identities.{key}")

    checkpoint_files = _expect_sequence(profile["checkpoint_files"], "profile.checkpoint_files")
    actual_checkpoint: list[tuple[str, int, str]] = []
    for index, item in enumerate(checkpoint_files):
        path = f"profile.checkpoint_files[{index}]"
        entry = _expect_mapping(item, frozenset({"bytes", "filename", "sha256"}), path)
        filename = _expect_string(entry["filename"], f"{path}.filename", maximum=128)
        if PurePosixPath(filename).name != filename or PureWindowsPath(filename).name != filename:
            _fail("unsafe_path", f"{path}.filename", "expected a basename")
        actual_checkpoint.append(
            (
                filename,
                _expect_int(entry["bytes"], f"{path}.bytes", minimum=1),
                _expect_sha256(entry["sha256"], f"{path}.sha256"),
            )
        )
    _expect_exact(tuple(actual_checkpoint), _EXPECTED_CHECKPOINT_FILES, "profile.checkpoint_files")

    installed_sources = _expect_sequence(profile["installed_sources"], "profile.installed_sources")
    actual_sources: list[tuple[str, str]] = []
    for index, item in enumerate(installed_sources):
        path = f"profile.installed_sources[{index}]"
        entry = _expect_mapping(item, frozenset({"path", "sha256"}), path)
        actual_sources.append(
            (
                _expect_relative_path(entry["path"], f"{path}.path"),
                _expect_sha256(entry["sha256"], f"{path}.sha256"),
            )
        )
    _expect_exact(tuple(actual_sources), _EXPECTED_INSTALLED_SOURCES, "profile.installed_sources")

    canonical = _expect_mapping(
        profile["canonical_input"],
        frozenset({"construction", "dtype", "image_ids", "normalized", "sha256", "shape"}),
        "profile.canonical_input",
    )
    construction = _expect_sequence(canonical["construction"], "profile.canonical_input.construction")
    for index, formula in enumerate(construction):
        _expect_string(formula, f"profile.canonical_input.construction[{index}]", maximum=1024)
    _expect_exact(
        tuple(construction),
        _EXPECTED_INPUT_CONSTRUCTION,
        "profile.canonical_input.construction",
    )
    _expect_exact(canonical["dtype"], "uint8", "profile.canonical_input.dtype")
    _expect_exact(
        _expect_tuple(canonical["image_ids"], "profile.canonical_input.image_ids"),
        ("image-b", "image-a"),
        "profile.canonical_input.image_ids",
    )
    _expect_shape(canonical["shape"], (2, 224, 224, 3), "profile.canonical_input.shape")
    _expect_sha256(canonical["sha256"], "profile.canonical_input.sha256")
    _expect_exact(canonical["sha256"], INPUT_SHA256, "profile.canonical_input.sha256")
    normalized = _expect_mapping(
        canonical["normalized"],
        frozenset({"c_contiguous", "dtype", "finite", "sha256", "shape"}),
        "profile.canonical_input.normalized",
    )
    _expect_exact(normalized["c_contiguous"], True, "profile.canonical_input.normalized.c_contiguous")
    _expect_exact(normalized["dtype"], "float32", "profile.canonical_input.normalized.dtype")
    _expect_exact(normalized["finite"], True, "profile.canonical_input.normalized.finite")
    _expect_sha256(normalized["sha256"], "profile.canonical_input.normalized.sha256")
    _expect_exact(normalized["sha256"], NORMALIZED_SHA256, "profile.canonical_input.normalized.sha256")
    _expect_shape(normalized["shape"], (2, 3, 224, 224), "profile.canonical_input.normalized.shape")

    geometry = _expect_mapping(
        profile["geometry"],
        frozenset(
            {
                "candidate_cls_indices",
                "candidate_patch_end_index",
                "candidate_patch_start_index",
                "candidate_register_indices",
                "grid_height",
                "grid_width",
                "hidden_size",
                "input_height",
                "input_width",
                "layer_count",
                "patch_count",
                "patch_size",
                "special_token_policy",
                "total_token_count",
            }
        ),
        "profile.geometry",
    )
    expected_geometry = {
        "candidate_cls_indices": [0],
        "candidate_patch_end_index": 200,
        "candidate_patch_start_index": 5,
        "candidate_register_indices": [1, 2, 3, 4],
        "grid_height": 14,
        "grid_width": 14,
        "hidden_size": 768,
        "input_height": 224,
        "input_width": 224,
        "layer_count": 12,
        "patch_count": 196,
        "patch_size": 16,
        "special_token_policy": SPECIAL_TOKEN_POLICY,
        "total_token_count": 201,
    }
    for key, expected in expected_geometry.items():
        _expect_exact(geometry[key], expected, f"profile.geometry.{key}")

    taps = _expect_mapping(
        profile["taps"], frozenset({"artifact_selected", "catalog"}), "profile.taps"
    )
    _expect_exact(
        _expect_tuple(taps["catalog"], "profile.taps.catalog"),
        TAP_CATALOG,
        "profile.taps.catalog",
    )
    _expect_exact(
        _expect_tuple(taps["artifact_selected"], "profile.taps.artifact_selected"),
        SELECTED_ARTIFACT_TAPS,
        "profile.taps.artifact_selected",
    )

    tolerances = _expect_mapping(
        profile["tolerances"], frozenset({"atol", "rtol"}), "profile.tolerances"
    )
    for key in ("atol", "rtol"):
        tolerance = _expect_float(tolerances[key], f"profile.tolerances.{key}")
        if tolerance != 1e-6:
            _fail("schema_validation_failed", f"profile.tolerances.{key}", "expected 1e-6")

    authorization = _expect_mapping(
        profile["authorization"],
        frozenset(
            {
                "forward_allowed",
                "max_forward_count_per_process",
                "max_independent_processes",
                "purpose",
                "raw_tensor_retention_allowed",
                "synthetic_input_only",
                "training_allowed",
            }
        ),
        "profile.authorization",
    )
    expected_authorization = {
        "forward_allowed": True,
        "max_forward_count_per_process": 2,
        "max_independent_processes": 2,
        "purpose": "pretrained_runtime_semantic_validation_only",
        "raw_tensor_retention_allowed": False,
        "synthetic_input_only": True,
        "training_allowed": False,
    }
    for key, expected in expected_authorization.items():
        _expect_exact(authorization[key], expected, f"profile.authorization.{key}")

    policies = _expect_mapping(
        profile["policies"],
        frozenset(
            {
                "cpu_only",
                "evaluation_mode",
                "fp32_only",
                "frozen_parameters",
                "image_processor_allowed",
                "inference_mode",
                "local_files_only",
                "model_cache_creation_allowed",
                "model_id_load_allowed",
                "network_allowed",
                "raw_tensor_retention_allowed",
                "token_allowed",
                "torchvision_allowed",
                "trust_remote_code",
                "use_safetensors",
            }
        ),
        "profile.policies",
    )
    expected_true = {
        "cpu_only",
        "evaluation_mode",
        "fp32_only",
        "frozen_parameters",
        "inference_mode",
        "local_files_only",
        "use_safetensors",
    }
    for key in policies:
        expected = key in expected_true
        _expect_exact(policies[key], expected, f"profile.policies.{key}")

    protocol = _expect_mapping(
        profile["protocol"],
        frozenset({"request_schema_id", "response_schema_id"}),
        "profile.protocol",
    )
    _expect_exact(protocol["request_schema_id"], SEMANTIC_REQUEST_SCHEMA_ID, "profile.protocol.request_schema_id")
    _expect_exact(protocol["response_schema_id"], SEMANTIC_RESPONSE_SCHEMA_ID, "profile.protocol.response_schema_id")

    manual = _expect_mapping(
        profile["manual_preprocessing"],
        frozenset(
            {
                "channel_order",
                "formula_steps",
                "image_mean",
                "image_standard_deviation",
                "profile_id",
                "prohibited_operations",
                "rescale_factor",
                "rescale_factor_dtype",
            }
        ),
        "profile.manual_preprocessing",
    )
    _expect_exact(manual["channel_order"], "RGB", "profile.manual_preprocessing.channel_order")
    _expect_exact(
        _expect_tuple(manual["image_mean"], "profile.manual_preprocessing.image_mean"),
        (0.485, 0.456, 0.406),
        "profile.manual_preprocessing.image_mean",
    )
    _expect_exact(
        _expect_tuple(
            manual["image_standard_deviation"],
            "profile.manual_preprocessing.image_standard_deviation",
        ),
        (0.229, 0.224, 0.225),
        "profile.manual_preprocessing.image_standard_deviation",
    )
    _expect_exact(manual["profile_id"], "dinov3-manual-fp32-imagenet-v1", "profile.manual_preprocessing.profile_id")
    _expect_exact(
        _expect_tuple(
            manual["prohibited_operations"],
            "profile.manual_preprocessing.prohibited_operations",
        ),
        ("crop", "image_processor", "pil_conversion", "resize", "torchvision"),
        "profile.manual_preprocessing.prohibited_operations",
    )
    _expect_exact(manual["rescale_factor_dtype"], "float32", "profile.manual_preprocessing.rescale_factor_dtype")
    if _expect_float(manual["rescale_factor"], "profile.manual_preprocessing.rescale_factor") != 0.003921568859368563:
        _fail("schema_validation_failed", "profile.manual_preprocessing.rescale_factor", "unexpected factor")
    steps = _expect_sequence(manual["formula_steps"], "profile.manual_preprocessing.formula_steps")
    for index, step in enumerate(steps):
        _expect_string(step, f"profile.manual_preprocessing.formula_steps[{index}]", maximum=256)
    _expect_exact(
        tuple(steps),
        _EXPECTED_FORMULA_STEPS,
        "profile.manual_preprocessing.formula_steps",
    )


def load_semantic_profile(path: str | Path) -> DINOv3SemanticProfile:
    """Load the exact canonical semantic profile from an absolute regular path."""

    _, payload = _read_regular_file(path, maximum_bytes=_MAX_PROFILE_BYTES, context="semantic_profile")
    value = _parse_json(payload, "semantic_profile")
    _scan_sensitive(value, "profile")
    _validate_profile(value)
    canonical = _canonical_json_bytes(value)
    if payload != canonical:
        _fail("invalid_json", "semantic_profile", "expected canonical sorted-key UTF-8 JSON")
    return DINOv3SemanticProfile(
        document=_freeze(value), sha256=hashlib.sha256(payload).hexdigest()
    )


def _validate_request(value: Mapping[str, Any]) -> DINOv3SemanticRequest:
    request = _expect_mapping(
        value,
        frozenset(
            {
                "run_id",
                "schema_id",
                "semantic_profile_sha256",
                "workspace_relative_artifact_directory",
            }
        ),
        "request",
    )
    _expect_exact(request["schema_id"], SEMANTIC_REQUEST_SCHEMA_ID, "request.schema_id")
    return DINOv3SemanticRequest(
        run_id=_expect_identifier(request["run_id"], "request.run_id"),
        semantic_profile_sha256=_expect_sha256(
            request["semantic_profile_sha256"], "request.semantic_profile_sha256"
        ),
        workspace_relative_artifact_directory=_expect_relative_path(
            request["workspace_relative_artifact_directory"],
            "request.workspace_relative_artifact_directory",
        ),
    )


def _safe_new_output(path: str | Path, context: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        _fail("unsafe_path", context, "expected an absolute path")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise DINOv3SemanticBoundaryError(
            "unsafe_path", f"{context}: parent is not accessible"
        ) from exc
    if _is_reparse(candidate.parent):
        _fail("unsafe_path", context, "reparse parent is forbidden")
    if candidate.exists() or candidate.is_symlink():
        _fail("unsafe_path", context, "refusing to overwrite an existing path")
    return parent / candidate.name


def write_semantic_request(
    path: str | Path,
    *,
    run_id: str,
    semantic_profile_sha256: str,
    workspace_relative_artifact_directory: str = "actual-output",
) -> DINOv3SemanticRequest:
    """Validate and exclusively write one canonical persistent-probe request."""

    request = _validate_request(
        {
            "run_id": run_id,
            "schema_id": SEMANTIC_REQUEST_SCHEMA_ID,
            "semantic_profile_sha256": semantic_profile_sha256,
            "workspace_relative_artifact_directory": workspace_relative_artifact_directory,
        }
    )
    output = _safe_new_output(path, "semantic_request")
    try:
        with output.open("xb") as stream:
            stream.write(_canonical_json_bytes(request.to_dict()))
    except OSError as exc:
        raise DINOv3SemanticBoundaryError(
            "io_error", "semantic_request: unable to write request"
        ) from exc
    return request


def read_semantic_request(path: str | Path) -> DINOv3SemanticRequest:
    """Read a strict semantic request (used by the persistent probe)."""

    _, payload = _read_regular_file(path, maximum_bytes=_MAX_REQUEST_BYTES, context="semantic_request")
    return _validate_request(_parse_json(payload, "semantic_request"))


def _validate_response_identities(
    value: object, profile: DINOv3SemanticProfile
) -> None:
    expected_fields = frozenset(
        {
            *profile.identities.keys(),
            "semantic_profile_sha256",
            "source_hashes",
        }
    )
    identities = _expect_mapping(value, expected_fields, "response.identities")
    _expect_exact(
        _expect_sha256(identities["semantic_profile_sha256"], "response.identities.semantic_profile_sha256"),
        profile.sha256,
        "response.identities.semantic_profile_sha256",
    )
    for key in profile.identities:
        _expect_exact(
            _expect_sha256(identities[key], f"response.identities.{key}"),
            profile.identities[key],
            f"response.identities.{key}",
        )
    sources = _expect_sequence(identities["source_hashes"], "response.identities.source_hashes")
    actual: list[tuple[str, str]] = []
    for index, item in enumerate(sources):
        path = f"response.identities.source_hashes[{index}]"
        entry = _expect_mapping(item, frozenset({"path", "sha256"}), path)
        actual.append(
            (
                _expect_relative_path(entry["path"], f"{path}.path"),
                _expect_sha256(entry["sha256"], f"{path}.sha256"),
            )
        )
    expected = tuple(
        (str(item["path"]), str(item["sha256"]))
        for item in profile.document.installed_sources
    )
    _expect_exact(tuple(actual), expected, "response.identities.source_hashes")


def _validate_response_model(value: object, profile: DINOv3SemanticProfile) -> None:
    model = _expect_mapping(
        value,
        frozenset(
            {
                "compute_dtype",
                "config_class",
                "device",
                "evaluation_mode",
                "gradients_enabled",
                "loading_error_count",
                "mismatched_key_count",
                "missing_key_count",
                "model_class",
                "output_dtype",
                "parameter_count",
                "parameter_dtype",
                "parameter_tensor_count",
                "parameters_frozen",
                "state_sha256",
                "unexpected_key_count",
            }
        ),
        "response.model",
    )
    exact = {
        "compute_dtype": "float32",
        "config_class": profile.model.config_class,
        "device": "cpu",
        "evaluation_mode": True,
        "gradients_enabled": False,
        "loading_error_count": 0,
        "mismatched_key_count": 0,
        "missing_key_count": 0,
        "model_class": profile.model.model_class,
        "output_dtype": "float32",
        "parameter_count": profile.model.expected_parameter_count,
        "parameter_dtype": "float32",
        "parameter_tensor_count": profile.model.expected_parameter_tensor_count,
        "parameters_frozen": True,
        "state_sha256": profile.model.state_sha256,
        "unexpected_key_count": 0,
    }
    _expect_sha256(model["state_sha256"], "response.model.state_sha256")
    for key, expected in exact.items():
        _expect_exact(model[key], expected, f"response.model.{key}")


def _validate_response_input(value: object, profile: DINOv3SemanticProfile) -> None:
    item = _expect_mapping(
        value,
        frozenset(
            {
                "crop_count",
                "image_ids",
                "image_processor_construction_count",
                "input_dtype",
                "input_sha256",
                "input_shape",
                "input_unchanged",
                "normalized_c_contiguous",
                "normalized_dtype",
                "normalized_finite",
                "normalized_sha256",
                "normalized_shape",
                "pil_conversion_count",
                "resize_count",
                "torchvision_import_count",
            }
        ),
        "response.input",
    )
    exact = {
        "crop_count": 0,
        "image_ids": list(profile.canonical_input.image_ids),
        "image_processor_construction_count": 0,
        "input_dtype": "uint8",
        "input_sha256": INPUT_SHA256,
        "input_unchanged": True,
        "normalized_c_contiguous": True,
        "normalized_dtype": "float32",
        "normalized_finite": True,
        "normalized_sha256": NORMALIZED_SHA256,
        "pil_conversion_count": 0,
        "resize_count": 0,
        "torchvision_import_count": 0,
    }
    _expect_sha256(item["input_sha256"], "response.input.input_sha256")
    _expect_sha256(item["normalized_sha256"], "response.input.normalized_sha256")
    for key, expected in exact.items():
        _expect_exact(item[key], expected, f"response.input.{key}")
    _expect_shape(item["input_shape"], (2, 224, 224, 3), "response.input.input_shape")
    _expect_shape(item["normalized_shape"], (2, 3, 224, 224), "response.input.normalized_shape")


def _validate_components(value: object) -> None:
    components = _expect_mapping(
        value,
        frozenset(
            {
                "cls_token",
                "embeddings",
                "encoder_layers",
                "model",
                "norm",
                "patch_embeddings",
                "pooler",
                "register_tokens",
            }
        ),
        "response.components",
    )
    expected = {
        "cls_token": "embeddings.cls_token",
        "embeddings": "embeddings",
        "encoder_layers": "layer",
        "model": "self",
        "norm": "norm",
        "patch_embeddings": "embeddings.patch_embeddings",
        "pooler": "pooler_output",
        "register_tokens": "embeddings.register_tokens",
    }
    for key, item in components.items():
        text = _expect_string(item, f"response.components.{key}", maximum=256)
        if "/" in text or "\\" in text:
            _fail("unsafe_path", f"response.components.{key}", "expected a Python component path")
        _expect_exact(text, expected[key], f"response.components.{key}")


def _validate_embedding(value: object, atol: float) -> None:
    embedding = _expect_mapping(
        value,
        frozenset(
            {
                "hidden_state_0_matches",
                "hidden_state_0_max_abs_diff",
                "hidden_state_0_shape",
                "patch_embedding_dtype",
                "patch_embedding_hook_matches",
                "patch_embedding_hook_max_abs_diff",
                "patch_embedding_shape",
                "sequence_dtype",
                "sequence_shape",
            }
        ),
        "response.embedding",
    )
    _expect_shape(embedding["sequence_shape"], (2, 201, 768), "response.embedding.sequence_shape")
    _expect_shape(embedding["hidden_state_0_shape"], (2, 201, 768), "response.embedding.hidden_state_0_shape")
    _expect_shape(embedding["patch_embedding_shape"], (2, 196, 768), "response.embedding.patch_embedding_shape")
    _expect_exact(embedding["sequence_dtype"], "float32", "response.embedding.sequence_dtype")
    _expect_exact(embedding["patch_embedding_dtype"], "float32", "response.embedding.patch_embedding_dtype")
    for name in ("hidden_state_0", "patch_embedding_hook"):
        _expect_exact(embedding[f"{name}_matches"], True, f"response.embedding.{name}_matches")
        if _expect_float(embedding[f"{name}_max_abs_diff"], f"response.embedding.{name}_max_abs_diff") > atol:
            _fail("semantic_validation_failed", f"response.embedding.{name}_max_abs_diff", "exceeds tolerance")


def _validate_token_partition(value: object, profile: DINOv3SemanticProfile, path: str) -> None:
    item = _expect_mapping(
        value,
        frozenset(
            {
                "cls_indices",
                "overlap_count",
                "partition_complete",
                "patch_indices",
                "register_indices",
                "sequence_order",
                "special_indices",
            }
        ),
        path,
    )
    cls_indices = _expect_tuple(item["cls_indices"], f"{path}.cls_indices")
    register_indices = _expect_tuple(item["register_indices"], f"{path}.register_indices")
    patch_indices = _expect_tuple(item["patch_indices"], f"{path}.patch_indices")
    special_indices = _expect_tuple(item["special_indices"], f"{path}.special_indices")
    _expect_exact(cls_indices, (0,), f"{path}.cls_indices")
    _expect_exact(register_indices, (1, 2, 3, 4), f"{path}.register_indices")
    _expect_exact(patch_indices, profile.patch_token_indices, f"{path}.patch_indices")
    _expect_exact(special_indices, profile.special_token_indices, f"{path}.special_indices")
    _expect_exact(item["sequence_order"], "cls_register_patch", f"{path}.sequence_order")
    _expect_exact(item["overlap_count"], 0, f"{path}.overlap_count")
    _expect_exact(item["partition_complete"], True, f"{path}.partition_complete")
    combined = special_indices + patch_indices
    if len(combined) != 201 or len(set(combined)) != 201 or set(combined) != set(range(201)):
        _fail("semantic_validation_failed", path, "token partition is incomplete or overlapping")


def _validate_patch_order(value: object, profile: DINOv3SemanticProfile) -> None:
    order = _expect_mapping(
        value,
        frozenset(
            {
                "atol",
                "conv2d_hook_shape",
                "embedding_patch_slice_shape",
                "matches",
                "max_abs_diff",
                "reconstructed_shape",
                "row_major",
                "rtol",
                "source_relation",
            }
        ),
        "response.patch_order",
    )
    _expect_shape(
        order["conv2d_hook_shape"],
        (2, 768, 14, 14),
        "response.patch_order.conv2d_hook_shape",
    )
    _expect_shape(
        order["embedding_patch_slice_shape"],
        (2, 196, 768),
        "response.patch_order.embedding_patch_slice_shape",
    )
    _expect_shape(order["reconstructed_shape"], (2, 196, 768), "response.patch_order.reconstructed_shape")
    _expect_exact(order["row_major"], True, "response.patch_order.row_major")
    _expect_exact(order["matches"], True, "response.patch_order.matches")
    _expect_exact(order["source_relation"], "conv2d_output_flatten_spatial_then_transpose", "response.patch_order.source_relation")
    for key in ("atol", "rtol"):
        _expect_exact(_expect_float(order[key], f"response.patch_order.{key}"), profile.tolerances[key], f"response.patch_order.{key}")
    if _expect_float(order["max_abs_diff"], "response.patch_order.max_abs_diff") > profile.tolerances.atol:
        _fail("semantic_validation_failed", "response.patch_order.max_abs_diff", "exceeds tolerance")


def _validate_hidden_states(value: object, profile: DINOv3SemanticProfile) -> None:
    hidden = _expect_mapping(
        value,
        frozenset({"blocks", "count", "embedding_hidden_state_index", "embedding_matches", "embedding_max_abs_diff", "shapes"}),
        "response.hidden_states",
    )
    _expect_exact(hidden["count"], 13, "response.hidden_states.count")
    _expect_exact(hidden["embedding_hidden_state_index"], 0, "response.hidden_states.embedding_hidden_state_index")
    _expect_exact(hidden["embedding_matches"], True, "response.hidden_states.embedding_matches")
    if _expect_float(hidden["embedding_max_abs_diff"], "response.hidden_states.embedding_max_abs_diff") > profile.tolerances.atol:
        _fail("semantic_validation_failed", "response.hidden_states.embedding_max_abs_diff", "exceeds tolerance")
    shapes = _expect_sequence(hidden["shapes"], "response.hidden_states.shapes")
    if len(shapes) != 13:
        _fail("semantic_validation_failed", "response.hidden_states.shapes", "expected 13 shapes")
    for index, shape in enumerate(shapes):
        _expect_shape(shape, (2, 201, 768), f"response.hidden_states.shapes[{index}]")
    blocks = _expect_sequence(hidden["blocks"], "response.hidden_states.blocks")
    if len(blocks) != 12:
        _fail("semantic_validation_failed", "response.hidden_states.blocks", "expected 12 block mappings")
    for index, raw in enumerate(blocks, start=1):
        path = f"response.hidden_states.blocks[{index - 1}]"
        block = _expect_mapping(
            raw,
            frozenset({"block_index", "hidden_state_index", "hidden_state_matches_hook", "hidden_state_shape", "hook_shape", "max_abs_diff", "tap_name"}),
            path,
        )
        _expect_exact(block["block_index"], index, f"{path}.block_index")
        _expect_exact(block["hidden_state_index"], index, f"{path}.hidden_state_index")
        _expect_exact(block["tap_name"], f"block_{index:02d}", f"{path}.tap_name")
        _expect_shape(block["hook_shape"], (2, 201, 768), f"{path}.hook_shape")
        _expect_shape(block["hidden_state_shape"], (2, 201, 768), f"{path}.hidden_state_shape")
        difference = _expect_float(block["max_abs_diff"], f"{path}.max_abs_diff")
        if index < 12:
            _expect_exact(block["hidden_state_matches_hook"], True, f"{path}.hidden_state_matches_hook")
            if difference > profile.tolerances.atol:
                _fail("semantic_validation_failed", f"{path}.max_abs_diff", "exceeds tolerance")
        else:
            _expect_exact(block["hidden_state_matches_hook"], False, f"{path}.hidden_state_matches_hook")
            if difference <= profile.tolerances.atol:
                _fail("semantic_validation_failed", f"{path}.max_abs_diff", "raw block 12 must differ from final-normalized hidden state")


def _validate_final_norm(value: object, profile: DINOv3SemanticProfile) -> None:
    final = _expect_mapping(
        value,
        frozenset(
            {
                "canonical_block_12_source",
                "canonical_final_norm_source",
                "explicit_norm_matches_last_hidden",
                "explicit_norm_max_abs_diff",
                "explicit_norm_shape",
                "hidden_state_12_matches_last_hidden",
                "hidden_state_12_max_abs_diff",
                "last_hidden_shape",
                "raw_final_differs_from_last_hidden",
                "raw_final_shape",
                "raw_final_vs_last_hidden_max_abs_diff",
            }
        ),
        "response.final_norm",
    )
    _expect_exact(final["canonical_block_12_source"], "raw_final_block_hook", "response.final_norm.canonical_block_12_source")
    _expect_exact(final["canonical_final_norm_source"], "explicit_model_norm", "response.final_norm.canonical_final_norm_source")
    for name in ("explicit_norm", "last_hidden", "raw_final"):
        _expect_shape(final[f"{name}_shape"], (2, 201, 768), f"response.final_norm.{name}_shape")
    for name in ("explicit_norm", "hidden_state_12"):
        _expect_exact(final[f"{name}_matches_last_hidden"], True, f"response.final_norm.{name}_matches_last_hidden")
        if _expect_float(final[f"{name}_max_abs_diff"], f"response.final_norm.{name}_max_abs_diff") > profile.tolerances.atol:
            _fail("semantic_validation_failed", f"response.final_norm.{name}_max_abs_diff", "exceeds tolerance")
    _expect_exact(final["raw_final_differs_from_last_hidden"], True, "response.final_norm.raw_final_differs_from_last_hidden")
    if _expect_float(final["raw_final_vs_last_hidden_max_abs_diff"], "response.final_norm.raw_final_vs_last_hidden_max_abs_diff") <= profile.tolerances.atol:
        _fail("semantic_validation_failed", "response.final_norm.raw_final_vs_last_hidden_max_abs_diff", "raw final output must differ")


def _validate_pooler(value: object, profile: DINOv3SemanticProfile) -> None:
    pooler = _expect_mapping(
        value,
        frozenset(
            {
                "canonical_global_source",
                "cls_matches",
                "cls_max_abs_diff",
                "excluded_from_canonical_global",
                "max_abs_difference_from_patch_mean",
                "mean_abs_difference_from_patch_mean",
                "patch_mean_shape",
                "pooler_shape",
                "separate_head_used",
                "source_relation",
            }
        ),
        "response.pooler",
    )
    _expect_shape(pooler["pooler_shape"], (2, 768), "response.pooler.pooler_shape")
    _expect_shape(pooler["patch_mean_shape"], (2, 768), "response.pooler.patch_mean_shape")
    _expect_exact(pooler["source_relation"], "last_hidden_state_cls", "response.pooler.source_relation")
    _expect_exact(pooler["separate_head_used"], False, "response.pooler.separate_head_used")
    _expect_exact(pooler["cls_matches"], True, "response.pooler.cls_matches")
    if _expect_float(pooler["cls_max_abs_diff"], "response.pooler.cls_max_abs_diff") > profile.tolerances.atol:
        _fail("semantic_validation_failed", "response.pooler.cls_max_abs_diff", "exceeds tolerance")
    _expect_exact(pooler["canonical_global_source"], "patch_axis_mean", "response.pooler.canonical_global_source")
    _expect_exact(pooler["excluded_from_canonical_global"], True, "response.pooler.excluded_from_canonical_global")
    _expect_float(pooler["max_abs_difference_from_patch_mean"], "response.pooler.max_abs_difference_from_patch_mean")
    _expect_float(pooler["mean_abs_difference_from_patch_mean"], "response.pooler.mean_abs_difference_from_patch_mean")


def _validate_token_layout(value: object, profile: DINOv3SemanticProfile) -> None:
    layout = _expect_mapping(
        value,
        frozenset({"grid_height", "grid_width", "partition_complete", "patch_token_indices", "special_token_indices", "special_token_policy", "total_token_count"}),
        "response.token_layout",
    )
    exact = {
        "grid_height": 14,
        "grid_width": 14,
        "partition_complete": True,
        "patch_token_indices": list(profile.patch_token_indices),
        "special_token_indices": list(profile.special_token_indices),
        "special_token_policy": SPECIAL_TOKEN_POLICY,
        "total_token_count": 201,
    }
    for key, expected in exact.items():
        _expect_exact(layout[key], expected, f"response.token_layout.{key}")


def _validate_taps(value: object, profile: DINOv3SemanticProfile) -> dict[str, Mapping[str, Any]]:
    taps = _expect_sequence(value, "response.taps")
    if len(taps) != 14:
        _fail("semantic_validation_failed", "response.taps", "expected 14 taps")
    by_name: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(taps):
        path = f"response.taps[{index}]"
        tap = _expect_mapping(
            raw,
            frozenset({"dtype", "finite", "full_sequence_shape", "global_matches_patch_mean", "global_sha256", "global_shape", "name", "patch_sha256", "patch_shape"}),
            path,
        )
        name = _expect_identifier(tap["name"], f"{path}.name")
        _expect_exact(name, TAP_CATALOG[index], f"{path}.name")
        if name in by_name:
            _fail("semantic_validation_failed", path, "duplicate tap")
        _expect_shape(tap["full_sequence_shape"], (2, 201, 768), f"{path}.full_sequence_shape")
        _expect_shape(tap["patch_shape"], (2, 196, 768), f"{path}.patch_shape")
        _expect_shape(tap["global_shape"], (2, 768), f"{path}.global_shape")
        _expect_exact(tap["dtype"], "float32", f"{path}.dtype")
        _expect_exact(tap["finite"], True, f"{path}.finite")
        _expect_exact(tap["global_matches_patch_mean"], True, f"{path}.global_matches_patch_mean")
        _expect_sha256(tap["patch_sha256"], f"{path}.patch_sha256")
        _expect_sha256(tap["global_sha256"], f"{path}.global_sha256")
        by_name[name] = tap
    return by_name


def _validate_artifact(
    value: object,
    taps: Mapping[str, Mapping[str, Any]],
    workspace_paths_value: object,
) -> None:
    artifact = _expect_mapping(
        value,
        frozenset({"entries", "roundtrip_validated", "selected_taps", "workspace_relative_directory"}),
        "response.artifact",
    )
    directory = _expect_relative_path(
        artifact["workspace_relative_directory"],
        "response.artifact.workspace_relative_directory",
    )
    _expect_exact(directory, "actual-output", "response.artifact.workspace_relative_directory")
    _expect_exact(
        _expect_tuple(artifact["selected_taps"], "response.artifact.selected_taps"),
        SELECTED_ARTIFACT_TAPS,
        "response.artifact.selected_taps",
    )
    _expect_exact(artifact["roundtrip_validated"], True, "response.artifact.roundtrip_validated")
    entries = _expect_sequence(artifact["entries"], "response.artifact.entries")
    if len(entries) != 3:
        _fail("semantic_validation_failed", "response.artifact.entries", "expected three artifacts")
    expected_paths: list[str] = []
    for index, raw in enumerate(entries):
        path = f"response.artifact.entries[{index}]"
        entry = _expect_mapping(
            raw,
            frozenset({"checksum_sha256", "global_sha256", "patch_sha256", "safetensors_relative_path", "sidecar_relative_path", "tap_name"}),
            path,
        )
        tap_name = _expect_identifier(entry["tap_name"], f"{path}.tap_name")
        _expect_exact(tap_name, SELECTED_ARTIFACT_TAPS[index], f"{path}.tap_name")
        sidecar = _expect_relative_path(entry["sidecar_relative_path"], f"{path}.sidecar_relative_path")
        tensors = _expect_relative_path(entry["safetensors_relative_path"], f"{path}.safetensors_relative_path")
        _expect_exact(sidecar, f"{directory}/{tap_name}.json", f"{path}.sidecar_relative_path")
        _expect_exact(tensors, f"{directory}/{tap_name}.safetensors", f"{path}.safetensors_relative_path")
        _expect_sha256(entry["checksum_sha256"], f"{path}.checksum_sha256")
        _expect_exact(_expect_sha256(entry["patch_sha256"], f"{path}.patch_sha256"), taps[tap_name]["patch_sha256"], f"{path}.patch_sha256")
        _expect_exact(_expect_sha256(entry["global_sha256"], f"{path}.global_sha256"), taps[tap_name]["global_sha256"], f"{path}.global_sha256")
        expected_paths.extend((sidecar, tensors))
    paths = tuple(
        _expect_relative_path(item, f"response.workspace_paths[{index}]")
        for index, item in enumerate(_expect_sequence(workspace_paths_value, "response.workspace_paths"))
    )
    _expect_exact(paths, tuple(expected_paths), "response.workspace_paths")


def _validate_all_true_mapping(value: object, fields: frozenset[str], path: str) -> None:
    mapping = _expect_mapping(value, fields, path)
    for key in fields:
        _expect_exact(mapping[key], True, f"{path}.{key}")


def validate_semantic_response(
    value: Mapping[str, Any], profile: DINOv3SemanticProfile
) -> DINOv3SemanticResponse:
    """Validate a strict metadata-only response against one exact profile."""

    if not isinstance(profile, DINOv3SemanticProfile):
        _fail("schema_validation_failed", "profile", "expected DINOv3SemanticProfile")
    _scan_sensitive(value, "response")
    response = _expect_mapping(
        value,
        frozenset(
            {
                "artifact",
                "components",
                "embedding",
                "final_norm",
                "forward_count",
                "hidden_states",
                "identities",
                "input",
                "invariants",
                "markers",
                "model",
                "patch_order",
                "pooler",
                "prefix_tokens",
                "repeat",
                "run_id",
                "schema_id",
                "semantic_marker",
                "socket_attempts",
                "status",
                "taps",
                "token_layout",
                "workspace_paths",
            }
        ),
        "response",
    )
    _expect_exact(response["schema_id"], SEMANTIC_RESPONSE_SCHEMA_ID, "response.schema_id")
    _expect_identifier(response["run_id"], "response.run_id")
    _expect_exact(response["status"], "success", "response.status")
    _expect_exact(response["semantic_marker"], SEMANTIC_SUCCESS_MARKER, "response.semantic_marker")
    _validate_response_identities(response["identities"], profile)
    _validate_response_model(response["model"], profile)
    _validate_response_input(response["input"], profile)
    _validate_components(response["components"])
    _validate_embedding(response["embedding"], profile.tolerances.atol)
    _validate_token_partition(response["prefix_tokens"], profile, "response.prefix_tokens")
    _validate_patch_order(response["patch_order"], profile)
    _validate_hidden_states(response["hidden_states"], profile)
    _validate_final_norm(response["final_norm"], profile)
    _validate_pooler(response["pooler"], profile)
    _validate_token_layout(response["token_layout"], profile)
    taps = _validate_taps(response["taps"], profile)
    _validate_artifact(response["artifact"], taps, response["workspace_paths"])
    _validate_all_true_mapping(
        response["repeat"],
        frozenset({"embedding_exact", "global_hashes_exact", "hidden_states_exact", "last_hidden_exact", "patch_hashes_exact", "pooler_exact"}),
        "response.repeat",
    )
    _validate_all_true_mapping(
        response["invariants"],
        frozenset({"model_state_unchanged", "numpy_rng_unchanged", "python_rng_unchanged", "torch_rng_unchanged", "training_false"}),
        "response.invariants",
    )
    _expect_exact(response["socket_attempts"], 0, "response.socket_attempts")
    _expect_exact(
        response["forward_count"],
        profile.document.authorization.max_forward_count_per_process,
        "response.forward_count",
    )
    _expect_exact(
        _expect_tuple(response["markers"], "response.markers"),
        REQUIRED_SEMANTIC_MARKERS,
        "response.markers",
    )
    return DINOv3SemanticResponse(document=_freeze(dict(value)))


def read_semantic_response(
    path: str | Path,
    *,
    profile: DINOv3SemanticProfile,
    workspace_root: str | Path | None = None,
) -> DINOv3SemanticResponse:
    """Read and validate a persistent response and optional artifact paths."""

    _, payload = _read_regular_file(path, maximum_bytes=_MAX_RESPONSE_BYTES, context="semantic_response")
    response = validate_semantic_response(_parse_json(payload, "semantic_response"), profile)
    if workspace_root is not None:
        for relative in response.document.workspace_paths:
            resolved = resolve_workspace_relative_path(workspace_root, relative, must_exist=True)
            if not resolved.is_file():
                _fail("unsafe_path", "response.workspace_paths", "expected regular artifact file")
    return response


def semantic_response_projection(
    response: DINOv3SemanticResponse,
) -> dict[str, Any]:
    """Return deterministic evidence, excluding only the run-specific run ID."""

    if not isinstance(response, DINOv3SemanticResponse):
        _fail("schema_validation_failed", "response", "expected DINOv3SemanticResponse")
    return response.semantic_projection()


__all__ = [
    "DINOv3SemanticBoundaryError",
    "DINOv3SemanticProfile",
    "DINOv3SemanticRequest",
    "DINOv3SemanticResponse",
    "INPUT_SHA256",
    "MODEL_ID",
    "MODEL_KEY",
    "NORMALIZED_SHA256",
    "REQUIRED_SEMANTIC_MARKERS",
    "REVISION",
    "SELECTED_ARTIFACT_TAPS",
    "SEMANTIC_PROFILE_SCHEMA_ID",
    "SEMANTIC_REQUEST_SCHEMA_ID",
    "SEMANTIC_RESPONSE_SCHEMA_ID",
    "SEMANTIC_SUCCESS_MARKER",
    "TAP_CATALOG",
    "load_semantic_profile",
    "read_semantic_request",
    "read_semantic_response",
    "resolve_workspace_relative_path",
    "semantic_response_projection",
    "sha256_file",
    "validate_semantic_response",
    "write_semantic_request",
]
