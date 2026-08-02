"""Run the exact DINOv3 local-checkpoint load twice in persistent Python."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping

from vision_encoder_alignment.runtime_boundary import (
    build_sanitized_worker_environment,
    load_runtime_profile,
    resolve_default_runtime_root,
    runtime_python_path,
    validate_runtime_root,
)
from vision_encoder_alignment.snapshot_boundary import (
    load_snapshot_profile,
    resolve_default_snapshot_root,
    validate_snapshot_root,
)


PROBE_RESPONSE_SCHEMA_ID = "vea.dinov3_checkpoint_load_probe_response.v0"

PROBE_MARKERS = (
    "DINOV3_SNAPSHOT_MANIFEST_OK",
    "DINOV3_CHECKPOINT_CONFIG_LOAD_OK",
    "DINOV3_CHECKPOINT_MODEL_LOAD_OK",
    "DINOV3_CHECKPOINT_LOADING_INFO_OK",
    "DINOV3_CHECKPOINT_CPU_FP32_OK",
    "DINOV3_CHECKPOINT_FROZEN_EVAL_OK",
    "DINOV3_CHECKPOINT_RNG_ISOLATION_OK",
    "DINOV3_CHECKPOINT_NETWORK_GUARD_OK",
    "DINOV3_CHECKPOINT_NO_FORWARD_OK",
)
SMOKE_MARKER = "DINOV3_CHECKPOINT_LOAD_SMOKE_OK"

_MAX_RESPONSE_BYTES = 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EMAIL_PATTERN = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
)
_SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+\S+|hf_[A-Za-z0-9]{10,}|"
    r"(?:token|secret|password|passwd|credential|api[_-]?key)\s*[:=])"
)
_CREDENTIAL_LIKE_NAMES = frozenset(
    {
        ".hf_token",
        ".netrc",
        "auth.json",
        "credentials",
        "credentials.ini",
        "credentials.json",
        "stored_tokens",
        "stored_tokens.json",
        "token",
        "tokens.json",
    }
)
_RESPONSE_FIELDS = frozenset(
    {
        "components",
        "config_class",
        "device",
        "dtype",
        "evaluation_mode",
        "forward_count",
        "forward_executed",
        "frozen",
        "load_options",
        "loading_info",
        "markers",
        "model_cache_unchanged",
        "model_class",
        "model_id",
        "model_state_sha256",
        "parameter_count",
        "parameter_tensor_count",
        "persistent_runtime_unchanged",
        "processor_object_count",
        "revision",
        "rng",
        "runtime_id",
        "runtime_manifest_sha256",
        "runtime_profile_sha256",
        "schema_id",
        "snapshot_id",
        "snapshot_manifest_sha256",
        "snapshot_profile_sha256",
        "snapshot_unchanged",
        "socket_attempt_count",
        "state_sha256",
        "status",
        "trainable_parameter_count",
        "training",
        "torchvision_import_count",
        "warnings",
    }
)


class CheckpointLoadSmokeError(RuntimeError):
    """Raised when checkpoint-load orchestration violates its contract."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def _fail(classification: str, message: str) -> None:
    raise CheckpointLoadSmokeError(classification, message)


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
        raise CheckpointLoadSmokeError(
            "path_validation_failed", "unable to inspect a filesystem path"
        ) from exc


def _resolved_directory(path: Path, context: str) -> Path:
    if not path.is_absolute() or _is_reparse_point(path):
        _fail("unsafe_path", f"{context} must be an absolute non-link directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CheckpointLoadSmokeError(
            "path_validation_failed", f"{context} does not exist"
        ) from exc
    if not resolved.is_dir():
        _fail("unsafe_path", f"{context} is not a directory")
    return resolved


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _roots_overlap(left: Path, right: Path) -> bool:
    return _contains(left, right) or _contains(right, left)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointLoadSmokeError(
            "inventory_failed", "unable to hash a protected file"
        ) from exc
    return digest.hexdigest()


def _credential_like(relative_path: str) -> bool:
    name = PurePosixPath(relative_path).name.casefold()
    return name in _CREDENTIAL_LIKE_NAMES or name.startswith(".env")


def _directory_inventory(root: Path) -> str:
    """Hash a path-free protected inventory without opening credential files."""

    if not root.exists() and not root.is_symlink():
        records: list[dict[str, Any]] = []
    else:
        resolved = _resolved_directory(root, "inventory_root")
        records = []
        for directory, directory_names, file_names in os.walk(
            resolved, topdown=True, followlinks=False
        ):
            parent = Path(directory)
            retained: list[str] = []
            for name in sorted(directory_names):
                child = parent / name
                relative = child.relative_to(resolved).as_posix()
                if relative == ".git":
                    records.append({"path": relative, "type": "excluded-directory"})
                    continue
                if _is_reparse_point(child):
                    _fail("unsafe_path", "protected inventory contains a link")
                retained.append(name)
            directory_names[:] = retained
            for name in sorted(file_names):
                child = parent / name
                relative = child.relative_to(resolved).as_posix()
                if relative == ".git":
                    records.append(
                        {
                            "path": relative,
                            "size": child.lstat().st_size,
                            "type": "excluded-file",
                        }
                    )
                    continue
                if _is_reparse_point(child):
                    _fail("unsafe_path", "protected inventory contains a link")
                try:
                    file_stat = child.stat()
                except OSError as exc:
                    raise CheckpointLoadSmokeError(
                        "inventory_failed", "unable to inspect a protected file"
                    ) from exc
                if not stat.S_ISREG(file_stat.st_mode):
                    _fail("unsafe_path", "protected inventory contains a non-file")
                if _credential_like(relative):
                    records.append(
                        {
                            "path": relative,
                            "size": file_stat.st_size,
                            "type": "credential-like-excluded",
                        }
                    )
                else:
                    records.append(
                        {
                            "path": relative,
                            "sha256": _sha256_file(child),
                            "size": file_stat.st_size,
                            "type": "file",
                        }
                    )
    payload = json.dumps(
        records,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("probe_response_invalid", "probe response has a duplicate field")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    _fail("probe_response_invalid", f"non-finite JSON value {value!r} is forbidden")


def _read_probe_response(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or _is_reparse_point(path):
        _fail("probe_response_invalid", "probe response path is unsafe")
    try:
        resolved = path.resolve(strict=True)
        file_stat = resolved.stat()
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > _MAX_RESPONSE_BYTES:
            _fail("probe_response_invalid", "probe response is not a bounded regular file")
        payload = resolved.read_bytes()
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite,
        )
    except CheckpointLoadSmokeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointLoadSmokeError(
            "probe_response_invalid", "probe response is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        _fail("probe_response_invalid", "probe response must be an object")
    return value


def _reject_path_or_secret(value: object, path: str = "response") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_path_or_secret(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_path_or_secret(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail("probe_response_invalid", f"{path} contains a non-finite value")
    elif isinstance(value, str):
        windows = PureWindowsPath(value)
        if PurePosixPath(value).is_absolute() or windows.is_absolute() or windows.drive:
            _fail("probe_response_invalid", f"{path} contains an absolute path")
        if _EMAIL_PATTERN.search(value) or _SECRET_VALUE_PATTERN.search(value):
            _fail("probe_response_invalid", f"{path} contains sensitive data")


def _expect_exact_fields(value: object, fields: frozenset[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("probe_response_invalid", f"{context} must be an object")
    actual = set(value)
    if actual != fields:
        _fail("probe_response_invalid", f"{context} fields do not match the schema")
    return value


def _expect_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        _fail("probe_response_invalid", f"{field} must be a boolean")
    return value


def _expect_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail("probe_response_invalid", f"{field} must be an integer >= {minimum}")
    return value


def _expect_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        _fail("probe_response_invalid", f"{field} must be a non-empty string")
    return value


def _expect_sha256(value: object, field: str) -> str:
    digest = _expect_string(value, field)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        _fail("probe_response_invalid", f"{field} must be a lowercase SHA-256")
    return digest


def _attribute_or_key(value: object, name: str) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, Mapping) and name in value:
        return value[name]
    _fail("probe_fact_mismatch", f"required contract field {name} is unavailable")


def _snapshot_fact(profile: object, name: str) -> Any:
    if hasattr(profile, name):
        return getattr(profile, name)
    if isinstance(profile, Mapping):
        if name in profile:
            return profile[name]
        model = profile.get("model")
        if isinstance(model, Mapping) and name in model:
            return model[name]
        persistent = profile.get("persistent_runtime")
        aliases = {
            "runtime_id": "runtime_id",
            "runtime_profile_sha256": "profile_sha256",
        }
        if isinstance(persistent, Mapping) and name in aliases:
            return persistent[aliases[name]]
    _fail("probe_fact_mismatch", f"snapshot profile field {name} is unavailable")


def _manifest_source_sha256(manifest: object | None) -> str | None:
    if manifest is None:
        return None
    if hasattr(manifest, "source_sha256"):
        value = getattr(manifest, "source_sha256")
        return value if isinstance(value, str) and value else None
    if isinstance(manifest, Mapping):
        value = manifest.get("source_sha256")
        return value if isinstance(value, str) and value else None
    return None


def validate_probe_response(
    value: object,
    *,
    snapshot_profile: object,
    runtime_profile: object,
    snapshot_manifest: object | None = None,
    runtime_manifest: object | None = None,
) -> dict[str, Any]:
    """Strictly validate a path-free probe response and all expected facts."""

    response = dict(_expect_exact_fields(value, _RESPONSE_FIELDS, "response"))
    _reject_path_or_secret(response)
    if response["schema_id"] != PROBE_RESPONSE_SCHEMA_ID:
        _fail("probe_response_invalid", "probe response schema ID is unexpected")
    if response["status"] != "success":
        _fail("probe_fact_mismatch", "probe did not report success")

    expected_strings = {
        "config_class": _snapshot_fact(snapshot_profile, "config_class"),
        "device": "cpu",
        "dtype": "float32",
        "model_class": _snapshot_fact(snapshot_profile, "model_class"),
        "model_id": _snapshot_fact(snapshot_profile, "model_id"),
        "revision": _snapshot_fact(snapshot_profile, "revision"),
        "runtime_id": _attribute_or_key(runtime_profile, "runtime_id"),
        "runtime_profile_sha256": _attribute_or_key(runtime_profile, "source_sha256"),
        "snapshot_id": _snapshot_fact(snapshot_profile, "snapshot_id"),
        "snapshot_profile_sha256": _attribute_or_key(snapshot_profile, "source_sha256"),
    }
    for field, expected in expected_strings.items():
        if _expect_string(response[field], field) != expected:
            _fail("probe_fact_mismatch", f"probe field {field} does not match")
    if response["runtime_profile_sha256"] != _snapshot_fact(
        snapshot_profile, "runtime_profile_sha256"
    ):
        _fail("probe_fact_mismatch", "runtime profile correlation is invalid")

    for field, manifest in (
        ("snapshot_manifest_sha256", snapshot_manifest),
        ("runtime_manifest_sha256", runtime_manifest),
    ):
        actual = _expect_sha256(response[field], field)
        expected = _manifest_source_sha256(manifest)
        if expected is not None and actual != expected:
            _fail("probe_fact_mismatch", f"probe field {field} does not match")
    _expect_sha256(response["state_sha256"], "state_sha256")
    if _expect_sha256(
        response["model_state_sha256"], "model_state_sha256"
    ) != response["state_sha256"]:
        _fail("probe_fact_mismatch", "model state hashes disagree")

    expected_ints = {
        "forward_count": 0,
        "parameter_count": _snapshot_fact(snapshot_profile, "expected_parameter_count"),
        "parameter_tensor_count": _snapshot_fact(
            snapshot_profile, "expected_parameter_tensor_count"
        ),
        "processor_object_count": 0,
        "socket_attempt_count": 0,
        "trainable_parameter_count": 0,
        "torchvision_import_count": 0,
    }
    for field, expected in expected_ints.items():
        if _expect_int(response[field], field) != expected:
            _fail("probe_fact_mismatch", f"probe field {field} does not match")

    expected_bools = {
        "evaluation_mode": True,
        "forward_executed": False,
        "frozen": True,
        "model_cache_unchanged": True,
        "persistent_runtime_unchanged": True,
        "snapshot_unchanged": True,
        "training": False,
    }
    for field, expected in expected_bools.items():
        if _expect_bool(response[field], field) is not expected:
            _fail("probe_fact_mismatch", f"probe field {field} does not match")

    if response["load_options"] != {
        "local_files_only": True,
        "output_loading_info": True,
        "token": False,
        "torch_dtype": "float32",
        "trust_remote_code": False,
        "use_safetensors": True,
    }:
        _fail("probe_fact_mismatch", "checkpoint load options are invalid")
    if response["loading_info"] != {
        "error_message_count": 0,
        "mismatched_key_count": 0,
        "missing_key_count": 0,
        "unexpected_key_count": 0,
    }:
        _fail("probe_fact_mismatch", "checkpoint loading information is not clean")
    if response["components"] != {
        "cls_token": True,
        "encoder_layer_count": 12,
        "final_norm": True,
        "patch_embedding": True,
        "register_tokens": True,
    }:
        _fail("probe_fact_mismatch", "required model components do not match")
    if response["rng"] != {
        "numpy_unchanged": True,
        "python_unchanged": True,
        "torch_unchanged": True,
    }:
        _fail("probe_fact_mismatch", "checkpoint load changed RNG state")
    if response["markers"] != list(PROBE_MARKERS):
        _fail("probe_fact_mismatch", "probe markers do not match")
    if response["warnings"] != []:
        _fail("probe_fact_mismatch", "probe reported warnings")
    return response


def _safe_response_name(value: str) -> str:
    if (
        not value
        or PurePosixPath(value).name != value
        or PureWindowsPath(value).name != value
        or PureWindowsPath(value).drive
    ):
        _fail("unsafe_path", "response_name must be a safe basename")
    return value


def invoke_checkpoint_load_probe(
    *,
    runtime_root: Path,
    repository_root: Path,
    snapshot_profile_path: Path,
    runtime_profile_path: Path,
    snapshot_root: Path,
    probe_script: Path,
    workspace_root: Path,
    response_name: str = "probe-response.json",
    timeout_seconds: float = 900.0,
    process_runner: Callable[..., Any] = subprocess.run,
    environment_builder: Callable[..., dict[str, str]] = build_sanitized_worker_environment,
    python_resolver: Callable[[str | Path], Path] = runtime_python_path,
) -> dict[str, Any]:
    """Invoke persistent Python with an argument list and parse strict JSON."""

    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        _fail("schema_validation_failed", "timeout_seconds must be positive and finite")
    repository = _resolved_directory(repository_root, "repository_root")
    runtime = _resolved_directory(runtime_root, "runtime_root")
    snapshot = _resolved_directory(snapshot_root, "snapshot_root")
    workspace = _resolved_directory(workspace_root, "workspace_root")
    if any(
        _roots_overlap(left, right)
        for index, left in enumerate((repository, runtime, snapshot, workspace))
        for right in (repository, runtime, snapshot, workspace)[index + 1 :]
    ):
        _fail("unsafe_path", "probe workspace and protected roots must be disjoint")
    response_file = workspace / _safe_response_name(response_name)
    if response_file.exists() or response_file.is_symlink():
        _fail("unsafe_path", "probe response path must be fresh")
    python = python_resolver(runtime).resolve(strict=True)
    environment = dict(
        environment_builder(
            runtime_python=python,
            repository_root=repository,
            workspace_root=workspace,
        )
    )
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    command = [
        str(python),
        "-B",
        str(probe_script.resolve(strict=True)),
        "--snapshot-profile",
        str(snapshot_profile_path.resolve(strict=True)),
        "--runtime-profile",
        str(runtime_profile_path.resolve(strict=True)),
        "--snapshot-root",
        str(snapshot),
        "--runtime-root",
        str(runtime),
        "--repository-root",
        str(repository),
        "--workspace-root",
        str(workspace),
        "--response",
        response_file.name,
    ]
    try:
        result = process_runner(
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
        raise CheckpointLoadSmokeError(
            "probe_timeout", "checkpoint-load probe exceeded its timeout"
        ) from exc
    except OSError as exc:
        raise CheckpointLoadSmokeError(
            "probe_launch_failed", "checkpoint-load probe could not be launched"
        ) from exc
    return_code = getattr(result, "returncode", None)
    if return_code != 0:
        cause = subprocess.CalledProcessError(
            int(return_code) if isinstance(return_code, int) else -1,
            command,
            output=getattr(result, "stdout", None),
            stderr=getattr(result, "stderr", None),
        )
        raise CheckpointLoadSmokeError(
            "probe_nonzero_exit", "checkpoint-load probe returned a nonzero exit"
        ) from cause
    try:
        return _read_probe_response(response_file)
    except CheckpointLoadSmokeError as exc:
        raise CheckpointLoadSmokeError(
            "probe_response_invalid", "checkpoint-load probe response is invalid"
        ) from exc


def _canonical_response(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CheckpointLoadSmokeError(
            "probe_response_invalid", "probe response is not canonical JSON data"
        ) from exc


def run_checkpoint_load_smoke(
    *,
    snapshot_profile_path: Path,
    runtime_profile_path: Path,
    snapshot_root: Path | None = None,
    runtime_root: Path | None = None,
    repository_root: Path | None = None,
    probe_script: Path | None = None,
    timeout_seconds: float = 900.0,
    snapshot_profile_loader: Callable[[str | Path], Any] = load_snapshot_profile,
    runtime_profile_loader: Callable[[str | Path], Any] = load_runtime_profile,
    snapshot_validator: Callable[..., Any] = validate_snapshot_root,
    runtime_validator: Callable[..., Any] = validate_runtime_root,
    snapshot_root_resolver: Callable[[Any], Path] = resolve_default_snapshot_root,
    runtime_root_resolver: Callable[[Any], Path] = resolve_default_runtime_root,
    probe_invoker: Callable[..., dict[str, Any]] = invoke_checkpoint_load_probe,
    inventory_reader: Callable[[Path], str] = _directory_inventory,
    temporary_directory_factory: Callable[..., Any] = tempfile.TemporaryDirectory,
    enforce_repository_sources: bool = False,
) -> dict[str, Any]:
    """Verify, load twice, enforce determinism, and prove cleanup/immutability."""

    repository = _resolved_directory(
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root,
        "repository_root",
    )
    snapshot_profile_file = snapshot_profile_path.resolve(strict=True)
    runtime_profile_file = runtime_profile_path.resolve(strict=True)
    snapshot_profile = snapshot_profile_loader(snapshot_profile_file)
    runtime_profile = runtime_profile_loader(runtime_profile_file)
    runtime = _resolved_directory(
        runtime_root_resolver(runtime_profile) if runtime_root is None else runtime_root,
        "runtime_root",
    )
    snapshot = _resolved_directory(
        snapshot_root_resolver(snapshot_profile)
        if snapshot_root is None
        else snapshot_root,
        "snapshot_root",
    )
    probe = (
        repository / "scripts" / "probe_dinov3_checkpoint_load.py"
        if probe_script is None
        else probe_script
    ).resolve(strict=True)
    if enforce_repository_sources:
        expected_snapshot_profile = (
            repository
            / "configs"
            / "model_snapshots"
            / "dinov3_vit_b16_r5931719_v0.json"
        ).resolve(strict=True)
        expected_runtime_profile = (
            repository
            / "configs"
            / "runtime_profiles"
            / "dinov3_isolated_manual_v0.json"
        ).resolve(strict=True)
        expected_probe = (
            repository / "scripts" / "probe_dinov3_checkpoint_load.py"
        ).resolve(strict=True)
        if (
            snapshot_profile_file != expected_snapshot_profile
            or runtime_profile_file != expected_runtime_profile
            or probe != expected_probe
        ):
            _fail(
                "source_identity_mismatch",
                "smoke inputs are not the exact repository deliverables",
            )
    if _snapshot_fact(snapshot_profile, "runtime_id") != _attribute_or_key(
        runtime_profile, "runtime_id"
    ):
        _fail("probe_fact_mismatch", "snapshot and runtime IDs differ")
    if _snapshot_fact(
        snapshot_profile, "runtime_profile_sha256"
    ) != _attribute_or_key(runtime_profile, "source_sha256"):
        _fail("probe_fact_mismatch", "snapshot references a different runtime profile")

    runtime_manifest = runtime_validator(
        runtime, runtime_profile, repository_root=repository
    )
    snapshot_manifest = snapshot_validator(
        snapshot,
        snapshot_profile,
        repository_root=repository,
        persistent_runtime_root=runtime,
    )
    protected_before = {
        "repository": inventory_reader(repository),
        "runtime": inventory_reader(runtime),
        "snapshot": inventory_reader(snapshot),
    }

    workspace_path: Path | None = None
    responses: list[dict[str, Any]] = []
    try:
        with temporary_directory_factory(prefix="vea_t007a3a_checkpoint_") as temporary:
            workspace_path = _resolved_directory(Path(temporary), "workspace_root")
            if any(
                _roots_overlap(workspace_path, protected)
                for protected in (repository, runtime, snapshot)
            ):
                _fail("unsafe_path", "task workspace overlaps a protected root")
            for repeat_index in range(2):
                repeat_root = workspace_path / f"repeat-{repeat_index}"
                repeat_root.mkdir()
                raw_response = probe_invoker(
                    runtime_root=runtime,
                    repository_root=repository,
                    snapshot_profile_path=snapshot_profile_file,
                    runtime_profile_path=runtime_profile_file,
                    snapshot_root=snapshot,
                    probe_script=probe,
                    workspace_root=repeat_root,
                    response_name="probe-response.json",
                    timeout_seconds=timeout_seconds,
                )
                responses.append(
                    validate_probe_response(
                        raw_response,
                        snapshot_profile=snapshot_profile,
                        runtime_profile=runtime_profile,
                        snapshot_manifest=snapshot_manifest,
                        runtime_manifest=runtime_manifest,
                    )
                )
            if _canonical_response(responses[0]) != _canonical_response(responses[1]):
                _fail("probe_nondeterministic", "repeated checkpoint-load facts differ")
    except CheckpointLoadSmokeError:
        raise
    except OSError as exc:
        raise CheckpointLoadSmokeError(
            "workspace_cleanup_failed", "task workspace setup or cleanup failed"
        ) from exc

    if workspace_path is None or workspace_path.exists():
        _fail("workspace_cleanup_failed", "task workspace was not removed")
    protected_after = {
        "repository": inventory_reader(repository),
        "runtime": inventory_reader(runtime),
        "snapshot": inventory_reader(snapshot),
    }
    if protected_after != protected_before:
        _fail("protected_inventory_changed", "checkpoint smoke changed protected content")

    final_runtime_manifest = runtime_validator(
        runtime, runtime_profile, repository_root=repository
    )
    final_snapshot_manifest = snapshot_validator(
        snapshot,
        snapshot_profile,
        repository_root=repository,
        persistent_runtime_root=runtime,
    )
    if (
        _manifest_source_sha256(final_runtime_manifest)
        != _manifest_source_sha256(runtime_manifest)
        or _manifest_source_sha256(final_snapshot_manifest)
        != _manifest_source_sha256(snapshot_manifest)
    ):
        _fail("protected_inventory_changed", "validated manifest identity changed")

    response = responses[0]
    repeat_response_sha256 = hashlib.sha256(_canonical_response(response)).hexdigest()
    evidence = {
        "config_class": response["config_class"],
        "device": response["device"],
        "dtype": response["dtype"],
        "evaluation_mode": response["evaluation_mode"],
        "forward_count": response["forward_count"],
        "forward_executed": response["forward_executed"],
        "frozen": response["frozen"],
        "load_options": response["load_options"],
        "loading_info": response["loading_info"],
        "markers": [*PROBE_MARKERS, SMOKE_MARKER],
        "model_class": response["model_class"],
        "model_id": response["model_id"],
        "model_state_sha256": response["model_state_sha256"],
        "parameter_count": response["parameter_count"],
        "parameter_tensor_count": response["parameter_tensor_count"],
        "persistent_runtime_unchanged": True,
        "processor_object_count": response["processor_object_count"],
        "repeat_count": 2,
        "repeat_deterministic": True,
        "repeat_model_facts_equal": True,
        "repeat_response_sha256": repeat_response_sha256,
        "repeat_state_hash_equal": True,
        "repository_unchanged": True,
        "revision": response["revision"],
        "rng": response["rng"],
        "runtime_id": response["runtime_id"],
        "runtime_manifest_sha256": response["runtime_manifest_sha256"],
        "runtime_profile_sha256": response["runtime_profile_sha256"],
        "snapshot_id": response["snapshot_id"],
        "snapshot_manifest_sha256": response["snapshot_manifest_sha256"],
        "snapshot_profile_sha256": response["snapshot_profile_sha256"],
        "snapshot_unchanged": True,
        "socket_attempt_count": response["socket_attempt_count"],
        "state_sha256": response["state_sha256"],
        "status": "PASS",
        "trainable_parameter_count": response["trainable_parameter_count"],
        "training": response["training"],
        "torchvision_import_count": response["torchvision_import_count"],
        "warnings": [],
        "workspace_cleaned": True,
    }
    _reject_path_or_secret(evidence)
    return evidence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-profile", required=True, type=Path)
    parser.add_argument("--runtime-profile", required=True, type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        evidence = run_checkpoint_load_smoke(
            snapshot_profile_path=args.snapshot_profile,
            runtime_profile_path=args.runtime_profile,
            snapshot_root=args.snapshot_root,
            runtime_root=args.runtime_root,
            timeout_seconds=args.timeout_seconds,
            enforce_repository_sources=True,
        )
    except Exception as exc:
        classification = getattr(exc, "classification", type(exc).__name__)
        print(f"DINOV3_CHECKPOINT_LOAD_SMOKE_FAILED:{classification}", file=sys.stderr)
        return 1
    for marker in evidence["markers"]:
        print(marker)
    print(
        json.dumps(
            evidence,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
