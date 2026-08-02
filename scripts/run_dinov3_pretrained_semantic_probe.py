"""Run the exact DINOv3 pretrained semantic probe in two isolated processes.

This orchestrator remains on the current Python side.  It verifies the pinned
identity graph and protected roots, launches the persistent Python with a
minimal offline environment, strictly validates each response, independently
reads the temporary feature artifacts, and removes every task-owned byte.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vision_encoder_alignment.dinov3_semantic_boundary import (
    DINOv3SemanticResponse,
    load_semantic_profile,
    read_semantic_response,
    write_semantic_request,
)
from vision_encoder_alignment.feature_artifacts import (
    read_feature_artifact,
    sha256_file,
)
from vision_encoder_alignment.identity_rebind import verify_rebound_identity
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
    "DINOV3_PRETRAINED_RUNTIME_SEMANTICS_OK",
)

SEMANTIC_STATUS = "DINOV3_PRETRAINED_RUNTIME_SEMANTICS_VERIFIED"
MODEL_KEY = "dinov3_vit_b16"
IMAGE_IDS = ("image-b", "image-a")
SELECTED_ARTIFACT_TAPS = ("embedding_output", "block_06", "final_norm")
PATCH_SHAPE = (2, 196, 768)
GLOBAL_SHAPE = (2, 768)
REPEAT_COUNT = 2

_MAX_CAPTURE_BYTES = 256 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
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
_ALLOWED_REPOSITORY_PATHS = frozenset(
    {
        "configs/semantic_probes/dinov3_vit_b16_r5931719_semantics_v0.json",
        "src/vision_encoder_alignment/dinov3_semantic_boundary.py",
        "scripts/probe_dinov3_pretrained_semantics.py",
        "scripts/run_dinov3_pretrained_semantic_probe.py",
        "tests/unit/test_dinov3_semantic_probe_profile.py",
        "tests/integration/test_dinov3_semantic_probe_boundary.py",
        "tasks/VEA-P0-T007A-4-dinov3-pretrained-runtime-semantic-probe.md",
        "reports/runs/VEA-P0-T007A-4-dinov3-pretrained-runtime-semantic-probe.md",
        "reports/runs/VEA-P0-T007A-4-dinov3-pretrained-runtime-semantic-probe.json",
        "reports/runs/VEA-P0-T007A-4-current-environment-freeze.txt",
        "reports/runs/VEA-P0-T007A-4-persistent-environment-freeze.txt",
    }
)


class DINOv3SemanticOrchestrationError(RuntimeError):
    """Raised when current-side semantic orchestration violates its contract."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def _fail(classification: str, message: str) -> None:
    raise DINOv3SemanticOrchestrationError(classification, message)


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
        raise DINOv3SemanticOrchestrationError(
            "path_validation_failed", "unable to inspect a filesystem path"
        ) from exc


def _resolved_directory(path: str | Path, context: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or _is_reparse_point(candidate):
        _fail("unsafe_path", f"{context} must be an absolute non-link directory")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise DINOv3SemanticOrchestrationError(
            "path_validation_failed", f"{context} does not exist"
        ) from exc
    if not resolved.is_dir():
        _fail("unsafe_path", f"{context} is not a directory")
    return resolved


def _resolved_regular_file(path: str | Path, context: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or _is_reparse_point(candidate):
        _fail("unsafe_path", f"{context} must be an absolute non-link file")
    try:
        resolved = candidate.resolve(strict=True)
        file_stat = resolved.stat()
    except OSError as exc:
        raise DINOv3SemanticOrchestrationError(
            "path_validation_failed", f"{context} does not exist"
        ) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        _fail("unsafe_path", f"{context} is not a regular file")
    return resolved


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _roots_overlap(left: Path, right: Path) -> bool:
    return _contains(left, right) or _contains(right, left)


def _validate_timeout(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        _fail("schema_validation_failed", "timeout_seconds must be positive and finite")
    return float(value)


def _safe_identifier(value: str, context: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_PATTERN.fullmatch(value) is None:
        _fail("schema_validation_failed", f"{context} must be a safe identifier")
    return value


def _safe_relative_path(value: str, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail("unsafe_path", f"{context} must be a non-empty POSIX relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        _fail("unsafe_path", f"{context} must be relative")
    if any(part in {"", ".", ".."} or ":" in part for part in value.split("/")):
        _fail("unsafe_path", f"{context} contains unsafe components")
    return posix


def _safe_workspace_file(workspace: Path, relative_path: str, context: str) -> Path:
    relative = _safe_relative_path(relative_path, context)
    current = workspace
    for part in relative.parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            _fail("artifact_validation_failed", f"{context} does not exist")
        if _is_reparse_point(current):
            _fail("unsafe_path", f"{context} contains a link or reparse point")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise DINOv3SemanticOrchestrationError(
            "artifact_validation_failed", f"{context} cannot be resolved"
        ) from exc
    if not _contains(workspace, resolved) or not resolved.is_file():
        _fail("unsafe_path", f"{context} escapes the task workspace")
    return resolved


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise DINOv3SemanticOrchestrationError(
            "response_validation_failed", "value is not canonical JSON data"
        ) from exc


def _response_dict(response: object) -> dict[str, Any]:
    if hasattr(response, "to_dict"):
        value = response.to_dict()
    elif isinstance(response, Mapping):
        value = dict(response)
    else:
        _fail("response_validation_failed", "unsupported response object")
    if not isinstance(value, dict):
        _fail("response_validation_failed", "response serialization is not an object")
    return value


def _semantic_projection(response: object) -> dict[str, Any]:
    if hasattr(response, "semantic_projection"):
        value = response.semantic_projection()
    else:
        value = _response_dict(response)
        value.pop("run_id", None)
    if not isinstance(value, Mapping):
        _fail("response_validation_failed", "semantic projection is not an object")
    return dict(value)


def _reject_sensitive_output(value: object, path: str = "evidence") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_sensitive_output(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive_output(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail("sensitive_data_rejected", f"{path} contains a non-finite value")
    elif isinstance(value, str):
        windows = PureWindowsPath(value)
        if PurePosixPath(value).is_absolute() or windows.is_absolute() or windows.drive:
            _fail("sensitive_data_rejected", f"{path} contains an absolute path")
        if _EMAIL_PATTERN.search(value) or _SECRET_VALUE_PATTERN.search(value):
            _fail("sensitive_data_rejected", f"{path} contains sensitive data")


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    """One path-relative protection record retained only in memory."""

    relative_path: str
    kind: str
    size_bytes: int
    sha256: str
    link_target: str
    metadata_stamp: int

    def canonical_value(self) -> tuple[object, ...]:
        return (
            self.relative_path,
            self.kind,
            self.size_bytes,
            self.sha256,
            self.link_target,
            self.metadata_stamp,
        )


@dataclass(frozen=True, slots=True)
class TreeInventory:
    """Exact path-free-comparable inventory of one protected root."""

    exists: bool
    entries: tuple[InventoryEntry, ...]
    credential_like_count: int

    @property
    def sha256(self) -> str:
        value = {
            "credential_like_count": self.credential_like_count,
            "entries": [entry.canonical_value() for entry in self.entries],
            "exists": self.exists,
        }
        return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "credential_like_count": self.credential_like_count,
            "directory_count": sum(entry.kind == "directory" for entry in self.entries),
            "exists": self.exists,
            "file_count": sum(entry.kind in {"file", "credential-like"} for entry in self.entries),
            "reparse_count": sum(entry.kind == "reparse" for entry in self.entries),
            "sha256": self.sha256,
            "total_file_bytes": sum(
                entry.size_bytes
                for entry in self.entries
                if entry.kind in {"file", "credential-like"}
            ),
        }


def _credential_like(relative_path: str) -> bool:
    name = PurePosixPath(relative_path).name.casefold()
    return name in _CREDENTIAL_LIKE_NAMES or name.startswith(".env")


def _excluded_path(relative_path: str, exclusions: frozenset[str]) -> bool:
    folded = relative_path.casefold()
    return any(
        folded == excluded.casefold()
        or folded.startswith(excluded.casefold().rstrip("/") + "/")
        for excluded in exclusions
    )


def _tree_inventory(
    root: Path,
    *,
    exclusions: frozenset[str] = frozenset(),
    hash_cache: dict[tuple[str, int, int], str] | None = None,
) -> TreeInventory:
    """Inventory one root without following links or opening credential files."""

    candidate = Path(root)
    if not candidate.is_absolute():
        _fail("unsafe_path", "inventory root must be absolute")
    if not candidate.exists() and not candidate.is_symlink():
        return TreeInventory(False, (), 0)
    if _is_reparse_point(candidate):
        try:
            target = os.readlink(candidate)
            size = candidate.lstat().st_size
        except OSError as exc:
            raise DINOv3SemanticOrchestrationError(
                "inventory_failed", "unable to inspect a protected root link"
            ) from exc
        return TreeInventory(
            True,
            (InventoryEntry(".", "reparse", size, "", target, 0),),
            0,
        )
    resolved = _resolved_directory(candidate, "inventory_root")
    cache = {} if hash_cache is None else hash_cache
    entries: list[InventoryEntry] = []
    credential_count = 0
    for directory, directory_names, file_names in os.walk(
        resolved, topdown=True, followlinks=False
    ):
        parent = Path(directory)
        retained: list[str] = []
        for name in sorted(directory_names):
            child = parent / name
            relative = child.relative_to(resolved).as_posix()
            if _excluded_path(relative, exclusions):
                continue
            if _is_reparse_point(child):
                try:
                    entries.append(
                        InventoryEntry(
                            relative,
                            "reparse",
                            child.lstat().st_size,
                            "",
                            os.readlink(child),
                            0,
                        )
                    )
                except OSError as exc:
                    raise DINOv3SemanticOrchestrationError(
                        "inventory_failed", "unable to inspect a protected link"
                    ) from exc
                continue
            entries.append(InventoryEntry(relative, "directory", 0, "", "", 0))
            retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            child = parent / name
            relative = child.relative_to(resolved).as_posix()
            if _excluded_path(relative, exclusions):
                continue
            if _is_reparse_point(child):
                try:
                    entries.append(
                        InventoryEntry(
                            relative,
                            "reparse",
                            child.lstat().st_size,
                            "",
                            os.readlink(child),
                            0,
                        )
                    )
                except OSError as exc:
                    raise DINOv3SemanticOrchestrationError(
                        "inventory_failed", "unable to inspect a protected link"
                    ) from exc
                continue
            try:
                file_stat = child.stat()
            except OSError as exc:
                raise DINOv3SemanticOrchestrationError(
                    "inventory_failed", "unable to inspect a protected file"
                ) from exc
            if not stat.S_ISREG(file_stat.st_mode):
                _fail("inventory_failed", "protected path is not a regular file")
            if _credential_like(relative):
                credential_count += 1
                entries.append(
                    InventoryEntry(
                        relative,
                        "credential-like",
                        file_stat.st_size,
                        "",
                        "",
                        file_stat.st_mtime_ns,
                    )
                )
                continue
            cache_key = (str(child.resolve(strict=True)), file_stat.st_size, file_stat.st_mtime_ns)
            digest = cache.get(cache_key)
            if digest is None:
                digest = sha256_file(child)
                cache[cache_key] = digest
            entries.append(
                InventoryEntry(relative, "file", file_stat.st_size, digest, "", 0)
            )
    return TreeInventory(True, tuple(sorted(entries, key=lambda item: item.relative_path)), credential_count)


def _default_cache_roots(environ: Mapping[str, str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    local_app_data = environ.get("LOCALAPPDATA")
    user_profile = environ.get("USERPROFILE")
    if isinstance(local_app_data, str) and local_app_data:
        local = Path(local_app_data)
        if local.is_absolute():
            roots["pip_cache"] = local / "pip" / "Cache"
            roots["transaction_root"] = (
                local
                / "VEA"
                / "transactions"
                / "dinov3-r5931719-registry-rebind-v1"
            )
    hub_value = environ.get("HUGGINGFACE_HUB_CACHE")
    if not isinstance(hub_value, str) or not hub_value:
        hf_home = environ.get("HF_HOME")
        if isinstance(hf_home, str) and hf_home:
            hub_value = str(Path(hf_home) / "hub")
        elif isinstance(user_profile, str) and user_profile:
            hub_value = str(Path(user_profile) / ".cache" / "huggingface" / "hub")
    if isinstance(hub_value, str) and hub_value and Path(hub_value).is_absolute():
        hub = Path(hub_value)
        roots["user_hub_cache"] = hub
        roots["clip_cache"] = hub / "models--openai--clip-vit-base-patch16"
        roots["siglip2_cache"] = hub / "models--google--siglip2-base-patch16-224"
    return roots


def capture_protection_inventories(
    *,
    repository_root: Path,
    runtime_root: Path,
    snapshot_root: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, TreeInventory]:
    """Capture all task-protected trees without serializing machine paths."""

    environment = os.environ if environ is None else environ
    hash_cache: dict[tuple[str, int, int], str] = {}
    result = {
        "repository_protected": _tree_inventory(
            repository_root,
            exclusions=frozenset({".git", *_ALLOWED_REPOSITORY_PATHS}),
            hash_cache=hash_cache,
        ),
        "repository_artifacts": _tree_inventory(
            repository_root / "artifacts", hash_cache=hash_cache
        ),
        "persistent_runtime": _tree_inventory(runtime_root, hash_cache=hash_cache),
        "external_snapshot": _tree_inventory(snapshot_root, hash_cache=hash_cache),
    }
    for label, root in _default_cache_roots(environment).items():
        result[label] = _tree_inventory(root, hash_cache=hash_cache)
    return result


@dataclass(frozen=True, slots=True)
class ProbeInvocation:
    response: DINOv3SemanticResponse
    process: Mapping[str, Any]


def _validate_captured_stdout(stdout: object, stderr: object) -> dict[str, Any]:
    if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
        _fail("probe_output_invalid", "probe process output must be captured as bytes")
    if len(stdout) > _MAX_CAPTURE_BYTES or len(stderr) > _MAX_CAPTURE_BYTES:
        _fail("probe_output_invalid", "probe process output exceeds its bound")
    try:
        lines = tuple(line for line in stdout.decode("utf-8", errors="strict").splitlines() if line)
        error_text = stderr.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise DINOv3SemanticOrchestrationError(
            "probe_output_invalid", "probe process output is not strict UTF-8"
        ) from exc
    if lines != PROBE_MARKERS:
        _fail("probe_output_invalid", "probe stdout markers do not match")
    if error_text:
        _fail("probe_output_invalid", "successful probe wrote to stderr")
    return {
        "return_code": 0,
        "shell": False,
        "stderr_bytes": 0,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
    }


def invoke_pretrained_semantic_probe(
    *,
    semantic_profile: object,
    semantic_profile_path: Path,
    runtime_profile_path: Path,
    snapshot_profile_path: Path,
    runtime_root: Path,
    snapshot_root: Path,
    repository_root: Path,
    probe_script: Path,
    workspace_root: Path,
    run_id: str,
    timeout_seconds: float,
    process_runner: Callable[..., Any] = subprocess.run,
    environment_builder: Callable[..., dict[str, str]] = build_sanitized_worker_environment,
    python_resolver: Callable[[str | Path], Path] = runtime_python_path,
    request_writer: Callable[..., Any] = write_semantic_request,
    response_reader: Callable[..., DINOv3SemanticResponse] = read_semantic_response,
) -> ProbeInvocation:
    """Launch one persistent probe using an explicit argv and strict protocol."""

    timeout = _validate_timeout(timeout_seconds)
    run = _safe_identifier(run_id, "run_id")
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
    profile_file = _resolved_regular_file(semantic_profile_path, "semantic_profile")
    runtime_profile_file = _resolved_regular_file(runtime_profile_path, "runtime_profile")
    snapshot_profile_file = _resolved_regular_file(snapshot_profile_path, "snapshot_profile")
    probe = _resolved_regular_file(probe_script, "probe_script")
    request_file = workspace / "request.json"
    response_file = workspace / "response.json"
    if any(path.exists() or path.is_symlink() for path in (request_file, response_file)):
        _fail("unsafe_path", "probe protocol paths must be fresh")
    profile_sha256 = getattr(
        semantic_profile,
        "sha256",
        getattr(semantic_profile, "source_sha256", None),
    )
    if not isinstance(profile_sha256, str) or _SHA256_PATTERN.fullmatch(profile_sha256) is None:
        _fail("profile_identity_mismatch", "semantic profile source hash is unavailable")
    request_writer(
        request_file,
        run_id=run,
        semantic_profile_sha256=profile_sha256,
        workspace_relative_artifact_directory="actual-output",
    )
    python = _resolved_regular_file(python_resolver(runtime), "runtime_python")
    environment = dict(
        environment_builder(
            runtime_python=python,
            repository_root=repository,
            workspace_root=workspace,
        )
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "DO_NOT_TRACK": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    command = [
        str(python),
        "-B",
        str(probe),
        "--semantic-profile",
        str(profile_file),
        "--runtime-profile",
        str(runtime_profile_file),
        "--snapshot-profile",
        str(snapshot_profile_file),
        "--runtime-root",
        str(runtime),
        "--snapshot-root",
        str(snapshot),
        "--workspace-root",
        str(workspace),
        "--request",
        request_file.name,
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
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DINOv3SemanticOrchestrationError(
            "probe_timeout", "pretrained semantic probe exceeded its timeout"
        ) from exc
    except OSError as exc:
        raise DINOv3SemanticOrchestrationError(
            "probe_launch_failed", "pretrained semantic probe could not be launched"
        ) from exc
    return_code = getattr(result, "returncode", None)
    if return_code != 0:
        cause = subprocess.CalledProcessError(
            int(return_code) if isinstance(return_code, int) else -1,
            command,
        )
        raise DINOv3SemanticOrchestrationError(
            "probe_nonzero_exit", "pretrained semantic probe returned a nonzero exit"
        ) from cause
    process = _validate_captured_stdout(
        getattr(result, "stdout", None), getattr(result, "stderr", None)
    )
    try:
        response = response_reader(
            response_file,
            profile=semantic_profile,
            workspace_root=workspace,
        )
    except Exception as exc:
        raise DINOv3SemanticOrchestrationError(
            "probe_response_invalid", "pretrained semantic response is invalid"
        ) from exc
    response_value = _response_dict(response)
    if response_value.get("run_id") != run:
        _fail("probe_response_mismatch", "response run ID does not match request")
    if tuple(response_value.get("markers", ())) != PROBE_MARKERS:
        _fail("probe_response_mismatch", "response markers do not match")
    return ProbeInvocation(response=response, process=process)


def validate_actual_output_artifacts(
    *,
    response: object,
    workspace_root: Path,
) -> dict[str, Any]:
    """Read every actual-output artifact again using the current environment."""

    workspace = _resolved_directory(workspace_root, "workspace_root")
    value = _response_dict(response)
    artifact = value.get("artifact")
    if not isinstance(artifact, Mapping):
        _fail("artifact_validation_failed", "response artifact summary is missing")
    entries = artifact.get("entries")
    if not isinstance(entries, list) or len(entries) != len(SELECTED_ARTIFACT_TAPS):
        _fail("artifact_validation_failed", "artifact entry count is invalid")
    if tuple(artifact.get("selected_taps", ())) != SELECTED_ARTIFACT_TAPS:
        _fail("artifact_validation_failed", "artifact selected taps are invalid")
    if artifact.get("roundtrip_validated") is not True:
        _fail("artifact_validation_failed", "persistent artifact round-trip failed")
    directory_relative = artifact.get("workspace_relative_directory")
    if directory_relative != "actual-output":
        _fail("artifact_validation_failed", "artifact directory is invalid")
    workspace_paths = value.get("workspace_paths")
    if not isinstance(workspace_paths, list):
        _fail("artifact_validation_failed", "workspace path inventory is invalid")
    expected_paths: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            _fail("artifact_validation_failed", "artifact entry is not an object")
        tap_name = entry.get("tap_name")
        if tap_name != SELECTED_ARTIFACT_TAPS[index]:
            _fail("artifact_validation_failed", "artifact tap order is invalid")
        sidecar_relative = entry.get("sidecar_relative_path")
        tensor_relative = entry.get("safetensors_relative_path")
        if not isinstance(sidecar_relative, str) or not isinstance(tensor_relative, str):
            _fail("artifact_validation_failed", "artifact paths are invalid")
        sidecar = _safe_workspace_file(
            workspace, sidecar_relative, f"artifact.entries[{index}].sidecar"
        )
        tensor = _safe_workspace_file(
            workspace, tensor_relative, f"artifact.entries[{index}].safetensors"
        )
        expected_paths.update({sidecar_relative, tensor_relative})
        loaded = read_feature_artifact(sidecar)
        metadata = loaded.metadata
        if metadata.run_id != value.get("run_id"):
            _fail("artifact_validation_failed", "artifact run ID does not match")
        if (
            metadata.model_key != MODEL_KEY
            or metadata.tap_namespace != "backbone"
            or metadata.tap_name != tap_name
            or tuple(metadata.image_ids) != IMAGE_IDS
            or metadata.grid_height != 14
            or metadata.grid_width != 14
            or metadata.patch_count != 196
            or metadata.hidden_dimension != 768
            or metadata.dtype != "float32"
            or metadata.shard_index != index
            or metadata.sidecar_file_name != sidecar.name
            or metadata.safetensors_file_name != tensor.name
            or metadata.run_status != "complete"
        ):
            _fail("artifact_validation_failed", "artifact metadata is inconsistent")
        if loaded.patch.shape != PATCH_SHAPE or loaded.global_mean.shape != GLOBAL_SHAPE:
            _fail("artifact_validation_failed", "artifact tensor shape is invalid")
        if loaded.patch.dtype != np.dtype(np.float32) or loaded.global_mean.dtype != np.dtype(np.float32):
            _fail("artifact_validation_failed", "artifact tensor dtype is invalid")
        patch_sha256 = _sha256_array(loaded.patch)
        global_sha256 = _sha256_array(loaded.global_mean)
        checksum_sha256 = sha256_file(tensor)
        for field, actual in (
            ("patch_sha256", patch_sha256),
            ("global_sha256", global_sha256),
            ("checksum_sha256", checksum_sha256),
        ):
            expected = entry.get(field)
            if not isinstance(expected, str) or not _SHA256_PATTERN.fullmatch(expected):
                _fail("artifact_validation_failed", f"artifact {field} is invalid")
            if actual != expected:
                _fail("artifact_validation_failed", f"artifact {field} does not match")
        if metadata.checksum_value != checksum_sha256:
            _fail("artifact_validation_failed", "sidecar checksum does not match")
        summaries.append(
            {
                "checksum_sha256": checksum_sha256,
                "global_sha256": global_sha256,
                "global_shape": list(loaded.global_mean.shape),
                "patch_sha256": patch_sha256,
                "patch_shape": list(loaded.patch.shape),
                "tap_name": tap_name,
            }
        )
    if set(workspace_paths) != expected_paths or len(workspace_paths) != len(expected_paths):
        _fail("artifact_validation_failed", "workspace path inventory does not match artifacts")
    artifact_directory = workspace / str(directory_relative)
    if not artifact_directory.is_dir() or _is_reparse_point(artifact_directory):
        _fail("artifact_validation_failed", "artifact directory is unsafe")
    actual_files = {
        path.relative_to(workspace).as_posix()
        for path in artifact_directory.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_paths:
        _fail("artifact_validation_failed", "artifact directory contains unexpected files")
    return {
        "entry_count": len(summaries),
        "entries": summaries,
        "image_ids": list(IMAGE_IDS),
        "roundtrip_validated": True,
        "selected_taps": list(SELECTED_ARTIFACT_TAPS),
    }


def _feature_hashes(response: object) -> tuple[str, ...]:
    taps = _response_dict(response).get("taps")
    if not isinstance(taps, list) or len(taps) != 14:
        _fail("response_validation_failed", "response must contain 14 tap records")
    hashes: list[str] = []
    for index, tap in enumerate(taps):
        if not isinstance(tap, Mapping):
            _fail("response_validation_failed", f"tap {index} is invalid")
        for field in ("patch_sha256", "global_sha256"):
            digest = tap.get(field)
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                _fail("response_validation_failed", f"tap {index} {field} is invalid")
            hashes.append(digest)
    return tuple(hashes)


def _manifest_hash(value: object) -> str:
    digest = getattr(value, "source_sha256", None)
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        _fail("identity_validation_failed", "manifest source hash is unavailable")
    return digest


def _protection_summaries(
    inventories: Mapping[str, TreeInventory],
) -> dict[str, dict[str, Any]]:
    return {label: inventories[label].summary() for label in sorted(inventories)}


def run_pretrained_semantic_probe(
    *,
    semantic_profile_path: Path,
    runtime_profile_path: Path,
    snapshot_profile_path: Path,
    runtime_root: Path | None = None,
    snapshot_root: Path | None = None,
    repository_root: Path | None = None,
    probe_script: Path | None = None,
    timeout_seconds: float = 900.0,
    semantic_profile_loader: Callable[[str | Path], Any] = load_semantic_profile,
    runtime_profile_loader: Callable[[str | Path], Any] = load_runtime_profile,
    snapshot_profile_loader: Callable[[str | Path], Any] = load_snapshot_profile,
    identity_verifier: Callable[..., Any] = verify_rebound_identity,
    runtime_validator: Callable[..., Any] = validate_runtime_root,
    snapshot_validator: Callable[..., Any] = validate_snapshot_root,
    runtime_root_resolver: Callable[[Any], Path] = resolve_default_runtime_root,
    snapshot_root_resolver: Callable[[Any], Path] = resolve_default_snapshot_root,
    probe_invoker: Callable[..., ProbeInvocation] = invoke_pretrained_semantic_probe,
    artifact_validator: Callable[..., dict[str, Any]] = validate_actual_output_artifacts,
    protection_capture: Callable[..., dict[str, TreeInventory]] = capture_protection_inventories,
    temporary_directory_factory: Callable[..., Any] = tempfile.TemporaryDirectory,
    enforce_repository_sources: bool = False,
) -> dict[str, Any]:
    """Run two independent probes and return path-free, validated evidence."""

    timeout = _validate_timeout(timeout_seconds)
    repository = _resolved_directory(
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root,
        "repository_root",
    )
    def repository_input(path: Path, context: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = repository / candidate
        return _resolved_regular_file(candidate, context)

    semantic_profile_file = repository_input(
        semantic_profile_path, "semantic_profile"
    )
    runtime_profile_file = repository_input(runtime_profile_path, "runtime_profile")
    snapshot_profile_file = repository_input(snapshot_profile_path, "snapshot_profile")
    semantic_profile = semantic_profile_loader(semantic_profile_file)
    runtime_profile = runtime_profile_loader(runtime_profile_file)
    snapshot_profile = snapshot_profile_loader(snapshot_profile_file)
    runtime = _resolved_directory(
        runtime_root_resolver(runtime_profile) if runtime_root is None else runtime_root,
        "runtime_root",
    )
    snapshot = _resolved_directory(
        snapshot_root_resolver(snapshot_profile) if snapshot_root is None else snapshot_root,
        "snapshot_root",
    )
    probe = _resolved_regular_file(
        repository / "scripts" / "probe_dinov3_pretrained_semantics.py"
        if probe_script is None
        else probe_script,
        "probe_script",
    )
    if any(
        _roots_overlap(left, right)
        for index, left in enumerate((repository, runtime, snapshot))
        for right in (repository, runtime, snapshot)[index + 1 :]
    ):
        _fail("unsafe_path", "repository, runtime, and snapshot roots must be disjoint")
    if enforce_repository_sources:
        expected = (
            repository
            / "configs"
            / "semantic_probes"
            / "dinov3_vit_b16_r5931719_semantics_v0.json",
            repository
            / "configs"
            / "runtime_profiles"
            / "dinov3_isolated_manual_v0.json",
            repository
            / "configs"
            / "model_snapshots"
            / "dinov3_vit_b16_r5931719_v0.json",
            repository / "scripts" / "probe_dinov3_pretrained_semantics.py",
        )
        actual = (
            semantic_profile_file,
            runtime_profile_file,
            snapshot_profile_file,
            probe,
        )
        if tuple(path.resolve(strict=True) for path in expected) != actual:
            _fail(
                "source_identity_mismatch",
                "semantic probe inputs are not the exact repository deliverables",
            )

    identity_verifier(
        repository_root=repository,
        runtime_profile_path=runtime_profile_file,
        snapshot_profile_path=snapshot_profile_file,
        runtime_root=runtime,
        snapshot_root=snapshot,
        persistent_runtime_root=runtime,
    )
    runtime_manifest = runtime_validator(
        runtime, runtime_profile, repository_root=repository
    )
    snapshot_manifest = snapshot_validator(
        snapshot,
        snapshot_profile,
        repository_root=repository,
        persistent_runtime_root=runtime,
    )
    protected_before = protection_capture(
        repository_root=repository,
        runtime_root=runtime,
        snapshot_root=snapshot,
    )

    workspace_path: Path | None = None
    invocations: list[ProbeInvocation] = []
    artifact_results: list[dict[str, Any]] = []
    scratch_cache_summaries: list[dict[str, Any]] = []
    primary_error: Exception | None = None
    try:
        with temporary_directory_factory(prefix="vea_t007a4_semantic_") as temporary:
            workspace_path = _resolved_directory(Path(temporary), "workspace_root")
            if any(
                _roots_overlap(workspace_path, protected)
                for protected in (repository, runtime, snapshot)
            ):
                _fail("unsafe_path", "task workspace overlaps a protected root")
            for repeat_index in range(REPEAT_COUNT):
                repeat_root = workspace_path / f"repeat-{repeat_index + 1:02d}"
                repeat_root.mkdir()
                run_id = f"t007a4-semantic-repeat-{repeat_index + 1:02d}"
                invocation = probe_invoker(
                    semantic_profile=semantic_profile,
                    semantic_profile_path=semantic_profile_file,
                    runtime_profile_path=runtime_profile_file,
                    snapshot_profile_path=snapshot_profile_file,
                    runtime_root=runtime,
                    snapshot_root=snapshot,
                    repository_root=repository,
                    probe_script=probe,
                    workspace_root=repeat_root,
                    run_id=run_id,
                    timeout_seconds=timeout,
                )
                invocations.append(invocation)
                artifact_results.append(
                    artifact_validator(
                        response=invocation.response,
                        workspace_root=repeat_root,
                    )
                )
                scratch = repeat_root / ".runtime-scratch" / "cache"
                scratch_inventory = _tree_inventory(scratch)
                scratch_summary = scratch_inventory.summary()
                if scratch_summary["file_count"] != 0 or scratch_summary["reparse_count"] != 0:
                    _fail("model_cache_changed", "probe created task scratch cache files")
                scratch_cache_summaries.append(scratch_summary)
            first_projection = _canonical_json_bytes(
                _semantic_projection(invocations[0].response)
            )
            second_projection = _canonical_json_bytes(
                _semantic_projection(invocations[1].response)
            )
            if first_projection != second_projection:
                _fail(
                    "cross_process_nondeterministic",
                    "independent semantic projections differ",
                )
            first_hashes = _feature_hashes(invocations[0].response)
            second_hashes = _feature_hashes(invocations[1].response)
            if len(first_hashes) != 28 or first_hashes != second_hashes:
                _fail(
                    "cross_process_nondeterministic",
                    "independent 28-feature hashes differ",
                )
            if _canonical_json_bytes(artifact_results[0]) != _canonical_json_bytes(
                artifact_results[1]
            ):
                _fail(
                    "cross_process_nondeterministic",
                    "current-side artifact projections differ",
                )
    except Exception as exc:
        primary_error = exc

    workspace_cleaned = workspace_path is not None and not workspace_path.exists()
    try:
        protected_after = protection_capture(
            repository_root=repository,
            runtime_root=runtime,
            snapshot_root=snapshot,
        )
        final_runtime_manifest = runtime_validator(
            runtime, runtime_profile, repository_root=repository
        )
        final_snapshot_manifest = snapshot_validator(
            snapshot,
            snapshot_profile,
            repository_root=repository,
            persistent_runtime_root=runtime,
        )
        identity_verifier(
            repository_root=repository,
            runtime_profile_path=runtime_profile_file,
            snapshot_profile_path=snapshot_profile_file,
            runtime_root=runtime,
            snapshot_root=snapshot,
            persistent_runtime_root=runtime,
        )
    except Exception as exc:
        if primary_error is None:
            primary_error = exc
        else:
            raise DINOv3SemanticOrchestrationError(
                "post_verification_failed",
                "post-run verification failed after another probe failure",
            ) from primary_error
        protected_after = {}
        final_runtime_manifest = None
        final_snapshot_manifest = None
    if not workspace_cleaned:
        raise DINOv3SemanticOrchestrationError(
            "workspace_cleanup_failed", "task-owned semantic workspace was not removed"
        ) from primary_error
    if protected_after != protected_before:
        raise DINOv3SemanticOrchestrationError(
            "protected_inventory_changed", "semantic probe changed protected content"
        ) from primary_error
    if (
        final_runtime_manifest is None
        or final_snapshot_manifest is None
        or _manifest_hash(final_runtime_manifest) != _manifest_hash(runtime_manifest)
        or _manifest_hash(final_snapshot_manifest) != _manifest_hash(snapshot_manifest)
    ):
        raise DINOv3SemanticOrchestrationError(
            "protected_inventory_changed", "validated manifest identity changed"
        ) from primary_error
    if primary_error is not None:
        raise primary_error

    first_response = _response_dict(invocations[0].response)
    projection = _canonical_json_bytes(_semantic_projection(invocations[0].response))
    feature_hashes = _feature_hashes(invocations[0].response)
    evidence = {
        "artifact": artifact_results[0],
        "credential_direct_read_count": 0,
        "feature_hash_count": len(feature_hashes),
        "feature_hashes": list(feature_hashes),
        "markers": list(PROBE_MARKERS),
        "network_operation_count": 0,
        "package_operation_count": 0,
        "processes": [dict(invocation.process) for invocation in invocations],
        "protected": _protection_summaries(protected_before),
        "protected_unchanged": True,
        "ready_for_t007b": True,
        "repeat_count": REPEAT_COUNT,
        "repeat_deterministic": True,
        "response": first_response,
        "scratch_model_caches": scratch_cache_summaries,
        "semantic_projection_sha256": hashlib.sha256(projection).hexdigest(),
        "semantic_status": SEMANTIC_STATUS,
        "status": "PASS",
        "workspace_cleaned": True,
    }
    _reject_sensitive_output(evidence)
    return evidence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-profile", required=True, type=Path)
    parser.add_argument("--runtime-profile", required=True, type=Path)
    parser.add_argument("--snapshot-profile", required=True, type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        evidence = run_pretrained_semantic_probe(
            semantic_profile_path=args.semantic_profile,
            runtime_profile_path=args.runtime_profile,
            snapshot_profile_path=args.snapshot_profile,
            runtime_root=args.runtime_root,
            snapshot_root=args.snapshot_root,
            timeout_seconds=args.timeout_seconds,
            enforce_repository_sources=True,
        )
    except Exception as exc:
        classification = getattr(exc, "classification", type(exc).__name__)
        print(
            f"DINOV3_PRETRAINED_SEMANTIC_ORCHESTRATION_FAILED:{classification}",
            file=sys.stderr,
        )
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
