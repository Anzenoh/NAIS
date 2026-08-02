"""Load the exact local DINOv3 checkpoint inside the persistent runtime.

The module has a standard-library-only import surface.  Offline policy and the
socket guard are established before PyTorch or Transformers is imported.
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
from typing import Any, Callable, Mapping


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

_MAX_RESPONSE_BYTES = 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EMAIL_PATTERN = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
)
_SECRET_VALUE_PATTERN = re.compile(
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


class CheckpointProbeError(RuntimeError):
    """Raised when the isolated checkpoint-load contract is violated."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def _fail(classification: str, message: str) -> None:
    raise CheckpointProbeError(classification, message)


class _SocketGuard:
    """Deny network operations while retaining an auditable attempt count."""

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
                raise RuntimeError("network access is disabled for checkpoint load")

            def connect_ex(self, address: object) -> int:
                guard.attempt_count += 1
                raise RuntimeError("network access is disabled for checkpoint load")

        def guarded_create_connection(*args: object, **kwargs: object) -> None:
            guard.attempt_count += 1
            raise RuntimeError("network access is disabled for checkpoint load")

        def guarded_getaddrinfo(*args: object, **kwargs: object) -> None:
            guard.attempt_count += 1
            raise RuntimeError("network access is disabled for checkpoint load")

        socket.socket = GuardedSocket
        socket.create_connection = guarded_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]

    def restore(self) -> None:
        socket.socket = self._socket_class
        socket.create_connection = self._create_connection
        socket.getaddrinfo = self._getaddrinfo


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
        raise CheckpointProbeError(
            "path_validation_failed", "unable to inspect a protected path"
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointProbeError(
            "inventory_failed", "unable to hash a protected file"
        ) from exc
    return digest.hexdigest()


def _credential_like(relative_path: str) -> bool:
    name = PurePosixPath(relative_path).name.casefold()
    return name in _CREDENTIAL_LIKE_NAMES or name.startswith(".env")


def _directory_inventory(root: Path) -> str:
    """Return a deterministic content inventory without reading credentials."""

    if not root.exists() and not root.is_symlink():
        records: list[dict[str, Any]] = []
    else:
        if not root.is_absolute() or _is_reparse_point(root):
            _fail("unsafe_path", "protected inventory root is unsafe")
        try:
            resolved = root.resolve(strict=True)
        except OSError as exc:
            raise CheckpointProbeError(
                "path_validation_failed", "protected inventory root is unavailable"
            ) from exc
        if not resolved.is_dir():
            _fail("unsafe_path", "protected inventory root is not a directory")
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
                    raise CheckpointProbeError(
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
    serialized = json.dumps(
        records,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolved_directory(path: Path, context: str) -> Path:
    if not path.is_absolute() or _is_reparse_point(path):
        _fail("unsafe_path", f"{context} must be an absolute non-link directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CheckpointProbeError(
            "path_validation_failed", f"{context} does not exist"
        ) from exc
    if not resolved.is_dir():
        _fail("unsafe_path", f"{context} is not a directory")
    return resolved


def _validate_offline_environment(workspace: Path) -> None:
    required = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    for key, expected in required.items():
        if os.environ.get(key) != expected:
            _fail("offline_environment_invalid", f"required offline variable {key} is invalid")
    for key in os.environ:
        upper = key.upper()
        if key not in {
            "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            "TOKENIZERS_PARALLELISM",
        } and any(
            fragment in upper for fragment in _SECRET_ENV_FRAGMENTS
        ):
            _fail("offline_environment_invalid", "a secret-like variable survived sanitization")
    for key in ("HF_HOME", "XDG_CACHE_HOME", "TEMP", "TMP"):
        value = os.environ.get(key)
        if not isinstance(value, str) or not value:
            _fail("offline_environment_invalid", f"required scratch variable {key} is absent")
        candidate = Path(value)
        if not candidate.is_absolute():
            _fail("offline_environment_invalid", f"scratch variable {key} is not absolute")
        try:
            normalized = candidate.resolve(strict=False)
        except OSError as exc:
            raise CheckpointProbeError(
                "offline_environment_invalid", "unable to resolve a scratch variable"
            ) from exc
        if not _contains(workspace, normalized):
            _fail("offline_environment_invalid", "scratch storage escapes the task workspace")


def _numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    import numpy as np

    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sequence_length(value: object, field: str) -> int:
    if not isinstance(value, (list, tuple)):
        _fail("loading_info_invalid", f"{field} must be a sequence")
    return len(value)


def _load_model_and_facts(files_directory: Path, snapshot_profile: Any) -> dict[str, Any]:
    """Perform the sole heavy-import and checkpoint-load portion of the probe."""

    import numpy as np
    import torch
    from transformers.models.dinov3_vit.configuration_dinov3_vit import (
        DINOv3ViTConfig,
    )
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()

    with torch.random.fork_rng(devices=[], enabled=True):
        config = DINOv3ViTConfig.from_pretrained(
            str(files_directory),
            local_files_only=True,
            trust_remote_code=False,
            token=False,
        )
        loaded = DINOv3ViTModel.from_pretrained(
            str(files_directory),
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            token=False,
            use_safetensors=True,
            torch_dtype=torch.float32,
            output_loading_info=True,
        )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        _fail("model_load_failed", "model loader did not return loading information")
    model, loading_info = loaded
    if not isinstance(loading_info, Mapping):
        _fail("loading_info_invalid", "loading information is not a mapping")

    model.eval()
    model.requires_grad_(False)

    parameters = tuple(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    parameter_tensor_count = len(parameters)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    devices = sorted({parameter.device.type for parameter in parameters})
    dtypes = sorted({str(parameter.dtype).removeprefix("torch.") for parameter in parameters})

    components = {
        "cls_token": getattr(getattr(model, "embeddings", None), "cls_token", None)
        is not None,
        "encoder_layer_count": len(getattr(model, "layer", ())),
        "final_norm": getattr(model, "norm", None) is not None,
        "patch_embedding": getattr(
            getattr(model, "embeddings", None), "patch_embeddings", None
        )
        is not None,
        "register_tokens": getattr(
            getattr(model, "embeddings", None), "register_tokens", None
        )
        is not None,
    }

    python_rng_after = random.getstate()
    numpy_rng_after = np.random.get_state()
    torch_rng_after = torch.random.get_rng_state().clone()
    rng = {
        "numpy_unchanged": _numpy_rng_equal(numpy_rng_before, numpy_rng_after),
        "python_unchanged": python_rng_before == python_rng_after,
        "torch_unchanged": bool(torch.equal(torch_rng_before, torch_rng_after)),
    }

    forbidden_root = "torch" + "vision"
    forbidden_import_count = sum(
        1
        for name in sys.modules
        if name == forbidden_root or name.startswith(forbidden_root + ".")
    )
    return {
        "components": components,
        "config_class": type(config).__name__,
        "device": devices[0] if len(devices) == 1 else "mixed",
        "dtype": dtypes[0] if len(dtypes) == 1 else "mixed",
        "evaluation_mode": not bool(model.training),
        "forward_count": 0,
        "forward_executed": False,
        "frozen": trainable_parameter_count == 0,
        "loading_info": {
            "error_message_count": _sequence_length(
                loading_info.get("error_msgs", []), "loading_info.error_msgs"
            ),
            "mismatched_key_count": _sequence_length(
                loading_info.get("mismatched_keys", []),
                "loading_info.mismatched_keys",
            ),
            "missing_key_count": _sequence_length(
                loading_info.get("missing_keys", []), "loading_info.missing_keys"
            ),
            "unexpected_key_count": _sequence_length(
                loading_info.get("unexpected_keys", []),
                "loading_info.unexpected_keys",
            ),
        },
        "model_class": type(model).__name__,
        "parameter_count": parameter_count,
        "parameter_tensor_count": parameter_tensor_count,
        "processor_object_count": 0,
        "rng": rng,
        "state_sha256": _state_sha256(model),
        "trainable_parameter_count": trainable_parameter_count,
        "training": bool(model.training),
        "torchvision_import_count": forbidden_import_count,
    }


def _attribute_or_key(value: object, name: str) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, Mapping) and name in value:
        return value[name]
    _fail("schema_validation_failed", f"required field {name} is unavailable")


def _expect_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail("schema_validation_failed", f"{field} is not a lowercase SHA-256")
    return value


def _validate_model_facts(facts: Mapping[str, Any], snapshot_profile: Any) -> None:
    expected = {
        "config_class": _attribute_or_key(snapshot_profile, "config_class"),
        "device": "cpu",
        "dtype": "float32",
        "evaluation_mode": True,
        "forward_count": 0,
        "forward_executed": False,
        "frozen": True,
        "model_class": _attribute_or_key(snapshot_profile, "model_class"),
        "parameter_count": _attribute_or_key(
            snapshot_profile, "expected_parameter_count"
        ),
        "parameter_tensor_count": _attribute_or_key(
            snapshot_profile, "expected_parameter_tensor_count"
        ),
        "processor_object_count": 0,
        "trainable_parameter_count": 0,
        "training": False,
        "torchvision_import_count": 0,
    }
    for key, expected_value in expected.items():
        if facts.get(key) != expected_value:
            _fail("model_fact_mismatch", f"model fact {key} does not match")
    if facts.get("loading_info") != {
        "error_message_count": 0,
        "mismatched_key_count": 0,
        "missing_key_count": 0,
        "unexpected_key_count": 0,
    }:
        _fail("loading_info_invalid", "checkpoint loading information is not clean")
    if facts.get("components") != {
        "cls_token": True,
        "encoder_layer_count": 12,
        "final_norm": True,
        "patch_embedding": True,
        "register_tokens": True,
    }:
        _fail("model_fact_mismatch", "required model components do not match")
    if facts.get("rng") != {
        "numpy_unchanged": True,
        "python_unchanged": True,
        "torch_unchanged": True,
    }:
        _fail("rng_changed", "checkpoint load changed caller RNG state")
    _expect_hash(facts.get("state_sha256"), "state_sha256")


def _reject_path_or_secret(value: object, path: str = "response") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_path_or_secret(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_path_or_secret(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail("response_invalid", f"{path} contains a non-finite number")
    elif isinstance(value, str):
        windows = PureWindowsPath(value)
        if PurePosixPath(value).is_absolute() or windows.is_absolute() or windows.drive:
            _fail("response_invalid", f"{path} contains an absolute path")
        if _EMAIL_PATTERN.search(value) or _SECRET_VALUE_PATTERN.search(value):
            _fail("response_invalid", f"{path} contains sensitive data")


def _write_response(path: Path, response: dict[str, Any]) -> None:
    _reject_path_or_secret(response)
    try:
        payload = (
            json.dumps(
                response,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointProbeError(
            "response_invalid", "unable to serialize probe response"
        ) from exc
    if len(payload) > _MAX_RESPONSE_BYTES:
        _fail("response_invalid", "probe response exceeds the size limit")
    if path.exists() or path.is_symlink():
        _fail("unsafe_path", "probe response path must be fresh")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise CheckpointProbeError(
            "response_write_failed", "unable to write probe response"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def run_checkpoint_load_probe(
    *,
    snapshot_profile_path: Path,
    runtime_profile_path: Path,
    snapshot_root: Path,
    runtime_root: Path,
    repository_root: Path,
    workspace_root: Path,
    snapshot_profile_loader: Callable[[str | Path], Any] | None = None,
    runtime_profile_loader: Callable[[str | Path], Any] | None = None,
    snapshot_validator: Callable[..., Any] | None = None,
    runtime_validator: Callable[..., Any] | None = None,
    files_directory_resolver: Callable[[str | Path], Path] | None = None,
    runtime_python_resolver: Callable[[str | Path], Path] | None = None,
    model_loader: Callable[[Path, Any], dict[str, Any]] = _load_model_and_facts,
    inventory_reader: Callable[[Path], str] = _directory_inventory,
    enforce_runtime_python: bool = True,
    enforce_offline_environment: bool = True,
    enforce_repository_sources: bool = False,
) -> dict[str, Any]:
    """Load and validate one exact local checkpoint without model inference."""

    workspace = _resolved_directory(workspace_root, "workspace_root")
    if enforce_offline_environment:
        _validate_offline_environment(workspace)

    guard = _SocketGuard()
    guard.install()
    try:
        from vision_encoder_alignment.runtime_boundary import (
            load_runtime_profile,
            runtime_python_path,
            validate_runtime_root,
        )
        from vision_encoder_alignment.snapshot_boundary import (
            load_snapshot_profile,
            snapshot_files_directory,
            validate_snapshot_root,
        )

        load_snapshot = snapshot_profile_loader or load_snapshot_profile
        load_runtime = runtime_profile_loader or load_runtime_profile
        verify_snapshot = snapshot_validator or validate_snapshot_root
        verify_runtime = runtime_validator or validate_runtime_root
        resolve_files = files_directory_resolver or snapshot_files_directory
        resolve_python = runtime_python_resolver or runtime_python_path

        repository = _resolved_directory(repository_root, "repository_root")
        runtime = _resolved_directory(runtime_root, "runtime_root")
        snapshot = _resolved_directory(snapshot_root, "snapshot_root")
        if any(
            _contains(left, right) or _contains(right, left)
            for index, left in enumerate((workspace, repository, runtime, snapshot))
            for right in (workspace, repository, runtime, snapshot)[index + 1 :]
        ):
            _fail("unsafe_path", "workspace and protected roots must be disjoint")

        snapshot_profile_file = snapshot_profile_path.resolve(strict=True)
        runtime_profile_file = runtime_profile_path.resolve(strict=True)
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
                or Path(__file__).resolve(strict=True) != expected_probe
            ):
                _fail(
                    "source_identity_mismatch",
                    "probe inputs are not the exact repository deliverables",
                )
        snapshot_profile = load_snapshot(snapshot_profile_file)
        runtime_profile = load_runtime(runtime_profile_file)
        if _attribute_or_key(snapshot_profile, "runtime_id") != _attribute_or_key(
            runtime_profile, "runtime_id"
        ):
            _fail("runtime_identity_mismatch", "snapshot and runtime IDs differ")
        if _attribute_or_key(
            snapshot_profile, "runtime_profile_sha256"
        ) != _attribute_or_key(runtime_profile, "source_sha256"):
            _fail("runtime_identity_mismatch", "runtime profile identity differs")

        runtime_manifest = verify_runtime(
            runtime, runtime_profile, repository_root=repository
        )
        snapshot_manifest = verify_snapshot(
            snapshot,
            snapshot_profile,
            repository_root=repository,
            persistent_runtime_root=runtime,
        )
        files_directory = resolve_files(snapshot)
        expected_python = resolve_python(runtime).resolve(strict=True)
        if enforce_runtime_python and Path(sys.executable).resolve(strict=True) != expected_python:
            _fail("wrong_python_runtime", "probe is not running under persistent Python")

        cache_root = Path(os.environ.get("HF_HOME", workspace / ".missing-cache"))
        before = {
            "model_cache": inventory_reader(cache_root),
            "persistent_runtime": inventory_reader(runtime),
            "snapshot": inventory_reader(snapshot),
        }
        facts = model_loader(files_directory, snapshot_profile)
        _validate_model_facts(facts, snapshot_profile)
        after = {
            "model_cache": inventory_reader(cache_root),
            "persistent_runtime": inventory_reader(runtime),
            "snapshot": inventory_reader(snapshot),
        }
        if before != after:
            _fail("protected_inventory_changed", "checkpoint load changed protected content")
        if guard.attempt_count != 0:
            _fail("network_attempted", "checkpoint load attempted network access")

        response = {
            "components": facts["components"],
            "config_class": facts["config_class"],
            "device": facts["device"],
            "dtype": facts["dtype"],
            "evaluation_mode": facts["evaluation_mode"],
            "forward_count": facts["forward_count"],
            "forward_executed": facts["forward_executed"],
            "frozen": facts["frozen"],
            "load_options": {
                "local_files_only": True,
                "output_loading_info": True,
                "token": False,
                "torch_dtype": "float32",
                "trust_remote_code": False,
                "use_safetensors": True,
            },
            "loading_info": facts["loading_info"],
            "markers": list(PROBE_MARKERS),
            "model_cache_unchanged": True,
            "model_class": facts["model_class"],
            "model_id": _attribute_or_key(snapshot_profile, "model_id"),
            "model_state_sha256": facts["state_sha256"],
            "parameter_count": facts["parameter_count"],
            "parameter_tensor_count": facts["parameter_tensor_count"],
            "persistent_runtime_unchanged": True,
            "processor_object_count": facts["processor_object_count"],
            "revision": _attribute_or_key(snapshot_profile, "revision"),
            "rng": facts["rng"],
            "runtime_id": _attribute_or_key(runtime_profile, "runtime_id"),
            "runtime_manifest_sha256": _attribute_or_key(
                runtime_manifest, "source_sha256"
            ),
            "runtime_profile_sha256": _attribute_or_key(
                runtime_profile, "source_sha256"
            ),
            "schema_id": PROBE_RESPONSE_SCHEMA_ID,
            "snapshot_id": _attribute_or_key(snapshot_profile, "snapshot_id"),
            "snapshot_manifest_sha256": _attribute_or_key(
                snapshot_manifest, "source_sha256"
            ),
            "snapshot_profile_sha256": _attribute_or_key(
                snapshot_profile, "source_sha256"
            ),
            "snapshot_unchanged": True,
            "socket_attempt_count": guard.attempt_count,
            "state_sha256": facts["state_sha256"],
            "status": "success",
            "trainable_parameter_count": facts["trainable_parameter_count"],
            "training": facts["training"],
            "torchvision_import_count": facts["torchvision_import_count"],
            "warnings": [],
        }
        _reject_path_or_secret(response)
        return response
    finally:
        guard.restore()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-profile", required=True, type=Path)
    parser.add_argument("--runtime-profile", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    return parser


def _response_path(workspace: Path, value: Path) -> Path:
    root = _resolved_directory(workspace, "workspace_root")
    candidate = value if value.is_absolute() else root / value
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise CheckpointProbeError(
            "path_validation_failed", "probe response parent is unavailable"
        ) from exc
    destination = resolved_parent / candidate.name
    if not _contains(root, destination):
        _fail("unsafe_path", "probe response escapes the workspace")
    if (
        not candidate.name
        or PurePosixPath(candidate.name).name != candidate.name
        or PureWindowsPath(candidate.name).name != candidate.name
    ):
        _fail("unsafe_path", "probe response must use a safe basename")
    return destination


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        workspace = _resolved_directory(args.workspace_root, "workspace_root")
        destination = _response_path(workspace, args.response)
        response = run_checkpoint_load_probe(
            snapshot_profile_path=args.snapshot_profile,
            runtime_profile_path=args.runtime_profile,
            snapshot_root=args.snapshot_root,
            runtime_root=args.runtime_root,
            repository_root=args.repository_root,
            workspace_root=workspace,
            enforce_repository_sources=True,
        )
        _write_response(destination, response)
    except Exception as exc:
        classification = getattr(exc, "classification", type(exc).__name__)
        print(f"DINOV3_CHECKPOINT_LOAD_PROBE_FAILED:{classification}", file=sys.stderr)
        return 1
    for marker in response["markers"]:
        print(marker)
    print(
        json.dumps(
            response,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
