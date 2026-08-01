"""Offline-safe Phase 1 bulk-runtime and checkpoint preflight contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

RUNTIME_SCHEMA_ID = "vea.p1_bulk_runtime_preflight.v0"
INVENTORY_SCHEMA_ID = "vea.p1_checkpoint_cache_inventory.v0"
DECISION_SCHEMA_ID = "vea.p1_bulk_preflight_decision.v0"
RUNTIME_TEMPLATE_SCHEMA_ID = "vea.p1_bulk_runtime_preflight_request.v0"
CHECKPOINT_PLAN_SCHEMA_ID = "vea.p1_three_encoder_checkpoint_plan.v0"
RUNTIME_LOCK_SCHEMA_ID = "vea.p1_bulk_runtime_lock.v0"
DIRECT_HOST_TEMPLATE_SCHEMA_ID = "vea.p1_direct_host_preflight_request.v1"
DIRECT_HOST_SCHEMA_ID = "vea.p1_direct_host_preflight.v1"
DIRECT_HOST_DECISION_SCHEMA_ID = "vea.p1_direct_host_preflight_decision.v1"

DIRECT_HOST_TARGET = {
    "target_id": "lab_apple_rtx5090",
    "execution_kind": "direct_gpu_host",
    "scheduler_kind": "none",
    "expected_hostname": "apple",
    "expected_gpu_name": "NVIDIA GeForce RTX 5090",
    "minimum_visible_gpu_count": 1,
}

EXPECTED_MODEL_KEYS = (
    "clip_vit_b16",
    "siglip2_base_p16_224",
    "dinov3_vit_b16",
)
EXPECTED_LANES = {
    "primary_lane": ("clip_vit_b16", "siglip2_base_p16_224"),
    "dinov3_isolated_lane": ("dinov3_vit_b16",),
}
PACKAGE_KEYS = (
    "numpy",
    "pillow",
    "safetensors",
    "pyyaml",
    "transformers",
    "tokenizers",
    "huggingface_hub",
)
PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pillow": "Pillow",
    "safetensors": "safetensors",
    "pyyaml": "PyYAML",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "huggingface_hub": "huggingface-hub",
}

GIB = 1024**3
STORAGE_BUDGET_BYTES = 100 * GIB
STORAGE_WARNING_BYTES = 90 * GIB
NOT_INSTALLED = "NOT_INSTALLED"
NOT_AVAILABLE = "NOT_AVAILABLE"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_EXACT_VERSION = re.compile(r"[A-Za-z0-9]+(?:[A-Za-z0-9.+_-]*[A-Za-z0-9])?\Z")
_MUTABLE_VERSION = re.compile(r"(?:latest|main|master|[<>=!~,*^]|\s)", re.I)
_TOKEN_VALUE = re.compile(
    r"(?:bearer\s+\S+|hf_[A-Za-z0-9]{10,}|"
    r"(?:token|secret|password|passwd|credential|api[_-]?key)\s*[:=])",
    re.I,
)
_HOME_OR_ABSOLUTE = re.compile(
    r"(?:^[A-Za-z]:[\\/]|^/|^~[/\\]|(?:^|[/\\])Users[/\\]|"
    r"(?:^|[/\\])home[/\\])"
)
_CREDENTIAL_BASENAMES = frozenset(
    {".netrc", "credentials", "credentials.json", "token", ".token"}
)


class PreflightError(RuntimeError):
    """A strict schema, safety, command, or filesystem validation failure."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def _fail(classification: str, path: str, message: str) -> None:
    raise PreflightError(classification, f"{path}: {message}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("invalid_json", "json", f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail("invalid_json", "json", f"non-finite value {value!r}")


def strict_json_loads(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except PreflightError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PreflightError("invalid_json", f"{label}: invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        _fail("invalid_json", label, "expected an object")
    return value


def canonical_json_bytes(value: object, *, trailing_lf: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreflightError("invalid_json", "value is not finite JSON") from exc
    return raw + (b"\n" if trailing_lf else b"")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PreflightError("filesystem_error", "unable to hash required file") from exc
    return digest.hexdigest()


def _exact_fields(value: object, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("schema_error", path, "expected an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing:
        _fail("schema_error", path, f"missing fields {missing}")
    if unknown:
        _fail("schema_error", path, f"unknown fields {unknown}")
    return value


def _string(value: object, path: str, *, allow_markers: bool = True) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        _fail("schema_error", path, "expected a non-empty string")
    if any(ord(char) < 32 for char in value):
        _fail("schema_error", path, "control characters are forbidden")
    if not allow_markers and value in {NOT_AVAILABLE, NOT_INSTALLED}:
        _fail("schema_error", path, "an observed value is required")
    return value


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail("schema_error", path, f"expected integer >= {minimum}")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail("schema_error", path, "expected boolean")
    return value


def _sha(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256.fullmatch(digest) is None:
        _fail("schema_error", path, "expected lowercase SHA-256")
    return digest


def _revision(value: object, path: str) -> str:
    revision = _string(value, path)
    if _REVISION.fullmatch(revision) is None:
        _fail("schema_error", path, "expected exact lowercase 40-hex revision")
    return revision


def _timestamp(value: object, path: str) -> str:
    timestamp = _string(value, path)
    if _UTC_TIMESTAMP.fullmatch(timestamp) is None:
        _fail("schema_error", path, "expected UTC timestamp YYYY-MM-DDTHH:MM:SSZ")
    return timestamp


def _exact_version(value: object, path: str) -> str:
    version = _string(value, path)
    if (
        version in {NOT_AVAILABLE, NOT_INSTALLED}
        or _MUTABLE_VERSION.search(version)
        or _EXACT_VERSION.fullmatch(version) is None
    ):
        _fail("schema_error", path, "expected one exact resolved version")
    return version


def _safe_relative_path(value: object, path: str) -> str:
    relative = _string(value, path)
    if "\\" in relative or relative.startswith("/") or PureWindowsPath(relative).drive:
        _fail("unsafe_path", path, "expected canonical POSIX relative path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail("unsafe_path", path, "path traversal is forbidden")
    if PurePosixPath(relative).as_posix() != relative:
        _fail("unsafe_path", path, "expected canonical POSIX relative path")
    return relative


def _check_safe_strings(value: object, path: str = "evidence") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_safe_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_safe_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if _TOKEN_VALUE.search(value):
            _fail("secret_leakage", path, "credential-like value is forbidden")
        if _HOME_OR_ABSOLUTE.search(value):
            _fail("path_leakage", path, "absolute or home path is forbidden")


def load_canonical_json(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise PreflightError("filesystem_error", f"{label}: unavailable") from exc
    value = strict_json_loads(raw, label=label)
    if raw != canonical_json_bytes(value):
        _fail("invalid_json", label, "expected canonical UTF-8 JSON with one final LF")
    _check_safe_strings(value, label)
    return value


def _atomic_publish(path: str | Path, raw: bytes) -> Path:
    destination = Path(path)
    if destination.exists():
        _fail("output_collision", "output", "refusing to overwrite existing path")
    if not destination.parent.is_dir():
        _fail("unsafe_path", "output", "parent directory must already exist")
    stage: Path | None = None
    reserved = False
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".staging",
            delete=False,
        )
        stage = Path(handle.name)
        with handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        reserved = True
        os.replace(stage, destination)
        stage = None
        return destination
    except PreflightError:
        raise
    except OSError as exc:
        raise PreflightError("filesystem_error", "unable to publish output") from exc
    finally:
        if stage is not None:
            stage.unlink(missing_ok=True)
        if reserved and (not destination.exists() or destination.stat().st_size == 0):
            destination.unlink(missing_ok=True)


def write_json_output(path: str | Path, value: Mapping[str, Any]) -> Path:
    _check_safe_strings(value)
    return _atomic_publish(path, canonical_json_bytes(value))


def write_text_output(path: str | Path, lines: Sequence[str]) -> Path:
    for index, line in enumerate(lines):
        _check_safe_strings(line, f"summary[{index}]")
    return _atomic_publish(path, ("\n".join(lines) + "\n").encode("utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        junction = getattr(path, "is_junction", None)
        if junction is not None and junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except OSError as exc:
        raise PreflightError("filesystem_error", "unable to inspect filesystem link") from exc


def _within(child: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((str(child), str(parent))) == str(parent)
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str = ""


CommandRunner = Callable[[Sequence[str]], CommandResult]


def subprocess_runner(arguments: Sequence[str]) -> CommandResult:
    """Run one bounded read-only command without a shell."""

    if not arguments or arguments[0] not in {"nvidia-smi", "nvcc", "du"}:
        _fail("unsafe_command", "command", "command is outside the read-only allowlist")
    try:
        result = subprocess.run(
            list(arguments),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=120,
        )
    except FileNotFoundError:
        return CommandResult(127, "", "command not found")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreflightError("command_failure", "read-only command failed") from exc
    return CommandResult(result.returncode, result.stdout, result.stderr)


@dataclass(frozen=True, slots=True)
class SystemProbe:
    os_name: str
    kernel: str
    architecture: str
    glibc: str
    python_version: str
    hostname: str
    packages: Mapping[str, str]
    torch: Mapping[str, Any]
    repository_mirror_exists: bool


@dataclass(frozen=True, slots=True)
class RuntimeDerivedState:
    allocation_observed: bool
    collection_complete: bool
    storage_warning_exceeded: bool
    storage_budget_exceeded: bool


@dataclass(frozen=True, slots=True)
class StorageSpace:
    total_bytes: int
    free_bytes: int


StorageProbe = Callable[[str | Path], StorageSpace]


@dataclass(frozen=True, slots=True)
class DirectHostDerivedState:
    host_evidence_complete: bool
    runtime_status: str
    ready_to_author_apple_runtime_setup: bool
    storage_warning_exceeded: bool
    storage_budget_exceeded: bool


def _installed_packages() -> dict[str, str]:
    result: dict[str, str] = {}
    for key, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            result[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[key] = NOT_INSTALLED
    return result


def _torch_probe() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "import_status": NOT_INSTALLED,
            "version": NOT_INSTALLED,
            "build": NOT_AVAILABLE,
            "cuda_version": NOT_AVAILABLE,
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "cudnn_version": NOT_AVAILABLE,
        }
    except Exception:
        return {
            "import_status": "IMPORT_ERROR",
            "version": NOT_AVAILABLE,
            "build": NOT_AVAILABLE,
            "cuda_version": NOT_AVAILABLE,
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "cudnn_version": NOT_AVAILABLE,
        }
    cuda_available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if cuda_available else 0
    names = [str(torch.cuda.get_device_name(index)) for index in range(count)]
    cudnn = torch.backends.cudnn.version() if cuda_available else None
    return {
        "import_status": "INSTALLED",
        "version": str(torch.__version__),
        "build": str(getattr(torch.version, "git_version", None) or NOT_AVAILABLE),
        "cuda_version": str(torch.version.cuda or NOT_AVAILABLE),
        "cuda_available": cuda_available,
        "device_count": count,
        "device_names": names,
        "cudnn_version": str(cudnn if cudnn is not None else NOT_AVAILABLE),
    }


def system_probe(repository_root: str | Path) -> SystemProbe:
    libc_name, libc_version = platform.libc_ver()
    glibc = f"{libc_name}-{libc_version}" if libc_name and libc_version else NOT_AVAILABLE
    return SystemProbe(
        os_name=platform.system() or NOT_AVAILABLE,
        kernel=platform.release() or NOT_AVAILABLE,
        architecture=platform.machine() or NOT_AVAILABLE,
        glibc=glibc,
        python_version=platform.python_version(),
        hostname=platform.node() or NOT_AVAILABLE,
        packages=_installed_packages(),
        torch=_torch_probe(),
        repository_mirror_exists=Path(repository_root).is_dir(),
    )


def system_storage_probe(root: str | Path) -> StorageSpace:
    try:
        usage = shutil.disk_usage(Path(root))
    except OSError as exc:
        raise PreflightError(
            "filesystem_error", "direct_host.storage: unable to inspect filesystem"
        ) from exc
    return StorageSpace(total_bytes=int(usage.total), free_bytes=int(usage.free))


def load_runtime_template(path: str | Path) -> dict[str, Any]:
    value = load_canonical_json(path, label="runtime_template")
    root = _exact_fields(
        value,
        {
            "schema_id", "storage", "roots", "runtime_lanes",
            "required_packages", "policies",
        },
        "runtime_template",
    )
    if root["schema_id"] != RUNTIME_TEMPLATE_SCHEMA_ID:
        _fail("schema_error", "runtime_template.schema_id", "unexpected schema")
    storage = _exact_fields(root["storage"], {"budget_bytes", "warning_bytes"}, "runtime_template.storage")
    if _integer(storage["budget_bytes"], "runtime_template.storage.budget_bytes") != STORAGE_BUDGET_BYTES:
        _fail("schema_error", "runtime_template.storage.budget_bytes", "expected 100 GiB")
    if _integer(storage["warning_bytes"], "runtime_template.storage.warning_bytes") != STORAGE_WARNING_BYTES:
        _fail("schema_error", "runtime_template.storage.warning_bytes", "expected 90 GiB")
    roots = _exact_fields(root["roots"], {"vea_root_env", "repository_root_env", "classifications"}, "runtime_template.roots")
    for key in ("vea_root_env", "repository_root_env"):
        _string(roots[key], f"runtime_template.roots.{key}")
    classifications = _exact_fields(roots["classifications"], {"vea_root", "repository_mirror"}, "runtime_template.roots.classifications")
    for key in classifications:
        _string(classifications[key], f"runtime_template.roots.classifications.{key}")
    _validate_lanes(root["runtime_lanes"], "runtime_template.runtime_lanes", versions=False)
    if tuple(root["required_packages"]) != PACKAGE_KEYS:
        _fail("schema_error", "runtime_template.required_packages", "unexpected package order")
    policies = _exact_fields(root["policies"], {"read_only", "network_allowed", "package_operations_allowed", "model_load_allowed", "gpu_tensor_allocation_allowed"}, "runtime_template.policies")
    expected = {"read_only": True, "network_allowed": False, "package_operations_allowed": False, "model_load_allowed": False, "gpu_tensor_allocation_allowed": False}
    if dict(policies) != expected:
        _fail("schema_error", "runtime_template.policies", "unsafe policy")
    return dict(root)


def _validate_direct_host_target(value: object, path: str) -> Mapping[str, Any]:
    target = _exact_fields(
        value, set(DIRECT_HOST_TARGET), path
    )
    for key in (
        "target_id", "execution_kind", "scheduler_kind", "expected_hostname",
        "expected_gpu_name",
    ):
        _string(target[key], f"{path}.{key}")
    _integer(target["minimum_visible_gpu_count"], f"{path}.minimum_visible_gpu_count", minimum=1)
    if dict(target) != DIRECT_HOST_TARGET:
        _fail("schema_error", path, "target identity mismatch")
    return target


def load_direct_host_template(path: str | Path) -> dict[str, Any]:
    value = load_canonical_json(path, label="direct_host_template")
    root = _exact_fields(
        value,
        {
            "schema_id", "target", "storage", "roots",
            "observed_packages", "policies",
        },
        "direct_host_template",
    )
    if root["schema_id"] != DIRECT_HOST_TEMPLATE_SCHEMA_ID:
        _fail("schema_error", "direct_host_template.schema_id", "unexpected schema")
    _validate_direct_host_target(root["target"], "direct_host_template.target")
    storage = _exact_fields(
        root["storage"], {"budget_bytes", "warning_bytes"},
        "direct_host_template.storage",
    )
    if (
        _integer(storage["budget_bytes"], "direct_host_template.storage.budget_bytes")
        != STORAGE_BUDGET_BYTES
    ):
        _fail("schema_error", "direct_host_template.storage.budget_bytes", "expected 100 GiB")
    if (
        _integer(storage["warning_bytes"], "direct_host_template.storage.warning_bytes")
        != STORAGE_WARNING_BYTES
    ):
        _fail("schema_error", "direct_host_template.storage.warning_bytes", "expected 90 GiB")
    roots = _exact_fields(
        root["roots"],
        {"vea_root_env", "repository_root_env", "classifications"},
        "direct_host_template.roots",
    )
    if roots["vea_root_env"] != "VEA_ROOT" or roots["repository_root_env"] != "REPO_ROOT":
        _fail("schema_error", "direct_host_template.roots", "unexpected root variables")
    classifications = _exact_fields(
        roots["classifications"], {"vea_root", "repository_root"},
        "direct_host_template.roots.classifications",
    )
    for key, item in classifications.items():
        _string(item, f"direct_host_template.roots.classifications.{key}")
    if tuple(root["observed_packages"]) != PACKAGE_KEYS:
        _fail("schema_error", "direct_host_template.observed_packages", "unexpected package order")
    policies = _exact_fields(
        root["policies"],
        {
            "read_only", "network_allowed", "package_operations_allowed",
            "docker_commands_allowed", "slurm_commands_allowed",
            "model_load_allowed", "gpu_tensor_allocation_allowed",
            "dataset_access_allowed", "cache_write_allowed",
            "credential_discovery_allowed", "repository_write_allowed",
        },
        "direct_host_template.policies",
    )
    expected_policies = {
        "read_only": True,
        "network_allowed": False,
        "package_operations_allowed": False,
        "docker_commands_allowed": False,
        "slurm_commands_allowed": False,
        "model_load_allowed": False,
        "gpu_tensor_allocation_allowed": False,
        "dataset_access_allowed": False,
        "cache_write_allowed": False,
        "credential_discovery_allowed": False,
        "repository_write_allowed": False,
    }
    if dict(policies) != expected_policies:
        _fail("schema_error", "direct_host_template.policies", "unsafe policy")
    return dict(root)


def _validate_lanes(value: object, path: str, *, versions: bool) -> None:
    if not isinstance(value, list) or len(value) != 2:
        _fail("schema_error", path, "expected two runtime lanes")
    observed: dict[str, tuple[str, ...]] = {}
    for index, raw in enumerate(value):
        fields = {"lane_id", "models"} | ({"python_version", "packages"} if versions else set())
        lane = _exact_fields(raw, fields, f"{path}[{index}]")
        lane_id = _string(lane["lane_id"], f"{path}[{index}].lane_id")
        if not isinstance(lane["models"], list):
            _fail("schema_error", f"{path}[{index}].models", "expected array")
        observed[lane_id] = tuple(_string(item, f"{path}[{index}].models") for item in lane["models"])
        if versions:
            _exact_version(lane["python_version"], f"{path}[{index}].python_version")
            packages = lane["packages"]
            if not isinstance(packages, Mapping) or not packages:
                _fail("schema_error", f"{path}[{index}].packages", "expected package map")
            for key, version in packages.items():
                _string(key, f"{path}[{index}].packages.key")
                _exact_version(version, f"{path}[{index}].packages.{key}")
    if observed != EXPECTED_LANES:
        _fail("schema_error", path, "runtime lane model assignment mismatch")


def _parse_nvidia_query(result: CommandResult) -> tuple[list[dict[str, Any]], str]:
    if result.returncode != 0:
        return [], NOT_AVAILABLE
    gpus: list[dict[str, Any]] = []
    drivers: set[str] = set()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            _fail("command_failure", "nvidia-smi", "unexpected query output")
        index, name, uuid, memory_mib, driver = parts
        try:
            memory_bytes = int(memory_mib) * 1024 * 1024
            gpu_index = int(index)
        except ValueError as exc:
            raise PreflightError("command_failure", "nvidia-smi: invalid numeric field") from exc
        gpus.append({
            "index": gpu_index, "name": name, "uuid": uuid,
            "memory_total_bytes": memory_bytes,
        })
        drivers.add(driver)
    if len(drivers) > 1:
        _fail("command_failure", "nvidia-smi", "inconsistent driver versions")
    return gpus, next(iter(drivers), NOT_AVAILABLE)


def _parse_cuda_compatibility(result: CommandResult) -> str:
    if result.returncode != 0:
        return NOT_AVAILABLE
    match = re.search(r"CUDA Version:\s*([0-9.]+)", result.stdout)
    return match.group(1) if match else NOT_AVAILABLE


def _parse_nvcc(result: CommandResult) -> str:
    if result.returncode != 0:
        return NOT_AVAILABLE
    match = re.search(r"release\s+([0-9.]+)", result.stdout)
    return match.group(1) if match else NOT_AVAILABLE


def _parse_du(result: CommandResult) -> int | None:
    if result.returncode != 0:
        return None
    first = result.stdout.strip().split(maxsplit=1)[0] if result.stdout.strip() else ""
    try:
        return int(first)
    except ValueError:
        return None


def _observed(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value not in {
        NOT_AVAILABLE,
        NOT_INSTALLED,
        "IMPORT_ERROR",
    }


def _derive_runtime_state(runtime: Mapping[str, Any]) -> RuntimeDerivedState:
    system = runtime["system"]
    slurm = runtime["slurm"]
    nvidia = runtime["nvidia_smi"]
    torch = runtime["pytorch"]
    storage = runtime["storage"]
    repository = runtime["repository"]

    system_complete = all(
        _observed(system[key])
        for key in (
            "operating_system",
            "kernel",
            "architecture",
            "glibc",
            "python_version",
        )
    )
    slurm_identity_complete = all(
        _observed(slurm[key])
        for key in ("job_id", "partition", "compute_node", "current_hostname")
    )
    a100_visible = any(
        re.search(r"\bA100\b", gpu["name"], re.I) for gpu in nvidia["gpus"]
    )
    allocation_observed = bool(
        slurm_identity_complete
        and slurm["current_process_on_compute_node"]
        and nvidia["available"]
        and nvidia["exit_code"] == 0
        and nvidia["visible_gpu_count"] > 0
        and a100_visible
    )
    torch_complete = bool(
        torch["import_status"] == "INSTALLED"
        and _observed(torch["version"])
        and _observed(torch["build"])
        and _observed(torch["cuda_version"])
        and torch["cuda_available"]
        and torch["device_count"] > 0
        and len(torch["device_names"]) == torch["device_count"]
        and any(re.search(r"\bA100\b", name, re.I) for name in torch["device_names"])
        and _observed(torch["cudnn_version"])
    )
    usage_available = bool(storage["usage_available"])
    warning_exceeded = bool(
        usage_available and storage["used_bytes"] > STORAGE_WARNING_BYTES
    )
    budget_exceeded = bool(
        usage_available and storage["used_bytes"] > STORAGE_BUDGET_BYTES
    )
    collection_complete = bool(
        system_complete
        and allocation_observed
        and _observed(nvidia["driver_version"])
        and _observed(nvidia["cuda_compatibility"])
        and torch_complete
        and usage_available
        and not budget_exceeded
        and repository["mirror_exists"]
    )
    return RuntimeDerivedState(
        allocation_observed=allocation_observed,
        collection_complete=collection_complete,
        storage_warning_exceeded=warning_exceeded,
        storage_budget_exceeded=budget_exceeded,
    )


def collect_runtime_preflight(
    template: Mapping[str, Any],
    *,
    vea_root: str | Path,
    repository_root: str | Path,
    environment: Mapping[str, str],
    runner: CommandRunner,
    probe: SystemProbe,
    timestamp_utc: str,
) -> tuple[dict[str, Any], int]:
    """Collect read-only runtime facts; return evidence and collector exit code."""

    _timestamp(timestamp_utc, "timestamp_utc")
    root = Path(vea_root)
    repo = Path(repository_root)
    if not root.is_absolute() or not repo.is_absolute():
        _fail("unsafe_invocation", "roots", "absolute input roots are required")
    query = runner((
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ))
    general = runner(("nvidia-smi",))
    nvcc_result = runner(("nvcc", "--version"))
    du_result = runner(("du", "-sb", "--", str(root)))
    gpus, driver = _parse_nvidia_query(query)
    cuda_compatibility = _parse_cuda_compatibility(general)
    used_bytes = _parse_du(du_result)

    slurm_job_id = environment.get("SLURM_JOB_ID") or NOT_AVAILABLE
    slurm_node = environment.get("SLURMD_NODENAME") or NOT_AVAILABLE
    partition = environment.get("SLURM_JOB_PARTITION") or NOT_AVAILABLE
    compute_node = (
        slurm_job_id != NOT_AVAILABLE
        and slurm_node != NOT_AVAILABLE
        and probe.hostname != NOT_AVAILABLE
        and slurm_node.casefold() == probe.hostname.casefold()
    )
    torch = dict(probe.torch)
    packages = {key: probe.packages.get(key, NOT_INSTALLED) for key in PACKAGE_KEYS}
    storage_complete = used_bytes is not None
    classifications = template["roots"]["classifications"]
    evidence: dict[str, Any] = {
        "schema_id": RUNTIME_SCHEMA_ID,
        "created_at_utc": timestamp_utc,
        "status": "NO_CURRENT_A100_ALLOCATION",
        "collection_complete": False,
        "system": {
            "operating_system": probe.os_name,
            "kernel": probe.kernel,
            "architecture": probe.architecture,
            "glibc": probe.glibc,
            "python_version": probe.python_version,
        },
        "slurm": {
            "job_id": slurm_job_id,
            "partition": partition,
            "compute_node": slurm_node,
            "current_hostname": probe.hostname,
            "current_process_on_compute_node": compute_node,
        },
        "nvidia_smi": {
            "available": query.returncode != 127,
            "exit_code": query.returncode,
            "visible_gpu_count": len(gpus),
            "gpus": gpus,
            "driver_version": driver,
            "cuda_compatibility": cuda_compatibility,
        },
        "nvcc": {
            "available": nvcc_result.returncode != 127,
            "exit_code": nvcc_result.returncode,
            "version": _parse_nvcc(nvcc_result),
        },
        "pytorch": torch,
        "packages": packages,
        "storage": {
            "root_classification": classifications["vea_root"],
            "usage_available": storage_complete,
            "used_bytes": used_bytes if used_bytes is not None else 0,
            "budget_bytes": STORAGE_BUDGET_BYTES,
            "warning_threshold_bytes": STORAGE_WARNING_BYTES,
            "warning_threshold_exceeded": bool(used_bytes is not None and used_bytes > STORAGE_WARNING_BYTES),
            "budget_exceeded": bool(used_bytes is not None and used_bytes > STORAGE_BUDGET_BYTES),
        },
        "repository": {
            "root_classification": classifications["repository_mirror"],
            "mirror_exists": probe.repository_mirror_exists,
        },
        "runtime_lanes": template["runtime_lanes"],
        "operation_counts": {
            "network": 0, "package": 0, "model_load": 0,
            "gpu_tensor_allocation": 0, "dataset_read": 0,
            "credential_discovery": 0,
        },
    }
    derived = _derive_runtime_state(evidence)
    evidence["status"] = (
        "A100_ALLOCATION_OBSERVED"
        if derived.allocation_observed
        else "NO_CURRENT_A100_ALLOCATION"
    )
    evidence["collection_complete"] = derived.collection_complete
    if derived.collection_complete:
        exit_code = 0
    elif not derived.allocation_observed and (
        slurm_job_id == NOT_AVAILABLE or not compute_node
    ):
        exit_code = 2
    else:
        exit_code = 4
    validate_runtime_evidence(evidence)
    return evidence, exit_code


def validate_runtime_evidence(value: object) -> dict[str, Any]:
    root = _exact_fields(value, {
        "schema_id", "created_at_utc", "status", "collection_complete",
        "system", "slurm", "nvidia_smi", "nvcc", "pytorch", "packages",
        "storage", "repository", "runtime_lanes", "operation_counts",
    }, "runtime")
    if root["schema_id"] != RUNTIME_SCHEMA_ID:
        _fail("schema_error", "runtime.schema_id", "unexpected schema")
    _timestamp(root["created_at_utc"], "runtime.created_at_utc")
    if root["status"] not in {"A100_ALLOCATION_OBSERVED", "NO_CURRENT_A100_ALLOCATION"}:
        _fail("schema_error", "runtime.status", "unexpected status")
    complete = _boolean(root["collection_complete"], "runtime.collection_complete")
    system = _exact_fields(root["system"], {"operating_system", "kernel", "architecture", "glibc", "python_version"}, "runtime.system")
    for key in system:
        _string(system[key], f"runtime.system.{key}")
    slurm = _exact_fields(root["slurm"], {"job_id", "partition", "compute_node", "current_hostname", "current_process_on_compute_node"}, "runtime.slurm")
    for key in ("job_id", "partition", "compute_node", "current_hostname"):
        _string(slurm[key], f"runtime.slurm.{key}")
    process_on_compute_node = _boolean(
        slurm["current_process_on_compute_node"],
        "runtime.slurm.current_process_on_compute_node",
    )
    expected_process_on_compute_node = bool(
        slurm["job_id"] != NOT_AVAILABLE
        and slurm["compute_node"] != NOT_AVAILABLE
        and slurm["current_hostname"] != NOT_AVAILABLE
        and slurm["compute_node"].casefold() == slurm["current_hostname"].casefold()
    )
    if process_on_compute_node != expected_process_on_compute_node:
        _fail(
            "schema_error",
            "runtime.slurm.current_process_on_compute_node",
            "inconsistent Slurm process/node identity",
        )
    nvidia = _exact_fields(root["nvidia_smi"], {"available", "exit_code", "visible_gpu_count", "gpus", "driver_version", "cuda_compatibility"}, "runtime.nvidia_smi")
    _boolean(nvidia["available"], "runtime.nvidia_smi.available")
    _integer(nvidia["exit_code"], "runtime.nvidia_smi.exit_code")
    _integer(nvidia["visible_gpu_count"], "runtime.nvidia_smi.visible_gpu_count")
    if not isinstance(nvidia["gpus"], list):
        _fail("schema_error", "runtime.nvidia_smi.gpus", "expected array")
    for index, gpu_raw in enumerate(nvidia["gpus"]):
        gpu = _exact_fields(gpu_raw, {"index", "name", "uuid", "memory_total_bytes"}, f"runtime.nvidia_smi.gpus[{index}]")
        _integer(gpu["index"], f"runtime.nvidia_smi.gpus[{index}].index")
        _string(gpu["name"], f"runtime.nvidia_smi.gpus[{index}].name")
        _string(gpu["uuid"], f"runtime.nvidia_smi.gpus[{index}].uuid")
        _integer(gpu["memory_total_bytes"], f"runtime.nvidia_smi.gpus[{index}].memory_total_bytes", minimum=1)
    if len(nvidia["gpus"]) != nvidia["visible_gpu_count"]:
        _fail("schema_error", "runtime.nvidia_smi.visible_gpu_count", "GPU count mismatch")
    if nvidia["available"] != (nvidia["exit_code"] != 127):
        _fail(
            "schema_error",
            "runtime.nvidia_smi.available",
            "availability/exit-code mismatch",
        )
    if nvidia["exit_code"] != 0 and nvidia["gpus"]:
        _fail(
            "schema_error",
            "runtime.nvidia_smi.gpus",
            "failed query cannot report visible GPUs",
        )
    gpu_indices = [gpu["index"] for gpu in nvidia["gpus"]]
    if len(gpu_indices) != len(set(gpu_indices)):
        _fail("schema_error", "runtime.nvidia_smi.gpus", "duplicate GPU index")
    _string(nvidia["driver_version"], "runtime.nvidia_smi.driver_version")
    _string(nvidia["cuda_compatibility"], "runtime.nvidia_smi.cuda_compatibility")
    nvcc = _exact_fields(root["nvcc"], {"available", "exit_code", "version"}, "runtime.nvcc")
    _boolean(nvcc["available"], "runtime.nvcc.available")
    _integer(nvcc["exit_code"], "runtime.nvcc.exit_code")
    _string(nvcc["version"], "runtime.nvcc.version")
    if nvcc["available"] != (nvcc["exit_code"] != 127):
        _fail("schema_error", "runtime.nvcc.available", "availability/exit-code mismatch")
    if nvcc["exit_code"] == 0 and not _observed(nvcc["version"]):
        _fail("schema_error", "runtime.nvcc.version", "successful nvcc requires a version")
    torch = _exact_fields(root["pytorch"], {"import_status", "version", "build", "cuda_version", "cuda_available", "device_count", "device_names", "cudnn_version"}, "runtime.pytorch")
    for key in ("import_status", "version", "build", "cuda_version", "cudnn_version"):
        _string(torch[key], f"runtime.pytorch.{key}")
    _boolean(torch["cuda_available"], "runtime.pytorch.cuda_available")
    _integer(torch["device_count"], "runtime.pytorch.device_count")
    if not isinstance(torch["device_names"], list) or any(not isinstance(item, str) or not item for item in torch["device_names"]):
        _fail("schema_error", "runtime.pytorch.device_names", "expected string array")
    if len(torch["device_names"]) != torch["device_count"]:
        _fail("schema_error", "runtime.pytorch.device_count", "device count mismatch")
    if torch["import_status"] not in {"INSTALLED", NOT_INSTALLED, "IMPORT_ERROR"}:
        _fail("schema_error", "runtime.pytorch.import_status", "unexpected import status")
    if torch["import_status"] != "INSTALLED":
        if (
            torch["cuda_available"]
            or torch["device_count"] != 0
            or torch["device_names"]
            or torch["build"] != NOT_AVAILABLE
            or torch["cuda_version"] != NOT_AVAILABLE
            or torch["cudnn_version"] != NOT_AVAILABLE
        ):
            _fail(
                "schema_error",
                "runtime.pytorch",
                "unavailable PyTorch cannot report CUDA/build evidence",
            )
        expected_version = (
            NOT_INSTALLED
            if torch["import_status"] == NOT_INSTALLED
            else NOT_AVAILABLE
        )
        if torch["version"] != expected_version:
            _fail("schema_error", "runtime.pytorch.version", "import/version mismatch")
    else:
        if not _observed(torch["version"]):
            _fail("schema_error", "runtime.pytorch.version", "installed version required")
        if torch["cuda_available"]:
            if (
                not _observed(torch["cuda_version"])
                or torch["device_count"] == 0
                or not _observed(torch["cudnn_version"])
            ):
                _fail(
                    "schema_error",
                    "runtime.pytorch",
                    "enabled CUDA requires version, device, and cuDNN evidence",
                )
        elif (
            torch["device_count"] != 0
            or torch["device_names"]
            or torch["cudnn_version"] != NOT_AVAILABLE
        ):
            _fail(
                "schema_error",
                "runtime.pytorch",
                "CUDA-unavailable PyTorch cannot report devices or cuDNN",
            )
    packages = _exact_fields(root["packages"], set(PACKAGE_KEYS), "runtime.packages")
    for key in PACKAGE_KEYS:
        _string(packages[key], f"runtime.packages.{key}")
    storage = _exact_fields(root["storage"], {"root_classification", "usage_available", "used_bytes", "budget_bytes", "warning_threshold_bytes", "warning_threshold_exceeded", "budget_exceeded"}, "runtime.storage")
    _string(storage["root_classification"], "runtime.storage.root_classification")
    for key in ("usage_available", "warning_threshold_exceeded", "budget_exceeded"):
        _boolean(storage[key], f"runtime.storage.{key}")
    for key in ("used_bytes", "budget_bytes", "warning_threshold_bytes"):
        _integer(storage[key], f"runtime.storage.{key}")
    if storage["budget_bytes"] != STORAGE_BUDGET_BYTES or storage["warning_threshold_bytes"] != STORAGE_WARNING_BYTES:
        _fail("schema_error", "runtime.storage", "storage policy mismatch")
    expected_warning = bool(
        storage["usage_available"] and storage["used_bytes"] > STORAGE_WARNING_BYTES
    )
    expected_budget = bool(
        storage["usage_available"] and storage["used_bytes"] > STORAGE_BUDGET_BYTES
    )
    if not storage["usage_available"] and storage["used_bytes"] != 0:
        _fail("schema_error", "runtime.storage.used_bytes", "unavailable usage must be zero")
    if storage["warning_threshold_exceeded"] != expected_warning:
        _fail(
            "schema_error",
            "runtime.storage.warning_threshold_exceeded",
            "derived warning flag mismatch",
        )
    if storage["budget_exceeded"] != expected_budget:
        _fail(
            "schema_error",
            "runtime.storage.budget_exceeded",
            "derived budget flag mismatch",
        )
    repository = _exact_fields(root["repository"], {"root_classification", "mirror_exists"}, "runtime.repository")
    _string(repository["root_classification"], "runtime.repository.root_classification")
    _boolean(repository["mirror_exists"], "runtime.repository.mirror_exists")
    _validate_lanes(root["runtime_lanes"], "runtime.runtime_lanes", versions=False)
    operations = _exact_fields(root["operation_counts"], {"network", "package", "model_load", "gpu_tensor_allocation", "dataset_read", "credential_discovery"}, "runtime.operation_counts")
    if any(_integer(value, f"runtime.operation_counts.{key}") != 0 for key, value in operations.items()):
        _fail("unsafe_operation", "runtime.operation_counts", "all prohibited operation counts must be zero")
    derived = _derive_runtime_state(root)
    if (root["status"] == "A100_ALLOCATION_OBSERVED") != derived.allocation_observed:
        _fail("schema_error", "runtime.status", "allocation marker is inconsistent")
    if complete != derived.collection_complete:
        _fail(
            "schema_error",
            "runtime.collection_complete",
            "caller value does not match derived runtime completeness",
        )
    _check_safe_strings(root, "runtime")
    return dict(root)


def _derive_direct_host_state(
    evidence: Mapping[str, Any],
) -> DirectHostDerivedState:
    target = evidence["target"]
    system = evidence["system"]
    scheduler = evidence["scheduler"]
    nvidia = evidence["nvidia_smi"]
    torch = evidence["pytorch"]
    packages = evidence["packages"]
    storage = evidence["storage"]
    repository = evidence["repository"]

    system_complete = all(
        _observed(system[key])
        for key in (
            "operating_system", "kernel", "architecture", "glibc",
            "python_version", "hostname",
        )
    )
    scheduler_complete = bool(
        scheduler["kind"] == "none"
        and not scheduler["slurm_environment_present"]
        and not scheduler["current_process_requires_slurm_allocation"]
    )
    expected_gpu_visible = any(
        gpu["name"] == target["expected_gpu_name"] for gpu in nvidia["gpus"]
    )
    gpu_complete = bool(
        nvidia["available"]
        and nvidia["exit_code"] == 0
        and nvidia["visible_gpu_count"] >= target["minimum_visible_gpu_count"]
        and len(nvidia["gpus"]) == nvidia["visible_gpu_count"]
        and expected_gpu_visible
        and _observed(nvidia["driver_version"])
        and _observed(nvidia["cuda_compatibility"])
    )
    warning_exceeded = bool(
        storage["usage_available"]
        and storage["used_bytes"] > STORAGE_WARNING_BYTES
    )
    budget_exceeded = bool(
        storage["usage_available"]
        and storage["used_bytes"] > STORAGE_BUDGET_BYTES
    )
    storage_complete = bool(
        storage["usage_available"]
        and storage["space_available"]
        and storage["total_bytes"] > 0
        and 0 <= storage["free_bytes"] <= storage["total_bytes"]
        and not budget_exceeded
    )
    host_complete = bool(
        target == DIRECT_HOST_TARGET
        and system_complete
        and system["hostname"] == target["expected_hostname"]
        and scheduler_complete
        and gpu_complete
        and storage_complete
        and repository["root_exists"]
    )

    packages_missing = any(value == NOT_INSTALLED for value in packages.values())
    torch_cuda_ready = bool(
        torch["import_status"] == "INSTALLED"
        and _observed(torch["version"])
        and _observed(torch["build"])
        and _observed(torch["cuda_version"])
        and torch["cuda_available"]
        and torch["device_count"] > 0
        and len(torch["device_names"]) == torch["device_count"]
        and any(name == target["expected_gpu_name"] for name in torch["device_names"])
        and _observed(torch["cudnn_version"])
    )
    if torch["import_status"] == NOT_INSTALLED or packages_missing:
        runtime_status = "RUNTIME_ENVIRONMENT_MISSING"
    elif torch["import_status"] == "IMPORT_ERROR":
        runtime_status = "RUNTIME_ENVIRONMENT_ERROR"
    elif torch_cuda_ready and all(_observed(value) for value in packages.values()):
        runtime_status = "RUNTIME_ENVIRONMENT_CUDA_READY"
    else:
        runtime_status = "RUNTIME_ENVIRONMENT_NOT_CUDA_READY"
    return DirectHostDerivedState(
        host_evidence_complete=host_complete,
        runtime_status=runtime_status,
        ready_to_author_apple_runtime_setup=host_complete,
        storage_warning_exceeded=warning_exceeded,
        storage_budget_exceeded=budget_exceeded,
    )


def collect_direct_host_preflight(
    template: Mapping[str, Any],
    *,
    vea_root: str | Path,
    repository_root: str | Path,
    environment: Mapping[str, str],
    runner: CommandRunner,
    probe: SystemProbe,
    storage_space: StorageSpace,
    timestamp_utc: str,
) -> tuple[dict[str, Any], int]:
    """Collect direct-host facts without Slurm, Docker, network, or GPU work."""

    _timestamp(timestamp_utc, "timestamp_utc")
    root = Path(vea_root)
    repository = Path(repository_root)
    if not root.is_absolute() or not repository.is_absolute():
        _fail("unsafe_invocation", "roots", "absolute input roots are required")
    if type(storage_space.total_bytes) is not int or type(storage_space.free_bytes) is not int:
        _fail("schema_error", "direct_host.storage", "storage values must be integers")
    space_available = bool(
        storage_space.total_bytes > 0
        and 0 <= storage_space.free_bytes <= storage_space.total_bytes
    )
    query = runner((
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ))
    general = runner(("nvidia-smi",))
    du_result = runner(("du", "-sb", "--", str(root)))
    gpus, driver = _parse_nvidia_query(query)
    cuda_compatibility = _parse_cuda_compatibility(general)
    used_bytes = _parse_du(du_result)
    classifications = template["roots"]["classifications"]
    packages = {key: probe.packages.get(key, NOT_INSTALLED) for key in PACKAGE_KEYS}
    slurm_environment_present = any(key.startswith("SLURM_") for key in environment)
    evidence: dict[str, Any] = {
        "schema_id": DIRECT_HOST_SCHEMA_ID,
        "created_at_utc": timestamp_utc,
        "target": dict(template["target"]),
        "status": "DIRECT_HOST_EVIDENCE_INCOMPLETE",
        "host_evidence_complete": False,
        "runtime_status": "RUNTIME_ENVIRONMENT_MISSING",
        "ready_to_author_apple_runtime_setup": False,
        "ready_for_p1_extraction": False,
        "system": {
            "operating_system": probe.os_name,
            "kernel": probe.kernel,
            "architecture": probe.architecture,
            "glibc": probe.glibc,
            "python_version": probe.python_version,
            "hostname": probe.hostname,
        },
        "scheduler": {
            "kind": template["target"]["scheduler_kind"],
            "slurm_environment_present": slurm_environment_present,
            "current_process_requires_slurm_allocation": False,
        },
        "nvidia_smi": {
            "available": query.returncode != 127,
            "exit_code": query.returncode,
            "visible_gpu_count": len(gpus),
            "gpus": gpus,
            "driver_version": driver,
            "cuda_compatibility": cuda_compatibility,
        },
        "pytorch": dict(probe.torch),
        "packages": packages,
        "storage": {
            "root_classification": classifications["vea_root"],
            "usage_available": used_bytes is not None,
            "space_available": space_available,
            "used_bytes": used_bytes if used_bytes is not None else 0,
            "total_bytes": storage_space.total_bytes if space_available else 0,
            "free_bytes": storage_space.free_bytes if space_available else 0,
            "budget_bytes": STORAGE_BUDGET_BYTES,
            "warning_threshold_bytes": STORAGE_WARNING_BYTES,
            "warning_threshold_exceeded": bool(
                used_bytes is not None and used_bytes > STORAGE_WARNING_BYTES
            ),
            "budget_exceeded": bool(
                used_bytes is not None and used_bytes > STORAGE_BUDGET_BYTES
            ),
        },
        "repository": {
            "root_classification": classifications["repository_root"],
            "root_exists": probe.repository_mirror_exists,
        },
        "operation_counts": {
            "network": 0,
            "package": 0,
            "docker": 0,
            "slurm": 0,
            "model_load": 0,
            "gpu_tensor_allocation": 0,
            "dataset_read": 0,
            "cache_write": 0,
            "credential_discovery": 0,
            "repository_write": 0,
        },
    }
    derived = _derive_direct_host_state(evidence)
    evidence["status"] = (
        "DIRECT_HOST_EVIDENCE_COMPLETE"
        if derived.host_evidence_complete
        else "DIRECT_HOST_EVIDENCE_INCOMPLETE"
    )
    evidence["host_evidence_complete"] = derived.host_evidence_complete
    evidence["runtime_status"] = derived.runtime_status
    evidence["ready_to_author_apple_runtime_setup"] = (
        derived.ready_to_author_apple_runtime_setup
    )
    validate_direct_host_evidence(evidence, template, repository_root=repository)
    if derived.host_evidence_complete:
        return evidence, 0
    required_observation_failed = bool(
        query.returncode != 0
        or general.returncode != 0
        or not _observed(driver)
        or not _observed(cuda_compatibility)
        or used_bytes is None
        or not space_available
    )
    return evidence, 4 if required_observation_failed else 2


def _validate_direct_torch(value: object) -> Mapping[str, Any]:
    torch = _exact_fields(
        value,
        {
            "import_status", "version", "build", "cuda_version",
            "cuda_available", "device_count", "device_names", "cudnn_version",
        },
        "direct_host.pytorch",
    )
    for key in ("import_status", "version", "build", "cuda_version", "cudnn_version"):
        _string(torch[key], f"direct_host.pytorch.{key}")
    _boolean(torch["cuda_available"], "direct_host.pytorch.cuda_available")
    _integer(torch["device_count"], "direct_host.pytorch.device_count")
    if not isinstance(torch["device_names"], list) or any(
        not isinstance(item, str) or not item for item in torch["device_names"]
    ):
        _fail("schema_error", "direct_host.pytorch.device_names", "expected string array")
    if len(torch["device_names"]) != torch["device_count"]:
        _fail("schema_error", "direct_host.pytorch.device_count", "device count mismatch")
    if torch["import_status"] not in {"INSTALLED", NOT_INSTALLED, "IMPORT_ERROR"}:
        _fail("schema_error", "direct_host.pytorch.import_status", "unexpected import status")
    if torch["import_status"] != "INSTALLED":
        expected_version = NOT_INSTALLED if torch["import_status"] == NOT_INSTALLED else NOT_AVAILABLE
        if (
            torch["version"] != expected_version
            or torch["build"] != NOT_AVAILABLE
            or torch["cuda_version"] != NOT_AVAILABLE
            or torch["cuda_available"]
            or torch["device_count"] != 0
            or torch["device_names"]
            or torch["cudnn_version"] != NOT_AVAILABLE
        ):
            _fail("schema_error", "direct_host.pytorch", "unavailable PyTorch contradiction")
    else:
        if not _observed(torch["version"]):
            _fail("schema_error", "direct_host.pytorch.version", "installed version required")
        if torch["cuda_available"]:
            if (
                not _observed(torch["cuda_version"])
                or torch["device_count"] == 0
                or not _observed(torch["cudnn_version"])
            ):
                _fail("schema_error", "direct_host.pytorch", "CUDA evidence is incomplete")
        elif torch["device_count"] or torch["device_names"] or torch["cudnn_version"] != NOT_AVAILABLE:
            _fail("schema_error", "direct_host.pytorch", "CUDA-unavailable device contradiction")
    return torch


def validate_direct_host_evidence(
    value: object,
    template: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _exact_fields(
        value,
        {
            "schema_id", "created_at_utc", "target", "status",
            "host_evidence_complete", "runtime_status",
            "ready_to_author_apple_runtime_setup", "ready_for_p1_extraction",
            "system", "scheduler", "nvidia_smi", "pytorch", "packages",
            "storage", "repository", "operation_counts",
        },
        "direct_host",
    )
    if root["schema_id"] != DIRECT_HOST_SCHEMA_ID:
        _fail("schema_error", "direct_host.schema_id", "unexpected schema")
    _timestamp(root["created_at_utc"], "direct_host.created_at_utc")
    target = _validate_direct_host_target(root["target"], "direct_host.target")
    if dict(target) != dict(template["target"]) or dict(target) != DIRECT_HOST_TARGET:
        _fail("schema_error", "direct_host.target", "target identity mismatch")
    if root["status"] not in {
        "DIRECT_HOST_EVIDENCE_COMPLETE", "DIRECT_HOST_EVIDENCE_INCOMPLETE",
    }:
        _fail("schema_error", "direct_host.status", "unexpected status")
    complete = _boolean(root["host_evidence_complete"], "direct_host.host_evidence_complete")
    ready = _boolean(
        root["ready_to_author_apple_runtime_setup"],
        "direct_host.ready_to_author_apple_runtime_setup",
    )
    if _boolean(root["ready_for_p1_extraction"], "direct_host.ready_for_p1_extraction"):
        _fail("schema_error", "direct_host.ready_for_p1_extraction", "must remain false")
    system = _exact_fields(
        root["system"],
        {"operating_system", "kernel", "architecture", "glibc", "python_version", "hostname"},
        "direct_host.system",
    )
    for key in system:
        _string(system[key], f"direct_host.system.{key}")
    scheduler = _exact_fields(
        root["scheduler"],
        {"kind", "slurm_environment_present", "current_process_requires_slurm_allocation"},
        "direct_host.scheduler",
    )
    _string(scheduler["kind"], "direct_host.scheduler.kind")
    _boolean(scheduler["slurm_environment_present"], "direct_host.scheduler.slurm_environment_present")
    requires_slurm = _boolean(
        scheduler["current_process_requires_slurm_allocation"],
        "direct_host.scheduler.current_process_requires_slurm_allocation",
    )
    if scheduler["kind"] != "none" or requires_slurm:
        _fail("schema_error", "direct_host.scheduler", "direct target cannot require Slurm")
    nvidia = _exact_fields(
        root["nvidia_smi"],
        {"available", "exit_code", "visible_gpu_count", "gpus", "driver_version", "cuda_compatibility"},
        "direct_host.nvidia_smi",
    )
    available = _boolean(nvidia["available"], "direct_host.nvidia_smi.available")
    exit_code = _integer(nvidia["exit_code"], "direct_host.nvidia_smi.exit_code")
    visible_count = _integer(
        nvidia["visible_gpu_count"], "direct_host.nvidia_smi.visible_gpu_count"
    )
    if available != (exit_code != 127):
        _fail("schema_error", "direct_host.nvidia_smi.available", "availability mismatch")
    if not isinstance(nvidia["gpus"], list):
        _fail("schema_error", "direct_host.nvidia_smi.gpus", "expected array")
    indices: list[int] = []
    for index, raw_gpu in enumerate(nvidia["gpus"]):
        gpu = _exact_fields(
            raw_gpu, {"index", "name", "uuid", "memory_total_bytes"},
            f"direct_host.nvidia_smi.gpus[{index}]",
        )
        indices.append(_integer(gpu["index"], f"direct_host.nvidia_smi.gpus[{index}].index"))
        _string(gpu["name"], f"direct_host.nvidia_smi.gpus[{index}].name")
        _string(gpu["uuid"], f"direct_host.nvidia_smi.gpus[{index}].uuid")
        _integer(gpu["memory_total_bytes"], f"direct_host.nvidia_smi.gpus[{index}].memory_total_bytes", minimum=1)
    if len(nvidia["gpus"]) != visible_count:
        _fail("schema_error", "direct_host.nvidia_smi.visible_gpu_count", "GPU count mismatch")
    if len(indices) != len(set(indices)):
        _fail("schema_error", "direct_host.nvidia_smi.gpus", "duplicate GPU index")
    if exit_code != 0 and nvidia["gpus"]:
        _fail("schema_error", "direct_host.nvidia_smi.gpus", "failed query reported GPUs")
    _string(nvidia["driver_version"], "direct_host.nvidia_smi.driver_version")
    _string(nvidia["cuda_compatibility"], "direct_host.nvidia_smi.cuda_compatibility")
    _validate_direct_torch(root["pytorch"])
    packages = _exact_fields(root["packages"], set(PACKAGE_KEYS), "direct_host.packages")
    for key in PACKAGE_KEYS:
        _string(packages[key], f"direct_host.packages.{key}")
    storage = _exact_fields(
        root["storage"],
        {
            "root_classification", "usage_available", "space_available",
            "used_bytes", "total_bytes", "free_bytes", "budget_bytes",
            "warning_threshold_bytes", "warning_threshold_exceeded", "budget_exceeded",
        },
        "direct_host.storage",
    )
    _string(storage["root_classification"], "direct_host.storage.root_classification")
    for key in ("usage_available", "space_available", "warning_threshold_exceeded", "budget_exceeded"):
        _boolean(storage[key], f"direct_host.storage.{key}")
    for key in ("used_bytes", "total_bytes", "free_bytes", "budget_bytes", "warning_threshold_bytes"):
        _integer(storage[key], f"direct_host.storage.{key}")
    if storage["budget_bytes"] != STORAGE_BUDGET_BYTES or storage["warning_threshold_bytes"] != STORAGE_WARNING_BYTES:
        _fail("schema_error", "direct_host.storage", "storage policy mismatch")
    if not storage["usage_available"] and storage["used_bytes"] != 0:
        _fail("schema_error", "direct_host.storage.used_bytes", "unavailable usage must be zero")
    if not storage["space_available"]:
        if storage["total_bytes"] != 0 or storage["free_bytes"] != 0:
            _fail("schema_error", "direct_host.storage", "unavailable space must be zero")
    elif storage["total_bytes"] < 1 or storage["free_bytes"] > storage["total_bytes"]:
        _fail("schema_error", "direct_host.storage", "invalid total/free space")
    expected_warning = bool(storage["usage_available"] and storage["used_bytes"] > STORAGE_WARNING_BYTES)
    expected_budget = bool(storage["usage_available"] and storage["used_bytes"] > STORAGE_BUDGET_BYTES)
    if storage["warning_threshold_exceeded"] != expected_warning:
        _fail("schema_error", "direct_host.storage.warning_threshold_exceeded", "derived warning flag mismatch")
    if storage["budget_exceeded"] != expected_budget:
        _fail("schema_error", "direct_host.storage.budget_exceeded", "derived budget flag mismatch")
    repository = _exact_fields(
        root["repository"], {"root_classification", "root_exists"},
        "direct_host.repository",
    )
    _string(repository["root_classification"], "direct_host.repository.root_classification")
    repository_exists = _boolean(repository["root_exists"], "direct_host.repository.root_exists")
    if repository_root is not None:
        candidate = Path(repository_root)
        if not candidate.is_absolute():
            _fail("unsafe_invocation", "repository_root", "absolute path required")
        if repository_exists != candidate.is_dir():
            _fail("schema_error", "direct_host.repository.root_exists", "filesystem state mismatch")
    operations = _exact_fields(
        root["operation_counts"],
        {
            "network", "package", "docker", "slurm", "model_load",
            "gpu_tensor_allocation", "dataset_read", "cache_write",
            "credential_discovery", "repository_write",
        },
        "direct_host.operation_counts",
    )
    if any(_integer(item, f"direct_host.operation_counts.{key}") != 0 for key, item in operations.items()):
        _fail("unsafe_operation", "direct_host.operation_counts", "prohibited operation count")
    derived = _derive_direct_host_state(root)
    expected_status = (
        "DIRECT_HOST_EVIDENCE_COMPLETE"
        if derived.host_evidence_complete
        else "DIRECT_HOST_EVIDENCE_INCOMPLETE"
    )
    if root["status"] != expected_status:
        _fail("schema_error", "direct_host.status", "derived host status mismatch")
    if complete != derived.host_evidence_complete:
        _fail("schema_error", "direct_host.host_evidence_complete", "derived completeness mismatch")
    if root["runtime_status"] != derived.runtime_status:
        _fail("schema_error", "direct_host.runtime_status", "derived runtime status mismatch")
    if ready != derived.ready_to_author_apple_runtime_setup:
        _fail("schema_error", "direct_host.ready_to_author_apple_runtime_setup", "derived readiness mismatch")
    _check_safe_strings(root, "direct_host")
    return dict(root)


def decide_direct_host_preflight(
    evidence: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    timestamp_utc: str,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    validated = validate_direct_host_evidence(
        evidence, template, repository_root=repository_root
    )
    _timestamp(timestamp_utc, "timestamp_utc")
    derived = _derive_direct_host_state(validated)
    decision = {
        "schema_id": DIRECT_HOST_DECISION_SCHEMA_ID,
        "created_at_utc": timestamp_utc,
        "target_id": DIRECT_HOST_TARGET["target_id"],
        "host_status": validated["status"],
        "runtime_status": derived.runtime_status,
        "decision": (
            "READY_TO_AUTHOR_APPLE_RUNTIME_SETUP"
            if derived.ready_to_author_apple_runtime_setup
            else "NOT_READY_TO_AUTHOR_APPLE_RUNTIME_SETUP"
        ),
        "ready_to_author_apple_runtime_setup": derived.ready_to_author_apple_runtime_setup,
        "ready_for_p1_extraction": False,
        "source_evidence_sha256": sha256_bytes(canonical_json_bytes(validated)),
        "operation_counts": {
            "network": 0, "package": 0, "docker": 0, "slurm": 0,
            "model_load": 0, "gpu_tensor_allocation": 0,
        },
    }
    validate_direct_host_decision(decision)
    return decision


def validate_direct_host_decision(value: object) -> dict[str, Any]:
    root = _exact_fields(
        value,
        {
            "schema_id", "created_at_utc", "target_id", "host_status",
            "runtime_status", "decision", "ready_to_author_apple_runtime_setup",
            "ready_for_p1_extraction", "source_evidence_sha256", "operation_counts",
        },
        "direct_host_decision",
    )
    if root["schema_id"] != DIRECT_HOST_DECISION_SCHEMA_ID:
        _fail("schema_error", "direct_host_decision.schema_id", "unexpected schema")
    _timestamp(root["created_at_utc"], "direct_host_decision.created_at_utc")
    if root["target_id"] != DIRECT_HOST_TARGET["target_id"]:
        _fail("schema_error", "direct_host_decision.target_id", "target mismatch")
    if root["host_status"] not in {"DIRECT_HOST_EVIDENCE_COMPLETE", "DIRECT_HOST_EVIDENCE_INCOMPLETE"}:
        _fail("schema_error", "direct_host_decision.host_status", "unexpected status")
    if root["runtime_status"] not in {
        "RUNTIME_ENVIRONMENT_MISSING", "RUNTIME_ENVIRONMENT_ERROR",
        "RUNTIME_ENVIRONMENT_NOT_CUDA_READY", "RUNTIME_ENVIRONMENT_CUDA_READY",
    }:
        _fail("schema_error", "direct_host_decision.runtime_status", "unexpected status")
    ready = _boolean(
        root["ready_to_author_apple_runtime_setup"],
        "direct_host_decision.ready_to_author_apple_runtime_setup",
    )
    expected_ready = root["host_status"] == "DIRECT_HOST_EVIDENCE_COMPLETE"
    if ready != expected_ready:
        _fail("schema_error", "direct_host_decision.ready_to_author_apple_runtime_setup", "host mismatch")
    expected_decision = (
        "READY_TO_AUTHOR_APPLE_RUNTIME_SETUP"
        if ready else "NOT_READY_TO_AUTHOR_APPLE_RUNTIME_SETUP"
    )
    if root["decision"] != expected_decision:
        _fail("schema_error", "direct_host_decision.decision", "decision mismatch")
    if _boolean(root["ready_for_p1_extraction"], "direct_host_decision.ready_for_p1_extraction"):
        _fail("schema_error", "direct_host_decision.ready_for_p1_extraction", "must remain false")
    _sha(root["source_evidence_sha256"], "direct_host_decision.source_evidence_sha256")
    operations = _exact_fields(
        root["operation_counts"],
        {"network", "package", "docker", "slurm", "model_load", "gpu_tensor_allocation"},
        "direct_host_decision.operation_counts",
    )
    if any(_integer(item, f"direct_host_decision.operation_counts.{key}") != 0 for key, item in operations.items()):
        _fail("unsafe_operation", "direct_host_decision.operation_counts", "prohibited operation count")
    _check_safe_strings(root, "direct_host_decision")
    return dict(root)


def _load_model_registry_lazily(registry_path: str | Path) -> dict[str, Any]:
    registry = Path(registry_path)
    if not registry.is_file():
        _fail("filesystem_error", "checkpoint_plan.registry", "registry is unavailable")
    try:
        from vision_encoder_alignment.config.loader import load_model_registry
    except ImportError as exc:
        raise PreflightError(
            "dependency_unavailable",
            "checkpoint_plan.registry: PyYAML/config loader is unavailable",
        ) from exc
    try:
        return load_model_registry(registry)
    except ImportError as exc:
        raise PreflightError(
            "dependency_unavailable",
            "checkpoint_plan.registry: PyYAML dependency is unavailable",
        ) from exc
    except Exception as exc:
        if type(exc).__name__ == "ConfigValidationError":
            raise PreflightError(
                "schema_error", "checkpoint_plan.registry: invalid model registry"
            ) from exc
        raise PreflightError(
            "filesystem_error", "checkpoint_plan.registry: unable to load registry"
        ) from exc


def load_checkpoint_plan(path: str | Path, registry_path: str | Path) -> dict[str, Any]:
    plan = load_canonical_json(path, label="checkpoint_plan")
    root = _exact_fields(plan, {"schema_id", "models", "runtime_lanes", "cache_root_env", "cache_root_classification", "policies"}, "checkpoint_plan")
    if root["schema_id"] != CHECKPOINT_PLAN_SCHEMA_ID:
        _fail("schema_error", "checkpoint_plan.schema_id", "unexpected schema")
    registry = _load_model_registry_lazily(registry_path)
    if not isinstance(root["models"], list) or len(root["models"]) != 3:
        _fail("schema_error", "checkpoint_plan.models", "expected three models")
    for index, (raw, expected_key) in enumerate(zip(root["models"], EXPECTED_MODEL_KEYS, strict=True)):
        model = _exact_fields(raw, {"registry_key", "model_id", "revision", "lane_id"}, f"checkpoint_plan.models[{index}]")
        if model["registry_key"] != expected_key:
            _fail("schema_error", f"checkpoint_plan.models[{index}].registry_key", "model order/key mismatch")
        expected = registry["models"][expected_key]
        if model["model_id"] != expected["model_id"]:
            _fail("schema_error", f"checkpoint_plan.models[{index}].model_id", "registry mismatch")
        if _revision(model["revision"], f"checkpoint_plan.models[{index}].revision") != expected["revision"]:
            _fail("schema_error", f"checkpoint_plan.models[{index}].revision", "registry mismatch")
        lane_id = _string(model["lane_id"], f"checkpoint_plan.models[{index}].lane_id")
        if expected_key not in EXPECTED_LANES.get(lane_id, ()):
            _fail("schema_error", f"checkpoint_plan.models[{index}].lane_id", "lane mismatch")
    _validate_lanes(root["runtime_lanes"], "checkpoint_plan.runtime_lanes", versions=False)
    _string(root["cache_root_env"], "checkpoint_plan.cache_root_env")
    _string(root["cache_root_classification"], "checkpoint_plan.cache_root_classification")
    policies = _exact_fields(root["policies"], {"local_files_only", "trust_remote_code", "no_import_time_download", "weights_outside_repository", "credentials_not_serialized"}, "checkpoint_plan.policies")
    expected_policies = {"local_files_only": True, "trust_remote_code": False, "no_import_time_download": True, "weights_outside_repository": True, "credentials_not_serialized": True}
    if dict(policies) != expected_policies:
        _fail("schema_error", "checkpoint_plan.policies", "unsafe checkpoint policy")
    return dict(root)


def _snapshot_path(cache_root: Path, model_id: str, revision: str) -> Path:
    repository_dir = "models--" + model_id.replace("/", "--")
    return cache_root / repository_dir / "snapshots" / revision


PathLinkProbe = Callable[[Path], bool]
PathResolver = Callable[[Path], Path]


def _strict_path_resolver(path: Path) -> Path:
    return path.resolve(strict=True)


def _directory_components(path: Path) -> tuple[Path, ...]:
    if not path.is_absolute():
        _fail("unsafe_invocation", "path", "absolute path is required")
    current = Path(path.anchor)
    components: list[Path] = [current]
    for part in path.parts[1:]:
        current = current / part
        components.append(current)
    return tuple(components)


def _non_link_directory(
    path: Path,
    *,
    label: str,
    link_probe: PathLinkProbe,
    resolver: PathResolver,
) -> Path | None:
    """Return a strict, non-link directory or ``None`` when it is absent.

    Every existing component is inspected before strict resolution.  This keeps
    cache and snapshot roots distinct from ordinary Hugging Face leaf links.
    """

    missing = False
    for component in _directory_components(path):
        try:
            linked = link_probe(component)
        except PreflightError:
            if not component.exists():
                missing = True
                continue
            raise
        if linked:
            message = (
                "linked cache root is forbidden"
                if label == "cache_root"
                else "linked directory is forbidden"
            )
            _fail("unsafe_path", label, message)
        if not component.exists():
            missing = True
            continue
        try:
            mode = component.stat().st_mode
        except OSError as exc:
            raise PreflightError("filesystem_error", "checkpoint directory unavailable") from exc
        if not stat.S_ISDIR(mode):
            _fail("unsafe_path", label, "expected directory")
    if missing:
        return None
    try:
        resolved = resolver(path)
    except OSError as exc:
        raise PreflightError("filesystem_error", "checkpoint directory unavailable") from exc
    if resolved != path:
        _fail("unsafe_path", label, "directory resolution is noncanonical")
    return resolved


def _inventory_files(
    snapshot: Path,
    cache_root: Path,
    *,
    link_probe: PathLinkProbe,
    resolver: PathResolver,
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(snapshot, followlinks=False):
        directory_path = Path(directory)
        for name in list(dirnames):
            child = directory_path / name
            if name.casefold() in _CREDENTIAL_BASENAMES:
                _fail("credential_discovery", "checkpoint_cache", "credential-like directory name is forbidden")
            if _is_link_or_reparse(child):
                _fail("unsafe_path", "checkpoint_cache", "linked directories are forbidden")
        for name in filenames:
            logical = directory_path / name
            if name.casefold() in _CREDENTIAL_BASENAMES:
                _fail("credential_discovery", "checkpoint_cache", "credential-like file name is forbidden")
            try:
                # A standard Hugging Face snapshot uses a leaf link into its
                # cache.  The link itself is allowed only after strict target
                # resolution, cache containment, and regular-file validation.
                resolved = resolver(logical)
                if not _within(resolved, cache_root):
                    _fail("root_escape", "checkpoint_cache", "link escapes cache root")
                item_stat = resolved.stat()
            except OSError as exc:
                raise PreflightError("filesystem_error", "checkpoint file unavailable") from exc
            if not stat.S_ISREG(item_stat.st_mode):
                _fail("unsafe_path", "checkpoint_cache", "expected regular file")
            relative = logical.relative_to(snapshot).as_posix()
            _safe_relative_path(relative, "checkpoint_cache.relative_path")
            files.append({"relative_path": relative, "bytes": item_stat.st_size, "sha256": sha256_file(resolved)})
    files.sort(key=lambda item: item["relative_path"])
    return files


def checkpoint_snapshot_files(
    cache_root: str | Path,
    *,
    model_id: str,
    revision: str,
    link_probe: PathLinkProbe | None = None,
    strict_resolver: PathResolver | None = None,
) -> list[dict[str, Any]]:
    """Inventory one bounded Hugging Face snapshot without following directories.

    Cache and snapshot directory components must be ordinary, canonical
    directories.  Snapshot leaf files may be Hugging Face links only when their
    strict target stays under that same cache root and is a regular file.
    Optional probes make the safety cases deterministic on platforms that do
    not permit test symlink creation.
    """

    candidate = Path(cache_root)
    probe = _is_link_or_reparse if link_probe is None else link_probe
    resolver = _strict_path_resolver if strict_resolver is None else strict_resolver
    resolved_cache = _non_link_directory(
        candidate, label="cache_root", link_probe=probe, resolver=resolver,
    )
    if resolved_cache is None:
        return []
    snapshot = _snapshot_path(resolved_cache, model_id, revision)
    resolved_snapshot = _non_link_directory(
        snapshot, label="checkpoint_cache", link_probe=probe, resolver=resolver,
    )
    if resolved_snapshot is None:
        return []
    if not _within(resolved_snapshot, resolved_cache):
        _fail("root_escape", "checkpoint_cache", "snapshot escapes cache root")
    return _inventory_files(
        snapshot, resolved_cache, link_probe=probe, resolver=resolver,
    )


def inventory_checkpoint_cache(
    plan: Mapping[str, Any],
    *,
    cache_root: str | Path,
    repository_root: str | Path,
    timestamp_utc: str,
) -> dict[str, Any]:
    _timestamp(timestamp_utc, "timestamp_utc")
    candidate = Path(cache_root)
    repository = Path(repository_root)
    if not candidate.is_absolute() or not repository.is_absolute():
        _fail("unsafe_invocation", "roots", "absolute input roots are required")
    resolved_cache = _non_link_directory(
        candidate,
        label="cache_root",
        link_probe=_is_link_or_reparse,
        resolver=_strict_path_resolver,
    )
    if resolved_cache is None:
        resolved_cache = candidate
    resolved_repository = repository.resolve(strict=False)
    if _within(resolved_cache, resolved_repository) or _within(resolved_repository, resolved_cache):
        _fail("unsafe_path", "cache_root", "checkpoint cache must be outside repository")
    models: list[dict[str, Any]] = []
    present = 0
    for planned in plan["models"]:
        files = checkpoint_snapshot_files(
            resolved_cache,
            model_id=planned["model_id"],
            revision=planned["revision"],
        )
        status = "PRESENT_AND_INVENTORIED" if files else "CACHE_MISSING"
        present += int(bool(files))
        aggregate = sha256_bytes(canonical_json_bytes(files, trailing_lf=False))
        models.append({
            "registry_key": planned["registry_key"],
            "model_id": planned["model_id"],
            "revision": planned["revision"],
            "status": status,
            "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files),
            "files": files,
            "aggregate_sha256": aggregate,
        })
    overall = "COMPLETE" if present == 3 else "CACHE_MISSING" if present == 0 else "PARTIAL"
    inventory = {
        "schema_id": INVENTORY_SCHEMA_ID,
        "created_at_utc": timestamp_utc,
        "status": overall,
        "cache_root_classification": plan["cache_root_classification"],
        "models": models,
        "present_model_count": present,
        "missing_model_count": 3 - present,
        "operation_counts": {"network": 0, "package": 0, "model_load": 0, "cache_write": 0, "credential_read": 0},
    }
    validate_checkpoint_inventory(inventory, plan)
    return inventory


def validate_checkpoint_inventory(value: object, plan: Mapping[str, Any]) -> dict[str, Any]:
    root = _exact_fields(value, {"schema_id", "created_at_utc", "status", "cache_root_classification", "models", "present_model_count", "missing_model_count", "operation_counts"}, "inventory")
    if root["schema_id"] != INVENTORY_SCHEMA_ID:
        _fail("schema_error", "inventory.schema_id", "unexpected schema")
    _timestamp(root["created_at_utc"], "inventory.created_at_utc")
    if root["status"] not in {"COMPLETE", "PARTIAL", "CACHE_MISSING"}:
        _fail("schema_error", "inventory.status", "unexpected status")
    if root["cache_root_classification"] != plan["cache_root_classification"]:
        _fail("schema_error", "inventory.cache_root_classification", "plan mismatch")
    if not isinstance(root["models"], list) or len(root["models"]) != 3:
        _fail("schema_error", "inventory.models", "expected three models")
    present = 0
    for index, (raw, planned) in enumerate(zip(root["models"], plan["models"], strict=True)):
        model = _exact_fields(raw, {"registry_key", "model_id", "revision", "status", "file_count", "total_bytes", "files", "aggregate_sha256"}, f"inventory.models[{index}]")
        for key in ("registry_key", "model_id", "revision"):
            if model[key] != planned[key]:
                _fail("schema_error", f"inventory.models[{index}].{key}", "plan mismatch")
        _revision(model["revision"], f"inventory.models[{index}].revision")
        if model["status"] not in {"CACHE_MISSING", "PRESENT_AND_INVENTORIED"}:
            _fail("schema_error", f"inventory.models[{index}].status", "unexpected status")
        if not isinstance(model["files"], list):
            _fail("schema_error", f"inventory.models[{index}].files", "expected array")
        previous = ""
        total = 0
        for file_index, raw_file in enumerate(model["files"]):
            item = _exact_fields(raw_file, {"relative_path", "bytes", "sha256"}, f"inventory.models[{index}].files[{file_index}]")
            relative = _safe_relative_path(item["relative_path"], f"inventory.models[{index}].files[{file_index}].relative_path")
            if relative <= previous:
                _fail("schema_error", f"inventory.models[{index}].files", "noncanonical order")
            previous = relative
            total += _integer(item["bytes"], f"inventory.models[{index}].files[{file_index}].bytes")
            _sha(item["sha256"], f"inventory.models[{index}].files[{file_index}].sha256")
        if model["file_count"] != len(model["files"]) or model["total_bytes"] != total:
            _fail("schema_error", f"inventory.models[{index}]", "file summary mismatch")
        aggregate = sha256_bytes(canonical_json_bytes(model["files"], trailing_lf=False))
        if _sha(model["aggregate_sha256"], f"inventory.models[{index}].aggregate_sha256") != aggregate:
            _fail("hash_mismatch", f"inventory.models[{index}].aggregate_sha256", "inventory aggregate mismatch")
        if model["status"] == "PRESENT_AND_INVENTORIED":
            if not model["files"]:
                _fail(
                    "schema_error",
                    f"inventory.models[{index}].status",
                    "present status requires a non-empty file inventory",
                )
            present += 1
        elif model["files"] or model["file_count"] != 0 or model["total_bytes"] != 0:
            _fail(
                "schema_error",
                f"inventory.models[{index}].status",
                "missing status requires an empty zero-byte inventory",
            )
    present_count = _integer(
        root["present_model_count"], "inventory.present_model_count"
    )
    missing_count = _integer(
        root["missing_model_count"], "inventory.missing_model_count"
    )
    if present_count != present or missing_count != 3 - present:
        _fail("schema_error", "inventory", "model count mismatch")
    expected_status = "COMPLETE" if present == 3 else "CACHE_MISSING" if present == 0 else "PARTIAL"
    if root["status"] != expected_status:
        _fail("schema_error", "inventory.status", "summary mismatch")
    operations = _exact_fields(root["operation_counts"], {"network", "package", "model_load", "cache_write", "credential_read"}, "inventory.operation_counts")
    if any(_integer(value, f"inventory.operation_counts.{key}") != 0 for key, value in operations.items()):
        _fail("unsafe_operation", "inventory.operation_counts", "all prohibited operation counts must be zero")
    _check_safe_strings(root, "inventory")
    return dict(root)


def verify_inventory_against_cache(
    expected: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    cache_root: str | Path,
    repository_root: str | Path,
) -> None:
    rebuilt = inventory_checkpoint_cache(
        plan,
        cache_root=cache_root,
        repository_root=repository_root,
        timestamp_utc=expected["created_at_utc"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(expected):
        _fail("hash_mismatch", "checkpoint_cache", "cache differs from expected inventory")


def validate_final_runtime_lock(value: object) -> dict[str, Any]:
    root = _exact_fields(value, {"schema_id", "runtime_lanes"}, "runtime_lock")
    if root["schema_id"] != RUNTIME_LOCK_SCHEMA_ID:
        _fail("schema_error", "runtime_lock.schema_id", "unexpected schema")
    _validate_lanes(root["runtime_lanes"], "runtime_lock.runtime_lanes", versions=True)
    _check_safe_strings(root, "runtime_lock")
    return dict(root)


def _runtime_complete(runtime: Mapping[str, Any]) -> bool:
    return _derive_runtime_state(runtime).collection_complete


def decide_preflight(
    runtime: Mapping[str, Any],
    inventory: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    timestamp_utc: str,
) -> dict[str, Any]:
    validated_runtime = validate_runtime_evidence(runtime)
    validated_inventory = validate_checkpoint_inventory(inventory, plan)
    _timestamp(timestamp_utc, "timestamp_utc")
    runtime_status = (
        "RUNTIME_EVIDENCE_COMPLETE"
        if _runtime_complete(validated_runtime)
        else "RUNTIME_EVIDENCE_INCOMPLETE"
    )
    checkpoint_status = {
        "COMPLETE": "CHECKPOINTS_PRESENT_AND_INVENTORIED",
        "PARTIAL": "CHECKPOINTS_PARTIAL",
        "CACHE_MISSING": "CHECKPOINTS_MISSING",
    }[validated_inventory["status"]]
    ready = runtime_status == "RUNTIME_EVIDENCE_COMPLETE"
    decision = {
        "schema_id": DECISION_SCHEMA_ID,
        "created_at_utc": timestamp_utc,
        "runtime_status": runtime_status,
        "checkpoint_status": checkpoint_status,
        "decision": "READY_TO_AUTHOR_P1_T002B" if ready else "NOT_READY_TO_AUTHOR_P1_T002B",
        "ready_to_author_p1_t002b": ready,
        "ready_for_p1_extraction": False,
        "model_plan_valid": True,
        "runtime_lanes": plan["runtime_lanes"],
        "operation_counts": {"network": 0, "package": 0, "model_load": 0, "gpu_tensor_allocation": 0},
    }
    validate_preflight_decision(decision)
    return decision


def validate_preflight_decision(value: object) -> dict[str, Any]:
    root = _exact_fields(value, {"schema_id", "created_at_utc", "runtime_status", "checkpoint_status", "decision", "ready_to_author_p1_t002b", "ready_for_p1_extraction", "model_plan_valid", "runtime_lanes", "operation_counts"}, "decision")
    if root["schema_id"] != DECISION_SCHEMA_ID:
        _fail("schema_error", "decision.schema_id", "unexpected schema")
    _timestamp(root["created_at_utc"], "decision.created_at_utc")
    if root["runtime_status"] not in {"RUNTIME_EVIDENCE_COMPLETE", "RUNTIME_EVIDENCE_INCOMPLETE"}:
        _fail("schema_error", "decision.runtime_status", "unexpected status")
    if root["checkpoint_status"] not in {"CHECKPOINTS_PRESENT_AND_INVENTORIED", "CHECKPOINTS_PARTIAL", "CHECKPOINTS_MISSING"}:
        _fail("schema_error", "decision.checkpoint_status", "unexpected status")
    ready = _boolean(root["ready_to_author_p1_t002b"], "decision.ready_to_author_p1_t002b")
    if root["decision"] != ("READY_TO_AUTHOR_P1_T002B" if ready else "NOT_READY_TO_AUTHOR_P1_T002B"):
        _fail("schema_error", "decision.decision", "decision marker mismatch")
    if ready != (root["runtime_status"] == "RUNTIME_EVIDENCE_COMPLETE"):
        _fail("schema_error", "decision.ready_to_author_p1_t002b", "runtime readiness mismatch")
    if _boolean(root["ready_for_p1_extraction"], "decision.ready_for_p1_extraction"):
        _fail("schema_error", "decision.ready_for_p1_extraction", "must remain false")
    if not _boolean(root["model_plan_valid"], "decision.model_plan_valid"):
        _fail("schema_error", "decision.model_plan_valid", "must be true")
    _validate_lanes(root["runtime_lanes"], "decision.runtime_lanes", versions=False)
    operations = _exact_fields(root["operation_counts"], {"network", "package", "model_load", "gpu_tensor_allocation"}, "decision.operation_counts")
    if any(_integer(value, f"decision.operation_counts.{key}") != 0 for key, value in operations.items()):
        _fail("unsafe_operation", "decision.operation_counts", "prohibited operation count")
    _check_safe_strings(root, "decision")
    return dict(root)


def runtime_summary(evidence: Mapping[str, Any]) -> list[str]:
    return [
        f"schema_id={evidence['schema_id']}",
        f"status={evidence['status']}",
        f"collection_complete={str(evidence['collection_complete']).lower()}",
        f"slurm_job_id={evidence['slurm']['job_id']}",
        f"compute_node={evidence['slurm']['compute_node']}",
        f"visible_gpu_count={evidence['nvidia_smi']['visible_gpu_count']}",
        f"pytorch_cuda_available={str(evidence['pytorch']['cuda_available']).lower()}",
        f"storage_used_bytes={evidence['storage']['used_bytes']}",
        f"storage_warning_exceeded={str(evidence['storage']['warning_threshold_exceeded']).lower()}",
        "READY_FOR_P1_EXTRACTION=false",
    ]


def direct_host_summary(evidence: Mapping[str, Any]) -> list[str]:
    return [
        f"schema_id={evidence['schema_id']}",
        f"target_id={evidence['target']['target_id']}",
        f"status={evidence['status']}",
        f"runtime_status={evidence['runtime_status']}",
        f"hostname={evidence['system']['hostname']}",
        f"visible_gpu_count={evidence['nvidia_smi']['visible_gpu_count']}",
        f"storage_used_bytes={evidence['storage']['used_bytes']}",
        f"storage_free_bytes={evidence['storage']['free_bytes']}",
        "READY_TO_AUTHOR_APPLE_RUNTIME_SETUP="
        f"{str(evidence['ready_to_author_apple_runtime_setup']).lower()}",
        "READY_FOR_P1_EXTRACTION=false",
    ]


def direct_host_decision_summary(evidence: Mapping[str, Any]) -> list[str]:
    return [
        f"schema_id={evidence['schema_id']}",
        f"target_id={evidence['target_id']}",
        f"host_status={evidence['host_status']}",
        f"runtime_status={evidence['runtime_status']}",
        f"decision={evidence['decision']}",
        "READY_TO_AUTHOR_APPLE_RUNTIME_SETUP="
        f"{str(evidence['ready_to_author_apple_runtime_setup']).lower()}",
        "READY_FOR_P1_EXTRACTION=false",
    ]


def inventory_summary(evidence: Mapping[str, Any]) -> list[str]:
    lines = [
        f"schema_id={evidence['schema_id']}",
        f"status={evidence['status']}",
        f"present_model_count={evidence['present_model_count']}",
        f"missing_model_count={evidence['missing_model_count']}",
    ]
    lines.extend(f"{model['registry_key']}={model['status']}" for model in evidence["models"])
    return lines


def decision_summary(evidence: Mapping[str, Any]) -> list[str]:
    return [
        f"schema_id={evidence['schema_id']}",
        f"runtime_status={evidence['runtime_status']}",
        f"checkpoint_status={evidence['checkpoint_status']}",
        f"decision={evidence['decision']}",
        f"READY_FOR_P1_T002B_PREFLIGHT_EXECUTION={str(evidence['ready_to_author_p1_t002b']).lower()}",
        "READY_FOR_P1_EXTRACTION=false",
    ]


__all__ = [
    "CHECKPOINT_PLAN_SCHEMA_ID", "CommandResult", "DECISION_SCHEMA_ID",
    "DIRECT_HOST_DECISION_SCHEMA_ID", "DIRECT_HOST_SCHEMA_ID",
    "DIRECT_HOST_TARGET", "DIRECT_HOST_TEMPLATE_SCHEMA_ID",
    "DirectHostDerivedState",
    "EXPECTED_LANES", "EXPECTED_MODEL_KEYS", "GIB", "INVENTORY_SCHEMA_ID",
    "NOT_AVAILABLE", "NOT_INSTALLED", "PACKAGE_KEYS", "PreflightError",
    "RUNTIME_LOCK_SCHEMA_ID", "RUNTIME_SCHEMA_ID", "RUNTIME_TEMPLATE_SCHEMA_ID",
    "RuntimeDerivedState", "STORAGE_BUDGET_BYTES", "STORAGE_WARNING_BYTES",
    "StorageProbe", "StorageSpace", "SystemProbe",
    "canonical_json_bytes", "collect_direct_host_preflight",
    "checkpoint_snapshot_files",
    "collect_runtime_preflight", "decide_direct_host_preflight", "decide_preflight",
    "decision_summary", "direct_host_decision_summary", "direct_host_summary",
    "inventory_checkpoint_cache", "inventory_summary",
    "load_canonical_json", "load_checkpoint_plan", "load_direct_host_template",
    "load_runtime_template",
    "runtime_summary", "sha256_bytes", "sha256_file", "strict_json_loads",
    "subprocess_runner", "system_probe", "system_storage_probe", "utc_now",
    "validate_checkpoint_inventory", "validate_direct_host_decision",
    "validate_direct_host_evidence", "validate_final_runtime_lock",
    "validate_preflight_decision", "validate_runtime_evidence", "verify_inventory_against_cache",
    "write_json_output", "write_text_output",
]
