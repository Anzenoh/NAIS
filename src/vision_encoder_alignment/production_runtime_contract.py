"""Strict, import-light production runtime contract for explicit Apple lanes.

The module uses only the standard library at import time.  Hardware and
package observations are injected, which keeps repository tests offline and
prevents import-time Torch, CUDA, model, or network work.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, NoReturn


PROFILE_SCHEMA_ID = "vea.p1.production_runtime_profile.v1"
APPLE_TARGET_ID = "lab_apple_rtx5090"
APPLE_HOSTNAME = "apple"
APPLE_GPU_NAME = "NVIDIA GeForce RTX 5090"
APPLE_PYTHON_VERSION = "3.12.0"
APPLE_GLIBC_VERSION = "2.35"
APPLE_DEVICE = "cuda:0"
APPLE_DTYPE = "float32"

MAIN_LANE_ID = "p1_apple_main"
MAIN_SELECTOR = "apple_main_cuda_fp32_v0"
MAIN_RUNTIME_ROOT = "/home/chltjsdh/vea/runtime/p1_apple/main"

DINOV3_LANE_ID = "p1_apple_dinov3_isolated"
DINOV3_SELECTOR = "dinov3_apple_cuda_fp32_v1"
DINOV3_RUNTIME_ROOT = "/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)*(?:\+[A-Za-z0-9._-]+)?$")
_MAX_PROFILE_BYTES = 512 * 1024
_LOCK_LINE = re.compile(
    r"^([a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)==([^\s]+) --hash=sha256:([0-9a-f]{64})$"
)


class ProductionRuntimeContractError(RuntimeError):
    """Fail-closed runtime-contract error with a stable classification."""

    def __init__(self, message: str, classification: str = "RUNTIME_CONTRACT_INVALID") -> None:
        super().__init__(message)
        self.classification = classification


def _fail(message: str, classification: str = "RUNTIME_CONTRACT_INVALID") -> NoReturn:
    raise ProductionRuntimeContractError(message, classification)


def _reject_nonfinite(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant is forbidden: {value}", "NONCANONICAL_JSON")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is forbidden: {key}", "NONCANONICAL_JSON")
        result[key] = value
    return result


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionRuntimeContractError(
            "runtime profile is not finite JSON", "NONCANONICAL_JSON"
        ) from exc


def _strict_json(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > _MAX_PROFILE_BYTES or payload.startswith(b"\xef\xbb\xbf"):
        _fail("runtime profile has invalid size or UTF-8 BOM", "NONCANONICAL_JSON")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionRuntimeContractError(
            "runtime profile is not strict UTF-8 JSON", "NONCANONICAL_JSON"
        ) from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        _fail("runtime profile is not canonical JSON", "NONCANONICAL_JSON")
    return value


def _exact(value: object, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(f"{path} has missing or unknown fields")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{path} must be a non-empty canonical string")
    if any(ord(character) < 32 for character in value):
        _fail(f"{path} contains a control character")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail(f"{path} must be a boolean")
    return value


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{path} must be an integer >= {minimum}")
    return value


def _sha(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256.fullmatch(digest) is None:
        _fail(f"{path} must be a lowercase SHA-256")
    return digest


def _safe_relative(value: object, path: str) -> str:
    text = _string(value, path)
    windows = PureWindowsPath(text)
    posix = PurePosixPath(text)
    if (
        "\\" in text
        or windows.is_absolute()
        or windows.drive
        or posix.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in text.split("/"))
    ):
        _fail(f"{path} must be a safe repository-relative path")
    return text


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, order=True)
class SourceBinding:
    path: str
    role: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ProductionRuntimeProfile:
    schema_id: str
    profile_id: str
    target_id: str
    hostname: str
    lane_id: str
    selector: str
    runtime_root: str
    interpreter_layout: str
    system_site_packages: bool
    python_version: str
    package_state: str
    required_distributions: tuple[str, ...]
    unresolved_packages: tuple[str, ...]
    direct_requirements: tuple[str, ...]
    resolved_versions: Mapping[str, str]
    resolved_artifacts: Mapping[str, Mapping[str, Any]]
    requirements_path: str | None
    requirements_sha256: str | None
    setup_plan_path: str | None
    setup_plan_sha256: str | None
    authoritative_inventory_sha256: str | None
    forbidden_distributions: tuple[str, ...]
    device: str
    dtype: str
    visible_gpu_count: int
    gpu_name: str
    cpu_fallback: bool
    mixed_precision: bool
    network_allowed: bool
    ready_for_p1_extraction: bool
    source_bindings: tuple[SourceBinding, ...]
    source_sha256: str
    document: Mapping[str, Any]

    @property
    def package_metadata_resolved(self) -> bool:
        return self.package_state == "resolved"


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    hostname: str
    operating_system: str
    architecture: str
    glibc_version: str
    python_version: str
    torch_runtime_version: str
    torch_cuda_build: str | None
    cuda_available: bool
    visible_gpu_count: int
    selected_device_index: int
    gpu_names: tuple[str, ...]
    offline: bool
    autocast_enabled: bool
    installed_packages: Mapping[str, str]
    torchvision_distribution_present: bool = False
    torchvision_module_spec_present: bool = False
    torchvision_imported_module_present: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeValidation:
    status: str
    profile_id: str
    selector: str
    lane_id: str
    target_id: str
    device: str
    dtype: str
    host_hardware_validated: bool
    package_metadata_resolved: bool
    installable: bool
    ready_for_p1_extraction: bool


def _validate_profile(value: Mapping[str, Any], digest: str) -> ProductionRuntimeProfile:
    top = _exact(
        value,
        {
            "device",
            "lane",
            "packages",
            "policies",
            "profile_id",
            "provenance",
            "python",
            "schema_id",
            "source_bindings",
            "target",
        },
        "profile",
    )
    schema_id = _string(top["schema_id"], "profile.schema_id")
    if schema_id != PROFILE_SCHEMA_ID:
        _fail("profile.schema_id is unsupported")
    profile_id = _string(top["profile_id"], "profile.profile_id")
    target = _exact(
        top["target"],
        {"architecture", "glibc", "gpu_name", "hostname", "operating_system", "target_id"},
        "profile.target",
    )
    target_id = _string(target["target_id"], "profile.target.target_id")
    hostname = _string(target["hostname"], "profile.target.hostname")
    if (
        target_id != APPLE_TARGET_ID
        or hostname != APPLE_HOSTNAME
        or target["operating_system"] != "Linux"
        or target["architecture"] != "x86_64"
        or target["glibc"] != APPLE_GLIBC_VERSION
        or target["gpu_name"] != APPLE_GPU_NAME
    ):
        _fail("profile.target is not the exact accepted Apple target")

    lane = _exact(
        top["lane"],
        {"interpreter_layout", "lane_id", "runtime_root", "selector", "system_site_packages"},
        "profile.lane",
    )
    lane_id = _string(lane["lane_id"], "profile.lane.lane_id")
    selector = _string(lane["selector"], "profile.lane.selector")
    runtime_root = _string(lane["runtime_root"], "profile.lane.runtime_root")
    interpreter_layout = _string(lane["interpreter_layout"], "profile.lane.interpreter_layout")
    system_site_packages = _boolean(lane["system_site_packages"], "profile.lane.system_site_packages")
    allowed_lanes = {
        (MAIN_LANE_ID, MAIN_SELECTOR, MAIN_RUNTIME_ROOT, "linux_bin", False),
        (DINOV3_LANE_ID, DINOV3_SELECTOR, DINOV3_RUNTIME_ROOT, "linux_bin", False),
    }
    if (lane_id, selector, runtime_root, interpreter_layout, system_site_packages) not in allowed_lanes:
        _fail("profile.lane is unknown, hybrid, or unsafe", "HYBRID_RUNTIME_PROFILE")

    python = _exact(top["python"], {"version"}, "profile.python")
    python_version = _string(python["version"], "profile.python.version")
    if python_version != APPLE_PYTHON_VERSION:
        _fail("profile.python.version must be exactly 3.12.0")

    if not isinstance(top["packages"], Mapping):
        _fail("profile.packages must be an object")
    packages = top["packages"]
    package_state = _string(packages["state"], "profile.packages.state")
    unresolved: list[str] = []
    direct: tuple[str, ...] = ()
    resolved_versions: dict[str, str] = {}
    resolved_artifacts: dict[str, Mapping[str, Any]] = {}
    requirements_path: str | None = None
    requirements_sha256: str | None = None
    setup_plan_path: str | None = None
    setup_plan_sha256: str | None = None
    inventory_sha256: str | None = None
    forbidden: tuple[str, ...] = ()
    if package_state == "unresolved":
        packages = _exact(
            packages,
            {"required_distributions", "state", "unresolved"},
            "profile.packages",
        )
        required_raw = packages["required_distributions"]
        unresolved_raw = packages["unresolved"]
        if not isinstance(required_raw, list) or not isinstance(unresolved_raw, list):
            _fail("profile.packages arrays are invalid")
        required = tuple(_string(item, "profile.packages.required_distributions") for item in required_raw)
        if (
            not required
            or required != tuple(sorted(required, key=str.lower))
            or len(required) != len(set(map(str.lower, required)))
            or any(_PACKAGE.fullmatch(item) is None for item in required)
        ):
            _fail("profile.packages.required_distributions is not an exact sorted set")
        for index, item in enumerate(unresolved_raw):
            row = _exact(item, {"distribution", "reason", "source"}, f"profile.packages.unresolved[{index}]")
            distribution = _string(row["distribution"], "unresolved.distribution")
            if distribution not in required:
                _fail("unresolved package is not required")
            if row["reason"] != "exact_linux_x86_64_cp312_artifact_not_yet_authorized":
                _fail("unresolved package reason is not exact")
            if row["source"] != "VEA-P1-T002C-1R-unresolved-metadata":
                _fail("unresolved package provenance is not exact")
            unresolved.append(distribution)
        if tuple(unresolved) != required:
            _fail("unresolved package inventory is incomplete")
    elif package_state == "resolved":
        packages = _exact(
            packages,
            {
                "artifact_count", "artifacts", "authoritative_inventory_sha256",
                "direct_requirements", "forbidden_distributions", "requirements",
                "setup_plan", "state",
            },
            "profile.packages",
        )
        if _integer(packages["artifact_count"], "profile.packages.artifact_count") != 40:
            _fail("resolved artifact count must be exactly 40")
        inventory_sha256 = _sha(packages["authoritative_inventory_sha256"], "profile.packages.authoritative_inventory_sha256")
        direct_raw = packages["direct_requirements"]
        artifacts_raw = packages["artifacts"]
        forbidden_raw = packages["forbidden_distributions"]
        if not isinstance(direct_raw, list) or not isinstance(artifacts_raw, list) or not isinstance(forbidden_raw, list):
            _fail("resolved package arrays are invalid")
        direct = tuple(_string(item, "profile.packages.direct_requirements") for item in direct_raw)
        if direct != tuple(sorted(direct)) or len(direct) != len(set(direct)):
            _fail("direct requirements must be a canonical exact set")
        for item in direct:
            if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*==[^\s]+", item):
                _fail("direct requirement is not an exact pin")
        artifact_rows: list[tuple[str, Mapping[str, Any]]] = []
        for index, item in enumerate(artifacts_raw):
            row = _exact(item, {"bytes", "distribution", "filename", "sha256", "version"}, f"profile.packages.artifacts[{index}]")
            distribution = _string(row["distribution"], "artifact.distribution")
            version = _string(row["version"], "artifact.version")
            if distribution != distribution.lower() or _PACKAGE.fullmatch(distribution) is None:
                _fail("artifact distribution is not normalized")
            _integer(row["bytes"], "artifact.bytes", minimum=1)
            filename = _string(row["filename"], "artifact.filename")
            if not filename.endswith(".whl"):
                _fail("resolved artifacts must be wheels")
            _sha(row["sha256"], "artifact.sha256")
            artifact_rows.append((distribution, row))
            resolved_versions[distribution] = version
            resolved_artifacts[distribution] = _freeze(row)
        if len(artifact_rows) != 40 or tuple(name for name, _ in artifact_rows) != tuple(sorted(name for name, _ in artifact_rows)) or len(resolved_versions) != 40:
            _fail("resolved package closure must contain 40 sorted unique distributions")
        required = tuple(resolved_versions)
        direct_names = tuple(item.split("==", 1)[0] for item in direct)
        if any(resolved_versions.get(name) != item.split("==", 1)[1] for name, item in zip(direct_names, direct)):
            _fail("direct requirements do not match the resolved closure")
        forbidden = tuple(_string(item, "forbidden_distribution") for item in forbidden_raw)
        if forbidden != tuple(sorted(forbidden)) or any(item in resolved_versions for item in forbidden):
            _fail("forbidden distribution is present or noncanonical")
        if "colorama" in resolved_versions or "torchvision" in resolved_versions:
            _fail("forbidden inactive distribution is present")
        if resolved_versions.get("torch") != "2.8.0+cu128":
            _fail("resolved Torch identity is not exact")
        if selector == DINOV3_SELECTOR and (resolved_versions.get("hf-xet") != "1.5.2" or "torchvision" not in forbidden):
            _fail("DINOv3 resolved package invariants are not exact")
        cuda_dependencies = [
            name for name in resolved_versions if name.startswith("nvidia-") or name == "triton"
        ]
        if len(cuda_dependencies) != 15:
            _fail("Torch Linux CUDA dependency closure is not exact")
        req = _exact(packages["requirements"], {"entry_count", "path", "sha256"}, "profile.packages.requirements")
        if _integer(req["entry_count"], "requirements.entry_count") != 40:
            _fail("requirements entry count is not exact")
        requirements_path = _safe_relative(req["path"], "requirements.path")
        requirements_sha256 = _sha(req["sha256"], "requirements.sha256")
        plan = _exact(packages["setup_plan"], {"path", "sha256"}, "profile.packages.setup_plan")
        setup_plan_path = _safe_relative(plan["path"], "setup_plan.path")
        setup_plan_sha256 = _sha(plan["sha256"], "setup_plan.sha256")
    else:
        _fail("profile.packages.state is unsupported")

    device = _exact(
        top["device"],
        {"cuda_required", "device", "dtype", "gpu_name", "visible_gpu_count"},
        "profile.device",
    )
    if device["cuda_required"] is not True:
        _fail("profile.device.cuda_required must be true")
    selected_device = _string(device["device"], "profile.device.device")
    dtype = _string(device["dtype"], "profile.device.dtype")
    visible = _integer(device["visible_gpu_count"], "profile.device.visible_gpu_count", minimum=1)
    gpu_name = _string(device["gpu_name"], "profile.device.gpu_name")
    if (selected_device, dtype, visible, gpu_name) != (APPLE_DEVICE, APPLE_DTYPE, 1, APPLE_GPU_NAME):
        _fail("profile.device is not exact cuda:0 FP32 RTX 5090")

    policies = _exact(
        top["policies"],
        {"cpu_fallback", "mixed_precision", "network_allowed", "ready_for_p1_extraction"},
        "profile.policies",
    )
    cpu_fallback = _boolean(policies["cpu_fallback"], "profile.policies.cpu_fallback")
    mixed_precision = _boolean(policies["mixed_precision"], "profile.policies.mixed_precision")
    network_allowed = _boolean(policies["network_allowed"], "profile.policies.network_allowed")
    ready = _boolean(policies["ready_for_p1_extraction"], "profile.policies.ready_for_p1_extraction")
    if cpu_fallback or mixed_precision or network_allowed or ready:
        _fail("profile policies must fail closed")

    provenance_fields = {"decision", "driver_version", "evidence_run_tag", "reported_cuda_compatibility"}
    if package_state == "resolved":
        provenance_fields.add("package_resolution_task")
    provenance = _exact(top["provenance"], provenance_fields, "profile.provenance")
    if (
        provenance["decision"] != "READY_TO_AUTHOR_APPLE_RUNTIME_SETUP"
        or provenance["driver_version"] != "580.173.02"
        or provenance["reported_cuda_compatibility"] != "13.0"
        or provenance["evidence_run_tag"] != "apple_20260728T094536Z"
    ):
        _fail("profile provenance is not the accepted host evidence")
    if package_state == "resolved" and provenance["package_resolution_task"] != "VEA-P1-T002C-3R2":
        _fail("resolved package provenance is not exact")

    rows_raw = top["source_bindings"]
    if not isinstance(rows_raw, list) or not rows_raw:
        _fail("profile.source_bindings must be a non-empty array")
    rows: list[SourceBinding] = []
    for index, item in enumerate(rows_raw):
        row = _exact(item, {"path", "role", "sha256"}, f"profile.source_bindings[{index}]")
        rows.append(
            SourceBinding(
                _safe_relative(row["path"], "source_binding.path"),
                _string(row["role"], "source_binding.role"),
                _sha(row["sha256"], "source_binding.sha256"),
            )
        )
    if tuple(row.path for row in rows) != tuple(sorted(row.path for row in rows)) or len({row.path for row in rows}) != len(rows):
        _fail("profile.source_bindings must be an exact sorted set")

    return ProductionRuntimeProfile(
        schema_id=schema_id,
        profile_id=profile_id,
        target_id=target_id,
        hostname=hostname,
        lane_id=lane_id,
        selector=selector,
        runtime_root=runtime_root,
        interpreter_layout=interpreter_layout,
        system_site_packages=system_site_packages,
        python_version=python_version,
        package_state=package_state,
        required_distributions=required,
        unresolved_packages=tuple(unresolved),
        direct_requirements=direct,
        resolved_versions=MappingProxyType(resolved_versions),
        resolved_artifacts=MappingProxyType(resolved_artifacts),
        requirements_path=requirements_path,
        requirements_sha256=requirements_sha256,
        setup_plan_path=setup_plan_path,
        setup_plan_sha256=setup_plan_sha256,
        authoritative_inventory_sha256=inventory_sha256,
        forbidden_distributions=forbidden,
        device=selected_device,
        dtype=dtype,
        visible_gpu_count=visible,
        gpu_name=gpu_name,
        cpu_fallback=cpu_fallback,
        mixed_precision=mixed_precision,
        network_allowed=network_allowed,
        ready_for_p1_extraction=ready,
        source_bindings=tuple(rows),
        source_sha256=digest,
        document=_freeze(value),
    )


def load_production_runtime_profile(path: str | Path) -> ProductionRuntimeProfile:
    """Load one canonical Apple profile without importing runtime packages."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        _fail("runtime profile must be a regular non-link file", "UNSAFE_PROFILE_PATH")
    try:
        payload = candidate.read_bytes()
    except OSError as exc:
        raise ProductionRuntimeContractError("unable to read runtime profile") from exc
    return _validate_profile(_strict_json(payload), hashlib.sha256(payload).hexdigest())


def validate_profile_source_bindings(
    profile: ProductionRuntimeProfile,
    repository_root: str | Path,
) -> tuple[SourceBinding, ...]:
    """Verify every profile source row against the supplied repository root."""

    root = Path(repository_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        _fail("repository root must be an absolute non-link directory", "UNSAFE_PROFILE_PATH")
    root = root.resolve(strict=True)
    for row in profile.source_bindings:
        path = root.joinpath(*row.path.split("/"))
        if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
            _fail(f"source binding is missing or unsafe: {row.path}", "SOURCE_BINDING_MISMATCH")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
            _fail(f"source binding SHA mismatch: {row.path}", "SOURCE_BINDING_MISMATCH")
    if profile.package_metadata_resolved:
        for relative, expected, label in (
            (profile.requirements_path, profile.requirements_sha256, "requirements"),
            (profile.setup_plan_path, profile.setup_plan_sha256, "setup plan"),
        ):
            assert relative is not None and expected is not None
            path = root.joinpath(*relative.split("/"))
            if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
                _fail(f"{label} path is missing or unsafe", "PACKAGE_BINDING_MISMATCH")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                _fail(f"{label} SHA mismatch", "PACKAGE_BINDING_MISMATCH")
            if label == "requirements":
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ProductionRuntimeContractError("requirements are not UTF-8", "PACKAGE_BINDING_MISMATCH") from exc
                if payload.startswith(b"\xef\xbb\xbf") or not text.endswith("\n"):
                    _fail("requirements are noncanonical", "PACKAGE_BINDING_MISMATCH")
                lines = text.splitlines()
                parsed: dict[str, tuple[str, str]] = {}
                for line in lines:
                    match = _LOCK_LINE.fullmatch(line)
                    if match is None or match.group(1) in parsed:
                        _fail("requirements contain an unsafe or duplicate entry", "PACKAGE_BINDING_MISMATCH")
                    parsed[match.group(1)] = (match.group(2), match.group(3))
                expected_rows = {
                    name: (str(row["version"]), str(row["sha256"]))
                    for name, row in profile.resolved_artifacts.items()
                }
                if len(lines) != 40 or list(parsed) != sorted(parsed) or parsed != expected_rows:
                    _fail("requirements do not match resolved artifacts", "PACKAGE_BINDING_MISMATCH")
    return profile.source_bindings


def validate_runtime_probe(
    profile: ProductionRuntimeProfile,
    probe: RuntimeProbe,
    *,
    require_resolved_packages: bool = False,
) -> RuntimeValidation:
    """Validate an injected host/runtime/CUDA observation against one profile."""

    exact = (
        (probe.hostname, profile.hostname, "hostname"),
        (probe.operating_system, "Linux", "operating_system"),
        (probe.architecture, "x86_64", "architecture"),
        (probe.glibc_version, APPLE_GLIBC_VERSION, "glibc"),
        (probe.python_version, profile.python_version, "python_version"),
        (probe.visible_gpu_count, profile.visible_gpu_count, "visible_gpu_count"),
        (probe.selected_device_index, 0, "selected_device_index"),
        (probe.gpu_names, (profile.gpu_name,), "gpu_names"),
        (probe.offline, True, "offline"),
        (probe.autocast_enabled, False, "autocast_enabled"),
    )
    for actual, expected, label in exact:
        if actual != expected:
            _fail(f"runtime probe {label} mismatch", "RUNTIME_PROBE_MISMATCH")
    if (
        not isinstance(probe.torch_runtime_version, str)
        or not probe.torch_runtime_version
        or not isinstance(probe.torch_cuda_build, str)
        or not probe.torch_cuda_build
        or not probe.cuda_available
    ):
        _fail("runtime probe is not CUDA-capable", "CUDA_RUNTIME_UNAVAILABLE")
    if require_resolved_packages and not profile.package_metadata_resolved:
        _fail("Apple package metadata remains unresolved", "UNRESOLVED_PACKAGE_METADATA")
    if profile.package_metadata_resolved:
        if set(probe.installed_packages) != set(profile.resolved_versions):
            _fail("installed package set mismatch", "PACKAGE_IDENTITY_MISMATCH")
        for distribution, expected_version in profile.resolved_versions.items():
            version = probe.installed_packages.get(distribution)
            if version != expected_version:
                _fail("installed package identity mismatch", "PACKAGE_IDENTITY_MISMATCH")
        if probe.torch_runtime_version != "2.8.0+cu128" or probe.torch_cuda_build != "12.8":
            _fail("Torch/CUDA build identity mismatch", "PACKAGE_IDENTITY_MISMATCH")
        if profile.selector == DINOV3_SELECTOR and (
            probe.torchvision_distribution_present
            or probe.torchvision_module_spec_present
            or probe.torchvision_imported_module_present
        ):
            _fail("DINOv3 torchvision contamination detected", "PACKAGE_IDENTITY_MISMATCH")
    return RuntimeValidation(
        status=(
            "APPLE_RUNTIME_CAPABILITY_VALIDATED_PACKAGE_METADATA_UNRESOLVED"
            if not profile.package_metadata_resolved
            else "APPLE_RESOLVED_METADATA_AND_RUNTIME_PROBE_VALIDATED"
        ),
        profile_id=profile.profile_id,
        selector=profile.selector,
        lane_id=profile.lane_id,
        target_id=profile.target_id,
        device=profile.device,
        dtype=profile.dtype,
        host_hardware_validated=True,
        package_metadata_resolved=profile.package_metadata_resolved,
        installable=False,
        ready_for_p1_extraction=False,
    )


def tensor_device_string(value: object) -> str:
    """Normalize an injected Torch-like tensor/parameter device identity."""

    device = getattr(value, "device", None)
    if device is None:
        _fail("tensor has no device identity", "TENSOR_CONTRACT_MISMATCH")
    text = str(device)
    if text == "cuda":
        index = getattr(device, "index", None)
        text = f"cuda:{0 if index is None else index}"
    return text


def validate_apple_tensor_contract(
    values: Sequence[object],
    *,
    expected_dtype: object,
    finite_probe: Any,
) -> None:
    """Validate fake or real Torch-like tensors without allocating a tensor."""

    if not values:
        _fail("tensor contract is empty", "TENSOR_CONTRACT_MISMATCH")
    for value in values:
        if tensor_device_string(value) != APPLE_DEVICE or getattr(value, "dtype", None) != expected_dtype:
            _fail("tensor is not cuda:0 FP32", "TENSOR_CONTRACT_MISMATCH")
        finite = finite_probe(value)
        if hasattr(finite, "all"):
            finite = finite.all()
        if hasattr(finite, "item"):
            finite = finite.item()
        if finite is not True:
            _fail("tensor contains NaN or Inf", "TENSOR_CONTRACT_MISMATCH")


__all__ = [
    "APPLE_DEVICE",
    "APPLE_DTYPE",
    "APPLE_GPU_NAME",
    "APPLE_TARGET_ID",
    "DINOV3_LANE_ID",
    "DINOV3_RUNTIME_ROOT",
    "DINOV3_SELECTOR",
    "MAIN_LANE_ID",
    "MAIN_RUNTIME_ROOT",
    "MAIN_SELECTOR",
    "PROFILE_SCHEMA_ID",
    "ProductionRuntimeContractError",
    "ProductionRuntimeProfile",
    "RuntimeProbe",
    "RuntimeValidation",
    "SourceBinding",
    "canonical_json_bytes",
    "load_production_runtime_profile",
    "tensor_device_string",
    "validate_apple_tensor_contract",
    "validate_profile_source_bindings",
    "validate_runtime_probe",
]
