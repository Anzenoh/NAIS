"""Offline, fail-closed authoring and validation for the Apple P1 runtime.

The module is deliberately standard-library-only.  Structural validation never
imports Torch and all external commands are represented as argument tuples.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn

from vision_encoder_alignment.production_runtime_contract import (
    DINOV3_RUNTIME_ROOT,
    DINOV3_SELECTOR,
    MAIN_RUNTIME_ROOT,
    MAIN_SELECTOR,
    ProductionRuntimeContractError,
    RuntimeProbe,
    load_production_runtime_profile,
    validate_profile_source_bindings,
    validate_runtime_probe,
)


SETUP_SCHEMA_ID = "vea.p1.apple_runtime_setup_plan.v1"
PROBE_SCHEMA_ID = "vea.p1.apple_runtime_probe.v1"
RESULT_SCHEMA_ID = "vea.p1.apple_runtime_setup_validation.v1"
TARGET_ID = "lab_apple_rtx5090"
HOSTNAME = "apple"
GPU_NAME = "NVIDIA GeForce RTX 5090"
PYTHON_VERSION = "3.12.0"
TORCH_VERSION = "2.8.0+cu128"
TORCH_CUDA_BUILD = "12.8"
GLIBC_VERSION = "2.35"
PLAN_RELATIVE = "configs/runtime_profiles/p1_apple_rtx5090_runtime_setup.json"
MAIN_REQUIREMENTS_RELATIVE = "configs/runtime_profiles/p1_apple_primary.requirements.txt"
DINOV3_REQUIREMENTS_RELATIVE = (
    "configs/runtime_profiles/p1_apple_dinov3_isolated.requirements.txt"
)
MAIN_PROFILE_RELATIVE = "configs/runtime_profiles/p1_apple_main_cuda_fp32_v0.json"
DINOV3_PROFILE_RELATIVE = "configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCK = re.compile(
    r"^([a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)==([^\s]+) --hash=sha256:([0-9a-f]{64})$"
)
_MAX_JSON_BYTES = 2 * 1024 * 1024
_DISTRIBUTION_SEPARATOR = re.compile(r"[-_.]+")
_NORMALIZED_DISTRIBUTION = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class AppleRuntimeSetupError(RuntimeError):
    """Stable classified failure for setup authoring and validation."""

    def __init__(self, message: str, classification: str = "SETUP_CONTRACT_INVALID") -> None:
        super().__init__(message)
        self.classification = classification


def _fail(message: str, classification: str = "SETUP_CONTRACT_INVALID") -> NoReturn:
    raise AppleRuntimeSetupError(message, classification)


def _reject_constant(value: str) -> NoReturn:
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
        raise AppleRuntimeSetupError("value is not finite JSON", "NONCANONICAL_JSON") from exc


def strict_json_bytes(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > _MAX_JSON_BYTES or payload.startswith(b"\xef\xbb\xbf"):
        _fail("JSON has an invalid size or UTF-8 BOM", "NONCANONICAL_JSON")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AppleRuntimeSetupError("JSON is not strict UTF-8", "NONCANONICAL_JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        _fail("JSON is not canonical", "NONCANONICAL_JSON")
    return value


def load_strict_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        _fail("JSON input must be a regular non-link file", "UNSAFE_PATH")
    return strict_json_bytes(candidate.read_bytes())


def _exact(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(f"{label} has missing or unknown fields")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{label} must be a non-empty canonical string")
    if any(ord(character) < 32 for character in value):
        _fail(f"{label} contains a control character")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be a boolean")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if _SHA256.fullmatch(text) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return text


def _normalized_distribution_name(value: object, label: str) -> str:
    name = _string(value, label)
    normalized = _DISTRIBUTION_SEPARATOR.sub("-", name).lower()
    if _NORMALIZED_DISTRIBUTION.fullmatch(normalized) is None:
        _fail(f"{label} is not a valid distribution name", "RUNTIME_PROBE_MISMATCH")
    return normalized


def _parse_pyvenv_cfg(value: object) -> Mapping[str, str]:
    if not isinstance(value, str) or not value or "\r" in value or "\x00" in value:
        _fail("pyvenv.cfg is missing or malformed", "LANE_ISOLATION_FAILURE")
    text = value
    settings: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            _fail("pyvenv.cfg is malformed", "LANE_ISOLATION_FAILURE")
        key, raw_value = line.split("=", 1)
        key = key.strip().lower()
        setting = raw_value.strip().lower()
        if not key or not setting or key in settings:
            _fail("pyvenv.cfg is malformed or has duplicate settings", "LANE_ISOLATION_FAILURE")
        settings[key] = setting
    if settings.get("include-system-site-packages") != "false":
        _fail("pyvenv.cfg does not disable system site packages", "LANE_ISOLATION_FAILURE")
    return settings


def _canonical_probe_posix_path(value: object, label: str) -> PurePosixPath:
    text = _string(value, label)
    path = PurePosixPath(text)
    if not path.is_absolute() or str(path) != text or ".." in path.parts or "//" in text:
        _fail(f"{label} is not a canonical absolute POSIX path", "LANE_ISOLATION_FAILURE")
    return path


def _repository_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or PureWindowsPath(relative).drive:
        _fail("repository path is unsafe", "UNSAFE_PATH")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"repository file is missing or unsafe: {relative}", "UNSAFE_PATH")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AppleRuntimeSetupError("repository file cannot be resolved", "UNSAFE_PATH") from exc
    if not resolved.is_relative_to(root):
        _fail("repository file escapes root", "UNSAFE_PATH")
    return candidate


def _safe_repository_root(value: str | Path) -> Path:
    root = Path(value)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        _fail("repository root must be an absolute non-link directory", "UNSAFE_PATH")
    resolved = root.resolve(strict=True)
    if not (resolved / ".git").is_dir():
        _fail("repository root does not contain .git", "UNSAFE_PATH")
    return resolved


def validate_new_posix_root(value: str, *, forbidden_roots: Sequence[str]) -> PurePosixPath:
    """Validate a future no-overwrite user-owned Linux root lexically."""

    if not isinstance(value, str) or not value.startswith("/") or "//" in value:
        _fail("target root must be a canonical absolute POSIX path", "UNSAFE_RUNTIME_ROOT")
    path = PurePosixPath(value)
    if str(path) != value or ".." in path.parts or len(path.parts) < 5:
        _fail("target root is broad or noncanonical", "UNSAFE_RUNTIME_ROOT")
    broad = {"/", "/home", "/home/chltjsdh", "/home/chltjsdh/vea"}
    if value in broad:
        _fail("target root is too broad", "UNSAFE_RUNTIME_ROOT")
    for forbidden in forbidden_roots:
        base = PurePosixPath(forbidden)
        if path == base or path.is_relative_to(base):
            _fail("target root overlaps a forbidden root", "UNSAFE_RUNTIME_ROOT")
    return path


@dataclass(frozen=True, slots=True)
class ArtifactRow:
    distribution: str
    version: str
    filename: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LanePlan:
    lane_id: str
    selector: str
    runtime_root: str
    requirements_path: str
    requirements_sha256: str
    artifacts: tuple[ArtifactRow, ...]
    versions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SetupPlan:
    source_sha256: str
    document: Mapping[str, Any]
    repository_root: Path
    python_filename: str
    python_bytes: int
    python_sha256: str
    main: LanePlan
    dinov3: LanePlan

    @property
    def artifact_union(self) -> tuple[ArtifactRow, ...]:
        rows: dict[str, ArtifactRow] = {}
        for row in self.main.artifacts + self.dinov3.artifacts:
            prior = rows.get(row.filename)
            if prior is not None and prior != row:
                _fail("artifact filename collision has conflicting identities")
            rows[row.filename] = row
        return tuple(rows[name] for name in sorted(rows))


def _validate_requirements(path: Path, expected_sha: str, artifacts: tuple[ArtifactRow, ...]) -> None:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        _fail("requirements SHA mismatch", "REQUIREMENTS_MISMATCH")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AppleRuntimeSetupError("requirements are not UTF-8", "REQUIREMENTS_MISMATCH") from exc
    if payload.startswith(b"\xef\xbb\xbf") or not text.endswith("\n"):
        _fail("requirements are noncanonical", "REQUIREMENTS_MISMATCH")
    parsed: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        match = _LOCK.fullmatch(line)
        if match is None or match.group(1) in parsed:
            _fail("requirements contain a floating, unsafe, or duplicate entry", "REQUIREMENTS_MISMATCH")
        parsed[match.group(1)] = (match.group(2), match.group(3))
    expected = {row.distribution: (row.version, row.sha256) for row in artifacts}
    if len(parsed) != 40 or list(parsed) != sorted(parsed) or parsed != expected:
        _fail("requirements do not match the exact 40-artifact lane", "REQUIREMENTS_MISMATCH")


def _lane_plan(
    raw: object,
    *,
    name: str,
    selector: str,
    runtime_root: str,
    requirements_path: str,
    repository_root: Path,
) -> LanePlan:
    lane = _exact(
        raw,
        {
            "artifact_count", "artifacts", "authoritative_inventory_sha256", "no_overwrite",
            "offline_installation_required", "requirements_path", "requirements_sha256",
            "runtime_root", "selector", "system_site_packages",
        },
        f"plan.lanes.{name}",
    )
    if (
        _integer(lane["artifact_count"], "artifact_count") != 40
        or lane["selector"] != selector
        or lane["runtime_root"] != runtime_root
        or lane["requirements_path"] != requirements_path
        or lane["no_overwrite"] is not True
        or lane["offline_installation_required"] is not True
        or lane["system_site_packages"] is not False
    ):
        _fail(f"{name} lane identity or policy is not exact")
    _sha(lane["authoritative_inventory_sha256"], "authoritative_inventory_sha256")
    expected_req_sha = _sha(lane["requirements_sha256"], "requirements_sha256")
    raw_rows = lane["artifacts"]
    if not isinstance(raw_rows, list):
        _fail("lane artifacts must be an array")
    rows: list[ArtifactRow] = []
    for index, raw_row in enumerate(raw_rows):
        row = _exact(
            raw_row,
            {"bytes", "distribution", "filename", "sha256", "version"},
            f"artifact[{index}]",
        )
        distribution = _string(row["distribution"], "distribution")
        filename = _string(row["filename"], "filename")
        if distribution != distribution.lower() or not filename.endswith(".whl"):
            _fail("artifact must be a normalized wheel")
        if PurePosixPath(filename).name != filename or PureWindowsPath(filename).drive:
            _fail("artifact filename is unsafe")
        rows.append(
            ArtifactRow(
                distribution=distribution,
                version=_string(row["version"], "version"),
                filename=filename,
                bytes=_integer(row["bytes"], "bytes", 1),
                sha256=_sha(row["sha256"], "sha256"),
            )
        )
    if len(rows) != 40 or tuple(row.distribution for row in rows) != tuple(
        sorted(row.distribution for row in rows)
    ) or len({row.distribution for row in rows}) != 40:
        _fail("lane artifact closure must be 40 sorted unique distributions")
    if any(not math.isfinite(float(row.bytes)) for row in rows):
        _fail("artifact bytes are invalid")
    if name == "dinov3":
        versions = {row.distribution: row.version for row in rows}
        if versions.get("hf-xet") != "1.5.2" or "torchvision" in versions:
            _fail("DINOv3 isolation package identity is invalid")
    req_file = _repository_file(repository_root, requirements_path)
    _validate_requirements(req_file, expected_req_sha, tuple(rows))
    return LanePlan(
        lane_id=name,
        selector=selector,
        runtime_root=runtime_root,
        requirements_path=requirements_path,
        requirements_sha256=expected_req_sha,
        artifacts=tuple(rows),
        versions={row.distribution: row.version for row in rows},
    )


def load_setup_plan(repository_root: str | Path) -> SetupPlan:
    """Load and bind the canonical plan, locks, and resolved runtime profiles."""

    root = _safe_repository_root(repository_root)
    plan_path = _repository_file(root, PLAN_RELATIVE)
    payload = plan_path.read_bytes()
    value = strict_json_bytes(payload)
    top = _exact(
        value,
        {
            "artifact_inventory", "lanes", "plan_id", "policies", "provenance", "python",
            "readiness", "schema_id", "target", "task_id",
        },
        "plan",
    )
    if top["schema_id"] != SETUP_SCHEMA_ID or top["plan_id"] != "p1-apple-rtx5090-runtime-setup-v1":
        _fail("setup plan identity is unsupported")
    target = _exact(
        top["target"],
        {
            "architecture", "device", "driver_version", "dtype", "glibc", "gpu_name",
            "hostname", "operating_system", "reported_cuda_compatibility", "target_id",
        },
        "plan.target",
    )
    if (
        target["target_id"], target["hostname"], target["operating_system"],
        target["architecture"], target["glibc"], target["gpu_name"], target["device"],
        target["dtype"], target["driver_version"], target["reported_cuda_compatibility"],
    ) != (TARGET_ID, HOSTNAME, "Linux", "x86_64", GLIBC_VERSION, GPU_NAME, "cuda:0", "float32", "580.173.02", "13.0"):
        _fail("setup plan target is not the accepted Apple target")
    python = _exact(top["python"], {"bytes", "filename", "sha256", "source_url", "version"}, "plan.python")
    if (
        python["version"] != PYTHON_VERSION
        or python["filename"] != "Python-3.12.0.tar.xz"
        or python["bytes"] != 20575020
        or python["sha256"] != "795c34f44df45a0e9b9710c8c71c15c671871524cd412ca14def212e8ccb155d"
        or python["source_url"] != "https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tar.xz"
    ):
        _fail("CPython artifact identity is not exact")
    policies = _exact(
        top["policies"],
        {
            "authoring_only", "gpu_validated", "installation_executed", "model_loaded",
            "network_allowed", "no_overwrite", "ready_for_p1_extraction", "runtime_created",
            "server_accessed",
        },
        "plan.policies",
    )
    if policies != {
        "authoring_only": True,
        "gpu_validated": False,
        "installation_executed": False,
        "model_loaded": False,
        "network_allowed": False,
        "no_overwrite": True,
        "ready_for_p1_extraction": False,
        "runtime_created": False,
        "server_accessed": False,
    }:
        _fail("setup plan policies do not fail closed")
    readiness = _exact(
        top["readiness"],
        {
            "ready_for_apple_p1_t002d_runtime_setup_execution", "ready_for_p1_extraction",
            "ready_to_retry_vea_p1_t002c",
        },
        "plan.readiness",
    )
    if readiness != {
        "ready_for_apple_p1_t002d_runtime_setup_execution": False,
        "ready_for_p1_extraction": False,
        "ready_to_retry_vea_p1_t002c": True,
    }:
        _fail("setup plan readiness is forged")
    inventory = _exact(top["artifact_inventory"], {"artifact_count", "path", "sha256"}, "artifact_inventory")
    if inventory["artifact_count"] != 80 or inventory["path"] != "manifests/authoritative-artifact-inventory.json":
        _fail("authoritative artifact inventory identity is invalid")
    _sha(inventory["sha256"], "artifact_inventory.sha256")
    lanes = _exact(top["lanes"], {"dinov3", "main"}, "plan.lanes")
    main = _lane_plan(
        lanes["main"], name="main", selector=MAIN_SELECTOR, runtime_root=MAIN_RUNTIME_ROOT,
        requirements_path=MAIN_REQUIREMENTS_RELATIVE, repository_root=root,
    )
    dinov3 = _lane_plan(
        lanes["dinov3"], name="dinov3", selector=DINOV3_SELECTOR,
        runtime_root=DINOV3_RUNTIME_ROOT, requirements_path=DINOV3_REQUIREMENTS_RELATIVE,
        repository_root=root,
    )
    if main.runtime_root == dinov3.runtime_root:
        _fail("runtime lane roots collide")
    digest = hashlib.sha256(payload).hexdigest()
    for profile_relative, lane in (
        (MAIN_PROFILE_RELATIVE, main),
        (DINOV3_PROFILE_RELATIVE, dinov3),
    ):
        profile = load_production_runtime_profile(_repository_file(root, profile_relative))
        validate_profile_source_bindings(profile, root)
        if (
            profile.selector != lane.selector
            or profile.runtime_root != lane.runtime_root
            or profile.requirements_path != lane.requirements_path
            or profile.requirements_sha256 != lane.requirements_sha256
            or profile.setup_plan_path != PLAN_RELATIVE
            or profile.setup_plan_sha256 != digest
            or dict(profile.resolved_versions) != dict(lane.versions)
        ):
            _fail("resolved profile/setup/lock binding mismatch", "PROFILE_BINDING_MISMATCH")
    return SetupPlan(
        source_sha256=digest,
        document=value,
        repository_root=root,
        python_filename=str(python["filename"]),
        python_bytes=int(python["bytes"]),
        python_sha256=str(python["sha256"]),
        main=main,
        dinov3=dinov3,
    )


ArtifactProbe = Callable[[Path], tuple[int, str]]


def _default_artifact_probe(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        _fail(f"artifact is missing or unsafe: {path.name}", "ARTIFACT_MISMATCH")
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def validate_artifact_root(
    plan: SetupPlan,
    artifact_root: str | Path,
    *,
    probe: ArtifactProbe = _default_artifact_probe,
    observed_relative_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the exact local CPython/wheel layout without network fallback."""

    root = Path(artifact_root)
    lexical_absolute = root.is_absolute() or PurePosixPath(str(artifact_root)).is_absolute()
    if not lexical_absolute or root.is_symlink() or (probe is _default_artifact_probe and not root.is_dir()):
        _fail("artifact root must be an absolute non-link directory", "UNSAFE_PATH")
    expected: dict[str, tuple[int, str]] = {
        f"python/{plan.python_filename}": (plan.python_bytes, plan.python_sha256)
    }
    for row in plan.artifact_union:
        expected[f"wheels/{row.filename}"] = (row.bytes, row.sha256)
    if observed_relative_paths is None:
        observed: list[str] = []
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                _fail("artifact tree contains a symlink", "ARTIFACT_MISMATCH")
            if candidate.is_file():
                observed.append(candidate.relative_to(root).as_posix())
    else:
        observed = list(observed_relative_paths)
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        _fail("artifact root has missing, extra, duplicate, or conflicting files", "ARTIFACT_MISMATCH")
    for relative in sorted(expected):
        path = root.joinpath(*PurePosixPath(relative).parts)
        actual_bytes, actual_sha = probe(path)
        if (actual_bytes, actual_sha) != expected[relative]:
            _fail(f"artifact bytes/SHA mismatch: {relative}", "ARTIFACT_MISMATCH")
    return {
        "artifact_file_count": len(expected),
        "cpython_artifact_count": 1,
        "dinov3_artifact_count": len(plan.dinov3.artifacts),
        "main_artifact_count": len(plan.main.artifacts),
        "missing": 0,
        "extra": 0,
        "sha256_mismatch": 0,
    }


_LANE_PROBE_FIELDS = {
    "autocast_enabled", "base_prefix", "cuda_available", "cudnn_version",
    "distribution_metadata_paths", "gpu_names", "installed_packages", "interpreter",
    "offline", "python_version", "pyvenv_cfg", "runtime_root", "selected_device_index",
    "site_packages", "sys_path", "system_site_packages", "torch_cuda_build",
    "torch_runtime_version", "torchvision_distribution_present", "user_site_enabled",
    "torchvision_imported_module_present", "torchvision_module_spec_present",
    "visible_gpu_count",
}


def _runtime_probe_for_lane(
    raw: object,
    *,
    lane: LanePlan,
    host: Mapping[str, Any],
) -> RuntimeProbe:
    value = _exact(raw, _LANE_PROBE_FIELDS, f"probe.lanes.{lane.lane_id}")
    expected_interpreter = f"{lane.runtime_root}/bin/python"
    expected_site_packages = f"{lane.runtime_root}/lib/python3.12/site-packages"
    other_runtime_root = DINOV3_RUNTIME_ROOT if lane.runtime_root == MAIN_RUNTIME_ROOT else MAIN_RUNTIME_ROOT
    runtime_root = _canonical_probe_posix_path(value["runtime_root"], "runtime_root")
    interpreter = _canonical_probe_posix_path(value["interpreter"], "interpreter")
    site_packages = _canonical_probe_posix_path(value["site_packages"], "site_packages")
    base_prefix = _canonical_probe_posix_path(value["base_prefix"], "base_prefix")
    _parse_pyvenv_cfg(value["pyvenv_cfg"])
    if (
        str(runtime_root) != lane.runtime_root
        or str(interpreter) != expected_interpreter
        or str(site_packages) != expected_site_packages
        or base_prefix == runtime_root
        or value["system_site_packages"] is not False
        or value["user_site_enabled"] is not False
    ):
        _fail("runtime lane path or isolation mismatch", "LANE_ISOLATION_FAILURE")

    sys_path = value["sys_path"]
    if not isinstance(sys_path, list) or not sys_path:
        _fail("runtime sys.path observation is invalid", "LANE_ISOLATION_FAILURE")
    observed_paths: list[PurePosixPath] = []
    for index, item in enumerate(sys_path):
        path = _canonical_probe_posix_path(item, f"sys_path[{index}]")
        if path in observed_paths:
            _fail("runtime sys.path contains a duplicate", "LANE_ISOLATION_FAILURE")
        observed_paths.append(path)
    package_paths = {
        path for path in observed_paths if "site-packages" in path.parts or "dist-packages" in path.parts
    }
    if package_paths != {site_packages} or any(
        path == PurePosixPath(other_runtime_root) or path.is_relative_to(PurePosixPath(other_runtime_root))
        for path in observed_paths
    ):
        _fail("system, user, or other-lane package path is visible", "LANE_ISOLATION_FAILURE")

    packages = value["installed_packages"]
    if not isinstance(packages, Mapping):
        _fail("installed package map is invalid", "RUNTIME_PROBE_MISMATCH")
    normalized_packages: dict[str, str] = {}
    for raw_name, raw_version in packages.items():
        name = _normalized_distribution_name(raw_name, "installed package name")
        version = _string(raw_version, f"installed_packages.{name}")
        if name in normalized_packages:
            _fail("normalized distribution name collision", "PACKAGE_IDENTITY_MISMATCH")
        normalized_packages[name] = version

    metadata_paths = value["distribution_metadata_paths"]
    if not isinstance(metadata_paths, Mapping):
        _fail("distribution metadata path map is invalid", "RUNTIME_PROBE_MISMATCH")
    normalized_metadata_paths: dict[str, PurePosixPath] = {}
    for raw_name, raw_path in metadata_paths.items():
        name = _normalized_distribution_name(raw_name, "distribution metadata name")
        if name in normalized_metadata_paths:
            _fail("normalized metadata name collision", "PACKAGE_IDENTITY_MISMATCH")
        normalized_metadata_paths[name] = _canonical_probe_posix_path(
            raw_path, f"distribution_metadata_paths.{name}"
        )
    if set(normalized_metadata_paths) != set(normalized_packages):
        _fail("distribution metadata/package identity mismatch", "PACKAGE_IDENTITY_MISMATCH")
    if any(path != site_packages for path in normalized_metadata_paths.values()):
        _fail("distribution metadata escapes the exact lane site-packages", "LANE_ISOLATION_FAILURE")

    gpu_names = value["gpu_names"]
    if not isinstance(gpu_names, list) or any(not isinstance(item, str) for item in gpu_names):
        _fail("GPU names are invalid", "RUNTIME_PROBE_MISMATCH")
    return RuntimeProbe(
        hostname=str(host["hostname"]),
        operating_system=str(host["operating_system"]),
        architecture=str(host["architecture"]),
        glibc_version=str(host["glibc"]),
        python_version=_string(value["python_version"], "python_version"),
        torch_runtime_version=_string(value["torch_runtime_version"], "torch_runtime_version"),
        torch_cuda_build=_string(value["torch_cuda_build"], "torch_cuda_build"),
        cuda_available=_boolean(value["cuda_available"], "cuda_available"),
        visible_gpu_count=_integer(value["visible_gpu_count"], "visible_gpu_count"),
        selected_device_index=_integer(value["selected_device_index"], "selected_device_index"),
        gpu_names=tuple(gpu_names),
        offline=_boolean(value["offline"], "offline"),
        autocast_enabled=_boolean(value["autocast_enabled"], "autocast_enabled"),
        installed_packages=normalized_packages,
        torchvision_distribution_present=_boolean(value["torchvision_distribution_present"], "torchvision_distribution_present"),
        torchvision_module_spec_present=_boolean(value["torchvision_module_spec_present"], "torchvision_module_spec_present"),
        torchvision_imported_module_present=_boolean(value["torchvision_imported_module_present"], "torchvision_imported_module_present"),
    )


def validate_runtime_probe_document(plan: SetupPlan, document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an injected two-lane runtime observation; never grants extraction."""

    top = _exact(document, {"host", "lanes", "schema_id"}, "probe")
    if top["schema_id"] != PROBE_SCHEMA_ID:
        _fail("runtime probe schema is unsupported", "RUNTIME_PROBE_MISMATCH")
    host = _exact(top["host"], {"architecture", "glibc", "hostname", "operating_system"}, "probe.host")
    if host != {
        "architecture": "x86_64", "glibc": GLIBC_VERSION, "hostname": HOSTNAME,
        "operating_system": "Linux",
    }:
        _fail("runtime host identity mismatch", "RUNTIME_PROBE_MISMATCH")
    lanes = _exact(top["lanes"], {"dinov3", "main"}, "probe.lanes")
    main_probe = _runtime_probe_for_lane(lanes["main"], lane=plan.main, host=host)
    dino_probe = _runtime_probe_for_lane(lanes["dinov3"], lane=plan.dinov3, host=host)
    main_profile = load_production_runtime_profile(_repository_file(plan.repository_root, MAIN_PROFILE_RELATIVE))
    dino_profile = load_production_runtime_profile(_repository_file(plan.repository_root, DINOV3_PROFILE_RELATIVE))
    try:
        main_result = validate_runtime_probe(main_profile, main_probe, require_resolved_packages=True)
        dino_result = validate_runtime_probe(dino_profile, dino_probe, require_resolved_packages=True)
    except ProductionRuntimeContractError as exc:
        raise AppleRuntimeSetupError(str(exc), exc.classification) from exc
    if (
        lanes["main"]["site_packages"] == lanes["dinov3"]["site_packages"]
        or lanes["main"]["interpreter"] == lanes["dinov3"]["interpreter"]
    ):
        _fail("runtime lanes are not isolated", "LANE_ISOLATION_FAILURE")
    if lanes["main"]["cudnn_version"] in (None, "") or lanes["dinov3"]["cudnn_version"] in (None, ""):
        _fail("cuDNN observation is missing", "RUNTIME_PROBE_MISMATCH")
    return {
        "classification": "APPLE_TWO_LANE_RUNTIME_VALIDATED_NOT_EXTRACTION_READY",
        "lanes": {
            "dinov3": {"profile_id": dino_result.profile_id, "status": dino_result.status},
            "main": {"profile_id": main_result.profile_id, "status": main_result.status},
        },
        "ready_for_p1_extraction": False,
        "runtime_validated": True,
        "schema_id": RESULT_SCHEMA_ID,
        "target_id": TARGET_ID,
    }


def structural_result(plan: SetupPlan) -> dict[str, Any]:
    return {
        "artifact_validation_performed": False,
        "classification": "APPLE_RUNTIME_SETUP_STRUCTURE_VALID",
        "dinov3_artifact_count": len(plan.dinov3.artifacts),
        "dry_run": True,
        "main_artifact_count": len(plan.main.artifacts),
        "network_operations": 0,
        "package_manager_operations": 0,
        "plan_sha256": plan.source_sha256,
        "ready_for_p1_extraction": False,
        "runtime_created": False,
        "schema_id": RESULT_SCHEMA_ID,
        "target_id": TARGET_ID,
    }


def build_operation_plan(
    plan: SetupPlan,
    *,
    artifact_root: str,
    python_root: str,
    work_root: str,
) -> tuple[tuple[str, ...], ...]:
    """Return the exact shell-free offline command plan; perform no writes."""

    forbidden = (plan.repository_root.as_posix(), MAIN_RUNTIME_ROOT, DINOV3_RUNTIME_ROOT)
    py_root = validate_new_posix_root(python_root, forbidden_roots=forbidden)
    work = validate_new_posix_root(work_root, forbidden_roots=forbidden + (str(py_root),))
    artifacts = PurePosixPath(artifact_root)
    if not artifacts.is_absolute() or ".." in artifacts.parts or str(artifacts) != artifact_root:
        _fail("artifact root must be a canonical absolute POSIX path", "UNSAFE_PATH")
    archive = str(artifacts / "python" / plan.python_filename)
    wheels = str(artifacts / "wheels")
    source = str(work / "Python-3.12.0")
    python = str(py_root / "bin" / "python3.12")
    commands: list[tuple[str, ...]] = [
        ("mkdir", "--", str(work)),
        ("tar", "--extract", "--file", archive, "--directory", str(work)),
        (f"{source}/configure", f"--prefix={py_root}", "--with-ensurepip=install"),
        ("make", "-C", str(work), "-j1"),
        ("make", "-C", str(work), "install"),
        (python, "-c", "import platform,sys; assert platform.python_version() == '3.12.0'; assert sys.prefix == sys.base_prefix"),
    ]
    for lane in (plan.main, plan.dinov3):
        lane_python = f"{lane.runtime_root}/bin/python"
        commands.extend(
            [
                (python, "-m", "venv", "--without-pip", "--copies", lane.runtime_root),
                (
                    python, "-m", "pip", "--python", lane_python, "install", "--no-index",
                    "--only-binary=:all:", "--no-deps", "--require-hashes", "--find-links", wheels,
                    "--requirement", str(plan.repository_root.joinpath(*PurePosixPath(lane.requirements_path).parts)),
                ),
            ]
        )
    return tuple(commands)


CommandRunner = Callable[[Sequence[str], Mapping[str, str]], subprocess.CompletedProcess[str]]


def _run_command(argv: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), check=False, capture_output=True, text=True, env=dict(env), shell=False)


def execute_operation_plan(
    commands: Sequence[Sequence[str]],
    *,
    runner: CommandRunner = _run_command,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], ...]:
    """Execute an already validated command plan; intended only for T002D."""

    environment = {
        "PATH": (base_environment or os.environ).get("PATH", ""),
        "HOME": (base_environment or os.environ).get("HOME", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    results: list[subprocess.CompletedProcess[str]] = []
    for command in commands:
        if not command or any(not isinstance(item, str) or not item for item in command):
            _fail("command plan contains an invalid argument", "UNSAFE_COMMAND")
        result = runner(tuple(command), environment)
        results.append(result)
        if result.returncode != 0:
            _fail("setup command failed; partial state must not be reused", "SETUP_EXECUTION_FAILED")
    return tuple(results)


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Create a canonical result exactly once without leaking local paths."""

    def audit(item: Any, key: str = "") -> None:
        lowered_key = key.lower()
        if any(word in lowered_key for word in ("authorization", "access_token", "password")):
            _fail("result contains credential-shaped data", "SENSITIVE_OUTPUT")
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                audit(child, str(child_key))
        elif isinstance(item, (list, tuple)):
            for child in item:
                audit(child, key)
        elif isinstance(item, str) and re.match(r"^[A-Za-z]:[\\/]Users[\\/]", item, re.IGNORECASE):
            _fail("result contains a Windows user path", "SENSITIVE_OUTPUT")

    audit(value)
    payload = canonical_json_bytes(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise AppleRuntimeSetupError("output already exists", "OUTPUT_EXISTS") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            target.unlink()
        except OSError:
            pass
        raise


def live_host_identity() -> dict[str, str]:
    """Collect non-GPU host identity only when explicit live mode requests it."""

    libc_name, libc_version = platform.libc_ver()
    if libc_name != "glibc":
        _fail("live host does not report glibc", "RUNTIME_PROBE_MISMATCH")
    return {
        "architecture": platform.machine(),
        "glibc": libc_version,
        "hostname": platform.node(),
        "operating_system": platform.system(),
    }


__all__ = [
    "AppleRuntimeSetupError", "ArtifactRow", "LanePlan", "SetupPlan", "PROBE_SCHEMA_ID",
    "RESULT_SCHEMA_ID", "atomic_write_json", "build_operation_plan", "canonical_json_bytes",
    "execute_operation_plan", "load_setup_plan", "load_strict_json", "strict_json_bytes",
    "structural_result", "validate_artifact_root", "validate_new_posix_root",
    "validate_runtime_probe_document",
]
