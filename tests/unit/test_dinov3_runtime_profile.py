"""Unit tests for the import-light DINOv3 persistent-runtime boundary."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, dataclass
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import vision_encoder_alignment.runtime_boundary as boundary
from vision_encoder_alignment.runtime_boundary import (
    RuntimeBoundaryError,
    RuntimeManifest,
    RuntimeProfile,
    WorkerRequest,
    WorkerResponse,
    build_sanitized_worker_environment,
    invoke_isolated_worker,
    load_runtime_manifest,
    load_runtime_profile,
    read_worker_response,
    resolve_default_runtime_root,
    resolve_workspace_relative_path,
    validate_runtime_manifest,
    validate_runtime_root,
    write_worker_request,
)


PROFILE_ID = "dinov3-manual-fp32-imagenet-v1"
PROFILE_SCHEMA = "vea.dinov3_isolated_runtime_profile.v0"
MANIFEST_SCHEMA = "vea.dinov3_persistent_runtime.v0"
REQUEST_SCHEMA = "vea.dinov3_isolated_worker_request.v0"
RESPONSE_SCHEMA = "vea.dinov3_isolated_worker_response.v0"
HEAD = "e518a51ba83f70b42611cc4386a9f55dc9e97de3"
INPUT_SHA256 = "b0fa2f6e3b2fbca50cde7e4a2dc3a65a0f788a184aec382365e520d54b468411"
OUTPUT_SHA256 = "c0ebe2af6db2e61d052dc58272379beef749a4c5392d4e2204a11862cf172526"
ARTIFACT_MARKERS = [
    "DINOV3_PERSISTENT_RUNTIME_MANIFEST_OK",
    "DINOV3_PERSISTENT_INBOUND_ARTIFACT_OK",
    "DINOV3_PERSISTENT_OUTBOUND_ARTIFACT_OK",
    "DINOV3_PERSISTENT_RUNTIME_NETWORK_GUARD_OK",
]
RUNTIME_MARKERS = ARTIFACT_MARKERS + [
    "DINOV3_PERSISTENT_MANUAL_PREPROCESSING_OK",
    "DINOV3_PERSISTENT_RANDOM_FORWARD_OK",
]


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _profile_path(repository_root: Path) -> Path:
    return (
        repository_root
        / "configs"
        / "runtime_profiles"
        / "dinov3_isolated_manual_v0.json"
    )


def _profile_value(repository_root: Path) -> dict[str, Any]:
    return json.loads(_profile_path(repository_root).read_text(encoding="utf-8"))


def _load_provisioner(repository_root: Path) -> Any:
    path = repository_root / "scripts" / "provision_dinov3_runtime.py"
    module_name = "_test_provision_dinov3_runtime"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _request_value(profile_sha256: str, operation: str = "artifact_boundary_self_test") -> dict[str, Any]:
    return {
        "deterministic_seed": 0,
        "expected_preprocessing_profile_id": PROFILE_ID,
        "operation": operation,
        "repository_head": HEAD,
        "run_id": "t007a2-unit",
        "runtime_profile_sha256": profile_sha256,
        "schema_id": REQUEST_SCHEMA,
        "workspace_relative_inbound_artifact_path": "inbound",
        "workspace_relative_outbound_artifact_path": "outbound",
    }


def _artifact_entry(tap_name: str, offset: int) -> dict[str, Any]:
    return {
        "checksum": f"{offset + 1:064x}",
        "dtype": "float32",
        "global_shape": [2, 3],
        "global_values_sha256": f"{offset + 2:064x}",
        "image_ids": ["image-b", "image-a"],
        "patch_shape": [2, 4, 3],
        "patch_values_sha256": f"{offset + 3:064x}",
        "sidecar": f"{tap_name}.json",
        "tap_name": tap_name,
        "tensor_keys": [
            f"backbone.{tap_name}.patch",
            f"backbone.{tap_name}.global_mean",
        ],
    }


def _artifact_summary(offset: int) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "entries": [
            _artifact_entry("embedding_output", offset),
            _artifact_entry("block_06", offset + 10),
        ],
        "global_shape": [2, 3],
        "image_ids": ["image-b", "image-a"],
        "patch_shape": [2, 4, 3],
        "tap_order": ["embedding_output", "block_06"],
        "validated": True,
    }


def _response_value(
    *,
    operation: str,
    runtime_id: str = "fake-runtime-v1",
    manifest_sha256: str = "a" * 64,
    candidate_freeze_sha256: str = "b" * 64,
    seed: int = 0,
) -> dict[str, Any]:
    inbound = _artifact_summary(20)
    outbound = _artifact_summary(40)
    if operation == "runtime_self_test":
        manual = {
            "crop_count": 0,
            "executed": True,
            "image_processor_construction_count": 0,
            "input_dtype": "uint8",
            "input_sha256": INPUT_SHA256,
            "input_shape": [2, 224, 224, 3],
            "input_unchanged": True,
            "normalized_sha256": OUTPUT_SHA256,
            "output_c_contiguous": True,
            "output_dtype": "float32",
            "output_finite": True,
            "output_shape": [2, 3, 224, 224],
            "pil_conversion_count": 0,
            "profile_id": PROFILE_ID,
            "resize_count": 0,
        }
        shapes = {
            "hidden_states": [[2, 201, 768] for _ in range(13)],
            "last_hidden": [2, 201, 768],
            "pooler": [2, 768],
        }
        random_model = {
            "checkpoint_weight_bytes": 0,
            "class": "DINOv3ViTModel",
            "cuda_available": False,
            "cuda_device_count": 0,
            "device": "cpu",
            "dtype": "float32",
            "evaluation_mode": True,
            "executed": True,
            "finite": True,
            "frozen": True,
            "hidden_state_count": 13,
            "hidden_state_shapes": [[2, 201, 768] for _ in range(13)],
            "last_hidden_sha256": "c" * 64,
            "last_hidden_shape": [2, 201, 768],
            "model_state_sha256_after": "d" * 64,
            "model_state_sha256_before": "d" * 64,
            "model_state_unchanged": True,
            "parameter_count": 85_660_416,
            "parameter_tensor_count": 211,
            "pooler_shape": [2, 768],
            "random_initialization_seed": seed,
            "repeat_exact": True,
            "rng": {
                "numpy_unchanged": True,
                "python_unchanged": True,
                "torch_unchanged": True,
            },
            "total_tokens": 201,
        }
        manual_hash: str | None = OUTPUT_SHA256
        markers = RUNTIME_MARKERS
    else:
        manual = {
            "executed": False,
            "normalized_sha256": None,
            "profile_id": PROFILE_ID,
        }
        shapes = {"hidden_states": [], "last_hidden": None, "pooler": None}
        random_model = {"executed": False, "shapes": []}
        manual_hash = None
        markers = ARTIFACT_MARKERS
    return {
        "artifact_validation_result": {"inbound": inbound, "outbound": outbound},
        "candidate_freeze_sha256": candidate_freeze_sha256,
        "failure_classification": None,
        "inbound_checksums": [entry["checksum"] for entry in inbound["entries"]],
        "manual_preprocessing": manual,
        "manual_preprocessing_hash": manual_hash,
        "markers": markers,
        "model_weight_byte_count": 0,
        "operation": operation,
        "outbound_checksums": [entry["checksum"] for entry in outbound["entries"]],
        "random_model": random_model,
        "random_model_shapes": shapes,
        "run_id": "t007a2-unit",
        "runtime_id": runtime_id,
        "runtime_manifest_sha256": manifest_sha256,
        "schema_id": RESPONSE_SCHEMA,
        "socket_attempt_count": 0,
        "status": "success",
        "torchvision_use_count": 0,
        "warnings": [],
    }


@dataclass(frozen=True)
class FakeBoundary:
    repository: Path
    profile_path: Path
    profile: RuntimeProfile
    runtime_root: Path
    manifest: RuntimeManifest
    worker_script: Path
    workspace: Path


def _prepare_fake_boundary(tmp_path: Path, repository_root: Path) -> FakeBoundary:
    repository = tmp_path / "repository"
    profile_path = repository / "configs" / "runtime_profiles" / "profile.json"
    worker_script = repository / "scripts" / "fake_worker.py"
    worker_script.parent.mkdir(parents=True)
    worker_script.write_text("# fake worker; subprocess is mocked by tests\n", encoding="utf-8")

    value = _profile_value(repository_root)
    metadata_payloads: dict[str, bytes] = {}
    for row in value["metadata_files"]:
        payload = f"metadata:{row['filename']}\n".encode()
        row["bytes"] = len(payload)
        row["sha256"] = _digest(payload)
        metadata_payloads[row["filename"]] = payload
    wheel_payloads: dict[str, bytes] = {}
    for row in value["wheel_files"]:
        payload = f"wheel:{row['filename']}\n".encode()
        row["bytes"] = len(payload)
        row["sha256"] = _digest(payload)
        wheel_payloads[row["filename"]] = payload
    _write_json(profile_path, value)
    profile = load_runtime_profile(profile_path.resolve())

    runtime_root = tmp_path / "runtimes" / profile.runtime_id
    (runtime_root / "metadata").mkdir(parents=True)
    (runtime_root / "wheelhouse").mkdir()
    (runtime_root / "venv" / "Scripts").mkdir(parents=True)
    for filename, payload in metadata_payloads.items():
        (runtime_root / "metadata" / filename).write_bytes(payload)
    for filename, payload in wheel_payloads.items():
        (runtime_root / "wheelhouse" / filename).write_bytes(payload)
    (runtime_root / "venv" / "Scripts" / "python.exe").write_bytes(b"fake-python\n")
    (runtime_root / "venv" / "pyvenv.cfg").write_text(
        "include-system-site-packages = true\n", encoding="utf-8"
    )

    dependency = value["dependency_validation"]
    manifest_value = {
        "candidate_freeze": copy.deepcopy(value["candidate"]["expected_freeze"]),
        "candidate_local_packages": copy.deepcopy(value["candidate"]["local_packages"]),
        "current_base_freeze_sha256": value["current_base"]["expected_freeze"]["sha256"],
        "dependency_validation": {
            "active_requirement_count": dependency["active_requirement_count"],
            "inherited_conflict_count": dependency["expected_inherited_conflict_count"],
            "inherited_conflict_sha256": dependency["expected_inherited_conflict_sha256"],
            "new_conflict_count": dependency["expected_new_conflict_count"],
            "unsatisfied_active_requirement_count": dependency[
                "expected_unsatisfied_active_requirement_count"
            ],
        },
        "forbidden_candidate_package_count": 0,
        "installed_sources": copy.deepcopy(value["installed_sources"]),
        "manual_preprocessing_profile_id": PROFILE_ID,
        "metadata_files": copy.deepcopy(value["metadata_files"]),
        "model_weight_present": False,
        "network_independent_verification_status": "VERIFIED",
        "provisioned_at_utc": "2026-07-20T00:00:00Z",
        "python_version": value["candidate"]["python"]["version"],
        "repository_profile_sha256": profile.source_sha256,
        "runtime_id": profile.runtime_id,
        "schema_id": MANIFEST_SCHEMA,
        "shared_torch": copy.deepcopy(value["current_base"]["torch"]),
        "torchvision": {"distribution_count": 0, "module_count": 0},
        "wheel_files": copy.deepcopy(value["wheel_files"]),
    }
    manifest_path = runtime_root / "runtime-manifest.json"
    _write_json(manifest_path, manifest_value)
    manifest = load_runtime_manifest(manifest_path.resolve())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inbound").mkdir()
    return FakeBoundary(
        repository=repository.resolve(),
        profile_path=profile_path.resolve(),
        profile=profile,
        runtime_root=runtime_root.resolve(),
        manifest=manifest,
        worker_script=worker_script.resolve(),
        workspace=workspace.resolve(),
    )


def _assert_classification(error: pytest.ExceptionInfo[RuntimeBoundaryError], expected: str) -> None:
    assert error.value.classification == expected


def test_valid_profile_load_round_trip_and_immutability(repository_root: Path) -> None:
    path = _profile_path(repository_root).resolve()
    profile = load_runtime_profile(path)
    assert profile.schema_id == PROFILE_SCHEMA
    assert profile.manual_preprocessing_profile_id == PROFILE_ID
    assert profile.to_dict() == json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(FrozenInstanceError):
        profile.runtime_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        profile.manual_preprocessing["profile_id"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize("field", ["schema_id", "candidate", "worker_protocol"])
def test_profile_missing_field_rejected(repository_root: Path, field: str) -> None:
    value = _profile_value(repository_root)
    del value[field]
    with pytest.raises(RuntimeBoundaryError) as error:
        RuntimeProfile.from_dict(value)
    _assert_classification(error, "schema_validation_failed")


def test_profile_unknown_field_rejected(repository_root: Path) -> None:
    value = _profile_value(repository_root)
    value["unexpected"] = True
    with pytest.raises(RuntimeBoundaryError, match="unknown fields"):
        RuntimeProfile.from_dict(value)


def test_profile_wrong_schema_rejected(repository_root: Path) -> None:
    value = _profile_value(repository_root)
    value["schema_id"] = "vea.wrong.v0"
    with pytest.raises(RuntimeBoundaryError, match="unexpected schema"):
        RuntimeProfile.from_dict(value)


@pytest.mark.parametrize("version", [">=4.56", "4.56.*", "4.56,5", "~=4.56"])
def test_profile_mutable_version_rejected(repository_root: Path, version: str) -> None:
    value = _profile_value(repository_root)
    value["candidate"]["local_packages"]["transformers"] = version
    with pytest.raises(RuntimeBoundaryError, match="exact package version"):
        RuntimeProfile.from_dict(value)


def test_profile_rejects_wrong_preprocessing_constants(repository_root: Path) -> None:
    value = _profile_value(repository_root)
    value["manual_preprocessing"]["image_mean"][0] = 0.5
    with pytest.raises(RuntimeBoundaryError, match="ImageNet constants"):
        RuntimeProfile.from_dict(value)


def test_profile_requires_forbidden_package_set(repository_root: Path) -> None:
    value = _profile_value(repository_root)
    value["forbidden_packages"].remove("torchvision")
    with pytest.raises(RuntimeBoundaryError, match="required packages"):
        RuntimeProfile.from_dict(value)


def test_profile_rejects_duplicate_json_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_id":"a","schema_id":"b"}', encoding="utf-8")
    with pytest.raises(RuntimeBoundaryError) as duplicate_error:
        load_runtime_profile(duplicate.resolve())
    _assert_classification(duplicate_error, "invalid_json")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(RuntimeBoundaryError) as nonfinite_error:
        load_runtime_profile(nonfinite.resolve())
    _assert_classification(nonfinite_error, "invalid_json")


def test_default_runtime_root_resolution(repository_root: Path, tmp_path: Path) -> None:
    profile = load_runtime_profile(_profile_path(repository_root).resolve())
    assert resolve_default_runtime_root(
        profile, environ={"LOCALAPPDATA": str(tmp_path.resolve())}
    ) == tmp_path.resolve() / "VEA" / "runtimes" / profile.runtime_id
    with pytest.raises(RuntimeBoundaryError) as missing:
        resolve_default_runtime_root(profile, environ={})
    _assert_classification(missing, "runtime_root_unavailable")
    with pytest.raises(RuntimeBoundaryError) as relative:
        resolve_default_runtime_root(profile, environ={"LOCALAPPDATA": "relative"})
    _assert_classification(relative, "unsafe_path")


@pytest.mark.parametrize(
    "unsafe",
    ["../escape", "a/../escape", "/absolute", "C:/absolute", "//server/share", "a\\b", "a//b", "./a"],
)
def test_request_rejects_unsafe_relative_paths(unsafe: str) -> None:
    value = _request_value("1" * 64)
    value["workspace_relative_inbound_artifact_path"] = unsafe
    with pytest.raises(RuntimeBoundaryError) as error:
        WorkerRequest.from_dict(value)
    _assert_classification(error, "unsafe_path")


def test_request_schema_round_trip_and_unknown_operation() -> None:
    value = _request_value("1" * 64)
    assert WorkerRequest.from_dict(value).to_dict() == value
    unknown = copy.deepcopy(value)
    unknown["operation"] = "download_checkpoint"
    with pytest.raises(RuntimeBoundaryError) as error:
        WorkerRequest.from_dict(unknown)
    _assert_classification(error, "unknown_worker_operation")


def test_request_requires_accepted_deterministic_seed_zero() -> None:
    value = _request_value("1" * 64)
    value["deterministic_seed"] = 1

    with pytest.raises(RuntimeBoundaryError) as error:
        WorkerRequest.from_dict(value)

    _assert_classification(error, "schema_validation_failed")


def test_request_rejects_missing_unknown_and_same_paths() -> None:
    missing = _request_value("1" * 64)
    del missing["run_id"]
    with pytest.raises(RuntimeBoundaryError):
        WorkerRequest.from_dict(missing)
    unknown = _request_value("1" * 64)
    unknown["secret"] = "value"
    with pytest.raises(RuntimeBoundaryError, match="unknown fields"):
        WorkerRequest.from_dict(unknown)
    same = _request_value("1" * 64)
    same["workspace_relative_outbound_artifact_path"] = "inbound"
    with pytest.raises(RuntimeBoundaryError) as error:
        WorkerRequest.from_dict(same)
    _assert_classification(error, "unsafe_path")


@pytest.mark.parametrize("operation", ["artifact_boundary_self_test", "runtime_self_test"])
def test_response_schema_round_trip(operation: str) -> None:
    value = _response_value(operation=operation)
    assert WorkerResponse.from_dict(value).to_dict() == value


def test_response_rejects_unknown_field_checksum_disagreement_and_markers() -> None:
    unknown = _response_value(operation="artifact_boundary_self_test")
    unknown["extra"] = 1
    with pytest.raises(RuntimeBoundaryError, match="unknown fields"):
        WorkerResponse.from_dict(unknown)
    checksum = _response_value(operation="artifact_boundary_self_test")
    checksum["inbound_checksums"][0] = "f" * 64
    with pytest.raises(RuntimeBoundaryError, match="checksum lists disagree"):
        WorkerResponse.from_dict(checksum)
    markers = _response_value(operation="runtime_self_test")
    markers["markers"] = list(reversed(markers["markers"]))
    with pytest.raises(RuntimeBoundaryError, match="marker order"):
        WorkerResponse.from_dict(markers)


def test_response_rejects_unknown_operation_and_secret_like_warning() -> None:
    operation = _response_value(operation="artifact_boundary_self_test")
    operation["operation"] = "checkpoint_download"
    with pytest.raises(RuntimeBoundaryError) as error:
        WorkerResponse.from_dict(operation)
    _assert_classification(error, "unknown_worker_operation")
    secret = _response_value(operation="artifact_boundary_self_test")
    secret["status"] = "error"
    secret["failure_classification"] = "worker_failure"
    secret["warnings"] = ["token=hf_abcdefghijklmnopqrstuvwxyz"]
    with pytest.raises(RuntimeBoundaryError) as secret_error:
        WorkerResponse.from_dict(secret)
    _assert_classification(secret_error, "sensitive_data_rejected")


def test_runtime_manifest_and_layout_validate(tmp_path: Path, repository_root: Path) -> None:
    fake = _prepare_fake_boundary(tmp_path, repository_root)
    validate_runtime_manifest(fake.manifest, fake.profile)
    assert (
        validate_runtime_root(
            fake.runtime_root, fake.profile, repository_root=fake.repository
        ).source_sha256
        == fake.manifest.source_sha256
    )


def test_invalid_manifest_and_profile_mismatch_rejected(tmp_path: Path, repository_root: Path) -> None:
    fake = _prepare_fake_boundary(tmp_path, repository_root)
    value = fake.manifest.to_dict()
    value["network_independent_verification_status"] = "PENDING"
    with pytest.raises(RuntimeBoundaryError):
        RuntimeManifest.from_dict(value)
    mismatch = fake.manifest.to_dict()
    mismatch["repository_profile_sha256"] = "f" * 64
    parsed = RuntimeManifest.from_dict(mismatch)
    with pytest.raises(RuntimeBoundaryError) as error:
        validate_runtime_manifest(parsed, fake.profile)
    _assert_classification(error, "runtime_identity_mismatch")


@pytest.mark.parametrize("kind", ["metadata", "wheel"])
def test_runtime_rejects_file_size_or_hash_mismatch(
    kind: str, tmp_path: Path, repository_root: Path
) -> None:
    fake = _prepare_fake_boundary(tmp_path, repository_root)
    if kind == "metadata":
        target = fake.runtime_root / "metadata" / fake.profile.metadata[0].filename
    else:
        target = fake.runtime_root / "wheelhouse" / fake.profile.wheels[0].filename
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(RuntimeBoundaryError) as error:
        validate_runtime_root(
            fake.runtime_root, fake.profile, repository_root=fake.repository
        )
    _assert_classification(error, "runtime_identity_mismatch")


def test_partial_runtime_root_fails_closed(tmp_path: Path, repository_root: Path) -> None:
    fake = _prepare_fake_boundary(tmp_path, repository_root)
    (fake.runtime_root / "runtime-manifest.json").unlink()
    with pytest.raises(RuntimeBoundaryError) as error:
        validate_runtime_root(
            fake.runtime_root, fake.profile, repository_root=fake.repository
        )
    _assert_classification(error, "runtime_layout_mismatch")


def test_repository_inside_runtime_root_rejected(tmp_path: Path, repository_root: Path) -> None:
    profile = load_runtime_profile(_profile_path(repository_root).resolve())
    repository = tmp_path / "repository"
    runtime = repository / profile.runtime_id
    runtime.mkdir(parents=True)
    with pytest.raises(RuntimeBoundaryError) as error:
        validate_runtime_root(runtime.resolve(), profile, repository_root=repository.resolve())
    _assert_classification(error, "unsafe_runtime_root")


def test_workspace_symlink_escape_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    link = workspace / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(RuntimeBoundaryError) as error:
        resolve_workspace_relative_path(
            workspace.resolve(), "escape/file.json", must_exist=False
        )
    _assert_classification(error, "workspace_escape")


def test_request_write_is_canonical_exclusive_and_response_read_is_strict(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    request = WorkerRequest.from_dict(_request_value("1" * 64))
    request_path = workspace / "request.json"
    write_worker_request(request_path.resolve(), request, workspace_root=workspace.resolve())
    assert request_path.read_bytes() == _canonical_bytes(request.to_dict())
    with pytest.raises(RuntimeBoundaryError) as error:
        write_worker_request(request_path.resolve(), request, workspace_root=workspace.resolve())
    _assert_classification(error, "unsafe_path")
    response_path = workspace / "response.json"
    response_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeBoundaryError) as malformed:
        read_worker_response(response_path.resolve(), workspace_root=workspace.resolve())
    _assert_classification(malformed, "invalid_json")


def test_environment_is_rebuilt_offline_without_secrets(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    workspace = tmp_path / "workspace"
    repository.mkdir()
    workspace.mkdir()
    runtime_python = tmp_path / "runtime" / "Scripts" / "python.exe"
    environ = {
        "SystemRoot": str(tmp_path / "Windows"),
        "PATH": "untrusted",
        "PYTHONPATH": "untrusted",
        "HF_TOKEN": "hf_abcdefghijklmnopqrstuvwxyz",
        "API_KEY": "secret",
        "USERNAME": "private-user",
    }
    result = build_sanitized_worker_environment(
        runtime_python=runtime_python.resolve(),
        repository_root=repository.resolve(),
        workspace_root=workspace.resolve(),
        environ=environ,
    )
    assert result["HF_HUB_OFFLINE"] == "1"
    assert result["TRANSFORMERS_OFFLINE"] == "1"
    assert result["PYTHONNOUSERSITE"] == "1"
    assert "HF_TOKEN" not in result and "API_KEY" not in result
    assert "private-user" not in json.dumps(result)
    assert "untrusted" not in result["PATH"]


def test_runtime_boundary_import_isolation(repository_root: Path, tmp_path: Path) -> None:
    code = (
        "import json,sys;"
        "import vision_encoder_alignment.runtime_boundary;"
        "heavy={'numpy','torch','transformers','huggingface_hub','safetensors','PIL'};"
        "print(json.dumps(sorted(heavy.intersection(sys.modules))))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env=environment,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_verify_runtime_is_read_only(tmp_path: Path, repository_root: Path) -> None:
    fake = _prepare_fake_boundary(tmp_path, repository_root)
    before = {
        path.relative_to(fake.runtime_root).as_posix(): path.read_bytes()
        for path in fake.runtime_root.rglob("*")
        if path.is_file()
    }
    validate_runtime_root(fake.runtime_root, fake.profile, repository_root=fake.repository)
    after = {
        path.relative_to(fake.runtime_root).as_posix(): path.read_bytes()
        for path in fake.runtime_root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_provision_plan_is_side_effect_free_and_does_not_execute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repository_root: Path,
) -> None:
    provisioner = _load_provisioner(repository_root)
    runtime_root = tmp_path / "planned" / provisioner.RUNTIME_ID
    before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("planning attempted execution or network access")

    monkeypatch.setattr(provisioner, "run_command", forbidden)
    monkeypatch.setattr(provisioner, "_download_metadata", forbidden)
    plan = provisioner.plan_provision(
        _profile_path(repository_root).resolve(),
        runtime_root.resolve(),
        repository_root=repository_root.resolve(),
        environment={},
    )
    after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    assert plan.runtime_root == runtime_root.resolve()
    assert plan.staging_parent == runtime_root.resolve().parent
    assert plan.staging_prefix == f".{provisioner.RUNTIME_ID}.staging-"
    assert after == before
    assert not runtime_root.exists()


def test_provision_plan_rejects_wrong_runtime_basename(
    tmp_path: Path, repository_root: Path
) -> None:
    provisioner = _load_provisioner(repository_root)
    wrong_root = (tmp_path / "not-the-profile-runtime-id").resolve()

    with pytest.raises(provisioner.ProvisioningError) as error:
        provisioner.plan_provision(
            _profile_path(repository_root).resolve(),
            wrong_root,
            repository_root=repository_root.resolve(),
            environment={},
        )

    assert error.value.classification == "INVALID_RUNTIME_ROOT"
    assert not wrong_root.exists()


def test_provisioner_rejects_repository_nested_inside_runtime_root(
    tmp_path: Path, repository_root: Path
) -> None:
    provisioner = _load_provisioner(repository_root)
    runtime_root = tmp_path / provisioner.RUNTIME_ID
    nested_repository = runtime_root / "repository"
    nested_repository.mkdir(parents=True)

    with pytest.raises(provisioner.ProvisioningError) as error:
        provisioner._validate_runtime_root_path(runtime_root, nested_repository)

    assert error.value.classification == "INVALID_RUNTIME_ROOT"


def test_boundary_rejects_root_overlap_in_both_directions(tmp_path: Path) -> None:
    outer = tmp_path.resolve()
    inner = (tmp_path / "inner").resolve()
    inner.mkdir()

    for left, right in ((outer, inner), (inner, outer)):
        with pytest.raises(RuntimeBoundaryError) as error:
            boundary._reject_overlapping_roots(left, right)
        _assert_classification(error, "unsafe_runtime_root")


def test_provision_plan_only_allows_owned_staging_for_internal_verification(
    tmp_path: Path, repository_root: Path
) -> None:
    provisioner = _load_provisioner(repository_root)
    recognized = (
        tmp_path
        / f".{provisioner.RUNTIME_ID}.staging-{'a' * 32}"
    ).resolve()

    with pytest.raises(provisioner.ProvisioningError):
        provisioner.plan_provision(
            _profile_path(repository_root).resolve(),
            recognized,
            repository_root=repository_root.resolve(),
        )
    internal_plan = provisioner.plan_provision(
        _profile_path(repository_root).resolve(),
        recognized,
        repository_root=repository_root.resolve(),
        _allow_owned_staging=True,
    )

    assert internal_plan.runtime_root == recognized
    assert not recognized.exists()


def test_fresh_provision_uses_short_owned_hub_cache_and_internal_staging_verify(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repository_root: Path,
) -> None:
    provisioner = _load_provisioner(repository_root)
    runtime_root = (tmp_path / provisioner.RUNTIME_ID).resolve()
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        provisioner,
        "_validate_current_base",
        lambda *_args, **_kwargs: ({}, ()),
    )
    monkeypatch.setattr(provisioner, "_require_free_space", lambda *_args: None)

    def fake_wheels(wheelhouse: Path, pip_cache: Path, **_kwargs: Any) -> None:
        wheelhouse.mkdir()
        pip_cache.mkdir(parents=True)

    def fake_metadata(metadata: Path, hub_cache: Path) -> None:
        observed["hub_cache"] = hub_cache
        observed["hub_cache_existed"] = hub_cache.is_dir()
        observed["hub_cache_outside_staging"] = not hub_cache.is_relative_to(
            runtime_root.parent
        )
        metadata.mkdir()

    def fake_venv(staging: Path, **_kwargs: Any) -> None:
        (staging / "venv").mkdir()

    def fake_verify(*_args: Any, **kwargs: Any) -> None:
        observed["allow_owned_staging"] = kwargs.get("_allow_owned_staging")

    monkeypatch.setattr(provisioner, "_download_wheels", fake_wheels)
    monkeypatch.setattr(provisioner, "_download_metadata", fake_metadata)
    monkeypatch.setattr(provisioner, "_create_and_install_venv", fake_venv)
    monkeypatch.setattr(provisioner, "_validate_candidate", lambda *_a, **_k: {})
    monkeypatch.setattr(provisioner, "_manifest_value", lambda *_a, **_k: {})
    monkeypatch.setattr(
        provisioner,
        "_write_canonical_json",
        lambda path, _value: path.write_bytes(b"{}\n"),
    )
    monkeypatch.setattr(provisioner, "verify_runtime", fake_verify)

    marker = provisioner.provision_runtime(
        _profile_path(repository_root).resolve(),
        runtime_root,
        repository_root=repository_root.resolve(),
    )

    assert marker == "DINOV3_PERSISTENT_RUNTIME_PROVISIONED"
    assert observed["hub_cache_existed"] is True
    assert observed["hub_cache_outside_staging"] is True
    assert observed["allow_owned_staging"] is True
    assert not observed["hub_cache"].exists()
    assert runtime_root.is_dir()


def test_candidate_venv_is_built_at_short_sibling_then_moved(
    tmp_path: Path, repository_root: Path
) -> None:
    provisioner = _load_provisioner(repository_root)
    staging = tmp_path / f".{provisioner.RUNTIME_ID}.staging-{'b' * 32}"
    (staging / "wheelhouse").mkdir(parents=True)
    build_paths: list[Path] = []

    def fake_runner(args: Any, **_kwargs: Any) -> Any:
        command = tuple(Path(item) if isinstance(item, Path) else item for item in args)
        if "venv" in command:
            build_venv = Path(command[-1])
            build_paths.append(build_venv)
            python = build_venv / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"fake-python")
        else:
            assert "install" in command
            assert Path(command[0]).is_relative_to(build_paths[0])
        return provisioner.CommandResult(tuple(map(str, args)), 0, "", "")

    provisioner._create_and_install_venv(staging, runner=fake_runner)

    assert len(build_paths) == 1
    assert not build_paths[0].exists()
    assert (staging / "venv" / "Scripts" / "python.exe").read_bytes() == b"fake-python"
    assert not list(staging.parent.glob(".vea-venv-*"))


def test_metadata_response_writer_enforces_two_mib_cap(
    repository_root: Path,
) -> None:
    provisioner = _load_provisioner(repository_root)
    stream = io.BytesIO()
    writer = provisioner._BoundedResponseWriter(
        stream, provisioner.MAX_METADATA_RESPONSE_BYTES
    )
    payload = b"x" * provisioner.MAX_METADATA_RESPONSE_BYTES

    assert writer.write(payload) == len(payload)
    with pytest.raises(provisioner.ProvisioningError) as error:
        writer.write(b"x")

    assert error.value.classification == "METADATA_RESPONSE_TOO_LARGE"
    assert stream.getvalue() == payload


@pytest.mark.parametrize("filename", ["token", "stored_tokens", "credentials.json"])
def test_runtime_scan_rejects_exact_credential_filenames(
    tmp_path: Path, repository_root: Path, filename: str
) -> None:
    provisioner = _load_provisioner(repository_root)
    runtime_root = tmp_path / provisioner.RUNTIME_ID
    credential = runtime_root / "venv" / filename
    credential.parent.mkdir(parents=True)
    credential.write_bytes(b"redacted-test-value")

    with pytest.raises(provisioner.ProvisioningError) as error:
        provisioner._scan_for_forbidden_runtime_files(runtime_root)

    assert error.value.classification == "RUNTIME_IDENTITY_MISMATCH"


def test_provisioner_subprocess_environment_filters_common_secret_names(
    monkeypatch: pytest.MonkeyPatch, repository_root: Path
) -> None:
    provisioner = _load_provisioner(repository_root)
    secret_names = (
        "HF_TOKEN",
        "GITHUB_PAT",
        "SSH_PRIVATE_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "SERVICE_PASSWORD",
        "SESSION_COOKIE",
    )
    for name in secret_names:
        monkeypatch.setenv(name, "redacted-test-value")

    environment = provisioner._normalized_environment(offline=True)

    assert all(name not in environment for name in secret_names)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_provision_rejects_partial_existing_root_without_runner_or_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repository_root: Path,
) -> None:
    provisioner = _load_provisioner(repository_root)
    runtime_root = tmp_path / provisioner.RUNTIME_ID
    runtime_root.mkdir()
    marker = runtime_root / "partial.txt"
    marker.write_bytes(b"user-owned partial runtime\n")
    before = marker.read_bytes()
    calls = {"runner": 0, "network": 0}

    def forbidden_runner(*_args: Any, **_kwargs: Any) -> Any:
        calls["runner"] += 1
        raise AssertionError("runner must not execute for a partial-root collision")

    def forbidden_network(*_args: Any, **_kwargs: Any) -> Any:
        calls["network"] += 1
        raise AssertionError("network must not execute for a partial-root collision")

    monkeypatch.setattr(provisioner, "_download_metadata", forbidden_network)
    with pytest.raises(provisioner.RuntimeCollisionError) as error:
        provisioner.provision_runtime(
            _profile_path(repository_root).resolve(),
            runtime_root.resolve(),
            repository_root=repository_root.resolve(),
            runner=forbidden_runner,
        )
    assert error.value.classification == "DINOV3_PERSISTENT_RUNTIME_COLLISION"
    assert calls == {"runner": 0, "network": 0}
    assert marker.read_bytes() == before
    assert {path.name for path in runtime_root.iterdir()} == {"partial.txt"}


def test_provisioner_verify_path_is_read_only_and_mutation_guarded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repository_root: Path,
) -> None:
    provisioner = _load_provisioner(repository_root)
    fake = _prepare_fake_boundary(tmp_path, repository_root)
    manifest_value = json.loads(
        (fake.runtime_root / "runtime-manifest.json").read_text(encoding="utf-8")
    )
    candidate = {
        "candidate_freeze_sha256": fake.profile.expected_candidate_freeze_sha256,
        "inherited_conflict_sha256": fake.profile.dependency_validation[
            "expected_inherited_conflict_sha256"
        ],
    }
    calls = {"runner": 0, "mutator": 0}

    def forbidden_runner(*_args: Any, **_kwargs: Any) -> Any:
        calls["runner"] += 1
        raise AssertionError("verify attempted an unplanned process")

    def forbidden_mutator(*_args: Any, **_kwargs: Any) -> Any:
        calls["mutator"] += 1
        raise AssertionError("verify attempted a provisioning mutation")

    monkeypatch.setattr(provisioner, "_validate_exact_directory", lambda *_a, **_k: None)
    monkeypatch.setattr(
        provisioner,
        "_validate_current_base",
        lambda *_a, **_k: (copy.deepcopy(manifest_value["shared_torch"]), ()),
    )
    monkeypatch.setattr(
        provisioner,
        "_validate_candidate",
        lambda *_a, **_k: copy.deepcopy(candidate),
    )
    monkeypatch.setattr(
        provisioner,
        "_manifest_value",
        lambda *_a, **_k: copy.deepcopy(manifest_value),
    )
    for name in (
        "_write_canonical_json",
        "_create_parent_directories",
        "_download_wheels",
        "_download_metadata",
        "_create_and_install_venv",
    ):
        monkeypatch.setattr(provisioner, name, forbidden_mutator)

    before = {
        path.relative_to(fake.runtime_root).as_posix(): path.read_bytes()
        for path in fake.runtime_root.rglob("*")
        if path.is_file()
    }
    result = provisioner.verify_runtime(
        _profile_path(repository_root).resolve(),
        fake.runtime_root,
        repository_root=repository_root.resolve(),
        runner=forbidden_runner,
    )
    after = {
        path.relative_to(fake.runtime_root).as_posix(): path.read_bytes()
        for path in fake.runtime_root.rglob("*")
        if path.is_file()
    }
    assert result.runtime_id == fake.profile.runtime_id
    assert result.runtime_manifest_sha256 == fake.manifest.source_sha256
    assert calls == {"runner": 0, "mutator": 0}
    assert after == before


def test_invoke_plans_argument_list_and_validates_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repository_root: Path
) -> None:
    fake = _prepare_fake_boundary(tmp_path, repository_root)
    request = WorkerRequest.from_dict(_request_value(fake.profile.source_sha256))
    captured: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        response = _response_value(
            operation=request.operation,
            runtime_id=fake.profile.runtime_id,
            manifest_sha256=fake.manifest.source_sha256,
            candidate_freeze_sha256=fake.profile.expected_candidate_freeze_sha256,
            seed=request.deterministic_seed,
        )
        _write_json(fake.workspace / "worker-response.json", response)
        return subprocess.CompletedProcess(command, 0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(boundary.subprocess, "run", run)
    response = invoke_isolated_worker(
        runtime_root=fake.runtime_root,
        repository_root=fake.repository,
        profile_path=fake.profile_path,
        worker_script=fake.worker_script,
        workspace_root=fake.workspace,
        request=request,
        timeout_seconds=10,
        environ={"HF_TOKEN": "hf_abcdefghijklmnopqrstuvwxyz"},
    )
    assert response.status == "success"
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["timeout"] == 10.0
    assert "--workspace" in captured["command"]
    assert "--workspace-root" not in captured["command"]
    assert captured["command"][-4:] == [
        "--request",
        "worker-request.json",
        "--response",
        "worker-response.json",
    ]


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "malformed"])
def test_invoke_wraps_process_and_response_failures(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repository_root: Path,
) -> None:
    fake = _prepare_fake_boundary(tmp_path, repository_root)
    request = WorkerRequest.from_dict(_request_value(fake.profile.source_sha256))

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if failure == "malformed":
            (fake.workspace / "worker-response.json").write_text("{", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        return subprocess.CompletedProcess(command, 7, stdout=b"out", stderr=b"err")

    monkeypatch.setattr(boundary.subprocess, "run", run)
    expected = {
        "nonzero": "worker_nonzero_exit",
        "timeout": "worker_timeout",
        "malformed": "worker_response_invalid",
    }[failure]
    with pytest.raises(RuntimeBoundaryError) as error:
        invoke_isolated_worker(
            runtime_root=fake.runtime_root,
            repository_root=fake.repository,
            profile_path=fake.profile_path,
            worker_script=fake.worker_script,
            workspace_root=fake.workspace,
            request=request,
            timeout_seconds=1,
            environ={},
        )
    _assert_classification(error, expected)
    assert error.value.__cause__ is not None
    if failure == "nonzero":
        assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
