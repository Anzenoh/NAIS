"""Integration tests for the DINOv3 JSON/process/artifact boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vision_encoder_alignment.runtime_boundary import (
    RuntimeBoundaryError,
    WorkerResponse,
)


EXPECTED_INPUT_SHA256 = (
    "b0fa2f6e3b2fbca50cde7e4a2dc3a65a0f788a184aec382365e520d54b468411"
)
EXPECTED_NORMALIZED_SHA256 = (
    "c0ebe2af6db2e61d052dc58272379beef749a4c5392d4e2204a11862cf172526"
)
EXPECTED_CANDIDATE_FREEZE_SHA256 = (
    "995687ed934c0a94b52a040af048f87bc8de4e5089475f70565ad37f1f33ecd9"
)
EXPECTED_HEAD = "e518a51ba83f70b42611cc4386a9f55dc9e97de3"
PROFILE_ID = "dinov3-manual-fp32-imagenet-v1"


# This bootstrap exists only for fake-worker integration tests.  It executes the
# unmodified production worker as __main__ in a separate process while making
# that child process's import/distribution view match the declared fake runtime.
# The production worker reads no test control and retains every guard.
_TEST_ONLY_WORKER_BOOTSTRAP = r"""
import importlib.metadata
import importlib.util
import runpy
import sys

worker_path, view, *worker_arguments = sys.argv[1:]
if view not in {"isolated", "host"}:
    raise RuntimeError("unsupported test-only worker view")

if view == "isolated":
    real_distribution = importlib.metadata.distribution
    real_find_spec = importlib.util.find_spec

    def isolated_distribution(name):
        normalized = str(name).replace("_", "-").lower()
        if normalized == "torchvision":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_distribution(name)

    def isolated_find_spec(name, package=None):
        if str(name).split(".", 1)[0] == "torchvision":
            return None
        return real_find_spec(name, package)

    importlib.metadata.distribution = isolated_distribution
    importlib.util.find_spec = isolated_find_spec
    for module_name in tuple(sys.modules):
        if module_name == "torchvision" or module_name.startswith("torchvision."):
            del sys.modules[module_name]

    try:
        importlib.metadata.distribution("torchvision")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("test-only distribution isolation failed")
    if importlib.util.find_spec("torchvision") is not None:
        raise RuntimeError("test-only module-spec isolation failed")
    if any(
        name == "torchvision" or name.startswith("torchvision.")
        for name in sys.modules
    ):
        raise RuntimeError("test-only imported-module isolation failed")

sys.argv = [worker_path, *worker_arguments]
runpy.run_path(worker_path, run_name="__main__")
"""


def _load_script(repository_root: Path, name: str) -> Any:
    path = repository_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{name[:-3]}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker_module(repository_root: Path) -> Any:
    return _load_script(repository_root, "dinov3_isolated_worker.py")


@pytest.fixture
def orchestrator_module(repository_root: Path) -> Any:
    return _load_script(repository_root, "run_dinov3_isolated_smoke.py")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _manifest_value(runtime_id: str, profile_sha256: str) -> dict[str, Any]:
    return {
        "candidate_freeze": {
            "bytes": 5206,
            "line_count": 275,
            "sha256": EXPECTED_CANDIDATE_FREEZE_SHA256,
        },
        "candidate_local_packages": {},
        "current_base_freeze_sha256": "4" * 64,
        "dependency_validation": {},
        "forbidden_candidate_package_count": 0,
        "installed_sources": [],
        "manual_preprocessing_profile_id": PROFILE_ID,
        "metadata_files": [],
        "model_weight_present": False,
        "network_independent_verification_status": "VERIFIED",
        "provisioned_at_utc": "2026-07-20T00:00:00Z",
        "python_version": "3.12.0",
        "repository_profile_sha256": profile_sha256,
        "runtime_id": runtime_id,
        "schema_id": "vea.dinov3_persistent_runtime.v0",
        "shared_torch": {},
        "torchvision": {"distribution_count": 0, "module_count": 0},
        "wheel_files": [],
    }


def _prepare_fake_runtime(
    tmp_path: Path, repository_root: Path
) -> tuple[Path, Path, str, str]:
    source_profile = (
        repository_root
        / "configs"
        / "runtime_profiles"
        / "dinov3_isolated_manual_v0.json"
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(source_profile.read_bytes())
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_sha256 = _sha256_file(profile_path)
    runtime_id = profile["runtime_id"]
    runtime_root = tmp_path / runtime_id
    metadata = runtime_root / "metadata"
    metadata.mkdir(parents=True)
    _write_json(metadata / "config.json", {})
    _write_json(metadata / "preprocessor_config.json", {})
    _write_json(
        runtime_root / "runtime-manifest.json",
        _manifest_value(runtime_id, profile_sha256),
    )
    return profile_path, runtime_root, runtime_id, profile_sha256


def _request_value(profile_sha256: str, *, inbound: str = "inbound") -> dict[str, Any]:
    return {
        "deterministic_seed": 0,
        "expected_preprocessing_profile_id": PROFILE_ID,
        "operation": "artifact_boundary_self_test",
        "repository_head": EXPECTED_HEAD,
        "run_id": "t007a2-boundary-test",
        "runtime_profile_sha256": profile_sha256,
        "schema_id": "vea.dinov3_isolated_worker_request.v0",
        "workspace_relative_inbound_artifact_path": inbound,
        "workspace_relative_outbound_artifact_path": "outbound",
    }


def _worker_environment(repository_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repository_root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return environment


def _run_fake_worker_process(
    *,
    repository_root: Path,
    worker_script: Path,
    profile_path: Path,
    runtime_root: Path,
    workspace_root: Path,
    request_path: Path,
    response_path: Path,
    inbound: Path,
    outbound: Path,
    timeout_seconds: float,
    isolate_torchvision: bool = True,
    additional_python_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the real worker CLI under a test-owned child import view."""

    worker_arguments = [
        "--profile",
        str(profile_path),
        "--runtime-root",
        str(runtime_root),
        "--workspace-root",
        str(workspace_root),
        "--request",
        str(request_path),
        "--response",
        str(response_path),
        "--inbound-artifact",
        str(inbound),
        "--outbound-artifact",
        str(outbound),
    ]
    command = [
        sys.executable,
        "-B",
        "-c",
        _TEST_ONLY_WORKER_BOOTSTRAP,
        str(worker_script),
        "isolated" if isolate_torchvision else "host",
        *worker_arguments,
    ]
    environment = _worker_environment(repository_root)
    if additional_python_path is not None:
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(additional_python_path), environment["PYTHONPATH"])
        )
    return subprocess.run(
        command,
        cwd=workspace_root,
        env=environment,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _invoke_fake_worker(
    *,
    repository_root: Path,
    worker_script: Path,
    profile_path: Path,
    runtime_root: Path,
    workspace_root: Path,
    request: Any,
    timeout_seconds: float,
) -> dict[str, Any]:
    request_path = workspace_root / "worker-request.json"
    response_path = workspace_root / "worker-response.json"
    _write_json(request_path, request.to_dict())
    inbound = workspace_root / request.workspace_relative_inbound_artifact_path
    outbound = workspace_root / request.workspace_relative_outbound_artifact_path
    result = _run_fake_worker_process(
        repository_root=repository_root,
        worker_script=worker_script,
        profile_path=profile_path,
        runtime_root=runtime_root,
        workspace_root=workspace_root,
        request_path=request_path,
        response_path=response_path,
        inbound=inbound,
        outbound=outbound,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeBoundaryError(
            "worker_nonzero_exit", f"fake worker exited with {result.returncode}"
        )
    return json.loads(response_path.read_text(encoding="utf-8"))


def test_worker_module_import_is_light(repository_root: Path, tmp_path: Path) -> None:
    worker = repository_root / "scripts" / "dinov3_isolated_worker.py"
    code = (
        "import importlib.util,json,sys;"
        f"s=importlib.util.spec_from_file_location('worker',{str(worker)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.split('.')[0] in {'numpy','torch','transformers','safetensors','PIL'})))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize(
    "payload",
    [b'{"operation":"a","operation":"b"}\n', b'{"value":NaN}\n'],
)
def test_worker_strict_json_rejects_duplicates_and_nonfinite_values(
    worker_module: Any, tmp_path: Path, payload: bytes
) -> None:
    document = tmp_path / "strict.json"
    document.write_bytes(payload)

    with pytest.raises(worker_module.WorkerValidationError):
        worker_module._read_json(document)


def test_exact_t007a1r_images_and_manual_hashes(worker_module: Any) -> None:
    images = worker_module.generate_deterministic_images()
    before = images.copy()
    normalized = worker_module.manual_preprocess(images)
    assert images.shape == (2, 224, 224, 3)
    assert images.dtype == np.uint8
    assert hashlib.sha256(images.tobytes()).hexdigest() == EXPECTED_INPUT_SHA256
    np.testing.assert_array_equal(images, before)
    assert normalized.shape == (2, 3, 224, 224)
    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous
    assert np.isfinite(normalized).all()
    assert hashlib.sha256(normalized.tobytes()).hexdigest() == EXPECTED_NORMALIZED_SHA256


def test_exact_synthetic_artifact_values(
    orchestrator_module: Any, tmp_path: Path
) -> None:
    result = orchestrator_module._write_inbound_artifacts(
        tmp_path / "inbound", "t007a2-boundary-test"
    )
    assert result["image_ids"] == ["image-b", "image-a"]
    assert result["tap_order"] == ["embedding_output", "block_06"]
    assert result["patch_shape"] == [2, 4, 3]
    assert result["global_shape"] == [2, 3]
    assert [entry["checksum"] for entry in result["entries"]] == [
        "31c2f1c7aab1bfddd9aa025a406c8425418155ff7b57339365b5139fe1ea69df",
        "ac88cb5d2a8a8a28779ec6444270db0e8352953ff2edd8a7d634d15c60268865",
    ]


def test_fake_runtime_worker_subprocess_round_trip(
    repository_root: Path,
    tmp_path: Path,
    orchestrator_module: Any,
) -> None:
    profile_path, runtime_root, runtime_id, profile_sha256 = _prepare_fake_runtime(
        tmp_path, repository_root
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inbound = orchestrator_module._write_inbound_artifacts(
        workspace / "inbound", "t007a2-boundary-test"
    )
    request_path = workspace / "worker-request.json"
    response_path = workspace / "worker-response.json"
    _write_json(request_path, _request_value(profile_sha256))
    worker = repository_root / "scripts" / "dinov3_isolated_worker.py"
    result = _run_fake_worker_process(
        repository_root=repository_root,
        worker_script=worker,
        profile_path=profile_path,
        runtime_root=runtime_root,
        workspace_root=workspace,
        request_path=request_path,
        response_path=response_path,
        inbound=workspace / "inbound",
        outbound=workspace / "outbound",
        timeout_seconds=60,
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(response_path.read_text(encoding="utf-8"))
    parsed_response = WorkerResponse.from_dict(response)
    assert response["status"] == "success"
    assert parsed_response.status == "success"
    assert response["runtime_id"] == runtime_id
    assert response["socket_attempt_count"] == 0
    assert response["model_weight_byte_count"] == 0
    assert response["torchvision_use_count"] == 0
    assert response["inbound_checksums"] == [
        entry["checksum"] for entry in inbound["entries"]
    ]
    outbound = orchestrator_module._validate_outbound_artifacts(
        workspace / "outbound", "t007a2-boundary-test"
    )
    assert response["outbound_checksums"] == [
        entry["checksum"] for entry in outbound["entries"]
    ]


def test_worker_rejects_workspace_path_escape(
    repository_root: Path,
    tmp_path: Path,
    orchestrator_module: Any,
) -> None:
    profile_path, runtime_root, _, profile_sha256 = _prepare_fake_runtime(
        tmp_path, repository_root
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orchestrator_module._write_inbound_artifacts(
        workspace / "inbound", "t007a2-boundary-test"
    )
    request_path = workspace / "worker-request.json"
    response_path = workspace / "worker-response.json"
    _write_json(request_path, _request_value(profile_sha256, inbound="../escape"))
    result = _run_fake_worker_process(
        repository_root=repository_root,
        worker_script=repository_root / "scripts" / "dinov3_isolated_worker.py",
        profile_path=profile_path,
        runtime_root=runtime_root,
        workspace_root=workspace,
        request_path=request_path,
        response_path=response_path,
        inbound=workspace / "inbound",
        outbound=workspace / "outbound",
        timeout_seconds=30,
    )
    assert result.returncode != 0
    assert "WorkerValidationError" in result.stderr
    assert not response_path.exists()
    assert not (workspace / "outbound").exists()
    assert json.loads(request_path.read_text(encoding="utf-8"))[
        "workspace_relative_inbound_artifact_path"
    ] == "../escape"


def test_worker_rejects_deterministic_torchvision_present_child_view(
    repository_root: Path,
    tmp_path: Path,
    orchestrator_module: Any,
) -> None:
    profile_path, runtime_root, _, profile_sha256 = _prepare_fake_runtime(
        tmp_path, repository_root
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orchestrator_module._write_inbound_artifacts(
        workspace / "inbound", "t007a2-boundary-test"
    )
    request_path = workspace / "worker-request.json"
    response_path = workspace / "worker-response.json"
    _write_json(request_path, _request_value(profile_sha256))

    contaminated_path = tmp_path / "test-only-contamination"
    package_path = contaminated_path / "torchvision"
    distribution_path = contaminated_path / "torchvision-0.0.dist-info"
    package_path.mkdir(parents=True)
    distribution_path.mkdir()
    (package_path / "__init__.py").write_text(
        "raise RuntimeError('test-only torchvision must never be imported')\n",
        encoding="utf-8",
        newline="\n",
    )
    (distribution_path / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torchvision\nVersion: 0.0\n",
        encoding="utf-8",
        newline="\n",
    )

    result = _run_fake_worker_process(
        repository_root=repository_root,
        worker_script=repository_root / "scripts" / "dinov3_isolated_worker.py",
        profile_path=profile_path,
        runtime_root=runtime_root,
        workspace_root=workspace,
        request_path=request_path,
        response_path=response_path,
        inbound=workspace / "inbound",
        outbound=workspace / "outbound",
        timeout_seconds=30,
        isolate_torchvision=False,
        additional_python_path=contaminated_path,
    )
    assert result.returncode != 0
    assert "WorkerValidationError" in result.stderr
    assert not response_path.exists()
    assert not (workspace / "outbound").exists()


def test_orchestrator_repeat_determinism_and_cleanup(
    repository_root: Path,
    tmp_path: Path,
    orchestrator_module: Any,
) -> None:
    profile_path, runtime_root, runtime_id, profile_sha256 = _prepare_fake_runtime(
        tmp_path, repository_root
    )
    fake_profile = SimpleNamespace(
        runtime_id=runtime_id,
        source_sha256=profile_sha256,
        manual_preprocessing_profile_id=PROFILE_ID,
    )
    fake_manifest = SimpleNamespace(
        source_sha256=_sha256_file(runtime_root / "runtime-manifest.json")
    )
    workspaces: list[Path] = []

    def invoke(**kwargs: Any) -> dict[str, Any]:
        workspaces.append(Path(kwargs["workspace_root"]))
        return _invoke_fake_worker(
            repository_root=Path(kwargs["repository_root"]),
            worker_script=Path(kwargs["worker_script"]),
            profile_path=Path(kwargs["profile_path"]),
            runtime_root=Path(kwargs["runtime_root"]),
            workspace_root=Path(kwargs["workspace_root"]),
            request=kwargs["request"],
            timeout_seconds=float(kwargs["timeout_seconds"]),
        )

    evidence = orchestrator_module.run_boundary_smoke(
        profile_path=profile_path,
        runtime_root=runtime_root,
        repository_root=repository_root,
        operation="artifact_boundary_self_test",
        profile_loader=lambda _path: fake_profile,
        runtime_validator=lambda *_args, **_kwargs: fake_manifest,
        worker_invoker=invoke,
        head_reader=lambda _root: EXPECTED_HEAD,
        timeout_seconds=60,
    )
    assert evidence["status"] == "PASS"
    assert evidence["repeat_count"] == 2
    assert evidence["repeat_deterministic"] is True
    assert evidence["workspace_cleaned"] is True
    assert evidence["repository_artifacts_unchanged"] is True
    assert evidence["persistent_runtime_unchanged"] is True
    assert len(workspaces) == 2
    assert all(not path.exists() for path in workspaces)
    assert evidence["markers"][-2:] == [
        "DINOV3_PERSISTENT_REPEAT_DETERMINISM_OK",
        "DINOV3_PERSISTENT_RUNTIME_BOUNDARY_OK",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeBoundaryError("worker_nonzero_exit", "worker failed"),
        RuntimeBoundaryError("worker_timeout", "worker timed out"),
    ],
)
def test_orchestrator_propagates_worker_failures_and_cleans_workspace(
    failure: RuntimeBoundaryError,
    repository_root: Path,
    tmp_path: Path,
    orchestrator_module: Any,
) -> None:
    profile_path, runtime_root, runtime_id, profile_sha256 = _prepare_fake_runtime(
        tmp_path, repository_root
    )
    fake_profile = SimpleNamespace(
        runtime_id=runtime_id,
        source_sha256=profile_sha256,
        manual_preprocessing_profile_id=PROFILE_ID,
    )
    captured: list[Path] = []

    def fail(**kwargs: Any) -> Any:
        captured.append(Path(kwargs["workspace_root"]))
        raise failure

    with pytest.raises(RuntimeBoundaryError) as raised:
        orchestrator_module.run_boundary_smoke(
            profile_path=profile_path,
            runtime_root=runtime_root,
            repository_root=repository_root,
            operation="artifact_boundary_self_test",
            profile_loader=lambda _path: fake_profile,
            runtime_validator=lambda *_args, **_kwargs: SimpleNamespace(
                source_sha256="5" * 64
            ),
            worker_invoker=fail,
            head_reader=lambda _root: EXPECTED_HEAD,
        )
    assert raised.value is failure
    assert len(captured) == 1
    assert not captured[0].exists()


def test_orchestrator_rejects_malformed_response_and_cleans_workspace(
    repository_root: Path,
    tmp_path: Path,
    orchestrator_module: Any,
) -> None:
    profile_path, runtime_root, runtime_id, profile_sha256 = _prepare_fake_runtime(
        tmp_path, repository_root
    )
    fake_profile = SimpleNamespace(
        runtime_id=runtime_id,
        source_sha256=profile_sha256,
        manual_preprocessing_profile_id=PROFILE_ID,
    )
    captured: list[Path] = []

    def malformed(**kwargs: Any) -> dict[str, Any]:
        captured.append(Path(kwargs["workspace_root"]))
        return {"status": "success"}

    with pytest.raises(orchestrator_module.SmokeValidationError):
        orchestrator_module.run_boundary_smoke(
            profile_path=profile_path,
            runtime_root=runtime_root,
            repository_root=repository_root,
            operation="artifact_boundary_self_test",
            profile_loader=lambda _path: fake_profile,
            runtime_validator=lambda *_args, **_kwargs: SimpleNamespace(
                source_sha256="5" * 64
            ),
            worker_invoker=malformed,
            head_reader=lambda _root: EXPECTED_HEAD,
        )
    assert len(captured) == 1
    assert not captured[0].exists()


def test_boundary_sources_use_argument_list_without_shell_true(
    repository_root: Path,
) -> None:
    sources = [
        repository_root / "src" / "vision_encoder_alignment" / "runtime_boundary.py",
        repository_root / "scripts" / "dinov3_isolated_worker.py",
        repository_root / "scripts" / "run_dinov3_isolated_smoke.py",
    ]
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "shell=True" not in text
    worker_text = sources[1].read_text(encoding="utf-8")
    assert "from_pretrained(" not in worker_text
    assert "load_state_dict(" not in worker_text
    assert "AutoImageProcessor" not in worker_text
