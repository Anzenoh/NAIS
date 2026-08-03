from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from vision_encoder_alignment.dinov3_extraction_boundary import (
    build_dinov3_apple_invocation_plan,
)
from vision_encoder_alignment.dinov3_production_identity import (
    APPLE_GRAPH_ID,
    APPLE_UNRESOLVED_GRAPH_ID,
    LEGACY_GRAPH_ID,
    DINOv3ProductionIdentityError,
    validate_versioned_production_graph,
)
from vision_encoder_alignment.production_runtime_contract import (
    load_production_runtime_profile,
)
from vision_encoder_alignment.runtime_boundary import (
    build_sanitized_worker_environment,
    runtime_python_path,
)


REPOSITORY = Path(__file__).resolve().parents[2]
LEGACY_RUNTIME = REPOSITORY / "configs/runtime_profiles/dinov3_isolated_manual_v0.json"
APPLE_RUNTIME = REPOSITORY / "configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json"
SNAPSHOT = REPOSITORY / "configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json"
LEGACY_EXTRACTOR = REPOSITORY / "configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json"
APPLE_EXTRACTOR = REPOSITORY / "configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json"


def test_complete_legacy_graph_remains_valid() -> None:
    graph = validate_versioned_production_graph(
        REPOSITORY,
        graph_id=LEGACY_GRAPH_ID,
        runtime_profile_path=LEGACY_RUNTIME,
        extractor_profile_path=LEGACY_EXTRACTOR,
        snapshot_profile_path=SNAPSHOT,
    )
    assert graph.graph_id == LEGACY_GRAPH_ID
    assert graph.ready_for_p1_extraction is False


def test_complete_apple_graph_is_valid_but_not_ready() -> None:
    graph = validate_versioned_production_graph(
        REPOSITORY,
        graph_id=APPLE_GRAPH_ID,
        runtime_profile_path=APPLE_RUNTIME,
        extractor_profile_path=APPLE_EXTRACTOR,
        snapshot_profile_path=SNAPSHOT,
    )
    assert graph.graph_id == APPLE_GRAPH_ID
    assert graph.unknown_nodes == graph.dangling_edges == ()
    assert graph.ready_for_p1_extraction is False


def test_resolved_live_profile_cannot_masquerade_as_unresolved_history() -> None:
    with pytest.raises(DINOv3ProductionIdentityError) as captured:
        validate_versioned_production_graph(
            REPOSITORY,
            graph_id=APPLE_UNRESOLVED_GRAPH_ID,
            runtime_profile_path=APPLE_RUNTIME,
            extractor_profile_path=APPLE_EXTRACTOR,
            snapshot_profile_path=SNAPSHOT,
        )
    assert captured.value.classification == "DINOV3_DANGLING_IDENTITY_EDGE"


def test_legacy_apple_hybrid_graph_is_rejected() -> None:
    with pytest.raises(DINOv3ProductionIdentityError) as captured:
        validate_versioned_production_graph(
            REPOSITORY,
            graph_id=APPLE_GRAPH_ID,
            runtime_profile_path=LEGACY_RUNTIME,
            extractor_profile_path=APPLE_EXTRACTOR,
            snapshot_profile_path=SNAPSHOT,
        )
    assert captured.value.classification == "DINOV3_HYBRID_PRODUCTION_GRAPH"


def test_linux_interpreter_layout_is_explicit(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    interpreter = runtime / "bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"fake-python")
    assert runtime_python_path(runtime, interpreter_layout="linux_bin") == interpreter.resolve()
    assert not (runtime / "venv/bin/python").exists()


def test_apple_environment_is_offline_and_cuda_explicit(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    repository = tmp_path / "repository"
    workspace = tmp_path / "workspace"
    for directory in (runtime / "bin", repository, workspace):
        directory.mkdir(parents=True)
    interpreter = runtime / "bin/python"
    interpreter.write_bytes(b"fake-python")
    environment = build_sanitized_worker_environment(
        runtime_python=interpreter,
        repository_root=repository,
        workspace_root=workspace,
        environ={"PATH": "/usr/bin", "TOKEN": "must-not-survive"},
        cuda_visible_devices="0",
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert "TOKEN" not in environment


def test_dinov3_apple_invocation_is_argument_list_only(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    workspace = tmp_path / "workspace"
    snapshot_root = tmp_path / "snapshot"
    for directory in (runtime / "bin", workspace, snapshot_root):
        directory.mkdir(parents=True)
    (runtime / "bin/python").write_bytes(b"fake-python")
    profile = load_production_runtime_profile(APPLE_RUNTIME)
    plan = build_dinov3_apple_invocation_plan(
        runtime_profile=profile,
        runtime_profile_path=APPLE_RUNTIME,
        runtime_root=runtime,
        repository_root=REPOSITORY,
        workspace_root=workspace,
        worker_script=REPOSITORY / "scripts/dinov3_extraction_worker.py",
        extractor_profile_path=APPLE_EXTRACTOR,
        snapshot_profile_path=SNAPSHOT,
        snapshot_root=snapshot_root,
        request_relative_path="request.json",
        response_relative_path="response.json",
        environ={"PATH": "/usr/bin"},
    )
    assert isinstance(plan.command, tuple)
    assert plan.command[1] == "-B"
    assert Path(plan.command[0]) == (runtime / "bin/python").resolve()
    assert plan.environment["VEA_DINOV3_EXECUTION_SELECTOR"] == profile.selector


def test_worker_fake_cuda_capability_branch() -> None:
    path = REPOSITORY / "scripts/dinov3_extraction_worker.py"
    spec = importlib.util.spec_from_file_location("_vea_worker_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _Version:
        cuda = "12.8"

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "NVIDIA GeForce RTX 5090"

    class _Torch:
        version = _Version()
        cuda = _Cuda()

    assert module._execution_device(_Torch(), module.APPLE_EXECUTION_SELECTOR) == "cuda:0"


def test_apple_graph_cli_verify_is_read_only() -> None:
    command = [
        sys.executable,
        "-B",
        str(REPOSITORY / "scripts/rebind_dinov3_production_identity.py"),
        "verify",
        "--graph-id",
        APPLE_GRAPH_ID,
    ]
    result = subprocess.run(
        command,
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert result.returncode == 0, result.stderr
    assert "DINOV3_APPLE_PRODUCTION_GRAPH_VERIFY_OK" in result.stdout


def test_migration_sources_contain_no_ambient_profile_selection() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPOSITORY / "src/vision_encoder_alignment/production_runtime_contract.py",
            REPOSITORY / "src/vision_encoder_alignment/dinov3_production_identity.py",
            REPOSITORY / "scripts/rebind_dinov3_production_identity.py",
        )
    )
    assert "requests." not in sources
    assert "shell=True" not in sources
    assert "socket." not in sources
