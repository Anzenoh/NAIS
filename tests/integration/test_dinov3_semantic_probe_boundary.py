"""Fake-only integration tests for the pretrained DINOv3 semantic boundary.

No test in this module imports the persistent environment, opens the real
checkpoint, or performs a network operation.  The subprocess, model response,
and current-side artifact boundary are exercised with injected fakes and
temporary files only.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from vision_encoder_alignment.dinov3_semantic_boundary import (
    DINOv3SemanticBoundaryError,
    load_semantic_profile,
    validate_semantic_response,
)
from vision_encoder_alignment.feature_artifacts import (
    ARTIFACT_SCHEMA_ID,
    FeatureArtifactMetadata,
    canonical_tensor_keys,
    write_feature_artifact,
)


def _load_script(repository_root: Path, relative_path: str) -> ModuleType:
    path = repository_root / relative_path
    module_name = f"_test_t007a4_{path.stem}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture
def runner_module(repository_root: Path) -> ModuleType:
    return _load_script(
        repository_root, "scripts/run_dinov3_pretrained_semantic_probe.py"
    )


@pytest.fixture
def semantic_profile(repository_root: Path) -> Any:
    return load_semantic_profile(
        (
            repository_root
            / "configs"
            / "semantic_probes"
            / "dinov3_vit_b16_r5931719_semantics_v0.json"
        ).resolve()
    )


@pytest.fixture
def valid_response_factory(repository_root: Path) -> Callable[..., dict[str, Any]]:
    """Reuse the task's single strict fake-response builder without drift."""

    module = _load_script(
        repository_root, "tests/unit/test_dinov3_semantic_probe_profile.py"
    )
    return module._valid_response


def _valid_response(
    semantic_profile: Any,
    valid_response_factory: Callable[..., dict[str, Any]],
    *,
    run_id: str = "t007a4-integration",
) -> dict[str, Any]:
    return valid_response_factory(semantic_profile, run_id=run_id)


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _assert_boundary_rejects(
    semantic_profile: Any,
    value: dict[str, Any],
) -> None:
    with pytest.raises(DINOv3SemanticBoundaryError):
        validate_semantic_response(value, semantic_profile)


def test_correct_cls_register_patch_and_hidden_state_mapping_is_accepted(
    semantic_profile: Any,
    valid_response_factory: Callable[..., dict[str, Any]],
) -> None:
    response = validate_semantic_response(
        _valid_response(semantic_profile, valid_response_factory), semantic_profile
    )
    value = response.to_dict()
    assert value["prefix_tokens"]["cls_indices"] == [0]
    assert value["prefix_tokens"]["register_indices"] == [1, 2, 3, 4]
    assert value["prefix_tokens"]["patch_indices"] == list(range(5, 201))
    assert value["hidden_states"]["count"] == 13
    assert value["hidden_states"]["blocks"][-1]["hidden_state_matches_hook"] is False


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        (
            "wrong-prefix-order",
            lambda value: value["prefix_tokens"].update(
                {"sequence_order": "register_cls_patch"}
            ),
        ),
        (
            "wrong-patch-start",
            lambda value: value["prefix_tokens"]["patch_indices"].__setitem__(0, 6),
        ),
        (
            "wrong-hidden-count",
            lambda value: value["hidden_states"].update({"count": 12}),
        ),
        (
            "block-hook-mismatch",
            lambda value: value["hidden_states"]["blocks"][0].update(
                {"hidden_state_matches_hook": False, "max_abs_diff": 1.0}
            ),
        ),
        (
            "final-norm-mismatch",
            lambda value: value["final_norm"].update(
                {"explicit_norm_matches_last_hidden": False, "explicit_norm_max_abs_diff": 1.0}
            ),
        ),
        (
            "pooler-cls-mismatch",
            lambda value: value["pooler"].update(
                {"cls_matches": False, "cls_max_abs_diff": 1.0}
            ),
        ),
        (
            "native-pooler-as-global",
            lambda value: value["pooler"].update(
                {"canonical_global_source": "pooler_output"}
            ),
        ),
        (
            "special-token-in-patch",
            lambda value: value["prefix_tokens"]["patch_indices"].__setitem__(0, 4),
        ),
        (
            "row-major-mismatch",
            lambda value: value["patch_order"].update({"row_major": False}),
        ),
        (
            "non-fp32-output",
            lambda value: value["taps"][0].update({"dtype": "float16"}),
        ),
        (
            "non-finite-output",
            lambda value: value["taps"][0].update({"finite": False}),
        ),
        (
            "per-tap-hash-mismatch",
            lambda value: value["artifact"]["entries"][0].update(
                {"patch_sha256": "f" * 64}
            ),
        ),
        (
            "repeat-nondeterminism",
            lambda value: value["repeat"].update({"patch_hashes_exact": False}),
        ),
        (
            "rng-mutation",
            lambda value: value["invariants"].update({"torch_rng_unchanged": False}),
        ),
        (
            "model-state-mutation",
            lambda value: value["invariants"].update({"model_state_unchanged": False}),
        ),
        (
            "socket-attempt",
            lambda value: value.update({"socket_attempts": 1}),
        ),
        (
            "absolute-path",
            lambda value: value.update({"future_path": "C:/private/result.json"}),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_invalid_semantic_evidence_is_rejected(
    case: str,
    mutate: Callable[[dict[str, Any]], None],
    semantic_profile: Any,
    valid_response_factory: Callable[..., dict[str, Any]],
) -> None:
    del case
    value = _valid_response(semantic_profile, valid_response_factory)
    mutate(value)
    _assert_boundary_rejects(semantic_profile, value)


def _write_actual_output(
    workspace: Path,
    value: dict[str, Any],
) -> None:
    output = workspace / "actual-output"
    tap_by_name = {tap["name"]: tap for tap in value["taps"]}
    for shard_index, entry in enumerate(value["artifact"]["entries"]):
        tap_name = entry["tap_name"]
        patch = (
            np.arange(2 * 196 * 768, dtype=np.float32).reshape(2, 196, 768)
            + np.float32(shard_index)
        )
        global_mean = patch.mean(axis=1, dtype=np.float32)
        patch_key, global_key = canonical_tensor_keys(tap_name)
        metadata = FeatureArtifactMetadata(
            schema_id=ARTIFACT_SCHEMA_ID,
            run_id=value["run_id"],
            model_key="dinov3_vit_b16",
            tap_namespace="backbone",
            tap_name=tap_name,
            patch_tensor_key=patch_key,
            global_mean_tensor_key=global_key,
            image_ids=["image-b", "image-a"],
            grid_height=14,
            grid_width=14,
            patch_count=196,
            hidden_dimension=768,
            dtype="float32",
            shard_index=shard_index,
            safetensors_file_name=f"{tap_name}.safetensors",
            sidecar_file_name=f"{tap_name}.json",
            checksum_algorithm="sha256",
            checksum_value="",
            created_at_utc="2026-07-21T00:00:00Z",
            run_status="incomplete",
        )
        completed = write_feature_artifact(output, metadata, patch, global_mean)
        patch_sha256 = _sha256_array(patch)
        global_sha256 = _sha256_array(global_mean)
        tap_by_name[tap_name]["patch_sha256"] = patch_sha256
        tap_by_name[tap_name]["global_sha256"] = global_sha256
        entry.update(
            {
                "checksum_sha256": completed.checksum_value,
                "global_sha256": global_sha256,
                "patch_sha256": patch_sha256,
            }
        )


def test_current_side_artifact_round_trip(
    tmp_path: Path,
    runner_module: ModuleType,
    semantic_profile: Any,
    valid_response_factory: Callable[..., dict[str, Any]],
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    value = _valid_response(semantic_profile, valid_response_factory)
    _write_actual_output(workspace, value)
    response = validate_semantic_response(value, semantic_profile)
    result = runner_module.validate_actual_output_artifacts(
        response=response, workspace_root=workspace
    )
    assert result["roundtrip_validated"] is True
    assert result["entry_count"] == 3
    assert result["selected_taps"] == ["embedding_output", "block_06", "final_norm"]


def test_artifact_checksum_mismatch_is_rejected(
    tmp_path: Path,
    runner_module: ModuleType,
    semantic_profile: Any,
    valid_response_factory: Callable[..., dict[str, Any]],
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    value = _valid_response(semantic_profile, valid_response_factory)
    _write_actual_output(workspace, value)
    response = validate_semantic_response(value, semantic_profile)
    tensor_path = workspace / value["artifact"]["entries"][0]["safetensors_relative_path"]
    tensor_path.write_bytes(tensor_path.read_bytes() + b"corruption")
    with pytest.raises(Exception):
        runner_module.validate_actual_output_artifacts(
            response=response, workspace_root=workspace
        )


def _fake_invocation_layout(tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: (tmp_path / name).resolve()
        for name in ("repository", "runtime", "snapshot", "workspace")
    }
    for path in paths.values():
        path.mkdir()
    files = {
        "semantic_profile_path": tmp_path / "semantic-profile.json",
        "runtime_profile_path": tmp_path / "runtime-profile.json",
        "snapshot_profile_path": tmp_path / "snapshot-profile.json",
        "probe_script": paths["repository"] / "probe.py",
        "python": paths["runtime"] / "python.exe",
    }
    for path in files.values():
        path.write_text("fake\n", encoding="utf-8")
    return {**paths, **{key: path.resolve() for key, path in files.items()}}


def _invoke_with_process(
    runner_module: ModuleType,
    semantic_profile: Any,
    layout: dict[str, Path],
    process_runner: Callable[..., Any],
    *,
    response_reader: Callable[..., Any],
) -> Any:
    return runner_module.invoke_pretrained_semantic_probe(
        semantic_profile=semantic_profile,
        semantic_profile_path=layout["semantic_profile_path"],
        runtime_profile_path=layout["runtime_profile_path"],
        snapshot_profile_path=layout["snapshot_profile_path"],
        runtime_root=layout["runtime"],
        snapshot_root=layout["snapshot"],
        repository_root=layout["repository"],
        probe_script=layout["probe_script"],
        workspace_root=layout["workspace"],
        run_id="fake-run",
        timeout_seconds=1.0,
        process_runner=process_runner,
        environment_builder=lambda **kwargs: {},
        python_resolver=lambda runtime: layout["python"],
        response_reader=response_reader,
    )


def test_malformed_response_is_wrapped(
    tmp_path: Path,
    runner_module: ModuleType,
    semantic_profile: Any,
) -> None:
    layout = _fake_invocation_layout(tmp_path)
    markers = ("\n".join(runner_module.PROBE_MARKERS) + "\n").encode("ascii")
    process = lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=markers, stderr=b"")
    with pytest.raises(runner_module.DINOv3SemanticOrchestrationError) as error:
        _invoke_with_process(
            runner_module,
            semantic_profile,
            layout,
            process,
            response_reader=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")),
        )
    assert error.value.classification == "probe_response_invalid"


def test_nonzero_subprocess_is_wrapped(
    tmp_path: Path,
    runner_module: ModuleType,
    semantic_profile: Any,
) -> None:
    layout = _fake_invocation_layout(tmp_path)
    process = lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout=b"", stderr=b"safe")
    with pytest.raises(runner_module.DINOv3SemanticOrchestrationError) as error:
        _invoke_with_process(
            runner_module,
            semantic_profile,
            layout,
            process,
            response_reader=lambda *args, **kwargs: None,
        )
    assert error.value.classification == "probe_nonzero_exit"
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)


def test_timeout_is_wrapped(
    tmp_path: Path,
    runner_module: ModuleType,
    semantic_profile: Any,
) -> None:
    layout = _fake_invocation_layout(tmp_path)

    def timeout(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    with pytest.raises(runner_module.DINOv3SemanticOrchestrationError) as error:
        _invoke_with_process(
            runner_module,
            semantic_profile,
            layout,
            timeout,
            response_reader=lambda *args, **kwargs: None,
        )
    assert error.value.classification == "probe_timeout"


def _fake_orchestrator_layout(tmp_path: Path) -> dict[str, Path]:
    layout = {
        name: (tmp_path / name).resolve()
        for name in ("repository", "runtime", "snapshot")
    }
    for path in layout.values():
        path.mkdir()
    for name in ("semantic.json", "runtime.json", "snapshot.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    probe = layout["repository"] / "probe.py"
    probe.write_text("fake\n", encoding="utf-8")
    artifacts = layout["repository"] / "artifacts"
    artifacts.mkdir()
    (artifacts / "sentinel.txt").write_text("unchanged\n", encoding="utf-8")
    layout.update(
        {
            "semantic_profile_path": (tmp_path / "semantic.json").resolve(),
            "runtime_profile_path": (tmp_path / "runtime.json").resolve(),
            "snapshot_profile_path": (tmp_path / "snapshot.json").resolve(),
            "probe_script": probe.resolve(),
        }
    )
    return layout


def _run_fake_orchestrator(
    runner_module: ModuleType,
    semantic_profile: Any,
    valid_response_factory: Callable[..., dict[str, Any]],
    layout: dict[str, Path],
    seen_workspaces: list[Path],
) -> dict[str, Any]:
    manifest = SimpleNamespace(source_sha256="a" * 64)

    def invoke(**kwargs: Any) -> Any:
        workspace = Path(kwargs["workspace_root"])
        seen_workspaces.append(workspace)
        value = _valid_response(
            semantic_profile, valid_response_factory, run_id=kwargs["run_id"]
        )
        response = validate_semantic_response(value, semantic_profile)
        return runner_module.ProbeInvocation(
            response=response,
            process={
                "return_code": 0,
                "shell": False,
                "stderr_bytes": 0,
                "stdout_bytes": 1,
                "stdout_sha256": "b" * 64,
            },
        )

    def protection_capture(**kwargs: Any) -> dict[str, Any]:
        return {
            "repository_artifacts": runner_module._tree_inventory(
                Path(kwargs["repository_root"]) / "artifacts"
            )
        }

    return runner_module.run_pretrained_semantic_probe(
        semantic_profile_path=layout["semantic_profile_path"],
        runtime_profile_path=layout["runtime_profile_path"],
        snapshot_profile_path=layout["snapshot_profile_path"],
        runtime_root=layout["runtime"],
        snapshot_root=layout["snapshot"],
        repository_root=layout["repository"],
        probe_script=layout["probe_script"],
        semantic_profile_loader=lambda path: semantic_profile,
        runtime_profile_loader=lambda path: SimpleNamespace(),
        snapshot_profile_loader=lambda path: SimpleNamespace(),
        identity_verifier=lambda **kwargs: None,
        runtime_validator=lambda *args, **kwargs: manifest,
        snapshot_validator=lambda *args, **kwargs: manifest,
        probe_invoker=invoke,
        artifact_validator=lambda **kwargs: {
            "entry_count": 3,
            "entries": [],
            "image_ids": ["image-b", "image-a"],
            "roundtrip_validated": True,
            "selected_taps": ["embedding_output", "block_06", "final_norm"],
        },
        protection_capture=protection_capture,
    )


def test_workspace_cleanup_after_two_fake_processes(
    tmp_path: Path,
    runner_module: ModuleType,
    semantic_profile: Any,
    valid_response_factory: Callable[..., dict[str, Any]],
) -> None:
    layout = _fake_orchestrator_layout(tmp_path)
    seen_workspaces: list[Path] = []
    result = _run_fake_orchestrator(
        runner_module,
        semantic_profile,
        valid_response_factory,
        layout,
        seen_workspaces,
    )
    assert result["workspace_cleaned"] is True
    assert result["repeat_count"] == 2
    assert len(seen_workspaces) == 2
    assert len(set(seen_workspaces)) == 2
    assert all(not path.exists() for path in seen_workspaces)


def test_repository_artifact_inventory_remains_unchanged(
    tmp_path: Path,
    runner_module: ModuleType,
    semantic_profile: Any,
    valid_response_factory: Callable[..., dict[str, Any]],
) -> None:
    layout = _fake_orchestrator_layout(tmp_path)
    sentinel = layout["repository"] / "artifacts" / "sentinel.txt"
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    result = _run_fake_orchestrator(
        runner_module,
        semantic_profile,
        valid_response_factory,
        layout,
        [],
    )
    assert result["protected_unchanged"] is True
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before


def test_repository_relative_profile_paths_are_resolved(
    tmp_path: Path,
    runner_module: ModuleType,
    semantic_profile: Any,
    valid_response_factory: Callable[..., dict[str, Any]],
) -> None:
    """The required CLI uses repository-relative profile arguments."""

    layout = _fake_orchestrator_layout(tmp_path)
    for name in ("semantic.json", "runtime.json", "snapshot.json"):
        (layout["repository"] / name).write_text("{}\n", encoding="utf-8")
    layout["semantic_profile_path"] = Path("semantic.json")
    layout["runtime_profile_path"] = Path("runtime.json")
    layout["snapshot_profile_path"] = Path("snapshot.json")
    result = _run_fake_orchestrator(
        runner_module,
        semantic_profile,
        valid_response_factory,
        layout,
        [],
    )
    assert result["status"] == "PASS"
