"""Integration tests for DINOv3 snapshot and checkpoint-load boundaries.

These tests use byte-sized fake snapshot members and injected probe/process
fakes.  They must never require Hub access, credentials, the persistent
runtime, or the real checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest

import vision_encoder_alignment.snapshot_boundary as snapshot_boundary
from vision_encoder_alignment.snapshot_boundary import (
    SnapshotFileIdentity,
    SnapshotProfile,
    load_snapshot_profile,
)


def _load_script(repository_root: Path, filename: str) -> ModuleType:
    path = repository_root / "scripts" / filename
    module_name = f"_test_t007a3a_{path.stem}"
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
def acquisition_module(repository_root: Path) -> ModuleType:
    return _load_script(repository_root, "acquire_dinov3_snapshot.py")


@pytest.fixture
def smoke_module(repository_root: Path) -> ModuleType:
    return _load_script(repository_root, "run_dinov3_checkpoint_load_smoke.py")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tree_inventory(root: Path) -> tuple[tuple[str, str, int], ...]:
    if not root.exists():
        return ()
    entries: list[tuple[str, str, int]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            payload = path.read_bytes()
            entries.append((relative, _sha256(payload), len(payload)))
        else:
            entries.append((relative + "/", "directory", 0))
    return tuple(entries)


@dataclass(frozen=True, slots=True)
class FakeAcquisition:
    profile: SnapshotProfile
    profile_path: Path
    runtime_profile_path: Path
    repository_root: Path
    runtime_root: Path
    snapshot_root: Path
    payloads: dict[str, bytes]
    boundary_module: Any
    runtime_provisioner_module: Any


def _prepare_fake_acquisition(
    tmp_path: Path, repository_root: Path
) -> FakeAcquisition:
    real_profile = load_snapshot_profile(
        repository_root
        / "configs"
        / "model_snapshots"
        / "dinov3_vit_b16_r5931719_v0.json"
    )
    payloads = {
        "config.json": b'{"model_type":"dinov3_vit"}\n',
        "preprocessor_config.json": b'{"do_resize":true}\n',
        "model.safetensors": b"small-fake-safetensors-payload\n",
    }
    identities = tuple(
        SnapshotFileIdentity(
            filename=identity.filename,
            size_bytes=len(payloads[identity.filename]),
            sha256=_sha256(payloads[identity.filename]),
        )
        for identity in real_profile.files
    )
    profile = replace(
        real_profile,
        files=identities,
        selected_total_bytes=sum(identity.size_bytes for identity in identities),
    )

    fake_repository = (tmp_path / "repository").resolve()
    fake_repository.mkdir()
    configs = fake_repository / "configs"
    (configs / "model_snapshots").mkdir(parents=True)
    (configs / "runtime_profiles").mkdir()
    profile_path = (
        configs / "model_snapshots" / "dinov3_vit_b16_r5931719_v0.json"
    )
    profile_path.write_bytes(b"fake profile loader sentinel\n")
    runtime_profile_path = (
        configs / "runtime_profiles" / "dinov3_isolated_manual_v0.json"
    )
    runtime_profile_path.write_bytes(b"fake runtime profile loader sentinel\n")
    (configs / "models.yaml").write_bytes(b"dinov3_vit_b16:\n  revision: null\n")

    runtime_root = (tmp_path / "runtime" / profile.runtime_id).resolve()
    metadata = runtime_root / "metadata"
    metadata.mkdir(parents=True)
    for filename in ("config.json", "preprocessor_config.json"):
        (metadata / filename).write_bytes(payloads[filename])
    (runtime_root / "runtime-manifest.json").write_bytes(b"protected runtime\n")

    snapshot_root = (tmp_path / "snapshots" / profile.snapshot_id).resolve()
    boundary_module = SimpleNamespace(
        SnapshotManifest=snapshot_boundary.SnapshotManifest,
        canonical_json_bytes=snapshot_boundary.canonical_json_bytes,
        load_snapshot_profile=lambda _path: profile,
        resolve_default_snapshot_root=snapshot_boundary.resolve_default_snapshot_root,
        validate_snapshot_manifest=snapshot_boundary.validate_snapshot_manifest,
        validate_snapshot_root=snapshot_boundary.validate_snapshot_root,
    )
    runtime_profile = SimpleNamespace(
        path=runtime_profile_path.resolve(),
        sha256=profile.runtime_profile_sha256,
        runtime_id=profile.runtime_id,
    )
    runtime_plan = SimpleNamespace(runtime_root=runtime_root)
    runtime_provisioner_module = SimpleNamespace(
        load_profile=lambda _path: runtime_profile,
        plan_provision=lambda *_args, **_kwargs: runtime_plan,
    )
    return FakeAcquisition(
        profile=profile,
        profile_path=profile_path,
        runtime_profile_path=runtime_profile_path,
        repository_root=fake_repository,
        runtime_root=runtime_root,
        snapshot_root=snapshot_root,
        payloads=payloads,
        boundary_module=boundary_module,
        runtime_provisioner_module=runtime_provisioner_module,
    )


def _remote_identity(module: ModuleType, fake: FakeAcquisition) -> Any:
    weight = fake.profile.weight_file
    return module.RemoteWeightIdentity(
        commit_sha=fake.profile.revision,
        filename=weight.filename,
        size_bytes=weight.size_bytes,
        sha256=weight.sha256,
    )


def _acquire_fake(
    module: ModuleType,
    fake: FakeAcquisition,
    monkeypatch: pytest.MonkeyPatch,
    *,
    weight_downloader: Callable[[Any, Path], Any] | None = None,
    boundary_module: Any | None = None,
    remote_identity_query: Callable[[Any], Any] | None = None,
) -> str:
    monkeypatch.setattr(module, "_assert_exact_profile", lambda _profile: None)

    def download(_profile: Any, cache_directory: Path) -> Path:
        target = cache_directory / fake.profile.weight_file.filename
        target.write_bytes(fake.payloads[fake.profile.weight_file.filename])
        return target

    return module.acquire_snapshot(
        fake.profile_path,
        fake.runtime_profile_path,
        fake.snapshot_root,
        fake.runtime_root,
        repository_root=fake.repository_root,
        boundary_module=boundary_module or fake.boundary_module,
        runtime_provisioner_module=fake.runtime_provisioner_module,
        repository_verifier=lambda _plan: None,
        persistent_runtime_verifier=lambda _plan: SimpleNamespace(status="verified"),
        remote_identity_query=remote_identity_query
        or (lambda _profile: _remote_identity(module, fake)),
        weight_downloader=weight_downloader or download,
        clock=lambda: "2026-07-21T00:00:00Z",
    )


def _staging_roots(fake: FakeAcquisition) -> tuple[Path, ...]:
    parent = fake.snapshot_root.parent
    if not parent.exists():
        return ()
    return tuple(parent.glob(f"{fake.profile.staging_name_prefix}*"))


def test_staging_to_final_atomic_publication(
    tmp_path: Path,
    repository_root: Path,
    acquisition_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _prepare_fake_acquisition(tmp_path, repository_root)
    staging_observations: list[tuple[Path, bool]] = []

    def validate(root: Path, *args: Any, **kwargs: Any) -> Any:
        root = Path(root)
        if kwargs.get("_allow_owned_staging"):
            staging_observations.append((root, fake.snapshot_root.exists()))
        return snapshot_boundary.validate_snapshot_root(root, *args, **kwargs)

    proxy = SimpleNamespace(**vars(fake.boundary_module))
    proxy.validate_snapshot_root = validate
    marker = _acquire_fake(
        acquisition_module,
        fake,
        monkeypatch,
        boundary_module=proxy,
    )

    assert marker == "DINOV3_EXACT_SNAPSHOT_ACQUIRED"
    assert len(staging_observations) == 1
    staging, final_existed_during_staging_validation = staging_observations[0]
    assert staging.parent == fake.snapshot_root.parent
    assert staging.name.startswith(fake.profile.staging_name_prefix)
    assert final_existed_during_staging_validation is False
    assert not staging.exists()
    assert fake.snapshot_root.is_dir()
    assert snapshot_boundary.validate_snapshot_root(
        fake.snapshot_root,
        fake.profile,
        repository_root=fake.repository_root,
        persistent_runtime_root=fake.runtime_root,
    ).snapshot_id == fake.profile.snapshot_id


def test_acquisition_failure_leaves_final_root_absent(
    tmp_path: Path,
    repository_root: Path,
    acquisition_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _prepare_fake_acquisition(tmp_path, repository_root)

    def corrupt_download(_profile: Any, cache_directory: Path) -> Path:
        result = cache_directory / "model.safetensors"
        result.write_bytes(b"wrong")
        return result

    with pytest.raises(acquisition_module.SnapshotAcquisitionError) as error:
        _acquire_fake(
            acquisition_module,
            fake,
            monkeypatch,
            weight_downloader=corrupt_download,
        )
    assert error.value.classification == "DINOV3_SNAPSHOT_INTEGRITY_FAILED"
    assert not fake.snapshot_root.exists()
    assert _staging_roots(fake) == ()


def test_task_owned_download_cache_is_cleaned(
    tmp_path: Path,
    repository_root: Path,
    acquisition_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _prepare_fake_acquisition(tmp_path, repository_root)
    observed_caches: list[Path] = []

    def download(_profile: Any, cache_directory: Path) -> Path:
        observed_caches.append(cache_directory)
        result = cache_directory / "model.safetensors"
        result.write_bytes(fake.payloads["model.safetensors"])
        return result

    _acquire_fake(
        acquisition_module,
        fake,
        monkeypatch,
        weight_downloader=download,
    )
    assert len(observed_caches) == 1
    assert not observed_caches[0].exists()
    assert _staging_roots(fake) == ()


def test_existing_exact_valid_snapshot_is_reused_without_remote_call(
    tmp_path: Path,
    repository_root: Path,
    acquisition_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _prepare_fake_acquisition(tmp_path, repository_root)
    assert _acquire_fake(acquisition_module, fake, monkeypatch) == (
        "DINOV3_EXACT_SNAPSHOT_ACQUIRED"
    )
    before = _tree_inventory(fake.snapshot_root)

    def forbidden_remote(_profile: Any) -> Any:
        raise AssertionError("exact-valid reuse must not access the network")

    marker = _acquire_fake(
        acquisition_module,
        fake,
        monkeypatch,
        remote_identity_query=forbidden_remote,
        weight_downloader=lambda *_args: (_ for _ in ()).throw(
            AssertionError("exact-valid reuse must not download")
        ),
    )
    assert marker == "DINOV3_EXACT_SNAPSHOT_ALREADY_VALID"
    assert _tree_inventory(fake.snapshot_root) == before


def test_mismatched_existing_root_is_rejected_without_repair(
    tmp_path: Path,
    repository_root: Path,
    acquisition_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _prepare_fake_acquisition(tmp_path, repository_root)
    fake.snapshot_root.mkdir(parents=True)
    residue = fake.snapshot_root / "partial-download.tmp"
    residue.write_bytes(b"do not repair")
    before = _tree_inventory(fake.snapshot_root)

    with pytest.raises(acquisition_module.SnapshotCollisionError) as error:
        _acquire_fake(
            acquisition_module,
            fake,
            monkeypatch,
            remote_identity_query=lambda _profile: (_ for _ in ()).throw(
                AssertionError("collision must fail before network access")
            ),
        )
    assert error.value.classification == "DINOV3_SNAPSHOT_COLLISION"
    assert _tree_inventory(fake.snapshot_root) == before


def test_registry_bytes_are_immutable_across_acquisition(
    tmp_path: Path,
    repository_root: Path,
    acquisition_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _prepare_fake_acquisition(tmp_path, repository_root)
    registry = fake.repository_root / "configs" / "models.yaml"
    before = registry.read_bytes()
    _acquire_fake(acquisition_module, fake, monkeypatch)
    assert registry.read_bytes() == before


def test_persistent_runtime_is_immutable_across_acquisition(
    tmp_path: Path,
    repository_root: Path,
    acquisition_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _prepare_fake_acquisition(tmp_path, repository_root)
    before = _tree_inventory(fake.runtime_root)
    _acquire_fake(acquisition_module, fake, monkeypatch)
    assert _tree_inventory(fake.runtime_root) == before


def test_acquisition_creates_no_repository_artifact(
    tmp_path: Path,
    repository_root: Path,
    acquisition_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _prepare_fake_acquisition(tmp_path, repository_root)
    before = _tree_inventory(fake.repository_root)
    _acquire_fake(acquisition_module, fake, monkeypatch)
    assert _tree_inventory(fake.repository_root) == before
    assert not (fake.repository_root / "artifacts").exists()


def test_repository_command_decodes_non_ascii_git_output_as_utf8(
    acquisition_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(command: Any, **kwargs: Any) -> Any:
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout="G:/내 드라이브/2026NAIS\n".encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(acquisition_module.subprocess, "run", fake_run)
    result = acquisition_module._run_command(["git", "rev-parse", "--show-toplevel"])

    assert result.stdout == "G:/내 드라이브/2026NAIS\n"
    assert observed["command"] == ("git", "rev-parse", "--show-toplevel")
    assert observed["text"] is False
    assert observed["shell"] is False


@dataclass(frozen=True, slots=True)
class FakeProbeInvocation:
    repository_root: Path
    runtime_root: Path
    snapshot_root: Path
    workspace_root: Path
    snapshot_profile_path: Path
    runtime_profile_path: Path
    probe_script: Path


def _prepare_fake_probe_invocation(tmp_path: Path) -> FakeProbeInvocation:
    repository = (tmp_path / "repository").resolve()
    runtime = (tmp_path / "runtime").resolve()
    snapshot = (tmp_path / "snapshot").resolve()
    workspace = (tmp_path / "workspace").resolve()
    for root in (repository, runtime, snapshot, workspace):
        root.mkdir()
    snapshot_profile = repository / "snapshot-profile.json"
    runtime_profile = repository / "runtime-profile.json"
    probe_script = repository / "probe.py"
    snapshot_profile.write_bytes(b"{}\n")
    runtime_profile.write_bytes(b"{}\n")
    probe_script.write_bytes(b"# fake probe\n")
    return FakeProbeInvocation(
        repository_root=repository,
        runtime_root=runtime,
        snapshot_root=snapshot,
        workspace_root=workspace,
        snapshot_profile_path=snapshot_profile,
        runtime_profile_path=runtime_profile,
        probe_script=probe_script,
    )


def _invoke_probe(
    module: ModuleType,
    fake: FakeProbeInvocation,
    process_runner: Callable[..., Any],
) -> dict[str, Any]:
    return module.invoke_checkpoint_load_probe(
        runtime_root=fake.runtime_root,
        repository_root=fake.repository_root,
        snapshot_profile_path=fake.snapshot_profile_path,
        runtime_profile_path=fake.runtime_profile_path,
        snapshot_root=fake.snapshot_root,
        probe_script=fake.probe_script,
        workspace_root=fake.workspace_root,
        timeout_seconds=1.0,
        process_runner=process_runner,
        environment_builder=lambda **_kwargs: {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        python_resolver=lambda _root: Path(sys.executable),
    )


def _successful_process_runner(
    captured: list[tuple[Any, dict[str, Any]]],
) -> Callable[..., Any]:
    def run(command: Any, **kwargs: Any) -> Any:
        captured.append((command, kwargs))
        response_name = command[command.index("--response") + 1]
        (Path(kwargs["cwd"]) / response_name).write_bytes(b"{}\n")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    return run


def test_probe_subprocess_uses_argument_list(
    tmp_path: Path, smoke_module: ModuleType
) -> None:
    fake = _prepare_fake_probe_invocation(tmp_path)
    captured: list[tuple[Any, dict[str, Any]]] = []
    _invoke_probe(smoke_module, fake, _successful_process_runner(captured))

    assert len(captured) == 1
    command, _kwargs = captured[0]
    assert isinstance(command, list)
    assert command[:2] == [str(Path(sys.executable).resolve()), "-B"]
    assert command[2] == str(fake.probe_script)
    assert command[-2:] == ["--response", "probe-response.json"]
    assert "--snapshot-root" in command
    assert "--runtime-root" in command
    assert "--workspace-root" in command


def test_probe_subprocess_explicitly_disables_shell(
    tmp_path: Path, smoke_module: ModuleType
) -> None:
    fake = _prepare_fake_probe_invocation(tmp_path)
    captured: list[tuple[Any, dict[str, Any]]] = []
    _invoke_probe(smoke_module, fake, _successful_process_runner(captured))

    _command, kwargs = captured[0]
    assert kwargs["shell"] is False
    assert kwargs["check"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is False


def test_probe_timeout_is_wrapped(
    tmp_path: Path, smoke_module: ModuleType
) -> None:
    fake = _prepare_fake_probe_invocation(tmp_path)

    def timeout(command: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(command, timeout=1.0)

    with pytest.raises(smoke_module.CheckpointLoadSmokeError) as error:
        _invoke_probe(smoke_module, fake, timeout)
    assert error.value.classification == "probe_timeout"
    assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)


def test_nonzero_probe_exit_is_wrapped(
    tmp_path: Path, smoke_module: ModuleType
) -> None:
    fake = _prepare_fake_probe_invocation(tmp_path)

    def nonzero(_command: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(returncode=23, stdout=b"safe stdout", stderr=b"safe stderr")

    with pytest.raises(smoke_module.CheckpointLoadSmokeError) as error:
        _invoke_probe(smoke_module, fake, nonzero)
    assert error.value.classification == "probe_nonzero_exit"
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
    assert error.value.__cause__.returncode == 23


def test_malformed_probe_json_is_rejected(
    tmp_path: Path, smoke_module: ModuleType
) -> None:
    fake = _prepare_fake_probe_invocation(tmp_path)

    def malformed(command: list[str], **kwargs: Any) -> Any:
        response_name = command[command.index("--response") + 1]
        (Path(kwargs["cwd"]) / response_name).write_bytes(b'{"status": NaN}\n')
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with pytest.raises(smoke_module.CheckpointLoadSmokeError) as error:
        _invoke_probe(smoke_module, fake, malformed)
    assert error.value.classification == "probe_response_invalid"


@dataclass(frozen=True, slots=True)
class FakeSmokeContract:
    snapshot_profile: Any
    runtime_profile: Any
    snapshot_manifest: Any
    runtime_manifest: Any


def _fake_smoke_contract() -> FakeSmokeContract:
    runtime_id = "dinov3-fake-runtime-v1"
    runtime_profile_sha256 = "1" * 64
    return FakeSmokeContract(
        snapshot_profile=SimpleNamespace(
            config_class="DINOv3ViTConfig",
            expected_parameter_count=85_660_416,
            expected_parameter_tensor_count=211,
            model_class="DINOv3ViTModel",
            model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
            revision="5931719e67bbdb9737e363e781fb0c67687896bc",
            runtime_id=runtime_id,
            runtime_profile_sha256=runtime_profile_sha256,
            snapshot_id="dinov3-vitb16-r5931719-safetensors-v1",
            source_sha256="2" * 64,
        ),
        runtime_profile=SimpleNamespace(
            runtime_id=runtime_id,
            source_sha256=runtime_profile_sha256,
        ),
        snapshot_manifest=SimpleNamespace(source_sha256="3" * 64),
        runtime_manifest=SimpleNamespace(source_sha256="4" * 64),
    )


def _valid_probe_response(
    module: ModuleType, contract: FakeSmokeContract
) -> dict[str, Any]:
    snapshot = contract.snapshot_profile
    runtime = contract.runtime_profile
    return {
        "components": {
            "cls_token": True,
            "encoder_layer_count": 12,
            "final_norm": True,
            "patch_embedding": True,
            "register_tokens": True,
        },
        "config_class": snapshot.config_class,
        "device": "cpu",
        "dtype": "float32",
        "evaluation_mode": True,
        "forward_count": 0,
        "forward_executed": False,
        "frozen": True,
        "load_options": {
            "local_files_only": True,
            "output_loading_info": True,
            "token": False,
            "torch_dtype": "float32",
            "trust_remote_code": False,
            "use_safetensors": True,
        },
        "loading_info": {
            "error_message_count": 0,
            "mismatched_key_count": 0,
            "missing_key_count": 0,
            "unexpected_key_count": 0,
        },
        "markers": list(module.PROBE_MARKERS),
        "model_cache_unchanged": True,
        "model_class": snapshot.model_class,
        "model_id": snapshot.model_id,
        "model_state_sha256": "5" * 64,
        "parameter_count": snapshot.expected_parameter_count,
        "parameter_tensor_count": snapshot.expected_parameter_tensor_count,
        "persistent_runtime_unchanged": True,
        "processor_object_count": 0,
        "revision": snapshot.revision,
        "rng": {
            "numpy_unchanged": True,
            "python_unchanged": True,
            "torch_unchanged": True,
        },
        "runtime_id": runtime.runtime_id,
        "runtime_manifest_sha256": contract.runtime_manifest.source_sha256,
        "runtime_profile_sha256": runtime.source_sha256,
        "schema_id": module.PROBE_RESPONSE_SCHEMA_ID,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_manifest_sha256": contract.snapshot_manifest.source_sha256,
        "snapshot_profile_sha256": snapshot.source_sha256,
        "snapshot_unchanged": True,
        "socket_attempt_count": 0,
        "state_sha256": "5" * 64,
        "status": "success",
        "trainable_parameter_count": 0,
        "training": False,
        "torchvision_import_count": 0,
        "warnings": [],
    }


def _validate_response(
    module: ModuleType,
    contract: FakeSmokeContract,
    response: dict[str, Any],
) -> dict[str, Any]:
    return module.validate_probe_response(
        response,
        snapshot_profile=contract.snapshot_profile,
        runtime_profile=contract.runtime_profile,
        snapshot_manifest=contract.snapshot_manifest,
        runtime_manifest=contract.runtime_manifest,
    )


def test_wrong_model_facts_are_rejected(smoke_module: ModuleType) -> None:
    contract = _fake_smoke_contract()
    response = _valid_probe_response(smoke_module, contract)
    response["parameter_count"] += 1

    with pytest.raises(smoke_module.CheckpointLoadSmokeError) as error:
        _validate_response(smoke_module, contract, response)
    assert error.value.classification == "probe_fact_mismatch"


def test_socket_attempt_report_is_rejected(smoke_module: ModuleType) -> None:
    contract = _fake_smoke_contract()
    response = _valid_probe_response(smoke_module, contract)
    response["socket_attempt_count"] = 1

    with pytest.raises(smoke_module.CheckpointLoadSmokeError) as error:
        _validate_response(smoke_module, contract, response)
    assert error.value.classification == "probe_fact_mismatch"


def test_forward_executed_report_is_rejected(smoke_module: ModuleType) -> None:
    contract = _fake_smoke_contract()
    response = _valid_probe_response(smoke_module, contract)
    response["forward_count"] = 1
    response["forward_executed"] = True

    with pytest.raises(smoke_module.CheckpointLoadSmokeError) as error:
        _validate_response(smoke_module, contract, response)
    assert error.value.classification == "probe_fact_mismatch"


def test_checkpoint_smoke_cleans_workspace_after_two_fake_probes(
    tmp_path: Path, smoke_module: ModuleType
) -> None:
    contract = _fake_smoke_contract()
    repository = (tmp_path / "repository").resolve()
    runtime = (tmp_path / "runtime").resolve()
    snapshot = (tmp_path / "snapshot").resolve()
    for root in (repository, runtime, snapshot):
        root.mkdir()
    scripts = repository / "scripts"
    scripts.mkdir()
    probe_script = scripts / "probe_dinov3_checkpoint_load.py"
    probe_script.write_bytes(b"# fake probe\n")
    snapshot_profile_path = repository / "snapshot-profile.json"
    runtime_profile_path = repository / "runtime-profile.json"
    snapshot_profile_path.write_bytes(b"{}\n")
    runtime_profile_path.write_bytes(b"{}\n")
    observed_repeat_roots: list[Path] = []

    def invoke(**kwargs: Any) -> dict[str, Any]:
        observed_repeat_roots.append(Path(kwargs["workspace_root"]))
        return _valid_probe_response(smoke_module, contract)

    evidence = smoke_module.run_checkpoint_load_smoke(
        snapshot_profile_path=snapshot_profile_path,
        runtime_profile_path=runtime_profile_path,
        snapshot_root=snapshot,
        runtime_root=runtime,
        repository_root=repository,
        probe_script=probe_script,
        snapshot_profile_loader=lambda _path: contract.snapshot_profile,
        runtime_profile_loader=lambda _path: contract.runtime_profile,
        snapshot_validator=lambda *_args, **_kwargs: contract.snapshot_manifest,
        runtime_validator=lambda *_args, **_kwargs: contract.runtime_manifest,
        probe_invoker=invoke,
        inventory_reader=lambda root: _tree_inventory(Path(root)),
    )

    assert evidence["status"] == "PASS"
    assert evidence["repeat_count"] == 2
    assert evidence["repeat_deterministic"] is True
    assert evidence["workspace_cleaned"] is True
    assert len(observed_repeat_roots) == 2
    assert all(not root.exists() for root in observed_repeat_roots)
    assert len({root.parent for root in observed_repeat_roots}) == 1
    assert not observed_repeat_roots[0].parent.exists()
