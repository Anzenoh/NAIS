from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import vision_encoder_alignment.p1_runtime_preflight as preflight
from vision_encoder_alignment.p1_runtime_preflight import (
    GIB,
    NOT_AVAILABLE,
    NOT_INSTALLED,
    CommandResult,
    PreflightError,
    SystemProbe,
    canonical_json_bytes,
    collect_runtime_preflight,
    decide_preflight,
    inventory_checkpoint_cache,
    load_checkpoint_plan,
    load_runtime_template,
    strict_json_loads,
    validate_checkpoint_inventory,
    validate_final_runtime_lock,
    validate_runtime_evidence,
    verify_inventory_against_cache,
    write_json_output,
)


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_TEMPLATE = ROOT / "configs/runtime_profiles/p1_bulk_cuda_a100_preflight.template.json"
DIRECT_TEMPLATE = ROOT / "configs/runtime_profiles/p1_bulk_cuda_rtx5090_direct_preflight.template.json"
PLAN = ROOT / "configs/model_snapshots/p1_three_encoder_checkpoint_plan.template.json"
REGISTRY = ROOT / "configs/models.yaml"
TIMESTAMP = "2026-07-28T00:00:00Z"


def _plan() -> dict[str, object]:
    return load_checkpoint_plan(PLAN, REGISTRY)


def _template() -> dict[str, object]:
    return load_runtime_template(RUNTIME_TEMPLATE)


def _probe(*, torch_mode: str = "cuda", repository_exists: bool = True, hostname: str = "a100-node-1") -> SystemProbe:
    torch = {
        "import_status": "INSTALLED",
        "version": "2.4.1+cu124",
        "build": "a" * 40,
        "cuda_version": "12.4",
        "cuda_available": True,
        "device_count": 1,
        "device_names": ["NVIDIA A100-SXM4-80GB"],
        "cudnn_version": "90100",
    }
    if torch_mode == "missing":
        torch.update(
            import_status=NOT_INSTALLED,
            version=NOT_INSTALLED,
            build=NOT_AVAILABLE,
            cuda_version=NOT_AVAILABLE,
            cuda_available=False,
            device_count=0,
            device_names=[],
            cudnn_version=NOT_AVAILABLE,
        )
    elif torch_mode == "cpu":
        torch.update(
            version="2.4.1+cpu",
            cuda_version=NOT_AVAILABLE,
            cuda_available=False,
            device_count=0,
            device_names=[],
            cudnn_version=NOT_AVAILABLE,
        )
    return SystemProbe(
        os_name="Linux",
        kernel="6.8.0",
        architecture="x86_64",
        glibc="glibc-2.35",
        python_version="3.12.4",
        hostname=hostname,
        packages={key: "1.2.3" for key in preflight.PACKAGE_KEYS},
        torch=torch,
        repository_mirror_exists=repository_exists,
    )


def _runner(
    *, gpu_name: str = "NVIDIA A100-SXM4-80GB", nvidia_exit: int = 0,
    used_bytes: int = 80 * GIB,
) -> tuple[preflight.CommandRunner, list[tuple[str, ...]]]:
    calls: list[tuple[str, ...]] = []

    def run(args: object) -> CommandResult:
        command = tuple(args)  # type: ignore[arg-type]
        calls.append(command)
        if command[0] == "nvidia-smi" and len(command) > 1:
            output = f"0, {gpu_name}, GPU-0001, 81920, 550.54\n" if nvidia_exit == 0 else ""
            return CommandResult(nvidia_exit, output)
        if command[0] == "nvidia-smi":
            output = "NVIDIA-SMI 550.54 CUDA Version: 12.4\n" if nvidia_exit == 0 else ""
            return CommandResult(nvidia_exit, output)
        if command[0] == "nvcc":
            return CommandResult(0, "Cuda compilation tools, release 12.4, V12.4.99\n")
        if command[0] == "du":
            return CommandResult(0, f"{used_bytes}\tignored\n")
        raise AssertionError(command)

    return run, calls


def _collect(
    tmp_path: Path,
    *,
    environment: dict[str, str] | None = None,
    probe: SystemProbe | None = None,
    runner: preflight.CommandRunner | None = None,
) -> tuple[dict[str, object], int]:
    vea = tmp_path / "vea"
    repo = tmp_path / "repo"
    vea.mkdir(exist_ok=True)
    repo.mkdir(exist_ok=True)
    default_runner, _ = _runner()
    return collect_runtime_preflight(
        _template(),
        vea_root=vea.resolve(),
        repository_root=repo.resolve(),
        environment=(
            {
                "SLURM_JOB_ID": "4001",
                "SLURMD_NODENAME": "a100-node-1",
                "SLURM_JOB_PARTITION": "p1",
            }
            if environment is None
            else environment
        ),
        runner=runner or default_runner,
        probe=probe or _probe(),
        timestamp_utc=TIMESTAMP,
    )


def _direct_probe(
    *,
    torch_mode: str = "cuda",
    repository_exists: bool = True,
    hostname: str = "apple",
) -> SystemProbe:
    packages = {key: "1.2.3" for key in preflight.PACKAGE_KEYS}
    torch = {
        "import_status": "INSTALLED",
        "version": "2.8.0+cu128",
        "build": "b" * 40,
        "cuda_version": "12.8",
        "cuda_available": True,
        "device_count": 1,
        "device_names": ["NVIDIA GeForce RTX 5090"],
        "cudnn_version": "91002",
    }
    if torch_mode == "missing":
        packages = {key: NOT_INSTALLED for key in preflight.PACKAGE_KEYS}
        torch.update(
            import_status=NOT_INSTALLED,
            version=NOT_INSTALLED,
            build=NOT_AVAILABLE,
            cuda_version=NOT_AVAILABLE,
            cuda_available=False,
            device_count=0,
            device_names=[],
            cudnn_version=NOT_AVAILABLE,
        )
    elif torch_mode == "cpu":
        torch.update(
            version="2.8.0+cpu",
            cuda_version=NOT_AVAILABLE,
            cuda_available=False,
            device_count=0,
            device_names=[],
            cudnn_version=NOT_AVAILABLE,
        )
    return SystemProbe(
        os_name="Linux",
        kernel="6.8.0",
        architecture="x86_64",
        glibc="glibc-2.35",
        python_version="3.10.12",
        hostname=hostname,
        packages=packages,
        torch=torch,
        repository_mirror_exists=repository_exists,
    )


def _direct_runner(
    *,
    gpu_name: str = "NVIDIA GeForce RTX 5090",
    nvidia_exit: int = 0,
    cuda_compatibility: str | None = "13.0",
    include_gpu: bool = True,
    used_bytes: int = 80 * GIB,
) -> tuple[preflight.CommandRunner, list[tuple[str, ...]]]:
    calls: list[tuple[str, ...]] = []

    def run(args: object) -> CommandResult:
        command = tuple(args)  # type: ignore[arg-type]
        calls.append(command)
        if command[0] == "nvidia-smi" and len(command) > 1:
            output = (
                f"0, {gpu_name}, GPU-5090, 32607, 580.173.02\n"
                if nvidia_exit == 0 and include_gpu
                else ""
            )
            return CommandResult(nvidia_exit, output)
        if command[0] == "nvidia-smi":
            output = (
                f"NVIDIA-SMI 580.173.02 CUDA Version: {cuda_compatibility}\n"
                if nvidia_exit == 0 and cuda_compatibility is not None
                else "NVIDIA-SMI 580.173.02\n" if nvidia_exit == 0 else ""
            )
            return CommandResult(nvidia_exit, output)
        if command[0] == "du":
            return CommandResult(0, f"{used_bytes}\tignored\n")
        raise AssertionError(command)

    return run, calls


def _direct_collect(
    tmp_path: Path,
    *,
    environment: dict[str, str] | None = None,
    probe: SystemProbe | None = None,
    runner: preflight.CommandRunner | None = None,
    storage_space: preflight.StorageSpace | None = None,
) -> tuple[dict[str, object], int]:
    vea = tmp_path / "direct-vea"
    repository = tmp_path / "direct-repository"
    vea.mkdir(exist_ok=True)
    repository.mkdir(exist_ok=True)
    default_runner, _ = _direct_runner()
    return preflight.collect_direct_host_preflight(
        preflight.load_direct_host_template(DIRECT_TEMPLATE),
        vea_root=vea.resolve(),
        repository_root=repository.resolve(),
        environment={} if environment is None else environment,
        runner=runner or default_runner,
        probe=probe or _direct_probe(),
        storage_space=storage_space or preflight.StorageSpace(400 * GIB, 290 * GIB),
        timestamp_utc=TIMESTAMP,
    )


def _write_plan(tmp_path: Path, mutate: object) -> Path:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    mutate(value)  # type: ignore[operator]
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(value))
    return path


def _make_cache(root: Path, plan: dict[str, object], count: int = 3) -> None:
    for index, model in enumerate(plan["models"][:count]):  # type: ignore[index]
        snapshot = (
            root
            / ("models--" + model["model_id"].replace("/", "--"))
            / "snapshots"
            / model["revision"]
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_bytes(f"config-{index}".encode())
        (snapshot / "model.safetensors").write_bytes(f"weights-{index}".encode())


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "server-repository"
    repository.mkdir(exist_ok=True)
    return repository.resolve()


def test_exact_three_model_revision_plan_acceptance() -> None:
    plan = _plan()
    assert [model["registry_key"] for model in plan["models"]] == list(preflight.EXPECTED_MODEL_KEYS)


def test_runtime_module_import_isolated_from_optional_packages() -> None:
    code = """
import builtins
import sys
real_import = builtins.__import__
blocked = {'yaml', 'torch', 'transformers', 'tokenizers', 'huggingface_hub', 'safetensors', 'PIL', 'numpy'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise ModuleNotFoundError(f'blocked optional package: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import vision_encoder_alignment.p1_runtime_preflight as module
import scripts.collect_p1_bulk_runtime_preflight as collector
import scripts.validate_p1_direct_host_preflight as direct_validator
assert 'vision_encoder_alignment.config' not in sys.modules
assert 'yaml' not in sys.modules
assert collector.collect_runtime_preflight is module.collect_runtime_preflight
assert direct_validator.validate_direct_host_evidence is module.validate_direct_host_evidence
try:
    module.load_checkpoint_plan(sys.argv[1], sys.argv[2])
except module.PreflightError as exc:
    assert exc.classification == 'dependency_unavailable'
else:
    raise AssertionError('checkpoint plan unexpectedly loaded without PyYAML')
print('IMPORT_ISOLATION_OK')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(PLAN), str(REGISTRY)],
        cwd=ROOT,
        env=environment,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "IMPORT_ISOLATION_OK"
    assert "Traceback" not in result.stderr


def test_pyyaml_missing_is_collected_as_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def version(distribution: str) -> str:
        if distribution == "PyYAML":
            raise preflight.importlib.metadata.PackageNotFoundError(distribution)
        return "1.2.3"

    monkeypatch.setattr(preflight.importlib.metadata, "version", version)
    assert preflight._installed_packages()["pyyaml"] == NOT_INSTALLED


def test_floating_revision_rejected(tmp_path: Path) -> None:
    path = _write_plan(tmp_path, lambda value: value["models"][0].update(revision="main"))
    with pytest.raises(PreflightError, match="40-hex"):
        load_checkpoint_plan(path, REGISTRY)


def test_model_id_mismatch_rejected(tmp_path: Path) -> None:
    path = _write_plan(tmp_path, lambda value: value["models"][1].update(model_id="wrong/model"))
    with pytest.raises(PreflightError, match="registry mismatch"):
        load_checkpoint_plan(path, REGISTRY)


def test_revision_mismatch_rejected(tmp_path: Path) -> None:
    path = _write_plan(tmp_path, lambda value: value["models"][2].update(revision="0" * 40))
    with pytest.raises(PreflightError, match="registry mismatch"):
        load_checkpoint_plan(path, REGISTRY)


def test_duplicate_json_key_rejected() -> None:
    with pytest.raises(PreflightError, match="duplicate key"):
        strict_json_loads(b'{"a":1,"a":2}', label="fixture")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_rejected(constant: str) -> None:
    with pytest.raises(PreflightError, match="non-finite"):
        strict_json_loads(f'{{"value":{constant}}}'.encode(), label="fixture")


def test_canonical_json_byte_determinism() -> None:
    first = canonical_json_bytes({"z": 1, "a": [2, 3]})
    second = canonical_json_bytes({"a": [2, 3], "z": 1})
    assert first == second == b'{"a":[2,3],"z":1}\n'


@pytest.mark.parametrize("unsafe", ["/home/researcher/cache", r"C:\\Users\\person", "hf_1234567890abcdef", "token=secret"])
def test_path_username_and_token_like_values_rejected(tmp_path: Path, unsafe: str) -> None:
    with pytest.raises(PreflightError):
        write_json_output(tmp_path / "unsafe.json", {"value": unsafe})


def test_a100_allocation_success(tmp_path: Path) -> None:
    evidence, exit_code = _collect(tmp_path)
    assert exit_code == 0
    assert evidence["status"] == "A100_ALLOCATION_OBSERVED"
    assert evidence["collection_complete"] is True


def test_missing_slurm_job_id_is_valid_incomplete_collection(tmp_path: Path) -> None:
    evidence, exit_code = _collect(tmp_path, environment={})
    assert exit_code == 2
    assert evidence["status"] == "NO_CURRENT_A100_ALLOCATION"


def test_login_node_is_fail_closed(tmp_path: Path) -> None:
    evidence, exit_code = _collect(
        tmp_path,
        environment={"SLURM_JOB_ID": "4001", "SLURMD_NODENAME": "a100-node-1"},
        probe=_probe(hostname="login-1"),
    )
    assert exit_code == 2
    assert evidence["slurm"]["current_process_on_compute_node"] is False


def test_nvidia_smi_failure_on_compute_node_is_required_evidence_failure(tmp_path: Path) -> None:
    runner, _ = _runner(nvidia_exit=9)
    evidence, exit_code = _collect(tmp_path, runner=runner)
    assert exit_code == 4
    assert evidence["nvidia_smi"]["exit_code"] == 9


def test_non_a100_gpu_is_not_accepted(tmp_path: Path) -> None:
    runner, _ = _runner(gpu_name="NVIDIA H100 80GB HBM3")
    evidence, exit_code = _collect(tmp_path, runner=runner)
    assert exit_code == 4
    assert evidence["status"] == "NO_CURRENT_A100_ALLOCATION"


def test_pytorch_missing_is_incomplete(tmp_path: Path) -> None:
    evidence, exit_code = _collect(tmp_path, probe=_probe(torch_mode="missing"))
    assert exit_code == 4
    assert evidence["pytorch"]["import_status"] == NOT_INSTALLED


def test_cpu_only_pytorch_is_incomplete(tmp_path: Path) -> None:
    evidence, exit_code = _collect(tmp_path, probe=_probe(torch_mode="cpu"))
    assert exit_code == 4
    assert evidence["pytorch"]["cuda_available"] is False


def test_cuda_enabled_pytorch_is_recorded(tmp_path: Path) -> None:
    evidence, _ = _collect(tmp_path)
    assert evidence["pytorch"]["version"] == "2.4.1+cu124"
    assert evidence["pytorch"]["device_names"] == ["NVIDIA A100-SXM4-80GB"]


def test_ninety_gib_boundary_does_not_warn(tmp_path: Path) -> None:
    runner, _ = _runner(used_bytes=90 * GIB)
    evidence, exit_code = _collect(tmp_path, runner=runner)
    assert exit_code == 0
    assert evidence["storage"]["warning_threshold_exceeded"] is False


def test_hundred_gib_budget_exceeded(tmp_path: Path) -> None:
    runner, _ = _runner(used_bytes=100 * GIB + 1)
    evidence, exit_code = _collect(tmp_path, runner=runner)
    assert exit_code == 4
    assert evidence["storage"]["budget_exceeded"] is True


@pytest.mark.parametrize("field", ["driver_version", "cuda_compatibility"])
def test_complete_runtime_rejects_missing_driver_or_cuda(
    tmp_path: Path, field: str
) -> None:
    runtime, _ = _collect(tmp_path)
    runtime["nvidia_smi"][field] = NOT_AVAILABLE
    with pytest.raises(PreflightError, match="derived runtime completeness"):
        validate_runtime_evidence(runtime)


@pytest.mark.parametrize(
    ("used_bytes", "field", "tampered"),
    [
        (90 * GIB + 1, "warning_threshold_exceeded", False),
        (100 * GIB + 1, "budget_exceeded", False),
        (80 * GIB, "warning_threshold_exceeded", True),
        (80 * GIB, "budget_exceeded", True),
    ],
)
def test_storage_derived_flags_cannot_be_tampered(
    tmp_path: Path, used_bytes: int, field: str, tampered: bool
) -> None:
    runner, _ = _runner(used_bytes=used_bytes)
    runtime, _ = _collect(tmp_path, runner=runner)
    runtime["storage"][field] = tampered
    with pytest.raises(PreflightError, match="derived .* flag mismatch"):
        validate_runtime_evidence(runtime)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda torch: torch.update(cuda_available=False),
        lambda torch: torch.update(cuda_version=NOT_AVAILABLE),
        lambda torch: torch.update(device_count=0, device_names=[]),
        lambda torch: torch.update(cudnn_version=NOT_AVAILABLE),
        lambda torch: torch.update(import_status=NOT_INSTALLED),
    ],
)
def test_pytorch_state_contradictions_are_rejected(
    tmp_path: Path, mutation: object
) -> None:
    runtime, _ = _collect(tmp_path)
    mutation(runtime["pytorch"])  # type: ignore[operator]
    with pytest.raises(PreflightError):
        validate_runtime_evidence(runtime)


def test_repository_mirror_complete_marker_contradiction_rejected(
    tmp_path: Path,
) -> None:
    runtime, _ = _collect(tmp_path)
    runtime["repository"]["mirror_exists"] = False
    with pytest.raises(PreflightError, match="derived runtime completeness"):
        validate_runtime_evidence(runtime)


def test_tampered_runtime_cannot_obtain_readiness(tmp_path: Path) -> None:
    runtime, _ = _collect(tmp_path)
    runtime["nvidia_smi"]["driver_version"] = NOT_AVAILABLE
    cache = tmp_path / "cache"
    cache.mkdir()
    plan = _plan()
    inventory = inventory_checkpoint_cache(
        plan,
        cache_root=cache.resolve(),
        repository_root=_repository(tmp_path),
        timestamp_utc=TIMESTAMP,
    )
    with pytest.raises(PreflightError):
        decide_preflight(runtime, inventory, plan, timestamp_utc=TIMESTAMP)


def test_checkpoint_cache_miss(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    inventory = inventory_checkpoint_cache(_plan(), cache_root=cache.resolve(), repository_root=_repository(tmp_path), timestamp_utc=TIMESTAMP)
    assert inventory["status"] == "CACHE_MISSING"


def test_partial_checkpoint_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    plan = _plan()
    _make_cache(cache, plan, 1)
    inventory = inventory_checkpoint_cache(plan, cache_root=cache.resolve(), repository_root=_repository(tmp_path), timestamp_utc=TIMESTAMP)
    assert inventory["status"] == "PARTIAL"
    assert inventory["present_model_count"] == 1


def test_complete_cache_inventory_and_aggregate_hash(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    plan = _plan()
    _make_cache(cache, plan)
    repository = _repository(tmp_path)
    first = inventory_checkpoint_cache(plan, cache_root=cache.resolve(), repository_root=repository, timestamp_utc=TIMESTAMP)
    second = inventory_checkpoint_cache(plan, cache_root=cache.resolve(), repository_root=repository, timestamp_utc=TIMESTAMP)
    assert first["status"] == "COMPLETE"
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert all(len(model["aggregate_sha256"]) == 64 for model in first["models"])


def test_snapshot_inventory_allows_safe_hf_leaf_link_with_injected_probe(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"; cache.mkdir()
    model = _plan()["models"][0]
    snapshot = preflight._snapshot_path(cache, model["model_id"], model["revision"])
    snapshot.mkdir(parents=True)
    logical = snapshot / "model.safetensors"
    logical.write_bytes(b"link-name-only")
    blob = cache / "blobs" / "content-addressed"
    blob.parent.mkdir(); blob.write_bytes(b"checkpoint-bytes")

    files = preflight.checkpoint_snapshot_files(
        cache.resolve(), model_id=model["model_id"], revision=model["revision"],
        link_probe=lambda path: path == logical,
        strict_resolver=lambda path: blob if path == logical else path.resolve(strict=True),
    )

    assert files == [{
        "relative_path": "model.safetensors", "bytes": len(b"checkpoint-bytes"),
        "sha256": preflight.sha256_file(blob),
    }]


@pytest.mark.parametrize("target", ("broken", "directory", "escaping"))
def test_snapshot_inventory_rejects_unsafe_leaf_link_targets(
    tmp_path: Path, target: str,
) -> None:
    cache = tmp_path / "cache"; cache.mkdir()
    model = _plan()["models"][0]
    snapshot = preflight._snapshot_path(cache, model["model_id"], model["revision"])
    snapshot.mkdir(parents=True)
    logical = snapshot / "model.safetensors"; logical.write_bytes(b"link-name-only")
    directory = cache / "blobs" / "directory"; directory.mkdir(parents=True)
    outside = tmp_path / "outside"; outside.write_bytes(b"outside")

    def resolve(path: Path) -> Path:
        if path != logical:
            return path.resolve(strict=True)
        if target == "broken":
            raise FileNotFoundError("broken")
        return directory if target == "directory" else outside

    with pytest.raises(PreflightError):
        preflight.checkpoint_snapshot_files(
            cache.resolve(), model_id=model["model_id"], revision=model["revision"],
            link_probe=lambda path: path == logical, strict_resolver=resolve,
        )


def test_snapshot_inventory_rejects_linked_snapshot_directory_with_injected_probe(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"; cache.mkdir()
    model = _plan()["models"][0]
    snapshot = preflight._snapshot_path(cache, model["model_id"], model["revision"])
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"{}");
    with pytest.raises(PreflightError, match="linked directory"):
        preflight.checkpoint_snapshot_files(
            cache.resolve(), model_id=model["model_id"], revision=model["revision"],
            link_probe=lambda path: path == snapshot,
        )


def _complete_inventory(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    cache = tmp_path / "cache"
    cache.mkdir()
    plan = _plan()
    _make_cache(cache, plan)
    inventory = inventory_checkpoint_cache(
        plan,
        cache_root=cache.resolve(),
        repository_root=_repository(tmp_path),
        timestamp_utc=TIMESTAMP,
    )
    return plan, inventory


def test_empty_present_and_inventoried_status_rejected(tmp_path: Path) -> None:
    plan, inventory = _complete_inventory(tmp_path)
    model = inventory["models"][0]
    model["files"] = []
    model["file_count"] = 0
    model["total_bytes"] = 0
    model["aggregate_sha256"] = preflight.sha256_bytes(
        canonical_json_bytes([], trailing_lf=False)
    )
    with pytest.raises(PreflightError, match="non-empty"):
        validate_checkpoint_inventory(inventory, plan)


def test_nonempty_cache_missing_status_rejected(tmp_path: Path) -> None:
    plan, inventory = _complete_inventory(tmp_path)
    inventory["models"][0]["status"] = "CACHE_MISSING"
    with pytest.raises(PreflightError, match="empty zero-byte"):
        validate_checkpoint_inventory(inventory, plan)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda inventory: inventory["models"][0].update(file_count=99),
        lambda inventory: inventory.update(present_model_count=2),
        lambda inventory: inventory.update(missing_model_count=1),
        lambda inventory: inventory.update(status="PARTIAL"),
    ],
)
def test_inventory_count_and_status_tampering_rejected(
    tmp_path: Path, mutation: object
) -> None:
    plan, inventory = _complete_inventory(tmp_path)
    mutation(inventory)  # type: ignore[operator]
    with pytest.raises(PreflightError):
        validate_checkpoint_inventory(inventory, plan)


def test_tampered_checkpoint_detected_against_inventory(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    plan = _plan()
    _make_cache(cache, plan)
    repository = _repository(tmp_path)
    inventory = inventory_checkpoint_cache(plan, cache_root=cache.resolve(), repository_root=repository, timestamp_utc=TIMESTAMP)
    first = plan["models"][0]
    weight = cache / ("models--" + first["model_id"].replace("/", "--")) / "snapshots" / first["revision"] / "model.safetensors"
    weight.write_bytes(b"tampered")
    with pytest.raises(PreflightError, match="differs"):
        verify_inventory_against_cache(inventory, plan, cache_root=cache.resolve(), repository_root=repository)


def test_repository_internal_checkpoint_root_rejected(tmp_path: Path) -> None:
    cache = tmp_path / "repository" / "cache"
    cache.mkdir(parents=True)
    with pytest.raises(PreflightError, match="outside repository"):
        inventory_checkpoint_cache(_plan(), cache_root=cache.resolve(), repository_root=(tmp_path / "repository").resolve(), timestamp_utc=TIMESTAMP)


def test_symlink_or_reparse_root_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(preflight, "_is_link_or_reparse", lambda path: path == cache)
    with pytest.raises(PreflightError, match="linked cache root"):
        inventory_checkpoint_cache(_plan(), cache_root=cache, repository_root=ROOT.resolve(), timestamp_utc=TIMESTAMP)


def _runtime_lock(version: object = "3.12.4") -> dict[str, object]:
    return {
        "schema_id": preflight.RUNTIME_LOCK_SCHEMA_ID,
        "runtime_lanes": [
            {"lane_id": "primary_lane", "models": ["clip_vit_b16", "siglip2_base_p16_224"], "python_version": version, "packages": {"torch": "2.4.1+cu124"}},
            {"lane_id": "dinov3_isolated_lane", "models": ["dinov3_vit_b16"], "python_version": "3.12.4", "packages": {"torch": "2.4.1+cu124"}},
        ],
    }


def test_final_runtime_lock_accepts_exact_versions() -> None:
    assert validate_final_runtime_lock(_runtime_lock())["schema_id"] == preflight.RUNTIME_LOCK_SCHEMA_ID


@pytest.mark.parametrize("version", [None, "", "latest", "3.12.*", ">=3.12", "3.12,3.11"])
def test_final_runtime_lock_rejects_unresolved_or_mutable_versions(version: object) -> None:
    with pytest.raises(PreflightError):
        validate_final_runtime_lock(_runtime_lock(version))


def test_checkpoint_missing_does_not_block_t002b_authoring_when_runtime_complete(tmp_path: Path) -> None:
    runtime, _ = _collect(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    plan = _plan()
    inventory = inventory_checkpoint_cache(plan, cache_root=cache.resolve(), repository_root=_repository(tmp_path), timestamp_utc=TIMESTAMP)
    decision = decide_preflight(runtime, inventory, plan, timestamp_utc=TIMESTAMP)
    assert decision["decision"] == "READY_TO_AUTHOR_P1_T002B"
    assert decision["checkpoint_status"] == "CHECKPOINTS_MISSING"
    assert decision["ready_for_p1_extraction"] is False


def test_runtime_evidence_tamper_is_rejected(tmp_path: Path) -> None:
    runtime, _ = _collect(tmp_path)
    runtime["status"] = "A100_ALLOCATION_OBSERVED"
    runtime["slurm"]["job_id"] = NOT_AVAILABLE
    with pytest.raises(PreflightError, match="inconsistent"):
        validate_runtime_evidence(runtime)


def test_a100_v0_profile_and_decision_semantics_are_preserved(tmp_path: Path) -> None:
    template = _template()
    runtime, exit_code = _collect(tmp_path)
    assert template["schema_id"] == "vea.p1_bulk_runtime_preflight_request.v0"
    assert runtime["schema_id"] == "vea.p1_bulk_runtime_preflight.v0"
    assert runtime["status"] == "A100_ALLOCATION_OBSERVED"
    assert exit_code == 0


def test_apple_rtx5090_direct_host_evidence_succeeds(tmp_path: Path) -> None:
    evidence, exit_code = _direct_collect(tmp_path)
    assert exit_code == 0
    assert evidence["status"] == "DIRECT_HOST_EVIDENCE_COMPLETE"
    assert evidence["host_evidence_complete"] is True
    assert evidence["runtime_status"] == "RUNTIME_ENVIRONMENT_CUDA_READY"
    assert evidence["ready_to_author_apple_runtime_setup"] is True
    assert evidence["ready_for_p1_extraction"] is False


def test_direct_host_wrong_hostname_is_rejected(tmp_path: Path) -> None:
    evidence, exit_code = _direct_collect(
        tmp_path, probe=_direct_probe(hostname="not-apple")
    )
    assert exit_code == 2
    assert evidence["host_evidence_complete"] is False


def test_direct_host_non_rtx5090_gpu_is_rejected(tmp_path: Path) -> None:
    runner, _ = _direct_runner(gpu_name="NVIDIA A100-SXM4-80GB")
    evidence, exit_code = _direct_collect(tmp_path, runner=runner)
    assert exit_code == 2
    assert evidence["host_evidence_complete"] is False


def test_direct_host_slurm_environment_is_not_accepted_as_apple_evidence(
    tmp_path: Path,
) -> None:
    evidence, exit_code = _direct_collect(
        tmp_path, environment={"SLURM_JOB_ID": "5001"}
    )
    assert exit_code == 2
    assert evidence["scheduler"]["slurm_environment_present"] is True
    assert evidence["host_evidence_complete"] is False


def test_direct_host_nvidia_smi_failure_is_required_evidence_failure(
    tmp_path: Path,
) -> None:
    runner, _ = _direct_runner(nvidia_exit=9)
    evidence, exit_code = _direct_collect(tmp_path, runner=runner)
    assert exit_code == 4
    assert evidence["nvidia_smi"]["exit_code"] == 9
    assert evidence["host_evidence_complete"] is False


def test_direct_host_gpu_count_inventory_contradiction_is_rejected(
    tmp_path: Path,
) -> None:
    evidence, _ = _direct_collect(tmp_path)
    evidence["nvidia_smi"]["visible_gpu_count"] = 2
    with pytest.raises(PreflightError, match="GPU count mismatch"):
        preflight.validate_direct_host_evidence(
            evidence, preflight.load_direct_host_template(DIRECT_TEMPLATE)
        )


@pytest.mark.parametrize("missing", ["driver", "cuda"])
def test_direct_host_missing_driver_or_cuda_never_gets_readiness(
    tmp_path: Path, missing: str
) -> None:
    runner, _ = (
        _direct_runner(include_gpu=False)
        if missing == "driver"
        else _direct_runner(cuda_compatibility=None)
    )
    evidence, exit_code = _direct_collect(tmp_path, runner=runner)
    assert exit_code == 4
    assert evidence["status"] == "DIRECT_HOST_EVIDENCE_INCOMPLETE"
    assert evidence["ready_to_author_apple_runtime_setup"] is False


def test_direct_host_repository_state_tampering_is_rejected(tmp_path: Path) -> None:
    evidence, _ = _direct_collect(tmp_path)
    evidence["repository"]["root_exists"] = False
    with pytest.raises(PreflightError, match="derived"):
        preflight.validate_direct_host_evidence(
            evidence, preflight.load_direct_host_template(DIRECT_TEMPLATE)
        )


def test_direct_host_repository_primitive_must_match_filesystem(tmp_path: Path) -> None:
    evidence, _ = _direct_collect(tmp_path)
    with pytest.raises(PreflightError, match="filesystem state mismatch"):
        preflight.validate_direct_host_evidence(
            evidence,
            preflight.load_direct_host_template(DIRECT_TEMPLATE),
            repository_root=(tmp_path / "missing-repository").resolve(),
        )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [("warning_threshold_exceeded", True), ("budget_exceeded", True)],
)
def test_direct_host_storage_derived_flag_tampering_is_rejected(
    tmp_path: Path, field: str, tampered: bool
) -> None:
    evidence, _ = _direct_collect(tmp_path)
    evidence["storage"][field] = tampered
    with pytest.raises(PreflightError, match="derived .* flag mismatch"):
        preflight.validate_direct_host_evidence(
            evidence, preflight.load_direct_host_template(DIRECT_TEMPLATE)
        )


def test_direct_host_storage_over_budget_never_gets_readiness(tmp_path: Path) -> None:
    runner, _ = _direct_runner(used_bytes=100 * GIB + 1)
    evidence, exit_code = _direct_collect(tmp_path, runner=runner)
    assert exit_code == 2
    assert evidence["storage"]["budget_exceeded"] is True
    assert evidence["ready_to_author_apple_runtime_setup"] is False


def test_missing_runtime_packages_preserve_complete_host_evidence(
    tmp_path: Path,
) -> None:
    evidence, exit_code = _direct_collect(
        tmp_path, probe=_direct_probe(torch_mode="missing")
    )
    assert exit_code == 0
    assert evidence["host_evidence_complete"] is True
    assert evidence["runtime_status"] == "RUNTIME_ENVIRONMENT_MISSING"
    assert evidence["ready_to_author_apple_runtime_setup"] is True


def test_cpu_only_pytorch_is_not_extraction_or_cuda_ready(tmp_path: Path) -> None:
    evidence, exit_code = _direct_collect(
        tmp_path, probe=_direct_probe(torch_mode="cpu")
    )
    assert exit_code == 0
    assert evidence["host_evidence_complete"] is True
    assert evidence["runtime_status"] == "RUNTIME_ENVIRONMENT_NOT_CUDA_READY"
    assert evidence["ready_for_p1_extraction"] is False


def test_cpu_only_pytorch_cannot_be_marked_cuda_ready(tmp_path: Path) -> None:
    evidence, _ = _direct_collect(tmp_path, probe=_direct_probe(torch_mode="cpu"))
    evidence["runtime_status"] = "RUNTIME_ENVIRONMENT_CUDA_READY"
    with pytest.raises(PreflightError, match="derived runtime status mismatch"):
        preflight.validate_direct_host_evidence(
            evidence, preflight.load_direct_host_template(DIRECT_TEMPLATE)
        )


def test_coherent_cuda_pytorch_is_classified_separately(tmp_path: Path) -> None:
    evidence, _ = _direct_collect(tmp_path)
    assert evidence["runtime_status"] == "RUNTIME_ENVIRONMENT_CUDA_READY"
    assert evidence["ready_for_p1_extraction"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "DIRECT_HOST_EVIDENCE_INCOMPLETE"),
        ("host_evidence_complete", False),
        ("runtime_status", "RUNTIME_ENVIRONMENT_MISSING"),
        ("ready_to_author_apple_runtime_setup", False),
    ],
)
def test_direct_host_caller_derived_state_tampering_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    evidence, _ = _direct_collect(tmp_path)
    evidence[field] = value
    with pytest.raises(PreflightError, match="derived"):
        preflight.validate_direct_host_evidence(
            evidence, preflight.load_direct_host_template(DIRECT_TEMPLATE)
        )


def test_direct_host_noncanonical_json_is_rejected(tmp_path: Path) -> None:
    evidence, _ = _direct_collect(tmp_path)
    path = tmp_path / "noncanonical-direct.json"
    path.write_bytes(canonical_json_bytes(evidence) + b"\n")
    with pytest.raises(PreflightError, match="canonical"):
        preflight.load_canonical_json(path, label="direct_host")
