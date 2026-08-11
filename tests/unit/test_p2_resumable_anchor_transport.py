from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as rat


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/vlm_compat/p2_universal_anchor_transport_v0.json"
MANIFEST_PATH = ROOT / "data/manifests/p1_1000.jsonl"
BASELINE = "e9afe5820dd03149811f06f0f0daa374d1c459e2"


def config():
    return rat.load_config(CONFIG_PATH)


def manifest():
    return rat.read_manifest(MANIFEST_PATH, config()["manifest"]["sha256"])


def plan(tmp_path: Path):
    roots = {name: tmp_path / name for name in ("dataset", "cache", "repo", "dino_runtime", "dino_snapshot", "run")}
    for name, path in roots.items():
        if name != "run":
            path.mkdir(exist_ok=True)
    return rat.build_run_plan(
        config(), manifest(), repository_commit=BASELINE,
        config_sha256=rat.sha256_file(CONFIG_PATH), manifest_sha256=rat.sha256_file(MANIFEST_PATH),
        dataset_root=str(roots["dataset"]), cache_root=str(roots["cache"]),
        repository_root=str(roots["repo"]), dino_runtime_root=str(roots["dino_runtime"]),
        dino_snapshot_root=str(roots["dino_snapshot"]), run_root=str(roots["run"]),
        device="cuda:0", storage_available_bytes=None,
    )


def mini_plan(tmp_path: Path, stage: str = "05_TRANSPORT_AND_CONNECTOR_VALIDATION"):
    value = plan(tmp_path)
    chosen = next(unit for unit in value["expected_units"] if unit["stage"] == stage)
    value["expected_units"] = [chosen]
    value["expected_unit_count"] = 1
    return value


def initialized(tmp_path: Path, stage: str = "05_TRANSPORT_AND_CONNECTOR_VALIDATION"):
    value = mini_plan(tmp_path, stage)
    root = tmp_path / "run"
    plan_id = rat.initialize_run(root, value)
    return root, value, plan_id, value["expected_units"][0]


def patch(value: float, count: int = 1):
    return np.full((count, 196, 768), value, dtype=np.float32)


def coordinator_module(name="p2_coordinator"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/run_resumable_anchor_transport.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_config_validation():
    value = config()
    assert value["schema_id"] == "vea.p2.universal_anchor_transport_config.v0"
    assert tuple(value["condition_order"]) == rat.CONDITIONS
    rat.validate_identity_sources(value, ROOT)


def test_config_rejects_condition_mutation(tmp_path):
    value = config()
    value["condition_order"] = list(reversed(value["condition_order"]))
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(rat.ContractError):
        rat.load_config(path)


def test_plan_is_deterministic(tmp_path):
    first = plan(tmp_path)
    second = plan(tmp_path)
    assert rat.canonical_json_bytes(first) == rat.canonical_json_bytes(second)
    assert rat.sha256_bytes(rat.canonical_json_bytes(first)) == rat.sha256_bytes(rat.canonical_json_bytes(second))


def test_plan_id_binds_canonical_bytes(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    assert rat.sha256_file(root / "run_plan.json") == plan_id


def test_plan_mismatch_rejected(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    (root / "run_plan.json").write_bytes((root / "run_plan.json").read_bytes() + b" ")
    with pytest.raises(rat.PlanMismatchError):
        rat.load_bound_plan(root)


def test_exact_anchor_selection(tmp_path):
    value = plan(tmp_path)
    expected = [row["image_id"] for row in sorted(
        (row for row in manifest() if row["split"] == "train"), key=lambda row: row["selection_index"]
    )[:128]]
    assert value["anchor_image_ids"] == expected


def test_anchor_and_evaluation_are_disjoint(tmp_path):
    value = plan(tmp_path)
    evaluation = set(value["evaluation_image_ids"]["validation"] + value["evaluation_image_ids"]["test"])
    assert set(value["anchor_image_ids"]).isdisjoint(evaluation)


def test_manifest_order_is_preserved(tmp_path):
    value = plan(tmp_path)
    assert [row["image_id"] for row in value["manifest_rows"]] == [row["image_id"] for row in manifest()]


def test_feature_chunks_are_sixteen_with_partial_tail(tmp_path):
    units = [unit for unit in plan(tmp_path)["expected_units"] if unit["stage"] == "01_DINOV3_FEATURE_BANK"]
    assert len(units) == 63
    assert [len(unit["image_ids"]) for unit in units[:2]] == [16, 16]
    assert len(units[-1]["image_ids"]) == 8


def test_storage_estimate_has_exact_margin():
    value = rat.estimate_storage_bytes(1000, 200, 1900)
    assert value["required_with_30_percent_margin_bytes"] == int(np.ceil(value["minimum_bytes"] * 1.30))
    assert value["same_directory_partial_allowance_bytes"] > 0


def test_storage_estimate_covers_independently_calculated_persisted_payload():
    f32 = 4
    patch_bytes = 196 * 768 * f32
    global_bytes = 768 * f32
    connector_bytes = 196 * 1536 * f32
    image_id_bytes = 96 * 4
    feature_payload = 1000 * (3 * patch_bytes + 3 * global_bytes + connector_bytes)
    feature_payload += 1000 * 3 * image_id_bytes
    anchor_payload = 128 * (4 * patch_bytes + 4 * global_bytes + connector_bytes)
    anchor_payload += 128 * 4 * image_id_bytes
    stage05_payload = 200 * 3 * 6 * 2 * (patch_bytes + connector_bytes)
    stage05_payload += 200 * 3 * 6 * image_id_bytes
    stage05_payload += 252 * 32768 * 4
    generation_payload = 1900 * 16 * 1024
    independently_calculated = feature_payload + anchor_payload + stage05_payload + generation_payload
    observed = rat.estimate_storage_bytes(1000, 200, 1900)
    assert observed["durable_payload_bytes"] >= independently_calculated
    assert observed["required_with_30_percent_margin_bytes"] > observed["previous_underestimated_required_bytes"]


def test_storage_old_pass_new_fail_boundary(tmp_path):
    estimate = rat.estimate_storage_bytes(1000, 200, 1900)
    available = (
        estimate["previous_underestimated_required_bytes"]
        + estimate["required_with_30_percent_margin_bytes"]
    ) // 2
    assert available > estimate["previous_underestimated_required_bytes"]
    with pytest.raises(rat.ContractError, match="insufficient run-root storage"):
        rat.build_run_plan(
            config(), manifest(), repository_commit=BASELINE,
            config_sha256=rat.sha256_file(CONFIG_PATH), manifest_sha256=rat.sha256_file(MANIFEST_PATH),
            dataset_root=str(tmp_path / "dataset"), cache_root=str(tmp_path / "cache"),
            repository_root=str(tmp_path / "repo"), dino_runtime_root=str(tmp_path / "dino_runtime"),
            dino_snapshot_root=str(tmp_path / "dino_snapshot"), run_root=str(tmp_path / "run"),
            device="fake-cpu", storage_available_bytes=available,
        )


@pytest.mark.parametrize("encoder,tap", [("TARGET_CLIP", "block_11"), ("SIGLIP2", "final_norm"), ("DINOV3", "block_11")])
def test_encoder_identity_and_tap_contract(encoder, tap):
    spec = config()["encoders"][encoder]
    assert spec["tap_name"] == tap
    assert spec["patch_shape"] == [196, 768]
    assert len(spec["revision"]) == 40


@pytest.mark.parametrize("bad", [np.zeros((196, 768), np.float32), np.zeros((1, 196, 767), np.float32), np.zeros((1, 196, 768), np.float64)])
def test_patch_shape_and_dtype_fail_closed(bad):
    with pytest.raises(rat.ContractError):
        rat.validate_patch_batch(bad, name="bad")


def test_patch_requires_finite_contiguous_writable():
    bad = patch(1.0)
    bad[0, 0, 0] = np.nan
    with pytest.raises(rat.ContractError):
        rat.validate_patch_batch(bad, name="bad")
    readonly = patch(1.0)
    readonly.flags.writeable = False
    with pytest.raises(rat.ContractError):
        rat.validate_patch_batch(readonly, name="bad")


def test_sample_rms_is_per_patch_and_anchor_free():
    source = patch(2.0)
    source[:, 1, :] = 4.0
    value = rat.sample_rms(source, epsilon=1e-6)
    assert np.allclose(value[:, 0, :], 2.0 / np.sqrt(4.0 + 1e-6))
    assert np.allclose(value[:, 1, :], 4.0 / np.sqrt(16.0 + 1e-6))


def test_global_1nn_tie_breaks_to_lower_anchor():
    source = patch(1.0)
    anchors = np.concatenate([patch(1.0), patch(1.0)])
    target = np.concatenate([patch(3.0), patch(7.0)])
    assert np.array_equal(rat.global_anchor_transport(source, anchors, target, top_k=1), patch(3.0))


def test_global_top8_uses_stable_unscaled_softmax():
    source = patch(1.0)
    anchors = np.concatenate([patch(1.0) for _ in range(8)])
    target = np.concatenate([patch(float(index)) for index in range(8)])
    result = rat.global_anchor_transport(source, anchors, target, top_k=8)
    assert np.allclose(result, patch(3.5))


def test_positional_1nn_uses_same_position():
    source = patch(1.0)
    anchors = np.concatenate([patch(1.0), patch(-1.0)])
    target = np.concatenate([patch(2.0), patch(9.0)])
    assert np.array_equal(rat.positional_anchor_transport(source, anchors, target, top_k=1), patch(2.0))


def test_positional_top8_stable_weights():
    source = patch(1.0)
    anchors = np.concatenate([patch(1.0) for _ in range(8)])
    target = np.concatenate([patch(float(index)) for index in range(8)])
    assert np.allclose(rat.positional_anchor_transport(source, anchors, target, top_k=8), patch(3.5))


def test_all_encoders_share_one_condition_dispatch():
    source = patch(1.0)
    anchors = np.concatenate([patch(1.0) for _ in range(8)])
    target = np.concatenate([patch(2.0) for _ in range(8)])
    results = [rat.apply_transport_condition("GLOBAL_TOP8", source, anchors, target) for _ in rat.ENCODERS]
    assert all(np.array_equal(results[0], value) for value in results[1:])


def test_raw_target_self_control_is_exact():
    source = np.arange(196 * 768, dtype=np.float32).reshape(1, 196, 768).copy()
    result = rat.apply_transport_condition("RAW", source, patch(0.0), patch(0.0))
    assert result is not source and np.array_equal(result, source)


def test_core_contains_no_fitting_primitives():
    source = (ROOT / "src/vision_encoder_alignment/vlm_compat/resumable_anchor_transport.py").read_text(encoding="utf-8").lower()
    assert "np.linalg.svd" not in source
    assert "torch.optim" not in source
    assert "requires_grad" not in source


def test_atomic_artifact_then_marker(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    marker = rat.atomic_publish_unit(root, unit, b"payload", plan_id, elapsed_seconds=1.25)
    assert rat.validate_unit_completion(root, unit, plan_id)
    assert marker["artifact_sha256"] == rat.sha256_bytes(b"payload")
    assert marker["artifact_bytes"] == 7


def test_injected_validation_failure_leaves_no_marker(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    with pytest.raises(RuntimeError):
        rat.atomic_publish_unit(
            root, unit, b"bad", plan_id, elapsed_seconds=0,
            validate_payload=lambda payload, spec: (_ for _ in ()).throw(RuntimeError("write validation")),
        )
    assert not rat.marker_path(root, unit["unit_id"]).exists()
    assert not (root / unit["artifact_relative_path"]).exists()


def test_valid_completed_unit_is_skipped(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    first = rat.atomic_publish_unit(root, unit, b"first", plan_id, elapsed_seconds=0)
    second = rat.atomic_publish_unit(root, unit, b"different", plan_id, elapsed_seconds=0)
    assert first == second
    assert (root / unit["artifact_relative_path"]).read_bytes() == b"first"


def test_orphan_partial_is_quarantined(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    artifact = root / unit["artifact_relative_path"]
    artifact.parent.mkdir(parents=True, exist_ok=True)
    orphan = artifact.with_name(artifact.name + ".partial.999")
    orphan.write_bytes(b"partial")
    moved = rat.recover_orphans(root, value, plan_id)
    assert moved and not orphan.exists()


def test_artifact_without_marker_is_quarantined(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    artifact = root / unit["artifact_relative_path"]
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"orphan")
    assert rat.recover_orphans(root, value, plan_id)
    assert not artifact.exists()


def test_marker_without_artifact_is_corruption(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    rat.atomic_publish_unit(root, unit, b"payload", plan_id, elapsed_seconds=0)
    (root / unit["artifact_relative_path"]).unlink()
    with pytest.raises(rat.CorruptionError):
        rat.validate_unit_completion(root, unit, plan_id)


def test_hash_mismatch_is_corruption(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    rat.atomic_publish_unit(root, unit, b"payload", plan_id, elapsed_seconds=0)
    (root / unit["artifact_relative_path"]).write_bytes(b"changed")
    with pytest.raises(rat.CorruptionError):
        rat.validate_unit_completion(root, unit, plan_id)


def test_resume_binding_rejects_root_change(tmp_path):
    value = plan(tmp_path)
    with pytest.raises(rat.PlanMismatchError):
        rat.validate_resume_bindings(
            value, repository_commit=BASELINE, config_sha256=value["config_sha256"],
            manifest_sha256=value["manifest_sha256"],
            roots={**value["roots"], "dataset_root": str(tmp_path / "different")}, device="cuda:0",
        )


def test_optional_reuse_root_null_matches_null(tmp_path):
    value = plan(tmp_path)
    assert value["roots"]["reuse_root"] is None
    rat.validate_resume_bindings(
        value,
        repository_commit=BASELINE,
        config_sha256=value["config_sha256"],
        manifest_sha256=value["manifest_sha256"],
        roots=dict(value["roots"]),
        device="cuda:0",
    )


def test_explicit_reuse_root_requires_exact_path(tmp_path):
    reuse = tmp_path / "reuse"
    reuse.mkdir()
    value = rat.build_run_plan(
        config(), manifest(), repository_commit=BASELINE,
        config_sha256=rat.sha256_file(CONFIG_PATH), manifest_sha256=rat.sha256_file(MANIFEST_PATH),
        dataset_root=str(tmp_path / "dataset"), cache_root=str(tmp_path / "cache"),
        repository_root=str(tmp_path / "repo"), dino_runtime_root=str(tmp_path / "dino_runtime"),
        dino_snapshot_root=str(tmp_path / "dino_snapshot"), run_root=str(tmp_path / "run"),
        device="fake-cpu", storage_available_bytes=None, reuse_root=str(reuse),
    )
    rat.validate_resume_bindings(
        value, repository_commit=BASELINE, config_sha256=value["config_sha256"],
        manifest_sha256=value["manifest_sha256"], roots=dict(value["roots"]), device="fake-cpu",
    )
    for observed in (None, str(tmp_path / "different")):
        roots = dict(value["roots"])
        roots["reuse_root"] = observed
        with pytest.raises(rat.PlanMismatchError):
            rat.validate_resume_bindings(
                value, repository_commit=BASELINE, config_sha256=value["config_sha256"],
                manifest_sha256=value["manifest_sha256"], roots=roots, device="fake-cpu",
            )


def test_all_existing_plan_commands_validate_without_reuse_root(tmp_path, monkeypatch):
    value = plan(tmp_path)
    root = tmp_path / "run"
    rat.initialize_run(root, value)
    coordinator = coordinator_module("p2_coordinator_no_reuse")
    monkeypatch.setattr(
        coordinator,
        "_bindings",
        lambda args, core: (
            config(), manifest(), BASELINE, value["config_sha256"], value["manifest_sha256"]
        ),
    )
    monkeypatch.setattr(coordinator, "_validated_runtime_bindings", lambda device: None)
    roots = value["roots"]
    for command in ("run", "resume", "verify", "status", "finalize"):
        args = SimpleNamespace(
            command=command,
            config=CONFIG_PATH,
            manifest=MANIFEST_PATH,
            dataset_root=Path(roots["dataset_root"]),
            cache_root=Path(roots["cache_root"]),
            repository_root=Path(roots["repository_root"]),
            dinov3_runtime_root=Path(roots["dino_runtime_root"]),
            dinov3_snapshot_root=Path(roots["dino_snapshot_root"]),
            run_root=root,
            device="cuda:0",
            reuse_root=None,
        )
        assert coordinator._validate_existing(args, rat)[1] == rat.sha256_file(root / "run_plan.json")


def test_stage04_units_bind_exact_ordered_anchors(tmp_path):
    value = plan(tmp_path)
    units = [unit for unit in value["expected_units"] if unit["stage"] == "04_ANCHOR_REGISTRY"]
    assert len(units) == 4
    assert all(unit["image_ids"] == value["anchor_image_ids"] for unit in units)
    assert value["expected_unit_count"] == 2345


def test_exclusive_live_lock(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    first = rat.CoordinatorLock(root, plan_id, identity_provider=lambda pid: "start", liveness_provider=lambda h, p, s: True)
    first.acquire()
    try:
        second = rat.CoordinatorLock(root, plan_id, identity_provider=lambda pid: "other", liveness_provider=lambda h, p, s: True)
        with pytest.raises(rat.LockBusyError):
            second.acquire()
    finally:
        first.release()
    assert not (root / "coordinator.lock").exists()


def test_proven_stale_lock_is_quarantined(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    lock = root / "coordinator.lock"
    lock.write_bytes(rat.canonical_json_bytes({
        "schema_id": rat.LOCK_SCHEMA, "plan_id": plan_id, "host": socket.gethostname(),
        "pid": 999999, "process_start_identity": "dead", "started_at_utc": "now",
    }))
    owner = rat.CoordinatorLock(root, plan_id, identity_provider=lambda pid: "new", liveness_provider=lambda h, p, s: False)
    owner.acquire()
    owner.release()
    assert list((root / "failures/quarantine").glob("proven_stale_lock*"))


def test_ambiguous_lock_is_not_deleted(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    lock = root / "coordinator.lock"
    lock.write_bytes(rat.canonical_json_bytes({
        "schema_id": rat.LOCK_SCHEMA, "plan_id": plan_id, "host": "remote",
        "pid": 2, "process_start_identity": "unknown", "started_at_utc": "now",
    }))
    owner = rat.CoordinatorLock(root, plan_id, identity_provider=lambda pid: "new", liveness_provider=lambda h, p, s: None)
    with pytest.raises(rat.AmbiguousLockError):
        owner.acquire()
    assert lock.exists()


def test_progress_is_append_only_canonical_jsonl(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    before = (root / "progress.jsonl").read_bytes()
    rat.append_event(root, rat.progress_event(plan_id, "RUN_STARTED"))
    after = (root / "progress.jsonl").read_bytes()
    assert after.startswith(before) and after.count(b"\n") == before.count(b"\n") + 1
    assert json.loads(after.splitlines()[-1])["plan_id"] == plan_id


def test_retryable_failure_is_exit_75():
    record = rat.failure_record(rat.IncompleteRunError("resource"), "p", "resume now")
    assert record["retryable"] and record["exit_code"] == 75


def test_contract_failure_is_nonretryable():
    record = rat.failure_record(rat.ContractError("shape"), "p", "resume now")
    assert not record["retryable"] and record["exit_code"] == 1


def test_exact_resume_command_quotes_paths():
    command = rat.format_recovery_command(["python", "run script.py"], ["--run-root", "a b"])
    assert "resume" in command and '"run script.py"' in command and '"a b"' in command


def test_generation_inventory_is_per_image_condition(tmp_path):
    units = [unit for unit in plan(tmp_path)["expected_units"] if unit["stage"] == "06_HELD_OUT_GENERATION"]
    assert len(units) == 1900
    assert all(len(unit["image_ids"]) == 1 for unit in units)
    assert len({unit["unit_id"] for unit in units}) == 1900


def test_expected_unit_and_generation_counts(tmp_path):
    value = plan(tmp_path)
    assert value["expected_unit_count"] == 2345
    assert value["expected_generation_count"] == 100 * (1 + 3 * 6)


def test_fake_main_stage_loads_once(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path, "02_SIGLIP2_FEATURE_BANK")
    spec = importlib.util.spec_from_file_location("p2_worker", ROOT / "scripts/p2_anchor_transport_stage_worker.py")
    worker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(worker)
    loaded = []
    class Model:
        def close(self):
            loaded.append("closed")
    result = worker.execute_stage(
        rat, value, plan_id, root, "02_SIGLIP2_FEATURE_BANK",
        fake_loader=lambda stage: (loaded.append(stage) or Model()),
        fake_processor=lambda model, unit: rat.deterministic_fake_payload(plan_id, unit),
    )
    assert result["model_load_count"] == 1
    assert result["stage_worker_invocation_count"] == 1
    assert loaded == ["02_SIGLIP2_FEATURE_BANK", "closed"]


def test_fake_dino_63_chunks_count_63_model_loads(tmp_path):
    value = plan(tmp_path)
    value["expected_units"] = [
        unit for unit in value["expected_units"] if unit["stage"] == "01_DINOV3_FEATURE_BANK"
    ]
    value["expected_unit_count"] = len(value["expected_units"])
    root = tmp_path / "run"
    plan_id = rat.initialize_run(root, value)
    worker_spec = importlib.util.spec_from_file_location(
        "p2_worker_dino_63", ROOT / "scripts/p2_anchor_transport_stage_worker.py"
    )
    worker = importlib.util.module_from_spec(worker_spec)
    assert worker_spec.loader is not None
    worker_spec.loader.exec_module(worker)
    result = worker.execute_stage(
        rat, value, plan_id, root, "01_DINOV3_FEATURE_BANK",
        fake_loader=lambda stage: object(),
        fake_processor=lambda model, unit: rat.deterministic_fake_payload(plan_id, unit),
    )
    assert result["completed_new"] == 63
    assert result["model_load_count"] == 63
    assert result["stage_worker_invocation_count"] == 1
    assert result["extractor_worker_invocation_count"] == 63


def test_stage_subprocess_environment_isolation_is_injected():
    coordinator = coordinator_module("p2_coordinator_environment")
    binding = {
        "main_python": coordinator.MAIN_PY,
        "main_runtime_root": coordinator.MAIN_RUNTIME_ROOT,
        "main_overlay": coordinator.MAIN_OVERLAY,
        "cache_bindings": dict(coordinator.APPROVED_CACHE_BINDINGS),
        "offline_environment": dict(coordinator.REQUIRED_OFFLINE_ENVIRONMENT),
        "package_versions": {name: "test" for name in coordinator.METADATA_PACKAGES},
    }
    value = {"runtime_bindings": binding}
    base = {
        "PYTHONPATH": os.pathsep.join((coordinator.MAIN_OVERLAY, "/approved/other")),
        **coordinator.APPROVED_CACHE_BINDINGS,
        **coordinator.REQUIRED_OFFLINE_ENVIRONMENT,
    }
    captured = []
    def runner(command, *, check, env):
        captured.append(env)
        return SimpleNamespace(returncode=0)
    coordinator._launch_stage_worker(
        ["dino-python", "worker"], value, "01_DINOV3_FEATURE_BANK",
        runner=runner, base_environment=base,
    )
    coordinator._launch_stage_worker(
        ["main-python", "worker"], value, "02_SIGLIP2_FEATURE_BANK",
        runner=runner, base_environment=base,
    )
    assert coordinator.MAIN_OVERLAY not in captured[0].get("PYTHONPATH", "")
    assert coordinator.MAIN_OVERLAY in captured[1]["PYTHONPATH"].split(os.pathsep)
    assert captured[0]["HF_HUB_OFFLINE"] == captured[1]["HF_HUB_OFFLINE"] == "1"


def test_approved_runtime_binding_uses_metadata_only():
    coordinator = coordinator_module("p2_coordinator_runtime")
    environment = {
        "PYTHONPATH": coordinator.MAIN_OVERLAY,
        **coordinator.APPROVED_CACHE_BINDINGS,
        **coordinator.REQUIRED_OFFLINE_ENVIRONMENT,
    }
    result = coordinator._validated_runtime_bindings(
        "cuda:0",
        executable=coordinator.MAIN_PY,
        environment=environment,
        is_file=lambda path: True,
        is_directory=lambda path: True,
        metadata_version=lambda package: f"metadata-{package}",
    )
    assert result["main_python"] == coordinator.MAIN_PY
    assert result["main_overlay"] == coordinator.MAIN_OVERLAY


def _parent_shell_without_generated_children(runbook: str) -> str:
    retained = []
    terminator = None
    for line in runbook.splitlines():
        if terminator is not None:
            if line == terminator:
                terminator = None
            continue
        if "<<'VEA_GPU_GATE'" in line:
            terminator = "VEA_GPU_GATE"
            retained.append(line)
            continue
        if "<<'VEA_T001D_SUPERVISOR'" in line:
            terminator = "VEA_T001D_SUPERVISOR"
            retained.append(line)
            continue
        retained.append(line)
    return "\n".join(retained)


def test_runbook_persistent_runtime_and_parent_safety():
    text = (ROOT / "reports/runbooks/VEA-P2-T001D-apple-resumable-anchor-transport.md").read_text(encoding="utf-8")
    assert "/home/chltjsdh/vea/p2_repos" in text
    assert "/home/chltjsdh/vea/p2_runs" in text
    assert "/tmp" not in text
    assert "MAIN_PY='/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3'" in text
    assert '"$MAIN_PY" "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py"' in text
    assert "bash -n \"$GPU_GATE_SCRIPT\"" in text and "bash -n \"$SUPERVISOR\"" in text
    assert "nohup setsid bash" in text
    parent = _parent_shell_without_generated_children(text)
    for forbidden in ("set -e", "set -u", "set -o pipefail", "\nexit ", "\nexec ", "trap '", "tmux new", "tmux attach"):
        assert forbidden not in parent


def test_runbook_resource_resume_and_pipeline_status_contract():
    text = (ROOT / "reports/runbooks/VEA-P2-T001D-apple-resumable-anchor-transport.md").read_text(encoding="utf-8")
    assert 'PIPE_CODES=("${PIPESTATUS[@]}")' in text
    assert 'WORKER_EXIT_CODE="${PIPE_CODES[0]}"' in text
    assert 'TEE_EXIT_CODE="${PIPE_CODES[1]}"' in text
    assert '[ "$WORKER_EXIT_CODE" -eq 75 ] && [ "$RESOURCE_RESUMES" -lt 3 ]' in text
    assert '[ "$WORKER_EXIT_CODE" -eq 130 ]' in text
    assert "FINAL_VERIFY_EXIT_CODE" in text
    assert "EXACT_MANUAL_RESUME=%q %q resume" in text


def test_signal_flag_stops_only_after_atomic_unit(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path, "01_DINOV3_FEATURE_BANK")
    spec = importlib.util.spec_from_file_location("p2_worker_signal", ROOT / "scripts/p2_anchor_transport_stage_worker.py")
    worker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(worker)

    def process(model, current):
        worker._STOP_REQUESTED = True
        return rat.deterministic_fake_payload(plan_id, current)

    with pytest.raises(worker.StageInterrupted):
        worker.execute_stage(
            rat, value, plan_id, root, "01_DINOV3_FEATURE_BANK",
            fake_loader=lambda stage: object(), fake_processor=process,
        )
    assert rat.validate_unit_completion(root, unit, plan_id)
    events = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    assert events[-1]["event_type"] == "INTERRUPTED"


def test_representation_metric_definitions():
    candidate, reference = patch(2.0), patch(1.0)
    metrics = rat.representation_metrics(candidate, reference, patch(0.0, 2))
    assert metrics["flattened_cosine_mean"] == pytest.approx(1.0)
    assert metrics["norm_ratio_mean"] == pytest.approx(2.0)
    assert metrics["nmse_mean"] == pytest.approx(1.0)


def test_same_image_retrieval_r1_r5():
    values = np.zeros((6, 196, 768), dtype=np.float32)
    for index in range(6):
        values[index, :, index] = 1.0
    metrics = rat.retrieval_recall(values.copy(), values.copy())
    assert metrics == {"recall_at_1": 1.0, "recall_at_5": 1.0}


def test_token_prefix_lcs_and_unigram_f1():
    metrics = rat.generation_metrics([1, 2, 4], [1, 2, 3])
    assert metrics["common_token_prefix_length"] == 2
    assert metrics["token_LCS_length"] == 2
    assert metrics["token_unigram_F1"] == pytest.approx(2 / 3)


def test_finalization_refuses_incomplete_plan(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    with pytest.raises(rat.IncompleteRunError):
        rat.finalize_run(root)


def test_exact_final_artifact_inventory(tmp_path):
    root, value, plan_id, unit = initialized(tmp_path)
    rat.atomic_publish_unit(root, unit, rat.deterministic_fake_payload(plan_id, unit), plan_id, elapsed_seconds=0)
    rat.finalize_run(root)
    assert {path.name for path in (root / "final").iterdir()} == set(rat.FINAL_ARTIFACTS)


def test_module_import_does_not_load_heavy_runtimes():
    code = (
        "import sys; import vision_encoder_alignment.vlm_compat.resumable_anchor_transport; "
        "print(any(x in sys.modules for x in ['torch','transformers','PIL','safetensors']))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        capture_output=True, text=True, check=True,
    )
    assert process.stdout.strip() == "False"


@pytest.mark.parametrize("script", ["scripts/run_resumable_anchor_transport.py", "scripts/p2_anchor_transport_stage_worker.py"])
def test_cli_help_is_import_safe(script):
    process = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0 and "usage:" in process.stdout.lower()


def test_no_real_runtime_execution_in_unit_suite(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert rat.estimate_storage_bytes(1000, 200, 1900)["minimum_bytes"] > 0
