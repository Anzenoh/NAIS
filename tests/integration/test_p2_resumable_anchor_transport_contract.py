from __future__ import annotations

import json
import importlib.util
import io
import shutil
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as rat


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/vlm_compat/p2_universal_anchor_transport_v0.json"
MANIFEST = ROOT / "data/manifests/p1_1000.jsonl"
BASELINE = "9f488cc46f1711c3f43e8ed24561198757988927"


def _plan(tmp_path: Path, run_root: Path):
    config = rat.load_config(CONFIG)
    rows = rat.read_manifest(MANIFEST, config["manifest"]["sha256"])
    roots = {}
    for name in ("dataset", "cache", "repository", "dino_runtime", "dino_snapshot"):
        roots[name] = tmp_path / name
        roots[name].mkdir(exist_ok=True)
    return rat.build_run_plan(
        config,
        rows,
        repository_commit=BASELINE,
        config_sha256=rat.sha256_file(CONFIG),
        manifest_sha256=rat.sha256_file(MANIFEST),
        dataset_root=str(roots["dataset"]),
        cache_root=str(roots["cache"]),
        repository_root=str(roots["repository"]),
        dino_runtime_root=str(roots["dino_runtime"]),
        dino_snapshot_root=str(roots["dino_snapshot"]),
        run_root=str(run_root),
        device="fake-cpu",
        storage_available_bytes=None,
    )


def test_interrupted_resume_equals_uninterrupted_and_skips_completed_generation(tmp_path):
    resumed = tmp_path / "resumed"
    uninterrupted = tmp_path / "uninterrupted"
    plan = _plan(tmp_path, resumed)
    assert plan["roots"]["reuse_root"] is None
    plan_id = rat.initialize_run(resumed, plan)
    immutable_plan = (resumed / "run_plan.json").read_bytes()

    # Clone the pristine bound plan before either counterfactual execution.
    shutil.copytree(resumed, uninterrupted)
    with pytest.raises(rat.InjectedInterruption):
        rat.execute_fake_units(
            resumed,
            interrupt_after_new_units=500,
            leave_orphan_partial=True,
        )
    partial = rat.verify_run(resumed)
    assert partial["status"] == "PARTIAL" and partial["completed_unit_count"] == 500
    completed_generation = next(
        unit for unit in plan["expected_units"]
        if unit["stage"] == "06_HELD_OUT_GENERATION"
        and rat.marker_path(resumed, unit["unit_id"]).exists()
    )
    marker = rat.marker_path(resumed, completed_generation["unit_id"])
    artifact = resumed / completed_generation["artifact_relative_path"]
    marker_before = marker.read_bytes()
    artifact_before = artifact.read_bytes()
    marker_mtime = marker.stat().st_mtime_ns
    artifact_mtime = artifact.stat().st_mtime_ns

    resumed_counts = rat.execute_fake_units(resumed)
    assert resumed_counts["completed_new"] == plan["expected_unit_count"] - 500
    assert marker.read_bytes() == marker_before and artifact.read_bytes() == artifact_before
    assert marker.stat().st_mtime_ns == marker_mtime and artifact.stat().st_mtime_ns == artifact_mtime
    assert list((resumed / "failures/quarantine").glob("orphan_partial*"))
    resumed_verify = rat.verify_run(resumed)
    assert resumed_verify["status"] == "PASS" and resumed_verify["remaining_unit_count"] == 0
    resumed_result = rat.finalize_run(resumed)
    assert resumed_result["status"] == "PASS_RESUMABLE_ANCHOR_TRANSPORT"
    assert resumed_result["encoder_order"] == list(rat.ENCODERS)
    assert resumed_result["condition_order"] == list(rat.CONDITIONS)
    assert resumed_result["generation_count"] == 1900
    assert resumed_result["completed_unit_count"] == 2345
    assert resumed_result["fake_cpu_execution"] is True
    assert json.loads((resumed / "final/artifact_manifest.json").read_text())["unit_artifact_count"] == 2345

    uninterrupted_counts = rat.execute_fake_units(uninterrupted)
    assert uninterrupted_counts["completed_new"] == plan["expected_unit_count"]
    uninterrupted_result = rat.finalize_run(uninterrupted)

    assert resumed_result == uninterrupted_result
    assert (resumed / "final/result.json").read_bytes() == (uninterrupted / "final/result.json").read_bytes()
    assert (resumed / "run_plan.json").read_bytes() == immutable_plan
    assert (uninterrupted / "run_plan.json").read_bytes() == immutable_plan
    assert rat.load_bound_plan(resumed)[1] == plan_id == rat.load_bound_plan(uninterrupted)[1]
    assert rat.verify_run(resumed, require_final=True)["status"] == "PASS"
    assert rat.verify_run(uninterrupted, require_final=True)["status"] == "PASS"
    assert not (resumed / "coordinator.lock").exists()
    assert not (uninterrupted / "coordinator.lock").exists()
    coordinator_spec = importlib.util.spec_from_file_location(
        "p2_coordinator_integration", ROOT / "scripts/run_resumable_anchor_transport.py"
    )
    coordinator = importlib.util.module_from_spec(coordinator_spec)
    assert coordinator_spec.loader is not None
    coordinator_spec.loader.exec_module(coordinator)
    status = coordinator._status(resumed, rat)
    assert status["plan_id"] == plan_id
    assert sum(row["completed"] for row in status["stages"]) == 2345


def test_stage01_first_chunk_main_orchestration_isolated_forward_and_resume_skip(tmp_path, monkeypatch):
    run_root = tmp_path / "stage01-run"
    value = _plan(tmp_path, run_root)
    stage_units = [unit for unit in value["expected_units"] if unit["stage"] == "01_DINOV3_FEATURE_BANK"][:2]
    value["expected_units"] = stage_units
    value["expected_unit_count"] = len(stage_units)

    coordinator_spec = importlib.util.spec_from_file_location(
        "p2_coordinator_stage01_boundary", ROOT / "scripts/run_resumable_anchor_transport.py"
    )
    coordinator = importlib.util.module_from_spec(coordinator_spec)
    assert coordinator_spec.loader is not None
    coordinator_spec.loader.exec_module(coordinator)
    worker_spec = importlib.util.spec_from_file_location(
        "p2_worker_stage01_boundary", ROOT / "scripts/p2_anchor_transport_stage_worker.py"
    )
    worker = importlib.util.module_from_spec(worker_spec)
    assert worker_spec.loader is not None
    worker_spec.loader.exec_module(worker)

    value["runtime_bindings"] = {
        "main_python": coordinator.MAIN_PY,
        "main_runtime_root": coordinator.MAIN_RUNTIME_ROOT,
        "main_overlay": coordinator.MAIN_OVERLAY,
        "cache_bindings": dict(coordinator.APPROVED_CACHE_BINDINGS),
        "offline_environment": dict(coordinator.REQUIRED_OFFLINE_ENVIRONMENT),
        "package_versions": {name: "test" for name in coordinator.METADATA_PACKAGES},
        "orchestration_module_specs": {"PIL": "available", "numpy": "available"},
    }
    command = coordinator._stage_worker_command(
        value,
        worker.STAGE01,
        ROOT / "scripts/p2_anchor_transport_stage_worker.py",
        run_root / "run_plan.json",
        run_root,
    )
    assert command[0] == coordinator.MAIN_PY
    availability = worker._validate_stage01_orchestration_runtime(
        value,
        worker.STAGE01,
        executable=coordinator.MAIN_PY,
        module_spec=lambda module: object(),
        is_directory=lambda path: True,
    )
    assert availability == {"interpreter": coordinator.MAIN_PY, "PIL": "available", "numpy": "available"}

    plan_id = rat.initialize_run(run_root, value)
    from vision_encoder_alignment.vlm_compat import dinov3_raw_injection as raw

    boundary_calls: list[str] = []
    decoded_units: list[str] = []
    dino_interpreter = Path(value["roots"]["dino_runtime_root"]) / "bin/python3"
    bindings = SimpleNamespace(
        runtime_root=Path(value["roots"]["dino_runtime_root"]),
        snapshot_root=Path(value["roots"]["dino_snapshot_root"]),
        interpreter=dino_interpreter,
    )

    class Result:
        def __init__(self, batch_size):
            self.batch_size = batch_size

        def get_tap(self, tap):
            return SimpleNamespace(patch=np.zeros((self.batch_size, 196, 768), dtype=np.float32))

    class Extractor:
        def extract(self, batch, *, tap_names):
            boundary_calls.append(str(bindings.interpreter))
            return Result(len(batch.image_ids))

    monkeypatch.setattr(raw, "validate_dinov3_roots", lambda **kwargs: bindings)
    monkeypatch.setattr(raw, "apple_dinov3_interpreter_compatibility", lambda observed: nullcontext())
    monkeypatch.setattr(raw, "_construct_dinov3_extractor", lambda *args, **kwargs: Extractor())

    first_attempt = True

    def canonical_batch(plan, unit, rows, core):
        nonlocal first_attempt
        decoded_units.append(unit["unit_id"])
        if first_attempt:
            worker._STOP_REQUESTED = True
            first_attempt = False
        return SimpleNamespace(image_ids=tuple(unit["image_ids"]))

    monkeypatch.setattr(worker, "_canonical_batch", canonical_batch)
    with pytest.raises(worker.StageInterrupted):
        worker.execute_stage(rat, value, plan_id, run_root, worker.STAGE01)

    first = stage_units[0]
    first_artifact = run_root / first["artifact_relative_path"]
    first_marker = rat.marker_path(run_root, first["unit_id"])
    assert rat.validate_unit_completion(run_root, first, plan_id)
    artifact_bytes = first_artifact.read_bytes()
    marker_bytes = first_marker.read_bytes()
    artifact_mtime = first_artifact.stat().st_mtime_ns
    marker_mtime = first_marker.stat().st_mtime_ns

    worker._STOP_REQUESTED = False
    resumed = worker.execute_stage(rat, value, plan_id, run_root, worker.STAGE01)
    assert resumed["completed_new"] == 1
    assert resumed["model_load_count"] == 1
    assert resumed["extractor_worker_invocation_count"] == 1
    assert resumed["stage_worker_invocation_count"] == 1
    assert decoded_units == [stage_units[0]["unit_id"], stage_units[1]["unit_id"]]
    assert boundary_calls == [str(dino_interpreter), str(dino_interpreter)]
    assert first_artifact.read_bytes() == artifact_bytes
    assert first_marker.read_bytes() == marker_bytes
    assert first_artifact.stat().st_mtime_ns == artifact_mtime
    assert first_marker.stat().st_mtime_ns == marker_mtime
    assert not list(run_root.rglob("*.partial"))


def test_r3_like_193_to_189_reuse_and_first_stage05_resume_checkpoint(tmp_path):
    old_space = tmp_path / "old"
    new_space = tmp_path / "new"
    old_space.mkdir()
    new_space.mkdir()
    old_root = old_space / "run"
    old_plan = _plan(old_space, old_root)
    old_plan_id = rat.initialize_run(old_root, old_plan)
    old_completed = [unit for unit in old_plan["expected_units"] if unit["stage"] in rat.STAGES[1:5]]
    assert len(old_completed) == 193
    for index, unit in enumerate(old_completed):
        if unit["stage"] in rat.STAGES[1:4]:
            buffer = io.BytesIO()
            np.savez(
                buffer,
                scientific_value=np.asarray([index], dtype=np.float32),
                plan_id=np.asarray(old_plan_id, dtype="U64"),
            )
            payload = buffer.getvalue()
        else:
            payload = rat.deterministic_fake_payload(old_plan_id, unit)
        rat.atomic_publish_unit(old_root, unit, payload, old_plan_id, elapsed_seconds=0)

    old_evidence = {
        path.relative_to(old_root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in old_root.rglob("*")
        if path.is_file()
    }
    new_root = new_space / "run"
    new_plan = _plan(new_space, new_root)
    new_plan["roots"]["reuse_root"] = str(old_root.resolve())
    new_plan["repository_commit"] = "2" * 40
    assert new_plan["expected_unit_count"] == 2345

    inspection = rat.inspect_exact_feature_reuse(new_plan)
    assert inspection["status"] == "ELIGIBLE"
    assert inspection["source_plan_id"] == old_plan_id
    assert inspection["source_completed_unit_count"] == 193
    assert inspection["eligible_feature_unit_count"] == 189
    new_plan_id = rat.initialize_run(new_root, new_plan)
    reuse = rat.reuse_exact_feature_units(new_root, new_plan, new_plan_id)
    assert reuse == {"eligible": 189, "reused": 189, "ignored": 0}

    worker_spec = importlib.util.spec_from_file_location(
        "p2_worker_r4_reuse", ROOT / "scripts/p2_anchor_transport_stage_worker.py"
    )
    worker = importlib.util.module_from_spec(worker_spec)
    assert worker_spec.loader is not None
    worker_spec.loader.exec_module(worker)
    for stage in rat.STAGES[1:4]:
        result = worker.execute_stage(rat, new_plan, new_plan_id, new_root, stage)
        assert result["completed_new"] == 0
        assert result["model_load_count"] == 0
        assert result["extractor_worker_invocation_count"] == 0

    feature_units = [unit for unit in new_plan["expected_units"] if unit["stage"] in rat.STAGES[1:4]]
    for unit in feature_units:
        with np.load(old_root / unit["artifact_relative_path"], allow_pickle=False) as old_archive:
            with np.load(new_root / unit["artifact_relative_path"], allow_pickle=False) as new_archive:
                assert np.array_equal(old_archive["scientific_value"], new_archive["scientific_value"])
                assert new_archive["plan_id"].item() == new_plan_id

    anchor_result = worker.execute_stage(
        rat,
        new_plan,
        new_plan_id,
        new_root,
        "04_ANCHOR_REGISTRY",
        fake_loader=lambda stage: object(),
        fake_processor=lambda model, unit: rat.deterministic_fake_payload(new_plan_id, unit),
    )
    assert anchor_result["completed_new"] == 4

    stage05_units = [unit for unit in new_plan["expected_units"] if unit["stage"] == "05_TRANSPORT_AND_CONNECTOR_VALIDATION"]
    first = stage05_units[0]
    processed: list[str] = []

    def interrupt_after_first(model, unit):
        processed.append(unit["unit_id"])
        worker._STOP_REQUESTED = True
        return rat.deterministic_fake_payload(new_plan_id, unit)

    worker._STOP_REQUESTED = False
    with pytest.raises(worker.StageInterrupted):
        worker.execute_stage(
            rat,
            new_plan,
            new_plan_id,
            new_root,
            "05_TRANSPORT_AND_CONNECTOR_VALIDATION",
            fake_loader=lambda stage: object(),
            fake_processor=interrupt_after_first,
        )
    artifact = new_root / first["artifact_relative_path"]
    marker = rat.marker_path(new_root, first["unit_id"])
    artifact_bytes = artifact.read_bytes()
    marker_bytes = marker.read_bytes()
    artifact_mtime = artifact.stat().st_mtime_ns
    marker_mtime = marker.stat().st_mtime_ns
    assert rat.verify_run(new_root)["status"] == "PARTIAL"
    assert rat.reuse_exact_feature_units(new_root, new_plan, new_plan_id) == {
        "eligible": 189,
        "reused": 0,
        "ignored": 0,
    }

    worker._STOP_REQUESTED = False
    resumed = worker.execute_stage(
        rat,
        new_plan,
        new_plan_id,
        new_root,
        "05_TRANSPORT_AND_CONNECTOR_VALIDATION",
        fake_loader=lambda stage: object(),
        fake_processor=lambda model, unit: (
            processed.append(unit["unit_id"]) or rat.deterministic_fake_payload(new_plan_id, unit)
        ),
    )
    assert resumed["completed_new"] == len(stage05_units) - 1
    assert processed.count(first["unit_id"]) == 1
    assert artifact.read_bytes() == artifact_bytes and marker.read_bytes() == marker_bytes
    assert artifact.stat().st_mtime_ns == artifact_mtime
    assert marker.stat().st_mtime_ns == marker_mtime
    for relative, (payload, modified) in old_evidence.items():
        path = old_root / relative
        assert path.read_bytes() == payload and path.stat().st_mtime_ns == modified
