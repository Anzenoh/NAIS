from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as rat


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/vlm_compat/p2_universal_anchor_transport_v0.json"
MANIFEST = ROOT / "data/manifests/p1_1000.jsonl"
BASELINE = "9bec9c62093bc06b13682259883713b96c69b2dc"


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
