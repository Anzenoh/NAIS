from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.p1_feature_extraction import (
    EXIT_ARTIFACT,
    EXIT_EXECUTION,
    EXIT_NOT_READY,
    FeatureExtractionError,
    GLOBAL_TAPS,
    MODEL_ORDER,
    PATCH_DEPTHS,
    PATCH_TAPS,
    atomic_write,
    canonical_json_bytes,
    complete_run,
    execution_plan,
    initial_run_state,
    initialize_run_artifacts,
    lane_command_plan,
    load_extraction_profile,
    offline_environment,
    ordered_batches,
    plan_shards,
    reconcile_resume,
    stream_manifest,
    strict_json_bytes,
    tensor_keys,
    validate_mode,
    validate_non_link_path,
    validate_output_root,
    validate_run,
    validate_tensor_mapping,
    write_execution_metrics,
    write_failure_evidence,
    write_shard,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/extraction_profiles/p1_1000_apple_fp32_v0.json"
RUN_ID = "20260730T120000Z_p1_1000_three_encoder_e518a51"


@pytest.fixture(autouse=True)
def _supported_no_clobber_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Emulate link(2) only when the external Windows test filesystem lacks it."""
    probe_source = tmp_path / ".link-source"
    probe_target = tmp_path / ".link-target"
    probe_source.write_bytes(b"probe")
    try:
        os.link(probe_source, probe_target)
        probe_target.unlink()
        return
    except OSError:
        pass

    def exclusive_copy(source, destination):
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with open(source, "rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
                outgoing.write(incoming.read())
                outgoing.flush()
                os.fsync(outgoing.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    monkeypatch.setattr(os, "link", exclusive_copy)


def _document() -> dict[str, object]:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def _write_profile(tmp_path: Path, mutate=None) -> Path:
    value = _document()
    if mutate is not None:
        mutate(value)
    path = tmp_path / "profile.json"
    path.write_bytes(canonical_json_bytes(value))
    return path


def _tensors(count: int = 2) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for tap_index, tap in enumerate(GLOBAL_TAPS):
        values[f"backbone.{tap}.global_mean"] = np.full((count, 768), tap_index, np.float32)
    for tap_index, tap in enumerate(PATCH_TAPS):
        patch = np.full((count, 196, 768), tap_index + 2, np.float32)
        values[f"backbone.{tap}.patch"] = patch
        values[f"backbone.{tap}.global_mean"] = patch.mean(axis=1, dtype=np.float32)
    return values


def _make_run(tmp_path: Path, count: int = 10):
    profile = load_extraction_profile(PROFILE)
    run = tmp_path / RUN_ID
    run.mkdir()
    ids = tuple(f"image-{index:04d}" for index in range(count))
    state = initialize_run_artifacts(
        run, profile, ROOT, run_id=RUN_ID, mode="smoke",
        batch_size=1, image_ids=ids,
    )
    plans = plan_shards(ids)
    for model in profile.models:
        directory = run / "models" / model["registry_key"]
        for plan in plans:
            write_shard(directory, run_id=RUN_ID, model=model, lane_id=model["lane_id"], shard=plan, tensors=_tensors(len(plan.image_ids)), created_at_utc="2026-07-30T03:00:00Z")
        write_execution_metrics(
            directory / "execution_metrics.json", run_id=RUN_ID, model=model,
            image_count=count, elapsed_seconds=1.25,
            peak_allocated_bytes=1024, peak_reserved_bytes=2048,
        )
    return profile, run, state


def _make_incomplete_run(tmp_path: Path, count: int = 10):
    profile = load_extraction_profile(PROFILE)
    run = tmp_path / RUN_ID
    run.mkdir()
    ids = tuple(f"image-{index:04d}" for index in range(count))
    state = initialize_run_artifacts(
        run, profile, ROOT, run_id=RUN_ID, mode="smoke",
        batch_size=1, image_ids=ids,
    )
    return profile, run, state, ids


def test_canonical_profile_acceptance() -> None:
    profile = load_extraction_profile(PROFILE)
    assert profile.document["profile_id"] == "p1-1000-apple-fp32-v0"


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("models", lambda v: v["models"][0].__setitem__("revision", "0" * 40)),
        ("runtime", lambda v: v["runtime"]["main"].__setitem__("interpreter", "/tmp/python3")),
        ("runtime", lambda v: v["runtime"]["main"]["profile"].__setitem__("sha256", "0" * 64)),
        ("source_bindings", lambda v: v["source_bindings"][0].__setitem__("sha256", "0" * 64)),
        ("source_bindings", lambda v: v["source_bindings"][0].__setitem__("path", "scripts/wrong.py")),
        ("canonical_geometry", lambda v: v["canonical_geometry"].__setitem__("crop_width", 225)),
        ("output", lambda v: v["output"].__setitem__("run_id_pattern", ".*")),
    ],
)
def test_exact_production_profile_graph_rejects_mutation(tmp_path: Path, label: str, mutate) -> None:
    with pytest.raises(FeatureExtractionError, match=label):
        load_extraction_profile(_write_profile(tmp_path, mutate))


@pytest.mark.parametrize("kind", ["missing", "unknown"])
def test_missing_or_unknown_profile_key_rejected(tmp_path: Path, kind: str) -> None:
    def mutate(value):
        value.pop("output") if kind == "missing" else value.__setitem__("unknown", 1)
    with pytest.raises(FeatureExtractionError):
        load_extraction_profile(_write_profile(tmp_path, mutate))


def test_duplicate_profile_key_rejected(tmp_path: Path) -> None:
    raw = PROFILE.read_bytes().replace(b'{"canonical_geometry"', b'{"schema_id":"x","canonical_geometry"')
    path = tmp_path / "duplicate.json"; path.write_bytes(raw)
    with pytest.raises(FeatureExtractionError, match="duplicate"):
        load_extraction_profile(path)


def test_noncanonical_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pretty.json"; path.write_text(json.dumps(_document(), indent=2) + "\n", encoding="utf-8")
    with pytest.raises(FeatureExtractionError, match="noncanonical"):
        load_extraction_profile(path)


def test_dataset_identity_mismatch_rejected(tmp_path: Path) -> None:
    with pytest.raises(FeatureExtractionError, match="immutable identity"):
        load_extraction_profile(_write_profile(tmp_path, lambda v: v["dataset"]["manifest"].__setitem__("sha256", "0" * 64)))


def test_checkpoint_identity_mismatch_rejected(tmp_path: Path) -> None:
    with pytest.raises(FeatureExtractionError, match="checkpoint"):
        load_extraction_profile(_write_profile(tmp_path, lambda v: v["checkpoint_plan"].__setitem__("sha256", "0" * 64)))


def test_tap_lists_and_order_are_exact() -> None:
    profile = load_extraction_profile(PROFILE)
    assert tuple(profile.document["tensors"]["global_taps"]) == GLOBAL_TAPS
    assert tuple(profile.document["tensors"]["patch_taps"]) == PATCH_TAPS


def test_patch_depths_map_to_expected_blocks() -> None:
    assert PATCH_DEPTHS == (0.25, 0.5, 0.75, 1.0)
    assert tuple(f"block_{int(depth * 12):02d}" for depth in PATCH_DEPTHS) == PATCH_TAPS


def test_fp32_only_enforced(tmp_path: Path) -> None:
    with pytest.raises(FeatureExtractionError, match="FP32"):
        load_extraction_profile(_write_profile(tmp_path, lambda v: v["policies"].__setitem__("compute_dtype", "float16")))


def test_smoke_count_and_batch_size_enforced() -> None:
    profile = load_extraction_profile(PROFILE)
    assert validate_mode(profile, "smoke", 1) == 10
    with pytest.raises(FeatureExtractionError):
        validate_mode(profile, "smoke", 2)


def test_unresolved_full_batch_size_cannot_execute() -> None:
    with pytest.raises(FeatureExtractionError) as caught:
        validate_mode(load_extraction_profile(PROFILE), "full", None)
    assert caught.value.exit_code == EXIT_NOT_READY


def test_positive_full_batch_size_is_accepted() -> None:
    assert validate_mode(load_extraction_profile(PROFILE), "full", 8) == 1000


def test_deterministic_twenty_five_image_shards() -> None:
    plans = plan_shards(tuple(f"i-{i}" for i in range(1000)))
    assert len(plans) == 40 and plans[0].start == 0 and plans[-1].stop == 1000
    assert all(plan.stop - plan.start == 25 for plan in plans)


def test_ordered_batches_preserve_image_order() -> None:
    rows = tuple(range(11))
    assert tuple(item for batch in ordered_batches(rows, 4) for item in batch) == rows


def test_output_path_escape_or_bad_run_id_rejected(tmp_path: Path) -> None:
    with pytest.raises(FeatureExtractionError):
        validate_output_root(tmp_path, "../bad", protected_roots=[])


def test_output_existing_root_rejected(tmp_path: Path) -> None:
    (tmp_path / RUN_ID).mkdir()
    with pytest.raises(FeatureExtractionError, match="exists"):
        validate_output_root(tmp_path, RUN_ID, protected_roots=[])


def test_output_overlap_rejected(tmp_path: Path) -> None:
    with pytest.raises(FeatureExtractionError, match="overlaps"):
        validate_output_root(tmp_path, RUN_ID, protected_roots=[tmp_path])


def test_lane_plan_is_sequential_and_separate() -> None:
    commands = lane_command_plan(load_extraction_profile(PROFILE), ROOT)
    assert tuple(command[-1] for command in commands) == MODEL_ORDER
    assert commands[0][0] == commands[1][0] != commands[2][0]
    assert all("-I" not in command and command[1] == "-B" for command in commands)


def test_offline_environment_is_allowlisted() -> None:
    environment = offline_environment(ROOT)
    assert environment["HF_HUB_OFFLINE"] == "1" and environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert "PATH" not in environment and "HOME" not in environment


def test_no_overwrite_atomic_publication(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    atomic_write(path, b"first")
    with pytest.raises(FeatureExtractionError):
        atomic_write(path, b"second")
    assert path.read_bytes() == b"first"


def test_atomic_publication_race_preserves_competing_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "race.json"
    original = os.link

    def competing_link(source, destination):
        Path(destination).write_bytes(b"competitor")
        return original(source, destination)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FeatureExtractionError, match="overwrite"):
        atomic_write(path, b"ours")
    assert path.read_bytes() == b"competitor"


def test_exact_resume_acceptance(tmp_path: Path) -> None:
    profile = load_extraction_profile(PROFILE); run = tmp_path / RUN_ID; run.mkdir()
    ids = [f"i-{i}" for i in range(10)]
    state = initialize_run_artifacts(run, profile, ROOT, run_id=RUN_ID, mode="smoke", batch_size=1, image_ids=ids)
    assert reconcile_resume(run, state, profile) == {model: 0 for model in MODEL_ORDER}


@pytest.mark.parametrize("field", ["profile_sha256", "run_id", "mode", "batch_size", "image_ids"])
def test_resume_identity_mismatch_rejected(tmp_path: Path, field: str) -> None:
    profile = load_extraction_profile(PROFILE); run = tmp_path / RUN_ID; run.mkdir()
    state = initialize_run_artifacts(run, profile, ROOT, run_id=RUN_ID, mode="smoke", batch_size=1, image_ids=[f"i-{i}" for i in range(10)])
    changed = dict(state); changed[field] = "wrong" if field != "image_ids" else list(reversed(state[field]))
    with pytest.raises(FeatureExtractionError, match="identity"):
        reconcile_resume(run, changed, profile)


def test_incomplete_run_rejected_by_default(tmp_path: Path) -> None:
    profile, run, _ = _make_run(tmp_path)
    with pytest.raises(FeatureExtractionError) as caught:
        validate_run(run, profile)
    assert caught.value.exit_code == EXIT_ARTIFACT


def test_missing_or_extra_shard_rejected(tmp_path: Path) -> None:
    profile, run, state = _make_run(tmp_path)
    extra = run / "models" / MODEL_ORDER[0] / "shards" / "extra"
    extra.parent.mkdir(parents=True)
    extra.write_text("x")
    with pytest.raises(FeatureExtractionError):
        reconcile_resume(run, state, profile)


def test_checksum_mismatch_rejected(tmp_path: Path) -> None:
    profile, run, state = _make_run(tmp_path)
    tensor = next((run / "models" / MODEL_ORDER[0]).glob("*.safetensors"))
    tensor.write_bytes(tensor.read_bytes() + b"x")
    with pytest.raises(FeatureExtractionError, match="checksum"):
        reconcile_resume(run, state, profile)


@pytest.mark.parametrize("mutation", ["key", "dtype", "shape", "nan", "inf", "mean"])
def test_tensor_contract_rejections(mutation: str) -> None:
    values = _tensors()
    if mutation == "key": values["unexpected"] = values.pop(next(iter(values)))
    elif mutation == "dtype": values[next(iter(values))] = values[next(iter(values))].astype(np.float64)
    elif mutation == "shape": values[next(iter(values))] = values[next(iter(values))][:, :-1]
    elif mutation == "nan": values[next(iter(values))][0, 0] = np.nan
    elif mutation == "inf": values[next(iter(values))][0, 0] = np.inf
    else: values["backbone.block_03.global_mean"][0, 0] += 1
    with pytest.raises(FeatureExtractionError):
        validate_tensor_mapping(values, 2)


def test_complete_manifest_is_emitted_only_after_validation(tmp_path: Path) -> None:
    profile, run, _ = _make_run(tmp_path)
    complete_run(run, profile)
    assert (run / "run_manifest.json").is_file()
    assert validate_run(run, profile)["classification"] == "P1_FEATURE_EXTRACTION_COMPLETE_VALIDATED"


def test_run_manifest_is_last_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vision_encoder_alignment.p1_feature_extraction as extraction
    profile, run, _ = _make_run(tmp_path)
    published: list[str] = []
    original = extraction.atomic_write_json

    def recording_write(path, value):
        published.append(Path(path).relative_to(run).as_posix())
        return original(path, value)

    monkeypatch.setattr(extraction, "atomic_write_json", recording_write)
    complete_run(run, profile)
    assert published[-1] == "run_manifest.json"


def test_tensor_publication_race_preserves_competing_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = load_extraction_profile(PROFILE); model = profile.models[0]
    directory = tmp_path / "model"; directory.mkdir()
    original = os.link

    def competing_link(source, destination):
        if str(destination).endswith(".safetensors"):
            Path(destination).write_bytes(b"competitor")
        return original(source, destination)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FeatureExtractionError, match="collision"):
        write_shard(
            directory, run_id=RUN_ID, model=model, lane_id=model["lane_id"],
            shard=plan_shards([f"i-{index}" for index in range(2)])[0],
            tensors=_tensors(2), created_at_utc="2026-07-30T03:00:00Z",
        )
    assert (directory / "shard-00000.safetensors").read_bytes() == b"competitor"


def test_exact_artifact_inventory_and_required_evidence(tmp_path: Path) -> None:
    profile, run, _ = _make_run(tmp_path)
    complete_run(run, profile)
    files = {path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file()}
    assert {"run_state.json", "dataset_manifest.snapshot.jsonl", "environment.json", "extraction_plan.json", "validation/artifact_validation.json", "run_manifest.json"} <= files
    assert all("/shards/" not in name for name in files)
    assert all(f"models/{model}/execution_metrics.json" in files for model in MODEL_ORDER)


def test_structured_failure_evidence_is_preserved_by_resume(tmp_path: Path) -> None:
    profile = load_extraction_profile(PROFILE); run = tmp_path / RUN_ID; run.mkdir()
    ids = [f"i-{i}" for i in range(10)]
    state = initialize_run_artifacts(run, profile, ROOT, run_id=RUN_ID, mode="smoke", batch_size=1, image_ids=ids)
    model = profile.models[0]; shard = plan_shards(ids)[0]
    failure = write_failure_evidence(
        run, state=state, stage="worker_process", model=model, shard=shard,
        exception_class="WorkerProcessFailure", stable_exit_code=EXIT_EXECUTION,
        raw_return_code=9, sanitized_message="worker_process_failed",
        created_at_utc="2026-07-30T03:00:00Z",
    )
    assert reconcile_resume(run, state, profile) == {model: 0 for model in MODEL_ORDER}
    assert json.loads(failure.read_text(encoding="utf-8"))["image_ids"] == ids


def test_non_link_path_accepts_canonical_root_and_rejects_leaf_link(tmp_path: Path) -> None:
    root = tmp_path / "root"; root.mkdir()
    assert validate_non_link_path(root, expected_kind="directory", label="root") == root

    def probe(path: Path):
        if path == root:
            return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0)
        return os.lstat(path)

    with pytest.raises(FeatureExtractionError, match="link or reparse"):
        validate_non_link_path(root, expected_kind="directory", label="root", lstat_probe=probe)


@pytest.mark.parametrize("label", [
    "dataset root", "checkpoint cache root", "runtime root", "output parent",
])
def test_non_link_path_rejects_injected_ancestor_reparse(tmp_path: Path, label: str) -> None:
    ancestor = tmp_path / "ancestor"; leaf = ancestor / "leaf"
    leaf.mkdir(parents=True)

    def probe(path: Path):
        if path == ancestor:
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_file_attributes=0x400)
        return os.lstat(path)

    with pytest.raises(FeatureExtractionError, match="link or reparse"):
        validate_non_link_path(leaf, expected_kind="directory", label=label, lstat_probe=probe)


@pytest.mark.parametrize("raw_return_code", [-9, 300, 9])
def test_worker_raw_return_code_is_separate_from_stable_exit(tmp_path: Path, raw_return_code: int) -> None:
    profile, run, state, ids = _make_incomplete_run(tmp_path)
    model = profile.models[0]; shard = plan_shards(ids)[0]
    path = write_failure_evidence(
        run, state=state, stage="worker_process", model=model, shard=shard,
        exception_class="WorkerProcessFailure", raw_return_code=raw_return_code,
        sanitized_message="worker_process_failed", created_at_utc="2026-07-30T03:00:00Z",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["stable_exit_code"] == EXIT_EXECUTION
    assert value["raw_return_code"] == raw_return_code


@pytest.mark.parametrize("partial_name", ["shard-00000.safetensors", "shard-00000.json"])
def test_partial_pair_is_preserved_when_failure_evidence_is_written(tmp_path: Path, partial_name: str) -> None:
    profile, run, state, ids = _make_incomplete_run(tmp_path)
    model = profile.models[0]; shard = plan_shards(ids)[0]
    directory = run / "models" / model["registry_key"]
    directory.mkdir(parents=True)
    partial = directory / partial_name; partial.write_bytes(b"competing-partial")
    write_failure_evidence(
        run, state=state, stage="worker_artifact_reconciliation",
        model=model, shard=shard, exception_class="WorkerArtifactReconciliationError",
        sanitized_message="worker_artifact_reconciliation_failed",
        created_at_utc="2026-07-30T03:00:00Z",
    )
    assert partial.read_bytes() == b"competing-partial"
    assert not (run / "run_manifest.json").exists()


@pytest.mark.parametrize("stage", ["completion", "model_free_validation"])
def test_orchestration_failure_uses_nullable_context(tmp_path: Path, stage: str) -> None:
    profile, run, state, _ = _make_incomplete_run(tmp_path)
    path = write_failure_evidence(
        run, state=state, stage=stage, exception_class="OrchestrationError",
        sanitized_message="orchestration_failed", created_at_utc="2026-07-30T03:00:00Z",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["failure_scope"] == "orchestration"
    assert all(value[key] is None for key in (
        "model_key", "lane_id", "shard_index", "start_index",
        "end_index_exclusive", "image_ids", "raw_return_code",
    ))


def test_failure_attempt_numbering_is_append_only(tmp_path: Path) -> None:
    profile, run, state, _ = _make_incomplete_run(tmp_path)
    for stage in ("completion", "model_free_validation"):
        write_failure_evidence(
            run, state=state, stage=stage, exception_class="OrchestrationError",
            sanitized_message="orchestration_failed", created_at_utc="2026-07-30T03:00:00Z",
        )
    assert [path.name for path in sorted((run / "failures").iterdir())] == [
        "attempt-0001.json", "attempt-0002.json",
    ]


def test_finalization_residue_blocks_automatic_resume(tmp_path: Path) -> None:
    profile, run, state, _ = _make_incomplete_run(tmp_path)
    (run / "validation").mkdir()
    with pytest.raises(FeatureExtractionError, match="finalization residue"):
        reconcile_resume(run, state, profile)


def test_all_shards_without_metrics_blocks_resume(tmp_path: Path) -> None:
    profile, run, state, ids = _make_incomplete_run(tmp_path)
    model = profile.models[0]; shard = plan_shards(ids)[0]
    write_shard(
        run / "models" / model["registry_key"], run_id=RUN_ID, model=model,
        lane_id=model["lane_id"], shard=shard, tensors=_tensors(10),
        created_at_utc="2026-07-30T03:00:00Z",
    )
    with pytest.raises(FeatureExtractionError, match="trustworthy metrics"):
        reconcile_resume(run, state, profile)


def test_run_manifest_collision_is_never_overwritten(tmp_path: Path) -> None:
    profile, run, _ = _make_run(tmp_path)
    (run / "run_manifest.json").write_bytes(b"competitor")
    with pytest.raises(FeatureExtractionError):
        complete_run(run, profile)
    assert (run / "run_manifest.json").read_bytes() == b"competitor"


def test_strict_json_rejects_nonfinite() -> None:
    with pytest.raises(FeatureExtractionError):
        strict_json_bytes(b'{"x":NaN}\n', label="x")
