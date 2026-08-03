from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import conftest as _test_support

import vision_encoder_alignment.dinov3_production_identity as identity


ROOT = Path(__file__).resolve().parents[2]
_LIVE_PROFILE_BYTES = (ROOT / identity.EXTRACTOR_PROFILE_RELATIVE_PATH).read_bytes()
_test_support._DINOV3_LIVE_PROFILE_BYTES = len(_LIVE_PROFILE_BYTES)
_test_support._DINOV3_LIVE_PROFILE_SHA256 = hashlib.sha256(_LIVE_PROFILE_BYTES).hexdigest()

# Keep this historical transaction fixture independent of live production-source
# evolution while still requiring the exact current bytes declared by the live profile.
_live_profile = json.loads(_LIVE_PROFILE_BYTES)
for _row in _live_profile["production_sources"]:
    if _row["path"] in _test_support._DINOV3_REGISTRY_PIN_HASHES:
        _test_support._DINOV3_REGISTRY_PIN_HASHES[_row["path"]] = _row["sha256"]
_snapshot_boundary = _test_support._replace_assignment_value(
    (ROOT / identity.SNAPSHOT_BOUNDARY_RELATIVE_PATH).read_bytes(),
    "PERSISTENT_RUNTIME_PROFILE_SHA256",
    f'''(
    "{_test_support._DINOV3_REGISTRY_PIN_HASHES[identity.RUNTIME_PROFILE_RELATIVE_PATH]}"
)''',
)
_test_support._DINOV3_REGISTRY_PIN_HASHES[identity.SNAPSHOT_BOUNDARY_RELATIVE_PATH] = (
    hashlib.sha256(_snapshot_boundary).hexdigest()
)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


@pytest.fixture
def layout(
    tmp_path: Path,
    dinov3_registry_pin_graph_path: Path,
) -> dict[str, object]:
    repository = tmp_path / "repository"
    runtime = tmp_path / "runtime"
    snapshot = tmp_path / "snapshot"
    runtime.mkdir()
    snapshot.mkdir()
    repository.mkdir()
    repository_files = set(identity.PRODUCTION_SOURCE_PATHS) | {
        identity.RUNTIME_PROFILE_RELATIVE_PATH,
        identity.SNAPSHOT_PROFILE_RELATIVE_PATH,
        identity.EXTRACTOR_PROFILE_RELATIVE_PATH,
        "scripts/provision_dinov3_runtime.py",
        "scripts/acquire_dinov3_snapshot.py",
    }
    registry_repository = dinov3_registry_pin_graph_path / "repository"
    for relative in repository_files:
        _copy_file(registry_repository / relative, repository / relative)

    real_runtime_manifest = (
        dinov3_registry_pin_graph_path / "runtime/runtime-manifest.json"
    )
    real_snapshot_manifest = (
        dinov3_registry_pin_graph_path / "snapshot/snapshot-manifest.json"
    )
    _copy_file(real_runtime_manifest, runtime / "runtime-manifest.json")
    _copy_file(real_snapshot_manifest, snapshot / "snapshot-manifest.json")
    (runtime / "runtime-data.bin").write_bytes(b"runtime-data-preserved")
    (snapshot / "checkpoint-data.bin").write_bytes(b"checkpoint-data-preserved")

    runtime_profile = repository / identity.RUNTIME_PROFILE_RELATIVE_PATH

    arguments = {
        "repository_root": repository,
        "runtime_profile_path": runtime_profile,
        "snapshot_profile_path": repository / identity.SNAPSHOT_PROFILE_RELATIVE_PATH,
        "extractor_profile_path": repository / identity.EXTRACTOR_PROFILE_RELATIVE_PATH,
        "transaction_root": tmp_path / "transaction",
        "runtime_root": runtime,
        "snapshot_root": snapshot,
    }
    paths = (
        runtime_profile,
        repository / identity.SNAPSHOT_PROFILE_RELATIVE_PATH,
        repository / "scripts/provision_dinov3_runtime.py",
        repository / identity.SNAPSHOT_BOUNDARY_RELATIVE_PATH,
        repository / "scripts/acquire_dinov3_snapshot.py",
        runtime / "runtime-manifest.json",
        snapshot / "snapshot-manifest.json",
        repository / identity.EXTRACTOR_PROFILE_RELATIVE_PATH,
    )
    return {
        "arguments": arguments,
        "paths": paths,
        "repository": repository,
        "runtime": runtime,
        "snapshot": snapshot,
        "transaction": tmp_path / "transaction",
    }


def _hashes(paths: tuple[Path, ...]) -> tuple[str, ...]:
    return tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)


def _apply(layout: dict[str, object]) -> tuple[identity.ProductionIdentityPlan, identity.ProductionIdentityResult]:
    plan = identity.build_production_identity_plan(**layout["arguments"])
    result = identity.apply_production_identity_transaction(plan)
    return plan, result


def test_successful_complete_transaction(layout: dict[str, object]) -> None:
    plan, result = _apply(layout)
    assert result.status == "DINOV3_PRODUCTION_IDENTITY_REBIND_COMMITTED"
    assert _hashes(layout["paths"]) == tuple(digest for _, digest in plan.new_hashes)


def test_all_candidates_prepared_before_first_replace(layout: dict[str, object]) -> None:
    plan = identity.build_production_identity_plan(**layout["arguments"])
    observed: list[bool] = []

    def inject(stage: str, index: int) -> None:
        if stage == "before_replace" and index == 0:
            root = layout["transaction"]
            observed.append(all((root / f"candidates/{i:02d}.bin").is_file() for i in range(8)))
            raise RuntimeError("stop")

    with pytest.raises(identity.DINOv3ProductionIdentityError) as caught:
        identity.apply_production_identity_transaction(plan, failure_injector=inject)
    assert caught.value.classification == "DINOV3_PRODUCTION_IDENTITY_FAILED_ROLLED_BACK"
    assert observed == [True]


def test_external_runtime_manifest_rebound(layout: dict[str, object]) -> None:
    plan, _ = _apply(layout)
    manifest = json.loads((layout["runtime"] / "runtime-manifest.json").read_bytes())
    assert manifest["repository_profile_sha256"] == dict(plan.new_hashes)["runtime_profile"]


def test_external_snapshot_manifest_rebound(layout: dict[str, object]) -> None:
    plan, _ = _apply(layout)
    manifest = json.loads((layout["snapshot"] / "snapshot-manifest.json").read_bytes())
    assert manifest["repository_profile_sha256"] == dict(plan.new_hashes)["snapshot_profile"]


def test_extractor_profile_final_binding(layout: dict[str, object]) -> None:
    plan, _ = _apply(layout)
    profile = json.loads(plan._paths[-1].read_bytes())
    assert profile["schema_id"] == identity.EXTRACTOR_PROFILE_SCHEMA_ID
    assert len(profile["production_sources"]) == 18
    assert any(
        row["path"] == "src/vision_encoder_alignment/production_runtime_contract.py"
        for row in profile["production_sources"]
    )
    assert profile["identities"]["runtime_profile_sha256"] == dict(plan.new_hashes)["runtime_profile"]


def test_compare_and_swap_rejects_concurrent_mutation(layout: dict[str, object]) -> None:
    plan = identity.build_production_identity_plan(**layout["arguments"])
    plan._paths[0].write_bytes(plan._paths[0].read_bytes() + b" ")
    with pytest.raises(identity.DINOv3ProductionIdentityError) as caught:
        identity.apply_production_identity_transaction(plan)
    assert caught.value.classification == "DINOV3_PRODUCTION_IDENTITY_CONCURRENT_MODIFICATION"


def test_second_apply_is_idempotent(layout: dict[str, object]) -> None:
    _apply(layout)
    second = identity.build_production_identity_plan(**layout["arguments"])
    before = _hashes(layout["paths"])
    result = identity.apply_production_identity_transaction(second)
    assert result.status == "DINOV3_PRODUCTION_IDENTITY_ALREADY_REBOUND"
    assert _hashes(layout["paths"]) == before


def test_failure_before_first_replace_restores_start(layout: dict[str, object]) -> None:
    plan = identity.build_production_identity_plan(**layout["arguments"])
    before = _hashes(layout["paths"])

    def inject(stage: str, index: int) -> None:
        if stage == "after_prepare":
            raise RuntimeError("stop")

    with pytest.raises(identity.DINOv3ProductionIdentityError):
        identity.apply_production_identity_transaction(plan, failure_injector=inject)
    assert _hashes(layout["paths"]) == before


@pytest.mark.parametrize("target_index", range(5))
def test_failure_after_each_repository_target_rolls_back(
    layout: dict[str, object], target_index: int
) -> None:
    plan = identity.build_production_identity_plan(**layout["arguments"])
    before = tuple(path.read_bytes() for path in layout["paths"])

    def inject(stage: str, index: int) -> None:
        if stage == "after_replace" and index == target_index:
            raise RuntimeError("stop")

    with pytest.raises(identity.DINOv3ProductionIdentityError):
        identity.apply_production_identity_transaction(plan, failure_injector=inject)
    assert tuple(path.read_bytes() for path in layout["paths"]) == before


def test_failure_after_runtime_manifest_rolls_back(layout: dict[str, object]) -> None:
    _assert_failure_after_target(layout, 5)


def test_failure_after_snapshot_manifest_rolls_back(layout: dict[str, object]) -> None:
    _assert_failure_after_target(layout, 6)


def _assert_failure_after_target(layout: dict[str, object], target_index: int) -> None:
    plan = identity.build_production_identity_plan(**layout["arguments"])
    before = tuple(path.read_bytes() for path in layout["paths"])

    def inject(stage: str, index: int) -> None:
        if stage == "after_replace" and index == target_index:
            raise RuntimeError("stop")

    with pytest.raises(identity.DINOv3ProductionIdentityError):
        identity.apply_production_identity_transaction(plan, failure_injector=inject)
    assert tuple(path.read_bytes() for path in layout["paths"]) == before


def test_rollback_runs_in_exact_reverse_order(layout: dict[str, object]) -> None:
    plan = identity.build_production_identity_plan(**layout["arguments"])
    rolled_back: list[int] = []

    def inject(stage: str, index: int) -> None:
        if stage == "after_replace" and index == 7:
            raise RuntimeError("stop")
        if stage == "before_rollback":
            rolled_back.append(index)

    with pytest.raises(identity.DINOv3ProductionIdentityError):
        identity.apply_production_identity_transaction(plan, failure_injector=inject)
    assert rolled_back == list(reversed(range(8)))


def test_rollback_restores_exact_raw_bytes(layout: dict[str, object]) -> None:
    plan = identity.build_production_identity_plan(**layout["arguments"])
    before = tuple(path.read_bytes() for path in layout["paths"])

    def inject(stage: str, index: int) -> None:
        if stage == "before_verify":
            raise RuntimeError("stop")

    with pytest.raises(identity.DINOv3ProductionIdentityError):
        identity.apply_production_identity_transaction(plan, failure_injector=inject)
    assert tuple(path.read_bytes() for path in layout["paths"]) == before


def test_incomplete_journal_recovery(layout: dict[str, object]) -> None:
    plan = identity.build_production_identity_plan(**layout["arguments"])
    before = _hashes(layout["paths"])

    def inject(stage: str, index: int) -> None:
        if stage == "after_replace" and index == 2:
            raise RuntimeError("apply stop")
        if stage == "before_rollback":
            raise RuntimeError("rollback stop")

    with pytest.raises(identity.DINOv3ProductionIdentityError) as caught:
        identity.apply_production_identity_transaction(plan, failure_injector=inject)
    assert caught.value.classification == "DINOV3_PRODUCTION_IDENTITY_ROLLBACK_FAILED"
    result = identity.recover_production_identity_transaction(**layout["arguments"])
    assert result.status == "DINOV3_PRODUCTION_IDENTITY_RECOVERED"
    assert _hashes(layout["paths"]) == before


def test_malformed_journal_is_rejected(layout: dict[str, object]) -> None:
    root = layout["transaction"]
    root.mkdir()
    (root / "journal.json").write_text('{"schema_id":"wrong"}', encoding="utf-8")
    with pytest.raises(identity.DINOv3ProductionIdentityError):
        identity.recover_production_identity_transaction(**layout["arguments"])


def test_rollback_failure_retains_transaction(layout: dict[str, object]) -> None:
    plan = identity.build_production_identity_plan(**layout["arguments"])

    def inject(stage: str, index: int) -> None:
        if stage == "after_replace" and index == 1:
            raise RuntimeError("apply stop")
        if stage == "before_rollback":
            raise RuntimeError("rollback stop")

    with pytest.raises(identity.DINOv3ProductionIdentityError):
        identity.apply_production_identity_transaction(plan, failure_injector=inject)
    assert (layout["transaction"] / "journal.json").is_file()
    journal = json.loads((layout["transaction"] / "journal.json").read_bytes())
    assert journal["state"] == "rollback_failed"


def test_success_cleans_transaction_workspace(layout: dict[str, object]) -> None:
    _apply(layout)
    assert not layout["transaction"].exists()


def test_runtime_and_snapshot_data_are_preserved(layout: dict[str, object]) -> None:
    runtime_data = layout["runtime"] / "runtime-data.bin"
    snapshot_data = layout["snapshot"] / "checkpoint-data.bin"
    before = (runtime_data.read_bytes(), snapshot_data.read_bytes())
    _apply(layout)
    assert (runtime_data.read_bytes(), snapshot_data.read_bytes()) == before


def test_transaction_module_has_no_network_package_or_git_operation() -> None:
    source = (ROOT / "src/vision_encoder_alignment/dinov3_production_identity.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "subprocess.run", "requests.", "urllib.", "socket.", "pip install",
        "git commit", "git checkout", "git reset", "huggingface_hub",
    )
    assert not any(value in source for value in forbidden)


def test_resolved_graph_transaction_supports_new_files_and_exact_rollback(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    candidate = tmp_path / "candidate"
    repository.mkdir()
    candidate.mkdir()
    (repository / "existing.txt").write_bytes(b"before")
    (candidate / "existing.txt").write_bytes(b"after")
    (candidate / "new.txt").write_bytes(b"new")
    plan = identity.build_resolved_graph_migration_plan(
        repository_root=repository,
        candidate_root=candidate,
        relative_paths=("existing.txt", "new.txt"),
        transaction_root=tmp_path / "transaction",
    )

    def fail(stage: str, index: int) -> None:
        if stage == "after_replace" and index == 1:
            raise RuntimeError("deterministic stop")

    with pytest.raises(identity.DINOv3ProductionIdentityError) as captured:
        identity.apply_resolved_graph_migration(plan, failure_injector=fail)
    assert captured.value.classification == "DINOV3_PRODUCTION_IDENTITY_FAILED_ROLLED_BACK"
    assert (repository / "existing.txt").read_bytes() == b"before"
    assert not (repository / "new.txt").exists()
    assert not (tmp_path / "transaction").exists()


def test_resolved_graph_transaction_commits_and_cleans(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    candidate = tmp_path / "candidate"
    repository.mkdir()
    candidate.mkdir()
    (repository / "a.txt").write_bytes(b"before")
    (candidate / "a.txt").write_bytes(b"after")
    plan = identity.build_resolved_graph_migration_plan(
        repository_root=repository,
        candidate_root=candidate,
        relative_paths=("a.txt",),
        transaction_root=tmp_path / "transaction",
    )
    assert identity.apply_resolved_graph_migration(plan) == ("a.txt",)
    assert (repository / "a.txt").read_bytes() == b"after"
    assert not (tmp_path / "transaction").exists()
