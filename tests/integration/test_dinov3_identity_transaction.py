"""Integration tests for the compensating DINOv3 identity transaction.

Every test operates on byte-sized roots below ``tmp_path``.  The accepted
runtime, snapshot, credentials, network, package manager, and Git are never
used.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any, Callable, Mapping

import pytest

import vision_encoder_alignment.identity_rebind as identity


TARGET_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
RUNTIME_ID = "dinov3-vitb16-r5931719-py312-torch230-tf4560-manual-v1"
SNAPSHOT_ID = "dinov3-vitb16-r5931719-safetensors-v1"


class SimulatedInterruption(BaseException):
    """Escape normal Exception rollback handling to leave a recovery journal."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _registry_bytes() -> bytes:
    return b"""schema_version: 1
revision_policy: "pin_exact_revision_before_phase1"
trust_remote_code_default: false

models:
  clip_vit_b16:
    provider: huggingface
    model_id: openai/clip-vit-base-patch16
    revision: 57c216476eefef5ab752ec549e440a49ae4ae5f3f
    revision_status: pinned
    access: public
    sentinel: clip-preserved

  siglip2_base_p16_224:
    provider: huggingface
    model_id: google/siglip2-base-patch16-224
    revision: 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
    revision_status: pinned
    access: public
    sentinel: siglip-preserved

  dinov3_vit_b16:
    provider: huggingface
    model_id: facebook/dinov3-vitb16-pretrain-lvd1689m
    revision: null
    revision_status: unresolved
    access: user_must_confirm_license_and_auth
    component: vision_backbone_only
    sentinel: dinov3-preserved
"""


def _runtime_profile_value(registry_sha256: str) -> dict[str, Any]:
    return {
        "current_base": {
            "source_files": [
                {"path": "configs/models.yaml", "sha256": registry_sha256},
                {"path": "src/sentinel.py", "sha256": "1" * 64},
            ]
        },
        "runtime_id": RUNTIME_ID,
        "schema_id": "vea.dinov3_isolated_runtime_profile.v0",
        "sentinel": {"preserved": [True, 7, "runtime"]},
        "target": {
            "evidence_revision": TARGET_REVISION,
            "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "registry_key": "dinov3_vit_b16",
        },
    }


def _snapshot_profile_value(runtime_profile_sha256: str) -> dict[str, Any]:
    return {
        "files": [
            {"bytes": 7, "filename": "config.json", "sha256": "2" * 64},
            {
                "bytes": 11,
                "filename": "model.safetensors",
                "sha256": "3" * 64,
            },
        ],
        "load_policy": {
            "forward_allowed": False,
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "model": {
            "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "registry_key": "dinov3_vit_b16",
            "revision": TARGET_REVISION,
        },
        "persistent_runtime": {
            "profile_sha256": runtime_profile_sha256,
            "runtime_id": RUNTIME_ID,
        },
        "schema_id": "vea.model_snapshot_profile.v0",
        "sentinel": {"preserved": [False, 9, "snapshot"]},
        "snapshot_id": SNAPSHOT_ID,
    }


def _runtime_manifest_value(runtime_profile_sha256: str) -> dict[str, Any]:
    return {
        "candidate_freeze": {"bytes": 13, "sha256": "4" * 64},
        "provisioned_at_utc": "2026-07-20T15:40:51Z",
        "repository_profile_sha256": runtime_profile_sha256,
        "runtime_id": RUNTIME_ID,
        "schema_id": "vea.dinov3_persistent_runtime.v0",
        "sentinel": {"preserved": [1, 2, 3]},
    }


def _snapshot_manifest_value(snapshot_profile_sha256: str) -> dict[str, Any]:
    return {
        "credential_backed_logical_request_count": 2,
        "integrity_status": "VERIFIED",
        "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "model_weight_transferred_bytes": 11,
        "provisioned_at_utc": "2026-07-20T17:53:45Z",
        "repository_profile_sha256": snapshot_profile_sha256,
        "revision": TARGET_REVISION,
        "schema_id": "vea.external_model_snapshot.v0",
        "sentinel": {"preserved": [4, 5, 6]},
        "snapshot_id": SNAPSHOT_ID,
    }


def _provisioner_bytes(runtime_sha256: str, registry_sha256: str) -> bytes:
    return f'''"""Small fake provisioner."""

PROFILE_SHA256 = "{runtime_sha256}"
EXPECTED_CURRENT_SOURCES = {{
    "configs/models.yaml": "{registry_sha256}",
    "src/sentinel.py": "{'1' * 64}",
}}
UNCHANGED_SENTINEL = "provisioner-preserved"
'''.encode("utf-8")


def _snapshot_boundary_bytes(runtime_sha256: str) -> bytes:
    return f'''"""Small fake snapshot boundary."""

PERSISTENT_RUNTIME_PROFILE_SHA256 = "{runtime_sha256}"
UNCHANGED_SENTINEL = "snapshot-boundary-preserved"
'''.encode("utf-8")


def _acquire_bytes(
    snapshot_sha256: str,
    runtime_sha256: str,
    registry_sha256: str,
) -> bytes:
    return f'''"""Small fake snapshot acquisition script."""

EXPECTED_PROFILE_SHA256 = "{snapshot_sha256}"
EXPECTED_RUNTIME_PROFILE_SHA256 = "{runtime_sha256}"
EXPECTED_REGISTRY_SHA256 = "{registry_sha256}"
UNCHANGED_SENTINEL = "acquire-preserved"
'''.encode("utf-8")


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


@dataclass(frozen=True, slots=True)
class FakeIdentityRoots:
    repository: Path
    runtime: Path
    snapshot: Path
    transaction: Path
    runtime_profile_path: Path
    snapshot_profile_path: Path
    targets: Mapping[str, Path]
    original_bytes: Mapping[str, bytes]
    expected_old_hashes: Mapping[str, str]
    runtime_data: Mapping[str, bytes]
    snapshot_data: Mapping[str, bytes]


@pytest.fixture
def fake_identity_roots(
    tmp_path: Path,
) -> FakeIdentityRoots:
    repository = (tmp_path / "fake-repository").resolve()
    runtime = (tmp_path / "fake-runtime").resolve()
    snapshot = (tmp_path / "fake-snapshot").resolve()
    transaction = (tmp_path / "fake-transaction").resolve()
    repository.mkdir()
    runtime.mkdir()
    snapshot.mkdir()

    targets = {
        "registry": repository / "configs" / "models.yaml",
        "runtime_profile": (
            repository
            / "configs"
            / "runtime_profiles"
            / "dinov3_isolated_manual_v0.json"
        ),
        "snapshot_profile": (
            repository
            / "configs"
            / "model_snapshots"
            / "dinov3_vit_b16_r5931719_v0.json"
        ),
        "snapshot_boundary": (
            repository
            / "src"
            / "vision_encoder_alignment"
            / "snapshot_boundary.py"
        ),
        "provisioner": repository / "scripts" / "provision_dinov3_runtime.py",
        "acquire_script": repository / "scripts" / "acquire_dinov3_snapshot.py",
        "runtime_manifest": runtime / "runtime-manifest.json",
        "snapshot_manifest": snapshot / "snapshot-manifest.json",
    }

    runtime_profile_path = targets["runtime_profile"]
    snapshot_profile_path = targets["snapshot_profile"]
    registry = _registry_bytes()
    runtime_profile = _canonical_json(_runtime_profile_value(_sha256(registry)))
    snapshot_profile = _canonical_json(
        _snapshot_profile_value(_sha256(runtime_profile))
    )
    original_bytes = {
        "registry": registry,
        "runtime_profile": runtime_profile,
        "snapshot_profile": snapshot_profile,
        "snapshot_boundary": _snapshot_boundary_bytes(_sha256(runtime_profile)),
        "provisioner": _provisioner_bytes(
            _sha256(runtime_profile), _sha256(registry)
        ),
        "acquire_script": _acquire_bytes(
            _sha256(snapshot_profile),
            _sha256(runtime_profile),
            _sha256(registry),
        ),
        "runtime_manifest": _canonical_json(
            _runtime_manifest_value(_sha256(runtime_profile))
        ),
        "snapshot_manifest": _canonical_json(
            _snapshot_manifest_value(_sha256(snapshot_profile))
        ),
    }
    for label, path in targets.items():
        _write(path, original_bytes[label])

    runtime_data_paths = (
        runtime / "metadata" / "config.json",
        runtime / "wheelhouse" / "wheel.whl",
        runtime / "venv" / "runtime.bin",
    )
    snapshot_data_paths = (
        snapshot / "files" / "config.json",
        snapshot / "files" / "preprocessor_config.json",
        snapshot / "files" / "model.safetensors",
    )
    for index, path in enumerate(runtime_data_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"runtime-protected-{index}\n".encode("ascii"))
    for index, path in enumerate(snapshot_data_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"snapshot-protected-{index}\n".encode("ascii"))

    expected_old_hashes = {
        label: _sha256(payload) for label, payload in original_bytes.items()
    }
    return FakeIdentityRoots(
        repository=repository,
        runtime=runtime,
        snapshot=snapshot,
        transaction=transaction,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        targets=targets,
        original_bytes=original_bytes,
        expected_old_hashes=expected_old_hashes,
        runtime_data={
            path.relative_to(runtime).as_posix(): path.read_bytes()
            for path in runtime_data_paths
        },
        snapshot_data={
            path.relative_to(snapshot).as_posix(): path.read_bytes()
            for path in snapshot_data_paths
        },
    )


def _build_plan(fake: FakeIdentityRoots) -> Any:
    plan = identity.build_identity_plan(
        repository_root=fake.repository,
        runtime_profile_path=fake.runtime_profile_path,
        snapshot_profile_path=fake.snapshot_profile_path,
        transaction_root=fake.transaction,
        runtime_root=fake.runtime,
        snapshot_root=fake.snapshot,
        expected_old_hashes=fake.expected_old_hashes,
    )
    identity.validate_identity_plan(plan)
    return plan


def _status(value: object) -> str:
    if isinstance(value, str):
        return value
    for name in ("transaction_status", "status", "marker"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str):
            return candidate
    raise AssertionError("transaction result does not expose a status marker")


def _classification(error: BaseException) -> str:
    value = getattr(error, "classification", None)
    return value if isinstance(value, str) else type(error).__name__


def _post_verify(*_args: Any, **_kwargs: Any) -> bool:
    return True


def _rollback_verify(*_args: Any, **_kwargs: Any) -> bool:
    return True


def _apply(
    fake: FakeIdentityRoots,
    plan: Any,
    *,
    post_verify: Callable[..., Any] | None = _post_verify,
    rollback_verify: Callable[..., Any] | None = _rollback_verify,
    failure_injector: Callable[..., Any] | None = None,
) -> Any:
    return identity.apply_identity_transaction(
        plan,
        repository_root=fake.repository,
        runtime_profile_path=fake.runtime_profile_path,
        snapshot_profile_path=fake.snapshot_profile_path,
        transaction_root=fake.transaction,
        runtime_root=fake.runtime,
        snapshot_root=fake.snapshot,
        post_verify=post_verify,
        rollback_verify=rollback_verify,
        failure_injector=failure_injector,
    )


def _apply_failure(
    fake: FakeIdentityRoots,
    plan: Any,
    **kwargs: Any,
) -> tuple[str, BaseException | None]:
    try:
        result = _apply(fake, plan, **kwargs)
    except identity.IdentityRebindError as error:
        return _classification(error), error
    return _status(result), None


def _event(arguments: tuple[Any, ...], keywords: Mapping[str, Any]) -> tuple[str, int]:
    name = keywords.get("event")
    index = keywords.get("index")
    if name is None and arguments and isinstance(arguments[0], str):
        name = arguments[0]
    if index is None:
        for value in arguments[1:] if name is not None else arguments:
            if type(value) is int:
                index = value
                break
    return str(name or ""), int(index if type(index) is int else -1)


def _fail_on(
    event_name: str, target_index: int, error: BaseException
) -> Callable[..., None]:
    def inject(*args: Any, **kwargs: Any) -> None:
        event, index = _event(args, kwargs)
        if event == event_name and index == target_index:
            raise error

    return inject


def _assert_originals(fake: FakeIdentityRoots) -> None:
    assert {
        label: path.read_bytes() for label, path in fake.targets.items()
    } == fake.original_bytes


def _assert_old_hashes(fake: FakeIdentityRoots) -> None:
    assert {
        label: _sha256_file(path) for label, path in fake.targets.items()
    } == fake.expected_old_hashes


def _candidate_hashes(plan: Any) -> Mapping[str, str]:
    graph = getattr(plan, "candidate_graph", None) or getattr(plan, "new_graph", None)
    if graph is not None:
        for name in ("hashes", "identity_hashes", "files"):
            values = getattr(graph, name, None)
            if isinstance(values, Mapping):
                return values
    value = getattr(plan, "candidate_hashes", None)
    if isinstance(value, Mapping):
        return value
    mutations = getattr(plan, "mutations", None)
    if mutations is not None:
        return {
            str(mutation.relative_path): str(mutation.new_sha256)
            for mutation in mutations
        }
    raise AssertionError("plan does not expose candidate hashes")


def _mutation_destination(fake: FakeIdentityRoots, mutation: Any) -> Path:
    roots = {
        "repository": fake.repository,
        "persistent_runtime": fake.runtime,
        "external_snapshot": fake.snapshot,
    }
    classification = str(mutation.path_classification)
    if classification not in roots:
        raise AssertionError(f"unknown path classification: {classification!r}")
    return roots[classification] / Path(str(mutation.relative_path))


def _verify_final(fake: FakeIdentityRoots) -> Any:
    return identity.verify_rebound_identity(
        repository_root=fake.repository,
        runtime_profile_path=fake.runtime_profile_path,
        snapshot_profile_path=fake.snapshot_profile_path,
        transaction_root=fake.transaction,
        runtime_root=fake.runtime,
        snapshot_root=fake.snapshot,
        expected_old_hashes=fake.expected_old_hashes,
    )


def _recover(fake: FakeIdentityRoots) -> Any:
    return identity.recover_identity_transaction(
        repository_root=fake.repository,
        runtime_profile_path=fake.runtime_profile_path,
        snapshot_profile_path=fake.snapshot_profile_path,
        transaction_root=fake.transaction,
        runtime_root=fake.runtime,
        snapshot_root=fake.snapshot,
        expected_old_hashes=fake.expected_old_hashes,
    )


def _journal_path(fake: FakeIdentityRoots) -> Path:
    journal = fake.transaction / "journal.json"
    if not journal.is_file():
        raise AssertionError("transaction journal.json is missing")
    return journal


def _protected_data(
    fake: FakeIdentityRoots,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    runtime = {
        relative: (fake.runtime / relative).read_bytes()
        for relative in fake.runtime_data
    }
    snapshot = {
        relative: (fake.snapshot / relative).read_bytes()
        for relative in fake.snapshot_data
    }
    return runtime, snapshot


def test_successful_complete_transaction_rebinds_all_identities(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    result = _apply(fake, plan)

    assert _status(result) == "DINOV3_IDENTITY_REBIND_COMMITTED"
    assert result.applied == tuple(mutation.label for mutation in plan.mutations)
    assert result.journal_transitions == (
        "prepared",
        "applying",
        "applied",
        "verifying",
        "committed",
    )
    assert result.compare_and_swap == "VERIFIED_ALL_DESTINATIONS"
    assert result.rollback_attempted is False
    assert result.rollback_succeeded is None
    assert result.cleanup_succeeded is True
    assert result.already_rebound is False
    assert len(plan.mutations) == 8
    assert [
        mutation.path_classification for mutation in plan.mutations
    ].count("repository") == 6
    assert [
        mutation.path_classification for mutation in plan.mutations
    ].count("persistent_runtime") == 1
    assert [
        mutation.path_classification for mutation in plan.mutations
    ].count("external_snapshot") == 1
    for mutation in plan.mutations:
        destination = _mutation_destination(fake, mutation)
        assert _sha256_file(destination) == mutation.new_sha256
        assert destination.stat().st_size == mutation.new_size
    verified = _verify_final(fake)
    assert isinstance(verified, identity.IdentityGraph)
    assert verified == plan.new_graph


def test_all_candidate_bytes_are_prepared_before_first_replace(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    candidates = _candidate_hashes(plan)
    events: list[str] = []

    def inspect_candidates(*args: Any, **kwargs: Any) -> None:
        event, index = _event(args, kwargs)
        if event == "after_prepare" and index == -1:
            staged = tuple(
                path
                for path in (fake.transaction / "candidates").rglob("*")
                if path.is_file()
            )
            journal_text = _journal_path(fake).read_text(encoding="utf-8")
            assert len(candidates) == 8
            assert sorted(_sha256_file(path) for path in staged) == sorted(
                candidates.values()
            )
            assert "raw_bytes" not in journal_text
            assert all(
                str(root) not in journal_text
                for root in (
                    fake.repository,
                    fake.runtime,
                    fake.snapshot,
                    fake.transaction,
                )
            )
            _assert_originals(fake)
            events.append("prepared")
        if event == "before_replace" and index == 0:
            events.append("before_first_replace")

    _apply(fake, plan, failure_injector=inspect_candidates)
    assert events[:2] == ["prepared", "before_first_replace"]


def test_compare_and_swap_rejects_changed_baseline_before_apply(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    fake.targets["registry"].write_bytes(b"concurrent baseline mutation\n")

    status, _ = _apply_failure(fake, plan)

    assert status in {
        "DINOV3_IDENTITY_BASELINE_MISMATCH",
        "DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK",
    }
    assert all(
        fake.targets[label].read_bytes() == payload
        for label, payload in fake.original_bytes.items()
        if label != "registry"
    )


def test_second_apply_is_exact_final_state_idempotent(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    assert _status(_apply(fake, plan)) == "DINOV3_IDENTITY_REBIND_COMMITTED"
    before = {label: path.read_bytes() for label, path in fake.targets.items()}

    second_plan = _build_plan(fake)
    second = _apply(fake, second_plan)

    assert second_plan.already_rebound is True
    assert _status(second) == "DINOV3_IDENTITY_ALREADY_REBOUND"
    assert second.applied == ()
    assert second.journal_transitions == ()
    assert second.compare_and_swap == "EXACT_FINAL_STATE"
    assert second.rollback_attempted is False
    assert second.cleanup_succeeded is True
    assert second.already_rebound is True
    assert {label: path.read_bytes() for label, path in fake.targets.items()} == before


@pytest.mark.parametrize("replace_index", range(8))
def test_failure_after_each_replace_restores_complete_old_graph(
    replace_index: int,
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)

    status, _ = _apply_failure(
        fake,
        plan,
        failure_injector=_fail_on(
            "after_replace", replace_index, RuntimeError("injected apply failure")
        ),
    )

    assert status == "DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK"
    _assert_originals(fake)
    _assert_old_hashes(fake)


def test_failure_before_first_replace_changes_no_destination(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    status, _ = _apply_failure(
        fake,
        _build_plan(fake),
        failure_injector=_fail_on(
            "before_replace", 0, RuntimeError("injected pre-apply failure")
        ),
    )

    assert status == "DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK"
    _assert_originals(fake)


def test_rollback_visits_applied_destinations_in_reverse_order(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    rollback_indexes: list[int] = []

    def inject(*args: Any, **kwargs: Any) -> None:
        event, index = _event(args, kwargs)
        if event == "after_replace" and index == 5:
            raise RuntimeError("start rollback")
        if event == "after_rollback_replace":
            rollback_indexes.append(index)

    status, _ = _apply_failure(fake, plan, failure_injector=inject)

    assert status == "DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK"
    assert rollback_indexes == [5, 4, 3, 2, 1, 0]


def test_rollback_restores_exact_original_raw_bytes(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    status, _ = _apply_failure(
        fake,
        _build_plan(fake),
        failure_injector=_fail_on("after_replace", 7, RuntimeError("rollback")),
    )

    assert status == "DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK"
    _assert_originals(fake)


def test_rollback_verifies_every_restored_sha256(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    verification_calls: list[dict[str, str]] = []

    def verify(*_args: Any, **_kwargs: Any) -> bool:
        hashes = {label: _sha256_file(path) for label, path in fake.targets.items()}
        verification_calls.append(hashes)
        return hashes == fake.expected_old_hashes

    status, _ = _apply_failure(
        fake,
        _build_plan(fake),
        rollback_verify=verify,
        failure_injector=_fail_on("after_replace", 4, RuntimeError("rollback")),
    )

    assert status == "DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK"
    assert verification_calls == [dict(fake.expected_old_hashes)]


def test_concurrent_destination_mutation_is_detected_and_compensated(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    concurrent_bytes = b"concurrent destination mutation\n"
    concurrent_mutation = plan.mutations[3]
    concurrent_path = _mutation_destination(fake, concurrent_mutation)

    def mutate_before_cas(*args: Any, **kwargs: Any) -> None:
        event, index = _event(args, kwargs)
        if event == "before_replace" and index == 3:
            concurrent_path.write_bytes(concurrent_bytes)

    status, _ = _apply_failure(fake, plan, failure_injector=mutate_before_cas)

    assert status == "DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED"
    assert concurrent_path.read_bytes() == concurrent_bytes
    assert all(
        path.read_bytes() == fake.original_bytes[label]
        for label, path in fake.targets.items()
        if path != concurrent_path
    )
    assert fake.transaction.is_dir()
    assert _journal_path(fake).is_file()


def test_incomplete_journal_recovery_restores_old_identity(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    with pytest.raises(SimulatedInterruption):
        _apply(
            fake,
            plan,
            failure_injector=_fail_on(
                "after_replace", 2, SimulatedInterruption("simulated process loss")
            ),
        )
    assert fake.transaction.is_dir()

    result = _recover(fake)

    assert _status(result) == "DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK"
    assert result.compare_and_swap == "RECOVERY_VERIFIED"
    assert result.rollback_attempted is True
    assert result.rollback_succeeded is True
    assert result.cleanup_succeeded is True
    _assert_originals(fake)


def test_post_replace_hash_exception_rolls_back_replaced_destination(
    fake_identity_roots: FakeIdentityRoots,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    first = plan.mutations[0]
    first_destination = _mutation_destination(fake, first)
    real_replace = identity.os.replace
    real_hash_file = identity._sha256_file
    state = {"replace_completed": False, "hash_failed": False}

    def tracking_replace(source: Any, destination: Any) -> None:
        real_replace(source, destination)
        if Path(destination) == first_destination and Path(source).name.endswith(
            ".candidate.tmp"
        ):
            state["replace_completed"] = True

    def fail_first_post_replace_hash(path: Path) -> str:
        if (
            state["replace_completed"]
            and not state["hash_failed"]
            and Path(path) == first_destination
        ):
            state["hash_failed"] = True
            raise RuntimeError("injected post-replace hash failure")
        return real_hash_file(path)

    monkeypatch.setattr(identity.os, "replace", tracking_replace)
    monkeypatch.setattr(identity, "_sha256_file", fail_first_post_replace_hash)

    status, _ = _apply_failure(fake, plan)

    assert state == {"replace_completed": True, "hash_failed": True}
    assert status == "DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK"
    _assert_originals(fake)
    _assert_old_hashes(fake)
    assert not fake.transaction.exists()


@pytest.mark.parametrize("terminal_state", ["committed", "rolled_back"])
def test_terminal_journal_recovery_is_rejected_without_live_mutation(
    terminal_state: str,
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    with pytest.raises(SimulatedInterruption):
        _apply(
            fake,
            plan,
            failure_injector=_fail_on(
                "after_replace", 0, SimulatedInterruption("leave journal")
            ),
        )
    journal = _journal_path(fake)
    value = json.loads(journal.read_text(encoding="utf-8"))
    value["state"] = terminal_state
    journal.write_bytes(_canonical_json(value))
    before = {label: path.read_bytes() for label, path in fake.targets.items()}

    with pytest.raises(identity.IdentityRebindError):
        _recover(fake)

    assert {label: path.read_bytes() for label, path in fake.targets.items()} == before
    assert journal.is_file()


def test_recovery_rejects_noncanonical_payload_path_before_live_mutation(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    with pytest.raises(SimulatedInterruption):
        _apply(
            fake,
            plan,
            failure_injector=_fail_on(
                "after_replace", 0, SimulatedInterruption("leave journal")
            ),
        )
    journal = _journal_path(fake)
    value = json.loads(journal.read_text(encoding="utf-8"))
    canonical_backup = fake.transaction / "backups" / "000.bin"
    alternate_backup = fake.transaction / "backups" / "alternate.bin"
    alternate_backup.write_bytes(canonical_backup.read_bytes())
    value["destinations"][0]["backup_relative_path"] = (
        "backups/alternate.bin"
    )
    journal.write_bytes(_canonical_json(value))
    canonical_backup.write_bytes(b"corrupt canonical backup\n")
    before = {label: path.read_bytes() for label, path in fake.targets.items()}

    with pytest.raises(identity.IdentityRebindError):
        _recover(fake)

    assert {label: path.read_bytes() for label, path in fake.targets.items()} == before
    assert journal.is_file()


def test_recovery_rejects_forged_plan_graph_before_live_mutation(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    with pytest.raises(SimulatedInterruption):
        _apply(
            fake,
            plan,
            failure_injector=_fail_on(
                "after_replace", 0, SimulatedInterruption("leave journal")
            ),
        )
    forged = replace(plan, old_graph=plan.new_graph)
    before = {label: path.read_bytes() for label, path in fake.targets.items()}

    with pytest.raises(identity.IdentityRebindError):
        identity.recover_identity_transaction(
            repository_root=fake.repository,
            runtime_profile_path=fake.runtime_profile_path,
            snapshot_profile_path=fake.snapshot_profile_path,
            transaction_root=fake.transaction,
            runtime_root=fake.runtime,
            snapshot_root=fake.snapshot,
            expected_old_hashes=fake.expected_old_hashes,
            plan=forged,
        )

    assert {label: path.read_bytes() for label, path in fake.targets.items()} == before
    assert _journal_path(fake).is_file()


def test_recovery_rejects_journal_temporary_symlink_without_overwrite(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    with pytest.raises(SimulatedInterruption):
        _apply(
            fake,
            plan,
            failure_injector=_fail_on(
                "after_replace", 0, SimulatedInterruption("leave journal")
            ),
        )
    outside = fake.repository.parent / "outside-journal-target.txt"
    outside.write_bytes(b"outside-preserved\n")
    temporary = fake.transaction / "journal.json.tmp"
    try:
        os.symlink(outside, temporary)
    except OSError as exc:  # pragma: no cover - host capability is explicit
        pytest.fail(f"test host could not create a file symlink: {exc}")
    before = {label: path.read_bytes() for label, path in fake.targets.items()}

    with pytest.raises(identity.IdentityRebindError):
        _recover(fake)

    assert outside.read_bytes() == b"outside-preserved\n"
    assert {label: path.read_bytes() for label, path in fake.targets.items()} == before
    assert _journal_path(fake).is_file()


def test_recovery_removes_regular_stale_journal_temporary(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    with pytest.raises(SimulatedInterruption):
        _apply(
            fake,
            plan,
            failure_injector=_fail_on(
                "after_replace", 0, SimulatedInterruption("leave journal")
            ),
        )
    temporary = fake.transaction / "journal.json.tmp"
    temporary.write_bytes(b'{"partial":')

    result = _recover(fake)

    assert _status(result) == "DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK"
    _assert_originals(fake)
    assert not fake.transaction.exists()


def test_recovery_validates_reconstructed_candidate_graph_before_mutation(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    with pytest.raises(SimulatedInterruption):
        _apply(
            fake,
            plan,
            failure_injector=_fail_on(
                "after_replace", 0, SimulatedInterruption("leave journal")
            ),
        )
    journal = _journal_path(fake)
    value = json.loads(journal.read_text(encoding="utf-8"))
    corrupt_candidate = fake.transaction / "candidates" / "003.bin"
    corrupt_candidate.write_bytes(b"not a valid provisioner candidate\n")
    value["destinations"][3]["new_sha256"] = _sha256_file(corrupt_candidate)
    journal.write_bytes(_canonical_json(value))
    before = {label: path.read_bytes() for label, path in fake.targets.items()}

    with pytest.raises(identity.IdentityRebindError):
        _recover(fake)

    assert {label: path.read_bytes() for label, path in fake.targets.items()} == before
    assert journal.is_file()


def test_transaction_root_creation_race_preserves_competing_root(
    fake_identity_roots: FakeIdentityRoots,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    real_mkdir = Path.mkdir

    def racing_mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == fake.transaction:
            real_mkdir(path)
            (path / "competitor-owned.txt").write_bytes(b"preserved\n")
            raise FileExistsError("injected transaction-root race")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", racing_mkdir)
    status, _ = _apply_failure(fake, plan)

    assert status == "DINOV3_IDENTITY_TRANSACTION_COLLISION"
    assert (fake.transaction / "competitor-owned.txt").read_bytes() == b"preserved\n"
    _assert_originals(fake)


def test_malformed_journal_is_rejected_without_live_mutation(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)
    with pytest.raises(SimulatedInterruption):
        _apply(
            fake,
            plan,
            failure_injector=_fail_on(
                "after_replace", 0, SimulatedInterruption("leave journal")
            ),
        )
    journal = _journal_path(fake)
    journal.write_bytes(b'{"state":NaN,"state":"duplicate"}\n')
    before = {label: path.read_bytes() for label, path in fake.targets.items()}

    with pytest.raises(identity.IdentityRebindError):
        _recover(fake)

    assert {label: path.read_bytes() for label, path in fake.targets.items()} == before
    assert fake.transaction.exists()


def test_rollback_verification_failure_retains_journal_and_backups(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    plan = _build_plan(fake)

    def reject_rollback(*_args: Any, **_kwargs: Any) -> bool:
        raise RuntimeError("rollback verification failed")

    status, _ = _apply_failure(
        fake,
        plan,
        rollback_verify=reject_rollback,
        failure_injector=_fail_on("after_replace", 2, RuntimeError("apply failed")),
    )

    assert status == "DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED"
    assert fake.transaction.is_dir()
    assert _journal_path(fake).is_file()
    assert any(path.is_file() for path in fake.transaction.rglob("*"))


def test_transaction_root_is_removed_after_success(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    assert _status(_apply(fake, _build_plan(fake))) == (
        "DINOV3_IDENTITY_REBIND_COMMITTED"
    )
    assert not fake.transaction.exists()


def test_transaction_root_is_removed_after_successful_rollback(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    status, _ = _apply_failure(
        fake,
        _build_plan(fake),
        failure_injector=_fail_on("after_replace", 1, RuntimeError("rollback")),
    )
    assert status == "DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK"
    assert not fake.transaction.exists()


def test_transaction_uses_no_git_network_or_package_operation(
    fake_identity_roots: FakeIdentityRoots,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = fake_identity_roots
    source = Path(identity.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {
            "huggingface_hub",
            "requests",
            "urllib",
            "http",
            "socket",
            "subprocess",
            "ensurepip",
        }
    )
    forbidden_commands = (
        "pip install",
        "git reset",
        "git restore",
        "git checkout",
        "git stash",
    )
    literals = "\n".join(
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    assert all(value not in literals for value in forbidden_commands)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr in {"system", "popen"}
        for node in ast.walk(tree)
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("forbidden external operation")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert _status(_apply(fake, _build_plan(fake))) == (
        "DINOV3_IDENTITY_REBIND_COMMITTED"
    )


def test_runtime_and_snapshot_data_files_are_preserved(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    before = _protected_data(fake)
    _apply(fake, _build_plan(fake))
    assert _protected_data(fake) == before


def test_transaction_creates_no_repository_artifact(
    fake_identity_roots: FakeIdentityRoots,
) -> None:
    fake = fake_identity_roots
    assert not (fake.repository / "artifacts").exists()
    _apply(fake, _build_plan(fake))
    assert not (fake.repository / "artifacts").exists()
