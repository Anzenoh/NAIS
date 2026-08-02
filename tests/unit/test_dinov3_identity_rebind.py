"""Unit tests for the import-light DINOv3 identity rebind transaction."""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import pytest

import vision_encoder_alignment.identity_rebind as rebind
from vision_encoder_alignment.identity_rebind import (
    IdentityFile,
    IdentityGraph,
    IdentityMutation,
    IdentityRebindError,
    IdentityTransactionPlan,
    IdentityTransactionResult,
    apply_identity_transaction,
    build_identity_plan,
    validate_identity_plan,
    verify_rebound_identity,
)


TARGET_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
RUNTIME_ID = "dinov3-vitb16-r5931719-py312-torch230-tf4560-manual-v1"
SNAPSHOT_ID = "dinov3-vitb16-r5931719-safetensors-v1"
OLD_REGISTRY_PAIR = (
    b"    revision: null\n"
    b"    revision_status: unresolved\n"
)
NEW_REGISTRY_PAIR = (
    f"    revision: {TARGET_REVISION}\n"
    "    revision_status: pinned\n"
).encode("ascii")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _json_leaf_diffs(
    before: Any,
    after: Any,
    prefix: tuple[str | int, ...] = (),
) -> set[tuple[str | int, ...]]:
    if type(before) is not type(after):
        return {prefix}
    if isinstance(before, dict):
        if set(before) != set(after):
            return {prefix}
        result: set[tuple[str | int, ...]] = set()
        for key in before:
            result.update(_json_leaf_diffs(before[key], after[key], (*prefix, key)))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return {prefix}
        result = set()
        for index, (old_item, new_item) in enumerate(zip(before, after, strict=True)):
            result.update(
                _json_leaf_diffs(old_item, new_item, (*prefix, index))
            )
        return result
    return set() if before == after else {prefix}


def _registry_bytes(*, line_ending: bytes = b"\n") -> bytes:
    value = """schema_version: 1
revision_policy: "pin_exact_revision_before_phase1"
trust_remote_code_default: false

models:
  clip_vit_b16:
    provider: huggingface
    model_id: openai/clip-vit-base-patch16
    revision: 57c216476eefef5ab752ec549e440a49ae4ae5f3
    revision_status: pinned
    access: public
    canonical_input:
      height: 224
      width: 224
      patch_size_hint: 16
      grid_hint: [14, 14]
      patch_count_hint: 196
    hidden_size_hint: 768
    layer_count_hint: 12
    loader_policy:
      trust_remote_code: false
    sentinel: clip-preserved

  siglip2_base_p16_224:
    provider: huggingface
    model_id: google/siglip2-base-patch16-224
    revision: 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
    revision_status: pinned
    access: public
    canonical_input:
      height: 224
      width: 224
      patch_size_hint: 16
      grid_hint: [14, 14]
      patch_count_hint: 196
    hidden_size_hint: 768
    layer_count_hint: 12
    loader_policy:
      trust_remote_code: false
    sentinel: siglip-preserved

  dinov3_vit_b16:
    provider: huggingface
    model_id: facebook/dinov3-vitb16-pretrain-lvd1689m
    revision: null
    revision_status: unresolved
    access: user_must_confirm_license_and_auth
    component: vision_backbone_only
    canonical_input:
      height: 224
      width: 224
      patch_size_hint: 16
      grid_hint: [14, 14]
      patch_count_hint: 196
    hidden_size_hint: 768
    layer_count_hint: 12
    loader_policy:
      trust_remote_code: false
    sentinel: dinov3-preserved
""".encode("utf-8")
    return value if line_ending == b"\n" else value.replace(b"\n", line_ending)


def _runtime_profile_value(registry_sha256: str) -> dict[str, Any]:
    return {
        "current_base": {
            "source_files": [
                {
                    "path": "configs/models.yaml",
                    "sha256": registry_sha256,
                },
                {
                    "path": "src/sentinel.py",
                    "sha256": "1" * 64,
                },
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
            {
                "bytes": 7,
                "filename": "config.json",
                "sha256": "2" * 64,
            },
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
    return f'''"""Small fake provisioner used only by identity-rebind tests."""

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


@dataclass(frozen=True, slots=True)
class FakeIdentityLayout:
    repository_root: Path
    runtime_root: Path
    snapshot_root: Path
    transaction_root: Path
    paths: Mapping[str, Path]
    old_bytes: Mapping[str, bytes]
    expected_old_hashes: Mapping[str, str]


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _prepare_layout(tmp_path: Path) -> FakeIdentityLayout:
    repository_root = (tmp_path / "repository").resolve()
    runtime_root = (
        tmp_path / "external" / "runtimes" / RUNTIME_ID
    ).resolve()
    snapshot_root = (
        tmp_path / "external" / "snapshots" / SNAPSHOT_ID
    ).resolve()
    transaction_root = (
        tmp_path
        / "external"
        / "transactions"
        / "dinov3-r5931719-registry-rebind-v1"
    ).resolve()

    paths = {
        "registry": repository_root / "configs" / "models.yaml",
        "runtime_profile": (
            repository_root
            / "configs"
            / "runtime_profiles"
            / "dinov3_isolated_manual_v0.json"
        ),
        "snapshot_profile": (
            repository_root
            / "configs"
            / "model_snapshots"
            / "dinov3_vit_b16_r5931719_v0.json"
        ),
        "snapshot_boundary": (
            repository_root
            / "src"
            / "vision_encoder_alignment"
            / "snapshot_boundary.py"
        ),
        "provisioner": repository_root / "scripts" / "provision_dinov3_runtime.py",
        "acquire_script": repository_root / "scripts" / "acquire_dinov3_snapshot.py",
        "runtime_manifest": runtime_root / "runtime-manifest.json",
        "snapshot_manifest": snapshot_root / "snapshot-manifest.json",
    }

    registry = _registry_bytes()
    runtime_profile = _canonical_json_bytes(_runtime_profile_value(_digest(registry)))
    snapshot_profile = _canonical_json_bytes(
        _snapshot_profile_value(_digest(runtime_profile))
    )
    old_bytes = {
        "registry": registry,
        "runtime_profile": runtime_profile,
        "snapshot_profile": snapshot_profile,
        "snapshot_boundary": _snapshot_boundary_bytes(_digest(runtime_profile)),
        "provisioner": _provisioner_bytes(
            _digest(runtime_profile), _digest(registry)
        ),
        "acquire_script": _acquire_bytes(
            _digest(snapshot_profile),
            _digest(runtime_profile),
            _digest(registry),
        ),
        "runtime_manifest": _canonical_json_bytes(
            _runtime_manifest_value(_digest(runtime_profile))
        ),
        "snapshot_manifest": _canonical_json_bytes(
            _snapshot_manifest_value(_digest(snapshot_profile))
        ),
    }
    for name, path in paths.items():
        _write(path, old_bytes[name])

    _write(runtime_root / "metadata" / "config.json", b"runtime-data-preserved\n")
    _write(snapshot_root / "files" / "config.json", b"snapshot-data-preserved\n")
    expected_old_hashes = {name: _digest(payload) for name, payload in old_bytes.items()}
    return FakeIdentityLayout(
        repository_root=repository_root,
        runtime_root=runtime_root,
        snapshot_root=snapshot_root,
        transaction_root=transaction_root,
        paths=paths,
        old_bytes=old_bytes,
        expected_old_hashes=expected_old_hashes,
    )


def _replace_registry_baseline(
    layout: FakeIdentityLayout,
    registry: bytes,
) -> FakeIdentityLayout:
    """Recompute the complete old DAG around alternate registry bytes."""

    runtime_value = json.loads(layout.old_bytes["runtime_profile"])
    runtime_value["current_base"]["source_files"][0]["sha256"] = _digest(registry)
    runtime_profile = _canonical_json_bytes(runtime_value)
    snapshot_value = json.loads(layout.old_bytes["snapshot_profile"])
    snapshot_value["persistent_runtime"]["profile_sha256"] = _digest(runtime_profile)
    snapshot_profile = _canonical_json_bytes(snapshot_value)
    runtime_manifest_value = json.loads(layout.old_bytes["runtime_manifest"])
    runtime_manifest_value["repository_profile_sha256"] = _digest(runtime_profile)
    snapshot_manifest_value = json.loads(layout.old_bytes["snapshot_manifest"])
    snapshot_manifest_value["repository_profile_sha256"] = _digest(snapshot_profile)
    old_bytes = {
        "registry": registry,
        "runtime_profile": runtime_profile,
        "snapshot_profile": snapshot_profile,
        "snapshot_boundary": _snapshot_boundary_bytes(_digest(runtime_profile)),
        "provisioner": _provisioner_bytes(_digest(runtime_profile), _digest(registry)),
        "acquire_script": _acquire_bytes(
            _digest(snapshot_profile),
            _digest(runtime_profile),
            _digest(registry),
        ),
        "runtime_manifest": _canonical_json_bytes(runtime_manifest_value),
        "snapshot_manifest": _canonical_json_bytes(snapshot_manifest_value),
    }
    for name, path in layout.paths.items():
        path.write_bytes(old_bytes[name])
    return FakeIdentityLayout(
        repository_root=layout.repository_root,
        runtime_root=layout.runtime_root,
        snapshot_root=layout.snapshot_root,
        transaction_root=layout.transaction_root,
        paths=layout.paths,
        old_bytes=old_bytes,
        expected_old_hashes={
            name: _digest(payload) for name, payload in old_bytes.items()
        },
    )


def _build(
    layout: FakeIdentityLayout,
    *,
    expected_old_hashes: Mapping[str, str] | None = None,
) -> IdentityTransactionPlan:
    return build_identity_plan(
        repository_root=layout.repository_root,
        runtime_profile_path=layout.paths["runtime_profile"],
        snapshot_profile_path=layout.paths["snapshot_profile"],
        transaction_root=layout.transaction_root,
        runtime_root=layout.runtime_root,
        snapshot_root=layout.snapshot_root,
        expected_old_hashes=(
            layout.expected_old_hashes
            if expected_old_hashes is None
            else expected_old_hashes
        ),
    )


def _inventory(*roots: Path) -> dict[str, tuple[str, int]]:
    rows: dict[str, tuple[str, int]] = {}
    for root_index, root in enumerate(roots):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            relative = f"{root_index}:{path.relative_to(root).as_posix()}"
            rows[relative] = (
                _digest(path.read_bytes()) if path.is_file() else "directory",
                path.stat().st_size if path.is_file() else 0,
            )
    return rows


def _assert_safe_public_tree(value: Any, roots: tuple[Path, ...]) -> None:
    assert not isinstance(value, (os.PathLike, bytes, bytearray, memoryview))
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            assert field.name not in {"destination", "raw_bytes", "transaction_root"}
            _assert_safe_public_tree(getattr(value, field.name), roots)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_safe_public_tree(key, roots)
            _assert_safe_public_tree(child, roots)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for child in value:
            _assert_safe_public_tree(child, roots)
        return
    if isinstance(value, str):
        assert not PurePosixPath(value).is_absolute()
        assert not PureWindowsPath(value).is_absolute()
        assert all(str(root) not in value for root in roots)


def _apply(layout: FakeIdentityLayout) -> tuple[IdentityTransactionPlan, IdentityTransactionResult]:
    plan = _build(layout)
    validate_identity_plan(plan)
    result = apply_identity_transaction(
        plan,
        repository_root=layout.repository_root,
        runtime_profile_path=layout.paths["runtime_profile"],
        snapshot_profile_path=layout.paths["snapshot_profile"],
        transaction_root=layout.transaction_root,
        runtime_root=layout.runtime_root,
        snapshot_root=layout.snapshot_root,
    )
    return plan, result


def _assignment_values(payload: bytes) -> dict[str, Any]:
    tree = ast.parse(payload.decode("utf-8"))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        values[targets[0].id] = ast.literal_eval(node.value)
    return values


def _assert_rebind_error(callable_: Any) -> None:
    with pytest.raises(IdentityRebindError):
        callable_()


def test_valid_old_identity_graph_loads_and_validates(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)

    plan = _build(layout)

    assert isinstance(plan, IdentityTransactionPlan)
    assert isinstance(plan.old_graph, IdentityGraph)
    assert isinstance(plan.new_graph, IdentityGraph)
    assert plan.already_rebound is False
    validate_identity_plan(plan)


def test_wrong_old_registry_hash_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    expected = dict(layout.expected_old_hashes)
    expected["registry"] = "0" * 64

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_already_pinned_unexpected_baseline_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    pinned = layout.old_bytes["registry"].replace(
        OLD_REGISTRY_PAIR,
        NEW_REGISTRY_PAIR,
        1,
    )
    layout.paths["registry"].write_bytes(pinned)
    expected = dict(layout.expected_old_hashes)
    expected["registry"] = _digest(pinned)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_registry_patch_changes_exactly_two_fields(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    _apply(layout)

    actual = layout.paths["registry"].read_bytes()
    expected = layout.old_bytes["registry"].replace(
        OLD_REGISTRY_PAIR,
        NEW_REGISTRY_PAIR,
        1,
    )
    assert actual == expected


def test_duplicate_registry_anchor_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    duplicate = layout.old_bytes["registry"].replace(
        OLD_REGISTRY_PAIR,
        OLD_REGISTRY_PAIR + OLD_REGISTRY_PAIR,
        1,
    )
    layout.paths["registry"].write_bytes(duplicate)
    expected = dict(layout.expected_old_hashes)
    expected["registry"] = _digest(duplicate)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_missing_registry_anchor_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    missing = layout.old_bytes["registry"].replace(
        OLD_REGISTRY_PAIR,
        b"    revision: missing\n    revision_status: unresolved\n",
        1,
    )
    layout.paths["registry"].write_bytes(missing)
    expected = dict(layout.expected_old_hashes)
    expected["registry"] = _digest(missing)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_clip_and_siglip_registry_bytes_are_preserved(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    before = layout.old_bytes["registry"]
    dinov3_offset = before.index(b"  dinov3_vit_b16:")
    _apply(layout)

    after = layout.paths["registry"].read_bytes()
    assert after[:dinov3_offset] == before[:dinov3_offset]


def test_registry_access_field_is_preserved(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    _apply(layout)

    after = layout.paths["registry"].read_bytes()
    assert after.count(b"access: user_must_confirm_license_and_auth") == 1
    assert b"access: public" in after


def test_runtime_profile_has_one_leaf_diff(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    before = json.loads(layout.old_bytes["runtime_profile"])
    _apply(layout)
    after = json.loads(layout.paths["runtime_profile"].read_bytes())

    assert _json_leaf_diffs(before, after) == {
        ("current_base", "source_files", 0, "sha256")
    }
    assert layout.paths["runtime_profile"].read_bytes() == _canonical_json_bytes(after)
    assert after["current_base"]["source_files"][0]["sha256"] == _digest(
        layout.paths["registry"].read_bytes()
    )


@pytest.mark.parametrize("row_count", [0, 2])
def test_missing_or_duplicate_models_source_row_is_rejected(
    tmp_path: Path,
    row_count: int,
) -> None:
    layout = _prepare_layout(tmp_path)
    value = json.loads(layout.old_bytes["runtime_profile"])
    rows = value["current_base"]["source_files"]
    model_row = rows[0]
    value["current_base"]["source_files"] = [rows[1]] + [model_row] * row_count
    payload = _canonical_json_bytes(value)
    layout.paths["runtime_profile"].write_bytes(payload)
    expected = dict(layout.expected_old_hashes)
    expected["runtime_profile"] = _digest(payload)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_snapshot_profile_has_one_leaf_diff(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    before = json.loads(layout.old_bytes["snapshot_profile"])
    _apply(layout)
    after = json.loads(layout.paths["snapshot_profile"].read_bytes())

    assert _json_leaf_diffs(before, after) == {
        ("persistent_runtime", "profile_sha256")
    }
    assert layout.paths["snapshot_profile"].read_bytes() == _canonical_json_bytes(after)
    assert after["persistent_runtime"]["profile_sha256"] == _digest(
        layout.paths["runtime_profile"].read_bytes()
    )


def test_source_ast_constants_are_replaced_exactly(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    _apply(layout)

    registry_sha = _digest(layout.paths["registry"].read_bytes())
    runtime_sha = _digest(layout.paths["runtime_profile"].read_bytes())
    snapshot_sha = _digest(layout.paths["snapshot_profile"].read_bytes())
    provisioner = _assignment_values(layout.paths["provisioner"].read_bytes())
    boundary = _assignment_values(layout.paths["snapshot_boundary"].read_bytes())
    acquire = _assignment_values(layout.paths["acquire_script"].read_bytes())
    assert provisioner["PROFILE_SHA256"] == runtime_sha
    assert provisioner["EXPECTED_CURRENT_SOURCES"]["configs/models.yaml"] == registry_sha
    assert provisioner["UNCHANGED_SENTINEL"] == "provisioner-preserved"
    assert boundary["PERSISTENT_RUNTIME_PROFILE_SHA256"] == runtime_sha
    assert boundary["UNCHANGED_SENTINEL"] == "snapshot-boundary-preserved"
    assert acquire["EXPECTED_PROFILE_SHA256"] == snapshot_sha
    assert acquire["EXPECTED_RUNTIME_PROFILE_SHA256"] == runtime_sha
    assert acquire["EXPECTED_REGISTRY_SHA256"] == registry_sha
    assert acquire["UNCHANGED_SENTINEL"] == "acquire-preserved"


def test_missing_source_assignment_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    payload = layout.old_bytes["snapshot_boundary"].replace(
        b"PERSISTENT_RUNTIME_PROFILE_SHA256",
        b"RENAMED_PROFILE_SHA256",
        1,
    )
    layout.paths["snapshot_boundary"].write_bytes(payload)
    expected = dict(layout.expected_old_hashes)
    expected["snapshot_boundary"] = _digest(payload)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_duplicate_source_assignment_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    payload = layout.old_bytes["snapshot_boundary"] + (
        b"PERSISTENT_RUNTIME_PROFILE_SHA256 = \""
        + layout.expected_old_hashes["runtime_profile"].encode("ascii")
        + b"\"\n"
    )
    layout.paths["snapshot_boundary"].write_bytes(payload)
    expected = dict(layout.expected_old_hashes)
    expected["snapshot_boundary"] = _digest(payload)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_wrong_source_assignment_type_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    old_sha = layout.expected_old_hashes["runtime_profile"].encode("ascii")
    payload = layout.old_bytes["snapshot_boundary"].replace(
        b'"' + old_sha + b'"',
        b"None",
        1,
    )
    layout.paths["snapshot_boundary"].write_bytes(payload)
    expected = dict(layout.expected_old_hashes)
    expected["snapshot_boundary"] = _digest(payload)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_external_runtime_manifest_has_one_leaf_diff(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    before = json.loads(layout.old_bytes["runtime_manifest"])
    _apply(layout)
    after = json.loads(layout.paths["runtime_manifest"].read_bytes())

    assert _json_leaf_diffs(before, after) == {("repository_profile_sha256",)}
    assert layout.paths["runtime_manifest"].read_bytes() == _canonical_json_bytes(after)
    assert after["repository_profile_sha256"] == _digest(
        layout.paths["runtime_profile"].read_bytes()
    )


def test_external_snapshot_manifest_has_one_leaf_diff(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    before = json.loads(layout.old_bytes["snapshot_manifest"])
    _apply(layout)
    after = json.loads(layout.paths["snapshot_manifest"].read_bytes())

    assert _json_leaf_diffs(before, after) == {("repository_profile_sha256",)}
    assert layout.paths["snapshot_manifest"].read_bytes() == _canonical_json_bytes(after)
    assert after["repository_profile_sha256"] == _digest(
        layout.paths["snapshot_profile"].read_bytes()
    )


def test_candidate_dag_hashes_are_calculated_from_candidate_bytes(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    plan, _ = _apply(layout)

    assert plan.new_graph.registry_sha256 == _digest(
        layout.paths["registry"].read_bytes()
    )
    assert plan.new_graph.runtime_profile_sha256 == _digest(
        layout.paths["runtime_profile"].read_bytes()
    )
    assert plan.new_graph.snapshot_profile_sha256 == _digest(
        layout.paths["snapshot_profile"].read_bytes()
    )


def test_old_and_new_identity_graphs_are_consistent(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    plan = _build(layout)

    assert plan.old_graph.registry_sha256 == layout.expected_old_hashes["registry"]
    assert plan.old_graph.runtime_profile_sha256 == layout.expected_old_hashes[
        "runtime_profile"
    ]
    assert plan.old_graph.snapshot_profile_sha256 == layout.expected_old_hashes[
        "snapshot_profile"
    ]
    assert plan.new_graph.registry_sha256 != plan.old_graph.registry_sha256
    assert plan.new_graph.runtime_profile_sha256 != plan.old_graph.runtime_profile_sha256
    assert plan.new_graph.snapshot_profile_sha256 != plan.old_graph.snapshot_profile_sha256


def test_historical_paths_are_excluded_from_old_residue_check(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    historical = layout.repository_root / "reports" / "runs" / "historical.json"
    _write(
        historical,
        _canonical_json_bytes(
            {
                "old_registry": layout.expected_old_hashes["registry"],
                "old_runtime": layout.expected_old_hashes["runtime_profile"],
                "old_snapshot": layout.expected_old_hashes["snapshot_profile"],
            }
        ),
    )

    plan = _build(layout)
    validate_identity_plan(plan)


def test_active_source_old_residue_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    payload = layout.old_bytes["provisioner"] + (
        b'OLD_IDENTITY_RESIDUE = "'
        + layout.expected_old_hashes["runtime_profile"].encode("ascii")
        + b'"\n'
    )
    layout.paths["provisioner"].write_bytes(payload)
    expected = dict(layout.expected_old_hashes)
    expected["provisioner"] = _digest(payload)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def _journal_value() -> dict[str, Any]:
    old_identity = {
        "registry": "4" * 64,
        "runtime_profile": "1" * 64,
        "snapshot_profile": "2" * 64,
    }
    candidate_identity = {
        "registry": "3" * 64,
        "runtime_profile": "6" * 64,
        "snapshot_profile": "7" * 64,
    }
    return {
        "applied_index": -1,
        "candidate_identity_hashes": candidate_identity,
        "destinations": [
            {
                "backup_relative_path": "backups/000.bin",
                "candidate_relative_path": "candidates/000.bin",
                "new_sha256": "3" * 64,
                "old_sha256": "4" * 64,
                "path_classification": "repository",
                "relative_path": "configs/models.yaml",
            }
        ],
        "last_transition_utc": "2026-07-21T00:00:00Z",
        "old_identity_hashes": old_identity,
        "rollback_index": -1,
        "schema_id": "vea.dinov3_identity_rebind_journal.v0",
        "started_at_utc": "2026-07-21T00:00:00Z",
        "state": "prepared",
        "transaction_id": "dinov3-r5931719-registry-rebind-v1",
    }


def test_strict_journal_schema_accepts_exact_value() -> None:
    rebind._validate_journal_value(_journal_value())


def test_unknown_journal_field_is_rejected() -> None:
    value = _journal_value()
    value["unknown"] = True

    _assert_rebind_error(lambda: rebind._validate_journal_value(value))


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
    ],
)
def test_duplicate_key_and_non_finite_json_are_rejected(
    tmp_path: Path,
    payload: bytes,
) -> None:
    layout = _prepare_layout(tmp_path)
    layout.paths["runtime_profile"].write_bytes(payload)
    expected = dict(layout.expected_old_hashes)
    expected["runtime_profile"] = _digest(payload)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_sensitive_field_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    value = json.loads(layout.old_bytes["runtime_manifest"])
    value["access_token"] = "must-not-enter-a-plan"
    payload = _canonical_json_bytes(value)
    layout.paths["runtime_manifest"].write_bytes(payload)
    expected = dict(layout.expected_old_hashes)
    expected["runtime_manifest"] = _digest(payload)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_plan_uses_safe_path_classifications(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    plan = _build(layout)
    classifications = {mutation.path_classification for mutation in plan.mutations}

    assert classifications == {
        "repository",
        "persistent_runtime",
        "external_snapshot",
    }
    for mutation in plan.mutations:
        assert not Path(mutation.relative_path).is_absolute()
        assert ".." not in Path(mutation.relative_path).parts


def test_overlapping_roots_are_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    overlapping_runtime = layout.repository_root / "nested-runtime"
    _write(
        overlapping_runtime / "runtime-manifest.json",
        layout.old_bytes["runtime_manifest"],
    )

    _assert_rebind_error(
        lambda: build_identity_plan(
            repository_root=layout.repository_root,
            runtime_profile_path=layout.paths["runtime_profile"],
            snapshot_profile_path=layout.paths["snapshot_profile"],
            transaction_root=layout.transaction_root,
            runtime_root=overlapping_runtime,
            snapshot_root=layout.snapshot_root,
            expected_old_hashes=layout.expected_old_hashes,
        )
    )


def test_symlink_destination_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    target = layout.runtime_root / "real-runtime-manifest.json"
    target.write_bytes(layout.paths["runtime_manifest"].read_bytes())
    layout.paths["runtime_manifest"].unlink()
    try:
        os.symlink(target, layout.paths["runtime_manifest"])
    except OSError as exc:  # pragma: no cover - host capability is asserted explicitly
        pytest.fail(f"test host could not create a file symlink: {exc}")

    _assert_rebind_error(lambda: _build(layout))


def test_plan_is_read_only(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    before = _inventory(
        layout.repository_root,
        layout.runtime_root,
        layout.snapshot_root,
    )

    plan = _build(layout)
    validate_identity_plan(plan)

    assert _inventory(
        layout.repository_root,
        layout.runtime_root,
        layout.snapshot_root,
    ) == before
    assert not layout.transaction_root.exists()


def test_exact_final_state_is_detected_idempotently(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    first_plan, first_result = _apply(layout)
    before_second = _inventory(
        layout.repository_root,
        layout.runtime_root,
        layout.snapshot_root,
    )

    second_plan = build_identity_plan(
        repository_root=layout.repository_root,
        runtime_profile_path=layout.paths["runtime_profile"],
        snapshot_profile_path=layout.paths["snapshot_profile"],
        transaction_root=layout.transaction_root,
        runtime_root=layout.runtime_root,
        snapshot_root=layout.snapshot_root,
        expected_old_hashes=layout.expected_old_hashes,
    )
    second_result = apply_identity_transaction(
        second_plan,
        repository_root=layout.repository_root,
        runtime_profile_path=layout.paths["runtime_profile"],
        snapshot_profile_path=layout.paths["snapshot_profile"],
        transaction_root=layout.transaction_root,
        runtime_root=layout.runtime_root,
        snapshot_root=layout.snapshot_root,
    )

    assert first_plan.already_rebound is False
    assert first_result.status == "DINOV3_IDENTITY_REBIND_COMMITTED"
    assert second_plan.already_rebound is True
    assert second_result.status == "DINOV3_IDENTITY_ALREADY_REBOUND"
    assert _inventory(
        layout.repository_root,
        layout.runtime_root,
        layout.snapshot_root,
    ) == before_second


def test_verify_rebound_identity_is_read_only(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    _apply(layout)
    before = _inventory(
        layout.repository_root,
        layout.runtime_root,
        layout.snapshot_root,
    )

    graph = verify_rebound_identity(
        repository_root=layout.repository_root,
        runtime_profile_path=layout.paths["runtime_profile"],
        snapshot_profile_path=layout.paths["snapshot_profile"],
        transaction_root=layout.transaction_root,
        runtime_root=layout.runtime_root,
        snapshot_root=layout.snapshot_root,
        expected_old_hashes=layout.expected_old_hashes,
    )

    assert isinstance(graph, IdentityGraph)
    assert _inventory(
        layout.repository_root,
        layout.runtime_root,
        layout.snapshot_root,
    ) == before


@pytest.mark.parametrize(
    "class_",
    [
        IdentityFile,
        IdentityGraph,
        IdentityMutation,
        IdentityTransactionPlan,
        IdentityTransactionResult,
    ],
)
def test_public_identity_value_objects_are_frozen_dataclasses(class_: type[Any]) -> None:
    assert is_dataclass(class_)
    assert class_.__dataclass_params__.frozen is True
    assert hasattr(class_, "__slots__")


def test_registry_patch_preserves_crlf_line_endings(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    registry = _registry_bytes(line_ending=b"\r\n")
    layout = _replace_registry_baseline(layout, registry)

    _apply(layout)
    after = layout.paths["registry"].read_bytes()
    assert after.count(b"\r\n") == registry.count(b"\r\n")
    assert after.replace(b"\r\n", b"").count(b"\n") == 0


def test_invalid_utf8_registry_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    invalid = layout.old_bytes["registry"] + b"\xff"
    layout.paths["registry"].write_bytes(invalid)
    expected = dict(layout.expected_old_hashes)
    expected["registry"] = _digest(invalid)

    _assert_rebind_error(lambda: _build(layout, expected_old_hashes=expected))


def test_plan_object_graph_excludes_absolute_paths_and_raw_bytes(
    tmp_path: Path,
) -> None:
    layout = _prepare_layout(tmp_path)
    plan = _build(layout)
    value = plan.to_dict()
    serialized = json.dumps(value, sort_keys=True)
    representation = repr(plan)
    roots = (
        layout.repository_root,
        layout.runtime_root,
        layout.snapshot_root,
        layout.transaction_root,
    )

    _assert_safe_public_tree(plan, roots)
    _assert_safe_public_tree(value, roots)
    for root in roots:
        assert str(root) not in representation
    assert "user_must_confirm_license_and_auth" not in serialized
    assert "destination" not in serialized
    assert "raw_bytes" not in serialized
    assert "transaction_root" not in serialized
    assert "user_must_confirm_license_and_auth" not in representation
    assert "destination" not in representation
    assert "raw_bytes" not in representation
    assert "transaction_root" not in representation


def test_apply_rejects_roots_that_do_not_materialize_plan(
    tmp_path: Path,
) -> None:
    layout = _prepare_layout(tmp_path)
    plan = _build(layout)
    unrelated_runtime = (tmp_path / "unrelated-runtime").resolve()
    _write(
        unrelated_runtime / "runtime-manifest.json",
        b'{"unrelated": true}\n',
    )

    _assert_rebind_error(
        lambda: apply_identity_transaction(
            plan,
            repository_root=layout.repository_root,
            runtime_profile_path=layout.paths["runtime_profile"],
            snapshot_profile_path=layout.paths["snapshot_profile"],
            transaction_root=layout.transaction_root,
            runtime_root=unrelated_runtime,
            snapshot_root=layout.snapshot_root,
        )
    )
    assert all(
        path.read_bytes() == layout.old_bytes[label]
        for label, path in layout.paths.items()
    )


def test_overlapping_apply_transaction_root_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    plan = _build(layout)

    _assert_rebind_error(
        lambda: apply_identity_transaction(
            plan,
            repository_root=layout.repository_root,
            runtime_profile_path=layout.paths["runtime_profile"],
            snapshot_profile_path=layout.paths["snapshot_profile"],
            transaction_root=layout.repository_root / "transaction",
            runtime_root=layout.runtime_root,
            snapshot_root=layout.snapshot_root,
        )
    )


def test_recovery_rejects_disagreeing_runtime_root_aliases(
    tmp_path: Path,
) -> None:
    layout = _prepare_layout(tmp_path)

    _assert_rebind_error(
        lambda: rebind.recover_identity_transaction(
            repository_root=layout.repository_root,
            runtime_profile_path=layout.paths["runtime_profile"],
            snapshot_profile_path=layout.paths["snapshot_profile"],
            transaction_root=layout.transaction_root,
            runtime_root=layout.runtime_root,
            persistent_runtime_root=tmp_path / "different-runtime",
            snapshot_root=layout.snapshot_root,
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("backup_relative_path", r"\escape.bin"),
        ("candidate_relative_path", r"C:\escape.bin"),
        ("backup_relative_path", "backups/../escape.bin"),
    ],
)
def test_journal_payload_path_escape_is_rejected(
    field: str,
    value: str,
) -> None:
    journal = _journal_value()
    journal["destinations"][0][field] = value

    _assert_rebind_error(lambda: rebind._validate_journal_value(journal))


def test_journal_progress_and_timestamp_are_strict() -> None:
    bad_index = _journal_value()
    bad_index["applied_index"] = 99
    _assert_rebind_error(lambda: rebind._validate_journal_value(bad_index))

    bad_timestamp = _journal_value()
    bad_timestamp["last_transition_utc"] = "Z"
    _assert_rebind_error(lambda: rebind._validate_journal_value(bad_timestamp))


def test_journal_identity_hash_must_match_destination() -> None:
    journal = _journal_value()
    journal["candidate_identity_hashes"]["registry"] = "9" * 64

    _assert_rebind_error(lambda: rebind._validate_journal_value(journal))


def test_registry_pair_and_access_must_be_inside_dinov3_block(
    tmp_path: Path,
) -> None:
    layout = _prepare_layout(tmp_path)
    registry = layout.old_bytes["registry"].replace(
        OLD_REGISTRY_PAIR,
        b"    revision: decoy\n    revision_status: unresolved\n",
        1,
    ).replace(
        b"    access: user_must_confirm_license_and_auth\n",
        b"    access: public\n",
        1,
    )
    registry += (
        b"\n  unrelated_model:\n"
        + OLD_REGISTRY_PAIR
        + b"    access: user_must_confirm_license_and_auth\n"
    )
    layout = _replace_registry_baseline(layout, registry)

    _assert_rebind_error(lambda: _build(layout))


def test_transaction_root_symlink_is_rejected(tmp_path: Path) -> None:
    layout = _prepare_layout(tmp_path)
    real_root = tmp_path / "real-transaction"
    real_root.mkdir()
    layout.transaction_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(real_root, layout.transaction_root, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host capability is explicit
        pytest.fail(f"test host could not create a directory symlink: {exc}")

    _assert_rebind_error(lambda: _build(layout))
