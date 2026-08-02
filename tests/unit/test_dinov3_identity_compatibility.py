from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest
import conftest as _test_support

import vision_encoder_alignment.dinov3_production_identity as production
import vision_encoder_alignment.identity_rebind as legacy


ROOT = Path(__file__).resolve().parents[2]
_LIVE_PROFILE_BYTES = (
    ROOT / production.EXTRACTOR_PROFILE_RELATIVE_PATH
).read_bytes()
_test_support._DINOV3_LIVE_PROFILE_BYTES = len(_LIVE_PROFILE_BYTES)
_test_support._DINOV3_LIVE_PROFILE_SHA256 = hashlib.sha256(_LIVE_PROFILE_BYTES).hexdigest()


def test_compatibility_layer_names_resolved_graph_explicitly() -> None:
    assert production.APPLE_GRAPH_ID != production.APPLE_UNRESOLVED_GRAPH_ID
    assert "resolved" in production.APPLE_GRAPH_ID


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class CompatibilityLayout:
    repository: Path
    runtime: Path
    snapshot: Path
    transaction: Path
    runtime_profile: Path
    snapshot_profile: Path
    extractor_profile: Path
    paths: tuple[Path, ...]
    original_bytes: tuple[bytes, ...]

    @property
    def compatibility_arguments(self) -> dict[str, Any]:
        return {
            "repository_root": self.repository,
            "runtime_profile_path": self.runtime_profile,
            "snapshot_profile_path": self.snapshot_profile,
            "extractor_profile_path": self.extractor_profile,
            "transaction_root": self.transaction,
            "runtime_root": self.runtime,
            "snapshot_root": self.snapshot,
        }


@pytest.fixture
def layout(
    tmp_path: Path,
    dinov3_registry_pin_graph_path: Path,
) -> CompatibilityLayout:
    repository = (tmp_path / "repository").resolve()
    runtime = (tmp_path / "runtime").resolve()
    snapshot = (tmp_path / "snapshot").resolve()
    transaction = (tmp_path / "transaction").resolve()
    repository.mkdir()
    runtime.mkdir()
    snapshot.mkdir()
    repository_files = set(production.PRODUCTION_SOURCE_PATHS) | {
        production.RUNTIME_PROFILE_RELATIVE_PATH,
        production.SNAPSHOT_PROFILE_RELATIVE_PATH,
        production.EXTRACTOR_PROFILE_RELATIVE_PATH,
        "scripts/provision_dinov3_runtime.py",
        "scripts/acquire_dinov3_snapshot.py",
    }
    registry_repository = dinov3_registry_pin_graph_path / "repository"
    for relative in repository_files:
        _copy(registry_repository / relative, repository / relative)

    runtime_manifest = dinov3_registry_pin_graph_path / "runtime/runtime-manifest.json"
    snapshot_manifest = dinov3_registry_pin_graph_path / "snapshot/snapshot-manifest.json"
    _copy(runtime_manifest, runtime / "runtime-manifest.json")
    _copy(snapshot_manifest, snapshot / "snapshot-manifest.json")
    runtime_profile = repository / production.RUNTIME_PROFILE_RELATIVE_PATH
    snapshot_profile = repository / production.SNAPSHOT_PROFILE_RELATIVE_PATH
    extractor_profile = repository / production.EXTRACTOR_PROFILE_RELATIVE_PATH
    paths = (
        repository / "configs/models.yaml",
        runtime_profile,
        snapshot_profile,
        repository / "scripts/provision_dinov3_runtime.py",
        repository / production.SNAPSHOT_BOUNDARY_RELATIVE_PATH,
        repository / "scripts/acquire_dinov3_snapshot.py",
        runtime / "runtime-manifest.json",
        snapshot / "snapshot-manifest.json",
        extractor_profile,
    )
    return CompatibilityLayout(
        repository=repository,
        runtime=runtime,
        snapshot=snapshot,
        transaction=transaction,
        runtime_profile=runtime_profile,
        snapshot_profile=snapshot_profile,
        extractor_profile=extractor_profile,
        paths=paths,
        original_bytes=tuple(path.read_bytes() for path in paths),
    )


def _production_arguments(layout: CompatibilityLayout) -> dict[str, Any]:
    return dict(layout.compatibility_arguments)


def _apply_production(layout: CompatibilityLayout) -> production.ProductionIdentityPlan:
    plan = production.build_production_identity_plan(**_production_arguments(layout))
    result = production.apply_production_identity_transaction(plan)
    assert result.status == "DINOV3_PRODUCTION_IDENTITY_REBIND_COMMITTED"
    return plan


def _verify(layout: CompatibilityLayout) -> legacy.AcceptedIdentityVerification:
    return legacy.verify_accepted_identity_state(**layout.compatibility_arguments)


def test_registry_pin_exact_final_is_accepted(layout: CompatibilityLayout) -> None:
    result = _verify(layout)
    assert result.state == "registry_pin_final"
    assert len(result.final_hashes) == 8


def test_production_exact_final_is_accepted(layout: CompatibilityLayout) -> None:
    _apply_production(layout)
    result = _verify(layout)
    assert result.state == "production_final"
    assert len(result.final_hashes) == 8
    assert callable(legacy.verify_versioned_production_graph)


@pytest.mark.parametrize(
    ("production_state", "expected"),
    [
        (False, "DINOV3_IDENTITY_STATE_REGISTRY_PIN_FINAL_OK"),
        (True, "DINOV3_IDENTITY_STATE_PRODUCTION_FINAL_OK"),
    ],
)
def test_state_marker_is_exact(
    layout: CompatibilityLayout,
    production_state: bool,
    expected: str,
) -> None:
    if production_state:
        _apply_production(layout)
    assert _verify(layout).marker == expected


def test_old_unpinned_graph_is_rejected(layout: CompatibilityLayout) -> None:
    registry = layout.paths[0]
    text = registry.read_text(encoding="utf-8")
    registry.write_text(
        text.replace(
            "revision: 5931719e67bbdb9737e363e781fb0c67687896bc",
            "revision: null",
        ),
        encoding="utf-8",
    )
    with pytest.raises(legacy.IdentityRebindError):
        _verify(layout)


def test_hybrid_graph_is_rejected(layout: CompatibilityLayout) -> None:
    _apply_production(layout)
    layout.runtime_profile.write_bytes(layout.original_bytes[1])
    with pytest.raises(legacy.IdentityRebindError) as caught:
        _verify(layout)
    assert caught.value.classification == "DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED", str(
        caught.value
    )


def test_partial_production_graph_is_rejected(layout: CompatibilityLayout) -> None:
    _apply_production(layout)
    for path, payload in zip(layout.paths[3:6], layout.original_bytes[3:6], strict=True):
        path.write_bytes(payload)
    with pytest.raises(legacy.IdentityRebindError) as caught:
        _verify(layout)
    assert caught.value.classification == "DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED", str(
        caught.value
    )


def test_tampered_runtime_row_is_rejected(layout: CompatibilityLayout) -> None:
    value = json.loads(layout.runtime_profile.read_bytes())
    value["current_base"]["source_files"][0]["sha256"] = "0" * 64
    layout.runtime_profile.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(legacy.IdentityRebindError):
        _verify(layout)


def test_tampered_extractor_profile_is_rejected(layout: CompatibilityLayout) -> None:
    _apply_production(layout)
    layout.extractor_profile.write_bytes(layout.extractor_profile.read_bytes() + b" ")
    with pytest.raises(legacy.IdentityRebindError):
        _verify(layout)


def test_tampered_manifest_is_rejected(layout: CompatibilityLayout) -> None:
    _apply_production(layout)
    manifest = layout.runtime / "runtime-manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(legacy.IdentityRebindError):
        _verify(layout)


def test_legacy_exact_verifier_still_rejects_production(
    layout: CompatibilityLayout,
) -> None:
    _apply_production(layout)
    arguments = dict(layout.compatibility_arguments)
    arguments.pop("extractor_profile_path")
    with pytest.raises(legacy.IdentityRebindError):
        legacy.verify_rebound_identity(**arguments)


def test_legacy_planner_still_rejects_production(layout: CompatibilityLayout) -> None:
    _apply_production(layout)
    arguments = dict(layout.compatibility_arguments)
    arguments.pop("extractor_profile_path")
    with pytest.raises(legacy.IdentityRebindError):
        legacy.build_identity_plan(**arguments)


@pytest.mark.parametrize("production_state", [False, True])
def test_compatibility_verification_writes_no_file(
    layout: CompatibilityLayout,
    production_state: bool,
) -> None:
    if production_state:
        _apply_production(layout)
    before = tuple(_sha256(path) for path in layout.paths)
    _verify(layout)
    assert tuple(_sha256(path) for path in layout.paths) == before


def test_compatibility_verification_creates_no_transaction_root(
    layout: CompatibilityLayout,
) -> None:
    assert not layout.transaction.exists()
    _verify(layout)
    assert not layout.transaction.exists()


def test_compatibility_verification_accepts_a_distinct_root_during_production_apply(
    layout: CompatibilityLayout,
) -> None:
    compatibility_transaction = layout.transaction.with_name(
        "compatibility-transaction"
    )
    plan = production.build_production_identity_plan(**_production_arguments(layout))

    def post_verify() -> bool:
        assert layout.transaction.is_dir()
        assert not compatibility_transaction.exists()
        arguments = dict(layout.compatibility_arguments)
        arguments["transaction_root"] = compatibility_transaction
        result = legacy.verify_accepted_identity_state(**arguments)
        assert result.state == "production_final"
        assert not compatibility_transaction.exists()
        return True

    result = production.apply_production_identity_transaction(
        plan,
        post_verify=post_verify,
    )
    assert result.status == "DINOV3_PRODUCTION_IDENTITY_REBIND_COMMITTED"
    assert not layout.transaction.exists()
    assert not compatibility_transaction.exists()


def test_registry_verification_keeps_production_module_lazy(
    layout: CompatibilityLayout,
) -> None:
    code = f"""
import sys
from pathlib import Path
from vision_encoder_alignment.identity_rebind import verify_accepted_identity_state
assert 'vision_encoder_alignment.dinov3_production_identity' not in sys.modules
r=verify_accepted_identity_state(
    repository_root=Path({str(layout.repository)!r}),
    runtime_profile_path=Path({str(layout.runtime_profile)!r}),
    snapshot_profile_path=Path({str(layout.snapshot_profile)!r}),
    extractor_profile_path=Path({str(layout.extractor_profile)!r}),
    transaction_root=Path({str(layout.transaction)!r}),
    runtime_root=Path({str(layout.runtime)!r}),
    snapshot_root=Path({str(layout.snapshot)!r}),
)
assert r.state == 'registry_pin_final'
assert 'vision_encoder_alignment.dinov3_production_identity' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=layout.repository,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_production_import_is_function_local() -> None:
    source = (ROOT / "src/vision_encoder_alignment/identity_rebind.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "dinov3_production_identity"
        for node in top_level_imports
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "verify_accepted_identity_state"
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "dinov3_production_identity"
        for node in ast.walk(function)
    )


def test_identity_modules_have_no_circular_import() -> None:
    for order in (
        "identity_rebind,dinov3_production_identity",
        "dinov3_production_identity,identity_rebind",
    ):
        names = order.split(",")
        code = ";".join(
            f"import vision_encoder_alignment.{name}" for name in names
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": "src"},
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_legacy_top_level_imports_remain_standard_library_only() -> None:
    tree = ast.parse(
        (ROOT / "src/vision_encoder_alignment/identity_rebind.py").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        not (isinstance(node, ast.ImportFrom) and node.level > 0)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )


def test_combined_failure_has_stable_path_free_classification(
    layout: CompatibilityLayout,
) -> None:
    _apply_production(layout)
    layout.runtime_profile.write_bytes(layout.original_bytes[1])
    with pytest.raises(legacy.IdentityRebindError) as caught:
        _verify(layout)
    assert caught.value.classification == "DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED", str(
        caught.value
    )
    message = str(caught.value)
    assert str(layout.repository) not in message
    assert str(layout.runtime) not in message
    assert "{" not in message and "raw" not in message.lower()


def test_accepted_result_is_frozen_and_slotted(layout: CompatibilityLayout) -> None:
    result = _verify(layout)
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.state = "production_final"  # type: ignore[misc]


def test_compatibility_result_is_deterministic(layout: CompatibilityLayout) -> None:
    first = _verify(layout)
    second = _verify(layout)
    assert first == second


def test_transaction_collision_does_not_fallback(layout: CompatibilityLayout) -> None:
    layout.transaction.mkdir()
    with pytest.raises(legacy.IdentityRebindError) as caught:
        _verify(layout)
    assert caught.value.classification == "DINOV3_IDENTITY_TRANSACTION_COLLISION"
