from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import vision_encoder_alignment.dinov3_production_identity as identity
import vision_encoder_alignment.dinov3_extraction_boundary as extraction_boundary
from vision_encoder_alignment.dinov3_extraction_boundary import (
    DINOv3ExtractionBoundaryError,
    SanitizedProcessFailure,
)
from vision_encoder_alignment.extractors.base import ExtractorValidationError


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PROFILE = ROOT / identity.RUNTIME_PROFILE_RELATIVE_PATH
SNAPSHOT_PROFILE = ROOT / identity.SNAPSHOT_PROFILE_RELATIVE_PATH
EXTRACTOR_PROFILE = ROOT / identity.EXTRACTOR_PROFILE_RELATIVE_PATH


@pytest.fixture
def plan(tmp_path: Path) -> identity.ProductionIdentityPlan:
    return identity.build_production_identity_plan(
        repository_root=ROOT,
        runtime_profile_path=RUNTIME_PROFILE,
        snapshot_profile_path=SNAPSHOT_PROFILE,
        extractor_profile_path=EXTRACTOR_PROFILE,
        transaction_root=tmp_path / "transaction",
    )


def _candidate_profile(plan: identity.ProductionIdentityPlan) -> dict[str, object]:
    return json.loads(plan._new_bytes[-1])


def _live_rows(plan: identity.ProductionIdentityPlan) -> tuple[identity.ProductionSourceIdentity, ...]:
    rows = []
    for row in plan.source_identities:
        digest = hashlib.sha256((ROOT / row.path).read_bytes()).hexdigest()
        rows.append(replace(row, sha256=digest))
    return tuple(rows)


def test_versioned_graph_ids_keep_legacy_unresolved_and_resolved_distinct() -> None:
    assert len({
        identity.LEGACY_GRAPH_ID,
        identity.APPLE_UNRESOLVED_GRAPH_ID,
        identity.APPLE_GRAPH_ID,
    }) == 3
    assert identity.APPLE_GRAPH_ID.endswith("resolved_v2")


def test_valid_production_source_map(plan: identity.ProductionIdentityPlan) -> None:
    rows = identity.load_production_source_identities(_candidate_profile(plan))
    assert rows == plan.source_identities


def test_live_production_sources_validate(plan: identity.ProductionIdentityPlan) -> None:
    assert identity.validate_production_sources(ROOT, _live_rows(plan)) == _live_rows(plan)


def test_missing_source_is_rejected(plan: identity.ProductionIdentityPlan) -> None:
    value = _candidate_profile(plan)
    value["production_sources"].pop()
    with pytest.raises(identity.DINOv3ProductionIdentityError, match="allowlist"):
        identity.load_production_source_identities(value)


def test_unknown_source_is_rejected(plan: identity.ProductionIdentityPlan) -> None:
    value = _candidate_profile(plan)
    value["production_sources"][0]["path"] = "unknown.py"
    with pytest.raises(identity.DINOv3ProductionIdentityError, match="unknown"):
        identity.load_production_source_identities(value)


def test_duplicate_source_is_rejected(plan: identity.ProductionIdentityPlan) -> None:
    value = _candidate_profile(plan)
    value["production_sources"][1] = dict(value["production_sources"][0])
    with pytest.raises(identity.DINOv3ProductionIdentityError, match="duplicate"):
        identity.load_production_source_identities(value)


def test_wrong_source_hash_is_rejected(plan: identity.ProductionIdentityPlan) -> None:
    rows = list(_live_rows(plan))
    rows[0] = replace(rows[0], sha256="0" * 64)
    with pytest.raises(identity.DINOv3ProductionIdentityError) as caught:
        identity.validate_production_sources(ROOT, rows)
    assert caught.value.classification == "DINOV3_PRODUCTION_SOURCE_MISMATCH"


def test_reparse_source_is_rejected(
    plan: identity.ProductionIdentityPlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = identity._is_reparse
    monkeypatch.setattr(
        identity,
        "_is_reparse",
        lambda path: path.name == "models.yaml" or original(path),
    )
    with pytest.raises(identity.DINOv3ProductionIdentityError, match="link"):
        identity.validate_production_sources(ROOT, _live_rows(plan))


def test_extractor_profile_v1_strict_parse(plan: identity.ProductionIdentityPlan) -> None:
    value = _candidate_profile(plan)
    assert value["schema_id"] == identity.EXTRACTOR_PROFILE_SCHEMA_ID
    assert len(identity.load_production_source_identities(value)) == 18


def test_extractor_profile_v0_is_rejected() -> None:
    value = json.loads(EXTRACTOR_PROFILE.read_bytes())
    value["schema_id"] = "vea.dinov3_isolated_extractor_profile.v0"
    value.pop("production_sources", None)
    with pytest.raises(identity.DINOv3ProductionIdentityError, match="schema v1"):
        identity.load_production_source_identities(value)


def test_candidate_dag_binds_runtime_then_snapshot(plan: identity.ProductionIdentityPlan) -> None:
    runtime_hash = hashlib.sha256(plan._new_bytes[0]).hexdigest()
    snapshot = json.loads(plan._new_bytes[1])
    assert snapshot["persistent_runtime"]["profile_sha256"] == runtime_hash


def test_runtime_source_list_is_exact(plan: identity.ProductionIdentityPlan) -> None:
    runtime = json.loads(plan._new_bytes[0])
    assert tuple(row["path"] for row in runtime["current_base"]["source_files"]) == identity.RUNTIME_SOURCE_PATHS


def test_cycle_prone_sources_are_excluded() -> None:
    forbidden = {
        "scripts/provision_dinov3_runtime.py",
        "src/vision_encoder_alignment/snapshot_boundary.py",
        "scripts/acquire_dinov3_snapshot.py",
    }
    assert forbidden.isdisjoint(identity.RUNTIME_SOURCE_PATHS)


def test_profile_semantic_diff_allowlist(plan: identity.ProductionIdentityPlan) -> None:
    old_runtime, new_runtime = map(json.loads, (plan._old_bytes[0], plan._new_bytes[0]))
    old_runtime["current_base"]["source_files"] = new_runtime["current_base"]["source_files"]
    assert old_runtime == new_runtime
    old_snapshot, new_snapshot = map(json.loads, (plan._old_bytes[1], plan._new_bytes[1]))
    old_snapshot["persistent_runtime"]["profile_sha256"] = new_snapshot["persistent_runtime"]["profile_sha256"]
    assert old_snapshot == new_snapshot


def test_strict_journal_rejects_duplicate_key(tmp_path: Path) -> None:
    path = tmp_path / "journal.json"
    path.write_text('{"schema_id":"x","schema_id":"y"}', encoding="utf-8")
    with pytest.raises(identity.DINOv3ProductionIdentityError, match="duplicate"):
        identity._read_journal(path)


def test_plan_is_read_only(tmp_path: Path) -> None:
    targets = (RUNTIME_PROFILE, SNAPSHOT_PROFILE, EXTRACTOR_PROFILE)
    before = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in targets)
    identity.build_production_identity_plan(
        repository_root=ROOT,
        runtime_profile_path=RUNTIME_PROFILE,
        snapshot_profile_path=SNAPSHOT_PROFILE,
        extractor_profile_path=EXTRACTOR_PROFILE,
        transaction_root=tmp_path / "transaction",
    )
    assert before == tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in targets)
    assert not (tmp_path / "transaction").exists()


def test_candidate_graph_is_deterministic(tmp_path: Path) -> None:
    arguments = dict(
        repository_root=ROOT,
        runtime_profile_path=RUNTIME_PROFILE,
        snapshot_profile_path=SNAPSHOT_PROFILE,
        extractor_profile_path=EXTRACTOR_PROFILE,
    )
    first = identity.build_production_identity_plan(
        **arguments, transaction_root=tmp_path / "one"
    )
    second = identity.build_production_identity_plan(
        **arguments, transaction_root=tmp_path / "two"
    )
    assert first.new_hashes == second.new_hashes


def test_current_side_validation_precedes_worker_launch() -> None:
    source = (ROOT / "src/vision_encoder_alignment/extractors/dinov3.py").read_text(encoding="utf-8")
    assert source.index("validate_production_sources(repository") < source.index(
        "invoke_dinov3_extraction_worker("
    )


def test_boundary_validation_precedes_subprocess() -> None:
    source = (ROOT / "src/vision_encoder_alignment/dinov3_extraction_boundary.py").read_text(encoding="utf-8")
    function = source[source.index("def invoke_dinov3_extraction_worker(") :]
    assert function.index("validate_production_sources(repository") < function.index("subprocess.run(")


def test_worker_validation_precedes_heavy_import() -> None:
    source = (ROOT / "scripts/dinov3_extraction_worker.py").read_text(encoding="utf-8")
    function = source[source.index("def run_worker(") :]
    assert function.index("validate_production_sources(repository") < function.index("import numpy as np")
    assert function.index("validate_production_sources(repository") < function.index("import torch")


def test_sanitized_timeout_cause_contains_no_payload() -> None:
    error = SanitizedProcessFailure("worker_timeout", timed_out=True)
    assert error.timed_out is True
    assert "command" not in repr(error).lower()


def test_sanitized_nonzero_cause_retains_only_return_code() -> None:
    error = SanitizedProcessFailure("worker_nonzero_exit", return_code=7)
    assert error.return_code == 7
    assert not hasattr(error, "stdout") and not hasattr(error, "stderr")


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
def test_worker_process_failure_uses_sanitized_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    repository = tmp_path / "repository"
    runtime = tmp_path / "runtime"
    snapshot = tmp_path / "snapshot"
    workspace = tmp_path / "workspace"
    for root in (repository, runtime, snapshot, workspace):
        root.mkdir()
    profile_path = repository / "profile.json"
    worker_path = repository / "worker.py"
    profile_path.write_text("{}", encoding="utf-8")
    worker_path.write_text("pass", encoding="utf-8")
    sources = SimpleNamespace(
        runtime_profile_path="runtime.json", snapshot_profile_path="snapshot.json"
    )
    identities = {
        "runtime_profile_sha256": "1" * 64,
        "snapshot_profile_sha256": "2" * 64,
        "persistent_runtime_manifest_sha256": "3" * 64,
        "external_snapshot_manifest_sha256": "4" * 64,
        "persistent_freeze_sha256": "5" * 64,
    }
    profile = SimpleNamespace(sha256="a" * 64, sources=sources, identities=identities)
    runtime_profile = SimpleNamespace(source_sha256="1" * 64)
    snapshot_profile = SimpleNamespace(source_sha256="2" * 64)
    runtime_manifest = SimpleNamespace(source_sha256="3" * 64, candidate_freeze_sha256="5" * 64)
    snapshot_manifest = SimpleNamespace(source_sha256="4" * 64)
    monkeypatch.setattr(extraction_boundary, "load_dinov3_extractor_profile", lambda path: profile)
    monkeypatch.setattr(extraction_boundary, "validate_production_sources", lambda *args: ())
    monkeypatch.setattr(extraction_boundary, "load_runtime_profile", lambda path: runtime_profile)
    monkeypatch.setattr(extraction_boundary, "load_snapshot_profile", lambda path: snapshot_profile)
    monkeypatch.setattr(extraction_boundary, "validate_runtime_root", lambda *args, **kwargs: runtime_manifest)
    monkeypatch.setattr(extraction_boundary, "validate_snapshot_root", lambda *args, **kwargs: snapshot_manifest)
    monkeypatch.setattr(extraction_boundary, "runtime_python_path", lambda path: runtime / "python.exe")
    monkeypatch.setattr(extraction_boundary, "build_sanitized_worker_environment", lambda **kwargs: {})
    monkeypatch.setattr(
        extraction_boundary,
        "write_dinov3_extraction_request",
        lambda *args, **kwargs: workspace / "request.json",
    )

    def run(*args: object, **kwargs: object) -> object:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(["secret-command"], 1, output=b"secret-output")
        return SimpleNamespace(returncode=9, stdout=b"secret-output", stderr=b"secret-error")

    monkeypatch.setattr(extraction_boundary.subprocess, "run", run)
    request = SimpleNamespace(
        extractor_profile_sha256="a" * 64,
        workspace_relative_input_path="input.npy",
        workspace_relative_output_directory="output",
    )
    monkeypatch.setattr(extraction_boundary, "resolve_workspace_relative_path", lambda *args, **kwargs: workspace / "input.npy")
    with pytest.raises(DINOv3ExtractionBoundaryError) as caught:
        extraction_boundary.invoke_dinov3_extraction_worker(
            runtime_root=runtime,
            snapshot_root=snapshot,
            repository_root=repository,
            extractor_profile_path=profile_path,
            worker_script=worker_path,
            workspace_root=workspace,
            request=request,
            timeout_seconds=1,
        )
    assert isinstance(caught.value.__cause__, SanitizedProcessFailure)
    assert caught.value.__context__ is None
    assert caught.value.__cause__.__context__ is None
    rendered = repr(caught.value) + repr(caught.value.__cause__)
    assert "secret-command" not in rendered and "secret-output" not in rendered


def test_recursive_public_cause_chain_is_redacted() -> None:
    safe = SanitizedProcessFailure("worker_timeout", timed_out=True)
    boundary = DINOv3ExtractionBoundaryError("worker_timeout", "isolated process failed")
    boundary.__cause__ = safe
    public = ExtractorValidationError("dinov3_worker.worker_timeout: isolated extraction failed")
    public.__cause__ = boundary
    chain: list[BaseException] = []
    current: BaseException | None = public
    while current is not None:
        chain.append(current)
        current = current.__cause__
    rendered = " ".join(f"{item!r} {item}" for item in chain)
    assert str(ROOT.resolve()) not in rendered
    assert "stdout" not in rendered and "stderr" not in rendered


def test_production_identity_module_uses_standard_library_only() -> None:
    source = (ROOT / "src/vision_encoder_alignment/dinov3_production_identity.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module not in {None, "__future__"}:
            imports.add(node.module.split(".", 1)[0])
    assert imports <= {
        "ast", "hashlib", "json", "math", "os", "pprint", "re", "shutil", "stat",
        "collections", "dataclasses", "pathlib", "typing",
    }


def test_production_allowlist_is_sorted_and_unique() -> None:
    assert identity.PRODUCTION_SOURCE_PATHS == tuple(sorted(identity.PRODUCTION_SOURCE_PATHS))
    assert len(identity.PRODUCTION_SOURCE_PATHS) == len(set(identity.PRODUCTION_SOURCE_PATHS)) == 18
    assert "src/vision_encoder_alignment/production_runtime_contract.py" in identity.PRODUCTION_SOURCE_PATHS
