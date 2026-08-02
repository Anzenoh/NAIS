from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from vision_encoder_alignment.dinov3_production_identity import (
    DINOv3ProductionIdentityError,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "t007b3a_production_identity_cli_unit",
        ROOT / "scripts/rebind_dinov3_production_identity.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _load_script()


def _namespace(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        runtime_profile=tmp_path / "runtime-profile.json",
        snapshot_profile=tmp_path / "snapshot-profile.json",
        extractor_profile=tmp_path / "extractor-profile.json",
        transaction_root=tmp_path / "production-transaction",
        compatibility_transaction_root=None,
        runtime_root=tmp_path / "runtime",
        snapshot_root=tmp_path / "snapshot",
    )


def _validated_root(
    tmp_path: Path,
    candidate: Path,
    *,
    production: Path | None = None,
    legacy: Path | None = None,
) -> Path:
    repository = (tmp_path / "repository").resolve()
    runtime = (tmp_path / "runtime").resolve()
    snapshot = (tmp_path / "snapshot").resolve()
    for root in (repository, runtime, snapshot):
        root.mkdir(exist_ok=True)
    return CLI._validate_compatibility_transaction_root(
        candidate,
        production_root=production or (tmp_path / "production-transaction").resolve(),
        legacy_root=legacy or (tmp_path / "legacy-transaction").resolve(),
        protected_roots=(repository, runtime, snapshot),
    )


def _assert_root_collision(caught: pytest.ExceptionInfo[DINOv3ProductionIdentityError]) -> None:
    assert caught.value.classification == "DINOV3_PRODUCTION_POST_VERIFY_ROOT_COLLISION"
    assert caught.value.failure_stage == "preflight_root_validation"


def test_default_compatibility_root_is_a_distinct_sibling(tmp_path: Path) -> None:
    production = (tmp_path / "production-transaction").resolve()
    candidate = CLI._default_compatibility_transaction_root(production)
    assert candidate.parent == production.parent
    assert candidate.name == "dinov3-production-identity-compatibility-verify-v1"
    assert candidate != production
    assert not candidate.exists()


def test_explicit_compatibility_root_is_respected_without_creation(tmp_path: Path) -> None:
    candidate = (tmp_path / "explicit-compatibility-transaction").resolve()
    assert _validated_root(tmp_path, candidate) == candidate
    assert not candidate.exists()


@pytest.mark.parametrize(
    "relation",
    ["same", "parent", "child"],
)
def test_production_root_overlap_is_rejected_before_creation(
    tmp_path: Path,
    relation: str,
) -> None:
    production = (tmp_path / "outer" / "production").resolve()
    production.parent.mkdir()
    candidate = {
        "same": production,
        "parent": production.parent,
        "child": production / "compatibility",
    }[relation]
    with pytest.raises(DINOv3ProductionIdentityError) as caught:
        _validated_root(tmp_path, candidate, production=production)
    _assert_root_collision(caught)
    assert not production.exists()


def test_existing_compatibility_root_is_rejected(tmp_path: Path) -> None:
    candidate = (tmp_path / "existing").resolve()
    candidate.mkdir()
    with pytest.raises(DINOv3ProductionIdentityError) as caught:
        _validated_root(tmp_path, candidate)
    _assert_root_collision(caught)


@pytest.mark.parametrize("relation", ["same", "parent", "child"])
def test_legacy_transaction_root_overlap_is_rejected(
    tmp_path: Path,
    relation: str,
) -> None:
    legacy = (tmp_path / "legacy-parent" / "legacy").resolve()
    legacy.parent.mkdir()
    candidate = {
        "same": legacy,
        "parent": legacy.parent,
        "child": legacy / "compatibility",
    }[relation]
    with pytest.raises(DINOv3ProductionIdentityError) as caught:
        _validated_root(tmp_path, candidate, legacy=legacy)
    _assert_root_collision(caught)
    assert not legacy.exists()


def test_protected_root_overlap_is_rejected(tmp_path: Path) -> None:
    repository = (tmp_path / "repository").resolve()
    repository.mkdir()
    candidate = repository / "transaction"
    with pytest.raises(DINOv3ProductionIdentityError) as caught:
        CLI._validate_compatibility_transaction_root(
            candidate,
            production_root=(tmp_path / "production").resolve(),
            legacy_root=(tmp_path / "legacy").resolve(),
            protected_roots=(repository,),
        )
    _assert_root_collision(caught)
    assert not candidate.exists()


def test_reparse_ancestor_is_rejected_without_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = (tmp_path / "linked-parent" / "compatibility").resolve()
    candidate.parent.mkdir()
    monkeypatch.setattr(CLI, "_is_reparse", lambda path: path == candidate.parent)
    with pytest.raises(DINOv3ProductionIdentityError) as caught:
        _validated_root(tmp_path, candidate)
    _assert_root_collision(caught)
    assert not candidate.exists()


def test_child_command_contains_exactly_one_distinct_transaction_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    namespace = _namespace(tmp_path)
    compatibility = (tmp_path / "compatibility-transaction").resolve()
    observations: list[tuple[list[str], dict[str, Any]]] = []

    def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observations.append((command, kwargs))
        assert command.count("--transaction-root") == 1
        selected = Path(command[command.index("--transaction-root") + 1])
        assert selected == compatibility
        assert selected != namespace.transaction_root
        assert not selected.exists()
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(CLI._POST_VERIFY_MARKERS),
            stderr="",
        )

    monkeypatch.setattr(CLI.subprocess, "run", child)
    assert CLI._legacy_post_verify(namespace, compatibility)() is True
    output = capsys.readouterr().out
    assert CLI._POST_VERIFY_ROOT_ISOLATION_MARKER in output
    assert observations[0][1]["shell"] is False
    assert 0 < observations[0][1]["timeout"] <= 1_800
    assert not compatibility.exists()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            "DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED: bounded",
            "DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED",
        ),
        ("secret=/private/value", CLI._UNCLASSIFIED_CHILD),
        ("UNKNOWN_STABLE_LOOKING_TOKEN: raw", CLI._UNCLASSIFIED_CHILD),
    ],
)
def test_child_classification_extraction_is_allowlisted(
    payload: str,
    expected: str,
) -> None:
    assert CLI._extract_child_classification(payload, "") == expected


def test_failure_evidence_is_bounded_and_path_free() -> None:
    namespace = argparse.Namespace(
        _failure_stage="legacy_compatibility_post_verify",
        _child_classification="DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED",
    )
    error = DINOv3ProductionIdentityError(
        "C:/Users/private secret-token raw stdout raw stderr",
        "DINOV3_PRODUCTION_IDENTITY_FAILED_ROLLED_BACK",
    )
    evidence = CLI._failure_evidence(error, namespace)
    assert evidence == {
        "stage": "legacy_compatibility_post_verify",
        "outer_classification": "DINOV3_PRODUCTION_IDENTITY_FAILED_ROLLED_BACK",
        "child_classification": "DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED",
        "rollback_status": "exact",
    }
    encoded = json.dumps(evidence, sort_keys=True)
    for forbidden in ("C:/", "private", "secret", "stdout", "stderr", "command"):
        assert forbidden not in encoded


def test_plan_verify_and_recover_ignore_compatibility_root_semantics(tmp_path: Path) -> None:
    namespace = _namespace(tmp_path)
    namespace.compatibility_transaction_root = tmp_path / "ignored"
    assert set(CLI._arguments(namespace)) == {
        "repository_root",
        "runtime_profile_path",
        "snapshot_profile_path",
        "extractor_profile_path",
        "transaction_root",
        "runtime_root",
        "snapshot_root",
    }
    for command in ("plan", "verify", "recover"):
        parsed = CLI._parser().parse_args(
            [command, "--compatibility-transaction-root", str(tmp_path / "ignored")]
        )
        assert parsed.command == command


def test_resolver_uses_distinct_default_and_does_not_create_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = (tmp_path / "repository").resolve()
    runtime = (tmp_path / "runtime").resolve()
    snapshot = (tmp_path / "snapshot").resolve()
    for root in (repository, runtime, snapshot):
        root.mkdir()
    production = (tmp_path / "production").resolve()
    namespace = _namespace(tmp_path)
    namespace.transaction_root = production
    monkeypatch.setattr(CLI, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        CLI,
        "default_legacy_transaction_root",
        lambda *, environ: (tmp_path / "legacy").resolve(),
    )
    plan = SimpleNamespace(
        _transaction_root=production,
        _paths=(None, None, None, None, None, runtime / "manifest", snapshot / "manifest"),
    )
    candidate = CLI._resolve_compatibility_transaction_root(namespace, plan)
    assert candidate == CLI._default_compatibility_transaction_root(production)
    assert candidate != production
    assert not candidate.exists()


def test_production_target_graph_source_is_not_modified_by_cli_contract() -> None:
    source = (ROOT / "scripts/rebind_dinov3_production_identity.py").read_text(
        encoding="utf-8"
    )
    assert "--compatibility-transaction-root" in source
    assert '"--transaction-root",\n            str(compatibility_root)' in source
    for forbidden in ("shell=True", "requests.", "urllib.", "socket.", "pip install"):
        assert forbidden not in source
