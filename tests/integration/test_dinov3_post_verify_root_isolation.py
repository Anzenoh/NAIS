from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable

import pytest

import vision_encoder_alignment.dinov3_production_identity as production
import vision_encoder_alignment.identity_rebind as legacy


ROOT = Path(__file__).resolve().parents[2]


def _load_existing_contract() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "t007b3a_existing_compatibility_integration",
        ROOT / "tests/integration/test_dinov3_identity_compatibility.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_existing_contract()
CLI = BASE.PRODUCTION_CLI


@pytest.fixture
def layout(tmp_path: Path) -> Any:
    return BASE.layout.__wrapped__(tmp_path)


def _exact_bytes(layout: Any) -> tuple[bytes, ...]:
    return tuple(path.read_bytes() for path in layout.paths)


def _compatibility_root_from(command: list[str]) -> Path:
    assert command.count("--transaction-root") == 1
    return Path(command[command.index("--transaction-root") + 1])


def _evidence(captured: pytest.CaptureResult[str]) -> dict[str, str]:
    lines = [
        line
        for line in captured.err.splitlines()
        if line.startswith(CLI._FAILURE_EVIDENCE_MARKER)
    ]
    assert len(lines) == 1
    return json.loads(lines[0][len(CLI._FAILURE_EVIDENCE_MARKER) :])


def test_pre_fix_omitted_root_alias_reproduces_collision_and_exact_rollback(
    tmp_path: Path,
    layout: Any,
) -> None:
    fake_local = (tmp_path / "local-app-data").resolve()
    fake_local.mkdir()
    environment = {"LOCALAPPDATA": str(fake_local)}
    active = legacy.default_transaction_root(environ=environment)
    arguments = dict(layout.arguments)
    arguments["transaction_root"] = active
    plan = production.build_production_identity_plan(**arguments)

    def omitted_root_compatibility_verify() -> bool:
        assert active.is_dir()
        with pytest.raises(legacy.IdentityRebindError) as caught:
            legacy.verify_accepted_identity_state(
                repository_root=layout.repository,
                runtime_profile_path=layout.runtime_profile,
                snapshot_profile_path=layout.snapshot_profile,
                extractor_profile_path=layout.extractor_profile,
                transaction_root=None,
                runtime_root=layout.runtime,
                snapshot_root=layout.snapshot,
                environ=environment,
            )
        assert caught.value.classification == "DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED"
        assert "DINOV3_PRODUCTION_IDENTITY_TRANSACTION_COLLISION" in str(caught.value)
        raise caught.value

    with pytest.raises(production.DINOv3ProductionIdentityError) as caught:
        production.apply_production_identity_transaction(
            plan,
            post_verify=omitted_root_compatibility_verify,
        )
    assert caught.value.classification == "DINOV3_PRODUCTION_IDENTITY_FAILED_ROLLED_BACK"
    assert _exact_bytes(layout) == layout.original_bytes
    assert not active.exists()


def test_distinct_root_control_verifies_production_final_and_commits(layout: Any) -> None:
    plan = production.build_production_identity_plan(**layout.arguments)
    observed: list[str] = []

    def compatibility_verify() -> bool:
        assert layout.transaction.is_dir()
        assert layout.compatibility_transaction != layout.transaction
        assert not layout.compatibility_transaction.exists()
        state = legacy.verify_accepted_identity_state(**layout.compatibility_arguments)
        observed.append(state.state)
        assert not layout.compatibility_transaction.exists()
        return True

    result = production.apply_production_identity_transaction(
        plan,
        post_verify=compatibility_verify,
    )
    assert result.status == "DINOV3_PRODUCTION_IDENTITY_REBIND_COMMITTED"
    assert observed == ["production_final"]
    assert not layout.transaction.exists()
    assert not layout.compatibility_transaction.exists()


def test_command_derived_child_verifies_while_active_root_exists(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    layout: Any,
) -> None:
    observations: list[dict[str, Any]] = []

    def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        compatibility = _compatibility_root_from(command)
        observations.append(
            {
                "active_exists": layout.transaction.is_dir(),
                "compatibility_distinct": compatibility != layout.transaction,
                "compatibility_absent_before": not compatibility.exists(),
            }
        )

        def value(option: str) -> Path:
            assert command.count(option) == 1
            return Path(command[command.index(option) + 1])

        repository = Path(command[2]).resolve(strict=True).parents[1]
        state = legacy.verify_accepted_identity_state(
            repository_root=repository,
            runtime_profile_path=value("--runtime-profile"),
            snapshot_profile_path=value("--snapshot-profile"),
            extractor_profile_path=value("--extractor-profile"),
            transaction_root=compatibility,
            runtime_root=value("--runtime-root"),
            snapshot_root=value("--snapshot-root"),
        )
        observations[-1]["state"] = state.state
        observations[-1]["compatibility_absent_after"] = not compatibility.exists()
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(BASE.POST_VERIFY_MARKERS),
            stderr="",
        )

    assert BASE._apply_with_child(monkeypatch, layout, child) == 0
    output = capsys.readouterr().out
    assert observations == [
        {
            "active_exists": True,
            "compatibility_distinct": True,
            "compatibility_absent_before": True,
            "state": "production_final",
            "compatibility_absent_after": True,
        }
    ]
    assert "DINOV3_PRODUCTION_POST_VERIFY_ROOT_ISOLATION_OK" in output
    assert not layout.transaction.exists()


@pytest.mark.parametrize("collision", ["same", "existing"])
def test_compatibility_root_collision_is_preflight_and_mutation_free(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    layout: Any,
    collision: str,
) -> None:
    BASE._configure_production_cli(monkeypatch, layout)
    compatibility = (
        layout.transaction
        if collision == "same"
        else (layout.transaction.parent / "already-existing").resolve()
    )
    if collision == "existing":
        compatibility.mkdir()
    launched: list[list[str]] = []
    monkeypatch.setattr(
        CLI.subprocess,
        "run",
        lambda command, **kwargs: launched.append(command),
    )
    arguments = BASE._production_cli_arguments(layout, "apply") + [
        "--compatibility-transaction-root",
        str(compatibility),
    ]
    assert CLI.main(arguments) == 2
    captured = capsys.readouterr()
    evidence = _evidence(captured)
    assert evidence == {
        "stage": "preflight_root_validation",
        "outer_classification": "DINOV3_PRODUCTION_POST_VERIFY_ROOT_COLLISION",
        "child_classification": "DINOV3_PRODUCTION_POST_VERIFY_ROOT_COLLISION",
        "rollback_status": "not_required",
    }
    assert launched == []
    assert _exact_bytes(layout) == layout.original_bytes
    assert not layout.transaction.exists()


def test_child_nonzero_preserves_classification_with_bounded_path_free_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    layout: Any,
) -> None:
    secret_path = str((layout.repository / "private-secret-token").resolve())

    def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            7,
            stdout=f"raw stdout {secret_path}",
            stderr=(
                "DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED: "
                f"raw stderr token=credential {secret_path}"
            ),
        )

    assert BASE._apply_with_child(monkeypatch, layout, child) == 2
    captured = capsys.readouterr()
    evidence = _evidence(captured)
    assert evidence == {
        "stage": "legacy_compatibility_post_verify",
        "outer_classification": "DINOV3_PRODUCTION_IDENTITY_FAILED_ROLLED_BACK",
        "child_classification": "DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED",
        "rollback_status": "exact",
    }
    rendered = json.dumps(evidence, sort_keys=True)
    for forbidden in (secret_path, "raw stdout", "raw stderr", "credential"):
        assert forbidden not in rendered
        assert forbidden not in captured.out
        assert forbidden not in captured.err
    assert _exact_bytes(layout) == layout.original_bytes
    assert not layout.transaction.exists()


@pytest.mark.parametrize("failure", ["missing-marker", "timeout"])
def test_post_verify_failure_modes_rollback_exactly_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    layout: Any,
    failure: str,
) -> None:
    secret = str((layout.snapshot / "private-checkpoint").resolve())

    def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command + [secret],
                kwargs["timeout"],
                output=f"secret output {secret}",
                stderr=f"secret stderr {secret}",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(BASE.POST_VERIFY_MARKERS[:-1]),
            stderr=f"ignored payload {secret}",
        )

    assert BASE._apply_with_child(monkeypatch, layout, child) == 2
    captured = capsys.readouterr()
    evidence = _evidence(captured)
    expected_child = {
        "missing-marker": "DINOV3_PRODUCTION_POST_VERIFY_MARKERS_MISSING",
        "timeout": "DINOV3_PRODUCTION_POST_VERIFY_TIMEOUT",
    }[failure]
    assert evidence["stage"] == "legacy_compatibility_post_verify"
    assert evidence["outer_classification"] == (
        "DINOV3_PRODUCTION_IDENTITY_FAILED_ROLLED_BACK"
    )
    assert evidence["child_classification"] == expected_child
    assert evidence["rollback_status"] == "exact"
    assert secret not in captured.out and secret not in captured.err
    assert _exact_bytes(layout) == layout.original_bytes
    assert not layout.transaction.exists()
    assert not layout.compatibility_transaction.exists()


def test_runtime_snapshot_sentinels_and_second_fake_apply_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    layout: Any,
) -> None:
    sentinels = (layout.runtime_data.read_bytes(), layout.snapshot_data.read_bytes())
    observations: list[dict[str, Any]] = []
    child = BASE._successful_child(layout, observations)
    assert BASE._apply_with_child(monkeypatch, layout, child) == 0
    assert BASE._apply_with_child(monkeypatch, layout, child) == 0
    assert len(observations) == 1
    assert (layout.runtime_data.read_bytes(), layout.snapshot_data.read_bytes()) == sentinels
    assert not layout.transaction.exists()


def test_legacy_registry_and_production_terminal_semantics_are_preserved(layout: Any) -> None:
    assert legacy.verify_accepted_identity_state(**layout.compatibility_arguments).state == (
        "registry_pin_final"
    )
    plan = production.build_production_identity_plan(**layout.arguments)
    production.apply_production_identity_transaction(plan)
    assert legacy.verify_accepted_identity_state(**layout.compatibility_arguments).state == (
        "production_final"
    )


def test_isolation_path_performs_no_network_package_or_git_operation() -> None:
    source = (ROOT / "scripts/rebind_dinov3_production_identity.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "requests.",
        "urllib.",
        "socket.",
        "pip install",
        "git commit",
        "git push",
        "huggingface_hub",
    ):
        assert forbidden not in source
