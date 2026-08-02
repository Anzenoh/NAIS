from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
from types import ModuleType
from typing import Any, Callable

import pytest
import conftest as _test_support

from conftest import reconstruct_dinov3_registry_pin_graph
import vision_encoder_alignment.dinov3_production_identity as production
import vision_encoder_alignment.identity_rebind as legacy


ROOT = Path(__file__).resolve().parents[2]
_LIVE_PROFILE_BYTES = (
    ROOT / production.EXTRACTOR_PROFILE_RELATIVE_PATH
).read_bytes()
_test_support._DINOV3_LIVE_PROFILE_BYTES = len(_LIVE_PROFILE_BYTES)
_test_support._DINOV3_LIVE_PROFILE_SHA256 = hashlib.sha256(_LIVE_PROFILE_BYTES).hexdigest()
POST_VERIFY_MARKERS = (
    "DINOV3_IDENTITY_STATE_PRODUCTION_FINAL_OK",
    "DINOV3_IDENTITY_REBIND_VERIFY_OK",
    "DINOV3_PERSISTENT_RUNTIME_VERIFY_OK",
    "DINOV3_SNAPSHOT_MANIFEST_OK",
    "DINOV3_EXACT_SNAPSHOT_VERIFY_OK",
)


def test_production_cli_exposes_explicit_resolved_graph_identity() -> None:
    assert production.APPLE_GRAPH_ID == "dinov3_apple_linux_cuda_fp32_resolved_v2"


def _load_script(relative: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY_CLI = _load_script(
    "scripts/rebind_dinov3_identity.py", "t007b2_legacy_identity_cli"
)
PRODUCTION_CLI = _load_script(
    "scripts/rebind_dinov3_production_identity.py", "t007b2_production_identity_cli"
)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class IntegrationLayout:
    repository: Path
    runtime: Path
    snapshot: Path
    transaction: Path
    compatibility_transaction: Path
    runtime_profile: Path
    snapshot_profile: Path
    extractor_profile: Path
    paths: tuple[Path, ...]
    original_bytes: tuple[bytes, ...]
    runtime_data: Path
    snapshot_data: Path

    @property
    def arguments(self) -> dict[str, Any]:
        return {
            "repository_root": self.repository,
            "runtime_profile_path": self.runtime_profile,
            "snapshot_profile_path": self.snapshot_profile,
            "extractor_profile_path": self.extractor_profile,
            "transaction_root": self.transaction,
            "runtime_root": self.runtime,
            "snapshot_root": self.snapshot,
        }

    @property
    def compatibility_arguments(self) -> dict[str, Any]:
        value = dict(self.arguments)
        value["transaction_root"] = self.compatibility_transaction
        return value


@pytest.fixture
def layout(tmp_path: Path) -> IntegrationLayout:
    repository = (tmp_path / "repository").resolve()
    runtime = (tmp_path / "runtime").resolve()
    snapshot = (tmp_path / "snapshot").resolve()
    transaction = (tmp_path / "production-transaction").resolve()
    compatibility_transaction = (tmp_path / "compatibility-transaction").resolve()
    repository.mkdir()
    runtime.mkdir()
    snapshot.mkdir()
    dinov3_registry_pin_graph_path = reconstruct_dinov3_registry_pin_graph(
        tmp_path / "registry-pin-graph"
    )
    repository_files = set(production.PRODUCTION_SOURCE_PATHS) | {
        production.RUNTIME_PROFILE_RELATIVE_PATH,
        production.SNAPSHOT_PROFILE_RELATIVE_PATH,
        production.EXTRACTOR_PROFILE_RELATIVE_PATH,
        "scripts/provision_dinov3_runtime.py",
        "scripts/acquire_dinov3_snapshot.py",
        "scripts/rebind_dinov3_identity.py",
    }
    registry_repository = dinov3_registry_pin_graph_path / "repository"
    for relative in repository_files:
        source = registry_repository / relative
        _copy(source if source.is_file() else ROOT / relative, repository / relative)

    _copy(
        dinov3_registry_pin_graph_path / "runtime/runtime-manifest.json",
        runtime / "runtime-manifest.json",
    )
    _copy(
        dinov3_registry_pin_graph_path / "snapshot/snapshot-manifest.json",
        snapshot / "snapshot-manifest.json",
    )
    runtime_data = runtime / "runtime-data.bin"
    snapshot_data = snapshot / "checkpoint-data.bin"
    runtime_data.write_bytes(b"runtime-data-preserved")
    snapshot_data.write_bytes(b"snapshot-data-preserved")
    runtime_profile = repository / production.RUNTIME_PROFILE_RELATIVE_PATH
    snapshot_profile = repository / production.SNAPSHOT_PROFILE_RELATIVE_PATH
    extractor_profile = repository / production.EXTRACTOR_PROFILE_RELATIVE_PATH
    paths = (
        runtime_profile,
        snapshot_profile,
        repository / "scripts/provision_dinov3_runtime.py",
        repository / production.SNAPSHOT_BOUNDARY_RELATIVE_PATH,
        repository / "scripts/acquire_dinov3_snapshot.py",
        runtime / "runtime-manifest.json",
        snapshot / "snapshot-manifest.json",
        extractor_profile,
    )
    return IntegrationLayout(
        repository=repository,
        runtime=runtime,
        snapshot=snapshot,
        transaction=transaction,
        compatibility_transaction=compatibility_transaction,
        runtime_profile=runtime_profile,
        snapshot_profile=snapshot_profile,
        extractor_profile=extractor_profile,
        paths=paths,
        original_bytes=tuple(path.read_bytes() for path in paths),
        runtime_data=runtime_data,
        snapshot_data=snapshot_data,
    )


def _production_cli_arguments(layout: IntegrationLayout, command: str) -> list[str]:
    return [
        command,
        "--runtime-profile",
        str(layout.runtime_profile),
        "--snapshot-profile",
        str(layout.snapshot_profile),
        "--extractor-profile",
        str(layout.extractor_profile),
        "--transaction-root",
        str(layout.transaction),
        "--runtime-root",
        str(layout.runtime),
        "--snapshot-root",
        str(layout.snapshot),
    ]


def _legacy_cli_arguments(layout: IntegrationLayout, command: str) -> list[str]:
    return [
        command,
        "--runtime-profile",
        str(layout.runtime_profile),
        "--snapshot-profile",
        str(layout.snapshot_profile),
        "--extractor-profile",
        str(layout.extractor_profile),
        "--transaction-root",
        str(layout.compatibility_transaction),
        "--runtime-root",
        str(layout.runtime),
        "--snapshot-root",
        str(layout.snapshot),
    ]


def _configure_legacy_cli(monkeypatch: pytest.MonkeyPatch, layout: IntegrationLayout) -> None:
    monkeypatch.setattr(LEGACY_CLI, "REPOSITORY_ROOT", layout.repository)
    monkeypatch.setattr(LEGACY_CLI, "_validate_repository_config", lambda: {})
    monkeypatch.setattr(LEGACY_CLI, "_runtime_verify", lambda arguments: None)
    monkeypatch.setattr(LEGACY_CLI, "_snapshot_verify", lambda arguments: None)


def _configure_production_cli(
    monkeypatch: pytest.MonkeyPatch, layout: IntegrationLayout
) -> None:
    monkeypatch.setattr(PRODUCTION_CLI, "REPOSITORY_ROOT", layout.repository)
    monkeypatch.setattr(PRODUCTION_CLI, "SOURCE_ROOT", ROOT / "src")


def _successful_child(
    layout: IntegrationLayout,
    observations: list[dict[str, Any]],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observations.append({"command": command, "kwargs": kwargs})
        assert layout.transaction.is_dir()

        def value(option: str) -> Path:
            assert command.count(option) == 1
            index = command.index(option)
            assert index + 1 < len(command)
            return Path(command[index + 1])

        compatibility_transaction = value("--transaction-root")
        assert compatibility_transaction != layout.transaction
        assert not compatibility_transaction.exists()
        repository = Path(command[2]).resolve(strict=True).parents[1]
        state = legacy.verify_accepted_identity_state(
            repository_root=repository,
            runtime_profile_path=value("--runtime-profile"),
            snapshot_profile_path=value("--snapshot-profile"),
            extractor_profile_path=value("--extractor-profile"),
            transaction_root=compatibility_transaction,
            runtime_root=value("--runtime-root"),
            snapshot_root=value("--snapshot-root"),
        )
        assert state.state == "production_final"
        assert not compatibility_transaction.exists()
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(POST_VERIFY_MARKERS) + "\n",
            stderr="",
        )

    return run


def _apply_with_child(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
    child: Callable[..., subprocess.CompletedProcess[str]],
) -> int:
    _configure_production_cli(monkeypatch, layout)
    monkeypatch.setattr(PRODUCTION_CLI.subprocess, "run", child)
    return PRODUCTION_CLI.main(_production_cli_arguments(layout, "apply"))


def test_legacy_cli_registry_state_marker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    layout: IntegrationLayout,
) -> None:
    _configure_legacy_cli(monkeypatch, layout)
    assert LEGACY_CLI.main(_legacy_cli_arguments(layout, "verify")) == 0
    output = capsys.readouterr().out
    assert "DINOV3_IDENTITY_STATE_REGISTRY_PIN_FINAL_OK" in output
    assert "DINOV3_IDENTITY_STATE_PRODUCTION_FINAL_OK" not in output


def test_legacy_cli_production_state_and_common_markers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    layout: IntegrationLayout,
) -> None:
    plan = production.build_production_identity_plan(**layout.arguments)
    production.apply_production_identity_transaction(plan)
    _configure_legacy_cli(monkeypatch, layout)
    assert LEGACY_CLI.main(_legacy_cli_arguments(layout, "verify")) == 0
    output = capsys.readouterr().out
    assert "DINOV3_IDENTITY_STATE_PRODUCTION_FINAL_OK" in output
    assert "DINOV3_IDENTITY_REBIND_VERIFY_OK" in output


def test_hybrid_graph_cli_is_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    plan = production.build_production_identity_plan(**layout.arguments)
    production.apply_production_identity_transaction(plan)
    layout.runtime_profile.write_bytes(layout.original_bytes[0])
    _configure_legacy_cli(monkeypatch, layout)
    with pytest.raises(legacy.IdentityRebindError):
        LEGACY_CLI.main(_legacy_cli_arguments(layout, "verify"))


def test_production_transaction_commits_with_internal_post_verifier(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    observations: list[dict[str, Any]] = []
    assert _apply_with_child(
        monkeypatch, layout, _successful_child(layout, observations)
    ) == 0
    assert observations
    command = observations[0]["command"]
    assert command.count("--transaction-root") == 1
    child_root = Path(command[command.index("--transaction-root") + 1])
    assert child_root != layout.transaction
    assert not child_root.exists()
    assert legacy.verify_accepted_identity_state(
        **layout.compatibility_arguments
    ).state == "production_final"


def test_post_verifier_runs_before_transaction_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    observed: list[bool] = []

    def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.append(layout.transaction.is_dir())
        return subprocess.CompletedProcess(
            command, 0, stdout="\n".join(POST_VERIFY_MARKERS), stderr=""
        )

    assert _apply_with_child(monkeypatch, layout, child) == 0
    assert observed == [True]
    assert not layout.transaction.exists()


def test_post_verifier_nonzero_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="redacted")

    assert _apply_with_child(monkeypatch, layout, child) == 2
    assert tuple(path.read_bytes() for path in layout.paths) == layout.original_bytes


def test_post_verifier_missing_marker_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(POST_VERIFY_MARKERS[:-1]),
            stderr="",
        )

    assert _apply_with_child(monkeypatch, layout, child) == 2
    assert tuple(path.read_bytes() for path in layout.paths) == layout.original_bytes


def test_production_apply_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    observations: list[dict[str, Any]] = []
    child = _successful_child(layout, observations)
    assert _apply_with_child(monkeypatch, layout, child) == 0
    assert _apply_with_child(monkeypatch, layout, child) == 0
    assert len(observations) == 1


def test_failed_post_verifier_restores_exact_old_bytes(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    assert _apply_with_child(monkeypatch, layout, child) == 2
    assert tuple(path.read_bytes() for path in layout.paths) == layout.original_bytes


def test_failed_post_verifier_leaves_no_transaction_residue(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    assert _apply_with_child(monkeypatch, layout, child) == 2
    assert not layout.transaction.exists()
    assert not layout.compatibility_transaction.exists()


def test_runtime_and_snapshot_data_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    before = (_sha256(layout.runtime_data), _sha256(layout.snapshot_data))
    assert _apply_with_child(monkeypatch, layout, _successful_child(layout, [])) == 0
    assert (_sha256(layout.runtime_data), _sha256(layout.snapshot_data)) == before


def test_cli_source_has_no_network_package_or_git_operation() -> None:
    source = (
        ROOT / "scripts/rebind_dinov3_production_identity.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "requests.",
        "urllib.",
        "socket.",
        "pip install",
        "git commit",
        "git push",
    ):
        assert forbidden not in source
    assert "--graph-id" in source


def test_post_verifier_uses_argument_list_and_root_overrides(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    observations: list[dict[str, Any]] = []
    assert _apply_with_child(
        monkeypatch, layout, _successful_child(layout, observations)
    ) == 0
    command = observations[0]["command"]
    assert isinstance(command, list)
    assert "--runtime-root" in command and str(layout.runtime) in command
    assert "--snapshot-root" in command and str(layout.snapshot) in command
    assert "--extractor-profile" in command and str(layout.extractor_profile) in command


def test_post_verifier_uses_shell_false_and_finite_timeout(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    observations: list[dict[str, Any]] = []
    assert _apply_with_child(
        monkeypatch, layout, _successful_child(layout, observations)
    ) == 0
    kwargs = observations[0]["kwargs"]
    assert kwargs["shell"] is False
    assert isinstance(kwargs["timeout"], int) and 0 < kwargs["timeout"] <= 1_800


def test_post_verifier_timeout_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
) -> None:
    def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    assert _apply_with_child(monkeypatch, layout, child) == 2
    assert tuple(path.read_bytes() for path in layout.paths) == layout.original_bytes
    assert not layout.transaction.exists()


@pytest.mark.parametrize("command", ["plan", "apply", "recover"])
def test_legacy_mutating_commands_reject_production_final(
    monkeypatch: pytest.MonkeyPatch,
    layout: IntegrationLayout,
    command: str,
) -> None:
    plan = production.build_production_identity_plan(**layout.arguments)
    production.apply_production_identity_transaction(plan)
    _configure_legacy_cli(monkeypatch, layout)
    with pytest.raises(legacy.IdentityRebindError):
        LEGACY_CLI.main(_legacy_cli_arguments(layout, command))
