"""Plan, apply, verify, or recover the local DINOv3 identity rebind."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from vision_encoder_alignment.identity_rebind import (  # noqa: E402
    IdentityRebindError,
    apply_identity_transaction,
    build_identity_plan,
    default_transaction_root,
    recover_identity_transaction,
    verify_accepted_identity_state,
)


DEFAULT_RUNTIME_PROFILE = (
    REPOSITORY_ROOT
    / "configs"
    / "runtime_profiles"
    / "dinov3_isolated_manual_v0.json"
)
DEFAULT_SNAPSHOT_PROFILE = (
    REPOSITORY_ROOT
    / "configs"
    / "model_snapshots"
    / "dinov3_vit_b16_r5931719_v0.json"
)
DEFAULT_EXTRACTOR_PROFILE = (
    REPOSITORY_ROOT
    / "configs"
    / "extractor_profiles"
    / "dinov3_vit_b16_r5931719_isolated_v0.json"
)
EXPERIMENT_CONFIG = REPOSITORY_ROOT / "configs" / "smoke_cpu.yaml"
TARGET_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"

_OFFLINE_FLAGS = {
    "CUDA_VISIBLE_DEVICES": "",
    "DO_NOT_TRACK": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
_SAFE_INHERITED_ENVIRONMENT_NAMES = (
    "ALLUSERSPROFILE",
    "APPDATA",
    "COMSPEC",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "NUMBER_OF_PROCESSORS",
    "OS",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
)
_CHECKPOINT_MARKERS = (
    "DINOV3_SNAPSHOT_MANIFEST_OK",
    "DINOV3_CHECKPOINT_CONFIG_LOAD_OK",
    "DINOV3_CHECKPOINT_MODEL_LOAD_OK",
    "DINOV3_CHECKPOINT_LOADING_INFO_OK",
    "DINOV3_CHECKPOINT_CPU_FP32_OK",
    "DINOV3_CHECKPOINT_FROZEN_EVAL_OK",
    "DINOV3_CHECKPOINT_RNG_ISOLATION_OK",
    "DINOV3_CHECKPOINT_NETWORK_GUARD_OK",
    "DINOV3_CHECKPOINT_NO_FORWARD_OK",
    "DINOV3_CHECKPOINT_LOAD_SMOKE_OK",
)
_PROCESS_MARKERS = (
    "DINOV3_PERSISTENT_RUNTIME_MANIFEST_OK",
    "DINOV3_PERSISTENT_INBOUND_ARTIFACT_OK",
    "DINOV3_PERSISTENT_OUTBOUND_ARTIFACT_OK",
    "DINOV3_PERSISTENT_RUNTIME_NETWORK_GUARD_OK",
    "DINOV3_PERSISTENT_MANUAL_PREPROCESSING_OK",
    "DINOV3_PERSISTENT_RANDOM_FORWARD_OK",
    "DINOV3_PERSISTENT_REPEAT_DETERMINISM_OK",
    "DINOV3_PERSISTENT_RUNTIME_BOUNDARY_OK",
)


def _offline_environment() -> dict[str, str]:
    # Construct a non-secret allowlist instead of copying the process
    # environment and then trying to remove credential-shaped names.
    environment = {
        name: os.environ[name]
        for name in _SAFE_INHERITED_ENVIRONMENT_NAMES
        if name in os.environ
    }
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment.update(_OFFLINE_FLAGS)
    return environment


def _append_root(arguments: list[str], option: str, root: Path | None) -> None:
    if root is not None:
        arguments.extend((option, str(root)))


def _run_checked(
    label: str,
    arguments: Sequence[str],
    required_markers: Sequence[str],
) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=REPOSITORY_ROOT,
        env=_offline_environment(),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if completed.returncode != 0:
        raise IdentityRebindError(
            f"{label} exited with status {completed.returncode}"
        )
    combined = completed.stdout + "\n" + completed.stderr
    missing = [marker for marker in required_markers if marker not in combined]
    if missing:
        raise IdentityRebindError(
            f"{label} omitted required markers: {', '.join(missing)}"
        )
    for marker in required_markers:
        print(marker)
    return combined


def _validate_repository_config() -> Mapping[str, Any]:
    from vision_encoder_alignment.config import (
        ReadinessStatus,
        assess_runtime_readiness,
        load_experiment_config,
        load_model_registry,
        validate_model_registry,
    )

    registry_path = REPOSITORY_ROOT / "configs" / "models.yaml"
    registry = load_model_registry(registry_path)
    validate_model_registry(registry)
    experiment = load_experiment_config(EXPERIMENT_CONFIG, registry)
    model = registry["models"]["dinov3_vit_b16"]
    if model.get("revision") != TARGET_REVISION:
        raise IdentityRebindError("repository config API returned the wrong revision")
    if model.get("revision_status") != "pinned":
        raise IdentityRebindError(
            "repository config API returned a non-pinned revision status"
        )
    if model.get("access") != "user_must_confirm_license_and_auth":
        raise IdentityRebindError("repository config API access policy changed")
    readiness_config = copy.deepcopy(experiment)
    readiness_config["models"] = ["dinov3_vit_b16"]
    with tempfile.TemporaryDirectory(prefix="vea_t007a3b_readiness_") as temp:
        temporary_root = Path(temp)
        dataset_root = temporary_root / "dataset"
        dataset_root.mkdir()
        manifest = temporary_root / "smoke_10.jsonl"
        manifest.write_text(
            "".join(
                json.dumps({"image_id": f"temporary-{index}"}) + "\n"
                for index in range(10)
            ),
            encoding="utf-8",
        )
        readiness_config["dataset"]["manifest"] = str(manifest)
        root_environment = {
            str(readiness_config["dataset"]["root_env"]): str(dataset_root)
        }
        readiness = assess_runtime_readiness(
            readiness_config,
            registry,
            repository_root=REPOSITORY_ROOT,
            environ=root_environment,
            cached_model_keys={"dinov3_vit_b16"},
        )
    issue_codes = tuple(issue.code for issue in readiness.issues)
    if "checkpoint_revision_unresolved" in issue_codes:
        raise IdentityRebindError("generic readiness still reports unresolved revision")
    if "checkpoint_not_cached" in issue_codes:
        raise IdentityRebindError("generic readiness still reports uncached DINOv3")
    if "gated_model_access_unconfirmed" not in issue_codes:
        raise IdentityRebindError("generic readiness lost the conservative gated issue")
    if readiness.status is not ReadinessStatus.BLOCKED_BY_USER_ACTION:
        raise IdentityRebindError("generic readiness returned an unexpected status")
    print("DINOV3_REGISTRY_CONFIG_API_OK")
    print("DINOV3_GENERIC_READINESS_BOUNDARY_OK")
    return {
        "issue_codes": list(issue_codes),
        "status": readiness.status.value,
    }


def _runtime_verify(arguments: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-B",
        str(REPOSITORY_ROOT / "scripts" / "provision_dinov3_runtime.py"),
        "verify",
        "--profile",
        str(arguments.runtime_profile),
    ]
    _append_root(command, "--runtime-root", arguments.runtime_root)
    _run_checked(
        "persistent runtime verify",
        command,
        ("DINOV3_PERSISTENT_RUNTIME_VERIFY_OK",),
    )


def _snapshot_verify(arguments: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-B",
        str(REPOSITORY_ROOT / "scripts" / "acquire_dinov3_snapshot.py"),
        "verify",
        "--profile",
        str(arguments.snapshot_profile),
        "--runtime-profile",
        str(arguments.runtime_profile),
    ]
    _append_root(command, "--snapshot-root", arguments.snapshot_root)
    _append_root(command, "--runtime-root", arguments.runtime_root)
    _run_checked(
        "external snapshot verify",
        command,
        (
            "DINOV3_SNAPSHOT_MANIFEST_OK",
            "DINOV3_EXACT_SNAPSHOT_VERIFY_OK",
        ),
    )


def _checkpoint_smoke(arguments: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-B",
        str(REPOSITORY_ROOT / "scripts" / "run_dinov3_checkpoint_load_smoke.py"),
        "--snapshot-profile",
        str(arguments.snapshot_profile),
        "--runtime-profile",
        str(arguments.runtime_profile),
    ]
    _append_root(command, "--snapshot-root", arguments.snapshot_root)
    _append_root(command, "--runtime-root", arguments.runtime_root)
    _run_checked("checkpoint-load smoke", command, _CHECKPOINT_MARKERS)


def _process_smoke(arguments: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-B",
        str(REPOSITORY_ROOT / "scripts" / "run_dinov3_isolated_smoke.py"),
        "--profile",
        str(arguments.runtime_profile),
    ]
    _append_root(command, "--runtime-root", arguments.runtime_root)
    _run_checked("process-boundary smoke", command, _PROCESS_MARKERS)


def _post_verify(arguments: argparse.Namespace) -> Callable[[], bool]:
    def verify() -> bool:
        _validate_repository_config()
        _runtime_verify(arguments)
        _snapshot_verify(arguments)
        _checkpoint_smoke(arguments)
        _process_smoke(arguments)
        return True

    return verify


def _rollback_verify(arguments: argparse.Namespace) -> Callable[[], bool]:
    def verify() -> bool:
        _runtime_verify(arguments)
        _snapshot_verify(arguments)
        return True

    return verify


def _identity_arguments(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "repository_root": REPOSITORY_ROOT,
        "runtime_profile_path": arguments.runtime_profile,
        "snapshot_profile_path": arguments.snapshot_profile,
        "transaction_root": arguments.transaction_root,
        "runtime_root": arguments.runtime_root,
        "snapshot_root": arguments.snapshot_root,
    }


def _verification_arguments(arguments: argparse.Namespace) -> dict[str, Any]:
    values = _identity_arguments(arguments)
    values["extractor_profile_path"] = arguments.extractor_profile
    return values


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verified compensating DINOv3 identity rebind"
    )
    parser.add_argument(
        "command",
        choices=("plan", "apply", "verify", "recover", "print-default-transaction-root"),
    )
    parser.add_argument(
        "--runtime-profile",
        type=Path,
        default=DEFAULT_RUNTIME_PROFILE,
    )
    parser.add_argument(
        "--snapshot-profile",
        type=Path,
        default=DEFAULT_SNAPSHOT_PROFILE,
    )
    parser.add_argument(
        "--extractor-profile",
        type=Path,
        default=DEFAULT_EXTRACTOR_PROFILE,
    )
    parser.add_argument("--transaction-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "print-default-transaction-root":
        print(default_transaction_root())
        return 0
    values = _identity_arguments(arguments)
    if arguments.command == "plan":
        plan = build_identity_plan(**values)
        _print_json(plan.to_dict())
        print("DINOV3_IDENTITY_REBIND_PLAN_OK")
        print("DINOV3_IDENTITY_REBIND_CANDIDATE_GRAPH_OK")
        return 0
    if arguments.command == "apply":
        plan = build_identity_plan(**values)
        print("DINOV3_IDENTITY_REBIND_PLAN_OK")
        print("DINOV3_IDENTITY_REBIND_CANDIDATE_GRAPH_OK")
        result = apply_identity_transaction(
            plan,
            **values,
            post_verify=_post_verify(arguments),
            rollback_verify=_rollback_verify(arguments),
        )
        _print_json(result.to_dict())
        print(result.status)
        return 0
    if arguments.command == "verify":
        verification = verify_accepted_identity_state(
            **_verification_arguments(arguments)
        )
        print(verification.marker)
        _validate_repository_config()
        _runtime_verify(arguments)
        _snapshot_verify(arguments)
        _print_json(verification.to_dict())
        print("DINOV3_IDENTITY_REBIND_VERIFY_OK")
        return 0
    result = recover_identity_transaction(
        **values,
        rollback_verify=_rollback_verify(arguments),
    )
    _print_json(result.to_dict())
    print(result.status)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IdentityRebindError as error:
        print(f"{error.classification}:{error}", file=sys.stderr)
        raise SystemExit(1) from error
