"""Plan, apply, verify, or recover the DINOv3 production identity graph."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from vision_encoder_alignment.dinov3_production_identity import (  # noqa: E402
    APPLE_EXTRACTOR_PROFILE_RELATIVE_PATH,
    APPLE_GRAPH_ID,
    APPLE_UNRESOLVED_GRAPH_ID,
    APPLE_RUNTIME_PROFILE_RELATIVE_PATH,
    DINOv3ProductionIdentityError,
    LEGACY_GRAPH_ID,
    apply_production_identity_transaction,
    apply_resolved_graph_migration,
    build_production_identity_plan,
    build_resolved_graph_migration_plan,
    default_transaction_root,
    recover_production_identity_transaction,
    recover_resolved_graph_migration,
    verify_production_identity,
    validate_versioned_production_graph,
)
from vision_encoder_alignment.identity_rebind import (  # noqa: E402
    default_transaction_root as default_legacy_transaction_root,
)


DEFAULT_RUNTIME_PROFILE = REPOSITORY_ROOT / "configs/runtime_profiles/dinov3_isolated_manual_v0.json"
DEFAULT_SNAPSHOT_PROFILE = REPOSITORY_ROOT / "configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json"
DEFAULT_EXTRACTOR_PROFILE = REPOSITORY_ROOT / "configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json"
DEFAULT_APPLE_RUNTIME_PROFILE = REPOSITORY_ROOT / APPLE_RUNTIME_PROFILE_RELATIVE_PATH
DEFAULT_APPLE_EXTRACTOR_PROFILE = REPOSITORY_ROOT / APPLE_EXTRACTOR_PROFILE_RELATIVE_PATH
_POST_VERIFY_TIMEOUT_SECONDS = 1_800
_COMPATIBILITY_TRANSACTION_ID = "dinov3-production-identity-compatibility-verify-v1"
_POST_VERIFY_ROOT_ISOLATION_MARKER = (
    "DINOV3_PRODUCTION_POST_VERIFY_ROOT_ISOLATION_OK"
)
_FAILURE_EVIDENCE_MARKER = "DINOV3_PRODUCTION_IDENTITY_FAILURE_EVIDENCE:"
_ROOT_COLLISION_CLASSIFICATION = "DINOV3_PRODUCTION_POST_VERIFY_ROOT_COLLISION"
_UNCLASSIFIED_CHILD = "DINOV3_CHILD_FAILURE_UNCLASSIFIED"
_CLASSIFICATION_RE = re.compile(r"^([A-Z][A-Z0-9_]+)(?::|$)")
_SAFE_CHILD_CLASSIFICATIONS = frozenset(
    {
        "DINOV3_IDENTITY_BASELINE_MISMATCH",
        "DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED",
        "DINOV3_IDENTITY_FINAL_VERIFY_FAILED",
        "DINOV3_IDENTITY_REBIND_ERROR",
        "DINOV3_IDENTITY_TRANSACTION_COLLISION",
        "DINOV3_PRODUCTION_IDENTITY_TRANSACTION_COLLISION",
        "DINOV3_PRODUCTION_POST_VERIFY_LAUNCH_FAILED",
        "DINOV3_PRODUCTION_POST_VERIFY_MARKERS_MISSING",
        "DINOV3_PRODUCTION_POST_VERIFY_NONZERO",
        "DINOV3_PRODUCTION_POST_VERIFY_TIMEOUT",
        _ROOT_COLLISION_CLASSIFICATION,
        _UNCLASSIFIED_CHILD,
    }
)
_SAFE_FAILURE_STAGES = frozenset(
    {
        "preflight_root_validation",
        "transaction_prepare",
        "transaction_apply",
        "final_hash_verify",
        "legacy_compatibility_post_verify",
        "transaction_cleanup",
    }
)
_POST_VERIFY_MARKERS = (
    "DINOV3_IDENTITY_STATE_PRODUCTION_FINAL_OK",
    "DINOV3_IDENTITY_REBIND_VERIFY_OK",
    "DINOV3_PERSISTENT_RUNTIME_VERIFY_OK",
    "DINOV3_SNAPSHOT_MANIFEST_OK",
    "DINOV3_EXACT_SNAPSHOT_VERIFY_OK",
)
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


def _offline_environment() -> dict[str, str]:
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


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(os.path.normpath(str(first))) == os.path.normcase(
        os.path.normpath(str(second))
    )


def _normalized_absolute(value: Path | str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute():
        raise _classified_error(
            _ROOT_COLLISION_CLASSIFICATION,
            "compatibility transaction root must be absolute",
            stage="preflight_root_validation",
        )
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        Path(os.path.normcase(str(path))).relative_to(
            Path(os.path.normcase(str(root)))
        )
    except ValueError:
        return False
    return True


def _roots_overlap(first: Path, second: Path) -> bool:
    return _is_relative_to(first, second) or _is_relative_to(second, first)


def _is_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return bool(flag and attributes & flag)
    except OSError:
        return True


def _classified_error(
    classification: str,
    message: str,
    *,
    stage: str,
    child_classification: str | None = None,
) -> DINOv3ProductionIdentityError:
    error = DINOv3ProductionIdentityError(message, classification)
    error.failure_stage = stage
    error.child_classification = child_classification or classification
    return error


def _validate_safe_absent_root(path: Path, *, stage: str) -> None:
    if path.exists() or path.is_symlink():
        raise _classified_error(
            _ROOT_COLLISION_CLASSIFICATION,
            "compatibility transaction root must be absent",
            stage=stage,
        )
    cursor = path
    while not cursor.exists():
        if cursor.is_symlink():
            raise _classified_error(
                _ROOT_COLLISION_CLASSIFICATION,
                "compatibility transaction root crosses an unsafe link",
                stage=stage,
            )
        parent = cursor.parent
        if parent == cursor:
            raise _classified_error(
                _ROOT_COLLISION_CLASSIFICATION,
                "compatibility transaction root has no safe ancestor",
                stage=stage,
            )
        cursor = parent
    if _is_reparse(cursor):
        raise _classified_error(
            _ROOT_COLLISION_CLASSIFICATION,
            "compatibility transaction root crosses an unsafe link",
            stage=stage,
        )
    try:
        canonical = cursor.resolve(strict=True)
    except OSError:
        raise _classified_error(
            _ROOT_COLLISION_CLASSIFICATION,
            "compatibility transaction root ancestor cannot be resolved",
            stage=stage,
        ) from None
    if not _same_path(cursor, canonical):
        raise _classified_error(
            _ROOT_COLLISION_CLASSIFICATION,
            "compatibility transaction root crosses an unsafe link",
            stage=stage,
        )


def _default_compatibility_transaction_root(production_root: Path) -> Path:
    return production_root.parent / _COMPATIBILITY_TRANSACTION_ID


def _validate_compatibility_transaction_root(
    compatibility_root: Path | str,
    *,
    production_root: Path | str,
    legacy_root: Path | str,
    protected_roots: Sequence[Path | str],
) -> Path:
    candidate = _normalized_absolute(compatibility_root)
    active = _normalized_absolute(production_root)
    legacy = _normalized_absolute(legacy_root)
    if _roots_overlap(candidate, active) or _roots_overlap(candidate, legacy):
        raise _classified_error(
            _ROOT_COLLISION_CLASSIFICATION,
            "compatibility transaction root collides with an active transaction root",
            stage="preflight_root_validation",
        )
    normalized_protected = tuple(
        Path(root).resolve(strict=True) for root in protected_roots
    )
    if any(_roots_overlap(candidate, root) for root in normalized_protected):
        raise _classified_error(
            _ROOT_COLLISION_CLASSIFICATION,
            "compatibility transaction root overlaps a protected root",
            stage="preflight_root_validation",
        )
    if any(_is_relative_to(candidate.parent, root) for root in normalized_protected):
        raise _classified_error(
            _ROOT_COLLISION_CLASSIFICATION,
            "compatibility transaction root parent is inside a protected root",
            stage="preflight_root_validation",
        )
    _validate_safe_absent_root(candidate, stage="preflight_root_validation")
    return candidate


def _resolve_compatibility_transaction_root(
    namespace: argparse.Namespace,
    plan: Any,
) -> Path:
    production_root = _normalized_absolute(plan._transaction_root)
    selected = namespace.compatibility_transaction_root
    if selected is None:
        selected = _default_compatibility_transaction_root(production_root)
    protected_roots = (
        REPOSITORY_ROOT.resolve(strict=True),
        plan._paths[5].parent.resolve(strict=True),
        plan._paths[6].parent.resolve(strict=True),
    )
    return _validate_compatibility_transaction_root(
        selected,
        production_root=production_root,
        legacy_root=default_legacy_transaction_root(environ=os.environ),
        protected_roots=protected_roots,
    )


def _extract_child_classification(stdout: str, stderr: str) -> str:
    for payload in (stdout, stderr):
        for line in payload.splitlines():
            match = _CLASSIFICATION_RE.match(line)
            if match is not None and match.group(1) in _SAFE_CHILD_CLASSIFICATIONS:
                return match.group(1)
    return _UNCLASSIFIED_CHILD


def _record_post_verify_failure(
    namespace: argparse.Namespace,
    *,
    classification: str,
    child_classification: str,
) -> DINOv3ProductionIdentityError:
    namespace._failure_stage = "legacy_compatibility_post_verify"
    namespace._child_classification = child_classification
    return _classified_error(
        classification,
        "legacy compatibility post-verifier failed",
        stage="legacy_compatibility_post_verify",
        child_classification=child_classification,
    )


def _failure_evidence(
    error: DINOv3ProductionIdentityError,
    namespace: argparse.Namespace,
) -> Mapping[str, str] | None:
    stage = getattr(namespace, "_failure_stage", None)
    child = getattr(namespace, "_child_classification", None)
    cursor: BaseException | None = error
    for _ in range(8):
        if cursor is None:
            break
        candidate_stage = getattr(cursor, "failure_stage", None)
        candidate_child = getattr(cursor, "child_classification", None)
        if stage is None and candidate_stage in _SAFE_FAILURE_STAGES:
            stage = candidate_stage
        if child is None and candidate_child in _SAFE_CHILD_CLASSIFICATIONS:
            child = candidate_child
        cursor = cursor.__cause__
    if stage not in _SAFE_FAILURE_STAGES:
        return None
    if child not in _SAFE_CHILD_CLASSIFICATIONS:
        child = _UNCLASSIFIED_CHILD
    rollback_status = {
        "DINOV3_PRODUCTION_IDENTITY_FAILED_ROLLED_BACK": "exact",
        "DINOV3_PRODUCTION_IDENTITY_ROLLBACK_FAILED": "failed",
        _ROOT_COLLISION_CLASSIFICATION: "not_required",
    }.get(error.classification, "not_required")
    return {
        "stage": stage,
        "outer_classification": error.classification,
        "child_classification": child,
        "rollback_status": rollback_status,
    }


def _legacy_post_verify(
    namespace: argparse.Namespace,
    compatibility_root: Path,
):
    def verify() -> bool:
        _validate_safe_absent_root(
            compatibility_root,
            stage="legacy_compatibility_post_verify",
        )
        command = [
            sys.executable,
            "-B",
            str(REPOSITORY_ROOT / "scripts" / "rebind_dinov3_identity.py"),
            "verify",
            "--runtime-profile",
            str(namespace.runtime_profile),
            "--snapshot-profile",
            str(namespace.snapshot_profile),
            "--extractor-profile",
            str(namespace.extractor_profile),
            "--transaction-root",
            str(compatibility_root),
        ]
        _append_root(command, "--runtime-root", namespace.runtime_root)
        _append_root(command, "--snapshot-root", namespace.snapshot_root)
        process_failure: DINOv3ProductionIdentityError | None = None
        try:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=_offline_environment(),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_POST_VERIFY_TIMEOUT_SECONDS,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            process_failure = _record_post_verify_failure(
                namespace,
                classification="DINOV3_PRODUCTION_POST_VERIFY_TIMEOUT",
                child_classification="DINOV3_PRODUCTION_POST_VERIFY_TIMEOUT",
            )
        except OSError:
            process_failure = _record_post_verify_failure(
                namespace,
                classification="DINOV3_PRODUCTION_POST_VERIFY_LAUNCH_FAILED",
                child_classification="DINOV3_PRODUCTION_POST_VERIFY_LAUNCH_FAILED",
            )
        if process_failure is not None:
            raise process_failure from None
        _validate_safe_absent_root(
            compatibility_root,
            stage="legacy_compatibility_post_verify",
        )
        if completed.returncode != 0:
            child = _extract_child_classification(
                completed.stdout or "", completed.stderr or ""
            )
            raise _record_post_verify_failure(
                namespace,
                classification="DINOV3_PRODUCTION_POST_VERIFY_NONZERO",
                child_classification=child,
            )
        combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
        missing = [marker for marker in _POST_VERIFY_MARKERS if marker not in combined]
        if missing:
            raise _record_post_verify_failure(
                namespace,
                classification="DINOV3_PRODUCTION_POST_VERIFY_MARKERS_MISSING",
                child_classification="DINOV3_PRODUCTION_POST_VERIFY_MARKERS_MISSING",
            )
        for marker in _POST_VERIFY_MARKERS:
            print(marker)
        print(_POST_VERIFY_ROOT_ISOLATION_MARKER)
        return True

    return verify


def _arguments(namespace: argparse.Namespace) -> dict[str, Any]:
    return {
        "repository_root": REPOSITORY_ROOT,
        "runtime_profile_path": namespace.runtime_profile,
        "snapshot_profile_path": namespace.snapshot_profile,
        "extractor_profile_path": namespace.extractor_profile,
        "transaction_root": namespace.transaction_root,
        "runtime_root": namespace.runtime_root,
        "snapshot_root": namespace.snapshot_root,
    }


def _print_plan(plan: Any) -> None:
    value = {
        "already_rebound": plan.already_rebound,
        "candidate_validated": plan.candidate_validated,
        "mutations": [
            {
                "label": row.label,
                "new_sha256": row.new_sha256,
                "old_sha256": row.old_sha256,
                "path_classification": row.path_classification,
                "relative_path": row.relative_path,
            }
            for row in plan.mutations
        ],
        "production_sources": [
            {"path": row.path, "role": row.role, "sha256": row.sha256}
            for row in plan.source_identities
        ],
        "transaction_id": plan.transaction_id,
    }
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan", "apply", "verify", "recover", "print-default-transaction-root"),
    )
    parser.add_argument("--runtime-profile", type=Path, default=DEFAULT_RUNTIME_PROFILE)
    parser.add_argument("--snapshot-profile", type=Path, default=DEFAULT_SNAPSHOT_PROFILE)
    parser.add_argument("--extractor-profile", type=Path, default=DEFAULT_EXTRACTOR_PROFILE)
    parser.add_argument("--transaction-root", type=Path)
    parser.add_argument("--compatibility-transaction-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument(
        "--graph-id",
        choices=(LEGACY_GRAPH_ID, APPLE_UNRESOLVED_GRAPH_ID, APPLE_GRAPH_ID),
        default=LEGACY_GRAPH_ID,
        help="explicit graph selector; never inferred from hostname or CUDA",
    )
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--migration-path", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.graph_id == APPLE_UNRESOLVED_GRAPH_ID:
            if args.command not in {"plan", "verify"}:
                raise DINOv3ProductionIdentityError(
                    "historical unresolved Apple graph is permanently read-only",
                    "DINOV3_APPLE_GRAPH_MUTATION_FORBIDDEN",
                )
        if args.graph_id in {APPLE_UNRESOLVED_GRAPH_ID, APPLE_GRAPH_ID}:
            if args.graph_id == APPLE_GRAPH_ID and args.command in {"plan", "apply"}:
                if args.candidate_root is None or args.transaction_root is None or not args.migration_path:
                    raise DINOv3ProductionIdentityError(
                        "resolved migration requires candidate root, transaction root, and explicit paths"
                    )
                migration = build_resolved_graph_migration_plan(
                    repository_root=REPOSITORY_ROOT,
                    candidate_root=args.candidate_root,
                    relative_paths=tuple(sorted(args.migration_path)),
                    transaction_root=args.transaction_root,
                )
                if args.command == "plan":
                    print(json.dumps({
                        "graph_id": migration.graph_id,
                        "mutation_count": len(migration.mutations),
                        "paths": [item.relative_path for item in migration.mutations],
                    }, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
                    print("DINOV3_RESOLVED_APPLE_GRAPH_PLAN_OK")
                    return 0
                apply_resolved_graph_migration(migration)
                print("DINOV3_RESOLVED_APPLE_GRAPH_APPLY_OK")
                return 0
            if args.graph_id == APPLE_GRAPH_ID and args.command == "recover":
                if args.transaction_root is None:
                    raise DINOv3ProductionIdentityError("resolved recovery requires transaction root")
                recover_resolved_graph_migration(
                    repository_root=REPOSITORY_ROOT,
                    transaction_root=args.transaction_root,
                )
                print("DINOV3_RESOLVED_APPLE_GRAPH_RECOVER_OK")
                return 0
            runtime = (
                DEFAULT_APPLE_RUNTIME_PROFILE
                if args.runtime_profile == DEFAULT_RUNTIME_PROFILE
                else args.runtime_profile
            )
            extractor = (
                DEFAULT_APPLE_EXTRACTOR_PROFILE
                if args.extractor_profile == DEFAULT_EXTRACTOR_PROFILE
                else args.extractor_profile
            )
            graph = validate_versioned_production_graph(
                REPOSITORY_ROOT,
                graph_id=args.graph_id,
                runtime_profile_path=runtime,
                extractor_profile_path=extractor,
                snapshot_profile_path=args.snapshot_profile,
            )
            print(json.dumps({
                "graph_id": graph.graph_id,
                "nodes": graph.nodes,
                "edges": graph.edges,
                "ready_for_p1_extraction": graph.ready_for_p1_extraction,
            }, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
            print("DINOV3_APPLE_PRODUCTION_GRAPH_VERIFY_OK")
            return 0
        if args.command == "print-default-transaction-root":
            print(default_transaction_root())
            return 0
        arguments = _arguments(args)
        if args.command == "recover":
            result = recover_production_identity_transaction(**arguments)
            print(result.status)
            return 0
        if args.command == "verify":
            result = verify_production_identity(**arguments)
            validate_versioned_production_graph(
                REPOSITORY_ROOT,
                graph_id=LEGACY_GRAPH_ID,
                runtime_profile_path=args.runtime_profile,
                extractor_profile_path=args.extractor_profile,
                snapshot_profile_path=args.snapshot_profile,
            )
            print(result.status)
            return 0
        plan = build_production_identity_plan(**arguments)
        if args.command == "plan":
            _print_plan(plan)
            print("DINOV3_PRODUCTION_IDENTITY_PLAN_OK")
            print("DINOV3_PRODUCTION_IDENTITY_CANDIDATE_GRAPH_OK")
            return 0
        compatibility_root = _resolve_compatibility_transaction_root(args, plan)
        result = apply_production_identity_transaction(
            plan,
            post_verify=_legacy_post_verify(args, compatibility_root),
        )
        print(result.status)
        return 0
    except DINOv3ProductionIdentityError as exc:
        if args.command == "apply":
            evidence = _failure_evidence(exc, args)
            if evidence is not None:
                print(
                    _FAILURE_EVIDENCE_MARKER
                    + json.dumps(
                        evidence,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                )
        print(exc.classification, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
