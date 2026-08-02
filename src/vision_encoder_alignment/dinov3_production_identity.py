"""Import-light DINOv3 production-source identity and transaction boundary.

This module deliberately uses only the Python standard library.  It derives
the complete identity graph from live production source bytes, then applies
the bounded repository/external-manifest update as a compensating transaction.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import pprint
import re
import shutil
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn


TRANSACTION_ID = "dinov3-production-identity-rebind-v1"
JOURNAL_SCHEMA_ID = "vea.dinov3_production_identity_transaction.v0"
EXTRACTOR_PROFILE_SCHEMA_ID = "vea.dinov3_isolated_extractor_profile.v1"
APPLE_EXTRACTOR_PROFILE_SCHEMA_ID = "vea.dinov3_apple_extractor_profile.v1"
LEGACY_GRAPH_ID = "dinov3_legacy_windows_cpu_v1"
APPLE_UNRESOLVED_GRAPH_ID = "dinov3_apple_linux_cuda_fp32_unresolved_v1"
APPLE_GRAPH_ID = "dinov3_apple_linux_cuda_fp32_resolved_v2"

RUNTIME_PROFILE_RELATIVE_PATH = (
    "configs/runtime_profiles/dinov3_isolated_manual_v0.json"
)
SNAPSHOT_PROFILE_RELATIVE_PATH = (
    "configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json"
)
EXTRACTOR_PROFILE_RELATIVE_PATH = (
    "configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json"
)
APPLE_RUNTIME_PROFILE_RELATIVE_PATH = "configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json"
APPLE_EXTRACTOR_PROFILE_RELATIVE_PATH = (
    "configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json"
)
SEMANTIC_PROFILE_RELATIVE_PATH = (
    "configs/semantic_probes/dinov3_vit_b16_r5931719_semantics_v0.json"
)

RUNTIME_SOURCE_PATHS = (
    "configs/models.yaml",
    "pyproject.toml",
    "scripts/dinov3_extraction_worker.py",
    "src/vision_encoder_alignment/dinov3_extraction_boundary.py",
    "src/vision_encoder_alignment/dinov3_production_identity.py",
    "src/vision_encoder_alignment/extractors/__init__.py",
    "src/vision_encoder_alignment/extractors/base.py",
    "src/vision_encoder_alignment/extractors/clip.py",
    "src/vision_encoder_alignment/extractors/dinov3.py",
    "src/vision_encoder_alignment/extractors/siglip2.py",
    "src/vision_encoder_alignment/feature_artifacts/__init__.py",
    "src/vision_encoder_alignment/feature_artifacts/checksum.py",
    "src/vision_encoder_alignment/feature_artifacts/io.py",
    "src/vision_encoder_alignment/feature_artifacts/schema.py",
    "src/vision_encoder_alignment/feature_artifacts/validation.py",
    "src/vision_encoder_alignment/production_runtime_contract.py",
    "src/vision_encoder_alignment/runtime_boundary.py",
)
SNAPSHOT_BOUNDARY_RELATIVE_PATH = "src/vision_encoder_alignment/snapshot_boundary.py"
PRODUCTION_SOURCE_PATHS = tuple(sorted((*RUNTIME_SOURCE_PATHS, SNAPSHOT_BOUNDARY_RELATIVE_PATH)))

_SOURCE_ROLES = {
    "configs/models.yaml": "model_registry",
    "pyproject.toml": "package_manifest",
    "scripts/dinov3_extraction_worker.py": "isolated_worker",
    "src/vision_encoder_alignment/dinov3_extraction_boundary.py": "extraction_boundary",
    "src/vision_encoder_alignment/dinov3_production_identity.py": "production_identity",
    "src/vision_encoder_alignment/extractors/__init__.py": "extractor_exports",
    "src/vision_encoder_alignment/extractors/base.py": "extractor_contract",
    "src/vision_encoder_alignment/extractors/clip.py": "peer_extractor",
    "src/vision_encoder_alignment/extractors/dinov3.py": "public_extractor",
    "src/vision_encoder_alignment/extractors/siglip2.py": "peer_extractor",
    "src/vision_encoder_alignment/feature_artifacts/__init__.py": "feature_artifact_boundary",
    "src/vision_encoder_alignment/feature_artifacts/checksum.py": "feature_artifact_boundary",
    "src/vision_encoder_alignment/feature_artifacts/io.py": "feature_artifact_boundary",
    "src/vision_encoder_alignment/feature_artifacts/schema.py": "feature_artifact_boundary",
    "src/vision_encoder_alignment/feature_artifacts/validation.py": "feature_artifact_boundary",
    "src/vision_encoder_alignment/production_runtime_contract.py": "production_runtime_contract",
    "src/vision_encoder_alignment/runtime_boundary.py": "runtime_boundary",
    SNAPSHOT_BOUNDARY_RELATIVE_PATH: "snapshot_boundary",
}
_TARGET_ROWS = (
    ("runtime_profile", "repository", RUNTIME_PROFILE_RELATIVE_PATH),
    ("snapshot_profile", "repository", SNAPSHOT_PROFILE_RELATIVE_PATH),
    ("provisioner", "repository", "scripts/provision_dinov3_runtime.py"),
    ("snapshot_boundary", "repository", SNAPSHOT_BOUNDARY_RELATIVE_PATH),
    ("acquire_script", "repository", "scripts/acquire_dinov3_snapshot.py"),
    ("runtime_manifest", "persistent_runtime", "runtime-manifest.json"),
    ("snapshot_manifest", "external_snapshot", "snapshot-manifest.json"),
    ("extractor_profile", "repository", EXTRACTOR_PROFILE_RELATIVE_PATH),
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 2 * 1024 * 1024


class DINOv3ProductionIdentityError(RuntimeError):
    """Fail-closed production identity error with a stable classification."""

    def __init__(self, message: str, classification: str = "DINOV3_PRODUCTION_IDENTITY_ERROR") -> None:
        super().__init__(message)
        self.classification = classification


@dataclass(frozen=True, slots=True, order=True)
class ProductionSourceIdentity:
    path: str
    role: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ProductionIdentityMutation:
    label: str
    path_classification: str
    relative_path: str
    old_sha256: str
    new_sha256: str


@dataclass(frozen=True, slots=True)
class ProductionIdentityPlan:
    transaction_id: str
    source_identities: tuple[ProductionSourceIdentity, ...]
    mutations: tuple[ProductionIdentityMutation, ...]
    old_hashes: tuple[tuple[str, str], ...]
    new_hashes: tuple[tuple[str, str], ...]
    rollback_order: tuple[str, ...]
    candidate_validated: bool
    already_rebound: bool
    _paths: tuple[Path, ...] = field(repr=False, compare=False)
    _old_bytes: tuple[bytes, ...] = field(repr=False, compare=False)
    _new_bytes: tuple[bytes, ...] = field(repr=False, compare=False)
    _transaction_root: Path = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ProductionIdentityResult:
    status: str
    applied: tuple[str, ...]
    journal_transitions: tuple[str, ...]
    compare_and_swap: str
    rollback_attempted: bool
    rollback_succeeded: bool | None
    cleanup_succeeded: bool
    already_rebound: bool
    final_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class VersionedProductionGraph:
    graph_id: str
    runtime_profile_sha256: str
    extractor_profile_sha256: str
    snapshot_profile_sha256: str
    semantic_profile_sha256: str
    source_identities: tuple[ProductionSourceIdentity, ...]
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    unknown_nodes: tuple[str, ...]
    dangling_edges: tuple[tuple[str, str], ...]
    ready_for_p1_extraction: bool


@dataclass(frozen=True, slots=True)
class ResolvedGraphMutation:
    relative_path: str
    old_sha256: str | None
    new_sha256: str


@dataclass(frozen=True, slots=True)
class ResolvedGraphMigrationPlan:
    graph_id: str
    mutations: tuple[ResolvedGraphMutation, ...]
    candidate_validated: bool
    _repository: Path = field(repr=False, compare=False)
    _transaction_root: Path = field(repr=False, compare=False)
    _candidate_bytes: tuple[bytes, ...] = field(repr=False, compare=False)


def _fail(message: str, classification: str = "DINOV3_PRODUCTION_IDENTITY_ERROR") -> NoReturn:
    raise DINOv3ProductionIdentityError(message, classification)


def build_resolved_graph_migration_plan(
    *,
    repository_root: Path | str,
    candidate_root: Path | str,
    relative_paths: Sequence[str],
    transaction_root: Path | str,
) -> ResolvedGraphMigrationPlan:
    """Bind an explicit resolved-graph candidate to current live bytes.

    This authoring transaction never chooses a graph from host or CUDA state.
    It supports both replacements and no-overwrite creation of new paths.
    """

    repository = _existing_root(repository_root, "repository_root")
    candidate = _existing_root(candidate_root, "candidate_root")
    transaction = Path(transaction_root)
    if not transaction.is_absolute() or transaction.exists():
        _fail("resolved graph transaction root is unsafe or already exists")
    normalized = tuple(_safe_relative(item, "relative_path") for item in relative_paths)
    if not normalized or normalized != tuple(sorted(normalized)) or len(set(normalized)) != len(normalized):
        _fail("resolved graph paths must be a sorted unique non-empty set")
    mutations: list[ResolvedGraphMutation] = []
    payloads: list[bytes] = []
    for relative in normalized:
        source = candidate.joinpath(*relative.split("/"))
        if source.is_symlink() or not source.is_file() or not source.resolve(strict=True).is_relative_to(candidate):
            _fail(f"resolved graph candidate is missing or unsafe: {relative}")
        destination = repository.joinpath(*relative.split("/"))
        if destination.exists() and (destination.is_symlink() or not destination.is_file()):
            _fail(f"resolved graph destination is unsafe: {relative}")
        payload = source.read_bytes()
        if not payload:
            _fail(f"resolved graph candidate is empty: {relative}")
        old_sha = _sha256_file(destination) if destination.is_file() else None
        mutations.append(ResolvedGraphMutation(relative, old_sha, _sha256(payload)))
        payloads.append(payload)
    return ResolvedGraphMigrationPlan(
        APPLE_GRAPH_ID,
        tuple(mutations),
        True,
        repository,
        transaction.resolve(strict=False),
        tuple(payloads),
    )


def apply_resolved_graph_migration(
    plan: ResolvedGraphMigrationPlan,
    *,
    post_verify: Callable[[], Any] | None = None,
    failure_injector: Callable[[str, int], Any] | None = None,
) -> tuple[str, ...]:
    """Atomically apply a resolved graph plan or restore exact starting state."""

    if not isinstance(plan, ResolvedGraphMigrationPlan) or not plan.candidate_validated:
        _fail("resolved graph apply requires a validated plan")
    root = plan._transaction_root
    if root.exists():
        _fail("resolved graph transaction root already exists")
    for mutation in plan.mutations:
        destination = plan._repository.joinpath(*mutation.relative_path.split("/"))
        actual = _sha256_file(destination) if destination.is_file() else None
        if actual != mutation.old_sha256:
            _fail("resolved graph compare-and-swap precondition failed", "DINOV3_PRODUCTION_IDENTITY_CONCURRENT_MODIFICATION")
    applied: list[int] = []
    journal: dict[str, Any] | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
        (root / "backups").mkdir()
        (root / "candidates").mkdir()
        targets = []
        for index, (mutation, payload) in enumerate(zip(plan.mutations, plan._candidate_bytes, strict=True)):
            if mutation.old_sha256 is not None:
                destination = plan._repository.joinpath(*mutation.relative_path.split("/"))
                _write_file(root / f"backups/{index:02d}.bin", destination.read_bytes(), exclusive=True)
            _write_file(root / f"candidates/{index:02d}.bin", payload, exclusive=True)
            targets.append({
                "backup_present": mutation.old_sha256 is not None,
                "new_sha256": mutation.new_sha256,
                "old_sha256": mutation.old_sha256,
                "relative_path": mutation.relative_path,
            })
        journal = {
            "applied_index": -1,
            "graph_id": APPLE_GRAPH_ID,
            "schema_id": "vea.dinov3_resolved_graph_transaction.v1",
            "state": "prepared",
            "targets": targets,
        }
        _write_file(root / "journal.json", _canonical_json(journal), exclusive=True)
        _inject(failure_injector, "after_prepare", -1)
        for index, (mutation, payload) in enumerate(zip(plan.mutations, plan._candidate_bytes, strict=True)):
            destination = plan._repository.joinpath(*mutation.relative_path.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            actual = _sha256_file(destination) if destination.is_file() else None
            if actual != mutation.old_sha256:
                _fail("resolved graph compare-and-swap failed", "DINOV3_PRODUCTION_IDENTITY_CONCURRENT_MODIFICATION")
            _inject(failure_injector, "before_replace", index)
            _same_parent_replace(destination, payload, mutation.new_sha256, index)
            applied.append(index)
            journal["applied_index"] = index
            journal["state"] = "applying"
            _write_journal(root, journal)
            _inject(failure_injector, "after_replace", index)
        if post_verify is not None and post_verify() is False:
            _fail("resolved graph post-verifier returned false")
        for mutation in plan.mutations:
            destination = plan._repository.joinpath(*mutation.relative_path.split("/"))
            if _sha256_file(destination) != mutation.new_sha256:
                _fail("resolved graph final identity mismatch")
        _cleanup_transaction(root)
        return tuple(mutation.relative_path for mutation in plan.mutations)
    except Exception as apply_error:
        rollback_error: Exception | None = None
        try:
            for index in reversed(applied):
                mutation = plan.mutations[index]
                destination = plan._repository.joinpath(*mutation.relative_path.split("/"))
                if mutation.old_sha256 is None:
                    if destination.is_file() and _sha256_file(destination) == mutation.new_sha256:
                        destination.unlink()
                else:
                    backup = (root / f"backups/{index:02d}.bin").read_bytes()
                    _same_parent_replace(destination, backup, mutation.old_sha256, index)
            _cleanup_transaction(root, require_journal=journal is not None)
        except Exception as exc:  # pragma: no cover - preserved journal is the evidence
            rollback_error = exc
        if rollback_error is not None:
            raise DINOv3ProductionIdentityError(
                "resolved graph rollback failed", "DINOV3_PRODUCTION_IDENTITY_ROLLBACK_FAILED"
            ) from rollback_error
        raise DINOv3ProductionIdentityError(
            "resolved graph migration failed and rolled back",
            "DINOV3_PRODUCTION_IDENTITY_FAILED_ROLLED_BACK",
        ) from apply_error


def recover_resolved_graph_migration(
    *, repository_root: Path | str, transaction_root: Path | str
) -> tuple[str, ...]:
    """Restore exact pre-migration bytes from a preserved resolved-graph journal."""

    repository = _existing_root(repository_root, "repository_root")
    root = Path(transaction_root)
    if not root.is_absolute() or not root.is_dir() or _is_reparse(root):
        _fail("resolved graph recovery root is missing or unsafe")
    journal_path = root / "journal.json"
    if journal_path.is_symlink() or not journal_path.is_file():
        _fail("resolved graph recovery journal is missing or unsafe")
    value = _strict_json(journal_path.read_bytes(), "resolved graph transaction journal")
    if set(value) != {"applied_index", "graph_id", "schema_id", "state", "targets"}:
        _fail("resolved graph recovery journal fields are invalid")
    if value["graph_id"] != APPLE_GRAPH_ID or value["schema_id"] != "vea.dinov3_resolved_graph_transaction.v1":
        _fail("resolved graph recovery journal identity mismatch")
    targets = value["targets"]
    applied_index = value["applied_index"]
    if not isinstance(targets, list) or not isinstance(applied_index, int):
        _fail("resolved graph recovery journal contents are invalid")
    restored: list[str] = []
    for index in reversed(range(min(applied_index + 1, len(targets)))):
        row = targets[index]
        if not isinstance(row, dict) or set(row) != {"backup_present", "new_sha256", "old_sha256", "relative_path"}:
            _fail("resolved graph recovery target is invalid")
        relative = _safe_relative(row["relative_path"], "recovery.relative_path")
        destination = repository.joinpath(*relative.split("/"))
        new_sha = row["new_sha256"]
        if not isinstance(new_sha, str) or _SHA256_RE.fullmatch(new_sha) is None:
            _fail("resolved graph recovery new identity is invalid")
        if not destination.is_file() or _sha256_file(destination) != new_sha:
            _fail("resolved graph recovery found an unknown live identity")
        if row["backup_present"] is True:
            old_sha = row["old_sha256"]
            if not isinstance(old_sha, str) or _SHA256_RE.fullmatch(old_sha) is None:
                _fail("resolved graph recovery old identity is invalid")
            backup = root / f"backups/{index:02d}.bin"
            if backup.is_symlink() or not backup.is_file() or _sha256_file(backup) != old_sha:
                _fail("resolved graph recovery backup identity mismatch")
            _same_parent_replace(destination, backup.read_bytes(), old_sha, index)
        elif row["backup_present"] is False and row["old_sha256"] is None:
            destination.unlink()
        else:
            _fail("resolved graph recovery backup state is invalid")
        restored.append(relative)
    _cleanup_transaction(root)
    return tuple(restored)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256(path.read_bytes())
    except OSError as exc:
        raise DINOv3ProductionIdentityError("unable to hash an identity file") from exc


def _reject_constant(value: str) -> NoReturn:
    _fail("non-finite JSON constant is forbidden")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate JSON object key")
        result[key] = value
    return result


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    if len(payload) > _MAX_JSON_BYTES:
        _fail(f"{label} exceeds the bounded JSON size")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DINOv3ProductionIdentityError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


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
    except OSError as exc:
        raise DINOv3ProductionIdentityError("unable to inspect an identity path") from exc


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_relative(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{label} is not a repository-relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} or ":" in part for part in value.split("/"))
    ):
        _fail(f"{label} is not a repository-relative path")
    return value


def _existing_root(value: Path | str, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute() or _is_reparse(candidate):
        _fail(f"{label} must be an absolute non-link directory")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise DINOv3ProductionIdentityError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        _fail(f"{label} is not a directory")
    return resolved


def _safe_file(root: Path, relative: str, label: str) -> Path:
    _safe_relative(relative, label)
    current = root
    for part in relative.split("/"):
        current = current / part
        if not current.exists() or _is_reparse(current):
            _fail(f"{label} is missing or crosses a link")
    resolved = current.resolve(strict=True)
    if not _contains(root, resolved) or not resolved.is_file():
        _fail(f"{label} is not a regular file within its root")
    return resolved


def _default_external_root(value: Mapping[str, Any], environ: Mapping[str, str], label: str) -> Path:
    variable = value.get("environment_variable")
    parts = value.get("relative_parts")
    if not isinstance(variable, str) or not isinstance(parts, list) or not parts:
        _fail(f"{label} external-root contract is invalid")
    base = environ.get(variable)
    if not isinstance(base, str) or not base:
        _fail(f"{label} environment variable is unavailable")
    if any(not isinstance(part, str) or not part or part in {".", ".."} for part in parts):
        _fail(f"{label} relative parts are invalid")
    return Path(base).joinpath(*parts)


def default_transaction_root(environ: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if environ is None else environ
    base = environment.get("LOCALAPPDATA")
    if not isinstance(base, str) or not base:
        _fail("LOCALAPPDATA is unavailable for the default transaction root")
    return Path(base) / "VEA" / "transactions" / TRANSACTION_ID


def _read_profile(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    if _is_reparse(path) or not path.is_file():
        _fail(f"{label} must be a regular non-link file")
    payload = path.read_bytes()
    return payload, _strict_json(payload, label)


def _parse_source_rows(value: object) -> tuple[ProductionSourceIdentity, ...]:
    if not isinstance(value, list):
        _fail("profile.production_sources must be a list")
    rows: list[ProductionSourceIdentity] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"path", "role", "sha256"}:
            _fail(f"profile.production_sources[{index}] has invalid fields")
        path = _safe_relative(item["path"], f"profile.production_sources[{index}].path")
        role = item["role"]
        digest = item["sha256"]
        if path in seen:
            _fail("profile.production_sources contains a duplicate path")
        if path not in _SOURCE_ROLES or role != _SOURCE_ROLES[path]:
            _fail("profile.production_sources contains an unknown path or role")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            _fail("profile.production_sources contains an invalid SHA-256")
        seen.add(path)
        rows.append(ProductionSourceIdentity(path, role, digest))
    if tuple(row.path for row in rows) != PRODUCTION_SOURCE_PATHS:
        _fail("profile.production_sources is not the exact sorted production allowlist")
    return tuple(rows)


def load_production_source_identities(
    extractor_profile: Path | str | Mapping[str, Any],
) -> tuple[ProductionSourceIdentity, ...]:
    """Load the strict v1 production source rows from a profile or mapping."""

    if isinstance(extractor_profile, Mapping):
        value = dict(extractor_profile)
    else:
        _, value = _read_profile(Path(extractor_profile), "extractor profile")
    if value.get("schema_id") not in {
        EXTRACTOR_PROFILE_SCHEMA_ID,
        APPLE_EXTRACTOR_PROFILE_SCHEMA_ID,
    }:
        _fail("extractor profile is not an accepted versioned schema v1")
    return _parse_source_rows(value.get("production_sources"))


def validate_production_sources(
    repository_root: Path | str,
    identities: Sequence[ProductionSourceIdentity] | Path | str | Mapping[str, Any],
) -> tuple[ProductionSourceIdentity, ...]:
    """Validate the exact source allowlist, roles, regular files, sizes, and hashes."""

    repository = _existing_root(repository_root, "repository_root")
    if isinstance(identities, (str, Path, Mapping)):
        rows = load_production_source_identities(identities)
    else:
        rows = tuple(identities)
        normalized = [
            {"path": row.path, "role": row.role, "sha256": row.sha256}
            if isinstance(row, ProductionSourceIdentity)
            else row
            for row in rows
        ]
        rows = _parse_source_rows(normalized)
    for row in rows:
        source = _safe_file(repository, row.path, f"production source {row.path}")
        try:
            size = source.stat().st_size
        except OSError as exc:
            raise DINOv3ProductionIdentityError("unable to inspect a production source") from exc
        if size <= 0 or _sha256_file(source) != row.sha256:
            _fail(
                f"production source identity mismatch: {row.path}",
                "DINOV3_PRODUCTION_SOURCE_MISMATCH",
            )
    return rows


def validate_versioned_production_graph(
    repository_root: Path | str,
    *,
    graph_id: str,
    runtime_profile_path: Path | str,
    extractor_profile_path: Path | str,
    snapshot_profile_path: Path | str | None = None,
    semantic_profile_path: Path | str | None = None,
) -> VersionedProductionGraph:
    """Validate one complete legacy or Apple DINOv3 graph without auto-selection.

    Every path is compared with its graph-specific repository destination.  A
    caller therefore cannot combine a legacy runtime with an Apple extractor,
    or silently select a graph from hostname or CUDA observations.
    """

    repository = _existing_root(repository_root, "repository_root")
    if graph_id == LEGACY_GRAPH_ID:
        runtime_relative = RUNTIME_PROFILE_RELATIVE_PATH
        extractor_relative = EXTRACTOR_PROFILE_RELATIVE_PATH
        runtime_schema = "vea.dinov3_isolated_runtime_profile.v0"
        extractor_schema = EXTRACTOR_PROFILE_SCHEMA_ID
    elif graph_id in {APPLE_UNRESOLVED_GRAPH_ID, APPLE_GRAPH_ID}:
        runtime_relative = APPLE_RUNTIME_PROFILE_RELATIVE_PATH
        extractor_relative = APPLE_EXTRACTOR_PROFILE_RELATIVE_PATH
        runtime_schema = "vea.p1.production_runtime_profile.v1"
        extractor_schema = APPLE_EXTRACTOR_PROFILE_SCHEMA_ID
    else:
        _fail("unknown versioned production graph", "DINOV3_UNKNOWN_PRODUCTION_GRAPH")

    supplied = (
        (runtime_profile_path, runtime_relative, "runtime profile"),
        (extractor_profile_path, extractor_relative, "extractor profile"),
        (
            SNAPSHOT_PROFILE_RELATIVE_PATH if snapshot_profile_path is None else snapshot_profile_path,
            SNAPSHOT_PROFILE_RELATIVE_PATH,
            "snapshot profile",
        ),
        (
            SEMANTIC_PROFILE_RELATIVE_PATH if semantic_profile_path is None else semantic_profile_path,
            SEMANTIC_PROFILE_RELATIVE_PATH,
            "semantic profile",
        ),
    )
    paths: list[Path] = []
    for candidate, relative, label in supplied:
        path = Path(candidate)
        if not path.is_absolute():
            path = repository.joinpath(*str(path).replace("\\", "/").split("/"))
        expected = _safe_file(repository, relative, label)
        if path.resolve(strict=True) != expected:
            _fail(
                f"{label} is incompatible with {graph_id}",
                "DINOV3_HYBRID_PRODUCTION_GRAPH",
            )
        paths.append(expected)

    runtime_payload, runtime_value = _read_profile(paths[0], "runtime profile")
    extractor_payload, extractor_value = _read_profile(paths[1], "extractor profile")
    snapshot_payload, snapshot_value = _read_profile(paths[2], "snapshot profile")
    semantic_payload, semantic_value = _read_profile(paths[3], "semantic profile")
    for label, payload, value in (
        ("runtime profile", runtime_payload, runtime_value),
        ("extractor profile", extractor_payload, extractor_value),
        ("snapshot profile", snapshot_payload, snapshot_value),
        ("semantic profile", semantic_payload, semantic_value),
    ):
        if _canonical_json(value) != payload:
            _fail(f"{label} is not canonical JSON")
    if runtime_value.get("schema_id") != runtime_schema or extractor_value.get("schema_id") != extractor_schema:
        _fail("runtime/extractor schemas form a hybrid graph", "DINOV3_HYBRID_PRODUCTION_GRAPH")
    if snapshot_value.get("schema_id") != "vea.model_snapshot_profile.v0" or semantic_value.get("schema_id") != "vea.dinov3_semantic_probe_profile.v0":
        _fail("snapshot or semantic profile schema is unsupported")

    source_rows = validate_production_sources(repository, extractor_value)
    runtime_sha = _sha256(runtime_payload)
    snapshot_sha = _sha256(snapshot_payload)
    semantic_sha = _sha256(semantic_payload)
    identities = extractor_value.get("identities")
    if not isinstance(identities, dict):
        _fail("extractor identities are missing")
    if (
        identities.get("runtime_profile_sha256") != runtime_sha
        or identities.get("snapshot_profile_sha256") != snapshot_sha
        or identities.get("semantic_profile_sha256") != semantic_sha
    ):
        _fail("extractor profile has a stale identity edge", "DINOV3_DANGLING_IDENTITY_EDGE")

    if graph_id in {APPLE_UNRESOLVED_GRAPH_ID, APPLE_GRAPH_ID}:
        from vision_encoder_alignment.production_runtime_contract import (
            DINOV3_SELECTOR,
            load_production_runtime_profile,
            validate_profile_source_bindings,
        )

        runtime = load_production_runtime_profile(paths[0])
        if graph_id == APPLE_GRAPH_ID and not runtime.package_metadata_resolved:
            _fail("resolved Apple graph requires resolved package metadata", "DINOV3_DANGLING_IDENTITY_EDGE")
        if graph_id == APPLE_UNRESOLVED_GRAPH_ID and runtime.package_metadata_resolved:
            _fail("unresolved Apple history graph cannot bind resolved metadata", "DINOV3_DANGLING_IDENTITY_EDGE")
        validate_profile_source_bindings(runtime, repository)
        link = extractor_value.get("runtime_profile")
        if (
            runtime.selector != DINOV3_SELECTOR
            or not isinstance(link, dict)
            or set(link) != {"path", "profile_id", "selector", "sha256"}
            or link.get("path") != APPLE_RUNTIME_PROFILE_RELATIVE_PATH
            or link.get("profile_id") != runtime.profile_id
            or link.get("selector") != runtime.selector
            or link.get("sha256") != runtime_sha
        ):
            _fail("Apple runtime/extractor link is incomplete", "DINOV3_HYBRID_PRODUCTION_GRAPH")
        legacy_payload, legacy = _read_profile(
            _safe_file(repository, EXTRACTOR_PROFILE_RELATIVE_PATH, "legacy extractor profile"),
            "legacy extractor profile",
        )
        del legacy_payload
        for key in (
            "checkpoint_files",
            "descriptor",
            "manual_preprocessing",
            "model",
            "tap_catalog",
            "tap_mapping",
            "token_layout",
        ):
            if extractor_value.get(key) != legacy.get(key):
                _fail("Apple graph changed scientific feature identity")
        policies = extractor_value.get("policies")
        if (
            not isinstance(policies, dict)
            or policies.get("cpu_only") is not False
            or policies.get("cuda_required") is not True
            or policies.get("device") != "cuda:0"
            or policies.get("dtype") != "float32"
            or policies.get("ready_for_p1_extraction") is not False
            or policies.get("network_allowed") is not False
            or policies.get("torchvision_allowed") is not False
        ):
            _fail("Apple extractor policies are incomplete")
    else:
        policies = extractor_value.get("policies")
        if not isinstance(policies, dict) or policies.get("cpu_only") is not True:
            _fail("legacy graph lost its CPU-only policy")

    nodes = (
        "extractor_profile",
        "model_registry",
        "package_manifest",
        "production_sources",
        "runtime_profile",
        "semantic_profile",
        "snapshot_profile",
    )
    edges = (
        ("extractor_profile", "production_sources"),
        ("extractor_profile", "runtime_profile"),
        ("extractor_profile", "semantic_profile"),
        ("extractor_profile", "snapshot_profile"),
        ("production_sources", "model_registry"),
        ("production_sources", "package_manifest"),
    )
    return VersionedProductionGraph(
        graph_id=graph_id,
        runtime_profile_sha256=runtime_sha,
        extractor_profile_sha256=_sha256(extractor_payload),
        snapshot_profile_sha256=snapshot_sha,
        semantic_profile_sha256=semantic_sha,
        source_identities=source_rows,
        nodes=nodes,
        edges=edges,
        unknown_nodes=(),
        dangling_edges=(),
        ready_for_p1_extraction=False,
    )


def _assignment_span(text: str, name: str) -> tuple[int, int]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise DINOv3ProductionIdentityError("identity source is not valid Python") from exc
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name) and node.target.id == name
        )
    ]
    if len(matches) != 1:
        _fail(f"expected one top-level assignment for {name}")
    node = matches[0]
    if node.end_lineno is None or node.end_col_offset is None:
        _fail(f"source offsets are unavailable for {name}")
    lines = text.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset
    end = sum(len(line) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
    return start, end


def _replace_assignments(payload: bytes, replacements: Mapping[str, Any]) -> bytes:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DINOv3ProductionIdentityError("identity source is not UTF-8") from exc
    edits: list[tuple[int, int, str]] = []
    for name, value in replacements.items():
        start, end = _assignment_span(text, name)
        rendered = pprint.pformat(value, width=100, sort_dicts=True)
        edits.append((start, end, f"{name} = {rendered}"))
    for start, end, rendered in sorted(edits, reverse=True):
        text = text[:start] + rendered + text[end:]
    candidate = text.encode("utf-8")
    ast.parse(candidate.decode("utf-8"))
    return candidate


def _source_values(payload: bytes, names: Sequence[str]) -> dict[str, Any]:
    tree = ast.parse(payload.decode("utf-8"))
    result: dict[str, Any] = {}
    for name in names:
        matches: list[ast.AST] = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                    matches.append(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
                matches.append(node.value)
        if len(matches) != 1:
            _fail(f"source assignment {name} is not unique")
        result[name] = ast.literal_eval(matches[0])
    return result


def _json_leaf_diffs(left: Any, right: Any, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                result.add((*prefix, str(key)))
            else:
                result.update(_json_leaf_diffs(left[key], right[key], (*prefix, str(key))))
        return result
    if left != right:
        return {prefix}
    return set()


def _layout(
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    extractor_profile_path: Path | str,
    transaction_root: Path | str | None,
    runtime_root: Path | str | None,
    snapshot_root: Path | str | None,
    environ: Mapping[str, str] | None,
) -> tuple[Path, Path, Path, tuple[Path, ...]]:
    environment = os.environ if environ is None else environ
    repository = _existing_root(repository_root, "repository_root")
    expected_profiles = (
        (runtime_profile_path, RUNTIME_PROFILE_RELATIVE_PATH, "runtime profile"),
        (snapshot_profile_path, SNAPSHOT_PROFILE_RELATIVE_PATH, "snapshot profile"),
        (extractor_profile_path, EXTRACTOR_PROFILE_RELATIVE_PATH, "extractor profile"),
    )
    resolved_profiles: list[Path] = []
    for supplied, relative, label in expected_profiles:
        resolved = Path(supplied).resolve(strict=True)
        expected = _safe_file(repository, relative, label)
        if resolved != expected:
            _fail(f"{label} is not the fixed repository destination")
        resolved_profiles.append(resolved)
    runtime_payload, runtime_value = _read_profile(resolved_profiles[0], "runtime profile")
    del runtime_payload
    snapshot_payload, snapshot_value = _read_profile(resolved_profiles[1], "snapshot profile")
    del snapshot_payload
    selected_runtime = (
        _default_external_root(runtime_value["runtime_root"], environment, "runtime")
        if runtime_root is None
        else Path(runtime_root)
    )
    selected_snapshot = (
        _default_external_root(snapshot_value["external_root"], environment, "snapshot")
        if snapshot_root is None
        else Path(snapshot_root)
    )
    runtime = _existing_root(selected_runtime, "runtime_root")
    snapshot = _existing_root(selected_snapshot, "snapshot_root")
    tx = default_transaction_root(environment) if transaction_root is None else Path(transaction_root)
    if not tx.is_absolute():
        _fail("transaction_root must be absolute")
    tx = tx.resolve(strict=False)
    for left, right in ((repository, runtime), (repository, snapshot), (runtime, snapshot)):
        if _contains(left, right) or _contains(right, left):
            _fail("repository, runtime, and snapshot roots must be disjoint")
    if any(_contains(root, tx) or _contains(tx, root) for root in (repository, runtime, snapshot)):
        _fail("transaction_root must not overlap a protected root")
    paths = (
        resolved_profiles[0],
        resolved_profiles[1],
        _safe_file(repository, "scripts/provision_dinov3_runtime.py", "provisioner"),
        _safe_file(repository, SNAPSHOT_BOUNDARY_RELATIVE_PATH, "snapshot boundary"),
        _safe_file(repository, "scripts/acquire_dinov3_snapshot.py", "acquisition script"),
        _safe_file(runtime, "runtime-manifest.json", "runtime manifest"),
        _safe_file(snapshot, "snapshot-manifest.json", "snapshot manifest"),
        resolved_profiles[2],
    )
    return repository, runtime, snapshot, paths


def _derive_candidates(repository: Path, paths: tuple[Path, ...]) -> tuple[tuple[bytes, ...], tuple[ProductionSourceIdentity, ...]]:
    old = tuple(path.read_bytes() for path in paths)
    runtime_value = _strict_json(old[0], "runtime profile")
    snapshot_value = _strict_json(old[1], "snapshot profile")
    runtime_manifest_value = _strict_json(old[5], "runtime manifest")
    snapshot_manifest_value = _strict_json(old[6], "snapshot manifest")
    extractor_value = _strict_json(old[7], "extractor profile")

    source_hashes = {
        relative: _sha256_file(_safe_file(repository, relative, f"source {relative}"))
        for relative in RUNTIME_SOURCE_PATHS
    }
    runtime_before = json.loads(json.dumps(runtime_value))
    runtime_value["current_base"]["source_files"] = [
        {"path": path, "sha256": source_hashes[path]} for path in RUNTIME_SOURCE_PATHS
    ]
    if _json_leaf_diffs(runtime_before, runtime_value) - {
        ("current_base", "source_files"),
        *{
            ("current_base", "source_files", str(index), key)
            for index in range(max(len(runtime_before["current_base"]["source_files"]), len(RUNTIME_SOURCE_PATHS)))
            for key in ("path", "sha256")
        },
    }:
        _fail("runtime profile candidate exceeded its semantic diff allowlist")
    runtime_candidate = _canonical_json(runtime_value)
    runtime_hash = _sha256(runtime_candidate)

    snapshot_before = json.loads(json.dumps(snapshot_value))
    snapshot_value["persistent_runtime"]["profile_sha256"] = runtime_hash
    if _json_leaf_diffs(snapshot_before, snapshot_value) - {
        ("persistent_runtime", "profile_sha256"),
    }:
        _fail("snapshot profile candidate exceeded its semantic diff allowlist")
    snapshot_candidate = _canonical_json(snapshot_value)
    snapshot_hash = _sha256(snapshot_candidate)

    provisioner_candidate = _replace_assignments(
        old[2],
        {"PROFILE_SHA256": runtime_hash, "EXPECTED_CURRENT_SOURCES": source_hashes},
    )
    boundary_candidate = _replace_assignments(
        old[3], {"PERSISTENT_RUNTIME_PROFILE_SHA256": runtime_hash}
    )
    registry_hash = source_hashes["configs/models.yaml"]
    acquire_candidate = _replace_assignments(
        old[4],
        {
            "EXPECTED_PROFILE_SHA256": snapshot_hash,
            "EXPECTED_RUNTIME_PROFILE_SHA256": runtime_hash,
            "EXPECTED_REGISTRY_SHA256": registry_hash,
        },
    )

    runtime_manifest_before = json.loads(json.dumps(runtime_manifest_value))
    runtime_manifest_value["repository_profile_sha256"] = runtime_hash
    if _json_leaf_diffs(runtime_manifest_before, runtime_manifest_value) - {
        ("repository_profile_sha256",)
    }:
        _fail("runtime manifest candidate exceeded its semantic diff allowlist")
    runtime_manifest_candidate = _canonical_json(runtime_manifest_value)

    snapshot_manifest_before = json.loads(json.dumps(snapshot_manifest_value))
    snapshot_manifest_value["repository_profile_sha256"] = snapshot_hash
    if _json_leaf_diffs(snapshot_manifest_before, snapshot_manifest_value) - {
        ("repository_profile_sha256",)
    }:
        _fail("snapshot manifest candidate exceeded its semantic diff allowlist")
    snapshot_manifest_candidate = _canonical_json(snapshot_manifest_value)

    production_hashes = dict(source_hashes)
    production_hashes[SNAPSHOT_BOUNDARY_RELATIVE_PATH] = _sha256(boundary_candidate)
    identities = tuple(
        ProductionSourceIdentity(path, _SOURCE_ROLES[path], production_hashes[path])
        for path in PRODUCTION_SOURCE_PATHS
    )
    extractor_value["schema_id"] = EXTRACTOR_PROFILE_SCHEMA_ID
    extractor_value["production_sources"] = [
        {"path": row.path, "role": row.role, "sha256": row.sha256} for row in identities
    ]
    identity_updates = {
        "runtime_profile_sha256": runtime_hash,
        "snapshot_profile_sha256": snapshot_hash,
        "persistent_runtime_manifest_sha256": _sha256(runtime_manifest_candidate),
        "external_snapshot_manifest_sha256": _sha256(snapshot_manifest_candidate),
    }
    for key, digest in identity_updates.items():
        extractor_value["identities"][key] = digest
    extractor_candidate = _canonical_json(extractor_value)

    return (
        runtime_candidate,
        snapshot_candidate,
        provisioner_candidate,
        boundary_candidate,
        acquire_candidate,
        runtime_manifest_candidate,
        snapshot_manifest_candidate,
        extractor_candidate,
    ), identities


def build_production_identity_plan(
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    extractor_profile_path: Path | str,
    transaction_root: Path | str | None = None,
    runtime_root: Path | str | None = None,
    snapshot_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProductionIdentityPlan:
    """Build the complete candidate graph without mutating any live file."""

    repository, _, _, paths = _layout(
        repository_root=repository_root,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        extractor_profile_path=extractor_profile_path,
        transaction_root=transaction_root,
        runtime_root=runtime_root,
        snapshot_root=snapshot_root,
        environ=environ,
    )
    tx = default_transaction_root(environ) if transaction_root is None else Path(transaction_root)
    tx = tx.resolve(strict=False)
    if tx.exists():
        _fail(
            "transaction root already exists; use recover for an incomplete journal",
            "DINOV3_PRODUCTION_IDENTITY_TRANSACTION_COLLISION",
        )
    old = tuple(path.read_bytes() for path in paths)
    candidates, identities = _derive_candidates(repository, paths)
    mutations = tuple(
        ProductionIdentityMutation(label, classification, relative, _sha256(before), _sha256(after))
        for (label, classification, relative), before, after in zip(
            _TARGET_ROWS, old, candidates, strict=True
        )
    )
    hashes_old = tuple((mutation.label, mutation.old_sha256) for mutation in mutations)
    hashes_new = tuple((mutation.label, mutation.new_sha256) for mutation in mutations)
    return ProductionIdentityPlan(
        transaction_id=TRANSACTION_ID,
        source_identities=identities,
        mutations=mutations,
        old_hashes=hashes_old,
        new_hashes=hashes_new,
        rollback_order=tuple(row[0] for row in reversed(_TARGET_ROWS)),
        candidate_validated=True,
        already_rebound=old == candidates,
        _paths=paths,
        _old_bytes=old,
        _new_bytes=candidates,
        _transaction_root=tx,
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        return


def _write_file(path: Path, payload: bytes, *, exclusive: bool = False) -> None:
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _journal_value(plan: ProductionIdentityPlan, state: str, applied_index: int) -> dict[str, Any]:
    return {
        "applied_index": applied_index,
        "schema_id": JOURNAL_SCHEMA_ID,
        "state": state,
        "targets": [
            {
                "backup": f"backups/{index:02d}.bin",
                "candidate": f"candidates/{index:02d}.bin",
                "label": mutation.label,
                "new_sha256": mutation.new_sha256,
                "old_sha256": mutation.old_sha256,
                "path_classification": mutation.path_classification,
                "relative_path": mutation.relative_path,
            }
            for index, mutation in enumerate(plan.mutations)
        ],
        "transaction_id": TRANSACTION_ID,
    }


def _write_journal(root: Path, value: Mapping[str, Any]) -> None:
    temporary = root / ".journal.tmp"
    _write_file(temporary, _canonical_json(value))
    os.replace(temporary, root / "journal.json")
    _fsync_directory(root)


def _read_journal(path: Path) -> dict[str, Any]:
    value = _strict_json(path.read_bytes(), "transaction journal")
    if set(value) != {"applied_index", "schema_id", "state", "targets", "transaction_id"}:
        _fail("transaction journal has unknown or missing fields")
    if value["schema_id"] != JOURNAL_SCHEMA_ID or value["transaction_id"] != TRANSACTION_ID:
        _fail("transaction journal identity mismatch")
    if value["state"] not in {"prepared", "applying", "applied", "verifying", "rolling_back", "rollback_failed"}:
        _fail("transaction journal state is not recoverable")
    if not isinstance(value["applied_index"], int) or isinstance(value["applied_index"], bool):
        _fail("transaction journal applied index is invalid")
    if not isinstance(value["targets"], list) or len(value["targets"]) != len(_TARGET_ROWS):
        _fail("transaction journal target count mismatch")
    expected_fields = {
        "backup", "candidate", "label", "new_sha256", "old_sha256",
        "path_classification", "relative_path",
    }
    for index, (row, expected) in enumerate(zip(value["targets"], _TARGET_ROWS, strict=True)):
        if not isinstance(row, dict) or set(row) != expected_fields:
            _fail("transaction journal target fields are invalid")
        label, classification, relative = expected
        if (
            row["label"] != label
            or row["path_classification"] != classification
            or row["relative_path"] != relative
            or row["backup"] != f"backups/{index:02d}.bin"
            or row["candidate"] != f"candidates/{index:02d}.bin"
            or not isinstance(row["old_sha256"], str)
            or _SHA256_RE.fullmatch(row["old_sha256"]) is None
            or not isinstance(row["new_sha256"], str)
            or _SHA256_RE.fullmatch(row["new_sha256"]) is None
        ):
            _fail("transaction journal target identity mismatch")
    return value


def _inject(callback: Callable[[str, int], Any] | None, stage: str, index: int) -> None:
    if callback is not None:
        callback(stage, index)


def _same_parent_replace(destination: Path, payload: bytes, expected: str, index: int) -> None:
    stage = destination.with_name(f".{destination.name}.{TRANSACTION_ID}.{index}.tmp")
    try:
        _write_file(stage, payload, exclusive=True)
        if _sha256_file(stage) != expected:
            _fail("same-parent candidate hash mismatch")
        os.replace(stage, destination)
        _fsync_directory(destination.parent)
    finally:
        stage.unlink(missing_ok=True)


def _cleanup_transaction(root: Path, *, require_journal: bool = True) -> None:
    if not root.exists():
        return
    if _is_reparse(root) or (require_journal and not (root / "journal.json").is_file()):
        _fail("refusing to clean an unexpected transaction root")
    shutil.rmtree(root)
    _fsync_directory(root.parent)


def _final_hashes(plan: ProductionIdentityPlan) -> tuple[tuple[str, str], ...]:
    result = tuple(
        (mutation.label, _sha256_file(path))
        for mutation, path in zip(plan.mutations, plan._paths, strict=True)
    )
    if result != plan.new_hashes:
        _fail(
            "final production identity graph is inconsistent",
            "DINOV3_PRODUCTION_IDENTITY_FINAL_VERIFY_FAILED",
        )
    validate_production_sources(plan._paths[0].parents[2], plan.source_identities)
    return result


def apply_production_identity_transaction(
    plan: ProductionIdentityPlan,
    *,
    post_verify: Callable[[], Any] | None = None,
    rollback_verify: Callable[[], Any] | None = None,
    failure_injector: Callable[[str, int], Any] | None = None,
) -> ProductionIdentityResult:
    """Apply a validated plan with compare-and-swap and reverse compensation."""

    if not isinstance(plan, ProductionIdentityPlan) or not plan.candidate_validated:
        _fail("apply requires a validated production identity plan")
    live = tuple(_sha256_file(path) for path in plan._paths)
    old = tuple(digest for _, digest in plan.old_hashes)
    new = tuple(digest for _, digest in plan.new_hashes)
    if live == new:
        final = _final_hashes(plan)
        return ProductionIdentityResult(
            "DINOV3_PRODUCTION_IDENTITY_ALREADY_REBOUND", (), (), "EXACT_FINAL_STATE",
            False, None, not plan._transaction_root.exists(), True, final,
        )
    if live != old:
        _fail(
            "live identity changed after plan construction",
            "DINOV3_PRODUCTION_IDENTITY_CONCURRENT_MODIFICATION",
        )
    root = plan._transaction_root
    if root.exists():
        _fail("transaction root already exists", "DINOV3_PRODUCTION_IDENTITY_TRANSACTION_COLLISION")
    transitions: list[str] = []
    applied: list[int] = []
    journal: dict[str, Any] | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
        (root / "backups").mkdir()
        (root / "candidates").mkdir()
        for index, (before, after) in enumerate(zip(plan._old_bytes, plan._new_bytes, strict=True)):
            _write_file(root / f"backups/{index:02d}.bin", before, exclusive=True)
            _write_file(root / f"candidates/{index:02d}.bin", after, exclusive=True)
        journal = _journal_value(plan, "prepared", -1)
        _write_journal(root, journal)
        transitions.append("prepared")
        _inject(failure_injector, "after_prepare", -1)
        journal["state"] = "applying"
        _write_journal(root, journal)
        transitions.append("applying")
        for index, (mutation, destination, candidate) in enumerate(
            zip(plan.mutations, plan._paths, plan._new_bytes, strict=True)
        ):
            _inject(failure_injector, "before_replace", index)
            if _sha256_file(destination) != mutation.old_sha256:
                _fail(
                    f"compare-and-swap failed for {mutation.label}",
                    "DINOV3_PRODUCTION_IDENTITY_CONCURRENT_MODIFICATION",
                )
            applied.append(index)
            _same_parent_replace(destination, candidate, mutation.new_sha256, index)
            if _sha256_file(destination) != mutation.new_sha256:
                _fail(f"applied hash mismatch for {mutation.label}")
            journal["applied_index"] = index
            _write_journal(root, journal)
            _inject(failure_injector, "after_replace", index)
        journal["state"] = "applied"
        _write_journal(root, journal)
        transitions.append("applied")
        journal["state"] = "verifying"
        _write_journal(root, journal)
        transitions.append("verifying")
        _inject(failure_injector, "before_verify", -1)
        final = _final_hashes(plan)
        if post_verify is not None and post_verify() is False:
            _fail("post-transaction verifier returned false")
        _cleanup_transaction(root)
        transitions.append("committed")
        return ProductionIdentityResult(
            "DINOV3_PRODUCTION_IDENTITY_REBIND_COMMITTED",
            tuple(plan.mutations[index].label for index in applied),
            tuple(transitions),
            "VERIFIED_ALL_DESTINATIONS",
            False,
            None,
            not root.exists(),
            False,
            final,
        )
    except Exception as apply_error:
        if journal is None:
            try:
                _cleanup_transaction(root, require_journal=False)
            except Exception:
                pass
            if isinstance(apply_error, DINOv3ProductionIdentityError):
                raise
            raise DINOv3ProductionIdentityError("transaction preparation failed") from apply_error
        try:
            journal["state"] = "rolling_back"
            _write_journal(root, journal)
            transitions.append("rolling_back")
            for index in reversed(range(len(plan.mutations))):
                mutation = plan.mutations[index]
                destination = plan._paths[index]
                current = _sha256_file(destination)
                if current == mutation.old_sha256:
                    continue
                if current != mutation.new_sha256:
                    _fail("rollback detected an unknown destination identity")
                _inject(failure_injector, "before_rollback", index)
                backup = (root / f"backups/{index:02d}.bin").read_bytes()
                _same_parent_replace(destination, backup, mutation.old_sha256, index)
                _inject(failure_injector, "after_rollback", index)
            if tuple(_sha256_file(path) for path in plan._paths) != old:
                _fail("rollback did not restore exact starting bytes")
            if rollback_verify is not None and rollback_verify() is False:
                _fail("rollback verifier returned false")
            _cleanup_transaction(root)
        except Exception as rollback_error:
            try:
                journal["state"] = "rollback_failed"
                _write_journal(root, journal)
            except Exception:
                pass
            raise DINOv3ProductionIdentityError(
                "production identity rollback failed",
                "DINOV3_PRODUCTION_IDENTITY_ROLLBACK_FAILED",
            ) from rollback_error
        raise DINOv3ProductionIdentityError(
            "production identity transaction failed and rolled back",
            "DINOV3_PRODUCTION_IDENTITY_FAILED_ROLLED_BACK",
        ) from apply_error


def verify_production_identity(
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    extractor_profile_path: Path | str,
    transaction_root: Path | str | None = None,
    runtime_root: Path | str | None = None,
    snapshot_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProductionIdentityResult:
    """Read-only verification of the exact final production graph."""

    plan = build_production_identity_plan(
        repository_root=repository_root,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        extractor_profile_path=extractor_profile_path,
        transaction_root=transaction_root,
        runtime_root=runtime_root,
        snapshot_root=snapshot_root,
        environ=environ,
    )
    if not plan.already_rebound:
        _fail(
            "production identity graph is not at the exact final state",
            "DINOV3_PRODUCTION_IDENTITY_FINAL_VERIFY_FAILED",
        )
    final = _final_hashes(plan)
    return ProductionIdentityResult(
        "DINOV3_PRODUCTION_IDENTITY_REBIND_VERIFY_OK", (), (), "EXACT_FINAL_STATE",
        False, None, not plan._transaction_root.exists(), True, final,
    )


def recover_production_identity_transaction(
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    extractor_profile_path: Path | str,
    transaction_root: Path | str | None = None,
    runtime_root: Path | str | None = None,
    snapshot_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    rollback_verify: Callable[[], Any] | None = None,
    failure_injector: Callable[[str, int], Any] | None = None,
) -> ProductionIdentityResult:
    """Recover only a valid incomplete journal by restoring exact backup bytes."""

    _, _, _, paths = _layout(
        repository_root=repository_root,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        extractor_profile_path=extractor_profile_path,
        transaction_root=transaction_root,
        runtime_root=runtime_root,
        snapshot_root=snapshot_root,
        environ=environ,
    )
    selected_root = default_transaction_root(environ) if transaction_root is None else Path(transaction_root)
    try:
        root = selected_root.resolve(strict=True)
    except OSError as exc:
        raise DINOv3ProductionIdentityError("recover requires an existing transaction root") from exc
    if _is_reparse(root):
        _fail("recover refuses an unexpected transaction root")
    journal_path = root / "journal.json"
    if not journal_path.is_file() or _is_reparse(journal_path):
        _fail("recover requires a regular transaction journal")
    journal = _read_journal(journal_path)
    old_hashes = tuple(row["old_sha256"] for row in journal["targets"])
    new_hashes = tuple(row["new_sha256"] for row in journal["targets"])
    backups: list[bytes] = []
    for index, row in enumerate(journal["targets"]):
        backup_path = _safe_file(root, row["backup"], f"backup {index}")
        candidate_path = _safe_file(root, row["candidate"], f"candidate {index}")
        backup = backup_path.read_bytes()
        candidate = candidate_path.read_bytes()
        if _sha256(backup) != old_hashes[index] or _sha256(candidate) != new_hashes[index]:
            _fail("recovery payload hash mismatch")
        backups.append(backup)
    transitions = ["rolling_back"]
    try:
        journal["state"] = "rolling_back"
        _write_journal(root, journal)
        for index in reversed(range(len(paths))):
            current = _sha256_file(paths[index])
            if current == old_hashes[index]:
                continue
            if current != new_hashes[index]:
                _fail("recovery found an unknown live identity")
            _inject(failure_injector, "before_rollback", index)
            _same_parent_replace(paths[index], backups[index], old_hashes[index], index)
            _inject(failure_injector, "after_rollback", index)
        if tuple(_sha256_file(path) for path in paths) != old_hashes:
            _fail("recovery did not restore exact starting bytes")
        if rollback_verify is not None and rollback_verify() is False:
            _fail("recovery rollback verifier returned false")
        _cleanup_transaction(root)
        transitions.append("rolled_back")
    except Exception as exc:
        try:
            journal["state"] = "rollback_failed"
            _write_journal(root, journal)
        except Exception:
            pass
        raise DINOv3ProductionIdentityError(
            "production identity recovery failed",
            "DINOV3_PRODUCTION_IDENTITY_ROLLBACK_FAILED",
        ) from exc
    return ProductionIdentityResult(
        "DINOV3_PRODUCTION_IDENTITY_RECOVERED",
        (),
        tuple(transitions),
        "RESTORED_STARTING_BYTES",
        True,
        True,
        not root.exists(),
        False,
        tuple((row[0], digest) for row, digest in zip(_TARGET_ROWS, old_hashes, strict=True)),
    )


__all__ = [
    "APPLE_EXTRACTOR_PROFILE_RELATIVE_PATH",
    "APPLE_GRAPH_ID",
    "APPLE_UNRESOLVED_GRAPH_ID",
    "APPLE_RUNTIME_PROFILE_RELATIVE_PATH",
    "DINOv3ProductionIdentityError",
    "EXTRACTOR_PROFILE_SCHEMA_ID",
    "LEGACY_GRAPH_ID",
    "PRODUCTION_SOURCE_PATHS",
    "ProductionIdentityPlan",
    "ProductionIdentityResult",
    "ResolvedGraphMigrationPlan",
    "ResolvedGraphMutation",
    "ProductionSourceIdentity",
    "RUNTIME_SOURCE_PATHS",
    "VersionedProductionGraph",
    "apply_production_identity_transaction",
    "apply_resolved_graph_migration",
    "build_production_identity_plan",
    "build_resolved_graph_migration_plan",
    "recover_resolved_graph_migration",
    "default_transaction_root",
    "load_production_source_identities",
    "recover_production_identity_transaction",
    "validate_production_sources",
    "validate_versioned_production_graph",
    "verify_production_identity",
]
