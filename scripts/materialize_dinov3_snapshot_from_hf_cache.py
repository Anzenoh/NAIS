#!/usr/bin/env python3
"""Offline atomic DINOv3 contract-snapshot materializer.

Only the standard library is imported at module load time. The repository's
existing snapshot boundary is loaded directly from the supplied repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import stat
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


SNAPSHOT_ID = "dinov3-vitb16-r5931719-safetensors-v1"
SOURCE_SNAPSHOT_RELATIVE_PATH = Path(
    "models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/"
    "5931719e67bbdb9737e363e781fb0c67687896bc"
)
PROFILE_RELATIVE_PATH = Path("configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json")
PROFILE_SHA256 = "df90c58f2fd603b1e7979576270e67924bd84594eb08ad337edc3fdcdd2b80ac"
MANIFEST_BYTES = 1129
MANIFEST_SHA256 = "ec13f3f45b2b0976319122d123f3bacdd583ed13c68f29eec6a79462e4e7d6ec"
PROVISIONED_AT_UTC = "2026-07-20T17:53:45.969924Z"
MINIMUM_FREE_BYTES = 1024**3
MATERIALIZED_BYTES = 342_663_521


class MaterializationError(RuntimeError):
    classification = "DINOV3_CONTRACT_SNAPSHOT_FAILED"


class MaterializationCollisionError(MaterializationError):
    classification = "DINOV3_CONTRACT_SNAPSHOT_COLLISION"


@dataclass(frozen=True, slots=True)
class MaterializationPlan:
    repository_root: Path
    source_cache_root: Path
    source_snapshot_root: Path
    destination_root: Path
    runtime_root: Path
    profile_path: Path
    profile: Any = field(repr=False, compare=False)
    boundary: ModuleType = field(repr=False, compare=False)


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    return bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _absolute_directory(value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute() or _is_reparse(candidate):
        raise MaterializationError(f"{label} must be an absolute non-link directory")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MaterializationError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise MaterializationError(f"{label} must be a directory")
    return resolved


def _reject_existing_reparse_chain(path: Path) -> None:
    current = path
    while True:
        if (current.exists() or current.is_symlink()) and _is_reparse(current):
            raise MaterializationError("destination path contains a link or reparse point")
        if current.parent == current:
            return
        current = current.parent


def _load_boundary(repository_root: Path) -> ModuleType:
    path = repository_root / "src/vision_encoder_alignment/snapshot_boundary.py"
    specification = importlib.util.spec_from_file_location(
        "_vea_p2_t001b_r2_snapshot_boundary", path
    )
    if specification is None or specification.loader is None:
        raise MaterializationError("snapshot boundary is unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(specification.name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_materialization(
    *,
    repository_root: str | Path,
    source_cache_root: str | Path,
    source_snapshot_root: str | Path,
    destination_root: str | Path,
    runtime_root: str | Path,
    boundary_module: ModuleType | None = None,
) -> MaterializationPlan:
    repository = _absolute_directory(repository_root, label="repository root")
    cache = _absolute_directory(source_cache_root, label="source cache root")
    source = _absolute_directory(source_snapshot_root, label="source snapshot root")
    runtime = _absolute_directory(runtime_root, label="runtime root")
    if not _contains(cache, source):
        raise MaterializationError("source snapshot root must be inside source cache root")
    try:
        expected_source = (cache / SOURCE_SNAPSHOT_RELATIVE_PATH).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MaterializationError("exact Hugging Face source snapshot is unavailable") from exc
    if source != expected_source:
        raise MaterializationError("source snapshot is not the exact model revision")
    requested = Path(destination_root)
    if not requested.is_absolute():
        raise MaterializationError("destination root must be absolute")
    _reject_existing_reparse_chain(requested)
    destination = requested.resolve(strict=False)
    if destination.name != SNAPSHOT_ID:
        raise MaterializationError("destination basename must be the exact snapshot ID")
    for name, protected in (
        ("repository", repository),
        ("runtime", runtime),
        ("source cache", cache),
    ):
        if _contains(destination, protected) or _contains(protected, destination):
            raise MaterializationError(f"destination root must not overlap {name} root")
    boundary = boundary_module or _load_boundary(repository)
    profile_path = (repository / PROFILE_RELATIVE_PATH).resolve(strict=True)
    if not profile_path.is_file() or _is_reparse(profile_path):
        raise MaterializationError("snapshot profile is unavailable or unsafe")
    profile = boundary.load_snapshot_profile(profile_path)
    if profile.source_sha256 != PROFILE_SHA256 or profile.snapshot_id != SNAPSHOT_ID:
        raise MaterializationError("repository snapshot profile identity mismatch")
    if profile.selected_total_bytes != MATERIALIZED_BYTES:
        raise MaterializationError("repository snapshot byte contract mismatch")
    return MaterializationPlan(
        repository, cache, source, destination, runtime, profile_path, profile, boundary
    )


def validate_source_files(plan: MaterializationPlan) -> Mapping[str, Path]:
    """Validate only the three selected cache files and return resolved targets."""

    selected: dict[str, Path] = {}
    for identity in plan.profile.files:
        leaf = plan.source_snapshot_root / identity.filename
        if not leaf.exists() and not leaf.is_symlink():
            raise MaterializationError(f"source file is missing: {identity.filename}")
        try:
            resolved = leaf.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise MaterializationError(f"source file cannot be resolved: {identity.filename}") from exc
        if not _contains(plan.source_cache_root, resolved) or not resolved.is_file():
            raise MaterializationError(f"source file escapes cache: {identity.filename}")
        try:
            size = resolved.stat().st_size
            digest = _sha256_file(resolved)
        except OSError as exc:
            raise MaterializationError(f"source file cannot be read: {identity.filename}") from exc
        if size != identity.size_bytes or digest != identity.sha256:
            raise MaterializationError(f"source file identity mismatch: {identity.filename}")
        selected[identity.filename] = resolved
    if set(selected) != {item.filename for item in plan.profile.files} or len(selected) != 3:
        raise MaterializationError("source allowlist is not the exact three-file contract")
    return selected


def manifest_value(plan: MaterializationPlan) -> dict[str, Any]:
    return {
        "credential_backed_logical_request_count": 2,
        "files": [item.to_dict() for item in plan.profile.files],
        "integrity_status": "VERIFIED",
        "model_id": plan.profile.model_id,
        "model_weight_transferred_bytes": plan.profile.weight_file.size_bytes,
        "network_independent_verification_status": "VERIFIED",
        "provisioned_at_utc": PROVISIONED_AT_UTC,
        "repository_profile_sha256": plan.profile.source_sha256,
        "revision": plan.profile.revision,
        "schema_id": plan.boundary.SNAPSHOT_MANIFEST_SCHEMA_ID,
        "snapshot_id": plan.profile.snapshot_id,
        "total_bytes": plan.profile.selected_total_bytes,
        "weight_format": plan.profile.weight_format,
    }


def manifest_payload(plan: MaterializationPlan) -> bytes:
    value = manifest_value(plan)
    manifest = plan.boundary.SnapshotManifest.from_dict(value)
    plan.boundary.validate_snapshot_manifest(manifest, plan.profile)
    payload = plan.boundary.canonical_json_bytes(value)
    if len(payload) != MANIFEST_BYTES or hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise MaterializationError("canonical manifest identity mismatch")
    return payload


def _copy_plain_file(source: Path, destination: Path) -> None:
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def _write_manifest(plan: MaterializationPlan, staging: Path) -> None:
    with (staging / "snapshot-manifest.json").open("xb") as stream:
        stream.write(manifest_payload(plan))
        stream.flush()
        os.fsync(stream.fileno())


def _verify_plain_file(path: Path, source: Path, identity: Any) -> None:
    if _is_reparse(path) or not path.is_file():
        raise MaterializationError("destination file is not a plain regular file")
    info = path.stat()
    source_info = source.stat()
    if getattr(info, "st_nlink", 1) != 1:
        raise MaterializationError("destination file is hard-linked")
    if (info.st_dev, info.st_ino) == (source_info.st_dev, source_info.st_ino):
        raise MaterializationError("destination file aliases its cache source")
    if info.st_size != identity.size_bytes or _sha256_file(path) != identity.sha256:
        raise MaterializationError("destination file identity mismatch")


def _require_free_space(path: Path, *, disk_usage: Callable[[Path], Any] = shutil.disk_usage) -> None:
    existing = path
    while not existing.exists():
        if existing.parent == existing:
            raise MaterializationError("destination volume is unavailable")
        existing = existing.parent
    if not existing.is_dir() or _is_reparse(existing):
        raise MaterializationError("destination parent is unsafe")
    if disk_usage(existing).free < MINIMUM_FREE_BYTES:
        raise MaterializationError("destination volume has less than 1 GiB free")


def _cleanup_staging(staging: Path, parent: Path, prefix: str, nonce: str) -> None:
    lexical = Path(os.path.abspath(staging))
    if lexical.parent != Path(os.path.abspath(parent)) or lexical.name != prefix + nonce:
        raise MaterializationError("refusing unsafe staging cleanup")
    if not lexical.exists():
        return
    if _is_reparse(lexical):
        raise MaterializationError("refusing linked staging cleanup")
    shutil.rmtree(lexical)


def materialize_snapshot(
    plan: MaterializationPlan,
    *,
    copy_file: Callable[[Path, Path], None] = _copy_plain_file,
    write_manifest: Callable[[MaterializationPlan, Path], None] = _write_manifest,
    rename: Callable[[Path, Path], None] | None = None,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> str:
    sources = validate_source_files(plan)
    final = plan.destination_root
    if final.exists() or final.is_symlink():
        try:
            plan.boundary.validate_snapshot_root(
                final,
                plan.profile,
                repository_root=plan.repository_root,
                persistent_runtime_root=plan.runtime_root,
            )
        except Exception as exc:
            raise MaterializationCollisionError("existing destination is not exact-valid") from exc
        return "DINOV3_CONTRACT_SNAPSHOT_ALREADY_VALID"

    _require_free_space(final.parent, disk_usage=disk_usage)
    final.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    prefix = plan.profile.staging_name_prefix
    staging = final.parent / f"{prefix}{nonce}"
    staging.mkdir(exist_ok=False)
    try:
        files = staging / "files"
        files.mkdir()
        for identity in plan.profile.files:
            source = sources[identity.filename]
            destination = files / identity.filename
            copy_file(source, destination)
            _verify_plain_file(destination, source, identity)
        write_manifest(plan, staging)
        plan.boundary.validate_snapshot_root(
            staging,
            plan.profile,
            repository_root=plan.repository_root,
            persistent_runtime_root=plan.runtime_root,
            _allow_owned_staging=True,
        )
        if final.exists() or final.is_symlink():
            raise MaterializationCollisionError("destination appeared before publication")
        (rename or (lambda source, destination: source.rename(destination)))(staging, final)
    except Exception:
        _cleanup_staging(staging, final.parent, prefix, nonce)
        raise
    plan.boundary.validate_snapshot_root(
        final,
        plan.profile,
        repository_root=plan.repository_root,
        persistent_runtime_root=plan.runtime_root,
    )
    return "DINOV3_CONTRACT_SNAPSHOT_MATERIALIZED"


def verify_snapshot(plan: MaterializationPlan) -> str:
    validate_source_files(plan)
    plan.boundary.validate_snapshot_root(
        plan.destination_root,
        plan.profile,
        repository_root=plan.repository_root,
        persistent_runtime_root=plan.runtime_root,
    )
    return "DINOV3_CONTRACT_SNAPSHOT_VERIFY_OK"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline atomic DINOv3 snapshot materializer.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("materialize", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--repository-root", required=True)
        child.add_argument("--source-cache-root", required=True)
        child.add_argument("--source-snapshot-root", required=True)
        child.add_argument("--destination-root", required=True)
        child.add_argument("--runtime-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = plan_materialization(
            repository_root=args.repository_root,
            source_cache_root=args.source_cache_root,
            source_snapshot_root=args.source_snapshot_root,
            destination_root=args.destination_root,
            runtime_root=args.runtime_root,
        )
        status = materialize_snapshot(plan) if args.command == "materialize" else verify_snapshot(plan)
    except Exception as exc:
        print(f"STATUS={getattr(exc, 'classification', MaterializationError.classification)} ERROR={type(exc).__name__}")
        return 2
    print(status)
    if args.command == "materialize":
        print("source_kind=huggingface_cache_exact_bytes")
        print("network_attempt_count=0")
        print(f"materialized_bytes={MATERIALIZED_BYTES}")
        print("accepted_acquisition_provenance_preserved=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
