"""Atomically acquire or verify the exact external DINOv3 snapshot.

Normal imports and ``verify`` are standard-library-only and network-free.  The
Hugging Face client is imported only inside the fresh-root acquisition path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit


SNAPSHOT_ID = "dinov3-vitb16-r5931719-safetensors-v1"
TARGET_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
TARGET_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
WEIGHT_FORMAT = "single_file_safetensors"
WEIGHT_FILENAME = "model.safetensors"
SNAPSHOT_MANIFEST_SCHEMA_ID = "vea.external_model_snapshot.v0"
HUB_ENDPOINT = "https://huggingface.co"

EXPECTED_PROFILE_SHA256 = 'df90c58f2fd603b1e7979576270e67924bd84594eb08ad337edc3fdcdd2b80ac'
EXPECTED_RUNTIME_PROFILE_SHA256 = '52fc589cc47127c4c764f9631f17777f31f9784f7d4b92426f5b31ac4e3a2163'
EXPECTED_RUNTIME_ID = (
    "dinov3-vitb16-r5931719-py312-torch230-tf4560-manual-v1"
)
EXPECTED_PERSISTENT_FREEZE_SHA256 = (
    "995687ed934c0a94b52a040af048f87bc8de4e5089475f70565ad37f1f33ecd9"
)
EXPECTED_REGISTRY_SHA256 = '2c75cc26a27febc633b2bad479779327f35abbf36160df2805f43a4e33f48244'
EXPECTED_REPOSITORY_HEAD = "e518a51ba83f70b42611cc4386a9f55dc9e97de3"
EXPECTED_REPOSITORY_BRANCH = "main"

EXPECTED_FILES: dict[str, tuple[int, str]] = {
    "config.json": (
        744,
        "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec",
    ),
    "preprocessor_config.json": (
        585,
        "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
    ),
    WEIGHT_FILENAME: (
        342_662_192,
        "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
    ),
}
SELECTED_TOTAL_BYTES = 342_663_521
MAX_SELECTED_MODEL_BYTES = 350 * 1024 * 1024
MINIMUM_FREE_BYTES = 2 * 1024**3

EXPECTED_TRACKED_MODIFICATIONS = frozenset(
    {
        "configs/models.yaml",
        "docs/DECISION_LOG.md",
        "docs/METRIC_DEFINITIONS.md",
        "docs/REPRODUCIBILITY.md",
        "docs/VALIDATION_PLAN.md",
        "src/vision_encoder_alignment/extractors/__init__.py",
        "src/vision_encoder_alignment/extractors/base.py",
        "src/vision_encoder_alignment/extractors/clip.py",
        "src/vision_encoder_alignment/extractors/siglip2.py",
        "src/vision_encoder_alignment/metrics/__init__.py",
        "src/vision_encoder_alignment/metrics/cka.py",
        "src/vision_encoder_alignment/metrics/linear_alignment.py",
        "src/vision_encoder_alignment/metrics/procrustes.py",
    }
)


class SnapshotAcquisitionError(RuntimeError):
    """A fail-closed acquisition error carrying only a safe classification."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


class SnapshotCollisionError(SnapshotAcquisitionError):
    """Raised when a pre-existing final root is not exact-valid."""

    def __init__(self, message: str = "existing snapshot root is not exact-valid") -> None:
        super().__init__("DINOV3_SNAPSHOT_COLLISION", message)


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class RemoteWeightIdentity:
    commit_sha: str
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DownloadedWeight:
    path: Path
    final_host: str


@dataclass(frozen=True, slots=True)
class AcquisitionPlan:
    profile_path: Path
    runtime_profile_path: Path
    repository_root: Path
    snapshot_root: Path
    persistent_runtime_root: Path
    staging_parent: Path
    staging_prefix: str
    profile: Any = field(repr=False, compare=False)
    boundary: ModuleType = field(repr=False, compare=False)
    runtime_provisioner: ModuleType = field(repr=False, compare=False)


class _BoundedResponseWriter:
    """Delegate binary writes while enforcing the exact authorized body cap."""

    def __init__(self, stream: Any, maximum_bytes: int) -> None:
        self._stream = stream
        self._maximum_bytes = maximum_bytes

    def write(self, payload: bytes) -> int:
        if self._stream.tell() + len(payload) > self._maximum_bytes:
            raise SnapshotAcquisitionError(
                "DINOV3_SNAPSHOT_DOWNLOAD_FAILED",
                "downloaded response exceeded the authorized byte cap",
            )
        return int(self._stream.write(payload))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


Runner = Callable[..., CommandResult]
_HOSTNAME_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)(?:\."
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_point(path: Path) -> bool:
    try:
        result = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(result.st_mode):
        return True
    attributes = getattr(result, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_reparse_chain(path: Path) -> None:
    current = path
    while True:
        if current.exists() and _is_reparse_point(current):
            raise SnapshotAcquisitionError(
                "UNSAFE_SNAPSHOT_ROOT",
                "snapshot path contains a symbolic link or reparse point",
            )
        if current.parent == current:
            break
        current = current.parent


def _run_command(
    args: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    timeout: float = 30.0,
) -> CommandResult:
    command = tuple(os.fspath(item) for item in args)
    try:
        completed = subprocess.run(
            command,
            cwd=None if cwd is None else os.fspath(cwd),
            capture_output=True,
            check=False,
            shell=False,
            text=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SnapshotAcquisitionError(
            "REPOSITORY_IDENTITY_MISMATCH",
            "unable to inspect repository identity",
        ) from exc
    try:
        stdout = completed.stdout.decode("utf-8", errors="strict")
        stderr = completed.stderr.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise SnapshotAcquisitionError(
            "REPOSITORY_IDENTITY_MISMATCH",
            "repository identity output is not strict UTF-8",
        ) from exc
    result = CommandResult(command, completed.returncode, stdout, stderr)
    if result.returncode != 0:
        raise SnapshotAcquisitionError(
            "REPOSITORY_IDENTITY_MISMATCH",
            "repository identity command failed",
        )
    return result


def _load_module(path: Path, module_name: str) -> ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == path:
        return existing
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise SnapshotAcquisitionError(
            "CURRENT_RUNTIME_IDENTITY_CHANGED", "required boundary module is unavailable"
        )
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
    return module


def _load_snapshot_boundary(repository_root: Path) -> ModuleType:
    return _load_module(
        repository_root / "src" / "vision_encoder_alignment" / "snapshot_boundary.py",
        "_vea_t007a3a_snapshot_boundary",
    )


def _load_runtime_provisioner(repository_root: Path) -> ModuleType:
    return _load_module(
        repository_root / "scripts" / "provision_dinov3_runtime.py",
        "_vea_t007a3a_runtime_provisioner",
    )


def _assert_exact_profile(profile: Any) -> None:
    actual_files = {
        item.filename: (item.size_bytes, item.sha256) for item in profile.files
    }
    checks = (
        (profile.source_sha256, EXPECTED_PROFILE_SHA256, "profile hash"),
        (profile.snapshot_id, SNAPSHOT_ID, "snapshot ID"),
        (profile.model_id, TARGET_MODEL_ID, "model ID"),
        (profile.revision, TARGET_REVISION, "revision"),
        (profile.weight_format, WEIGHT_FORMAT, "weight format"),
        (profile.selected_total_bytes, SELECTED_TOTAL_BYTES, "selected bytes"),
        (profile.runtime_id, EXPECTED_RUNTIME_ID, "persistent runtime ID"),
        (
            profile.runtime_profile_sha256,
            EXPECTED_RUNTIME_PROFILE_SHA256,
            "persistent runtime profile hash",
        ),
        (
            profile.required_persistent_freeze_sha256,
            EXPECTED_PERSISTENT_FREEZE_SHA256,
            "persistent freeze hash",
        ),
        (actual_files, EXPECTED_FILES, "selected file allowlist"),
        (
            profile.staging_name_prefix,
            f".{SNAPSHOT_ID}.staging-",
            "staging prefix",
        ),
    )
    for actual, expected, description in checks:
        if actual != expected:
            raise SnapshotAcquisitionError(
                "DINOV3_SNAPSHOT_INTEGRITY_FAILED",
                f"repository snapshot {description} is not exact",
            )
    if profile.weight_file.size_bytes > MAX_SELECTED_MODEL_BYTES:
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_INTEGRITY_FAILED",
            "selected model payload exceeds the authorized cap",
        )


def _validate_requested_snapshot_root(
    requested: str | Path,
    *,
    profile: Any,
    repository_root: Path,
    persistent_runtime_root: Path,
) -> Path:
    candidate = Path(requested)
    if not candidate.is_absolute():
        raise SnapshotAcquisitionError(
            "UNSAFE_SNAPSHOT_ROOT", "snapshot root must be absolute"
        )
    lexical = Path(os.path.abspath(candidate))
    _reject_reparse_chain(lexical)
    resolved = lexical.resolve(strict=False)
    if resolved.name != profile.snapshot_id:
        raise SnapshotAcquisitionError(
            "UNSAFE_SNAPSHOT_ROOT", "snapshot root basename is not the snapshot ID"
        )
    roots = (resolved, repository_root.resolve(strict=True), persistent_runtime_root)
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if _contains(left, right) or _contains(right, left):
                raise SnapshotAcquisitionError(
                    "UNSAFE_SNAPSHOT_ROOT",
                    "snapshot, repository, and persistent runtime roots must not overlap",
                )
    return resolved


def _resolve_exact_repository_file(
    value: str | Path,
    repository_root: Path,
    expected_relative_path: str,
) -> Path:
    requested = Path(value)
    candidate = requested if requested.is_absolute() else repository_root / requested
    expected = repository_root / Path(expected_relative_path)
    if _is_reparse_point(candidate):
        raise SnapshotAcquisitionError(
            "REPOSITORY_IDENTITY_MISMATCH",
            "repository profile path is a symbolic link or reparse point",
        )
    try:
        resolved = candidate.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise SnapshotAcquisitionError(
            "REPOSITORY_IDENTITY_MISMATCH", "required repository profile is missing"
        ) from exc
    if resolved != expected_resolved or not resolved.is_file():
        raise SnapshotAcquisitionError(
            "REPOSITORY_IDENTITY_MISMATCH", "unexpected repository profile path"
        )
    return resolved


def plan_acquisition(
    profile_path: str | Path,
    runtime_profile_path: str | Path,
    snapshot_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    boundary_module: ModuleType | None = None,
    runtime_provisioner_module: ModuleType | None = None,
) -> AcquisitionPlan:
    """Return a side-effect-free exact acquisition plan."""

    repository = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    boundary = boundary_module or _load_snapshot_boundary(repository)
    provisioner = runtime_provisioner_module or _load_runtime_provisioner(repository)
    resolved_profile_path = _resolve_exact_repository_file(
        profile_path,
        repository,
        "configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json",
    )
    resolved_runtime_profile_path = _resolve_exact_repository_file(
        runtime_profile_path,
        repository,
        "configs/runtime_profiles/dinov3_isolated_manual_v0.json",
    )
    profile = boundary.load_snapshot_profile(resolved_profile_path)
    _assert_exact_profile(profile)
    runtime_profile = provisioner.load_profile(resolved_runtime_profile_path)
    if (
        runtime_profile.sha256 != profile.runtime_profile_sha256
        or runtime_profile.runtime_id != profile.runtime_id
    ):
        raise SnapshotAcquisitionError(
            "CURRENT_RUNTIME_IDENTITY_CHANGED",
            "persistent runtime profile does not match the snapshot profile",
        )
    runtime_plan = provisioner.plan_provision(
        runtime_profile.path,
        runtime_root,
        repository_root=repository,
        environment=environment,
    )
    requested = (
        boundary.resolve_default_snapshot_root(profile, environ=environment)
        if snapshot_root is None
        else Path(snapshot_root)
    )
    final = _validate_requested_snapshot_root(
        requested,
        profile=profile,
        repository_root=repository,
        persistent_runtime_root=runtime_plan.runtime_root,
    )
    return AcquisitionPlan(
        profile_path=resolved_profile_path,
        runtime_profile_path=runtime_profile.path,
        repository_root=repository,
        snapshot_root=final,
        persistent_runtime_root=runtime_plan.runtime_root,
        staging_parent=final.parent,
        staging_prefix=profile.staging_name_prefix,
        profile=profile,
        boundary=boundary,
        runtime_provisioner=provisioner,
    )


def _verify_repository_identity(plan: AcquisitionPlan, *, runner: Runner | None = None) -> None:
    execute = runner or _run_command
    repository = plan.repository_root
    top = Path(execute(["git", "rev-parse", "--show-toplevel"], cwd=repository).stdout.strip())
    branch = execute(["git", "branch", "--show-current"], cwd=repository).stdout.strip()
    head = execute(["git", "rev-parse", "HEAD"], cwd=repository).stdout.strip()
    modified = frozenset(
        line.strip().replace("\\", "/")
        for line in execute(["git", "diff", "--name-only"], cwd=repository).stdout.splitlines()
        if line.strip()
    )
    staged = tuple(
        line.strip()
        for line in execute(
            ["git", "diff", "--cached", "--name-only"], cwd=repository
        ).stdout.splitlines()
        if line.strip()
    )
    if (
        top.resolve(strict=True) != repository
        or branch != EXPECTED_REPOSITORY_BRANCH
        or head != EXPECTED_REPOSITORY_HEAD
        or modified != EXPECTED_TRACKED_MODIFICATIONS
        or staged
    ):
        raise SnapshotAcquisitionError(
            "REPOSITORY_IDENTITY_MISMATCH",
            "repository branch, commit, tracked modifications, or index changed",
        )
    registry = repository / "configs" / "models.yaml"
    if _sha256_file(registry) != EXPECTED_REGISTRY_SHA256:
        raise SnapshotAcquisitionError(
            "REPOSITORY_IDENTITY_MISMATCH", "model registry bytes changed"
        )


def _verify_persistent_runtime(plan: AcquisitionPlan) -> Any:
    result = plan.runtime_provisioner.verify_runtime(
        plan.runtime_profile_path,
        plan.persistent_runtime_root,
        repository_root=plan.repository_root,
    )
    if (
        result.runtime_id != plan.profile.runtime_id
        or result.repository_profile_sha256 != plan.profile.runtime_profile_sha256
        or result.candidate_freeze_sha256
        != plan.profile.required_persistent_freeze_sha256
    ):
        raise SnapshotAcquisitionError(
            "CURRENT_RUNTIME_IDENTITY_CHANGED",
            "persistent runtime verification result changed",
        )
    return result


def _verify_regular_file(path: Path, identity: Any, *, classification: str) -> None:
    if not path.is_file() or _is_reparse_point(path):
        raise SnapshotAcquisitionError(classification, "required file is missing or unsafe")
    try:
        size = path.stat().st_size
        digest = _sha256_file(path)
    except OSError as exc:
        raise SnapshotAcquisitionError(classification, "unable to inspect required file") from exc
    if size != identity.size_bytes or digest != identity.sha256:
        raise SnapshotAcquisitionError(classification, "required file identity mismatch")


def _persistent_metadata_directory(plan: AcquisitionPlan) -> Path:
    metadata = plan.persistent_runtime_root / "metadata"
    if not metadata.is_dir() or _is_reparse_point(metadata):
        raise SnapshotAcquisitionError(
            "CURRENT_RUNTIME_IDENTITY_CHANGED", "persistent metadata directory is unsafe"
        )
    entries = tuple(metadata.iterdir())
    expected = {plan.profile.config_file.filename, plan.profile.preprocessor_file.filename}
    if {item.name for item in entries} != expected or any(
        _is_reparse_point(item) for item in entries
    ):
        raise SnapshotAcquisitionError(
            "CURRENT_RUNTIME_IDENTITY_CHANGED", "persistent metadata layout changed"
        )
    for identity in (plan.profile.config_file, plan.profile.preprocessor_file):
        _verify_regular_file(
            metadata / identity.filename,
            identity,
            classification="CURRENT_RUNTIME_IDENTITY_CHANGED",
        )
    return metadata


def _copy_plain_file(source: Path, destination: Path) -> None:
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def _copy_persistent_metadata(plan: AcquisitionPlan, destination: Path) -> None:
    metadata = _persistent_metadata_directory(plan)
    for identity in (plan.profile.config_file, plan.profile.preprocessor_file):
        _copy_plain_file(metadata / identity.filename, destination / identity.filename)


def _hub_access_failure(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code in {401, 403} or type(error).__name__ in {
        "GatedRepoError",
        "RepositoryNotFoundError",
    }


def _validate_download_host(value: object) -> str:
    if not isinstance(value, str):
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_DOWNLOAD_FAILED", "download host is unavailable"
        )
    host = value.lower()
    if _HOSTNAME_PATTERN.fullmatch(host) is None:
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_DOWNLOAD_FAILED", "download host is not a safe hostname"
        )
    return host


def _query_remote_weight_identity(profile: Any) -> RemoteWeightIdentity:
    endpoint_override = os.environ.get("HF_ENDPOINT")
    if endpoint_override and endpoint_override.rstrip("/") != HUB_ENDPOINT:
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_DOWNLOAD_FAILED", "an unauthorized Hub endpoint is configured"
        )
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise SnapshotAcquisitionError(
                "DINOV3_SNAPSHOT_DOWNLOAD_FAILED", "Hugging Face Hub client is unavailable"
            ) from exc
        try:
            information = HfApi(endpoint=HUB_ENDPOINT).model_info(
                repo_id=profile.model_id,
                revision=profile.revision,
                timeout=30.0,
                files_metadata=True,
                token=True,
            )
        except Exception as exc:
            classification = (
                "DINOV3_ACCESS_NOT_ACTIVE"
                if _hub_access_failure(exc)
                else "DINOV3_SNAPSHOT_DOWNLOAD_FAILED"
            )
            raise SnapshotAcquisitionError(
                classification, "authenticated exact-revision model metadata request failed"
            ) from exc
    finally:
        sys.dont_write_bytecode = previous

    if information.id != profile.model_id or information.sha != profile.revision:
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_INTEGRITY_FAILED", "remote model identity is not exact"
        )
    siblings = information.siblings or []
    selected = [item for item in siblings if item.rfilename == profile.weight_file.filename]
    if len(selected) != 1:
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_INTEGRITY_FAILED", "exact selected model file is not unique"
        )
    sibling = selected[0]
    lfs = sibling.lfs
    if (
        type(sibling.size) is not int
        or lfs is None
        or type(lfs.size) is not int
        or sibling.size != profile.weight_file.size_bytes
        or lfs.size != profile.weight_file.size_bytes
        or lfs.sha256 != profile.weight_file.sha256
        or sibling.size > MAX_SELECTED_MODEL_BYTES
    ):
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_INTEGRITY_FAILED", "remote model file identity is not exact"
        )
    return RemoteWeightIdentity(
        commit_sha=information.sha,
        filename=sibling.rfilename,
        size_bytes=sibling.size,
        sha256=lfs.sha256,
    )


def _download_exact_weight(profile: Any, cache_directory: Path) -> DownloadedWeight:
    if (
        not cache_directory.is_dir()
        or _is_reparse_point(cache_directory)
        or any(cache_directory.iterdir())
    ):
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_DOWNLOAD_FAILED", "task-owned Hub cache must start empty"
        )
    previous = sys.dont_write_bytecode
    hub_file_download: Any | None = None
    original_http_get: Any | None = None
    original_hf_transfer: bool | None = None
    observed_hosts: list[str] = []
    sys.dont_write_bytecode = True
    try:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub import file_download as hub_file_download
        except Exception as exc:
            raise SnapshotAcquisitionError(
                "DINOV3_SNAPSHOT_DOWNLOAD_FAILED", "Hugging Face Hub client is unavailable"
            ) from exc
        original_http_get = hub_file_download.http_get
        original_hf_transfer = bool(hub_file_download.HF_HUB_ENABLE_HF_TRANSFER)

        def bounded_http_get(url: str, temp_file: Any, **kwargs: Any) -> None:
            expected_size = kwargs.get("expected_size")
            if expected_size != profile.weight_file.size_bytes:
                raise SnapshotAcquisitionError(
                    "DINOV3_SNAPSHOT_INTEGRITY_FAILED",
                    "download response size is not the exact authorized size",
                )
            parsed_url = urlsplit(url)
            host = (parsed_url.hostname or "").lower()
            if parsed_url.scheme.lower() != "https" or not host:
                raise SnapshotAcquisitionError(
                    "DINOV3_SNAPSHOT_DOWNLOAD_FAILED",
                    "download target is not an authorized HTTPS redirect",
                )
            observed_hosts.append(_validate_download_host(host))
            original_http_get(
                url,
                _BoundedResponseWriter(temp_file, profile.weight_file.size_bytes),
                **kwargs,
            )

        hub_file_download.HF_HUB_ENABLE_HF_TRANSFER = False
        hub_file_download.http_get = bounded_http_get
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=profile.model_id,
                    filename=profile.weight_file.filename,
                    revision=profile.revision,
                    token=True,
                    cache_dir=os.fspath(cache_directory),
                    local_files_only=False,
                    force_download=False,
                    resume_download=False,
                    endpoint=HUB_ENDPOINT,
                )
            )
        except SnapshotAcquisitionError:
            raise
        except Exception as exc:
            classification = (
                "DINOV3_ACCESS_NOT_ACTIVE"
                if _hub_access_failure(exc)
                else "DINOV3_SNAPSHOT_DOWNLOAD_FAILED"
            )
            raise SnapshotAcquisitionError(
                classification, "exact model weight download failed"
            ) from exc
    finally:
        if hub_file_download is not None and original_http_get is not None:
            hub_file_download.http_get = original_http_get
        if hub_file_download is not None and original_hf_transfer is not None:
            hub_file_download.HF_HUB_ENABLE_HF_TRANSFER = original_hf_transfer
        sys.dont_write_bytecode = previous

    try:
        resolved_cache = cache_directory.resolve(strict=True)
        resolved_download = downloaded.resolve(strict=True)
    except OSError as exc:
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_DOWNLOAD_FAILED", "downloaded cache file is unavailable"
        ) from exc
    if not _contains(resolved_cache, resolved_download):
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_DOWNLOAD_FAILED", "downloaded file escaped the task-owned cache"
        )
    if not observed_hosts:
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_DOWNLOAD_FAILED",
            "fresh task-owned cache did not observe a model payload transfer",
        )
    _verify_regular_file(
        resolved_download,
        profile.weight_file,
        classification="DINOV3_SNAPSHOT_INTEGRITY_FAILED",
    )
    return DownloadedWeight(
        path=resolved_download,
        final_host=observed_hosts[-1],
    )


def _require_free_space(path: Path) -> None:
    existing = path
    while not existing.exists():
        if existing.parent == existing:
            raise SnapshotAcquisitionError(
                "INSUFFICIENT_FREE_SPACE", "snapshot volume is unavailable"
            )
        existing = existing.parent
    if not existing.is_dir() or _is_reparse_point(existing):
        raise SnapshotAcquisitionError(
            "UNSAFE_SNAPSHOT_ROOT", "snapshot parent is not a safe directory"
        )
    try:
        free_bytes = shutil.disk_usage(existing).free
    except OSError as exc:
        raise SnapshotAcquisitionError(
            "INSUFFICIENT_FREE_SPACE", "unable to inspect snapshot-volume free space"
        ) from exc
    if free_bytes < MINIMUM_FREE_BYTES:
        raise SnapshotAcquisitionError(
            "INSUFFICIENT_FREE_SPACE", "snapshot volume has less than 2 GiB free"
        )


def _create_parent_directories(parent: Path) -> tuple[Path, ...]:
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    _reject_reparse_chain(current)
    created: list[Path] = []
    for directory in reversed(missing):
        directory.mkdir()
        created.append(directory)
    return tuple(created)


def _remove_created_empty_parents(created: Sequence[Path]) -> None:
    for directory in reversed(created):
        try:
            directory.rmdir()
        except OSError:
            break


def _cleanup_owned_staging(
    staging: Path,
    parent: Path,
    prefix: str,
    nonce: str,
) -> None:
    lexical = Path(os.path.abspath(staging))
    expected_parent = Path(os.path.abspath(parent))
    if lexical.parent != expected_parent or lexical.name != prefix + nonce:
        raise SnapshotAcquisitionError(
            "UNSAFE_CLEANUP_REFUSED", "staging cleanup target is not task-owned"
        )
    if not lexical.exists():
        return
    if _is_reparse_point(lexical):
        raise SnapshotAcquisitionError(
            "UNSAFE_CLEANUP_REFUSED", "staging cleanup target is a reparse point"
        )
    shutil.rmtree(lexical)


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _manifest_value(profile: Any, provisioned_at_utc: str) -> dict[str, Any]:
    return {
        "credential_backed_logical_request_count": 2,
        "files": [item.to_dict() for item in profile.files],
        "integrity_status": "VERIFIED",
        "model_id": profile.model_id,
        "model_weight_transferred_bytes": profile.weight_file.size_bytes,
        "network_independent_verification_status": "VERIFIED",
        "provisioned_at_utc": provisioned_at_utc,
        "repository_profile_sha256": profile.source_sha256,
        "revision": profile.revision,
        "schema_id": SNAPSHOT_MANIFEST_SCHEMA_ID,
        "snapshot_id": profile.snapshot_id,
        "total_bytes": profile.selected_total_bytes,
        "weight_format": profile.weight_format,
    }


def _write_manifest(plan: AcquisitionPlan, staging: Path, timestamp: str) -> None:
    value = _manifest_value(plan.profile, timestamp)
    manifest = plan.boundary.SnapshotManifest.from_dict(value)
    plan.boundary.validate_snapshot_manifest(manifest, plan.profile)
    payload = plan.boundary.canonical_json_bytes(value)
    path = staging / "snapshot-manifest.json"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def verify_snapshot(
    profile_path: str | Path,
    runtime_profile_path: str | Path,
    snapshot_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    boundary_module: ModuleType | None = None,
    runtime_provisioner_module: ModuleType | None = None,
) -> Any:
    """Read-only, network-free verification of one exact final snapshot."""

    plan = plan_acquisition(
        profile_path,
        runtime_profile_path,
        snapshot_root,
        runtime_root,
        repository_root=repository_root,
        environment=environment,
        boundary_module=boundary_module,
        runtime_provisioner_module=runtime_provisioner_module,
    )
    return plan.boundary.validate_snapshot_root(
        plan.snapshot_root,
        plan.profile,
        repository_root=plan.repository_root,
        persistent_runtime_root=plan.persistent_runtime_root,
    )


def acquire_snapshot(
    profile_path: str | Path,
    runtime_profile_path: str | Path,
    snapshot_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    boundary_module: ModuleType | None = None,
    runtime_provisioner_module: ModuleType | None = None,
    repository_verifier: Callable[[AcquisitionPlan], None] | None = None,
    persistent_runtime_verifier: Callable[[AcquisitionPlan], Any] | None = None,
    remote_identity_query: Callable[[Any], RemoteWeightIdentity] | None = None,
    weight_downloader: Callable[[Any, Path], DownloadedWeight | Path] | None = None,
    download_host_reporter: Callable[[str], None] | None = None,
    clock: Callable[[], str] | None = None,
) -> str:
    """Acquire a fresh exact root or reuse an exact-valid existing root."""

    plan = plan_acquisition(
        profile_path,
        runtime_profile_path,
        snapshot_root,
        runtime_root,
        repository_root=repository_root,
        environment=environment,
        boundary_module=boundary_module,
        runtime_provisioner_module=runtime_provisioner_module,
    )
    final = plan.snapshot_root
    if final.exists() or final.is_symlink():
        try:
            plan.boundary.validate_snapshot_root(
                final,
                plan.profile,
                repository_root=plan.repository_root,
                persistent_runtime_root=plan.persistent_runtime_root,
            )
        except Exception as exc:
            raise SnapshotCollisionError() from exc
        (repository_verifier or _verify_repository_identity)(plan)
        (persistent_runtime_verifier or _verify_persistent_runtime)(plan)
        return "DINOV3_EXACT_SNAPSHOT_ALREADY_VALID"

    (repository_verifier or _verify_repository_identity)(plan)
    (persistent_runtime_verifier or _verify_persistent_runtime)(plan)
    _persistent_metadata_directory(plan)
    remote = (remote_identity_query or _query_remote_weight_identity)(plan.profile)
    if (
        remote.commit_sha != plan.profile.revision
        or remote.filename != plan.profile.weight_file.filename
        or remote.size_bytes != plan.profile.weight_file.size_bytes
        or remote.sha256 != plan.profile.weight_file.sha256
    ):
        raise SnapshotAcquisitionError(
            "DINOV3_SNAPSHOT_INTEGRITY_FAILED", "remote model identity changed"
        )
    _require_free_space(plan.staging_parent)
    created_parents = _create_parent_directories(plan.staging_parent)
    nonce = uuid.uuid4().hex
    staging = plan.staging_parent / f"{plan.staging_prefix}{nonce}"
    try:
        staging.mkdir(exist_ok=False)
    except Exception:
        _remove_created_empty_parents(created_parents)
        raise

    try:
        files_directory = staging / "files"
        files_directory.mkdir()
        _copy_persistent_metadata(plan, files_directory)
        with tempfile.TemporaryDirectory(prefix="vea_t007a3a_hf_") as cache_text:
            cache_directory = Path(cache_text)
            downloaded = (weight_downloader or _download_exact_weight)(
                plan.profile, cache_directory
            )
            if isinstance(downloaded, DownloadedWeight):
                downloaded_path = downloaded.path
                final_host = _validate_download_host(downloaded.final_host)
                if download_host_reporter is not None:
                    download_host_reporter(final_host)
            else:
                downloaded_path = Path(downloaded)
            _verify_regular_file(
                downloaded_path,
                plan.profile.weight_file,
                classification="DINOV3_SNAPSHOT_INTEGRITY_FAILED",
            )
            _copy_plain_file(
                downloaded_path,
                files_directory / plan.profile.weight_file.filename,
            )
        for identity in plan.profile.files:
            _verify_regular_file(
                files_directory / identity.filename,
                identity,
                classification="DINOV3_SNAPSHOT_INTEGRITY_FAILED",
            )
        _write_manifest(plan, staging, (clock or _timestamp_now)())
        plan.boundary.validate_snapshot_root(
            staging,
            plan.profile,
            repository_root=plan.repository_root,
            persistent_runtime_root=plan.persistent_runtime_root,
            _allow_owned_staging=True,
        )
        if final.exists() or final.is_symlink():
            raise SnapshotCollisionError("snapshot root appeared before publication")
        try:
            staging.rename(final)
        except OSError as exc:
            if final.exists() or final.is_symlink():
                raise SnapshotCollisionError(
                    "snapshot root appeared during publication"
                ) from exc
            raise SnapshotAcquisitionError(
                "DINOV3_SNAPSHOT_INTEGRITY_FAILED",
                "same-parent atomic snapshot publication failed",
            ) from exc
    except Exception:
        _cleanup_owned_staging(staging, plan.staging_parent, plan.staging_prefix, nonce)
        _remove_created_empty_parents(created_parents)
        raise
    return "DINOV3_EXACT_SNAPSHOT_ACQUIRED"


def _default_profile_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "model_snapshots"
        / "dinov3_vit_b16_r5931719_v0.json"
    )


def _default_runtime_profile_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "runtime_profiles"
        / "dinov3_isolated_manual_v0.json"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("print-default-root", "acquire", "verify"))
    parser.add_argument("--profile", type=Path, default=_default_profile_path())
    parser.add_argument(
        "--runtime-profile", type=Path, default=_default_runtime_profile_path()
    )
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    return parser


def _safe_error_classification(error: BaseException) -> str:
    classification = getattr(error, "classification", None)
    if isinstance(classification, str) and classification:
        return classification
    return "DINOV3_SNAPSHOT_INTEGRITY_FAILED"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    try:
        if args.command == "print-default-root":
            plan = plan_acquisition(
                args.profile,
                args.runtime_profile,
                args.snapshot_root,
                args.runtime_root,
                repository_root=repository,
            )
            print(plan.snapshot_root)
            return 0
        if args.command == "verify":
            verify_snapshot(
                args.profile,
                args.runtime_profile,
                args.snapshot_root,
                args.runtime_root,
                repository_root=repository,
            )
            print("DINOV3_SNAPSHOT_MANIFEST_OK")
            print("DINOV3_EXACT_SNAPSHOT_VERIFY_OK")
            return 0
        download_hosts: list[str] = []
        marker = acquire_snapshot(
            args.profile,
            args.runtime_profile,
            args.snapshot_root,
            args.runtime_root,
            repository_root=repository,
            download_host_reporter=download_hosts.append,
        )
        print(marker)
        for host in download_hosts:
            print(f"DINOV3_DOWNLOAD_FINAL_HOST={host}")
        return 0
    except Exception as exc:
        classification = _safe_error_classification(exc)
        print(f"{classification}: operation failed safely", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
