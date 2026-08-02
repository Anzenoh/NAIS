"""Verified compensating transaction for the accepted DINOv3 identity graph.

This module is deliberately import-light.  It prepares every candidate from
raw local bytes, validates the complete hash DAG, and only then performs
compare-and-swap replacements.  It never imports a model, a package manager,
Git, or a network client.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
import shutil
import stat
from typing import Any, Callable, Mapping, NoReturn


TRANSACTION_ID = "dinov3-r5931719-registry-rebind-v1"
TARGET_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
JOURNAL_SCHEMA_ID = "vea.dinov3_identity_rebind_journal.v0"

_LABELS = (
    "registry",
    "runtime_profile",
    "snapshot_profile",
    "provisioner",
    "snapshot_boundary",
    "acquire_script",
    "runtime_manifest",
    "snapshot_manifest",
)
_REPOSITORY_PATHS = {
    "registry": "configs/models.yaml",
    "runtime_profile": (
        "configs/runtime_profiles/dinov3_isolated_manual_v0.json"
    ),
    "snapshot_profile": (
        "configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json"
    ),
    "provisioner": "scripts/provision_dinov3_runtime.py",
    "snapshot_boundary": "src/vision_encoder_alignment/snapshot_boundary.py",
    "acquire_script": "scripts/acquire_dinov3_snapshot.py",
}
_CLASSIFICATIONS = {
    **{name: "repository" for name in _REPOSITORY_PATHS},
    "runtime_manifest": "persistent_runtime",
    "snapshot_manifest": "external_snapshot",
}
_RELATIVE_PATHS = {
    **_REPOSITORY_PATHS,
    "runtime_manifest": "runtime-manifest.json",
    "snapshot_manifest": "snapshot-manifest.json",
}

# Accepted T007A-2/T007A-3A byte identities.  Callers may supply an alternate
# complete mapping for isolated tests, but partial overrides are never allowed.
DEFAULT_OLD_HASHES: Mapping[str, str] = {
    "registry": "4a1b0b1367abc39845bcd577701c713017c9a7c817efd3ed30e5e762c248b2ea",
    "runtime_profile": "e9d45852406c5b51348682c6580d35379d76191b5434e4873349346177a0a7e9",
    "snapshot_profile": "df8396073bb60be6da138b3f8da43431b5aef75895fa424d70dd7697c785ad14",
    "provisioner": "4f32770d1d955abb6481e3024c5a7e568ec49bca261f506b714fbe3b3e551269",
    "snapshot_boundary": "01b62e87f959beff64ee11bcb13f5997b4ed394bbed9b04e85345a28eb3b287b",
    "acquire_script": "805a512df665ce200da9a995796813ed37c516739b4bc61a3560e2ffc03bdfc9",
    "runtime_manifest": "8775d840bf1e2de4a3b74e7baa79b66551eb0264d231f11a2649bfc4dc5bab29",
    "snapshot_manifest": "411f9ffe0d65abe727b27e20a4350cb9c1f989d8a323a41e07a852f7f5773c25",
}

_HASH_FIELDS = {
    "registry": "registry_sha256",
    "runtime_profile": "runtime_profile_sha256",
    "snapshot_profile": "snapshot_profile_sha256",
    "provisioner": "provisioner_sha256",
    "snapshot_boundary": "snapshot_boundary_sha256",
    "acquire_script": "acquire_script_sha256",
    "runtime_manifest": "runtime_manifest_sha256",
    "snapshot_manifest": "snapshot_manifest_sha256",
}
_SEMANTIC_DIFFS = {
    "registry": (
        "models.dinov3_vit_b16.revision",
        "models.dinov3_vit_b16.revision_status",
    ),
    "runtime_profile": (
        "current_base.source_files[path=configs/models.yaml].sha256",
    ),
    "snapshot_profile": ("persistent_runtime.profile_sha256",),
    "provisioner": (
        "PROFILE_SHA256",
        "EXPECTED_CURRENT_SOURCES[configs/models.yaml]",
    ),
    "snapshot_boundary": ("PERSISTENT_RUNTIME_PROFILE_SHA256",),
    "acquire_script": (
        "EXPECTED_PROFILE_SHA256",
        "EXPECTED_RUNTIME_PROFILE_SHA256",
        "EXPECTED_REGISTRY_SHA256",
    ),
    "runtime_manifest": ("repository_profile_sha256",),
    "snapshot_manifest": ("repository_profile_sha256",),
}
_JOURNAL_STATES = {
    "prepared",
    "applying",
    "applied",
    "verifying",
    "committed",
    "rolling_back",
    "rolled_back",
    "rollback_failed",
}
_JOURNAL_FIELDS = {
    "applied_index",
    "candidate_identity_hashes",
    "destinations",
    "last_transition_utc",
    "old_identity_hashes",
    "rollback_index",
    "schema_id",
    "started_at_utc",
    "state",
    "transaction_id",
}
_JOURNAL_DESTINATION_FIELDS = {
    "backup_relative_path",
    "candidate_relative_path",
    "new_sha256",
    "old_sha256",
    "path_classification",
    "relative_path",
}
_IDENTITY_HASH_KEYS = {"registry", "runtime_profile", "snapshot_profile"}
_SECRET_KEYS = {
    "access_key",
    "access_key_id",
    "access_token",
    "api_key",
    "api_keys",
    "auth_token",
    "authorization",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "password",
    "passwd",
    "private_key",
    "secret",
    "secrets",
    "token",
    "tokens",
}
_FINAL_VERIFICATION_COMMANDS = (
    "rebind_dinov3_identity.py verify",
    "provision_dinov3_runtime.py verify",
    "acquire_dinov3_snapshot.py verify",
    "run_dinov3_checkpoint_load_smoke.py",
    "run_dinov3_isolated_smoke.py",
)
_COMPATIBILITY_FALLBACK_CLASSIFICATIONS = frozenset(
    {
        "DINOV3_IDENTITY_BASELINE_MISMATCH",
        "DINOV3_IDENTITY_FINAL_VERIFY_FAILED",
    }
)
_COMPATIBILITY_FALLBACK_MESSAGES = frozenset(
    {
        "live identity is neither the accepted old graph nor the exact final graph",
        "provisioner PROFILE_SHA256 does not match the expected identity",
    }
)
_REGISTRY_PIN_FINAL_MARKER = "DINOV3_IDENTITY_STATE_REGISTRY_PIN_FINAL_OK"
_PRODUCTION_FINAL_MARKER = "DINOV3_IDENTITY_STATE_PRODUCTION_FINAL_OK"


class IdentityRebindError(RuntimeError):
    """Fail-closed identity validation or transaction error."""

    def __init__(
        self,
        message: str,
        *,
        classification: str = "DINOV3_IDENTITY_REBIND_ERROR",
    ) -> None:
        super().__init__(message)
        self.classification = classification


@dataclass(frozen=True, slots=True)
class IdentityFile:
    path_classification: str
    relative_path: str
    size_bytes: int
    sha256: str
    encoding: str
    line_ending: str


@dataclass(frozen=True, slots=True)
class IdentityGraph:
    registry_sha256: str
    runtime_profile_sha256: str
    snapshot_profile_sha256: str
    provisioner_sha256: str
    snapshot_boundary_sha256: str
    acquire_script_sha256: str
    runtime_manifest_sha256: str
    snapshot_manifest_sha256: str

    @property
    def hashes(self) -> Mapping[str, str]:
        return {
            label: getattr(self, field)
            for label, field in _HASH_FIELDS.items()
        }

    @property
    def identity_hashes(self) -> Mapping[str, str]:
        return self.hashes

    @property
    def status(self) -> str:
        return "DINOV3_IDENTITY_REBIND_VERIFY_OK"

    def to_dict(self) -> dict[str, str]:
        return dict(self.hashes)


@dataclass(frozen=True, slots=True)
class AcceptedIdentityVerification:
    """Path-free result for one of the two accepted terminal identity states."""

    state: str
    marker: str
    final_hashes: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_hashes": [
                {"label": label, "sha256": digest}
                for label, digest in self.final_hashes
            ],
            "marker": self.marker,
            "state": self.state,
        }


@dataclass(frozen=True, slots=True)
class IdentityMutation:
    label: str
    path_classification: str
    relative_path: str
    old: IdentityFile
    candidate: IdentityFile
    semantic_diff_paths: tuple[str, ...]

    @property
    def old_sha256(self) -> str:
        return self.old.sha256

    @property
    def new_sha256(self) -> str:
        return self.candidate.sha256

    @property
    def old_size(self) -> int:
        return self.old.size_bytes

    @property
    def new_size(self) -> int:
        return self.candidate.size_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "new_sha256": self.new_sha256,
            "new_size": self.new_size,
            "old_sha256": self.old_sha256,
            "old_size": self.old_size,
            "path_classification": self.path_classification,
            "relative_path": self.relative_path,
            "semantic_diff_paths": list(self.semantic_diff_paths),
        }


@dataclass(frozen=True, slots=True)
class IdentityTransactionPlan:
    transaction_id: str
    target_revision: str
    old_graph: IdentityGraph
    new_graph: IdentityGraph
    mutations: tuple[IdentityMutation, ...]
    rollback_order: tuple[str, ...]
    candidate_validated: bool
    already_rebound: bool
    final_verification_commands: tuple[str, ...]

    @property
    def candidate_graph(self) -> IdentityGraph:
        return self.new_graph

    @property
    def candidate_hashes(self) -> Mapping[str, str]:
        return self.new_graph.hashes

    def to_dict(self) -> dict[str, Any]:
        return {
            "already_rebound": self.already_rebound,
            "candidate_validated": self.candidate_validated,
            "final_verification_commands": list(self.final_verification_commands),
            "mutations": [mutation.to_dict() for mutation in self.mutations],
            "new_graph": self.new_graph.to_dict(),
            "old_graph": self.old_graph.to_dict(),
            "rollback_order": list(self.rollback_order),
            "target_revision": self.target_revision,
            "transaction_id": self.transaction_id,
        }


@dataclass(frozen=True, slots=True)
class IdentityTransactionResult:
    status: str
    old_graph: IdentityGraph
    final_graph: IdentityGraph
    applied: tuple[str, ...]
    journal_transitions: tuple[str, ...]
    compare_and_swap: str
    rollback_attempted: bool
    rollback_succeeded: bool | None
    cleanup_succeeded: bool
    already_rebound: bool

    @property
    def transaction_status(self) -> str:
        return self.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "already_rebound": self.already_rebound,
            "applied": list(self.applied),
            "cleanup_succeeded": self.cleanup_succeeded,
            "compare_and_swap": self.compare_and_swap,
            "final_graph": self.final_graph.to_dict(),
            "journal_transitions": list(self.journal_transitions),
            "old_graph": self.old_graph.to_dict(),
            "rollback_attempted": self.rollback_attempted,
            "rollback_succeeded": self.rollback_succeeded,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class _Layout:
    repository_root: Path
    runtime_root: Path
    snapshot_root: Path
    transaction_root: Path
    paths: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class _ExecutableIdentityFile:
    path_classification: str
    relative_path: str
    destination: Path
    raw_bytes: bytes
    size_bytes: int
    sha256: str
    encoding: str
    line_ending: str

    def public(self) -> IdentityFile:
        return IdentityFile(
            path_classification=self.path_classification,
            relative_path=self.relative_path,
            size_bytes=self.size_bytes,
            sha256=self.sha256,
            encoding=self.encoding,
            line_ending=self.line_ending,
        )


@dataclass(frozen=True, slots=True)
class _ExecutableIdentityMutation:
    public: IdentityMutation
    destination: Path
    old: _ExecutableIdentityFile
    candidate: _ExecutableIdentityFile

    @property
    def label(self) -> str:
        return self.public.label

    @property
    def path_classification(self) -> str:
        return self.public.path_classification

    @property
    def relative_path(self) -> str:
        return self.public.relative_path

    @property
    def semantic_diff_paths(self) -> tuple[str, ...]:
        return self.public.semantic_diff_paths

    @property
    def old_sha256(self) -> str:
        return self.public.old_sha256

    @property
    def new_sha256(self) -> str:
        return self.public.new_sha256


@dataclass(frozen=True, slots=True)
class _IdentityTransactionExecution:
    public: IdentityTransactionPlan
    transaction_root: Path
    mutations: tuple[_ExecutableIdentityMutation, ...]

    @property
    def transaction_id(self) -> str:
        return self.public.transaction_id

    @property
    def target_revision(self) -> str:
        return self.public.target_revision

    @property
    def old_graph(self) -> IdentityGraph:
        return self.public.old_graph

    @property
    def new_graph(self) -> IdentityGraph:
        return self.public.new_graph

    @property
    def rollback_order(self) -> tuple[str, ...]:
        return self.public.rollback_order

    @property
    def candidate_validated(self) -> bool:
        return self.public.candidate_validated

    @property
    def already_rebound(self) -> bool:
        return self.public.already_rebound


def _fail(
    message: str,
    classification: str = "DINOV3_IDENTITY_REBIND_ERROR",
) -> NoReturn:
    raise IdentityRebindError(message, classification=classification)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()
    except OSError as error:
        _fail(f"cannot hash identity destination {path.name!r}: {error}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _scan_sensitive_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.casefold()
            if lowered in _SECRET_KEYS:
                _fail(f"secret-like JSON field is forbidden at {path}.{key}")
            _scan_sensitive_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_sensitive_fields(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(f"non-finite JSON value is forbidden at {path}")


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"{label} is not strict UTF-8: {error}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except IdentityRebindError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        _fail(f"{label} is not strict JSON: {error}")
    if not isinstance(value, dict):
        _fail(f"{label} JSON root must be an object")
    _scan_sensitive_fields(value)
    return value


def _line_ending(payload: bytes) -> str:
    crlf = payload.count(b"\r\n")
    bare_lf = payload.replace(b"\r\n", b"").count(b"\n")
    if crlf and bare_lf:
        return "MIXED"
    if crlf:
        return "CRLF"
    if bare_lf:
        return "LF"
    return "NONE"


def _has_reparse_attribute(path: Path) -> bool:
    try:
        result = path.lstat()
    except OSError as error:
        _fail(f"cannot inspect path {path!s}: {error}")
    attributes = getattr(result, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse)


def _reject_link(path: Path, label: str) -> None:
    if path.is_symlink() or _has_reparse_attribute(path):
        _fail(f"{label} must not be a symbolic link, junction, or reparse point")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _roots_overlap(first: Path, second: Path) -> bool:
    return _is_relative_to(first, second) or _is_relative_to(second, first)


def _lexical_absolute(value: Path | str) -> Path:
    """Return an absolute normalized path without following a final link."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(os.path.normpath(str(first))) == os.path.normcase(
        os.path.normpath(str(second))
    )


def _existing_root(value: Path | str, label: str) -> Path:
    lexical = _lexical_absolute(value)
    if not lexical.is_dir():
        _fail(f"{label} is not an existing directory")
    _reject_link(lexical, label)
    try:
        canonical = lexical.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve {label}: {error}")
    if not _same_path(lexical, canonical):
        _fail(f"{label} traverses a symbolic link, junction, or reparse point")
    return lexical


def _safe_relative(value: str, label: str) -> None:
    if not value or "\x00" in value or "\\" in value:
        _fail(f"unsafe {label} relative path: {value!r}")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or value != posix.as_posix()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        _fail(f"unsafe {label} relative path: {value!r}")


def _default_local_appdata(environ: Mapping[str, str]) -> Path:
    value = environ.get("LOCALAPPDATA")
    if not value:
        _fail("LOCALAPPDATA is required to resolve default external roots")
    return _existing_root(value, "LOCALAPPDATA root")


def default_transaction_root(
    *, environ: Mapping[str, str] | None = None
) -> Path:
    environment = os.environ if environ is None else environ
    return _default_local_appdata(environment) / "VEA" / "transactions" / TRANSACTION_ID


def _resolve_layout(
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    transaction_root: Path | str | None,
    runtime_root: Path | str | None,
    snapshot_root: Path | str | None,
    environ: Mapping[str, str] | None,
) -> _Layout:
    environment = os.environ if environ is None else environ
    repository = _existing_root(repository_root, "repository root")
    runtime_profile = _lexical_absolute(runtime_profile_path)
    snapshot_profile = _lexical_absolute(snapshot_profile_path)
    expected_runtime_profile = repository / _REPOSITORY_PATHS["runtime_profile"]
    expected_snapshot_profile = repository / _REPOSITORY_PATHS["snapshot_profile"]
    if not _same_path(runtime_profile, expected_runtime_profile):
        _fail("runtime profile path is outside the exact repository allowlist")
    if not _same_path(snapshot_profile, expected_snapshot_profile):
        _fail("snapshot profile path is outside the exact repository allowlist")
    if not runtime_profile.is_file() or not snapshot_profile.is_file():
        _fail("runtime and snapshot profile paths must be existing files")
    _reject_link(runtime_profile, "runtime profile")
    _reject_link(snapshot_profile, "snapshot profile")
    if runtime_root is None or snapshot_root is None:
        runtime_value = _strict_json(runtime_profile.read_bytes(), "runtime profile")
        snapshot_value = _strict_json(snapshot_profile.read_bytes(), "snapshot profile")
        local_appdata = _default_local_appdata(environment)
        if runtime_root is None:
            runtime_id = runtime_value.get("runtime_id")
            if not isinstance(runtime_id, str) or not runtime_id:
                _fail("runtime profile does not contain a valid runtime_id")
            runtime_root = local_appdata / "VEA" / "runtimes" / runtime_id
        if snapshot_root is None:
            snapshot_id = snapshot_value.get("snapshot_id")
            if not isinstance(snapshot_id, str) or not snapshot_id:
                _fail("snapshot profile does not contain a valid snapshot_id")
            snapshot_root = local_appdata / "VEA" / "snapshots" / snapshot_id
    runtime = _existing_root(runtime_root, "persistent runtime root")
    snapshot = _existing_root(snapshot_root, "external snapshot root")
    transaction = (
        default_transaction_root(environ=environment)
        if transaction_root is None
        else _lexical_absolute(transaction_root)
    )
    roots = (repository, runtime, snapshot, transaction)
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if _roots_overlap(first, second):
                _fail(f"identity roots overlap: {first.name!r} and {second.name!r}")
    # Keep the lexical destination path.  ``Path.resolve()`` would follow a
    # leaf symlink before the mandatory lstat/reparse check below.
    paths = {
        name: repository / Path(relative)
        for name, relative in _REPOSITORY_PATHS.items()
    }
    paths["runtime_manifest"] = runtime / "runtime-manifest.json"
    paths["snapshot_manifest"] = snapshot / "snapshot-manifest.json"
    for name, path in paths.items():
        root = {
            "repository": repository,
            "persistent_runtime": runtime,
            "external_snapshot": snapshot,
        }[_CLASSIFICATIONS[name]]
        if not _is_relative_to(path, root):
            _fail(f"{name} destination escapes its classified root")
        if not path.is_file():
            _fail(f"required identity destination is absent: {_RELATIVE_PATHS[name]}")
        _reject_link(path, name)
        cursor = path.parent
        while cursor != root:
            _reject_link(cursor, f"{name} ancestor")
            cursor = cursor.parent
    if transaction.exists():
        _reject_link(transaction, "transaction root")
    return _Layout(repository, runtime, snapshot, transaction, paths)


def _identity_file(
    label: str,
    destination: Path,
    payload: bytes | None = None,
) -> _ExecutableIdentityFile:
    raw = destination.read_bytes() if payload is None else payload
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"{label} is not strict UTF-8: {error}")
    return _ExecutableIdentityFile(
        path_classification=_CLASSIFICATIONS[label],
        relative_path=_RELATIVE_PATHS[label],
        destination=destination,
        raw_bytes=raw,
        size_bytes=len(raw),
        sha256=_sha256(raw),
        encoding="utf-8",
        line_ending=_line_ending(raw),
    )


def _read_files(layout: _Layout) -> dict[str, _ExecutableIdentityFile]:
    return {
        label: _identity_file(label, layout.paths[label]) for label in _LABELS
    }


def _graph(files: Mapping[str, _ExecutableIdentityFile]) -> IdentityGraph:
    return IdentityGraph(
        **{
            field: files[label].sha256
            for label, field in _HASH_FIELDS.items()
        }
    )


def _validate_expected_hashes(
    expected_old_hashes: Mapping[str, str] | None,
) -> dict[str, str]:
    values = dict(DEFAULT_OLD_HASHES if expected_old_hashes is None else expected_old_hashes)
    if set(values) != set(_LABELS):
        _fail("expected old hash mapping must contain the exact identity allowlist")
    for label, value in values.items():
        if not _is_sha256(value):
            _fail(f"expected old hash for {label} is not lowercase SHA-256")
    return values


def _registry_pairs(payload: bytes) -> tuple[bytes, bytes]:
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"registry is not strict UTF-8: {error}")
    line_ending = _line_ending(payload)
    if line_ending not in {"LF", "CRLF"}:
        _fail("registry must use one consistent LF or CRLF line ending")
    newline = b"\r\n" if line_ending == "CRLF" else b"\n"
    old = b"    revision: null" + newline + b"    revision_status: unresolved" + newline
    new = (
        b"    revision: "
        + TARGET_REVISION.encode("ascii")
        + newline
        + b"    revision_status: pinned"
        + newline
    )
    marker = b"  dinov3_vit_b16:" + newline
    if payload.count(marker) != 1:
        _fail("registry must contain exactly one DINOv3 block anchor")
    return old, new


def _dinov3_block_bounds(payload: bytes) -> tuple[int, int]:
    """Return the exact byte interval occupied by the DINOv3 model block."""

    line_ending = _line_ending(payload)
    newline = b"\r\n" if line_ending == "CRLF" else b"\n"
    marker = b"  dinov3_vit_b16:" + newline
    if payload.count(marker) != 1:
        _fail("registry must contain exactly one DINOv3 block anchor")
    start = payload.index(marker)
    cursor = start + len(marker)
    end = len(payload)
    while cursor < len(payload):
        next_newline = payload.find(newline, cursor)
        line_end = len(payload) if next_newline < 0 else next_newline
        line = payload[cursor:line_end]
        if (
            line.startswith(b"  ")
            and not line.startswith(b"   ")
            and line.rstrip().endswith(b":")
        ):
            end = cursor
            break
        if next_newline < 0:
            break
        cursor = next_newline + len(newline)
    return start, end


def _patch_registry(payload: bytes, *, reverse: bool = False) -> bytes:
    old, new = _registry_pairs(payload)
    source, target = (new, old) if reverse else (old, new)
    if payload.count(source) != 1:
        _fail("registry revision pair is missing or duplicated")
    block_start, block_end = _dinov3_block_bounds(payload)
    pair = payload.index(source)
    if pair < block_start or pair + len(source) > block_end:
        _fail("registry revision pair is outside the DINOv3 block")
    block = payload[block_start:block_end]
    if block.count(b"    access: user_must_confirm_license_and_auth") != 1:
        _fail("DINOv3 gated-access classification is missing or duplicated")
    result = payload[:pair] + target + payload[pair + len(source) :]
    if result.count(target) != 1:
        _fail("registry candidate does not contain the exact target pair")
    return result


def _json_leaf_diffs(
    before: Any,
    after: Any,
    prefix: tuple[str | int, ...] = (),
) -> set[tuple[str | int, ...]]:
    if type(before) is not type(after):
        return {prefix}
    if isinstance(before, dict):
        if set(before) != set(after):
            return {prefix}
        result: set[tuple[str | int, ...]] = set()
        for key in before:
            result.update(_json_leaf_diffs(before[key], after[key], (*prefix, key)))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return {prefix}
        result = set()
        for index, (old_item, new_item) in enumerate(zip(before, after, strict=True)):
            result.update(_json_leaf_diffs(old_item, new_item, (*prefix, index)))
        return result
    return set() if before == after else {prefix}


def _runtime_profile_candidate(payload: bytes, registry_hash: str) -> bytes:
    value = _strict_json(payload, "runtime profile")
    try:
        rows = value["current_base"]["source_files"]
    except (KeyError, TypeError):
        _fail("runtime profile source_files path is missing")
    if not isinstance(rows, list):
        _fail("runtime profile source_files must be a list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("path") == "configs/models.yaml"
    ]
    if len(matches) != 1:
        _fail("runtime profile must contain exactly one configs/models.yaml row")
    before = json.loads(json.dumps(value, ensure_ascii=False))
    matches[0]["sha256"] = registry_hash
    if _json_leaf_diffs(before, value) != {
        ("current_base", "source_files", rows.index(matches[0]), "sha256")
    }:
        _fail("runtime profile candidate changed an unexpected semantic leaf")
    return _canonical_json(value)


def _snapshot_profile_candidate(payload: bytes, runtime_hash: str) -> bytes:
    value = _strict_json(payload, "snapshot profile")
    try:
        old_value = value["persistent_runtime"]["profile_sha256"]
    except (KeyError, TypeError):
        _fail("snapshot profile persistent runtime hash path is missing")
    if not isinstance(old_value, str):
        _fail("snapshot profile persistent runtime hash must be a string")
    before = json.loads(json.dumps(value, ensure_ascii=False))
    value["persistent_runtime"]["profile_sha256"] = runtime_hash
    if _json_leaf_diffs(before, value) != {
        ("persistent_runtime", "profile_sha256")
    }:
        _fail("snapshot profile candidate changed an unexpected semantic leaf")
    return _canonical_json(value)


def _manifest_candidate(payload: bytes, profile_hash: str, label: str) -> bytes:
    value = _strict_json(payload, label)
    old_value = value.get("repository_profile_sha256")
    if not isinstance(old_value, str):
        _fail(f"{label} repository_profile_sha256 must be a string")
    before = json.loads(json.dumps(value, ensure_ascii=False))
    value["repository_profile_sha256"] = profile_hash
    if _json_leaf_diffs(before, value) != {("repository_profile_sha256",)}:
        _fail(f"{label} candidate changed an unexpected semantic leaf")
    return _canonical_json(value)


def _source_tree(payload: bytes, label: str) -> tuple[str, ast.Module]:
    try:
        text = payload.decode("utf-8")
        tree = ast.parse(text, filename=label)
    except (UnicodeDecodeError, SyntaxError) as error:
        _fail(f"{label} is not valid UTF-8 Python source: {error}")
    return text, tree


def _top_level_assignment(tree: ast.Module, name: str, label: str) -> ast.AST:
    matches: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                matches.append(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                matches.append(node.value)
    if len(matches) != 1:
        _fail(f"{label} must assign {name} exactly once at module scope")
    return matches[0]


def _string_node(
    tree: ast.Module,
    assignment: str,
    label: str,
    *,
    mapping_key: str | None = None,
) -> ast.Constant:
    value = _top_level_assignment(tree, assignment, label)
    if mapping_key is not None:
        if not isinstance(value, ast.Dict):
            _fail(f"{label} {assignment} must be a dictionary literal")
        matches: list[ast.AST] = []
        for key, child in zip(value.keys, value.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == mapping_key:
                matches.append(child)
        if len(matches) != 1:
            _fail(f"{label} must contain {assignment}[{mapping_key!r}] exactly once")
        value = matches[0]
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        _fail(f"{label} {assignment} target value must be a string literal")
    if value.end_lineno is None or value.end_col_offset is None:
        _fail(f"{label} AST does not expose a complete string location")
    return value


def _node_offsets(text: str, node: ast.Constant) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    # AST column offsets are UTF-8 byte offsets.  Identity source anchors are
    # ASCII, so encoding each prefix gives the exact byte position.
    start = sum(len(line.encode("utf-8")) for line in lines[: node.lineno - 1])
    start += node.col_offset
    end = sum(len(line.encode("utf-8")) for line in lines[: node.end_lineno - 1])
    end += int(node.end_col_offset)
    return start, end


def _replace_source_values(
    payload: bytes,
    label: str,
    replacements: tuple[tuple[str, str | None, str, str], ...],
) -> bytes:
    text, tree = _source_tree(payload, label)
    edits: list[tuple[int, int, bytes]] = []
    for assignment, mapping_key, expected, candidate in replacements:
        node = _string_node(
            tree,
            assignment,
            label,
            mapping_key=mapping_key,
        )
        if node.value != expected:
            _fail(f"{label} {assignment} does not match the expected identity")
        start, end = _node_offsets(text, node)
        token = payload[start:end]
        try:
            quote = token.decode("utf-8")
            parsed = ast.literal_eval(quote)
        except (UnicodeDecodeError, SyntaxError, ValueError) as error:
            _fail(f"{label} string anchor cannot be decoded: {error}")
        if parsed != expected or not quote:
            _fail(f"{label} source anchor is not the expected literal")
        prefix_length = 0
        while prefix_length < len(quote) and quote[prefix_length] in "rRuUbBfF":
            prefix_length += 1
        quote_character = quote[prefix_length : prefix_length + 1]
        if quote_character not in {"'", '"'} or quote.endswith(quote_character * 3):
            _fail(f"{label} source anchor must use a simple quoted literal")
        replacement = (
            quote[:prefix_length]
            + quote_character
            + candidate
            + quote_character
        ).encode("utf-8")
        edits.append((start, end, replacement))
    if len({(start, end) for start, end, _ in edits}) != len(edits):
        _fail(f"{label} source replacement anchors overlap")
    result = payload
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start] + replacement + result[end:]
    _source_tree(result, f"candidate {label}")
    return result


def _source_candidates(
    files: Mapping[str, _ExecutableIdentityFile],
    old_graph: IdentityGraph,
    new_registry_hash: str,
    new_runtime_hash: str,
    new_snapshot_hash: str,
    *,
    reverse: bool = False,
) -> dict[str, bytes]:
    old_registry = old_graph.registry_sha256
    old_runtime = old_graph.runtime_profile_sha256
    old_snapshot = old_graph.snapshot_profile_sha256
    if reverse:
        registry_pair = (new_registry_hash, old_registry)
        runtime_pair = (new_runtime_hash, old_runtime)
        snapshot_pair = (new_snapshot_hash, old_snapshot)
    else:
        registry_pair = (old_registry, new_registry_hash)
        runtime_pair = (old_runtime, new_runtime_hash)
        snapshot_pair = (old_snapshot, new_snapshot_hash)
    return {
        "provisioner": _replace_source_values(
            files["provisioner"].raw_bytes,
            "provisioner",
            (
                ("PROFILE_SHA256", None, *runtime_pair),
                (
                    "EXPECTED_CURRENT_SOURCES",
                    "configs/models.yaml",
                    *registry_pair,
                ),
            ),
        ),
        "snapshot_boundary": _replace_source_values(
            files["snapshot_boundary"].raw_bytes,
            "snapshot boundary",
            (("PERSISTENT_RUNTIME_PROFILE_SHA256", None, *runtime_pair),),
        ),
        "acquire_script": _replace_source_values(
            files["acquire_script"].raw_bytes,
            "snapshot acquisition script",
            (
                ("EXPECTED_PROFILE_SHA256", None, *snapshot_pair),
                ("EXPECTED_RUNTIME_PROFILE_SHA256", None, *runtime_pair),
                ("EXPECTED_REGISTRY_SHA256", None, *registry_pair),
            ),
        ),
    }


def _derive_candidates(
    old_files: Mapping[str, _ExecutableIdentityFile],
    old_graph: IdentityGraph,
) -> dict[str, _ExecutableIdentityFile]:
    registry = _patch_registry(old_files["registry"].raw_bytes)
    registry_hash = _sha256(registry)
    runtime = _runtime_profile_candidate(
        old_files["runtime_profile"].raw_bytes, registry_hash
    )
    runtime_hash = _sha256(runtime)
    snapshot = _snapshot_profile_candidate(
        old_files["snapshot_profile"].raw_bytes, runtime_hash
    )
    snapshot_hash = _sha256(snapshot)
    source_payloads = _source_candidates(
        old_files,
        old_graph,
        registry_hash,
        runtime_hash,
        snapshot_hash,
    )
    payloads = {
        "registry": registry,
        "runtime_profile": runtime,
        "snapshot_profile": snapshot,
        **source_payloads,
        "runtime_manifest": _manifest_candidate(
            old_files["runtime_manifest"].raw_bytes,
            runtime_hash,
            "runtime manifest",
        ),
        "snapshot_manifest": _manifest_candidate(
            old_files["snapshot_manifest"].raw_bytes,
            snapshot_hash,
            "snapshot manifest",
        ),
    }
    return {
        label: _identity_file(label, old_files[label].destination, payloads[label])
        for label in _LABELS
    }


def _reverse_final_files(
    final_files: Mapping[str, _ExecutableIdentityFile],
    expected: Mapping[str, str],
) -> dict[str, _ExecutableIdentityFile]:
    final_graph = _graph(final_files)
    synthetic_graph = IdentityGraph(
        **{_HASH_FIELDS[label]: expected[label] for label in _LABELS}
    )
    registry = _patch_registry(final_files["registry"].raw_bytes, reverse=True)
    runtime = _runtime_profile_candidate(
        final_files["runtime_profile"].raw_bytes,
        expected["registry"],
    )
    snapshot = _snapshot_profile_candidate(
        final_files["snapshot_profile"].raw_bytes,
        expected["runtime_profile"],
    )
    source_payloads = _source_candidates(
        final_files,
        synthetic_graph,
        final_graph.registry_sha256,
        final_graph.runtime_profile_sha256,
        final_graph.snapshot_profile_sha256,
        reverse=True,
    )
    payloads = {
        "registry": registry,
        "runtime_profile": runtime,
        "snapshot_profile": snapshot,
        **source_payloads,
        "runtime_manifest": _manifest_candidate(
            final_files["runtime_manifest"].raw_bytes,
            expected["runtime_profile"],
            "runtime manifest",
        ),
        "snapshot_manifest": _manifest_candidate(
            final_files["snapshot_manifest"].raw_bytes,
            expected["snapshot_profile"],
            "snapshot manifest",
        ),
    }
    result = {
        label: _identity_file(label, final_files[label].destination, payloads[label])
        for label in _LABELS
    }
    if {label: item.sha256 for label, item in result.items()} != dict(expected):
        _fail("live identity is neither the accepted old graph nor the exact final graph")
    return result


def _mapping_source_hash(value: Mapping[str, Any]) -> str:
    try:
        rows = value["current_base"]["source_files"]
    except (KeyError, TypeError):
        _fail("runtime profile source file list is missing")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("path") == "configs/models.yaml"
    ] if isinstance(rows, list) else []
    if len(matches) != 1 or not _is_sha256(matches[0].get("sha256")):
        _fail("runtime profile registry source identity is invalid")
    return str(matches[0]["sha256"])


def _source_values(payload: bytes, label: str) -> dict[str, Any]:
    _, tree = _source_tree(payload, label)
    result: dict[str, Any] = {}
    names = {
        "provisioner": ("PROFILE_SHA256", "EXPECTED_CURRENT_SOURCES"),
        "snapshot boundary": ("PERSISTENT_RUNTIME_PROFILE_SHA256",),
        "snapshot acquisition script": (
            "EXPECTED_PROFILE_SHA256",
            "EXPECTED_RUNTIME_PROFILE_SHA256",
            "EXPECTED_REGISTRY_SHA256",
        ),
    }[label]
    for name in names:
        node = _top_level_assignment(tree, name, label)
        try:
            result[name] = ast.literal_eval(node)
        except (ValueError, SyntaxError) as error:
            _fail(f"{label} {name} is not a literal value: {error}")
    return result


def _validate_graph(
    files: Mapping[str, _ExecutableIdentityFile],
    graph: IdentityGraph,
    *,
    final: bool,
) -> None:
    for label in _LABELS:
        if files[label].sha256 != graph.hashes[label]:
            _fail(f"{label} bytes do not match their identity graph hash")
    old_pair, new_pair = _registry_pairs(files["registry"].raw_bytes)
    required_pair = new_pair if final else old_pair
    forbidden_pair = old_pair if final else new_pair
    registry_payload = files["registry"].raw_bytes
    block_start, block_end = _dinov3_block_bounds(registry_payload)
    block = registry_payload[block_start:block_end]
    if block.count(required_pair) != 1:
        _fail("registry does not contain its required exact revision pair")
    if forbidden_pair in block:
        _fail("registry contains a contradictory revision pair")
    if block.count(b"    access: user_must_confirm_license_and_auth") != 1:
        _fail("registry lost the DINOv3 access classification")
    runtime = _strict_json(files["runtime_profile"].raw_bytes, "runtime profile")
    snapshot = _strict_json(files["snapshot_profile"].raw_bytes, "snapshot profile")
    runtime_manifest = _strict_json(
        files["runtime_manifest"].raw_bytes, "runtime manifest"
    )
    snapshot_manifest = _strict_json(
        files["snapshot_manifest"].raw_bytes, "snapshot manifest"
    )
    if _mapping_source_hash(runtime) != graph.registry_sha256:
        _fail("runtime profile is not bound to the registry graph hash")
    try:
        snapshot_runtime_hash = snapshot["persistent_runtime"]["profile_sha256"]
    except (KeyError, TypeError):
        _fail("snapshot profile persistent runtime identity is missing")
    if snapshot_runtime_hash != graph.runtime_profile_sha256:
        _fail("snapshot profile is not bound to the runtime profile graph hash")
    if runtime_manifest.get("repository_profile_sha256") != graph.runtime_profile_sha256:
        _fail("runtime manifest is not bound to the runtime profile graph hash")
    if snapshot_manifest.get("repository_profile_sha256") != graph.snapshot_profile_sha256:
        _fail("snapshot manifest is not bound to the snapshot profile graph hash")
    provisioner = _source_values(files["provisioner"].raw_bytes, "provisioner")
    boundary = _source_values(
        files["snapshot_boundary"].raw_bytes, "snapshot boundary"
    )
    acquire = _source_values(
        files["acquire_script"].raw_bytes, "snapshot acquisition script"
    )
    expected_sources = provisioner.get("EXPECTED_CURRENT_SOURCES")
    if not isinstance(expected_sources, dict):
        _fail("provisioner EXPECTED_CURRENT_SOURCES must be a literal dictionary")
    source_rows = [key for key in expected_sources if key == "configs/models.yaml"]
    if len(source_rows) != 1:
        _fail("provisioner registry source constant is missing or duplicated")
    checks = (
        (provisioner.get("PROFILE_SHA256"), graph.runtime_profile_sha256),
        (expected_sources.get("configs/models.yaml"), graph.registry_sha256),
        (
            boundary.get("PERSISTENT_RUNTIME_PROFILE_SHA256"),
            graph.runtime_profile_sha256,
        ),
        (acquire.get("EXPECTED_PROFILE_SHA256"), graph.snapshot_profile_sha256),
        (
            acquire.get("EXPECTED_RUNTIME_PROFILE_SHA256"),
            graph.runtime_profile_sha256,
        ),
        (acquire.get("EXPECTED_REGISTRY_SHA256"), graph.registry_sha256),
    )
    if any(actual != expected for actual, expected in checks):
        _fail("active source constants are inconsistent with the identity graph")
    for label in ("provisioner", "snapshot_boundary", "acquire_script"):
        try:
            compile(
                files[label].raw_bytes,
                files[label].relative_path,
                "exec",
                dont_inherit=True,
            )
        except (SyntaxError, ValueError) as error:
            _fail(f"candidate source does not compile: {label}: {error}")


def _validate_candidate_diffs(
    old_files: Mapping[str, _ExecutableIdentityFile],
    new_files: Mapping[str, _ExecutableIdentityFile],
) -> None:
    old_registry, _ = _registry_pairs(old_files["registry"].raw_bytes)
    expected_registry = old_files["registry"].raw_bytes.replace(
        old_registry,
        _registry_pairs(old_files["registry"].raw_bytes)[1],
        1,
    )
    if new_files["registry"].raw_bytes != expected_registry:
        _fail("registry candidate contains bytes beyond the exact two-field patch")
    expected_json_diffs = {
        "runtime_profile": {
            ("current_base", "source_files", _runtime_row_index(old_files), "sha256")
        },
        "snapshot_profile": {("persistent_runtime", "profile_sha256")},
        "runtime_manifest": {("repository_profile_sha256",)},
        "snapshot_manifest": {("repository_profile_sha256",)},
    }
    for label, expected in expected_json_diffs.items():
        before = _strict_json(old_files[label].raw_bytes, label)
        after = _strict_json(new_files[label].raw_bytes, label)
        if _json_leaf_diffs(before, after) != expected:
            _fail(f"{label} candidate contains an unexpected semantic diff")
    old_hashes = (
        old_files["registry"].sha256,
        old_files["runtime_profile"].sha256,
        old_files["snapshot_profile"].sha256,
    )
    active_source = b"\n".join(
        new_files[label].raw_bytes
        for label in ("provisioner", "snapshot_boundary", "acquire_script")
    )
    for old_hash in old_hashes:
        if old_hash.encode("ascii") in active_source:
            _fail("candidate active source retains an old identity residue")


def _runtime_row_index(files: Mapping[str, _ExecutableIdentityFile]) -> int:
    value = _strict_json(files["runtime_profile"].raw_bytes, "runtime profile")
    rows = value.get("current_base", {}).get("source_files", [])
    matches = [
        index
        for index, row in enumerate(rows)
        if isinstance(row, dict) and row.get("path") == "configs/models.yaml"
    ] if isinstance(rows, list) else []
    if len(matches) != 1:
        _fail("runtime profile must contain one registry source row")
    return matches[0]


def build_identity_plan(
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    transaction_root: Path | str | None = None,
    runtime_root: Path | str | None = None,
    snapshot_root: Path | str | None = None,
    persistent_runtime_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    expected_old_hashes: Mapping[str, str] | None = None,
) -> IdentityTransactionPlan:
    """Build and validate the complete candidate graph without writing files."""

    if runtime_root is not None and persistent_runtime_root is not None:
        if Path(runtime_root).resolve() != Path(persistent_runtime_root).resolve():
            _fail("runtime_root and persistent_runtime_root disagree")
    selected_runtime_root = (
        runtime_root if runtime_root is not None else persistent_runtime_root
    )
    expected = _validate_expected_hashes(expected_old_hashes)
    layout = _resolve_layout(
        repository_root=repository_root,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        transaction_root=transaction_root,
        runtime_root=selected_runtime_root,
        snapshot_root=snapshot_root,
        environ=environ,
    )
    if layout.transaction_root.exists():
        _fail(
            "transaction root already exists; use recover for a valid journal",
            "DINOV3_IDENTITY_TRANSACTION_COLLISION",
        )
    live_files = _read_files(layout)
    live_hashes = {label: file.sha256 for label, file in live_files.items()}
    already_rebound = False
    if live_hashes == expected:
        old_files = live_files
        old_graph = _graph(old_files)
        _validate_graph(old_files, old_graph, final=False)
        new_files = _derive_candidates(old_files, old_graph)
    else:
        old_files = _reverse_final_files(live_files, expected)
        old_graph = _graph(old_files)
        _validate_graph(old_files, old_graph, final=False)
        new_files = _derive_candidates(old_files, old_graph)
        if any(
            new_files[label].raw_bytes != live_files[label].raw_bytes
            for label in _LABELS
        ):
            _fail(
                "live identity is not the exact old or exact final graph",
                "DINOV3_IDENTITY_BASELINE_MISMATCH",
            )
        already_rebound = True
    new_graph = _graph(new_files)
    _validate_graph(new_files, new_graph, final=True)
    _validate_candidate_diffs(old_files, new_files)
    mutations = _public_mutations(old_files, new_files)
    plan = IdentityTransactionPlan(
        transaction_id=TRANSACTION_ID,
        target_revision=TARGET_REVISION,
        old_graph=old_graph,
        new_graph=new_graph,
        mutations=mutations,
        rollback_order=tuple(reversed(_LABELS)),
        candidate_validated=True,
        already_rebound=already_rebound,
        final_verification_commands=_FINAL_VERIFICATION_COMMANDS,
    )
    validate_identity_plan(plan)
    return plan


def _classified_root(destination: Path, relative_path: str) -> Path:
    root = destination
    for _ in PurePosixPath(relative_path).parts:
        root = root.parent
    return root


def _validate_transaction_ancestor(path: Path) -> None:
    cursor = path
    while not cursor.exists():
        parent = cursor.parent
        if parent == cursor:
            _fail("transaction root has no existing safe ancestor")
        cursor = parent
    _reject_link(cursor, "transaction root ancestor")
    try:
        canonical = cursor.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve transaction root ancestor: {error}")
    if not _same_path(cursor, canonical):
        _fail("transaction root ancestor traverses a link or reparse point")


def _validate_execution_plan_paths(plan: _IdentityTransactionExecution) -> None:
    by_label = {mutation.label: mutation for mutation in plan.mutations}
    if set(by_label) != set(_LABELS):
        _fail("identity plan labels do not match the exact allowlist")
    for mutation in plan.mutations:
        destination = _lexical_absolute(mutation.destination)
        if not mutation.destination.is_absolute() or not _same_path(
            destination, mutation.destination
        ):
            _fail(f"{mutation.label} destination is not a normalized absolute path")
        if (
            not _same_path(mutation.old.destination, destination)
            or not _same_path(mutation.candidate.destination, destination)
        ):
            _fail(f"{mutation.label} nested identity destination is inconsistent")
        if not destination.is_file():
            _fail(f"{mutation.label} destination is not an existing file")
        _reject_link(destination, f"{mutation.label} destination")
    repository = _classified_root(
        by_label["registry"].destination,
        by_label["registry"].relative_path,
    )
    runtime = _classified_root(
        by_label["runtime_manifest"].destination,
        by_label["runtime_manifest"].relative_path,
    )
    snapshot = _classified_root(
        by_label["snapshot_manifest"].destination,
        by_label["snapshot_manifest"].relative_path,
    )
    classified_roots = {
        "repository": repository,
        "persistent_runtime": runtime,
        "external_snapshot": snapshot,
    }
    for classification, root in classified_roots.items():
        if not root.is_dir():
            _fail(f"derived {classification} root is not an existing directory")
        _reject_link(root, f"derived {classification} root")
    for mutation in plan.mutations:
        root = classified_roots[mutation.path_classification]
        expected_destination = root / PurePosixPath(mutation.relative_path)
        if not _same_path(mutation.destination, expected_destination):
            _fail(f"{mutation.label} destination is not bound to its allowlisted root")
        cursor = mutation.destination.parent
        while cursor != root:
            _reject_link(cursor, f"{mutation.label} destination ancestor")
            cursor = cursor.parent
    roots = tuple(classified_roots.values())
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if _roots_overlap(first, second):
                _fail("derived identity roots overlap")
    transaction = _lexical_absolute(plan.transaction_root)
    if not plan.transaction_root.is_absolute() or not _same_path(
        transaction, plan.transaction_root
    ):
        _fail("transaction root is not a normalized absolute path")
    if any(_roots_overlap(transaction, root) for root in roots):
        _fail("transaction root overlaps an identity root")
    if transaction.exists():
        _reject_link(transaction, "transaction root")
    _validate_transaction_ancestor(transaction)


def validate_identity_plan(plan: IdentityTransactionPlan) -> None:
    if not isinstance(plan, IdentityTransactionPlan):
        _fail("identity transaction plan has the wrong type")
    if plan.transaction_id != TRANSACTION_ID or plan.target_revision != TARGET_REVISION:
        _fail("identity transaction plan targets an unexpected identity")
    if not plan.candidate_validated or len(plan.mutations) != len(_LABELS):
        _fail("identity transaction plan is incomplete")
    if tuple(mutation.label for mutation in plan.mutations) != _LABELS:
        _fail("identity transaction destinations are not in the required order")
    if plan.rollback_order != tuple(reversed(_LABELS)):
        _fail("identity rollback order is invalid")
    if plan.final_verification_commands != _FINAL_VERIFICATION_COMMANDS:
        _fail("identity final verification commands are invalid")
    for mutation in plan.mutations:
        _safe_relative(mutation.relative_path, mutation.label)
        if mutation.path_classification != _CLASSIFICATIONS[mutation.label]:
            _fail(f"unsafe path classification for {mutation.label}")
        if mutation.relative_path != _RELATIVE_PATHS[mutation.label]:
            _fail(f"unexpected relative path for {mutation.label}")
        for state, file in (("old", mutation.old), ("candidate", mutation.candidate)):
            if (
                file.path_classification != mutation.path_classification
                or file.relative_path != mutation.relative_path
            ):
                _fail(f"{state} metadata path mismatch for {mutation.label}")
            if not _is_sha256(file.sha256):
                _fail(f"{state} metadata hash is invalid for {mutation.label}")
            if type(file.size_bytes) is not int or file.size_bytes < 0:
                _fail(f"{state} metadata size is invalid for {mutation.label}")
            if file.encoding != "utf-8" or file.line_ending not in {
                "LF",
                "CRLF",
                "MIXED",
                "NONE",
            }:
                _fail(f"{state} metadata encoding is invalid for {mutation.label}")
        if mutation.semantic_diff_paths != _SEMANTIC_DIFFS[mutation.label]:
            _fail(f"unexpected semantic diff declaration for {mutation.label}")
    old_hashes = {
        mutation.label: mutation.old.sha256 for mutation in plan.mutations
    }
    new_hashes = {
        mutation.label: mutation.candidate.sha256 for mutation in plan.mutations
    }
    if old_hashes != dict(plan.old_graph.hashes) or new_hashes != dict(
        plan.new_graph.hashes
    ):
        _fail("identity plan graph hashes are internally inconsistent")


def _public_mutations(
    old_files: Mapping[str, _ExecutableIdentityFile],
    new_files: Mapping[str, _ExecutableIdentityFile],
) -> tuple[IdentityMutation, ...]:
    return tuple(
        IdentityMutation(
            label=label,
            path_classification=_CLASSIFICATIONS[label],
            relative_path=_RELATIVE_PATHS[label],
            old=old_files[label].public(),
            candidate=new_files[label].public(),
            semantic_diff_paths=_SEMANTIC_DIFFS[label],
        )
        for label in _LABELS
    )


def _bind_execution_plan(
    plan: IdentityTransactionPlan,
    layout: _Layout,
    old_files: Mapping[str, _ExecutableIdentityFile],
    new_files: Mapping[str, _ExecutableIdentityFile],
) -> _IdentityTransactionExecution:
    old_graph = _graph(old_files)
    new_graph = _graph(new_files)
    mutations = _public_mutations(old_files, new_files)
    if (
        old_graph != plan.old_graph
        or new_graph != plan.new_graph
        or mutations != plan.mutations
    ):
        _fail(
            "caller roots do not materialize the supplied identity plan",
            "DINOV3_IDENTITY_PLAN_BINDING_MISMATCH",
        )
    executable_mutations = tuple(
        _ExecutableIdentityMutation(
            public=mutation,
            destination=layout.paths[mutation.label],
            old=old_files[mutation.label],
            candidate=new_files[mutation.label],
        )
        for mutation in mutations
    )
    executable = _IdentityTransactionExecution(
        public=plan,
        transaction_root=layout.transaction_root,
        mutations=executable_mutations,
    )
    _validate_execution_plan_paths(executable)
    return executable


def _materialize_execution_plan(
    plan: IdentityTransactionPlan,
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    transaction_root: Path | str | None,
    runtime_root: Path | str | None,
    snapshot_root: Path | str | None,
    persistent_runtime_root: Path | str | None,
    environ: Mapping[str, str] | None,
) -> _IdentityTransactionExecution:
    validate_identity_plan(plan)
    if runtime_root is not None and persistent_runtime_root is not None:
        if Path(runtime_root).resolve() != Path(persistent_runtime_root).resolve():
            _fail("runtime_root and persistent_runtime_root disagree")
    selected_runtime_root = (
        runtime_root if runtime_root is not None else persistent_runtime_root
    )
    layout = _resolve_layout(
        repository_root=repository_root,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        transaction_root=transaction_root,
        runtime_root=selected_runtime_root,
        snapshot_root=snapshot_root,
        environ=environ,
    )
    live_files = _read_files(layout)
    live_graph = _graph(live_files)
    if live_graph == plan.old_graph:
        old_files = live_files
        _validate_graph(old_files, plan.old_graph, final=False)
        new_files = _derive_candidates(old_files, plan.old_graph)
    elif live_graph == plan.new_graph:
        new_files = live_files
        _validate_graph(new_files, plan.new_graph, final=True)
        old_files = _reverse_final_files(live_files, plan.old_graph.hashes)
    else:
        _fail(
            "caller roots are neither the plan old graph nor exact final graph",
            "DINOV3_IDENTITY_BASELINE_MISMATCH",
        )
    _validate_graph(old_files, plan.old_graph, final=False)
    _validate_graph(new_files, plan.new_graph, final=True)
    _validate_candidate_diffs(old_files, new_files)
    return _bind_execution_plan(plan, layout, old_files, new_files)


def _identity_hash_subset(graph: IdentityGraph) -> dict[str, str]:
    return {
        "registry": graph.registry_sha256,
        "runtime_profile": graph.runtime_profile_sha256,
        "snapshot_profile": graph.snapshot_profile_sha256,
    }


def _validate_journal_value(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != _JOURNAL_FIELDS:
        _fail("transaction journal has unknown or missing top-level fields")
    if value["schema_id"] != JOURNAL_SCHEMA_ID:
        _fail("transaction journal schema is invalid")
    if value["transaction_id"] != TRANSACTION_ID:
        _fail("transaction journal ID is invalid")
    if value["state"] not in _JOURNAL_STATES:
        _fail("transaction journal state is invalid")
    destinations = value["destinations"]
    if not isinstance(destinations, list) or not destinations:
        _fail("transaction journal destinations must be a non-empty list")
    final_index = len(destinations) - 1
    for field in ("applied_index", "rollback_index"):
        if (
            type(value[field]) is not int
            or value[field] < -1
            or value[field] > final_index
        ):
            _fail(f"transaction journal {field} is invalid")
    for field in ("started_at_utc", "last_transition_utc"):
        timestamp = value[field]
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            _fail(f"transaction journal {field} is invalid")
        try:
            parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except ValueError:
            _fail(f"transaction journal {field} is invalid")
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            _fail(f"transaction journal {field} is not UTC")
    for field in ("old_identity_hashes", "candidate_identity_hashes"):
        hashes = value[field]
        if not isinstance(hashes, dict) or set(hashes) != _IDENTITY_HASH_KEYS:
            _fail(f"transaction journal {field} keys are invalid")
        if not all(_is_sha256(item) for item in hashes.values()):
            _fail(f"transaction journal {field} contains an invalid hash")
    seen_destinations: set[tuple[str, str]] = set()
    identity_by_path = {
        _RELATIVE_PATHS[label]: label
        for label in ("registry", "runtime_profile", "snapshot_profile")
    }
    for index, destination in enumerate(destinations):
        if not isinstance(destination, dict) or set(destination) != _JOURNAL_DESTINATION_FIELDS:
            _fail("transaction journal destination has unknown or missing fields")
        if destination["path_classification"] not in set(_CLASSIFICATIONS.values()):
            _fail("transaction journal destination classification is invalid")
        for field in (
            "relative_path",
            "backup_relative_path",
            "candidate_relative_path",
        ):
            if not isinstance(destination[field], str):
                _fail(f"transaction journal destination {field} is invalid")
            _safe_relative(destination[field], field)
        if destination["backup_relative_path"] != f"backups/{index:03d}.bin":
            _fail("transaction journal backup path is not canonical")
        if destination["candidate_relative_path"] != f"candidates/{index:03d}.bin":
            _fail("transaction journal candidate path is not canonical")
        if not _is_sha256(destination["old_sha256"]) or not _is_sha256(
            destination["new_sha256"]
        ):
            _fail("transaction journal destination contains an invalid hash")
        destination_key = (
            destination["path_classification"],
            destination["relative_path"],
        )
        if destination_key in seen_destinations:
            _fail("transaction journal contains a duplicate destination")
        seen_destinations.add(destination_key)
        identity_label = identity_by_path.get(destination["relative_path"])
        if identity_label is not None:
            if (
                value["old_identity_hashes"][identity_label]
                != destination["old_sha256"]
                or value["candidate_identity_hashes"][identity_label]
                != destination["new_sha256"]
            ):
                _fail("transaction journal identity hashes disagree with a destination")
    state = value["state"]
    applied_index = value["applied_index"]
    rollback_index = value["rollback_index"]
    if state == "prepared" and (applied_index != -1 or rollback_index != -1):
        _fail("prepared journal has impossible progress indexes")
    if state in {"applied", "verifying", "committed"} and applied_index != final_index:
        _fail(f"{state} journal does not show every destination applied")
    if state in {"prepared", "applying", "applied", "verifying", "committed"} and rollback_index != -1:
        _fail(f"{state} journal has an impossible rollback index")
    if rollback_index > applied_index and applied_index != -1:
        _fail("transaction journal rollback index exceeds applied progress")
    _scan_sensitive_fields(value)


def _journal_value(
    plan: _IdentityTransactionExecution, started: str
) -> dict[str, Any]:
    return {
        "applied_index": -1,
        "candidate_identity_hashes": _identity_hash_subset(plan.new_graph),
        "destinations": [
            {
                "backup_relative_path": f"backups/{index:03d}.bin",
                "candidate_relative_path": f"candidates/{index:03d}.bin",
                "new_sha256": mutation.new_sha256,
                "old_sha256": mutation.old_sha256,
                "path_classification": mutation.path_classification,
                "relative_path": mutation.relative_path,
            }
            for index, mutation in enumerate(plan.mutations)
        ],
        "last_transition_utc": started,
        "old_identity_hashes": _identity_hash_subset(plan.old_graph),
        "rollback_index": -1,
        "schema_id": JOURNAL_SCHEMA_ID,
        "started_at_utc": started,
        "state": "prepared",
        "transaction_id": TRANSACTION_ID,
    }


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


def _write_fsynced(path: Path, payload: bytes, *, exclusive: bool = False) -> None:
    mode = "xb" if exclusive else "wb"
    try:
        with path.open(mode) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        _fail(f"cannot write transaction file {path.name!r}: {error}")


def _write_journal(root: Path, value: dict[str, Any]) -> None:
    _reject_link(root, "transaction root")
    if not _same_path(root, root.resolve(strict=True)):
        _fail("transaction root traverses a link or reparse point")
    value["last_transition_utc"] = _utc_now()
    _validate_journal_value(value)
    path = root / "journal.json"
    temporary = root / "journal.json.tmp"
    if temporary.exists() or temporary.is_symlink():
        _fail("transaction journal temporary path already exists")
    if path.exists() and (path.is_symlink() or _has_reparse_attribute(path)):
        _fail("transaction journal path is a link or reparse point")
    _write_fsynced(temporary, _canonical_json(value), exclusive=True)
    os.replace(temporary, path)
    _fsync_directory(root)


def _read_journal(path: Path) -> dict[str, Any]:
    value = _strict_json(path.read_bytes(), "transaction journal")
    _validate_journal_value(value)
    return value


def _remove_recoverable_journal_temporary(root: Path) -> None:
    temporary = root / "journal.json.tmp"
    if not temporary.exists() and not temporary.is_symlink():
        return
    if temporary.is_symlink() or _has_reparse_attribute(temporary):
        _fail("transaction journal temporary path is a link or reparse point")
    if not temporary.is_file():
        _fail("transaction journal temporary path is not a regular file")
    try:
        canonical = temporary.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve transaction journal temporary path: {error}")
    if not _same_path(temporary, canonical) or not _is_relative_to(temporary, root):
        _fail("transaction journal temporary path escapes its transaction root")
    temporary.unlink()
    _fsync_directory(root)


def _journal_payload_path(root: Path, relative_path: str, label: str) -> Path:
    _safe_relative(relative_path, label)
    path = root.joinpath(*PurePosixPath(relative_path).parts)
    if not _is_relative_to(path, root):
        _fail(f"{label} escapes the transaction root")
    if not path.is_file():
        _fail(f"{label} is missing")
    _reject_link(path, label)
    cursor = path.parent
    while cursor != root:
        _reject_link(cursor, f"{label} ancestor")
        cursor = cursor.parent
    try:
        canonical = path.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve {label}: {error}")
    if not _same_path(path, canonical):
        _fail(f"{label} traverses a link or reparse point")
    return path


def _same_parent_stage(
    mutation: _ExecutableIdentityMutation, index: int
) -> Path:
    return mutation.destination.parent / (
        f".vea-{TRANSACTION_ID}-{index:03d}.candidate.tmp"
    )


def _same_parent_rollback(
    mutation: _ExecutableIdentityMutation, index: int
) -> Path:
    return mutation.destination.parent / (
        f".vea-{TRANSACTION_ID}-{index:03d}.rollback.tmp"
    )


def _invoke(
    callback: Callable[..., Any] | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if callback is None:
        return True
    return callback(*args, **kwargs)


def _inject(
    callback: Callable[..., Any] | None,
    event: str,
    index: int,
) -> None:
    if callback is not None:
        callback(event, index)


def _verify_plan_final(plan: _IdentityTransactionExecution) -> IdentityGraph:
    files: dict[str, _ExecutableIdentityFile] = {}
    for mutation in plan.mutations:
        current_hash = _sha256_file(mutation.destination)
        if current_hash != mutation.new_sha256:
            _fail(
                f"final identity hash mismatch for {mutation.label}",
                "DINOV3_IDENTITY_FINAL_VERIFY_FAILED",
            )
        files[mutation.label] = _identity_file(
            mutation.label, mutation.destination
        )
    graph = _graph(files)
    if graph != plan.new_graph:
        _fail(
            "final identity graph differs from the candidate graph",
            "DINOV3_IDENTITY_FINAL_VERIFY_FAILED",
        )
    _validate_graph(files, graph, final=True)
    return graph


def _cleanup_transaction(plan: _IdentityTransactionExecution) -> None:
    for index, mutation in enumerate(plan.mutations):
        for path in (
            _same_parent_stage(mutation, index),
            _same_parent_rollback(mutation, index),
        ):
            if path.exists():
                _reject_link(path, "transaction staging file")
                path.unlink()
    root = _lexical_absolute(plan.transaction_root)
    if root.exists():
        _reject_link(root, "transaction root")
        try:
            canonical = root.resolve(strict=True)
        except OSError as error:
            _fail(f"cannot resolve transaction root for cleanup: {error}")
        if not _same_path(root, canonical):
            _fail("refusing to clean a transaction root that traverses a link")
        if any(
            _is_relative_to(root, mutation.destination.parent)
            or _is_relative_to(mutation.destination, root)
            for mutation in plan.mutations
        ):
            _fail("refusing to remove an overlapping transaction root")
        shutil.rmtree(root)


def _prepare_transaction(
    plan: _IdentityTransactionExecution,
) -> tuple[dict[str, Any], list[str]]:
    root = plan.transaction_root
    root.parent.mkdir(parents=True, exist_ok=True)
    _reject_link(root.parent, "transaction parent")
    if not _same_path(root.parent, root.parent.resolve(strict=True)):
        _fail("transaction parent traverses a link or reparse point")
    try:
        root.mkdir(exist_ok=False)
    except FileExistsError:
        _fail(
            "transaction root already exists",
            "DINOV3_IDENTITY_TRANSACTION_COLLISION",
        )
    (root / "backups").mkdir()
    (root / "candidates").mkdir()
    for index, mutation in enumerate(plan.mutations):
        backup = root / "backups" / f"{index:03d}.bin"
        candidate = root / "candidates" / f"{index:03d}.bin"
        stage = _same_parent_stage(mutation, index)
        if stage.exists():
            _fail(f"same-parent candidate staging collision for {mutation.label}")
        _write_fsynced(backup, mutation.old.raw_bytes, exclusive=True)
        _write_fsynced(candidate, mutation.candidate.raw_bytes, exclusive=True)
        _write_fsynced(stage, mutation.candidate.raw_bytes, exclusive=True)
        if _sha256_file(backup) != mutation.old_sha256:
            _fail(f"backup verification failed for {mutation.label}")
        if _sha256_file(candidate) != mutation.new_sha256:
            _fail(f"candidate workspace verification failed for {mutation.label}")
        if _sha256_file(stage) != mutation.new_sha256:
            _fail(f"same-parent candidate verification failed for {mutation.label}")
    started = _utc_now()
    journal = _journal_value(plan, started)
    _write_journal(root, journal)
    return journal, ["prepared"]


def _rollback(
    plan: _IdentityTransactionExecution,
    journal: dict[str, Any],
    applied_indexes: list[int],
    transitions: list[str],
    rollback_verify: Callable[..., Any] | None,
    failure_injector: Callable[..., Any] | None,
) -> None:
    journal["applied_index"] = max(applied_indexes, default=-1)
    journal["state"] = "rolling_back"
    _write_journal(plan.transaction_root, journal)
    transitions.append("rolling_back")
    for index in reversed(applied_indexes):
        mutation = plan.mutations[index]
        current_hash = _sha256_file(mutation.destination)
        if current_hash == mutation.old_sha256:
            continue
        if current_hash != mutation.new_sha256:
            _fail(
                f"rollback compare-and-swap failed for {mutation.label}",
                "DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED",
            )
        _inject(failure_injector, "before_rollback_replace", index)
        backup = _journal_payload_path(
            plan.transaction_root,
            str(journal["destinations"][index]["backup_relative_path"]),
            f"rollback backup {index}",
        )
        if _sha256_file(backup) != mutation.old_sha256:
            _fail(
                f"rollback backup hash failed for {mutation.label}",
                "DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED",
            )
        temporary = _same_parent_rollback(mutation, index)
        if temporary.exists():
            _fail(
                f"rollback staging collision for {mutation.label}",
                "DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED",
            )
        _write_fsynced(temporary, backup.read_bytes(), exclusive=True)
        os.replace(temporary, mutation.destination)
        _fsync_directory(mutation.destination.parent)
        if _sha256_file(mutation.destination) != mutation.old_sha256:
            _fail(
                f"restored hash mismatch for {mutation.label}",
                "DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED",
            )
        journal["rollback_index"] = index
        _write_journal(plan.transaction_root, journal)
        _inject(failure_injector, "after_rollback_replace", index)
    for mutation in plan.mutations:
        if _sha256_file(mutation.destination) != mutation.old_sha256:
            _fail(
                f"rollback did not restore the complete old graph at {mutation.label}",
                "DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED",
            )
    verified = _invoke(rollback_verify)
    if verified is False:
        _fail(
            "rollback verifier returned false",
            "DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED",
        )
    journal["state"] = "rolled_back"
    _write_journal(plan.transaction_root, journal)
    transitions.append("rolled_back")


def apply_identity_transaction(
    plan: IdentityTransactionPlan,
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    transaction_root: Path | str | None = None,
    runtime_root: Path | str | None = None,
    snapshot_root: Path | str | None = None,
    persistent_runtime_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    post_verify: Callable[..., Any] | None = None,
    rollback_verify: Callable[..., Any] | None = None,
    failure_injector: Callable[..., Any] | None = None,
) -> IdentityTransactionResult:
    """Apply a fully validated plan, compensating on every ordinary failure."""

    plan = _materialize_execution_plan(
        plan,
        repository_root=repository_root,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        transaction_root=transaction_root,
        runtime_root=runtime_root,
        snapshot_root=snapshot_root,
        persistent_runtime_root=persistent_runtime_root,
        environ=environ,
    )
    current = tuple(_sha256_file(item.destination) for item in plan.mutations)
    old = tuple(item.old_sha256 for item in plan.mutations)
    new = tuple(item.new_sha256 for item in plan.mutations)
    if current == new:
        graph = _verify_plan_final(plan)
        return IdentityTransactionResult(
            status="DINOV3_IDENTITY_ALREADY_REBOUND",
            old_graph=plan.old_graph,
            final_graph=graph,
            applied=(),
            journal_transitions=(),
            compare_and_swap="EXACT_FINAL_STATE",
            rollback_attempted=False,
            rollback_succeeded=None,
            cleanup_succeeded=not plan.transaction_root.exists(),
            already_rebound=True,
        )
    if current != old:
        _fail(
            "live identity changed after plan construction",
            "DINOV3_IDENTITY_BASELINE_MISMATCH",
        )
    if plan.transaction_root.exists():
        _fail(
            "transaction root already exists",
            "DINOV3_IDENTITY_TRANSACTION_COLLISION",
        )
    journal: dict[str, Any] | None = None
    transitions: list[str] = []
    applied_indexes: list[int] = []
    try:
        journal, transitions = _prepare_transaction(plan)
        _inject(failure_injector, "after_prepare", -1)
        journal["state"] = "applying"
        _write_journal(plan.transaction_root, journal)
        transitions.append("applying")
        for index, mutation in enumerate(plan.mutations):
            _inject(failure_injector, "before_replace", index)
            if _sha256_file(mutation.destination) != mutation.old_sha256:
                raise IdentityRebindError(
                    f"compare-and-swap baseline changed for {mutation.label}",
                    classification="DINOV3_IDENTITY_CONCURRENT_MODIFICATION",
                )
            stage = _same_parent_stage(mutation, index)
            if _sha256_file(stage) != mutation.new_sha256:
                raise IdentityRebindError(
                    f"candidate staging hash changed for {mutation.label}"
                )
            # Register the destination before replacement.  If replacement
            # succeeds and any immediately following durability/hash probe
            # fails, compensation must still visit this index.  A failed
            # replace leaves old bytes and rollback safely treats that as a
            # no-op.
            applied_indexes.append(index)
            os.replace(stage, mutation.destination)
            _fsync_directory(mutation.destination.parent)
            if _sha256_file(mutation.destination) != mutation.new_sha256:
                raise IdentityRebindError(
                    f"applied hash mismatch for {mutation.label}"
                )
            journal["applied_index"] = index
            _write_journal(plan.transaction_root, journal)
            _inject(failure_injector, "after_replace", index)
        journal["state"] = "applied"
        _write_journal(plan.transaction_root, journal)
        transitions.append("applied")
        journal["state"] = "verifying"
        _write_journal(plan.transaction_root, journal)
        transitions.append("verifying")
        _inject(failure_injector, "before_verify", -1)
        verified = _invoke(post_verify)
        if verified is False:
            raise IdentityRebindError("post-transaction verifier returned false")
        final_graph = _verify_plan_final(plan)
        journal["state"] = "committed"
        _write_journal(plan.transaction_root, journal)
        transitions.append("committed")
        _cleanup_transaction(plan)
        return IdentityTransactionResult(
            status="DINOV3_IDENTITY_REBIND_COMMITTED",
            old_graph=plan.old_graph,
            final_graph=final_graph,
            applied=tuple(plan.mutations[index].label for index in applied_indexes),
            journal_transitions=tuple(transitions),
            compare_and_swap="VERIFIED_ALL_DESTINATIONS",
            rollback_attempted=False,
            rollback_succeeded=None,
            cleanup_succeeded=not plan.transaction_root.exists(),
            already_rebound=False,
        )
    except Exception as apply_error:
        if journal is None:
            if (
                isinstance(apply_error, IdentityRebindError)
                and apply_error.classification
                == "DINOV3_IDENTITY_TRANSACTION_COLLISION"
            ):
                raise
            # Preparation can fail before a valid journal exists.  Remove only
            # deterministic task-owned staging files and retain no partial root.
            try:
                _cleanup_transaction(plan)
            except Exception:
                pass
            if isinstance(apply_error, IdentityRebindError):
                raise
            raise IdentityRebindError(str(apply_error)) from apply_error
        try:
            _rollback(
                plan,
                journal,
                applied_indexes,
                transitions,
                rollback_verify,
                failure_injector,
            )
            _cleanup_transaction(plan)
        except Exception as rollback_error:
            try:
                journal["state"] = "rollback_failed"
                _write_journal(plan.transaction_root, journal)
            except Exception:
                pass
            raise IdentityRebindError(
                f"identity transaction rollback failed: {rollback_error}",
                classification="DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED",
            ) from apply_error
        raise IdentityRebindError(
            f"identity transaction failed and rolled back: {apply_error}",
            classification="DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK",
        ) from apply_error


def verify_rebound_identity(
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    transaction_root: Path | str | None = None,
    runtime_root: Path | str | None = None,
    snapshot_root: Path | str | None = None,
    persistent_runtime_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    expected_old_hashes: Mapping[str, str] | None = None,
    plan: IdentityTransactionPlan | None = None,
) -> IdentityGraph:
    """Read-only verification of the exact final graph."""

    if plan is not None:
        executable = _materialize_execution_plan(
            plan,
            repository_root=repository_root,
            runtime_profile_path=runtime_profile_path,
            snapshot_profile_path=snapshot_profile_path,
            transaction_root=transaction_root,
            runtime_root=runtime_root,
            snapshot_root=snapshot_root,
            persistent_runtime_root=persistent_runtime_root,
            environ=environ,
        )
        return _verify_plan_final(executable)
    candidate_plan = build_identity_plan(
        repository_root=repository_root,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        transaction_root=transaction_root,
        runtime_root=runtime_root,
        snapshot_root=snapshot_root,
        persistent_runtime_root=persistent_runtime_root,
        environ=environ,
        expected_old_hashes=expected_old_hashes,
    )
    if not candidate_plan.already_rebound:
        _fail(
            "identity graph is still at the accepted old state",
            "DINOV3_IDENTITY_FINAL_VERIFY_FAILED",
        )
    executable = _materialize_execution_plan(
        candidate_plan,
        repository_root=repository_root,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        transaction_root=transaction_root,
        runtime_root=runtime_root,
        snapshot_root=snapshot_root,
        persistent_runtime_root=persistent_runtime_root,
        environ=environ,
    )
    return _verify_plan_final(executable)


def verify_accepted_identity_state(
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    extractor_profile_path: Path | str,
    transaction_root: Path | str | None = None,
    runtime_root: Path | str | None = None,
    snapshot_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> AcceptedIdentityVerification:
    """Read-only classification of an exact registry-pin or production final graph."""

    legacy_arguments = {
        "repository_root": repository_root,
        "runtime_profile_path": runtime_profile_path,
        "snapshot_profile_path": snapshot_profile_path,
        "transaction_root": transaction_root,
        "runtime_root": runtime_root,
        "snapshot_root": snapshot_root,
        "environ": environ,
    }
    try:
        graph = verify_rebound_identity(**legacy_arguments)
    except IdentityRebindError as legacy_error:
        graph_state_failure = (
            legacy_error.classification in _COMPATIBILITY_FALLBACK_CLASSIFICATIONS
            or (
                legacy_error.classification == "DINOV3_IDENTITY_REBIND_ERROR"
                and str(legacy_error) in _COMPATIBILITY_FALLBACK_MESSAGES
            )
        )
        if not graph_state_failure:
            raise
        legacy_classification = legacy_error.classification
    else:
        return AcceptedIdentityVerification(
            state="registry_pin_final",
            marker=_REGISTRY_PIN_FINAL_MARKER,
            final_hashes=tuple(sorted(graph.hashes.items())),
        )

    # Keep the production transaction implementation outside the legacy
    # module's import-time graph.  This is the sole compatibility-only edge.
    from .dinov3_production_identity import (  # noqa: PLC0415
        DINOv3ProductionIdentityError,
        verify_production_identity,
    )

    try:
        production = verify_production_identity(
            repository_root=repository_root,
            runtime_profile_path=runtime_profile_path,
            snapshot_profile_path=snapshot_profile_path,
            extractor_profile_path=extractor_profile_path,
            transaction_root=transaction_root,
            runtime_root=runtime_root,
            snapshot_root=snapshot_root,
            environ=environ,
        )
    except DINOv3ProductionIdentityError as production_error:
        raise IdentityRebindError(
            "neither accepted DINOv3 identity terminal state verified "
            f"(legacy={legacy_classification}; "
            f"production={production_error.classification})",
            classification="DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED",
        ) from None
    return AcceptedIdentityVerification(
        state="production_final",
        marker=_PRODUCTION_FINAL_MARKER,
        final_hashes=tuple(production.final_hashes),
    )


def verify_versioned_production_graph(
    *,
    repository_root: Path | str,
    graph_id: str,
    runtime_profile_path: Path | str,
    extractor_profile_path: Path | str,
    snapshot_profile_path: Path | str | None = None,
    semantic_profile_path: Path | str | None = None,
) -> Any:
    """Verify one explicitly selected production graph without host inference.

    The import remains local so the legacy registry-rebind module keeps its
    standard-library-only import boundary and does not acquire a second graph
    through ambient hostname, CUDA, or environment state.
    """

    from .dinov3_production_identity import (  # noqa: PLC0415
        APPLE_GRAPH_ID,
        APPLE_UNRESOLVED_GRAPH_ID,
        DINOv3ProductionIdentityError,
        LEGACY_GRAPH_ID,
        validate_versioned_production_graph,
    )

    try:
        if graph_id not in {LEGACY_GRAPH_ID, APPLE_UNRESOLVED_GRAPH_ID, APPLE_GRAPH_ID}:
            raise DINOv3ProductionIdentityError(
                "graph ID must be an explicit versioned identity",
                "DINOV3_UNKNOWN_PRODUCTION_GRAPH",
            )
        return validate_versioned_production_graph(
            repository_root,
            graph_id=graph_id,
            runtime_profile_path=runtime_profile_path,
            extractor_profile_path=extractor_profile_path,
            snapshot_profile_path=snapshot_profile_path,
            semantic_profile_path=semantic_profile_path,
        )
    except DINOv3ProductionIdentityError as exc:
        raise IdentityRebindError(
            "explicit versioned production graph did not verify",
            classification="DINOV3_IDENTITY_COMPATIBILITY_VERIFY_FAILED",
        ) from exc


def _recovery_layout(
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    transaction_root: Path | str | None,
    runtime_root: Path | str | None,
    snapshot_root: Path | str | None,
    persistent_runtime_root: Path | str | None,
    environ: Mapping[str, str] | None,
) -> _Layout:
    if runtime_root is not None and persistent_runtime_root is not None:
        if Path(runtime_root).resolve() != Path(persistent_runtime_root).resolve():
            _fail("runtime_root and persistent_runtime_root disagree")
    selected_runtime = runtime_root if runtime_root is not None else persistent_runtime_root
    layout = _resolve_layout(
        repository_root=repository_root,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        transaction_root=transaction_root,
        runtime_root=selected_runtime,
        snapshot_root=snapshot_root,
        environ=environ,
    )
    if not layout.transaction_root.is_dir():
        _fail("recover requires an existing transaction root")
    _reject_link(layout.transaction_root, "transaction root")
    return layout


def recover_identity_transaction(
    *,
    repository_root: Path | str,
    runtime_profile_path: Path | str,
    snapshot_profile_path: Path | str,
    transaction_root: Path | str | None = None,
    runtime_root: Path | str | None = None,
    snapshot_root: Path | str | None = None,
    persistent_runtime_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    expected_old_hashes: Mapping[str, str] | None = None,
    plan: IdentityTransactionPlan | None = None,
    rollback_verify: Callable[..., Any] | None = None,
    failure_injector: Callable[..., Any] | None = None,
) -> IdentityTransactionResult:
    """Recover a strict incomplete journal by restoring every original byte."""

    expected = _validate_expected_hashes(expected_old_hashes)
    layout = _recovery_layout(
        repository_root=repository_root,
        runtime_profile_path=runtime_profile_path,
        snapshot_profile_path=snapshot_profile_path,
        transaction_root=transaction_root,
        runtime_root=runtime_root,
        snapshot_root=snapshot_root,
        persistent_runtime_root=persistent_runtime_root,
        environ=environ,
    )
    journal_path = layout.transaction_root / "journal.json"
    if not journal_path.is_file():
        _fail("recover requires a transaction journal")
    _reject_link(journal_path, "transaction journal")
    journal = _read_journal(journal_path)
    if journal["state"] in {"committed", "rolled_back"}:
        _fail("recover refuses a terminal transaction journal")
    if len(journal["destinations"]) != len(_LABELS):
        _fail("recovery journal does not contain the complete destination allowlist")
    old_files: dict[str, _ExecutableIdentityFile] = {}
    journal_candidates: dict[str, _ExecutableIdentityFile] = {}
    for index, (label, row) in enumerate(
        zip(_LABELS, journal["destinations"], strict=True)
    ):
        if (
            row["path_classification"] != _CLASSIFICATIONS[label]
            or row["relative_path"] != _RELATIVE_PATHS[label]
        ):
            _fail("recovery journal destination allowlist mismatch")
        if row["old_sha256"] != expected[label]:
            _fail("recovery journal old identity mismatch")
        backup = _journal_payload_path(
            layout.transaction_root,
            str(row["backup_relative_path"]),
            f"recovery backup {index}",
        )
        candidate = _journal_payload_path(
            layout.transaction_root,
            str(row["candidate_relative_path"]),
            f"recovery candidate {index}",
        )
        old_file = _identity_file(label, layout.paths[label], backup.read_bytes())
        candidate_file = _identity_file(
            label, layout.paths[label], candidate.read_bytes()
        )
        if (
            old_file.sha256 != row["old_sha256"]
            or candidate_file.sha256 != row["new_sha256"]
        ):
            _fail("recovery journal payload hash mismatch")
        old_files[label] = old_file
        journal_candidates[label] = candidate_file

    old_graph = _graph(old_files)
    if old_graph.hashes != expected:
        _fail("recovery plan old graph does not match accepted old hashes")
    _validate_graph(old_files, old_graph, final=False)
    candidate_files = _derive_candidates(old_files, old_graph)
    if any(
        candidate_files[label].raw_bytes != journal_candidates[label].raw_bytes
        for label in _LABELS
    ):
        _fail("recovery candidate payload is not the canonical derived graph")
    new_graph = _graph(candidate_files)
    _validate_graph(candidate_files, new_graph, final=True)
    _validate_candidate_diffs(old_files, candidate_files)
    derived_plan = IdentityTransactionPlan(
        transaction_id=TRANSACTION_ID,
        target_revision=TARGET_REVISION,
        old_graph=old_graph,
        new_graph=new_graph,
        mutations=_public_mutations(old_files, candidate_files),
        rollback_order=tuple(reversed(_LABELS)),
        candidate_validated=True,
        already_rebound=False,
        final_verification_commands=_FINAL_VERIFICATION_COMMANDS,
    )
    validate_identity_plan(derived_plan)
    if plan is not None:
        validate_identity_plan(plan)
        if plan != derived_plan:
            _fail("recovery plan does not exactly match the journal payload graph")
    executable = _bind_execution_plan(
        derived_plan, layout, old_files, candidate_files
    )
    mutations = executable.mutations
    if derived_plan.old_graph.hashes != expected:
        _fail("recovery plan old graph does not match accepted old hashes")
    if (
        _identity_hash_subset(derived_plan.old_graph)
        != journal["old_identity_hashes"]
        or _identity_hash_subset(derived_plan.new_graph)
        != journal["candidate_identity_hashes"]
    ):
        _fail("recovery plan graph does not match journal identity hashes")
    for index, (label, row, mutation) in enumerate(
        zip(_LABELS, journal["destinations"], mutations, strict=True)
    ):
        if (
            mutation.label != label
            or row["path_classification"] != mutation.path_classification
            or row["relative_path"] != mutation.relative_path
            or row["old_sha256"] != mutation.old_sha256
            or row["new_sha256"] != mutation.new_sha256
        ):
            _fail("recovery journal does not match the supplied identity plan")
        backup = _journal_payload_path(
            layout.transaction_root,
            str(row["backup_relative_path"]),
            f"recovery backup {index}",
        )
        candidate = _journal_payload_path(
            layout.transaction_root,
            str(row["candidate_relative_path"]),
            f"recovery candidate {index}",
        )
        if (
            _sha256_file(backup) != mutation.old_sha256
            or _sha256_file(candidate) != mutation.new_sha256
        ):
            _fail("recovery payload hash differs from the journal")
    _remove_recoverable_journal_temporary(layout.transaction_root)
    transitions = [str(journal["state"]), "rolling_back"]
    journal["state"] = "rolling_back"
    _write_journal(layout.transaction_root, journal)
    restored: list[int] = []
    try:
        for index in reversed(range(len(mutations))):
            mutation = mutations[index]
            current = _sha256_file(mutation.destination)
            if current == mutation.old_sha256:
                continue
            if current != mutation.new_sha256:
                _fail(
                    f"recovery found an unknown live hash for {mutation.label}",
                    "DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED",
                )
            _inject(failure_injector, "before_rollback_replace", index)
            row = journal["destinations"][index]
            backup = _journal_payload_path(
                layout.transaction_root,
                str(row["backup_relative_path"]),
                f"recovery backup {index}",
            )
            temporary = _same_parent_rollback(mutation, index)
            if temporary.exists():
                _fail("recovery rollback staging path already exists")
            _write_fsynced(temporary, backup.read_bytes(), exclusive=True)
            os.replace(temporary, mutation.destination)
            _fsync_directory(mutation.destination.parent)
            if _sha256_file(mutation.destination) != mutation.old_sha256:
                _fail("recovery restored hash verification failed")
            restored.append(index)
            journal["rollback_index"] = index
            _write_journal(layout.transaction_root, journal)
            _inject(failure_injector, "after_rollback_replace", index)
        if any(_sha256_file(item.destination) != item.old_sha256 for item in mutations):
            _fail("recovery did not restore the complete old identity graph")
        if _invoke(rollback_verify) is False:
            _fail("recovery rollback verifier returned false")
        journal["state"] = "rolled_back"
        _write_journal(layout.transaction_root, journal)
        transitions.append("rolled_back")
        _cleanup_transaction(executable)
    except Exception as error:
        try:
            journal["state"] = "rollback_failed"
            _write_journal(layout.transaction_root, journal)
        except Exception:
            pass
        raise IdentityRebindError(
            f"identity recovery rollback failed: {error}",
            classification="DINOV3_IDENTITY_TRANSACTION_ROLLBACK_FAILED",
        ) from error
    return IdentityTransactionResult(
        status="DINOV3_IDENTITY_TRANSACTION_FAILED_ROLLED_BACK",
        old_graph=old_graph,
        final_graph=old_graph,
        applied=tuple(mutations[index].label for index in restored),
        journal_transitions=tuple(transitions),
        compare_and_swap="RECOVERY_VERIFIED",
        rollback_attempted=True,
        rollback_succeeded=True,
        cleanup_succeeded=not layout.transaction_root.exists(),
        already_rebound=False,
    )


__all__ = [
    "AcceptedIdentityVerification",
    "IdentityFile",
    "IdentityGraph",
    "IdentityMutation",
    "IdentityRebindError",
    "IdentityTransactionPlan",
    "IdentityTransactionResult",
    "apply_identity_transaction",
    "build_identity_plan",
    "default_transaction_root",
    "recover_identity_transaction",
    "validate_identity_plan",
    "verify_accepted_identity_state",
    "verify_versioned_production_graph",
    "verify_rebound_identity",
]
