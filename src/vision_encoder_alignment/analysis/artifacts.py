"""Monotonic campaign state, atomic artifacts, and deterministic package sealing."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .access_control import sha256_file
from .config import canonical_json_bytes


STATE_SEQUENCE = (
    "PREFLIGHT", "BLIND_INPUT_INTEGRITY", "REPEAT_CONTROL", "T004_DEV",
    "T005_DEV", "T006_DEV", "EXTENSION_DEV", "FREEZE",
    "T007_TEST_UNSEAL", "REPORT", "SEAL", "VERIFY",
)

STAGE_DIRECTORIES = {
    "PREFLIGHT": "preflight",
    "BLIND_INPUT_INTEGRITY": "blind_input_integrity",
    "REPEAT_CONTROL": "repeat_control",
    "T004_DEV": "t004_dev",
    "T005_DEV": "t005_dev",
    "T006_DEV": "t006_dev",
    "EXTENSION_DEV": "extension_dev",
    "FREEZE": "freeze",
    "T007_TEST_UNSEAL": "t007_test",
    "REPORT": "report",
    "SEAL": "seal_stage",
    "VERIFY": "verify_stage",
}


class ArtifactError(RuntimeError):
    """Raised for state, publication, or package-integrity failures."""


def atomic_write_bytes(path: str | Path, payload: bytes, *, replace: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and destination.exists():
            raise ArtifactError(f"output collision: {destination.name}")
        if replace:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise ArtifactError(f"output collision: {destination.name}") from exc
            temporary.unlink()
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: object, *, replace: bool = False) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(value), replace=replace)


def append_jsonl(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as stream:
        stream.write(canonical_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def deterministic_result_name(run_id: str, commit_sha: str, config_sha: str) -> str:
    if len(commit_sha) != 40 or len(config_sha) != 64:
        raise ArtifactError("result identity requires full commit and config SHA")
    return f"{run_id}_analysis_{commit_sha[:12]}_{config_sha[:12]}"


class CampaignStore:
    """Exclusive, resumable owner of one deterministic campaign result root."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        commit_sha: str,
        tree_sha: str,
        config_sha: str,
        input_checksum_manifest_sha: str,
    ) -> None:
        self.root = Path(root)
        self.identity = {
            "run_id": run_id,
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
            "config_sha256": config_sha,
            "input_checksum_manifest_sha256": input_checksum_manifest_sha,
        }
        self.state_path = self.root / "campaign_state.json"
        self.events_path = self.root / "campaign_events.jsonl"
        self.lock_path = self.root / ".campaign.lock"
        self._locked = False

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ArtifactError("concurrent campaign lock exists") from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes({"schema_id": "vea.p1.campaign_lock.v1", **self.identity}))
            stream.flush()
            os.fsync(stream.fileno())
        self._locked = True
        try:
            if self.state_path.exists():
                state = self.load_state()
                for key, expected in self.identity.items():
                    if state.get(key) != expected:
                        raise ArtifactError(f"resume identity mismatch: {key}")
                self.assert_resume_integrity()
            else:
                state = {
                "schema_id": "vea.p1.campaign_state.v1",
                **self.identity,
                "state_sequence": list(STATE_SEQUENCE),
                "current_state": STATE_SEQUENCE[0],
                "completed_states": [],
                "published_stage_markers": {},
                "test_unseal_authorization_count": 0,
                "test_unseal_event_count": 0,
                "test_unseal_first_read_started": False,
                "unseal_event_id": None,
                "unseal_binding": None,
                "frozen": False,
                "sealed": False,
                "verified": False,
                }
                atomic_write_json(self.state_path, state)
                append_jsonl(self.events_path, {
                    "schema_id": "vea.p1.campaign_event.v1",
                    "event": "campaign_initialized",
                    "state": STATE_SEQUENCE[0],
                    **self.identity,
                })
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        if self._locked:
            self.lock_path.unlink(missing_ok=True)
            self._locked = False

    def __enter__(self) -> "CampaignStore":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()

    def load_state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError("campaign state is unreadable") from exc
        if not isinstance(value, dict) or value.get("state_sequence") != list(STATE_SEQUENCE):
            raise ArtifactError("campaign state schema or sequence mismatch")
        return value

    @property
    def current_state(self) -> str:
        return str(self.load_state()["current_state"])

    def _write_state(self, state: Mapping[str, Any]) -> None:
        atomic_write_json(self.state_path, dict(state), replace=True)

    def begin_stage(self, state_name: str) -> Path | None:
        state = self.load_state()
        completed = tuple(state["completed_states"])
        if state_name in completed:
            marker = self.root / STAGE_DIRECTORIES[state_name] / "COMPLETE.json"
            if not marker.is_file():
                raise ArtifactError("completed stage marker is missing")
            return None
        if state.get("current_state") != state_name:
            raise ArtifactError(f"stage order mismatch: expected {state.get('current_state')}, got {state_name}")
        if state_name == "T007_TEST_UNSEAL" and (
            state.get("test_unseal_authorization_count") != 1
            or not isinstance(state.get("unseal_event_id"), str)
            or not state.get("unseal_event_id")
        ):
            raise ArtifactError("test unseal requires one immutable authorization")
        if state_name == "T007_TEST_UNSEAL" and state["test_unseal_event_count"] == 0:
            if not state.get("frozen"):
                raise ArtifactError("test unseal requires one prior freeze")
            state["test_unseal_event_count"] = 1
            self._write_state(state)
            append_jsonl(self.events_path, {
                "schema_id": "vea.p1.campaign_event.v1",
                "event": "test_unseal",
                "state": state_name,
                "logical_event_count": 1,
                "unseal_event_id": state["unseal_event_id"],
                "config_sha256": self.identity["config_sha256"],
            })
        elif state_name == "T007_TEST_UNSEAL" and state["test_unseal_event_count"] != 1:
            raise ArtifactError("invalid logical test-unseal count")
        final = self.root / STAGE_DIRECTORIES[state_name]
        if final.exists():
            raise ArtifactError("incomplete stage output collides with canonical path")
        temporary = self.root / f".{STAGE_DIRECTORIES[state_name]}.in_progress"
        if temporary.exists():
            evidence = self.root / "logs" / "interrupted_stages"
            evidence.mkdir(parents=True, exist_ok=True)
            target = evidence / f"{STAGE_DIRECTORIES[state_name]}-{len(tuple(evidence.iterdir())):04d}"
            os.replace(temporary, target)
        temporary.mkdir()
        append_jsonl(self.events_path, {
            "schema_id": "vea.p1.campaign_event.v1",
            "event": "stage_started",
            "state": state_name,
            "config_sha256": self.identity["config_sha256"],
        })
        return temporary

    def authorize_test_unseal(self, binding: Mapping[str, Any]) -> str:
        """Create or verify the one immutable authorization bound to frozen state."""

        state = self.load_state()
        if state.get("current_state") != "T007_TEST_UNSEAL" or not state.get("frozen"):
            raise ArtifactError("test unseal authorization requires the frozen T007 boundary")
        required = {
            "analysis_commit", "analysis_tree", "analysis_config_sha256",
            "input_checksum_manifest_sha256", "freeze_sha256",
            "fitted_state_manifest_sha256",
        }
        document = dict(binding)
        if set(document) != required or any(not isinstance(document[key], str) or not document[key] for key in required):
            raise ArtifactError("test unseal binding fields are invalid")
        event_id = hashlib.sha256(canonical_json_bytes({
            "schema_id": "vea.p1.test_unseal_binding.v1",
            **document,
        })).hexdigest()
        authorization_count = state.get("test_unseal_authorization_count", 0)
        if authorization_count == 1:
            if state.get("unseal_event_id") != event_id or state.get("unseal_binding") != document:
                raise ArtifactError("test unseal authorization identity changed")
            return event_id
        if authorization_count != 0 or state.get("unseal_event_id") is not None or state.get("unseal_binding") is not None:
            raise ArtifactError("invalid test unseal authorization state")
        state["test_unseal_authorization_count"] = 1
        state["unseal_event_id"] = event_id
        state["unseal_binding"] = document
        self._write_state(state)
        append_jsonl(self.events_path, {
            "schema_id": "vea.p1.campaign_event.v1",
            "event": "test_unseal_authorized",
            "state": "T007_TEST_UNSEAL",
            "authorization_count": 1,
            "unseal_event_id": event_id,
            "binding": document,
        })
        return event_id

    def mark_test_read_started(self, event_id: str) -> None:
        """Bind the first decoded test read to the sole authorized event."""

        state = self.load_state()
        if (
            state.get("current_state") != "T007_TEST_UNSEAL"
            or state.get("test_unseal_authorization_count") != 1
            or state.get("test_unseal_event_count") != 1
            or state.get("unseal_event_id") != event_id
        ):
            raise ArtifactError("test read is not bound to the authorized unseal event")
        if state.get("test_unseal_first_read_started"):
            return
        state["test_unseal_first_read_started"] = True
        self._write_state(state)
        append_jsonl(self.events_path, {
            "schema_id": "vea.p1.campaign_event.v1",
            "event": "test_unseal_first_value_read",
            "state": "T007_TEST_UNSEAL",
            "logical_event_count": 1,
            "unseal_event_id": event_id,
        })

    @property
    def unseal_event_id(self) -> str | None:
        value = self.load_state().get("unseal_event_id")
        return value if isinstance(value, str) and value else None

    def complete_stage(self, state_name: str, temporary: Path, summary: Mapping[str, Any]) -> Path:
        state = self.load_state()
        if state["current_state"] != state_name or state_name in state["completed_states"]:
            raise ArtifactError("stage completion is stale or duplicated")
        output_members = [
            {
                "path": path.relative_to(temporary).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(temporary.rglob("*"), key=lambda item: item.relative_to(temporary).as_posix())
            if path.is_file()
        ]
        marker = {
            "schema_id": "vea.p1.stage_completion.v1",
            "state": state_name,
            "config_sha256": self.identity["config_sha256"],
            "output_members": output_members,
            "summary": dict(summary),
        }
        marker_path = atomic_write_json(temporary / "COMPLETE.json", marker)
        marker_sha = sha256_file(marker_path)
        final = self.root / STAGE_DIRECTORIES[state_name]
        if final.exists():
            raise ArtifactError("stage output collision")
        os.replace(temporary, final)
        completed = list(state["completed_states"])
        completed.append(state_name)
        index = STATE_SEQUENCE.index(state_name)
        state["completed_states"] = completed
        published = dict(state.get("published_stage_markers", {}))
        published[state_name] = marker_sha
        state["published_stage_markers"] = published
        if state_name == "FREEZE":
            state["frozen"] = True
        if state_name == "T007_TEST_UNSEAL" and (
            state["test_unseal_event_count"] != 1
            or not state.get("test_unseal_first_read_started")
        ):
            raise ArtifactError("test stage completed without one bound logical unseal")
        if state_name == "SEAL":
            state["sealed"] = True
        if state_name == "VERIFY":
            state["verified"] = True
        state["current_state"] = STATE_SEQUENCE[index + 1] if index + 1 < len(STATE_SEQUENCE) else "COMPLETE"
        self._write_state(state)
        append_jsonl(self.events_path, {
            "schema_id": "vea.p1.campaign_event.v1",
            "event": "stage_completed",
            "state": state_name,
            "next_state": state["current_state"],
            "config_sha256": self.identity["config_sha256"],
        })
        return final

    def assert_resume_integrity(self) -> None:
        state = self.load_state()
        if state.get("test_unseal_authorization_count", 0) not in (0, 1):
            raise ArtifactError("invalid test-unseal authorization count")
        if state["test_unseal_event_count"] not in (0, 1):
            raise ArtifactError("invalid test-unseal event count")
        if state["test_unseal_event_count"] > state.get("test_unseal_authorization_count", 0):
            raise ArtifactError("test unseal occurred without authorization")
        if state.get("test_unseal_first_read_started") and state["test_unseal_event_count"] != 1:
            raise ArtifactError("test value read occurred without one unseal")
        if state.get("test_unseal_authorization_count", 0) == 1:
            if not isinstance(state.get("unseal_event_id"), str) or not isinstance(state.get("unseal_binding"), dict):
                raise ArtifactError("test-unseal authorization identity is missing")
        elif state.get("unseal_event_id") is not None or state.get("unseal_binding") is not None:
            raise ArtifactError("unauthorized test-unseal identity is present")
        completed = tuple(state["completed_states"])
        if completed != STATE_SEQUENCE[: len(completed)]:
            raise ArtifactError("non-monotonic completed-stage sequence")
        published = state.get("published_stage_markers")
        if not isinstance(published, dict) or set(published) != set(completed):
            raise ArtifactError("published-stage marker inventory mismatch")
        for state_name in completed:
            directory = self.root / STAGE_DIRECTORIES[state_name]
            marker_path = directory / "COMPLETE.json"
            if not marker_path.is_file() or sha256_file(marker_path) != published[state_name]:
                raise ArtifactError(f"completed stage marker changed: {state_name}")
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"completed stage marker unreadable: {state_name}") from exc
            if marker.get("state") != state_name or marker.get("config_sha256") != self.identity["config_sha256"]:
                raise ArtifactError(f"completed stage marker identity mismatch: {state_name}")
            expected = marker.get("output_members")
            if not isinstance(expected, list):
                raise ArtifactError(f"completed stage manifest missing: {state_name}")
            actual_paths = {
                path.relative_to(directory).as_posix()
                for path in directory.rglob("*")
                if path.is_file() and path.name != "COMPLETE.json"
            }
            if actual_paths != {item.get("path") for item in expected if isinstance(item, dict)}:
                raise ArtifactError(f"completed stage output inventory changed: {state_name}")
            for item in expected:
                path = directory.joinpath(*Path(item["path"]).parts)
                if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                    raise ArtifactError(f"completed stage output changed: {state_name}:{item['path']}")
        if state["test_unseal_event_count"] == 1 and "FREEZE" not in completed:
            raise ArtifactError("test unseal occurred without freeze")
        event_count = 0
        authorization_count = 0
        first_read_count = 0
        if self.events_path.exists():
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                event_count += value.get("event") == "test_unseal"
                authorization_count += value.get("event") == "test_unseal_authorized"
                first_read_count += value.get("event") == "test_unseal_first_value_read"
        if event_count != state["test_unseal_event_count"]:
            raise ArtifactError("logical test-unseal ledger mismatch")
        if authorization_count != state.get("test_unseal_authorization_count", 0):
            raise ArtifactError("test-unseal authorization ledger mismatch")
        if first_read_count != int(bool(state.get("test_unseal_first_read_started"))):
            raise ArtifactError("test-unseal first-read ledger mismatch")


def _covered_files(root: Path, excluded: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactError("payload contains a symbolic link")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def build_payload_manifest(payload_root: Path) -> dict[str, Any]:
    excluded = {"payload_manifest.json", "seal.json"}
    members = [
        {
            "path": path.relative_to(payload_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _covered_files(payload_root, excluded)
    ]
    return {
        "schema_id": "vea.p1.payload_manifest.v1",
        "excluded_members": sorted(excluded),
        "covered_member_count": len(members),
        "members": members,
    }


def _tar_bytes(root: Path, *, fixed_mtime: int) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=fixed_mtime, compresslevel=9) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
            paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
            for path in paths:
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    raise ArtifactError("archive source contains a symbolic link")
                info = tarfile.TarInfo(relative + ("/" if path.is_dir() else ""))
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = fixed_mtime
                if path.is_dir():
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    info.size = 0
                    archive.addfile(info)
                elif path.is_file():
                    info.type = tarfile.REGTYPE
                    info.mode = 0o644
                    info.size = path.stat().st_size
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
    return buffer.getvalue()


def seal_payload(
    payload_root: str | Path,
    archive_path: str | Path,
    *,
    code_commit: str,
    code_tree: str,
    config_sha256: str,
    input_checksum_manifest_sha256: str,
    fixed_mtime: int,
) -> dict[str, str | int]:
    root = Path(payload_root)
    if (root / "payload_manifest.json").exists() or (root / "seal.json").exists():
        raise ArtifactError("payload metadata collision")
    manifest = build_payload_manifest(root)
    manifest_path = atomic_write_json(root / "payload_manifest.json", manifest)
    manifest_sha = sha256_file(manifest_path)
    seal = {
        "schema_id": "vea.p1.internal_seal.v1",
        "payload_manifest_sha256": manifest_sha,
        "code_commit": code_commit,
        "code_tree": code_tree,
        "analysis_config_sha256": config_sha256,
        "input_checksum_manifest_sha256": input_checksum_manifest_sha256,
        "outer_archive_sha256_included": False,
    }
    atomic_write_json(root / "seal.json", seal)
    payload = _tar_bytes(root, fixed_mtime=fixed_mtime)
    archive = atomic_write_bytes(archive_path, payload)
    return {
        "payload_manifest_sha256": manifest_sha,
        "sealed_package_sha256": sha256_file(archive),
        "covered_member_count": int(manifest["covered_member_count"]),
    }


def _verify_sealed_bytes(payload: bytes) -> dict[str, str | int]:
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ArtifactError("sealed package contains duplicate members")
            for member in members:
                if member.issym() or member.islnk() or member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ArtifactError("sealed package contains an unsafe member")
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ArtifactError("sealed package member is unreadable")
                    files[member.name] = stream.read()
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ArtifactError("sealed package archive is unreadable") from exc
    try:
        manifest = json.loads(files["payload_manifest.json"])
        seal = json.loads(files["seal.json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ArtifactError("sealed package metadata is missing or invalid") from exc
    canonical_manifest = canonical_json_bytes(manifest)
    manifest_sha = hashlib.sha256(canonical_manifest).hexdigest()
    if files["payload_manifest.json"] != canonical_manifest:
        raise ArtifactError("payload manifest is not canonical")
    if seal.get("payload_manifest_sha256") != manifest_sha:
        raise ArtifactError("internal seal pointer mismatch")
    if manifest.get("excluded_members") != ["payload_manifest.json", "seal.json"]:
        raise ArtifactError("payload exclusions mismatch")
    expected_paths = {item["path"] for item in manifest.get("members", [])}
    actual_covered = set(files) - {"payload_manifest.json", "seal.json"}
    if expected_paths != actual_covered:
        raise ArtifactError("payload member inventory mismatch")
    for item in manifest["members"]:
        member_payload = files[item["path"]]
        if len(member_payload) != item["bytes"] or hashlib.sha256(member_payload).hexdigest() != item["sha256"]:
            raise ArtifactError(f"payload member checksum mismatch: {item['path']}")
    archive_sha = hashlib.sha256(payload).hexdigest()
    if archive_sha.encode("ascii") in b"".join(files.values()):
        raise ArtifactError("outer package hash is self-referential")
    return {
        "sealed_package_sha256": archive_sha,
        "payload_manifest_sha256": manifest_sha,
        "covered_member_count": len(expected_paths),
    }


def verify_sealed_package(path: str | Path) -> dict[str, str | int]:
    archive_path = Path(path)
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ArtifactError("sealed package is not a regular file")
    return _verify_sealed_bytes(archive_path.read_bytes())


_REVIEW_MEMBERS = (
    "WINDOWS_VERIFY.txt",
    "delivery_summary.json",
    "sealed_package.sha256",
    "sealed_package.tar.gz",
)


def build_review_bundle(result_root: str | Path, bundle_path: str | Path) -> dict[str, str | int]:
    """Build a deterministic review ZIP embedding the independently sealed package."""

    root = Path(result_root)
    bundle = Path(bundle_path)
    if bundle.exists():
        raise ArtifactError("review bundle collision")
    members: list[dict[str, str | int]] = []
    payloads: dict[str, bytes] = {}
    for name in _REVIEW_MEMBERS:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ArtifactError(f"review bundle member is missing or unsafe: {name}")
        payload = path.read_bytes()
        payloads[name] = payload
        members.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "schema_id": "vea.p1.review_bundle_manifest.v1",
        "excluded_members": ["review_bundle_manifest.json"],
        "covered_member_count": len(members),
        "members": members,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    payloads["review_bundle_manifest.json"] = manifest_bytes
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payloads[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    bundle_bytes = buffer.getvalue()
    atomic_write_bytes(bundle, bundle_bytes)
    return {
        "review_bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "review_bundle_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "review_bundle_covered_member_count": len(members),
    }


def verify_review_bundle(path: str | Path) -> dict[str, str | int]:
    bundle = Path(path)
    if not bundle.is_file() or bundle.is_symlink():
        raise ArtifactError("review bundle is not a regular file")
    bundle_bytes = bundle.read_bytes()
    files: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(bundle_bytes), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ArtifactError("review bundle contains duplicate members")
            for info in infos:
                if info.is_dir() or info.filename.startswith("/") or ".." in Path(info.filename).parts:
                    raise ArtifactError("review bundle contains an unsafe member")
                if info.date_time != (1980, 1, 1, 0, 0, 0) or info.compress_type != zipfile.ZIP_DEFLATED:
                    raise ArtifactError("review bundle metadata is not deterministic")
                files[info.filename] = archive.read(info)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ArtifactError("review bundle archive is unreadable") from exc
    try:
        manifest = json.loads(files["review_bundle_manifest.json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ArtifactError("review bundle manifest is missing or invalid") from exc
    canonical = canonical_json_bytes(manifest)
    if files["review_bundle_manifest.json"] != canonical:
        raise ArtifactError("review bundle manifest is not canonical")
    if manifest.get("excluded_members") != ["review_bundle_manifest.json"]:
        raise ArtifactError("review bundle exclusions mismatch")
    expected = {item["path"] for item in manifest.get("members", [])}
    if expected != set(_REVIEW_MEMBERS) or set(files) != expected | {"review_bundle_manifest.json"}:
        raise ArtifactError("review bundle member inventory mismatch")
    for item in manifest["members"]:
        payload = files[item["path"]]
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ArtifactError(f"review bundle member checksum mismatch: {item['path']}")
    package = _verify_sealed_bytes(files["sealed_package.tar.gz"])
    package_line = files["sealed_package.sha256"].decode("ascii")
    if package_line != f"{package['sealed_package_sha256']}  sealed_package.tar.gz\n":
        raise ArtifactError("review bundle package checksum file mismatch")
    try:
        delivery = json.loads(files["delivery_summary.json"])
    except json.JSONDecodeError as exc:
        raise ArtifactError("review bundle delivery summary is invalid") from exc
    if delivery.get("sealed_package_sha256") != package["sealed_package_sha256"]:
        raise ArtifactError("review bundle delivery/package identity mismatch")
    bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()
    if bundle_sha.encode("ascii") in b"".join(files.values()):
        raise ArtifactError("review bundle hash is self-referential")
    return {
        **package,
        "review_bundle_sha256": bundle_sha,
        "review_bundle_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "review_bundle_covered_member_count": len(expected),
    }


__all__ = [
    "ArtifactError", "CampaignStore", "STAGE_DIRECTORIES", "STATE_SEQUENCE", "append_jsonl",
    "atomic_write_bytes", "atomic_write_json", "build_payload_manifest",
    "build_review_bundle", "deterministic_result_name", "seal_payload",
    "verify_review_bundle", "verify_sealed_package",
]
