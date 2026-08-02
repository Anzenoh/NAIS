"""Unit tests for the strict, import-light DINOv3 snapshot boundary."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import vision_encoder_alignment.snapshot_boundary as boundary
from vision_encoder_alignment.snapshot_boundary import (
    MODEL_ID,
    REVISION,
    SNAPSHOT_ID,
    SNAPSHOT_MANIFEST_SCHEMA_ID,
    SNAPSHOT_PROFILE_SCHEMA_ID,
    SnapshotBoundaryError,
    SnapshotFileIdentity,
    SnapshotManifest,
    SnapshotProfile,
    canonical_json_bytes,
    load_snapshot_manifest,
    load_snapshot_profile,
    resolve_default_snapshot_root,
    snapshot_files_directory,
    validate_snapshot_manifest,
    validate_snapshot_root,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _profile_path(repository_root: Path) -> Path:
    return (
        repository_root
        / "configs"
        / "model_snapshots"
        / "dinov3_vit_b16_r5931719_v0.json"
    )


def _profile_value(repository_root: Path) -> dict[str, Any]:
    return json.loads(_profile_path(repository_root).read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _assert_classification(
    error: pytest.ExceptionInfo[SnapshotBoundaryError], expected: str
) -> None:
    assert error.value.classification == expected


@dataclass(frozen=True, slots=True)
class FakeSnapshot:
    repository: Path
    persistent_runtime: Path
    snapshot_root: Path
    profile: SnapshotProfile
    payloads: dict[str, bytes]

    @property
    def manifest_path(self) -> Path:
        return self.snapshot_root / "snapshot-manifest.json"


def _manifest_value(profile: SnapshotProfile) -> dict[str, Any]:
    return {
        "credential_backed_logical_request_count": 2,
        "files": [identity.to_dict() for identity in profile.files],
        "integrity_status": "VERIFIED",
        "model_id": profile.model_id,
        "model_weight_transferred_bytes": profile.weight_file.size_bytes,
        "network_independent_verification_status": "VERIFIED",
        "provisioned_at_utc": "2026-07-21T00:00:00Z",
        "repository_profile_sha256": profile.source_sha256,
        "revision": profile.revision,
        "schema_id": SNAPSHOT_MANIFEST_SCHEMA_ID,
        "snapshot_id": profile.snapshot_id,
        "total_bytes": profile.selected_total_bytes,
        "weight_format": profile.weight_format,
    }


def _prepare_fake_snapshot(tmp_path: Path, repository_root: Path) -> FakeSnapshot:
    profile = load_snapshot_profile(_profile_path(repository_root).resolve())
    payloads = {
        "config.json": b'{"model_type":"dinov3_vit"}\n',
        "preprocessor_config.json": b'{"do_resize":true}\n',
        "model.safetensors": b"small-safe-fake-weight-payload\n",
    }
    identities = tuple(
        SnapshotFileIdentity(
            filename=filename,
            size_bytes=len(payload),
            sha256=_digest(payload),
        )
        for filename, payload in payloads.items()
    )
    fake_profile = replace(
        profile,
        files=identities,
        selected_total_bytes=sum(item.size_bytes for item in identities),
    )

    repository = (tmp_path / "repository").resolve()
    persistent_runtime = (tmp_path / "runtimes" / fake_profile.runtime_id).resolve()
    snapshot_root = (tmp_path / "snapshots" / fake_profile.snapshot_id).resolve()
    repository.mkdir(parents=True)
    persistent_runtime.mkdir(parents=True)
    files_directory = snapshot_root / "files"
    files_directory.mkdir(parents=True)
    for filename, payload in payloads.items():
        (files_directory / filename).write_bytes(payload)
    _write_json(snapshot_root / "snapshot-manifest.json", _manifest_value(fake_profile))
    return FakeSnapshot(
        repository=repository,
        persistent_runtime=persistent_runtime,
        snapshot_root=snapshot_root,
        profile=fake_profile,
        payloads=payloads,
    )


def _inventory(root: Path) -> tuple[tuple[str, str, int, int], ...]:
    rows: list[tuple[str, str, int, int]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            file_stat = path.stat()
            rows.append((relative, _digest(path.read_bytes()), file_stat.st_size, file_stat.st_mtime_ns))
        else:
            rows.append((relative + "/", "directory", 0, path.stat().st_mtime_ns))
    return tuple(rows)


def test_valid_profile_round_trip_canonical_identity_and_immutability(
    repository_root: Path,
) -> None:
    path = _profile_path(repository_root).resolve()
    profile = load_snapshot_profile(path)

    assert profile.schema_id == SNAPSHOT_PROFILE_SCHEMA_ID
    assert profile.snapshot_id == SNAPSHOT_ID
    assert profile.model_id == MODEL_ID
    assert profile.revision == REVISION
    assert profile.weight_file.filename == "model.safetensors"
    assert profile.weight_file.size_bytes == 342_662_192
    assert profile.selected_total_bytes == 342_663_521
    assert profile.to_dict() == json.loads(path.read_text(encoding="utf-8"))
    assert path.read_bytes() == canonical_json_bytes(profile.to_dict())

    with pytest.raises(FrozenInstanceError):
        profile.snapshot_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        profile.load_policy["cpu_only"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.external_root["relative_parts"][0] = "changed"  # type: ignore[index]


@pytest.mark.parametrize("field", ["schema_id", "files", "persistent_runtime"])
def test_profile_missing_field_rejected(
    repository_root: Path, field: str
) -> None:
    value = _profile_value(repository_root)
    del value[field]
    with pytest.raises(SnapshotBoundaryError) as error:
        SnapshotProfile.from_dict(value)
    _assert_classification(error, "schema_validation_failed")


def test_profile_unknown_top_level_and_nested_fields_rejected(
    repository_root: Path,
) -> None:
    top_level = _profile_value(repository_root)
    top_level["unexpected"] = True
    with pytest.raises(SnapshotBoundaryError, match="unknown fields"):
        SnapshotProfile.from_dict(top_level)

    nested = _profile_value(repository_root)
    nested["model"]["unexpected"] = True
    with pytest.raises(SnapshotBoundaryError, match="unknown fields"):
        SnapshotProfile.from_dict(nested)


def test_profile_loader_rejects_duplicate_keys_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    duplicate = (tmp_path / "duplicate.json").resolve()
    duplicate.write_text('{"schema_id":"a","schema_id":"b"}', encoding="utf-8")
    with pytest.raises(SnapshotBoundaryError) as duplicate_error:
        load_snapshot_profile(duplicate)
    _assert_classification(duplicate_error, "invalid_json")

    nonfinite = (tmp_path / "nonfinite.json").resolve()
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(SnapshotBoundaryError) as nonfinite_error:
        load_snapshot_profile(nonfinite)
    _assert_classification(nonfinite_error, "invalid_json")


def test_profile_wrong_schema_model_and_revision_rejected(
    repository_root: Path,
) -> None:
    cases = (
        ("schema_id", "vea.wrong.v0"),
        ("model.model_id", "other/model"),
        ("model.revision", "f" * 40),
    )
    for path, wrong_value in cases:
        value = _profile_value(repository_root)
        if path == "schema_id":
            value[path] = wrong_value
        else:
            _, field = path.split(".")
            value["model"][field] = wrong_value
        with pytest.raises(SnapshotBoundaryError) as error:
            SnapshotProfile.from_dict(value)
        _assert_classification(error, "snapshot_identity_mismatch")


@pytest.mark.parametrize("revision", ["main", "refs/heads/main", "latest", "5931719"])
def test_mutable_or_abbreviated_revision_rejected(
    repository_root: Path, revision: str
) -> None:
    value = _profile_value(repository_root)
    value["model"]["revision"] = revision
    with pytest.raises(SnapshotBoundaryError, match="immutable 40-hex revision"):
        SnapshotProfile.from_dict(value)


@pytest.mark.parametrize("kind", ["size", "sha256"])
def test_wrong_selected_weight_identity_rejected(
    repository_root: Path, kind: str
) -> None:
    value = _profile_value(repository_root)
    weight = value["files"][2]
    if kind == "size":
        weight["bytes"] += 1
        value["selected_total_bytes"] += 1
    else:
        weight["sha256"] = "f" * 64
    with pytest.raises(SnapshotBoundaryError) as error:
        SnapshotProfile.from_dict(value)
    _assert_classification(error, "snapshot_identity_mismatch")


def test_default_root_planning_is_exact_and_side_effect_free(
    tmp_path: Path, repository_root: Path
) -> None:
    profile = load_snapshot_profile(_profile_path(repository_root).resolve())
    base = (tmp_path / "local-app-data").resolve()
    before = tuple(tmp_path.rglob("*"))
    result = resolve_default_snapshot_root(
        profile,
        environ={"LOCALAPPDATA": str(base)},
    )
    after = tuple(tmp_path.rglob("*"))
    assert result == base / "VEA" / "snapshots" / profile.snapshot_id
    assert before == after
    assert not result.exists()

    with pytest.raises(SnapshotBoundaryError) as missing:
        resolve_default_snapshot_root(profile, environ={})
    _assert_classification(missing, "snapshot_root_unavailable")
    with pytest.raises(SnapshotBoundaryError) as relative:
        resolve_default_snapshot_root(profile, environ={"LOCALAPPDATA": "relative"})
    _assert_classification(relative, "unsafe_path")


@pytest.mark.parametrize("overlap", ["repository", "runtime"])
def test_snapshot_root_overlap_rejected(
    tmp_path: Path, repository_root: Path, overlap: str
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    parent = fake.repository if overlap == "repository" else fake.persistent_runtime
    overlapping_root = parent / fake.profile.snapshot_id
    fake.snapshot_root.rename(overlapping_root)
    with pytest.raises(SnapshotBoundaryError) as error:
        validate_snapshot_root(
            overlapping_root.resolve(),
            fake.profile,
            repository_root=fake.repository,
            persistent_runtime_root=fake.persistent_runtime,
        )
    _assert_classification(error, "unsafe_snapshot_root")


def test_root_basename_must_match_snapshot_id(
    tmp_path: Path, repository_root: Path
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    wrong = fake.snapshot_root.with_name("wrong-snapshot")
    fake.snapshot_root.rename(wrong)
    with pytest.raises(SnapshotBoundaryError) as error:
        validate_snapshot_root(
            wrong.resolve(),
            fake.profile,
            repository_root=fake.repository,
            persistent_runtime_root=fake.persistent_runtime,
        )
    _assert_classification(error, "snapshot_identity_mismatch")


def test_symlink_and_reparse_paths_rejected(
    tmp_path: Path,
    repository_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    weight = fake.snapshot_root / "files" / "model.safetensors"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(fake.payloads["model.safetensors"])
    weight.unlink()
    try:
        weight.symlink_to(outside)
    except OSError:
        pytest.skip("file symlink creation is unavailable")
    with pytest.raises(SnapshotBoundaryError) as symlink_error:
        validate_snapshot_root(
            fake.snapshot_root,
            fake.profile,
            repository_root=fake.repository,
            persistent_runtime_root=fake.persistent_runtime,
        )
    _assert_classification(symlink_error, "unsafe_path")

    weight.unlink()
    weight.write_bytes(fake.payloads["model.safetensors"])
    original = boundary._is_reparse_point
    monkeypatch.setattr(
        boundary,
        "_is_reparse_point",
        lambda path: Path(path).name == "files" or original(Path(path)),
    )
    with pytest.raises(SnapshotBoundaryError) as reparse_error:
        snapshot_files_directory(fake.snapshot_root)
    _assert_classification(reparse_error, "unsafe_path")


def test_exact_layout_and_files_directory(
    tmp_path: Path, repository_root: Path
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    manifest = validate_snapshot_root(
        fake.snapshot_root,
        fake.profile,
        repository_root=fake.repository,
        persistent_runtime_root=fake.persistent_runtime,
    )
    assert manifest.schema_id == SNAPSHOT_MANIFEST_SCHEMA_ID
    assert snapshot_files_directory(fake.snapshot_root) == fake.snapshot_root / "files"


@pytest.mark.parametrize("location", ["root", "files"])
def test_extra_file_rejected(
    tmp_path: Path, repository_root: Path, location: str
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    directory = fake.snapshot_root if location == "root" else fake.snapshot_root / "files"
    (directory / "residue.tmp").write_bytes(b"partial")
    with pytest.raises(SnapshotBoundaryError) as error:
        validate_snapshot_root(
            fake.snapshot_root,
            fake.profile,
            repository_root=fake.repository,
            persistent_runtime_root=fake.persistent_runtime,
        )
    _assert_classification(error, "snapshot_layout_mismatch")


@pytest.mark.parametrize("missing", ["manifest", "config", "weight"])
def test_missing_or_partial_snapshot_fails_closed(
    tmp_path: Path, repository_root: Path, missing: str
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    target = {
        "manifest": fake.manifest_path,
        "config": fake.snapshot_root / "files" / "config.json",
        "weight": fake.snapshot_root / "files" / "model.safetensors",
    }[missing]
    target.unlink()
    before = _inventory(fake.snapshot_root)
    with pytest.raises(SnapshotBoundaryError) as error:
        validate_snapshot_root(
            fake.snapshot_root,
            fake.profile,
            repository_root=fake.repository,
            persistent_runtime_root=fake.persistent_runtime,
        )
    _assert_classification(error, "snapshot_layout_mismatch")
    assert _inventory(fake.snapshot_root) == before


@pytest.mark.parametrize("kind", ["size", "sha256"])
def test_snapshot_file_size_or_sha_mismatch_rejected(
    tmp_path: Path, repository_root: Path, kind: str
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    target = fake.snapshot_root / "files" / "model.safetensors"
    if kind == "size":
        target.write_bytes(target.read_bytes() + b"tamper")
    else:
        payload = bytearray(target.read_bytes())
        payload[0] ^= 1
        target.write_bytes(payload)
    with pytest.raises(SnapshotBoundaryError) as error:
        validate_snapshot_root(
            fake.snapshot_root,
            fake.profile,
            repository_root=fake.repository,
            persistent_runtime_root=fake.persistent_runtime,
        )
    _assert_classification(error, "snapshot_integrity_failed")


def test_manifest_missing_unknown_status_duplicate_and_nonfinite_rejected(
    tmp_path: Path, repository_root: Path
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    original = _manifest_value(fake.profile)

    missing = copy.deepcopy(original)
    del missing["integrity_status"]
    with pytest.raises(SnapshotBoundaryError, match="missing required fields"):
        SnapshotManifest.from_dict(missing)

    unknown = copy.deepcopy(original)
    unknown["unexpected"] = True
    with pytest.raises(SnapshotBoundaryError, match="unknown fields"):
        SnapshotManifest.from_dict(unknown)

    invalid_status = copy.deepcopy(original)
    invalid_status["integrity_status"] = "PENDING"
    with pytest.raises(SnapshotBoundaryError) as status_error:
        SnapshotManifest.from_dict(invalid_status)
    _assert_classification(status_error, "snapshot_identity_mismatch")

    duplicate = (tmp_path / "duplicate-manifest.json").resolve()
    duplicate.write_text('{"schema_id":"a","schema_id":"b"}', encoding="utf-8")
    with pytest.raises(SnapshotBoundaryError) as duplicate_error:
        load_snapshot_manifest(duplicate)
    _assert_classification(duplicate_error, "invalid_json")

    nonfinite = (tmp_path / "nonfinite-manifest.json").resolve()
    nonfinite.write_text('{"value":Infinity}', encoding="utf-8")
    with pytest.raises(SnapshotBoundaryError) as nonfinite_error:
        load_snapshot_manifest(nonfinite)
    _assert_classification(nonfinite_error, "invalid_json")


def test_manifest_profile_mismatch_rejected(
    tmp_path: Path, repository_root: Path
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    mismatch = _manifest_value(fake.profile)
    mismatch["repository_profile_sha256"] = "f" * 64
    manifest = SnapshotManifest.from_dict(mismatch)
    with pytest.raises(SnapshotBoundaryError) as error:
        validate_snapshot_manifest(manifest, fake.profile)
    _assert_classification(error, "snapshot_identity_mismatch")


@pytest.mark.parametrize(
    "sensitive",
    ["hf_abcdefghijklmnopqrstuvwxyz", "person@example.com", r"C:\\Users\\person\\file"],
)
def test_sensitive_manifest_value_rejected(
    tmp_path: Path, repository_root: Path, sensitive: str
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    value = _manifest_value(fake.profile)
    value["model_id"] = sensitive
    with pytest.raises(SnapshotBoundaryError) as error:
        SnapshotManifest.from_dict(value)
    _assert_classification(error, "sensitive_data_rejected")


def test_verify_is_read_only_and_exact_valid_root_is_idempotent(
    tmp_path: Path, repository_root: Path
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    before = _inventory(fake.snapshot_root)
    first = validate_snapshot_root(
        fake.snapshot_root,
        fake.profile,
        repository_root=fake.repository,
        persistent_runtime_root=fake.persistent_runtime,
    )
    middle = _inventory(fake.snapshot_root)
    second = validate_snapshot_root(
        fake.snapshot_root,
        fake.profile,
        repository_root=fake.repository,
        persistent_runtime_root=fake.persistent_runtime,
    )
    after = _inventory(fake.snapshot_root)
    assert first.source_sha256 == second.source_sha256
    assert before == middle == after


def test_owned_staging_validation_is_explicit_and_read_only(
    tmp_path: Path, repository_root: Path
) -> None:
    fake = _prepare_fake_snapshot(tmp_path, repository_root)
    staging = fake.snapshot_root.with_name(fake.profile.staging_name_prefix + "unit")
    fake.snapshot_root.rename(staging)
    before = _inventory(staging)

    with pytest.raises(SnapshotBoundaryError) as public_error:
        validate_snapshot_root(
            staging,
            fake.profile,
            repository_root=fake.repository,
            persistent_runtime_root=fake.persistent_runtime,
        )
    _assert_classification(public_error, "snapshot_identity_mismatch")

    manifest = validate_snapshot_root(
        staging,
        fake.profile,
        repository_root=fake.repository,
        persistent_runtime_root=fake.persistent_runtime,
        _allow_owned_staging=True,
    )
    assert manifest.snapshot_id == fake.profile.snapshot_id
    assert _inventory(staging) == before


def test_snapshot_boundary_import_is_standard_library_only(
    repository_root: Path, tmp_path: Path
) -> None:
    code = (
        "import json,sys;"
        "import vision_encoder_alignment.snapshot_boundary;"
        "heavy={'numpy','torch','transformers','huggingface_hub','safetensors','PIL'};"
        "print(json.dumps(sorted(heavy.intersection(sys.modules))))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env=environment,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
