from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vision_encoder_alignment.snapshot_boundary import (
    APPLE_HF_SNAPSHOT_LAYOUT_ID,
    MODEL_ID,
    REVISION,
    SnapshotBoundaryError,
    validate_apple_hf_snapshot,
    canonical_json_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
APPLE_PROFILE = ROOT / "configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json"
CANONICAL_APPLE_SNAPSHOT_SHA256 = "d1326e9358930d826eaae6c18b44e384fa38b854140304951210347d7ecdac0d"


def _identity(name: str, payload: bytes) -> dict[str, object]:
    return {
        "bytes": len(payload),
        "filename": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _snapshot(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    cache = tmp_path / "cache"
    root = cache / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / REVISION
    root.mkdir(parents=True)
    payloads = {
        "config.json": b"config",
        "preprocessor_config.json": b"preprocessor",
        "model.safetensors": b"weights",
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return cache, root, [_identity(name, payload) for name, payload in payloads.items()]


def test_exact_huggingface_cache_snapshot_is_validated(tmp_path: Path) -> None:
    cache, root, files = _snapshot(tmp_path)
    result = validate_apple_hf_snapshot(
        root, cache_root=cache, model_id=MODEL_ID, revision=REVISION, files=files
    )
    assert result.root == root.resolve()
    assert APPLE_HF_SNAPSHOT_LAYOUT_ID == "huggingface_cache_snapshot_v1"
    assert len(result.files) == 3


def test_live_apple_profile_pins_the_validator_canonical_snapshot_identity() -> None:
    profile = json.loads(APPLE_PROFILE.read_text(encoding="utf-8"))
    identity_document = {
        "files": profile["checkpoint_files"],
        "layout": APPLE_HF_SNAPSHOT_LAYOUT_ID,
        "model_id": profile["model"]["model_id"],
        "revision": profile["model"]["revision"],
    }
    actual = hashlib.sha256(canonical_json_bytes(identity_document)).hexdigest()
    assert actual == CANONICAL_APPLE_SNAPSHOT_SHA256
    assert profile["identities"]["external_snapshot_manifest_sha256"] == actual


def test_contained_leaf_link_is_allowed_but_escape_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, root, files = _snapshot(tmp_path)
    leaf = root / "config.json"
    blobs = cache / "blobs"
    blobs.mkdir()
    target = blobs / "config"
    target.write_bytes(b"config")
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve
    monkeypatch.setattr(
        Path, "is_symlink", lambda self: self == leaf or original_is_symlink(self)
    )
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, *args, **kwargs: target
        if self == leaf
        else original_resolve(self, *args, **kwargs),
    )
    validate_apple_hf_snapshot(
        root, cache_root=cache, model_id=MODEL_ID, revision=REVISION, files=files
    )
    outside = tmp_path / "outside"
    outside.write_bytes(b"config")
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, *args, **kwargs: outside
        if self == leaf
        else original_resolve(self, *args, **kwargs),
    )
    with pytest.raises(SnapshotBoundaryError, match="approved cache"):
        validate_apple_hf_snapshot(
            root, cache_root=cache, model_id=MODEL_ID, revision=REVISION, files=files
        )


def test_missing_size_hash_and_broken_link_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, root, files = _snapshot(tmp_path)
    (root / "config.json").unlink()
    with pytest.raises(SnapshotBoundaryError):
        validate_apple_hf_snapshot(
            root, cache_root=cache, model_id=MODEL_ID, revision=REVISION, files=files
        )
    (root / "config.json").write_bytes(b"wrong-size")
    with pytest.raises(SnapshotBoundaryError):
        validate_apple_hf_snapshot(
            root, cache_root=cache, model_id=MODEL_ID, revision=REVISION, files=files
        )
    files[0] = {**files[0], "bytes": len(b"wrong-size")}
    with pytest.raises(SnapshotBoundaryError):
        validate_apple_hf_snapshot(
            root, cache_root=cache, model_id=MODEL_ID, revision=REVISION, files=files
        )
    leaf = root / "config.json"
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve
    monkeypatch.setattr(
        Path, "is_symlink", lambda self: self == leaf or original_is_symlink(self)
    )
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError())
        if self == leaf
        else original_resolve(self, *args, **kwargs),
    )
    with pytest.raises(SnapshotBoundaryError):
        validate_apple_hf_snapshot(
            root, cache_root=cache, model_id=MODEL_ID, revision=REVISION, files=files
        )


def test_wrong_model_or_revision_path_is_rejected(tmp_path: Path) -> None:
    cache, root, files = _snapshot(tmp_path)
    for model_id, revision in (("wrong/model", REVISION), (MODEL_ID, "0" * 40)):
        with pytest.raises(SnapshotBoundaryError, match="model or revision"):
            validate_apple_hf_snapshot(
                root, cache_root=cache, model_id=model_id, revision=revision, files=files
            )
