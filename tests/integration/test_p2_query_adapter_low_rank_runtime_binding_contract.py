"""Synthetic end-to-end contract for the T004C handoff; no real artifacts."""

from __future__ import annotations

import inspect
import hashlib
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_runtime_binding as binding


MIXED_SPLITS = (
    "test", "train", "validation", "train", "test", "train", "train", "validation",
    "test", "train", "validation", "train", "test", "train", "validation", "train",
)
MIXED_IDS = tuple(f"mixed-{index:02d}" for index in range(16))


def _mixed_unit(tmp_path: Path, bank: str = "SIGLIP2", *, compressed: bool = False,
                changed: str | None = None) -> binding._Unit:
    path = tmp_path / f"{bank.lower()}-{changed or 'valid'}.npz"
    patches = np.empty((16, 196, 768), dtype=np.float32)
    for index in range(16):
        patches[index].fill(1000 + index if MIXED_SPLITS[index] == "test" else index + 1)
    if changed == "fortran":
        patches = np.asfortranarray(patches)
    if changed == "dtype":
        patches = patches.astype(np.float64)
    if changed == "shape":
        patches = patches[:, :, :-1]
    ids = np.asarray(MIXED_IDS, dtype="U96")
    if changed == "ids":
        ids[[0, 1]] = ids[[1, 0]]
    arrays = {
        "image_ids": ids, "patches": patches,
        "globals": np.zeros((16, 768), dtype=np.float32),
        "plan_id": np.asarray("a" * 64 if changed != "plan" else "b" * 64, dtype="U64"),
    }
    if bank == "TARGET_CLIP":
        arrays["connectors"] = np.zeros((16, 196, 1536), dtype=np.float32)
    if changed == "extra":
        arrays["rogue"] = np.zeros((1,), dtype=np.float32)
    (np.savez_compressed if compressed else np.savez)(path, **arrays)
    if changed == "encrypted":
        raw = bytearray(path.read_bytes())
        central = raw.index(b"PK\x01\x02")
        raw[central + 8] |= 1
        raw[6] |= 1
        path.write_bytes(raw)
    return binding._Unit(bank, MIXED_IDS, MIXED_SPLITS, path,
                         hashlib.sha256(path.read_bytes()).hexdigest(), "a" * 64)


def _blocked(code: str, action) -> None:
    with pytest.raises(binding.LowRankRuntimeBindingError) as error:
        action()
    assert error.value.classification == code


def test_public_entry_point_has_only_explicit_roots() -> None:
    parameters = list(inspect.signature(binding.bind_low_rank_runtime).parameters)
    assert parameters == ["repository_root", "t001d_run_root", "t002d_validation_root"]
    assert "fixture" not in parameters
    assert binding.SOURCE_ORDER == ("SIGLIP2", "DINOV3")
    assert core.MAX_CHUNK_ROWS == 4096


def test_factory_caps_real_patch_rows_without_opening_any_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = tuple(f"train-{index:02d}" for index in range(21))
    factory = binding.TrainChunkFactory("SIGLIP2", (), (), ids, (196, 768))

    def rows(units):
        value = np.zeros((196, 768), dtype=np.float32)
        for image_id in ids:
            yield image_id, value.copy()

    monkeypatch.setattr(factory, "_rows", rows)
    first = list(factory())
    second = list(factory())
    assert [len(chunk.source_image_ids) for chunk in first] == [20, 1]
    assert [len(chunk.source_image_ids) for chunk in second] == [20, 1]
    assert all(chunk.source.shape[0] * 196 <= 4096 for chunk in first)
    assert first[0].source_image_ids == ids[:20]
    assert first[1].source_image_ids == ids[20:]
    assert all(np.array_equal(a.source, b.source) for a, b in zip(first, second, strict=True))


def test_no_public_test_or_validation_data_api() -> None:
    assert set(binding.BoundLowRankInputs.__dataclass_fields__) == {
        "config", "ols", "train_factories", "provenance"
    }
    assert not hasattr(binding.BoundLowRankInputs, "validation")
    assert not hasattr(binding.BoundLowRankInputs, "test")
    assert not hasattr(binding, "fit_low_rank_rrr")


def test_realistic_mixed_npz_reads_exact_train_ranges_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _mixed_unit(tmp_path)
    target = _mixed_unit(tmp_path, "TARGET_CLIP")
    original = binding._read_exact_at
    forbidden: list[tuple[int, int, Path]] = []
    for unit in (source, target):
        raw = unit.path.read_bytes()
        with zipfile.ZipFile(unit.path) as archive:
            info = archive.getinfo("patches.npy")
            local = struct.unpack_from("<IHHHHHIIIHH", raw, info.header_offset)
            member_start = info.header_offset + 30 + local[-2] + local[-1]
            assert raw[member_start:member_start + 6] == b"\x93NUMPY"
            header_length_size = 2 if raw[member_start + 6] == 1 else 4
            header_length = int.from_bytes(
                raw[member_start + 8:member_start + 8 + header_length_size], "little"
            )
            start = member_start + 8 + header_length_size + header_length
            row_bytes = 196 * 768 * 4
            forbidden.extend((start + index * row_bytes, start + (index + 1) * row_bytes, unit.path)
                             for index, split in enumerate(MIXED_SPLITS) if split == "test")
            if unit.bank == "TARGET_CLIP":
                connector = archive.getinfo("connectors.npy")
                connector_local = struct.unpack_from("<IHHHHHIIIHH", raw, connector.header_offset)
                connector_start = connector.header_offset + 30 + connector_local[-2] + connector_local[-1]
                forbidden.append((connector_start, connector_start + connector.file_size, unit.path))
    observed: list[tuple[Path, int, int]] = []

    def observed_read(stream, offset, size):
        observed.append((Path(stream.name), offset, offset + size))
        return original(stream, offset, size)

    monkeypatch.setattr(binding, "_read_exact_at", observed_read)
    ids = tuple(image_id for image_id, split in zip(MIXED_IDS, MIXED_SPLITS, strict=True) if split == "train")
    factory = binding.TrainChunkFactory("SIGLIP2", (source,), (target,), ids, (196, 768))
    first, second = list(factory()), list(factory())
    assert len(first) == len(second) == 1
    assert first[0].source_image_ids == second[0].source_image_ids == ids
    assert first[0].source.shape == (len(ids), 196, 768)
    assert np.array_equal(first[0].source, second[0].source)
    assert [float(row[0, 0]) for row in first[0].source] == [index + 1 for index, split in enumerate(MIXED_SPLITS) if split == "train"]
    assert first[0].source.shape[0] * 196 <= 4096
    assert observed
    for path, start, stop in observed:
        assert all(path != forbidden_path or stop <= low or start >= high
                   for low, high, forbidden_path in forbidden)


@pytest.mark.parametrize("changed,code", [
    ("extra", "BLOCKED_NPZ_LAYOUT"),
    ("encrypted", "BLOCKED_NPZ_LAYOUT"),
    ("ids", "BLOCKED_IMAGE_ORDER"),
    ("plan", "BLOCKED_PLAN"),
    ("fortran", "BLOCKED_SHAPE"),
    ("dtype", "BLOCKED_SHAPE"),
    ("shape", "BLOCKED_SHAPE"),
])
def test_mixed_npz_adversaries_fail_closed(tmp_path: Path, changed: str, code: str) -> None:
    unit = _mixed_unit(tmp_path, changed=changed)
    _blocked(code, lambda: binding._select_train_rows(unit, (196, 768)))


def test_compressed_mixed_npz_rejected(tmp_path: Path) -> None:
    unit = _mixed_unit(tmp_path, compressed=True)
    _blocked("BLOCKED_NPZ_LAYOUT", lambda: binding._select_train_rows(unit, (196, 768)))


def test_mixed_npz_hash_drift_rejected_before_selective_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unit = _mixed_unit(tmp_path)
    unit.path.write_bytes(unit.path.read_bytes() + b"tamper")
    monkeypatch.setattr(binding, "_select_train_rows", lambda *args: pytest.fail("reader called after hash drift"))
    ids = tuple(image_id for image_id, split in zip(MIXED_IDS, MIXED_SPLITS, strict=True) if split == "train")
    factory = binding.TrainChunkFactory("SIGLIP2", (unit,), (unit,), ids, (196, 768))
    _blocked("BLOCKED_ARTIFACT_HASH", lambda: list(factory()))
