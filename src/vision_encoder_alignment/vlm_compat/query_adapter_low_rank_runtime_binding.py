"""Identity-checked, read-only inputs for the frozen low-rank Train runtime.

``bind_low_rank_runtime`` takes three explicit existing absolute directories. It
does not fit or select an adapter. Test-only units are hashed as opaque artifact
bytes for lineage, never opened as NumPy payloads. Mixed units use exact byte
offsets in the accepted ZIP_STORED np.savez layout to read Train patch rows only.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_bank_preflight as preflight
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


LOW_RANK_PATH = Path("configs/vlm_compat/p2_query_adapter_low_rank_linear_v1.json")
LINEAR_PATH = Path("configs/vlm_compat/p2_query_adapter_linear_v1.json")
BANK_PATH = Path("configs/vlm_compat/p2_query_adapter_linear_bank_binding_v1.json")
TEST_BINDING_PATH = Path("configs/vlm_compat/p2_query_adapter_linear_test_binding_v1.json")
TRACKED_SHA = {
    LINEAR_PATH: "28835f065f9a00d756316ced0a1d5ad3e74a3709d88a83abc47ddbe1d6f142a1",
    BANK_PATH: "d2742c93f8eeae286d490fc229ff57fa8cd6701aab81dadde656ff23433129be",
    TEST_BINDING_PATH: "7688f7f14bdf6058f314e8b23ab60cc7ce9ca59dc48df4844b328ed7844662d3",
}
SOURCE_ORDER = ("SIGLIP2", "DINOV3")
BANK_ORDER = (*SOURCE_ORDER, "TARGET_CLIP")
SCHEMA_ID = "vea.p2.query_adapter_low_rank_runtime_binding.v1"


class LowRankRuntimeBindingError(RuntimeError):
    """A stable fail-closed identity, payload, or access rejection."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def _fail(code: str, message: str) -> None:
    raise LowRankRuntimeBindingError(code, message)


def _root(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        _fail("BLOCKED_ROOT", f"{label} must be an existing explicit absolute directory")
    return path.resolve()


def _tracked_json(root: Path, relative: Path, expected: str) -> dict[str, Any]:
    try:
        raw = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            _fail("BLOCKED_TRACKED_SHA", f"accepted tracked-file SHA drift: {relative}")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_TRACKED_SHA", f"accepted tracked file unavailable: {relative}") from error
    if not isinstance(value, dict):
        _fail("BLOCKED_TRACKED_SHA", f"accepted tracked JSON is not an object: {relative}")
    return value


def _artifact(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _fail("BLOCKED_UNSAFE_PATH", f"{label} must be a nonempty POSIX relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or ":" in relative:
        _fail("BLOCKED_UNSAFE_PATH", f"{label} contains an unsafe path component")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        _fail("BLOCKED_UNSAFE_PATH", f"{label} resolves outside its root")
    if not path.is_file():
        _fail("BLOCKED_ARTIFACT_MISSING", f"{label} is missing")
    return path


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise LowRankRuntimeBindingError("BLOCKED_ARTIFACT_MISSING", f"cannot hash {path.name}") from error
    return digest.hexdigest()


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_RESULT", f"cannot read {path.name}") from error
    if not isinstance(value, dict):
        _fail("BLOCKED_RESULT", f"{path.name} is not a JSON object")
    return value


def _array(archive: Any, key: str, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        value = np.array(archive[key], copy=True, order="C")
    except (KeyError, ValueError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_SHAPE", f"{label} {key} missing or unreadable") from error
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        _fail("BLOCKED_SHAPE", f"{label} {key} shape, float32 dtype, or finiteness mismatch")
    return value


@dataclass(frozen=True)
class _StoredMember:
    data_offset: int
    size: int
    crc32: int


def _read_exact_at(stream: Any, offset: int, size: int) -> bytes:
    """One bounded positional read; tests observe these ranges for Test exclusion."""
    if offset < 0 or size < 0:
        _fail("BLOCKED_NPZ_LAYOUT", "negative NPZ byte offset or length")
    stream.seek(offset)
    value = stream.read(size)
    if len(value) != size:
        _fail("BLOCKED_NPZ_LAYOUT", "truncated NPZ member")
    return value


def _stored_members(stream: Any, names: frozenset[str]) -> dict[str, _StoredMember]:
    """Parse only the strict, unencrypted ZIP_STORED layout emitted by np.savez."""
    stream.seek(0, 2)
    total = stream.tell()
    if total < 22:
        _fail("BLOCKED_NPZ_LAYOUT", "NPZ has no end-of-central-directory record")
    end = struct.unpack("<IHHHHIIH", _read_exact_at(stream, total - 22, 22))
    signature, disk, central_disk, disk_count, entry_count, central_size, central_offset, comment_size = end
    if (signature != 0x06054B50 or disk != 0 or central_disk != 0 or comment_size != 0
            or disk_count != entry_count or entry_count != len(names)
            or central_size > 4096 or central_offset + central_size != total - 22):
        _fail("BLOCKED_NPZ_LAYOUT", "NPZ central directory differs from supported np.savez layout")
    central = _read_exact_at(stream, central_offset, central_size)
    cursor = 0
    entries: dict[str, tuple[int, int, int]] = {}
    for _ in range(entry_count):
        if cursor + 46 > len(central):
            _fail("BLOCKED_NPZ_LAYOUT", "truncated NPZ central entry")
        fields = struct.unpack_from("<IHHHHHHIIIHHHHHII", central, cursor)
        (magic, _made, needed, flags, method, _time, _date, crc, packed, size,
         name_len, extra_len, comment_len, disk_start, _internal, _external, local_offset) = fields
        cursor += 46
        if (magic != 0x02014B50 or needed != 45 or flags != 0 or method != zipfile.ZIP_STORED
                or packed != size or size == 0xFFFFFFFF or disk_start != 0
                or extra_len != 0 or comment_len != 0 or not 0 < name_len <= 64
                or cursor + name_len > len(central)):
            _fail("BLOCKED_NPZ_LAYOUT", "compressed, encrypted, or unsupported NPZ central entry")
        try:
            name = central[cursor:cursor + name_len].decode("ascii")
        except UnicodeDecodeError as error:
            raise LowRankRuntimeBindingError("BLOCKED_NPZ_LAYOUT", "non-ASCII NPZ member name") from error
        cursor += name_len
        if name not in names or name in entries:
            _fail("BLOCKED_NPZ_LAYOUT", "NPZ members differ from exact allowlist")
        entries[name] = (local_offset, size, crc)
    if cursor != len(central) or set(entries) != names:
        _fail("BLOCKED_NPZ_LAYOUT", "NPZ members or central directory length differ")
    members: dict[str, _StoredMember] = {}
    previous_end = 0
    for name, (offset, size, crc) in sorted(entries.items(), key=lambda item: item[1][0]):
        if offset != previous_end or offset + 30 > central_offset:
            _fail("BLOCKED_NPZ_LAYOUT", "NPZ member offsets overlap or contain gaps")
        local = struct.unpack("<IHHHHHIIIHH", _read_exact_at(stream, offset, 30))
        (magic, needed, flags, method, _time, _date, local_crc, packed, unpacked,
         name_len, extra_len) = local
        if (magic != 0x04034B50 or needed != 45 or flags != 0 or method != zipfile.ZIP_STORED
                or local_crc != crc or packed != 0xFFFFFFFF or unpacked != 0xFFFFFFFF
                or name_len != len(name) or extra_len != 20):
            _fail("BLOCKED_NPZ_LAYOUT", "NPZ local header differs from np.savez ZIP64 stored layout")
        if _read_exact_at(stream, offset + 30, name_len) != name.encode("ascii"):
            _fail("BLOCKED_NPZ_LAYOUT", "NPZ local and central member names differ")
        extra = _read_exact_at(stream, offset + 30 + name_len, 20)
        if struct.unpack("<HHQQ", extra) != (1, 16, size, size):
            _fail("BLOCKED_NPZ_LAYOUT", "NPZ ZIP64 size metadata differs")
        data_offset = offset + 30 + name_len + 20
        previous_end = data_offset + size
        if previous_end > central_offset:
            _fail("BLOCKED_NPZ_LAYOUT", "NPZ member extends into central directory")
        members[name] = _StoredMember(data_offset, size, crc)
    if previous_end != central_offset:
        _fail("BLOCKED_NPZ_LAYOUT", "NPZ has trailing unbound member bytes")
    return members


def _metadata_array(stream: Any, member: _StoredMember, maximum: int, label: str) -> np.ndarray:
    if member.size > maximum:
        _fail("BLOCKED_NPZ_LAYOUT", f"{label} metadata member exceeds size bound")
    payload = _read_exact_at(stream, member.data_offset, member.size)
    if zlib.crc32(payload) != member.crc32:
        _fail("BLOCKED_NPZ_LAYOUT", f"{label} metadata CRC differs")
    try:
        value = np.lib.format.read_array(io.BytesIO(payload), allow_pickle=False, max_header_size=10000)
    except (OSError, ValueError, EOFError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_NPZ_LAYOUT", f"{label} metadata NPY is invalid") from error
    if not isinstance(value, np.ndarray):
        _fail("BLOCKED_NPZ_LAYOUT", f"{label} metadata is not an NPY array")
    return value


def _patch_data_offset(stream: Any, member: _StoredMember, shape: tuple[int, int, int]) -> int:
    prefix = _read_exact_at(stream, member.data_offset, 8)
    if prefix[:6] != b"\x93NUMPY" or tuple(prefix[6:]) not in ((1, 0), (2, 0)):
        _fail("BLOCKED_NPZ_LAYOUT", "patch NPY magic or version differs")
    length_size = 2 if prefix[6] == 1 else 4
    length_raw = _read_exact_at(stream, member.data_offset + 8, length_size)
    header_size = int.from_bytes(length_raw, "little")
    if header_size > 10000 or 8 + length_size + header_size > member.size:
        _fail("BLOCKED_NPZ_LAYOUT", "patch NPY header length is invalid")
    header = _read_exact_at(stream, member.data_offset + 8 + length_size, header_size)
    try:
        reader = np.lib.format.read_array_header_1_0 if prefix[6] == 1 else np.lib.format.read_array_header_2_0
        observed_shape, fortran, dtype = reader(io.BytesIO(length_raw + header), max_header_size=10000)
    except (ValueError, SyntaxError, TypeError, EOFError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_NPZ_LAYOUT", "patch NPY header is invalid") from error
    if observed_shape != shape or fortran or dtype.str != "<f4":
        _fail("BLOCKED_SHAPE", "patch NPY must be C-contiguous little-endian float32 with approved shape")
    data_offset = member.data_offset + 8 + length_size + header_size
    if data_offset + int(np.prod(shape)) * 4 != member.data_offset + member.size:
        _fail("BLOCKED_NPZ_LAYOUT", "patch NPY data size differs from approved shape")
    return data_offset


def _select_partition_rows(unit: _Unit, shape: tuple[int, int], partition: str) -> list[tuple[str, np.ndarray]]:
    """Decode only the approved partition's patch ranges from a checked unit."""
    if partition not in ("train", "validation"):
        _fail("BLOCKED_TEST_GUARD", "only Train or Validation patch rows may be decoded")
    names = (preflight.TARGET_FIELDS if unit.bank == "TARGET_CLIP" else preflight.SOURCE_FIELDS)
    allowed = frozenset(f"{name}.npy" for name in names)
    count = len(unit.image_ids)
    try:
        with unit.path.open("rb") as stream:
            members = _stored_members(stream, allowed)
            ids = _metadata_array(stream, members["image_ids.npy"], count * 384 + 4096, "image_ids")
            if (ids.shape != (count,) or ids.dtype != np.dtype("U96")
                    or tuple(map(str, ids.tolist())) != unit.image_ids):
                _fail("BLOCKED_IMAGE_ORDER", f"{unit.bank} unit image IDs differ from plan")
            plan_id = _metadata_array(stream, members["plan_id.npy"], 4096, "plan_id")
            if plan_id.shape != () or plan_id.dtype != np.dtype("U64") or str(plan_id.item()) != unit.plan_id:
                _fail("BLOCKED_PLAN", f"{unit.bank} persisted plan identity differs")
            patches = members["patches.npy"]
            data_offset = _patch_data_offset(stream, patches, (count, *shape))
            row_bytes = shape[0] * shape[1] * 4
            selected: list[tuple[str, np.ndarray]] = []
            for index, (image_id, split) in enumerate(zip(unit.image_ids, unit.splits, strict=True)):
                if split != partition:
                    continue
                raw = _read_exact_at(stream, data_offset + index * row_bytes, row_bytes)
                value = np.frombuffer(raw, dtype="<f4").copy().reshape(shape)
                if not np.isfinite(value).all():
                    _fail("BLOCKED_SHAPE", f"{unit.bank} {partition} patch row contains non-finite values")
                selected.append((image_id, value))
            return selected
    except LowRankRuntimeBindingError:
        raise
    except (OSError, ValueError, struct.error) as error:
        raise LowRankRuntimeBindingError("BLOCKED_NPZ_LAYOUT", f"{unit.bank} stored NPZ cannot be read safely") from error


def _select_train_rows(unit: _Unit, shape: tuple[int, int]) -> list[tuple[str, np.ndarray]]:
    """Preserve the reviewed Train-only reader entry point."""
    return _select_partition_rows(unit, shape, "train")


def _select_validation_rows(unit: _Unit, shape: tuple[int, int]) -> list[tuple[str, np.ndarray]]:
    """Read Validation patches only; Test, Train, globals and connectors stay opaque."""
    return _select_partition_rows(unit, shape, "validation")


@dataclass(frozen=True)
class AcceptedOLS:
    weights: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    artifact_sha256: str


@dataclass(frozen=True)
class _Unit:
    bank: str
    image_ids: tuple[str, ...]
    splits: tuple[str, ...]
    path: Path
    sha256: str
    plan_id: str


class TrainChunkFactory:
    """Re-iterable paired Train stream; each pass checks prior chunk bytes."""

    def __init__(self, source: str, source_units: tuple[_Unit, ...], target_units: tuple[_Unit, ...],
                 train_ids: tuple[str, ...], patch_shape: tuple[int, int]) -> None:
        self.source = source
        self._source_units = source_units
        self._target_units = target_units
        self._train_ids = train_ids
        self._patch_shape = patch_shape
        self._first_pass: tuple[str, ...] | None = None

    def __call__(self) -> Iterator[core.TrainChunk]:
        return self._iterate()

    def _iterate(self) -> Iterator[core.TrainChunk]:
        source_rows = self._rows(self._source_units)
        target_rows = self._rows(self._target_units)
        seen: list[str] = []
        ids_seen: list[str] = []
        max_images = max(1, core.MAX_CHUNK_ROWS // self._patch_shape[0])
        pending_source: list[np.ndarray] = []
        pending_target: list[np.ndarray] = []
        pending_ids: list[str] = []
        for source_item, target_item in zip(source_rows, target_rows, strict=True):
            source_id, source_value = source_item
            target_id, target_value = target_item
            if source_id != target_id:
                _fail("BLOCKED_IMAGE_ORDER", f"{self.source} Train source/target image order differs")
            ids_seen.append(source_id)
            pending_ids.append(source_id)
            pending_source.append(source_value)
            pending_target.append(target_value)
            if len(pending_ids) == max_images:
                chunk, digest = self._chunk(pending_source, pending_target, pending_ids)
                self._check_replay(len(seen), digest)
                seen.append(digest)
                yield chunk
                pending_source, pending_target, pending_ids = [], [], []
        if pending_ids:
            chunk, digest = self._chunk(pending_source, pending_target, pending_ids)
            self._check_replay(len(seen), digest)
            seen.append(digest)
            yield chunk
        if tuple(ids_seen) != self._train_ids:
            _fail("BLOCKED_SPLIT", f"{self.source} Train stream count/order differs from manifest")
        if self._first_pass is None:
            self._first_pass = tuple(seen)
        elif tuple(seen) != self._first_pass:
            _fail("BLOCKED_STREAM_REPLAY", f"{self.source} Train stream pass length changed")

    def _check_replay(self, index: int, digest: str) -> None:
        if self._first_pass is not None and (index >= len(self._first_pass) or digest != self._first_pass[index]):
            _fail("BLOCKED_STREAM_REPLAY", f"{self.source} Train chunk {index} changed across passes")

    def _chunk(self, source: list[np.ndarray], target: list[np.ndarray], ids: list[str]) -> tuple[core.TrainChunk, str]:
        s = np.stack(source)
        t = np.stack(target)
        identity = json.dumps(ids, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(identity + s.tobytes() + t.tobytes()).hexdigest()
        return core.TrainChunk(s, t, tuple(ids), tuple(ids), "train"), digest

    def _rows(self, units: tuple[_Unit, ...]) -> Iterator[tuple[str, np.ndarray]]:
        for unit in units:
            if "train" not in unit.splits:
                continue
            if _sha_file(unit.path) != unit.sha256:
                _fail("BLOCKED_ARTIFACT_HASH", f"{unit.bank} Train unit changed after binding")
            selected = _select_train_rows(unit, self._patch_shape)
            if _sha_file(unit.path) != unit.sha256:
                _fail("BLOCKED_ARTIFACT_HASH", f"{unit.bank} Train unit changed during payload read")
            yield from selected


@dataclass(frozen=True)
class BoundLowRankInputs:
    config: Mapping[str, Any]
    ols: Mapping[str, AcceptedOLS]
    train_factories: Mapping[str, TrainChunkFactory]
    provenance: Mapping[str, Any]

    def train_factory(self, source: str) -> TrainChunkFactory:
        if source not in SOURCE_ORDER:
            _fail("BLOCKED_TEST_GUARD", "only the two frozen Train sources are available")
        return self.train_factories[source]

    def split(self, name: str) -> None:
        _fail("BLOCKED_TEST_GUARD", f"split access is not provided by this Train-only binding: {name}")


@dataclass(frozen=True)
class ValidationChunk:
    source: np.ndarray
    target: np.ndarray
    image_ids: tuple[str, ...]
    split: str = "validation"


class ValidationChunkFactory:
    """Paired, ordered, bounded Validation patches with per-unit SHA checks."""

    def __init__(self, source: str, source_units: tuple[_Unit, ...], target_units: tuple[_Unit, ...],
                 validation_ids: tuple[str, ...], patch_shape: tuple[int, int]) -> None:
        self.source = source
        self._source_units = source_units
        self._target_units = target_units
        self._validation_ids = validation_ids
        self._patch_shape = patch_shape

    def __call__(self) -> Iterator[ValidationChunk]:
        source_rows = self._rows(self._source_units)
        target_rows = self._rows(self._target_units)
        observed: list[str] = []
        pending_source: list[np.ndarray] = []
        pending_target: list[np.ndarray] = []
        pending_ids: list[str] = []
        maximum = max(1, core.MAX_CHUNK_ROWS // self._patch_shape[0])
        for (source_id, source_value), (target_id, target_value) in zip(source_rows, target_rows, strict=True):
            if source_id != target_id:
                _fail("BLOCKED_IMAGE_ORDER", f"{self.source} Validation pairing differs")
            observed.append(source_id)
            pending_ids.append(source_id)
            pending_source.append(source_value)
            pending_target.append(target_value)
            if len(pending_ids) == maximum:
                yield ValidationChunk(np.stack(pending_source), np.stack(pending_target), tuple(pending_ids))
                pending_source, pending_target, pending_ids = [], [], []
        if pending_ids:
            yield ValidationChunk(np.stack(pending_source), np.stack(pending_target), tuple(pending_ids))
        if tuple(observed) != self._validation_ids or len(set(observed)) != len(observed):
            _fail("BLOCKED_IMAGE_ORDER", f"{self.source} Validation count/order differs from manifest")

    def _rows(self, units: tuple[_Unit, ...]) -> Iterator[tuple[str, np.ndarray]]:
        for unit in units:
            if "validation" not in unit.splits:
                continue
            if _sha_file(unit.path) != unit.sha256:
                _fail("BLOCKED_ARTIFACT_HASH", f"{unit.bank} Validation unit changed before read")
            selected = _select_validation_rows(unit, self._patch_shape)
            if _sha_file(unit.path) != unit.sha256:
                _fail("BLOCKED_ARTIFACT_HASH", f"{unit.bank} Validation unit changed during read")
            yield from selected


@dataclass(frozen=True)
class BoundLowRankValidationInputs:
    config: Mapping[str, Any]
    ols: Mapping[str, AcceptedOLS]
    validation_factories: Mapping[str, ValidationChunkFactory]
    provenance: Mapping[str, Any]

    def validation_factory(self, source: str) -> ValidationChunkFactory:
        if source not in SOURCE_ORDER:
            _fail("BLOCKED_TEST_GUARD", "only frozen Validation sources are available")
        return self.validation_factories[source]

    def split(self, name: str) -> None:
        _fail("BLOCKED_TEST_GUARD", f"arbitrary split access is forbidden: {name}")


def _lineage(repository_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        config = core.load_low_rank_config(repository_root / LOW_RANK_PATH)
    except core.LowRankLinearError as error:
        raise LowRankRuntimeBindingError("BLOCKED_CONFIG", str(error)) from error
    linear = _tracked_json(repository_root, LINEAR_PATH, TRACKED_SHA[LINEAR_PATH])
    bank = _tracked_json(repository_root, BANK_PATH, TRACKED_SHA[BANK_PATH])
    test = _tracked_json(repository_root, TEST_BINDING_PATH, TRACKED_SHA[TEST_BINDING_PATH])
    provenance = config["provenance"]
    if (provenance["predecessor_linear_config"]["git_blob_sha256"] != TRACKED_SHA[LINEAR_PATH]
            or provenance["accepted_linear_test_binding"]["git_blob_sha256"] != TRACKED_SHA[TEST_BINDING_PATH]
            or bank.get("adapter_config_sha256") != TRACKED_SHA[LINEAR_PATH]
            or bank.get("adapter_config_path") != str(LINEAR_PATH).replace("\\", "/")
            or test.get("bank_binding_config_path") != str(BANK_PATH).replace("\\", "/")
            or test.get("validation_result_sha256") != provenance["accepted_linear_test_binding"]["validation_result_sha256"]
            or test.get("validation_runtime_identity_sha256") != provenance["accepted_linear_test_binding"]["validation_runtime_identity_sha256"]
            or bank.get("manifest_sha256") != config["split"]["manifest_sha256"]
            or linear.get("manifest", {}).get("split_identity_sha256") != config["split"]["split_identity_sha256"]
            or bank.get("t001d_plan_id") is None):
        _fail("BLOCKED_LINEAGE", "T004A, predecessor, bank binding, and Validation identities differ")
    for source in SOURCE_ORDER:
        spec = test.get("finalized_adapter_artifacts", {}).get(source, {}).get("FULL_LINEAR", {})
        if (spec.get("sha256") != config["ols_control"]["exact_artifact_sha256_by_source"][source]
                or spec.get("relative_path") != config["ols_control"]["relative_artifact_path_by_source"][source]):
            _fail("BLOCKED_LINEAGE", f"{source} accepted Full Linear provenance differs")
        for key in ("model_id", "revision"):
            if linear["sources"][source][key] != config["sources"][source][key]:
                _fail("BLOCKED_LINEAGE", f"{source} model identity differs")
        if linear["sources"][source]["tap_name"] != config["sources"][source]["tap_name"]:
            _fail("BLOCKED_LINEAGE", f"{source} tap identity differs")
    for key in ("model_id", "revision", "tap_name"):
        if linear["target"][key] != config["target"][key]:
            _fail("BLOCKED_LINEAGE", "target CLIP identity differs")
    for key in ("model_id", "revision", "connector_input_shape", "connector_output_shape"):
        if linear["target_vlm"][key] != config["target_vlm"][key]:
            _fail("BLOCKED_LINEAGE", "TinyLLaVA identity differs")
    return config, linear, bank, test


def _ols(validation_root: Path, test: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, AcceptedOLS]:
    result_path = _artifact(validation_root, "result.json", "T002D Validation result")
    if _sha_file(result_path) != expected["result"]:
        _fail("BLOCKED_RESULT_HASH", "T002D Validation result SHA differs from accepted identity")
    result = _json_file(result_path)
    if (result.get("schema_id") != "vea.p2.query_adapter_linear_validation_runtime.v1"
            or result.get("status") != "PASS_QUERY_ADAPTER_LINEAR_VALIDATION"
            or result.get("runtime_identity_sha256") != test["validation_runtime_identity_sha256"]
            or result.get("t001d_plan_id") != expected["plan"]
            or any(type(result.get(key)) is not int or result[key] != 0 for key in
                   ("test_scientific_access", "test_metric_count", "generation_call_count", "connector_vision_call_count"))):
        _fail("BLOCKED_RESULT", "T002D Validation result status, runtime, plan, or access counters differ")
    manifest = result.get("fit_artifacts")
    if not isinstance(manifest, dict):
        _fail("BLOCKED_RESULT", "T002D fit-artifact manifest missing")
    output: dict[str, AcceptedOLS] = {}
    for source in SOURCE_ORDER:
        spec = test["finalized_adapter_artifacts"][source]["FULL_LINEAR"]
        relative = spec["relative_path"]
        sha = expected["ols"][source]
        path = _artifact(validation_root, relative, f"{source} Full Linear")
        if manifest.get(relative) != sha or _sha_file(path) != sha:
            _fail("BLOCKED_OLS_HASH", f"{source} Full Linear manifest or artifact SHA differs")
        try:
            with np.load(path, allow_pickle=False) as archive:
                fields = {"runtime_identity", "weights", "source_mean", "target_mean",
                          "rank", "residual_sum_squares"}
                names = archive.zip.namelist()
                if (len(archive.files) != len(fields) or set(archive.files) != fields
                        or len(names) != len(fields) or set(names) != {f"{field}.npy" for field in fields}):
                    _fail("BLOCKED_SHAPE", f"{source} Full Linear fields differ")
                identity = archive["runtime_identity"]
                if identity.shape != () or identity.dtype.kind not in "US" or str(identity.item()) != test["validation_runtime_identity_sha256"]:
                    _fail("BLOCKED_OLS_IDENTITY", f"{source} Full Linear persisted runtime identity differs")
                rank = archive["rank"]
                if rank.shape != () or rank.dtype.kind not in "iu" or not 0 <= int(rank.item()) <= 768:
                    _fail("BLOCKED_SHAPE", f"{source} Full Linear rank must be an integer scalar in 0..768")
                residuals = archive["residual_sum_squares"]
                if (residuals.ndim != 1 or residuals.shape[0] not in (0, 768)
                        or residuals.dtype != np.float32 or not np.isfinite(residuals).all()
                        or (residuals.shape[0] == 768 and int(rank.item()) != 768)):
                    _fail("BLOCKED_SHAPE", f"{source} Full Linear residuals differ from the producer contract")
                output[source] = AcceptedOLS(
                    _array(archive, "weights", (768, 768), source),
                    _array(archive, "source_mean", (768,), source),
                    _array(archive, "target_mean", (768,), source), sha,
                )
        except LowRankRuntimeBindingError:
            raise
        except (OSError, ValueError, EOFError, KeyError) as error:
            raise LowRankRuntimeBindingError("BLOCKED_SHAPE", f"{source} Full Linear cannot be safely read") from error
    return output


def _banks(run_root: Path, plan: Mapping[str, Any], plan_id: str, linear: Mapping[str, Any],
           patch_shape: tuple[int, int], expected: Mapping[str, Any]
           ) -> tuple[dict[str, tuple[_Unit, ...]], tuple[str, ...]]:
    try:
        rows, counts = preflight._validate_plan(plan, linear, preflight.PRODUCTION_EXPECTATIONS)
        identities = preflight._validate_identities(plan, linear)
    except preflight.BankPreflightError as error:
        code = ("BLOCKED_SPLIT" if "SPLIT" in error.classification else
                "BLOCKED_IMAGE_ORDER" if "IMAGE_ORDER" in error.classification else
                "BLOCKED_IDENTITY" if "MODEL_IDENTITY" in error.classification else "BLOCKED_PLAN")
        raise LowRankRuntimeBindingError(code, f"T001D plan/split/model identity invalid: {error}") from error
    except (KeyError, TypeError, ValueError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_PLAN", f"T001D plan/split/model identity invalid: {error}") from error
    if counts != {"train": 800, "validation": 100, "test": 100}:
        _fail("BLOCKED_SPLIT", "T001D partition counts differ")
    split_by_id = {row["image_id"]: row["split"] for row in rows}
    manifest_ids = tuple(row["image_id"] for row in rows)
    train_ids = tuple(row["image_id"] for row in rows if row["split"] == "train")
    all_units: dict[str, tuple[_Unit, ...]] = {}
    for bank in BANK_ORDER:
        try:
            planned = preflight._ordered_units(plan, bank)
        except preflight.BankPreflightError as error:
            raise LowRankRuntimeBindingError("BLOCKED_PLAN", str(error)) from error
        units: list[_Unit] = []
        ids: list[str] = []
        digest_units: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        seen_unit_ids: set[str] = set()
        for unit in planned:
            unit_ids = unit.get("image_ids")
            relative = unit.get("artifact_relative_path")
            uid = unit.get("unit_id")
            if (unit.get("encoder") != bank or not isinstance(unit_ids, list) or not unit_ids
                    or not isinstance(uid, str) or uid in seen_unit_ids
                    or not isinstance(relative, str) or relative in seen_paths):
                _fail("BLOCKED_PLAN", f"{bank} unit identity/IDs invalid or duplicate")
            seen_paths.add(relative)
            seen_unit_ids.add(uid)
            if any(image_id not in split_by_id for image_id in unit_ids):
                _fail("BLOCKED_SPLIT", f"{bank} unit contains an unapproved image ID")
            splits = tuple(split_by_id[image_id] for image_id in unit_ids)
            n = len(unit_ids)
            if "train" in splits and n * patch_shape[0] > core.MAX_CHUNK_ROWS:
                _fail("BLOCKED_CHUNK_CAP", f"{bank} Train-bearing NPZ unit exceeds 4096 patch rows")
            shape = {"patches": [n, *patch_shape], "globals": [n, patch_shape[1]]}
            if bank == "TARGET_CLIP":
                shape["connectors"] = [n, 196, 1536]
            if unit.get("shape") != shape or unit.get("dtype") != {key: "float32" for key in shape}:
                _fail("BLOCKED_SHAPE", f"{bank} unit shape/dtype metadata differs")
            path = _artifact(run_root, relative, f"{bank} unit")
            sha = _sha_file(path)  # Opaque bytes only, including Test-only units.
            units.append(_Unit(bank, tuple(unit_ids), splits, path, sha, plan_id))
            ids.extend(unit_ids)
            digest_units.append({"unit_id": uid, "artifact_relative_path": relative, "artifact_sha256": sha})
        if tuple(ids) != manifest_ids or len(set(ids)) != 1000:
            _fail("BLOCKED_IMAGE_ORDER", f"{bank} image order differs from approved manifest")
        digest_material = {
            "bank": bank, "encoder_identity": identities[bank], "t001d_plan_id": plan_id,
            "manifest_sha256": plan["manifest_sha256"], "ordered_units": digest_units,
            "ordered_image_ids": ids, "patch_shape": list(patch_shape), "patch_dtype": "float32",
            "connector_shape": [196, 1536] if bank == "TARGET_CLIP" else None,
            "connector_dtype": "float32" if bank == "TARGET_CLIP" else None,
        }
        digest = hashlib.sha256(t001d.canonical_json_bytes(digest_material)).hexdigest()
        if digest != expected["banks"][bank]:
            _fail("BLOCKED_BANK_HASH", f"{bank} bank identity differs from accepted binding")
        all_units[bank] = tuple(units)
    return all_units, train_ids


def _bind(repository_root: str | Path, t001d_run_root: str | Path, t002d_validation_root: str | Path,
          *, fixture: Mapping[str, Any] | None = None) -> BoundLowRankInputs:
    repository = _root(repository_root, "repository_root")
    run_root = _root(t001d_run_root, "t001d_run_root")
    validation_root = _root(t002d_validation_root, "t002d_validation_root")
    config, linear, bank, test = _lineage(repository)
    try:
        plan, plan_id = t001d.load_bound_plan(run_root)
    except (OSError, ValueError, TypeError, t001d.HarnessError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_PLAN", "T001D bound plan unavailable") from error
    expected = {
        "plan": bank["t001d_plan_id"], "manifest": bank["manifest_sha256"], "banks": bank["banks"],
        "result": test["validation_result_sha256"],
        "ols": {source: test["finalized_adapter_artifacts"][source]["FULL_LINEAR"]["sha256"] for source in SOURCE_ORDER},
    }
    if fixture is not None:
        expected = fixture["expected"]
    if plan_id != expected["plan"] or plan.get("manifest_sha256") != expected["manifest"]:
        _fail("BLOCKED_PLAN", "T001D plan or manifest SHA differs from accepted binding")
    patch_shape = (196, 768) if fixture is None else tuple(fixture["patch_shape"])
    units, train_ids = _banks(run_root, plan, plan_id, linear, patch_shape, expected)
    ols = _ols(validation_root, test, expected)
    factories = {source: TrainChunkFactory(source, units[source], units["TARGET_CLIP"], train_ids, patch_shape)
                 for source in SOURCE_ORDER}
    return BoundLowRankInputs(config, ols, factories, {
        "schema_id": SCHEMA_ID, "t001d_plan_id": plan_id,
        "manifest_sha256": expected["manifest"], "bank_identity_sha256": dict(expected["banks"]),
        "validation_result_sha256": expected["result"],
        "validation_runtime_identity_sha256": test["validation_runtime_identity_sha256"],
        "ols_artifact_sha256": dict(expected["ols"]),
        "train_image_count": len(train_ids), "test_scientific_access": 0,
        "test_payload_open_count": 0, "test_only_units_hashed_as_opaque_bytes": True,
        "float32_latency_evidence": False,
        "numeric_note": "T004B apply_low_rank accumulates float64 and returns float32; no float32 adapter-only latency was measured.",
    })


def bind_low_rank_runtime(repository_root: str | Path, t001d_run_root: str | Path,
                          t002d_validation_root: str | Path) -> BoundLowRankInputs:
    """Bind accepted real identities without fitting, evaluating, or opening Test tensors."""
    return _bind(repository_root, t001d_run_root, t002d_validation_root)


def _bind_synthetic_fixture_for_tests(repository_root: str | Path, t001d_run_root: str | Path,
                                      t002d_validation_root: str | Path, *,
                                      fixture: Mapping[str, Any]) -> BoundLowRankInputs:
    """Private offline seam: fixture hashes/geometry cannot reach the public entry point."""
    return _bind(repository_root, t001d_run_root, t002d_validation_root, fixture=fixture)


def _bind_validation(repository_root: str | Path, t001d_run_root: str | Path,
                     t002d_validation_root: str | Path, *,
                     fixture: Mapping[str, Any] | None = None) -> BoundLowRankValidationInputs:
    """Add Validation selection over the already accepted Train/OLS lineage."""
    accepted = _bind(repository_root, t001d_run_root, t002d_validation_root, fixture=fixture)
    repository = _root(repository_root, "repository_root")
    run_root = _root(t001d_run_root, "t001d_run_root")
    _config, linear, _bank, _test = _lineage(repository)
    try:
        plan, plan_id = t001d.load_bound_plan(run_root)
    except (OSError, ValueError, TypeError, t001d.HarnessError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_PLAN", "T001D bound plan unavailable") from error
    provenance = accepted.provenance
    if plan_id != provenance["t001d_plan_id"] or plan.get("manifest_sha256") != provenance["manifest_sha256"]:
        _fail("BLOCKED_PLAN", "Validation plan changed after accepted binding")
    expected = {"plan": plan_id, "manifest": provenance["manifest_sha256"],
                "banks": provenance["bank_identity_sha256"]}
    patch_shape = (196, 768) if fixture is None else tuple(fixture["patch_shape"])
    units, _train_ids = _banks(run_root, plan, plan_id, linear, patch_shape, expected)
    validation_ids = tuple(image_id for unit in units["TARGET_CLIP"]
                           for image_id, split in zip(unit.image_ids, unit.splits, strict=True)
                           if split == "validation")
    if len(validation_ids) != 100 or len(set(validation_ids)) != 100:
        _fail("BLOCKED_SPLIT", "Validation image count or uniqueness differs")
    factories = {source: ValidationChunkFactory(source, units[source], units["TARGET_CLIP"],
                                                 validation_ids, patch_shape)
                 for source in SOURCE_ORDER}
    return BoundLowRankValidationInputs(accepted.config, accepted.ols, factories, {
        **provenance, "schema_id": "vea.p2.query_adapter_low_rank_validation_binding.v1",
        "validation_image_count": len(validation_ids),
    })


def bind_low_rank_validation_runtime(repository_root: str | Path, t001d_run_root: str | Path,
                                     t002d_validation_root: str | Path) -> BoundLowRankValidationInputs:
    """Bind accepted OLS and Validation-only preconnector patch streams."""
    return _bind_validation(repository_root, t001d_run_root, t002d_validation_root)


def _bind_validation_synthetic_fixture_for_tests(repository_root: str | Path, t001d_run_root: str | Path,
                                                 t002d_validation_root: str | Path, *,
                                                 fixture: Mapping[str, Any]) -> BoundLowRankValidationInputs:
    """Private offline seam; production identities remain fixed."""
    return _bind_validation(repository_root, t001d_run_root, t002d_validation_root, fixture=fixture)
