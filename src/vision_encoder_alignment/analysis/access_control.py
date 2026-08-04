"""Fail-closed P1 input validation and split-authorized partial tensor reads."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from safetensors import safe_open

from .config import AnalysisConfig, canonical_json_bytes, strict_json_document


class AccessError(RuntimeError):
    """Raised for an input-identity or test-access contract violation."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _link_or_reparse(path: Path) -> bool:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise AccessError(f"unable to stat input path: {path.name}: {exc}") from exc
    if stat.S_ISLNK(value.st_mode):
        return True
    attributes = getattr(value, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _regular_non_link(path: Path, *, label: str) -> None:
    if not path.is_file() or _link_or_reparse(path):
        raise AccessError(f"{label}: expected a regular non-link file")


def _strict_json_file(path: Path, *, label: str, canonical: bool = True) -> dict[str, Any]:
    _regular_non_link(path, label=label)
    raw = path.read_bytes()
    value = strict_json_document(raw, label=label)
    if canonical and raw != canonical_json_bytes(value):
        raise AccessError(f"{label}: JSON is not canonical")
    return value


@dataclass(frozen=True, slots=True)
class ManifestRow:
    image_id: str
    split: str
    selection_index: int


@dataclass(frozen=True, slots=True)
class ShardDescriptor:
    model_key: str
    shard_index: int
    start_index: int
    stop_index: int
    image_ids: tuple[str, ...]
    sidecar_path: Path
    tensor_path: Path
    tensor_keys: tuple[str, ...]
    header: Mapping[str, Mapping[str, Any]]

    @property
    def relative_tensor_path(self) -> str:
        return f"models/{self.model_key}/{self.tensor_path.name}"


@dataclass(frozen=True, slots=True)
class ValidatedInput:
    run_root: Path
    evidence_root: Path
    checksum_manifest_path: Path
    checksum_manifest_sha256: str
    ordered_rows: tuple[ManifestRow, ...]
    shards: Mapping[str, tuple[ShardDescriptor, ...]]
    verified_file_count: int

    @property
    def split_by_image_id(self) -> Mapping[str, str]:
        return {row.image_id: row.split for row in self.ordered_rows}


def _manifest_rows(path: Path, *, expected_sha: str) -> tuple[ManifestRow, ...]:
    if sha256_file(path) != expected_sha:
        raise AccessError("image manifest checksum mismatch")
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    for index, raw in enumerate(path.read_bytes().splitlines()):
        value = strict_json_document(raw + b"\n", label=f"image manifest row {index}")
        image_id = value.get("image_id")
        split = value.get("split")
        selection = value.get("selection_index")
        if not isinstance(image_id, str) or not image_id or image_id in seen:
            raise AccessError("image manifest contains invalid or duplicate image ID")
        if split not in {"train", "validation", "test"} or selection != index:
            raise AccessError("image manifest order or split mismatch")
        seen.add(image_id)
        rows.append(ManifestRow(image_id, split, index))
    if len(rows) != 1000:
        raise AccessError("image manifest row count mismatch")
    counts = {split: sum(row.split == split for row in rows) for split in ("train", "validation", "test")}
    if counts != {"train": 800, "validation": 100, "test": 100}:
        raise AccessError("image manifest split counts mismatch")
    return tuple(rows)


def _parse_checksum_manifest(path: Path, *, run_id: str) -> dict[str, str]:
    _regular_non_link(path, label="input checksum manifest")
    prefix = f"/home/chltjsdh/vea/artifacts/p1_t003/{run_id}/"
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdef" for c in parts[0]):
            raise AccessError(f"checksum manifest line {line_number}: invalid")
        if not parts[1].startswith(prefix):
            raise AccessError("checksum manifest path is outside the accepted Apple root")
        relative = parts[1][len(prefix):]
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative or relative in result:
            raise AccessError("checksum manifest relative path is unsafe or duplicate")
        result[relative] = parts[0]
    if len(result) != 252:
        raise AccessError("checksum manifest must contain exactly 252 files")
    return result


def _verify_evidence(evidence_root: Path, config: AnalysisConfig) -> Path:
    if not evidence_root.is_dir() or _link_or_reparse(evidence_root):
        raise AccessError("evidence root must be a non-link directory")
    manifest = evidence_root / "24-final-evidence-manifest.txt"
    _regular_non_link(manifest, label="evidence manifest")
    if sha256_file(manifest) != config.document["input"]["evidence_manifest_sha256"]:
        raise AccessError("evidence manifest checksum mismatch")
    expected_names = {
        "20-full-run.json", "21-full-validation.json",
        "22-full-run-file-sha256.txt", "23-final-summary.json",
    }
    observed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or parts[1] not in expected_names or parts[1] in observed:
            raise AccessError("evidence manifest inventory mismatch")
        observed[parts[1]] = parts[0]
    if set(observed) != expected_names:
        raise AccessError("evidence manifest must cover exactly four files")
    for name, expected in observed.items():
        evidence = evidence_root / name
        _regular_non_link(evidence, label="evidence member")
        if sha256_file(evidence) != expected:
            raise AccessError(f"evidence member checksum mismatch: {name}")
    return evidence_root / "22-full-run-file-sha256.txt"


def _safetensors_header(path: Path) -> dict[str, Mapping[str, Any]]:
    _regular_non_link(path, label="Safetensors shard")
    size = path.stat().st_size
    if size < 9:
        raise AccessError("Safetensors shard is truncated")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise AccessError("Safetensors header prefix is truncated")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length < 2 or header_length > size - 8:
            raise AccessError("Safetensors header length is invalid")
        header_raw = stream.read(header_length)
    try:
        value = json.loads(
            header_raw.decode("utf-8").rstrip(" "),
            object_pairs_hook=lambda pairs: _header_pairs(pairs),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, AccessError) as exc:
        if isinstance(exc, AccessError):
            raise
        raise AccessError("Safetensors header JSON is invalid") from exc
    if not isinstance(value, dict):
        raise AccessError("Safetensors header must be an object")
    value.pop("__metadata__", None)
    descriptors: dict[str, Mapping[str, Any]] = {}
    spans: list[tuple[int, int, str]] = []
    for key, descriptor in value.items():
        if not isinstance(key, str) or not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise AccessError("Safetensors tensor descriptor is invalid")
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if dtype != "F32" or not isinstance(shape, list) or any(type(item) is not int or item < 1 for item in shape):
            raise AccessError("Safetensors dtype or shape is invalid")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(item) is not int or item < 0 for item in offsets):
            raise AccessError("Safetensors data offsets are invalid")
        start, stop = offsets
        expected_bytes = 4 * int(np.prod(shape, dtype=np.int64))
        if stop <= start or stop - start != expected_bytes:
            raise AccessError("Safetensors tensor byte span is invalid")
        spans.append((start, stop, key))
        descriptors[key] = descriptor
    cursor = 0
    for start, stop, _ in sorted(spans):
        if start != cursor:
            raise AccessError("Safetensors offsets contain a gap or overlap")
        cursor = stop
    if 8 + header_length + cursor != size:
        raise AccessError("Safetensors final offset does not match file size")
    return descriptors


def _header_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AccessError(f"duplicate Safetensors header key: {key}")
        result[key] = value
    return result


def _validate_shard_sidecar(
    sidecar: Path,
    *,
    tensor: Path,
    model: Mapping[str, Any],
    run_id: str,
    ordered_ids: Sequence[str],
    expected_keys: tuple[str, ...],
) -> ShardDescriptor:
    value = _strict_json_file(sidecar, label="shard sidecar")
    required = {
        "created_at_utc", "dtype", "end_index_exclusive", "hidden_size", "image_ids",
        "lane_id", "model_id", "model_key", "patch_count", "revision", "run_id",
        "safetensors_file", "safetensors_sha256", "schema_id", "shard_index",
        "start_index", "tensor_keys", "tensor_sha256",
    }
    if set(value) != required:
        raise AccessError("shard sidecar fields mismatch")
    index = value["shard_index"]
    start = value["start_index"]
    stop = value["end_index_exclusive"]
    if type(index) is not int or type(start) is not int or type(stop) is not int:
        raise AccessError("shard index fields must be integers")
    if start != index * 25 or stop != min(start + 25, 1000):
        raise AccessError("shard boundaries mismatch")
    expected_ids = tuple(ordered_ids[start:stop])
    identity = {
        "schema_id": "vea.p1.feature_shard.v0",
        "run_id": run_id,
        "model_key": model["key"],
        "model_id": model["model_id"],
        "revision": model["revision"],
        "lane_id": model["lane_id"],
        "dtype": "float32",
        "hidden_size": 768,
        "patch_count": 196,
        "safetensors_file": tensor.name,
    }
    for key, expected in identity.items():
        if value[key] != expected:
            raise AccessError(f"shard identity mismatch: {key}")
    if tuple(value["image_ids"]) != expected_ids:
        raise AccessError("shard image order mismatch")
    if tuple(value["tensor_keys"]) != expected_keys or set(value["tensor_sha256"]) != set(expected_keys):
        raise AccessError("shard tensor-key inventory mismatch")
    if value["safetensors_sha256"] != sha256_file(tensor):
        raise AccessError("shard sidecar Safetensors checksum mismatch")
    header = _safetensors_header(tensor)
    if set(header) != set(expected_keys):
        raise AccessError("Safetensors header key inventory mismatch")
    batch = stop - start
    for key in expected_keys:
        expected_shape = [batch, 196, 768] if key.endswith(".patch") else [batch, 768]
        if header[key]["shape"] != expected_shape or header[key]["dtype"] != "F32":
            raise AccessError(f"Safetensors tensor contract mismatch: {key}")
    return ShardDescriptor(
        model_key=model["key"],
        shard_index=index,
        start_index=start,
        stop_index=stop,
        image_ids=expected_ids,
        sidecar_path=sidecar,
        tensor_path=tensor,
        tensor_keys=expected_keys,
        header=header,
    )


def validate_analysis_input(
    *,
    run_root: str | Path,
    evidence_root: str | Path,
    repository_root: str | Path,
    config: AnalysisConfig,
) -> ValidatedInput:
    root = Path(run_root).resolve(strict=True)
    evidence = Path(evidence_root).resolve(strict=True)
    repository = Path(repository_root).resolve(strict=True)
    if not root.is_dir() or _link_or_reparse(root):
        raise AccessError("run root must be a regular non-link directory")
    checksum_path = _verify_evidence(evidence, config)
    if sha256_file(checksum_path) != config.document["input"]["checksum_manifest_sha256"]:
        raise AccessError("input checksum manifest identity mismatch")
    checksums = _parse_checksum_manifest(checksum_path, run_id=config.document["input"]["run_id"])
    actual: set[str] = set()
    for path in root.rglob("*"):
        if _link_or_reparse(path):
            raise AccessError("input tree contains a link or reparse point")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(checksums):
        missing = sorted(set(checksums) - actual)
        extra = sorted(actual - set(checksums))
        raise AccessError(f"input inventory mismatch: missing={missing[:1]} extra={extra[:1]}")
    for relative in sorted(checksums):
        path = root.joinpath(*PurePosixPath(relative).parts)
        _regular_non_link(path, label="input member")
        if sha256_file(path) != checksums[relative]:
            raise AccessError(f"input checksum mismatch: {relative}")

    run_manifest = root / "run_manifest.json"
    if sha256_file(run_manifest) != config.document["input"]["run_manifest_sha256"]:
        raise AccessError("run manifest checksum mismatch")
    run_document = _strict_json_file(run_manifest, label="run manifest")
    if run_document.get("schema_id") != "vea.p1.feature_run_manifest.v0" or run_document.get("status") != "complete":
        raise AccessError("run manifest is not a complete P1 run")
    if run_document.get("run_id") != config.document["input"]["run_id"] or run_document.get("image_count") != 1000:
        raise AccessError("run manifest identity mismatch")
    manifest_path = repository / config.document["splits"]["manifest_path"]
    rows = _manifest_rows(manifest_path, expected_sha=config.document["input"]["image_manifest_sha256"])
    ordered_ids = tuple(row.image_id for row in rows)
    if tuple(run_document.get("image_ids", ())) != ordered_ids:
        raise AccessError("run manifest image order mismatch")

    expected_keys = tuple(config.document["input"]["tensor_keys"])
    shard_map: dict[str, tuple[ShardDescriptor, ...]] = {}
    for model in config.document["input"]["encoders"]:
        directory = root / "models" / model["key"]
        descriptors: list[ShardDescriptor] = []
        for index in range(40):
            basename = f"shard-{index:05d}"
            descriptors.append(_validate_shard_sidecar(
                directory / f"{basename}.json",
                tensor=directory / f"{basename}.safetensors",
                model=model,
                run_id=config.document["input"]["run_id"],
                ordered_ids=ordered_ids,
                expected_keys=expected_keys,
            ))
        if tuple(item.shard_index for item in descriptors) != tuple(range(40)):
            raise AccessError("shard order mismatch")
        shard_map[model["key"]] = tuple(descriptors)
    return ValidatedInput(
        run_root=root,
        evidence_root=evidence,
        checksum_manifest_path=checksum_path,
        checksum_manifest_sha256=sha256_file(checksum_path),
        ordered_rows=rows,
        shards=shard_map,
        verified_file_count=len(checksums),
    )


_DEV_STATES = {"T004_DEV", "T005_DEV", "T006_DEV", "EXTENSION_DEV", "FREEZE"}


class AuthorizedFeatureReader:
    """Read only explicit row spans authorized by the monotonic campaign state."""

    def __init__(
        self,
        validated_input: ValidatedInput,
        *,
        config: AnalysisConfig,
        state_getter: Callable[[], str],
        ledger_path: str | Path,
        on_first_test_read: Callable[[str], None] | None = None,
        unseal_event_getter: Callable[[], str | None] | None = None,
    ) -> None:
        self.input = validated_input
        self.config = config
        self.state_getter = state_getter
        self.ledger_path = Path(ledger_path)
        self.on_first_test_read = on_first_test_read
        self.unseal_event_getter = unseal_event_getter
        self._test_read_callback_completed = False
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._split_by_id = validated_input.split_by_image_id

    def _authorize(self, splits: tuple[str, ...]) -> str:
        if not splits or any(split not in {"train", "validation", "test"} for split in splits):
            raise AccessError("feature read requested an invalid split")
        state = self.state_getter()
        if state in _DEV_STATES:
            if "test" in splits:
                raise AccessError("BLOCKED_LEAKAGE: test row requested before test unseal")
        elif state == "T007_TEST_UNSEAL":
            if splits != ("test",):
                raise AccessError("post-unseal reads are restricted to frozen test endpoints")
        else:
            raise AccessError(f"value reads are forbidden in campaign state {state}")
        return state

    @staticmethod
    def _spans(indices: Sequence[int], row_count: int) -> tuple[tuple[int, int], ...]:
        if not indices:
            return ()
        spans: list[tuple[int, int]] = []
        start = previous = indices[0]
        for index in indices[1:]:
            if index == previous + 1:
                previous = index
                continue
            spans.append((start, previous + 1))
            start = previous = index
        spans.append((start, previous + 1))
        result: list[tuple[int, int]] = []
        for start, stop in spans:
            if start == 0 and stop == row_count and row_count > 1:
                midpoint = row_count // 2
                result.extend(((0, midpoint), (midpoint, row_count)))
            else:
                result.append((start, stop))
        return tuple(result)

    def _ledger(
        self,
        *,
        state: str,
        split: str,
        model_key: str,
        tensor_key: str,
        image_count: int,
        purpose: str,
        shard: ShardDescriptor,
        start: int,
        stop: int,
        unseal_event_id: str | None,
    ) -> None:
        if not isinstance(purpose, str) or not purpose or any(char in purpose for char in "\r\n"):
            raise AccessError("ledger purpose must be one sanitized line")
        record = {
            "schema_id": "vea.p1.data_access.v1",
            "stage": state,
            "split": split,
            "image_count": image_count,
            "model": model_key,
            "tap": tensor_key,
            "state": state,
            "purpose": purpose,
            "config_sha256": self.config.sha256,
            "shard": shard.relative_tensor_path,
            "row_start": start,
            "row_stop_exclusive": stop,
        }
        if unseal_event_id is not None:
            record["unseal_event_id"] = unseal_event_id
        with self.ledger_path.open("ab") as stream:
            stream.write(canonical_json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())

    def iter_blocks(
        self,
        model_key: str,
        tensor_key: str,
        *,
        splits: tuple[str, ...],
        purpose: str,
    ) -> Iterator[tuple[tuple[str, ...], np.ndarray]]:
        state = self._authorize(splits)
        if model_key not in self.input.shards:
            raise AccessError("unknown model key")
        if tensor_key not in self.config.document["input"]["tensor_keys"]:
            raise AccessError("unknown tensor key")
        split_set = set(splits)
        for shard in self.input.shards[model_key]:
            local_indices = [
                index for index, image_id in enumerate(shard.image_ids)
                if self._split_by_id[image_id] in split_set
            ]
            for start, stop in self._spans(local_indices, len(shard.image_ids)):
                span_ids = shard.image_ids[start:stop]
                span_splits = {self._split_by_id[image_id] for image_id in span_ids}
                if len(span_splits) != 1:
                    raise AccessError("authorized slice crosses split boundary")
                split = next(iter(span_splits))
                if split not in split_set:
                    raise AccessError("authorized slice contains a forbidden row")
                unseal_event_id: str | None = None
                if split == "test":
                    if self.unseal_event_getter is None:
                        raise AccessError("test read has no unseal-event identity provider")
                    unseal_event_id = self.unseal_event_getter()
                    if not isinstance(unseal_event_id, str) or not unseal_event_id:
                        raise AccessError("test read has no immutable unseal-event identity")
                    if not self._test_read_callback_completed:
                        if self.on_first_test_read is None:
                            raise AccessError("test read has no first-read binding callback")
                        self.on_first_test_read(unseal_event_id)
                        self._test_read_callback_completed = True
                try:
                    with safe_open(str(shard.tensor_path), framework="np") as handle:
                        value = np.asarray(handle.get_slice(tensor_key)[start:stop])
                except Exception as exc:
                    raise AccessError("authorized Safetensors partial read failed") from exc
                expected_shape = (stop - start, 196, 768) if tensor_key.endswith(".patch") else (stop - start, 768)
                if value.shape != expected_shape or value.dtype != np.float32:
                    raise AccessError("authorized tensor slice shape or dtype mismatch")
                if not np.all(np.isfinite(value)):
                    raise AccessError("authorized tensor slice contains NaN or Inf")
                if np.all(value == 0.0):
                    raise AccessError("authorized tensor slice is all zero")
                self._ledger(
                    state=state,
                    split=split,
                    model_key=model_key,
                    tensor_key=tensor_key,
                    image_count=stop - start,
                    purpose=purpose,
                    shard=shard,
                    start=start,
                    stop=stop,
                    unseal_event_id=unseal_event_id,
                )
                yield span_ids, value

    def read_array(
        self,
        model_key: str,
        tensor_key: str,
        *,
        split: str,
        purpose: str,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        identifiers: list[str] = []
        arrays: list[np.ndarray] = []
        for image_ids, value in self.iter_blocks(
            model_key,
            tensor_key,
            splits=(split,),
            purpose=purpose,
        ):
            identifiers.extend(image_ids)
            arrays.append(value)
        expected = tuple(row.image_id for row in self.input.ordered_rows if row.split == split)
        if tuple(identifiers) != expected:
            raise AccessError("feature reader did not restore canonical split order")
        if not arrays:
            raise AccessError("feature reader returned no rows")
        return tuple(identifiers), np.concatenate(arrays, axis=0)


__all__ = [
    "AccessError", "AuthorizedFeatureReader", "ManifestRow", "ShardDescriptor",
    "ValidatedInput", "sha256_file", "validate_analysis_input",
]
