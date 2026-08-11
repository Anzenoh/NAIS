"""Immutable, atomically checkpointed Phase-2 anchor-transport orchestration.

This module deliberately has a standard-library-only import surface.  Numerical
dependencies are imported inside the fitting-free transport functions so the
plan, status, resume, and verification commands remain usable before any model
runtime is entered.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import socket
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


PLAN_SCHEMA = "vea.p2.t001d.run_plan.v1"
STATE_SCHEMA = "vea.p2.t001d.resume.v1"
UNIT_SCHEMA = "vea.p2.t001d.unit_artifact.v1"
RESULT_SCHEMA = "vea.p2.t001d.resumable_anchor_transport.v1"
MARKER_SCHEMA = "vea.p2.t001d.done_marker.v1"
EVENT_SCHEMA = "vea.p2.t001d.progress_event.v1"
LOCK_SCHEMA = "vea.p2.t001d.coordinator_lock.v1"
RETRYABLE_EXIT_CODE = 75

STAGES = (
    "00_PLAN",
    "01_DINOV3_FEATURE_BANK",
    "02_SIGLIP2_FEATURE_BANK",
    "03_TARGET_CLIP_AND_NATIVE_CONNECTOR_BANK",
    "04_ANCHOR_REGISTRY",
    "05_TRANSPORT_AND_CONNECTOR_VALIDATION",
    "06_HELD_OUT_GENERATION",
    "07_METRICS_AND_FINALIZATION",
)
ENCODERS = ("TARGET_CLIP", "SIGLIP2", "DINOV3")
CONDITIONS = (
    "RAW",
    "SAMPLE_RMS",
    "GLOBAL_1NN",
    "GLOBAL_TOP8",
    "POSITIONAL_1NN",
    "POSITIONAL_TOP8",
)
FINAL_ARTIFACTS = (
    "result.json",
    "summary.csv",
    "per_image.csv",
    "retrieval.csv",
    "runtime_report.json",
    "resume_report.json",
    "artifact_manifest.json",
)


class HarnessError(RuntimeError):
    """Base class carrying a stable failure classification."""

    classification = "BLOCKED_HARNESS_ERROR"
    retryable = False


class PlanMismatchError(HarnessError):
    classification = "BLOCKED_PLAN_ID_MISMATCH"


class CorruptionError(HarnessError):
    classification = "BLOCKED_CHECKPOINT_CORRUPTION"


class IncompleteRunError(HarnessError):
    classification = "PARTIAL_INCOMPLETE_RUN"
    retryable = True


class RunInterruptedError(HarnessError):
    classification = "PARTIAL_INTERRUPTED"
    exit_code = 130


class LockBusyError(HarnessError):
    classification = "PARTIAL_LIVE_COORDINATOR"
    retryable = True


class AmbiguousLockError(HarnessError):
    classification = "BLOCKED_AMBIGUOUS_COORDINATOR_LOCK"


class ContractError(HarnessError):
    classification = "BLOCKED_CONTRACT_VIOLATION"


class InjectedInterruption(IncompleteRunError):
    classification = "PARTIAL_INJECTED_INTERRUPTION"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_load(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {token}")

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    validate_partial: Callable[[bytes], None] | None = None,
) -> str:
    """Publish bytes with a same-directory partial and durable rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        observed_payload = partial.read_bytes()
        if validate_partial is not None:
            validate_partial(observed_payload)
        observed = sha256_file(partial)
        expected = sha256_bytes(payload)
        if observed != expected:
            raise CorruptionError(f"partial write hash mismatch for {path}")
        os.replace(partial, path)
        _fsync_directory(path.parent)
        return expected
    finally:
        if partial.exists():
            partial.unlink()


def atomic_write_json(path: Path, value: Any) -> str:
    return atomic_write_bytes(path, canonical_json_bytes(value))


def append_event(run_root: Path, event: Mapping[str, Any]) -> None:
    path = run_root / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(event))
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def progress_event(plan_id: str, event_type: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_id": EVENT_SCHEMA,
        "plan_id": plan_id,
        "event_type": event_type,
        "recorded_at_utc": utc_now(),
        **fields,
    }


def load_config(path: Path) -> dict[str, Any]:
    value = strict_json_load(path)
    if not isinstance(value, dict):
        raise ContractError("configuration must be a JSON object")
    required = {
        "schema_id",
        "manifest",
        "source_encoder_order",
        "encoders",
        "target_vlm",
        "anchor_count",
        "feature_chunk_size",
        "condition_order",
        "top_k",
        "softmax_temperature",
        "rms_epsilon",
        "prompt",
        "generation",
        "runtime_dtype",
        "stored_feature_dtype",
        "similarity_metric_dtype",
        "resume_schema",
        "artifact_schema",
        "plan_schema",
        "result_schema",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ContractError(f"configuration keys missing: {missing}")
    if value["plan_schema"] != PLAN_SCHEMA or value["result_schema"] != RESULT_SCHEMA:
        raise ContractError("configuration schema binding is not the T001D contract")
    if tuple(value["source_encoder_order"]) != ENCODERS:
        raise ContractError("source encoder order must be TARGET_CLIP,SIGLIP2,DINOV3")
    if tuple(value["condition_order"]) != CONDITIONS:
        raise ContractError("condition order differs from the frozen contract")
    if int(value["anchor_count"]) != 128 or int(value["feature_chunk_size"]) != 16:
        raise ContractError("anchor count and feature chunk size are frozen at 128 and 16")
    if int(value["top_k"]) != 8 or float(value["softmax_temperature"]) != 1.0:
        raise ContractError("top-k/temperature contract mismatch")
    if value["stored_feature_dtype"] != "float32" or value["similarity_metric_dtype"] != "float64":
        raise ContractError("feature/metric dtype contract mismatch")
    for encoder in ENCODERS:
        spec = value["encoders"].get(encoder)
        if not isinstance(spec, dict) or spec.get("patch_shape") != [196, 768]:
            raise ContractError(f"{encoder} must bind an exact [196,768] patch tensor")
    if value["target_vlm"].get("connector_output_shape") != [196, 1536]:
        raise ContractError("native connector output shape must be [196,1536]")
    return value


def read_manifest(path: Path, expected_sha256: str | None = None) -> list[dict[str, Any]]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ContractError("manifest SHA-256 differs from the frozen configuration")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ContractError(f"blank manifest line at {line_number}")
            row = json.loads(line)
            required = {"image_id", "relative_path", "source_sha256", "split", "selection_index"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ContractError(f"invalid manifest row at {line_number}")
            image_id = str(row["image_id"])
            if image_id in seen_ids:
                raise ContractError(f"duplicate image_id: {image_id}")
            seen_ids.add(image_id)
            if row["split"] not in {"train", "validation", "test"}:
                raise ContractError(f"unsupported split at line {line_number}")
            rows.append(row)
    if len(rows) != 1000:
        raise ContractError(f"manifest must contain 1000 rows, observed {len(rows)}")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}
    if counts != {"train": 800, "validation": 100, "test": 100}:
        raise ContractError(f"manifest split counts differ: {counts}")
    selection_indices = [int(row["selection_index"]) for row in rows]
    if len(set(selection_indices)) != len(rows):
        raise ContractError("selection_index must be unique")
    return rows


def validate_identity_sources(config: Mapping[str, Any], repository_root: Path) -> None:
    repository = repository_root.resolve()
    bindings: list[tuple[str, str, str]] = []
    for encoder, spec in config["encoders"].items():
        for path_key, sha_key in (
            ("profile_path", "profile_sha256"),
            ("runtime_profile_path", "runtime_profile_sha256"),
            ("snapshot_profile_path", "snapshot_profile_sha256"),
            ("extractor_source", "extractor_sha256"),
            ("injection_boundary_source", "injection_boundary_sha256"),
        ):
            if path_key in spec:
                bindings.append((f"{encoder}.{path_key}", str(spec[path_key]), str(spec[sha_key])))
    target = config["target_vlm"]
    bindings.append(("target_vlm.config_path", str(target["config_path"]), str(target["config_sha256"])))
    bindings.append(("target_vlm.native_capture_source", str(target["native_capture_source"]), str(target["native_capture_sha256"])))
    bindings.append(("target_vlm.native_replay_source", str(target["native_replay_source"]), str(target["native_replay_sha256"])))
    for label, relative, expected_sha in bindings:
        candidate = (repository / relative).resolve()
        try:
            candidate.relative_to(repository)
        except ValueError as error:
            raise ContractError(f"identity path escapes repository: {label}") from error
        if not candidate.is_file() or sha256_file(candidate) != expected_sha:
            raise ContractError(f"identity source SHA mismatch: {label}")


def _chunks(values: Sequence[Any], size: int) -> Iterator[tuple[int, Sequence[Any]]]:
    for offset in range(0, len(values), size):
        yield offset // size, values[offset : offset + size]


def _manifest_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "image_id": str(row["image_id"]),
        "relative_path": str(row["relative_path"]),
        "source_sha256": str(row["source_sha256"]),
        "selection_index": int(row["selection_index"]),
        "split": str(row["split"]),
    }


def _unit(
    unit_id: str,
    stage: str,
    relative_path: str,
    *,
    encoder: str | None = None,
    condition: str | None = None,
    split: str | None = None,
    image_ids: Sequence[str] = (),
    shape: Any = None,
    dtype: Any = None,
    chunk_index: int | None = None,
    image_id: str | None = None,
) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "stage": stage,
        "artifact_relative_path": relative_path,
        "encoder": encoder,
        "condition": condition,
        "split": split,
        "image_ids": list(image_ids),
        "shape": shape,
        "dtype": dtype,
        "chunk_index": chunk_index,
        "image_id": image_id,
    }


def build_unit_inventory(rows: Sequence[Mapping[str, Any]], chunk_size: int = 16) -> list[dict[str, Any]]:
    ordered = list(rows)
    anchor_image_ids = [
        str(row["image_id"])
        for row in sorted(
            (row for row in ordered if row["split"] == "train"),
            key=lambda row: int(row["selection_index"]),
        )[:128]
    ]
    if len(anchor_image_ids) != 128 or len(set(anchor_image_ids)) != 128:
        raise ContractError("unit inventory requires exactly 128 distinct ordered train anchors")
    units: list[dict[str, Any]] = []
    stage_specs = (
        ("01_DINOV3_FEATURE_BANK", "DINOV3", "features/dinov3"),
        ("02_SIGLIP2_FEATURE_BANK", "SIGLIP2", "features/siglip2"),
        ("03_TARGET_CLIP_AND_NATIVE_CONNECTOR_BANK", "TARGET_CLIP", "features/target_clip"),
    )
    for stage, encoder, directory in stage_specs:
        for chunk_index, chunk in _chunks(ordered, chunk_size):
            ids = [str(row["image_id"]) for row in chunk]
            units.append(
                _unit(
                    f"{stage.lower()}-{chunk_index:04d}",
                    stage,
                    f"{directory}/chunk_{chunk_index:04d}.npz",
                    encoder=encoder,
                    image_ids=ids,
                    shape=(
                        {"patches": [len(ids), 196, 768], "globals": [len(ids), 768]}
                        if encoder != "TARGET_CLIP"
                        else {
                            "patches": [len(ids), 196, 768],
                            "globals": [len(ids), 768],
                            "connectors": [len(ids), 196, 1536],
                        }
                    ),
                    dtype={key: "float32" for key in (
                        ("patches", "globals") if encoder != "TARGET_CLIP" else ("patches", "globals", "connectors")
                    )},
                    chunk_index=chunk_index,
                )
            )
    registry_names = ("TARGET_DICTIONARY", "TARGET_CLIP", "SIGLIP2", "DINOV3")
    for name in registry_names:
        units.append(
            _unit(
                f"04_anchor_registry-{name.lower()}",
                "04_ANCHOR_REGISTRY",
                f"anchors/{name.lower()}.npz",
                encoder=name if name in ENCODERS else None,
                image_ids=anchor_image_ids,
                shape=(
                    {"patches": [128, 196, 768], "globals": [128, 768], "connectors": [128, 196, 1536]}
                    if name == "TARGET_DICTIONARY"
                    else {"patches": [128, 196, 768], "globals": [128, 768]}
                ),
                dtype={key: "float32" for key in (
                    ("patches", "globals", "connectors") if name == "TARGET_DICTIONARY" else ("patches", "globals")
                )},
            )
        )
    for split in ("validation", "test"):
        split_rows = [row for row in ordered if row["split"] == split]
        for chunk_index, chunk in _chunks(split_rows, chunk_size):
            ids = [str(row["image_id"]) for row in chunk]
            for encoder in ENCODERS:
                for condition in CONDITIONS:
                    slug = f"{split}-{encoder.lower()}-{condition.lower()}-{chunk_index:04d}"
                    units.append(
                        _unit(
                            f"05_transport-{slug}",
                            "05_TRANSPORT_AND_CONNECTOR_VALIDATION",
                            f"transport/{split}/{encoder.lower()}/{condition.lower()}/chunk_{chunk_index:04d}.npz",
                            encoder=encoder,
                            condition=condition,
                            split=split,
                            image_ids=ids,
                            shape={
                                "transported": [len(ids), 196, 768],
                                "connector": [len(ids), 196, 1536],
                                "native_patch": [len(ids), 196, 768],
                                "native_connector": [len(ids), 196, 1536],
                            },
                            dtype={
                                "transported": "float32",
                                "connector": "float32",
                                "native_patch": "float32",
                                "native_connector": "float32",
                                "metrics": "float64",
                            },
                            chunk_index=chunk_index,
                        )
                    )
    test_rows = [row for row in ordered if row["split"] == "test"]
    for test_index, row in enumerate(test_rows):
        image_id = str(row["image_id"])
        units.append(
            _unit(
                f"06_generation-{test_index:04d}-native",
                "06_HELD_OUT_GENERATION",
                f"generation/native/{test_index:04d}.json",
                split="test",
                image_ids=[image_id],
                image_id=image_id,
            )
        )
        for encoder in ENCODERS:
            for condition in CONDITIONS:
                slug = f"{encoder.lower()}-{condition.lower()}"
                units.append(
                    _unit(
                        f"06_generation-{test_index:04d}-{slug}",
                        "06_HELD_OUT_GENERATION",
                        f"generation/{encoder.lower()}/{condition.lower()}/{test_index:04d}.json",
                        encoder=encoder,
                        condition=condition,
                        split="test",
                        image_ids=[image_id],
                        image_id=image_id,
                    )
                )
    if len(units) != 2345:
        raise ContractError(f"unit inventory cardinality mismatch: expected 2345, observed {len(units)}")
    if len({unit["unit_id"] for unit in units}) != len(units):
        raise ContractError("unit inventory contains duplicate identifiers")
    return units


def estimate_storage_bytes(row_count: int, eval_count: int, generation_count: int) -> dict[str, int | float]:
    """Conservatively account for every persisted T001D array and checkpoint."""

    f32 = 4
    patch = 196 * 768 * f32
    global_mean = 768 * f32
    connector = 196 * 1536 * f32
    unicode_image_id = 96 * 4
    unicode_plan_id = 64 * 4
    npz_overhead = 4 * 1024
    feature_unit_count = math.ceil(row_count / 16) * len(ENCODERS)
    per_split = eval_count // 2
    if eval_count % 2 or per_split <= 0:
        raise ContractError("storage accounting requires equal nonempty validation/test counts")
    transport_unit_count = 2 * math.ceil(per_split / 16) * len(ENCODERS) * len(CONDITIONS)
    anchor_unit_count = 4
    unit_count = feature_unit_count + anchor_unit_count + transport_unit_count + generation_count

    feature_tensor_bytes = row_count * (3 * patch + 3 * global_mean + connector)
    feature_identity_bytes = row_count * len(ENCODERS) * unicode_image_id
    feature_container_bytes = feature_unit_count * (unicode_plan_id + npz_overhead)
    feature_banks = feature_tensor_bytes + feature_identity_bytes + feature_container_bytes

    anchor_tensor_bytes = 128 * (4 * patch + 4 * global_mean + connector)
    anchor_identity_bytes = 128 * anchor_unit_count * unicode_image_id
    anchor_container_bytes = anchor_unit_count * (unicode_plan_id + npz_overhead)
    anchor_registry = anchor_tensor_bytes + anchor_identity_bytes + anchor_container_bytes

    stage05_tensor_bytes = (
        eval_count
        * len(ENCODERS)
        * len(CONDITIONS)
        * 2
        * (patch + connector)
    )
    stage05_identity_bytes = eval_count * len(ENCODERS) * len(CONDITIONS) * unicode_image_id
    stage05_metric_bytes = transport_unit_count * 32768 * 4
    stage05_container_bytes = transport_unit_count * (unicode_plan_id + npz_overhead)
    transported = (
        stage05_tensor_bytes
        + stage05_identity_bytes
        + stage05_metric_bytes
        + stage05_container_bytes
    )

    generation_allowance = generation_count * 16 * 1024
    marker_allowance = unit_count * 16 * 1024
    progress_allowance = (2 * unit_count + 64) * 8 * 1024
    failure_final_allowance = 256 * 1024 * 1024
    metadata_allowance = marker_allowance + progress_allowance + failure_final_allowance
    durable_payload = (
        feature_banks
        + anchor_registry
        + transported
        + generation_allowance
        + metadata_allowance
    )

    largest_feature_partial = 16 * (patch + global_mean + connector) + 16 * unicode_image_id + unicode_plan_id + npz_overhead
    largest_anchor_partial = 128 * (patch + global_mean + connector) + 128 * unicode_image_id + unicode_plan_id + npz_overhead
    largest_stage05_partial = 16 * 2 * (patch + connector) + 16 * unicode_image_id + 32768 * 4 + unicode_plan_id + npz_overhead
    partial_allowance = max(
        largest_feature_partial,
        largest_anchor_partial,
        largest_stage05_partial,
        16 * 1024,
    )
    minimum = durable_payload + partial_allowance
    required = math.ceil(minimum * 1.30)

    legacy_feature = row_count * (3 * patch + connector)
    legacy_anchor = 128 * (4 * patch)
    legacy_transport = eval_count * len(ENCODERS) * len(CONDITIONS) * (patch + connector)
    legacy_durable = legacy_feature + legacy_anchor + legacy_transport + generation_allowance + 256 * 1024 * 1024
    legacy_partial = 16 * (patch + connector)
    legacy_minimum = legacy_durable + legacy_partial
    legacy_required = math.ceil(legacy_minimum * 1.30)
    return {
        "feature_banks_bytes": feature_banks,
        "feature_tensor_bytes": feature_tensor_bytes,
        "feature_identity_and_container_bytes": feature_identity_bytes + feature_container_bytes,
        "anchor_registry_bytes": anchor_registry,
        "anchor_tensor_bytes": anchor_tensor_bytes,
        "anchor_identity_and_container_bytes": anchor_identity_bytes + anchor_container_bytes,
        "transport_and_connector_bytes": transported,
        "stage05_tensor_bytes": stage05_tensor_bytes,
        "stage05_identity_metric_container_bytes": stage05_identity_bytes + stage05_metric_bytes + stage05_container_bytes,
        "generation_allowance_bytes": generation_allowance,
        "marker_allowance_bytes": marker_allowance,
        "progress_allowance_bytes": progress_allowance,
        "failure_and_final_allowance_bytes": failure_final_allowance,
        "metadata_allowance_bytes": metadata_allowance,
        "same_directory_partial_allowance_bytes": partial_allowance,
        "durable_payload_bytes": durable_payload,
        "minimum_bytes": minimum,
        "required_with_30_percent_margin_bytes": required,
        "required_with_30_percent_margin_gib": round(required / (1024**3), 6),
        "previous_underestimated_minimum_bytes": legacy_minimum,
        "previous_underestimated_required_bytes": legacy_required,
        "previous_underestimated_required_gib": round(legacy_required / (1024**3), 6),
    }


def build_run_plan(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    repository_commit: str,
    config_sha256: str,
    manifest_sha256: str,
    dataset_root: str,
    cache_root: str,
    repository_root: str,
    dino_runtime_root: str,
    dino_snapshot_root: str,
    run_root: str,
    device: str,
    storage_available_bytes: int | None,
    reuse_root: str | None = None,
    runtime_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if len(repository_commit) != 40:
        raise ContractError("repository commit must be an exact 40-character identity")
    ordered = list(rows)
    train = sorted(
        (row for row in ordered if row["split"] == "train"),
        key=lambda row: int(row["selection_index"]),
    )
    validation = [row for row in ordered if row["split"] == "validation"]
    test = [row for row in ordered if row["split"] == "test"]
    anchors = train[:128]
    anchor_ids = {str(row["image_id"]) for row in anchors}
    eval_ids = {str(row["image_id"]) for row in validation + test}
    if anchor_ids & eval_ids:
        raise ContractError("anchor and held-out image IDs overlap")
    units = build_unit_inventory(ordered, int(config["feature_chunk_size"]))
    storage = estimate_storage_bytes(len(ordered), len(validation) + len(test), 1900)
    if storage_available_bytes is not None and storage_available_bytes < int(storage["required_with_30_percent_margin_bytes"]):
        raise ContractError(
            "insufficient run-root storage: "
            f"required={storage['required_with_30_percent_margin_bytes']} available={storage_available_bytes}"
        )
    roots = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "cache_root": str(Path(cache_root).resolve()),
        "repository_root": str(Path(repository_root).resolve()),
        "dino_runtime_root": str(Path(dino_runtime_root).resolve()),
        "dino_snapshot_root": str(Path(dino_snapshot_root).resolve()),
        "run_root": str(Path(run_root).resolve()),
        "reuse_root": str(Path(reuse_root).resolve()) if reuse_root else None,
    }
    return {
        "schema_id": PLAN_SCHEMA,
        "task_id": "VEA-P2-T001D",
        "repository_commit": repository_commit,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "manifest_row_count": len(ordered),
        "manifest_split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "manifest_rows": [_manifest_binding(row) for row in ordered],
        "anchor_policy": "first_128_train_by_ascending_selection_index",
        "anchor_image_ids": [str(row["image_id"]) for row in anchors],
        "evaluation_image_ids": {
            "validation": [str(row["image_id"]) for row in validation],
            "test": [str(row["image_id"]) for row in test],
        },
        "stage_order": list(STAGES),
        "source_encoder_order": list(ENCODERS),
        "condition_order": list(CONDITIONS),
        "expected_generation_count": 1900,
        "feature_chunk_size": 16,
        "device": device,
        "roots": roots,
        "identity_bindings": {
            "encoders": config["encoders"],
            "target_vlm": config["target_vlm"],
        },
        "runtime_bindings": dict(runtime_bindings) if runtime_bindings is not None else None,
        "frozen_hyperparameters": {
            "top_k": config["top_k"],
            "softmax_temperature": config["softmax_temperature"],
            "rms_epsilon": config["rms_epsilon"],
            "prompt": config["prompt"],
            "generation": config["generation"],
            "runtime_dtype": config["runtime_dtype"],
            "stored_feature_dtype": config["stored_feature_dtype"],
            "similarity_metric_dtype": config["similarity_metric_dtype"],
        },
        "storage_preflight": {**storage, "available_bytes_at_plan": storage_available_bytes},
        "expected_units": units,
        "expected_unit_count": len(units),
        "artifact_schemas": {
            "unit": config["artifact_schema"],
            "done_marker": MARKER_SCHEMA,
            "resume": config["resume_schema"],
            "result": config["result_schema"],
        },
        "final_artifacts": list(FINAL_ARTIFACTS),
    }


def initialize_run(run_root: Path, plan: Mapping[str, Any]) -> str:
    if run_root.exists() and any(run_root.iterdir()):
        raise ContractError(f"run root must be absent or empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    for directory in (
        "state",
        "state/locks",
        "markers",
        "features/target_clip",
        "features/siglip2",
        "features/dinov3",
        "native_connectors",
        "anchors",
        "transport",
        "connectors",
        "generation/native",
        "generation/target_clip",
        "generation/siglip2",
        "generation/dinov3",
        "metrics",
        "logs",
        "failures",
        "failures/quarantine",
        "final",
    ):
        (run_root / directory).mkdir(parents=True, exist_ok=True)
    plan_bytes = canonical_json_bytes(dict(plan))
    plan_id = sha256_bytes(plan_bytes)
    atomic_write_bytes(run_root / "run_plan.json", plan_bytes)
    state = {
        "schema_id": STATE_SCHEMA,
        "plan_id": plan_id,
        "created_at_utc": utc_now(),
        "expected_unit_count": plan["expected_unit_count"],
    }
    atomic_write_json(run_root / "state" / "plan_binding.json", state)
    append_event(run_root, progress_event(plan_id, "PLAN_CREATED", expected_unit_count=plan["expected_unit_count"]))
    return plan_id


def load_bound_plan(run_root: Path) -> tuple[dict[str, Any], str]:
    plan_path = run_root / "run_plan.json"
    state_path = run_root / "state" / "plan_binding.json"
    if not plan_path.is_file() or not state_path.is_file():
        raise PlanMismatchError("run_plan.json and state/plan_binding.json are both required")
    plan_bytes = plan_path.read_bytes()
    plan_id = sha256_bytes(plan_bytes)
    state = strict_json_load(state_path)
    if state.get("schema_id") != STATE_SCHEMA or state.get("plan_id") != plan_id:
        raise PlanMismatchError("run plan bytes do not match the bound plan_id")
    try:
        plan = json.loads(plan_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanMismatchError("run plan is not valid canonical JSON") from error
    if canonical_json_bytes(plan) != plan_bytes:
        raise PlanMismatchError("run plan is not in canonical immutable form")
    if plan.get("schema_id") != PLAN_SCHEMA:
        raise PlanMismatchError("run plan schema mismatch")
    return plan, plan_id


def validate_resume_bindings(
    plan: Mapping[str, Any],
    *,
    repository_commit: str,
    config_sha256: str,
    manifest_sha256: str,
    roots: Mapping[str, str | None],
    device: str,
    runtime_bindings: Mapping[str, Any] | None = None,
) -> None:
    expected = {
        "repository_commit": plan["repository_commit"],
        "config_sha256": plan["config_sha256"],
        "manifest_sha256": plan["manifest_sha256"],
        "device": plan["device"],
    }
    observed = {
        "repository_commit": repository_commit,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "device": device,
    }
    if expected != observed:
        raise PlanMismatchError(f"resume identity mismatch: expected={expected} observed={observed}")
    planned_roots = plan["roots"]
    if set(planned_roots) != set(roots):
        raise PlanMismatchError(
            f"resume root keys mismatch: expected={sorted(planned_roots)} observed={sorted(roots)}"
        )
    for name, planned in planned_roots.items():
        value = roots[name]
        if planned is None or value is None:
            if planned is value:
                continue
            raise PlanMismatchError(f"resume root mismatch for {name}: expected={planned} observed={value}")
        resolved = str(Path(value).resolve())
        if planned != resolved:
            raise PlanMismatchError(f"resume root mismatch for {name}: expected={planned} observed={resolved}")
    if plan.get("runtime_bindings") != (dict(runtime_bindings) if runtime_bindings is not None else None):
        raise PlanMismatchError("resume runtime/overlay binding differs from immutable plan")


def marker_path(run_root: Path, unit_id: str) -> Path:
    return run_root / "markers" / f"{unit_id}.done.json"


def validate_unit_completion(run_root: Path, unit: Mapping[str, Any], plan_id: str) -> bool:
    marker = marker_path(run_root, str(unit["unit_id"]))
    artifact = run_root / str(unit["artifact_relative_path"])
    if not marker.exists():
        return False
    if not artifact.is_file():
        raise CorruptionError(f"done marker exists but artifact is missing: {unit['unit_id']}")
    value = strict_json_load(marker)
    required = {
        "schema_id",
        "plan_id",
        "unit_id",
        "stage",
        "artifact_relative_path",
        "artifact_sha256",
        "artifact_size_bytes",
        "artifact_bytes",
        "encoder",
        "condition",
        "split",
        "image_ids",
        "shape",
        "dtype",
        "chunk_index",
        "image_id",
        "elapsed_seconds",
        "completed_at_utc",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise CorruptionError(f"invalid done marker fields: {unit['unit_id']}")
    exact = {
        "schema_id": MARKER_SCHEMA,
        "plan_id": plan_id,
        "unit_id": unit["unit_id"],
        "stage": unit["stage"],
        "artifact_relative_path": unit["artifact_relative_path"],
        "encoder": unit["encoder"],
        "condition": unit["condition"],
        "split": unit["split"],
        "image_ids": unit["image_ids"],
        "shape": unit["shape"],
        "dtype": unit["dtype"],
        "chunk_index": unit["chunk_index"],
        "image_id": unit["image_id"],
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise CorruptionError(f"marker {key} mismatch for {unit['unit_id']}")
    if value["artifact_size_bytes"] != artifact.stat().st_size or value["artifact_bytes"] != artifact.stat().st_size:
        raise CorruptionError(f"artifact size mismatch for {unit['unit_id']}")
    if value["artifact_sha256"] != sha256_file(artifact):
        raise CorruptionError(f"artifact hash mismatch for {unit['unit_id']}")
    if not isinstance(value["elapsed_seconds"], (int, float)) or value["elapsed_seconds"] < 0:
        raise CorruptionError(f"invalid elapsed time for {unit['unit_id']}")
    return True


def atomic_publish_unit(
    run_root: Path,
    unit: Mapping[str, Any],
    payload: bytes,
    plan_id: str,
    *,
    elapsed_seconds: float,
    validate_payload: Callable[[bytes, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if validate_unit_completion(run_root, unit, plan_id):
        return strict_json_load(marker_path(run_root, str(unit["unit_id"])))
    artifact = run_root / str(unit["artifact_relative_path"])
    if artifact.exists():
        raise CorruptionError(f"unmarked artifact must be quarantined before publish: {artifact}")
    artifact_sha = atomic_write_bytes(
        artifact,
        payload,
        (lambda observed: validate_payload(observed, unit)) if validate_payload is not None else None,
    )
    marker = {
        "schema_id": MARKER_SCHEMA,
        "plan_id": plan_id,
        "unit_id": unit["unit_id"],
        "stage": unit["stage"],
        "artifact_relative_path": unit["artifact_relative_path"],
        "artifact_sha256": artifact_sha,
        "artifact_size_bytes": len(payload),
        "artifact_bytes": len(payload),
        "encoder": unit["encoder"],
        "condition": unit["condition"],
        "split": unit["split"],
        "image_ids": unit["image_ids"],
        "shape": unit["shape"],
        "dtype": unit["dtype"],
        "chunk_index": unit["chunk_index"],
        "image_id": unit["image_id"],
        "elapsed_seconds": round(float(elapsed_seconds), 9),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(marker_path(run_root, str(unit["unit_id"])), marker)
    append_event(
        run_root,
        progress_event(
            plan_id,
            "UNIT_COMPLETED",
            unit_id=unit["unit_id"],
            stage=unit["stage"],
            artifact_sha256=artifact_sha,
        ),
    )
    return marker


def _quarantine_destination(run_root: Path, path: Path, reason: str) -> Path:
    relative = path.relative_to(run_root).as_posix().replace("/", "__")
    digest = hashlib.sha256(f"{relative}:{time.time_ns()}".encode("utf-8")).hexdigest()[:12]
    return run_root / "failures" / "quarantine" / f"{reason}__{digest}__{path.name}"


def quarantine_path(run_root: Path, path: Path, reason: str, plan_id: str) -> Path:
    try:
        path.relative_to(run_root)
    except ValueError as error:
        raise ContractError("refusing to quarantine a path outside the run root") from error
    destination = _quarantine_destination(run_root, path, reason)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, destination)
    _fsync_directory(destination.parent)
    return destination


def recover_orphans(run_root: Path, plan: Mapping[str, Any], plan_id: str) -> list[str]:
    quarantined: list[str] = []
    for path in sorted(run_root.rglob("*.partial.*")):
        if "failures" in path.parts:
            continue
        destination = quarantine_path(run_root, path, "orphan_partial", plan_id)
        quarantined.append(destination.relative_to(run_root).as_posix())
    for unit in plan["expected_units"]:
        artifact = run_root / unit["artifact_relative_path"]
        marker = marker_path(run_root, unit["unit_id"])
        if artifact.exists() and not marker.exists():
            destination = quarantine_path(run_root, artifact, "unmarked_artifact", plan_id)
            quarantined.append(destination.relative_to(run_root).as_posix())
        elif marker.exists():
            validate_unit_completion(run_root, unit, plan_id)
    return quarantined


def inspect_exact_feature_reuse(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Read and validate an explicit reuse source without copying or mutating it."""

    reuse_value = plan["roots"].get("reuse_root")
    if reuse_value is None:
        return {
            "status": "NO_REUSE_ROOT",
            "reuse_root": None,
            "source_plan_id": None,
            "source_repository_commit": None,
            "source_completed_unit_count": 0,
            "eligible_feature_unit_count": 0,
            "eligible_unit_ids": [],
            "incompatibilities": [],
        }
    reuse_root = Path(reuse_value)
    try:
        old_plan, old_plan_id = load_bound_plan(reuse_root)
    except HarnessError as error:
        raise CorruptionError(f"explicit reuse source plan binding is invalid: {reuse_root}") from error
    identity_fields = (
        "schema_id",
        "manifest_sha256",
        "manifest_rows",
        "identity_bindings",
        "feature_chunk_size",
        "artifact_schemas",
    )
    incompatibilities = [name for name in identity_fields if old_plan.get(name) != plan.get(name)]
    feature_units = [unit for unit in plan["expected_units"] if unit["stage"] in STAGES[1:4]]
    old_feature_units = [unit for unit in old_plan["expected_units"] if unit["stage"] in STAGES[1:4]]
    if old_feature_units != feature_units:
        incompatibilities.append("stage_01_03_unit_definitions")
    base = {
        "reuse_root": str(reuse_root),
        "source_plan_id": old_plan_id,
        "source_repository_commit": old_plan.get("repository_commit"),
    }
    if incompatibilities:
        return {
            "status": "INCOMPATIBLE",
            **base,
            "source_completed_unit_count": 0,
            "eligible_feature_unit_count": 0,
            "eligible_unit_ids": [],
            "incompatibilities": sorted(set(incompatibilities)),
        }
    completed_unit_ids: set[str] = set()
    for old_unit in old_plan["expected_units"]:
        artifact = reuse_root / old_unit["artifact_relative_path"]
        marker = marker_path(reuse_root, old_unit["unit_id"])
        if artifact.exists() != marker.exists():
            raise CorruptionError(f"reuse marker/artifact mismatch for {old_unit['unit_id']}")
        if marker.exists():
            if not validate_unit_completion(reuse_root, old_unit, old_plan_id):
                raise CorruptionError(f"reuse completion validation failed for {old_unit['unit_id']}")
            completed_unit_ids.add(str(old_unit["unit_id"]))
    eligible = [str(unit["unit_id"]) for unit in feature_units if str(unit["unit_id"]) in completed_unit_ids]
    return {
        "status": "ELIGIBLE",
        **base,
        "source_completed_unit_count": len(completed_unit_ids),
        "eligible_feature_unit_count": len(eligible),
        "eligible_unit_ids": eligible,
        "incompatibilities": [],
    }


def reuse_exact_feature_units(run_root: Path, plan: Mapping[str, Any], plan_id: str) -> dict[str, int]:
    """Publish only the exact eligible units returned by the read-only inspector."""

    inspection = inspect_exact_feature_reuse(plan)
    if inspection["status"] == "NO_REUSE_ROOT":
        return {"eligible": 0, "reused": 0, "ignored": 0}
    if inspection["status"] == "INCOMPATIBLE":
        return {"eligible": 0, "reused": 0, "ignored": 1}
    reuse_root = Path(str(inspection["reuse_root"]))
    old_plan_id = str(inspection["source_plan_id"])
    new_units = {unit["unit_id"]: unit for unit in plan["expected_units"]}
    reused = 0
    for unit_id in inspection["eligible_unit_ids"]:
        unit = new_units[unit_id]
        if validate_unit_completion(run_root, unit, plan_id):
            continue
        if not validate_unit_completion(reuse_root, unit, old_plan_id):
            raise CorruptionError(f"inspected reuse unit disappeared before publication: {unit_id}")
        source_path = reuse_root / unit["artifact_relative_path"]
        payload = source_path.read_bytes()
        if source_path.suffix == ".npz":
            np = _numpy()
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files if name != "plan_id"}
            arrays["plan_id"] = np.asarray(plan_id, dtype="U64")
            buffer = io.BytesIO()
            np.savez(buffer, **arrays)
            payload = buffer.getvalue()
        atomic_publish_unit(run_root, unit, payload, plan_id, elapsed_seconds=0.0)
        reused += 1
    if reused:
        append_event(
            run_root,
            progress_event(plan_id, "RESUMED", reuse_root=str(reuse_root), reused_feature_units=reused),
        )
    return {"eligible": int(inspection["eligible_feature_unit_count"]), "reused": reused, "ignored": 0}


def current_process_start_identity(pid: int | None = None) -> str | None:
    pid = os.getpid() if pid is None else pid
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return None
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            try:
                if not ctypes.windll.kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ):
                    return None
                ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                return f"windows-filetime:{ticks}"
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except (AttributeError, OSError):
            return None
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        fields = proc_stat.read_text(encoding="utf-8").split()
        return f"proc-start-ticks:{fields[21]}"
    except (OSError, IndexError):
        return None


def process_liveness(host: str, pid: int, start_identity: str | None) -> bool | None:
    if host != socket.gethostname():
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return False
    if start_identity is None:
        return None
    observed = current_process_start_identity(pid)
    if observed is None:
        return None
    return observed == start_identity


@dataclass
class CoordinatorLock:
    run_root: Path
    plan_id: str
    identity_provider: Callable[[int | None], str | None] = current_process_start_identity
    liveness_provider: Callable[[str, int, str | None], bool | None] = process_liveness

    def __post_init__(self) -> None:
        self.path = self.run_root / "coordinator.lock"
        self.record: dict[str, Any] | None = None

    def acquire(self) -> None:
        record = {
            "schema_id": LOCK_SCHEMA,
            "plan_id": self.plan_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "process_start_identity": self.identity_provider(os.getpid()),
            "started_at_utc": utc_now(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            existing = strict_json_load(self.path)
            if existing.get("plan_id") != self.plan_id:
                raise PlanMismatchError("coordinator lock carries a different plan_id")
            live = self.liveness_provider(
                str(existing.get("host")),
                int(existing.get("pid")),
                existing.get("process_start_identity"),
            )
            if live is True:
                raise LockBusyError("a live coordinator already owns this run")
            if live is None:
                raise AmbiguousLockError("coordinator liveness cannot be proven")
            quarantine_path(self.run_root, self.path, "proven_stale_lock", self.plan_id)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            payload = canonical_json_bytes(record)
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.path.parent)
        self.record = record

    def release(self) -> None:
        if self.record is None:
            return
        if not self.path.exists():
            self.record = None
            return
        existing = strict_json_load(self.path)
        ownership = ("plan_id", "host", "pid", "process_start_identity", "started_at_utc")
        if any(existing.get(key) != self.record.get(key) for key in ownership):
            raise AmbiguousLockError("coordinator lock ownership changed before release")
        self.path.unlink()
        _fsync_directory(self.path.parent)
        self.record = None

    def __enter__(self) -> "CoordinatorLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def _numpy() -> Any:
    import numpy as np

    return np


def validate_patch_batch(value: Any, *, name: str, batch_size: int | None = None) -> Any:
    np = _numpy()
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise ContractError(f"{name} must be float32, observed {array.dtype}")
    if array.ndim != 3 or array.shape[1:] != (196, 768):
        raise ContractError(f"{name} must have shape [N,196,768], observed {array.shape}")
    if batch_size is not None and array.shape[0] != batch_size:
        raise ContractError(f"{name} batch size mismatch: expected {batch_size}, observed {array.shape[0]}")
    if not array.flags.c_contiguous or not array.flags.writeable:
        raise ContractError(f"{name} must be writable and C-contiguous")
    if not np.isfinite(array).all():
        raise ContractError(f"{name} contains non-finite values")
    return array


def sample_rms(source: Any, *, epsilon: float = 1e-6) -> Any:
    """Normalize each patch token by its own channel RMS; use no anchors."""

    np = _numpy()
    source = validate_patch_batch(source, name="source")
    source64 = source.astype(np.float64, copy=False)
    rms = np.sqrt(np.mean(source64 * source64, axis=2, keepdims=True) + float(epsilon))
    return np.ascontiguousarray((source64 / rms).astype(np.float32))


def _cosine_rows(query: Any, keys: Any) -> Any:
    np = _numpy()
    query64 = np.asarray(query, dtype=np.float64)
    keys64 = np.asarray(keys, dtype=np.float64)
    qnorm = np.linalg.norm(query64, axis=1, keepdims=True)
    knorm = np.linalg.norm(keys64, axis=1, keepdims=True).T
    return (query64 @ keys64.T) / np.maximum(qnorm * knorm, np.finfo(np.float64).tiny)


def _stable_topk_weights(similarity: Any, k: int, temperature: float) -> tuple[Any, Any]:
    np = _numpy()
    if temperature <= 0:
        raise ContractError("softmax temperature must be positive")
    # Stable sorting makes lower anchor indices win all similarity ties.
    indices = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    selected = np.take_along_axis(similarity, indices, axis=1) / temperature
    selected = selected - selected.max(axis=1, keepdims=True)
    weights = np.exp(selected)
    weights /= weights.sum(axis=1, keepdims=True)
    return indices, weights


def global_anchor_transport(source: Any, source_anchor: Any, target_anchor: Any, *, top_k: int) -> Any:
    np = _numpy()
    source = validate_patch_batch(source, name="source")
    source_anchor = validate_patch_batch(source_anchor, name="source_anchor")
    target_anchor = validate_patch_batch(target_anchor, name="target_anchor", batch_size=source_anchor.shape[0])
    query = source.astype(np.float64).mean(axis=1)
    keys = source_anchor.astype(np.float64).mean(axis=1)
    similarity = _cosine_rows(query, keys)
    indices, weights = _stable_topk_weights(similarity, top_k, 1.0)
    target64 = target_anchor.astype(np.float64)
    result = np.empty(source.shape, dtype=np.float64)
    for row in range(source.shape[0]):
        result[row] = np.tensordot(weights[row], target64[indices[row]], axes=(0, 0))
    return np.ascontiguousarray(result.astype(np.float32))


def positional_anchor_transport(source: Any, source_anchor: Any, target_anchor: Any, *, top_k: int) -> Any:
    np = _numpy()
    source = validate_patch_batch(source, name="source")
    source_anchor = validate_patch_batch(source_anchor, name="source_anchor")
    target_anchor = validate_patch_batch(target_anchor, name="target_anchor", batch_size=source_anchor.shape[0])
    result = np.empty(source.shape, dtype=np.float64)
    source64 = source.astype(np.float64)
    source_anchor64 = source_anchor.astype(np.float64)
    target_anchor64 = target_anchor.astype(np.float64)
    for position in range(196):
        similarity = _cosine_rows(source64[:, position, :], source_anchor64[:, position, :])
        indices, weights = _stable_topk_weights(similarity, top_k, 1.0)
        for row in range(source.shape[0]):
            result[row, position] = np.tensordot(
                weights[row], target_anchor64[indices[row], position], axes=(0, 0)
            )
    return np.ascontiguousarray(result.astype(np.float32))


def apply_transport_condition(
    condition: str,
    source: Any,
    source_anchor: Any,
    target_anchor: Any,
    *,
    rms_epsilon: float = 1e-6,
) -> Any:
    source = validate_patch_batch(source, name="source")
    if condition == "RAW":
        return source.copy(order="C")
    if condition == "SAMPLE_RMS":
        return sample_rms(source, epsilon=rms_epsilon)
    if condition == "GLOBAL_1NN":
        return global_anchor_transport(source, source_anchor, target_anchor, top_k=1)
    if condition == "GLOBAL_TOP8":
        return global_anchor_transport(source, source_anchor, target_anchor, top_k=8)
    if condition == "POSITIONAL_1NN":
        return positional_anchor_transport(source, source_anchor, target_anchor, top_k=1)
    if condition == "POSITIONAL_TOP8":
        return positional_anchor_transport(source, source_anchor, target_anchor, top_k=8)
    raise ContractError(f"unknown transport condition: {condition}")


def representation_metrics(candidate: Any, reference: Any, nmse_denominator: Any | None = None) -> dict[str, float]:
    np = _numpy()
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    for name, value in (("candidate", candidate), ("reference", reference)):
        if value.dtype != np.float32 or value.ndim != 3 or not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ContractError(f"{name} metric tensor must be finite contiguous float32 [N,T,D]")
    if candidate.shape != reference.shape:
        raise ContractError("candidate/reference metric tensors must have identical shapes")
    a = candidate.astype(np.float64).reshape(candidate.shape[0], -1)
    b = reference.astype(np.float64).reshape(reference.shape[0], -1)
    cosine = np.sum(a * b, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), np.finfo(np.float64).tiny
    )
    denominator_source = reference if nmse_denominator is None else np.asarray(nmse_denominator)
    if denominator_source.dtype != np.float32 or denominator_source.ndim != 3 or not np.isfinite(denominator_source).all():
        raise ContractError("NMSE denominator bank must be finite float32 [N,T,D]")
    if denominator_source.shape[1:] != candidate.shape[1:]:
        raise ContractError("NMSE denominator feature geometry mismatch")
    anchor_mean = np.asarray(denominator_source, dtype=np.float64).mean(axis=(0, 1))
    denominator = np.sum(
        (reference.astype(np.float64) - anchor_mean.reshape(1, 1, -1)) ** 2,
        axis=(1, 2),
    )
    if np.any(denominator == 0.0):
        raise ContractError("NMSE target-anchor-mean denominator is zero")
    nmse = np.sum((a - b) ** 2, axis=1) / denominator
    candidate_norm = np.linalg.norm(a, axis=1)
    reference_norm = np.linalg.norm(b, axis=1)
    norm_ratio = candidate_norm / np.maximum(reference_norm, np.finfo(np.float64).tiny)
    return {
        "flattened_cosine_mean": float(cosine.mean()),
        "norm_ratio_mean": float(norm_ratio.mean()),
        "nmse_mean": float(nmse.mean()),
    }


def retrieval_recall(candidate: Any, reference: Any) -> dict[str, float]:
    np = _numpy()
    candidate = validate_patch_batch(candidate, name="candidate")
    reference = validate_patch_batch(reference, name="reference", batch_size=candidate.shape[0])
    a = candidate.astype(np.float64).mean(axis=1)
    b = reference.astype(np.float64).mean(axis=1)
    similarity = _cosine_rows(a, b)
    order = np.argsort(-similarity, axis=1, kind="stable")
    correct = np.arange(candidate.shape[0])[:, None]
    return {
        "recall_at_1": float(np.mean(np.any(order[:, :1] == correct, axis=1))),
        "recall_at_5": float(np.mean(np.any(order[:, : min(5, order.shape[1])] == correct, axis=1))),
    }


def generation_metrics(candidate_tokens: Sequence[str], native_tokens: Sequence[str]) -> dict[str, float | int]:
    prefix = 0
    for candidate, native in zip(candidate_tokens, native_tokens):
        if candidate != native:
            break
        prefix += 1
    previous = [0] * (len(native_tokens) + 1)
    for candidate in candidate_tokens:
        current = [0]
        for column, native in enumerate(native_tokens, 1):
            current.append(previous[column - 1] + 1 if candidate == native else max(previous[column], current[-1]))
        previous = current
    lcs = previous[-1]
    from collections import Counter

    candidate_counts = Counter(candidate_tokens)
    native_counts = Counter(native_tokens)
    overlap = sum((candidate_counts & native_counts).values())
    precision = overlap / len(candidate_tokens) if candidate_tokens else 0.0
    recall = overlap / len(native_tokens) if native_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_match": int(list(candidate_tokens) == list(native_tokens)),
        "common_token_prefix_length": prefix,
        "common_token_prefix_ratio": prefix / len(native_tokens) if native_tokens else float(not candidate_tokens),
        "token_LCS_length": lcs,
        "token_LCS_ratio": lcs / len(native_tokens) if native_tokens else float(not candidate_tokens),
        "token_unigram_F1": f1,
        "generated_token_count": len(candidate_tokens),
        "absolute_token_count_difference": abs(len(candidate_tokens) - len(native_tokens)),
    }


def deterministic_fake_payload(plan_id: str, unit: Mapping[str, Any]) -> bytes:
    seed = sha256_bytes(canonical_json_bytes({"plan_id": plan_id, "unit": unit}))
    return canonical_json_bytes(
        {
            "schema_id": UNIT_SCHEMA,
            "plan_id": plan_id,
            "unit_id": unit["unit_id"],
            "stage": unit["stage"],
            "encoder": unit["encoder"],
            "condition": unit["condition"],
            "split": unit["split"],
            "image_ids": unit["image_ids"],
            "shape": unit["shape"],
            "dtype": unit["dtype"],
            "fake_cpu_digest": seed,
        }
    )


def execute_fake_units(
    run_root: Path,
    *,
    interrupt_after_new_units: int | None = None,
    leave_orphan_partial: bool = False,
) -> dict[str, int]:
    plan, plan_id = load_bound_plan(run_root)
    completed_before = 0
    completed_new = 0
    with CoordinatorLock(run_root, plan_id):
        quarantined = recover_orphans(run_root, plan, plan_id)
        existing = sum(
            validate_unit_completion(run_root, unit, plan_id) for unit in plan["expected_units"]
        )
        append_event(
            run_root,
            progress_event(
                plan_id,
                "RESUMED" if existing else "RUN_STARTED",
                completed_before=existing,
                quarantined_paths=quarantined,
            ),
        )
        for stage in STAGES[1:7]:
            stage_units = [unit for unit in plan["expected_units"] if unit["stage"] == stage]
            incomplete = [unit for unit in stage_units if not validate_unit_completion(run_root, unit, plan_id)]
            if not incomplete:
                completed_before += len(stage_units)
                continue
            append_event(run_root, progress_event(plan_id, "STAGE_STARTED", stage=stage))
            for unit in stage_units:
                if validate_unit_completion(run_root, unit, plan_id):
                    completed_before += 1
                    continue
                append_event(
                    run_root,
                    progress_event(plan_id, "UNIT_STARTED", stage=stage, unit_id=unit["unit_id"]),
                )
                payload = deterministic_fake_payload(plan_id, unit)
                atomic_publish_unit(run_root, unit, payload, plan_id, elapsed_seconds=0.0)
                completed_new += 1
                if interrupt_after_new_units is not None and completed_new == interrupt_after_new_units:
                    if leave_orphan_partial:
                        artifact = run_root / unit["artifact_relative_path"]
                        orphan = artifact.with_name(f"{artifact.name}.partial.injected")
                        orphan.write_bytes(b"injected interrupted partial")
                    append_event(
                        run_root,
                        progress_event(plan_id, "INTERRUPTED", completed_new=completed_new),
                    )
                    raise InjectedInterruption(f"injected after {completed_new} new units")
            append_event(run_root, progress_event(plan_id, "STAGE_COMPLETED", stage=stage))
    return {"completed_before": completed_before, "completed_new": completed_new}


def verify_run(run_root: Path, *, require_final: bool = False) -> dict[str, Any]:
    plan, plan_id = load_bound_plan(run_root)
    completed = 0
    for unit in plan["expected_units"]:
        if validate_unit_completion(run_root, unit, plan_id):
            completed += 1
    missing_final = [name for name in FINAL_ARTIFACTS if not (run_root / "final" / name).is_file()]
    complete_units = completed == plan["expected_unit_count"]
    if require_final and missing_final:
        raise IncompleteRunError(f"final artifacts missing: {missing_final}")
    return {
        "plan_id": plan_id,
        "expected_unit_count": plan["expected_unit_count"],
        "completed_unit_count": completed,
        "remaining_unit_count": plan["expected_unit_count"] - completed,
        "complete_units": complete_units,
        "missing_final_artifacts": missing_final,
        "status": "PASS" if complete_units and (not require_final or not missing_final) else "PARTIAL",
    }


def _csv_bytes(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _read_resume_counts(run_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    path = run_root / "progress.jsonl"
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            event_type = str(event.get("event_type"))
            counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _runtime_counts(run_root: Path) -> dict[str, Any]:
    totals = {
        "model_load_count": 0,
        "stage_worker_invocation_count": 0,
        "extractor_worker_invocation_count": 0,
        "by_stage": {},
    }
    path = run_root / "progress.jsonl"
    if not path.exists():
        return totals
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event_type") != "STAGE_COMPLETED" or "model_load_count" not in event:
                continue
            stage = str(event["stage"])
            row = totals["by_stage"].setdefault(
                stage,
                {"model_load_count": 0, "stage_worker_invocation_count": 0, "extractor_worker_invocation_count": 0},
            )
            for field in ("model_load_count", "stage_worker_invocation_count", "extractor_worker_invocation_count"):
                value = int(event.get(field, 0))
                totals[field] += value
                row[field] += value
    return totals


def _aggregate_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import statistics

    coordinates: dict[tuple[str, str, str, str], list[tuple[str, float]]] = {}
    identity = {
        "image_id", "encoder", "condition", "split", "record_type", "plan_id",
        "schema_id", "unit_id", "token_ids", "decoded_text", "native_answer",
        "condition_answer", "injection_calls",
    }
    for record in records:
        if record.get("encoder") is None or record.get("condition") == "NATIVE":
            continue
        for metric, value in record.items():
            if metric in identity or not isinstance(value, (int, float)):
                continue
            key = (str(record["encoder"]), str(record["condition"]), str(record["split"]), metric)
            coordinates.setdefault(key, []).append((str(record["image_id"]), float(value)))
    raw_lookup: dict[tuple[str, str, str, str], float] = {}
    for (encoder, condition, split, metric), values in coordinates.items():
        if condition == "RAW":
            for image_id, value in values:
                raw_lookup[(encoder, split, metric, image_id)] = value
    rows = []
    for key in sorted(coordinates):
        encoder, condition, split, metric = key
        pairs = coordinates[key]
        values = [value for _, value in pairs]
        improved = 0
        for image_id, value in pairs:
            raw = raw_lookup.get((encoder, split, metric, image_id))
            if raw is None or condition == "RAW":
                continue
            metric_lower = metric.lower()
            if "norm_ratio" in metric_lower:
                improved += int(abs(value - 1.0) < abs(raw - 1.0))
            elif any(token in metric_lower for token in ("cosine", "f1", "equal", "prefix", "lcs", "generation_success")):
                improved += int(value > raw)
            else:
                improved += int(value < raw)
        rows.append(
            {
                "encoder": encoder,
                "condition": condition,
                "split": split,
                "metric": metric,
                "count": len(values),
                "mean": sum(values) / len(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
                "count_improved_over_raw": improved,
            }
        )
    return rows


def _collect_real_outputs(run_root: Path, plan: Mapping[str, Any], plan_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Collect persisted scientific rows; fake artifacts take a deterministic CPU path."""

    stage_five = [unit for unit in plan["expected_units"] if unit["stage"] == "05_TRANSPORT_AND_CONNECTOR_VALIDATION"]
    first_payload = (run_root / stage_five[0]["artifact_relative_path"]).read_bytes()
    if first_payload.startswith(b"{"):
        fake_records = [
            {
                "image_id": image_id,
                "encoder": encoder,
                "condition": condition,
                "split": split,
                "record_type": "fake_transport",
                "plan_id": plan_id,
            }
            for split in ("validation", "test")
            for image_id in plan["evaluation_image_ids"][split]
            for encoder in ENCODERS
            for condition in CONDITIONS
        ]
        fake_summary = [
            {
                "encoder": encoder,
                "condition": condition,
                "split": split,
                "metric": "fake_contract_only",
                "count": len(plan["evaluation_image_ids"][split]),
                "mean": 0.0,
                "median": 0.0,
                "minimum": 0.0,
                "maximum": 0.0,
                "count_improved_over_raw": 0,
            }
            for split in ("validation", "test")
            for encoder in ENCODERS
            for condition in CONDITIONS
        ]
        fake_retrieval = [
            {
                "encoder": encoder,
                "condition": condition,
                "split": split,
                "recall_at_1": 0.0,
                "recall_at_5": 0.0,
                "plan_id": plan_id,
            }
            for split in ("validation", "test")
            for encoder in ENCODERS
            for condition in CONDITIONS
        ]
        return fake_records, fake_summary, fake_retrieval, True

    np = _numpy()
    records: list[dict[str, Any]] = []
    retrieval_groups: dict[tuple[str, str, str], tuple[list[Any], list[Any]]] = {}
    for unit in stage_five:
        with np.load(run_root / unit["artifact_relative_path"], allow_pickle=False) as archive:
            metric_rows = json.loads(str(archive["per_image_metrics_json"].item()))
            transported = np.array(archive["transported"], dtype=np.float32, copy=True, order="C")
            native_patch = np.array(archive["native_patch"], dtype=np.float32, copy=True, order="C")
        for metric_row in metric_rows:
            records.append(
                {
                    "record_type": "transport",
                    "plan_id": plan_id,
                    "encoder": unit["encoder"],
                    "condition": unit["condition"],
                    "split": unit["split"],
                    **metric_row,
                }
            )
        key = (unit["encoder"], unit["condition"], unit["split"])
        pair = retrieval_groups.setdefault(key, ([], []))
        pair[0].append(transported)
        pair[1].append(native_patch)
    retrieval_rows = []
    for (encoder, condition, split), (candidate_chunks, reference_chunks) in sorted(retrieval_groups.items()):
        metrics = retrieval_recall(
            np.ascontiguousarray(np.concatenate(candidate_chunks)),
            np.ascontiguousarray(np.concatenate(reference_chunks)),
        )
        retrieval_rows.append(
            {
                "encoder": encoder,
                "condition": condition,
                "split": split,
                **metrics,
                "plan_id": plan_id,
            }
        )
    for unit in plan["expected_units"]:
        if unit["stage"] != "06_HELD_OUT_GENERATION":
            continue
        generation = strict_json_load(run_root / unit["artifact_relative_path"])
        records.append(
            {
                "record_type": "generation",
                "plan_id": plan_id,
                "split": "test",
                **generation,
            }
        )
    return records, _aggregate_rows(records), retrieval_rows, False


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> tuple[str, ...]:
    fields = set().union(*(row.keys() for row in rows)) if rows else set(preferred)
    return tuple([name for name in preferred if name in fields] + sorted(fields - set(preferred)))


def finalize_run(run_root: Path) -> dict[str, Any]:
    plan, plan_id = load_bound_plan(run_root)
    verification = verify_run(run_root)
    if not verification["complete_units"]:
        raise IncompleteRunError(
            f"cannot finalize: {verification['remaining_unit_count']} expected units remain"
        )
    with CoordinatorLock(run_root, plan_id):
        final_root = run_root / "final"
        final_root.mkdir(parents=True, exist_ok=True)
        unit_hashes = []
        for unit in plan["expected_units"]:
            marker = strict_json_load(marker_path(run_root, unit["unit_id"]))
            unit_hashes.append(marker["artifact_sha256"])
        scientific_digest = sha256_bytes(canonical_json_bytes(unit_hashes))
        per_image_rows, summary_rows, retrieval_rows, fake_execution = _collect_real_outputs(
            run_root, plan, plan_id
        )
        event_counts = _read_resume_counts(run_root)
        runtime_counts = _runtime_counts(run_root)
        elapsed_by_stage = {
            stage: round(
                sum(
                    float(strict_json_load(marker_path(run_root, unit["unit_id"]))["elapsed_seconds"])
                    for unit in plan["expected_units"]
                    if unit["stage"] == stage
                ),
                9,
            )
            for stage in STAGES[1:7]
        }
        result = {
            "schema_id": RESULT_SCHEMA,
            "status": "PASS_RESUMABLE_ANCHOR_TRANSPORT",
            "plan_id": plan_id,
            "repository_commit": plan["repository_commit"],
            "manifest_sha256": plan["manifest_sha256"],
            "config_sha256": plan["config_sha256"],
            "encoder_order": list(ENCODERS),
            "condition_order": list(CONDITIONS),
            "identity_bindings": plan["identity_bindings"],
            "anchor_image_ids": plan["anchor_image_ids"],
            "evaluation_image_ids": plan["evaluation_image_ids"],
            "condition_formulas": {
                "RAW": "source patch unchanged",
                "SAMPLE_RMS": "patch/sqrt(mean(channel_value^2)+1e-6), independently per patch token",
                "GLOBAL_1NN": "lower-index deterministic argmax of unscaled global cosine",
                "GLOBAL_TOP8": "stable-softmax weighted target dictionaries for top-8 unscaled global cosine",
                "POSITIONAL_1NN": "same-position lower-index deterministic argmax",
                "POSITIONAL_TOP8": "same-position stable-softmax top-8 target dictionary transport",
            },
            "anchor_count": 128,
            "evaluation_count": 200,
            "generation_count": 1900,
            "expected_generation_count": plan["expected_generation_count"],
            "expected_unit_count": plan["expected_unit_count"],
            "completed_unit_count": plan["expected_unit_count"],
            "aggregate_metric_row_count": len(summary_rows),
            "aggregate_metrics": summary_rows,
            "actual_runtime_counts": runtime_counts,
            "elapsed_seconds_by_stage": elapsed_by_stage,
            "fake_cpu_execution": fake_execution,
            "scientific_payload_sha256": scientific_digest,
            "metrics_location": "summary.csv",
            "per_image_location": "per_image.csv",
            "retrieval_location": "retrieval.csv",
            "runtime_report_location": "runtime_report.json",
            "resume_report_location": "resume_report.json",
        }
        runtime_report = {
            "schema_id": "vea.p2.t001d.runtime_report.v1",
            "plan_id": plan_id,
            "device": plan["device"],
            "sequential_encoder_residency_required": True,
            "peak_memory_source": "stage_worker_records",
            "stage_order": list(STAGES),
            "elapsed_seconds_by_stage": elapsed_by_stage,
            "progress_event_counts": event_counts,
            "actual_runtime_counts": runtime_counts,
        }
        resume_report = {
            "schema_id": "vea.p2.t001d.resume_report.v1",
            "plan_id": plan_id,
            "event_counts": event_counts,
            "quarantine_entries": sorted(
                path.relative_to(run_root).as_posix()
                for path in (run_root / "failures" / "quarantine").glob("*")
            ),
        }
        payloads = {
            "result.json": canonical_json_bytes(result),
            "summary.csv": _csv_bytes(
                _csv_fieldnames(
                    summary_rows,
                    ("encoder", "condition", "split", "metric", "count", "mean", "median", "minimum", "maximum", "count_improved_over_raw"),
                ),
                summary_rows,
            ),
            "per_image.csv": _csv_bytes(
                _csv_fieldnames(per_image_rows, ("record_type", "image_id", "encoder", "condition", "split", "plan_id")),
                per_image_rows,
            ),
            "retrieval.csv": _csv_bytes(
                _csv_fieldnames(
                    retrieval_rows,
                    ("encoder", "condition", "split", "recall_at_1", "recall_at_5", "plan_id"),
                ),
                retrieval_rows,
            ),
            "runtime_report.json": canonical_json_bytes(runtime_report),
            "resume_report.json": canonical_json_bytes(resume_report),
        }
        hashes: dict[str, str] = {}
        for name, payload in payloads.items():
            path = final_root / name
            if path.exists() and sha256_file(path) != sha256_bytes(payload):
                raise CorruptionError(f"existing final artifact differs: {name}")
            hashes[name] = atomic_write_bytes(path, payload) if not path.exists() else sha256_file(path)
        artifact_manifest = {
            "schema_id": "vea.p2.t001d.artifact_manifest.v1",
            "plan_id": plan_id,
            "artifacts": [
                {"relative_path": name, "sha256": hashes[name], "size_bytes": len(payloads[name])}
                for name in payloads
            ],
            "unit_artifact_count": plan["expected_unit_count"],
            "unit_artifact_hash_chain_sha256": scientific_digest,
        }
        manifest_payload = canonical_json_bytes(artifact_manifest)
        manifest_path = final_root / "artifact_manifest.json"
        if manifest_path.exists() and sha256_file(manifest_path) != sha256_bytes(manifest_payload):
            raise CorruptionError("existing final artifact_manifest.json differs")
        if not manifest_path.exists():
            atomic_write_bytes(manifest_path, manifest_payload)
        append_event(run_root, progress_event(plan_id, "FINALIZED", scientific_payload_sha256=scientific_digest))
    return result


def format_recovery_command(executable: str | Sequence[str], common_arguments: Sequence[str]) -> str:
    import subprocess

    prefix = [executable] if isinstance(executable, str) else list(executable)
    return subprocess.list2cmdline([*prefix, "resume", *common_arguments])


def failure_record(error: BaseException, plan_id: str | None, recovery_command: str | None) -> dict[str, Any]:
    classification = getattr(error, "classification", "BLOCKED_UNCLASSIFIED_FAILURE")
    retryable = bool(getattr(error, "retryable", False))
    chain: list[dict[str, str]] = []
    current: BaseException | None = error
    while current is not None:
        chain.append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    return {
        "schema_id": "vea.p2.t001d.failure.v1",
        "plan_id": plan_id,
        "classification": classification,
        "retryable": retryable,
        "exit_code": RETRYABLE_EXIT_CODE if retryable else int(getattr(error, "exit_code", 1)),
        "root_cause": chain[-1] if chain else None,
        "exception_chain": chain,
        "exact_recovery_command": recovery_command if retryable else None,
        "recorded_at_utc": utc_now(),
    }


def write_failure_record(run_root: Path, record: Mapping[str, Any]) -> Path:
    directory = run_root / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(canonical_json_bytes(record)).hexdigest()[:12]
    path = directory / f"failure_{token}.json"
    atomic_write_json(path, dict(record))
    return path


__all__ = [
    "AmbiguousLockError",
    "CONDITIONS",
    "ContractError",
    "CoordinatorLock",
    "CorruptionError",
    "ENCODERS",
    "FINAL_ARTIFACTS",
    "HarnessError",
    "IncompleteRunError",
    "InjectedInterruption",
    "LockBusyError",
    "PLAN_SCHEMA",
    "PlanMismatchError",
    "RETRYABLE_EXIT_CODE",
    "RunInterruptedError",
    "STAGES",
    "apply_transport_condition",
    "atomic_publish_unit",
    "atomic_write_bytes",
    "build_run_plan",
    "build_unit_inventory",
    "canonical_json_bytes",
    "current_process_start_identity",
    "deterministic_fake_payload",
    "estimate_storage_bytes",
    "execute_fake_units",
    "failure_record",
    "finalize_run",
    "format_recovery_command",
    "generation_metrics",
    "global_anchor_transport",
    "initialize_run",
    "inspect_exact_feature_reuse",
    "load_bound_plan",
    "load_config",
    "marker_path",
    "positional_anchor_transport",
    "process_liveness",
    "read_manifest",
    "recover_orphans",
    "representation_metrics",
    "retrieval_recall",
    "reuse_exact_feature_units",
    "sample_rms",
    "sha256_bytes",
    "sha256_file",
    "strict_json_load",
    "validate_patch_batch",
    "validate_identity_sources",
    "validate_resume_bindings",
    "validate_unit_completion",
    "verify_run",
    "write_failure_record",
]
