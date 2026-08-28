"""Read-only, memory-bounded binding of T001D banks to the Phase 2 adapter."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


SCHEMA_ID = "vea.p2.query_adapter_bank_preflight.v1"
SUCCESS_STATUS = "PASS_QUERY_ADAPTER_BANK_PREFLIGHT"
DIGEST_RECIPE = (
    "sha256(canonical-json-v1({bank,encoder_identity,t001d_plan_id,manifest_sha256,"
    "ordered_units:[{unit_id,artifact_relative_path,artifact_sha256}],"
    "ordered_image_ids,patch_shape,patch_dtype,connector_shape,connector_dtype}))"
)
REQUIRED_STAGES = {
    "DINOV3": "01_DINOV3_FEATURE_BANK",
    "SIGLIP2": "02_SIGLIP2_FEATURE_BANK",
    "TARGET_CLIP": "03_TARGET_CLIP_AND_NATIVE_CONNECTOR_BANK",
}
SOURCE_FIELDS = frozenset({"image_ids", "patches", "globals", "plan_id"})
TARGET_FIELDS = frozenset({*SOURCE_FIELDS, "connectors"})


class BankPreflightError(RuntimeError):
    """A stable, fail-closed bank binding failure."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification

    def as_dict(self) -> dict[str, str]:
        return {"status": "BLOCKED", "classification": self.classification, "message": str(self)}


@dataclass(frozen=True)
class PreflightExpectations:
    """Expected dimensions/counts; production defaults are immutable."""

    image_count: int = 1000
    split_counts: tuple[int, int, int] = (800, 100, 100)
    patch_shape: tuple[int, int] = (196, 768)
    connector_shape: tuple[int, int] = (196, 1536)


PRODUCTION_EXPECTATIONS = PreflightExpectations()


def _fail(classification: str, message: str) -> None:
    raise BankPreflightError(classification, message)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(t001d.canonical_json_bytes(value)).hexdigest()


def _read_config(path: Path) -> tuple[dict[str, Any], str]:
    try:
        config = linear.load_query_adapter_config(path)
        digest = t001d.sha256_file(path)
    except (OSError, ValueError, KeyError, TypeError, linear.QueryAdapterLinearError) as error:
        _fail("BLOCKED_BANK_PREFLIGHT_CONFIG", f"adapter config rejected: {error}")
    return config, digest


def _require_mapping(value: Any, label: str, classification: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(classification, f"{label} must be an object")
    return value


def _identity_projection(spec: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    missing = [key for key in keys if key not in spec]
    if missing:
        _fail("BLOCKED_BANK_MODEL_IDENTITY", f"model identity missing fields: {missing}")
    return {key: spec[key] for key in keys}


def _validate_identities(plan: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bindings = _require_mapping(plan.get("identity_bindings"), "T001D identity_bindings", "BLOCKED_T001D_PLAN_IDENTITY")
    plan_encoders = _require_mapping(bindings.get("encoders"), "T001D encoders", "BLOCKED_T001D_PLAN_IDENTITY")
    sources = _require_mapping(config.get("sources"), "adapter sources", "BLOCKED_BANK_PREFLIGHT_CONFIG")
    target = _require_mapping(config.get("target"), "adapter target", "BLOCKED_BANK_PREFLIGHT_CONFIG")
    keys = ("model_id", "revision", "tap", "tap_name", "preprocessing", "special_token_policy", "patch_shape")
    expected = {"SIGLIP2": sources["SIGLIP2"], "DINOV3": sources["DINOV3"], "TARGET_CLIP": target}
    projections: dict[str, dict[str, Any]] = {}
    for bank, wanted in expected.items():
        observed = _require_mapping(plan_encoders.get(bank), f"T001D {bank}", "BLOCKED_BANK_MODEL_IDENTITY")
        wanted_projection = _identity_projection(wanted, keys)
        observed_projection = _identity_projection(observed, keys)
        if observed_projection != wanted_projection:
            _fail("BLOCKED_BANK_MODEL_IDENTITY", f"{bank} model identity differs from frozen adapter config")
        projections[bank] = wanted_projection

    target_vlm = _require_mapping(config.get("target_vlm"), "adapter target_vlm", "BLOCKED_BANK_PREFLIGHT_CONFIG")
    plan_vlm = _require_mapping(bindings.get("target_vlm"), "T001D target_vlm", "BLOCKED_BANK_MODEL_IDENTITY")
    vlm_keys = ("model_id", "revision", "connector_input_shape", "connector_output_shape")
    if _identity_projection(plan_vlm, vlm_keys) != _identity_projection(target_vlm, vlm_keys):
        _fail("BLOCKED_BANK_MODEL_IDENTITY", "TinyLLaVA target identity differs from frozen adapter config")
    projections["TARGET_CLIP"] = {**projections["TARGET_CLIP"], "target_vlm": _identity_projection(target_vlm, vlm_keys)}
    return projections


def _validate_plan(
    plan: Mapping[str, Any], config: Mapping[str, Any], expectations: PreflightExpectations
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if plan.get("schema_id") != t001d.PLAN_SCHEMA or plan.get("task_id") != "VEA-P2-T001D":
        _fail("BLOCKED_T001D_PLAN_IDENTITY", "T001D plan schema/task identity mismatch")
    schemas = _require_mapping(plan.get("artifact_schemas"), "T001D artifact_schemas", "BLOCKED_T001D_PLAN_IDENTITY")
    if schemas.get("result") != t001d.RESULT_SCHEMA or schemas.get("done_marker") != t001d.MARKER_SCHEMA:
        _fail("BLOCKED_T001D_PLAN_IDENTITY", "T001D result/marker schema binding mismatch")

    manifest = _require_mapping(config.get("manifest"), "adapter manifest", "BLOCKED_BANK_PREFLIGHT_CONFIG")
    expected_sha = manifest.get("sha256")
    if plan.get("manifest_sha256") != expected_sha:
        _fail("BLOCKED_T001D_PLAN_IDENTITY", "T001D manifest SHA-256 differs from frozen adapter config")
    if manifest.get("row_count") != 1000:
        _fail("BLOCKED_BANK_PREFLIGHT_CONFIG", "frozen adapter manifest row_count is not 1000")
    if plan.get("manifest_row_count") != expectations.image_count:
        _fail("BLOCKED_T001D_PLAN_IDENTITY", "T001D manifest row count differs from required profile")
    rows_value = plan.get("manifest_rows")
    if not isinstance(rows_value, list) or len(rows_value) != expectations.image_count:
        _fail("BLOCKED_T001D_PLAN_IDENTITY", "T001D manifest rows are unavailable or incomplete")
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(rows_value):
        row = _require_mapping(value, f"manifest row {index}", "BLOCKED_T001D_PLAN_IDENTITY")
        if set(row) != {"image_id", "relative_path", "source_sha256", "selection_index", "split"}:
            _fail("BLOCKED_T001D_PLAN_IDENTITY", f"manifest row {index} fields differ from T001D contract")
        if not isinstance(row["image_id"], str) or not row["image_id"]:
            _fail("BLOCKED_T001D_PLAN_IDENTITY", f"manifest row {index} has invalid image_id")
        rows.append(dict(row))
    image_ids = [row["image_id"] for row in rows]
    if len(set(image_ids)) != len(image_ids):
        _fail("BLOCKED_BANK_IMAGE_ORDER", "T001D manifest contains duplicate image IDs")

    names = ("train", "validation", "test")
    counts = {name: sum(row["split"] == name for row in rows) for name in names}
    wanted_counts = dict(zip(names, expectations.split_counts, strict=True))
    if any(row["split"] not in names for row in rows) or counts != wanted_counts:
        _fail("BLOCKED_BANK_SPLIT_IDENTITY", f"T001D split identity differs: observed={counts}")
    if plan.get("manifest_split_counts") != counts:
        _fail("BLOCKED_BANK_SPLIT_IDENTITY", "T001D plan split counts do not bind manifest rows")
    roles = _require_mapping(manifest.get("partition_roles"), "adapter partition_roles", "BLOCKED_BANK_PREFLIGHT_CONFIG")
    frozen_counts = {name: roles[name]["image_count"] for name in names}
    if frozen_counts != {"train": 800, "validation": 100, "test": 100}:
        _fail("BLOCKED_BANK_PREFLIGHT_CONFIG", "adapter partition counts drifted from 800/100/100")
    return rows, counts


def _safe_artifact(run_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        _fail("BLOCKED_T001D_PLAN_IDENTITY", "feature unit artifact path is not relative")
    candidate = (run_root / relative).resolve()
    try:
        candidate.relative_to(run_root)
    except ValueError:
        _fail("BLOCKED_T001D_PLAN_IDENTITY", "feature unit artifact path escapes run root")
    return candidate


def _scalar_plan_id(archive: Any, expected: str) -> None:
    value = archive["plan_id"]
    if value.shape != () or value.dtype.kind not in "US" or str(value.item()) != expected:
        _fail("BLOCKED_BANK_PAYLOAD_CONTRACT", "persisted NPZ plan_id differs from validated T001D plan")


def _float_array(archive: Any, name: str, shape: tuple[int, ...]) -> None:
    value = archive[name]
    if value.shape != shape:
        _fail("BLOCKED_BANK_PAYLOAD_CONTRACT", f"{name} shape differs: observed={value.shape} expected={shape}")
    if value.dtype != np.dtype("float32"):
        _fail("BLOCKED_BANK_PAYLOAD_CONTRACT", f"{name} dtype differs: observed={value.dtype} expected=float32")
    if not np.isfinite(value).all():
        _fail("BLOCKED_BANK_PAYLOAD_CONTRACT", f"{name} contains non-finite values")


def _ordered_units(plan: Mapping[str, Any], bank: str) -> list[Mapping[str, Any]]:
    values = plan.get("expected_units")
    if not isinstance(values, list):
        _fail("BLOCKED_T001D_PLAN_IDENTITY", "T001D expected_units is unavailable")
    units = [value for value in values if isinstance(value, Mapping) and value.get("stage") == REQUIRED_STAGES[bank]]
    if not units:
        _fail("BLOCKED_T001D_UNIT_INCOMPLETE", f"no feature units found for {bank}")
    try:
        units.sort(key=lambda unit: int(unit["chunk_index"]))
    except (KeyError, TypeError, ValueError):
        _fail("BLOCKED_T001D_PLAN_IDENTITY", f"{bank} chunk indices are invalid")
    if [unit.get("chunk_index") for unit in units] != list(range(len(units))):
        _fail("BLOCKED_T001D_PLAN_IDENTITY", f"{bank} chunk indices are not unique and contiguous")
    return units


def _validate_bank(
    run_root: Path,
    plan: Mapping[str, Any],
    plan_id: str,
    manifest_sha: str,
    bank: str,
    identity: Mapping[str, Any],
    expectations: PreflightExpectations,
) -> tuple[dict[str, Any], list[str]]:
    units = _ordered_units(plan, bank)
    image_ids: list[str] = []
    digest_units: list[dict[str, str]] = []
    artifact_total_bytes = 0
    seen_unit_ids: set[str] = set()
    seen_paths: set[str] = set()
    for unit in units:
        if unit.get("encoder") != bank:
            _fail("BLOCKED_T001D_PLAN_IDENTITY", f"{bank} unit encoder binding mismatch")
        unit_id = unit.get("unit_id")
        relative = unit.get("artifact_relative_path")
        if not isinstance(unit_id, str) or unit_id in seen_unit_ids or not isinstance(relative, str) or relative in seen_paths:
            _fail("BLOCKED_T001D_PLAN_IDENTITY", f"{bank} has duplicate/invalid unit identity")
        seen_unit_ids.add(unit_id)
        seen_paths.add(relative)
        artifact = _safe_artifact(run_root, relative)
        try:
            complete = t001d.validate_unit_completion(run_root, unit, plan_id)
        except t001d.CorruptionError as error:
            classification = (
                "BLOCKED_T001D_ARTIFACT_HASH"
                if "artifact hash mismatch" in str(error) or "artifact size mismatch" in str(error)
                else "BLOCKED_T001D_UNIT_INCOMPLETE"
            )
            _fail(classification, str(error))
        except (OSError, ValueError, KeyError, TypeError, t001d.HarnessError) as error:
            _fail("BLOCKED_T001D_UNIT_INCOMPLETE", f"unit completion validation failed: {error}")
        if not complete:
            _fail("BLOCKED_T001D_UNIT_INCOMPLETE", f"unit is incomplete: {unit_id}")
        actual_sha = t001d.sha256_file(artifact)
        artifact_total_bytes += artifact.stat().st_size
        try:
            with np.load(artifact, allow_pickle=False) as archive:
                expected_fields = TARGET_FIELDS if bank == "TARGET_CLIP" else SOURCE_FIELDS
                if set(archive.files) != expected_fields:
                    _fail("BLOCKED_BANK_PAYLOAD_CONTRACT", f"{bank} NPZ fields differ: observed={sorted(archive.files)}")
                raw_ids = archive["image_ids"]
                if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "US":
                    _fail("BLOCKED_BANK_PAYLOAD_CONTRACT", f"{bank} image_ids must be a one-dimensional string array")
                chunk_ids = [str(value) for value in raw_ids.tolist()]
                planned_ids = unit.get("image_ids")
                if chunk_ids != planned_ids:
                    _fail("BLOCKED_BANK_IMAGE_ORDER", f"{bank} payload image order differs from its plan unit")
                n = len(chunk_ids)
                if n == 0:
                    _fail("BLOCKED_BANK_PAYLOAD_CONTRACT", f"{bank} contains an empty feature chunk")
                expected_unit_shape = {
                    "patches": [n, *expectations.patch_shape],
                    "globals": [n, expectations.patch_shape[1]],
                    **({"connectors": [n, *expectations.connector_shape]} if bank == "TARGET_CLIP" else {}),
                }
                expected_unit_dtype = {name: "float32" for name in expected_unit_shape}
                if unit.get("shape") != expected_unit_shape or unit.get("dtype") != expected_unit_dtype:
                    _fail("BLOCKED_T001D_PLAN_IDENTITY", f"{bank} unit shape/dtype metadata differs from payload contract")
                _scalar_plan_id(archive, plan_id)
                _float_array(archive, "patches", (n, *expectations.patch_shape))
                _float_array(archive, "globals", (n, expectations.patch_shape[1]))
                if bank == "TARGET_CLIP":
                    _float_array(archive, "connectors", (n, *expectations.connector_shape))
        except BankPreflightError:
            raise
        except (OSError, ValueError, EOFError) as error:
            _fail("BLOCKED_BANK_PAYLOAD_CONTRACT", f"{bank} NPZ cannot be read safely: {error}")
        image_ids.extend(chunk_ids)
        digest_units.append({"unit_id": unit_id, "artifact_relative_path": relative, "artifact_sha256": actual_sha})

    digest_material = {
        "bank": bank,
        "encoder_identity": identity,
        "t001d_plan_id": plan_id,
        "manifest_sha256": manifest_sha,
        "ordered_units": digest_units,
        "ordered_image_ids": image_ids,
        "patch_shape": list(expectations.patch_shape),
        "patch_dtype": "float32",
        "connector_shape": list(expectations.connector_shape) if bank == "TARGET_CLIP" else None,
        "connector_dtype": "float32" if bank == "TARGET_CLIP" else None,
    }
    summary: dict[str, Any] = {
        "model_id": identity["model_id"],
        "revision": identity["revision"],
        "tap": identity["tap"],
        "image_count": len(image_ids),
        "chunk_count": len(units),
        "patch_shape": list(expectations.patch_shape),
        "patch_dtype": "float32",
        "artifact_total_bytes": artifact_total_bytes,
        "bank_identity_sha256": _canonical_sha(digest_material),
        "digest_recipe": DIGEST_RECIPE,
    }
    if bank == "TARGET_CLIP":
        summary.update(connector_shape=list(expectations.connector_shape), connector_dtype="float32")
    return summary, image_ids


def preflight_query_adapter_banks(
    config_path: str | Path,
    t001d_run_root: str | Path,
    *,
    expectations: PreflightExpectations = PRODUCTION_EXPECTATIONS,
) -> dict[str, Any]:
    """Validate a completed T001D bank source without modifying or aggregating it."""

    config_path = Path(config_path)
    run_root = Path(t001d_run_root)
    if not run_root.is_absolute():
        _fail("BLOCKED_T001D_PLAN_IDENTITY", "T001D run root must be absolute")
    run_root = run_root.resolve()
    config, config_sha = _read_config(config_path)
    try:
        plan, plan_id = t001d.load_bound_plan(run_root)
    except (OSError, ValueError, TypeError, t001d.HarnessError) as error:
        _fail("BLOCKED_T001D_PLAN_IDENTITY", f"T001D plan binding rejected: {error}")
    rows, counts = _validate_plan(plan, config, expectations)
    identities = _validate_identities(plan, config)
    manifest_ids = [row["image_id"] for row in rows]
    banks: dict[str, dict[str, Any]] = {}
    bank_ids: dict[str, list[str]] = {}
    for bank in ("SIGLIP2", "DINOV3", "TARGET_CLIP"):
        banks[bank], bank_ids[bank] = _validate_bank(
            run_root, plan, plan_id, str(plan["manifest_sha256"]), bank, identities[bank], expectations
        )
    if any(len(ids) != expectations.image_count or len(set(ids)) != len(ids) for ids in bank_ids.values()):
        _fail("BLOCKED_BANK_IMAGE_ORDER", "a required bank does not contain the exact unique image count")
    cross_equal = bank_ids["SIGLIP2"] == bank_ids["DINOV3"] == bank_ids["TARGET_CLIP"]
    if not cross_equal:
        _fail("BLOCKED_BANK_IMAGE_ORDER", "feature-bank image ID orders differ")
    manifest_equal = bank_ids["SIGLIP2"] == manifest_ids
    if not manifest_equal:
        _fail("BLOCKED_BANK_IMAGE_ORDER", "feature-bank image order differs from T001D manifest order")
    return {
        "schema_id": SCHEMA_ID,
        "status": SUCCESS_STATUS,
        "adapter_experiment_id": config["experiment_id"],
        "adapter_config_sha256": config_sha,
        "t001d_plan_id": plan_id,
        "t001d_run_root": str(run_root),
        "manifest_sha256": plan["manifest_sha256"],
        "banks": banks,
        "partitions": {
            "train_count": counts["train"],
            "validation_count": counts["validation"],
            "test_count": counts["test"],
        },
        "cross_bank_image_order_equal": True,
        "manifest_order_equal": True,
        "all_required_units_valid": True,
        "bank_digest_algorithm": DIGEST_RECIPE,
    }


def synthetic_t001d_plan(
    config: Mapping[str, Any], expectations: PreflightExpectations, *, chunks: int = 2
) -> dict[str, Any]:
    """Build a tiny T001D-contract plan for explicit fake validation only."""

    if chunks < 2 or expectations.image_count < chunks:
        raise ValueError("synthetic fixture requires at least two non-empty chunks")
    names = ("train", "validation", "test")
    split_values = [name for name, count in zip(names, expectations.split_counts, strict=True) for _ in range(count)]
    if len(split_values) != expectations.image_count:
        raise ValueError("synthetic split counts must sum to image_count")
    rows = [
        {
            "image_id": f"fake-{index:04d}",
            "relative_path": f"fake/{index:04d}.png",
            "source_sha256": hashlib.sha256(f"fake-{index}".encode()).hexdigest(),
            "selection_index": index,
            "split": split_values[index],
        }
        for index in range(expectations.image_count)
    ]
    sizes = [expectations.image_count // chunks + (index < expectations.image_count % chunks) for index in range(chunks)]
    units: list[dict[str, Any]] = []
    offset = 0
    for bank, stage in REQUIRED_STAGES.items():
        offset = 0
        for chunk_index, size in enumerate(sizes):
            ids = [row["image_id"] for row in rows[offset : offset + size]]
            offset += size
            shape = {"patches": [size, *expectations.patch_shape], "globals": [size, expectations.patch_shape[1]]}
            if bank == "TARGET_CLIP":
                shape["connectors"] = [size, *expectations.connector_shape]
            units.append(
                {
                    "unit_id": f"{stage.lower()}-{chunk_index:04d}",
                    "stage": stage,
                    "artifact_relative_path": f"features/{bank.lower()}/chunk_{chunk_index:04d}.npz",
                    "encoder": bank,
                    "condition": None,
                    "split": None,
                    "image_ids": ids,
                    "shape": shape,
                    "dtype": {name: "float32" for name in shape},
                    "chunk_index": chunk_index,
                    "image_id": None,
                }
            )
    encoders = {
        "SIGLIP2": dict(config["sources"]["SIGLIP2"]),
        "DINOV3": dict(config["sources"]["DINOV3"]),
        "TARGET_CLIP": dict(config["target"]),
    }
    return {
        "schema_id": t001d.PLAN_SCHEMA,
        "task_id": "VEA-P2-T001D",
        "repository_commit": "0" * 40,
        "config_sha256": "0" * 64,
        "manifest_sha256": config["manifest"]["sha256"],
        "manifest_row_count": expectations.image_count,
        "manifest_split_counts": dict(zip(names, expectations.split_counts, strict=True)),
        "manifest_rows": rows,
        "stage_order": list(t001d.STAGES),
        "identity_bindings": {"encoders": encoders, "target_vlm": dict(config["target_vlm"])},
        "expected_units": units,
        "expected_unit_count": len(units),
        "artifact_schemas": {
            "unit": "vea.p2.t001d.unit_artifact.v1",
            "done_marker": t001d.MARKER_SCHEMA,
            "resume": "vea.p2.t001d.resume.v1",
            "result": t001d.RESULT_SCHEMA,
        },
    }


def publish_synthetic_t001d_fixture(
    run_root: Path,
    plan: Mapping[str, Any],
    *,
    payload_order_override: Mapping[str, Sequence[str]] | None = None,
    fill_offset: float = 0.0,
    array_mutator: Callable[[Mapping[str, Any], dict[str, Any]], None] | None = None,
) -> str:
    """Write fake NPZ units under an explicit temporary root for tests/smoke."""

    plan_id = t001d.initialize_run(run_root, plan)
    overrides = payload_order_override or {}
    for index, unit in enumerate(plan["expected_units"]):
        ids = list(overrides.get(str(unit["encoder"]), unit["image_ids"]))
        n = len(ids)
        arrays: dict[str, Any] = {
            "image_ids": np.asarray(ids, dtype="U96"),
            "patches": np.full(tuple(unit["shape"]["patches"]), fill_offset + index, dtype=np.float32),
            "globals": np.full(tuple(unit["shape"]["globals"]), fill_offset + index, dtype=np.float32),
            "plan_id": np.asarray(plan_id, dtype="U64"),
        }
        if "connectors" in unit["shape"]:
            arrays["connectors"] = np.full(tuple(unit["shape"]["connectors"]), fill_offset + index, dtype=np.float32)
        if n != len(unit["image_ids"]):
            raise ValueError("payload order override must preserve chunk length")
        if array_mutator is not None:
            array_mutator(unit, arrays)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        t001d.atomic_publish_unit(run_root, unit, buffer.getvalue(), plan_id, elapsed_seconds=0.0)
    return plan_id
