"""Train/Validation-only runtime for the frozen direct linear-adapter experiment."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_bank_preflight as preflight
from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


BINDING_SCHEMA_ID = "vea.p2.query_adapter_linear_bank_binding.v1"
RUNTIME_SCHEMA_ID = "vea.p2.query_adapter_linear_validation_runtime.v1"
SUCCESS_STATUS = "PASS_QUERY_ADAPTER_LINEAR_VALIDATION"


class ValidationRuntimeError(RuntimeError):
    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


class ConnectorProtocol(Protocol):
    generation_calls: int
    vision_calls: int

    def apply(self, patches: np.ndarray, *, batch_size: int) -> np.ndarray: ...


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    return t001d.sha256_file(path)


def _fail(classification: str, message: str) -> None:
    raise ValidationRuntimeError(classification, message)


def load_bank_binding(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("BLOCKED_BANK_BINDING_CONFIG", f"bank binding is unavailable or invalid: {error}")
    required = {
        "schema_id", "experiment_id", "adapter_config_path", "adapter_config_sha256",
        "accepted_preflight_json_sha256", "t001d_plan_id", "manifest_sha256", "banks", "path_identity_policy",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("schema_id") != BINDING_SCHEMA_ID:
        _fail("BLOCKED_BANK_BINDING_CONFIG", "bank binding schema/fields mismatch")
    if value.get("experiment_id") != linear.EXPERIMENT_ID:
        _fail("BLOCKED_BANK_BINDING_CONFIG", "bank binding experiment identity mismatch")
    banks = value.get("banks")
    if not isinstance(banks, dict) or set(banks) != set(linear.SOURCE_ORDER) | {"TARGET_CLIP"}:
        _fail("BLOCKED_BANK_BINDING_CONFIG", "bank binding bank keys mismatch")
    identities = [value[name] for name in ("adapter_config_sha256", "accepted_preflight_json_sha256", "t001d_plan_id", "manifest_sha256")]
    identities.extend(banks.values())
    for key in identities:
        if not isinstance(key, str) or len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            _fail("BLOCKED_BANK_BINDING_CONFIG", "bank binding contains invalid SHA-256 identity")
    if "path" not in str(value["path_identity_policy"]).lower():
        _fail("BLOCKED_BANK_BINDING_CONFIG", "bank binding must state non-normative path policy")
    return value


def verify_preflight_binding(
    report: Mapping[str, Any], binding: Mapping[str, Any], *, adapter_config_sha256: str
) -> None:
    if report.get("status") != preflight.SUCCESS_STATUS:
        _fail("BLOCKED_BANK_PREFLIGHT", "T002C preflight did not pass")
    required = {
        "adapter_experiment_id": linear.EXPERIMENT_ID,
        "adapter_config_sha256": adapter_config_sha256,
        "t001d_plan_id": binding["t001d_plan_id"],
        "manifest_sha256": binding["manifest_sha256"],
        "cross_bank_image_order_equal": True,
        "manifest_order_equal": True,
        "all_required_units_valid": True,
    }
    if any(report.get(key) != expected for key, expected in required.items()):
        _fail("BLOCKED_BANK_PREFLIGHT", "re-run T002C preflight identity differs from accepted bank binding")
    observed_banks = report.get("banks")
    if not isinstance(observed_banks, Mapping):
        _fail("BLOCKED_BANK_PREFLIGHT", "preflight bank summaries are unavailable")
    for name, expected in binding["banks"].items():
        summary = observed_banks.get(name)
        if not isinstance(summary, Mapping) or summary.get("bank_identity_sha256") != expected:
            _fail("BLOCKED_BANK_PREFLIGHT", f"preflight {name} digest differs from accepted bank binding")


def _ordered_stage_units(plan: Mapping[str, Any], bank: str) -> list[Mapping[str, Any]]:
    stage = preflight.REQUIRED_STAGES[bank]
    units = [unit for unit in plan.get("expected_units", []) if isinstance(unit, Mapping) and unit.get("stage") == stage]
    try:
        return sorted(units, key=lambda value: int(value["chunk_index"]))
    except (KeyError, TypeError, ValueError) as error:
        _fail("BLOCKED_T001D_BANK_PAYLOAD", f"{bank} chunk identity is invalid: {error}")


def _validate_selected(value: np.ndarray, shape_tail: tuple[int, int], label: str) -> np.ndarray:
    if value.ndim != 3 or value.shape[1:] != shape_tail or value.dtype != np.dtype("float32") or not np.isfinite(value).all():
        _fail("BLOCKED_T001D_BANK_PAYLOAD", f"{label} bank payload contract mismatch")
    return np.ascontiguousarray(value)


def load_train_validation_banks(run_root: str | Path) -> dict[str, Any]:
    """Read only Train/Validation selections from validated T001D bank chunks."""

    root = Path(run_root).resolve()
    try:
        plan, plan_id = t001d.load_bound_plan(root)
    except t001d.HarnessError as error:
        _fail("BLOCKED_T001D_BANK_PAYLOAD", f"T001D plan cannot be loaded: {error}")
    rows = plan.get("manifest_rows")
    if not isinstance(rows, list):
        _fail("BLOCKED_T001D_BANK_PAYLOAD", "T001D manifest rows are unavailable")
    split_by_id = {str(row["image_id"]): str(row["split"]) for row in rows if isinstance(row, Mapping)}
    selected_ids = {name: [str(row["image_id"]) for row in rows if row.get("split") == name] for name in ("train", "validation")}
    if {name: len(values) for name, values in selected_ids.items()} != {"train": 800, "validation": 100}:
        _fail("BLOCKED_T001D_BANK_PAYLOAD", "T001D Train/Validation split counts differ from frozen contract")
    values: dict[str, dict[str, Any]] = {}
    for bank in (*linear.SOURCE_ORDER, "TARGET_CLIP"):
        parts: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
        ids: dict[str, list[str]] = {"train": [], "validation": []}
        connector_parts: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
        for unit in _ordered_stage_units(plan, bank):
            try:
                if not t001d.validate_unit_completion(root, unit, plan_id):
                    _fail("BLOCKED_T001D_BANK_PAYLOAD", f"incomplete T001D unit: {unit['unit_id']}")
                with np.load(root / str(unit["artifact_relative_path"]), allow_pickle=False) as archive:
                    unit_ids = [str(value) for value in archive["image_ids"].tolist()]
                    for split in ("train", "validation"):
                        indices = [index for index, image_id in enumerate(unit_ids) if split_by_id.get(image_id) == split]
                        if not indices:
                            continue
                        parts[split].append(np.ascontiguousarray(archive["patches"][indices], dtype=np.float32))
                        ids[split].extend(unit_ids[index] for index in indices)
                        if bank == "TARGET_CLIP":
                            connector_parts[split].append(np.ascontiguousarray(archive["connectors"][indices], dtype=np.float32))
            except (OSError, ValueError, KeyError, t001d.HarnessError) as error:
                _fail("BLOCKED_T001D_BANK_PAYLOAD", f"cannot load {bank} T001D payload: {error}")
        values[bank] = {
            split: _validate_selected(np.concatenate(parts[split], axis=0), (196, 768), f"{bank} {split}")
            for split in ("train", "validation")
        }
        values[bank]["ids"] = ids
        if bank == "TARGET_CLIP":
            values[bank]["connectors"] = {
                split: _validate_selected(np.concatenate(connector_parts[split], axis=0), (196, 1536), f"TARGET_CLIP connector {split}")
                for split in ("train", "validation")
            }
    target = values["TARGET_CLIP"]
    for source in linear.SOURCE_ORDER:
        for split in ("train", "validation"):
            if values[source]["ids"][split] != target["ids"][split] or target["ids"][split] != selected_ids[split]:
                _fail("BLOCKED_T001D_BANK_PAYLOAD", f"{source} {split} image order differs from T001D manifest")
    return values


def _connector_metrics(candidate: np.ndarray, target: np.ndarray, train_target_mean: np.ndarray) -> dict[str, float]:
    metrics = linear.patch_metrics(candidate, target, train_target_mean)
    return {"connector_cosine": float(metrics["cosine"]), "connector_nmse": float(metrics["nmse"])}


def _fit_and_apply(method: str, train_source: np.ndarray, train_target: np.ndarray, validation_source: np.ndarray, lambda_value: float | None = None) -> tuple[Any, np.ndarray]:
    if method == "CALIBRATED":
        fit = linear.fit_calibration(train_source, train_target)
        return fit, linear.apply_calibration(validation_source, fit)
    if method == "ORTHOGONAL_PROCRUSTES":
        calibration = linear.fit_calibration(train_source, train_target)
        fit = linear.fit_orthogonal_procrustes(linear.apply_calibration(train_source, calibration), train_target)
        return (calibration, fit), linear.apply_orthogonal_procrustes(linear.apply_calibration(validation_source, calibration), fit)
    if method == "FULL_LINEAR":
        fit = linear.fit_full_linear(train_source, train_target)
        return fit, linear.apply_full_linear(validation_source, fit)
    if method == "RIDGE" and lambda_value is not None:
        fit = linear.fit_ridge(train_source, train_target, lambda_value)
        return fit, linear.apply_ridge(validation_source, fit)
    _fail("BLOCKED_RUNTIME_CONTRACT", f"unsupported frozen method: {method}")


def _fit_arrays(fit: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(fit, tuple):
        merged: dict[str, np.ndarray] = {}
        for index, item in enumerate(fit):
            merged.update(_fit_arrays(item, f"{prefix}part{index}_"))
        return merged
    if not is_dataclass(fit):
        _fail("BLOCKED_RUNTIME_CONTRACT", "fit artifact is not a T002B fit object")
    result: dict[str, np.ndarray] = {}
    for field in fields(fit):
        result[f"{prefix}{field.name}"] = np.asarray(getattr(fit, field.name))
    return result


def _write_new(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
    except FileExistsError:
        _fail("BLOCKED_RUNTIME_RESUME", f"immutable runtime artifact already exists: {path.name}")
    return hashlib.sha256(payload).hexdigest()


def _load_resumed_result(output_root: Path, identity: str) -> dict[str, Any] | None:
    plan_path, result_path = output_root / "validation_plan.json", output_root / "result.json"
    if not plan_path.is_file() or not result_path.is_file():
        return None
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail("BLOCKED_RUNTIME_RESUME", f"runtime resume metadata is invalid: {error}")
    if plan.get("runtime_identity_sha256") != identity or result.get("runtime_identity_sha256") != identity:
        _fail("BLOCKED_RUNTIME_RESUME", "runtime output identity differs; refusing unsafe resume")
    artifacts = result.get("fit_artifacts")
    if not isinstance(artifacts, Mapping):
        _fail("BLOCKED_RUNTIME_RESUME", "runtime result lacks fit artifact hashes")
    for relative, expected in artifacts.items():
        path = output_root / str(relative)
        if not path.is_file() or _sha_file(path) != expected:
            _fail("BLOCKED_RUNTIME_RESUME", f"fit artifact hash mismatch: {relative}")
    result["resumed"] = True
    return result


def run_train_validation(
    *,
    config: Mapping[str, Any],
    binding: Mapping[str, Any],
    banks: Mapping[str, Any],
    connector: ConnectorProtocol,
    output_root: str | Path,
    connector_batch_size: int,
    adapter_config_sha256: str,
    binding_config_sha256: str,
    fake: bool = False,
) -> dict[str, Any]:
    """Fit Train-only T002B adapters and evaluate/select on Validation only."""

    linear.validate_query_adapter_config(config)
    if binding.get("adapter_config_sha256") != adapter_config_sha256:
        _fail("BLOCKED_BANK_BINDING_CONFIG", "adapter config SHA differs from accepted bank binding")
    output = Path(output_root).resolve()
    identity_material = {
        "schema_id": RUNTIME_SCHEMA_ID,
        "adapter_config_sha256": adapter_config_sha256,
        "binding_config_sha256": binding_config_sha256,
        "accepted_preflight_json_sha256": binding["accepted_preflight_json_sha256"],
        "t001d_plan_id": binding["t001d_plan_id"],
        "banks": binding["banks"],
        "fake": fake,
    }
    identity = _sha(identity_material)
    resumed = _load_resumed_result(output, identity) if output.exists() else None
    if resumed is not None:
        return resumed
    if output.exists() and any(output.iterdir()):
        _fail("BLOCKED_RUNTIME_RESUME", "output root is nonempty without a valid resumable result")
    output.mkdir(parents=True, exist_ok=True)
    _write_new(output / "validation_plan.json", _canonical_bytes({"schema_id": RUNTIME_SCHEMA_ID, "runtime_identity_sha256": identity, **identity_material}))

    target = banks["TARGET_CLIP"]
    fit_artifacts: dict[str, str] = {}
    sources_result: dict[str, Any] = {}
    for source_name in linear.SOURCE_ORDER:
        source = banks[source_name]
        train_source, validation_source = source["train"], source["validation"]
        train_target, validation_target = target["train"], target["validation"]
        train_ids, validation_ids = source["ids"]["train"], source["ids"]["validation"]
        if train_ids != target["ids"]["train"] or validation_ids != target["ids"]["validation"]:
            _fail("BLOCKED_RUNTIME_CONTRACT", f"{source_name} source/target image order mismatch")
        if not fake and (train_source.shape[0], validation_source.shape[0]) != (800, 100):
            _fail("BLOCKED_RUNTIME_CONTRACT", "real runtime requires exactly 800 Train and 100 Validation images")
        methods: dict[str, Any] = {}
        ridge_records: list[dict[str, Any]] = []
        fit_specs: list[tuple[str, str, float | None]] = [
            ("CALIBRATED", "CALIBRATED", None),
            ("ORTHOGONAL_PROCRUSTES", "ORTHOGONAL_PROCRUSTES", None),
            ("FULL_LINEAR", "FULL_LINEAR", None),
            *[(f"RIDGE_{value:g}", "RIDGE", value) for value in linear.RIDGE_LAMBDA_NORMALIZED_GRID],
        ]
        for label, method, lambda_value in fit_specs:
            fit, adapted = _fit_and_apply(method, train_source, train_target, validation_source, lambda_value)
            connector_values = connector.apply(adapted, batch_size=connector_batch_size)
            patch = linear.patch_metrics(adapted, validation_target, train_target.mean(axis=(0, 1), dtype=np.float64))
            retrieval = linear.same_image_retrieval(adapted, validation_target, validation_ids, target["ids"]["validation"])
            connector_metrics = _connector_metrics(
                connector_values,
                target["connectors"]["validation"],
                target["connectors"]["train"].mean(axis=(0, 1), dtype=np.float64),
            )
            record = {
                "method": method,
                "lambda_normalized": lambda_value,
                "patch_cosine": float(patch["cosine"]), "patch_nmse": float(patch["nmse"]), "patch_r2": float(patch["r2"]),
                "same_image_recall_at_1": float(retrieval["recall_at_1"]),
                "same_image_recall_at_5": float(retrieval["recall_at_5"]),
                **connector_metrics,
            }
            methods[label] = record
            arrays = {"runtime_identity": np.asarray(identity, dtype="U64"), **_fit_arrays(fit)}
            buffer = io.BytesIO(); np.savez(buffer, **arrays)
            relative = f"fits/{source_name.lower()}_{label.lower()}.npz"
            fit_artifacts[relative] = _write_new(output / relative, buffer.getvalue())
            if method == "RIDGE":
                ridge_records.append({"split": "validation", **record})
        selection = linear.select_ridge_candidate(ridge_records)
        sources_result[source_name] = {"methods": methods, "ridge_selection": selection}
    if int(getattr(connector, "vision_calls", 0)) != 0 or int(getattr(connector, "generation_calls", 0)) != 0:
        _fail("BLOCKED_CONNECTOR_BOUNDARY", "connector boundary recorded vision/generation activity")
    result = {
        "schema_id": RUNTIME_SCHEMA_ID,
        "status": SUCCESS_STATUS,
        "runtime_identity_sha256": identity,
        "t001d_plan_id": binding["t001d_plan_id"],
        "sources": sources_result,
        "fit_artifacts": fit_artifacts,
        "test_scientific_access": 0,
        "test_metric_count": 0,
        "generation_call_count": 0,
        "connector_vision_call_count": 0,
        "resumed": False,
    }
    _write_new(output / "result.json", _canonical_bytes(result))
    return result
