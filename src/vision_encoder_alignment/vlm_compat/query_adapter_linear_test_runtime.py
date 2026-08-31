"""Finalized held-out Test and generation runtime for Phase 2 linear adapters."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_bank_preflight as bank_preflight
from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


TEST_BINDING_SCHEMA_ID = "vea.p2.query_adapter_linear_test_binding.v1"
TEST_RUNTIME_SCHEMA_ID = "vea.p2.query_adapter_linear_test_runtime.v1"
GENERATION_UNIT_SCHEMA_ID = "vea.p2.query_adapter_linear_generation_unit.v1"
SUCCESS_STATUS = "PASS_QUERY_ADAPTER_LINEAR_TEST_GENERATION"
FINALIZED_METHODS = ("CALIBRATED", "ORTHOGONAL_PROCRUSTES", "FULL_LINEAR", "RIDGE_SELECTED")


class TestRuntimeError(RuntimeError):
    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


class GeneratorProtocol(Protocol):
    native_calls: int
    adapted_calls: int

    def apply_connector(self, patches: np.ndarray, *, batch_size: int) -> np.ndarray: ...
    def generate_native(self, image_path: str | Path) -> Any: ...
    def generate_adapted(self, image_path: str | Path, patch: np.ndarray) -> Any: ...


def _fail(classification: str, message: str) -> None:
    raise TestRuntimeError(classification, message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def load_test_binding(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("BLOCKED_TEST_BINDING", f"Test binding is unavailable or invalid: {error}")
    required = {
        "schema_id", "experiment_id", "bank_binding_config_path", "validation_result_sha256",
        "validation_runtime_identity_sha256", "selected_ridge_lambda_normalized",
        "finalized_adapter_artifacts", "generation_plan", "path_identity_policy",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("schema_id") != TEST_BINDING_SCHEMA_ID:
        _fail("BLOCKED_TEST_BINDING", "Test binding schema/fields mismatch")
    if value.get("experiment_id") != linear.EXPERIMENT_ID:
        _fail("BLOCKED_TEST_BINDING", "Test binding experiment identity mismatch")
    if not _valid_sha(value.get("validation_result_sha256")) or not _valid_sha(value.get("validation_runtime_identity_sha256")):
        _fail("BLOCKED_TEST_BINDING", "Test binding Validation SHA identity is invalid")
    if value.get("selected_ridge_lambda_normalized") != {"SIGLIP2": 1e-6, "DINOV3": 1e-4}:
        _fail("BLOCKED_TEST_BINDING", "accepted Validation Ridge selection differs")
    artifacts = value.get("finalized_adapter_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(linear.SOURCE_ORDER):
        _fail("BLOCKED_TEST_BINDING", "finalized adapter source keys differ")
    for source in linear.SOURCE_ORDER:
        if not isinstance(artifacts[source], dict) or tuple(artifacts[source]) != FINALIZED_METHODS:
            _fail("BLOCKED_TEST_BINDING", f"{source} finalized method order differs")
        for spec in artifacts[source].values():
            if not isinstance(spec, dict) or set(spec) != {"relative_path", "sha256"} or not _valid_sha(spec.get("sha256")):
                _fail("BLOCKED_TEST_BINDING", f"{source} finalized artifact identity is invalid")
    if value.get("generation_plan") != {"test_images": 100, "native": 100, "adapted": 800, "total": 900}:
        _fail("BLOCKED_TEST_BINDING", "production generation plan differs from 100+800=900")
    return value


def verify_validation_run(binding: Mapping[str, Any], validation_root: str | Path) -> dict[str, Any]:
    root = Path(validation_root).resolve()
    result_path = root / "result.json"
    if not result_path.is_file() or t001d.sha256_file(result_path) != binding["validation_result_sha256"]:
        _fail("BLOCKED_VALIDATION_IDENTITY", "T002D result SHA differs from accepted Validation binding")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail("BLOCKED_VALIDATION_IDENTITY", f"T002D result cannot be read: {error}")
    if result.get("runtime_identity_sha256") != binding["validation_runtime_identity_sha256"]:
        _fail("BLOCKED_VALIDATION_IDENTITY", "T002D runtime identity differs")
    if result.get("test_scientific_access") != 0 or result.get("test_metric_count") != 0 or result.get("generation_call_count") != 0:
        _fail("BLOCKED_VALIDATION_IDENTITY", "T002D result contains forbidden Test/generation activity")
    for source, expected_lambda in binding["selected_ridge_lambda_normalized"].items():
        try:
            observed = result["sources"][source]["ridge_selection"]["selected_lambda_normalized"]
        except (KeyError, TypeError):
            _fail("BLOCKED_VALIDATION_IDENTITY", f"T002D {source} selection is unavailable")
        if observed != expected_lambda:
            _fail("BLOCKED_VALIDATION_IDENTITY", f"T002D {source} Ridge selection differs")
    result_artifacts = result.get("fit_artifacts")
    if not isinstance(result_artifacts, Mapping):
        _fail("BLOCKED_VALIDATION_IDENTITY", "T002D fit artifact manifest is unavailable")
    for methods in binding["finalized_adapter_artifacts"].values():
        for spec in methods.values():
            relative, expected = spec["relative_path"], spec["sha256"]
            path = root / relative
            if result_artifacts.get(relative) != expected or not path.is_file() or t001d.sha256_file(path) != expected:
                _fail("BLOCKED_VALIDATION_IDENTITY", f"finalized adapter artifact differs: {relative}")
    return result


def _array(archive: Any, name: str) -> np.ndarray:
    try:
        return np.array(archive[name], copy=True, order="C")
    except KeyError:
        _fail("BLOCKED_ADAPTER_ARTIFACT", f"finalized adapter field is missing: {name}")


def load_finalized_adapters(binding: Mapping[str, Any], validation_root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(validation_root).resolve()
    runtime_identity = binding["validation_runtime_identity_sha256"]
    result: dict[str, dict[str, Any]] = {}
    for source, methods in binding["finalized_adapter_artifacts"].items():
        result[source] = {}
        for method, spec in methods.items():
            path = root / spec["relative_path"]
            try:
                with np.load(path, allow_pickle=False) as archive:
                    persisted = archive["runtime_identity"]
                    if persisted.shape != () or str(persisted.item()) != runtime_identity:
                        _fail("BLOCKED_ADAPTER_ARTIFACT", f"{source} {method} runtime identity differs")
                    if method == "CALIBRATED":
                        fit = linear.CalibrationFit(*(_array(archive, name) for name in ("source_mean", "target_mean", "source_std", "target_std", "scale", "bias")))
                    elif method == "ORTHOGONAL_PROCRUSTES":
                        calibration = linear.CalibrationFit(*(_array(archive, f"part0_{name}") for name in ("source_mean", "target_mean", "source_std", "target_std", "scale", "bias")))
                        procrustes = linear.ProcrustesFit(*(_array(archive, f"part1_{name}") for name in ("rotation", "source_mean", "target_mean")))
                        fit = (calibration, procrustes)
                    elif method == "FULL_LINEAR":
                        fit = linear.FullLinearFit(
                            _array(archive, "weights"), _array(archive, "source_mean"), _array(archive, "target_mean"),
                            int(archive["rank"].item()), _array(archive, "residual_sum_squares"),
                        )
                    else:
                        fit = linear.RidgeFit(
                            _array(archive, "weights"), _array(archive, "source_mean"), _array(archive, "target_mean"),
                            float(archive["lambda_normalized"].item()), float(archive["alpha"].item()), int(archive["train_row_count"].item()),
                        )
                        if fit.lambda_normalized != binding["selected_ridge_lambda_normalized"][source]:
                            _fail("BLOCKED_ADAPTER_ARTIFACT", f"{source} persisted Ridge lambda differs")
            except TestRuntimeError:
                raise
            except (OSError, ValueError, KeyError) as error:
                _fail("BLOCKED_ADAPTER_ARTIFACT", f"cannot load finalized adapter {source}/{method}: {error}")
            result[source][method] = fit
    return result


def _apply_finalized(method: str, values: np.ndarray, fit: Any) -> np.ndarray:
    if method == "CALIBRATED":
        return linear.apply_calibration(values, fit)
    if method == "ORTHOGONAL_PROCRUSTES":
        calibration, procrustes = fit
        return linear.apply_orthogonal_procrustes(linear.apply_calibration(values, calibration), procrustes)
    if method == "FULL_LINEAR":
        return linear.apply_full_linear(values, fit)
    if method == "RIDGE_SELECTED":
        return linear.apply_ridge(values, fit)
    _fail("BLOCKED_TEST_CONTRACT", f"unexpected finalized method: {method}")


def _atomic_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    if path.exists() or partial.exists():
        _fail("BLOCKED_TEST_RESUME", f"atomic destination/partial already exists: {path.name}")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except BaseException:
        raise


def _load_generation(path: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail("BLOCKED_TEST_RESUME", f"generation artifact is invalid: {error}")
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            _fail("BLOCKED_TEST_RESUME", f"generation artifact identity differs: {path.name}/{key}")
    if value.get("schema_id") != GENERATION_UNIT_SCHEMA_ID or value.get("generation_success") is not True:
        _fail("BLOCKED_TEST_RESUME", f"generation artifact contract differs: {path.name}")
    tokens = value.get("token_ids")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens) or not isinstance(value.get("decoded_text"), str):
        _fail("BLOCKED_TEST_RESUME", f"generation payload contract differs: {path.name}")
    if value.get("method") == "NATIVE":
        if not isinstance(value.get("vision_calls"), int) or value["vision_calls"] < 1 or value.get("injection_calls") != 0:
            _fail("BLOCKED_TEST_RESUME", f"native generation path evidence differs: {path.name}")
    else:
        if value.get("vision_calls") != 0 or value.get("injection_calls") != 1 or value.get("injection_restored") is not True:
            _fail("BLOCKED_TEST_RESUME", f"adapted generation path evidence differs: {path.name}")
        if value.get("native_exact_match") not in (0, 1) or not isinstance(value.get("token_unigram_f1"), (int, float)):
            _fail("BLOCKED_TEST_RESUME", f"adapted generation metrics differ: {path.name}")
    return value


def _publish_generation(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    existing = _load_generation(path, {key: record[key] for key in ("runtime_identity_sha256", "image_id", "source", "method")})
    if existing is not None:
        return existing
    _atomic_new(path, _canonical_bytes(record))
    return dict(record)


def _metrics(adapted: np.ndarray, target: Mapping[str, Any], ids: Sequence[str], generator: GeneratorProtocol, batch_size: int) -> dict[str, float]:
    patch = linear.patch_metrics(adapted, target["test"], target["train_patch_mean"])
    retrieval = linear.same_image_retrieval(adapted, target["test"], ids, target["ids"])
    connector_values = generator.apply_connector(adapted, batch_size=batch_size)
    connector = linear.patch_metrics(connector_values, target["connectors"], target["train_connector_mean"])
    return {
        "patch_cosine": float(patch["cosine"]), "patch_nmse": float(patch["nmse"]), "patch_r2": float(patch["r2"]),
        "connector_cosine": float(connector["cosine"]), "connector_nmse": float(connector["nmse"]),
        "same_image_recall_at_1": float(retrieval["recall_at_1"]), "same_image_recall_at_5": float(retrieval["recall_at_5"]),
    }


def run_held_out_test(
    *, config: Mapping[str, Any], binding: Mapping[str, Any], banks: Mapping[str, Any], adapters: Mapping[str, Any],
    image_paths: Sequence[str | Path], generator: GeneratorProtocol, output_root: str | Path,
    connector_batch_size: int, adapter_config_sha256: str, test_binding_sha256: str, fake: bool = False,
) -> dict[str, Any]:
    """Apply exactly eight finalized adapters; no fit or selector is called."""

    linear.validate_query_adapter_config(config)
    production_plan = linear.generation_plan(config)
    if production_plan != binding["generation_plan"]:
        _fail("BLOCKED_TEST_BINDING", "derived production generation plan differs from binding")
    target = banks["TARGET_CLIP"]
    test_count = len(target["ids"])
    if len(image_paths) != test_count or (not fake and test_count != 100):
        _fail("BLOCKED_TEST_CONTRACT", "held-out Test image count/path contract mismatch")
    finalized = [(source, method) for source in linear.SOURCE_ORDER for method in FINALIZED_METHODS]
    if len(finalized) != 8 or any(set(adapters[source]) != set(FINALIZED_METHODS) for source in linear.SOURCE_ORDER):
        _fail("BLOCKED_TEST_CONTRACT", "exactly eight finalized adapter configurations are required")
    identity_material = {
        "schema_id": TEST_RUNTIME_SCHEMA_ID, "adapter_config_sha256": adapter_config_sha256,
        "test_binding_sha256": test_binding_sha256, "validation_result_sha256": binding["validation_result_sha256"],
        "validation_runtime_identity_sha256": binding["validation_runtime_identity_sha256"], "fake": fake,
    }
    runtime_identity = _sha(identity_material)
    output = Path(output_root).resolve()
    plan = {**identity_material, "runtime_identity_sha256": runtime_identity, "production_generation_plan": production_plan,
            "actual_generation_plan": {"test_images": test_count, "native": test_count, "adapted": test_count * 8, "total": test_count * 9}}
    plan_path = output / "test_plan.json"
    if plan_path.exists():
        if plan_path.read_bytes() != _canonical_bytes(plan):
            _fail("BLOCKED_TEST_RESUME", "existing Test plan identity differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_new(plan_path, _canonical_bytes(plan))
    result_path = output / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            _fail("BLOCKED_TEST_RESUME", f"final result is invalid: {error}")
        if result.get("runtime_identity_sha256") != runtime_identity:
            _fail("BLOCKED_TEST_RESUME", "final result runtime identity differs")
        expected_counts = {"native": test_count, "adapted": test_count * 8, "total": test_count * 9}
        if result.get("generation_counts") != expected_counts or result.get("finalized_configuration_count") != 8:
            _fail("BLOCKED_TEST_RESUME", "final result count contract differs")
        for index, image_id in enumerate(target["ids"]):
            native_path = output / "generations" / "native" / f"{index:04d}.json"
            if _load_generation(native_path, {"runtime_identity_sha256": runtime_identity, "image_id": image_id, "source": "TARGET_CLIP", "method": "NATIVE"}) is None:
                _fail("BLOCKED_TEST_RESUME", f"final result is missing native generation {index}")
            for source, method in finalized:
                adapted_path = output / "generations" / source.lower() / method.lower() / f"{index:04d}.json"
                if _load_generation(adapted_path, {"runtime_identity_sha256": runtime_identity, "image_id": image_id, "source": source, "method": method}) is None:
                    _fail("BLOCKED_TEST_RESUME", f"final result is missing adapted generation {source}/{method}/{index}")
        result["resumed"] = True
        return result

    metric_results: dict[str, Any] = {}
    for source, method in finalized:
        adapted = _apply_finalized(method, banks[source]["test"], adapters[source][method])
        metric_results[f"{source}/{method}"] = _metrics(adapted, target, target["ids"], generator, connector_batch_size)

    native_records: list[dict[str, Any]] = []
    skipped = 0
    for index, (image_id, image_path) in enumerate(zip(target["ids"], image_paths, strict=True)):
        path = output / "generations" / "native" / f"{index:04d}.json"
        expected = {"runtime_identity_sha256": runtime_identity, "image_id": image_id, "source": "TARGET_CLIP", "method": "NATIVE"}
        record = _load_generation(path, expected)
        if record is None:
            generated = generator.generate_native(image_path)
            record = _publish_generation(path, {
                "schema_id": GENERATION_UNIT_SCHEMA_ID, **expected, "token_ids": list(generated.token_ids),
                "decoded_text": generated.decoded_text, "vision_calls": generated.vision_calls,
                "connector_calls": generated.connector_calls, "injection_calls": 0,
                "injection_restored": True, "generation_success": True,
            })
        else:
            skipped += 1
        native_records.append(record)

    generation_metric_rows: list[dict[str, Any]] = []
    for source, method in finalized:
        adapted = _apply_finalized(method, banks[source]["test"], adapters[source][method])
        for index, (image_id, image_path, native_record) in enumerate(zip(target["ids"], image_paths, native_records, strict=True)):
            path = output / "generations" / source.lower() / method.lower() / f"{index:04d}.json"
            expected = {"runtime_identity_sha256": runtime_identity, "image_id": image_id, "source": source, "method": method}
            record = _load_generation(path, expected)
            if record is None:
                generated = generator.generate_adapted(image_path, adapted[index : index + 1])
                metric = t001d.generation_metrics(generated.token_ids, native_record["token_ids"])
                record = _publish_generation(path, {
                    "schema_id": GENERATION_UNIT_SCHEMA_ID, **expected, "token_ids": list(generated.token_ids),
                    "decoded_text": generated.decoded_text, "vision_calls": generated.vision_calls,
                    "connector_calls": generated.connector_calls, "injection_calls": generated.injection_calls,
                    "injection_restored": generated.injection_restored, "generation_success": True,
                    "native_exact_match": int(metric["exact_match"]),
                    "token_unigram_f1": float(metric["token_unigram_F1"]),
                })
            else:
                skipped += 1
            generation_metric_rows.append({
                "source": source, "method": method, "image_id": image_id,
                "native_exact_match": record["native_exact_match"], "token_unigram_f1": record["token_unigram_f1"],
            })
    expected_total = test_count * 9
    if len(native_records) + len(generation_metric_rows) != expected_total:
        _fail("BLOCKED_TEST_GENERATION", "final generation count differs from plan")
    result = {
        "schema_id": TEST_RUNTIME_SCHEMA_ID, "status": SUCCESS_STATUS, "runtime_identity_sha256": runtime_identity,
        "finalized_configuration_count": len(finalized), "test_image_count": test_count,
        "test_metrics": metric_results, "generation_metrics": generation_metric_rows,
        "generation_counts": {"native": test_count, "adapted": test_count * 8, "total": expected_total},
        "production_generation_plan": production_plan, "resume_skipped_generation_count": skipped, "resumed": False,
    }
    _atomic_new(result_path, _canonical_bytes(result))
    return result


def load_real_test_inputs(run_root: str | Path, dataset_root: str | Path) -> tuple[dict[str, Any], list[Path]]:
    """Load Test banks and Train target means after all identity gates have passed."""

    root, dataset = Path(run_root).resolve(), Path(dataset_root).resolve()
    plan, plan_id = t001d.load_bound_plan(root)
    rows = plan.get("manifest_rows", [])
    test_rows = [row for row in rows if row.get("split") == "test"]
    if len(test_rows) != 100:
        _fail("BLOCKED_TEST_DATA", "T001D Test manifest count differs from 100")
    test_ids = [str(row["image_id"]) for row in test_rows]
    image_paths: list[Path] = []
    for row in test_rows:
        path = (dataset / str(row["relative_path"])).resolve()
        try:
            path.relative_to(dataset)
        except ValueError:
            _fail("BLOCKED_TEST_DATA", "Test image path escapes dataset root")
        if not path.is_file() or t001d.sha256_file(path) != row["source_sha256"]:
            _fail("BLOCKED_TEST_DATA", f"Test image identity differs: {row['image_id']}")
        image_paths.append(path)
    split_by_id = {str(row["image_id"]): str(row["split"]) for row in rows}
    banks: dict[str, Any] = {}
    for bank in (*linear.SOURCE_ORDER, "TARGET_CLIP"):
        units = sorted(
            (unit for unit in plan["expected_units"] if unit["stage"] == bank_preflight.REQUIRED_STAGES[bank]),
            key=lambda unit: int(unit["chunk_index"]),
        )
        test_parts: list[np.ndarray] = []
        connector_parts: list[np.ndarray] = []
        observed_ids: list[str] = []
        train_patch_sum = np.zeros(768, dtype=np.float64)
        train_connector_sum = np.zeros(1536, dtype=np.float64)
        train_patch_rows = train_connector_rows = 0
        for unit in units:
            if not t001d.validate_unit_completion(root, unit, plan_id):
                _fail("BLOCKED_TEST_DATA", f"T001D feature unit is incomplete: {unit['unit_id']}")
            with np.load(root / unit["artifact_relative_path"], allow_pickle=False) as archive:
                ids = [str(value) for value in archive["image_ids"].tolist()]
                test_indices = [index for index, image_id in enumerate(ids) if split_by_id.get(image_id) == "test"]
                train_indices = [index for index, image_id in enumerate(ids) if split_by_id.get(image_id) == "train"]
                if test_indices:
                    test_parts.append(np.ascontiguousarray(archive["patches"][test_indices], dtype=np.float32))
                    observed_ids.extend(ids[index] for index in test_indices)
                    if bank == "TARGET_CLIP":
                        connector_parts.append(np.ascontiguousarray(archive["connectors"][test_indices], dtype=np.float32))
                if bank == "TARGET_CLIP" and train_indices:
                    train_patch = np.asarray(archive["patches"][train_indices], dtype=np.float64)
                    train_connector = np.asarray(archive["connectors"][train_indices], dtype=np.float64)
                    train_patch_sum += train_patch.sum(axis=(0, 1)); train_patch_rows += train_patch.shape[0] * train_patch.shape[1]
                    train_connector_sum += train_connector.sum(axis=(0, 1)); train_connector_rows += train_connector.shape[0] * train_connector.shape[1]
        test = np.concatenate(test_parts, axis=0)
        if observed_ids != test_ids or test.shape != (100, 196, 768) or test.dtype != np.dtype("float32") or not np.isfinite(test).all():
            _fail("BLOCKED_TEST_DATA", f"{bank} Test bank identity/shape differs")
        banks[bank] = {"test": test, "ids": observed_ids}
        if bank == "TARGET_CLIP":
            connectors = np.concatenate(connector_parts, axis=0)
            if connectors.shape != (100, 196, 1536) or connectors.dtype != np.dtype("float32") or not np.isfinite(connectors).all():
                _fail("BLOCKED_TEST_DATA", "TARGET_CLIP Test connector bank differs")
            if train_patch_rows != 800 * 196 or train_connector_rows != 800 * 196:
                _fail("BLOCKED_TEST_DATA", "Train target mean row count differs")
            banks[bank].update(
                connectors=connectors,
                train_patch_mean=train_patch_sum / train_patch_rows,
                train_connector_mean=train_connector_sum / train_connector_rows,
            )
    return banks, image_paths
