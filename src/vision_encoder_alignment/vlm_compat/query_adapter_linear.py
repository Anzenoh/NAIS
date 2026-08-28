"""Pure NumPy core for the frozen Phase 2 direct linear-adapter configuration.

This module intentionally contains no feature-bank, model, connector, or GPU
integration.  Runtime integration is responsible for resolving artifacts and
for applying the production `[N,196,768]` contract before calling this core.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONFIG_SCHEMA_ID = "vea.p2.query_adapter_linear_config.v1"
EXPERIMENT_ID = "vea.p2.query_adapter_linear.v1"
SOURCE_ORDER = ("SIGLIP2", "DINOV3")
METHOD_ORDER = ("CALIBRATED", "ORTHOGONAL_PROCRUSTES", "FULL_LINEAR", "RIDGE")
FINALIZED_METHOD_ORDER = ("CALIBRATED", "ORTHOGONAL_PROCRUSTES", "FULL_LINEAR", "RIDGE_SELECTED")
RIDGE_LAMBDA_NORMALIZED_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
GENERATION_PROMPT = "What is shown in this image? Answer briefly."


class QueryAdapterLinearError(ValueError):
    """Raised when frozen configuration or numerical contracts are violated."""


@dataclass(frozen=True)
class CalibrationFit:
    source_mean: np.ndarray
    target_mean: np.ndarray
    source_std: np.ndarray
    target_std: np.ndarray
    scale: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class ProcrustesFit:
    rotation: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray


@dataclass(frozen=True)
class FullLinearFit:
    weights: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    rank: int
    residual_sum_squares: np.ndarray


@dataclass(frozen=True)
class RidgeFit:
    weights: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    lambda_normalized: float
    alpha: float
    train_row_count: int


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QueryAdapterLinearError(f"{label} must be an object")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise QueryAdapterLinearError(f"frozen config drift: {label}")


def _as_float64(value: Any, label: str, *, ndim: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:  # pragma: no cover - NumPy raises varied exceptions.
        raise QueryAdapterLinearError(f"{label} is not an array") from exc
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise QueryAdapterLinearError(f"{label} must be a real numeric array")
    if ndim is not None and array.ndim != ndim:
        raise QueryAdapterLinearError(f"{label} must have {ndim} dimensions")
    converted = np.asarray(array, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise QueryAdapterLinearError(f"{label} contains NaN or Inf")
    return converted


def _as_train_rows(source: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    x = _as_float64(source, "source")
    y = _as_float64(target, "target")
    if x.shape != y.shape:
        raise QueryAdapterLinearError("source and target shapes must match")
    if x.ndim < 2:
        raise QueryAdapterLinearError("fitting arrays must have a sample axis and feature axis")
    if x.shape[-1] == 0:
        raise QueryAdapterLinearError("feature dimension must be nonempty")
    x_rows = x.reshape(-1, x.shape[-1])
    y_rows = y.reshape(-1, y.shape[-1])
    if x_rows.shape[0] == 0:
        raise QueryAdapterLinearError("fitting rows must be nonempty")
    return x_rows, y_rows


def _float32_contiguous(value: np.ndarray) -> np.ndarray:
    return np.array(value, dtype=np.float32, order="C", copy=True)


def load_query_adapter_config(path: str | Path) -> dict[str, Any]:
    """Load and fail-closed validate the authoritative frozen JSON config."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QueryAdapterLinearError("query adapter config is unavailable or invalid JSON") from exc
    return validate_query_adapter_config(value)


def validate_query_adapter_config(value: Any) -> dict[str, Any]:
    """Validate all scientifically load-bearing values without copying model IDs."""

    config = dict(_require_mapping(value, "config"))
    _require_equal(config.get("schema_id"), CONFIG_SCHEMA_ID, "schema_id")
    _require_equal(config.get("experiment_id"), EXPERIMENT_ID, "experiment_id")
    _require_equal(tuple(config.get("source_order", ())), SOURCE_ORDER, "source_order")
    _require_equal(tuple(config.get("method_order", ())), METHOD_ORDER, "method_order")
    target = _require_mapping(config.get("target"), "target")
    _require_equal(target.get("key"), "TARGET_CLIP", "target.key")
    _require_equal(target.get("patch_shape"), [196, 768], "target.patch_shape")
    sources = _require_mapping(config.get("sources"), "sources")
    for source_name in SOURCE_ORDER:
        source = _require_mapping(sources.get(source_name), f"sources.{source_name}")
        _require_equal(source.get("patch_shape"), [196, 768], f"sources.{source_name}.patch_shape")
        _require_equal(source.get("frozen"), True, f"sources.{source_name}.frozen")
    manifest = _require_mapping(config.get("manifest"), "manifest")
    _require_equal(manifest.get("row_count"), 1000, "manifest.row_count")
    _require_equal(manifest.get("patches_per_image"), 196, "manifest.patches_per_image")
    roles = _require_mapping(manifest.get("partition_roles"), "manifest.partition_roles")
    for name, count, fit, select, test in (
        ("train", 800, True, False, False),
        ("validation", 100, False, True, False),
        ("test", 100, False, False, True),
    ):
        role = _require_mapping(roles.get(name), f"manifest.partition_roles.{name}")
        _require_equal(role.get("image_count"), count, f"{name}.image_count")
        _require_equal(role.get("fit"), fit, f"{name}.fit")
        _require_equal(role.get("select"), select, f"{name}.select")
        _require_equal(role.get("test"), test, f"{name}.test")
    methods = _require_mapping(config.get("methods"), "methods")
    ridge = _require_mapping(methods.get("RIDGE"), "methods.RIDGE")
    _require_equal(tuple(ridge.get("lambda_normalized_grid", ())), RIDGE_LAMBDA_NORMALIZED_GRID, "RIDGE.lambda_normalized_grid")
    _require_equal(ridge.get("selection_partition"), "validation", "RIDGE.selection_partition")
    selection = _require_mapping(config.get("ridge_validation_selection"), "ridge_validation_selection")
    _require_equal(selection.get("partition"), "validation", "ridge_validation_selection.partition")
    _require_equal(selection.get("test_metrics_forbidden"), True, "ridge_validation_selection.test_metrics_forbidden")
    hierarchy = tuple(
        (item.get("metric"), item.get("direction"))
        for item in selection.get("hierarchy", ())
        if isinstance(item, Mapping)
    )
    _require_equal(
        hierarchy,
        (
            ("same_image_recall_at_1", "highest"),
            ("same_image_recall_at_5", "highest"),
            ("connector_cosine", "highest"),
            ("lambda_normalized", "smallest"),
        ),
        "ridge_validation_selection.hierarchy",
    )
    generation = _require_mapping(config.get("generation"), "generation")
    for key, expected in (
        ("prompt", GENERATION_PROMPT),
        ("max_new_tokens", 32),
        ("num_beams", 1),
        ("do_sample", False),
        ("top_p", None),
        ("temperature", 0),
        ("planned_native_generations", 100),
        ("planned_adapted_generations", 800),
        ("planned_total_generations", 900),
    ):
        _require_equal(generation.get(key), expected, f"generation.{key}")
    numeric = _require_mapping(config.get("numeric_policy"), "numeric_policy")
    for key, expected in (
        ("fit_accumulation_dtype", "float64"),
        ("linear_solve_dtype", "float64"),
        ("stored_adapter_parameter_dtype", "float32"),
        ("stored_adapted_feature_dtype", "float32"),
        ("similarity_metric_dtype", "float64"),
        ("target_vlm_runtime_dtype", "float16"),
        ("mixed_precision_fitting", False),
    ):
        _require_equal(numeric.get(key), expected, f"numeric_policy.{key}")
    evaluation = _require_mapping(config.get("test_evaluation"), "test_evaluation")
    _require_equal(tuple(evaluation.get("finalized_method_order", ())), FINALIZED_METHOD_ORDER, "test_evaluation.finalized_method_order")
    _require_equal(evaluation.get("source_count"), 2, "test_evaluation.source_count")
    _require_equal(evaluation.get("adapted_configuration_count"), 8, "test_evaluation.adapted_configuration_count")
    return config


def validate_feature_bank(
    features: Any,
    image_ids: Sequence[str],
    *,
    expected_image_count: int,
    expected_patch_count: int,
    expected_feature_dim: int,
) -> np.ndarray:
    """Validate a production-shaped bank before any patch flattening occurs."""

    values = _as_float64(features, "features", ndim=3)
    expected_shape = (expected_image_count, expected_patch_count, expected_feature_dim)
    if values.shape != expected_shape:
        raise QueryAdapterLinearError(f"features must have shape {expected_shape}, observed {values.shape}")
    if len(image_ids) != expected_image_count or len(set(image_ids)) != len(image_ids):
        raise QueryAdapterLinearError("image IDs must be unique and match the image axis")
    if any(not isinstance(image_id, str) or not image_id for image_id in image_ids):
        raise QueryAdapterLinearError("image IDs must be nonempty strings")
    return _float32_contiguous(values)


def validate_feature_pair(
    source: Any,
    target: Any,
    source_image_ids: Sequence[str],
    target_image_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Require exact source/target geometry and same-Query image ordering."""

    source_values = _as_float64(source, "source", ndim=3)
    target_values = _as_float64(target, "target", ndim=3)
    if source_values.shape != target_values.shape:
        raise QueryAdapterLinearError("source and target feature shapes must match")
    if tuple(source_image_ids) != tuple(target_image_ids):
        raise QueryAdapterLinearError("source and target image-ID order must match exactly")
    if len(source_image_ids) != source_values.shape[0] or len(set(source_image_ids)) != len(source_image_ids):
        raise QueryAdapterLinearError("image IDs must uniquely match the feature image axis")
    return _float32_contiguous(source_values), _float32_contiguous(target_values)


def validate_partition_image_ids(
    train_image_ids: Sequence[str], validation_image_ids: Sequence[str], test_image_ids: Sequence[str]
) -> None:
    """Fail closed on duplicates or any image-level partition overlap."""

    partitions = {
        "train": tuple(train_image_ids),
        "validation": tuple(validation_image_ids),
        "test": tuple(test_image_ids),
    }
    for name, values in partitions.items():
        if len(values) != len(set(values)) or any(not isinstance(value, str) or not value for value in values):
            raise QueryAdapterLinearError(f"{name} image IDs must be unique nonempty strings")
    names = tuple(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if set(partitions[left]) & set(partitions[right]):
                raise QueryAdapterLinearError(f"image-level split overlap: {left}/{right}")


def fit_calibration(source: Any, target: Any, *, std_floor: float = 1e-6) -> CalibrationFit:
    if not np.isfinite(std_floor) or std_floor <= 0.0:
        raise QueryAdapterLinearError("std_floor must be finite and positive")
    x, y = _as_train_rows(source, target)
    source_mean = x.mean(axis=0, dtype=np.float64)
    target_mean = y.mean(axis=0, dtype=np.float64)
    source_std = x.std(axis=0, ddof=0, dtype=np.float64)
    target_std = y.std(axis=0, ddof=0, dtype=np.float64)
    scale = target_std / np.maximum(source_std, float(std_floor))
    bias = target_mean - source_mean * scale
    return CalibrationFit(*(_float32_contiguous(value) for value in (source_mean, target_mean, source_std, target_std, scale, bias)))


def apply_calibration(features: Any, fit: CalibrationFit) -> np.ndarray:
    values = _as_float64(features, "features")
    if values.ndim < 1 or values.shape[-1] != fit.scale.shape[0]:
        raise QueryAdapterLinearError("calibration feature dimension mismatch")
    result = values * np.asarray(fit.scale, dtype=np.float64) + np.asarray(fit.bias, dtype=np.float64)
    return _float32_contiguous(result)


def fit_orthogonal_procrustes(calibrated_source: Any, target: Any) -> ProcrustesFit:
    x, y = _as_train_rows(calibrated_source, target)
    source_mean = x.mean(axis=0, dtype=np.float64)
    target_mean = y.mean(axis=0, dtype=np.float64)
    u, _singular_values, vt = np.linalg.svd((x - source_mean).T @ (y - target_mean), full_matrices=False)
    return ProcrustesFit(_float32_contiguous(u @ vt), _float32_contiguous(source_mean), _float32_contiguous(target_mean))


def apply_orthogonal_procrustes(features: Any, fit: ProcrustesFit) -> np.ndarray:
    values = _as_float64(features, "features")
    if values.ndim < 1 or values.shape[-1] != fit.rotation.shape[0]:
        raise QueryAdapterLinearError("Procrustes feature dimension mismatch")
    result = (values - np.asarray(fit.source_mean, dtype=np.float64)) @ np.asarray(fit.rotation, dtype=np.float64)
    result += np.asarray(fit.target_mean, dtype=np.float64)
    return _float32_contiguous(result)


def fit_full_linear(source: Any, target: Any) -> FullLinearFit:
    x, y = _as_train_rows(source, target)
    source_mean = x.mean(axis=0, dtype=np.float64)
    target_mean = y.mean(axis=0, dtype=np.float64)
    weights, residuals, rank, _singular_values = np.linalg.lstsq(x - source_mean, y - target_mean, rcond=None)
    return FullLinearFit(
        _float32_contiguous(weights),
        _float32_contiguous(source_mean),
        _float32_contiguous(target_mean),
        int(rank),
        _float32_contiguous(np.asarray(residuals, dtype=np.float64)),
    )


def apply_full_linear(features: Any, fit: FullLinearFit) -> np.ndarray:
    values = _as_float64(features, "features")
    if values.ndim < 1 or values.shape[-1] != fit.weights.shape[0]:
        raise QueryAdapterLinearError("full-linear feature dimension mismatch")
    result = (values - np.asarray(fit.source_mean, dtype=np.float64)) @ np.asarray(fit.weights, dtype=np.float64)
    result += np.asarray(fit.target_mean, dtype=np.float64)
    return _float32_contiguous(result)


def _validate_lambda(lambda_normalized: float, allowed_lambdas: Sequence[float]) -> float:
    if not isinstance(lambda_normalized, (int, float, np.floating)) or not np.isfinite(lambda_normalized):
        raise QueryAdapterLinearError("lambda_normalized must be finite")
    value = float(lambda_normalized)
    if value <= 0.0 or value not in tuple(float(item) for item in allowed_lambdas):
        raise QueryAdapterLinearError("lambda_normalized is not in the frozen grid")
    return value


def fit_ridge(
    source: Any,
    target: Any,
    lambda_normalized: float,
    *,
    allowed_lambdas: Sequence[float] = RIDGE_LAMBDA_NORMALIZED_GRID,
) -> RidgeFit:
    x, y = _as_train_rows(source, target)
    selected_lambda = _validate_lambda(lambda_normalized, allowed_lambdas)
    source_mean = x.mean(axis=0, dtype=np.float64)
    target_mean = y.mean(axis=0, dtype=np.float64)
    xc = x - source_mean
    yc = y - target_mean
    alpha = selected_lambda * xc.shape[0]
    system = xc.T @ xc + alpha * np.eye(xc.shape[1], dtype=np.float64)
    weights = np.linalg.solve(system, xc.T @ yc)
    return RidgeFit(
        _float32_contiguous(weights),
        _float32_contiguous(source_mean),
        _float32_contiguous(target_mean),
        selected_lambda,
        float(alpha),
        int(xc.shape[0]),
    )


def apply_ridge(features: Any, fit: RidgeFit) -> np.ndarray:
    values = _as_float64(features, "features")
    if values.ndim < 1 or values.shape[-1] != fit.weights.shape[0]:
        raise QueryAdapterLinearError("Ridge feature dimension mismatch")
    result = (values - np.asarray(fit.source_mean, dtype=np.float64)) @ np.asarray(fit.weights, dtype=np.float64)
    result += np.asarray(fit.target_mean, dtype=np.float64)
    return _float32_contiguous(result)


def patch_metrics(candidate: Any, reference: Any, train_target_mean: Any) -> dict[str, Any]:
    """Compute equal-image-weighted patch cosine, NMSE, and R-squared."""

    predicted = _as_float64(candidate, "candidate", ndim=3)
    expected = _as_float64(reference, "reference", ndim=3)
    mean = _as_float64(train_target_mean, "train_target_mean", ndim=1)
    if predicted.shape != expected.shape or predicted.shape[-1] != mean.shape[0]:
        raise QueryAdapterLinearError("patch metric shapes are incompatible")
    predicted_flat = predicted.reshape(predicted.shape[0], -1)
    expected_flat = expected.reshape(expected.shape[0], -1)
    cosine_denominator = np.linalg.norm(predicted_flat, axis=1) * np.linalg.norm(expected_flat, axis=1)
    if np.any(cosine_denominator == 0.0):
        raise QueryAdapterLinearError("patch cosine denominator is zero")
    sse = np.sum((predicted_flat - expected_flat) ** 2, axis=1)
    centered = expected - mean.reshape(1, 1, -1)
    sst = np.sum(centered**2, axis=(1, 2))
    if np.any(sst == 0.0):
        raise QueryAdapterLinearError("patch NMSE/R-squared denominator is zero")
    per_image_cosine = np.sum(predicted_flat * expected_flat, axis=1) / cosine_denominator
    per_image_nmse = sse / sst
    per_image_r2 = 1.0 - sse / sst
    return {
        "per_image_cosine": per_image_cosine,
        "per_image_nmse": per_image_nmse,
        "per_image_r2": per_image_r2,
        "cosine": float(per_image_cosine.mean()),
        "nmse": float(per_image_nmse.mean()),
        "r2": float(per_image_r2.mean()),
    }


def same_image_retrieval(
    candidate: Any,
    reference: Any,
    query_image_ids: Sequence[str],
    reference_image_ids: Sequence[str],
) -> dict[str, Any]:
    """Compute stable cosine ranks with lower reference indices winning ties."""

    queries = _as_float64(candidate, "candidate", ndim=3)
    gallery = _as_float64(reference, "reference", ndim=3)
    if queries.shape != gallery.shape:
        raise QueryAdapterLinearError("retrieval candidate and reference shapes must match")
    if tuple(query_image_ids) != tuple(reference_image_ids):
        raise QueryAdapterLinearError("retrieval requires exact Query/reference image-ID order")
    if len(query_image_ids) != queries.shape[0] or len(set(query_image_ids)) != len(query_image_ids):
        raise QueryAdapterLinearError("retrieval image IDs must uniquely match the image axis")
    query_vectors = queries.mean(axis=1)
    reference_vectors = gallery.mean(axis=1)
    denominator = np.linalg.norm(query_vectors, axis=1, keepdims=True) * np.linalg.norm(reference_vectors, axis=1)[None, :]
    if np.any(denominator == 0.0):
        raise QueryAdapterLinearError("retrieval cosine denominator is zero")
    similarity = (query_vectors @ reference_vectors.T) / denominator
    ranking = np.argsort(-similarity, axis=1, kind="stable")
    expected = np.arange(queries.shape[0])[:, None]
    ranks = (np.argmax(ranking == expected, axis=1) + 1).astype(np.int64)
    return {
        "ranks": ranks,
        "ranking": ranking,
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= min(5, queries.shape[0]))),
    }


def select_ridge_candidate(
    validation_records: Sequence[Mapping[str, Any]],
    *,
    lambda_grid: Sequence[float] = RIDGE_LAMBDA_NORMALIZED_GRID,
) -> dict[str, Any]:
    """Select one Validation-only Ridge candidate using the frozen hierarchy."""

    expected = tuple(float(item) for item in lambda_grid)
    if len(set(expected)) != len(expected):
        raise QueryAdapterLinearError("Ridge lambda grid contains duplicates")
    by_lambda: dict[float, dict[str, Any]] = {}
    required = ("lambda_normalized", "same_image_recall_at_1", "same_image_recall_at_5", "connector_cosine")
    for record_value in validation_records:
        record = dict(_require_mapping(record_value, "validation record"))
        if record.get("split", "validation") != "validation" or any(key.lower().startswith("test") for key in record):
            raise QueryAdapterLinearError("Test selector input is forbidden")
        if any(key not in record for key in required):
            raise QueryAdapterLinearError("Ridge validation record is missing a selector metric")
        selected_lambda = _validate_lambda(float(record["lambda_normalized"]), expected)
        if selected_lambda in by_lambda:
            raise QueryAdapterLinearError("duplicate Ridge lambda candidate")
        for key in required[1:]:
            metric = record[key]
            if not isinstance(metric, (int, float, np.floating)) or not np.isfinite(metric):
                raise QueryAdapterLinearError(f"Ridge selector metric is invalid: {key}")
        by_lambda[selected_lambda] = record
    if set(by_lambda) != set(expected):
        raise QueryAdapterLinearError("all frozen Ridge lambda candidates must be present exactly once")
    ordered = sorted(
        by_lambda.values(),
        key=lambda row: (
            -float(row["same_image_recall_at_1"]),
            -float(row["same_image_recall_at_5"]),
            -float(row["connector_cosine"]),
            float(row["lambda_normalized"]),
        ),
    )
    winner = dict(ordered[0])
    return {
        "selected_lambda_normalized": float(winner["lambda_normalized"]),
        "selected_record": winner,
        "comparison_provenance": [dict(row) for row in ordered],
        "selection_hierarchy": [
            "highest same_image_recall_at_1",
            "highest same_image_recall_at_5",
            "highest connector_cosine",
            "smallest lambda_normalized",
        ],
    }


def generation_plan(config: Mapping[str, Any]) -> dict[str, int]:
    """Derive and validate frozen Test generation counts without model execution."""

    validated = validate_query_adapter_config(config)
    test_images = int(validated["manifest"]["partition_roles"]["test"]["image_count"])
    source_count = len(validated["source_order"])
    method_count = len(validated["test_evaluation"]["finalized_method_order"])
    native = test_images
    adapted = source_count * method_count * test_images
    total = native + adapted
    generation = validated["generation"]
    expected = {
        "native": int(generation["planned_native_generations"]),
        "adapted": int(generation["planned_adapted_generations"]),
        "total": int(generation["planned_total_generations"]),
    }
    observed = {"native": native, "adapted": adapted, "total": total}
    if observed != expected:
        raise QueryAdapterLinearError(f"frozen generation plan mismatch: observed={observed} expected={expected}")
    return {"test_images": test_images, **observed}
