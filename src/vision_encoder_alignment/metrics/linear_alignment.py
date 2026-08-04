"""Train-only ridge, reduced-rank maps, and held-out mapping metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .statistics import MetricError, bootstrap_indices, paired_matrices, validate_int, validate_real


def _immutable(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return result.reshape(contiguous.shape)


@dataclass(frozen=True, slots=True, eq=False)
class RidgeMapping:
    weights: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    relative_alpha: float
    absolute_alpha: float
    training_sample_count: int
    source_name: str
    target_name: str
    condition_number: float
    rank: int | None = None

    def __post_init__(self) -> None:
        weights = _immutable(np.asarray(self.weights))
        source_mean = _immutable(np.asarray(self.source_mean))
        target_mean = _immutable(np.asarray(self.target_mean))
        if weights.ndim != 2 or source_mean.shape != (weights.shape[0],) or target_mean.shape != (weights.shape[1],):
            raise MetricError("ridge mapping: inconsistent shapes")
        if self.training_sample_count < 2 or not self.source_name or not self.target_name:
            raise MetricError("ridge mapping: invalid fit metadata")
        for name, value in (
            ("relative_alpha", self.relative_alpha),
            ("absolute_alpha", self.absolute_alpha),
            ("condition_number", self.condition_number),
        ):
            validate_real(value, name=name, minimum=0.0)
        if self.rank is not None and (type(self.rank) is not int or not 1 <= self.rank <= min(weights.shape)):
            raise MetricError("ridge mapping: invalid reduced rank")
        if not all(np.all(np.isfinite(item)) for item in (weights, source_mean, target_mean)):
            raise MetricError("ridge mapping: non-finite state")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "source_mean", source_mean)
        object.__setattr__(self, "target_mean", target_mean)

    @property
    def direction(self) -> str:
        return f"{self.source_name}-\u003e{self.target_name}"

    @property
    def source_dimension(self) -> int:
        return int(self.weights.shape[0])

    @property
    def target_dimension(self) -> int:
        return int(self.weights.shape[1])

    def transform(self, source: object) -> np.ndarray:
        value = np.asarray(source)
        if value.ndim != 2 or value.shape[1] != self.source_dimension:
            raise MetricError("ridge transform: source dimension mismatch")
        value = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise MetricError("ridge transform: non-finite source")
        result = (value - self.source_mean) @ self.weights + self.target_mean
        if not np.all(np.isfinite(result)):
            raise MetricError("ridge transform: non-finite prediction")
        return result


def ridge_alpha_scale(source_centered: np.ndarray) -> float:
    """Return trace(X.T X)/D, the frozen relative-alpha scale."""

    trace = float(np.sum(source_centered * source_centered, dtype=np.float64))
    if trace <= 0.0 or not math.isfinite(trace):
        raise MetricError("ridge alpha scale: zero or non-finite source variation")
    return trace / source_centered.shape[1]


def fit_ridge_mapping(
    source_train: object,
    target_train: object,
    *,
    relative_alpha: float,
    source_name: str = "source",
    target_name: str = "target",
    condition_number_limit: float = 1.0e12,
) -> RidgeMapping:
    source, target = paired_matrices(
        source_train,
        target_train,
        first_name="source_train",
        second_name="target_train",
    )
    relative = validate_real(relative_alpha, name="relative_alpha", minimum=0.0)
    limit = validate_real(condition_number_limit, name="condition_number_limit", minimum=1.0)
    source_mean = np.mean(source, axis=0, dtype=np.float64)
    target_mean = np.mean(target, axis=0, dtype=np.float64)
    x = source - source_mean
    y = target - target_mean
    if np.linalg.norm(y, "fro") == 0.0:
        raise MetricError("ridge fit: zero target variation")
    scale = ridge_alpha_scale(x)
    absolute = relative * scale
    gram = x.T @ x
    system = gram + absolute * np.eye(x.shape[1], dtype=np.float64)
    try:
        singular = np.linalg.svd(system, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise MetricError("ridge fit: condition SVD failed") from exc
    condition = math.inf if singular[-1] == 0.0 else float(singular[0] / singular[-1])
    if not math.isfinite(condition) or condition > limit:
        raise MetricError(f"ridge fit: condition number {condition} exceeds {limit}")
    try:
        weights = np.linalg.solve(system, x.T @ y)
    except np.linalg.LinAlgError as exc:
        raise MetricError("ridge fit: solve failed") from exc
    if not np.all(np.isfinite(weights)):
        raise MetricError("ridge fit: non-finite weights")
    return RidgeMapping(
        weights=weights,
        source_mean=source_mean,
        target_mean=target_mean,
        relative_alpha=relative,
        absolute_alpha=absolute,
        training_sample_count=source.shape[0],
        source_name=source_name,
        target_name=target_name,
        condition_number=condition,
    )


def reduced_rank_mapping(mapping: RidgeMapping, rank: int) -> RidgeMapping:
    maximum = min(
        mapping.source_dimension,
        mapping.target_dimension,
        mapping.training_sample_count - 1,
    )
    if type(rank) is not int or rank < 1 or rank > maximum:
        raise MetricError(f"reduced rank must be in [1, {maximum}]")
    try:
        left, singular, right = np.linalg.svd(mapping.weights, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise MetricError("reduced-rank mapping: SVD failed") from exc
    weights = (left[:, :rank] * singular[:rank][None, :]) @ right[:rank]
    return RidgeMapping(
        weights=weights,
        source_mean=mapping.source_mean,
        target_mean=mapping.target_mean,
        relative_alpha=mapping.relative_alpha,
        absolute_alpha=mapping.absolute_alpha,
        training_sample_count=mapping.training_sample_count,
        source_name=mapping.source_name,
        target_name=mapping.target_name,
        condition_number=mapping.condition_number,
        rank=rank,
    )


def select_reduced_rank_mapping(
    mapping: RidgeMapping,
    source_validation: object,
    target_validation: object,
    *,
    ranks: Sequence[int],
    bootstrap_count: int,
    seed: int,
) -> tuple[RidgeMapping, list[dict[str, float | int | bool]]]:
    """Select the least-complex reduced rank within one bootstrap SE of best."""

    if not ranks:
        raise MetricError("reduced-rank selection: empty rank grid")
    maximum = min(mapping.source_dimension, mapping.target_dimension, mapping.training_sample_count - 1)
    canonical = tuple(int(rank) for rank in ranks)
    if any(type(rank) is not int or rank < 1 or rank > maximum for rank in canonical):
        raise MetricError(f"reduced-rank selection: rank must be in [1, {maximum}]")
    if len(set(canonical)) != len(canonical) or tuple(sorted(canonical)) != canonical:
        raise MetricError("reduced-rank selection: ranks must be unique increasing integers")
    source, target = paired_matrices(
        source_validation,
        target_validation,
        first_name="source_validation",
        second_name="target_validation",
    )
    if source.shape[1] != mapping.source_dimension or target.shape[1] != mapping.target_dimension:
        raise MetricError("reduced-rank selection: validation width mismatch")
    try:
        left, singular, right = np.linalg.svd(mapping.weights, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise MetricError("reduced-rank selection: SVD failed") from exc
    candidates: list[tuple[RidgeMapping, np.ndarray]] = []
    for rank in canonical:
        weights = (left[:, :rank] * singular[:rank][None, :]) @ right[:rank]
        candidate = RidgeMapping(
            weights=weights,
            source_mean=mapping.source_mean,
            target_mean=mapping.target_mean,
            relative_alpha=mapping.relative_alpha,
            absolute_alpha=mapping.absolute_alpha,
            training_sample_count=mapping.training_sample_count,
            source_name=mapping.source_name,
            target_name=mapping.target_name,
            condition_number=mapping.condition_number,
            rank=rank,
        )
        candidates.append((candidate, per_image_squared_error(target, candidate.transform(source))))
    means = np.asarray([np.mean(loss, dtype=np.float64) for _, loss in candidates])
    minimum = int(np.argmin(means))
    count = validate_int(bootstrap_count, name="bootstrap_count", minimum=2)
    generator = np.random.Generator(np.random.PCG64(validate_int(seed, name="seed")))
    best = candidates[minimum][1]
    boot = np.empty(count, dtype=np.float64)
    for index in range(count):
        sample = generator.integers(0, best.size, size=best.size)
        boot[index] = float(np.mean(best[sample], dtype=np.float64))
    threshold = float(means[minimum] + np.std(boot, ddof=1))
    eligible = [index for index, value in enumerate(means) if value <= threshold]
    selected = min(eligible, key=lambda index: (candidates[index][0].rank, index))
    table = [
        {
            "rank": int(candidate.rank),
            "validation_mean_mse": float(means[index]),
            "eligible_one_standard_error": index in eligible,
            "selected": index == selected,
        }
        for index, (candidate, _) in enumerate(candidates)
    ]
    return candidates[selected][0], table


def per_image_squared_error(target: object, prediction: object) -> np.ndarray:
    y, estimate = paired_matrices(
        target,
        prediction,
        first_name="target",
        second_name="prediction",
        same_width=True,
    )
    return np.mean((y - estimate) ** 2, axis=1, dtype=np.float64)


def alignment_metrics(
    target: object,
    prediction: object,
    *,
    target_train_mean: object,
) -> dict[str, float | int]:
    y, estimate = paired_matrices(
        target,
        prediction,
        first_name="target",
        second_name="prediction",
        same_width=True,
    )
    mean = np.asarray(target_train_mean, dtype=np.float64)
    if mean.shape != (y.shape[1],) or not np.all(np.isfinite(mean)):
        raise MetricError("target_train_mean: invalid shape or value")
    residual = y - estimate
    sse_by_dimension = np.sum(residual * residual, axis=0, dtype=np.float64)
    baseline = y - mean
    sst_by_dimension = np.sum(baseline * baseline, axis=0, dtype=np.float64)
    if np.any(sst_by_dimension == 0.0):
        raise MetricError("mapping metrics: zero target denominator dimension")
    macro_r2 = float(np.mean(1.0 - sse_by_dimension / sst_by_dimension, dtype=np.float64))
    nmse = float(np.sum(sse_by_dimension, dtype=np.float64) / np.sum(sst_by_dimension, dtype=np.float64))
    ynorm = np.linalg.norm(y, axis=1)
    pnorm = np.linalg.norm(estimate, axis=1)
    if np.any(ynorm == 0.0) or np.any(pnorm == 0.0):
        raise MetricError("mapping metrics: zero-norm row")
    cosines = np.sum(y * estimate, axis=1, dtype=np.float64) / (ynorm * pnorm)
    return {
        "sample_count": int(y.shape[0]),
        "target_dimension": int(y.shape[1]),
        "macro_r2_train_target_mean": macro_r2,
        "nmse_train_target_mean": nmse,
        "mean_cosine": float(np.mean(cosines, dtype=np.float64)),
        "median_cosine": float(np.median(cosines)),
        "mean_per_image_mse": float(np.mean(per_image_squared_error(y, estimate), dtype=np.float64)),
    }


def select_ridge_mapping(
    source_train: object,
    target_train: object,
    source_validation: object,
    target_validation: object,
    *,
    relative_alphas: Sequence[float],
    bootstrap_count: int,
    seed: int,
    source_name: str,
    target_name: str,
    condition_number_limit: float = 1.0e12,
) -> tuple[RidgeMapping, list[dict[str, float | bool]]]:
    if not relative_alphas:
        raise MetricError("ridge selection: empty alpha grid")
    candidates: list[tuple[RidgeMapping, np.ndarray]] = []
    for alpha in relative_alphas:
        mapping = fit_ridge_mapping(
            source_train,
            target_train,
            relative_alpha=float(alpha),
            source_name=source_name,
            target_name=target_name,
            condition_number_limit=condition_number_limit,
        )
        losses = per_image_squared_error(target_validation, mapping.transform(source_validation))
        candidates.append((mapping, losses))
    means = np.array([np.mean(loss, dtype=np.float64) for _, loss in candidates])
    minimum = int(np.argmin(means))
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    boot = np.empty(bootstrap_count, dtype=np.float64)
    minimum_losses = candidates[minimum][1]
    for index in range(bootstrap_count):
        sample = generator.integers(0, minimum_losses.size, size=minimum_losses.size)
        boot[index] = float(np.mean(minimum_losses[sample], dtype=np.float64))
    standard_error = float(np.std(boot, ddof=1))
    threshold = means[minimum] + standard_error
    eligible = [index for index, value in enumerate(means) if value <= threshold]
    # Higher alpha is less complex; ties retain the earlier canonical grid position.
    selected = sorted(eligible, key=lambda index: (-candidates[index][0].relative_alpha, index))[0]
    table = [
        {
            "relative_alpha": item.relative_alpha,
            "absolute_alpha": item.absolute_alpha,
            "validation_mean_mse": float(means[index]),
            "eligible_one_standard_error": index in eligible,
            "selected": index == selected,
        }
        for index, (item, _) in enumerate(candidates)
    ]
    return candidates[selected][0], table


def bootstrap_mapping_metric(
    target: object,
    prediction: object,
    *,
    target_train_mean: object,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    y, estimate = paired_matrices(target, prediction, same_width=True)
    names = ("macro_r2_train_target_mean", "nmse_train_target_mean", "mean_cosine")
    values = {name: np.empty(replicates, dtype=np.float64) for name in names}
    for index, sample in enumerate(bootstrap_indices(y.shape[0], replicates, seed=seed)):
        result = alignment_metrics(y[sample], estimate[sample], target_train_mean=target_train_mean)
        for name in names:
            values[name][index] = float(result[name])
    intervals: dict[str, tuple[float, float]] = {}
    for name in names:
        low, high = np.quantile(values[name], [0.025, 0.975], method="linear")
        intervals[name] = (float(low), float(high))
    return intervals


__all__ = [
    "RidgeMapping", "alignment_metrics", "bootstrap_mapping_metric",
    "fit_ridge_mapping", "per_image_squared_error", "reduced_rank_mapping",
    "ridge_alpha_scale", "select_reduced_rank_mapping", "select_ridge_mapping",
]
