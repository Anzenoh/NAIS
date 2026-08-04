"""Deterministic validation, ranking, resampling, and inference utilities."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np


class MetricError(ValueError):
    """Raised when a metric is undefined or an input violates its contract."""


def as_float_matrix(value: Any, *, name: str, minimum_rows: int = 2) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise MetricError(f"{name}: expected a real rank-2 array") from exc
    if array.ndim != 2:
        raise MetricError(f"{name}: expected rank 2, got {array.ndim}")
    if array.shape[0] < minimum_rows or array.shape[1] < 1:
        raise MetricError(f"{name}: empty or insufficient shape {array.shape}")
    if array.dtype.kind not in "iuf":
        raise MetricError(f"{name}: expected a real numeric dtype")
    result = np.asarray(array, dtype=np.float64, order="C")
    if not np.all(np.isfinite(result)):
        raise MetricError(f"{name}: contains NaN or Inf")
    return result


def paired_matrices(
    first: Any,
    second: Any,
    *,
    first_name: str = "first",
    second_name: str = "second",
    same_width: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x = as_float_matrix(first, name=first_name)
    y = as_float_matrix(second, name=second_name)
    if x.shape[0] != y.shape[0]:
        raise MetricError(f"{first_name}, {second_name}: row counts differ")
    if same_width and x.shape[1] != y.shape[1]:
        raise MetricError(f"{first_name}, {second_name}: widths differ")
    return x, y


def validate_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise MetricError(f"{name}: expected an integer")
    result = int(value)
    if result < minimum:
        raise MetricError(f"{name}: expected >= {minimum}")
    return result


def validate_real(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise MetricError(f"{name}: expected a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise MetricError(f"{name}: invalid finite scalar")
    return result


def deterministic_seed(master_seed: int, label: str) -> int:
    seed = validate_int(master_seed, name="master_seed")
    if not isinstance(label, str) or not label:
        raise MetricError("label: expected a non-empty string")
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()[:8],
        "big",
    )


def rng_for(master_seed: int, label: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(deterministic_seed(master_seed, label)))


def finite_sum_squares(value: np.ndarray, *, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.sum(np.asarray(value, dtype=np.float64) ** 2, dtype=np.float64))
    if not math.isfinite(result):
        raise MetricError(f"{name}: non-finite sum of squares")
    return result


def rankdata_average(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise MetricError("rankdata: expected one finite vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman_correlation(first: object, second: object) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise MetricError("spearman: expected paired vectors")
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = math.sqrt(finite_sum_squares(rx, name="rank x") * finite_sum_squares(ry, name="rank y"))
    if denominator == 0.0:
        raise MetricError("spearman: constant ranks")
    return float(np.dot(rx, ry) / denominator)


def empirical_pvalue(observed: float, null: object, *, alternative: str) -> float:
    score = validate_real(observed, name="observed")
    values = np.asarray(null, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise MetricError("null: expected a non-empty finite vector")
    if alternative == "greater":
        extreme = int(np.count_nonzero(values >= score))
    elif alternative == "less":
        extreme = int(np.count_nonzero(values <= score))
    elif alternative == "two-sided":
        center = float(np.mean(values, dtype=np.float64))
        extreme = int(np.count_nonzero(np.abs(values - center) >= abs(score - center)))
    else:
        raise MetricError("alternative: expected greater, less, or two-sided")
    return float((extreme + 1) / (values.size + 1))


def percentile_interval(
    values: object,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    level = validate_real(confidence, name="confidence", minimum=0.0)
    if not 0.0 < level < 1.0 or array.ndim != 1 or array.size < 2:
        raise MetricError("percentile interval: invalid input")
    if not np.all(np.isfinite(array)):
        raise MetricError("percentile interval: non-finite input")
    alpha = (1.0 - level) / 2.0
    try:
        low, high = np.quantile(array, [alpha, 1.0 - alpha], method="linear")
    except TypeError:  # NumPy before the method keyword.
        low, high = np.quantile(array, [alpha, 1.0 - alpha], interpolation="linear")
    return float(low), float(high)


def bootstrap_indices(
    image_count: int,
    replicates: int,
    *,
    seed: int,
) -> Iterable[np.ndarray]:
    count = validate_int(image_count, name="image_count", minimum=2)
    reps = validate_int(replicates, name="replicates", minimum=1)
    generator = np.random.Generator(np.random.PCG64(validate_int(seed, name="seed")))
    for _ in range(reps):
        yield generator.integers(0, count, size=count, dtype=np.int64)


def bootstrap_scalar(
    image_count: int,
    replicates: int,
    *,
    seed: int,
    statistic: Callable[[np.ndarray], float],
) -> np.ndarray:
    values = np.empty(validate_int(replicates, name="replicates", minimum=1), dtype=np.float64)
    for index, sample in enumerate(bootstrap_indices(image_count, replicates, seed=seed)):
        value = float(statistic(sample))
        if not math.isfinite(value):
            raise MetricError(f"bootstrap replicate {index}: non-finite")
        values[index] = value
    return values


def holm_bonferroni(pvalues: Sequence[float], *, alpha: float = 0.05) -> tuple[bool, ...]:
    threshold = validate_real(alpha, name="alpha", minimum=0.0)
    if not 0.0 < threshold < 1.0 or not pvalues:
        raise MetricError("holm: invalid alpha or empty family")
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0.0) | (values > 1.0)):
        raise MetricError("holm: invalid p-value")
    order = np.argsort(values, kind="mergesort")
    rejected = np.zeros(values.size, dtype=bool)
    active = True
    for rank, index in enumerate(order):
        if active and values[index] <= threshold / (values.size - rank):
            rejected[index] = True
        else:
            active = False
    return tuple(bool(item) for item in rejected)


def one_standard_error_choice(
    means: Sequence[float],
    standard_errors: Sequence[float],
    complexities: Sequence[float],
) -> int:
    if not (len(means) == len(standard_errors) == len(complexities)) or not means:
        raise MetricError("one-standard-error: inconsistent candidates")
    loss = np.asarray(means, dtype=np.float64)
    errors = np.asarray(standard_errors, dtype=np.float64)
    complexity = np.asarray(complexities, dtype=np.float64)
    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise MetricError("one-standard-error: invalid values")
    best = int(np.argmin(loss))
    eligible = np.flatnonzero(loss <= loss[best] + errors[best])
    if eligible.size == 0:
        raise MetricError("one-standard-error: no eligible candidate")
    order = sorted((float(complexity[index]), int(index)) for index in eligible)
    return order[0][1]


def validate_image_ids(value: object, *, name: str = "image_ids") -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise MetricError(f"{name}: expected an ordered sequence")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise MetricError(f"{name}: invalid image ID")
    if len(set(result)) != len(result):
        raise MetricError(f"{name}: duplicate image ID")
    return result


__all__ = [
    "MetricError", "as_float_matrix", "bootstrap_indices", "bootstrap_scalar",
    "deterministic_seed", "empirical_pvalue", "finite_sum_squares",
    "holm_bonferroni", "one_standard_error_choice", "paired_matrices",
    "percentile_interval", "rankdata_average", "rng_for", "spearman_correlation",
    "validate_image_ids", "validate_int", "validate_real",
]
