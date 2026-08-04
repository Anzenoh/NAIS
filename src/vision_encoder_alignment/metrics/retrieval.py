"""Deterministic global and within-image patch retrieval metrics."""

from __future__ import annotations

import numpy as np

from .statistics import MetricError, paired_matrices


def _unit_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        raise MetricError(f"{name}: zero or non-finite row norm")
    return value / norms[:, None]


def cosine_retrieval(
    query: object,
    target: object,
    *,
    recall_at: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | int]:
    source, destination = paired_matrices(
        query,
        target,
        first_name="query",
        second_name="target",
        same_width=True,
    )
    if any(type(k) is not int or k < 1 or k > source.shape[0] for k in recall_at):
        raise MetricError("retrieval: invalid k")
    similarity = _unit_rows(source, name="query") @ _unit_rows(destination, name="target").T
    order = np.argsort(-similarity, axis=1, kind="mergesort")
    truth = np.arange(source.shape[0])[:, None]
    result: dict[str, float | int] = {"sample_count": int(source.shape[0])}
    for k in recall_at:
        result[f"recall_at_{k}"] = float(np.mean(np.any(order[:, :k] == truth, axis=1)))
    ranks = np.argmax(order == truth, axis=1) + 1
    result["mean_reciprocal_rank"] = float(np.mean(1.0 / ranks, dtype=np.float64))
    return result


def patch_coordinate_retrieval(
    source: object,
    target: object,
    *,
    grid: tuple[int, int] = (14, 14),
) -> dict[str, float | int]:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 3 or y.ndim != 3 or x.shape != y.shape:
        raise MetricError("patch retrieval: expected paired [images, patches, features]")
    image_count, patch_count, _ = x.shape
    if grid[0] * grid[1] != patch_count or image_count < 1:
        raise MetricError("patch retrieval: grid mismatch")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise MetricError("patch retrieval: NaN or Inf")
    rows, columns = np.unravel_index(np.arange(patch_count), grid)
    exact = 0
    within_one = 0
    top_five = 0
    reciprocal_rank = 0.0
    manhattan_values = np.empty(image_count * patch_count, dtype=np.float64)
    euclidean_values = np.empty(image_count * patch_count, dtype=np.float64)
    cursor = 0
    for image_index in range(image_count):
        source_unit = _unit_rows(x[image_index], name="source patch")
        target_unit = _unit_rows(y[image_index], name="target patch")
        similarity = source_unit @ target_unit.T
        order = np.argsort(-similarity, axis=1, kind="mergesort")
        predicted = order[:, 0]
        truth = np.arange(patch_count)
        distance_row = np.abs(rows[predicted] - rows[truth])
        distance_column = np.abs(columns[predicted] - columns[truth])
        manhattan = distance_row + distance_column
        euclidean = np.sqrt(distance_row * distance_row + distance_column * distance_column)
        exact += int(np.count_nonzero(predicted == truth))
        within_one += int(np.count_nonzero(manhattan <= 1))
        top_five += int(np.count_nonzero(np.any(order[:, :5] == truth[:, None], axis=1)))
        ranks = np.argmax(order == truth[:, None], axis=1) + 1
        reciprocal_rank += float(np.sum(1.0 / ranks, dtype=np.float64))
        stop = cursor + patch_count
        manhattan_values[cursor:stop] = manhattan
        euclidean_values[cursor:stop] = euclidean
        cursor = stop
    total = image_count * patch_count
    return {
        "image_count": int(image_count),
        "patch_count_per_image": int(patch_count),
        "query_count": int(total),
        "exact_coordinate_top1": exact / total,
        "manhattan_at_most_1": within_one / total,
        "top5_coordinate_recall": top_five / total,
        "mean_reciprocal_rank": reciprocal_rank / total,
        "mean_manhattan_error": float(np.mean(manhattan_values, dtype=np.float64)),
        "median_manhattan_error": float(np.median(manhattan_values)),
        "mean_euclidean_error": float(np.mean(euclidean_values, dtype=np.float64)),
        "median_euclidean_error": float(np.median(euclidean_values)),
    }


__all__ = ["cosine_retrieval", "patch_coordinate_retrieval"]
