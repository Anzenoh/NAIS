"""Spectrum, anisotropy, norm, and neighborhood geometry diagnostics."""

from __future__ import annotations

import math

import numpy as np

from .statistics import MetricError, as_float_matrix, paired_matrices


def covariance_geometry(value: object) -> dict[str, object]:
    x = as_float_matrix(value, name="value")
    centered = x - np.mean(x, axis=0, dtype=np.float64, keepdims=True)
    try:
        singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise MetricError("covariance geometry: SVD failed") from exc
    eigenvalues = singular * singular / (x.shape[0] - 1)
    total = float(np.sum(eigenvalues, dtype=np.float64))
    if total <= 0.0 or not math.isfinite(total):
        raise MetricError("covariance geometry: zero total variance")
    probabilities = eigenvalues / total
    positive = probabilities[probabilities > 0.0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive), dtype=np.float64)))
    participation = float(total * total / np.sum(eigenvalues * eigenvalues, dtype=np.float64))
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise MetricError("covariance geometry: zero or non-finite row norm")
    unit = x / norms[:, None]
    sum_unit = np.sum(unit, axis=0, dtype=np.float64)
    anisotropy = float((np.dot(sum_unit, sum_unit) - x.shape[0]) / (x.shape[0] * (x.shape[0] - 1)))
    return {
        "sample_count": int(x.shape[0]),
        "feature_dimension": int(x.shape[1]),
        "covariance_eigenvalues": [float(item) for item in eigenvalues],
        "effective_rank": effective_rank,
        "participation_ratio": participation,
        "anisotropy_mean_pairwise_cosine": anisotropy,
        "norm_mean": float(np.mean(norms, dtype=np.float64)),
        "norm_median": float(np.median(norms)),
        "norm_standard_deviation": float(np.std(norms, ddof=1)),
        "norm_minimum": float(np.min(norms)),
        "norm_maximum": float(np.max(norms)),
    }


def knn_overlap(
    source: object,
    target: object,
    *,
    k_values: tuple[int, ...] = (5, 10, 20, 50),
) -> dict[str, float | int]:
    x, y = paired_matrices(source, target, first_name="source", second_name="target")
    n = x.shape[0]
    if any(type(k) is not int or k < 1 or k >= n for k in k_values):
        raise MetricError("kNN overlap: invalid k")
    xnorm = np.linalg.norm(x, axis=1)
    ynorm = np.linalg.norm(y, axis=1)
    if np.any(xnorm == 0.0) or np.any(ynorm == 0.0):
        raise MetricError("kNN overlap: zero-norm row")
    xsim = (x / xnorm[:, None]) @ (x / xnorm[:, None]).T
    ysim = (y / ynorm[:, None]) @ (y / ynorm[:, None]).T
    np.fill_diagonal(xsim, -np.inf)
    np.fill_diagonal(ysim, -np.inf)
    xorder = np.argsort(-xsim, axis=1, kind="mergesort")
    yorder = np.argsort(-ysim, axis=1, kind="mergesort")
    result: dict[str, float | int] = {"sample_count": int(n)}
    for k in k_values:
        overlap = 0.0
        for index in range(n):
            overlap += len(set(xorder[index, :k]) & set(yorder[index, :k])) / k
        result[f"overlap_at_{k}"] = overlap / n
    return result


__all__ = ["covariance_geometry", "knn_overlap"]
