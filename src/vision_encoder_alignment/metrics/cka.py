"""Linear CKA, unbiased sensitivity CKA, RSA, and calibrated controls."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .statistics import (
    MetricError,
    as_float_matrix,
    bootstrap_indices,
    empirical_pvalue,
    paired_matrices,
    percentile_interval,
    spearman_correlation,
    validate_int,
)


def _center(value: np.ndarray) -> np.ndarray:
    return value - np.mean(value, axis=0, dtype=np.float64, keepdims=True)


def linear_cka(source: object, target: object) -> float:
    """Column-centered feature-space linear CKA with FP64 accumulation."""

    x, y = paired_matrices(source, target, first_name="source", second_name="target")
    xc = _center(x)
    yc = _center(y)
    with np.errstate(over="ignore", invalid="ignore"):
        cross = xc.T @ yc
        xx = xc.T @ xc
        yy = yc.T @ yc
        numerator = float(np.sum(cross * cross, dtype=np.float64))
        denominator = float(np.linalg.norm(xx, "fro") * np.linalg.norm(yy, "fro"))
    if denominator <= 0.0 or not math.isfinite(numerator) or not math.isfinite(denominator):
        raise MetricError("linear CKA: zero or non-finite denominator")
    value = numerator / denominator
    if not math.isfinite(value):
        raise MetricError("linear CKA: non-finite result")
    return float(value)


def linear_cka_from_moments(
    cross_product: object,
    source_product: object,
    target_product: object,
) -> float:
    cross = as_float_matrix(cross_product, name="cross_product", minimum_rows=1)
    xx = as_float_matrix(source_product, name="source_product", minimum_rows=1)
    yy = as_float_matrix(target_product, name="target_product", minimum_rows=1)
    if xx.shape[0] != xx.shape[1] or yy.shape[0] != yy.shape[1]:
        raise MetricError("linear CKA moments: covariance matrices must be square")
    if cross.shape != (xx.shape[0], yy.shape[0]):
        raise MetricError("linear CKA moments: inconsistent dimensions")
    numerator = float(np.sum(cross * cross, dtype=np.float64))
    denominator = float(np.linalg.norm(xx, "fro") * np.linalg.norm(yy, "fro"))
    if denominator <= 0.0 or not math.isfinite(numerator / denominator):
        raise MetricError("linear CKA moments: undefined")
    return float(numerator / denominator)


def _u_center(gram: np.ndarray) -> np.ndarray:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] < 4:
        raise MetricError("U-centering requires a square Gram matrix with n >= 4")
    value = np.array(gram, dtype=np.float64, copy=True, order="C")
    np.fill_diagonal(value, 0.0)
    n = value.shape[0]
    row_sum = np.sum(value, axis=1, dtype=np.float64)
    total = float(np.sum(row_sum, dtype=np.float64))
    centered = (
        value
        - row_sum[:, None] / (n - 2)
        - row_sum[None, :] / (n - 2)
        + total / ((n - 1) * (n - 2))
    )
    np.fill_diagonal(centered, 0.0)
    return centered


def ucentered_cka(source: object, target: object) -> float:
    """Unbiased-HSIC sensitivity CKA using U-centered linear kernels."""

    x, y = paired_matrices(source, target, first_name="source", second_name="target")
    if x.shape[0] < 4:
        raise MetricError("U-centered CKA requires at least four samples")
    k = _u_center(x @ x.T)
    l = _u_center(y @ y.T)
    scale = x.shape[0] * (x.shape[0] - 3)
    xy = float(np.sum(k * l, dtype=np.float64) / scale)
    xx = float(np.sum(k * k, dtype=np.float64) / scale)
    yy = float(np.sum(l * l, dtype=np.float64) / scale)
    if xx <= 0.0 or yy <= 0.0:
        raise MetricError("U-centered CKA: non-positive self-HSIC")
    result = xy / math.sqrt(xx * yy)
    if not math.isfinite(result):
        raise MetricError("U-centered CKA: non-finite result")
    return float(result)


def rsa_cosine_spearman(source: object, target: object) -> float:
    """Tie-aware Spearman RSA over upper-triangle cosine-distance RDMs."""

    x, y = paired_matrices(source, target, first_name="source", second_name="target")
    xnorm = np.linalg.norm(x, axis=1)
    ynorm = np.linalg.norm(y, axis=1)
    if np.any(xnorm == 0.0) or np.any(ynorm == 0.0):
        raise MetricError("RSA: zero-norm row")
    xunit = x / xnorm[:, None]
    yunit = y / ynorm[:, None]
    upper = np.triu_indices(x.shape[0], k=1)
    xrdm = 1.0 - (xunit @ xunit.T)[upper]
    yrdm = 1.0 - (yunit @ yunit.T)[upper]
    return spearman_correlation(xrdm, yrdm)


def spectrum_matched_row_null(value: object, *, seed: int) -> np.ndarray:
    """Replace centered left singular vectors by an independent seeded Haar basis."""

    x = as_float_matrix(value, name="value")
    mean = np.mean(x, axis=0, dtype=np.float64, keepdims=True)
    centered = x - mean
    try:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise MetricError("spectrum null: SVD failed") from exc
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * (singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tolerance))
    if rank < 1:
        raise MetricError("spectrum null: zero rank")
    generator = np.random.Generator(np.random.PCG64(validate_int(seed, name="seed")))
    random = generator.standard_normal((x.shape[0], rank))
    random -= np.mean(random, axis=0, dtype=np.float64, keepdims=True)
    try:
        basis, triangular = np.linalg.qr(random, mode="reduced")
    except np.linalg.LinAlgError as exc:
        raise MetricError("spectrum null: QR failed") from exc
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    basis *= signs[None, :]
    result = (basis * singular[:rank][None, :]) @ vt[:rank] + mean
    if not np.all(np.isfinite(result)):
        raise MetricError("spectrum null: non-finite result")
    return result


def _center_gram(gram: np.ndarray) -> np.ndarray:
    row = np.mean(gram, axis=1, dtype=np.float64, keepdims=True)
    column = np.mean(gram, axis=0, dtype=np.float64, keepdims=True)
    return gram - row - column + float(np.mean(gram, dtype=np.float64))


def linear_cka_from_grams(source_gram: object, target_gram: object) -> float:
    k, l = paired_matrices(
        source_gram,
        target_gram,
        first_name="source_gram",
        second_name="target_gram",
        same_width=True,
    )
    if k.shape[0] != k.shape[1]:
        raise MetricError("linear CKA Gram: expected square matrices")
    kc = _center_gram(k)
    lc = _center_gram(l)
    numerator = float(np.sum(kc * lc, dtype=np.float64))
    denominator = math.sqrt(
        float(np.sum(kc * kc, dtype=np.float64))
        * float(np.sum(lc * lc, dtype=np.float64))
    )
    if denominator <= 0.0:
        raise MetricError("linear CKA Gram: zero denominator")
    return float(numerator / denominator)


def permutation_calibration(
    source: object,
    target: object,
    *,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    """Calibrate paired CKA against deterministic image-label permutations."""

    x, y = paired_matrices(source, target, first_name="source", second_name="target")
    count = validate_int(permutations, name="permutations", minimum=1)
    observed = linear_cka(x, y)
    k = x @ x.T
    l = y @ y.T
    generator = np.random.Generator(np.random.PCG64(validate_int(seed, name="seed")))
    null = np.empty(count, dtype=np.float64)
    for index in range(count):
        order = generator.permutation(x.shape[0])
        null[index] = linear_cka_from_grams(k, l[np.ix_(order, order)])
    mean = float(np.mean(null, dtype=np.float64))
    standard_deviation = float(np.std(null, ddof=1)) if count > 1 else 0.0
    zscore = None if standard_deviation == 0.0 else float((observed - mean) / standard_deviation)
    return {
        "observed": observed,
        "null_mean": mean,
        "null_standard_deviation": standard_deviation,
        "null_centered_effect": float(observed - mean),
        "null_zscore": zscore,
        "empirical_pvalue_greater": empirical_pvalue(observed, null, alternative="greater"),
        "permutation_count": count,
    }


def cka_image_bootstrap(
    source: object,
    target: object,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    x, y = paired_matrices(source, target, first_name="source", second_name="target")
    count = validate_int(replicates, name="replicates", minimum=2)
    k = x @ x.T
    l = y @ y.T
    values = np.empty(count, dtype=np.float64)
    for index, sample in enumerate(bootstrap_indices(x.shape[0], count, seed=seed)):
        values[index] = linear_cka_from_grams(k[np.ix_(sample, sample)], l[np.ix_(sample, sample)])
    low, high = percentile_interval(values, confidence=confidence)
    return {
        "estimate": linear_cka(x, y),
        "confidence_level": float(confidence),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_count": count,
        "unit": "image_block",
    }


__all__ = [
    "cka_image_bootstrap", "linear_cka", "linear_cka_from_grams",
    "linear_cka_from_moments", "permutation_calibration", "rsa_cosine_spearman",
    "spectrum_matched_row_null", "ucentered_cka",
]
