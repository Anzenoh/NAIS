"""Centered train-only orthogonal Procrustes alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .statistics import MetricError, paired_matrices


def _immutable(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return result.reshape(contiguous.shape)


@dataclass(frozen=True, slots=True, eq=False)
class ProcrustesMapping:
    rotation: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    training_sample_count: int
    source_name: str
    target_name: str
    singular_values: np.ndarray

    def __post_init__(self) -> None:
        rotation = _immutable(np.asarray(self.rotation))
        source_mean = _immutable(np.asarray(self.source_mean))
        target_mean = _immutable(np.asarray(self.target_mean))
        singular = _immutable(np.asarray(self.singular_values))
        if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
            raise MetricError("Procrustes rotation must be square")
        dimension = rotation.shape[0]
        if source_mean.shape != (dimension,) or target_mean.shape != (dimension,):
            raise MetricError("Procrustes means have inconsistent dimensions")
        if singular.shape != (dimension,) or self.training_sample_count < 2:
            raise MetricError("Procrustes fit metadata is inconsistent")
        if not self.source_name or not self.target_name:
            raise MetricError("Procrustes endpoint names must be non-empty")
        if not all(np.all(np.isfinite(item)) for item in (rotation, source_mean, target_mean, singular)):
            raise MetricError("Procrustes state contains NaN or Inf")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "source_mean", source_mean)
        object.__setattr__(self, "target_mean", target_mean)
        object.__setattr__(self, "singular_values", singular)

    @property
    def direction(self) -> str:
        return f"{self.source_name}-\u003e{self.target_name}"

    @property
    def source_dimension(self) -> int:
        return int(self.rotation.shape[0])

    @property
    def target_dimension(self) -> int:
        return int(self.rotation.shape[1])

    def transform(self, source: object) -> np.ndarray:
        value = np.asarray(source)
        if value.ndim != 2 or value.shape[1] != self.source_dimension:
            raise MetricError("Procrustes transform: source dimension mismatch")
        value = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise MetricError("Procrustes transform: non-finite source")
        result = (value - self.source_mean) @ self.rotation + self.target_mean
        if not np.all(np.isfinite(result)):
            raise MetricError("Procrustes transform: non-finite prediction")
        return result


def fit_orthogonal_procrustes(
    source_train: object,
    target_train: object,
    *,
    source_name: str = "source",
    target_name: str = "target",
) -> ProcrustesMapping:
    source, target = paired_matrices(
        source_train,
        target_train,
        first_name="source_train",
        second_name="target_train",
        same_width=True,
    )
    source_mean = np.mean(source, axis=0, dtype=np.float64)
    target_mean = np.mean(target, axis=0, dtype=np.float64)
    source_centered = source - source_mean
    target_centered = target - target_mean
    if np.linalg.norm(source_centered, "fro") == 0.0 or np.linalg.norm(target_centered, "fro") == 0.0:
        raise MetricError("Procrustes fit: zero centered variation")
    covariance = source_centered.T @ target_centered
    try:
        left, singular, right_transpose = np.linalg.svd(covariance, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise MetricError("Procrustes fit: SVD failed") from exc
    rotation = left @ right_transpose
    if not np.all(np.isfinite(rotation)):
        raise MetricError("Procrustes fit: non-finite rotation")
    return ProcrustesMapping(
        rotation=rotation,
        source_mean=source_mean,
        target_mean=target_mean,
        training_sample_count=source.shape[0],
        source_name=source_name,
        target_name=target_name,
        singular_values=singular,
    )


__all__ = ["ProcrustesMapping", "fit_orthogonal_procrustes"]
