"""Streaming FP64 sufficient statistics for large feature matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_encoder_alignment.metrics.cka import linear_cka_from_moments
from vision_encoder_alignment.metrics.statistics import MetricError, as_float_matrix


@dataclass(slots=True)
class PairSufficientStatistics:
    """Mergeable raw sums that yield exactly centered CKA moments."""

    source_width: int
    target_width: int
    count: int = 0
    source_sum: np.ndarray | None = None
    target_sum: np.ndarray | None = None
    source_product: np.ndarray | None = None
    target_product: np.ndarray | None = None
    cross_product: np.ndarray | None = None

    def __post_init__(self) -> None:
        if type(self.source_width) is not int or self.source_width < 1:
            raise MetricError("source_width: expected positive integer")
        if type(self.target_width) is not int or self.target_width < 1:
            raise MetricError("target_width: expected positive integer")
        self.source_sum = np.zeros(self.source_width, dtype=np.float64)
        self.target_sum = np.zeros(self.target_width, dtype=np.float64)
        self.source_product = np.zeros((self.source_width, self.source_width), dtype=np.float64)
        self.target_product = np.zeros((self.target_width, self.target_width), dtype=np.float64)
        self.cross_product = np.zeros((self.source_width, self.target_width), dtype=np.float64)

    def update(self, source: object, target: object) -> None:
        x = as_float_matrix(source, name="source block", minimum_rows=1)
        y = as_float_matrix(target, name="target block", minimum_rows=1)
        if x.shape[0] != y.shape[0] or x.shape[1] != self.source_width or y.shape[1] != self.target_width:
            raise MetricError("sufficient-statistics block shape mismatch")
        self.count += x.shape[0]
        self.source_sum += np.sum(x, axis=0, dtype=np.float64)
        self.target_sum += np.sum(y, axis=0, dtype=np.float64)
        self.source_product += x.T @ x
        self.target_product += y.T @ y
        self.cross_product += x.T @ y

    def merge(self, other: "PairSufficientStatistics") -> None:
        if self.source_width != other.source_width or self.target_width != other.target_width:
            raise MetricError("sufficient-statistics merge shape mismatch")
        self.count += other.count
        self.source_sum += other.source_sum
        self.target_sum += other.target_sum
        self.source_product += other.source_product
        self.target_product += other.target_product
        self.cross_product += other.cross_product

    def centered_moments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.count < 2:
            raise MetricError("sufficient statistics require at least two rows")
        n = float(self.count)
        xx = self.source_product - np.outer(self.source_sum, self.source_sum) / n
        yy = self.target_product - np.outer(self.target_sum, self.target_sum) / n
        xy = self.cross_product - np.outer(self.source_sum, self.target_sum) / n
        if not all(np.all(np.isfinite(item)) for item in (xx, yy, xy)):
            raise MetricError("sufficient statistics produced non-finite moments")
        return xy, xx, yy

    def linear_cka(self) -> float:
        xy, xx, yy = self.centered_moments()
        return linear_cka_from_moments(xy, xx, yy)


@dataclass(slots=True)
class FeatureSufficientStatistics:
    width: int
    count: int = 0
    feature_sum: np.ndarray | None = None
    feature_product: np.ndarray | None = None
    squared_norm_sum: float = 0.0
    norm_sum: float = 0.0

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 1:
            raise MetricError("width: expected positive integer")
        self.feature_sum = np.zeros(self.width, dtype=np.float64)
        self.feature_product = np.zeros((self.width, self.width), dtype=np.float64)

    def update(self, value: object) -> None:
        x = as_float_matrix(value, name="feature block", minimum_rows=1)
        if x.shape[1] != self.width:
            raise MetricError("feature-statistics block width mismatch")
        norms = np.linalg.norm(x, axis=1)
        if not np.all(np.isfinite(norms)):
            raise MetricError("feature-statistics non-finite norm")
        self.count += x.shape[0]
        self.feature_sum += np.sum(x, axis=0, dtype=np.float64)
        self.feature_product += x.T @ x
        self.squared_norm_sum += float(np.sum(norms * norms, dtype=np.float64))
        self.norm_sum += float(np.sum(norms, dtype=np.float64))

    def mean(self) -> np.ndarray:
        if self.count < 1:
            raise MetricError("empty feature statistics")
        return self.feature_sum / self.count

    def centered_product(self) -> np.ndarray:
        if self.count < 2:
            raise MetricError("feature statistics require at least two rows")
        result = self.feature_product - np.outer(self.feature_sum, self.feature_sum) / self.count
        if not np.all(np.isfinite(result)):
            raise MetricError("feature statistics produced non-finite covariance")
        return result

    def covariance(self) -> np.ndarray:
        return self.centered_product() / (self.count - 1)


__all__ = ["FeatureSufficientStatistics", "PairSufficientStatistics"]
