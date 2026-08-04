"""Patch-state transforms, streaming CKA, spatial controls, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from vision_encoder_alignment.analysis.sufficient_statistics import PairSufficientStatistics

from .statistics import MetricError


def _patch_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 2 or array.shape[2] < 1:
        raise MetricError(f"{name}: expected [images, patches, features]")
    if array.dtype.kind not in "iuf":
        raise MetricError(f"{name}: expected a real numeric dtype")
    result = np.asarray(array, dtype=np.float64, order="C")
    if not np.all(np.isfinite(result)):
        raise MetricError(f"{name}: contains NaN or Inf")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class PositionTemplate:
    patch_mean: np.ndarray
    global_mean: np.ndarray
    training_image_count: int

    def __post_init__(self) -> None:
        patch = np.asarray(self.patch_mean, dtype=np.float64)
        global_mean = np.asarray(self.global_mean, dtype=np.float64)
        if patch.ndim != 2 or global_mean.shape != (patch.shape[1],):
            raise MetricError("position template: inconsistent shapes")
        if self.training_image_count < 1 or not np.all(np.isfinite(patch)) or not np.all(np.isfinite(global_mean)):
            raise MetricError("position template: invalid state")
        patch_copy = np.frombuffer(np.ascontiguousarray(patch).tobytes(), dtype=np.float64).reshape(patch.shape)
        global_copy = np.frombuffer(np.ascontiguousarray(global_mean).tobytes(), dtype=np.float64)
        object.__setattr__(self, "patch_mean", patch_copy)
        object.__setattr__(self, "global_mean", global_copy)


def fit_position_template(train_patch: object) -> PositionTemplate:
    x = _patch_array(train_patch, name="train_patch")
    patch_mean = np.mean(x, axis=0, dtype=np.float64)
    global_mean = np.mean(x, axis=(0, 1), dtype=np.float64)
    return PositionTemplate(patch_mean, global_mean, x.shape[0])


def patch_state(value: object, state: str, *, template: PositionTemplate | None = None) -> np.ndarray:
    x = _patch_array(value, name="patch")
    if state == "X":
        return np.array(x, copy=True, order="C")
    per_image = np.mean(x, axis=1, dtype=np.float64, keepdims=True)
    within = x - per_image
    if state == "W":
        return within
    if state != "D":
        raise MetricError("patch state: expected X, W, or D")
    if template is None or template.patch_mean.shape != x.shape[1:]:
        raise MetricError("patch state D: missing or incompatible train template")
    return within - template.patch_mean[None, :, :] + template.global_mean[None, None, :]


def position_template_energy(value: object, *, template: PositionTemplate) -> dict[str, float]:
    x = _patch_array(value, name="patch")
    if template.patch_mean.shape != x.shape[1:]:
        raise MetricError("position energy: template shape mismatch")
    position = template.patch_mean - template.global_mean[None, :]
    centered = x - template.global_mean[None, None, :]
    denominator = float(np.sum(centered * centered, dtype=np.float64))
    if denominator == 0.0:
        raise MetricError("position energy: zero denominator")
    numerator = float(x.shape[0] * np.sum(position * position, dtype=np.float64))
    return {
        "position_template_energy": numerator,
        "total_train_centered_energy": denominator,
        "position_energy_fraction": numerator / denominator,
    }


def streaming_patch_cka(
    blocks: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    source_width: int,
    target_width: int,
) -> float:
    statistics = PairSufficientStatistics(source_width, target_width)
    image_blocks = 0
    for source, target in blocks:
        x = _patch_array(source, name="source patch block")
        y = _patch_array(target, name="target patch block")
        if x.shape[:2] != y.shape[:2]:
            raise MetricError("streaming patch CKA: image/patch shape mismatch")
        statistics.update(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]))
        image_blocks += x.shape[0]
    if image_blocks < 1:
        raise MetricError("streaming patch CKA: empty input")
    return statistics.linear_cka()


def permute_patches_within_images(value: object, *, seed: int) -> np.ndarray:
    x = _patch_array(value, name="patch")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    result = np.empty_like(x)
    for image in range(x.shape[0]):
        result[image] = x[image, generator.permutation(x.shape[1])]
    return result


def decode_coordinates(patch_count: int, *, grid: tuple[int, int] = (14, 14)) -> np.ndarray:
    if type(patch_count) is not int or patch_count != grid[0] * grid[1]:
        raise MetricError("coordinate decode: patch count/grid mismatch")
    row, column = np.unravel_index(np.arange(patch_count), grid)
    return np.column_stack((row, column)).astype(np.int64, copy=False)


def patch_semivariogram(value: object, *, grid: tuple[int, int] = (14, 14)) -> dict[str, float]:
    x = _patch_array(value, name="patch")
    if x.shape[1] != grid[0] * grid[1]:
        raise MetricError("semivariogram: grid mismatch")
    spatial = x.reshape(x.shape[0], grid[0], grid[1], x.shape[2])
    horizontal = spatial[:, :, 1:, :] - spatial[:, :, :-1, :]
    vertical = spatial[:, 1:, :, :] - spatial[:, :-1, :, :]
    horizontal_gamma = 0.5 * float(np.mean(horizontal * horizontal, dtype=np.float64))
    vertical_gamma = 0.5 * float(np.mean(vertical * vertical, dtype=np.float64))
    return {
        "horizontal_lag1_semivariance": horizontal_gamma,
        "vertical_lag1_semivariance": vertical_gamma,
        "adjacent_lag1_semivariance": (horizontal_gamma + vertical_gamma) / 2.0,
    }


def patch_norm_diagnostics(value: object) -> dict[str, float | int]:
    x = _patch_array(value, name="patch")
    norms = np.linalg.norm(x, axis=2)
    return {
        "image_count": int(x.shape[0]),
        "patch_count": int(x.shape[1]),
        "zero_norm_count": int(np.count_nonzero(norms == 0.0)),
        "norm_mean": float(np.mean(norms, dtype=np.float64)),
        "norm_median": float(np.median(norms)),
        "norm_minimum": float(np.min(norms)),
        "norm_maximum": float(np.max(norms)),
    }


__all__ = [
    "PositionTemplate", "decode_coordinates", "fit_position_template",
    "patch_norm_diagnostics", "patch_semivariogram", "patch_state",
    "permute_patches_within_images", "position_template_energy",
    "streaming_patch_cka",
]
