"""Reduced-rank composition, cycles, and symmetric regularized MAXVAR GCCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .linear_alignment import RidgeMapping, per_image_squared_error, ridge_alpha_scale
from .statistics import MetricError, as_float_matrix


def compose_linear_maps(first: RidgeMapping, second: RidgeMapping) -> tuple[np.ndarray, np.ndarray]:
    """Compose compatible affine maps and return (weights, intercept)."""

    if first.target_dimension != second.source_dimension:
        raise MetricError("map composition: incompatible exact nodes")
    weights = first.weights @ second.weights
    first_intercept = first.target_mean - first.source_mean @ first.weights
    second_intercept = second.target_mean - second.source_mean @ second.weights
    intercept = first_intercept @ second.weights + second_intercept
    return weights, intercept


def apply_affine(value: object, weights: object, intercept: object) -> np.ndarray:
    x = as_float_matrix(value, name="value")
    matrix = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != x.shape[1] or bias.shape != (matrix.shape[1],):
        raise MetricError("affine map: shape mismatch")
    result = x @ matrix + bias
    if not np.all(np.isfinite(result)):
        raise MetricError("affine map: non-finite result")
    return result


def closed_cycle_error(value: object, mappings: Sequence[RidgeMapping]) -> dict[str, float]:
    if len(mappings) < 2:
        raise MetricError("closed cycle: at least two mappings required")
    x = as_float_matrix(value, name="cycle value")
    current = x
    for index, mapping in enumerate(mappings):
        if current.shape[1] != mapping.source_dimension:
            raise MetricError(f"closed cycle: incompatible mapping {index}")
        current = mapping.transform(current)
    if current.shape != x.shape:
        raise MetricError("closed cycle: output node differs from start node")
    residual = current - x
    denominator = float(np.sum((x - np.mean(x, axis=0)) ** 2, dtype=np.float64))
    if denominator == 0.0:
        raise MetricError("closed cycle: zero denominator")
    return {
        "cycle_nmse": float(np.sum(residual * residual, dtype=np.float64) / denominator),
        "cycle_mean_mse": float(np.mean(residual * residual, dtype=np.float64)),
    }


def _immutable(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(contiguous.shape)


@dataclass(frozen=True, slots=True, eq=False)
class MultiwayLatentModel:
    encoder_names: tuple[str, ...]
    means: tuple[np.ndarray, ...]
    encoders: tuple[np.ndarray, ...]
    decoders: tuple[np.ndarray, ...]
    relative_alpha: float
    absolute_alphas: tuple[float, ...]
    latent_dimension: int
    training_sample_count: int

    def __post_init__(self) -> None:
        count = len(self.encoder_names)
        if count < 2 or not (len(self.means) == len(self.encoders) == len(self.decoders) == len(self.absolute_alphas) == count):
            raise MetricError("multiway latent: inconsistent encoder state")
        if self.latent_dimension < 1 or self.training_sample_count < 2:
            raise MetricError("multiway latent: invalid dimensions")
        means: list[np.ndarray] = []
        encoders: list[np.ndarray] = []
        decoders: list[np.ndarray] = []
        for index in range(count):
            mean = _immutable(np.asarray(self.means[index]))
            encoder = _immutable(np.asarray(self.encoders[index]))
            decoder = _immutable(np.asarray(self.decoders[index]))
            if mean.ndim != 1 or encoder.shape != (mean.size, self.latent_dimension) or decoder.shape != (self.latent_dimension, mean.size):
                raise MetricError("multiway latent: incompatible map shape")
            if not all(np.all(np.isfinite(item)) for item in (mean, encoder, decoder)):
                raise MetricError("multiway latent: non-finite state")
            means.append(mean)
            encoders.append(encoder)
            decoders.append(decoder)
        object.__setattr__(self, "means", tuple(means))
        object.__setattr__(self, "encoders", tuple(encoders))
        object.__setattr__(self, "decoders", tuple(decoders))

    def transform(self, values: Sequence[object]) -> np.ndarray:
        if len(values) != len(self.encoder_names):
            raise MetricError("multiway transform: encoder count mismatch")
        latents: list[np.ndarray] = []
        for value, mean, encoder in zip(values, self.means, self.encoders, strict=True):
            x = as_float_matrix(value, name="multiway value")
            if x.shape[1] != mean.size:
                raise MetricError("multiway transform: feature width mismatch")
            latents.append((x - mean) @ encoder)
        if len({item.shape for item in latents}) != 1:
            raise MetricError("multiway transform: sample count mismatch")
        latent = np.mean(np.stack(latents, axis=0), axis=0, dtype=np.float64)
        latent -= np.mean(latent, axis=0, dtype=np.float64, keepdims=True)
        return latent

    def reconstruct(self, values: Sequence[object]) -> tuple[np.ndarray, ...]:
        latent = self.transform(values)
        return tuple(latent @ decoder + mean for decoder, mean in zip(self.decoders, self.means, strict=True))

    def shared_private(self, values: Sequence[object]) -> dict[str, object]:
        observed = tuple(as_float_matrix(item, name="multiway observed") for item in values)
        shared = self.reconstruct(observed)
        per_encoder: list[dict[str, float | str]] = []
        fractions: list[float] = []
        for name, value, reconstruction, mean in zip(self.encoder_names, observed, shared, self.means, strict=True):
            total = float(np.sum((value - mean) ** 2, dtype=np.float64))
            private = float(np.sum((value - reconstruction) ** 2, dtype=np.float64))
            if total == 0.0:
                raise MetricError("shared/private: zero total variance")
            fraction = 1.0 - private / total
            fractions.append(fraction)
            per_encoder.append({
                "encoder": name,
                "shared_variance_fraction": fraction,
                "private_residual_fraction": private / total,
            })
        return {
            "encoder_balanced_shared_variance_fraction": float(np.mean(fractions, dtype=np.float64)),
            "per_encoder": per_encoder,
        }


def fit_multiway_maxvar(
    values: Sequence[object],
    *,
    encoder_names: Sequence[str],
    latent_dimension: int,
    relative_alpha: float,
) -> MultiwayLatentModel:
    if len(values) < 2 or len(values) != len(encoder_names):
        raise MetricError("MAXVAR GCCA: encoder count mismatch")
    matrices = tuple(as_float_matrix(item, name="GCCA train") for item in values)
    if len({item.shape[0] for item in matrices}) != 1:
        raise MetricError("MAXVAR GCCA: sample counts differ")
    n = matrices[0].shape[0]
    maximum = min(n - 1, *(item.shape[1] for item in matrices))
    if type(latent_dimension) is not int or not 1 <= latent_dimension <= maximum:
        raise MetricError(f"MAXVAR GCCA: latent dimension must be <= {maximum}")
    if relative_alpha < 0.0 or not np.isfinite(relative_alpha):
        raise MetricError("MAXVAR GCCA: invalid regularization")
    means = tuple(np.mean(item, axis=0, dtype=np.float64) for item in matrices)
    centered = tuple(item - mean for item, mean in zip(matrices, means, strict=True))
    projections = np.zeros((n, n), dtype=np.float64)
    absolute: list[float] = []
    encoder_maps: list[np.ndarray] = []
    for matrix in centered:
        alpha = float(relative_alpha * ridge_alpha_scale(matrix))
        absolute.append(alpha)
        system = matrix.T @ matrix + alpha * np.eye(matrix.shape[1], dtype=np.float64)
        try:
            map_to_samples = np.linalg.solve(system, matrix.T)
        except np.linalg.LinAlgError as exc:
            raise MetricError("MAXVAR GCCA: regularized solve failed") from exc
        projections += matrix @ map_to_samples / len(centered)
        encoder_maps.append(map_to_samples)
    projections = (projections + projections.T) / 2.0
    centering = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    projections = centering @ projections @ centering
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(projections)
    except np.linalg.LinAlgError as exc:
        raise MetricError("MAXVAR GCCA: eigendecomposition failed") from exc
    order = np.argsort(-eigenvalues, kind="mergesort")[:latent_dimension]
    latent = eigenvectors[:, order]
    for column in range(latent.shape[1]):
        pivot = int(np.argmax(np.abs(latent[:, column])))
        if latent[pivot, column] < 0.0:
            latent[:, column] *= -1.0
    encoders = tuple(map_to_samples @ latent for map_to_samples in encoder_maps)
    decoders: list[np.ndarray] = []
    for matrix in centered:
        try:
            decoder, _, _, _ = np.linalg.lstsq(latent, matrix, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise MetricError("MAXVAR GCCA: decoder solve failed") from exc
        decoders.append(decoder)
    return MultiwayLatentModel(
        encoder_names=tuple(encoder_names),
        means=means,
        encoders=encoders,
        decoders=tuple(decoders),
        relative_alpha=float(relative_alpha),
        absolute_alphas=tuple(absolute),
        latent_dimension=latent_dimension,
        training_sample_count=n,
    )


def select_multiway_dimension(
    train_values: Sequence[object],
    validation_values: Sequence[object],
    *,
    encoder_names: Sequence[str],
    latent_candidates: Sequence[int],
    relative_alpha: float,
    bootstrap_count: int,
    seed: int,
) -> tuple[MultiwayLatentModel, list[dict[str, float | int | bool]]]:
    candidates: list[tuple[MultiwayLatentModel, np.ndarray]] = []
    observed = tuple(as_float_matrix(item, name="GCCA validation") for item in validation_values)
    for latent_dimension in latent_candidates:
        model = fit_multiway_maxvar(
            train_values,
            encoder_names=encoder_names,
            latent_dimension=int(latent_dimension),
            relative_alpha=relative_alpha,
        )
        reconstruction = model.reconstruct(observed)
        losses = np.mean(
            np.stack([
                per_image_squared_error(value, estimate)
                for value, estimate in zip(observed, reconstruction, strict=True)
            ], axis=0),
            axis=0,
            dtype=np.float64,
        )
        candidates.append((model, losses))
    means = np.array([np.mean(loss, dtype=np.float64) for _, loss in candidates])
    minimum = int(np.argmin(means))
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    boot = np.empty(bootstrap_count, dtype=np.float64)
    best_loss = candidates[minimum][1]
    for index in range(bootstrap_count):
        sample = generator.integers(0, best_loss.size, size=best_loss.size)
        boot[index] = np.mean(best_loss[sample], dtype=np.float64)
    threshold = means[minimum] + float(np.std(boot, ddof=1))
    eligible = [index for index, value in enumerate(means) if value <= threshold]
    selected = min(eligible, key=lambda index: (candidates[index][0].latent_dimension, index))
    table = [
        {
            "latent_dimension": model.latent_dimension,
            "validation_encoder_balanced_mse": float(means[index]),
            "eligible_one_standard_error": index in eligible,
            "selected": index == selected,
        }
        for index, (model, _) in enumerate(candidates)
    ]
    return candidates[selected][0], table


__all__ = [
    "MultiwayLatentModel", "apply_affine", "closed_cycle_error",
    "compose_linear_maps", "fit_multiway_maxvar", "select_multiway_dimension",
]
