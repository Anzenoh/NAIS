"""Deterministic NumPy-only mock implementation of the extractor contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from .base import (
    CanonicalImageBatch,
    CanonicalTapFeatures,
    ExtractionResult,
    ExtractorDescriptor,
    ExtractorValidationError,
    NumericNormalizationSpec,
    TapSpec,
    TokenLayout,
    build_vit_tap_specs,
    resolve_requested_taps,
    select_patch_tokens,
)


_SPECIAL_TOKEN_SENTINEL = np.float32(-1_000_000.0)


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ExtractorValidationError(f"{path}: expected a positive integer")
    result = int(value)
    if result < 1:
        raise ExtractorValidationError(f"{path}: expected a positive integer")
    return result


def _special_token_count(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ExtractorValidationError(
            "mock_config.special_token_count: expected a non-negative integer"
        )
    result = int(value)
    if result < 0:
        raise ExtractorValidationError(
            "mock_config.special_token_count: expected a non-negative integer"
        )
    return result


def _mock_token_layout(
    grid_height: int, grid_width: int, special_token_count: int
) -> TokenLayout:
    patch_count = grid_height * grid_width
    total = patch_count + special_token_count
    if special_token_count == 0:
        special_indices: tuple[int, ...] = ()
        policy = "explicit_no_special_tokens"
    elif special_token_count == 1:
        special_indices = (0,)
        policy = "explicit_synthetic_prefix"
    else:
        special_indices = tuple(
            round(index * (total - 1) / (special_token_count - 1))
            for index in range(special_token_count)
        )
        policy = "explicit_synthetic_prefix_interior_suffix"
    special_set = set(special_indices)
    patch_indices = tuple(index for index in range(total) if index not in special_set)
    return TokenLayout(
        grid_height=grid_height,
        grid_width=grid_width,
        total_token_count=total,
        patch_token_indices=patch_indices,
        special_token_indices=special_indices,
        special_token_policy=policy,
    )


@dataclass(frozen=True, slots=True)
class MockExtractorConfig:
    """Validated shape and identity settings for the synthetic mock extractor."""

    model_key: str = "mock_vit"
    model_id: str = "synthetic/mock-vit"
    revision: str = "synthetic-v1"
    family: str = "mock_vit"
    input_height: int = 224
    input_width: int = 224
    grid_height: int = 14
    grid_width: int = 14
    hidden_dimension: int = 8
    layer_count: int = 4
    special_token_count: int = 3
    normalization: NumericNormalizationSpec = field(
        default_factory=lambda: NumericNormalizationSpec(
            scale=1.0 / 255.0,
            mean=(0.0, 0.0, 0.0),
            standard_deviation=(1.0, 1.0, 1.0),
            channel_order="RGB",
        )
    )

    def __post_init__(self) -> None:
        input_height = _positive_integer(self.input_height, "mock_config.input_height")
        input_width = _positive_integer(self.input_width, "mock_config.input_width")
        grid_height = _positive_integer(self.grid_height, "mock_config.grid_height")
        grid_width = _positive_integer(self.grid_width, "mock_config.grid_width")
        hidden_dimension = _positive_integer(
            self.hidden_dimension, "mock_config.hidden_dimension"
        )
        layer_count = _positive_integer(self.layer_count, "mock_config.layer_count")
        special_token_count = _special_token_count(self.special_token_count)
        if input_height % grid_height != 0:
            raise ExtractorValidationError(
                "mock_config.input_height: must be divisible by grid_height"
            )
        if input_width % grid_width != 0:
            raise ExtractorValidationError(
                "mock_config.input_width: must be divisible by grid_width"
            )
        if not isinstance(self.normalization, NumericNormalizationSpec):
            raise ExtractorValidationError(
                "mock_config.normalization: expected NumericNormalizationSpec"
            )
        for field_name in ("model_key", "model_id", "revision", "family"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ExtractorValidationError(
                    f"mock_config.{field_name}: expected a non-empty string"
                )

        endpoints = np.array([0.0, 255.0], dtype=np.float64)[:, None]
        means = np.asarray(self.normalization.mean, dtype=np.float64)[None, :]
        standard_deviations = np.asarray(
            self.normalization.standard_deviation, dtype=np.float64
        )[None, :]
        with np.errstate(over="ignore", invalid="ignore"):
            normalized_bounds = (
                endpoints * self.normalization.scale - means
            ) / standard_deviations
        float32_limit = np.finfo(np.float32).max / max(16.0, hidden_dimension * 4.0)
        if (
            not np.isfinite(normalized_bounds).all()
            or np.max(np.abs(normalized_bounds)) > float32_limit
        ):
            raise ExtractorValidationError(
                "mock_config.normalization: produces non-finite FP32 intermediates"
            )

        object.__setattr__(self, "input_height", input_height)
        object.__setattr__(self, "input_width", input_width)
        object.__setattr__(self, "grid_height", grid_height)
        object.__setattr__(self, "grid_width", grid_width)
        object.__setattr__(self, "hidden_dimension", hidden_dimension)
        object.__setattr__(self, "layer_count", layer_count)
        object.__setattr__(self, "special_token_count", special_token_count)


class MockVisionEncoderExtractor:
    """A deterministic content-dependent extractor with no learned parameters."""

    def __init__(self, config: MockExtractorConfig | None = None) -> None:
        if config is None:
            config = MockExtractorConfig()
        if not isinstance(config, MockExtractorConfig):
            raise ExtractorValidationError(
                "config: expected MockExtractorConfig or None"
            )
        token_layout = _mock_token_layout(
            config.grid_height, config.grid_width, config.special_token_count
        )
        taps = build_vit_tap_specs(config.layer_count)
        self._config = config
        self._descriptor = ExtractorDescriptor(
            model_key=config.model_key,
            model_id=config.model_id,
            revision=config.revision,
            family=config.family,
            implementation_class="MockVisionEncoderExtractor",
            vision_component_path="synthetic.mock.backbone",
            input_height=config.input_height,
            input_width=config.input_width,
            grid_height=config.grid_height,
            grid_width=config.grid_width,
            hidden_dimension=config.hidden_dimension,
            layer_count=config.layer_count,
            parameter_dtype="float32",
            compute_dtype="float32",
            output_dtype="float32",
            evaluation_mode=True,
            gradients_enabled=False,
            normalization=config.normalization,
            token_layout=token_layout,
            taps=taps,
        )

    @property
    def descriptor(self) -> ExtractorDescriptor:
        return self._descriptor

    @property
    def available_taps(self) -> tuple[TapSpec, ...]:
        return self._descriptor.taps

    def extract(
        self,
        batch: CanonicalImageBatch,
        *,
        tap_names: Sequence[str] | None = None,
    ) -> ExtractionResult:
        if not isinstance(batch, CanonicalImageBatch):
            raise ExtractorValidationError(
                "batch: expected CanonicalImageBatch"
            )
        expected_shape = (
            self._descriptor.input_height,
            self._descriptor.input_width,
            3,
        )
        if batch.rgb_uint8.shape[1:] != expected_shape:
            raise ExtractorValidationError(
                "batch.rgb_uint8.spatial_dimensions: expected "
                f"{expected_shape[:2]}, got {batch.rgb_uint8.shape[1:3]}"
            )
        requested = resolve_requested_taps(self.available_taps, tap_names)
        base_patch = self._base_patch_features(batch.rgb_uint8)
        features = tuple(
            CanonicalTapFeatures.from_patch(tap, self._tap_features(base_patch, tap))
            for tap in requested
        )
        return ExtractionResult(
            descriptor=self._descriptor,
            image_ids=batch.image_ids,
            tap_features=features,
        )

    def _base_patch_features(self, rgb_uint8: np.ndarray) -> np.ndarray:
        normalization = self._descriptor.normalization
        rgb = rgb_uint8.astype(np.float32, copy=True)
        rgb *= np.float32(normalization.scale)
        rgb -= np.asarray(normalization.mean, dtype=np.float32)
        rgb /= np.asarray(normalization.standard_deviation, dtype=np.float32)
        if not np.isfinite(rgb).all():
            raise ExtractorValidationError(
                "mock_intermediate.normalized_rgb: contains NaN or Inf"
            )

        batch_size = rgb.shape[0]
        grid_height = self._descriptor.grid_height
        grid_width = self._descriptor.grid_width
        patch_height = self._descriptor.input_height // grid_height
        patch_width = self._descriptor.input_width // grid_width
        patches = rgb.reshape(
            batch_size,
            grid_height,
            patch_height,
            grid_width,
            patch_width,
            3,
        ).transpose(0, 1, 3, 2, 4, 5)
        summaries = patches.mean(axis=(3, 4), dtype=np.float32).reshape(
            batch_size, self._descriptor.patch_count, 3
        )

        channel_index = np.arange(1, 4, dtype=np.float32)[:, None]
        feature_index = np.arange(
            1, self._descriptor.hidden_dimension + 1, dtype=np.float32
        )[None, :]
        projection = (
            np.remainder(channel_index * feature_index, np.float32(7.0)) + 1.0
        ) / 8.0
        projected = np.matmul(summaries, projection).astype(np.float32, copy=False)

        rows = np.repeat(np.arange(grid_height, dtype=np.float32), grid_width)
        columns = np.tile(np.arange(grid_width, dtype=np.float32), grid_height)
        if grid_height > 1:
            rows = rows * np.float32(2.0 / (grid_height - 1)) - np.float32(1.0)
        else:
            rows.fill(0.0)
        if grid_width > 1:
            columns = (
                columns * np.float32(2.0 / (grid_width - 1)) - np.float32(1.0)
            )
        else:
            columns.fill(0.0)
        hidden_index = np.arange(
            1, self._descriptor.hidden_dimension + 1, dtype=np.float32
        )
        position = (
            rows[:, None]
            * (hidden_index[None, :] / np.float32(self._descriptor.hidden_dimension + 1))
            + columns[:, None]
            * (
                np.remainder(hidden_index[None, :], np.float32(3.0)) + 1.0
            )
            / np.float32(self._descriptor.hidden_dimension + 2)
        ) * np.float32(0.05)
        result = projected + position[None, :, :]
        if not np.isfinite(result).all():
            raise ExtractorValidationError(
                "mock_intermediate.base_patch: contains NaN or Inf"
            )
        return np.ascontiguousarray(result, dtype=np.float32)

    def _tap_features(self, base_patch: np.ndarray, tap: TapSpec) -> np.ndarray:
        if tap.kind == "embedding_output":
            depth = 0
        elif tap.kind == "block":
            if tap.report_index is None:
                raise ExtractorValidationError("tap.report_index: expected an integer")
            depth = tap.report_index
        else:
            depth = self._descriptor.layer_count + 1
        scale = np.float32(1.0 + 0.25 * depth / (self._descriptor.layer_count + 1))
        hidden_index = np.arange(self._descriptor.hidden_dimension, dtype=np.int32)
        signs = np.where(hidden_index % 2 == 0, -1.0, 1.0).astype(np.float32)
        transformed = base_patch * scale + signs[None, None, :] * np.float32(
            depth * 0.01
        )

        layout = self._descriptor.token_layout
        raw_tokens = np.full(
            (
                transformed.shape[0],
                layout.total_token_count,
                transformed.shape[2],
            ),
            _SPECIAL_TOKEN_SENTINEL - np.float32(depth),
            dtype=np.float32,
        )
        raw_tokens[:, layout.patch_token_indices, :] = transformed
        patch = select_patch_tokens(raw_tokens, layout)
        sentinel = _SPECIAL_TOKEN_SENTINEL - np.float32(depth)
        if np.any(patch == sentinel):
            raise ExtractorValidationError(
                "mock_intermediate.patch: special-token sentinel leaked into output"
            )
        return patch
