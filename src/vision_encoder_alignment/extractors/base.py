"""Canonical extractor contracts shared by real and synthetic adapters."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Protocol, runtime_checkable

import numpy as np


class ExtractorValidationError(ValueError):
    """Raised when extractor input or output violates the public contract."""


_CANONICAL_NAMESPACE = "backbone"
_SUPPORTED_FEATURE_DTYPES = frozenset({"float16", "float32"})
_SUPPORTED_DESCRIPTOR_DTYPES = frozenset({"float32"})
_SUPPORTED_TAP_KINDS = frozenset({"embedding_output", "block", "final_norm"})
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_BLOCK_NAME_PATTERN = re.compile(r"^block_([0-9]{2,})$")


def _fail(path: str, message: str) -> None:
    raise ExtractorValidationError(f"{path}: {message}")


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "expected a non-empty string")
    return value


def _identifier(value: object, path: str) -> str:
    result = _nonempty_string(value, path)
    if _IDENTIFIER_PATTERN.fullmatch(result) is None:
        _fail(path, "expected a canonical lowercase identifier")
    return result


def _integer(value: object, path: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        _fail(path, f"expected an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        _fail(path, f"expected an integer >= {minimum}")
    return result


def _real(value: object, path: str, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        qualifier = "positive " if positive else ""
        _fail(path, f"expected a finite {qualifier}real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        _fail(path, f"expected a finite {qualifier}real number")
    return result


def _sequence(value: object, path: str) -> tuple[object, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            _fail(path, "expected a one-dimensional sequence")
        return tuple(value)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(path, "expected a sequence")
    return tuple(value)


def _image_ids(value: object, path: str = "image_ids") -> tuple[str, ...]:
    items = _sequence(value, path)
    if not items:
        _fail(path, "expected at least one image ID")
    validated: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        image_id = _nonempty_string(item, f"{path}[{index}]")
        if image_id in seen:
            _fail(f"{path}[{index}]", f"duplicate image ID {image_id!r}")
        seen.add(image_id)
        validated.append(image_id)
    return tuple(validated)


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.array(value, copy=True, order="C")
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return result.reshape(contiguous.shape)


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalImageBatch:
    """An ordered batch of canonical RGB pixels supplied to an extractor."""

    image_ids: tuple[str, ...]
    rgb_uint8: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        image_ids = _image_ids(self.image_ids)
        if not isinstance(self.rgb_uint8, np.ndarray):
            _fail("rgb_uint8", "expected a NumPy array")
        rgb = self.rgb_uint8
        if rgb.ndim != 4:
            _fail("rgb_uint8", "expected rank 4 [B, H, W, 3]")
        if rgb.shape[0] != len(image_ids):
            _fail("rgb_uint8.batch_dimension", "does not match image_ids count")
        if rgb.shape[0] < 1:
            _fail("rgb_uint8.batch_dimension", "expected at least one image")
        if rgb.shape[1] < 1 or rgb.shape[2] < 1:
            _fail("rgb_uint8.spatial_dimensions", "height and width must be positive")
        if rgb.shape[3] != 3:
            _fail("rgb_uint8.channel_dimension", "expected exactly three RGB channels")
        if rgb.dtype != np.dtype(np.uint8):
            _fail("rgb_uint8.dtype", "expected uint8")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "rgb_uint8", _immutable_array(rgb))


@dataclass(frozen=True, slots=True)
class NumericNormalizationSpec:
    """Explicit per-channel numeric normalization applied after uint8 loading."""

    scale: float
    mean: tuple[float, float, float]
    standard_deviation: tuple[float, float, float]
    channel_order: str

    def __post_init__(self) -> None:
        scale = _real(self.scale, "normalization.scale", positive=True)
        mean_items = _sequence(self.mean, "normalization.mean")
        std_items = _sequence(
            self.standard_deviation, "normalization.standard_deviation"
        )
        if len(mean_items) != 3:
            _fail("normalization.mean", "expected exactly three channel values")
        if len(std_items) != 3:
            _fail(
                "normalization.standard_deviation",
                "expected exactly three channel values",
            )
        mean = tuple(
            _real(item, f"normalization.mean[{index}]", positive=False)
            for index, item in enumerate(mean_items)
        )
        standard_deviation = tuple(
            _real(
                item,
                f"normalization.standard_deviation[{index}]",
                positive=True,
            )
            for index, item in enumerate(std_items)
        )
        if self.channel_order != "RGB":
            _fail("normalization.channel_order", "expected 'RGB'")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", standard_deviation)


@dataclass(frozen=True, slots=True)
class TapSpec:
    """A canonical hidden-state tap and its model-library index mapping."""

    namespace: str
    name: str
    kind: str
    report_index: int | None
    library_index: int | None

    def __post_init__(self) -> None:
        if self.namespace != _CANONICAL_NAMESPACE:
            _fail("tap.namespace", f"expected {_CANONICAL_NAMESPACE!r}")
        name = _identifier(self.name, "tap.name")
        if self.kind not in _SUPPORTED_TAP_KINDS:
            _fail("tap.kind", f"unsupported kind {self.kind!r}")

        if self.kind == "embedding_output":
            if name != "embedding_output":
                _fail("tap.name", "embedding output must be named 'embedding_output'")
            if self.report_index is not None or self.library_index is not None:
                _fail("tap.indices", "embedding output indices must be None")
            return

        if self.kind == "final_norm":
            if name != "final_norm":
                _fail("tap.name", "final norm must be named 'final_norm'")
            if self.report_index is not None or self.library_index is not None:
                _fail("tap.indices", "final norm indices must be None")
            return

        report_index = _integer(self.report_index, "tap.report_index", minimum=1)
        library_index = _integer(self.library_index, "tap.library_index", minimum=0)
        if library_index != report_index - 1:
            _fail("tap.library_index", "expected report_index - 1")
        match = _BLOCK_NAME_PATTERN.fullmatch(name)
        if match is None or int(match.group(1)) != report_index:
            _fail("tap.name", "does not match the one-based block report index")
        object.__setattr__(self, "report_index", report_index)
        object.__setattr__(self, "library_index", library_index)


def build_vit_tap_specs(
    layer_count: int,
    *,
    include_embedding_output: bool = True,
    include_final_norm: bool = True,
) -> tuple[TapSpec, ...]:
    """Build the complete ordered canonical tap catalog for a ViT backbone."""

    layers = _integer(layer_count, "layer_count", minimum=1)
    if type(include_embedding_output) is not bool:
        _fail("include_embedding_output", "expected a boolean")
    if type(include_final_norm) is not bool:
        _fail("include_final_norm", "expected a boolean")
    width = max(2, len(str(layers)))
    taps: list[TapSpec] = []
    if include_embedding_output:
        taps.append(
            TapSpec(_CANONICAL_NAMESPACE, "embedding_output", "embedding_output", None, None)
        )
    taps.extend(
        TapSpec(
            _CANONICAL_NAMESPACE,
            f"block_{report_index:0{width}d}",
            "block",
            report_index,
            report_index - 1,
        )
        for report_index in range(1, layers + 1)
    )
    if include_final_norm:
        taps.append(TapSpec(_CANONICAL_NAMESPACE, "final_norm", "final_norm", None, None))
    return tuple(taps)


def _tap_sequence(value: object, path: str, *, nonempty: bool) -> tuple[TapSpec, ...]:
    items = _sequence(value, path)
    if nonempty and not items:
        _fail(path, "expected at least one tap")
    result: list[TapSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, TapSpec):
            _fail(f"{path}[{index}]", "expected TapSpec")
        if item.name in seen:
            _fail(f"{path}[{index}].name", f"duplicate tap {item.name!r}")
        seen.add(item.name)
        result.append(item)
    return tuple(result)


def resolve_requested_taps(
    available_taps: Sequence[TapSpec],
    requested_tap_names: Sequence[str] | None,
) -> tuple[TapSpec, ...]:
    """Resolve an ordered tap-name request against an available catalog."""

    available = _tap_sequence(available_taps, "available_taps", nonempty=True)
    if requested_tap_names is None:
        return available
    requested = _sequence(requested_tap_names, "requested_tap_names")
    if not requested:
        _fail("requested_tap_names", "expected at least one tap name")
    by_name = {tap.name: tap for tap in available}
    resolved: list[TapSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(requested):
        name = _nonempty_string(item, f"requested_tap_names[{index}]")
        if name in seen:
            _fail(f"requested_tap_names[{index}]", f"duplicate tap {name!r}")
        if name not in by_name:
            _fail(f"requested_tap_names[{index}]", f"unknown tap {name!r}")
        seen.add(name)
        resolved.append(by_name[name])
    return tuple(resolved)


def resolve_normalized_block_taps(
    layer_count: int, normalized_depths: Sequence[float]
) -> tuple[str, ...]:
    """Map normalized depths to unique one-based block tap names."""

    layers = _integer(layer_count, "layer_count", minimum=1)
    depths = _sequence(normalized_depths, "normalized_depths")
    if not depths:
        _fail("normalized_depths", "expected at least one depth")
    width = max(2, len(str(layers)))
    result: list[str] = []
    seen: set[int] = set()
    for index, item in enumerate(depths):
        depth = _real(item, f"normalized_depths[{index}]", positive=True)
        if depth > 1.0:
            _fail(f"normalized_depths[{index}]", "expected a value in (0, 1]")
        scaled_depth = depth * layers
        nearest_integer = int(round(scaled_depth))
        if 1 <= nearest_integer <= layers:
            # Recover only integer boundaries obscured by binary float roundoff.
            tolerance = 4.0 * max(
                math.ulp(scaled_depth),
                math.ulp(float(nearest_integer)),
            )
            if abs(scaled_depth - nearest_integer) <= tolerance:
                scaled_depth = float(nearest_integer)
        report_index = min(layers, max(1, math.ceil(scaled_depth)))
        if report_index not in seen:
            seen.add(report_index)
            result.append(f"block_{report_index:0{width}d}")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TokenLayout:
    """Explicit classification and canonical ordering of hidden-state tokens."""

    grid_height: int
    grid_width: int
    total_token_count: int
    patch_token_indices: tuple[int, ...]
    special_token_indices: tuple[int, ...]
    special_token_policy: str

    def __post_init__(self) -> None:
        grid_height = _integer(self.grid_height, "token_layout.grid_height", minimum=1)
        grid_width = _integer(self.grid_width, "token_layout.grid_width", minimum=1)
        patch_count = grid_height * grid_width
        total = _integer(
            self.total_token_count, "token_layout.total_token_count", minimum=patch_count
        )
        patch_items = _sequence(
            self.patch_token_indices, "token_layout.patch_token_indices"
        )
        special_items = _sequence(
            self.special_token_indices, "token_layout.special_token_indices"
        )

        def validate_indices(items: tuple[object, ...], path: str) -> tuple[int, ...]:
            result: list[int] = []
            seen: set[int] = set()
            for index, item in enumerate(items):
                token_index = _integer(item, f"{path}[{index}]", minimum=0)
                if token_index >= total:
                    _fail(f"{path}[{index}]", f"expected an index below {total}")
                if token_index in seen:
                    _fail(f"{path}[{index}]", f"duplicate index {token_index}")
                seen.add(token_index)
                result.append(token_index)
            return tuple(result)

        patch_indices = validate_indices(
            patch_items, "token_layout.patch_token_indices"
        )
        special_indices = validate_indices(
            special_items, "token_layout.special_token_indices"
        )
        if len(patch_indices) != patch_count:
            _fail(
                "token_layout.patch_token_indices",
                f"expected exactly {patch_count} patch indices",
            )
        overlap = set(patch_indices) & set(special_indices)
        if overlap:
            _fail("token_layout.indices", "patch and special token indices overlap")
        coverage = set(patch_indices) | set(special_indices)
        if coverage != set(range(total)):
            _fail("token_layout.indices", "patch and special indices must cover all tokens")
        policy = _nonempty_string(
            self.special_token_policy, "token_layout.special_token_policy"
        )
        object.__setattr__(self, "grid_height", grid_height)
        object.__setattr__(self, "grid_width", grid_width)
        object.__setattr__(self, "total_token_count", total)
        object.__setattr__(self, "patch_token_indices", patch_indices)
        object.__setattr__(self, "special_token_indices", special_indices)
        object.__setattr__(self, "special_token_policy", policy)

    @property
    def observed_non_patch_token_count(self) -> int:
        return len(self.special_token_indices)


def select_patch_tokens(hidden_state: np.ndarray, token_layout: TokenLayout) -> np.ndarray:
    """Select canonical row-major patch tokens using explicit token indices."""

    if not isinstance(token_layout, TokenLayout):
        _fail("token_layout", "expected TokenLayout")
    if not isinstance(hidden_state, np.ndarray):
        _fail("hidden_state", "expected a NumPy array")
    if hidden_state.ndim != 3:
        _fail("hidden_state", "expected rank 3 [B, T, D]")
    if hidden_state.shape[0] < 1:
        _fail("hidden_state.batch_dimension", "expected at least one item")
    if hidden_state.shape[1] != token_layout.total_token_count:
        _fail("hidden_state.token_dimension", "does not match token_layout")
    if hidden_state.shape[2] < 1:
        _fail("hidden_state.hidden_dimension", "expected a positive dimension")
    if hidden_state.dtype.name not in _SUPPORTED_FEATURE_DTYPES:
        _fail("hidden_state.dtype", f"unsupported dtype {hidden_state.dtype.name!r}")
    if not np.isfinite(hidden_state).all():
        _fail("hidden_state", "contains NaN or Inf")
    selected = hidden_state[:, token_layout.patch_token_indices, :]
    return np.array(selected, copy=True, order="C")


@dataclass(frozen=True, slots=True)
class ExtractorDescriptor:
    """Validated static metadata describing one extractor implementation."""

    model_key: str
    model_id: str
    revision: str
    family: str
    implementation_class: str
    vision_component_path: str
    input_height: int
    input_width: int
    grid_height: int
    grid_width: int
    hidden_dimension: int
    layer_count: int
    parameter_dtype: str
    compute_dtype: str
    output_dtype: str
    evaluation_mode: bool
    gradients_enabled: bool
    normalization: NumericNormalizationSpec
    token_layout: TokenLayout
    taps: tuple[TapSpec, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "model_key",
            "model_id",
            "revision",
            "family",
            "implementation_class",
            "vision_component_path",
        ):
            _nonempty_string(getattr(self, field_name), f"descriptor.{field_name}")
        input_height = _integer(self.input_height, "descriptor.input_height", minimum=1)
        input_width = _integer(self.input_width, "descriptor.input_width", minimum=1)
        grid_height = _integer(self.grid_height, "descriptor.grid_height", minimum=1)
        grid_width = _integer(self.grid_width, "descriptor.grid_width", minimum=1)
        hidden_dimension = _integer(
            self.hidden_dimension, "descriptor.hidden_dimension", minimum=1
        )
        layer_count = _integer(self.layer_count, "descriptor.layer_count", minimum=1)
        for field_name in ("parameter_dtype", "compute_dtype", "output_dtype"):
            dtype = getattr(self, field_name)
            if dtype not in _SUPPORTED_DESCRIPTOR_DTYPES:
                _fail(f"descriptor.{field_name}", f"unsupported dtype {dtype!r}")
        if self.evaluation_mode is not True:
            _fail("descriptor.evaluation_mode", "expected True")
        if self.gradients_enabled is not False:
            _fail("descriptor.gradients_enabled", "expected False")
        if not isinstance(self.normalization, NumericNormalizationSpec):
            _fail("descriptor.normalization", "expected NumericNormalizationSpec")
        if not isinstance(self.token_layout, TokenLayout):
            _fail("descriptor.token_layout", "expected TokenLayout")
        if (
            self.token_layout.grid_height != grid_height
            or self.token_layout.grid_width != grid_width
        ):
            _fail("descriptor.token_layout.grid", "does not match descriptor grid")
        taps = _tap_sequence(self.taps, "descriptor.taps", nonempty=True)
        include_embedding = any(tap.kind == "embedding_output" for tap in taps)
        include_final = any(tap.kind == "final_norm" for tap in taps)
        expected = build_vit_tap_specs(
            layer_count,
            include_embedding_output=include_embedding,
            include_final_norm=include_final,
        )
        if taps != expected:
            _fail(
                "descriptor.taps",
                "expected a complete ordered block catalog for layer_count",
            )
        object.__setattr__(self, "input_height", input_height)
        object.__setattr__(self, "input_width", input_width)
        object.__setattr__(self, "grid_height", grid_height)
        object.__setattr__(self, "grid_width", grid_width)
        object.__setattr__(self, "hidden_dimension", hidden_dimension)
        object.__setattr__(self, "layer_count", layer_count)
        object.__setattr__(self, "taps", taps)

    @property
    def patch_count(self) -> int:
        return self.grid_height * self.grid_width


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalTapFeatures:
    """Canonical patch and pooled features for one extractor tap."""

    tap: TapSpec
    patch: np.ndarray = field(repr=False)
    global_mean: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tap, TapSpec):
            _fail("tap_features.tap", "expected TapSpec")
        if not isinstance(self.patch, np.ndarray):
            _fail("tap_features.patch", "expected a NumPy array")
        if not isinstance(self.global_mean, np.ndarray):
            _fail("tap_features.global_mean", "expected a NumPy array")
        patch = self.patch
        global_mean = self.global_mean
        if patch.ndim != 3:
            _fail("tap_features.patch", "expected rank 3 [B, P, D]")
        if global_mean.ndim != 2:
            _fail("tap_features.global_mean", "expected rank 2 [B, D]")
        if min(patch.shape) < 1:
            _fail("tap_features.patch", "all dimensions must be positive")
        if patch.shape[0] != global_mean.shape[0]:
            _fail("tap_features.batch_dimension", "patch and global_mean differ")
        if patch.shape[2] != global_mean.shape[1]:
            _fail("tap_features.hidden_dimension", "patch and global_mean differ")
        if patch.dtype.name not in _SUPPORTED_FEATURE_DTYPES:
            _fail("tap_features.patch.dtype", f"unsupported dtype {patch.dtype.name!r}")
        if global_mean.dtype != patch.dtype:
            _fail("tap_features.dtype", "patch and global_mean dtypes differ")
        if not np.isfinite(patch).all():
            _fail("tap_features.patch", "contains NaN or Inf")
        if not np.isfinite(global_mean).all():
            _fail("tap_features.global_mean", "contains NaN or Inf")
        calculated = patch.mean(axis=1, dtype=np.float32)
        if patch.dtype == np.dtype(np.float16):
            rtol, atol = 1e-3, 1e-3
        else:
            rtol, atol = 1e-5, 1e-6
        if not np.allclose(
            global_mean.astype(np.float32, copy=False),
            calculated,
            rtol=rtol,
            atol=atol,
        ):
            _fail(
                "tap_features.global_mean",
                "does not match the patch-axis mean within dtype tolerance",
            )
        object.__setattr__(self, "patch", _immutable_array(patch))
        object.__setattr__(self, "global_mean", _immutable_array(global_mean))

    @classmethod
    def from_patch(cls, tap: TapSpec, patch: np.ndarray) -> "CanonicalTapFeatures":
        """Construct canonical features using an FP32 patch-axis mean."""

        if not isinstance(patch, np.ndarray):
            _fail("tap_features.patch", "expected a NumPy array")
        if patch.ndim != 3:
            _fail("tap_features.patch", "expected rank 3 [B, P, D]")
        if min(patch.shape) < 1:
            _fail("tap_features.patch", "all dimensions must be positive")
        if patch.dtype.name not in _SUPPORTED_FEATURE_DTYPES:
            _fail("tap_features.patch.dtype", f"unsupported dtype {patch.dtype.name!r}")
        global_mean = patch.mean(axis=1, dtype=np.float32).astype(patch.dtype, copy=False)
        return cls(tap=tap, patch=patch, global_mean=global_mean)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """An ordered set of canonical tap features for one image batch."""

    descriptor: ExtractorDescriptor
    image_ids: tuple[str, ...]
    tap_features: tuple[CanonicalTapFeatures, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ExtractorDescriptor):
            _fail("result.descriptor", "expected ExtractorDescriptor")
        image_ids = _image_ids(self.image_ids, "result.image_ids")
        items = _sequence(self.tap_features, "result.tap_features")
        if not items:
            _fail("result.tap_features", "expected at least one tap feature")
        descriptor_taps = {tap.name: tap for tap in self.descriptor.taps}
        features: list[CanonicalTapFeatures] = []
        seen: set[str] = set()
        for index, item in enumerate(items):
            path = f"result.tap_features[{index}]"
            if not isinstance(item, CanonicalTapFeatures):
                _fail(path, "expected CanonicalTapFeatures")
            name = item.tap.name
            if name in seen:
                _fail(f"{path}.tap.name", f"duplicate tap {name!r}")
            if name not in descriptor_taps or item.tap != descriptor_taps[name]:
                _fail(f"{path}.tap", "is not present in the descriptor catalog")
            if item.patch.shape[0] != len(image_ids):
                _fail(f"{path}.batch_dimension", "does not match image_ids count")
            if item.patch.shape[1] != self.descriptor.patch_count:
                _fail(f"{path}.patch_dimension", "does not match descriptor patch count")
            if item.patch.shape[2] != self.descriptor.hidden_dimension:
                _fail(
                    f"{path}.hidden_dimension",
                    "does not match descriptor hidden dimension",
                )
            if item.patch.dtype.name != self.descriptor.output_dtype:
                _fail(f"{path}.dtype", "does not match descriptor output dtype")
            seen.add(name)
            features.append(item)
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "tap_features", tuple(features))

    @property
    def tap_names(self) -> tuple[str, ...]:
        return tuple(feature.tap.name for feature in self.tap_features)

    def get_tap(self, tap_name: str) -> CanonicalTapFeatures:
        name = _nonempty_string(tap_name, "tap_name")
        for feature in self.tap_features:
            if feature.tap.name == name:
                return feature
        _fail("tap_name", f"unknown extracted tap {name!r}")


@runtime_checkable
class VisionEncoderExtractor(Protocol):
    """Structural interface implemented by canonical vision encoder adapters."""

    @property
    def descriptor(self) -> ExtractorDescriptor: ...

    @property
    def available_taps(self) -> tuple[TapSpec, ...]: ...

    def extract(
        self,
        batch: CanonicalImageBatch,
        *,
        tap_names: Sequence[str] | None = None,
    ) -> ExtractionResult: ...
