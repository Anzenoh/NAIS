"""Offline, exact-revision SigLIP2 vision feature extraction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import random
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
from ..production_runtime_contract import (
    MAIN_SELECTOR,
    ProductionRuntimeContractError,
    ProductionRuntimeProfile,
    load_production_runtime_profile,
    validate_profile_source_bindings,
)


_MODEL_KEY = "siglip2_base_p16_224"
_MODEL_ID = "google/siglip2-base-patch16-224"
_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
_INPUT_SIZE = 224
_PATCH_SIZE = 16
_GRID_SIZE = 14
_PATCH_COUNT = 196
_TOKEN_COUNT = 196
_HIDDEN_DIMENSION = 768
_LAYER_COUNT = 12
_ATTENTION_HEAD_COUNT = 12
_INTERMEDIATE_SIZE = 3_072
_INPUT_CHANNEL_COUNT = 3
_LAYER_NORM_EPSILON = 1e-6
_PARAMETER_COUNT = 92_884_224
_PARAMETER_TENSOR_COUNT = 208
_SPECIAL_TOKEN_POLICY = "siglip_no_special_tokens_verified_runtime_v1"
_PROCESSOR_SCALE = 1.0 / 255.0
_PROCESSOR_MEAN = (0.5, 0.5, 0.5)
_PROCESSOR_STD = (0.5, 0.5, 0.5)
_ALLOWED_UNEXPECTED_PREFIXES = ("text_model.",)
_ALLOWED_UNEXPECTED_KEYS = frozenset({"logit_scale", "logit_bias"})
_DEPENDENCY_ERROR = (
    "SigLIP2 runtime dependencies are unavailable; install the audited optional "
    "real-model runtime dependencies"
)


@dataclass(frozen=True, slots=True)
class _SnapshotFileSpec:
    size: int
    sha256: str
    json_document: bool = False


_EXPECTED_SNAPSHOT_FILES = {
    "config.json": _SnapshotFileSpec(
        253,
        "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7",
        True,
    ),
    "preprocessor_config.json": _SnapshotFileSpec(
        394,
        "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc",
        True,
    ),
    "model.safetensors": _SnapshotFileSpec(
        1_500_800_904,
        "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
    ),
}

_EXPECTED_SOURCE_HASHES = {
    "configuration_siglip": (
        "aeb6d6505a2c8238d3b2f3968de30db9ea888797f7a04820f3bcf0c67c12608c"
    ),
    "image_processing_siglip": (
        "c3050709210910d61a04f2b27923b7b843e2c64e1140254c154873ac2d90e413"
    ),
    "modeling_siglip": (
        "9bf1830e8d0d384fceebf4e03a7928189b348906746aed8c703e78fdecca4d4b"
    ),
}


def _fail(path: str, message: str) -> None:
    raise ExtractorValidationError(f"{path}: {message}")


def _numpy_rng_states_equal(
    left: tuple[object, ...], right: tuple[object, ...]
) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


@contextmanager
def _preserve_python_numpy_rng() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        yield
    finally:
        if random.getstate() != python_state:
            random.setstate(python_state)
        current_numpy_state = np.random.get_state()
        if not _numpy_rng_states_equal(current_numpy_state, numpy_state):
            np.random.set_state(numpy_state)


@dataclass(frozen=True, slots=True, repr=False)
class SigLIP2ExtractorConfig:
    """Identity and exact local snapshot used by the audited SigLIP2 extractor."""

    snapshot_path: Path
    model_key: str = _MODEL_KEY
    model_id: str = _MODEL_ID
    revision: str = _REVISION
    runtime_profile_path: Path | None = None
    execution_selector: str = "legacy_windows_cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_path, (str, Path)):
            _fail("siglip2_config.snapshot_path", "expected a string or Path")
        try:
            snapshot = Path(self.snapshot_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ExtractorValidationError(
                "siglip2_config.snapshot_path: directory does not exist"
            ) from exc
        if not snapshot.is_dir():
            _fail("siglip2_config.snapshot_path", "expected an existing directory")
        if snapshot.name != _REVISION:
            _fail(
                "siglip2_config.snapshot_path",
                "directory basename must equal the audited revision",
            )
        if self.model_key != _MODEL_KEY:
            _fail("siglip2_config.model_key", f"expected {_MODEL_KEY!r}")
        if self.model_id != _MODEL_ID:
            _fail("siglip2_config.model_id", f"expected {_MODEL_ID!r}")
        if self.revision != _REVISION:
            _fail("siglip2_config.revision", f"expected audited commit {_REVISION!r}")
        if self.execution_selector not in {"legacy_windows_cpu", MAIN_SELECTOR}:
            _fail("siglip2_config.execution_selector", "unknown explicit execution selector")
        runtime_profile: Path | None = None
        if self.execution_selector == MAIN_SELECTOR:
            if not isinstance(self.runtime_profile_path, (str, Path)):
                _fail(
                    "siglip2_config.runtime_profile_path",
                    "Apple selector requires an explicit runtime profile",
                )
            runtime_profile = Path(self.runtime_profile_path).resolve(strict=True)
            expected_runtime = (
                Path(__file__).resolve().parents[3]
                / "configs/runtime_profiles/p1_apple_main_cuda_fp32_v0.json"
            ).resolve(strict=True)
            if runtime_profile != expected_runtime or not runtime_profile.is_file():
                _fail(
                    "siglip2_config.runtime_profile_path",
                    "expected the repository Apple main runtime profile",
                )
        elif self.runtime_profile_path is not None:
            _fail(
                "siglip2_config.runtime_profile_path",
                "legacy selector forbids an Apple runtime profile",
            )
        object.__setattr__(self, "snapshot_path", snapshot)
        object.__setattr__(self, "runtime_profile_path", runtime_profile)

    def __repr__(self) -> str:
        return (
            "SigLIP2ExtractorConfig("
            "snapshot_path=<redacted-local-snapshot>, "
            f"model_key={self.model_key!r}, model_id={self.model_id!r}, "
            f"revision={self.revision!r})"
        )


def _selected_runtime_profile(
    config: SigLIP2ExtractorConfig,
) -> ProductionRuntimeProfile | None:
    if config.execution_selector == "legacy_windows_cpu":
        return None
    assert config.runtime_profile_path is not None
    try:
        profile = load_production_runtime_profile(config.runtime_profile_path)
        validate_profile_source_bindings(
            profile, Path(__file__).resolve().parents[3]
        )
    except ProductionRuntimeContractError as exc:
        raise ExtractorValidationError(
            f"siglip2_apple_runtime.{exc.classification}: runtime contract failed"
        ) from exc
    if profile.selector != MAIN_SELECTOR:
        _fail("siglip2_apple_runtime.selector", "not the explicit Apple main selector")
    if not profile.package_metadata_resolved:
        _fail(
            "siglip2_apple_runtime.packages",
            "exact Apple package metadata remains unresolved",
        )
    return profile


@dataclass(frozen=True, slots=True)
class _VerifiedSnapshot:
    file_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _RuntimeMetadata:
    model_class: str
    vision_component_path: str
    input_height: int
    input_width: int
    patch_size: int
    grid_height: int
    grid_width: int
    hidden_dimension: int
    layer_count: int
    parameter_count: int
    parameter_tensor_count: int
    parameter_dtype: str
    compute_dtype: str
    output_dtype: str
    evaluation_mode: bool
    gradients_enabled: bool
    normalization: NumericNormalizationSpec


@dataclass(frozen=True, slots=True)
class _RuntimeForwardResult:
    hidden_states: tuple[np.ndarray, ...]
    last_hidden_state: np.ndarray
    final_normalized: np.ndarray
    pooler_output: np.ndarray
    recomputed_pooler: np.ndarray


class _SigLIP2Runtime(Protocol):
    metadata: _RuntimeMetadata

    def forward(self, normalized_nchw_float32: np.ndarray) -> _RuntimeForwardResult: ...

    def model_state_sha256(self) -> str: ...


@dataclass(frozen=True, slots=True)
class _RuntimeImports:
    torch: Any
    siglip_config: Any
    siglip_vision_config: Any
    siglip_image_processor: Any
    siglip_vision_model: Any
    configuration_module: Any
    image_processing_module: Any
    modeling_module: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_siglip2_snapshot(snapshot_path: Path) -> _VerifiedSnapshot:
    if snapshot_path.name != _REVISION:
        _fail("siglip2_snapshot", "directory basename does not match audited revision")
    try:
        actual_entries = {item.name for item in snapshot_path.iterdir()}
    except OSError as exc:
        raise ExtractorValidationError(
            "siglip2_snapshot: unable to inspect snapshot directory"
        ) from exc
    expected_files = set(_EXPECTED_SNAPSHOT_FILES)
    if actual_entries != expected_files:
        missing = sorted(expected_files - actual_entries)
        unexpected = sorted(actual_entries - expected_files)
        details = []
        if missing:
            details.append(f"missing {missing!r}")
        if unexpected:
            details.append(f"unexpected {unexpected!r}")
        _fail("siglip2_snapshot.files", "; ".join(details))

    verified: list[tuple[str, str]] = []
    for name, expected in _EXPECTED_SNAPSHOT_FILES.items():
        path = snapshot_path / name
        try:
            if not path.is_file():
                _fail(f"siglip2_snapshot.{name}", "expected a regular file")
            size = path.stat().st_size
        except OSError as exc:
            raise ExtractorValidationError(
                f"siglip2_snapshot.{name}: unable to inspect file"
            ) from exc
        if size == 0:
            _fail(f"siglip2_snapshot.{name}", "zero-byte files are not allowed")
        if size != expected.size:
            _fail(
                f"siglip2_snapshot.{name}.size",
                f"expected {expected.size}, found {size}",
            )
        try:
            digest = _sha256_file(path)
        except OSError as exc:
            raise ExtractorValidationError(
                f"siglip2_snapshot.{name}: unable to hash file"
            ) from exc
        if digest != expected.sha256:
            _fail(f"siglip2_snapshot.{name}.sha256", "does not match audited digest")
        if expected.json_document:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ExtractorValidationError(
                    f"siglip2_snapshot.{name}: expected valid UTF-8 JSON"
                ) from exc
            if not isinstance(document, dict):
                _fail(f"siglip2_snapshot.{name}", "expected a JSON object")
        verified.append((name, digest))
    return _VerifiedSnapshot(tuple(verified))


def _import_siglip2_runtime() -> _RuntimeImports:
    try:
        import PIL  # noqa: F401
        import huggingface_hub  # noqa: F401
        import safetensors  # noqa: F401
        import torch
        from transformers import (
            SiglipConfig,
            SiglipImageProcessor,
            SiglipVisionConfig,
            SiglipVisionModel,
        )
        from transformers.models.siglip import (
            configuration_siglip,
            image_processing_siglip,
            modeling_siglip,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise ExtractorValidationError(_DEPENDENCY_ERROR) from exc
    return _RuntimeImports(
        torch=torch,
        siglip_config=SiglipConfig,
        siglip_vision_config=SiglipVisionConfig,
        siglip_image_processor=SiglipImageProcessor,
        siglip_vision_model=SiglipVisionModel,
        configuration_module=configuration_siglip,
        image_processing_module=image_processing_siglip,
        modeling_module=modeling_siglip,
    )


def _validate_runtime_identity(runtime: _RuntimeImports) -> None:
    if sys.version_info[:2] != (3, 12):
        _fail(
            "siglip2_runtime.python.version",
            f"expected Python 3.12, found {sys.version_info.major}.{sys.version_info.minor}",
        )
    expected_versions = {
        "torch": "2.3.0",
        "transformers": "4.40.1",
        "Pillow": "10.3.0",
        "huggingface-hub": "0.22.2",
        "safetensors": "0.4.3",
    }
    for package, expected in expected_versions.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ExtractorValidationError(_DEPENDENCY_ERROR) from exc
        if actual != expected:
            _fail(
                f"siglip2_runtime.{package}.version",
                f"expected {expected!r}, found {actual!r}",
            )

    torch = runtime.torch
    if torch.__version__ != "2.3.0+cpu":
        _fail(
            "siglip2_runtime.torch.runtime",
            f"expected '2.3.0+cpu', found {torch.__version__!r}",
        )
    if (
        torch.version.cuda is not None
        or torch.cuda.is_available()
        or torch.cuda.device_count() != 0
    ):
        _fail("siglip2_runtime.device", "expected the audited CPU-only runtime")

    modules = {
        "configuration_siglip": runtime.configuration_module,
        "image_processing_siglip": runtime.image_processing_module,
        "modeling_siglip": runtime.modeling_module,
    }
    for name, module in modules.items():
        try:
            source_path = Path(inspect.getfile(module))
            digest = _sha256_file(source_path)
        except (OSError, TypeError) as exc:
            raise ExtractorValidationError(
                f"siglip2_runtime.source.{name}: unable to inspect source"
            ) from exc
        if digest != _EXPECTED_SOURCE_HASHES[name]:
            _fail(f"siglip2_runtime.source.{name}", "source hash mismatch")


def _validate_config_and_processor(
    full_config: Any, processor: Any
) -> tuple[Any, NumericNormalizationSpec]:
    if full_config.__class__.__name__ != "SiglipConfig":
        _fail("siglip2_runtime.config.class", "expected 'SiglipConfig'")
    if getattr(full_config, "model_type", None) != "siglip":
        _fail("siglip2_runtime.config.model_type", "expected 'siglip'")
    vision_config = getattr(full_config, "vision_config", None)
    if vision_config is None or vision_config.__class__.__name__ != "SiglipVisionConfig":
        _fail(
            "siglip2_runtime.config.vision_config.class",
            "expected 'SiglipVisionConfig'",
        )
    expected_vision = {
        "model_type": "siglip_vision_model",
        "image_size": _INPUT_SIZE,
        "patch_size": _PATCH_SIZE,
        "hidden_size": _HIDDEN_DIMENSION,
        "num_hidden_layers": _LAYER_COUNT,
        "num_attention_heads": _ATTENTION_HEAD_COUNT,
        "intermediate_size": _INTERMEDIATE_SIZE,
        "num_channels": _INPUT_CHANNEL_COUNT,
        "layer_norm_eps": _LAYER_NORM_EPSILON,
    }
    for name, expected in expected_vision.items():
        if getattr(vision_config, name, None) != expected:
            _fail(
                f"siglip2_runtime.config.vision_config.{name}",
                f"expected {expected!r}",
            )
    if _INPUT_SIZE // _PATCH_SIZE != _GRID_SIZE:
        _fail("siglip2_runtime.config.grid", "audited geometry is inconsistent")

    if processor.__class__.__name__ != "SiglipImageProcessor":
        _fail("siglip2_runtime.processor.class", "expected 'SiglipImageProcessor'")
    for name in ("do_rescale", "do_normalize", "do_resize"):
        if getattr(processor, name, None) is not True:
            _fail(f"siglip2_runtime.processor.{name}", "expected True")
    if getattr(processor, "do_convert_rgb", None) not in (None, False):
        _fail("siglip2_runtime.processor.do_convert_rgb", "expected disabled")
    scale = float(getattr(processor, "rescale_factor", math.nan))
    try:
        mean = tuple(float(value) for value in processor.image_mean)
        std = tuple(float(value) for value in processor.image_std)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExtractorValidationError(
            "siglip2_runtime.processor.normalization: expected numeric channel metadata"
        ) from exc
    values = np.asarray((scale, *mean, *std), dtype=np.float64)
    if not np.isfinite(values).all():
        _fail("siglip2_runtime.processor.normalization", "contains NaN or Inf")
    if scale != _PROCESSOR_SCALE or mean != _PROCESSOR_MEAN or std != _PROCESSOR_STD:
        _fail("siglip2_runtime.processor.normalization", "audited values do not match")
    if getattr(processor, "size", None) != {
        "height": _INPUT_SIZE,
        "width": _INPUT_SIZE,
    }:
        _fail("siglip2_runtime.processor.size", "expected 224 x 224")
    if getattr(processor, "crop_size", None) is not None:
        _fail("siglip2_runtime.processor.crop_size", "expected center crop to be absent")
    if getattr(processor, "resample", None) != 2:
        _fail("siglip2_runtime.processor.resample", "expected 2")
    return vision_config, NumericNormalizationSpec(scale, mean, std, "RGB")


def _model_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(memoryview(tensor.numpy()))
    return digest.hexdigest()


class _TorchSigLIP2Runtime:
    __slots__ = ("_device", "_initial_state_sha256", "_model", "_torch", "metadata")

    def __init__(
        self,
        model: Any,
        torch_module: Any,
        metadata: _RuntimeMetadata,
        *,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._torch = torch_module
        self.metadata = metadata
        self._device = device
        self._initial_state_sha256 = _model_state_sha256(model)

    def model_state_sha256(self) -> str:
        return _model_state_sha256(self._model)

    def forward(self, normalized_nchw_float32: np.ndarray) -> _RuntimeForwardResult:
        torch = self._torch
        if self._model.training:
            _fail("siglip2_runtime.model.training", "expected evaluation mode")
        if any(parameter.requires_grad for parameter in self._model.parameters()):
            _fail("siglip2_runtime.model.parameters", "expected all parameters frozen")
        rng_before = torch.get_rng_state().clone()
        try:
            fork_devices = [0] if self._device == "cuda:0" else []
            with torch.random.fork_rng(devices=fork_devices, enabled=True):
                pixel_values = torch.from_numpy(normalized_nchw_float32).to(
                    device=self._device, dtype=torch.float32
                )
                with torch.inference_mode():
                    outputs = self._model(
                        pixel_values=pixel_values,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = tuple(outputs.hidden_states)
                    final_normalized = self._model.vision_model.post_layernorm(
                        hidden_states[-1]
                    )
                    recomputed_pooler = self._model.vision_model.head(
                        outputs.last_hidden_state
                    )
        except ExtractorValidationError:
            raise
        except Exception as exc:
            raise ExtractorValidationError(
                "siglip2_runtime.forward: model forward failed"
            ) from exc
        if not torch.equal(rng_before, torch.get_rng_state()):
            _fail("siglip2_runtime.forward.rng", "PyTorch CPU RNG state changed")
        if self.model_state_sha256() != self._initial_state_sha256:
            _fail("siglip2_runtime.forward.model_state", "model state changed")
        if self._model.training:
            _fail("siglip2_runtime.model.training", "model left evaluation mode")

        def to_numpy(tensor: Any) -> np.ndarray:
            if str(tensor.device) != self._device:
                _fail("siglip2_runtime.forward.device", "output left the selected device")
            if tensor.dtype != torch.float32:
                _fail("siglip2_runtime.forward.dtype", "expected float32 output")
            return np.array(tensor.detach().cpu().numpy(), copy=True, order="C")

        return _RuntimeForwardResult(
            hidden_states=tuple(to_numpy(value) for value in hidden_states),
            last_hidden_state=to_numpy(outputs.last_hidden_state),
            final_normalized=to_numpy(final_normalized),
            pooler_output=to_numpy(outputs.pooler_output),
            recomputed_pooler=to_numpy(recomputed_pooler),
        )


def _load_verified_siglip2_runtime(
    config: SigLIP2ExtractorConfig,
) -> _SigLIP2Runtime:
    _verify_siglip2_snapshot(config.snapshot_path)
    selected_profile = _selected_runtime_profile(config)
    selected_device = "cpu" if selected_profile is None else selected_profile.device
    runtime = _import_siglip2_runtime()
    if selected_profile is None:
        _validate_runtime_identity(runtime)
    load_kwargs = {
        "revision": config.revision,
        "local_files_only": True,
        "trust_remote_code": False,
        "token": False,
    }
    try:
        full_config = runtime.siglip_config.from_pretrained(
            config.snapshot_path,
            **load_kwargs,
        )
        processor = runtime.siglip_image_processor.from_pretrained(
            config.snapshot_path,
            **load_kwargs,
        )
    except Exception as exc:
        raise ExtractorValidationError(
            "siglip2_runtime.configuration: unable to load exact local snapshot"
        ) from exc
    vision_config, normalization = _validate_config_and_processor(
        full_config, processor
    )

    torch = runtime.torch
    rng_before = torch.get_rng_state().clone()
    try:
        with torch.random.fork_rng(devices=[], enabled=True):
            model, loading_info = runtime.siglip_vision_model.from_pretrained(
                config.snapshot_path,
                config=vision_config,
                revision=config.revision,
                local_files_only=True,
                trust_remote_code=False,
                token=False,
                use_safetensors=True,
                output_loading_info=True,
                torch_dtype=torch.float32,
            )
    except Exception as exc:
        raise ExtractorValidationError(
            "siglip2_runtime.model: unable to load exact local snapshot"
        ) from exc
    if not torch.equal(rng_before, torch.get_rng_state()):
        _fail("siglip2_runtime.model.rng", "PyTorch CPU RNG state changed")

    model.to(device=selected_device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    if model.__class__.__name__ != "SiglipVisionModel":
        _fail("siglip2_runtime.model.class", "expected 'SiglipVisionModel'")
    try:
        vision_model = model.vision_model
        components = (
            vision_model.embeddings,
            vision_model.embeddings.patch_embedding,
            vision_model.embeddings.position_embedding,
            vision_model.encoder,
            vision_model.encoder.layers,
            vision_model.post_layernorm,
            vision_model.head,
        )
    except AttributeError as exc:
        raise ExtractorValidationError(
            "siglip2_runtime.model.components: required vision component is missing"
        ) from exc
    if any(component is None for component in components):
        _fail("siglip2_runtime.model.components", "required vision component is missing")
    if len(vision_model.encoder.layers) != _LAYER_COUNT:
        _fail("siglip2_runtime.model.encoder.layers", "expected 12 layers")

    if not isinstance(loading_info, dict):
        _fail("siglip2_runtime.loading_info", "expected a mapping")
    missing = list(loading_info.get("missing_keys", ()))
    mismatched = list(loading_info.get("mismatched_keys", ()))
    errors = list(loading_info.get("error_msgs", ()))
    unexpected = list(loading_info.get("unexpected_keys", ()))
    unallowed = [
        key
        for key in unexpected
        if key not in _ALLOWED_UNEXPECTED_KEYS
        and not key.startswith(_ALLOWED_UNEXPECTED_PREFIXES)
    ]
    if missing:
        _fail("siglip2_runtime.loading_info.missing_keys", f"found {len(missing)}")
    if mismatched:
        _fail("siglip2_runtime.loading_info.mismatched_keys", f"found {len(mismatched)}")
    if errors:
        _fail("siglip2_runtime.loading_info.errors", f"found {len(errors)}")
    if unallowed:
        _fail(
            "siglip2_runtime.loading_info.unexpected_keys",
            f"found {len(unallowed)} unallowed keys",
        )

    parameters = tuple(model.parameters())
    if sum(parameter.numel() for parameter in parameters) != _PARAMETER_COUNT:
        _fail("siglip2_runtime.model.parameter_count", f"expected {_PARAMETER_COUNT}")
    if len(parameters) != _PARAMETER_TENSOR_COUNT:
        _fail(
            "siglip2_runtime.model.parameter_tensor_count",
            f"expected {_PARAMETER_TENSOR_COUNT}",
        )
    if any(str(parameter.device) != selected_device for parameter in parameters):
        _fail("siglip2_runtime.model.device", "parameters left the selected device")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        _fail("siglip2_runtime.model.dtype", "expected float32")
    if model.training:
        _fail("siglip2_runtime.model.training", "expected evaluation mode")
    if any(parameter.requires_grad for parameter in parameters):
        _fail("siglip2_runtime.model.parameters", "expected all parameters frozen")

    metadata = _RuntimeMetadata(
        model_class="SiglipVisionModel",
        vision_component_path="vision_model",
        input_height=_INPUT_SIZE,
        input_width=_INPUT_SIZE,
        patch_size=_PATCH_SIZE,
        grid_height=_GRID_SIZE,
        grid_width=_GRID_SIZE,
        hidden_dimension=_HIDDEN_DIMENSION,
        layer_count=_LAYER_COUNT,
        parameter_count=_PARAMETER_COUNT,
        parameter_tensor_count=_PARAMETER_TENSOR_COUNT,
        parameter_dtype="float32",
        compute_dtype="float32",
        output_dtype="float32",
        evaluation_mode=True,
        gradients_enabled=False,
        normalization=normalization,
    )
    return _TorchSigLIP2Runtime(model, torch, metadata, device=selected_device)


def _validate_runtime_metadata(metadata: object) -> _RuntimeMetadata:
    if not isinstance(metadata, _RuntimeMetadata):
        _fail("siglip2_runtime.metadata", "expected verified runtime metadata")
    expected = {
        "model_class": "SiglipVisionModel",
        "vision_component_path": "vision_model",
        "input_height": _INPUT_SIZE,
        "input_width": _INPUT_SIZE,
        "patch_size": _PATCH_SIZE,
        "grid_height": _GRID_SIZE,
        "grid_width": _GRID_SIZE,
        "hidden_dimension": _HIDDEN_DIMENSION,
        "layer_count": _LAYER_COUNT,
        "parameter_count": _PARAMETER_COUNT,
        "parameter_tensor_count": _PARAMETER_TENSOR_COUNT,
        "parameter_dtype": "float32",
        "compute_dtype": "float32",
        "output_dtype": "float32",
        "evaluation_mode": True,
        "gradients_enabled": False,
    }
    for name, value in expected.items():
        if getattr(metadata, name) != value:
            _fail(f"siglip2_runtime.metadata.{name}", f"expected {value!r}")
    if metadata.normalization != NumericNormalizationSpec(
        _PROCESSOR_SCALE,
        _PROCESSOR_MEAN,
        _PROCESSOR_STD,
        "RGB",
    ):
        _fail(
            "siglip2_runtime.metadata.normalization",
            "does not match audited SigLIP2 normalization",
        )
    return metadata


def _build_descriptor(
    config: SigLIP2ExtractorConfig, metadata: _RuntimeMetadata
) -> ExtractorDescriptor:
    token_layout = TokenLayout(
        grid_height=metadata.grid_height,
        grid_width=metadata.grid_width,
        total_token_count=_TOKEN_COUNT,
        patch_token_indices=tuple(range(_TOKEN_COUNT)),
        special_token_indices=(),
        special_token_policy=_SPECIAL_TOKEN_POLICY,
    )
    return ExtractorDescriptor(
        model_key=config.model_key,
        model_id=config.model_id,
        revision=config.revision,
        family="siglip2",
        implementation_class="SigLIP2VisionEncoderExtractor",
        vision_component_path=metadata.vision_component_path,
        input_height=metadata.input_height,
        input_width=metadata.input_width,
        grid_height=metadata.grid_height,
        grid_width=metadata.grid_width,
        hidden_dimension=metadata.hidden_dimension,
        layer_count=metadata.layer_count,
        parameter_dtype=metadata.parameter_dtype,
        compute_dtype=metadata.compute_dtype,
        output_dtype=metadata.output_dtype,
        evaluation_mode=metadata.evaluation_mode,
        gradients_enabled=metadata.gradients_enabled,
        normalization=metadata.normalization,
        token_layout=token_layout,
        taps=build_vit_tap_specs(metadata.layer_count),
    )


class SigLIP2VisionEncoderExtractor:
    """Frozen CPU FP32 SigLIP2 extractor for the one audited local checkpoint."""

    __slots__ = ("_config", "_descriptor", "_runtime")

    def __init__(self, config: SigLIP2ExtractorConfig) -> None:
        if not isinstance(config, SigLIP2ExtractorConfig):
            _fail("config", "expected SigLIP2ExtractorConfig")
        with _preserve_python_numpy_rng():
            try:
                runtime = _load_verified_siglip2_runtime(config)
                metadata = _validate_runtime_metadata(runtime.metadata)
                descriptor = _build_descriptor(config, metadata)
            except ExtractorValidationError:
                raise
            except Exception as exc:
                raise ExtractorValidationError(
                    "siglip2_runtime: unable to construct verified SigLIP2 extractor"
                ) from exc
        self._config = config
        self._runtime = runtime
        self._descriptor = descriptor

    @property
    def descriptor(self) -> ExtractorDescriptor:
        return self._descriptor

    @property
    def available_taps(self) -> tuple[TapSpec, ...]:
        return self._descriptor.taps

    def _model_state_sha256(self) -> str:
        return self._runtime.model_state_sha256()

    def _normalize(self, batch: CanonicalImageBatch) -> np.ndarray:
        expected = (
            self._descriptor.input_height,
            self._descriptor.input_width,
            3,
        )
        if batch.rgb_uint8.shape[1:] != expected:
            _fail(
                "batch.rgb_uint8.spatial_dimensions",
                f"expected {expected[:2]}, got {batch.rgb_uint8.shape[1:3]}",
            )
        normalization = self._descriptor.normalization
        rgb = batch.rgb_uint8.astype(np.float32, copy=True)
        rgb *= np.float32(normalization.scale)
        rgb -= np.asarray(normalization.mean, dtype=np.float32)
        rgb /= np.asarray(normalization.standard_deviation, dtype=np.float32)
        normalized = np.ascontiguousarray(rgb.transpose(0, 3, 1, 2), dtype=np.float32)
        if not np.isfinite(normalized).all():
            _fail("siglip2_input.normalized", "contains NaN or Inf")
        return normalized

    def _validate_forward(
        self, result: object, batch_size: int
    ) -> _RuntimeForwardResult:
        if not isinstance(result, _RuntimeForwardResult):
            _fail("siglip2_runtime.forward", "expected a verified forward result")
        if not isinstance(result.hidden_states, tuple):
            _fail("siglip2_runtime.hidden_states", "expected a tuple")
        expected_sequence_shape = (batch_size, _TOKEN_COUNT, _HIDDEN_DIMENSION)
        if len(result.hidden_states) != _LAYER_COUNT + 1:
            _fail("siglip2_runtime.hidden_states", "expected 13 tensors")
        for index, hidden_state in enumerate(result.hidden_states):
            self._validate_array(
                hidden_state,
                f"siglip2_runtime.hidden_states[{index}]",
                expected_sequence_shape,
            )

        arrays = {
            "last_hidden_state": (result.last_hidden_state, expected_sequence_shape),
            "final_normalized": (result.final_normalized, expected_sequence_shape),
            "pooler_output": (
                result.pooler_output,
                (batch_size, _HIDDEN_DIMENSION),
            ),
            "recomputed_pooler": (
                result.recomputed_pooler,
                (batch_size, _HIDDEN_DIMENSION),
            ),
        }
        for name, (array, shape) in arrays.items():
            self._validate_array(array, f"siglip2_runtime.{name}", shape)
        if not np.allclose(
            result.last_hidden_state,
            result.final_normalized,
            rtol=1e-6,
            atol=1e-6,
        ):
            _fail(
                "siglip2_runtime.last_hidden_state",
                "does not match explicit final normalization",
            )
        if not np.allclose(
            result.pooler_output,
            result.recomputed_pooler,
            rtol=1e-6,
            atol=1e-6,
        ):
            _fail(
                "siglip2_runtime.pooler_output",
                "does not match the explicit attention-pooling head",
            )
        return result

    @staticmethod
    def _validate_array(array: object, path: str, shape: tuple[int, ...]) -> None:
        if not isinstance(array, np.ndarray):
            _fail(path, "expected NumPy array")
        if array.shape != shape:
            _fail(f"{path}.shape", f"expected {shape}")
        if array.dtype != np.dtype(np.float32):
            _fail(f"{path}.dtype", "expected float32")
        if not np.isfinite(array).all():
            _fail(path, "contains NaN or Inf")

    def extract(
        self,
        batch: CanonicalImageBatch,
        *,
        tap_names: Sequence[str] | None = None,
    ) -> ExtractionResult:
        with _preserve_python_numpy_rng():
            return self._extract_preserving_input(batch, tap_names=tap_names)

    def _extract_preserving_input(
        self,
        batch: CanonicalImageBatch,
        *,
        tap_names: Sequence[str] | None,
    ) -> ExtractionResult:
        if not isinstance(batch, CanonicalImageBatch):
            _fail("batch", "expected CanonicalImageBatch")
        input_before = batch.rgb_uint8.tobytes(order="C")
        requested = resolve_requested_taps(self.available_taps, tap_names)
        normalized = self._normalize(batch)
        try:
            forward = self._runtime.forward(normalized)
        except ExtractorValidationError:
            raise
        except Exception as exc:
            raise ExtractorValidationError(
                "siglip2_runtime.forward: extraction failed"
            ) from exc
        validated = self._validate_forward(forward, len(batch.image_ids))
        features: list[CanonicalTapFeatures] = []
        for tap in requested:
            if tap.kind == "embedding_output":
                sequence = validated.hidden_states[0]
            elif tap.kind == "block":
                if tap.report_index is None:
                    _fail("tap.report_index", "expected a block report index")
                sequence = validated.hidden_states[tap.report_index]
            else:
                sequence = validated.final_normalized
            raw = np.array(sequence, dtype=np.float32, copy=True, order="C")
            patch = select_patch_tokens(raw, self._descriptor.token_layout)
            if patch.shape != (
                len(batch.image_ids),
                _PATCH_COUNT,
                _HIDDEN_DIMENSION,
            ):
                _fail("siglip2_output.patch.shape", "unexpected canonical patch shape")
            if not np.isfinite(patch).all():
                _fail("siglip2_output.patch", "contains NaN or Inf")
            features.append(CanonicalTapFeatures.from_patch(tap, patch))
        if batch.rgb_uint8.tobytes(order="C") != input_before:
            _fail("batch.rgb_uint8", "input was mutated during extraction")
        return ExtractionResult(
            descriptor=self._descriptor,
            image_ids=batch.image_ids,
            tap_features=tuple(features),
        )


__all__ = ["SigLIP2ExtractorConfig", "SigLIP2VisionEncoderExtractor"]
