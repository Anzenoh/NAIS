"""Offline, exact-revision CLIP ViT feature extraction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
from collections.abc import Sequence
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


_MODEL_KEY = "clip_vit_b16"
_MODEL_ID = "openai/clip-vit-base-patch16"
_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
_INPUT_SIZE = 224
_PATCH_SIZE = 16
_GRID_SIZE = 14
_PATCH_COUNT = 196
_TOKEN_COUNT = 197
_HIDDEN_DIMENSION = 768
_LAYER_COUNT = 12
_PARAMETER_COUNT = 85_799_424
_PARAMETER_TENSOR_COUNT = 199
_SPECIAL_TOKEN_POLICY = "clip_cls_first_verified_runtime_v1"
_PROCESSOR_SCALE = 1.0 / 255.0
_PROCESSOR_MEAN = (0.48145466, 0.4578275, 0.40821073)
_PROCESSOR_STD = (0.26862954, 0.26130258, 0.27577711)
_ALLOWED_UNEXPECTED_PREFIXES = (
    "text_model.",
    "text_projection",
    "visual_projection",
    "logit_scale",
)
_DEPENDENCY_ERROR = (
    "CLIP runtime dependencies are unavailable; install the project clip extra"
)


@dataclass(frozen=True, slots=True)
class _SnapshotFileSpec:
    size: int
    sha256: str
    json_document: bool = False


_EXPECTED_SNAPSHOT_FILES = {
    "config.json": _SnapshotFileSpec(
        4_104,
        "eaf1c9089a8553c913d27ea66407f8bfc2be9989c80c9f331ddb3d63d4c5e8ad",
        True,
    ),
    "preprocessor_config.json": _SnapshotFileSpec(
        316,
        "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
        True,
    ),
    "pytorch_model.bin": _SnapshotFileSpec(
        598_641_023,
        "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0",
    ),
}

_EXPECTED_SOURCE_HASHES = {
    "configuration_clip": (
        "bcc1473638df2e2a9e253b50ea337e981f2d3905364258b1c97e55197668e766"
    ),
    "image_processing_clip": (
        "ce628cc71feab61ad618904e3bd69e56f55a07bb3216c74d2c7c99631a342d00"
    ),
    "modeling_clip": (
        "c0d51bdc43657b74a5b6032cde7b6535fd1b6a4b6a7a89e6a33f7356b337e633"
    ),
}


def _fail(path: str, message: str) -> None:
    raise ExtractorValidationError(f"{path}: {message}")


@dataclass(frozen=True, slots=True, repr=False)
class CLIPExtractorConfig:
    """Identity and exact local snapshot used by the audited CLIP extractor."""

    snapshot_path: Path
    model_key: str = _MODEL_KEY
    model_id: str = _MODEL_ID
    revision: str = _REVISION
    runtime_profile_path: Path | None = None
    execution_selector: str = "legacy_windows_cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_path, (str, Path)):
            _fail("clip_config.snapshot_path", "expected a string or Path")
        try:
            snapshot = Path(self.snapshot_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ExtractorValidationError(
                "clip_config.snapshot_path: directory does not exist"
            ) from exc
        if not snapshot.is_dir():
            _fail("clip_config.snapshot_path", "expected an existing directory")
        if snapshot.name != _REVISION:
            _fail(
                "clip_config.snapshot_path",
                "directory basename must equal the audited revision",
            )
        if self.model_key != _MODEL_KEY:
            _fail("clip_config.model_key", f"expected {_MODEL_KEY!r}")
        if self.model_id != _MODEL_ID:
            _fail("clip_config.model_id", f"expected {_MODEL_ID!r}")
        if self.revision != _REVISION:
            _fail("clip_config.revision", f"expected audited commit {_REVISION!r}")
        if self.execution_selector not in {"legacy_windows_cpu", MAIN_SELECTOR}:
            _fail("clip_config.execution_selector", "unknown explicit execution selector")
        runtime_profile: Path | None = None
        if self.execution_selector == MAIN_SELECTOR:
            if not isinstance(self.runtime_profile_path, (str, Path)):
                _fail(
                    "clip_config.runtime_profile_path",
                    "Apple selector requires an explicit runtime profile",
                )
            runtime_profile = Path(self.runtime_profile_path).resolve(strict=True)
            expected_runtime = (
                Path(__file__).resolve().parents[3]
                / "configs/runtime_profiles/p1_apple_main_cuda_fp32_v0.json"
            ).resolve(strict=True)
            if runtime_profile != expected_runtime or not runtime_profile.is_file():
                _fail(
                    "clip_config.runtime_profile_path",
                    "expected the repository Apple main runtime profile",
                )
        elif self.runtime_profile_path is not None:
            _fail(
                "clip_config.runtime_profile_path",
                "legacy selector forbids an Apple runtime profile",
            )
        object.__setattr__(self, "snapshot_path", snapshot)
        object.__setattr__(self, "runtime_profile_path", runtime_profile)

    def __repr__(self) -> str:
        return (
            "CLIPExtractorConfig("
            "snapshot_path=<redacted-local-snapshot>, "
            f"model_key={self.model_key!r}, model_id={self.model_id!r}, "
            f"revision={self.revision!r})"
        )


def _selected_runtime_profile(
    config: CLIPExtractorConfig,
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
            f"clip_apple_runtime.{exc.classification}: runtime contract failed"
        ) from exc
    if profile.selector != MAIN_SELECTOR:
        _fail("clip_apple_runtime.selector", "not the explicit Apple main selector")
    if not profile.package_metadata_resolved:
        _fail(
            "clip_apple_runtime.packages",
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


class _CLIPRuntime(Protocol):
    metadata: _RuntimeMetadata

    def forward(self, normalized_nchw_float32: np.ndarray) -> _RuntimeForwardResult: ...

    def model_state_sha256(self) -> str: ...


@dataclass(frozen=True, slots=True)
class _RuntimeImports:
    torch: Any
    clip_vision_config: Any
    clip_image_processor: Any
    clip_vision_model: Any
    configuration_module: Any
    image_processing_module: Any
    modeling_module: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_clip_snapshot(snapshot_path: Path) -> _VerifiedSnapshot:
    if snapshot_path.name != _REVISION:
        _fail("clip_snapshot", "directory basename does not match audited revision")
    try:
        actual_files = {
            item.name for item in snapshot_path.iterdir() if item.is_file()
        }
    except OSError as exc:
        raise ExtractorValidationError(
            "clip_snapshot: unable to inspect snapshot directory"
        ) from exc
    expected_files = set(_EXPECTED_SNAPSHOT_FILES)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        details = []
        if missing:
            details.append(f"missing {missing!r}")
        if unexpected:
            details.append(f"unexpected {unexpected!r}")
        _fail("clip_snapshot.files", "; ".join(details))

    verified: list[tuple[str, str]] = []
    for name, expected in _EXPECTED_SNAPSHOT_FILES.items():
        path = snapshot_path / name
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ExtractorValidationError(
                f"clip_snapshot.{name}: unable to inspect file"
            ) from exc
        if size == 0:
            _fail(f"clip_snapshot.{name}", "zero-byte files are not allowed")
        if size != expected.size:
            _fail(
                f"clip_snapshot.{name}.size",
                f"expected {expected.size}, found {size}",
            )
        try:
            digest = _sha256_file(path)
        except OSError as exc:
            raise ExtractorValidationError(
                f"clip_snapshot.{name}: unable to hash file"
            ) from exc
        if digest != expected.sha256:
            _fail(f"clip_snapshot.{name}.sha256", "does not match audited digest")
        if expected.json_document:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ExtractorValidationError(
                    f"clip_snapshot.{name}: expected valid UTF-8 JSON"
                ) from exc
            if not isinstance(document, dict):
                _fail(f"clip_snapshot.{name}", "expected a JSON object")
        verified.append((name, digest))
    return _VerifiedSnapshot(tuple(verified))


def _import_clip_runtime() -> _RuntimeImports:
    try:
        import PIL  # noqa: F401
        import huggingface_hub  # noqa: F401
        import torch
        from transformers import (
            CLIPImageProcessor,
            CLIPVisionConfig,
            CLIPVisionModel,
        )
        from transformers.models.clip import (
            configuration_clip,
            image_processing_clip,
            modeling_clip,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise ExtractorValidationError(_DEPENDENCY_ERROR) from exc
    return _RuntimeImports(
        torch=torch,
        clip_vision_config=CLIPVisionConfig,
        clip_image_processor=CLIPImageProcessor,
        clip_vision_model=CLIPVisionModel,
        configuration_module=configuration_clip,
        image_processing_module=image_processing_clip,
        modeling_module=modeling_clip,
    )


def _validate_runtime_identity(runtime: _RuntimeImports) -> None:
    expected_versions = {
        "torch": "2.3.0",
        "transformers": "4.40.1",
        "Pillow": "10.3.0",
        "huggingface-hub": "0.22.2",
    }
    for package, expected in expected_versions.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ExtractorValidationError(_DEPENDENCY_ERROR) from exc
        if actual != expected:
            _fail(
                f"clip_runtime.{package}.version",
                f"expected {expected!r}, found {actual!r}",
            )
    torch = runtime.torch
    if torch.__version__ != "2.3.0+cpu":
        _fail(
            "clip_runtime.torch.runtime",
            f"expected '2.3.0+cpu', found {torch.__version__!r}",
        )
    if (
        torch.version.cuda is not None
        or torch.cuda.is_available()
        or torch.cuda.device_count() != 0
    ):
        _fail("clip_runtime.device", "expected the audited CPU-only runtime")

    modules = {
        "configuration_clip": runtime.configuration_module,
        "image_processing_clip": runtime.image_processing_module,
        "modeling_clip": runtime.modeling_module,
    }
    for name, module in modules.items():
        source_path = Path(inspect.getfile(module))
        if _sha256_file(source_path) != _EXPECTED_SOURCE_HASHES[name]:
            _fail(f"clip_runtime.source.{name}", "source hash mismatch")


def _validate_config_and_processor(config: Any, processor: Any) -> NumericNormalizationSpec:
    expected_config = {
        "model_type": "clip_vision_model",
        "image_size": _INPUT_SIZE,
        "patch_size": _PATCH_SIZE,
        "hidden_size": _HIDDEN_DIMENSION,
        "num_hidden_layers": _LAYER_COUNT,
    }
    if config.__class__.__name__ != "CLIPVisionConfig":
        _fail("clip_runtime.config.class", "expected 'CLIPVisionConfig'")
    for name, expected in expected_config.items():
        if getattr(config, name, None) != expected:
            _fail(f"clip_runtime.config.{name}", f"expected {expected!r}")
    if _INPUT_SIZE // _PATCH_SIZE != _GRID_SIZE:
        _fail("clip_runtime.config.grid", "audited geometry is inconsistent")

    if processor.__class__.__name__ != "CLIPImageProcessor":
        _fail("clip_runtime.processor.class", "expected 'CLIPImageProcessor'")
    for name in ("do_rescale", "do_normalize", "do_convert_rgb"):
        if getattr(processor, name, None) is not True:
            _fail(f"clip_runtime.processor.{name}", "expected True")
    scale = float(getattr(processor, "rescale_factor", math.nan))
    try:
        mean = tuple(float(value) for value in processor.image_mean)
        std = tuple(float(value) for value in processor.image_std)
    except (TypeError, ValueError) as exc:
        raise ExtractorValidationError(
            "clip_runtime.processor.normalization: expected numeric channel metadata"
        ) from exc
    values = np.asarray((scale, *mean, *std), dtype=np.float64)
    if not np.isfinite(values).all():
        _fail("clip_runtime.processor.normalization", "contains NaN or Inf")
    if scale != _PROCESSOR_SCALE or mean != _PROCESSOR_MEAN or std != _PROCESSOR_STD:
        _fail("clip_runtime.processor.normalization", "audited values do not match")
    if getattr(processor, "size", None) != {"shortest_edge": _INPUT_SIZE}:
        _fail("clip_runtime.processor.size", "expected shortest edge 224")
    if getattr(processor, "crop_size", None) != {
        "height": _INPUT_SIZE,
        "width": _INPUT_SIZE,
    }:
        _fail("clip_runtime.processor.crop_size", "expected 224 x 224")
    return NumericNormalizationSpec(scale, mean, std, "RGB")


def _model_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(memoryview(tensor.numpy()))
    return digest.hexdigest()


class _TorchCLIPRuntime:
    __slots__ = ("_device", "_model", "_torch", "metadata")

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

    def model_state_sha256(self) -> str:
        return _model_state_sha256(self._model)

    def forward(self, normalized_nchw_float32: np.ndarray) -> _RuntimeForwardResult:
        torch = self._torch
        if self._model.training:
            _fail("clip_runtime.model.training", "expected evaluation mode")
        if any(parameter.requires_grad for parameter in self._model.parameters()):
            _fail("clip_runtime.model.parameters", "expected all parameters frozen")
        state_before = self.model_state_sha256()
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
        except ExtractorValidationError:
            raise
        except Exception as exc:
            raise ExtractorValidationError(
                "clip_runtime.forward: model forward failed"
            ) from exc
        if not torch.equal(rng_before, torch.get_rng_state()):
            _fail("clip_runtime.forward.rng", "PyTorch CPU RNG state changed")
        if self.model_state_sha256() != state_before:
            _fail("clip_runtime.forward.model_state", "model state changed")
        if self._model.training:
            _fail("clip_runtime.model.training", "model left evaluation mode")

        def to_numpy(tensor: Any) -> np.ndarray:
            if str(tensor.device) != self._device:
                _fail("clip_runtime.forward.device", "output left the selected device")
            if tensor.dtype != torch.float32:
                _fail("clip_runtime.forward.dtype", "expected float32 output")
            return np.array(
                tensor.detach().cpu().numpy(),
                copy=True,
                order="C",
            )

        return _RuntimeForwardResult(
            hidden_states=tuple(to_numpy(value) for value in hidden_states),
            last_hidden_state=to_numpy(outputs.last_hidden_state),
            final_normalized=to_numpy(final_normalized),
            pooler_output=to_numpy(outputs.pooler_output),
        )


def _load_verified_clip_runtime(config: CLIPExtractorConfig) -> _CLIPRuntime:
    _verify_clip_snapshot(config.snapshot_path)
    selected_profile = _selected_runtime_profile(config)
    selected_device = "cpu" if selected_profile is None else selected_profile.device
    runtime = _import_clip_runtime()
    if selected_profile is None:
        _validate_runtime_identity(runtime)
    load_kwargs = {
        "revision": config.revision,
        "local_files_only": True,
        "trust_remote_code": False,
        "token": False,
    }
    try:
        model_config = runtime.clip_vision_config.from_pretrained(
            config.snapshot_path,
            **load_kwargs,
        )
        processor = runtime.clip_image_processor.from_pretrained(
            config.snapshot_path,
            **load_kwargs,
        )
    except Exception as exc:
        raise ExtractorValidationError(
            "clip_runtime.configuration: unable to load exact local snapshot"
        ) from exc
    normalization = _validate_config_and_processor(model_config, processor)
    torch = runtime.torch
    try:
        with torch.random.fork_rng(devices=[], enabled=True):
            model, loading_info = runtime.clip_vision_model.from_pretrained(
                config.snapshot_path,
                config=model_config,
                revision=config.revision,
                local_files_only=True,
                trust_remote_code=False,
                token=False,
                use_safetensors=False,
                output_loading_info=True,
                torch_dtype=torch.float32,
            )
    except Exception as exc:
        raise ExtractorValidationError(
            "clip_runtime.model: unable to load exact local snapshot"
        ) from exc
    model.to(device=selected_device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    if model.__class__.__name__ != "CLIPVisionModel":
        _fail("clip_runtime.model.class", "expected 'CLIPVisionModel'")
    try:
        vision_model = model.vision_model
        components = (
            vision_model.embeddings,
            vision_model.embeddings.patch_embedding,
            vision_model.embeddings.class_embedding,
            vision_model.embeddings.position_embedding,
            vision_model.pre_layrnorm,
            vision_model.encoder,
            vision_model.encoder.layers,
            vision_model.post_layernorm,
        )
    except AttributeError as exc:
        raise ExtractorValidationError(
            "clip_runtime.model.components: required vision component is missing"
        ) from exc
    if any(component is None for component in components):
        _fail("clip_runtime.model.components", "required vision component is missing")
    if len(vision_model.encoder.layers) != _LAYER_COUNT:
        _fail("clip_runtime.model.encoder.layers", "expected 12 layers")

    missing = list(loading_info.get("missing_keys", ()))
    mismatched = list(loading_info.get("mismatched_keys", ()))
    errors = list(loading_info.get("error_msgs", ()))
    unexpected = list(loading_info.get("unexpected_keys", ()))
    unallowed = [
        key
        for key in unexpected
        if not key.startswith(_ALLOWED_UNEXPECTED_PREFIXES)
    ]
    if missing:
        _fail("clip_runtime.loading_info.missing_keys", f"found {len(missing)}")
    if mismatched:
        _fail("clip_runtime.loading_info.mismatched_keys", f"found {len(mismatched)}")
    if errors:
        _fail("clip_runtime.loading_info.errors", f"found {len(errors)}")
    if unallowed:
        _fail(
            "clip_runtime.loading_info.unexpected_keys",
            f"found {len(unallowed)} unallowed keys",
        )

    parameters = tuple(model.parameters())
    if sum(parameter.numel() for parameter in parameters) != _PARAMETER_COUNT:
        _fail("clip_runtime.model.parameter_count", f"expected {_PARAMETER_COUNT}")
    if len(parameters) != _PARAMETER_TENSOR_COUNT:
        _fail(
            "clip_runtime.model.parameter_tensor_count",
            f"expected {_PARAMETER_TENSOR_COUNT}",
        )
    if any(str(parameter.device) != selected_device for parameter in parameters):
        _fail("clip_runtime.model.device", "parameters left the selected device")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        _fail("clip_runtime.model.dtype", "expected float32")
    if model.training:
        _fail("clip_runtime.model.training", "expected evaluation mode")
    if any(parameter.requires_grad for parameter in parameters):
        _fail("clip_runtime.model.parameters", "expected all parameters frozen")

    metadata = _RuntimeMetadata(
        model_class="CLIPVisionModel",
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
    return _TorchCLIPRuntime(model, torch, metadata, device=selected_device)


def _validate_runtime_metadata(metadata: object) -> _RuntimeMetadata:
    if not isinstance(metadata, _RuntimeMetadata):
        _fail("clip_runtime.metadata", "expected verified runtime metadata")
    expected = {
        "model_class": "CLIPVisionModel",
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
            _fail(f"clip_runtime.metadata.{name}", f"expected {value!r}")
    if not isinstance(metadata.normalization, NumericNormalizationSpec):
        _fail(
            "clip_runtime.metadata.normalization",
            "expected NumericNormalizationSpec",
        )
    return metadata


def _build_descriptor(
    config: CLIPExtractorConfig, metadata: _RuntimeMetadata
) -> ExtractorDescriptor:
    token_layout = TokenLayout(
        grid_height=metadata.grid_height,
        grid_width=metadata.grid_width,
        total_token_count=_TOKEN_COUNT,
        patch_token_indices=tuple(range(1, _TOKEN_COUNT)),
        special_token_indices=(0,),
        special_token_policy=_SPECIAL_TOKEN_POLICY,
    )
    return ExtractorDescriptor(
        model_key=config.model_key,
        model_id=config.model_id,
        revision=config.revision,
        family="clip",
        implementation_class="CLIPVisionEncoderExtractor",
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


class CLIPVisionEncoderExtractor:
    """Frozen CPU FP32 CLIP extractor for the one audited local checkpoint."""

    __slots__ = ("_config", "_descriptor", "_runtime")

    def __init__(self, config: CLIPExtractorConfig) -> None:
        if not isinstance(config, CLIPExtractorConfig):
            _fail("config", "expected CLIPExtractorConfig")
        try:
            runtime = _load_verified_clip_runtime(config)
            metadata = _validate_runtime_metadata(runtime.metadata)
            descriptor = _build_descriptor(config, metadata)
        except ExtractorValidationError:
            raise
        except Exception as exc:
            raise ExtractorValidationError(
                "clip_runtime: unable to construct verified CLIP extractor"
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
            _fail("clip_input.normalized", "contains NaN or Inf")
        return normalized

    def _validate_forward(
        self, result: object, batch_size: int
    ) -> _RuntimeForwardResult:
        if not isinstance(result, _RuntimeForwardResult):
            _fail("clip_runtime.forward", "expected a verified forward result")
        expected_sequence_shape = (batch_size, _TOKEN_COUNT, _HIDDEN_DIMENSION)
        if len(result.hidden_states) != _LAYER_COUNT + 1:
            _fail("clip_runtime.hidden_states", "expected 13 tensors")
        for index, hidden_state in enumerate(result.hidden_states):
            if not isinstance(hidden_state, np.ndarray):
                _fail(f"clip_runtime.hidden_states[{index}]", "expected NumPy array")
            if hidden_state.shape != expected_sequence_shape:
                _fail(
                    f"clip_runtime.hidden_states[{index}].shape",
                    f"expected {expected_sequence_shape}",
                )
            if hidden_state.dtype != np.dtype(np.float32):
                _fail(
                    f"clip_runtime.hidden_states[{index}].dtype",
                    "expected float32",
                )
            if not np.isfinite(hidden_state).all():
                _fail(f"clip_runtime.hidden_states[{index}]", "contains NaN or Inf")

        arrays = {
            "last_hidden_state": (result.last_hidden_state, expected_sequence_shape),
            "final_normalized": (result.final_normalized, expected_sequence_shape),
            "pooler_output": (
                result.pooler_output,
                (batch_size, _HIDDEN_DIMENSION),
            ),
        }
        for name, (array, shape) in arrays.items():
            if not isinstance(array, np.ndarray):
                _fail(f"clip_runtime.{name}", "expected NumPy array")
            if array.shape != shape:
                _fail(f"clip_runtime.{name}.shape", f"expected {shape}")
            if array.dtype != np.dtype(np.float32):
                _fail(f"clip_runtime.{name}.dtype", "expected float32")
            if not np.isfinite(array).all():
                _fail(f"clip_runtime.{name}", "contains NaN or Inf")
        if not np.array_equal(result.last_hidden_state, result.hidden_states[-1]):
            _fail(
                "clip_runtime.last_hidden_state",
                "does not exactly match hidden_states[-1]",
            )
        if not np.allclose(
            result.final_normalized[:, 0, :],
            result.pooler_output,
            rtol=1e-6,
            atol=1e-6,
        ):
            _fail(
                "clip_runtime.final_normalized",
                "class row does not match pooler_output",
            )
        return result

    def extract(
        self,
        batch: CanonicalImageBatch,
        *,
        tap_names: Sequence[str] | None = None,
    ) -> ExtractionResult:
        if not isinstance(batch, CanonicalImageBatch):
            _fail("batch", "expected CanonicalImageBatch")
        requested = resolve_requested_taps(self.available_taps, tap_names)
        normalized = self._normalize(batch)
        try:
            forward = self._runtime.forward(normalized)
        except ExtractorValidationError:
            raise
        except Exception as exc:
            raise ExtractorValidationError(
                "clip_runtime.forward: extraction failed"
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
                _fail("clip_output.patch.shape", "unexpected canonical patch shape")
            if not np.isfinite(patch).all():
                _fail("clip_output.patch", "contains NaN or Inf")
            features.append(CanonicalTapFeatures.from_patch(tap, patch))
        return ExtractionResult(
            descriptor=self._descriptor,
            image_ids=batch.image_ids,
            tap_features=tuple(features),
        )


__all__ = ["CLIPExtractorConfig", "CLIPVisionEncoderExtractor"]
