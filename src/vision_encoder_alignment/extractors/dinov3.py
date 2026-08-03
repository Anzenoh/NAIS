"""Exact DINOv3 extraction through the persistent isolated CPU runtime."""

from __future__ import annotations

import hashlib
import math
import os
import random
import stat
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..dinov3_extraction_boundary import (
    DINOv3ExtractionBoundaryError,
    DINOv3ExtractionRequest,
    DINOv3ExtractionResponse,
    DINOv3ExtractorProfile,
    MODEL_ID,
    MODEL_KEY,
    REVISION,
    invoke_dinov3_extraction_worker,
    LEGACY_EXECUTION_SELECTOR,
    load_dinov3_extractor_profile,
    resolve_workspace_relative_path,
    sha256_file,
    validate_dinov3_extraction_request,
)
from ..dinov3_production_identity import (
    DINOv3ProductionIdentityError,
    validate_production_sources,
)
from ..feature_artifacts import read_feature_artifact
from ..runtime_boundary import (
    RuntimeBoundaryError,
    load_runtime_profile,
    resolve_default_runtime_root,
    runtime_python_path,
    validate_runtime_root,
)
from ..production_runtime_contract import (
    DINOV3_SELECTOR,
    ProductionRuntimeContractError,
    load_production_runtime_profile,
    validate_profile_source_bindings,
)
from ..snapshot_boundary import (
    SnapshotBoundaryError,
    load_snapshot_profile,
    resolve_default_snapshot_root,
    validate_apple_hf_snapshot,
    validate_snapshot_root,
)
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
)


_PROFILE_RELATIVE_PATH = Path(
    "configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json"
)
_APPLE_PROFILE_RELATIVE_PATH = Path(
    "configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json"
)
_APPLE_RUNTIME_PROFILE_RELATIVE_PATH = Path(
    "configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json"
)
_WORKER_RELATIVE_PATH = Path("scripts/dinov3_extraction_worker.py")
_SEMANTIC_EVIDENCE_SHA256 = (
    "8308c0ca73a3cb31fcfcc1333abe4ad904ea3e96bb45e83662281630b53e0311"
)
_SPECIAL_TOKEN_POLICY = "dinov3_cls1_register4_prefix_verified_runtime_v1"
_MODEL_STATE_SHA256 = (
    "c382afd66ac658255f9f1406bad1e5e5fcf410af40cbc5c306b770792a4f30fd"
)
_EXPECTED_WORKER_POLICY = {
    "cpu_only": True,
    "fp32_only": True,
    "local_files_only": True,
    "network_allowed": False,
    "processor_allowed": False,
    "torchvision_allowed": False,
}
_APPLE_EXPECTED_WORKER_POLICY = {
    "cpu_fallback": False,
    "cpu_only": False,
    "cuda_required": True,
    "device": "cuda:0",
    "dtype": "float32",
    "fp32_only": True,
    "local_files_only": True,
    "mixed_precision": False,
    "network_allowed": False,
    "processor_allowed": False,
    "token_allowed": False,
    "torchvision_allowed": False,
}


def _fail(path: str, message: str) -> None:
    raise ExtractorValidationError(f"{path}: {message}")


def _is_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return bool(flag and attributes & flag)
    except OSError as exc:
        raise ExtractorValidationError("dinov3_config.path: unable to inspect path") from exc


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_root(value: str | Path, field: str, repository: Path) -> Path:
    if not isinstance(value, (str, Path)):
        _fail(field, "expected a string or Path")
    candidate = Path(value)
    if not candidate.is_absolute() or _is_reparse(candidate):
        _fail(field, "expected an absolute non-link directory")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExtractorValidationError(f"{field}: directory does not exist") from exc
    if not resolved.is_dir():
        _fail(field, "expected an existing directory")
    if _contains(repository, resolved) or _contains(resolved, repository):
        _fail(field, "must not overlap the repository")
    return resolved


@dataclass(frozen=True, slots=True, repr=False)
class DINOv3ExtractorConfig:
    """Exact profile and external roots for the isolated DINOv3 extractor."""

    profile_path: Path
    runtime_root: Path | None = None
    snapshot_root: Path | None = None
    apple_runtime_profile_path: Path | None = None
    execution_selector: str = LEGACY_EXECUTION_SELECTOR
    timeout_seconds: float = 900.0
    model_key: str = MODEL_KEY
    model_id: str = MODEL_ID
    revision: str = REVISION

    def __post_init__(self) -> None:
        repository = _repository_root()
        if not isinstance(self.profile_path, (str, Path)):
            _fail("dinov3_config.profile_path", "expected a string or Path")
        candidate = Path(self.profile_path)
        if _is_reparse(candidate):
            _fail("dinov3_config.profile_path", "links and reparse points are forbidden")
        try:
            profile = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ExtractorValidationError(
                "dinov3_config.profile_path: file does not exist"
            ) from exc
        if not profile.is_file():
            _fail("dinov3_config.profile_path", "expected an existing regular file")
        if self.execution_selector == LEGACY_EXECUTION_SELECTOR:
            expected = (repository / _PROFILE_RELATIVE_PATH).resolve(strict=True)
        elif self.execution_selector == DINOV3_SELECTOR:
            expected = (repository / _APPLE_PROFILE_RELATIVE_PATH).resolve(strict=True)
        else:
            _fail("dinov3_config.execution_selector", "unknown explicit execution selector")
        if profile != expected:
            _fail(
                "dinov3_config.profile_path",
                "expected the repository DINOv3 extractor profile",
            )
        apple_runtime_profile: Path | None = None
        if self.execution_selector == DINOV3_SELECTOR:
            if not isinstance(self.apple_runtime_profile_path, (str, Path)):
                _fail(
                    "dinov3_config.apple_runtime_profile_path",
                    "explicit Apple selector requires its repository runtime profile",
                )
            apple_runtime_profile = Path(self.apple_runtime_profile_path).resolve(strict=True)
            expected_runtime = (repository / _APPLE_RUNTIME_PROFILE_RELATIVE_PATH).resolve(strict=True)
            if apple_runtime_profile != expected_runtime or not apple_runtime_profile.is_file():
                _fail(
                    "dinov3_config.apple_runtime_profile_path",
                    "expected the exact repository Apple runtime profile",
                )
        elif self.apple_runtime_profile_path is not None:
            _fail(
                "dinov3_config.apple_runtime_profile_path",
                "legacy selector forbids an Apple runtime profile",
            )
        if self.model_key != MODEL_KEY:
            _fail("dinov3_config.model_key", f"expected {MODEL_KEY!r}")
        if self.model_id != MODEL_ID:
            _fail("dinov3_config.model_id", f"expected {MODEL_ID!r}")
        if self.revision != REVISION:
            _fail("dinov3_config.revision", f"expected {REVISION!r}")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            _fail("dinov3_config.timeout_seconds", "expected a positive finite value")
        runtime = (
            None
            if self.runtime_root is None
            else _resolve_root(self.runtime_root, "dinov3_config.runtime_root", repository)
        )
        snapshot = (
            None
            if self.snapshot_root is None
            else _resolve_root(self.snapshot_root, "dinov3_config.snapshot_root", repository)
        )
        if runtime is not None and snapshot is not None and (
            _contains(runtime, snapshot) or _contains(snapshot, runtime)
        ):
            _fail("dinov3_config.roots", "runtime and snapshot roots must be disjoint")
        object.__setattr__(self, "profile_path", profile)
        object.__setattr__(self, "runtime_root", runtime)
        object.__setattr__(self, "snapshot_root", snapshot)
        object.__setattr__(self, "apple_runtime_profile_path", apple_runtime_profile)
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

    def __repr__(self) -> str:
        return (
            "DINOv3ExtractorConfig("
            "profile_path=<redacted-repository-profile>, "
            f"runtime_root={'<default>' if self.runtime_root is None else '<redacted-external-runtime>'}, "
            f"snapshot_root={'<default>' if self.snapshot_root is None else '<redacted-external-snapshot>'}, "
            f"execution_selector={self.execution_selector!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, model_key={self.model_key!r}, "
            f"model_id={self.model_id!r}, revision={self.revision!r})"
        )


@dataclass(frozen=True, slots=True)
class _ExtractorContext:
    profile: DINOv3ExtractorProfile
    runtime_root: Path
    snapshot_root: Path
    repository_root: Path
    worker_script: Path
    descriptor: ExtractorDescriptor


def _build_descriptor() -> ExtractorDescriptor:
    return ExtractorDescriptor(
        model_key=MODEL_KEY,
        model_id=MODEL_ID,
        revision=REVISION,
        family="dinov3",
        implementation_class="DINOv3VisionEncoderExtractor",
        vision_component_path="self",
        input_height=224,
        input_width=224,
        grid_height=14,
        grid_width=14,
        hidden_dimension=768,
        layer_count=12,
        parameter_dtype="float32",
        compute_dtype="float32",
        output_dtype="float32",
        evaluation_mode=True,
        gradients_enabled=False,
        normalization=NumericNormalizationSpec(
            scale=float(np.float32(1.0 / 255.0)),
            mean=(0.485, 0.456, 0.406),
            standard_deviation=(0.229, 0.224, 0.225),
            channel_order="RGB",
        ),
        token_layout=TokenLayout(
            grid_height=14,
            grid_width=14,
            total_token_count=201,
            patch_token_indices=tuple(range(5, 201)),
            special_token_indices=(0, 1, 2, 3, 4),
            special_token_policy=_SPECIAL_TOKEN_POLICY,
        ),
        taps=build_vit_tap_specs(12),
    )


def _load_extractor_context(config: DINOv3ExtractorConfig) -> _ExtractorContext:
    repository = _repository_root()
    if config.execution_selector == DINOV3_SELECTOR:
        assert config.apple_runtime_profile_path is not None
        profile = load_dinov3_extractor_profile(
            config.profile_path, execution_selector=DINOV3_SELECTOR
        )
        validate_production_sources(repository, config.profile_path)
        apple_runtime = load_production_runtime_profile(config.apple_runtime_profile_path)
        validate_profile_source_bindings(apple_runtime, repository)
        if not apple_runtime.package_metadata_resolved:
            _fail(
                "dinov3_apple_runtime.packages",
                "exact Apple package metadata remains unresolved",
            )
        if (
            apple_runtime.selector != DINOV3_SELECTOR
            or apple_runtime.interpreter_layout != "linux_bin"
            or apple_runtime.system_site_packages
            or apple_runtime.device != "cuda:0"
            or apple_runtime.dtype != "float32"
            or apple_runtime.cpu_fallback
            or apple_runtime.mixed_precision
            or apple_runtime.network_allowed
            or apple_runtime.ready_for_p1_extraction
        ):
            _fail("dinov3_apple_runtime.contract", "Apple runtime policy mismatch")
        sources = profile.sources
        identities = profile.identities
        source_files = {
            "registry_sha256": repository / "configs/models.yaml",
            "runtime_profile_sha256": repository / str(sources.runtime_profile_path),
            "snapshot_profile_sha256": repository / str(sources.snapshot_profile_path),
            "semantic_profile_sha256": repository / str(sources.semantic_profile_path),
            "semantic_evidence_sha256": repository / str(sources.semantic_evidence_path),
        }
        for key, path in source_files.items():
            if sha256_file(path) != identities[key]:
                _fail(f"dinov3_profile.identities.{key}", "repository source hash mismatch")
        if apple_runtime.source_sha256 != identities.runtime_profile_sha256:
            _fail("dinov3_profile.identities.runtime_profile_sha256", "runtime identity mismatch")
        runtime = (
            Path(apple_runtime.runtime_root)
            if config.runtime_root is None
            else config.runtime_root
        )
        runtime = _resolve_root(runtime, "dinov3_config.runtime_root", repository)
        if runtime.as_posix() != apple_runtime.runtime_root:
            _fail("dinov3_config.runtime_root", "expected exact Apple isolated runtime root")
        runtime_python_path(runtime, interpreter_layout=apple_runtime.interpreter_layout)
        snapshot_profile = load_snapshot_profile(source_files["snapshot_profile_sha256"])
        snapshot = (
            resolve_default_snapshot_root(snapshot_profile)
            if config.snapshot_root is None
            else config.snapshot_root
        )
        snapshot = _resolve_root(snapshot, "dinov3_config.snapshot_root", repository)
        if _contains(runtime, snapshot) or _contains(snapshot, runtime):
            _fail("dinov3_config.roots", "runtime and snapshot roots overlap")
        snapshot_manifest = validate_apple_hf_snapshot(
            snapshot,
            cache_root=snapshot.parents[2],
            model_id=MODEL_ID,
            revision=REVISION,
            files=profile.checkpoint_files,
        )
        if (
            snapshot_profile.source_sha256 != identities.snapshot_profile_sha256
            or snapshot_manifest.source_sha256
            != identities.external_snapshot_manifest_sha256
        ):
            _fail("dinov3_profile.identities.snapshot", "live snapshot identity mismatch")
        worker = (repository / _WORKER_RELATIVE_PATH).resolve(strict=True)
        return _ExtractorContext(
            profile=profile,
            runtime_root=runtime,
            snapshot_root=snapshot,
            repository_root=repository,
            worker_script=worker,
            descriptor=_build_descriptor(),
        )
    profile = load_dinov3_extractor_profile(config.profile_path)
    validate_production_sources(repository, config.profile_path)
    sources = profile.sources
    identities = profile.identities
    source_files = {
        "registry_sha256": repository / "configs/models.yaml",
        "runtime_profile_sha256": repository / str(sources.runtime_profile_path),
        "snapshot_profile_sha256": repository / str(sources.snapshot_profile_path),
        "semantic_profile_sha256": repository / str(sources.semantic_profile_path),
        "semantic_evidence_sha256": repository / str(sources.semantic_evidence_path),
    }
    for key, path in source_files.items():
        if sha256_file(path) != identities[key]:
            _fail(f"dinov3_profile.identities.{key}", "repository source hash mismatch")
    if identities.semantic_evidence_sha256 != _SEMANTIC_EVIDENCE_SHA256:
        _fail("dinov3_profile.identities.semantic_evidence_sha256", "unexpected semantic evidence")
    runtime_profile = load_runtime_profile(source_files["runtime_profile_sha256"])
    snapshot_profile = load_snapshot_profile(source_files["snapshot_profile_sha256"])
    runtime = (
        resolve_default_runtime_root(runtime_profile)
        if config.runtime_root is None
        else config.runtime_root
    )
    snapshot = (
        resolve_default_snapshot_root(snapshot_profile)
        if config.snapshot_root is None
        else config.snapshot_root
    )
    runtime = _resolve_root(runtime, "dinov3_config.runtime_root", repository)
    snapshot = _resolve_root(snapshot, "dinov3_config.snapshot_root", repository)
    if _contains(runtime, snapshot) or _contains(snapshot, runtime):
        _fail("dinov3_config.roots", "runtime and snapshot roots overlap")
    runtime_manifest = validate_runtime_root(runtime, runtime_profile, repository_root=repository)
    snapshot_manifest = validate_snapshot_root(
        snapshot,
        snapshot_profile,
        repository_root=repository,
        persistent_runtime_root=runtime,
    )
    exact = {
        "runtime_profile_sha256": runtime_profile.source_sha256,
        "snapshot_profile_sha256": snapshot_profile.source_sha256,
        "persistent_runtime_manifest_sha256": runtime_manifest.source_sha256,
        "external_snapshot_manifest_sha256": snapshot_manifest.source_sha256,
        "persistent_freeze_sha256": runtime_manifest.candidate_freeze_sha256,
    }
    for key, actual in exact.items():
        if identities[key] != actual:
            _fail(f"dinov3_profile.identities.{key}", "live identity mismatch")
    worker = (repository / _WORKER_RELATIVE_PATH).resolve(strict=True)
    return _ExtractorContext(
        profile=profile,
        runtime_root=runtime,
        snapshot_root=snapshot,
        repository_root=repository,
        worker_script=worker,
        descriptor=_build_descriptor(),
    )


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


class DINOv3VisionEncoderExtractor:
    """Canonical DINOv3 adapter whose model executes only in an isolated process."""

    __slots__ = ("_config", "_context", "_last_worker_response")

    def __init__(self, config: DINOv3ExtractorConfig) -> None:
        if not isinstance(config, DINOv3ExtractorConfig):
            _fail("config", "expected DINOv3ExtractorConfig")
        try:
            context = _load_extractor_context(config)
        except ExtractorValidationError:
            raise
        except (
            DINOv3ExtractionBoundaryError,
            DINOv3ProductionIdentityError,
            ProductionRuntimeContractError,
            RuntimeBoundaryError,
            SnapshotBoundaryError,
        ) as exc:
            classification = getattr(exc, "classification", type(exc).__name__)
            raise ExtractorValidationError(
                f"dinov3_contract.{classification}: contract validation failed"
            ) from exc
        except Exception as exc:
            raise ExtractorValidationError(
                "dinov3_runtime: unable to construct verified isolated extractor"
            ) from exc
        self._config = config
        self._context = context
        self._last_worker_response: DINOv3ExtractionResponse | None = None

    @property
    def descriptor(self) -> ExtractorDescriptor:
        return self._context.descriptor

    @property
    def available_taps(self) -> tuple[TapSpec, ...]:
        return self.descriptor.taps

    def extract(
        self,
        batch: CanonicalImageBatch,
        *,
        tap_names: Sequence[str] | None = None,
    ) -> ExtractionResult:
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        try:
            result = self._extract(batch, tap_names=tap_names)
        finally:
            python_changed = random.getstate() != python_before
            numpy_changed = not _numpy_rng_equal(np.random.get_state(), numpy_before)
            if python_changed:
                random.setstate(python_before)
            if numpy_changed:
                np.random.set_state(numpy_before)
        if python_changed or numpy_changed:
            _fail("dinov3_runtime.rng", "current-process Python or NumPy RNG changed")
        return result

    def _extract(
        self,
        batch: CanonicalImageBatch,
        *,
        tap_names: Sequence[str] | None,
    ) -> ExtractionResult:
        if not isinstance(batch, CanonicalImageBatch):
            _fail("batch", "expected CanonicalImageBatch")
        if batch.rgb_uint8.shape[1:] != (224, 224, 3):
            _fail("batch.rgb_uint8.spatial_dimensions", "expected 224 x 224 RGB")
        requested = resolve_requested_taps(self.available_taps, tap_names)
        input_before = batch.rgb_uint8.tobytes(order="C")
        raw_sha256 = hashlib.sha256(input_before).hexdigest()
        workspace_path: Path | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="vea_dinov3_extract_") as temporary:
                workspace_path = Path(temporary).resolve(strict=True)
                temporary_input = workspace_path / ".input.npy.tmp"
                input_path = workspace_path / "input.npy"
                try:
                    with temporary_input.open("xb") as stream:
                        np.save(stream, batch.rgb_uint8, allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary_input, input_path)
                finally:
                    temporary_input.unlink(missing_ok=True)
                worker_policy = (
                    _APPLE_EXPECTED_WORKER_POLICY
                    if self._config.execution_selector == DINOV3_SELECTOR
                    else _EXPECTED_WORKER_POLICY
                )
                request_value = {
                        "expected_worker_policy": dict(worker_policy),
                        "extractor_profile_sha256": self._context.profile.sha256,
                        "image_ids": list(batch.image_ids),
                        "input_dtype": "uint8",
                        "input_file_sha256": sha256_file(input_path),
                        "input_shape": list(batch.rgb_uint8.shape),
                        "model_id": MODEL_ID,
                        "model_key": MODEL_KEY,
                        "operation": "extract_canonical_features",
                        "raw_input_sha256": raw_sha256,
                        "requested_tap_names": [tap.name for tap in requested],
                        "revision": REVISION,
                        "run_id": f"t007b-{uuid.uuid4().hex}",
                        "schema_id": "vea.dinov3_extraction_request.v0",
                        "workspace_relative_input_path": "input.npy",
                        "workspace_relative_output_directory": "output",
                    }
                request = (
                    validate_dinov3_extraction_request(
                        request_value, execution_selector=DINOV3_SELECTOR
                    )
                    if self._config.execution_selector == DINOV3_SELECTOR
                    else validate_dinov3_extraction_request(request_value)
                )
                try:
                    response = invoke_dinov3_extraction_worker(
                        runtime_root=self._context.runtime_root,
                        snapshot_root=self._context.snapshot_root,
                        repository_root=self._context.repository_root,
                        extractor_profile_path=self._config.profile_path,
                        worker_script=self._context.worker_script,
                        workspace_root=workspace_path,
                        request=request,
                        timeout_seconds=self._config.timeout_seconds,
                        execution_selector=self._config.execution_selector,
                    )
                except DINOv3ExtractionBoundaryError as exc:
                    raise ExtractorValidationError(
                        f"dinov3_worker.{exc.classification}: isolated extraction failed"
                    ) from exc
                self._last_worker_response = response
                features = self._read_artifacts(
                    response=response,
                    request=request,
                    requested=requested,
                    workspace=workspace_path,
                )
                result = ExtractionResult(
                    descriptor=self.descriptor,
                    image_ids=batch.image_ids,
                    tap_features=tuple(features),
                )
                if batch.rgb_uint8.tobytes(order="C") != input_before:
                    _fail("batch.rgb_uint8", "input was mutated during extraction")
                return result
        except ExtractorValidationError:
            raise
        except Exception as exc:
            raise ExtractorValidationError(
                "dinov3_runtime: isolated feature extraction failed"
            ) from exc
        finally:
            if workspace_path is not None and workspace_path.exists():
                _fail("dinov3_runtime.workspace", "temporary cleanup failed")

    def _read_artifacts(
        self,
        *,
        response: DINOv3ExtractionResponse,
        request: DINOv3ExtractionRequest,
        requested: tuple[TapSpec, ...],
        workspace: Path,
    ) -> list[CanonicalTapFeatures]:
        value = response.to_dict()
        entries = value["artifacts"]
        features: list[CanonicalTapFeatures] = []
        for index, (tap, entry) in enumerate(zip(requested, entries, strict=True)):
            sidecar = resolve_workspace_relative_path(
                workspace, entry["sidecar_relative_path"], must_exist=True
            )
            tensor = resolve_workspace_relative_path(
                workspace, entry["safetensors_relative_path"], must_exist=True
            )
            loaded = read_feature_artifact(sidecar)
            metadata = loaded.metadata
            expected = {
                "run_id": request.run_id,
                "model_key": MODEL_KEY,
                "tap_namespace": "backbone",
                "tap_name": tap.name,
                "image_ids": list(request.image_ids),
                "grid_height": 14,
                "grid_width": 14,
                "patch_count": 196,
                "hidden_dimension": 768,
                "dtype": "float32",
                "shard_index": index,
                "sidecar_file_name": sidecar.name,
                "safetensors_file_name": tensor.name,
                "run_status": "complete",
            }
            metadata_dict = metadata.to_dict()
            for key, required in expected.items():
                if metadata_dict[key] != required:
                    _fail(f"dinov3_artifact.{tap.name}.{key}", "metadata mismatch")
            patch_sha256 = _sha256_array(loaded.patch)
            global_sha256 = _sha256_array(loaded.global_mean)
            if (
                metadata.checksum_value != entry["checksum_sha256"]
                or sha256_file(tensor) != entry["checksum_sha256"]
                or patch_sha256 != entry["patch_sha256"]
                or global_sha256 != entry["global_sha256"]
            ):
                _fail(f"dinov3_artifact.{tap.name}", "checksum or value hash mismatch")
            if loaded.patch.shape != (len(request.image_ids), 196, 768):
                _fail(f"dinov3_artifact.{tap.name}.patch", "shape mismatch")
            if loaded.global_mean.shape != (len(request.image_ids), 768):
                _fail(f"dinov3_artifact.{tap.name}.global_mean", "shape mismatch")
            if loaded.patch.dtype != np.dtype(np.float32) or loaded.global_mean.dtype != np.dtype(np.float32):
                _fail(f"dinov3_artifact.{tap.name}.dtype", "expected float32")
            if not np.isfinite(loaded.patch).all() or not np.isfinite(loaded.global_mean).all():
                _fail(f"dinov3_artifact.{tap.name}", "contains NaN or Inf")
            exact_mean = loaded.patch.mean(axis=1, dtype=np.float32)
            if not np.array_equal(loaded.global_mean, exact_mean):
                _fail(f"dinov3_artifact.{tap.name}.global_mean", "not exact patch-axis mean")
            features.append(
                CanonicalTapFeatures(
                    tap=tap,
                    patch=loaded.patch,
                    global_mean=loaded.global_mean,
                )
            )
        return features


__all__ = ["DINOv3ExtractorConfig", "DINOv3VisionEncoderExtractor"]
