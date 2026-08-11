"""Deterministic 8/4 training-free DINOv3-to-CLIP alignment pilot.

Heavy runtime dependencies are deliberately imported only by the real runner.
The selection, fitting, metrics, aggregation, and orchestration contracts are
therefore testable on CPU with synthetic arrays and fakes.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from vision_encoder_alignment.vlm_compat import dinov3_raw_injection as raw
from vision_encoder_alignment.vlm_compat import tinyllava_native_replay as native

SCHEMA_ID = "vea.p2.t001c.training_free_alignment_pilot.v1"
STATUS = "PASS_TRAINING_FREE_ALIGNMENT_PILOT"
PROMPT = "What is shown in this image? Answer briefly."
CONDITIONS = ("NATIVE", "RAW", "CALIBRATED", "LINEAR")
ARTIFACT_NAMES = (
    "result.json",
    "per_image.csv",
    "summary.csv",
    "paired_patch_features.safetensors",
    "alignment_parameters.safetensors",
    "test_connector_outputs.safetensors",
)
DEVELOPMENT_IMAGE = "000000000139.jpg"
PATCH_SHAPE = (196, 768)
CONNECTOR_SHAPE = (196, 1536)


class AlignmentPilotError(RuntimeError):
    """A bounded pilot contract was violated."""


class PilotStageError(AlignmentPilotError):
    """An execution failure carrying bounded experiment coordinates."""

    def __init__(
        self,
        stage: str,
        *,
        image_id: str | None = None,
        relative_path: str | None = None,
        condition: str | None = None,
    ) -> None:
        super().__init__(f"{stage} failed")
        self.stage = stage
        self.image_id = image_id
        self.relative_path = Path(relative_path).name if relative_path is not None else None
        self.condition = condition


@contextmanager
def execution_context(
    stage: str,
    *,
    image_id: str | None = None,
    relative_path: str | None = None,
    condition: str | None = None,
) -> Iterable[None]:
    """Attach exact bounded coordinates while preserving the exception chain."""

    try:
        yield
    except PilotStageError:
        raise
    except Exception as exc:
        raise PilotStageError(
            stage,
            image_id=image_id,
            relative_path=relative_path,
            condition=condition,
        ) from exc


def _bounded_exception_message(error: BaseException) -> str:
    return " ".join(str(error).splitlines())[:512]


def bounded_cause_chain(error: BaseException) -> list[dict[str, str]]:
    """Return at most eight exception identities without traceback or runtime state."""

    result: list[dict[str, str]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and len(result) < 8 and id(current) not in seen:
        seen.add(id(current))
        result.append({"type": type(current).__name__[:128], "message": _bounded_exception_message(current)})
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return result


@dataclass(frozen=True, slots=True)
class ManifestRow:
    image_id: str
    relative_path: str
    source_sha256: str
    split: str
    selection_index: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ManifestRow":
        try:
            row = cls(
                image_id=str(value["image_id"]),
                relative_path=str(value["relative_path"]),
                source_sha256=str(value["source_sha256"]),
                split=str(value["split"]),
                selection_index=value["selection_index"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AlignmentPilotError("manifest row is malformed") from exc
        if type(row.selection_index) is not int or row.selection_index < 0:
            raise AlignmentPilotError("manifest selection_index must be a nonnegative integer")
        if not row.image_id or Path(row.relative_path).name != row.relative_path:
            raise AlignmentPilotError("manifest image identity or relative path is invalid")
        if len(row.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in row.source_sha256):
            raise AlignmentPilotError("manifest source_sha256 is invalid")
        return row


@dataclass(frozen=True, slots=True)
class PilotSelection:
    calibration: tuple[ManifestRow, ...]
    test: tuple[ManifestRow, ...]

    @property
    def all_rows(self) -> tuple[ManifestRow, ...]:
        return self.calibration + self.test


@dataclass(frozen=True, slots=True)
class MomentFit:
    scale: object
    bias: object
    clip_train_mean: object
    clip_train_std: object
    dinov3_train_mean: object
    dinov3_train_std: object


@dataclass(frozen=True, slots=True)
class LinearFit:
    rotation: object
    calibrated_train_mean: object
    clip_train_mean: object


def _load_manifest(path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AlignmentPilotError(f"manifest line {line_number} is not an object")
                rows.append(ManifestRow.from_mapping(value))
    except AlignmentPilotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AlignmentPilotError("manifest is unavailable or malformed") from exc
    return rows


def select_pilot_rows(rows_or_path: Sequence[ManifestRow] | str | Path) -> PilotSelection:
    """Select exactly the first eight train and first four test image rows."""

    rows = _load_manifest(rows_or_path) if isinstance(rows_or_path, (str, Path)) else list(rows_or_path)
    calibration = tuple(sorted((r for r in rows if r.split == "train"), key=lambda r: r.selection_index)[:8])
    test = tuple(sorted((r for r in rows if r.split == "test"), key=lambda r: r.selection_index)[:4])
    if len(calibration) != 8 or len(test) != 4:
        raise AlignmentPilotError("manifest does not provide exactly eight train and four test rows")
    if any(r.relative_path == DEVELOPMENT_IMAGE for r in calibration + test):
        raise AlignmentPilotError("the development image must not be selected")
    calibration_ids = {r.image_id for r in calibration}
    test_ids = {r.image_id for r in test}
    if len(calibration_ids) != 8 or len(test_ids) != 4 or calibration_ids & test_ids:
        raise AlignmentPilotError("calibration and held-out images must be unique and disjoint")
    if len({r.selection_index for r in calibration + test}) != 12:
        raise AlignmentPilotError("selected manifest rows have duplicate selection indices")
    return PilotSelection(calibration, test)


def _validated_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise AlignmentPilotError("dataset root must be an existing non-link directory")
    return root


def _validate_selected_source(row: ManifestRow, root: Path) -> Path:
    unresolved = root / row.relative_path
    if unresolved.is_symlink():
        raise AlignmentPilotError(f"selected source path is a link: {row.relative_path}")
    candidate = unresolved.resolve(strict=True)
    if candidate.parent != root or not candidate.is_file():
        raise AlignmentPilotError(f"selected source path is not a bounded regular file: {row.relative_path}")
    if native.sha256_file(candidate) != row.source_sha256:
        raise AlignmentPilotError(f"selected source SHA-256 mismatch: {row.relative_path}")
    return candidate


def validate_selected_sources(selection: PilotSelection, dataset_root: str | Path) -> tuple[Path, ...]:
    """Resolve and hash all selected files without allowing path escape or substitution."""

    root = _validated_dataset_root(dataset_root)
    return tuple(_validate_selected_source(row, root) for row in selection.all_rows)


def _require_image_partition(array: object, expected_images: int, label: str) -> Any:
    import numpy as np

    value = np.asarray(array)
    if value.shape != (expected_images, *PATCH_SHAPE):
        raise AlignmentPilotError(
            f"{label} must preserve image partition with shape {(expected_images, *PATCH_SHAPE)}"
        )
    if not np.isfinite(value).all():
        raise AlignmentPilotError(f"{label} contains NaN or Inf")
    return value


def fit_moment_calibration(dinov3_train: object, clip_train: object) -> MomentFit:
    """Fit exact dimension-wise population moments using image-partitioned train data."""

    import numpy as np

    source = _require_image_partition(dinov3_train, 8, "dinov3_train").astype(np.float64)
    target = _require_image_partition(clip_train, 8, "clip_train").astype(np.float64)
    source_rows = source.reshape(-1, source.shape[-1])
    target_rows = target.reshape(-1, target.shape[-1])
    source_mean = source_rows.mean(axis=0, dtype=np.float64)
    target_mean = target_rows.mean(axis=0, dtype=np.float64)
    source_std = source_rows.std(axis=0, ddof=0, dtype=np.float64)
    target_std = target_rows.std(axis=0, ddof=0, dtype=np.float64)
    scale = target_std / np.maximum(source_std, 1e-6)
    bias = target_mean - source_mean * scale
    values = (scale, bias, target_mean, target_std, source_mean, source_std)
    return MomentFit(*(np.asarray(value, dtype=np.float32) for value in values))


def apply_moment_calibration(value: object, fit: MomentFit) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.shape[-1:] != (768,) or not np.isfinite(array).all():
        raise AlignmentPilotError("moment input must be finite with final dimension 768")
    transformed = array.astype(np.float64) * np.asarray(fit.scale, dtype=np.float64)
    transformed += np.asarray(fit.bias, dtype=np.float64)
    return np.array(transformed, dtype=np.float32, order="C", copy=True)


def fit_orthogonal_procrustes(calibrated_train: object, clip_train: object) -> LinearFit:
    """Fit exact float64 orthogonal Procrustes without determinant correction."""

    import numpy as np

    source = _require_image_partition(calibrated_train, 8, "calibrated_train").astype(np.float64)
    target = _require_image_partition(clip_train, 8, "clip_train").astype(np.float64)
    source_rows = source.reshape(-1, 768)
    target_rows = target.reshape(-1, 768)
    source_mean = source_rows.mean(axis=0, dtype=np.float64)
    target_mean = target_rows.mean(axis=0, dtype=np.float64)
    u, _singular, vt = np.linalg.svd(
        (source_rows - source_mean).T @ (target_rows - target_mean),
        full_matrices=False,
    )
    rotation = u @ vt
    return LinearFit(
        np.asarray(rotation, dtype=np.float32),
        np.asarray(source_mean, dtype=np.float32),
        np.asarray(target_mean, dtype=np.float32),
    )


def apply_orthogonal_procrustes(value: object, fit: LinearFit) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.shape[-1:] != (768,) or not np.isfinite(array).all():
        raise AlignmentPilotError("linear input must be finite with final dimension 768")
    transformed = (
        (array.astype(np.float64) - np.asarray(fit.calibrated_train_mean, dtype=np.float64))
        @ np.asarray(fit.rotation, dtype=np.float64)
        + np.asarray(fit.clip_train_mean, dtype=np.float64)
    )
    return np.array(transformed, dtype=np.float32, order="C", copy=True)


def writable_contiguous_float32(value: object) -> Any:
    import numpy as np

    return np.array(value, dtype=np.float32, order="C", copy=True)


def held_out_patch_batch(dinov3_features: object, test_index: int) -> Any:
    """Select one held-out image while preserving its singleton batch axis."""

    import numpy as np

    source = np.asarray(dinov3_features)
    if source.shape != (12, *PATCH_SHAPE):
        raise AlignmentPilotError("DINO held-out source must have shape (12, 196, 768)")
    if type(test_index) is not int or test_index not in range(4):
        raise AlignmentPilotError("held-out test index must be an integer from zero through three")
    start = 8 + test_index
    result = source[start : start + 1]
    if result.shape != (1, *PATCH_SHAPE):
        raise AlignmentPilotError("held-out DINO batch axis was not preserved")
    return result


def validate_replacement_injection_array(value: object, *, condition: str) -> Any:
    """Fail fast on any replacement array outside the exact raw-injection ABI."""

    import numpy as np

    if condition not in CONDITIONS[1:]:
        raise AlignmentPilotError("replacement condition must be RAW, CALIBRATED, or LINEAR")
    if not isinstance(value, np.ndarray):
        raise AlignmentPilotError(f"{condition} replacement must be a NumPy array")
    if value.shape != (1, *PATCH_SHAPE):
        raise AlignmentPilotError(
            f"{condition} replacement shape is {value.shape}, not (1, 196, 768)"
        )
    if value.dtype != np.dtype(np.float32):
        raise AlignmentPilotError(f"{condition} replacement dtype must be float32")
    if not value.flags.c_contiguous:
        raise AlignmentPilotError(f"{condition} replacement must be C contiguous")
    if not value.flags.writeable:
        raise AlignmentPilotError(f"{condition} replacement must be writable")
    if not np.isfinite(value).all():
        raise AlignmentPilotError(f"{condition} replacement contains NaN or Inf")
    return value


def flattened_metrics(condition: object, native_value: object, train_target_mean: object) -> dict[str, float]:
    """Compute task-defined flattened float64 cosine, norm ratio, and NMSE."""

    import numpy as np

    candidate = np.asarray(condition, dtype=np.float64)
    reference = np.asarray(native_value, dtype=np.float64)
    mean = np.asarray(train_target_mean, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.shape[-1:] != mean.shape[-1:]:
        raise AlignmentPilotError("metric tensors have incompatible shapes")
    candidate_flat, reference_flat = candidate.reshape(-1), reference.reshape(-1)
    candidate_norm = float(np.linalg.norm(candidate_flat))
    reference_norm = float(np.linalg.norm(reference_flat))
    cosine_denominator = candidate_norm * reference_norm
    denominator = float(np.square(reference - mean).sum(dtype=np.float64))
    if cosine_denominator == 0.0 or reference_norm == 0.0 or denominator == 0.0:
        raise AlignmentPilotError("metric denominator is zero")
    return {
        "cosine": float(np.dot(candidate_flat, reference_flat) / cosine_denominator),
        "norm_ratio": candidate_norm / reference_norm,
        "nmse": float(np.square(candidate - reference).sum(dtype=np.float64) / denominator),
    }


def aggregate_metrics(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate each metric in fixed condition order and compare it with RAW per image."""

    numeric_metrics = (
        "patch_cosine", "patch_norm_ratio", "patch_nmse",
        "connector_cosine", "connector_norm_ratio", "connector_nmse",
        "generation_success", "token_ids_equal_to_native", "decoded_text_equal_to_native",
        "generated_token_count",
    )
    higher = {
        "patch_cosine", "connector_cosine", "generation_success",
        "token_ids_equal_to_native", "decoded_text_equal_to_native",
    }
    by_condition = {condition: [r for r in records if r["condition"] == condition] for condition in CONDITIONS}
    if any(len(rows) != 4 for rows in by_condition.values()):
        raise AlignmentPilotError("aggregation requires four held-out rows for every condition")
    raw_by_image = {str(r["image_id"]): r for r in by_condition["RAW"]}
    native_by_image = {str(r["image_id"]): r for r in by_condition["NATIVE"]}
    summary: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for metric in numeric_metrics:
            values = [float(r[metric]) for r in by_condition[condition]]
            improved = 0
            for record, value in zip(by_condition[condition], values, strict=True):
                raw_value = float(raw_by_image[str(record["image_id"])][metric])
                native_value = float(native_by_image[str(record["image_id"])][metric])
                if metric in {"patch_norm_ratio", "connector_norm_ratio"}:
                    is_improved = abs(value - 1.0) < abs(raw_value - 1.0)
                elif metric == "generated_token_count":
                    is_improved = abs(value - native_value) < abs(raw_value - native_value)
                else:
                    is_improved = value > raw_value if metric in higher else value < raw_value
                if is_improved:
                    improved += 1
            summary.append({
                "condition": condition,
                "metric": metric,
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
                "count_improved_over_raw": improved,
            })
    return summary


def fake_orchestration(
    *,
    extract_dino_batch: Callable[[tuple[str, ...]], object],
    load_tinyllava: Callable[[], object],
    capture_calibration_feature: Callable[[object, str], object],
    generate_test: Callable[[object, str, str], object],
    calibration_ids: Sequence[str],
    test_ids: Sequence[str],
) -> dict[str, int]:
    """Dependency-injected call plan used to prove the exact runtime cardinalities."""

    if len(calibration_ids) != 8 or len(test_ids) != 4 or set(calibration_ids) & set(test_ids):
        raise AlignmentPilotError("fake orchestration requires a disjoint 8/4 image split")
    extract_dino_batch(tuple(calibration_ids) + tuple(test_ids))
    model = load_tinyllava()
    for image_id in calibration_ids:
        capture_calibration_feature(model, image_id)
    generation_calls = 0
    for image_id in test_ids:
        for condition in CONDITIONS:
            generate_test(model, image_id, condition)
            generation_calls += 1
    return {
        "dinov3_batch_extractions": 1,
        "tinyllava_model_loads": 1,
        "calibration_native_feature_calls": 8,
        "calibration_generation_calls": 0,
        "test_generation_calls": generation_calls,
    }


def build_result(**values: object) -> dict[str, object]:
    result = {"schema_id": SCHEMA_ID, "status": STATUS, **values}
    native.canonical_json_bytes(result)
    return result


def _extract_dino_batch(
    pixels: object,
    image_ids: tuple[str, ...],
    *,
    repository_root: Path,
    runtime_root: Path,
    snapshot_root: Path,
) -> Any:
    """Reuse the verified R3 extractor boundary for exactly one 12-image invocation."""

    import numpy as np
    from vision_encoder_alignment.extractors.base import CanonicalImageBatch

    bindings = raw.validate_dinov3_roots(
        repository_root=repository_root, runtime_root=runtime_root, snapshot_root=snapshot_root
    )
    profile = bindings.repository_root / raw.DINO_PROFILE
    runtime_profile = bindings.repository_root / raw.DINO_RUNTIME_PROFILE
    batch = CanonicalImageBatch(image_ids, pixels)
    try:
        with raw.apple_dinov3_interpreter_compatibility(bindings):
            extractor = raw._construct_dinov3_extractor(
                bindings, profile_path=profile, runtime_profile_path=runtime_profile
            )
            result = extractor.extract(batch, tap_names=(raw.DINO_TAP,))
    except Exception as exc:
        raise AlignmentPilotError("verified isolated 12-image DINO extraction failed") from exc
    finally:
        try:
            del extractor
        except UnboundLocalError:
            pass
    descriptor = result.descriptor
    if (
        descriptor.model_id != raw.DINO_MODEL_ID
        or descriptor.revision != raw.DINO_REVISION
        or descriptor.token_layout.special_token_indices != (0, 1, 2, 3, 4)
        or descriptor.token_layout.special_token_policy != "dinov3_cls1_register4_prefix_verified_runtime_v1"
        or tuple(result.image_ids) != image_ids
    ):
        raise AlignmentPilotError("DINO identity, order, or token-removal policy mismatch")
    feature = result.get_tap(raw.DINO_TAP)
    patch = np.array(feature.patch, dtype=np.float32, order="C", copy=True)
    if feature.tap.name != raw.DINO_TAP or patch.shape != (12, *PATCH_SHAPE) or not np.isfinite(patch).all():
        raise AlignmentPilotError("DINO batch feature contract mismatch")
    return patch


def _repository_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise AlignmentPilotError("repository commit cannot be resolved")
    return completed.stdout.strip()


def _chat_kwargs(model: object, tokenizer: object, image: Path, config: native.AnchorConfig, max_tokens: int) -> dict[str, object]:
    kwargs = native._chat_kwargs(model=model, tokenizer=tokenizer, image_path=image, config=config)
    kwargs["prompt"] = PROMPT
    kwargs["max_new_tokens"] = max_tokens
    return kwargs


def _capture(
    model: object, vision_tower: object, connector: object, tokenizer: object,
    image: Path, config: native.AnchorConfig, max_tokens: int,
) -> tuple[native.CaptureState, list[list[int]], str]:
    kwargs = _chat_kwargs(model, tokenizer, image, config, max_tokens)
    return native._run_chat_capture(
        model=model, vision_tower=vision_tower, connector=connector,
        chat=lambda: model.chat(**kwargs),  # type: ignore[attr-defined]
    )


def _validate_capture(state: native.CaptureState, dtype: str, *, injected: bool) -> None:
    if state.patch_tensor is None or state.connector_tensor is None:
        raise AlignmentPilotError("TinyLLaVA capture is incomplete")
    if (state.vision_calls != 0) if injected else (state.vision_calls < 1):
        raise AlignmentPilotError("TinyLLaVA vision-call contract mismatch")
    if state.connector_calls < 1:
        raise AlignmentPilotError("TinyLLaVA connector was not invoked")
    finite = lambda value: bool(value.isfinite().all().item())
    native.validate_tensor_contract(
        state.patch_tensor, expected_shape=(1, *PATCH_SHAPE), expected_dtype=dtype,
        finite_check=finite, label="connector_input_patch",
    )
    native.validate_tensor_contract(
        state.connector_tensor, expected_shape=(1, *CONNECTOR_SHAPE), expected_dtype=dtype,
        finite_check=finite, label="connector_output",
    )


def _numpy(tensor: object) -> Any:
    return tensor.detach().float().cpu().numpy()  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class DirectCalibrationCapture:
    patch: object
    connector: object
    vision_calls: int
    connector_calls: int


def capture_native_calibration_features(
    *,
    model: object,
    vision_tower: object,
    connector: object,
    image_path: str | Path,
    dtype: str,
) -> DirectCalibrationCapture:
    """Capture native patch/connector pairs without invoking language generation."""

    module_name = getattr(model.__class__, "__module__", None)
    if not isinstance(module_name, str) or not module_name:
        raise native.RuntimeABIError("pinned model runtime module is unavailable")
    try:
        runtime_module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise native.RuntimeABIError("pinned model runtime module cannot be resolved") from exc
    load_image = getattr(runtime_module, "load_image", None)
    process_images = getattr(runtime_module, "process_images", None)
    if not callable(load_image) or not callable(process_images):
        raise native.RuntimeABIError("pinned runtime load_image and process_images are required")
    image_processor = getattr(vision_tower, "_image_processor", None)
    model_config = getattr(model, "config", None)
    model_device = getattr(model, "device", None)
    model_dtype = getattr(model, "dtype", None)
    if image_processor is None or model_config is None or model_device is None or model_dtype is None:
        raise native.RuntimeABIError("pinned preprocessing runtime attributes are unavailable")
    try:
        loaded_image = load_image(str(Path(image_path)))
        image_tensor = process_images([loaded_image], image_processor, model_config)
        image_tensor = image_tensor.to(device=model_device, dtype=model_dtype)
    except Exception as exc:
        raise native.RuntimeABIError("pinned TinyLLaVA image preprocessing failed") from exc

    state = native.CaptureState()
    handles: list[object] = []
    returned: object | None = None
    try:
        handles.append(vision_tower.register_forward_hook(state.vision_hook))
        handles.append(connector.register_forward_pre_hook(state.connector_pre_hook))
        handles.append(connector.register_forward_hook(state.connector_hook))
        encode_images = getattr(model, "encode_images", None)
        if not callable(encode_images):
            raise native.RuntimeABIError("model.encode_images is required for calibration capture")
        returned = encode_images(image_tensor)
    except (native.NativeReplayError, AlignmentPilotError):
        raise
    except Exception as exc:
        raise native.RuntimeABIError("direct native calibration capture failed") from exc
    finally:
        native._remove_hooks(handles)

    _validate_capture(state, dtype, injected=False)
    if state.vision_calls != 1 or state.connector_calls != 1:
        raise AlignmentPilotError("direct calibration must invoke vision and connector exactly once")
    if returned is None or state.connector_tensor is None:
        raise AlignmentPilotError("direct calibration connector return is unavailable")
    try:
        returned_matches_hook = bool(returned.detach().equal(state.connector_tensor))
    except AttributeError as exc:
        raise AlignmentPilotError("direct calibration return cannot be compared exactly") from exc
    if not returned_matches_hook:
        raise AlignmentPilotError("encode_images return differs from the connector hook output")

    patch = writable_contiguous_float32(_numpy(state.patch_tensor))
    connector_output = writable_contiguous_float32(_numpy(state.connector_tensor))
    state.patch_tensor = None
    state.connector_tensor = None
    del returned, image_tensor, loaded_image
    return DirectCalibrationCapture(
        patch=patch,
        connector=connector_output,
        vision_calls=state.vision_calls,
        connector_calls=state.connector_calls,
    )


def _record(
    *, image_id: str, relative_path: str, condition: str, state: native.CaptureState,
    tokens: list[list[int]], text: str, native_state: native.CaptureState,
    native_tokens: list[list[int]], native_text: str, patch_mean: object, connector_mean: object,
) -> dict[str, object]:
    patch = flattened_metrics(_numpy(state.patch_tensor), _numpy(native_state.patch_tensor), patch_mean)
    connector = flattened_metrics(
        _numpy(state.connector_tensor), _numpy(native_state.connector_tensor), connector_mean
    )
    token_count = len(tokens[0]) if tokens else 0
    return {
        "image_id": image_id, "relative_path": relative_path, "condition": condition,
        "patch_cosine": patch["cosine"], "patch_norm_ratio": patch["norm_ratio"], "patch_nmse": patch["nmse"],
        "connector_cosine": connector["cosine"], "connector_norm_ratio": connector["norm_ratio"], "connector_nmse": connector["nmse"],
        "generation_success": True, "token_ids_equal_to_native": tokens == native_tokens,
        "decoded_text_equal_to_native": text == native_text, "generated_token_count": token_count,
        "native_answer": native_text, "condition_answer": text, "token_ids": tokens,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise AlignmentPilotError(f"cannot write empty {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def run_training_free_alignment_pilot(
    *, tinyllava_config: str | Path, manifest: str | Path, dataset_root: str | Path,
    cache_root: str | Path, repository_root: str | Path, dinov3_runtime_root: str | Path,
    dinov3_snapshot_root: str | Path, output_dir: str | Path, device: str, dtype: str,
) -> dict[str, object]:
    """Execute the exact sequential Apple pilot and publish its six artifacts."""

    stage_start = time.perf_counter()
    elapsed: dict[str, float] = {}
    with execution_context("input_validation"):
        config = native.load_anchor_config(tinyllava_config)
        selection = select_pilot_rows(manifest)
        repository = Path(repository_root).resolve(strict=True)
        cache = Path(cache_root).resolve(strict=True)
        bindings = raw.validate_dinov3_roots(
            repository_root=repository, runtime_root=dinov3_runtime_root,
            snapshot_root=dinov3_snapshot_root,
        )
        if not cache.is_dir() or cache.is_symlink():
            raise AlignmentPilotError("cache root must be an existing non-link directory")
        if not device.startswith("cuda"):
            raise AlignmentPilotError("the Apple alignment pilot requires CUDA")
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise AlignmentPilotError("dtype must be float16, bfloat16, or float32")
        output = Path(output_dir)
        if output.exists():
            raise FileExistsError("output directory must be fresh")
        dataset = _validated_dataset_root(dataset_root)
    selected_paths: list[Path] = []
    for row in selection.all_rows:
        with execution_context(
            "input_validation", image_id=row.image_id, relative_path=row.relative_path
        ):
            selected_paths.append(_validate_selected_source(row, dataset))
    paths = tuple(selected_paths)
    elapsed["input_validation_seconds"] = time.perf_counter() - stage_start

    import numpy as np

    decode_start = time.perf_counter()
    decoded: list[Any] = []
    for row, path in zip(selection.all_rows, paths, strict=True):
        with execution_context(
            "input_validation", image_id=row.image_id, relative_path=row.relative_path
        ):
            decoded.append(raw.canonical_rgb_nhwc(path))
    with execution_context("input_validation"):
        pixels = np.concatenate(decoded, axis=0)
        if pixels.shape != (12, 224, 224, 3):
            raise AlignmentPilotError("canonical decoded batch shape mismatch")
    elapsed["decode_seconds"] = time.perf_counter() - decode_start
    dino_start = time.perf_counter()
    with execution_context("dinov3_extraction"):
        dino = _extract_dino_batch(
            pixels, tuple(row.image_id for row in selection.all_rows), repository_root=repository,
            runtime_root=bindings.runtime_root, snapshot_root=bindings.snapshot_root,
        )
    elapsed["dinov3_worker_seconds"] = time.perf_counter() - dino_start

    model_start = time.perf_counter()
    with execution_context("tinyllava_load"):
        try:
            import torch
            from safetensors.torch import save_file
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except (ImportError, ModuleNotFoundError) as exc:
            raise AlignmentPilotError("required offline TinyLLaVA dependencies are unavailable") from exc
        if not torch.cuda.is_available():
            raise AlignmentPilotError("CUDA is unavailable")
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        common = {
            "revision": config.revision, "cache_dir": str(cache),
            "local_files_only": config.local_files_only, "trust_remote_code": config.trust_remote_code,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch_dtype, **common)
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id, revision=config.revision, cache_dir=str(cache),
                local_files_only=config.local_files_only, trust_remote_code=config.trust_remote_code,
                use_fast=False, model_max_length=model.config.tokenizer_model_max_length,
                padding_side=model.config.tokenizer_padding_side,
            )
        except Exception as exc:
            raise AlignmentPilotError("exact local TinyLLaVA model or tokenizer could not be loaded") from exc
        model.to(device=device, dtype=torch_dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise AlignmentPilotError("TinyLLaVA is not fully frozen")
        vision_tower = model.get_vision_tower() if callable(getattr(model, "get_vision_tower", None)) else native._resolve_component(
            model, label="vision tower", paths=("vision_tower", "model.vision_tower")
        )
        connector = native._resolve_component(
            model, label="connector", paths=("connector", "model.connector", "mm_projector", "model.mm_projector")
        )
        runtime_config = native._resolve_runtime_config(model, vision_tower, config)
        if any(parameter.requires_grad for parameter in vision_tower.parameters()):
            raise AlignmentPilotError("TinyLLaVA vision tower is not frozen")
    elapsed["tinyllava_load_seconds"] = time.perf_counter() - model_start

    calibration_start = time.perf_counter()
    calibration_patches: list[Any] = []
    calibration_connectors: list[Any] = []
    with torch.inference_mode():
        for row, path in zip(selection.calibration, paths[:8], strict=True):
            with execution_context(
                "calibration_native_capture",
                image_id=row.image_id,
                relative_path=row.relative_path,
            ):
                captured = capture_native_calibration_features(
                    model=model,
                    vision_tower=vision_tower,
                    connector=connector,
                    image_path=path,
                    dtype=dtype,
                )
                calibration_patches.append(captured.patch[0])
                calibration_connectors.append(captured.connector[0])
    with execution_context("alignment_fit"):
        clip_train = np.stack(calibration_patches)
        connector_train = np.stack(calibration_connectors)
        moment = fit_moment_calibration(dino[:8], clip_train)
        calibrated_train = apply_moment_calibration(dino[:8], moment)
        linear = fit_orthogonal_procrustes(calibrated_train, clip_train)
        patch_mean = clip_train.reshape(-1, 768).mean(axis=0, dtype=np.float64)
        connector_mean = connector_train.reshape(-1, 1536).mean(axis=0, dtype=np.float64)
    elapsed["calibration_and_fit_seconds"] = time.perf_counter() - calibration_start

    test_start = time.perf_counter()
    records: list[dict[str, object]] = []
    test_clip: list[Any] = []
    connector_outputs: dict[str, list[Any]] = {condition: [] for condition in CONDITIONS}
    with torch.inference_mode():
        for test_index, (row, path) in enumerate(zip(selection.test, paths[8:], strict=True)):
            with execution_context(
                "held_out_generation",
                image_id=row.image_id,
                relative_path=row.relative_path,
                condition="NATIVE",
            ):
                native_state, native_tokens, native_text = _capture(
                    model, vision_tower, connector, tokenizer, path, config, 32
                )
                _validate_capture(native_state, dtype, injected=False)
                test_clip.append(_numpy(native_state.patch_tensor)[0])
            states: dict[str, tuple[native.CaptureState, list[list[int]], str]] = {
                "NATIVE": (native_state, native_tokens, native_text)
            }
            with execution_context(
                "held_out_generation",
                image_id=row.image_id,
                relative_path=row.relative_path,
                condition="RAW",
            ):
                raw_batch = held_out_patch_batch(dino, test_index)
            with execution_context(
                "held_out_generation",
                image_id=row.image_id,
                relative_path=row.relative_path,
                condition="CALIBRATED",
            ):
                calibrated_batch = apply_moment_calibration(raw_batch, moment)
            with execution_context(
                "held_out_generation",
                image_id=row.image_id,
                relative_path=row.relative_path,
                condition="LINEAR",
            ):
                linear_batch = apply_orthogonal_procrustes(calibrated_batch, linear)
            replacement_batches = {
                "RAW": raw_batch,
                "CALIBRATED": calibrated_batch,
                "LINEAR": linear_batch,
            }
            for condition in CONDITIONS[1:]:
                with execution_context(
                    "held_out_generation",
                    image_id=row.image_id,
                    relative_path=row.relative_path,
                    condition=condition,
                ):
                    array = writable_contiguous_float32(replacement_batches[condition])
                    array = validate_replacement_injection_array(array, condition=condition)
                    tensor = torch.from_numpy(array)
                    injection = native.ReplayInjection(
                        model, tensor, connector, prepare_patch=native._prepare_for_connector
                    )
                    with injection:
                        state, tokens, text = _capture(
                            model, vision_tower, connector, tokenizer, path, config, 32
                        )
                    _validate_capture(state, dtype, injected=True)
                    if injection.invocations != 1 or not injection.restored:
                        raise AlignmentPilotError("injection did not run once and restore encode_images")
                    states[condition] = (state, tokens, text)
            for condition in CONDITIONS:
                with execution_context(
                    "held_out_generation",
                    image_id=row.image_id,
                    relative_path=row.relative_path,
                    condition=condition,
                ):
                    state, tokens, text = states[condition]
                    connector_outputs[condition].append(_numpy(state.connector_tensor)[0])
                    records.append(_record(
                        image_id=row.image_id, relative_path=row.relative_path, condition=condition,
                        state=state, tokens=tokens, text=text, native_state=native_state,
                        native_tokens=native_tokens, native_text=native_text,
                        patch_mean=patch_mean, connector_mean=connector_mean,
                    ))
    elapsed["held_out_generation_seconds"] = time.perf_counter() - test_start
    with execution_context("held_out_generation"):
        if len(records) != 16:
            raise AlignmentPilotError("held-out generation count is not exactly 16")
        unload_start = time.perf_counter()
        del model, vision_tower, connector, tokenizer
        torch.cuda.empty_cache()
        elapsed["tinyllava_unload_seconds"] = time.perf_counter() - unload_start

    publish_start = time.perf_counter()
    with execution_context("artifact_publication"):
        summary = aggregate_metrics(records)
        output.mkdir(parents=True, exist_ok=False)
        _write_csv(output / "per_image.csv", records)
        _write_csv(output / "summary.csv", summary)
        save_file({
            "calibration_clip_block11": torch.from_numpy(writable_contiguous_float32(clip_train)),
            "calibration_dinov3_block11": torch.from_numpy(writable_contiguous_float32(dino[:8])),
            "test_clip_block11": torch.from_numpy(writable_contiguous_float32(np.stack(test_clip))),
            "test_dinov3_block11": torch.from_numpy(writable_contiguous_float32(dino[8:])),
        }, str(output / "paired_patch_features.safetensors"))
        save_file({
            "moment_scale": torch.from_numpy(writable_contiguous_float32(moment.scale)),
            "moment_bias": torch.from_numpy(writable_contiguous_float32(moment.bias)),
            "calibrated_train_mean": torch.from_numpy(writable_contiguous_float32(linear.calibrated_train_mean)),
            "clip_train_mean": torch.from_numpy(writable_contiguous_float32(moment.clip_train_mean)),
            "clip_train_std": torch.from_numpy(writable_contiguous_float32(moment.clip_train_std)),
            "dinov3_train_mean": torch.from_numpy(writable_contiguous_float32(moment.dinov3_train_mean)),
            "dinov3_train_std": torch.from_numpy(writable_contiguous_float32(moment.dinov3_train_std)),
            "procrustes_rotation": torch.from_numpy(writable_contiguous_float32(linear.rotation)),
        }, str(output / "alignment_parameters.safetensors"))
        save_file({
            condition.lower(): torch.from_numpy(writable_contiguous_float32(np.stack(connector_outputs[condition])))
            for condition in CONDITIONS
        }, str(output / "test_connector_outputs.safetensors"))
        elapsed["artifact_publication_seconds"] = time.perf_counter() - publish_start
        elapsed["total_seconds"] = time.perf_counter() - stage_start
        artifact_inventory = {}
        for name in ARTIFACT_NAMES[1:]:
            artifact = output / name
            artifact_inventory[name] = {"bytes": artifact.stat().st_size, "sha256": native.sha256_file(artifact)}
        result = build_result(
            repository_commit=_repository_commit(repository),
            tinyllava={"model_id": config.model_id, "revision": config.revision, "feature_tap": "hidden_states[-2]", "runtime_config": runtime_config},
            dinov3={"model_id": raw.DINO_MODEL_ID, "revision": raw.DINO_REVISION, "feature_tap": "hidden_states[11]", "special_tokens_removed": "CLS 1 + register tokens 4"},
            prompt=PROMPT,
            generation={"test_max_new_tokens": 32, "num_beams": 1, "do_sample": False, "top_p": None, "temperature": 0},
            calibration_capture={
                "method": "pinned runtime load_image -> process_images -> model.encode_images",
                "language_generation": False,
            },
            split_policy="first 8 train and first 4 test by selection_index; image-level partition before patch flattening",
            calibration_images=[asdict(row) for row in selection.calibration],
            test_images=[asdict(row) for row in selection.test],
            conditions=list(CONDITIONS),
            transformations={
                "CALIBRATED": "dimension-wise population-moment affine fit, float64, denominator floor 1e-6",
                "LINEAR": "orthogonal Procrustes after CALIBRATED, float64 SVD, R=U@Vt, reflection allowed",
            },
            parameter_contract={
                "moment_scale": {"shape": [768], "dtype": "float32"},
                "moment_bias": {"shape": [768], "dtype": "float32"},
                "calibrated_train_mean": {"shape": [768], "dtype": "float32"},
                "clip_train_mean": {"shape": [768], "dtype": "float32"},
                "clip_train_std": {"shape": [768], "dtype": "float32"},
                "dinov3_train_mean": {"shape": [768], "dtype": "float32"},
                "dinov3_train_std": {"shape": [768], "dtype": "float32"},
                "procrustes_rotation": {"shape": [768, 768], "dtype": "float32"},
            },
            per_image=records,
            aggregate_metrics=summary,
            runtime_call_counts={
                "dinov3_batch_extractions": 1,
                "tinyllava_model_loads": 1,
                "calibration_native_feature_calls": 8,
                "calibration_generation_calls": 0,
                "test_generation_calls": 16,
            },
            artifacts=artifact_inventory,
            elapsed_seconds=elapsed,
        )
        native._write_new(output / "result.json", native.canonical_json_bytes(result))
    return result


def write_failure_result(output_dir: str | Path, *, error: BaseException) -> Path | None:
    """Write bounded structured failure evidence without overwriting prior output."""

    try:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "result.json"
        context = error if isinstance(error, PilotStageError) else None
        native._write_new(path, native.canonical_json_bytes({
            "schema_id": SCHEMA_ID, "status": "BLOCKED_TRAINING_FREE_ALIGNMENT_PILOT",
            "stage": context.stage if context is not None else "unclassified",
            "image_id": context.image_id if context is not None else None,
            "relative_path": context.relative_path if context is not None else None,
            "condition": context.condition if context is not None else None,
            "error_type": type(error).__name__,
            "message": _bounded_exception_message(error),
            "cause_chain": bounded_cause_chain(error),
        }))
        return path
    except Exception:
        return None
