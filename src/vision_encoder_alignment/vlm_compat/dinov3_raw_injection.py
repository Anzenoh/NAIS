"""One-image, raw DINOv3 block-11 injection smoke.

Only standard-library modules and the import-safe TinyLLaVA replay boundary are
imported at module load time. DINOv3, Torch, Transformers, Pillow, and
Safetensors are runtime-only dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from . import tinyllava_native_replay as native_replay


RESULT_SCHEMA_ID = "vea.p2.t001b.dinov3_raw_injection.v1"
STATUS = "PASS_DINOV3_RAW_INJECTION_SMOKE"
GEOMETRY_ID = "canonical_224_center_crop_v0"
DINO_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
DINO_TAP = "block_11"
DINO_SELECTOR = "dinov3_apple_cuda_fp32_v1"
DINO_PROFILE = "configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json"
DINO_RUNTIME_PROFILE = "configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json"
DINO_SNAPSHOT_PROFILE = "configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json"


class DINOv3RawInjectionError(RuntimeError):
    """Base exception for the one-image raw-injection smoke."""

    classification = "BLOCKED_DINOV3_RAW_INJECTION"


class DINOv3RawContractError(DINOv3RawInjectionError):
    """Raised for an invalid DINO source or raw-injection contract."""

    classification = "FAILED_DINOV3_RAW_INJECTION_CONTRACT"


@dataclass(frozen=True, slots=True)
class GeometryPlan:
    resize_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class DINOv3RootBindings:
    repository_root: Path
    runtime_root: Path
    snapshot_root: Path
    interpreter: Path


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())
    except OSError as exc:
        raise DINOv3RawContractError("unable to inspect root path") from exc


def _absolute_non_link_directory(value: str | Path, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise DINOv3RawContractError(f"{label} must be a string or Path")
    candidate = Path(value)
    if not candidate.is_absolute() or _is_link(candidate):
        raise DINOv3RawContractError(f"{label} must be an absolute non-link directory")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DINOv3RawContractError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise DINOv3RawContractError(f"{label} must be an existing directory")
    return resolved


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_disjoint_roots(*roots: tuple[str, Path]) -> None:
    for index, (left_name, left) in enumerate(roots):
        for right_name, right in roots[index + 1 :]:
            if _contains(left, right) or _contains(right, left):
                raise DINOv3RawContractError(
                    f"{left_name} and {right_name} must not overlap"
                )


def _load_pinned_runtime_root(profile_path: Path) -> Path:
    from vision_encoder_alignment.production_runtime_contract import (
        load_production_runtime_profile,
    )

    profile = load_production_runtime_profile(profile_path)
    if profile.selector != DINO_SELECTOR or profile.interpreter_layout != "linux_bin":
        raise DINOv3RawContractError("pinned Apple DINOv3 runtime profile mismatch")
    return Path(profile.runtime_root)


def _validate_pinned_snapshot(
    snapshot_root: Path,
    *,
    profile_path: Path,
    repository_root: Path,
    runtime_root: Path,
) -> None:
    from vision_encoder_alignment.snapshot_boundary import (
        load_snapshot_profile,
        validate_snapshot_root,
    )

    profile = load_snapshot_profile(profile_path)
    validate_snapshot_root(
        snapshot_root,
        profile,
        repository_root=repository_root,
        persistent_runtime_root=runtime_root,
    )


def resolve_apple_dinov3_interpreter(
    runtime_root: str | Path,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> Path:
    """Validate the directly rooted Apple venv interpreter and its identity."""

    root = _absolute_non_link_directory(runtime_root, label="DINO runtime root")
    primary = root / "bin" / "python"
    fallback = root / "bin" / "python3"
    candidate = primary if primary.exists() or primary.is_symlink() else fallback
    if (
        not candidate.exists()
        or not candidate.is_file()
        or candidate.is_symlink()
        or not os.access(candidate, os.X_OK)
    ):
        raise DINOv3RawContractError(
            "Apple DINOv3 interpreter must be executable bin/python or bin/python3"
        )
    try:
        interpreter = candidate.resolve(strict=True)
        interpreter.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DINOv3RawContractError("Apple DINOv3 interpreter escapes its runtime root") from exc
    try:
        completed = run(
            [
                str(interpreter),
                "-I",
                "-c",
                "from pathlib import Path; import sys; print(Path(sys.prefix).resolve())",
            ],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DINOv3RawContractError("Apple DINOv3 interpreter prefix probe failed") from exc
    try:
        prefix = Path(completed.stdout.strip()).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DINOv3RawContractError("Apple DINOv3 interpreter prefix probe failed") from exc
    if completed.returncode != 0 or prefix != root:
        raise DINOv3RawContractError("Apple DINOv3 interpreter prefix is not the runtime root")
    return interpreter


def validate_dinov3_roots(
    *,
    repository_root: str | Path,
    runtime_root: str | Path,
    snapshot_root: str | Path,
) -> DINOv3RootBindings:
    """Validate all explicit Apple DINO roots without consulting host defaults."""

    repository = _absolute_non_link_directory(repository_root, label="repository root")
    runtime = _absolute_non_link_directory(runtime_root, label="DINO runtime root")
    snapshot = _absolute_non_link_directory(snapshot_root, label="DINO snapshot root")
    _require_disjoint_roots(
        ("repository root", repository),
        ("DINO runtime root", runtime),
        ("DINO snapshot root", snapshot),
    )
    runtime_profile = repository / DINO_RUNTIME_PROFILE
    snapshot_profile = repository / DINO_SNAPSHOT_PROFILE
    if not runtime_profile.is_file() or not snapshot_profile.is_file():
        raise DINOv3RawContractError("verified Apple DINOv3 root profiles are unavailable")
    expected_runtime = _load_pinned_runtime_root(runtime_profile)
    if not expected_runtime.is_absolute() or runtime.as_posix() != expected_runtime.as_posix():
        raise DINOv3RawContractError("DINO runtime root does not match the pinned Apple profile")
    interpreter = resolve_apple_dinov3_interpreter(runtime)
    _validate_pinned_snapshot(
        snapshot,
        profile_path=snapshot_profile,
        repository_root=repository,
        runtime_root=runtime,
    )
    return DINOv3RootBindings(repository, runtime, snapshot, interpreter)


def _runtime_python_modules() -> tuple[object, object, object]:
    from vision_encoder_alignment import dinov3_extraction_boundary, runtime_boundary
    from vision_encoder_alignment.extractors import dinov3

    return runtime_boundary, dinov3_extraction_boundary, dinov3


@contextmanager
def apple_dinov3_interpreter_compatibility(
    bindings: DINOv3RootBindings,
) -> Iterator[None]:
    """Temporarily adapt the shared Linux layout resolver for the accepted Apple venv."""

    modules = _runtime_python_modules()
    originals = tuple(getattr(module, "runtime_python_path") for module in modules)
    delegate = originals[0]

    def replacement(
        runtime_root: str | Path,
        *,
        interpreter_layout: str = "windows_scripts",
    ) -> Path:
        try:
            candidate = Path(runtime_root).resolve(strict=True)
        except (OSError, RuntimeError):
            candidate = Path(runtime_root)
        if interpreter_layout == "linux_bin" and candidate == bindings.runtime_root:
            return bindings.interpreter
        return delegate(runtime_root, interpreter_layout=interpreter_layout)

    try:
        for module in modules:
            setattr(module, "runtime_python_path", replacement)
        yield
    finally:
        for module, original in zip(modules, originals, strict=True):
            setattr(module, "runtime_python_path", original)


def canonical_geometry_plan(width: int, height: int) -> GeometryPlan:
    """Return the exact P1 shorter-side resize and centered 224 crop plan."""

    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise DINOv3RawContractError("image dimensions must be positive integers")
    short = min(width, height)
    if width <= height:
        size = (224, (height * 224 + short // 2) // short)
    else:
        size = ((width * 224 + short // 2) // short, 224)
    left, top = (size[0] - 224) // 2, (size[1] - 224) // 2
    return GeometryPlan(size, (left, top, left + 224, top + 224))


def canonical_rgb_nhwc(image_path: str | Path) -> Any:
    """Decode exactly one RGB image with the P1 canonical geometry."""

    import numpy as np
    from PIL import Image

    path = Path(image_path)
    if not path.is_file():
        raise DINOv3RawContractError("image path must be an existing regular file")
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            plan = canonical_geometry_plan(*rgb.size)
            resized = rgb.resize(plan.resize_size, Image.Resampling.BICUBIC)
            cropped = resized.crop(plan.crop_box)
            pixels = np.ascontiguousarray(np.asarray(cropped), dtype=np.uint8)
    except DINOv3RawInjectionError:
        raise
    except Exception as exc:
        raise DINOv3RawContractError("image cannot be decoded with canonical geometry") from exc
    batch = np.ascontiguousarray(np.stack((pixels,)), dtype=np.uint8)
    if batch.shape != (1, 224, 224, 3) or batch.dtype != np.dtype(np.uint8):
        raise DINOv3RawContractError("canonical image geometry did not produce [1,224,224,3] uint8")
    return batch


def _dtype_name(value: object) -> str:
    return str(value).removeprefix("torch.")


def validate_dinov3_block11_patch(
    patch: object,
    *,
    finite_check: Callable[[object], bool],
) -> None:
    """Validate the immutable raw DINOv3 source tensor contract."""

    try:
        shape = tuple(int(item) for item in patch.shape)  # type: ignore[attr-defined]
        dtype = _dtype_name(patch.dtype)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise DINOv3RawContractError("DINOv3 patch must be tensor-like") from exc
    if shape != (1, 196, 768):
        raise DINOv3RawContractError(f"DINOv3 block_11 patch shape is {shape}, not (1, 196, 768)")
    if dtype != "float32":
        raise DINOv3RawContractError(f"DINOv3 block_11 patch dtype is {dtype!r}, not 'float32'")
    if not finite_check(patch):
        raise DINOv3RawContractError("DINOv3 block_11 patch contains NaN or Inf")


def _construct_dinov3_extractor(
    bindings: DINOv3RootBindings,
    *,
    profile_path: Path,
    runtime_profile_path: Path,
) -> object:
    from vision_encoder_alignment.extractors.dinov3 import (
        DINOv3ExtractorConfig,
        DINOv3VisionEncoderExtractor,
    )

    return DINOv3VisionEncoderExtractor(
        DINOv3ExtractorConfig(
            profile_path,
            runtime_root=bindings.runtime_root,
            snapshot_root=bindings.snapshot_root,
            execution_selector=DINO_SELECTOR,
            apple_runtime_profile_path=runtime_profile_path,
        )
    )


def extract_dinov3_block11(
    pixels: object,
    *,
    repository_root: str | Path,
    dinov3_runtime_root: str | Path,
    dinov3_snapshot_root: str | Path,
) -> Any:
    """Run the verified isolated DINOv3 extractor and return raw block-11 patches."""

    import numpy as np

    from vision_encoder_alignment.extractors.base import CanonicalImageBatch
    bindings = validate_dinov3_roots(
        repository_root=repository_root,
        runtime_root=dinov3_runtime_root,
        snapshot_root=dinov3_snapshot_root,
    )
    repository = bindings.repository_root
    profile = repository / DINO_PROFILE
    runtime_profile = repository / DINO_RUNTIME_PROFILE
    if not profile.is_file() or not runtime_profile.is_file():
        raise DINOv3RawContractError("verified Apple DINOv3 profiles are unavailable")
    batch = CanonicalImageBatch(("dinov3_raw_smoke",), pixels)
    try:
        with apple_dinov3_interpreter_compatibility(bindings):
            extractor = _construct_dinov3_extractor(
                bindings,
                profile_path=profile,
                runtime_profile_path=runtime_profile,
            )
            result = extractor.extract(batch, tap_names=(DINO_TAP,))
    except Exception as exc:
        raise DINOv3RawInjectionError("verified isolated DINOv3 block_11 extraction failed") from exc
    finally:
        # The extractor synchronously waits for, validates, and cleans up its worker.
        try:
            del extractor
        except UnboundLocalError:
            pass
    descriptor = result.descriptor
    if (
        descriptor.model_id != DINO_MODEL_ID
        or descriptor.revision != DINO_REVISION
        or descriptor.token_layout.special_token_indices != (0, 1, 2, 3, 4)
        or descriptor.token_layout.special_token_policy
        != "dinov3_cls1_register4_prefix_verified_runtime_v1"
    ):
        raise DINOv3RawContractError("DINOv3 extractor identity or token-removal policy mismatch")
    feature = result.get_tap(DINO_TAP)
    if feature.tap.name != DINO_TAP or tuple(result.image_ids) != ("dinov3_raw_smoke",):
        raise DINOv3RawContractError("DINOv3 extraction did not return exactly block_11 for one image")
    patch = np.ascontiguousarray(feature.patch, dtype=np.float32)
    validate_dinov3_block11_patch(patch, finite_check=lambda value: bool(np.isfinite(value).all()))
    return patch


def raw_numpy_to_torch(patch: object, *, torch: object) -> object:
    """Perform the sole permitted pre-connector raw-feature conversion."""

    return torch.from_numpy(patch)


def _stats(tensor: object, *, torch: object) -> dict[str, float]:
    value = tensor.detach().to(dtype=torch.float64)  # type: ignore[attr-defined]
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "l2_norm": float(torch.linalg.vector_norm(value).item()),
    }


def _repository_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or len(value) != 40:
        raise DINOv3RawContractError("repository commit is unavailable")
    return value


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    with path.open("xb") as stream:
        stream.write(payload)


def build_result(**values: object) -> dict[str, object]:
    """Build the successful, non-comparative raw-injection result document."""

    required = {
        "status", "repository_commit", "image_sha256", "geometry_id",
        "tinyllava", "dinov3", "native", "raw_dinov3", "diagnostics", "artifacts",
    }
    missing = sorted(required - set(values))
    unknown = sorted(set(values) - required)
    if missing or unknown:
        raise DINOv3RawContractError(
            f"result fields mismatch: missing={missing!r}, unknown={unknown!r}"
        )
    result = {"schema_id": RESULT_SCHEMA_ID, **values}
    native_replay.canonical_json_bytes(result)
    return result


def run_dinov3_raw_injection_smoke(
    *,
    tinyllava_config: str | Path,
    image_path: str | Path,
    cache_root: str | Path,
    repository_root: str | Path,
    dinov3_runtime_root: str | Path,
    dinov3_snapshot_root: str | Path,
    output_dir: str | Path,
    device: str,
    dtype: str,
) -> dict[str, object]:
    """Run one sequential DINOv3 extraction, native path, and raw injection path."""

    config = native_replay.load_anchor_config(tinyllava_config)
    image = Path(image_path)
    cache = Path(cache_root)
    bindings = validate_dinov3_roots(
        repository_root=repository_root,
        runtime_root=dinov3_runtime_root,
        snapshot_root=dinov3_snapshot_root,
    )
    repository = bindings.repository_root
    output = Path(output_dir)
    if not image.is_file() or not cache.is_dir() or not repository.is_dir():
        raise DINOv3RawContractError("image, cache root, and repository root must exist")
    if output.exists():
        raise FileExistsError("output directory must be fresh")
    if not device.startswith("cuda"):
        raise DINOv3RawContractError("Apple raw-injection smoke requires CUDA")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise DINOv3RawContractError("dtype must be float16, bfloat16, or float32")
    output.mkdir(parents=True, exist_ok=False)

    # This completes before TinyLLaVA imports or model construction begin.
    pixels = canonical_rgb_nhwc(image)
    dinov3_patch = extract_dinov3_block11(
        pixels,
        repository_root=repository,
        dinov3_runtime_root=bindings.runtime_root,
        dinov3_snapshot_root=bindings.snapshot_root,
    )

    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, ModuleNotFoundError) as exc:
        raise native_replay.RuntimeABIError("required TinyLLaVA runtime dependency is unavailable") from exc
    if not torch.cuda.is_available():
        raise DINOv3RawContractError("CUDA is unavailable")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    common = {
        "revision": config.revision,
        "cache_dir": str(cache),
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch_dtype, **common)
        maximum = model.config.tokenizer_model_max_length
        padding_side = model.config.tokenizer_padding_side
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=str(cache),
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
            use_fast=False,
            model_max_length=maximum,
            padding_side=padding_side,
        )
    except AttributeError as exc:
        raise native_replay.RuntimeABIError("pinned model config lacks tokenizer ABI fields") from exc
    except Exception as exc:
        raise native_replay.RuntimeABIError("exact local TinyLLaVA model or tokenizer could not be loaded") from exc
    model.to(device=device, dtype=torch_dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise DINOv3RawContractError("TinyLLaVA model is not fully frozen")
    vision_tower = (
        model.get_vision_tower()
        if callable(getattr(model, "get_vision_tower", None))
        else native_replay._resolve_component(model, label="vision tower", paths=("vision_tower", "model.vision_tower"))
    )
    connector = native_replay._resolve_component(
        model,
        label="connector",
        paths=("connector", "model.connector", "mm_projector", "model.mm_projector"),
    )
    resolved = native_replay._resolve_runtime_config(model, vision_tower, config)
    if any(parameter.requires_grad for parameter in vision_tower.parameters()):
        raise DINOv3RawContractError("TinyLLaVA vision tower is not frozen")
    native_kwargs = native_replay._chat_kwargs(
        model=model, tokenizer=tokenizer, image_path=image, config=config
    )
    raw_kwargs = native_replay._chat_kwargs(
        model=model, tokenizer=tokenizer, image_path=image, config=config
    )
    with torch.inference_mode():
        native_state, native_tokens, native_text = native_replay._run_chat_capture(
            model=model,
            vision_tower=vision_tower,
            connector=connector,
            chat=lambda: model.chat(**native_kwargs),
        )
    if native_state.vision_calls < 1 or native_state.connector_calls < 1:
        raise DINOv3RawContractError("native CLIP path did not invoke required components")
    if native_state.patch_tensor is None or native_state.connector_tensor is None:
        raise DINOv3RawContractError("native CLIP capture is incomplete")
    finite = lambda value: bool(torch.isfinite(value).all().item())
    native_replay.validate_tensor_contract(
        native_state.patch_tensor,
        expected_shape=(1, 196, 768), expected_dtype=dtype,
        finite_check=finite, label="native_clip_patch",
    )
    native_replay.validate_tensor_contract(
        native_state.connector_tensor,
        expected_shape=(1, 196, 1536), expected_dtype=dtype,
        finite_check=finite, label="native_connector",
    )
    raw_tensor = raw_numpy_to_torch(dinov3_patch, torch=torch)
    validate_dinov3_block11_patch(raw_tensor.cpu(), finite_check=finite)
    injection = native_replay.ReplayInjection(
        model, raw_tensor, connector, prepare_patch=native_replay._prepare_for_connector
    )
    with torch.inference_mode():
        with injection:
            raw_state, raw_tokens, raw_text = native_replay._run_chat_capture(
                model=model,
                vision_tower=vision_tower,
                connector=connector,
                chat=lambda: model.chat(**raw_kwargs),
            )
    if raw_state.vision_calls != 0 or raw_state.connector_calls < 1 or not injection.restored:
        raise DINOv3RawContractError("raw injection did not bypass CLIP vision or restore encode_images")
    if raw_state.connector_tensor is None:
        raise DINOv3RawContractError("raw DINOv3 connector capture is incomplete")
    native_replay.validate_tensor_contract(
        raw_state.connector_tensor,
        expected_shape=(1, 196, 1536), expected_dtype=dtype,
        finite_check=finite, label="raw_connector",
    )
    native_patch = native_state.patch_tensor.detach().cpu()
    raw_patch = raw_tensor.detach().cpu()
    native_connector = native_state.connector_tensor.detach().cpu()
    raw_connector = raw_state.connector_tensor.detach().cpu()
    native_stats = _stats(native_patch, torch=torch)
    dino_stats = _stats(raw_patch, torch=torch)
    diagnostics = {
        "native_clip_patch": native_stats,
        "dinov3_raw_patch": dino_stats,
        "patch_cosine_float64": native_replay._connector_cosine(native_patch, raw_patch, torch=torch),
        "patch_norm_ratio": dino_stats["l2_norm"] / native_stats["l2_norm"],
        "native_connector": _stats(native_connector, torch=torch),
        "dinov3_connector": _stats(raw_connector, torch=torch),
        "connector_cosine_float64": native_replay._connector_cosine(native_connector, raw_connector, torch=torch),
    }
    artifact = output / "input_patch_features.safetensors"
    save_file(
        {
            "native_clip_block_11": native_patch.contiguous(),
            "dinov3_block_11": raw_patch.contiguous(),
        },
        str(artifact),
    )
    result = build_result(
        status=STATUS,
        repository_commit=_repository_commit(repository),
        image_sha256=native_replay.sha256_file(image),
        geometry_id=GEOMETRY_ID,
        tinyllava={"model_id": config.model_id, "revision": config.revision, "runtime_config": resolved},
        dinov3={"model_id": DINO_MODEL_ID, "revision": DINO_REVISION, "tap": DINO_TAP, "source": "hidden_states[11]", "special_tokens_removed": "CLS 1 + registers 4"},
        native={"patch_shape": list(native_patch.shape), "patch_dtype": native_replay._dtype_name(native_patch.dtype), "vision_calls": native_state.vision_calls, "connector_calls": native_state.connector_calls, "token_ids": native_tokens, "answer": native_text},
        raw_dinov3={"patch_shape": list(raw_patch.shape), "patch_dtype": native_replay._dtype_name(raw_patch.dtype), "clip_vision_calls": raw_state.vision_calls, "connector_calls": raw_state.connector_calls, "encode_images_restored": injection.restored, "connector_shape": list(raw_connector.shape), "token_ids": raw_tokens, "answer": raw_text, "answer_equal_to_native": raw_text == native_text},
        diagnostics=diagnostics,
        artifacts={"input_patch_features": {"relative_path": artifact.name, "size_bytes": artifact.stat().st_size, "sha256": native_replay.sha256_file(artifact)}},
    )
    _write_new(output / "result.json", native_replay.canonical_json_bytes(result))
    return result


def write_blocker_result(output_dir: str | Path, *, stage: str, error: BaseException) -> Path | None:
    """Write bounded failure evidence only in a fresh task-owned output directory."""

    try:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "result.json"
        _write_new(path, native_replay.canonical_json_bytes({
            "schema_id": RESULT_SCHEMA_ID,
            "status": getattr(error, "classification", "BLOCKED_DINOV3_RAW_INJECTION"),
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
        }))
        return path
    except Exception:
        return None


__all__ = [
    "DINO_MODEL_ID", "DINO_REVISION", "DINO_TAP", "GEOMETRY_ID", "RESULT_SCHEMA_ID",
    "STATUS", "DINOv3RawContractError", "DINOv3RawInjectionError", "DINOv3RootBindings", "GeometryPlan",
    "apple_dinov3_interpreter_compatibility",
    "build_result", "canonical_geometry_plan", "canonical_rgb_nhwc", "extract_dinov3_block11",
    "raw_numpy_to_torch", "resolve_apple_dinov3_interpreter", "run_dinov3_raw_injection_smoke",
    "validate_dinov3_block11_patch", "validate_dinov3_roots",
    "write_blocker_result",
]
