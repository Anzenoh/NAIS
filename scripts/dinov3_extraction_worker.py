"""Offline one-forward DINOv3 extraction worker for the persistent runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import socket
import stat
import sys
from typing import Any, Mapping


MODEL_KEY = "dinov3_vit_b16"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
MODEL_STATE_SHA256 = "c382afd66ac658255f9f1406bad1e5e5fcf410af40cbc5c306b770792a4f30fd"
EXPECTED_PARAMETER_COUNT = 85_660_416
EXPECTED_PARAMETER_TENSOR_COUNT = 211
LEGACY_EXECUTION_SELECTOR = "legacy_windows_cpu"
APPLE_EXECUTION_SELECTOR = "dinov3_apple_cuda_fp32_v1"
TAP_CATALOG = (
    "embedding_output",
    "block_01",
    "block_02",
    "block_03",
    "block_04",
    "block_05",
    "block_06",
    "block_07",
    "block_08",
    "block_09",
    "block_10",
    "block_11",
    "block_12",
    "final_norm",
)
MARKERS = (
    "DINOV3_REAL_EXTRACTOR_PROFILE_OK",
    "DINOV3_REAL_EXTRACTOR_DESCRIPTOR_OK",
    "DINOV3_REAL_EXTRACTOR_ALL_TAPS_OK",
    "DINOV3_REAL_EXTRACTOR_ARTIFACT_BOUNDARY_OK",
    "DINOV3_REAL_EXTRACTOR_RNG_STATE_OK",
    "DINOV3_REAL_EXTRACTOR_NETWORK_GUARD_OK",
)
_SECRET_ENV_FRAGMENTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
    "API_KEY",
    "PRIVATE_KEY",
)


class WorkerValidationError(RuntimeError):
    """Raised when the worker cannot prove the exact extraction contract."""


class _SocketGuard:
    def __init__(self) -> None:
        self.attempt_count = 0
        self._socket = socket.socket
        self._create_connection = socket.create_connection
        self._getaddrinfo = socket.getaddrinfo

    def install(self) -> None:
        guard = self
        original = self._socket

        class GuardedSocket(original):  # type: ignore[misc, valid-type]
            def connect(self, address: object) -> None:
                guard.attempt_count += 1
                raise RuntimeError("network is disabled for DINOv3 extraction")

            def connect_ex(self, address: object) -> int:
                guard.attempt_count += 1
                raise RuntimeError("network is disabled for DINOv3 extraction")

        def guarded_create_connection(*args: object, **kwargs: object) -> None:
            guard.attempt_count += 1
            raise RuntimeError("network is disabled for DINOv3 extraction")

        def guarded_getaddrinfo(*args: object, **kwargs: object) -> None:
            guard.attempt_count += 1
            raise RuntimeError("network is disabled for DINOv3 extraction")

        socket.socket = GuardedSocket
        socket.create_connection = guarded_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]

    def restore(self) -> None:
        socket.socket = self._socket
        socket.create_connection = self._create_connection
        socket.getaddrinfo = self._getaddrinfo


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and attributes & flag)


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolved_directory(path: Path, field: str) -> Path:
    if not path.is_absolute() or _is_reparse(path):
        raise WorkerValidationError(f"{field} must be an absolute non-link directory")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise WorkerValidationError(f"{field} is not a directory")
    return resolved


def _workspace_path(workspace: Path, value: str | Path, field: str, *, must_exist: bool) -> Path:
    text = str(value)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        not text
        or "\\" in text
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} or ":" in part for part in text.split("/"))
    ):
        raise WorkerValidationError(f"{field} is not a safe relative path")
    current = workspace
    for part in text.split("/"):
        current = current / part
        if current.exists() or current.is_symlink():
            if _is_reparse(current):
                raise WorkerValidationError(f"{field} crosses a reparse point")
    resolved = current.resolve(strict=must_exist)
    if not _contains(workspace, resolved):
        raise WorkerValidationError(f"{field} escapes the workspace")
    if must_exist and not resolved.is_file():
        raise WorkerValidationError(f"{field} is not a regular file")
    return resolved


def _validate_environment(
    workspace: Path,
    execution_selector: str = LEGACY_EXECUTION_SELECTOR,
) -> None:
    if execution_selector not in {LEGACY_EXECUTION_SELECTOR, APPLE_EXECUTION_SELECTOR}:
        raise WorkerValidationError("unknown execution selector")
    required = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "CUDA_VISIBLE_DEVICES": "0" if execution_selector == APPLE_EXECUTION_SELECTOR else "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, expected in required.items():
        if os.environ.get(key) != expected:
            raise WorkerValidationError(f"offline environment variable {key} is invalid")
    for key in os.environ:
        if key not in {"HF_HUB_DISABLE_IMPLICIT_TOKEN", "TOKENIZERS_PARALLELISM"} and any(
            fragment in key.upper() for fragment in _SECRET_ENV_FRAGMENTS
        ):
            raise WorkerValidationError("secret-like environment key survived sanitization")
    for key in ("HF_HOME", "XDG_CACHE_HOME", "TEMP", "TMP"):
        value = os.environ.get(key)
        if not isinstance(value, str) or not value:
            raise WorkerValidationError(f"scratch variable {key} is absent")
        resolved = Path(value).resolve(strict=False)
        if not _contains(workspace, resolved):
            raise WorkerValidationError(f"scratch variable {key} escapes workspace")


def _assert_import_light() -> None:
    forbidden = ("numpy", "torch", "transformers", "PIL", "safetensors", "torchvision")
    loaded = [name for name in sys.modules if name.split(".", 1)[0] in forbidden]
    if loaded:
        raise WorkerValidationError("heavy runtime imported before socket guard")


def _torchvision_presence() -> tuple[bool, bool, bool]:
    """Observe distribution, module-spec, and imported-module views."""

    try:
        importlib.metadata.distribution("torchvision")
    except importlib.metadata.PackageNotFoundError:
        distribution_present = False
    else:
        distribution_present = True
    spec_present = importlib.util.find_spec("torchvision") is not None
    imported_present = any(
        name == "torchvision" or name.startswith("torchvision.") for name in sys.modules
    )
    return distribution_present, spec_present, imported_present


def _execution_device(torch: Any, execution_selector: str) -> str:
    if execution_selector == LEGACY_EXECUTION_SELECTOR:
        if (
            getattr(torch.version, "cuda", None) is not None
            or torch.cuda.is_available()
            or torch.cuda.device_count() != 0
        ):
            raise WorkerValidationError("legacy worker requires the exact CPU-only runtime")
        return "cpu"
    if execution_selector != APPLE_EXECUTION_SELECTOR:
        raise WorkerValidationError("unknown execution selector")
    if (
        not isinstance(getattr(torch.version, "cuda", None), str)
        or not torch.version.cuda
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090"
    ):
        raise WorkerValidationError("Apple worker requires one exact RTX 5090 CUDA device")
    return "cuda:0"


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    import numpy as np

    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _manual_preprocess(images: Any) -> Any:
    import numpy as np

    normalized = images.astype(np.float32, copy=True)
    normalized *= np.float32(1.0 / 255.0)
    normalized -= np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    normalized /= np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = np.ascontiguousarray(normalized.transpose(0, 3, 1, 2), dtype=np.float32)
    if normalized.dtype != np.dtype(np.float32) or not normalized.flags.c_contiguous:
        raise WorkerValidationError("manual preprocessing dtype or layout mismatch")
    if not np.isfinite(normalized).all():
        raise WorkerValidationError("manual preprocessing produced NaN or Inf")
    return normalized


def _load_model(
    snapshot_files: Path,
    execution_selector: str = LEGACY_EXECUTION_SELECTOR,
) -> tuple[Any, dict[str, int]]:
    import torch
    from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

    device = _execution_device(torch, execution_selector)
    fork_devices = [0] if device == "cuda:0" else []
    with torch.random.fork_rng(devices=fork_devices, enabled=True):
        config = DINOv3ViTConfig.from_pretrained(
            str(snapshot_files),
            local_files_only=True,
            trust_remote_code=False,
            token=False,
        )
        loaded = DINOv3ViTModel.from_pretrained(
            str(snapshot_files),
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            token=False,
            use_safetensors=True,
            dtype=torch.float32,
            output_loading_info=True,
        )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise WorkerValidationError("model loader did not return loading information")
    model, loading = loaded
    if type(config).__name__ != "DINOv3ViTConfig" or type(model).__name__ != "DINOv3ViTModel":
        raise WorkerValidationError("model/config class mismatch")
    if not isinstance(loading, Mapping):
        raise WorkerValidationError("loading information is not an object")
    counts = {
        "error_message_count": len(loading.get("error_msgs", ())),
        "mismatched_key_count": len(loading.get("mismatched_keys", ())),
        "missing_key_count": len(loading.get("missing_keys", ())),
        "unexpected_key_count": len(loading.get("unexpected_keys", ())),
    }
    if any(counts.values()):
        raise WorkerValidationError("checkpoint loading information is not clean")
    model.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    parameters = tuple(model.parameters())
    if sum(item.numel() for item in parameters) != EXPECTED_PARAMETER_COUNT:
        raise WorkerValidationError("parameter count mismatch")
    if len(parameters) != EXPECTED_PARAMETER_TENSOR_COUNT:
        raise WorkerValidationError("parameter tensor count mismatch")
    if any(str(item.device) != device or item.dtype != torch.float32 for item in parameters):
        raise WorkerValidationError("model is not uniformly on the selected FP32 device")
    if model.training or any(item.requires_grad for item in parameters):
        raise WorkerValidationError("model is not frozen in evaluation mode")
    if len(model.layer) != 12 or model.norm is None:
        raise WorkerValidationError("required DINOv3 component is missing")
    return model, counts


def _one_forward(
    model: Any,
    normalized: Any,
    execution_selector: str = LEGACY_EXECUTION_SELECTOR,
) -> tuple[dict[str, Any], Any]:
    import torch

    captured: list[Any] = []

    def capture_raw_final(module: Any, arguments: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise WorkerValidationError("raw block_12 hook did not return a tensor")
        captured.append(output.detach().clone())

    handle = model.layer[11].register_forward_hook(capture_raw_final)
    try:
        with torch.inference_mode():
            device = _execution_device(torch, execution_selector)
            pixel_values = torch.from_numpy(normalized).to(device=device, dtype=torch.float32)
            outputs = model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise WorkerValidationError("raw block_12 hook count is not one")
    hidden = outputs.hidden_states
    last = outputs.last_hidden_state
    raw_final = captured[0]
    expected = (normalized.shape[0], 201, 768)
    if not isinstance(hidden, tuple) or len(hidden) != 13:
        raise WorkerValidationError("hidden-state count mismatch")
    if any(tuple(item.shape) != expected for item in hidden):
        raise WorkerValidationError("hidden-state shape mismatch")
    if tuple(raw_final.shape) != expected or tuple(last.shape) != expected:
        raise WorkerValidationError("final sequence shape mismatch")
    with torch.inference_mode():
        explicit_norm = model.norm(raw_final)
    if not torch.equal(hidden[12], last) or not torch.equal(explicit_norm, last):
        raise WorkerValidationError("final norm mapping mismatch")
    if torch.equal(raw_final, last):
        raise WorkerValidationError("raw block_12 was confused with final norm")
    sequences = {"embedding_output": hidden[0]}
    sequences.update({f"block_{index:02d}": hidden[index] for index in range(1, 12)})
    sequences["block_12"] = raw_final
    sequences["final_norm"] = last
    if tuple(sequences) != TAP_CATALOG:
        raise WorkerValidationError("canonical tap mapping order mismatch")
    expected_device = "cuda:0" if execution_selector == APPLE_EXECUTION_SELECTOR else "cpu"
    if any(
        str(value.device) != expected_device
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all().item())
        for value in sequences.values()
    ):
        raise WorkerValidationError("canonical full sequence is invalid")
    return sequences, outputs.pooler_output


def _array_sha256(value: Any) -> str:
    import numpy as np

    return _sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def _write_artifacts(
    *,
    workspace: Path,
    output_directory: Path,
    run_id: str,
    image_ids: tuple[str, ...],
    requested_taps: tuple[str, ...],
    sequences: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from datetime import datetime, timezone

    import numpy as np

    from vision_encoder_alignment.feature_artifacts import (
        ARTIFACT_SCHEMA_ID,
        FeatureArtifactMetadata,
        canonical_tensor_keys,
        read_feature_artifact,
        write_feature_artifact,
    )

    if output_directory.exists():
        raise WorkerValidationError("output directory must be fresh")
    output_directory.mkdir()
    entries: list[dict[str, Any]] = []
    for index, tap_name in enumerate(requested_taps):
        sequence = np.array(
            sequences[tap_name].detach().cpu().contiguous().numpy(),
            dtype=np.float32,
            copy=True,
            order="C",
        )
        if sequence.shape != (len(image_ids), 201, 768):
            raise WorkerValidationError("full sequence shape mismatch")
        patch = np.ascontiguousarray(sequence[:, 5:201, :], dtype=np.float32)
        global_mean = np.ascontiguousarray(patch.mean(axis=1, dtype=np.float32))
        if patch.shape != (len(image_ids), 196, 768) or global_mean.shape != (len(image_ids), 768):
            raise WorkerValidationError("canonical feature shape mismatch")
        directory_name = f"{index:02d}-{tap_name}"
        directory = output_directory / directory_name
        tensor_name = f"shard-{index:05d}.safetensors"
        sidecar_name = f"shard-{index:05d}.json"
        patch_key, global_key = canonical_tensor_keys(tap_name)
        metadata = FeatureArtifactMetadata(
            schema_id=ARTIFACT_SCHEMA_ID,
            run_id=run_id,
            model_key=MODEL_KEY,
            tap_namespace="backbone",
            tap_name=tap_name,
            patch_tensor_key=patch_key,
            global_mean_tensor_key=global_key,
            image_ids=list(image_ids),
            grid_height=14,
            grid_width=14,
            patch_count=196,
            hidden_dimension=768,
            dtype="float32",
            shard_index=index,
            safetensors_file_name=tensor_name,
            sidecar_file_name=sidecar_name,
            checksum_algorithm="sha256",
            checksum_value="",
            created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            run_status="incomplete",
        )
        completed = write_feature_artifact(directory, metadata, patch, global_mean)
        sidecar = directory / sidecar_name
        tensors = directory / tensor_name
        loaded = read_feature_artifact(sidecar)
        if not np.array_equal(loaded.patch, patch) or not np.array_equal(loaded.global_mean, global_mean):
            raise WorkerValidationError("feature artifact round-trip mismatch")
        relative_prefix = PurePosixPath("output") / directory_name
        entries.append(
            {
                "checksum_sha256": completed.checksum_value,
                "dtype": "float32",
                "global_sha256": _array_sha256(loaded.global_mean),
                "global_shape": list(loaded.global_mean.shape),
                "patch_sha256": _array_sha256(loaded.patch),
                "patch_shape": list(loaded.patch.shape),
                "safetensors_relative_path": str(relative_prefix / tensor_name),
                "sidecar_relative_path": str(relative_prefix / sidecar_name),
                "tap_name": tap_name,
            }
        )
    return entries


def _write_response(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_worker(
    *,
    extractor_profile_path: Path,
    runtime_profile_path: Path,
    snapshot_profile_path: Path,
    runtime_root: Path,
    snapshot_root: Path,
    workspace: Path,
    request_path: Path,
    response_path: Path,
    execution_selector: str = LEGACY_EXECUTION_SELECTOR,
) -> dict[str, Any]:
    """Execute exact local checkpoint extraction with one model forward."""

    workspace = _resolved_directory(workspace, "workspace_root")
    repository = _resolved_directory(Path(__file__).resolve().parents[1], "repository_root")
    runtime_root = _resolved_directory(runtime_root, "runtime_root")
    snapshot_root = _resolved_directory(snapshot_root, "snapshot_root")
    if any(
        _contains(left, right) or _contains(right, left)
        for index, left in enumerate((repository, runtime_root, snapshot_root, workspace))
        for right in (repository, runtime_root, snapshot_root, workspace)[index + 1 :]
    ):
        raise WorkerValidationError("protected roots overlap")
    request_path = _workspace_path(workspace, request_path, "request", must_exist=True)
    response_path = _workspace_path(workspace, response_path, "response", must_exist=False)
    if response_path.exists():
        raise WorkerValidationError("response path must be fresh")
    _validate_environment(workspace, execution_selector)
    _assert_import_light()
    if any(_torchvision_presence()):
        raise WorkerValidationError("torchvision is present in the isolated worker view")
    guard = _SocketGuard()
    guard.install()
    try:
        from vision_encoder_alignment.dinov3_extraction_boundary import (
            load_dinov3_extractor_profile,
            read_dinov3_extraction_request,
            sha256_file,
        )
        from vision_encoder_alignment.dinov3_production_identity import (
            validate_production_sources,
        )
        from vision_encoder_alignment.snapshot_boundary import (
            load_snapshot_profile,
            snapshot_files_directory,
            validate_apple_hf_snapshot,
            validate_snapshot_root,
        )

        profile = (
            load_dinov3_extractor_profile(
                extractor_profile_path,
                execution_selector=APPLE_EXECUTION_SELECTOR,
            )
            if execution_selector == APPLE_EXECUTION_SELECTOR
            else load_dinov3_extractor_profile(extractor_profile_path)
        )
        validate_production_sources(repository, extractor_profile_path)
        request = (
            read_dinov3_extraction_request(
                request_path,
                execution_selector=APPLE_EXECUTION_SELECTOR,
            )
            if execution_selector == APPLE_EXECUTION_SELECTOR
            else read_dinov3_extraction_request(request_path)
        )
        if request.extractor_profile_sha256 != profile.sha256:
            raise WorkerValidationError("extractor profile SHA-256 mismatch")
        snapshot_profile = load_snapshot_profile(snapshot_profile_path)
        snapshot_manifest = (
            validate_apple_hf_snapshot(
                snapshot_root,
                cache_root=snapshot_root.parents[2],
                model_id=MODEL_ID,
                revision=REVISION,
                files=profile.checkpoint_files,
            )
            if execution_selector == APPLE_EXECUTION_SELECTOR
            else validate_snapshot_root(
                snapshot_root,
                snapshot_profile,
                repository_root=repository,
                persistent_runtime_root=runtime_root,
            )
        )
        identities = profile.identities
        if execution_selector == APPLE_EXECUTION_SELECTOR:
            from vision_encoder_alignment.production_runtime_contract import (
                DINOV3_LANE_ID,
                DINOV3_SELECTOR,
                load_production_runtime_profile,
                validate_profile_source_bindings,
            )

            runtime_profile = load_production_runtime_profile(runtime_profile_path)
            validate_profile_source_bindings(runtime_profile, repository)
            if (
                not runtime_profile.package_metadata_resolved
                or runtime_profile.lane_id != DINOV3_LANE_ID
                or runtime_profile.selector != DINOV3_SELECTOR
                or runtime_profile.runtime_root != runtime_root.as_posix()
                or runtime_profile.interpreter_layout != "linux_bin"
                or runtime_profile.system_site_packages
                or runtime_profile.device != "cuda:0"
                or runtime_profile.dtype != "float32"
                or runtime_profile.cpu_fallback
                or runtime_profile.mixed_precision
                or runtime_profile.network_allowed
                or runtime_profile.ready_for_p1_extraction
            ):
                raise WorkerValidationError("Apple production runtime contract mismatch")
            actual_identities = {
                "registry_sha256": sha256_file(repository / "configs/models.yaml"),
                "runtime_profile_sha256": runtime_profile.source_sha256,
                "snapshot_profile_sha256": snapshot_profile.source_sha256,
                "semantic_profile_sha256": sha256_file(
                    repository / str(profile.sources.semantic_profile_path)
                ),
                "semantic_evidence_sha256": sha256_file(
                    repository / str(profile.sources.semantic_evidence_path)
                ),
                "external_snapshot_manifest_sha256": snapshot_manifest.source_sha256,
            }
        else:
            from vision_encoder_alignment.runtime_boundary import (
                load_runtime_profile,
                validate_runtime_root,
            )

            runtime_profile = load_runtime_profile(runtime_profile_path)
            runtime_manifest = validate_runtime_root(
                runtime_root, runtime_profile, repository_root=repository
            )
            actual_identities = {
                "registry_sha256": sha256_file(repository / "configs/models.yaml"),
                "runtime_profile_sha256": runtime_profile.source_sha256,
                "snapshot_profile_sha256": snapshot_profile.source_sha256,
                "semantic_profile_sha256": sha256_file(
                    repository / str(profile.sources.semantic_profile_path)
                ),
                "semantic_evidence_sha256": sha256_file(
                    repository / str(profile.sources.semantic_evidence_path)
                ),
                "persistent_runtime_manifest_sha256": runtime_manifest.source_sha256,
                "external_snapshot_manifest_sha256": snapshot_manifest.source_sha256,
                "persistent_freeze_sha256": runtime_manifest.candidate_freeze_sha256,
            }
        for key, actual in actual_identities.items():
            if identities[key] != actual:
                raise WorkerValidationError(f"live identity mismatch: {key}")
        input_path = _workspace_path(
            workspace, request.workspace_relative_input_path, "input", must_exist=True
        )
        output = _workspace_path(
            workspace,
            request.workspace_relative_output_directory,
            "output",
            must_exist=False,
        )
        if output.exists():
            raise WorkerValidationError("output directory must be fresh")
        input_file_before = _sha256_file(input_path)
        if input_file_before != request.input_file_sha256:
            raise WorkerValidationError("input file SHA-256 mismatch")

        import numpy as np
        import torch

        if any(name == "torchvision" or name.startswith("torchvision.") for name in sys.modules):
            raise WorkerValidationError("torchvision imported before extraction")
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()
        torch_rng_before = torch.random.get_rng_state().clone()
        images = np.load(input_path, allow_pickle=False)
        if (
            not isinstance(images, np.ndarray)
            or images.dtype != np.dtype(np.uint8)
            or tuple(images.shape) != request.input_shape
            or not images.flags.c_contiguous
        ):
            raise WorkerValidationError("input array contract mismatch")
        input_bytes_before = images.tobytes(order="C")
        if _sha256_bytes(input_bytes_before) != request.raw_input_sha256:
            raise WorkerValidationError("raw input SHA-256 mismatch")
        normalized = _manual_preprocess(images)
        normalized_sha256 = _sha256_bytes(normalized.tobytes(order="C"))
        snapshot_files = (
            snapshot_root
            if execution_selector == APPLE_EXECUTION_SELECTOR
            else snapshot_files_directory(snapshot_root)
        )
        if execution_selector == LEGACY_EXECUTION_SELECTOR:
            model, _ = _load_model(snapshot_files)
        else:
            model, _ = _load_model(snapshot_files, execution_selector)
        state_before = _state_sha256(model)
        if state_before != MODEL_STATE_SHA256:
            raise WorkerValidationError("model-state SHA-256 mismatch")
        if execution_selector == LEGACY_EXECUTION_SELECTOR:
            sequences, pooler_output = _one_forward(model, normalized)
        else:
            sequences, pooler_output = _one_forward(model, normalized, execution_selector)
        if pooler_output is None or tuple(pooler_output.shape) != (len(request.image_ids), 768):
            raise WorkerValidationError("native pooler shape mismatch")
        entries = _write_artifacts(
            workspace=workspace,
            output_directory=output,
            run_id=request.run_id,
            image_ids=request.image_ids,
            requested_taps=request.requested_tap_names,
            sequences=sequences,
        )
        state_after = _state_sha256(model)
        python_rng_after = random.getstate()
        numpy_rng_after = np.random.get_state()
        torch_rng_after = torch.random.get_rng_state().clone()
        invariants = {
            "model_state_unchanged": state_before == state_after == MODEL_STATE_SHA256,
            "numpy_rng_unchanged": _numpy_rng_equal(numpy_rng_before, numpy_rng_after),
            "parameters_frozen": not any(item.requires_grad for item in model.parameters()),
            "python_rng_unchanged": python_rng_before == python_rng_after,
            "torch_rng_unchanged": bool(torch.equal(torch_rng_before, torch_rng_after)),
            "training_false": not bool(model.training),
        }
        if not all(invariants.values()):
            raise WorkerValidationError("model or RNG invariant failed")
        if images.tobytes(order="C") != input_bytes_before or _sha256_file(input_path) != input_file_before:
            raise WorkerValidationError("input array or file changed")
        torchvision_count = sum(_torchvision_presence())
        if torchvision_count != 0:
            raise WorkerValidationError("torchvision became present during extraction")
        if guard.attempt_count != 0:
            raise WorkerValidationError("network operation was attempted")
        response = {
            "artifacts": entries,
            "canonical_mapping": {
                "block_12_source": "raw_final_block_hook",
                "final_norm_source": "last_hidden_state",
                "global_source": "patch_axis_mean",
                "hidden_state_12_relation": "final_norm",
                "native_pooler_excluded": True,
                "patch_token_indices": list(range(5, 201)),
                "raw_block_12_differs_from_final_norm": True,
                "special_tokens_excluded": True,
            },
            "descriptor": {
                "grid_height": 14,
                "grid_width": 14,
                "hidden_dimension": 768,
                "layer_count": 12,
                "output_dtype": "float32",
                "patch_count": 196,
                "total_token_count": 201,
            },
            "failure_classification": None,
            "forward_count": 1,
            "identities": {
                "external_snapshot_manifest_sha256": snapshot_manifest.source_sha256,
                "extractor_profile_sha256": profile.sha256,
                "persistent_freeze_sha256": profile.identities.persistent_freeze_sha256,
                "persistent_runtime_manifest_sha256": profile.identities.persistent_runtime_manifest_sha256,
                "runtime_profile_sha256": runtime_profile.source_sha256,
                "snapshot_profile_sha256": snapshot_profile.source_sha256,
            },
            "image_ids": list(request.image_ids),
            "input": {
                "input_file_sha256": input_file_before,
                "input_unchanged": True,
                "normalized_dtype": "float32",
                "normalized_sha256": normalized_sha256,
                "normalized_shape": list(normalized.shape),
                "raw_input_sha256": request.raw_input_sha256,
            },
            "invariants": invariants,
            "markers": list(MARKERS),
            "model_state_sha256": state_after,
            "operation": "extract_canonical_features",
            "processor_construction_count": 0,
            "requested_tap_names": list(request.requested_tap_names),
            "run_id": request.run_id,
            "schema_id": "vea.dinov3_extraction_response.v0",
            "socket_attempt_count": guard.attempt_count,
            "status": "success",
            "torchvision_import_count": torchvision_count,
            "warnings": [],
            "workspace_paths": [
                path
                for entry in entries
                for path in (
                    entry["sidecar_relative_path"],
                    entry["safetensors_relative_path"],
                )
            ],
        }
        from vision_encoder_alignment.dinov3_extraction_boundary import (
            validate_dinov3_extraction_response,
        )

        validate_dinov3_extraction_response(response, request=request, profile=profile)
        _write_response(response_path, response)
        return response
    finally:
        guard.restore()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor-profile", required=True, type=Path)
    parser.add_argument("--runtime-profile", required=True, type=Path)
    parser.add_argument("--snapshot-profile", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument(
        "--execution-selector",
        choices=(LEGACY_EXECUTION_SELECTOR, APPLE_EXECUTION_SELECTOR),
        default=LEGACY_EXECUTION_SELECTOR,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        response = run_worker(
            extractor_profile_path=args.extractor_profile.resolve(strict=True),
            runtime_profile_path=args.runtime_profile.resolve(strict=True),
            snapshot_profile_path=args.snapshot_profile.resolve(strict=True),
            runtime_root=args.runtime_root.resolve(strict=True),
            snapshot_root=args.snapshot_root.resolve(strict=True),
            workspace=args.workspace_root.resolve(strict=True),
            request_path=args.request,
            response_path=args.response,
            execution_selector=args.execution_selector,
        )
    except Exception as exc:
        print(f"DINOV3_EXTRACTION_WORKER_FAILED:{type(exc).__name__}", file=sys.stderr)
        return 1
    for marker in response["markers"]:
        print(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
