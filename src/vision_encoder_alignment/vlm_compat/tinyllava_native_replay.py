"""Offline-only frozen TinyLLaVA native patch capture and replay.

This module intentionally imports only the Python standard library. Runtime
model dependencies are loaded inside :func:`run_native_replay`.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import math
import subprocess
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


ANCHOR_SCHEMA_ID = "vea.p2.vlm_anchor.v1"
RESULT_SCHEMA_ID = "vea.p2.t001a.native_replay.v1"
MODEL_TYPE = "tinyllava"
PATCH_TENSOR_KEY = "patch_features"
_CONNECTOR_COSINE_THRESHOLD = 0.999999


class NativeReplayError(RuntimeError):
    """Base exception for the bounded native replay harness."""

    classification = "FAILED_NATIVE_REPLAY_ASSERTION"


class AnchorConfigError(NativeReplayError, ValueError):
    """Raised when the pinned anchor document is not exact."""

    classification = "BLOCKED_INVALID_ANCHOR_CONFIG"


class RuntimeABIError(NativeReplayError):
    """Raised when the pinned TinyLLaVA runtime lacks a required interface."""

    classification = "BLOCKED_TINYLLAVA_RUNTIME_ABI"


class ReplayContractError(NativeReplayError):
    """Raised when capture, serialization, or replay violates the contract."""


@dataclass(frozen=True, slots=True)
class AnchorConfig:
    schema_id: str
    model_id: str
    revision: str
    trust_remote_code: bool
    local_files_only: bool
    expected_vision_model: str
    expected_vision_feature_layer: int
    expected_vision_feature_select_strategy: str
    expected_connector_type: str
    expected_image_size: int
    expected_patch_size: int
    expected_patch_count: int
    expected_vision_hidden_size: int
    expected_connector_output_size: int
    expected_vision_tower_frozen: bool
    prompt: str
    max_new_tokens: int
    do_sample: bool
    num_beams: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AnchorConfig":
        expected = set(cls.__dataclass_fields__)
        actual = set(value)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing or unknown:
            raise AnchorConfigError(
                f"anchor fields mismatch: missing={missing!r}, unknown={unknown!r}"
            )
        try:
            result = cls(**value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise AnchorConfigError("anchor fields have invalid types") from exc
        result.validate()
        return result

    def validate(self) -> None:
        fixed: Mapping[str, object] = {
            "schema_id": ANCHOR_SCHEMA_ID,
            "model_id": "jiajunlong/TinyLLaVA-OpenELM-450M-CLIP-0.55B",
            "revision": "0bff7b92d5b9ef4520c61b1f7872d7d1e13537a6",
            "trust_remote_code": True,
            "local_files_only": True,
            "expected_vision_model": "openai/clip-vit-base-patch16",
            "expected_vision_feature_layer": -2,
            "expected_vision_feature_select_strategy": "patch",
            "expected_connector_type": "mlp2x_gelu",
            "expected_image_size": 224,
            "expected_patch_size": 16,
            "expected_patch_count": 196,
            "expected_vision_hidden_size": 768,
            "expected_connector_output_size": 1536,
            "expected_vision_tower_frozen": True,
            "prompt": "What is shown in this image? Answer briefly.",
            "max_new_tokens": 32,
            "do_sample": False,
            "num_beams": 1,
        }
        for name, expected in fixed.items():
            actual = getattr(self, name)
            if type(actual) is not type(expected) or actual != expected:
                raise AnchorConfigError(
                    f"anchor.{name}: expected {expected!r}, found {actual!r}"
                )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _duplicate_key_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise AnchorConfigError(f"anchor JSON has duplicate key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise AnchorConfigError(f"anchor JSON has non-finite constant {value!r}")


def load_anchor_config(path: str | Path) -> AnchorConfig:
    """Load the exact pinned anchor using only the standard library."""

    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_duplicate_key_hook,
            parse_constant=_reject_constant,
        )
    except AnchorConfigError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnchorConfigError("anchor config is unavailable or malformed") from exc
    if not isinstance(value, dict):
        raise AnchorConfigError("anchor config must be a JSON object")
    return AnchorConfig.from_mapping(value)


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic publication JSON bytes."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReplayContractError("result contains a non-JSON value") from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReplayContractError("unable to hash required file") from exc
    return digest.hexdigest()


def _dtype_name(value: object) -> str:
    return str(value).removeprefix("torch.")


def validate_tensor_contract(
    tensor: object,
    *,
    expected_shape: Sequence[int],
    expected_dtype: str | None,
    finite_check: Callable[[object], bool],
    label: str,
) -> None:
    """Validate a tensor-like object without importing a tensor library."""

    try:
        shape = tuple(int(item) for item in tensor.shape)  # type: ignore[attr-defined]
        dtype = _dtype_name(tensor.dtype)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ReplayContractError(f"{label}: expected a tensor") from exc
    expected = tuple(int(item) for item in expected_shape)
    if shape != expected:
        raise ReplayContractError(f"{label}.shape: expected {expected}, found {shape}")
    if expected_dtype is not None and dtype != expected_dtype:
        raise ReplayContractError(
            f"{label}.dtype: expected {expected_dtype!r}, found {dtype!r}"
        )
    if not finite_check(tensor):
        raise ReplayContractError(f"{label}: contains NaN or Inf")


def validate_serialization_round_trip(
    original: object,
    reloaded: object,
    *,
    finite_check: Callable[[object], bool],
    max_abs_diff: Callable[[object, object], float],
) -> dict[str, object]:
    """Validate exact shape, dtype, finite values, and element equality."""

    original_shape = tuple(int(item) for item in original.shape)  # type: ignore[attr-defined]
    original_dtype = _dtype_name(original.dtype)  # type: ignore[attr-defined]
    validate_tensor_contract(
        reloaded,
        expected_shape=original_shape,
        expected_dtype=original_dtype,
        finite_check=finite_check,
        label="serialized_patch",
    )
    difference = float(max_abs_diff(original, reloaded))
    if not math.isfinite(difference) or difference != 0.0:
        raise ReplayContractError(
            f"serialized_patch.max_abs_diff: expected 0, found {difference!r}"
        )
    return {
        "shape_equal": True,
        "dtype_equal": True,
        "all_finite": True,
        "max_abs_diff": difference,
    }


@dataclass(slots=True)
class CaptureState:
    vision_calls: int = 0
    connector_calls: int = 0
    patch_tensor: object | None = None
    connector_tensor: object | None = None
    vision_output_type: str | None = None
    hidden_state_tuple_length: int | None = None

    @staticmethod
    def _clone(value: object) -> object:
        try:
            return value.detach().clone()  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise ReplayContractError("hook output is not tensor-like") from exc

    def vision_hook(self, _module: object, _args: object, output: object) -> None:
        self.vision_calls += 1
        self.vision_output_type = type(output).__name__
        hidden_states = getattr(output, "hidden_states", None)
        if isinstance(hidden_states, (tuple, list)):
            self.hidden_state_tuple_length = len(hidden_states)

    def connector_pre_hook(self, _module: object, args: object) -> None:
        if not isinstance(args, tuple) or not args:
            raise ReplayContractError("connector hook did not receive patch features")
        self.patch_tensor = self._clone(args[0])

    def connector_hook(self, _module: object, _args: object, output: object) -> None:
        self.connector_calls += 1
        self.connector_tensor = self._clone(output)


def _same_bound_callable(left: object, right: object) -> bool:
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", left) is getattr(right, "__func__", right)
    )


class ReplayInjection:
    """Temporarily route one model instance's ``encode_images`` to a connector."""

    def __init__(
        self,
        model: object,
        patch_tensor: object,
        connector: Callable[[object], object],
        *,
        prepare_patch: Callable[[object, object], object] | None = None,
    ) -> None:
        self.model = model
        self.patch_tensor = patch_tensor
        self.connector = connector
        self.prepare_patch = prepare_patch or (lambda patch, _connector: patch)
        self.invocations = 0
        self.restored = False
        self._entered = False
        self._had_instance_attribute = False
        self._instance_value: object | None = None
        self._original: object | None = None

    def __enter__(self) -> "ReplayInjection":
        if self._entered:
            raise ReplayContractError("replay injection cannot be entered twice")
        try:
            original = getattr(self.model, "encode_images")
        except AttributeError as exc:
            raise RuntimeABIError("model.encode_images is required") from exc
        namespace = getattr(self.model, "__dict__", {})
        self._had_instance_attribute = "encode_images" in namespace
        if self._had_instance_attribute:
            self._instance_value = namespace["encode_images"]
        self._original = original

        def replay(instance: object, _placeholder: object, *_args: object, **_kwargs: object) -> object:
            if instance is not self.model:
                raise ReplayContractError("replay method escaped the loaded model instance")
            self.invocations += 1
            patch = self.prepare_patch(self.patch_tensor, self.connector)
            return self.connector(patch)

        setattr(self.model, "encode_images", types.MethodType(replay, self.model))
        self._entered = True
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> bool:
        if self._had_instance_attribute:
            setattr(self.model, "encode_images", self._instance_value)
        else:
            try:
                delattr(self.model, "encode_images")
            except AttributeError as exc:
                raise ReplayContractError("unable to restore model.encode_images") from exc
        self._entered = False
        restored = getattr(self.model, "encode_images", None)
        self.restored = _same_bound_callable(restored, self._original)
        if not self.restored:
            raise ReplayContractError("original model.encode_images was not restored")
        return False


@contextlib.contextmanager
def _capture_generate(model: object) -> Iterator[list[list[int]]]:
    try:
        original = getattr(model, "generate")
    except AttributeError as exc:
        raise RuntimeABIError("model.generate is required") from exc
    namespace = getattr(model, "__dict__", {})
    had_instance_attribute = "generate" in namespace
    instance_value = namespace.get("generate")
    captured: list[list[int]] = []

    def wrapped(instance: object, *args: object, **kwargs: object) -> object:
        if instance is not model:
            raise ReplayContractError("generate wrapper escaped the loaded model instance")
        output = original(*args, **kwargs)
        sequences = getattr(output, "sequences", output)
        try:
            values = sequences.detach().cpu().tolist()
        except AttributeError as exc:
            raise RuntimeABIError("model.generate did not return token IDs") from exc
        if values and isinstance(values[0], int):
            values = [values]
        if not isinstance(values, list) or not all(isinstance(row, list) for row in values):
            raise RuntimeABIError("generated token IDs have an unsupported structure")
        captured[:] = [[int(token) for token in row] for row in values]
        return output

    setattr(model, "generate", types.MethodType(wrapped, model))
    try:
        yield captured
    finally:
        if had_instance_attribute:
            setattr(model, "generate", instance_value)
        else:
            delattr(model, "generate")
        if not _same_bound_callable(getattr(model, "generate", None), original):
            raise ReplayContractError("original model.generate was not restored")


def _remove_hooks(handles: Sequence[object]) -> None:
    for handle in reversed(handles):
        try:
            handle.remove()  # type: ignore[attr-defined]
        except Exception:
            pass


def _run_chat_capture(
    *,
    model: object,
    vision_tower: object,
    connector: object,
    chat: Callable[[], object],
) -> tuple[CaptureState, list[list[int]], str]:
    state = CaptureState()
    handles: list[object] = []
    try:
        handles.append(vision_tower.register_forward_hook(state.vision_hook))  # type: ignore[attr-defined]
        handles.append(connector.register_forward_pre_hook(state.connector_pre_hook))  # type: ignore[attr-defined]
        handles.append(connector.register_forward_hook(state.connector_hook))  # type: ignore[attr-defined]
        with _capture_generate(model) as token_ids:
            text = chat()
    except NativeReplayError:
        raise
    except Exception as exc:
        raise RuntimeABIError("TinyLLaVA chat execution failed") from exc
    finally:
        _remove_hooks(handles)
    if not isinstance(text, str):
        if isinstance(text, tuple) and text and isinstance(text[0], str):
            text = text[0]
        else:
            raise RuntimeABIError("model.chat did not return decoded text")
    if not token_ids:
        raise RuntimeABIError("model.chat did not invoke the wrapped model.generate")
    return state, token_ids, text


def _nested_value(value: object, *paths: str) -> object:
    for path in paths:
        current = value
        found = True
        for part in path.split("."):
            if isinstance(current, Mapping):
                if part not in current:
                    found = False
                    break
                current = current[part]
            else:
                if not hasattr(current, part):
                    found = False
                    break
                current = getattr(current, part)
        if found:
            return current
    raise RuntimeABIError(f"runtime configuration is missing {paths[0]!r}")


def _resolve_component(model: object, *, label: str, paths: Sequence[str]) -> object:
    for path in paths:
        current = model
        try:
            for part in path.split("."):
                current = getattr(current, part)
            if current is not None:
                return current
        except AttributeError:
            continue
    raise RuntimeABIError(f"required {label} interface is absent")


def _resolve_runtime_config(
    model: object,
    vision_tower: object,
    config: AnchorConfig,
) -> dict[str, object]:
    runtime = getattr(model, "config", None)
    if runtime is None:
        raise RuntimeABIError("model.config is required")
    vision = getattr(vision_tower, "config", None)
    resolved = {
        "model_type": _nested_value(runtime, "model_type"),
        "vision_model": _nested_value(
            runtime,
            "vision_model_name_or_path",
            "vision_tower",
            "mm_vision_tower",
        ),
        "vision_feature_layer": _nested_value(runtime, "vision_feature_layer", "mm_vision_select_layer"),
        "vision_feature_select_strategy": _nested_value(
            runtime,
            "vision_feature_select_strategy",
            "mm_vision_select_feature",
        ),
        "connector_type": _nested_value(runtime, "connector_type", "mm_projector_type"),
        "image_size": _nested_value(vision, "image_size"),
        "patch_size": _nested_value(vision, "patch_size"),
        "vision_hidden_size": _nested_value(vision, "hidden_size"),
        "connector_output_size": _nested_value(
            runtime,
            "text_config.hidden_size",
            "llm_config.hidden_size",
            "hidden_size",
        ),
    }
    expected = {
        "model_type": MODEL_TYPE,
        "vision_model": config.expected_vision_model,
        "vision_feature_layer": config.expected_vision_feature_layer,
        "vision_feature_select_strategy": config.expected_vision_feature_select_strategy,
        "connector_type": config.expected_connector_type,
        "image_size": config.expected_image_size,
        "patch_size": config.expected_patch_size,
        "vision_hidden_size": config.expected_vision_hidden_size,
        "connector_output_size": config.expected_connector_output_size,
    }
    for name, expected_value in expected.items():
        if resolved[name] != expected_value:
            raise RuntimeABIError(
                f"runtime.{name}: expected {expected_value!r}, found {resolved[name]!r}"
            )
    return resolved


def _chat_kwargs(
    *,
    model: object,
    tokenizer: object,
    image_path: Path,
    config: AnchorConfig,
) -> dict[str, object]:
    """Build the exact kwargs accepted by the pinned TinyLLaVA ``chat`` ABI."""

    chat = getattr(model, "chat", None)
    if not callable(chat):
        raise RuntimeABIError("model.chat is required")
    try:
        signature = inspect.signature(chat)
    except (TypeError, ValueError) as exc:
        raise RuntimeABIError("model.chat signature is unavailable") from exc
    required = {
        "prompt",
        "tokenizer",
        "image",
        "max_new_tokens",
        "num_beams",
        "top_p",
        "temperature",
    }
    missing = sorted(required - set(signature.parameters))
    if missing:
        raise RuntimeABIError(
            f"pinned model.chat ABI is missing required parameters {missing!r}"
        )
    return {
        "prompt": config.prompt,
        "tokenizer": tokenizer,
        "image": str(image_path),
        "max_new_tokens": config.max_new_tokens,
        "num_beams": config.num_beams,
        "top_p": None,
        "temperature": 0,
    }


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as exc:
        raise ReplayContractError(f"unable to write {path.name}") from exc


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
        raise ReplayContractError("repository commit is unavailable")
    return value


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("torch", "transformers", "safetensors", "Pillow", "sentencepiece"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "NOT_INSTALLED"
    return result


def build_result(**values: object) -> dict[str, object]:
    """Build and validate the successful result schema."""

    required = {
        "status",
        "repository_commit",
        "model_id",
        "model_revision",
        "runtime_package_versions",
        "cuda_identity",
        "checkpoint_snapshot_identity",
        "image_path_basename",
        "image_sha256",
        "runtime_resolved_model_configuration",
        "trainable_parameter_count",
        "patch_shape",
        "connector_shape",
        "native_call_counts",
        "replay_call_counts",
        "serialization_comparison",
        "connector_comparison",
        "native_token_ids",
        "replay_token_ids",
        "native_decoded_text",
        "replay_decoded_text",
        "restoration_status",
        "artifacts",
    }
    missing = sorted(required - set(values))
    unknown = sorted(set(values) - required)
    if missing or unknown:
        raise ReplayContractError(
            f"result fields mismatch: missing={missing!r}, unknown={unknown!r}"
        )
    result = {"schema_id": RESULT_SCHEMA_ID, **values}
    canonical_json_bytes(result)
    return result


def _tensor_bytes(tensor: object, torch: object) -> bytes:
    try:
        byte_view = tensor.detach().cpu().contiguous().view(torch.uint8)  # type: ignore[attr-defined]
        return byte_view.numpy().tobytes(order="C")
    except Exception as exc:
        raise ReplayContractError("unable to canonicalize tensor bytes") from exc


def _tensor_sha256(tensor: object, torch: object) -> str:
    return hashlib.sha256(_tensor_bytes(tensor, torch)).hexdigest()


def _prepare_for_connector(patch: object, connector: object) -> object:
    try:
        parameter = next(iter(connector.parameters()))  # type: ignore[attr-defined]
        return patch.to(device=parameter.device, dtype=parameter.dtype)  # type: ignore[attr-defined]
    except (AttributeError, StopIteration) as exc:
        raise RuntimeABIError("connector device and dtype are unavailable") from exc


def _connector_cosine(native_connector: object, replay_connector: object, *, torch: object) -> float:
    """Compute the connector cosine using float64 inputs before reduction."""

    return float(
        torch.nn.functional.cosine_similarity(
            native_connector.to(dtype=torch.float64).reshape(1, -1),
            replay_connector.to(dtype=torch.float64).reshape(1, -1),
            dim=1,
        ).item()
    )


def _require_connector_cosine(cosine: float) -> None:
    if not math.isfinite(cosine) or cosine < _CONNECTOR_COSINE_THRESHOLD:
        raise ReplayContractError(
            "connector cosine similarity "
            f"{cosine!r} is below threshold {_CONNECTOR_COSINE_THRESHOLD}"
        )


def run_native_replay(
    *,
    config_path: str | Path,
    image_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    device: str,
    dtype: str,
    repository_root: str | Path,
) -> dict[str, object]:
    """Execute one offline frozen native capture and connector-boundary replay."""

    config = load_anchor_config(config_path)
    image_file = Path(image_path)
    cache = Path(cache_root)
    output = Path(output_dir)
    if not image_file.is_file():
        raise ReplayContractError("image path must be an existing regular file")
    if not cache.is_dir():
        raise ReplayContractError("cache root must be an existing directory")
    if output.exists():
        raise FileExistsError("output directory must not already exist")
    if not device.startswith("cuda"):
        raise ReplayContractError("the Apple validation profile requires CUDA")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ReplayContractError("dtype must be float16, bfloat16, or float32")
    output.mkdir(parents=True, exist_ok=False)

    try:
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeABIError("required TinyLLaVA runtime dependency is unavailable") from exc

    if not torch.cuda.is_available():
        raise ReplayContractError("CUDA is unavailable")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    common = {
        "revision": config.revision,
        "cache_dir": str(cache),
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch_dtype,
            **common,
        )
    except Exception as exc:
        raise RuntimeABIError("exact local TinyLLaVA checkpoint could not be loaded") from exc
    try:
        tokenizer_model_max_length = model.config.tokenizer_model_max_length
        tokenizer_padding_side = model.config.tokenizer_padding_side
    except AttributeError as exc:
        raise RuntimeABIError(
            "pinned model config is missing tokenizer_model_max_length or tokenizer_padding_side"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=str(cache),
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
            use_fast=False,
            model_max_length=tokenizer_model_max_length,
            padding_side=tokenizer_padding_side,
        )
    except Exception as exc:
        raise RuntimeABIError("exact local TinyLLaVA tokenizer could not be loaded") from exc
    model.to(device=device, dtype=torch_dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 0 or model.training:
        raise ReplayContractError("model is not fully frozen in evaluation mode")

    if callable(getattr(model, "get_vision_tower", None)):
        vision_tower = model.get_vision_tower()
    else:
        vision_tower = _resolve_component(
            model,
            label="vision tower",
            paths=("vision_tower", "model.vision_tower"),
        )
    connector = _resolve_component(
        model,
        label="connector",
        paths=("connector", "model.connector", "mm_projector", "model.mm_projector"),
    )
    for component, label in ((vision_tower, "vision tower"), (connector, "connector")):
        if not callable(getattr(component, "register_forward_hook", None)):
            raise RuntimeABIError(f"{label} does not support forward hooks")
    resolved = _resolve_runtime_config(model, vision_tower, config)
    if config.expected_vision_tower_frozen and any(
        parameter.requires_grad for parameter in vision_tower.parameters()
    ):
        raise ReplayContractError("vision tower is not frozen")
    resolved["vision_tower_frozen"] = True

    native_kwargs = _chat_kwargs(
        model=model,
        tokenizer=tokenizer,
        image_path=image_file,
        config=config,
    )
    replay_kwargs = _chat_kwargs(
        model=model,
        tokenizer=tokenizer,
        image_path=image_file,
        config=config,
    )

    with torch.inference_mode():
        native, native_tokens, native_text = _run_chat_capture(
            model=model,
            vision_tower=vision_tower,
            connector=connector,
            chat=lambda: model.chat(**native_kwargs),
        )
    if native.vision_calls < 1 or native.connector_calls < 1:
        raise ReplayContractError("native generation did not invoke required components")
    if native.patch_tensor is None or native.connector_tensor is None:
        raise ReplayContractError("native hooks did not capture required tensors")
    finite = lambda value: bool(torch.isfinite(value).all().item())
    validate_tensor_contract(
        native.patch_tensor,
        expected_shape=(1, config.expected_patch_count, config.expected_vision_hidden_size),
        expected_dtype=dtype,
        finite_check=finite,
        label="native_patch",
    )
    validate_tensor_contract(
        native.connector_tensor,
        expected_shape=(1, config.expected_patch_count, config.expected_connector_output_size),
        expected_dtype=dtype,
        finite_check=finite,
        label="native_connector",
    )

    tensor_path = output / "captured_patch_features.safetensors"
    save_file({PATCH_TENSOR_KEY: native.patch_tensor.detach().cpu().contiguous()}, str(tensor_path))
    reloaded = load_file(str(tensor_path), device="cpu")[PATCH_TENSOR_KEY]
    serialization = validate_serialization_round_trip(
        native.patch_tensor.detach().cpu(),
        reloaded,
        finite_check=finite,
        max_abs_diff=lambda left, right: float((left - right).abs().max().item()),
    )
    native_patch_hash = _tensor_sha256(native.patch_tensor, torch)
    reloaded_patch_hash = _tensor_sha256(reloaded, torch)
    if native_patch_hash != reloaded_patch_hash:
        raise ReplayContractError("serialized patch byte hash changed")
    serialization["byte_hash_equal"] = True
    serialization["sha256"] = native_patch_hash

    injection = ReplayInjection(
        model,
        reloaded,
        connector,
        prepare_patch=_prepare_for_connector,
    )
    with torch.inference_mode():
        with injection:
            replay, replay_tokens, replay_text = _run_chat_capture(
                model=model,
                vision_tower=vision_tower,
                connector=connector,
                chat=lambda: model.chat(**replay_kwargs),
            )
    if replay.vision_calls != 0 or replay.connector_calls < 1 or injection.invocations < 1:
        raise ReplayContractError("replay did not bypass vision tower through connector")
    if replay.connector_tensor is None:
        raise ReplayContractError("replay connector output was not captured")
    validate_tensor_contract(
        replay.connector_tensor,
        expected_shape=(1, config.expected_patch_count, config.expected_connector_output_size),
        expected_dtype=dtype,
        finite_check=finite,
        label="replay_connector",
    )
    native_connector = native.connector_tensor.detach().cpu()
    replay_connector = replay.connector_tensor.detach().cpu()
    connector_max = float((native_connector - replay_connector).abs().max().item())
    cosine = _connector_cosine(native_connector, replay_connector, torch=torch)
    if not math.isfinite(connector_max) or connector_max > 1e-5:
        raise ReplayContractError("connector maximum absolute difference exceeds 1e-5")
    _require_connector_cosine(cosine)
    if native_tokens != replay_tokens:
        raise ReplayContractError("native and replay generated token IDs differ")
    if native_text != replay_text:
        raise ReplayContractError("native and replay decoded text differ")

    feature_manifest = {
        "schema_id": "vea.p2.t001a.feature_manifest.v1",
        "model_id": config.model_id,
        "model_revision": config.revision,
        "image_path_basename": image_file.name,
        "image_sha256": sha256_file(image_file),
        "tensor_key": PATCH_TENSOR_KEY,
        "shape": list(reloaded.shape),
        "dtype": _dtype_name(reloaded.dtype),
        "ordered_patch_tokens": True,
        "vision_feature_layer": config.expected_vision_feature_layer,
        "vision_feature_select_strategy": config.expected_vision_feature_select_strategy,
        "vision_output_type": native.vision_output_type,
        "hidden_state_tuple_length": native.hidden_state_tuple_length,
        "resolved_hidden_state_index": (
            native.hidden_state_tuple_length + config.expected_vision_feature_layer
            if native.hidden_state_tuple_length is not None
            and config.expected_vision_feature_layer < 0
            else config.expected_vision_feature_layer
        ),
        "tensor_sha256": reloaded_patch_hash,
    }
    manifest_path = output / "feature_manifest.json"
    _write_new(manifest_path, canonical_json_bytes(feature_manifest))
    tensor_artifact = {
        "relative_path": tensor_path.name,
        "size_bytes": tensor_path.stat().st_size,
        "sha256": sha256_file(tensor_path),
    }
    manifest_artifact = {
        "relative_path": manifest_path.name,
        "size_bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }
    snapshot_name = "models--" + config.model_id.replace("/", "--")
    snapshot = cache / snapshot_name / "snapshots" / config.revision
    result = build_result(
        status="PASS_NATIVE_REPLAY",
        repository_commit=_repository_commit(Path(repository_root)),
        model_id=config.model_id,
        model_revision=config.revision,
        runtime_package_versions=_package_versions(),
        cuda_identity={
            "torch_cuda": torch.version.cuda,
            "device": device,
            "device_name": torch.cuda.get_device_name(torch.device(device)),
        },
        checkpoint_snapshot_identity={
            "revision": config.revision,
            "snapshot_directory_name": snapshot.name,
            "locally_resolved": snapshot.is_dir(),
        },
        image_path_basename=image_file.name,
        image_sha256=feature_manifest["image_sha256"],
        runtime_resolved_model_configuration=resolved,
        trainable_parameter_count=trainable,
        patch_shape=list(native.patch_tensor.shape),
        connector_shape=list(native.connector_tensor.shape),
        native_call_counts={
            "vision_tower": native.vision_calls,
            "connector": native.connector_calls,
        },
        replay_call_counts={
            "vision_tower": replay.vision_calls,
            "connector": replay.connector_calls,
            "encode_images_injection": injection.invocations,
        },
        serialization_comparison=serialization,
        connector_comparison={
            "max_abs_diff": connector_max,
            "cosine_similarity": cosine,
            "native_sha256": _tensor_sha256(native_connector, torch),
            "replay_sha256": _tensor_sha256(replay_connector, torch),
        },
        native_token_ids=native_tokens,
        replay_token_ids=replay_tokens,
        native_decoded_text=native_text,
        replay_decoded_text=replay_text,
        restoration_status={"encode_images_restored": injection.restored},
        artifacts={
            "captured_patch_features": tensor_artifact,
            "feature_manifest": manifest_artifact,
            "result": {"relative_path": "result.json"},
        },
    )
    _write_new(output / "result.json", canonical_json_bytes(result))
    return result


def write_blocker_result(output_dir: str | Path, error: BaseException) -> Path | None:
    """Best-effort structured failure evidence without overwriting any file."""

    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=True)
        path = output / "result.json"
        classification = getattr(error, "classification", "BLOCKED_RUNTIME_FAILURE")
        payload = {
            "schema_id": RESULT_SCHEMA_ID,
            "status": classification,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        _write_new(path, canonical_json_bytes(payload))
        return path
    except Exception:
        return None


__all__ = [
    "ANCHOR_SCHEMA_ID",
    "RESULT_SCHEMA_ID",
    "AnchorConfig",
    "AnchorConfigError",
    "CaptureState",
    "NativeReplayError",
    "ReplayContractError",
    "ReplayInjection",
    "RuntimeABIError",
    "build_result",
    "canonical_json_bytes",
    "load_anchor_config",
    "run_native_replay",
    "sha256_file",
    "validate_serialization_round_trip",
    "validate_tensor_contract",
    "write_blocker_result",
]
