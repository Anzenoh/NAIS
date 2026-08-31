"""Frozen TinyLLaVA native/adapted generation boundary using the T001D ABI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_tinyllava_connector as connector_boundary


class FrozenGenerationError(RuntimeError):
    """Raised when frozen generation or replay violates the established ABI."""


@dataclass(frozen=True)
class GenerationOutput:
    token_ids: tuple[int, ...]
    decoded_text: str
    vision_calls: int
    connector_calls: int
    injection_calls: int
    injection_restored: bool


@dataclass
class FrozenTinyLLaVAGenerator:
    model: Any
    tokenizer: Any
    vision_tower: Any
    connector: Any
    torch: Any
    anchor_config: Any
    device: str
    dtype: Any
    native_calls: int = 0
    adapted_calls: int = 0

    def _capture(self, image_path: Path) -> tuple[Any, list[list[int]], str]:
        from vision_encoder_alignment.vlm_compat import tinyllava_native_replay as native

        kwargs = native._chat_kwargs(
            model=self.model, tokenizer=self.tokenizer, image_path=image_path, config=self.anchor_config
        )
        return native._run_chat_capture(
            model=self.model,
            vision_tower=self.vision_tower,
            connector=self.connector,
            chat=lambda: self.model.chat(**kwargs),
        )

    @staticmethod
    def _tokens(values: list[list[int]]) -> tuple[int, ...]:
        return tuple(int(value) for row in values for value in row)

    def generate_native(self, image_path: str | Path) -> GenerationOutput:
        path = Path(image_path)
        if not path.is_file():
            raise FrozenGenerationError("native generation image is unavailable")
        with self.torch.inference_mode():
            state, tokens, text = self._capture(path)
        if state.vision_calls < 1 or state.connector_calls < 1:
            raise FrozenGenerationError("native path did not invoke vision and connector")
        self.native_calls += 1
        return GenerationOutput(self._tokens(tokens), text, state.vision_calls, state.connector_calls, 0, True)

    def generate_adapted(self, image_path: str | Path, patch: np.ndarray) -> GenerationOutput:
        from vision_encoder_alignment.vlm_compat import tinyllava_native_replay as native

        path = Path(image_path)
        values = np.asarray(patch)
        expected = (1, self.anchor_config.expected_patch_count, self.anchor_config.expected_vision_hidden_size)
        if not path.is_file() or values.shape != expected or values.dtype != np.dtype("float32") or not np.isfinite(values).all():
            raise FrozenGenerationError("adapted generation input contract mismatch")
        tensor = self.torch.from_numpy(np.ascontiguousarray(values))
        injection = native.ReplayInjection(
            self.model, tensor, self.connector, prepare_patch=native._prepare_for_connector
        )
        with self.torch.inference_mode():
            with injection:
                state, tokens, text = self._capture(path)
        if state.vision_calls != 0 or state.connector_calls < 1:
            raise FrozenGenerationError("adapted replay did not bypass vision through the connector")
        if injection.invocations != 1 or not injection.restored:
            raise FrozenGenerationError("adapted replay injection did not invoke once and restore")
        self.adapted_calls += 1
        return GenerationOutput(
            self._tokens(tokens), text, state.vision_calls, state.connector_calls,
            injection.invocations, injection.restored,
        )

    def apply_connector(self, patches: np.ndarray, *, batch_size: int) -> np.ndarray:
        runner = connector_boundary.FrozenTinyLLaVAConnector(
            model=self.model, connector=self.connector, torch=self.torch, device=self.device, dtype=self.dtype,
            expected_input_shape=(self.anchor_config.expected_patch_count, self.anchor_config.expected_vision_hidden_size),
            expected_output_shape=(self.anchor_config.expected_patch_count, self.anchor_config.expected_connector_output_size),
        )
        return runner.apply(patches, batch_size=batch_size)


def _resolve_component(model: Any, paths: tuple[tuple[str, ...], ...], label: str) -> Any:
    for path in paths:
        value = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is not None:
            return value
    raise FrozenGenerationError(f"TinyLLaVA {label} component is unavailable")


def load_frozen_tinyllava_generator(
    *, config_path: str | Path, cache_root: str | Path, device: str, dtype_name: str = "float16"
) -> FrozenTinyLLaVAGenerator:
    """Lazy-load the pinned offline model/tokenizer and keep every parameter frozen."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from vision_encoder_alignment.vlm_compat import tinyllava_native_replay as native
    except ImportError as error:  # pragma: no cover - real host only.
        raise FrozenGenerationError("TinyLLaVA generation dependencies are unavailable") from error
    if dtype_name != "float16":
        raise FrozenGenerationError("frozen generation dtype must be float16")
    config = native.load_anchor_config(config_path)
    common = {
        "revision": config.revision,
        "cache_dir": str(cache_root),
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch.float16, **common)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, use_fast=False,
            model_max_length=model.config.tokenizer_model_max_length,
            padding_side=model.config.tokenizer_padding_side,
            **common,
        )
    except Exception as error:  # pragma: no cover - real host only.
        raise FrozenGenerationError("exact local TinyLLaVA model/tokenizer could not be loaded") from error
    model.to(device=device, dtype=torch.float16)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise FrozenGenerationError("TinyLLaVA is not fully frozen in evaluation mode")
    vision = (
        model.get_vision_tower()
        if callable(getattr(model, "get_vision_tower", None))
        else _resolve_component(model, (("vision_tower",), ("model", "vision_tower")), "vision tower")
    )
    connector = _resolve_component(
        model, (("connector",), ("model", "connector"), ("mm_projector",), ("model", "mm_projector")), "connector"
    )
    native._resolve_runtime_config(model, vision, config)
    if any(parameter.requires_grad for parameter in vision.parameters()) or any(parameter.requires_grad for parameter in connector.parameters()):
        raise FrozenGenerationError("vision tower or connector is not frozen")
    return FrozenTinyLLaVAGenerator(model, tokenizer, vision, connector, torch, config, device, torch.float16)
