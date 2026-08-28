"""Frozen TinyLLaVA connector-only boundary for Phase 2 validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class FrozenConnectorError(RuntimeError):
    """Raised when the connector-only frozen-runtime contract is violated."""


@dataclass
class FrozenTinyLLaVAConnector:
    """A loaded model boundary that exposes only the frozen connector callable."""

    model: Any
    connector: Any
    torch: Any
    device: str
    dtype: Any
    expected_input_shape: tuple[int, int] = (196, 768)
    expected_output_shape: tuple[int, int] = (196, 1536)
    generation_calls: int = 0
    vision_calls: int = 0
    connector_calls: int = 0

    def apply(self, patches: np.ndarray, *, batch_size: int) -> np.ndarray:
        """Run connector batches; deliberately never invokes vision/chat/generation."""

        values = np.asarray(patches)
        if values.ndim != 3 or values.shape[1:] != self.expected_input_shape:
            raise FrozenConnectorError(f"connector input shape mismatch: {values.shape}")
        if values.dtype != np.dtype("float32") or not np.isfinite(values).all():
            raise FrozenConnectorError("connector input must be finite float32")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise FrozenConnectorError("connector batch_size must be positive")
        outputs: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, values.shape[0], batch_size):
                tensor = self.torch.from_numpy(np.ascontiguousarray(values[start : start + batch_size])).to(
                    device=self.device, dtype=self.dtype
                )
                result = self.connector(tensor)
                self.connector_calls += 1
                array = result.detach().to(device="cpu", dtype=self.torch.float32).numpy()
                if array.shape != (tensor.shape[0], *self.expected_output_shape) or not np.isfinite(array).all():
                    raise FrozenConnectorError("connector output contract mismatch")
                outputs.append(np.ascontiguousarray(array, dtype=np.float32))
        if self.vision_calls != 0 or self.generation_calls != 0:
            raise FrozenConnectorError("connector boundary invoked vision or generation")
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, *self.expected_output_shape), dtype=np.float32)


def _resolve_connector(model: Any) -> Any:
    for path in (("connector",), ("model", "connector"), ("mm_projector",), ("model", "mm_projector")):
        current = model
        for name in path:
            current = getattr(current, name, None)
            if current is None:
                break
        if current is not None and callable(current):
            return current
    raise FrozenConnectorError("TinyLLaVA connector component is unavailable")


def load_frozen_tinyllava_connector(
    *, config_path: str | Path,
    model_id: str,
    revision: str,
    cache_root: str | Path,
    device: str,
    dtype_name: str = "float16",
) -> FrozenTinyLLaVAConnector:
    """Lazy-load a frozen model and expose only its connector path.

    Heavy dependencies are intentionally imported inside this real-runtime-only
    function. No vision input, chat ABI, or generation API is called here.
    """

    try:
        import torch
        from transformers import AutoModelForCausalLM
        from vision_encoder_alignment.vlm_compat import tinyllava_native_replay as native
    except ImportError as error:  # pragma: no cover - exercised on real host only.
        raise FrozenConnectorError("connector runtime dependencies are unavailable") from error
    if dtype_name != "float16":
        raise FrozenConnectorError("frozen connector runtime dtype must be float16")
    anchor = native.load_anchor_config(config_path)
    if anchor.model_id != model_id or anchor.revision != revision:
        raise FrozenConnectorError("TinyLLaVA connector identity differs from the frozen adapter config")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(cache_root),
        local_files_only=anchor.local_files_only,
        trust_remote_code=anchor.trust_remote_code,
        torch_dtype=torch.float16,
    )
    model.to(device=device, dtype=torch.float16)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise FrozenConnectorError("TinyLLaVA model is not fully frozen in evaluation mode")
    connector = _resolve_connector(model)
    if hasattr(connector, "eval"):
        connector.eval()
    if hasattr(connector, "parameters") and any(parameter.requires_grad for parameter in connector.parameters()):
        raise FrozenConnectorError("TinyLLaVA connector is not frozen")
    return FrozenTinyLLaVAConnector(
        model=model, connector=connector, torch=torch, device=device, dtype=torch.float16,
        expected_input_shape=(anchor.expected_patch_count, anchor.expected_vision_hidden_size),
        expected_output_shape=(anchor.expected_patch_count, anchor.expected_connector_output_size),
    )
