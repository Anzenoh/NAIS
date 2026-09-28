from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_tinyllava_connector as inference


class SyntheticFrozenConnector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((768,), 0.75, dtype=torch.float16))
        self.register_buffer("offset", torch.full((768,), 0.125, dtype=torch.float16))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return torch.cat((patches * self.scale, patches + self.offset), dim=-1)


def test_config_provenance_and_train_validation_only_contract() -> None:
    config = core.load_config()
    assert hashlib.sha256(Path(core.CONFIG_PATH).read_bytes()).hexdigest() == core.CONFIG_SHA256
    for binding in config["provenance"].values():
        assert hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() == binding["sha256"]
    assert [config["split"][name]["image_count"] for name in ("train", "validation", "test")] == [800, 100, 100]
    assert config["split"]["test"]["access_authorized"] is False
    assert config["test_and_generation_policy"]["test_access_authorized"] is False
    assert config["test_and_generation_policy"]["generation_scheduled"] is False
    assert config["validation_selection"]["validation_llm_generation_authorized"] is False
    assert config["adapter"]["trainable_components"] == ["adapter.W", "adapter.b"]
    assert config["conditions"]["connector_aware_weight_grid"] == [0.25, 1.0, 4.0]


def test_training_boundary_matches_existing_inference_connector_on_synthetic_module() -> None:
    connector = SyntheticFrozenConnector()
    training = core.FrozenDifferentiableConnector(connector)
    existing = inference.FrozenTinyLLaVAConnector(
        model=nn.Identity(), connector=connector, torch=torch, device="cpu", dtype=torch.float16,
    )
    generator = torch.Generator().manual_seed(7)
    patches = torch.randn((1, 196, 768), generator=generator, dtype=torch.float32, requires_grad=True)
    differentiable = training(patches)
    inferred = existing.apply(patches.detach().numpy(), batch_size=1)
    np.testing.assert_allclose(differentiable.detach().numpy(), inferred, rtol=2e-3, atol=2e-3)
    assert existing.connector_calls == 1 and existing.vision_calls == existing.generation_calls == 0
    assert differentiable.requires_grad and differentiable.dtype == torch.float32
    assert connector.training is False and all(not p.requires_grad for p in connector.parameters())


def test_differentiable_boundary_rejects_disabled_grad_and_frozen_state_change() -> None:
    connector = SyntheticFrozenConnector()
    training = core.FrozenDifferentiableConnector(connector)
    candidate = torch.ones((1, 196, 768), dtype=torch.float32, requires_grad=True)
    with torch.no_grad(), pytest.raises(core.ConnectorAwareLinearError, match="differentiable"):
        training(candidate)
    with torch.inference_mode(), pytest.raises(core.ConnectorAwareLinearError, match="differentiable"):
        training(candidate)
    with torch.no_grad():
        connector.offset.add_(1)
    with pytest.raises(core.ConnectorAwareLinearError, match="buffer changed"):
        training(candidate)
