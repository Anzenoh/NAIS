from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_linear as predecessor


class SyntheticConnector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((768,), 1.125, dtype=torch.float16))
        self.register_buffer("offset", torch.full((768,), 0.25, dtype=torch.float16))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat((value * self.scale, value + self.offset), dim=-1)


def fixture_batch(count: int = 2, *, split: str = "train") -> tuple[core.FeatureBatch, SyntheticConnector]:
    generator = torch.Generator().manual_seed(43)
    source = torch.randn((count, 196, 768), generator=generator)
    target = source * 0.75 + 0.2
    connector = SyntheticConnector()
    with torch.inference_mode():
        target_connector = connector(target.half()).float().clone()
    ids = tuple(f"synthetic-{split}-{i}" for i in range(count))
    return core.FeatureBatch(source, target, target_connector, ids, ids, ids, split, True), connector


def adapter() -> core.AffineAdapter:
    return core.AffineAdapter(torch.eye(768), torch.zeros(768))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_id",), "wrong"),
        (("experiment_id",), "wrong"),
        (("conditions", "connector_aware_weight_grid"), [0.0, 0.25, 1.0]),
        (("adapter", "trainable_scalar_count_per_source"), 590593),
        (("losses", "D_patch"), "validation_mean"),
        (("gradient_training", "learning_rate"), 1e-3),
        (("split", "test", "access_authorized"), True),
        (("test_and_generation_policy", "generation_scheduled"), True),
        (("runtime_bindings", "target_feature_bank_path_and_hash"), "fabricated"),
    ],
)
def test_config_drift_fails_closed(path: tuple[str, ...], value: object) -> None:
    changed = copy.deepcopy(core.load_config())
    node = changed
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(core.ConnectorAwareLinearError, match="drift|schema"):
        core.validate_config(changed)


def test_config_byte_identity_and_open_bindings(tmp_path: Path) -> None:
    config = core.load_config()
    assert set(config["conditions"]["connector_aware_weight_grid"]) == {0.25, 1.0, 4.0}
    assert core.condition_weight("PATCH_ONLY_CONTROL", config) == 0.0
    assert all(value == core.OPEN_BINDING for key, value in config["runtime_bindings"].items() if key != "fail_closed_before_runtime_on_unbound_or_mismatched_identity")
    with pytest.raises(core.ConnectorAwareLinearError, match="condition"):
        core.condition_weight("OLS_REFERENCE", config)
    copied = tmp_path / "config.json"
    copied.write_text(Path(core.CONFIG_PATH).read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(core.ConnectorAwareLinearError, match="SHA-256"):
        core.load_config(copied)


def test_affine_capacity_and_zero_step_full_linear_equivalence() -> None:
    rng = np.random.default_rng(17)
    weights = np.eye(768, dtype=np.float32)
    source_mean = rng.normal(size=768).astype(np.float32)
    target_mean = rng.normal(size=768).astype(np.float32)
    fitted = predecessor.FullLinearFit(weights, source_mean, target_mean, 768, np.empty(0, dtype=np.float32))
    fitted_adapter = core.AffineAdapter.from_full_linear(weights, source_mean, target_mean)
    assert [(name, tuple(value.shape)) for name, value in fitted_adapter.named_parameters()] == [
        ("W", (768, 768)), ("b", (768,))
    ]
    assert sum(value.numel() for value in fitted_adapter.parameters() if value.requires_grad) == 590592
    source = rng.normal(size=(1, 196, 768)).astype(np.float32)
    actual = fitted_adapter(torch.from_numpy(source)).detach().numpy()
    np.testing.assert_allclose(actual, predecessor.apply_full_linear(source, fitted), rtol=1e-5, atol=1e-5)


def test_connector_loss_only_backward_reaches_adapter() -> None:
    config = core.load_config()
    batch, connector = fixture_batch()
    boundary = core.FrozenDifferentiableConnector(connector)
    fitted_adapter = adapter()
    normalizers = core.compute_train_normalizers(batch, config)
    losses = core.compute_losses(fitted_adapter(batch.source), batch.target, batch.target_connector,
                                 boundary, normalizers, "PATCH_CONNECTOR_1P0", config)
    losses["L_connector"].backward()
    assert losses["L_total"].dtype == torch.float32
    assert fitted_adapter.W.grad is not None and torch.isfinite(fitted_adapter.W.grad).all()
    assert float(fitted_adapter.W.grad.norm()) > 0
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in connector.parameters())


def test_one_step_updates_only_adapter_and_keeps_counters_zero() -> None:
    config = core.load_config()
    batch, connector = fixture_batch()
    boundary = core.FrozenDifferentiableConnector(connector)
    normalizers = core.compute_train_normalizers(batch, config)
    fitted_adapter = adapter()
    optimizer = core.make_optimizer(fitted_adapter, config)
    before_adapter = (fitted_adapter.W.clone(), fitted_adapter.b.clone())
    before_connector = {key: value.clone() for key, value in connector.state_dict().items()}
    guard = core.AccessGuard()
    losses = core.train_step(fitted_adapter, batch, boundary, normalizers,
                             "PATCH_CONNECTOR_0P25", optimizer, config, guard)
    assert losses["L_total"] > 0
    assert not torch.equal(fitted_adapter.W, before_adapter[0])
    assert not torch.equal(fitted_adapter.b, before_adapter[1])
    assert all(torch.equal(connector.state_dict()[key], value) for key, value in before_connector.items())
    assert guard.test_access_count == guard.generation_calls == 0


def test_optimizer_drift_and_non_adapter_parameter_are_rejected() -> None:
    config = core.load_config()
    batch, connector = fixture_batch(count=1)
    boundary = core.FrozenDifferentiableConnector(connector)
    normalizers = core.compute_train_normalizers(batch, config)
    fitted_adapter = adapter()
    optimizer = core.make_optimizer(fitted_adapter, config)
    optimizer.param_groups[0]["lr"] = 1e-3
    with pytest.raises(core.ConnectorAwareLinearError, match="settings"):
        core.train_step(fitted_adapter, batch, boundary, normalizers,
                        "PATCH_ONLY_CONTROL", optimizer, config, core.AccessGuard())
    extra = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([fitted_adapter.W, fitted_adapter.b, extra], lr=3e-5,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    with pytest.raises(core.ConnectorAwareLinearError, match="non-adapter"):
        core.train_step(fitted_adapter, batch, boundary, normalizers,
                        "PATCH_ONLY_CONTROL", optimizer, config, core.AccessGuard())


def test_train_denominators_reject_zero_and_nonfinite() -> None:
    config = core.load_config()
    batch, _ = fixture_batch()
    normalizers = core.compute_train_normalizers(batch, config)
    assert normalizers.patch.dtype == normalizers.connector.dtype == torch.float32
    assert normalizers.patch > 0 and normalizers.connector > 0
    with pytest.raises(core.ConnectorAwareLinearError, match="Train only"):
        core.compute_train_normalizers(replace(batch, split="validation"), config)
    with pytest.raises(core.ConnectorAwareLinearError, match="patch Train denominator"):
        core.compute_train_normalizers(replace(batch, target=torch.zeros_like(batch.target)), config)
    with pytest.raises(core.ConnectorAwareLinearError, match="connector Train denominator"):
        core.compute_train_normalizers(replace(batch, target_connector=torch.zeros_like(batch.target_connector)), config)
    with pytest.raises(core.ConnectorAwareLinearError, match="nonfinite"):
        core.compute_train_normalizers(replace(batch, target=batch.target.fill_(float("nan"))), config)


def test_checkpoint_selection_uses_frozen_losses_and_earliest_tie() -> None:
    config = core.load_config()
    records = [{"source": "SIGLIP2", "condition": "PATCH_ONLY_CONTROL", "split": "validation",
                "epoch": epoch, "L_patch": 1.0, "L_total": 0.2 if epoch == 8 else 2.0}
               for epoch in range(9)]
    assert core.select_checkpoint(records, "SIGLIP2", "PATCH_ONLY_CONTROL", config)["epoch"] == 0
    connector_records = [{**row, "condition": "PATCH_CONNECTOR_1P0", "L_patch": 0.0,
                          "L_total": 0.1 if row["epoch"] in (3, 5) else 1.0}
                         for row in records]
    assert core.select_checkpoint(connector_records, "SIGLIP2", "PATCH_CONNECTOR_1P0", config)["epoch"] == 3
    connector_records[0]["split"] = "test"
    with pytest.raises(core.ConnectorAwareLinearError, match="Validation"):
        core.select_checkpoint(connector_records, "SIGLIP2", "PATCH_CONNECTOR_1P0", config)


def test_lambda_selection_hierarchy_and_smallest_weight_tie() -> None:
    config = core.load_config()
    records = [{"source": "DINOV3", "split": "validation", "condition": condition,
                "connector_loss_weight": weight, "same_image_recall_at_1": 0.5,
                "same_image_recall_at_5": 0.8, "connector_cosine": 0.6}
               for condition, weight in zip(core.CONDITIONS[2:], (0.25, 1.0, 4.0))]
    assert core.select_connector_weight(records, "DINOV3", config)["connector_loss_weight"] == 0.25
    records[1]["same_image_recall_at_1"] = 0.6
    assert core.select_connector_weight(records, "DINOV3", config)["connector_loss_weight"] == 1.0
    records[0]["same_image_recall_at_1"] = 0.6
    records[0]["same_image_recall_at_5"] = 0.9
    assert core.select_connector_weight(records, "DINOV3", config)["connector_loss_weight"] == 0.25
    records[1]["same_image_recall_at_5"] = 0.9
    records[1]["connector_cosine"] = 0.7
    assert core.select_connector_weight(records, "DINOV3", config)["connector_loss_weight"] == 1.0
    with pytest.raises(core.ConnectorAwareLinearError, match="required"):
        core.select_connector_weight(records[:2], "DINOV3", config)


def test_safe_save_reload_and_hash_validation(tmp_path: Path) -> None:
    fitted_adapter = adapter()
    source, _ = fixture_batch(count=1)
    before = fitted_adapter(source.source).detach().numpy()
    directory = tmp_path / "selected"
    metadata = core.save_selected_adapter(directory, fitted_adapter, source="SIGLIP2",
                                          condition="PATCH_CONNECTOR_1P0", epoch=3,
                                          guard=core.AccessGuard())
    restored, loaded = core.load_selected_adapter(directory)
    assert loaded == metadata and loaded["runtime_identity_binding"] == core.OPEN_BINDING
    np.testing.assert_allclose(restored(source.source).detach().numpy(), before, rtol=1e-5, atol=1e-5)
    with pytest.raises(core.ConnectorAwareLinearError, match="exists"):
        core.save_selected_adapter(directory, fitted_adapter, source="SIGLIP2",
                                   condition="PATCH_CONNECTOR_1P0", epoch=3, guard=core.AccessGuard())
    with (directory / "affine.npz").open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(core.ConnectorAwareLinearError, match="hash mismatch"):
        core.load_selected_adapter(directory)


def test_shape_dtype_nonfinite_order_split_and_access_guard_fail_closed() -> None:
    config = core.load_config()
    batch, connector = fixture_batch()
    bad = (
        replace(batch, source=batch.source[:, :195]),
        replace(batch, source=batch.source.double()),
        replace(batch, source=torch.full_like(batch.source, float("inf"))),
        replace(batch, target_image_ids=tuple(reversed(batch.target_image_ids))),
        replace(batch, connector_image_ids=tuple(reversed(batch.connector_image_ids))),
        replace(batch, split="test"),
    )
    for candidate in bad:
        with pytest.raises(core.ConnectorAwareLinearError):
            core.validate_batch(candidate, config)
    validation = replace(batch, split="validation", source_image_ids=("v0", "v1"),
                         target_image_ids=("v0", "v1"), connector_image_ids=("v0", "v1"))
    core.validate_train_validation_pair(batch, validation, config)
    with pytest.raises(core.ConnectorAwareLinearError, match="overlap"):
        core.validate_train_validation_pair(batch, replace(batch, split="validation"), config)
    boundary = core.FrozenDifferentiableConnector(connector)
    fitted_adapter = adapter()
    with pytest.raises(core.ConnectorAwareLinearError, match="zero"):
        core.train_step(fitted_adapter, batch, boundary, core.compute_train_normalizers(batch, config),
                        "PATCH_ONLY_CONTROL", core.make_optimizer(fitted_adapter, config), config,
                        core.AccessGuard(generation_calls=1))


def test_deterministic_image_shuffle_epoch_budget_and_synthetic_epoch() -> None:
    config = core.load_config()
    first = core.image_batches(17, 1, config)
    assert first == core.image_batches(17, 1, config) and first != core.image_batches(17, 2, config)
    assert [len(part) for part in first] == [8, 8, 1]
    assert sorted(index for part in first for index in part) == list(range(17))
    with pytest.raises(core.ConnectorAwareLinearError, match="epoch"):
        core.image_batches(17, 9, config)
    batch, connector = fixture_batch(count=2)
    boundary = core.FrozenDifferentiableConnector(connector)
    fitted_adapter = adapter()
    records = core.train_epoch(fitted_adapter, batch, boundary, core.compute_train_normalizers(batch, config),
                               "PATCH_ONLY_CONTROL", core.make_optimizer(fitted_adapter, config),
                               1, config, core.AccessGuard())
    assert len(records) == 1 and records[0]["L_total"] == pytest.approx(records[0]["L_patch"])
