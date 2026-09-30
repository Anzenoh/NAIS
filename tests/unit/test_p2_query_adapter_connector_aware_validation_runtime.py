"""Focused deterministic offline checks for the T003D Validation runner."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_query_adapter_connector_aware_validation as cli
from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_validation_runtime as runtime


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("t003d_completed") / "run"
    config, binding, anchor = runtime.load_accepted_inputs(ROOT)
    inputs, inference = cli._fake_inputs(binding)
    captures = {"initial": [], "optimizer_calls": 0, "updates": {}}
    original_fresh = runtime._fresh_adapter
    original_optimizer = core.make_optimizer
    original_epoch = core.train_epoch
    patch = pytest.MonkeyPatch()
    def fresh(arrays, device):
        adapter = original_fresh(arrays, device)
        captures["initial"].append(runtime._snapshot(adapter))
        return adapter
    def optimizer(adapter, design):
        captures["optimizer_calls"] += 1
        return original_optimizer(adapter, design)
    def epoch(adapter, train, boundary, normalizers, condition, selected_optimizer, number, design, guard):
        before = runtime._snapshot(adapter)
        records = original_epoch(adapter, train, boundary, normalizers, condition,
                                 selected_optimizer, number, design, guard)
        after = runtime._snapshot(adapter)
        captures["updates"].setdefault(condition, []).append(any(
            not np.array_equal(before[name], after[name]) for name in before
        ))
        return records
    patch.setattr(runtime, "_fresh_adapter", fresh)
    patch.setattr(core, "make_optimizer", optimizer)
    patch.setattr(core, "train_epoch", epoch)
    try:
        result = runtime.run_validation(
            config=config, binding=binding, anchor=anchor, inputs=inputs,
            inference_connector=inference, output_root=output, device="cpu", fake=True,
        )
    finally:
        patch.undo()
    return output, result, config, binding, anchor, captures, inference


def test_initializations_optimizer_scope_and_frozen_connector(completed) -> None:
    *_, captures, inference = completed
    assert len(captures["initial"]) == 10
    for offset in (0, 5):
        reference = captures["initial"][offset]
        for state in captures["initial"][offset + 1:offset + 5]:
            assert all(np.array_equal(reference[name], state[name]) for name in ("W", "b"))
    assert captures["optimizer_calls"] == 8  # four gradient conditions per source; OLS has none
    assert set(captures["updates"]) == set(core.CONDITIONS[1:])
    assert all(any(changes) for changes in captures["updates"].values())
    assert inference.connector.training is False
    assert inference.connector.scale.item() == 1.0
    assert inference.vision_calls == inference.generation_calls == 0


def test_all_conditions_epochs_metrics_and_zero_counters(completed) -> None:
    _, result, _, _, _ = completed[:5]
    assert result["schema_id"] == runtime.SCHEMA_ID
    assert result["status"] == "PASS_CONNECTOR_AWARE_VALIDATION"
    assert result["fake"] is True
    assert all(result[key] == 0 for key in (
        "test_access_count", "test_metric_count", "generation_calls", "vision_calls"
    ))
    assert set(result["sources"]) == set(core.SOURCES)
    for source in core.SOURCES:
        row = result["sources"][source]
        assert row["normalizers"]["D_patch"] > 0
        assert row["normalizers"]["D_connector"] > 0
        assert set(runtime.METRIC_FIELDS) <= set(row["OLS_REFERENCE"])
        assert row["OLS_REFERENCE"]["epoch"] == 0
        assert set(row["gradient_conditions"]) == set(core.CONDITIONS[1:])
        for condition in core.CONDITIONS[1:]:
            records = row["gradient_conditions"][condition]["validation_records"]
            assert [item["epoch"] for item in records] == list(range(9))
            assert all(item["connector_loss_weight"] == core.CONNECTOR_WEIGHTS[condition] for item in records)
            assert all(np.isfinite(item[key]) for item in records for key in runtime.METRIC_FIELDS)
            assert row["gradient_conditions"][condition]["selected_checkpoint"] == core.select_checkpoint(
                records, source, condition, completed[2],
            )
        selected = [row["gradient_conditions"][condition]["selected_checkpoint"]
                    for condition in core.CONDITIONS[2:]]
        assert row["selected_connector_aware"] == core.select_connector_weight(selected, source, completed[2])


def test_artifacts_are_safe_hashed_and_reload_exactly(completed, monkeypatch) -> None:
    output, result, _, _, _ = completed[:5]
    actual_load = runtime.np.load
    calls = []
    def safe_load(*args, **kwargs):
        calls.append(kwargs.get("allow_pickle"))
        return actual_load(*args, **kwargs)
    monkeypatch.setattr(runtime.np, "load", safe_load)
    for source in core.SOURCES:
        for condition, spec in result["sources"][source]["selected_artifacts"].items():
            folder = (output / spec["relative_path"]).parent
            restored, metadata = runtime.load_selected_artifact(
                folder, runtime_identity=result["runtime_identity_sha256"]
            )
            assert metadata["affine_npz_sha256"] == spec["affine_npz_sha256"]
            assert metadata["binding_sha256"] == runtime.BINDING_SHA256
            assert restored.W.shape == (768, 768) and restored.b.shape == (768,)
            assert metadata["condition"] == condition
    assert calls == [False] * 8
    folder = (output / result["sources"]["SIGLIP2"]["selected_artifacts"]["PATCH_ONLY_CONTROL"]["relative_path"]).parent
    with (folder / "affine.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(runtime.ValidationError, match="SHA-256"):
        runtime.load_selected_artifact(folder, runtime_identity=result["runtime_identity_sha256"])


def test_result_is_final_and_output_collision_rejected(completed) -> None:
    output, result, config, binding, anchor = completed[:5]
    persisted = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert persisted == result
    inputs, inference = cli._fake_inputs(binding)
    with pytest.raises(runtime.ValidationError, match="output root"):
        runtime.run_validation(config=config, binding=binding, anchor=anchor,
                               inputs=inputs, inference_connector=inference,
                               output_root=output, device="cpu", fake=True)


def test_frozen_source_hash_and_identity_drift_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime, "BINDING_SHA256", "0" * 64)
    with pytest.raises(runtime.ValidationError, match="SHA-256"):
        runtime.load_accepted_inputs(ROOT)
    monkeypatch.undo()
    config, binding, anchor = runtime.load_accepted_inputs(ROOT)
    inputs, inference = cli._fake_inputs(binding)
    bad = replace(inputs, provenance={"binding": {**binding, "schema_id": "drift"}})
    with pytest.raises(runtime.ValidationError, match="bound input"):
        runtime.run_validation(config=config, binding=binding, anchor=anchor, inputs=bad,
                               inference_connector=inference, output_root=tmp_path / "wrong", device="cpu", fake=True)
    changed_binding = {**binding, "implementation_predecessor": "0" * 40}
    with pytest.raises(runtime.ValidationError, match="binding config in memory"):
        runtime.run_validation(config=config, binding=changed_binding, anchor=anchor, inputs=inputs,
                               inference_connector=inference, output_root=tmp_path / "binding", device="cpu", fake=True)
    changed_anchor = {**anchor, "revision": "wrong"}
    with pytest.raises(runtime.ValidationError, match="anchor config in memory"):
        runtime.run_validation(config=config, binding=binding, anchor=changed_anchor, inputs=inputs,
                               inference_connector=inference, output_root=tmp_path / "anchor", device="cpu", fake=True)


def test_wrong_connector_dtype_device_and_frozen_state_rejected(tmp_path: Path) -> None:
    config, binding, anchor = runtime.load_accepted_inputs(ROOT)
    inputs, inference = cli._fake_inputs(binding)
    inference.dtype = torch.float32
    with pytest.raises(runtime.ValidationError, match="device or float16"):
        runtime.run_validation(config=config, binding=binding, anchor=anchor, inputs=inputs,
                               inference_connector=inference, output_root=tmp_path / "dtype", device="cpu", fake=True)
    inference.dtype = torch.float16
    inference.device = "mps"
    with pytest.raises(runtime.ValidationError, match="device or float16"):
        runtime.run_validation(config=config, binding=binding, anchor=anchor, inputs=inputs,
                               inference_connector=inference, output_root=tmp_path / "device", device="cpu", fake=True)
    inference.device = "cpu"
    inference.connector.train()
    with pytest.raises(runtime.ValidationError, match="frozen"):
        runtime.run_validation(config=config, binding=binding, anchor=anchor, inputs=inputs,
                               inference_connector=inference, output_root=tmp_path / "training", device="cpu", fake=True)


@pytest.mark.parametrize("kind", ["order", "shape", "nonfinite", "split"])
def test_invalid_bound_tensor_rejected(kind: str, tmp_path: Path) -> None:
    config, binding, anchor = runtime.load_accepted_inputs(ROOT)
    inputs, inference = cli._fake_inputs(binding)
    train = dict(inputs.train)
    batch = train["SIGLIP2"]
    if kind == "order":
        train["SIGLIP2"] = replace(batch, target_image_ids=("wrong",))
    elif kind == "shape":
        train["SIGLIP2"] = replace(batch, source=batch.source[:, :195])
    elif kind == "nonfinite":
        values = batch.source.clone(); values[0, 0, 0] = float("nan")
        train["SIGLIP2"] = replace(batch, source=values)
    else:
        train["SIGLIP2"] = replace(batch, split="test")
    with pytest.raises(core.ConnectorAwareLinearError):
        runtime.run_validation(config=config, binding=binding, anchor=anchor,
                               inputs=replace(inputs, train=train), inference_connector=inference,
                               output_root=tmp_path / kind, device="cpu", fake=True)


def test_zero_step_training_inference_parity_fails_closed(tmp_path: Path, monkeypatch) -> None:
    config, binding, anchor = runtime.load_accepted_inputs(ROOT)
    inputs, inference = cli._fake_inputs(binding)
    monkeypatch.setattr(inference, "apply", lambda patches, *, batch_size: np.zeros(
        (patches.shape[0], 196, 1536), dtype=np.float32
    ))
    with pytest.raises(runtime.ValidationError, match="parity"):
        runtime.run_validation(config=config, binding=binding, anchor=anchor,
                               inputs=inputs, inference_connector=inference,
                               output_root=tmp_path / "parity", device="cpu", fake=True)


def test_real_mode_rejects_synthetic_fixture_without_gpu(tmp_path: Path, monkeypatch) -> None:
    config, binding, anchor = runtime.load_accepted_inputs(ROOT)
    inputs, inference = cli._fake_inputs(binding)
    monkeypatch.setattr(runtime, "_check_connector",
                        lambda connector, device, fake: core.FrozenDifferentiableConnector(connector.connector))
    with pytest.raises(runtime.ValidationError, match="production inputs"):
        runtime.run_validation(config=config, binding=binding, anchor=anchor,
                               inputs=inputs, inference_connector=inference,
                               output_root=tmp_path / "real", device="cpu", fake=False)


def test_selected_artifact_rejects_wrong_runtime_identity(completed) -> None:
    output, result, _, _, _ = completed[:5]
    spec = result["sources"]["DINOV3"]["selected_artifacts"]["PATCH_CONNECTOR_1P0"]
    with pytest.raises(runtime.ValidationError, match="metadata"):
        runtime.load_selected_artifact((output / spec["relative_path"]).parent, runtime_identity="0" * 64)
