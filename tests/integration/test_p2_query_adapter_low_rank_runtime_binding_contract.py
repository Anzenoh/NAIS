"""Synthetic end-to-end contract for the T004C handoff; no real artifacts."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_runtime_binding as binding


def test_public_entry_point_has_only_explicit_roots() -> None:
    parameters = list(inspect.signature(binding.bind_low_rank_runtime).parameters)
    assert parameters == ["repository_root", "t001d_run_root", "t002d_validation_root"]
    assert "fixture" not in parameters
    assert binding.SOURCE_ORDER == ("SIGLIP2", "DINOV3")
    assert core.MAX_CHUNK_ROWS == 4096


def test_factory_caps_real_patch_rows_without_opening_any_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = tuple(f"train-{index:02d}" for index in range(21))
    factory = binding.TrainChunkFactory("SIGLIP2", (), (), ids, (196, 768))

    def rows(units):
        value = np.zeros((196, 768), dtype=np.float32)
        for image_id in ids:
            yield image_id, value.copy()

    monkeypatch.setattr(factory, "_rows", rows)
    first = list(factory())
    second = list(factory())
    assert [len(chunk.source_image_ids) for chunk in first] == [20, 1]
    assert [len(chunk.source_image_ids) for chunk in second] == [20, 1]
    assert all(chunk.source.shape[0] * 196 <= 4096 for chunk in first)
    assert first[0].source_image_ids == ids[:20]
    assert first[1].source_image_ids == ids[20:]
    assert all(np.array_equal(a.source, b.source) for a, b in zip(first, second, strict=True))


def test_no_public_test_or_validation_data_api() -> None:
    assert set(binding.BoundLowRankInputs.__dataclass_fields__) == {
        "config", "ols", "train_factories", "provenance"
    }
    assert not hasattr(binding.BoundLowRankInputs, "validation")
    assert not hasattr(binding.BoundLowRankInputs, "test")
    assert not hasattr(binding, "fit_low_rank_rrr")
