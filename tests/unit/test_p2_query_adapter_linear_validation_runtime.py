from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import query_adapter_linear_validation_runtime as runtime
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


CONFIG = Path("configs/vlm_compat/p2_query_adapter_linear_v1.json")
BINDING = Path("configs/vlm_compat/p2_query_adapter_linear_bank_binding_v1.json")


class FakeConnector:
    def __init__(self) -> None:
        self.generation_calls = 0
        self.vision_calls = 0
        self.connector_calls = 0
        self.weights = np.array([[1.0, 0.0, 0.2, 0.1], [0.1, 1.0, 0.0, 0.2], [0.2, 0.1, 1.0, 0.0]], dtype=np.float32)

    def apply(self, patches: np.ndarray, *, batch_size: int) -> np.ndarray:
        self.connector_calls += (patches.shape[0] + batch_size - 1) // batch_size
        return np.ascontiguousarray(patches @ self.weights, dtype=np.float32)


def fake_banks() -> tuple[dict[str, object], FakeConnector]:
    rng = np.random.default_rng(22)
    target = rng.normal(size=(12, 2, 3)).astype(np.float32) + 1.0
    connector = FakeConnector()
    ids = [f"id-{index}" for index in range(12)]
    banks: dict[str, object] = {
        "TARGET_CLIP": {
            "train": target[:8], "validation": target[8:],
            "ids": {"train": ids[:8], "validation": ids[8:]},
            "connectors": {"train": connector.apply(target[:8], batch_size=3), "validation": connector.apply(target[8:], batch_size=3)},
        }
    }
    connector.connector_calls = 0
    for index, name in enumerate(linear.SOURCE_ORDER):
        value = np.ascontiguousarray(target * (1.0 + index * 0.1) + 0.05, dtype=np.float32)
        banks[name] = {"train": value[:8], "validation": value[8:], "ids": {"train": ids[:8], "validation": ids[8:]}}
    return banks, connector


def run(tmp_path: Path, connector: FakeConnector | None = None) -> dict[str, object]:
    banks, default_connector = fake_banks()
    return runtime.run_train_validation(
        config=linear.load_query_adapter_config(CONFIG), binding=runtime.load_bank_binding(BINDING), banks=banks,
        connector=connector or default_connector, output_root=tmp_path / "runtime", connector_batch_size=3,
        adapter_config_sha256=t001d.sha256_file(CONFIG), binding_config_sha256=t001d.sha256_file(BINDING), fake=True,
    )


def test_fake_train_validation_executes_all_frozen_methods_and_lanes(tmp_path: Path) -> None:
    result = run(tmp_path)
    assert result["status"] == runtime.SUCCESS_STATUS
    assert set(result["sources"]) == set(linear.SOURCE_ORDER)
    for source in result["sources"].values():
        assert len(source["methods"]) == 12
        assert source["ridge_selection"]["selected_lambda_normalized"] in linear.RIDGE_LAMBDA_NORMALIZED_GRID
    assert result["test_scientific_access"] == result["test_metric_count"] == result["generation_call_count"] == 0


def test_runtime_fit_artifacts_are_hash_bound_and_safe_resume(tmp_path: Path) -> None:
    first = run(tmp_path)
    second = run(tmp_path)
    assert second["resumed"] is True
    assert first["runtime_identity_sha256"] == second["runtime_identity_sha256"]
    for relative, digest in first["fit_artifacts"].items():
        assert t001d.sha256_file(tmp_path / "runtime" / relative) == digest


def test_binding_mismatch_rejects_preflight_report() -> None:
    binding = runtime.load_bank_binding(BINDING)
    report = {
        "status": "PASS_QUERY_ADAPTER_BANK_PREFLIGHT", "adapter_experiment_id": linear.EXPERIMENT_ID,
        "adapter_config_sha256": binding["adapter_config_sha256"], "t001d_plan_id": binding["t001d_plan_id"],
        "manifest_sha256": binding["manifest_sha256"], "cross_bank_image_order_equal": True,
        "manifest_order_equal": True, "all_required_units_valid": True,
        "banks": {name: {"bank_identity_sha256": digest} for name, digest in binding["banks"].items()},
    }
    report["banks"]["DINOV3"]["bank_identity_sha256"] = "0" * 64
    with pytest.raises(runtime.ValidationRuntimeError, match="differs"):
        runtime.verify_preflight_binding(report, binding, adapter_config_sha256=binding["adapter_config_sha256"])


def test_connector_generation_or_vision_activity_fails_closed(tmp_path: Path) -> None:
    connector = FakeConnector()
    connector.generation_calls = 1
    with pytest.raises(runtime.ValidationRuntimeError, match="vision/generation"):
        run(tmp_path, connector)
