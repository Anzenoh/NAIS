from __future__ import annotations

from pathlib import Path

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_bank_preflight as bank
from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as rat


CONFIG = Path("configs/vlm_compat/p2_query_adapter_linear_v1.json")
PRODUCTION_SHAPES_SMALL_COUNT = bank.PreflightExpectations(
    image_count=4,
    split_counts=(2, 1, 1),
    patch_shape=(196, 768),
    connector_shape=(196, 1536),
)


def test_preflight_accepts_current_t001d_npz_and_marker_contract(tmp_path: Path) -> None:
    config = linear.load_query_adapter_config(CONFIG)
    plan = bank.synthetic_t001d_plan(config, PRODUCTION_SHAPES_SMALL_COUNT)
    run_root = tmp_path / "t001d-contract"
    plan_id = bank.publish_synthetic_t001d_fixture(run_root, plan)

    source = next(unit for unit in plan["expected_units"] if unit["encoder"] == "DINOV3")
    target = next(unit for unit in plan["expected_units"] if unit["encoder"] == "TARGET_CLIP")
    assert rat.validate_unit_completion(run_root, source, plan_id)
    assert rat.validate_unit_completion(run_root, target, plan_id)
    with np.load(run_root / source["artifact_relative_path"], allow_pickle=False) as archive:
        assert set(archive.files) == {"image_ids", "patches", "globals", "plan_id"}
        assert archive["patches"].shape[1:] == (196, 768)
    with np.load(run_root / target["artifact_relative_path"], allow_pickle=False) as archive:
        assert set(archive.files) == {"image_ids", "patches", "globals", "connectors", "plan_id"}
        assert archive["patches"].shape[1:] == (196, 768)
        assert archive["connectors"].shape[1:] == (196, 1536)

    result = bank.preflight_query_adapter_banks(CONFIG, run_root, expectations=PRODUCTION_SHAPES_SMALL_COUNT)
    assert result["status"] == bank.SUCCESS_STATUS
    assert result["t001d_plan_id"] == plan_id
    assert result["banks"]["TARGET_CLIP"]["connector_shape"] == [196, 1536]
    assert result["cross_bank_image_order_equal"] is True
    assert result["manifest_order_equal"] is True


def test_preflight_uses_only_the_three_t001d_feature_stages(tmp_path: Path) -> None:
    config = linear.load_query_adapter_config(CONFIG)
    plan = bank.synthetic_t001d_plan(config, PRODUCTION_SHAPES_SMALL_COUNT)
    assert {unit["stage"] for unit in plan["expected_units"]} == set(bank.REQUIRED_STAGES.values())
    run_root = tmp_path / "three-stages"
    bank.publish_synthetic_t001d_fixture(run_root, plan)
    result = bank.preflight_query_adapter_banks(CONFIG, run_root, expectations=PRODUCTION_SHAPES_SMALL_COUNT)
    assert set(result["banks"]) == {"SIGLIP2", "DINOV3", "TARGET_CLIP"}
    assert all(value["chunk_count"] == 2 for value in result["banks"].values())
