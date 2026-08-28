from __future__ import annotations

import json
from pathlib import Path

from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear


CONFIG_PATH = Path("configs/vlm_compat/p2_query_adapter_linear_v1.json")
T001D_PATH = Path("configs/vlm_compat/p2_universal_anchor_transport_v0.json")


def test_checked_in_frozen_linear_config_matches_core_contract() -> None:
    config = linear.load_query_adapter_config(CONFIG_PATH)
    assert config["schema_id"] == "vea.p2.query_adapter_linear_config.v1"
    assert config["experiment_id"] == "vea.p2.query_adapter_linear.v1"
    assert config["source_order"] == ["SIGLIP2", "DINOV3"]
    assert config["target"]["key"] == "TARGET_CLIP"
    assert config["method_order"] == ["CALIBRATED", "ORTHOGONAL_PROCRUSTES", "FULL_LINEAR", "RIDGE"]
    assert len(config["methods"]["RIDGE"]["lambda_normalized_grid"]) == 9
    roles = config["manifest"]["partition_roles"]
    assert [roles[name]["image_count"] for name in ("train", "validation", "test")] == [800, 100, 100]
    assert linear.generation_plan(config)["total"] == 900


def test_checked_in_generation_block_remains_t001d_deterministic_values() -> None:
    config = linear.load_query_adapter_config(CONFIG_PATH)
    t001d = json.loads(T001D_PATH.read_text(encoding="utf-8"))
    generation = config["generation"]
    assert generation["prompt"] == t001d["prompt"]
    for key in ("max_new_tokens", "num_beams", "do_sample", "top_p", "temperature"):
        assert generation[key] == t001d["generation"][key]
