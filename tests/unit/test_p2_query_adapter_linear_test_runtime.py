from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_query_adapter_linear_test as cli
from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import query_adapter_linear_test_runtime as runtime


CONFIG = Path("configs/vlm_compat/p2_query_adapter_linear_v1.json")
BINDING = Path("configs/vlm_compat/p2_query_adapter_linear_test_binding_v1.json")


def test_checked_in_test_binding_freezes_exact_validation_selection() -> None:
    binding = runtime.load_test_binding(BINDING)
    assert binding["validation_result_sha256"] == "10777d3aa4b19683beecb3b78ae5cea21d20538ae126db9a65daf434aa604ac9"
    assert binding["validation_runtime_identity_sha256"] == "ac5438b0a38fb2a6ba8b36ea50215d1ff1df5bfa1bcee89a0f7a8d4887097026"
    assert binding["selected_ridge_lambda_normalized"] == {"SIGLIP2": 1e-6, "DINOV3": 1e-4}
    assert sum(len(methods) for methods in binding["finalized_adapter_artifacts"].values()) == 8


def test_production_generation_plan_is_exactly_100_plus_800() -> None:
    binding = runtime.load_test_binding(BINDING)
    config = linear.load_query_adapter_config(CONFIG)
    assert linear.generation_plan(config) == binding["generation_plan"] == {
        "test_images": 100, "native": 100, "adapted": 800, "total": 900
    }


def test_fake_e2e_has_eight_configs_distinct_paths_and_safe_resume(tmp_path: Path) -> None:
    result = cli.fake_smoke(CONFIG, BINDING, tmp_path / "fake")
    assert result["status"] == runtime.SUCCESS_STATUS
    assert result["finalized_configuration_count"] == 8
    assert result["generation_counts"] == {"native": 3, "adapted": 24, "total": 27}
    assert result["production_generation_plan"]["total"] == 900
    assert result["resume_skipped_generation_count"] == 5
    assert result["fake_validation"]["native_and_adapted_paths_distinct"] is True
    assert result["fake_validation"]["injection_restored"] is True


def test_fake_runtime_never_refits_or_reselects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fit/selection forbidden"))
    for name in ("fit_calibration", "fit_orthogonal_procrustes", "fit_full_linear", "fit_ridge", "select_ridge_candidate"):
        monkeypatch.setattr(linear, name, forbidden)
    result = cli.fake_smoke(CONFIG, BINDING, tmp_path / "no-refit")
    assert result["status"] == runtime.SUCCESS_STATUS


def test_validation_identity_must_pass_before_artifact_loading(tmp_path: Path) -> None:
    with pytest.raises(runtime.TestRuntimeError, match="result SHA"):
        runtime.verify_validation_run(runtime.load_test_binding(BINDING), tmp_path)


def test_generation_artifact_identity_corruption_fails_closed(tmp_path: Path) -> None:
    result = cli.fake_smoke(CONFIG, BINDING, tmp_path / "corrupt")
    assert result["status"] == runtime.SUCCESS_STATUS
    unit = tmp_path / "corrupt" / "generations" / "native" / "0000.json"
    value = unit.read_text(encoding="utf-8").replace('"image_id":"fake-0"', '"image_id":"wrong"')
    unit.write_text(value, encoding="utf-8")
    with pytest.raises(runtime.TestRuntimeError, match="identity differs"):
        runtime._load_generation(
            unit,
            {"runtime_identity_sha256": result["runtime_identity_sha256"], "image_id": "fake-0", "source": "TARGET_CLIP", "method": "NATIVE"},
        )
