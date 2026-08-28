from __future__ import annotations

import copy
import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_bank_preflight as bank
from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as rat


CONFIG = Path("configs/vlm_compat/p2_query_adapter_linear_v1.json")
SMALL = bank.PreflightExpectations(image_count=4, split_counts=(2, 1, 1), patch_shape=(2, 3), connector_shape=(2, 4))


def config() -> dict[str, object]:
    return linear.load_query_adapter_config(CONFIG)


def make_run(
    tmp_path: Path,
    *,
    mutate_plan=None,
    mutate_arrays=None,
    fill_offset: float = 0.0,
) -> tuple[Path, dict[str, object], str]:
    plan = bank.synthetic_t001d_plan(config(), SMALL)
    if mutate_plan is not None:
        mutate_plan(plan)
    root = tmp_path / "run"
    plan_id = bank.publish_synthetic_t001d_fixture(
        root, plan, fill_offset=fill_offset, array_mutator=mutate_arrays
    )
    return root, plan, plan_id


def preflight(root: Path) -> dict[str, object]:
    return bank.preflight_query_adapter_banks(CONFIG, root, expectations=SMALL)


def classification(error: pytest.ExceptionInfo[bank.BankPreflightError]) -> str:
    return error.value.classification


def test_valid_config_binding_and_config_hash(tmp_path: Path) -> None:
    root, _, _ = make_run(tmp_path)
    result = preflight(root)
    assert result["status"] == bank.SUCCESS_STATUS
    assert result["adapter_config_sha256"] == rat.sha256_file(CONFIG)
    assert result["adapter_experiment_id"] == "vea.p2.query_adapter_linear.v1"


def test_invalid_adapter_config_is_classified(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    root, _, _ = make_run(tmp_path / "fixture")
    with pytest.raises(bank.BankPreflightError) as caught:
        bank.preflight_query_adapter_banks(invalid, root, expectations=SMALL)
    assert classification(caught) == "BLOCKED_BANK_PREFLIGHT_CONFIG"


def test_wrong_manifest_sha_is_rejected(tmp_path: Path) -> None:
    root, _, _ = make_run(tmp_path, mutate_plan=lambda plan: plan.update(manifest_sha256="f" * 64))
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_T001D_PLAN_IDENTITY"


def test_wrong_t001d_plan_identity_is_rejected(tmp_path: Path) -> None:
    root, _, _ = make_run(tmp_path, mutate_plan=lambda plan: plan.update(task_id="VEA-P2-OTHER"))
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_T001D_PLAN_IDENTITY"


@pytest.mark.parametrize(("key", "value"), [("revision", "wrong"), ("model_id", "wrong/model")])
def test_encoder_model_or_revision_mismatch_is_rejected(tmp_path: Path, key: str, value: str) -> None:
    def mutate(plan):
        plan["identity_bindings"]["encoders"]["DINOV3"][key] = value

    root, _, _ = make_run(tmp_path, mutate_plan=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_MODEL_IDENTITY"


def test_tap_mismatch_is_rejected(tmp_path: Path) -> None:
    def mutate(plan):
        plan["identity_bindings"]["encoders"]["SIGLIP2"]["tap"] = "wrong tap"

    root, _, _ = make_run(tmp_path, mutate_plan=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_MODEL_IDENTITY"


def test_source_wrong_shape_is_rejected(tmp_path: Path) -> None:
    def mutate(unit, arrays):
        if unit["encoder"] == "DINOV3" and unit["chunk_index"] == 0:
            arrays["patches"] = np.zeros((2, 2, 4), dtype=np.float32)

    root, _, _ = make_run(tmp_path, mutate_arrays=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_PAYLOAD_CONTRACT"


def test_source_wrong_dtype_is_rejected(tmp_path: Path) -> None:
    def mutate(unit, arrays):
        if unit["encoder"] == "SIGLIP2" and unit["chunk_index"] == 0:
            arrays["globals"] = arrays["globals"].astype(np.float64)

    root, _, _ = make_run(tmp_path, mutate_arrays=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_PAYLOAD_CONTRACT"


def test_target_connector_wrong_shape_is_rejected(tmp_path: Path) -> None:
    def mutate(unit, arrays):
        if unit["encoder"] == "TARGET_CLIP" and unit["chunk_index"] == 0:
            arrays["connectors"] = np.zeros((2, 2, 5), dtype=np.float32)

    root, _, _ = make_run(tmp_path, mutate_arrays=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_PAYLOAD_CONTRACT"


def test_nonfinite_feature_is_rejected(tmp_path: Path) -> None:
    def mutate(unit, arrays):
        if unit["encoder"] == "DINOV3" and unit["chunk_index"] == 1:
            arrays["patches"].flat[0] = np.nan

    root, _, _ = make_run(tmp_path, mutate_arrays=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_PAYLOAD_CONTRACT"


def test_wrong_persisted_plan_id_is_rejected(tmp_path: Path) -> None:
    def mutate(unit, arrays):
        if unit["encoder"] == "SIGLIP2" and unit["chunk_index"] == 0:
            arrays["plan_id"] = np.asarray("0" * 64, dtype="U64")

    root, _, _ = make_run(tmp_path, mutate_arrays=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_PAYLOAD_CONTRACT"


def test_incomplete_unit_is_rejected(tmp_path: Path) -> None:
    root, plan, _ = make_run(tmp_path)
    rat.marker_path(root, plan["expected_units"][0]["unit_id"]).unlink()
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_T001D_UNIT_INCOMPLETE"


def test_artifact_byte_corruption_is_rejected(tmp_path: Path) -> None:
    root, plan, _ = make_run(tmp_path)
    artifact = root / plan["expected_units"][0]["artifact_relative_path"]
    with artifact.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_T001D_ARTIFACT_HASH"


def test_unexpected_scientific_payload_field_is_rejected(tmp_path: Path) -> None:
    def mutate(unit, arrays):
        if unit["encoder"] == "SIGLIP2" and unit["chunk_index"] == 0:
            arrays["substitute"] = np.zeros(1, dtype=np.float32)

    root, _, _ = make_run(tmp_path, mutate_arrays=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_PAYLOAD_CONTRACT"


def test_duplicate_manifest_image_id_is_rejected(tmp_path: Path) -> None:
    def mutate(plan):
        plan["manifest_rows"][1]["image_id"] = plan["manifest_rows"][0]["image_id"]

    root, _, _ = make_run(tmp_path, mutate_plan=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_IMAGE_ORDER"


def test_cross_bank_order_mismatch_is_rejected(tmp_path: Path) -> None:
    def mutate(plan):
        unit = next(value for value in plan["expected_units"] if value["encoder"] == "SIGLIP2")
        unit["image_ids"] = list(reversed(unit["image_ids"]))

    root, _, _ = make_run(tmp_path, mutate_plan=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_IMAGE_ORDER"


def test_manifest_order_mismatch_is_rejected(tmp_path: Path) -> None:
    def mutate(plan):
        plan["manifest_rows"][0], plan["manifest_rows"][1] = plan["manifest_rows"][1], plan["manifest_rows"][0]

    root, _, _ = make_run(tmp_path, mutate_plan=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_IMAGE_ORDER"


def test_split_overlap_or_count_failure_is_rejected(tmp_path: Path) -> None:
    def mutate(plan):
        plan["manifest_rows"][0]["split"] = "validation"

    root, _, _ = make_run(tmp_path, mutate_plan=mutate)
    with pytest.raises(bank.BankPreflightError) as caught:
        preflight(root)
    assert classification(caught) == "BLOCKED_BANK_SPLIT_IDENTITY"


def test_deterministic_digest_repeatability(tmp_path: Path) -> None:
    root, _, _ = make_run(tmp_path)
    first = preflight(root)
    second = preflight(root)
    assert first["banks"] == second["banks"]


def test_digest_changes_when_artifact_byte_identity_changes(tmp_path: Path) -> None:
    first_root, _, _ = make_run(tmp_path / "first", fill_offset=0.0)
    second_root, _, _ = make_run(tmp_path / "second", fill_offset=1.0)
    first = preflight(first_root)
    second = preflight(second_root)
    assert first["banks"]["SIGLIP2"]["bank_identity_sha256"] != second["banks"]["SIGLIP2"]["bank_identity_sha256"]


def test_digest_is_independent_of_absolute_run_root(tmp_path: Path) -> None:
    first_root, _, first_plan_id = make_run(tmp_path / "first")
    second_root, _, second_plan_id = make_run(tmp_path / "second")
    assert first_plan_id == second_plan_id
    first = preflight(first_root)
    second = preflight(second_root)
    assert {
        name: value["bank_identity_sha256"] for name, value in first["banks"].items()
    } == {
        name: value["bank_identity_sha256"] for name, value in second["banks"].items()
    }


def test_implementation_never_concatenates_full_bank_tensors() -> None:
    source = inspect.getsource(bank.preflight_query_adapter_banks) + inspect.getsource(bank._validate_bank)
    assert "concatenate" not in source and "np.stack" not in source


def test_import_has_no_heavy_model_runtime() -> None:
    code = (
        "import sys; import vision_encoder_alignment.vlm_compat.query_adapter_bank_preflight; "
        "assert not {'torch','transformers','huggingface_hub'}.intersection(sys.modules)"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment, check=False)
    assert completed.returncode == 0, completed.stderr
