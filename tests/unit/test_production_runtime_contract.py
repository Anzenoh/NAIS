from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vision_encoder_alignment.production_runtime_contract import (
    APPLE_DEVICE,
    APPLE_GPU_NAME,
    DINOV3_SELECTOR,
    MAIN_SELECTOR,
    ProductionRuntimeContractError,
    RuntimeProbe,
    canonical_json_bytes,
    load_production_runtime_profile,
    validate_apple_tensor_contract,
    validate_profile_source_bindings,
    validate_runtime_probe,
)


REPOSITORY = Path(__file__).resolve().parents[2]
MAIN_PROFILE = REPOSITORY / "configs/runtime_profiles/p1_apple_main_cuda_fp32_v0.json"
DINO_PROFILE = REPOSITORY / "configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json"


def _probe(**changes: object) -> RuntimeProbe:
    values: dict[str, object] = {
        "hostname": "apple",
        "operating_system": "Linux",
        "architecture": "x86_64",
        "glibc_version": "2.35",
        "python_version": "3.12.0",
        "torch_runtime_version": "2.8.0+cu128",
        "torch_cuda_build": "12.8",
        "cuda_available": True,
        "visible_gpu_count": 1,
        "selected_device_index": 0,
        "gpu_names": (APPLE_GPU_NAME,),
        "offline": True,
        "autocast_enabled": False,
        "installed_packages": {},
    }
    values.update(changes)
    return RuntimeProbe(**values)  # type: ignore[arg-type]


def _mutated_profile(tmp_path: Path, mutate: object) -> Path:
    value = json.loads(MAIN_PROFILE.read_text(encoding="utf-8"))
    mutate(value)  # type: ignore[operator]
    path = tmp_path / "profile.json"
    path.write_bytes(canonical_json_bytes(value))
    return path


def test_main_profile_loads_with_explicit_selector() -> None:
    profile = load_production_runtime_profile(MAIN_PROFILE)
    assert profile.selector == MAIN_SELECTOR
    assert profile.device == APPLE_DEVICE


def test_dinov3_profile_loads_as_independent_lane() -> None:
    profile = load_production_runtime_profile(DINO_PROFILE)
    assert profile.selector == DINOV3_SELECTOR
    assert profile.system_site_packages is False


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_id":"x","schema_id":"y"}\n', encoding="utf-8")
    with pytest.raises(ProductionRuntimeContractError, match="duplicate"):
        load_production_runtime_profile(path)


def test_nonfinite_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ProductionRuntimeContractError, match="non-finite"):
        load_production_runtime_profile(path)


def test_noncanonical_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "noncanonical.json"
    path.write_text(MAIN_PROFILE.read_text(encoding="utf-8").rstrip(), encoding="utf-8")
    with pytest.raises(ProductionRuntimeContractError, match="canonical"):
        load_production_runtime_profile(path)


def test_wrong_hostname_profile_is_rejected(tmp_path: Path) -> None:
    path = _mutated_profile(tmp_path, lambda value: value["target"].update(hostname="pear"))
    with pytest.raises(ProductionRuntimeContractError, match="Apple target"):
        load_production_runtime_profile(path)


def test_resolved_metadata_is_exact_but_not_installation_authority() -> None:
    profile = load_production_runtime_profile(MAIN_PROFILE)
    result = validate_runtime_probe(
        profile,
        _probe(installed_packages=dict(profile.resolved_versions)),
        require_resolved_packages=True,
    )
    assert len(profile.resolved_versions) == 40
    assert len(profile.direct_requirements) == 7
    assert result.package_metadata_resolved is True
    assert result.installable is False


def test_fake_cuda_probe_validates_capability_only() -> None:
    profile = load_production_runtime_profile(MAIN_PROFILE)
    result = validate_runtime_probe(
        profile, _probe(installed_packages=dict(profile.resolved_versions))
    )
    assert result.host_hardware_validated is True
    assert result.installable is False
    assert result.ready_for_p1_extraction is False


def test_cpu_only_torch_probe_is_rejected() -> None:
    profile = load_production_runtime_profile(MAIN_PROFILE)
    with pytest.raises(ProductionRuntimeContractError) as captured:
        validate_runtime_probe(
            profile,
            _probe(
                torch_cuda_build=None,
                cuda_available=False,
                installed_packages=dict(profile.resolved_versions),
            ),
        )
    assert captured.value.classification == "CUDA_RUNTIME_UNAVAILABLE"


def test_wrong_gpu_name_is_rejected() -> None:
    with pytest.raises(ProductionRuntimeContractError, match="gpu_names"):
        validate_runtime_probe(
            load_production_runtime_profile(MAIN_PROFILE), _probe(gpu_names=("A100",))
        )


def test_gpu_count_mismatch_is_rejected() -> None:
    with pytest.raises(ProductionRuntimeContractError, match="visible_gpu_count"):
        validate_runtime_probe(
            load_production_runtime_profile(MAIN_PROFILE), _probe(visible_gpu_count=2)
        )


def test_online_probe_is_rejected() -> None:
    with pytest.raises(ProductionRuntimeContractError, match="offline"):
        validate_runtime_probe(load_production_runtime_profile(MAIN_PROFILE), _probe(offline=False))


def test_runtime_probe_is_immutable_dataclass() -> None:
    assert replace(_probe(), hostname="pear").hostname == "pear"
    assert _probe().hostname == "apple"


def test_main_profile_source_bindings_match_repository() -> None:
    rows = validate_profile_source_bindings(load_production_runtime_profile(MAIN_PROFILE), REPOSITORY)
    assert len(rows) == 3


def test_dinov3_profile_source_bindings_match_repository() -> None:
    rows = validate_profile_source_bindings(load_production_runtime_profile(DINO_PROFILE), REPOSITORY)
    assert len(rows) == 5
    assert all("torchvision" not in row.path for row in rows)


class _Tensor:
    def __init__(self, device: str = "cuda:0", dtype: str = "float32") -> None:
        self.device = device
        self.dtype = dtype


def test_fake_apple_tensor_contract_accepts_cuda_fp32() -> None:
    validate_apple_tensor_contract([_Tensor()], expected_dtype="float32", finite_probe=lambda _: True)


def test_fake_apple_tensor_contract_rejects_cpu() -> None:
    with pytest.raises(ProductionRuntimeContractError):
        validate_apple_tensor_contract([_Tensor("cpu")], expected_dtype="float32", finite_probe=lambda _: True)


def test_fake_apple_tensor_contract_rejects_wrong_dtype() -> None:
    with pytest.raises(ProductionRuntimeContractError):
        validate_apple_tensor_contract([_Tensor(dtype="float16")], expected_dtype="float32", finite_probe=lambda _: True)


def test_fake_apple_tensor_contract_rejects_nonfinite() -> None:
    with pytest.raises(ProductionRuntimeContractError):
        validate_apple_tensor_contract([_Tensor()], expected_dtype="float32", finite_probe=lambda _: False)


def test_hybrid_lane_selector_is_rejected(tmp_path: Path) -> None:
    path = _mutated_profile(
        tmp_path,
        lambda value: value["lane"].update(selector=DINOV3_SELECTOR),
    )
    with pytest.raises(ProductionRuntimeContractError) as captured:
        load_production_runtime_profile(path)
    assert captured.value.classification == "HYBRID_RUNTIME_PROFILE"


def test_profiles_cannot_forge_extraction_readiness() -> None:
    for path in (MAIN_PROFILE, DINO_PROFILE):
        profile = load_production_runtime_profile(path)
        assert profile.ready_for_p1_extraction is False
        assert profile.package_metadata_resolved is True


def test_dinov3_resolved_closure_forbids_torchvision() -> None:
    profile = load_production_runtime_profile(DINO_PROFILE)
    assert len(profile.resolved_versions) == 40
    assert len(profile.direct_requirements) == 6
    assert profile.resolved_versions["hf-xet"] == "1.5.2"
    assert profile.resolved_versions["torch"] == "2.8.0+cu128"
    assert "torchvision" in profile.forbidden_distributions


def test_wrong_exact_installed_version_is_rejected() -> None:
    profile = load_production_runtime_profile(MAIN_PROFILE)
    installed = dict(profile.resolved_versions)
    installed["numpy"] = "0.0.0"
    with pytest.raises(ProductionRuntimeContractError) as captured:
        validate_runtime_probe(profile, _probe(installed_packages=installed))
    assert captured.value.classification == "PACKAGE_IDENTITY_MISMATCH"


@pytest.mark.parametrize(
    "field",
    [
        "torchvision_distribution_present",
        "torchvision_module_spec_present",
        "torchvision_imported_module_present",
    ],
)
def test_dinov3_torchvision_three_view_contamination_is_rejected(field: str) -> None:
    profile = load_production_runtime_profile(DINO_PROFILE)
    changes = {field: True, "installed_packages": dict(profile.resolved_versions)}
    with pytest.raises(ProductionRuntimeContractError) as captured:
        validate_runtime_probe(profile, _probe(**changes))
    assert captured.value.classification == "PACKAGE_IDENTITY_MISMATCH"
