from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from vision_encoder_alignment.dinov3_extraction_boundary import (
    DINOv3ExtractionBoundaryError,
    DINOv3ExtractionRequest,
    MODEL_ID,
    MODEL_KEY,
    MODEL_STATE_SHA256,
    REQUIRED_MARKERS,
    REVISION,
    TAP_CATALOG,
    load_dinov3_extractor_profile,
    read_dinov3_extraction_request,
    validate_dinov3_extraction_request,
    validate_dinov3_extraction_response,
)
from vision_encoder_alignment.extractors import VisionEncoderExtractor
from vision_encoder_alignment.extractors.base import ExtractorValidationError
import vision_encoder_alignment.extractors.dinov3 as dinov3


ROOT = Path(__file__).resolve().parents[2]
LIVE_PROFILE = (
    ROOT / "configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json"
)
LIVE_PROFILE_SHA256 = (
    "60d0d0d7d37c6b4d47997e09619204f9e1eddb96118d5ee3528d9e6fcd5fe829"
)
CANDIDATE_PROFILE_SHA256 = (
    "60d0d0d7d37c6b4d47997e09619204f9e1eddb96118d5ee3528d9e6fcd5fe829"
)
REGISTRY_PIN_PROFILE_SHA256 = (
    "3281b94e188230c521e674cae301768a335dd635e7401b51acafeae053a137bc"
)


def _request(profile_path: Path, **updates: object) -> DINOv3ExtractionRequest:
    value: dict[str, object] = {
        "expected_worker_policy": {
            "cpu_only": True,
            "fp32_only": True,
            "local_files_only": True,
            "network_allowed": False,
            "processor_allowed": False,
            "torchvision_allowed": False,
        },
        "extractor_profile_sha256": load_dinov3_extractor_profile(profile_path).sha256,
        "image_ids": ["image-b", "image-a"],
        "input_dtype": "uint8",
        "input_file_sha256": "1" * 64,
        "input_shape": [2, 224, 224, 3],
        "model_id": MODEL_ID,
        "model_key": MODEL_KEY,
        "operation": "extract_canonical_features",
        "raw_input_sha256": "2" * 64,
        "requested_tap_names": ["block_12", "final_norm"],
        "revision": REVISION,
        "run_id": "unit-run",
        "schema_id": "vea.dinov3_extraction_request.v0",
        "workspace_relative_input_path": "input.npy",
        "workspace_relative_output_directory": "output",
    }
    value.update(updates)
    return validate_dinov3_extraction_request(value)


def _artifact(index: int, tap: str) -> dict[str, object]:
    prefix = f"output/{index:02d}-{tap}"
    return {
        "checksum_sha256": "3" * 64,
        "dtype": "float32",
        "global_sha256": "4" * 64,
        "global_shape": [2, 768],
        "patch_sha256": "5" * 64,
        "patch_shape": [2, 196, 768],
        "safetensors_relative_path": f"{prefix}/shard-{index:05d}.safetensors",
        "sidecar_relative_path": f"{prefix}/shard-{index:05d}.json",
        "tap_name": tap,
    }


def _response(
    profile_path: Path,
    request: DINOv3ExtractionRequest,
    **updates: object,
) -> dict[str, object]:
    profile = load_dinov3_extractor_profile(profile_path)
    artifacts = [_artifact(index, tap) for index, tap in enumerate(request.requested_tap_names)]
    value: dict[str, object] = {
        "artifacts": artifacts,
        "canonical_mapping": {
            "block_12_source": "raw_final_block_hook",
            "final_norm_source": "last_hidden_state",
            "global_source": "patch_axis_mean",
            "hidden_state_12_relation": "final_norm",
            "native_pooler_excluded": True,
            "patch_token_indices": list(range(5, 201)),
            "raw_block_12_differs_from_final_norm": True,
            "special_tokens_excluded": True,
        },
        "descriptor": {
            "grid_height": 14,
            "grid_width": 14,
            "hidden_dimension": 768,
            "layer_count": 12,
            "output_dtype": "float32",
            "patch_count": 196,
            "total_token_count": 201,
        },
        "failure_classification": None,
        "forward_count": 1,
        "identities": {
            "external_snapshot_manifest_sha256": profile.identities.external_snapshot_manifest_sha256,
            "extractor_profile_sha256": profile.sha256,
            "persistent_freeze_sha256": profile.identities.persistent_freeze_sha256,
            "persistent_runtime_manifest_sha256": profile.identities.persistent_runtime_manifest_sha256,
            "runtime_profile_sha256": profile.identities.runtime_profile_sha256,
            "snapshot_profile_sha256": profile.identities.snapshot_profile_sha256,
        },
        "image_ids": list(request.image_ids),
        "input": {
            "input_file_sha256": request.input_file_sha256,
            "input_unchanged": True,
            "normalized_dtype": "float32",
            "normalized_sha256": "6" * 64,
            "normalized_shape": [2, 3, 224, 224],
            "raw_input_sha256": request.raw_input_sha256,
        },
        "invariants": {
            "model_state_unchanged": True,
            "numpy_rng_unchanged": True,
            "parameters_frozen": True,
            "python_rng_unchanged": True,
            "torch_rng_unchanged": True,
            "training_false": True,
        },
        "markers": list(REQUIRED_MARKERS),
        "model_state_sha256": MODEL_STATE_SHA256,
        "operation": "extract_canonical_features",
        "processor_construction_count": 0,
        "requested_tap_names": list(request.requested_tap_names),
        "run_id": request.run_id,
        "schema_id": "vea.dinov3_extraction_response.v0",
        "socket_attempt_count": 0,
        "status": "success",
        "torchvision_import_count": 0,
        "warnings": [],
        "workspace_paths": [
            path
            for item in artifacts
            for path in (item["sidecar_relative_path"], item["safetensors_relative_path"])
        ],
    }
    value.update(updates)
    return value


def test_package_lazy_import_has_no_forbidden_delta() -> None:
    code = """
import sys
before=set(sys.modules)
import vision_encoder_alignment.extractors as e
assert e.DINOv3ExtractorConfig
assert e.DINOv3VisionEncoderExtractor
added=set(sys.modules)-before
forbidden=('torch','transformers','huggingface_hub','PIL','torchvision')
assert not any(n==r or n.startswith(r+'.') for n in added for r in forbidden), sorted(added)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": "src"},
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "relative",
    [
        "src/vision_encoder_alignment/extractors/dinov3.py",
        "src/vision_encoder_alignment/dinov3_extraction_boundary.py",
    ],
)
def test_source_has_no_forbidden_top_level_import(relative: str) -> None:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    forbidden = {"torch", "transformers", "huggingface_hub", "PIL", "torchvision"}
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    assert forbidden.isdisjoint(imports)


def test_live_profile_exact_v0_identity(
    dinov3_live_production_profile_path: Path,
) -> None:
    assert dinov3_live_production_profile_path == LIVE_PROFILE.resolve()
    payload = dinov3_live_production_profile_path.read_bytes()
    value = json.loads(payload)
    assert len(payload) == 12_506
    assert hashlib.sha256(payload).hexdigest() == LIVE_PROFILE_SHA256
    assert value["schema_id"] == "vea.dinov3_isolated_extractor_profile.v1"
    assert len(load_dinov3_extractor_profile(LIVE_PROFILE).production_sources) == 18


def test_live_profile_has_no_production_sources() -> None:
    value = json.loads(LIVE_PROFILE.read_bytes())
    assert len(value["production_sources"]) == 18


def test_v1_loader_rejects_live_profile(
    dinov3_registry_pin_profile_path: Path,
) -> None:
    payload = dinov3_registry_pin_profile_path.read_bytes()
    assert len(payload) == 8_885
    assert hashlib.sha256(payload).hexdigest() == REGISTRY_PIN_PROFILE_SHA256
    with pytest.raises(
        DINOv3ExtractionBoundaryError,
        match=r"missing=\['production_sources'\]",
    ):
        load_dinov3_extractor_profile(dinov3_registry_pin_profile_path)


def test_candidate_profile_exact_identity(
    dinov3_candidate_profile_path: Path,
) -> None:
    payload = dinov3_candidate_profile_path.read_bytes()
    profile = load_dinov3_extractor_profile(dinov3_candidate_profile_path)
    assert len(payload) == 12_506
    assert hashlib.sha256(payload).hexdigest() == CANDIDATE_PROFILE_SHA256
    assert profile.document.schema_id == "vea.dinov3_isolated_extractor_profile.v1"
    assert len(profile.production_sources) == 18


def test_candidate_fixture_preserves_live_profile(
    dinov3_candidate_profile_path: Path,
) -> None:
    assert dinov3_candidate_profile_path.is_file()
    assert hashlib.sha256(LIVE_PROFILE.read_bytes()).hexdigest() == LIVE_PROFILE_SHA256


def test_candidate_fixture_is_outside_repository(
    dinov3_candidate_profile_path: Path,
    dinov3_registry_pin_profile_path: Path,
) -> None:
    for path in (dinov3_candidate_profile_path, dinov3_registry_pin_profile_path):
        assert path.is_absolute()
        assert not path.is_relative_to(ROOT.resolve())
        assert not path.is_symlink()


def test_profile_is_exact_and_has_fourteen_taps(
    dinov3_candidate_profile_path: Path,
) -> None:
    profile = load_dinov3_extractor_profile(dinov3_candidate_profile_path)
    assert profile.document.schema_id == "vea.dinov3_isolated_extractor_profile.v1"
    assert len(profile.production_sources) == 18
    assert profile.model.key == MODEL_KEY
    assert profile.model.revision == REVISION
    assert profile.tap_catalog == TAP_CATALOG
    assert profile.document.reference_feature_hash_count == 28


def test_profile_is_frozen(dinov3_candidate_profile_path: Path) -> None:
    profile = load_dinov3_extractor_profile(dinov3_candidate_profile_path)
    with pytest.raises(AttributeError):
        profile.document.model = None


def test_config_valid_and_path_redacted() -> None:
    config = dinov3.DINOv3ExtractorConfig(LIVE_PROFILE)
    assert config.profile_path == LIVE_PROFILE.resolve()
    assert str(LIVE_PROFILE.resolve()) not in repr(config)
    assert "<redacted-repository-profile>" in repr(config)


def test_config_is_frozen_and_slotted() -> None:
    config = dinov3.DINOv3ExtractorConfig(LIVE_PROFILE)
    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.timeout_seconds = 3.0  # type: ignore[misc]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_config_rejects_invalid_timeout(timeout: object) -> None:
    with pytest.raises(ExtractorValidationError):
        dinov3.DINOv3ExtractorConfig(LIVE_PROFILE, timeout_seconds=timeout)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [("model_key", "other"), ("model_id", "other/model"), ("revision", "0" * 40)],
)
def test_config_rejects_wrong_identity(field: str, value: str) -> None:
    kwargs = {field: value}
    with pytest.raises(ExtractorValidationError):
        dinov3.DINOv3ExtractorConfig(LIVE_PROFILE, **kwargs)


def test_config_rejects_missing_profile(tmp_path: Path) -> None:
    with pytest.raises(ExtractorValidationError):
        dinov3.DINOv3ExtractorConfig(tmp_path / "missing.json")


def test_config_rejects_repository_overlap_root() -> None:
    with pytest.raises(ExtractorValidationError):
        dinov3.DINOv3ExtractorConfig(LIVE_PROFILE, runtime_root=ROOT)


def test_profile_rejects_wrong_model_revision(
    tmp_path: Path, dinov3_candidate_profile_path: Path
) -> None:
    value = json.loads(dinov3_candidate_profile_path.read_text(encoding="utf-8"))
    value["model"]["revision"] = "0" * 40
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DINOv3ExtractionBoundaryError):
        load_dinov3_extractor_profile(path.resolve())


def test_profile_rejects_unknown_field(
    tmp_path: Path, dinov3_candidate_profile_path: Path
) -> None:
    value = json.loads(dinov3_candidate_profile_path.read_text(encoding="utf-8"))
    value["unknown"] = True
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DINOv3ExtractionBoundaryError):
        load_dinov3_extractor_profile(path.resolve())


def test_profile_rejects_duplicate_json_key(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"schema_id":"x","schema_id":"y"}', encoding="utf-8")
    with pytest.raises(DINOv3ExtractionBoundaryError, match="duplicate"):
        load_dinov3_extractor_profile(path.resolve())


def test_request_reader_rejects_duplicate_json_key(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    path.write_text('{"schema_id":"x","schema_id":"y"}', encoding="utf-8")
    with pytest.raises(DINOv3ExtractionBoundaryError, match="duplicate"):
        read_dinov3_extraction_request(path.resolve())


def test_descriptor_exact() -> None:
    descriptor = dinov3._build_descriptor()
    assert descriptor.model_key == MODEL_KEY
    assert descriptor.implementation_class == "DINOv3VisionEncoderExtractor"
    assert descriptor.vision_component_path == "self"
    assert descriptor.patch_count == 196
    assert descriptor.hidden_dimension == 768
    assert descriptor.normalization.scale == float(np.float32(1 / 255))


def test_token_layout_exact() -> None:
    layout = dinov3._build_descriptor().token_layout
    assert layout.total_token_count == 201
    assert layout.special_token_indices == (0, 1, 2, 3, 4)
    assert layout.patch_token_indices == tuple(range(5, 201))


def test_tap_catalog_exact() -> None:
    assert tuple(tap.name for tap in dinov3._build_descriptor().taps) == TAP_CATALOG


def test_protocol_conformance_with_fake_context(
    monkeypatch: pytest.MonkeyPatch, dinov3_candidate_profile_path: Path
) -> None:
    profile = load_dinov3_extractor_profile(dinov3_candidate_profile_path)
    context = dinov3._ExtractorContext(
        profile=profile,
        runtime_root=ROOT,
        snapshot_root=ROOT,
        repository_root=ROOT,
        worker_script=dinov3_candidate_profile_path,
        descriptor=dinov3._build_descriptor(),
    )
    monkeypatch.setattr(dinov3, "_load_extractor_context", lambda config: context)
    extractor = dinov3.DINOv3VisionEncoderExtractor(
        dinov3.DINOv3ExtractorConfig(LIVE_PROFILE)
    )
    assert isinstance(extractor, VisionEncoderExtractor)


@pytest.mark.parametrize(
    "updates",
    [
        {"requested_tap_names": []},
        {"requested_tap_names": ["block_01", "block_01"]},
        {"requested_tap_names": ["unknown"]},
        {"input_shape": [2, 225, 224, 3]},
        {"input_dtype": "float32"},
        {"model_key": "other"},
        {"workspace_relative_input_path": "../escape.npy"},
        {"raw_input_sha256": "bad"},
    ],
)
def test_request_rejects_invalid_contract(
    updates: dict[str, object], dinov3_candidate_profile_path: Path
) -> None:
    base = _request(dinov3_candidate_profile_path).to_dict()
    base.update(updates)
    with pytest.raises(DINOv3ExtractionBoundaryError):
        validate_dinov3_extraction_request(base)


def test_response_validates_exact_contract(
    dinov3_candidate_profile_path: Path,
) -> None:
    request = _request(dinov3_candidate_profile_path)
    profile = load_dinov3_extractor_profile(dinov3_candidate_profile_path)
    response = validate_dinov3_extraction_response(
        _response(dinov3_candidate_profile_path, request),
        request=request,
        profile=profile,
    )
    assert response.document.forward_count == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "wrong"),
        ("model_state_sha256", "0" * 64),
        ("forward_count", 2),
        ("socket_attempt_count", 1),
        ("torchvision_import_count", 1),
        ("processor_construction_count", 1),
        ("warnings", ["warning"]),
    ],
)
def test_response_rejects_top_level_mismatch(
    field: str, value: object, dinov3_candidate_profile_path: Path
) -> None:
    request = _request(dinov3_candidate_profile_path)
    with pytest.raises(DINOv3ExtractionBoundaryError):
        validate_dinov3_extraction_response(
            _response(dinov3_candidate_profile_path, request, **{field: value}),
            request=request,
            profile=load_dinov3_extractor_profile(dinov3_candidate_profile_path),
        )


def test_response_rejects_token_layout_or_special_token_leakage(
    dinov3_candidate_profile_path: Path,
) -> None:
    request = _request(dinov3_candidate_profile_path)
    value = _response(dinov3_candidate_profile_path, request)
    value["canonical_mapping"]["patch_token_indices"] = list(range(201))  # type: ignore[index]
    with pytest.raises(DINOv3ExtractionBoundaryError, match="mapping"):
        validate_dinov3_extraction_response(
            value,
            request=request,
            profile=load_dinov3_extractor_profile(dinov3_candidate_profile_path),
        )


def test_response_rejects_native_pooler_global_source(
    dinov3_candidate_profile_path: Path,
) -> None:
    request = _request(dinov3_candidate_profile_path)
    value = _response(dinov3_candidate_profile_path, request)
    value["canonical_mapping"]["global_source"] = "native_pooler"  # type: ignore[index]
    with pytest.raises(DINOv3ExtractionBoundaryError, match="mapping"):
        validate_dinov3_extraction_response(
            value,
            request=request,
            profile=load_dinov3_extractor_profile(dinov3_candidate_profile_path),
        )


def test_response_rejects_raw_final_norm_mixup(
    dinov3_candidate_profile_path: Path,
) -> None:
    request = _request(dinov3_candidate_profile_path)
    value = _response(dinov3_candidate_profile_path, request)
    value["canonical_mapping"]["block_12_source"] = "hidden_states[12]"  # type: ignore[index]
    with pytest.raises(DINOv3ExtractionBoundaryError, match="mapping"):
        validate_dinov3_extraction_response(
            value,
            request=request,
            profile=load_dinov3_extractor_profile(dinov3_candidate_profile_path),
        )


def test_response_rejects_wrong_patch_shape(
    dinov3_candidate_profile_path: Path,
) -> None:
    request = _request(dinov3_candidate_profile_path)
    value = _response(dinov3_candidate_profile_path, request)
    value["artifacts"][0]["patch_shape"] = [2, 201, 768]  # type: ignore[index]
    with pytest.raises(DINOv3ExtractionBoundaryError):
        validate_dinov3_extraction_response(
            value,
            request=request,
            profile=load_dinov3_extractor_profile(dinov3_candidate_profile_path),
        )


def test_response_rejects_absolute_artifact_path(
    dinov3_candidate_profile_path: Path,
) -> None:
    request = _request(dinov3_candidate_profile_path)
    value = _response(dinov3_candidate_profile_path, request)
    value["artifacts"][0]["sidecar_relative_path"] = str((ROOT / "x.json").resolve())  # type: ignore[index]
    with pytest.raises(DINOv3ExtractionBoundaryError):
        validate_dinov3_extraction_response(
            value,
            request=request,
            profile=load_dinov3_extractor_profile(dinov3_candidate_profile_path),
        )
