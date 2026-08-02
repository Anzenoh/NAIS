"""Unit tests for the import-light DINOv3 semantic-probe boundary."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import vision_encoder_alignment.dinov3_semantic_boundary as boundary
from vision_encoder_alignment.dinov3_semantic_boundary import (
    DINOv3SemanticBoundaryError,
    INPUT_SHA256,
    MODEL_STATE_SHA256,
    NORMALIZED_SHA256,
    REQUIRED_SEMANTIC_MARKERS,
    SELECTED_ARTIFACT_TAPS,
    SEMANTIC_REQUEST_SCHEMA_ID,
    SEMANTIC_RESPONSE_SCHEMA_ID,
    SEMANTIC_SUCCESS_MARKER,
    TAP_CATALOG,
    load_semantic_profile,
    read_semantic_request,
    resolve_workspace_relative_path,
    semantic_response_projection,
    validate_semantic_response,
    write_semantic_request,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _profile_path() -> Path:
    return (
        _repository_root()
        / "configs"
        / "semantic_probes"
        / "dinov3_vit_b16_r5931719_semantics_v0.json"
    )


def _profile_value() -> dict[str, Any]:
    return json.loads(_profile_path().read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _modified_profile(tmp_path: Path, mutate: Any) -> Path:
    value = _profile_value()
    mutate(value)
    path = (tmp_path / "profile.json").resolve()
    _write_json(path, value)
    return path


def _hash(number: int) -> str:
    return f"{number:064x}"


def _valid_response(profile: Any, *, run_id: str = "t007a4-unit") -> dict[str, Any]:
    patch_indices = list(range(5, 201))
    special_indices = [0, 1, 2, 3, 4]
    taps = []
    for index, name in enumerate(TAP_CATALOG, start=1):
        taps.append(
            {
                "dtype": "float32",
                "finite": True,
                "full_sequence_shape": [2, 201, 768],
                "global_matches_patch_mean": True,
                "global_sha256": _hash(index * 2),
                "global_shape": [2, 768],
                "name": name,
                "patch_sha256": _hash(index * 2 - 1),
                "patch_shape": [2, 196, 768],
            }
        )
    tap_by_name = {item["name"]: item for item in taps}
    entries = []
    workspace_paths = []
    for index, name in enumerate(SELECTED_ARTIFACT_TAPS, start=1):
        sidecar = f"actual-output/{name}.json"
        tensors = f"actual-output/{name}.safetensors"
        entries.append(
            {
                "checksum_sha256": _hash(100 + index),
                "global_sha256": tap_by_name[name]["global_sha256"],
                "patch_sha256": tap_by_name[name]["patch_sha256"],
                "safetensors_relative_path": tensors,
                "sidecar_relative_path": sidecar,
                "tap_name": name,
            }
        )
        workspace_paths.extend((sidecar, tensors))
    return {
        "artifact": {
            "entries": entries,
            "roundtrip_validated": True,
            "selected_taps": list(SELECTED_ARTIFACT_TAPS),
            "workspace_relative_directory": "actual-output",
        },
        "components": {
            "cls_token": "embeddings.cls_token",
            "embeddings": "embeddings",
            "encoder_layers": "layer",
            "model": "self",
            "norm": "norm",
            "patch_embeddings": "embeddings.patch_embeddings",
            "pooler": "pooler_output",
            "register_tokens": "embeddings.register_tokens",
        },
        "embedding": {
            "hidden_state_0_matches": True,
            "hidden_state_0_max_abs_diff": 0.0,
            "hidden_state_0_shape": [2, 201, 768],
            "patch_embedding_dtype": "float32",
            "patch_embedding_hook_matches": True,
            "patch_embedding_hook_max_abs_diff": 0.0,
            "patch_embedding_shape": [2, 196, 768],
            "sequence_dtype": "float32",
            "sequence_shape": [2, 201, 768],
        },
        "final_norm": {
            "canonical_block_12_source": "raw_final_block_hook",
            "canonical_final_norm_source": "explicit_model_norm",
            "explicit_norm_matches_last_hidden": True,
            "explicit_norm_max_abs_diff": 0.0,
            "explicit_norm_shape": [2, 201, 768],
            "hidden_state_12_matches_last_hidden": True,
            "hidden_state_12_max_abs_diff": 0.0,
            "last_hidden_shape": [2, 201, 768],
            "raw_final_differs_from_last_hidden": True,
            "raw_final_shape": [2, 201, 768],
            "raw_final_vs_last_hidden_max_abs_diff": 1.0,
        },
        "forward_count": 2,
        "hidden_states": {
            "blocks": [
                {
                    "block_index": index,
                    "hidden_state_index": index,
                    "hidden_state_matches_hook": index < 12,
                    "hidden_state_shape": [2, 201, 768],
                    "hook_shape": [2, 201, 768],
                    "max_abs_diff": 0.0 if index < 12 else 1.0,
                    "tap_name": f"block_{index:02d}",
                }
                for index in range(1, 13)
            ],
            "count": 13,
            "embedding_hidden_state_index": 0,
            "embedding_matches": True,
            "embedding_max_abs_diff": 0.0,
            "shapes": [[2, 201, 768] for _ in range(13)],
        },
        "identities": {
            **dict(profile.to_dict()["identities"]),
            "semantic_profile_sha256": profile.sha256,
            "source_hashes": profile.to_dict()["installed_sources"],
        },
        "input": {
            "crop_count": 0,
            "image_ids": ["image-b", "image-a"],
            "image_processor_construction_count": 0,
            "input_dtype": "uint8",
            "input_sha256": INPUT_SHA256,
            "input_shape": [2, 224, 224, 3],
            "input_unchanged": True,
            "normalized_c_contiguous": True,
            "normalized_dtype": "float32",
            "normalized_finite": True,
            "normalized_sha256": NORMALIZED_SHA256,
            "normalized_shape": [2, 3, 224, 224],
            "pil_conversion_count": 0,
            "resize_count": 0,
            "torchvision_import_count": 0,
        },
        "invariants": {
            "model_state_unchanged": True,
            "numpy_rng_unchanged": True,
            "python_rng_unchanged": True,
            "torch_rng_unchanged": True,
            "training_false": True,
        },
        "markers": list(REQUIRED_SEMANTIC_MARKERS),
        "model": {
            "compute_dtype": "float32",
            "config_class": "DINOv3ViTConfig",
            "device": "cpu",
            "evaluation_mode": True,
            "gradients_enabled": False,
            "loading_error_count": 0,
            "mismatched_key_count": 0,
            "missing_key_count": 0,
            "model_class": "DINOv3ViTModel",
            "output_dtype": "float32",
            "parameter_count": 85660416,
            "parameter_dtype": "float32",
            "parameter_tensor_count": 211,
            "parameters_frozen": True,
            "state_sha256": MODEL_STATE_SHA256,
            "unexpected_key_count": 0,
        },
        "patch_order": {
            "atol": 1e-6,
            "conv2d_hook_shape": [2, 768, 14, 14],
            "embedding_patch_slice_shape": [2, 196, 768],
            "matches": True,
            "max_abs_diff": 0.0,
            "reconstructed_shape": [2, 196, 768],
            "row_major": True,
            "rtol": 1e-6,
            "source_relation": "conv2d_output_flatten_spatial_then_transpose",
        },
        "pooler": {
            "canonical_global_source": "patch_axis_mean",
            "cls_matches": True,
            "cls_max_abs_diff": 0.0,
            "excluded_from_canonical_global": True,
            "max_abs_difference_from_patch_mean": 2.0,
            "mean_abs_difference_from_patch_mean": 0.5,
            "patch_mean_shape": [2, 768],
            "pooler_shape": [2, 768],
            "separate_head_used": False,
            "source_relation": "last_hidden_state_cls",
        },
        "prefix_tokens": {
            "cls_indices": [0],
            "overlap_count": 0,
            "partition_complete": True,
            "patch_indices": patch_indices,
            "register_indices": [1, 2, 3, 4],
            "sequence_order": "cls_register_patch",
            "special_indices": special_indices,
        },
        "repeat": {
            "embedding_exact": True,
            "global_hashes_exact": True,
            "hidden_states_exact": True,
            "last_hidden_exact": True,
            "patch_hashes_exact": True,
            "pooler_exact": True,
        },
        "run_id": run_id,
        "schema_id": SEMANTIC_RESPONSE_SCHEMA_ID,
        "semantic_marker": SEMANTIC_SUCCESS_MARKER,
        "socket_attempts": 0,
        "status": "success",
        "taps": taps,
        "token_layout": {
            "grid_height": 14,
            "grid_width": 14,
            "partition_complete": True,
            "patch_token_indices": patch_indices,
            "special_token_indices": special_indices,
            "special_token_policy": (
                "dinov3_cls1_register4_prefix_verified_runtime_v1"
            ),
            "total_token_count": 201,
        },
        "workspace_paths": workspace_paths,
    }


def _profile() -> Any:
    return load_semantic_profile(_profile_path().resolve())


def test_valid_semantic_profile_loads_with_exact_public_facts() -> None:
    profile = _profile()
    assert profile.source_sha256 == profile.sha256
    assert profile.model.key == "dinov3_vit_b16"
    assert profile.model.revision == "5931719e67bbdb9737e363e781fb0c67687896bc"
    assert profile.tap_catalog == TAP_CATALOG
    assert profile.selected_artifact_taps == SELECTED_ARTIFACT_TAPS
    assert profile.patch_token_indices == tuple(range(5, 201))
    assert profile.special_token_indices == (0, 1, 2, 3, 4)


def test_profile_is_canonical_sorted_key_json() -> None:
    value = _profile_value()
    expected = (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    assert _profile_path().read_bytes() == expected


def test_profile_missing_field_is_rejected(tmp_path: Path) -> None:
    path = _modified_profile(tmp_path, lambda value: value.pop("model"))
    with pytest.raises(DINOv3SemanticBoundaryError, match="missing fields"):
        load_semantic_profile(path)


def test_profile_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = _modified_profile(tmp_path, lambda value: value.update({"future": False}))
    with pytest.raises(DINOv3SemanticBoundaryError, match="unknown fields"):
        load_semantic_profile(path)


def test_profile_duplicate_key_is_rejected(tmp_path: Path) -> None:
    payload = _profile_path().read_text(encoding="utf-8").replace(
        "{\n", '{\n  "schema_id": "duplicate",\n', 1
    )
    path = (tmp_path / "duplicate.json").resolve()
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(DINOv3SemanticBoundaryError, match="duplicate"):
        load_semantic_profile(path)


def test_profile_nonfinite_json_is_rejected(tmp_path: Path) -> None:
    payload = _profile_path().read_text(encoding="utf-8").replace("1e-06", "NaN", 1)
    path = (tmp_path / "nan.json").resolve()
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(DINOv3SemanticBoundaryError, match="non-finite"):
        load_semantic_profile(path)


@pytest.mark.parametrize(
    "field",
    ["registry_sha256", "runtime_profile_sha256", "snapshot_profile_sha256"],
)
def test_profile_identity_hash_mismatch_is_rejected(
    tmp_path: Path, field: str
) -> None:
    path = _modified_profile(
        tmp_path, lambda value: value["identities"].update({field: "0" * 64})
    )
    with pytest.raises(DINOv3SemanticBoundaryError, match=field):
        load_semantic_profile(path)


@pytest.mark.parametrize("field", ["model_id", "revision"])
def test_profile_wrong_model_or_revision_is_rejected(
    tmp_path: Path, field: str
) -> None:
    path = _modified_profile(
        tmp_path, lambda value: value["model"].update({field: "wrong"})
    )
    with pytest.raises(DINOv3SemanticBoundaryError, match=field):
        load_semantic_profile(path)


def test_profile_invalid_checkpoint_hash_is_rejected(tmp_path: Path) -> None:
    path = _modified_profile(
        tmp_path,
        lambda value: value["checkpoint_files"][0].update({"sha256": "0" * 64}),
    )
    with pytest.raises(DINOv3SemanticBoundaryError, match="checkpoint_files"):
        load_semantic_profile(path)


@pytest.mark.parametrize(
    ("path_parts", "match"),
    [
        (("canonical_input", "sha256"), "canonical_input.sha256"),
        (("canonical_input", "normalized", "sha256"), "normalized.sha256"),
    ],
)
def test_profile_input_hash_mismatch_is_rejected(
    tmp_path: Path, path_parts: tuple[str, ...], match: str
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        target = value
        for part in path_parts[:-1]:
            target = target[part]
        target[path_parts[-1]] = "0" * 64

    with pytest.raises(DINOv3SemanticBoundaryError, match=match):
        load_semantic_profile(_modified_profile(tmp_path, mutate))


@pytest.mark.parametrize("field", ["catalog", "artifact_selected"])
def test_profile_invalid_tap_catalog_is_rejected(tmp_path: Path, field: str) -> None:
    path = _modified_profile(
        tmp_path, lambda value: value["taps"][field].reverse()
    )
    with pytest.raises(DINOv3SemanticBoundaryError, match=field):
        load_semantic_profile(path)


def test_profile_public_objects_are_recursively_frozen() -> None:
    profile = _profile()
    with pytest.raises(FrozenInstanceError):
        profile.sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        profile.model["key"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.taps.catalog[0] = "changed"  # type: ignore[index]


def test_boundary_import_isolation_in_fresh_interpreter() -> None:
    script = (
        "import sys; import vision_encoder_alignment.dinov3_semantic_boundary; "
        "forbidden={'numpy','torch','transformers','huggingface_hub','PIL','safetensors'}; "
        "print(sorted(forbidden.intersection(sys.modules)))"
    )
    environment = {**os.environ, "PYTHONPATH": str(_repository_root() / "src")}
    completed = subprocess.run(
        [sys.executable, "-S", "-B", "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_semantic_request_round_trip_is_canonical(tmp_path: Path) -> None:
    profile = _profile()
    path = (tmp_path / "request.json").resolve()
    written = write_semantic_request(
        path, run_id="t007a4-request", semantic_profile_sha256=profile.sha256
    )
    loaded = read_semantic_request(path)
    assert loaded == written
    assert loaded.to_dict()["schema_id"] == SEMANTIC_REQUEST_SCHEMA_ID
    assert loaded.workspace_relative_artifact_directory == "actual-output"


@pytest.mark.parametrize(
    "relative", ["../escape", "nested/../escape", "C:/escape", "/escape", "a\\b"]
)
def test_semantic_request_rejects_unsafe_relative_path(
    tmp_path: Path, relative: str
) -> None:
    with pytest.raises(DINOv3SemanticBoundaryError):
        write_semantic_request(
            (tmp_path / "request.json").resolve(),
            run_id="unsafe-request",
            semantic_profile_sha256="a" * 64,
            workspace_relative_artifact_directory=relative,
        )


def test_semantic_request_refuses_overwrite(tmp_path: Path) -> None:
    path = (tmp_path / "request.json").resolve()
    path.write_text("reserved", encoding="utf-8")
    with pytest.raises(DINOv3SemanticBoundaryError, match="overwrite"):
        write_semantic_request(
            path, run_id="no-overwrite", semantic_profile_sha256="a" * 64
        )


def test_valid_semantic_response_returns_frozen_object() -> None:
    profile = _profile()
    response = validate_semantic_response(_valid_response(profile), profile)
    assert response.status == "success"
    assert len(response.taps) == 14
    with pytest.raises((AttributeError, TypeError)):
        response.document.model["device"] = "cuda"  # type: ignore[index]


@pytest.mark.parametrize("kind", ["missing", "unknown"])
def test_response_rejects_missing_and_unknown_fields(kind: str) -> None:
    profile = _profile()
    value = _valid_response(profile)
    if kind == "missing":
        value.pop("model")
    else:
        value["future"] = False
    with pytest.raises(DINOv3SemanticBoundaryError, match=f"{kind} fields"):
        validate_semantic_response(value, profile)


def test_response_rejects_patch_special_overlap() -> None:
    profile = _profile()
    value = _valid_response(profile)
    value["prefix_tokens"]["patch_indices"][0] = 4
    with pytest.raises(DINOv3SemanticBoundaryError, match="patch_indices"):
        validate_semantic_response(value, profile)


def test_response_rejects_incomplete_token_coverage() -> None:
    profile = _profile()
    value = _valid_response(profile)
    value["prefix_tokens"]["patch_indices"].pop()
    with pytest.raises(DINOv3SemanticBoundaryError, match="patch_indices"):
        validate_semantic_response(value, profile)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("patch_shape", [2, 195, 768]),
        ("dtype", "float16"),
        ("patch_sha256", "invalid"),
    ],
)
def test_response_rejects_invalid_per_tap_evidence(
    field: str, replacement: object
) -> None:
    profile = _profile()
    value = _valid_response(profile)
    value["taps"][0][field] = replacement
    with pytest.raises(DINOv3SemanticBoundaryError, match=field):
        validate_semantic_response(value, profile)


def test_response_rejects_missing_semantic_marker() -> None:
    profile = _profile()
    value = _valid_response(profile)
    value.pop("semantic_marker")
    with pytest.raises(DINOv3SemanticBoundaryError, match="semantic_marker"):
        validate_semantic_response(value, profile)


def test_response_rejects_incomplete_marker_catalog() -> None:
    profile = _profile()
    value = _valid_response(profile)
    value["markers"].pop()
    with pytest.raises(DINOv3SemanticBoundaryError, match="markers"):
        validate_semantic_response(value, profile)


@pytest.mark.parametrize(
    ("field", "replacement", "classification"),
    [
        ("raw_tensor", [[[1.0]]], "sensitive_data_rejected"),
        ("api_key", "not-a-real-key", "sensitive_data_rejected"),
        ("future_path", "C:/private/output.json", "sensitive_data_rejected"),
    ],
)
def test_response_rejects_raw_secret_and_absolute_path_fields(
    field: str, replacement: object, classification: str
) -> None:
    profile = _profile()
    value = _valid_response(profile)
    value[field] = replacement
    with pytest.raises(DINOv3SemanticBoundaryError) as error:
        validate_semantic_response(value, profile)
    assert error.value.classification == classification


def test_semantic_projection_excludes_only_run_id() -> None:
    profile = _profile()
    first = validate_semantic_response(
        _valid_response(profile, run_id="run-one"), profile
    )
    second = validate_semantic_response(
        _valid_response(profile, run_id="run-two"), profile
    )
    assert semantic_response_projection(first) == second.semantic_projection()
    assert "run_id" not in first.semantic_projection()
    assert "workspace_paths" in first.semantic_projection()


def test_workspace_resolver_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(DINOv3SemanticBoundaryError, match="traversal"):
        resolve_workspace_relative_path(tmp_path.resolve(), "../outside")


def test_workspace_resolver_rejects_symlink_component(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(DINOv3SemanticBoundaryError, match="reparse"):
        resolve_workspace_relative_path(tmp_path.resolve(), "link/file.json", must_exist=False)


def test_response_rejects_nonfinite_programmatic_value() -> None:
    profile = _profile()
    value = _valid_response(profile)
    value["patch_order"]["max_abs_diff"] = float("nan")
    with pytest.raises(DINOv3SemanticBoundaryError, match="non-finite"):
        validate_semantic_response(value, profile)


def test_response_rejects_block_12_as_unormalized_hidden_state() -> None:
    profile = _profile()
    value = _valid_response(profile)
    value["hidden_states"]["blocks"][-1]["hidden_state_matches_hook"] = True
    value["hidden_states"]["blocks"][-1]["max_abs_diff"] = 0.0
    with pytest.raises(DINOv3SemanticBoundaryError, match="block"):
        validate_semantic_response(value, profile)


def test_response_rejects_native_pooler_as_canonical_global() -> None:
    profile = _profile()
    value = _valid_response(profile)
    value["pooler"]["canonical_global_source"] = "pooler_output"
    with pytest.raises(DINOv3SemanticBoundaryError, match="canonical_global_source"):
        validate_semantic_response(value, profile)
