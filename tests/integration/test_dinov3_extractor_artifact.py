from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import random
from typing import Any

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
    load_dinov3_extractor_profile,
    resolve_workspace_relative_path,
    validate_dinov3_extraction_response,
)
from vision_encoder_alignment.extractors import CanonicalImageBatch
from vision_encoder_alignment.extractors.base import ExtractorValidationError
from vision_encoder_alignment.feature_artifacts import (
    ARTIFACT_SCHEMA_ID,
    FeatureArtifactMetadata,
    canonical_tensor_keys,
    write_feature_artifact,
)
import vision_encoder_alignment.extractors.dinov3 as dinov3


ROOT = Path(__file__).resolve().parents[2]
LIVE_PROFILE = (
    ROOT / "configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json"
)


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def _context(profile_path: Path) -> dinov3._ExtractorContext:
    return dinov3._ExtractorContext(
        profile=load_dinov3_extractor_profile(profile_path),
        runtime_root=ROOT,
        snapshot_root=ROOT,
        repository_root=ROOT,
        worker_script=profile_path,
        descriptor=dinov3._build_descriptor(),
    )


def _batch(image_ids: tuple[str, str] = ("image-b", "image-a")) -> CanonicalImageBatch:
    pixels = np.arange(2 * 224 * 224 * 3, dtype=np.uint32).reshape(2, 224, 224, 3)
    return CanonicalImageBatch(image_ids, (pixels % 256).astype(np.uint8))


def _fake_invoker(
    captured: dict[str, Any],
    profile_path: Path,
    *,
    corrupt: str | None = None,
):
    def invoke(**kwargs: Any):
        request: DINOv3ExtractionRequest = kwargs["request"]
        workspace = Path(kwargs["workspace_root"])
        captured.setdefault("workspaces", []).append(workspace)
        captured.setdefault("requests", []).append(request)
        input_path = workspace / request.workspace_relative_input_path
        loaded_input = np.load(input_path, allow_pickle=False)
        captured["input"] = loaded_input.copy()
        output = workspace / request.workspace_relative_output_directory
        artifacts: list[dict[str, Any]] = []
        for index, tap in enumerate(request.requested_tap_names):
            tap_index = dinov3._build_descriptor().taps.index(
                next(item for item in dinov3._build_descriptor().taps if item.name == tap)
            )
            base = np.arange(len(request.image_ids) * 196 * 768, dtype=np.float32).reshape(
                len(request.image_ids), 196, 768
            )
            patch = np.ascontiguousarray(base * np.float32(1e-5) + np.float32(tap_index))
            global_mean = np.ascontiguousarray(patch.mean(axis=1, dtype=np.float32))
            directory_name = f"{index:02d}-{tap}"
            directory = output / directory_name
            tensor_name = f"shard-{index:05d}.safetensors"
            sidecar_name = f"shard-{index:05d}.json"
            patch_key, global_key = canonical_tensor_keys(tap)
            metadata = FeatureArtifactMetadata(
                schema_id=ARTIFACT_SCHEMA_ID,
                run_id=request.run_id,
                model_key=MODEL_KEY,
                tap_namespace="backbone",
                tap_name=tap,
                patch_tensor_key=patch_key,
                global_mean_tensor_key=global_key,
                image_ids=list(request.image_ids),
                grid_height=14,
                grid_width=14,
                patch_count=196,
                hidden_dimension=768,
                dtype="float32",
                shard_index=index,
                safetensors_file_name=tensor_name,
                sidecar_file_name=sidecar_name,
                checksum_algorithm="sha256",
                checksum_value="",
                created_at_utc="2026-07-21T00:00:00Z",
                run_status="incomplete",
            )
            completed = write_feature_artifact(directory, metadata, patch, global_mean)
            prefix = f"output/{directory_name}"
            artifacts.append(
                {
                    "checksum_sha256": completed.checksum_value,
                    "dtype": "float32",
                    "global_sha256": _array_hash(global_mean),
                    "global_shape": [len(request.image_ids), 768],
                    "patch_sha256": _array_hash(patch),
                    "patch_shape": [len(request.image_ids), 196, 768],
                    "safetensors_relative_path": f"{prefix}/{tensor_name}",
                    "sidecar_relative_path": f"{prefix}/{sidecar_name}",
                    "tap_name": tap,
                }
            )
        profile = load_dinov3_extractor_profile(profile_path)
        response: dict[str, Any] = {
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
                "normalized_sha256": "a" * 64,
                "normalized_shape": [len(request.image_ids), 3, 224, 224],
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
        validated = validate_dinov3_extraction_response(
            response, request=request, profile=profile
        )
        if corrupt == "tensor":
            tensor = workspace / artifacts[0]["safetensors_relative_path"]
            tensor.write_bytes(tensor.read_bytes() + b"corrupt")
        elif corrupt == "sidecar":
            sidecar = workspace / artifacts[0]["sidecar_relative_path"]
            sidecar.write_text("{}", encoding="utf-8")
        return validated

    return invoke


@pytest.fixture
def extractor_factory(
    monkeypatch: pytest.MonkeyPatch,
    dinov3_candidate_profile_path: Path,
):
    def create(*, corrupt: str | None = None, failure: str | None = None):
        captured: dict[str, Any] = {}
        load_dinov3_extractor_profile(dinov3_candidate_profile_path)
        monkeypatch.setattr(
            dinov3,
            "_load_extractor_context",
            lambda config: _context(dinov3_candidate_profile_path),
        )
        if failure is None:
            monkeypatch.setattr(
                dinov3,
                "invoke_dinov3_extraction_worker",
                _fake_invoker(
                    captured,
                    dinov3_candidate_profile_path,
                    corrupt=corrupt,
                ),
            )
        else:
            def fail(**kwargs: Any):
                captured.setdefault("workspaces", []).append(Path(kwargs["workspace_root"]))
                raise DINOv3ExtractionBoundaryError(failure, "simulated failure")
            monkeypatch.setattr(dinov3, "invoke_dinov3_extraction_worker", fail)
        extractor = dinov3.DINOv3VisionEncoderExtractor(
            dinov3.DINOv3ExtractorConfig(LIVE_PROFILE)
        )
        return extractor, captured
    return create


def test_live_v0_context_fails_closed_before_worker(
    monkeypatch: pytest.MonkeyPatch,
    dinov3_registry_pin_profile_path: Path,
) -> None:
    calls: list[object] = []

    def forbidden_worker(**kwargs: Any) -> None:
        calls.append(kwargs)
        raise AssertionError("worker must not be invoked for the rolled-back v0 profile")

    monkeypatch.setattr(
        dinov3,
        "invoke_dinov3_extraction_worker",
        forbidden_worker,
    )
    with pytest.raises(DINOv3ExtractionBoundaryError) as caught:
        load_dinov3_extractor_profile(dinov3_registry_pin_profile_path)
    assert "production_sources" in str(caught.value)
    assert calls == []


def test_public_extractor_round_trips_request_input_and_artifacts(extractor_factory: Any) -> None:
    extractor, captured = extractor_factory()
    batch = _batch()
    result = extractor.extract(batch, tap_names=("block_12", "final_norm"))
    assert result.image_ids == batch.image_ids
    assert result.tap_names == ("block_12", "final_norm")
    np.testing.assert_array_equal(captured["input"], batch.rgb_uint8)
    assert captured["requests"][0].workspace_relative_input_path == "input.npy"


def test_all_taps_return_canonical_shapes(extractor_factory: Any) -> None:
    extractor, _ = extractor_factory()
    result = extractor.extract(_batch())
    assert result.tap_names == tuple(tap.name for tap in extractor.available_taps)
    assert all(feature.patch.shape == (2, 196, 768) for feature in result.tap_features)
    assert all(feature.global_mean.shape == (2, 768) for feature in result.tap_features)


def test_subset_request_order_is_preserved(extractor_factory: Any) -> None:
    extractor, _ = extractor_factory()
    order = ("final_norm", "embedding_output", "block_06")
    assert extractor.extract(_batch(), tap_names=order).tap_names == order


@pytest.mark.parametrize("tap_names", [(), ("block_01", "block_01"), ("unknown",)])
def test_duplicate_unknown_or_empty_taps_fail(extractor_factory: Any, tap_names: tuple[str, ...]) -> None:
    extractor, captured = extractor_factory()
    with pytest.raises(ExtractorValidationError):
        extractor.extract(_batch(), tap_names=tap_names)
    assert not captured.get("requests")


def test_wrong_geometry_fails_before_worker(extractor_factory: Any) -> None:
    extractor, captured = extractor_factory()
    batch = CanonicalImageBatch(("x",), np.zeros((1, 223, 224, 3), dtype=np.uint8))
    with pytest.raises(ExtractorValidationError):
        extractor.extract(batch)
    assert not captured.get("requests")


def test_input_and_current_rng_are_preserved(extractor_factory: Any) -> None:
    extractor, _ = extractor_factory()
    batch = _batch()
    pixels_before = batch.rgb_uint8.copy()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    extractor.extract(batch, tap_names=("block_01",))
    np.testing.assert_array_equal(batch.rgb_uint8, pixels_before)
    assert random.getstate() == python_before
    assert _numpy_rng_equal(np.random.get_state(), numpy_before)


def test_image_ids_do_not_numerically_change_features(extractor_factory: Any) -> None:
    extractor, _ = extractor_factory()
    first = extractor.extract(_batch(("one", "two")), tap_names=("block_06",))
    second = extractor.extract(_batch(("alpha", "beta")), tap_names=("block_06",))
    np.testing.assert_array_equal(first.tap_features[0].patch, second.tap_features[0].patch)


def test_repeat_determinism(extractor_factory: Any) -> None:
    extractor, _ = extractor_factory()
    first = extractor.extract(_batch(), tap_names=("block_12", "final_norm"))
    second = extractor.extract(_batch(), tap_names=("block_12", "final_norm"))
    for left, right in zip(first.tap_features, second.tap_features, strict=True):
        np.testing.assert_array_equal(left.patch, right.patch)
        np.testing.assert_array_equal(left.global_mean, right.global_mean)


def test_output_arrays_are_immutable(extractor_factory: Any) -> None:
    extractor, _ = extractor_factory()
    feature = extractor.extract(_batch(), tap_names=("block_01",)).tap_features[0]
    assert not feature.patch.flags.writeable
    assert not feature.global_mean.flags.writeable


@pytest.mark.parametrize("corrupt", ["tensor", "sidecar"])
def test_corrupt_artifact_fails_closed_and_cleans_workspace(extractor_factory: Any, corrupt: str) -> None:
    extractor, captured = extractor_factory(corrupt=corrupt)
    with pytest.raises(ExtractorValidationError):
        extractor.extract(_batch(), tap_names=("block_01",))
    assert captured["workspaces"]
    assert all(not path.exists() for path in captured["workspaces"])


@pytest.mark.parametrize("classification", ["worker_timeout", "worker_nonzero_exit", "worker_response_invalid"])
def test_worker_failures_are_chained_and_cleaned(extractor_factory: Any, classification: str) -> None:
    extractor, captured = extractor_factory(failure=classification)
    with pytest.raises(ExtractorValidationError) as error:
        extractor.extract(_batch(), tap_names=("block_01",))
    assert isinstance(error.value.__cause__, DINOv3ExtractionBoundaryError)
    assert classification in str(error.value)
    assert all(not path.exists() for path in captured["workspaces"])


def test_success_workspace_cleanup(extractor_factory: Any) -> None:
    extractor, captured = extractor_factory()
    extractor.extract(_batch(), tap_names=("block_01",))
    assert all(not path.exists() for path in captured["workspaces"])


def test_repository_artifacts_unchanged(extractor_factory: Any) -> None:
    artifacts = ROOT / "artifacts"
    before = sorted(path.relative_to(artifacts).as_posix() for path in artifacts.rglob("*")) if artifacts.exists() else []
    live_before = hashlib.sha256(LIVE_PROFILE.read_bytes()).hexdigest()
    extractor, _ = extractor_factory()
    extractor.extract(_batch(), tap_names=("block_01",))
    after = sorted(path.relative_to(artifacts).as_posix() for path in artifacts.rglob("*")) if artifacts.exists() else []
    assert after == before
    assert hashlib.sha256(LIVE_PROFILE.read_bytes()).hexdigest() == live_before


def test_worker_source_uses_argument_list_and_no_shell_true() -> None:
    source = (ROOT / "src/vision_encoder_alignment/dinov3_extraction_boundary.py").read_text(encoding="utf-8")
    assert "subprocess.run(" in source
    assert "shell=False" in source
    assert "shell=True" not in source


def test_worker_source_has_raw_final_hook_and_one_forward_call() -> None:
    source = (ROOT / "scripts/dinov3_extraction_worker.py").read_text(encoding="utf-8")
    assert "model.layer[11].register_forward_hook" in source
    assert 'sequences["block_12"] = raw_final' in source
    assert 'sequences["final_norm"] = last' in source
    assert "sequence[:, 5:201, :]" in source
    assert "patch.mean(axis=1, dtype=np.float32)" in source


def test_worker_source_is_offline_and_processor_free() -> None:
    source = (ROOT / "scripts/dinov3_extraction_worker.py").read_text(encoding="utf-8")
    assert "local_files_only=True" in source
    assert "trust_remote_code=False" in source
    assert "token=False" in source
    assert "use_safetensors=True" in source
    assert "AutoImageProcessor" not in source
    assert "ImageProcessor.from_pretrained" not in source


def test_worker_source_guard_order_local_snapshot_and_no_mutation_clients() -> None:
    source = (ROOT / "scripts/dinov3_extraction_worker.py").read_text(encoding="utf-8")
    assert source.index("guard.install()") < source.index("import numpy as np", source.index("guard.install()"))
    assert "snapshot_files = snapshot_files_directory(snapshot_root)" in source
    assert "model, _ = _load_model(snapshot_files)" in source
    assert source.count("outputs = model(") == 1
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "torchvision" not in imported_roots
    for forbidden in ("hf_hub_download", "snapshot_download", "urlopen(", "pip install", "git commit"):
        assert forbidden not in source


def test_noncanonical_batch_object_fails_before_worker(extractor_factory: Any) -> None:
    extractor, captured = extractor_factory()
    with pytest.raises(ExtractorValidationError):
        extractor.extract(np.zeros((2, 224, 224, 3), dtype=np.uint8))  # type: ignore[arg-type]
    assert not captured.get("requests")


def test_request_carries_raw_and_file_hashes(extractor_factory: Any) -> None:
    extractor, captured = extractor_factory()
    batch = _batch()
    extractor.extract(batch, tap_names=("block_01",))
    request = captured["requests"][0]
    assert request.raw_input_sha256 == _array_hash(batch.rgb_uint8)
    assert request.input_file_sha256 != request.raw_input_sha256


def test_workspace_path_escape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DINOv3ExtractionBoundaryError):
        resolve_workspace_relative_path(tmp_path.resolve(), "../escape.json", must_exist=False)


def test_workspace_absolute_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DINOv3ExtractionBoundaryError):
        resolve_workspace_relative_path(
            tmp_path.resolve(), str((tmp_path / "absolute.json").resolve()), must_exist=False
        )


def test_response_protocol_source_contains_no_raw_tensor_transport() -> None:
    source = (ROOT / "src/vision_encoder_alignment/dinov3_extraction_boundary.py").read_text(encoding="utf-8")
    assert '"raw_tensor"' in source  # explicit rejection list
    assert "tensor_values" in source
    assert "DINOv3ExtractionResponse" in source
