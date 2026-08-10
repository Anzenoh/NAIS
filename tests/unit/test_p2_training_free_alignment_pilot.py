from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import training_free_alignment_pilot as pilot


def _row(index: int, split: str, name: str | None = None) -> pilot.ManifestRow:
    filename = name or f"{index:012d}.jpg"
    return pilot.ManifestRow(f"id-{index}", filename, "a" * 64, split, index)


def _records() -> list[dict[str, object]]:
    rows = []
    for image in range(4):
        for condition, delta in zip(pilot.CONDITIONS, (0.0, 2.0, 1.0, 3.0), strict=True):
            rows.append({
                "image_id": f"test-{image}", "condition": condition,
                "patch_cosine": delta, "patch_norm_ratio": delta,
                "patch_nmse": delta, "connector_cosine": delta,
                "connector_norm_ratio": delta, "connector_nmse": delta,
                "generation_success": delta, "token_ids_equal_to_native": delta,
                "decoded_text_equal_to_native": delta, "generated_token_count": delta,
            })
    return rows


def test_selects_first_eight_train_and_first_four_test_deterministically() -> None:
    rows = [_row(i, "train") for i in reversed(range(10))] + [_row(100 + i, "test") for i in reversed(range(6))]
    selected = pilot.select_pilot_rows(rows)
    assert [r.selection_index for r in selected.calibration] == list(range(8))
    assert [r.selection_index for r in selected.test] == [100, 101, 102, 103]


def test_manifest_fixed_selection_matches_contract() -> None:
    selected = pilot.select_pilot_rows(Path("data/manifests/p1_1000.jsonl"))
    assert [r.relative_path for r in selected.calibration] == [
        "000000206411.jpg", "000000312263.jpg", "000000049060.jpg", "000000161008.jpg",
        "000000572408.jpg", "000000230450.jpg", "000000571804.jpg", "000000328030.jpg",
    ]
    assert [r.relative_path for r in selected.test] == [
        "000000172946.jpg", "000000127270.jpg", "000000001818.jpg", "000000365886.jpg",
    ]


def test_rejects_overlapping_calibration_and_test_images() -> None:
    rows = [_row(i, "train") for i in range(8)] + [_row(20 + i, "test") for i in range(4)]
    rows[-1] = pilot.ManifestRow(rows[0].image_id, rows[-1].relative_path, "a" * 64, "test", 23)
    with pytest.raises(pilot.AlignmentPilotError, match="disjoint"):
        pilot.select_pilot_rows(rows)


def test_rejects_patch_level_leakage_by_requiring_image_axis() -> None:
    flattened = np.zeros((8 * 196, 768), dtype=np.float32)
    with pytest.raises(pilot.AlignmentPilotError, match="image partition"):
        pilot.fit_moment_calibration(flattened, flattened)


def test_validates_source_sha256(tmp_path: Path) -> None:
    content = b"bounded-image"
    digest = hashlib.sha256(content).hexdigest()
    rows = [pilot.ManifestRow(f"train-{i}", f"t{i}.jpg", digest, "train", i) for i in range(8)]
    rows += [pilot.ManifestRow(f"test-{i}", f"e{i}.jpg", digest, "test", 100 + i) for i in range(4)]
    for row in rows:
        (tmp_path / row.relative_path).write_bytes(content)
    selected = pilot.select_pilot_rows(rows)
    assert len(pilot.validate_selected_sources(selected, tmp_path)) == 12
    (tmp_path / rows[-1].relative_path).write_bytes(b"changed")
    with pytest.raises(pilot.AlignmentPilotError, match="SHA-256 mismatch"):
        pilot.validate_selected_sources(selected, tmp_path)


def test_rejects_development_image() -> None:
    rows = [_row(i, "train", pilot.DEVELOPMENT_IMAGE if i == 0 else None) for i in range(8)]
    rows += [_row(100 + i, "test") for i in range(4)]
    with pytest.raises(pilot.AlignmentPilotError, match="development"):
        pilot.select_pilot_rows(rows)


def test_exact_dimensionwise_moment_calibration() -> None:
    source = np.arange(8 * 196 * 768, dtype=np.float32).reshape(8, 196, 768) % 17
    target = source * 2.5 - 3.0
    fit = pilot.fit_moment_calibration(source, target)
    actual = pilot.apply_moment_calibration(source, fit)
    np.testing.assert_allclose(actual, target, rtol=2e-6, atol=2e-5)
    assert fit.scale.dtype == np.float32 and fit.bias.dtype == np.float32


def test_moment_uses_ddof_zero_and_denominator_floor() -> None:
    source = np.ones((8, 196, 768), dtype=np.float32)
    target = np.broadcast_to(np.arange(8, dtype=np.float32)[:, None, None], source.shape).copy()
    fit = pilot.fit_moment_calibration(source, target)
    expected = target.astype(np.float64).reshape(-1, 768).std(axis=0, ddof=0) / 1e-6
    np.testing.assert_allclose(fit.scale, expected.astype(np.float32))
    np.testing.assert_allclose(fit.dinov3_train_std, 0.0)


def test_exact_orthogonal_procrustes_recovery() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(8, 196, 768)).astype(np.float32)
    rotation = np.eye(768, dtype=np.float32)
    rotation[[0, 1]] = rotation[[1, 0]]
    target = source @ rotation + 0.25
    fit = pilot.fit_orthogonal_procrustes(source, target)
    actual = pilot.apply_orthogonal_procrustes(source, fit)
    np.testing.assert_allclose(actual, target, atol=2e-4, rtol=2e-4)
    assert fit.rotation.dtype == np.float32


def test_fit_functions_accept_only_eight_calibration_images() -> None:
    held_out_shaped = np.zeros((4, 196, 768), dtype=np.float32)
    with pytest.raises(pilot.AlignmentPilotError):
        pilot.fit_moment_calibration(held_out_shaped, held_out_shaped)
    with pytest.raises(pilot.AlignmentPilotError):
        pilot.fit_orthogonal_procrustes(held_out_shaped, held_out_shaped)


def test_exact_condition_order() -> None:
    assert pilot.CONDITIONS == ("NATIVE", "RAW", "CALIBRATED", "LINEAR")


def test_exact_nmse_uses_calibration_target_mean() -> None:
    native_value = np.array([[2.0, 4.0]])
    condition = np.array([[1.0, 6.0]])
    train_mean = np.array([1.0, 1.0])
    metrics = pilot.flattened_metrics(condition, native_value, train_mean)
    assert metrics["nmse"] == pytest.approx(5.0 / 10.0)


def test_higher_and_lower_improvement_count_logic() -> None:
    summary = pilot.aggregate_metrics(_records())
    calibrated_cosine = next(r for r in summary if r["condition"] == "CALIBRATED" and r["metric"] == "patch_cosine")
    calibrated_nmse = next(r for r in summary if r["condition"] == "CALIBRATED" and r["metric"] == "patch_nmse")
    assert calibrated_cosine["count_improved_over_raw"] == 0
    assert calibrated_nmse["count_improved_over_raw"] == 4


def test_result_schema_and_required_artifact_inventory() -> None:
    result = pilot.build_result(artifacts={name: {} for name in pilot.ARTIFACT_NAMES})
    assert result["schema_id"] == pilot.SCHEMA_ID
    assert result["status"] == pilot.STATUS
    assert tuple(result["artifacts"]) == pilot.ARTIFACT_NAMES


def test_numpy_boundary_is_writable_contiguous_copy() -> None:
    original = np.arange(24, dtype=np.float64).reshape(4, 6).T
    copied = pilot.writable_contiguous_float32(original)
    assert copied.dtype == np.float32 and copied.flags.c_contiguous and copied.flags.writeable
    assert not np.shares_memory(original, copied)


def test_fake_orchestration_one_dino_batch_and_one_model_load() -> None:
    calls = {"dino": 0, "load": 0}
    def extract(ids: tuple[str, ...]) -> None:
        calls["dino"] += 1
        assert len(ids) == 12
    def load() -> object:
        calls["load"] += 1
        return object()
    counts = pilot.fake_orchestration(
        extract_dino_batch=extract, load_tinyllava=load,
        capture_calibration=lambda _model, _image: None,
        generate_test=lambda _model, _image, _condition: None,
        calibration_ids=[f"c{i}" for i in range(8)], test_ids=[f"t{i}" for i in range(4)],
    )
    assert calls == {"dino": 1, "load": 1}
    assert counts["dinov3_batch_extractions"] == counts["tinyllava_model_loads"] == 1


def test_fake_orchestration_exactly_sixteen_held_out_generations() -> None:
    seen: list[tuple[str, str]] = []
    counts = pilot.fake_orchestration(
        extract_dino_batch=lambda _ids: None, load_tinyllava=lambda: object(),
        capture_calibration=lambda _model, _image: None,
        generate_test=lambda _model, image, condition: seen.append((image, condition)),
        calibration_ids=[f"c{i}" for i in range(8)], test_ids=[f"t{i}" for i in range(4)],
    )
    assert counts["test_generation_calls"] == len(seen) == 16
    assert [condition for _image, condition in seen[:4]] == list(pilot.CONDITIONS)


def test_differing_answers_still_produce_software_pass() -> None:
    result = pilot.build_result(per_image=[{"decoded_text_equal_to_native": False}])
    assert result["status"] == pilot.STATUS


def test_module_import_does_not_load_heavy_runtime_packages() -> None:
    code = (
        "import sys; import vision_encoder_alignment.vlm_compat.training_free_alignment_pilot; "
        "forbidden={'torch','transformers','PIL','safetensors'}; "
        "assert not forbidden.intersection(sys.modules)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False, env=env
    )
    assert completed.returncode == 0, completed.stderr


def test_cli_help_is_import_safe() -> None:
    env = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "scripts/run_training_free_alignment_pilot.py", "--help"],
        text=True, capture_output=True, check=False, env=env,
    )
    assert completed.returncode == 0
    assert "--dinov3-snapshot-root" in completed.stdout
    assert "torch" not in completed.stderr.lower()
