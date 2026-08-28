from __future__ import annotations

import copy
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear


CONFIG_PATH = Path("configs/vlm_compat/p2_query_adapter_linear_v1.json")


def _config() -> dict[str, object]:
    return linear.load_query_adapter_config(CONFIG_PATH)


def _records() -> list[dict[str, object]]:
    return [
        {
            "split": "validation",
            "lambda_normalized": value,
            "same_image_recall_at_1": 0.2,
            "same_image_recall_at_5": 0.2,
            "connector_cosine": 0.2,
        }
        for value in linear.RIDGE_LAMBDA_NORMALIZED_GRID
    ]


def test_frozen_config_acceptance_and_generation_plan() -> None:
    config = _config()
    assert linear.generation_plan(config) == {"test_images": 100, "native": 100, "adapted": 800, "total": 900}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source_order",), ["DINOV3", "SIGLIP2"]),
        (("methods", "RIDGE", "lambda_normalized_grid"), [1e-6]),
        (("generation", "temperature"), 0.5),
    ],
)
def test_scientifically_load_bearing_config_drift_is_rejected(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(_config())
    target: dict[str, object] = config
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = value
    with pytest.raises(linear.QueryAdapterLinearError, match="frozen config drift"):
        linear.validate_query_adapter_config(config)


def test_feature_validation_rejects_nonfinite_shape_and_image_order_mismatch() -> None:
    values = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(linear.QueryAdapterLinearError, match="shape"):
        linear.validate_feature_bank(values, ["a", "b"], expected_image_count=2, expected_patch_count=3, expected_feature_dim=5)
    with pytest.raises(linear.QueryAdapterLinearError, match="NaN"):
        linear.validate_feature_bank(np.full((2, 3, 4), np.nan), ["a", "b"], expected_image_count=2, expected_patch_count=3, expected_feature_dim=4)
    with pytest.raises(linear.QueryAdapterLinearError, match="order"):
        linear.validate_feature_pair(values, values, ["a", "b"], ["b", "a"])


def test_partition_overlap_is_rejected() -> None:
    with pytest.raises(linear.QueryAdapterLinearError, match="overlap"):
        linear.validate_partition_image_ids(["train"], ["validation"], ["train"])


def test_exact_calibration_recovery_ddof_zero_and_std_floor() -> None:
    source = np.array([[1.0, 2.0], [1.0, 6.0], [1.0, 10.0]], dtype=np.float32)
    target = np.array([[2.0, -1.0], [4.0, 1.0], [6.0, 3.0]], dtype=np.float32)
    fit = linear.fit_calibration(source, target)
    assert fit.source_std.dtype == fit.scale.dtype == np.float32
    np.testing.assert_allclose(fit.source_std, source.astype(np.float64).std(axis=0, ddof=0).astype(np.float32))
    np.testing.assert_allclose(fit.scale[0], target.astype(np.float64).std(axis=0, ddof=0)[0] / 1e-6)
    assert linear.apply_calibration(source, fit).dtype == np.float32
    np.testing.assert_allclose(linear.apply_calibration(source, fit)[:, 1], target[:, 1], rtol=2e-6)


def test_orthogonal_procrustes_recovery_and_float32_contiguous_output() -> None:
    rng = np.random.default_rng(12)
    source = rng.normal(size=(30, 3))
    rotation = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    target = source @ rotation + np.array([0.2, -1.5, 4.0])
    fit = linear.fit_orthogonal_procrustes(source, target)
    result = linear.apply_orthogonal_procrustes(source, fit)
    np.testing.assert_allclose(result, target, atol=2e-5, rtol=2e-5)
    assert fit.rotation.dtype == result.dtype == np.float32 and result.flags.c_contiguous


def test_full_linear_known_affine_recovery_and_never_uses_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(40, 3))
    weights = np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0], [2.0, 0.0, 1.0]])
    target = source @ weights + np.array([2.0, -3.0, 0.25])
    monkeypatch.setattr(linear, "apply_calibration", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")))
    fit = linear.fit_full_linear(source, target)
    np.testing.assert_allclose(linear.apply_full_linear(source, fit), target, atol=2e-5, rtol=2e-5)
    assert fit.rank == 3 and fit.weights.dtype == np.float32


def test_ridge_alpha_uses_actual_row_count_and_known_reference_solve() -> None:
    source = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    target = source @ np.array([[2.0, 1.0], [0.5, -1.0]]) + np.array([1.0, -2.0])
    fit = linear.fit_ridge(source, target, 1e-1)
    assert fit.train_row_count == 4 and fit.alpha == pytest.approx(0.4)
    xc = source.astype(np.float64) - source.mean(axis=0, dtype=np.float64)
    yc = target.astype(np.float64) - target.mean(axis=0, dtype=np.float64)
    expected = np.linalg.solve(xc.T @ xc + 0.4 * np.eye(2), xc.T @ yc)
    np.testing.assert_allclose(fit.weights, expected.astype(np.float32))
    result = linear.apply_ridge(source, fit)
    assert result.dtype == np.float32 and result.flags.c_contiguous


def test_ridge_rejects_invalid_lambda() -> None:
    with pytest.raises(linear.QueryAdapterLinearError, match="frozen grid"):
        linear.fit_ridge(np.ones((3, 2)), np.ones((3, 2)), 0.3)


def test_patch_metrics_reference_and_zero_denominator_rejection() -> None:
    candidate = np.array([[[1.0, 6.0]], [[2.0, 3.0]]])
    reference = np.array([[[2.0, 4.0]], [[2.0, 3.0]]])
    metrics = linear.patch_metrics(candidate, reference, np.array([1.0, 1.0]))
    assert metrics["per_image_nmse"][0] == pytest.approx(5.0 / 10.0)
    assert metrics["per_image_r2"][0] == pytest.approx(0.5)
    with pytest.raises(linear.QueryAdapterLinearError, match="denominator"):
        linear.patch_metrics(reference, reference, np.array([2.0, 4.0]))


def test_same_image_retrieval_and_stable_ties() -> None:
    values = np.eye(3, dtype=np.float64).reshape(3, 1, 3)
    result = linear.same_image_retrieval(values, values, ["a", "b", "c"], ["a", "b", "c"])
    assert result["ranks"].tolist() == [1, 1, 1] and result["recall_at_5"] == 1.0
    tied = np.broadcast_to(np.array([[[1.0, 0.0]]]), (3, 1, 2)).copy()
    stable = linear.same_image_retrieval(tied, tied, ["a", "b", "c"], ["a", "b", "c"])
    assert stable["ranking"][2].tolist() == [0, 1, 2]
    assert stable["ranks"].tolist() == [1, 2, 3]


def test_ridge_selection_requires_all_nine_and_rejects_test_input() -> None:
    records = _records()[:-1]
    with pytest.raises(linear.QueryAdapterLinearError, match="all frozen"):
        linear.select_ridge_candidate(records)
    records = _records()
    records[0]["split"] = "test"
    with pytest.raises(linear.QueryAdapterLinearError, match="Test"):
        linear.select_ridge_candidate(records)


def test_ridge_selection_all_tie_break_levels() -> None:
    records = _records()
    first, second = records[0], records[1]
    first.update({"same_image_recall_at_1": 0.8, "same_image_recall_at_5": 0.1, "connector_cosine": 0.1})
    second.update({"same_image_recall_at_1": 0.7, "same_image_recall_at_5": 1.0, "connector_cosine": 1.0})
    assert linear.select_ridge_candidate(records)["selected_lambda_normalized"] == 1e-6
    second["same_image_recall_at_1"] = 0.8
    assert linear.select_ridge_candidate(records)["selected_lambda_normalized"] == 1e-5
    first["same_image_recall_at_5"] = 1.0
    assert linear.select_ridge_candidate(records)["selected_lambda_normalized"] == 1e-5
    first["connector_cosine"] = 1.0
    assert linear.select_ridge_candidate(records)["selected_lambda_normalized"] == 1e-6


def test_core_import_has_no_heavy_runtime_modules() -> None:
    code = "import sys; import vision_encoder_alignment.vlm_compat.query_adapter_linear; assert not {'torch','transformers','huggingface_hub'}.intersection(sys.modules)"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False, env=env)
    assert completed.returncode == 0, completed.stderr


def test_cli_module_is_importable_without_runtime_dependencies() -> None:
    module = importlib.import_module("scripts.validate_query_adapter_linear")
    assert callable(module.synthetic_smoke)
