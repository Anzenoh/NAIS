"""Synthetic-only validation CLI for the Phase 2 direct linear adapter core."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear


def _orthogonal_matrix(dimension: int) -> np.ndarray:
    matrix = np.arange(1, dimension * dimension + 1, dtype=np.float64).reshape(dimension, dimension)
    q, _ = np.linalg.qr(matrix + np.eye(dimension))
    return q


def synthetic_smoke(config_path: str | Path) -> dict[str, object]:
    config = linear.load_query_adapter_config(config_path)
    rng = np.random.default_rng(20260829)
    source = rng.normal(size=(24, 4))

    calibrated_target = source * np.array([2.0, 0.5, 1.5, 3.0]) + np.array([1.0, -2.0, 0.25, 4.0])
    calibration = linear.fit_calibration(source, calibrated_target)
    if not np.allclose(linear.apply_calibration(source, calibration), calibrated_target, rtol=2e-6, atol=2e-6):
        raise RuntimeError("synthetic Calibration recovery failed")

    rotation = _orthogonal_matrix(4)
    procrustes_target = source @ rotation + np.array([0.5, -1.0, 2.0, 0.25])
    procrustes = linear.fit_orthogonal_procrustes(source, procrustes_target)
    if not np.allclose(linear.apply_orthogonal_procrustes(source, procrustes), procrustes_target, rtol=2e-5, atol=2e-5):
        raise RuntimeError("synthetic Procrustes recovery failed")

    weights = np.array([[1.5, -0.5, 0.0, 0.25], [0.0, 2.0, 0.5, -1.0], [1.0, 0.0, -1.0, 0.5], [0.25, 1.0, 0.0, 1.5]])
    affine_target = source @ weights + np.array([0.5, -0.25, 1.0, 2.0])
    full_linear = linear.fit_full_linear(source, affine_target)
    if not np.allclose(linear.apply_full_linear(source, full_linear), affine_target, rtol=2e-5, atol=2e-5):
        raise RuntimeError("synthetic Full Linear recovery failed")
    ridge = linear.fit_ridge(source, affine_target, 1e-6)
    if linear.apply_ridge(source, ridge).shape != affine_target.shape:
        raise RuntimeError("synthetic Ridge application failed")

    feature_bank = np.eye(4, dtype=np.float64).reshape(4, 1, 4)
    retrieval = linear.same_image_retrieval(feature_bank, feature_bank, ["a", "b", "c", "d"], ["a", "b", "c", "d"])
    if retrieval["recall_at_1"] != 1.0 or retrieval["recall_at_5"] != 1.0:
        raise RuntimeError("synthetic retrieval failed")
    records = [
        {"split": "validation", "lambda_normalized": value, "same_image_recall_at_1": 0.5, "same_image_recall_at_5": 0.5, "connector_cosine": float(index)}
        for index, value in enumerate(linear.RIDGE_LAMBDA_NORMALIZED_GRID)
    ]
    selection = linear.select_ridge_candidate(records)
    if selection["selected_lambda_normalized"] != 100.0:
        raise RuntimeError("synthetic Ridge selection failed")
    plan = linear.generation_plan(config)
    return {
        "status": "PASS_QUERY_ADAPTER_LINEAR_SYNTHETIC",
        "config_schema_id": config["schema_id"],
        "checks": ["calibration", "procrustes", "full_linear", "ridge", "retrieval", "ridge_selection", "generation_plan"],
        "planned_total_generations": plan["total"],
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate direct linear adapter logic using synthetic arrays only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args(arguments)
    if not args.synthetic_smoke:
        parser.error("--synthetic-smoke is required; no real-data mode exists")
    print(json.dumps(synthetic_smoke(args.config), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
