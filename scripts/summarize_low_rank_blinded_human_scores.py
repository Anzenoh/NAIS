#!/usr/bin/env python3
"""Read-only, identity-checked analysis of blinded Validation human score exports."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts import build_low_rank_blinded_human_viewer as blind
from scripts import build_low_rank_generation_viewer as prior_viewer
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_generation as runtime
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_train_fit_runtime as fit
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as transport

SCHEMA = "vea.p2.low_rank_blinded_human_analysis.v1"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 43


def percentile(sorted_values: Sequence[float], percent: float) -> float:
    index = (len(sorted_values) - 1) * percent / 100
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    return float(sorted_values[low] * (high - index) + sorted_values[high] * (index - low))


def paired(values: Sequence[int]) -> dict[str, Any]:
    runtime.require(values and all(type(value) is int for value in values),
                    "paired score differences required")
    count = len(values)
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(sum(values[rng.randrange(count)] for _ in range(count)) / count
                   for _ in range(BOOTSTRAP_RESAMPLES))
    return {
        "image_count": count,
        "mean_paired_delta": sum(values) / count,
        "median_paired_delta": statistics.median(values),
        "win_count": sum(value > 0 for value in values),
        "tie_count": sum(value == 0 for value in values),
        "loss_count": sum(value < 0 for value in values),
        "descriptive_paired_image_percentile_bootstrap_ci_95": [
            percentile(means, 2.5), percentile(means, 97.5)],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "inference_role": "post_hoc_exploratory_descriptive_only",
    }


def summarize(value: Mapping[str, Any], assignment: Mapping[str, Any],
              protocol: Mapping[str, Any]) -> dict[str, Any]:
    coverage = blind.validate_scores(value, assignment, protocol)
    summary: dict[str, Any] = {
        "schema_id": SCHEMA, "study_id": blind.PROTOCOL_SCHEMA,
        "source_t004k_result_sha256": blind.T004K_SHA,
        "rubric_sha256": assignment["rubric_sha256"],
        "mapping_sha256": assignment["mapping_sha256"],
        "split": "validation", "selection_status": "NO_RANK_ACCEPTED",
        "coverage": coverage,
        "test_access": "NOT_ACCESSED",
        "scientific_role": "post_hoc_exploratory_absolute_caption_quality_not_selection",
    }
    if not coverage["complete"]:
        summary["status"] = "INCOMPLETE_NO_FINAL_INFERENCE"
        return summary
    scores = {lane: [] for lane in runtime.LANES}
    for entry, mapping in zip(value["entries"], assignment["assignments"], strict=True):
        runtime.require(entry["image_id"] == mapping["image_id"], "score-image pairing differs")
        for label in blind.LABELS:
            lane = mapping["display_to_lane"][label]
            score = entry["ratings"][label]
            runtime.require(type(score) is int, "complete export has unrated slot")
            scores[lane].append(score)
    per_lane: dict[str, Any] = {}
    for lane in runtime.LANES:
        values = scores[lane]
        runtime.require(len(values) == len(assignment["ordered_image_ids"]), "lane score count differs")
        per_lane[lane] = {
            "count": len(values),
            "score_distribution_0_to_5": {str(score): values.count(score) for score in range(6)},
            "mean": statistics.mean(values),
            "median": statistics.median(values),
        }
    within = {}
    adapted_native = {}
    for source in runtime.binding.SOURCE_ORDER:
        ols = scores[f"{source}/OLS_REFERENCE"]
        low_rank = scores[f"{source}/LOW_RANK_256"]
        native = scores["NATIVE"]
        within[source] = paired([candidate - reference
                                 for candidate, reference in zip(low_rank, ols, strict=True)])
        adapted_native[source] = {
            "OLS_REFERENCE": paired([candidate - reference
                                     for candidate, reference in zip(ols, native, strict=True)]),
            "LOW_RANK_256": paired([candidate - reference
                                    for candidate, reference in zip(low_rank, native, strict=True)]),
        }
    summary.update({
        "status": "COMPLETE_EXPLORATORY_DESCRIPTIVE_ONLY",
        "native_absolute_image_grounded_quality": per_lane["NATIVE"],
        "per_lane": per_lane,
        "within_source_low_rank_minus_ols": within,
        "adapted_minus_native_human_score": adapted_native,
        "equivalence_or_significance_claim": None,
    })
    return summary


def fresh_json_path(value: Path, roots: Sequence[Path]) -> Path:
    runtime.require(value.is_absolute() and value.parent.is_dir() and not value.is_symlink(),
                    "analysis output path must be explicit and external")
    path = value.resolve()
    runtime.require(path.suffix.lower() == ".json" and not path.exists() and
                    all(not path.is_relative_to(root) for root in roots),
                    "analysis output JSON must be fresh and outside input roots")
    return path


def analyze(*, repository: Path, result_root: Path, result_sha256: str,
            t001d_root: Path, protocol_path: Path, assignment_path: Path,
            scores_path: Path, output_file: Path | None = None) -> dict[str, Any]:
    protocol = blind.load_protocol(repository, protocol_path)
    runtime.require(result_sha256 == protocol["source_t004k_result_sha256"],
                    "T004K reviewed result SHA differs")
    result, rows, images = prior_viewer.load_verified(
        repository=repository, result_root=result_root,
        result_sha256=result_sha256, t001d_root=t001d_root)
    runtime.require(len(rows) == 500 and len(images) == 100 and
                    result["selection_status"] == "NO_RANK_ACCEPTED" and
                    result["test_generation_count"] == 0 and
                    result["synthetic_test_only"] is False,
                    "verified T004K production identity differs")
    ids = tuple(result["ordered_image_ids"])
    runtime.require(len(ids) == 100 and tuple(image.image_id for image in images) == ids,
                    "verified Validation IDs differ")
    assignment = runtime.read_json(assignment_path)
    blind.validate_assignment(assignment, ids, protocol)
    value = runtime.read_json(scores_path)
    summary = summarize(value, assignment, protocol)
    if output_file is not None:
        plan, _ = transport.load_bound_plan(t001d_root)
        dataset_root = Path(plan["roots"]["dataset_root"]).resolve()
        path = fresh_json_path(output_file, (repository, result_root, t001d_root, dataset_root))
        runtime.require(path != assignment_path.resolve() and path != scores_path.resolve(),
                        "analysis output must not overwrite source input")
        runtime.write_new(path, runtime.canonical(summary))
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "result-root", "t001d-run-root", "protocol-config",
                 "assignment", "scores"):
        value.add_argument(f"--{name}", type=Path, required=True)
    value.add_argument("--result-sha256", required=True)
    value.add_argument("--output-file", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        repository = fit._input_root(args.repository_root, "repository_root")
        runtime.require(repository == ROOT.resolve(), "repository root differs")
        result_root = fit._input_root(args.result_root, "result_root")
        t001d_root = fit._input_root(args.t001d_run_root, "t001d_run_root")
        for label, path in (("protocol_config", args.protocol_config),
                            ("assignment", args.assignment), ("scores", args.scores)):
            runtime.require(path.is_absolute() and path.is_file(), f"{label} must be an existing absolute file")
        summary = analyze(repository=repository, result_root=result_root,
                          result_sha256=args.result_sha256, t001d_root=t001d_root,
                          protocol_path=args.protocol_config, assignment_path=args.assignment,
                          scores_path=args.scores, output_file=args.output_file)
    except (runtime.DiagnosticError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
