"""Read-only Validation patch and same-image identity metrics for frozen RRR fits."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_runtime_binding as binding
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_train_fit_runtime as train_fit


SCHEMA_ID = "vea.p2.query_adapter_low_rank_validation_metrics.v1"
STATUS = "PASS_VALIDATION_PATCH_IDENTITY_ONLY"
ACCEPTED_T004F_RESULT_SHA256 = "84549208e6e7a097e769b89b254bcf1bf939b1e29a3c06185a77f0afbdd858f3"


class LowRankValidationError(RuntimeError):
    """Accepted identity, Validation stream, or metric invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LowRankValidationError(message)


def _sha(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and
             all(character in "0123456789abcdef" for character in value), f"{label} is not SHA-256")
    return value


def _checked_bound(bound: binding.BoundLowRankValidationInputs) -> dict[str, Any]:
    _require(isinstance(bound, binding.BoundLowRankValidationInputs), "Validation binder returned unexpected type")
    config = core.validate_low_rank_config(bound.config)
    _require(tuple(config["sources"]["order"]) == binding.SOURCE_ORDER and
             tuple(bound.ols) == binding.SOURCE_ORDER and
             tuple(bound.validation_factories) == binding.SOURCE_ORDER,
             "Validation source order differs")
    provenance = bound.provenance
    _require(isinstance(provenance, Mapping) and
             provenance.get("schema_id") == "vea.p2.query_adapter_low_rank_validation_binding.v1",
             "Validation provenance schema differs")
    _sha(provenance.get("t001d_plan_id"), "T001D plan")
    _require(provenance.get("manifest_sha256") == config["split"]["manifest_sha256"],
             "Validation manifest differs")
    banks = provenance.get("bank_identity_sha256")
    _require(isinstance(banks, Mapping) and set(banks) == set(binding.BANK_ORDER),
             "Validation bank identity set differs")
    for bank in binding.BANK_ORDER:
        _sha(banks[bank], f"{bank} bank")
    _require(provenance.get("validation_result_sha256") ==
             config["provenance"]["accepted_linear_test_binding"]["validation_result_sha256"],
             "accepted T002D result differs")
    _sha(provenance.get("validation_runtime_identity_sha256"), "T002D runtime")
    hashes = provenance.get("ols_artifact_sha256")
    _require(isinstance(hashes, Mapping) and set(hashes) == set(binding.SOURCE_ORDER),
             "accepted OLS set differs")
    for source in binding.SOURCE_ORDER:
        expected = config["ols_control"]["exact_artifact_sha256_by_source"][source]
        _require(hashes[source] == expected and bound.ols[source].artifact_sha256 == expected,
                 f"{source} accepted OLS differs")
        _require(callable(bound.validation_factory(source)), f"{source} Validation factory missing")
    for key, value in (("train_image_count", 800), ("validation_image_count", 100),
                       ("test_scientific_access", 0), ("test_payload_open_count", 0)):
        _require(type(provenance.get(key)) is int and provenance[key] == value,
                 f"Validation provenance {key} differs")
    _require(provenance.get("test_only_units_hashed_as_opaque_bytes") is True,
             "Test-only units must remain opaque")
    return config


def _fit_manifest(root: Path, bound: binding.BoundLowRankValidationInputs, *,
                  result_sha: str, dim: int, patches: int, ranks: tuple[int, ...],
                  synthetic: bool) -> tuple[dict[str, dict[int, core.LowRankFactors]],
                                            dict[str, dict[int, str]], str]:
    path = binding._artifact(root, "result.json", "T004F result")
    observed_sha = binding._sha_file(path)
    _require(observed_sha == result_sha, "T004F result SHA differs from accepted identity")
    result = binding._json_file(path)
    provenance = bound.provenance
    fixed = {
        "schema_id": train_fit.SCHEMA_ID, "task_id": train_fit.TASK_ID,
        "status": train_fit.STATUS, "synthetic_test_only": synthetic,
        "config_sha256": core.CONFIG_SHA256,
        "t001d_plan_id": provenance["t001d_plan_id"],
        "manifest_sha256": provenance["manifest_sha256"],
        "bank_identity_sha256": dict(provenance["bank_identity_sha256"]),
        "validation_result_sha256": provenance["validation_result_sha256"],
        "validation_runtime_identity_sha256": provenance["validation_runtime_identity_sha256"],
        "ols_artifact_sha256": dict(provenance["ols_artifact_sha256"]),
        "train_image_count": 800 if not synthetic else result.get("train_image_count"),
        "train_row_count": 156800 if not synthetic else result.get("train_row_count"),
        "source_order": list(binding.SOURCE_ORDER), "rank_grid_ordered": list(ranks),
        "test_scientific_access": 0, "test_payload_open_count": 0,
        "validation_patch_access": 0, "generation_call_count": 0,
        "float32_latency_evidence": False,
    }
    for key, expected in fixed.items():
        _require(result.get(key) == expected and type(result.get(key)) is type(expected),
                 f"T004F result {key} differs")
    if synthetic:
        _require(type(result["train_image_count"]) is int and result["train_image_count"] > 0 and
                 type(result["train_row_count"]) is int and
                 result["train_row_count"] == result["train_image_count"] * patches,
                 "synthetic T004F Train counts differ")
    sources = result.get("sources")
    _require(isinstance(sources, list) and len(sources) == 2 and
             [item.get("source") for item in sources if isinstance(item, Mapping)] == list(binding.SOURCE_ORDER),
             "T004F source records differ")
    loaded: dict[str, dict[int, core.LowRankFactors]] = {}
    factor_hashes: dict[str, dict[int, str]] = {}
    seen_paths: set[str] = {"result.json"}
    for source_record, source in zip(sources, binding.SOURCE_ORDER, strict=True):
        _require(source_record.get("train_image_count") == result["train_image_count"] and
                 source_record.get("train_row_count") == result["train_row_count"] and
                 type(source_record.get("positive_eigenvalue_count")) is int and
                 source_record.get("positive_eigenvalue_count") >= max(ranks) and
                 source_record.get("candidate_disposition") == "UNSELECTED",
                 f"{source} T004F Train diagnostics differ")
        records = source_record.get("ranks")
        _require(isinstance(records, list) and len(records) == len(ranks) and
                 [record.get("rank") for record in records if isinstance(record, Mapping)] == list(ranks),
                 f"{source} T004F rank grid differs")
        loaded[source] = {}
        factor_hashes[source] = {}
        for record, rank in zip(records, ranks, strict=True):
            _require(type(record.get("rank")) is int and record.get("status") == "FITTED",
                     f"{source} rank {rank} was not fitted")
            artifact = record.get("artifact")
            _require(isinstance(artifact, Mapping), f"{source} rank {rank} artifact missing")
            relative = artifact.get("relative_path")
            expected_relative = f"factors/{source.lower()}_rank_{rank:03d}.npz"
            _require(relative == expected_relative and relative not in seen_paths,
                     "T004F factor path unsafe or duplicate")
            seen_paths.add(relative)
            sha = _sha(artifact.get("sha256"), f"{source} rank {rank} factor")
            count = 2 * dim * rank + dim
            _require(artifact.get("dtype") == "float32" and
                     artifact.get("shape") == {"U": [dim, rank], "V": [dim, rank], "bias": [dim]} and
                     type(artifact.get("scalar_count")) is int and artifact["scalar_count"] == count and
                     artifact.get("theoretical_raw_parameter_bytes") == count * 4,
                     f"{source} rank {rank} factor metadata differs")
            if not synthetic:
                _require(count == core.production_parameter_count(rank, bound.config),
                         "frozen factor parameter count differs")
            factor_path = binding._artifact(root, relative, f"{source} rank {rank} factor")
            _require(type(artifact.get("serialized_bytes")) is int and
                     artifact["serialized_bytes"] == factor_path.stat().st_size,
                     f"{source} rank {rank} serialized size differs")
            _require(binding._sha_file(factor_path) == sha, f"{source} rank {rank} factor SHA differs")
            try:
                with np.load(factor_path, allow_pickle=False) as archive:
                    names = archive.zip.namelist()
                    _require(len(names) == 3 and set(names) == {"U.npy", "V.npy", "bias.npy"} and
                             len(archive.files) == 3 and set(archive.files) == {"U", "V", "bias"},
                             f"{source} rank {rank} factor fields differ")
                    arrays = {name: np.array(archive[name], copy=True) for name in ("U", "V", "bias")}
            except (OSError, ValueError, EOFError, KeyError) as error:
                raise LowRankValidationError(f"{source} rank {rank} factor cannot be read") from error
            for name, shape in (("U", (dim, rank)), ("V", (dim, rank)), ("bias", (dim,))):
                value = arrays[name]
                _require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                         f"{source} rank {rank} {name} shape/dtype/finite mismatch")
            _require(binding._sha_file(factor_path) == sha, f"{source} rank {rank} factor changed during read")
            loaded[source][rank] = core.LowRankFactors(rank, arrays["U"], arrays["V"], arrays["bias"],
                                                        count, patches, dim, synthetic)
            factor_hashes[source][rank] = sha
    _require(binding._sha_file(path) == result_sha, "T004F result changed during read")
    return loaded, factor_hashes, observed_sha


def _validation_arrays(factory: binding.ValidationChunkFactory, *, images: int,
                       patches: int, dim: int) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ids: list[str] = []
    for chunk in factory():
        _require(isinstance(chunk, binding.ValidationChunk) and chunk.split == "validation",
                 "Validation factory returned non-Validation data")
        _require(isinstance(chunk.source, np.ndarray) and isinstance(chunk.target, np.ndarray) and
                 chunk.source.ndim == 3 and chunk.source.shape == chunk.target.shape and
                 chunk.source.shape[1:] == (patches, dim) and
                 chunk.source.dtype == chunk.target.dtype == np.float32 and
                 chunk.source.shape[0] * patches <= core.MAX_CHUNK_ROWS and
                 chunk.image_ids and len(chunk.image_ids) == chunk.source.shape[0],
                 "Validation patch shape, dtype or chunk bound differs")
        _require(np.isfinite(chunk.source).all() and np.isfinite(chunk.target).all(),
                 "Validation patches contain non-finite values")
        sources.append(chunk.source)
        targets.append(chunk.target)
        ids.extend(chunk.image_ids)
    _require(len(ids) == images and len(set(ids)) == images and
             all(isinstance(image_id, str) and image_id for image_id in ids),
             "Validation count or image IDs differ")
    return np.concatenate(sources), np.concatenate(targets), tuple(ids)


def _condition(candidate: np.ndarray, native: np.ndarray, ids: tuple[str, ...],
               train_target_mean: np.ndarray, *, label: str, rank: int | None,
               factor_sha: str | None) -> dict[str, Any]:
    patch = linear.patch_metrics(candidate, native, train_target_mean)
    retrieval = linear.same_image_retrieval(candidate, native, ids, ids)
    metrics = {key: float(patch[key]) for key in ("cosine", "nmse", "r2")}
    metrics.update({"recall_at_1": float(retrieval["recall_at_1"]),
                    "recall_at_5": float(retrieval["recall_at_5"])})
    _require(all(np.isfinite(value) for value in metrics.values()), "aggregate metric is non-finite")
    image_rows = []
    for index, image_id in enumerate(ids):
        row = {"image_id": image_id, "patch_cosine": float(patch["per_image_cosine"][index]),
               "patch_nmse": float(patch["per_image_nmse"][index]),
               "patch_r2": float(patch["per_image_r2"][index]),
               "same_image_rank": int(retrieval["ranks"][index])}
        _require(all(np.isfinite(value) for key, value in row.items() if key.startswith("patch_")),
                 "per-image metric is non-finite")
        image_rows.append(row)
    return {"condition": label, "rank": rank, "factor_sha256": factor_sha,
            "selection_status": "UNSELECTED", "patch_metrics": metrics,
            "per_image": image_rows,
            "connector_metrics": "NOT_EVALUATED", "generation_metrics": "NOT_EVALUATED",
            "float32_adapter_latency": "NOT_EVALUATED"}


def _evaluate(bound: binding.BoundLowRankValidationInputs, factors: Mapping[str, Mapping[int, core.LowRankFactors]],
              factor_hashes: Mapping[str, Mapping[int, str]], fit_sha: str, *, images: int, patches: int, dim: int,
              ranks: tuple[int, ...]) -> dict[str, Any]:
    _checked_bound(bound)
    source_records = []
    first_ids: tuple[str, ...] | None = None
    first_target_sha: str | None = None
    for source in binding.SOURCE_ORDER:
        raw, native, ids = _validation_arrays(bound.validation_factory(source),
                                               images=images, patches=patches, dim=dim)
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()
        target_sha = hashlib.sha256(native.tobytes()).hexdigest()
        if first_ids is None:
            first_ids, first_target_sha = ids, target_sha
        else:
            _require(ids == first_ids and target_sha == first_target_sha,
                     "Validation IDs or native target differ across sources")
        accepted = bound.ols[source]
        _require(accepted.weights.shape == (dim, dim) and
                 accepted.source_mean.shape == accepted.target_mean.shape == (dim,) and
                 all(value.dtype == np.float32 and np.isfinite(value).all() for value in
                     (accepted.weights, accepted.source_mean, accepted.target_mean)),
                 f"{source} accepted OLS arrays differ")
        ols = linear.FullLinearFit(accepted.weights, accepted.source_mean, accepted.target_mean,
                                   0, np.empty(0, dtype=np.float32))
        conditions = [_condition(linear.apply_full_linear(raw, ols), native, ids, accepted.target_mean,
                                 label="OLS_REFERENCE", rank=None, factor_sha=None)]
        _require(tuple(factors[source]) == ranks, f"{source} frozen rank order differs")
        for rank in ranks:
            compact = factors[source][rank]
            _require(isinstance(compact, core.LowRankFactors) and compact.rank == rank,
                     f"{source} rank {rank} compact factor differs")
            conditions.append(_condition(core.apply_low_rank(raw, compact), native, ids,
                                         accepted.target_mean, label=f"LOW_RANK_{rank}", rank=rank,
                                         factor_sha=factor_hashes[source][rank]))
        source_records.append({"source": source, "validation_image_count": images,
                               "ordered_image_id_sha256": digest, "conditions": conditions})
    provenance = bound.provenance
    return {"schema_id": SCHEMA_ID, "task_id": "VEA-P2-T004G", "status": STATUS,
            "split": "validation", "config_sha256": core.CONFIG_SHA256,
            "t004f_result_sha256": fit_sha, "t001d_plan_id": provenance["t001d_plan_id"],
            "manifest_sha256": provenance["manifest_sha256"],
            "bank_identity_sha256": dict(provenance["bank_identity_sha256"]),
            "t002d_validation_result_sha256": provenance["validation_result_sha256"],
            "ols_artifact_sha256": dict(provenance["ols_artifact_sha256"]),
            "source_order": list(binding.SOURCE_ORDER), "rank_grid_ordered": list(ranks),
            "validation_image_count": images, "ordered_image_id_sha256":
                hashlib.sha256(json.dumps(first_ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "feature_shape": [images, patches, dim], "feature_dtype": "float32",
            "aggregation_unit": "image", "sources": source_records,
            "selection_status": "UNSELECTED", "connector_metrics": "NOT_EVALUATED",
            "generation_metrics": "NOT_EVALUATED", "float32_adapter_latency": "NOT_EVALUATED",
            "test_scientific_access": 0, "test_payload_open_count": 0,
            "test_metric_count": 0, "generation_call_count": 0, "connector_call_count": 0}


def _publish(output: Path, payload: Mapping[str, Any]) -> None:
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        with (stage / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        _require(not output.exists() and not output.is_symlink(), "output collision before publish")
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def run_low_rank_validation_metrics(repository_root: str | Path, t001d_run_root: str | Path,
                                    t002d_validation_root: str | Path, t004f_fit_root: str | Path,
                                    output_root: str | Path) -> dict[str, Any]:
    """Evaluate frozen low-rank factors and accepted OLS on Validation patches only."""
    repository = train_fit._input_root(repository_root, "repository_root")
    t001d = train_fit._input_root(t001d_run_root, "t001d_run_root")
    t002d = train_fit._input_root(t002d_validation_root, "t002d_validation_root")
    fit_root = train_fit._input_root(t004f_fit_root, "t004f_fit_root")
    output = train_fit._output_root(output_root, repository, (t001d, t002d, fit_root))
    bound = binding.bind_low_rank_validation_runtime(repository, t001d, t002d)
    _checked_bound(bound)
    factors, factor_hashes, fit_sha = _fit_manifest(
        fit_root, bound, result_sha=ACCEPTED_T004F_RESULT_SHA256,
        dim=768, patches=196, ranks=core.RANK_GRID, synthetic=False)
    payload = _evaluate(bound, factors, factor_hashes, fit_sha,
                        images=100, patches=196, dim=768, ranks=core.RANK_GRID)
    _publish(output, payload)
    return payload


def _run_synthetic_fixture_for_tests(bound: binding.BoundLowRankValidationInputs, fit_root: Path,
                                     output_root: Path, *, images: int, patches: int, dim: int,
                                     ranks: tuple[int, ...]) -> dict[str, Any]:
    """Private small-fixture seam; public API and CLI have fixed production shape."""
    _require(fit_root.is_absolute() and fit_root.is_dir() and output_root.is_absolute() and
             output_root.parent.is_dir() and not output_root.exists(), "synthetic roots invalid")
    _checked_bound(bound)
    fit_sha = binding._sha_file(fit_root / "result.json")
    factors, factor_hashes, _ = _fit_manifest(fit_root, bound, result_sha=fit_sha,
                                              dim=dim, patches=patches, ranks=ranks, synthetic=True)
    payload = _evaluate(bound, factors, factor_hashes, fit_sha,
                        images=images, patches=patches, dim=dim, ranks=ranks)
    _publish(output_root, payload)
    return payload
