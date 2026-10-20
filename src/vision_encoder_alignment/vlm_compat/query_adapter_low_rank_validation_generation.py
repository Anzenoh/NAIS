"""Post-hoc Validation generation diagnostic over accepted OLS and rank-256 factors."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_runtime_binding as binding
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_train_fit_runtime as train_fit
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_metrics_runtime as metrics
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as transport

CONFIG_PATH = Path("configs/vlm_compat/p2_low_rank_validation_generation_diagnostic_v1.json")
CONFIG_SHA256 = "7e5e6a1c1aa0829c8f2ddbea647c0d762138a6dfdbdd8f6538edf64940cc12b5"
SCHEMA = "vea.p2.low_rank_ols256_validation_generation_diagnostic.v1"
ROW_SCHEMA = "vea.p2.low_rank_validation_generation_row.v1"
RESULT_SCHEMA = "vea.p2.low_rank_validation_generation_result.v1"
STATUS = "PASS_EXPLORATORY_VALIDATION_GENERATION"
LANES = ("NATIVE", "SIGLIP2/OLS_REFERENCE", "SIGLIP2/LOW_RANK_256",
         "DINOV3/OLS_REFERENCE", "DINOV3/LOW_RANK_256")
T004F_SHA = metrics.ACCEPTED_T004F_RESULT_SHA256
T004H_SHA = "ee4c36afd3597a4e737a3af33e5a27a7c944bad282331c2fa2a0e3558fa70fe2"
GENERATION = {"prompt": "What is shown in this image? Answer briefly.", "max_new_tokens": 32,
              "num_beams": 1, "do_sample": False, "top_p": None, "temperature": 0}
ZERO = {"test_scientific_access": 0, "test_payload_open_count": 0,
        "test_metric_count": 0, "test_generation_count": 0}
ROW_KEYS = frozenset({"schema_id", "image_id", "relative_path", "lane_key", "source", "method",
                      "token_ids", "decoded_text", "prompt_decoding_identity", "image_sha256",
                      "artifact_sha256", "vision_calls", "connector_calls", "injection_calls",
                      "injection_restored", "native_exact_match", "token_unigram_f1",
                      "same_image_rank"})
RESULT_KEYS = frozenset({"schema_id", "status", "task_id", "study_id", "role", "split",
                         "selection_status", "synthetic_test_only", "validation_image_count",
                         "ordered_image_ids", "lane_order", "generation_counts", "rows_sha256",
                         "aggregate_sha256", "config_sha256", "t004f_result_sha256",
                         "t004h_result_sha256", "provenance", "artifact_sha256", "aggregate",
                         *ZERO})


class DiagnosticError(RuntimeError):
    """The frozen Validation diagnostic or a required artifact differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("utf-8")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=_invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DiagnosticError(f"invalid JSON: {path.name}") from error
    require(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def load_config(repository_root: Path) -> dict[str, Any]:
    path = repository_root / CONFIG_PATH
    require(sha_file(path) == CONFIG_SHA256, "frozen T004J config SHA differs")
    config = read_json(path)
    fixed = {"schema_id": SCHEMA, "study_id": SCHEMA, "task_id": "VEA-P2-T004J",
             "role": "post_hoc_exploratory_validation_diagnostic_not_rank_selection",
             "dataset": "p1_1000", "split": "validation", "validation_image_count": 100,
             "source_order": list(binding.SOURCE_ORDER), "target": "TARGET_CLIP",
             "lane_order": list(LANES), "generation_plan": {"native": 100, "adapted": 400, "total": 500},
             "generation": GENERATION, "t004f_result_sha256": T004F_SHA,
             "t004h_result_sha256": T004H_SHA,
             "t004g_reviewed_commit": "a67d923b2b65c64f9782d70cd13f183312ed60fd",
             "t004i_reviewed_commit": "92a6babf316cdc4594dba97b1912ef1b4ed907aa",
             "selection_status": "NO_RANK_ACCEPTED", **ZERO}
    require(config == fixed, "frozen T004J config differs")
    return config


@dataclass(frozen=True)
class ValidationImage:
    image_id: str
    path: Path
    relative_path: str
    sha256: str


def validation_images(t001d_root: Path, ids: tuple[str, ...], provenance: Mapping[str, Any]) -> list[ValidationImage]:
    """Read only Validation manifest rows; each image must match its approved SHA."""
    plan, plan_id = transport.load_bound_plan(t001d_root)
    require(plan_id == provenance["t001d_plan_id"] and
            plan.get("manifest_sha256") == provenance["manifest_sha256"], "T001D plan identity differs")
    dataset_root = Path(plan["roots"]["dataset_root"])
    require(dataset_root.is_absolute() and dataset_root.is_dir(), "dataset root unavailable")
    dataset_root = dataset_root.resolve(strict=True)
    rows = [row for row in plan["manifest_rows"] if row.get("split") == "validation"]
    require(tuple(row.get("image_id") for row in rows) == ids and len(set(ids)) == len(ids),
            "Validation manifest image count/order differs")
    images = []
    seen: set[Path] = set()
    for row in rows:
        relative = row.get("relative_path")
        require(isinstance(relative, str) and relative and "\\" not in relative, "unsafe image path")
        pure = PurePosixPath(relative)
        require(not pure.is_absolute() and relative == pure.as_posix() and
                not any(part in (".", "..") for part in pure.parts)
                and ":" not in relative and pure.suffix.lower() in (".jpg", ".jpeg"), "unsafe Validation JPEG path")
        path = (dataset_root / relative).resolve()
        require(path.is_relative_to(dataset_root) and path.is_file() and path not in seen,
                "Validation image escapes root, is missing, or is repeated")
        seen.add(path)
        observed = sha_file(path)
        require(observed == row.get("source_sha256"), "Validation source-image SHA differs")
        images.append(ValidationImage(row["image_id"], path, relative, observed))
    return images


def h_metrics(root: Path, bound: binding.BoundLowRankValidationInputs,
              ids: tuple[str, ...], factor_hashes: Mapping[str, Mapping[int, str]],
              expected_sha: str = T004H_SHA) -> dict[str, dict[str, dict[str, int]]]:
    """Verify the complete T004H result and pair only OLS/rank-256 ranks by image ID."""
    path = binding._artifact(root, "result.json", "T004H result")
    require(sha_file(path) == expected_sha, "T004H result SHA differs")
    result = read_json(path)
    provenance = bound.provenance
    fixed = {"schema_id": metrics.SCHEMA_ID, "status": metrics.STATUS, "split": "validation",
             "config_sha256": core.CONFIG_SHA256, "t004f_result_sha256": T004F_SHA,
             "t001d_plan_id": provenance["t001d_plan_id"],
             "manifest_sha256": provenance["manifest_sha256"],
             "bank_identity_sha256": dict(provenance["bank_identity_sha256"]),
             "t002d_validation_result_sha256": provenance["validation_result_sha256"],
             "ols_artifact_sha256": dict(provenance["ols_artifact_sha256"]),
             "source_order": list(binding.SOURCE_ORDER),
             "rank_grid_ordered": list(core.RANK_GRID), "validation_image_count": 100,
             "feature_shape": [100, 196, 768], "feature_dtype": "float32",
             "selection_status": "UNSELECTED", "test_scientific_access": 0,
             "test_payload_open_count": 0, "test_metric_count": 0,
             "generation_call_count": 0, "connector_call_count": 0}
    for key, value in fixed.items():
        require(result.get(key) == value and type(result.get(key)) is type(value),
                f"T004H {key} differs")
    require(result.get("ordered_image_id_sha256") == hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "T004H ordered Validation image IDs differ")
    sources = result.get("sources")
    require(isinstance(sources, list) and len(sources) == 2, "T004H source count differs")
    paired: dict[str, dict[str, dict[str, int]]] = {}
    for item, source in zip(sources, binding.SOURCE_ORDER, strict=True):
        require(isinstance(item, dict) and item.get("source") == source and
                item.get("validation_image_count") == 100 and
                item.get("ordered_image_id_sha256") == result["ordered_image_id_sha256"],
                f"T004H {source} source identity differs")
        conditions = item.get("conditions")
        require(isinstance(conditions, list) and len(conditions) == 7 and
                [c.get("condition") for c in conditions if isinstance(c, dict)] ==
                ["OLS_REFERENCE", *(f"LOW_RANK_{rank}" for rank in core.RANK_GRID)],
                f"T004H {source} condition order differs")
        paired[source] = {}
        for condition in conditions:
            name = condition["condition"]
            expected_rank = None if name == "OLS_REFERENCE" else int(name.removeprefix("LOW_RANK_"))
            expected_factor = None if expected_rank is None else factor_hashes[source][expected_rank]
            require(condition.get("rank") == expected_rank and
                    condition.get("factor_sha256") == expected_factor and
                    condition.get("selection_status") == "UNSELECTED",
                    f"T004H {source}/{name} rank or factor differs")
            rows = condition.get("per_image")
            require(isinstance(rows, list) and len(rows) == 100 and
                    tuple(row.get("image_id") for row in rows if isinstance(row, dict)) == ids,
                    f"T004H {source}/{name} per-image pairing differs")
            if name in ("OLS_REFERENCE", "LOW_RANK_256"):
                ranks: dict[str, int] = {}
                for row in rows:
                    rank = row.get("same_image_rank")
                    require(type(rank) is int and 1 <= rank <= 100,
                            f"T004H {source}/{name} retrieval rank invalid")
                    ranks[row["image_id"]] = rank
                paired[source][name] = ranks
    require(sha_file(path) == expected_sha, "T004H result changed during read")
    return paired


def token_output(output: Any, adapted: bool) -> tuple[list[int], str, int, int, int, bool]:
    try:
        tokens = list(output.token_ids)
        value = output.decoded_text
        vision, connector, injection = output.vision_calls, output.connector_calls, output.injection_calls
        restored = output.injection_restored
    except AttributeError as error:
        raise DiagnosticError("generation output malformed") from error
    require(tokens and all(type(token) is int and token >= 0 for token in tokens) and
            isinstance(value, str) and isinstance(vision, int) and isinstance(connector, int) and
            isinstance(injection, int) and type(restored) is bool and restored and connector >= 1,
            "generation output/token/counter contract differs")
    require((vision == 0 and injection == 1) if adapted else (vision >= 1 and injection == 0),
            "frozen vision/injection boundary differs")
    return tokens, value, vision, connector, injection, restored


def patch(source: np.ndarray, method: str, ols: binding.AcceptedOLS,
          factor: core.LowRankFactors) -> np.ndarray:
    if method == "OLS_REFERENCE":
        fit = linear.FullLinearFit(ols.weights, ols.source_mean, ols.target_mean, 0,
                                   np.empty(0, dtype=np.float32))
        adapted = linear.apply_full_linear(source, fit)
    else:
        require(method == "LOW_RANK_256" and factor.rank == 256, "unsupported adapted lane")
        adapted = core.apply_low_rank(source, factor)
    require(adapted.shape == (1, 196, 768) and adapted.dtype == np.float32 and
            adapted.flags.c_contiguous and np.isfinite(adapted).all(), "adapted patch contract differs")
    return adapted


def aggregate(rows: Sequence[Mapping[str, Any]], ids: tuple[str, ...]) -> dict[str, Any]:
    by_key = {(row["image_id"], row["lane_key"]): row for row in rows}
    require(len(by_key) == len(rows) == len(ids) * 5 and
            tuple((row["image_id"], row["lane_key"]) for row in rows) ==
            tuple((image_id, lane) for image_id in ids for lane in LANES),
            "row count, order, or uniqueness differs")
    per_lane = {}
    for lane in LANES[1:]:
        members = [by_key[(image_id, lane)] for image_id in ids]
        per_lane[lane] = {"native_exact_match_count": sum(row["native_exact_match"] for row in members),
                          "mean_token_unigram_f1": sum(row["token_unigram_f1"] for row in members) / len(ids),
                          "generation_success_count": len(members)}
    within = {}
    for source in binding.SOURCE_ORDER:
        control = [by_key[(image_id, f"{source}/OLS_REFERENCE")] for image_id in ids]
        candidate = [by_key[(image_id, f"{source}/LOW_RANK_256")] for image_id in ids]
        within[source] = {
            "generated_text_equality_count": sum(a["decoded_text"] == b["decoded_text"]
                                                 for a, b in zip(control, candidate, strict=True)),
            "token_equality_count": sum(a["token_ids"] == b["token_ids"]
                                        for a, b in zip(control, candidate, strict=True)),
            "mean_paired_f1_delta_rank256_minus_ols": sum(
                b["token_unigram_f1"] - a["token_unigram_f1"]
                for a, b in zip(control, candidate, strict=True)) / len(ids)}
    return {"per_lane": per_lane, "within_source": within}


def validate_rows(rows: Sequence[Mapping[str, Any]], ids: tuple[str, ...],
                  images: Sequence[ValidationImage], hashes: Mapping[str, Mapping[str, str]],
                  ranks: Mapping[str, Mapping[str, Mapping[str, int]]], config: Mapping[str, Any]) -> dict[str, Any]:
    require(len(images) == len(ids) and tuple(image.image_id for image in images) == ids,
            "Validation image pairing differs")
    prompt_hash = digest(config["generation"])
    require(len(rows) == len(ids) * 5, "generation row count differs")
    for index, image in enumerate(images):
        native = rows[index * 5]
        for offset, lane in enumerate(LANES):
            row = rows[index * 5 + offset]
            require(set(row) == ROW_KEYS and row["schema_id"] == ROW_SCHEMA and
                    row["image_id"] == image.image_id and row["relative_path"] == image.relative_path and
                    row["image_sha256"] == image.sha256 and row["lane_key"] == lane and
                    row["prompt_decoding_identity"] == prompt_hash,
                    "row identity, schema, or generation contract differs")
            tokens = row["token_ids"]
            require(isinstance(tokens, list) and tokens and
                    all(type(token) is int and token >= 0 for token in tokens) and
                    isinstance(row["decoded_text"], str), "row tokens/text malformed")
            adapted = offset != 0
            require(row["injection_restored"] is True and
                    type(row["vision_calls"]) is int and type(row["connector_calls"]) is int and
                    type(row["injection_calls"]) is int and row["connector_calls"] >= 1 and
                    ((row["vision_calls"] == 0 and row["injection_calls"] == 1) if adapted else
                     (row["vision_calls"] >= 1 and row["injection_calls"] == 0)),
                    "row frozen-boundary counters differ")
            if not adapted:
                require(row["source"] == "TARGET_CLIP" and row["method"] == "NATIVE" and
                        row["artifact_sha256"] is None and row["native_exact_match"] is None and
                        row["token_unigram_f1"] is None and row["same_image_rank"] is None,
                        "native row fields differ")
                continue
            source, method = lane.split("/")
            expected_sha = hashes[source][method]
            expected_rank = ranks[source][method][image.image_id]
            metric = transport.generation_metrics(tokens, native["token_ids"])
            require(row["source"] == source and row["method"] == method and
                    type(row["same_image_rank"]) is int and 1 <= row["same_image_rank"] <=
                    (100 if len(ids) == 100 else len(ids)) and
                    row["artifact_sha256"] == expected_sha and row["same_image_rank"] == expected_rank and
                    type(row["native_exact_match"]) is int and
                    row["native_exact_match"] == metric["exact_match"] and
                    type(row["token_unigram_f1"]) is float and
                    row["token_unigram_f1"] == metric["token_unigram_F1"],
                    "adapted row metric, rank, or artifact differs")
    return aggregate(rows, ids)


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def run_diagnostic(*, config: Mapping[str, Any], ids: tuple[str, ...],
                   images: Sequence[ValidationImage], arrays: Mapping[str, np.ndarray],
                   ols: Mapping[str, binding.AcceptedOLS],
                   factors: Mapping[str, core.LowRankFactors],
                   hashes: Mapping[str, Mapping[str, str]],
                   ranks: Mapping[str, Mapping[str, Mapping[str, int]]],
                   generator: Any, output_root: Path, device: str, fake: bool = False,
                   provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    require(config["lane_order"] == list(LANES) and config["generation"] == GENERATION and
            config["t004f_result_sha256"] == T004F_SHA and config["t004h_result_sha256"] == T004H_SHA,
            "frozen config differs")
    require(len(ids) == (2 if fake else 100) and len(set(ids)) == len(ids) and
            set(arrays) == set(ols) == set(factors) == set(hashes) == set(ranks) == set(binding.SOURCE_ORDER),
            "Validation source or image set differs")
    require(device == ("cpu" if fake else "cuda:0") and generator.device == device,
            "generation device identity differs")
    if not fake:
        anchor = getattr(generator, "anchor_config", None)
        require(anchor is not None and all(getattr(anchor, key, object()) == GENERATION[key]
                                           for key in ("prompt", "max_new_tokens", "num_beams", "do_sample")),
                "frozen prompt/decoding differs")
    require(output_root.is_absolute() and output_root.parent.is_dir() and not output_root.exists(),
            "output root must be fresh, absolute, and absent")
    for source in binding.SOURCE_ORDER:
        value = arrays[source]
        require(isinstance(value, np.ndarray) and value.shape == (len(ids), 196, 768) and
                value.dtype == np.float32 and np.isfinite(value).all() and
                factors[source].rank == 256 and
                set(hashes[source]) == set(ranks[source]) == {"OLS_REFERENCE", "LOW_RANK_256"} and
                set(ranks[source]["OLS_REFERENCE"]) == set(ids) and
                set(ranks[source]["LOW_RANK_256"]) == set(ids),
                f"{source} same-image patches or artifact pairing differs")
    require(tuple(image.image_id for image in images) == ids, "image order differs")
    output_root.mkdir()
    rows: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        require(image.path.is_file() and sha_file(image.path) == image.sha256,
                "Validation JPEG changed before generation")
        native_output = generator.generate_native(image.path)
        native_tokens, native_text, vision, connector, injection, restored = token_output(native_output, False)
        common = {"schema_id": ROW_SCHEMA, "image_id": image.image_id,
                  "relative_path": image.relative_path, "image_sha256": image.sha256,
                  "prompt_decoding_identity": digest(config["generation"])}
        rows.append({**common, "lane_key": "NATIVE", "source": "TARGET_CLIP", "method": "NATIVE",
                     "artifact_sha256": None, "token_ids": native_tokens, "decoded_text": native_text,
                     "vision_calls": vision, "connector_calls": connector, "injection_calls": injection,
                     "injection_restored": restored, "native_exact_match": None,
                     "token_unigram_f1": None, "same_image_rank": None})
        for lane in LANES[1:]:
            source, method = lane.split("/")
            adapted_patch = patch(arrays[source][index:index + 1], method, ols[source], factors[source])
            require(image.path.is_file() and sha_file(image.path) == image.sha256,
                    "Validation JPEG changed during generation")
            output = generator.generate_adapted(image.path, adapted_patch)
            tokens, text, vision, connector, injection, restored = token_output(output, True)
            metric = transport.generation_metrics(tokens, native_tokens)
            rows.append({**common, "lane_key": lane, "source": source, "method": method,
                         "artifact_sha256": hashes[source][method], "token_ids": tokens,
                         "decoded_text": text, "vision_calls": vision, "connector_calls": connector,
                         "injection_calls": injection, "injection_restored": restored,
                         "native_exact_match": int(metric["exact_match"]),
                         "token_unigram_f1": float(metric["token_unigram_F1"]),
                         "same_image_rank": ranks[source][method][image.image_id]})
    require(generator.native_calls == len(ids) and generator.adapted_calls == len(ids) * 4,
            "generation call counts differ")
    summary = validate_rows(rows, ids, images, hashes, ranks, config)
    rows_path = output_root / "rows.jsonl"
    aggregate_path = output_root / "aggregate.json"
    write_new(rows_path, b"".join(canonical(row) for row in rows))
    write_new(aggregate_path, canonical(summary))
    parsed = read_rows(rows_path)
    require(validate_rows(parsed, ids, images, hashes, ranks, config) == read_json(aggregate_path),
            "published rows/aggregate disagree")
    result = {"schema_id": RESULT_SCHEMA, "status": STATUS, "task_id": "VEA-P2-T004J",
              "study_id": SCHEMA, "role": config["role"], "split": "validation",
              "selection_status": "NO_RANK_ACCEPTED", "synthetic_test_only": fake,
              "validation_image_count": len(ids), "ordered_image_ids": list(ids),
              "lane_order": list(LANES),
              "generation_counts": {"native": len(ids), "adapted": len(ids) * 4, "total": len(rows)},
              "rows_sha256": sha_file(rows_path), "aggregate_sha256": sha_file(aggregate_path),
              "config_sha256": digest(config), "t004f_result_sha256": T004F_SHA,
              "t004h_result_sha256": T004H_SHA, "provenance": dict(provenance or {}),
              "artifact_sha256": {source: dict(hashes[source]) for source in binding.SOURCE_ORDER},
              "aggregate": summary, **ZERO}
    write_new(output_root / "result.json", canonical(result))
    return result


def read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        require(lines and all(line for line in lines), "rows JSONL empty or malformed")
        rows = [json.loads(line, object_pairs_hook=_unique_pairs,
                           parse_constant=_invalid_constant) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DiagnosticError("rows JSONL malformed") from error
    require(all(isinstance(row, dict) for row in rows), "rows JSONL objects required")
    return rows


def production_inputs(repository: Path, t001d: Path, t002d: Path, fit_root: Path,
                      h_root: Path) -> tuple[Any, ...]:
    bound = binding.bind_low_rank_validation_runtime(repository, t001d, t002d)
    metrics._checked_bound(bound)
    all_factors, all_hashes, _ = metrics._fit_manifest(
        fit_root, bound, result_sha=T004F_SHA, dim=768, patches=196,
        ranks=core.RANK_GRID, synthetic=False)
    arrays: dict[str, np.ndarray] = {}
    ids: tuple[str, ...] | None = None
    target_sha: str | None = None
    for source in binding.SOURCE_ORDER:
        values, target, observed_ids = metrics._validation_arrays(
            bound.validation_factory(source), images=100, patches=196, dim=768)
        observed_target_sha = hashlib.sha256(target.tobytes()).hexdigest()
        if ids is None:
            ids, target_sha = observed_ids, observed_target_sha
        require(observed_ids == ids and observed_target_sha == target_sha,
                "source Validation IDs or native target differ")
        arrays[source] = values
    assert ids is not None
    paired = h_metrics(h_root, bound, ids, all_hashes)
    images = validation_images(t001d, ids, bound.provenance)
    factors = {source: all_factors[source][256] for source in binding.SOURCE_ORDER}
    hashes = {source: {"OLS_REFERENCE": bound.ols[source].artifact_sha256,
                       "LOW_RANK_256": all_hashes[source][256]} for source in binding.SOURCE_ORDER}
    return ids, images, arrays, bound.ols, factors, hashes, paired, bound.provenance
