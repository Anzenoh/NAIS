"""Strict, immutable configuration for the Phase 1 analysis campaign."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn


CONFIG_SCHEMA_ID = "vea.p1.analysis_campaign.v1"


class ConfigError(ValueError):
    """Raised when the frozen campaign configuration is not exact."""


def _fail(message: str) -> NoReturn:
    raise ConfigError(message)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON number: {value}")


def strict_json_document(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{label}: expected UTF-8") from exc
    if text.startswith("\ufeff"):
        _fail(f"{label}: UTF-8 BOM is forbidden")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ConfigError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label}: top level must be an object")
    return value


def canonical_json_bytes(value: object) -> bytes:
    def check(item: object, path: str = "document") -> None:
        if isinstance(item, float) and not math.isfinite(item):
            _fail(f"{path}: non-finite number")
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    _fail(f"{path}: non-string key")
                check(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                check(child, f"{path}[{index}]")

    check(value)
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def label_seed(master_seed: int, label: str) -> int:
    if type(master_seed) is not int or master_seed < 0:
        _fail("master seed must be a non-negative integer")
    if not isinstance(label, str) or not label:
        _fail("seed label must be a non-empty string")
    digest = hashlib.sha256(f"{master_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _exact(value: object, keys: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        _fail(f"{path}: expected exact keys {sorted(keys)}, got {actual}")
    return value


def _strings(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        _fail(f"{path}: expected a non-empty list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        _fail(f"{path}: expected non-empty strings")
    if len(set(result)) != len(result):
        _fail(f"{path}: duplicates are forbidden")
    return result


def _positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{path}: expected a positive integer")
    return value


def _sha(value: object, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        _fail(f"{path}: expected lowercase SHA-256")
    if any(char not in "0123456789abcdef" for char in value):
        _fail(f"{path}: expected lowercase SHA-256")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Validated frozen campaign configuration and its canonical identity."""

    document: Mapping[str, Any]
    canonical_bytes: bytes
    sha256: str

    @property
    def master_seed(self) -> int:
        return int(self.document["randomness"]["master_seed"])

    def seed(self, label: str) -> int:
        return label_seed(self.master_seed, label)

    @property
    def encoder_keys(self) -> tuple[str, ...]:
        return tuple(item["key"] for item in self.document["input"]["encoders"])

    @property
    def split_counts(self) -> Mapping[str, int]:
        return self.document["splits"]["counts"]


_TOP = {
    "schema_id", "campaign_id", "input", "splits", "randomness",
    "resampling", "mapping", "patch", "endpoints", "extension",
    "multiple_comparisons", "numerics", "conclusion_rules", "execution",
}


def validate_config_document(root: dict[str, Any]) -> None:
    _exact(root, _TOP, "config")
    if root["schema_id"] != CONFIG_SCHEMA_ID:
        _fail("config.schema_id: unsupported")
    if root["campaign_id"] != "p1_analysis_campaign_v1":
        _fail("config.campaign_id: unsupported")

    input_doc = _exact(root["input"], {
        "run_id", "extraction_commit", "extraction_tree", "run_manifest_sha256",
        "image_manifest_sha256", "split_sha256", "checksum_manifest_sha256",
        "evidence_manifest_sha256", "tensor_schema_id", "image_count",
        "shard_image_count", "tensor_keys", "encoders",
    }, "config.input")
    if input_doc["run_id"] != "20260803T192848Z_p1_1000_three_encoder_27688e67b0c6":
        _fail("config.input.run_id: unexpected identity")
    if input_doc["extraction_commit"] != "27688e67b0c639a81d07c31b7992dc8a09341d8d":
        _fail("config.input.extraction_commit: unexpected identity")
    if input_doc["extraction_tree"] != "b6fb3ef7fc99e0a4f16c351867f2da1fea100760":
        _fail("config.input.extraction_tree: unexpected identity")
    for key in (
        "run_manifest_sha256", "image_manifest_sha256", "split_sha256",
        "checksum_manifest_sha256", "evidence_manifest_sha256",
    ):
        _sha(input_doc[key], f"config.input.{key}")
    if input_doc["tensor_schema_id"] != "vea.p1.feature_shard.v0":
        _fail("config.input.tensor_schema_id: unsupported")
    if input_doc["image_count"] != 1000 or input_doc["shard_image_count"] != 25:
        _fail("config.input: unexpected image or shard count")
    keys = _strings(input_doc["tensor_keys"], "config.input.tensor_keys")
    if len(keys) != 18:
        _fail("config.input.tensor_keys: expected 18 keys")
    encoders = input_doc["encoders"]
    if not isinstance(encoders, list) or len(encoders) != 3:
        _fail("config.input.encoders: expected exactly three encoders")
    expected_encoder_keys = (
        "clip_vit_b16", "siglip2_base_p16_224", "dinov3_vit_b16",
    )
    for index, encoder in enumerate(encoders):
        item = _exact(encoder, {
            "key", "model_id", "revision", "lane_id", "hidden_size",
            "patch_count", "grid", "dtype", "global_taps", "patch_taps",
        }, f"config.input.encoders[{index}]")
        if item["key"] != expected_encoder_keys[index]:
            _fail("config.input.encoders: order or key mismatch")
        if item["hidden_size"] != 768 or item["patch_count"] != 196:
            _fail("config.input.encoders: unexpected tensor geometry")
        if item["grid"] != [14, 14] or item["dtype"] != "float32":
            _fail("config.input.encoders: unexpected grid or dtype")
        if len(_strings(item["global_taps"], "global_taps")) != 14:
            _fail("config.input.encoders.global_taps: expected 14")
        if tuple(_strings(item["patch_taps"], "patch_taps")) != (
            "block_03", "block_06", "block_09", "block_12",
        ):
            _fail("config.input.encoders.patch_taps: unexpected")

    splits = _exact(root["splits"], {
        "manifest_path", "algorithm_id", "seed", "counts", "unit",
    }, "config.splits")
    if splits["manifest_path"] != "data/manifests/p1_1000.jsonl":
        _fail("config.splits.manifest_path: unexpected")
    if splits["algorithm_id"] != "numpy-pcg64-largest-remainder-v1":
        _fail("config.splits.algorithm_id: unexpected")
    if splits["seed"] != 43 or splits["unit"] != "image_id":
        _fail("config.splits: unexpected seed or unit")
    counts = _exact(splits["counts"], {"train", "validation", "test"}, "counts")
    if counts != {"train": 800, "validation": 100, "test": 100}:
        _fail("config.splits.counts: unexpected")

    randomness = _exact(root["randomness"], {"master_seed", "derivation"}, "randomness")
    if randomness != {
        "master_seed": 20260804,
        "derivation": "sha256-first-u64-big-endian-v1",
    }:
        _fail("config.randomness: unexpected")
    resampling = _exact(root["resampling"], {
        "global_permutations", "global_primary_bootstraps",
        "mapping_shuffled_refits", "patch_permutations",
        "patch_selected_bootstraps", "confidence_level", "unit",
    }, "resampling")
    expected_counts = {
        "global_permutations": 1000, "global_primary_bootstraps": 2000,
        "mapping_shuffled_refits": 500, "patch_permutations": 500,
        "patch_selected_bootstraps": 1000,
    }
    for key, expected in expected_counts.items():
        if resampling[key] != expected:
            _fail(f"config.resampling.{key}: expected {expected}")
    if resampling["confidence_level"] != 0.95 or resampling["unit"] != "image_block":
        _fail("config.resampling: unexpected confidence level or unit")

    mapping = _exact(root["mapping"], {
        "ridge_relative_alphas", "ridge_scale_formula", "knn_values",
        "reduced_rank_candidates", "multiway_latent_candidates",
        "selection_rule", "tie_break",
    }, "mapping")
    for key in ("ridge_relative_alphas", "reduced_rank_candidates", "multiway_latent_candidates"):
        if not isinstance(mapping[key], list) or not mapping[key]:
            _fail(f"config.mapping.{key}: expected non-empty list")
    if mapping["reduced_rank_candidates"] != [32, 64, 128, 256, 512, 768]:
        _fail("config.mapping.reduced_rank_candidates: unexpected")
    if mapping["multiway_latent_candidates"] != [32, 64, 128, 256, 512, 768]:
        _fail("config.mapping.multiway_latent_candidates: unexpected")
    if mapping["knn_values"] != [5, 10, 20, 50]:
        _fail("config.mapping.knn_values: unexpected")
    if mapping["selection_rule"] != "validation_one_standard_error_image_bootstrap":
        _fail("config.mapping.selection_rule: unexpected")

    patch = _exact(root["patch"], {
        "taps", "states", "position_template", "retrieval_metrics",
        "not_applicable_label",
    }, "patch")
    if patch["taps"] != ["block_03", "block_06", "block_09", "block_12"]:
        _fail("config.patch.taps: unexpected")
    if patch["states"] != ["X", "W", "D"]:
        _fail("config.patch.states: unexpected")
    if patch["not_applicable_label"] != "NOT_APPLICABLE_PATCH_TAP_UNAVAILABLE":
        _fail("config.patch.not_applicable_label: unexpected")

    endpoints = _exact(root["endpoints"], {"primary", "secondary", "exploratory"}, "endpoints")
    for family in endpoints.values():
        _strings(family, "config.endpoints family")
    extension = _exact(root["extension"], {
        "authorized", "confirmatory", "families", "mlp_authorized",
        "cross_attention_authorized", "nonlinear_cli_enabled",
    }, "extension")
    if extension["authorized"] is not True or extension["confirmatory"] is not True:
        _fail("config.extension: linear extension must be confirmatory and authorized")
    if any(extension[key] is not False for key in (
        "mlp_authorized", "cross_attention_authorized", "nonlinear_cli_enabled",
    )):
        _fail("config.extension: nonlinear methods are forbidden")

    multiple = _exact(root["multiple_comparisons"], {"method", "alpha", "families"}, "multiple")
    if multiple["method"] != "holm_bonferroni" or multiple["alpha"] != 0.05:
        _fail("config.multiple_comparisons: unexpected")
    _strings(multiple["families"], "config.multiple_comparisons.families")
    numerics = _exact(root["numerics"], {
        "input_dtype", "accumulation_dtype", "zero_variance", "zero_norm",
        "undefined_metric", "condition_number_limit", "svd_failure",
        "nonfinite", "solver", "tie_break",
    }, "numerics")
    if numerics["input_dtype"] != "float32" or numerics["accumulation_dtype"] != "float64":
        _fail("config.numerics: unexpected dtype")
    if not isinstance(numerics["condition_number_limit"], (int, float)):
        _fail("config.numerics.condition_number_limit: expected number")
    if not math.isfinite(float(numerics["condition_number_limit"])):
        _fail("config.numerics.condition_number_limit: non-finite")

    rules = _exact(root["conclusion_rules"], {"version", "labels", "gates"}, "rules")
    if rules["version"] != "A-E-null-calibrated-v1":
        _fail("config.conclusion_rules.version: unexpected")
    if rules["labels"] != ["A", "B", "C", "D", "E", "INDETERMINATE"]:
        _fail("config.conclusion_rules.labels: unexpected")
    _exact(rules["gates"], {"A", "B", "C", "D", "E", "INDETERMINATE"}, "rules.gates")

    execution = _exact(root["execution"], {
        "state_sequence", "analysis_backend", "thread_count", "seal_eligible",
        "synthetic_mode_seal_eligible", "repeat_control", "result_schema_id",
        "archive_format", "fixed_archive_mtime", "value_chunk_images",
    }, "execution")
    expected_states = [
        "PREFLIGHT", "BLIND_INPUT_INTEGRITY", "REPEAT_CONTROL", "T004_DEV",
        "T005_DEV", "T006_DEV", "EXTENSION_DEV", "FREEZE",
        "T007_TEST_UNSEAL", "REPORT", "SEAL", "VERIFY",
    ]
    if execution["state_sequence"] != expected_states:
        _fail("config.execution.state_sequence: unexpected")
    if execution["analysis_backend"] != "numpy_cpu" or execution["thread_count"] != 1:
        _fail("config.execution: unexpected backend")
    if execution["seal_eligible"] is not True or execution["synthetic_mode_seal_eligible"] is not False:
        _fail("config.execution: unexpected sealing policy")
    repeat = _exact(execution["repeat_control"], {
        "apple_required", "windows_local_status", "checksum_integrity_is_repeat",
        "windows_local_authorized",
    }, "execution.repeat_control")
    if repeat != {
        "apple_required": True,
        "windows_local_status": "NOT_EXECUTED_LOCAL_RESOURCE_UNAVAILABLE",
        "checksum_integrity_is_repeat": False,
        "windows_local_authorized": True,
    }:
        _fail("config.execution.repeat_control: unexpected")
    _positive_int(execution["value_chunk_images"], "execution.value_chunk_images")


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"unable to read analysis config: {exc}") from exc
    root = strict_json_document(raw, label="analysis config")
    validate_config_document(root)
    canonical = canonical_json_bytes(root)
    if raw != canonical:
        _fail("analysis config is not canonical JSON")
    return AnalysisConfig(
        document=_freeze(root),
        canonical_bytes=canonical,
        sha256=sha256_bytes(canonical),
    )


__all__ = [
    "AnalysisConfig", "CONFIG_SCHEMA_ID", "ConfigError", "canonical_json_bytes",
    "label_seed", "load_analysis_config", "sha256_bytes", "strict_json_document",
    "validate_config_document",
]
