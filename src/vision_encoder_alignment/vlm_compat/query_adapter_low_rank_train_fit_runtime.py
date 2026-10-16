"""Train-only low-rank fit orchestration over the accepted runtime binding."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_runtime_binding as binding


SCHEMA_ID = "vea.p2.query_adapter_low_rank_train_fit_runtime.v1"
TASK_ID = "VEA-P2-T004E"
STATUS = "PASS_TRAIN_ONLY_LOW_RANK_FIT"
SOURCE_ORDER = ("SIGLIP2", "DINOV3")


class LowRankTrainFitError(RuntimeError):
    """A fail-closed input, output, or persisted-factor invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LowRankTrainFitError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_root(value: str | Path, name: str) -> Path:
    path = Path(value)
    _require(path.is_absolute() and path.is_dir(), f"{name} must be an existing absolute directory")
    return path.resolve(strict=True)


def _output_root(value: str | Path, repository: Path, sources: Sequence[Path]) -> Path:
    path = Path(value)
    _require(path.is_absolute(), "output_root must be explicit and absolute")
    _require(path.parent.is_dir(), "output_root parent must exist")
    parent = path.parent.resolve(strict=True)
    output = parent / path.name
    _require(path.name not in ("", ".", "..") and not output.exists() and not output.is_symlink(),
             "output_root must be fresh and absent")
    _require(not output.is_relative_to(repository) and not repository.is_relative_to(output),
             "output_root overlaps the repository")
    for source in sources:
        _require(not output.is_relative_to(source) and not source.is_relative_to(output),
                 "output_root overlaps a read-only input root")
    return output


def _sha(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and
             all(character in "0123456789abcdef" for character in value), f"{label} must be SHA-256")
    return value


def _validate_bound(bound: binding.BoundLowRankInputs) -> dict[str, Any]:
    _require(isinstance(bound, binding.BoundLowRankInputs), "binder returned unexpected type")
    config = core.validate_low_rank_config(bound.config)
    _require(tuple(config["sources"]["order"]) == SOURCE_ORDER and
             tuple(bound.ols) == SOURCE_ORDER and tuple(bound.train_factories) == SOURCE_ORDER,
             "source order or count differs from frozen order")
    provenance = bound.provenance
    _require(isinstance(provenance, Mapping) and provenance.get("schema_id") == binding.SCHEMA_ID,
             "binding provenance schema mismatch")
    _sha(provenance.get("t001d_plan_id"), "T001D plan ID")
    _require(provenance.get("manifest_sha256") == config["split"]["manifest_sha256"],
             "manifest identity mismatch")
    banks = provenance.get("bank_identity_sha256")
    _require(isinstance(banks, Mapping) and set(banks) == set(binding.BANK_ORDER),
             "bank identity set mismatch")
    for bank in binding.BANK_ORDER:
        _sha(banks[bank], f"{bank} bank identity")
    _require(provenance.get("validation_result_sha256") ==
             config["provenance"]["accepted_linear_test_binding"]["validation_result_sha256"],
             "accepted Validation result identity mismatch")
    _sha(provenance.get("validation_runtime_identity_sha256"), "Validation runtime identity")
    ols_hashes = provenance.get("ols_artifact_sha256")
    _require(isinstance(ols_hashes, Mapping) and set(ols_hashes) == set(SOURCE_ORDER),
             "OLS identity set mismatch")
    for source in SOURCE_ORDER:
        expected = config["ols_control"]["exact_artifact_sha256_by_source"][source]
        _require(ols_hashes[source] == expected and bound.ols[source].artifact_sha256 == expected,
                 f"{source} accepted OLS identity mismatch")
        _require(callable(bound.train_factory(source)), f"{source} Train factory missing")
    _require(type(provenance.get("train_image_count")) is int and
             provenance["train_image_count"] == config["split"]["train"]["image_count"],
             "Train image count mismatch")
    for key in ("test_scientific_access", "test_payload_open_count"):
        _require(type(provenance.get(key)) is int and provenance[key] == 0, f"{key} must be zero")
    _require(provenance.get("test_only_units_hashed_as_opaque_bytes") is True,
             "Test-only units must remain opaque")
    _require(provenance.get("float32_latency_evidence") is False,
             "float32 latency evidence cannot be claimed")
    return config


def _validate_fit(result: core.RRRResult, ranks: tuple[int, ...], images: int,
                  patches: int, dim: int, synthetic: bool, config: Mapping[str, Any]) -> None:
    _require(isinstance(result, core.RRRResult) and tuple(result.by_rank) == ranks,
             "RRR rank grid or order mismatch")
    _require(result.train_image_count == images and result.train_row_count == images * patches and
             result.synthetic_test_only is synthetic, "RRR Train count or mode mismatch")
    _require(type(result.positive_eigenvalue_count) is int and
             0 <= result.positive_eigenvalue_count <= dim, "RRR spectrum diagnostic invalid")
    for rank in ranks:
        item = result.by_rank[rank]
        if isinstance(item, core.DegenerateRank):
            _require(item.rank == rank and item.status == "DEGENERATE_RANK" and
                     rank > result.positive_eigenvalue_count, "invalid degenerate rank")
            continue
        _require(isinstance(item, core.LowRankFactors) and item.rank == rank and
                 rank <= result.positive_eigenvalue_count and item.feature_dim == dim and
                 item.patches_per_image == patches and item.synthetic_test_only is synthetic,
                 "compact rank geometry or disposition mismatch")
        _require(item.scalar_count == 2 * dim * rank + dim, "compact scalar count mismatch")
        _require(item.input_feature_class == "raw_source_preconnector_patch" and
                 item.output_feature_class == "adapted_target_like_preconnector_patch",
                 "compact feature class mismatch")
        for name, shape in (("U", (dim, rank)), ("V", (dim, rank)), ("bias", (dim,))):
            value = getattr(item, name)
            _require(isinstance(value, np.ndarray) and value.shape == shape and
                     value.dtype == np.float32 and np.isfinite(value).all(),
                     f"{name} shape, dtype, or finiteness mismatch")
        if not synthetic:
            _require(item.scalar_count == core.production_parameter_count(rank, config),
                     "frozen production scalar count mismatch")


def _sample(patches: int, dim: int) -> np.ndarray:
    # Bounded, deterministic raw-source-shaped input; no Train data is reused.
    coordinates = np.arange(patches * dim, dtype=np.float32).reshape(1, patches, dim)
    return ((coordinates % 23) - 11) / 23


def _factor_relative_path(source: str, rank: int) -> str:
    _require(source in SOURCE_ORDER and type(rank) is int and rank > 0,
             "invalid factor filename identity")
    return f"factors/{source.lower()}_rank_{rank:03d}.npz"


def _verify_npz(path: Path, item: core.LowRankFactors, expected_sha: str) -> None:
    _require(_sha_file(path) == expected_sha, "saved factor SHA mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(set(archive.files) == {"U", "V", "bias"} and len(archive.files) == 3 and
                     set(archive.zip.namelist()) == {"U.npy", "V.npy", "bias.npy"} and
                     len(archive.zip.namelist()) == 3, "saved factor field set mismatch")
            arrays = {name: np.array(archive[name], copy=True) for name in ("U", "V", "bias")}
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise LowRankTrainFitError("saved factor archive is unreadable") from error
    for name, expected in (("U", item.U), ("V", item.V), ("bias", item.bias)):
        actual = arrays[name]
        _require(actual.shape == expected.shape and actual.dtype == np.float32 and
                 np.isfinite(actual).all() and np.array_equal(actual, expected),
                 f"saved {name} differs from in-memory compact factor")
    persisted = replace(item, U=arrays["U"], V=arrays["V"], bias=arrays["bias"])
    sample = _sample(item.patches_per_image, item.feature_dim)
    _require(np.array_equal(core.apply_low_rank(sample, persisted), core.apply_low_rank(sample, item)),
             "saved compact factor application parity mismatch")
    _require(_sha_file(path) == expected_sha, "saved factor changed during verification")


def _run_bound(bound: binding.BoundLowRankInputs, output: Path, *,
               synthetic_shape: core.SyntheticTestShape | None = None,
               synthetic_ranks: tuple[int, ...] | None = None) -> dict[str, Any]:
    config = _validate_bound(bound)
    synthetic = synthetic_shape is not None
    ranks = synthetic_ranks if synthetic else core.RANK_GRID
    _require(isinstance(ranks, tuple) and len(ranks) > 0 and len(set(ranks)) == len(ranks),
             "rank grid malformed or duplicate")
    if synthetic:
        _require(isinstance(synthetic_shape, core.SyntheticTestShape), "synthetic geometry missing")
        patches, dim, images = (synthetic_shape.patches_per_image, synthetic_shape.feature_dim,
                                synthetic_shape.expected_train_images)
    else:
        _require(synthetic_ranks is None, "production rank override forbidden")
        patches, dim, images = core.PRODUCTION_PATCHES, core.PRODUCTION_DIM, core.PRODUCTION_IMAGES
    records: list[dict[str, Any]] = []
    used_paths: set[str] = {"result.json"}
    for source in SOURCE_ORDER:
        accepted = bound.ols[source]
        kwargs: dict[str, Any] = {}
        if synthetic:
            kwargs = {"synthetic_test_shape": synthetic_shape, "synthetic_test_ranks": ranks}
        result = core.fit_low_rank_rrr(bound.train_factory(source), accepted.weights,
                                       accepted.source_mean, accepted.target_mean, bound.config, **kwargs)
        _validate_fit(result, ranks, images, patches, dim, synthetic, config)
        rank_records: list[dict[str, Any]] = []
        for rank in ranks:
            item = result.by_rank[rank]
            if isinstance(item, core.DegenerateRank):
                rank_records.append({"rank": rank, "status": "DEGENERATE_RANK", "artifact": None})
                continue
            relative = _factor_relative_path(source, rank)
            _require(relative not in used_paths and not Path(relative).is_absolute() and
                     ".." not in Path(relative).parts and "\\" not in relative,
                     "duplicate or unsafe factor path")
            used_paths.add(relative)
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                np.savez(stream, U=item.U, V=item.V, bias=item.bias)
            sha = _sha_file(path)
            _verify_npz(path, item, sha)
            rank_records.append({"rank": rank, "status": "FITTED", "artifact": {
                "relative_path": relative, "sha256": sha, "dtype": "float32",
                "shape": {"U": [dim, rank], "V": [dim, rank], "bias": [dim]},
                "scalar_count": item.scalar_count,
                "theoretical_raw_parameter_bytes": item.scalar_count * 4,
                "serialized_bytes": path.stat().st_size,
            }})
        records.append({"source": source, "train_image_count": result.train_image_count,
                        "train_row_count": result.train_row_count,
                        "positive_eigenvalue_count": result.positive_eigenvalue_count,
                        "candidate_disposition": ("NO_RANK_ACCEPTED" if
                                                  result.positive_eigenvalue_count == 0 else "UNSELECTED"),
                        "ranks": rank_records})
    provenance = bound.provenance
    payload = {
        "schema_id": SCHEMA_ID, "task_id": TASK_ID, "status": STATUS,
        "synthetic_test_only": synthetic,
        "config_sha256": core.CONFIG_SHA256,
        "t001d_plan_id": provenance["t001d_plan_id"],
        "manifest_sha256": provenance["manifest_sha256"],
        "bank_identity_sha256": dict(provenance["bank_identity_sha256"]),
        "validation_result_sha256": provenance["validation_result_sha256"],
        "validation_runtime_identity_sha256": provenance["validation_runtime_identity_sha256"],
        "ols_artifact_sha256": dict(provenance["ols_artifact_sha256"]),
        "train_image_count": images, "train_row_count": images * patches,
        "source_order": list(SOURCE_ORDER), "rank_grid_ordered": list(ranks),
        "sources": records,
        "test_scientific_access": 0, "test_payload_open_count": 0,
        "validation_patch_access": 0, "generation_call_count": 0,
        "float32_latency_evidence": False,
        "numeric_note": provenance.get("numeric_note"),
        "fit_environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "platform": platform.platform(),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat()},
    }
    with (output / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return payload


def _publish(bound: binding.BoundLowRankInputs, output: Path, *,
             synthetic_shape: core.SyntheticTestShape | None = None,
             synthetic_ranks: tuple[int, ...] | None = None) -> dict[str, Any]:
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        payload = _run_bound(bound, stage, synthetic_shape=synthetic_shape,
                             synthetic_ranks=synthetic_ranks)
        _require(not output.exists() and not output.is_symlink(), "output collision before publish")
        os.rename(stage, output)  # directory rename; never replaces an existing destination
        return payload
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def run_low_rank_train_fit(repository_root: str | Path, t001d_run_root: str | Path,
                           t002d_validation_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Bind accepted production inputs, fit frozen ranks, and publish compact factors."""
    repository = _input_root(repository_root, "repository_root")
    t001d = _input_root(t001d_run_root, "t001d_run_root")
    t002d = _input_root(t002d_validation_root, "t002d_validation_root")
    output = _output_root(output_root, repository, (t001d, t002d))
    bound = binding.bind_low_rank_runtime(repository, t001d, t002d)
    return _publish(bound, output)


def _run_synthetic_bound_for_tests(bound: binding.BoundLowRankInputs, output_root: str | Path,
                                   *, shape: core.SyntheticTestShape,
                                   ranks: tuple[int, ...]) -> dict[str, Any]:
    """Private offline fixture entry: never exposed by production API or CLI."""
    output = Path(output_root)
    _require(output.is_absolute() and output.parent.is_dir() and not output.exists(),
             "synthetic output root must be fresh and absolute")
    return _publish(bound, output, synthetic_shape=shape, synthetic_ranks=ranks)
