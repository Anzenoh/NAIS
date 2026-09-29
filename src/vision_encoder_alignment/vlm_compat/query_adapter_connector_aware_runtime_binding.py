"""Fail-closed, Train/Validation-only binding for the accepted T003C identities.

The public preflight receives explicit roots. It never fits, generates, or reads
the held-out Test payload; the existing bank preflight proves current bank lineage.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np
import torch

from vision_encoder_alignment.vlm_compat import query_adapter_bank_preflight as bank_preflight
from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_linear_validation_runtime as validation
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


BINDING_PATH = Path("configs/vlm_compat/p2_query_adapter_connector_aware_runtime_binding_v1.json")
DESIGN_PATH = core.CONFIG_PATH
BANK_PATH = Path("configs/vlm_compat/p2_query_adapter_linear_bank_binding_v1.json")
TEST_BINDING_PATH = Path("configs/vlm_compat/p2_query_adapter_linear_test_binding_v1.json")
SCHEMA_ID = "vea.p2.query_adapter_connector_aware_runtime_binding.v1"
SOURCE_HASHES = {
    DESIGN_PATH: core.CONFIG_SHA256,
    BANK_PATH: "d2742c93f8eeae286d490fc229ff57fa8cd6701aab81dadde656ff23433129be",
    TEST_BINDING_PATH: "7688f7f14bdf6058f314e8b23ab60cc7ce9ca59dc48df4844b328ed7844662d3",
}
BASELINE = "fe135cd325c3321809d1d6d6f517cf3964e69525"
DENIED_POLICY = {
    "test_access_authorized": False, "test_metrics_authorized": False,
    "test_generation_authorized": False, "train_validation_generation_authorized": False,
}
PRODUCTION_COUNTS = (800, 100)


class RuntimeBindingError(RuntimeError):
    """An accepted identity or access boundary could not be proven."""


def _fail(message: str) -> None:
    raise RuntimeBindingError(message)


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeBindingError(f"required identity file unavailable: {path.name}") from error


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeBindingError(f"required JSON unavailable: {path.name}") from error
    if not isinstance(value, dict):
        _fail(f"required JSON is not an object: {path.name}")
    return value


def _no_absolute_paths(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _no_absolute_paths(item)
    elif isinstance(value, list):
        for item in value:
            _no_absolute_paths(item)
    elif isinstance(value, str) and (
        value.startswith(("/", "~", "\\\\", "file://"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        _fail("binding contains a machine-local absolute path")


def load_binding(repository_root: str | Path) -> dict[str, Any]:
    """Verify all three accepted repository files and every copied binding field."""
    root = Path(repository_root).resolve()
    source: dict[Path, dict[str, Any]] = {}
    for relative, expected in SOURCE_HASHES.items():
        path = root / relative
        if _sha(path) != expected:
            _fail(f"accepted source SHA-256 drift: {relative.name}")
        source[relative] = _json(path)
    binding = _json(root / BINDING_PATH)
    _no_absolute_paths(binding)
    bank = source[BANK_PATH]
    linear_test = source[TEST_BINDING_PATH]
    expected = {
        "schema_id": SCHEMA_ID,
        "experiment_id": core.EXPERIMENT_ID,
        "connector_aware_config_sha256": SOURCE_HASHES[DESIGN_PATH],
        "implementation_predecessor": BASELINE,
        "bank_binding_config_sha256": SOURCE_HASHES[BANK_PATH],
        "linear_test_binding_config_sha256": SOURCE_HASHES[TEST_BINDING_PATH],
        "t001d": {key: bank[key] for key in (
            "t001d_plan_id", "manifest_sha256", "accepted_preflight_json_sha256", "banks"
        )},
        "t002d": {
            "validation_result_sha256": linear_test["validation_result_sha256"],
            "validation_runtime_identity_sha256": linear_test["validation_runtime_identity_sha256"],
            "full_linear": {
                name: linear_test["finalized_adapter_artifacts"][name]["FULL_LINEAR"]
                for name in core.SOURCES
            },
        },
        "runtime_roots": {
            "t001d": "EXPLICIT_T001D_RUN_ROOT", "t002d": "EXPLICIT_T002D_VALIDATION_ROOT"
        },
        "policy": DENIED_POLICY,
    }
    if binding != expected or source[DESIGN_PATH].get("experiment_id") != core.EXPERIMENT_ID:
        _fail("runtime binding schema or copied accepted provenance differs")
    return binding


def _absolute_root(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        _fail(f"{label} must be an existing explicit absolute directory")
    return path.resolve()


def _relative_artifact(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in relative:
        _fail("Full Linear artifact path escapes Validation root")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        _fail("Full Linear artifact resolves outside Validation root")
    return resolved


def _array(archive: Any, key: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.array(archive[key], copy=True, order="C")
    except (KeyError, ValueError) as error:
        raise RuntimeBindingError(f"Full Linear {key} unavailable") from error
    if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
        _fail(f"Full Linear {key} shape, dtype, or finiteness mismatch")
    return array


def _load_full_linear(root: Path, result: Mapping[str, Any], binding: Mapping[str, Any],
                      expected_hashes: Mapping[str, str]) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    identity = binding["t002d"]["validation_runtime_identity_sha256"]
    artifacts = result.get("fit_artifacts")
    if not isinstance(artifacts, Mapping):
        _fail("Validation fit artifact manifest unavailable")
    for source in core.SOURCES:
        spec = binding["t002d"]["full_linear"][source]
        relative = spec["relative_path"]
        path = _relative_artifact(root, relative)
        expected = expected_hashes[source]
        if not path.is_file() or _sha(path) != expected or artifacts.get(relative) != expected:
            _fail(f"{source} Full Linear artifact identity mismatch")
        try:
            with np.load(path, allow_pickle=False) as archive:
                persisted = archive["runtime_identity"]
                if persisted.shape != () or str(persisted.item()) != identity:
                    _fail(f"{source} Full Linear persisted runtime identity mismatch")
                arrays = {
                    "weights": _array(archive, "weights", (768, 768)),
                    "source_mean": _array(archive, "source_mean", (768,)),
                    "target_mean": _array(archive, "target_mean", (768,)),
                }
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeBindingError(f"{source} Full Linear payload cannot be safely loaded") from error
        adapter = core.AffineAdapter.from_full_linear(**arrays)
        arrays["bias"] = adapter.b.detach().cpu().numpy().copy()
        output[source] = arrays
    return output


@dataclass(frozen=True)
class BoundInputs:
    train: Mapping[str, core.FeatureBatch]
    validation: Mapping[str, core.FeatureBatch]
    full_linear_initialization: Mapping[str, Mapping[str, np.ndarray]]
    provenance: Mapping[str, Any]

    def split(self, name: str) -> Mapping[str, core.FeatureBatch]:
        if name == "train":
            return self.train
        if name == "validation":
            return self.validation
        _fail("Test or unknown split is not authorized")


def _bind(
    repository_root: str | Path, t001d_run_root: str | Path, t002d_validation_root: str | Path,
    *, counts: tuple[int, int], preflight_fn: Callable[..., Mapping[str, Any]],
    bank_loader: Callable[[Path], Mapping[str, Any]],
    fixture_hashes: Mapping[str, Any] | None = None,
) -> BoundInputs:
    binding = load_binding(repository_root)
    run_root = _absolute_root(t001d_run_root, "T001D_RUN_ROOT")
    validation_root = _absolute_root(t002d_validation_root, "T002D_VALIDATION_ROOT")
    try:
        plan, plan_id = t001d.load_bound_plan(run_root)
    except (OSError, ValueError, t001d.HarnessError) as error:
        raise RuntimeBindingError("bound T001D plan cannot be validated") from error
    accepted = binding["t001d"]
    expected_plan = accepted["t001d_plan_id"] if fixture_hashes is None else fixture_hashes["plan"]
    expected_manifest = accepted["manifest_sha256"] if fixture_hashes is None else fixture_hashes["manifest"]
    if plan_id != expected_plan or plan.get("manifest_sha256") != expected_manifest:
        _fail("T001D plan or manifest identity mismatch")
    split_counts = plan.get("manifest_split_counts")
    if not isinstance(split_counts, Mapping) or tuple(split_counts.get(name) for name in ("train", "validation")) != counts:
        _fail("T001D Train/Validation count mismatch")
    bank_binding = validation.load_bank_binding(Path(repository_root).resolve() / BANK_PATH)
    adapter_config = Path(repository_root).resolve() / bank_binding["adapter_config_path"]
    if fixture_hashes is None and _sha(adapter_config) != bank_binding["adapter_config_sha256"]:
        _fail("accepted predecessor adapter config SHA-256 mismatch")
    # The existing preflight revalidates current units and recomputes the
    # three bank identities; the accepted historical report SHA alone does not.
    report = preflight_fn(adapter_config, run_root)
    validation.verify_preflight_binding(report, bank_binding,
                                       adapter_config_sha256=bank_binding["adapter_config_sha256"])
    banks = bank_loader(run_root)
    target = banks.get("TARGET_CLIP")
    if not isinstance(target, Mapping):
        _fail("target bank is unavailable")
    result_path = validation_root / "result.json"
    result_hash = binding["t002d"]["validation_result_sha256"] if fixture_hashes is None else fixture_hashes["result"]
    if _sha(result_path) != result_hash:
        _fail("T002D Validation result SHA-256 mismatch")
    result = _json(result_path)
    expected_runtime = binding["t002d"]["validation_runtime_identity_sha256"]
    if result.get("runtime_identity_sha256") != expected_runtime:
        _fail("T002D Validation runtime identity mismatch")
    if any(type(result.get(key)) is not int or result[key] != 0 for key in (
        "test_scientific_access", "test_metric_count", "generation_call_count"
    )):
        _fail("T002D Validation recorded Test or generation activity")
    if result.get("t001d_plan_id") != accepted["t001d_plan_id"]:
        _fail("T002D Validation result belongs to a different T001D plan")
    hashes = {name: binding["t002d"]["full_linear"][name]["sha256"] for name in core.SOURCES}
    if fixture_hashes is not None:
        hashes = fixture_hashes["full_linear"]
    initialization = _load_full_linear(validation_root, result, binding, hashes)
    design = core.validate_config(_json(Path(repository_root).resolve() / DESIGN_PATH))
    partitions: dict[str, dict[str, core.FeatureBatch]] = {"train": {}, "validation": {}}
    for split, count in zip(("train", "validation"), counts):
        ids = target["ids"][split]
        if len(ids) != count or len(set(ids)) != count:
            _fail("target bank image IDs/count mismatch")
        if target[split].shape != (count, 196, 768) or target["connectors"][split].shape != (count, 196, 1536):
            _fail("target bank tensor shape mismatch")
        for source in core.SOURCES:
            bank = banks[source]
            if bank["ids"][split] != ids or bank[split].shape != (count, 196, 768):
                _fail(f"{source} {split} image ID order or tensor shape mismatch")
            batch = core.FeatureBatch(
                torch.from_numpy(np.ascontiguousarray(bank[split])),
                torch.from_numpy(np.ascontiguousarray(target[split])),
                torch.from_numpy(np.ascontiguousarray(target["connectors"][split])),
                tuple(ids), tuple(ids), tuple(ids), split,
                synthetic_fixture=fixture_hashes is not None,
            )
            core.validate_batch(batch, design)
            partitions[split][source] = batch
    for source in core.SOURCES:
        core.validate_train_validation_pair(partitions["train"][source], partitions["validation"][source], design)
    return BoundInputs(partitions["train"], partitions["validation"], initialization,
                       {"binding": binding, "bank_preflight_status": report["status"], "policy": DENIED_POLICY})


def preflight_runtime(repository_root: str | Path, t001d_run_root: str | Path,
                      t002d_validation_root: str | Path) -> BoundInputs:
    """Production entry point; all identities/counts are frozen and no paths are inferred."""
    return _bind(repository_root, t001d_run_root, t002d_validation_root,
                 counts=PRODUCTION_COUNTS,
                 preflight_fn=bank_preflight.preflight_query_adapter_banks,
                 bank_loader=validation.load_train_validation_banks)


def _bind_synthetic_fixture_for_tests(
    repository_root: str | Path, t001d_run_root: str | Path, t002d_validation_root: str | Path,
    *, expected_counts: tuple[int, int], expected_hashes: Mapping[str, Any],
    synthetic_preflight: Callable[..., Mapping[str, Any]],
    synthetic_bank_loader: Callable[[Path], Mapping[str, Any]],
) -> BoundInputs:
    """Explicit offline fixture seam; production callers use preflight_runtime only."""
    return _bind(repository_root, t001d_run_root, t002d_validation_root,
                 counts=expected_counts, preflight_fn=synthetic_preflight,
                 bank_loader=synthetic_bank_loader, fixture_hashes=expected_hashes)
