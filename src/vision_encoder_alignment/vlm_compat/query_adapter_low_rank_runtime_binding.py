"""Identity-checked, read-only inputs for the frozen low-rank Train runtime.

``bind_low_rank_runtime`` takes three explicit existing absolute directories. It
does not fit or select an adapter. Test-only units are hashed as opaque artifact
bytes for lineage, never opened as NumPy payloads. A unit mixing Train and Test
is refused before its patch array can be decompressed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

import numpy as np

from vision_encoder_alignment.vlm_compat import query_adapter_bank_preflight as preflight
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


LOW_RANK_PATH = Path("configs/vlm_compat/p2_query_adapter_low_rank_linear_v1.json")
LINEAR_PATH = Path("configs/vlm_compat/p2_query_adapter_linear_v1.json")
BANK_PATH = Path("configs/vlm_compat/p2_query_adapter_linear_bank_binding_v1.json")
TEST_BINDING_PATH = Path("configs/vlm_compat/p2_query_adapter_linear_test_binding_v1.json")
TRACKED_SHA = {
    LINEAR_PATH: "28835f065f9a00d756316ced0a1d5ad3e74a3709d88a83abc47ddbe1d6f142a1",
    BANK_PATH: "d2742c93f8eeae286d490fc229ff57fa8cd6701aab81dadde656ff23433129be",
    TEST_BINDING_PATH: "7688f7f14bdf6058f314e8b23ab60cc7ce9ca59dc48df4844b328ed7844662d3",
}
SOURCE_ORDER = ("SIGLIP2", "DINOV3")
BANK_ORDER = (*SOURCE_ORDER, "TARGET_CLIP")
SCHEMA_ID = "vea.p2.query_adapter_low_rank_runtime_binding.v1"


class LowRankRuntimeBindingError(RuntimeError):
    """A stable fail-closed identity, payload, or access rejection."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def _fail(code: str, message: str) -> None:
    raise LowRankRuntimeBindingError(code, message)


def _root(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        _fail("BLOCKED_ROOT", f"{label} must be an existing explicit absolute directory")
    return path.resolve()


def _tracked_json(root: Path, relative: Path, expected: str) -> dict[str, Any]:
    try:
        raw = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            _fail("BLOCKED_TRACKED_SHA", f"accepted tracked-file SHA drift: {relative}")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_TRACKED_SHA", f"accepted tracked file unavailable: {relative}") from error
    if not isinstance(value, dict):
        _fail("BLOCKED_TRACKED_SHA", f"accepted tracked JSON is not an object: {relative}")
    return value


def _artifact(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _fail("BLOCKED_UNSAFE_PATH", f"{label} must be a nonempty POSIX relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or ":" in relative:
        _fail("BLOCKED_UNSAFE_PATH", f"{label} contains an unsafe path component")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        _fail("BLOCKED_UNSAFE_PATH", f"{label} resolves outside its root")
    if not path.is_file():
        _fail("BLOCKED_ARTIFACT_MISSING", f"{label} is missing")
    return path


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise LowRankRuntimeBindingError("BLOCKED_ARTIFACT_MISSING", f"cannot hash {path.name}") from error
    return digest.hexdigest()


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_RESULT", f"cannot read {path.name}") from error
    if not isinstance(value, dict):
        _fail("BLOCKED_RESULT", f"{path.name} is not a JSON object")
    return value


def _array(archive: Any, key: str, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        value = np.array(archive[key], copy=True, order="C")
    except (KeyError, ValueError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_SHAPE", f"{label} {key} missing or unreadable") from error
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        _fail("BLOCKED_SHAPE", f"{label} {key} shape, float32 dtype, or finiteness mismatch")
    return value


@dataclass(frozen=True)
class AcceptedOLS:
    weights: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    artifact_sha256: str


@dataclass(frozen=True)
class _Unit:
    bank: str
    image_ids: tuple[str, ...]
    splits: tuple[str, ...]
    path: Path
    sha256: str
    plan_id: str


class TrainChunkFactory:
    """Re-iterable paired Train stream; each pass checks prior chunk bytes."""

    def __init__(self, source: str, source_units: tuple[_Unit, ...], target_units: tuple[_Unit, ...],
                 train_ids: tuple[str, ...], patch_shape: tuple[int, int]) -> None:
        self.source = source
        self._source_units = source_units
        self._target_units = target_units
        self._train_ids = train_ids
        self._patch_shape = patch_shape
        self._first_pass: tuple[str, ...] | None = None

    def __call__(self) -> Iterator[core.TrainChunk]:
        return self._iterate()

    def _iterate(self) -> Iterator[core.TrainChunk]:
        source_rows = self._rows(self._source_units)
        target_rows = self._rows(self._target_units)
        seen: list[str] = []
        ids_seen: list[str] = []
        max_images = max(1, core.MAX_CHUNK_ROWS // self._patch_shape[0])
        pending_source: list[np.ndarray] = []
        pending_target: list[np.ndarray] = []
        pending_ids: list[str] = []
        for source_item, target_item in zip(source_rows, target_rows, strict=True):
            source_id, source_value = source_item
            target_id, target_value = target_item
            if source_id != target_id:
                _fail("BLOCKED_IMAGE_ORDER", f"{self.source} Train source/target image order differs")
            ids_seen.append(source_id)
            pending_ids.append(source_id)
            pending_source.append(source_value)
            pending_target.append(target_value)
            if len(pending_ids) == max_images:
                chunk, digest = self._chunk(pending_source, pending_target, pending_ids)
                self._check_replay(len(seen), digest)
                seen.append(digest)
                yield chunk
                pending_source, pending_target, pending_ids = [], [], []
        if pending_ids:
            chunk, digest = self._chunk(pending_source, pending_target, pending_ids)
            self._check_replay(len(seen), digest)
            seen.append(digest)
            yield chunk
        if tuple(ids_seen) != self._train_ids:
            _fail("BLOCKED_SPLIT", f"{self.source} Train stream count/order differs from manifest")
        if self._first_pass is None:
            self._first_pass = tuple(seen)
        elif tuple(seen) != self._first_pass:
            _fail("BLOCKED_STREAM_REPLAY", f"{self.source} Train stream pass length changed")

    def _check_replay(self, index: int, digest: str) -> None:
        if self._first_pass is not None and (index >= len(self._first_pass) or digest != self._first_pass[index]):
            _fail("BLOCKED_STREAM_REPLAY", f"{self.source} Train chunk {index} changed across passes")

    def _chunk(self, source: list[np.ndarray], target: list[np.ndarray], ids: list[str]) -> tuple[core.TrainChunk, str]:
        s = np.stack(source)
        t = np.stack(target)
        identity = json.dumps(ids, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(identity + s.tobytes() + t.tobytes()).hexdigest()
        return core.TrainChunk(s, t, tuple(ids), tuple(ids), "train"), digest

    def _rows(self, units: tuple[_Unit, ...]) -> Iterator[tuple[str, np.ndarray]]:
        for unit in units:
            if "train" not in unit.splits:
                continue
            if "test" in unit.splits:
                _fail("BLOCKED_TEST_GUARD", f"{unit.bank} unit mixes Train and Test; patch payload cannot be isolated")
            if _sha_file(unit.path) != unit.sha256:
                _fail("BLOCKED_ARTIFACT_HASH", f"{unit.bank} Train unit changed after binding")
            try:
                with np.load(unit.path, allow_pickle=False) as archive:
                    if set(archive.files) != (preflight.TARGET_FIELDS if unit.bank == "TARGET_CLIP" else preflight.SOURCE_FIELDS):
                        _fail("BLOCKED_SHAPE", f"{unit.bank} unit fields differ")
                    raw_ids = archive["image_ids"]
                    if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "US" or tuple(map(str, raw_ids.tolist())) != unit.image_ids:
                        _fail("BLOCKED_IMAGE_ORDER", f"{unit.bank} unit image IDs differ from plan")
                    raw_plan = archive["plan_id"]
                    if raw_plan.shape != () or raw_plan.dtype.kind not in "US" or str(raw_plan.item()) != unit.plan_id:
                        _fail("BLOCKED_PLAN", f"{unit.bank} persisted plan identity differs")
                    patches = _array(archive, "patches", (len(unit.image_ids), *self._patch_shape), unit.bank)
            except LowRankRuntimeBindingError:
                raise
            except (OSError, ValueError, EOFError, KeyError) as error:
                raise LowRankRuntimeBindingError("BLOCKED_SHAPE", f"{unit.bank} Train unit cannot be safely read") from error
            if _sha_file(unit.path) != unit.sha256:
                _fail("BLOCKED_ARTIFACT_HASH", f"{unit.bank} Train unit changed during payload read")
            for index, (image_id, split) in enumerate(zip(unit.image_ids, unit.splits, strict=True)):
                if split == "train":
                    yield image_id, patches[index].copy()


@dataclass(frozen=True)
class BoundLowRankInputs:
    config: Mapping[str, Any]
    ols: Mapping[str, AcceptedOLS]
    train_factories: Mapping[str, TrainChunkFactory]
    provenance: Mapping[str, Any]

    def train_factory(self, source: str) -> TrainChunkFactory:
        if source not in SOURCE_ORDER:
            _fail("BLOCKED_TEST_GUARD", "only the two frozen Train sources are available")
        return self.train_factories[source]

    def split(self, name: str) -> None:
        _fail("BLOCKED_TEST_GUARD", f"split access is not provided by this Train-only binding: {name}")


def _lineage(repository_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        config = core.load_low_rank_config(repository_root / LOW_RANK_PATH)
    except core.LowRankLinearError as error:
        raise LowRankRuntimeBindingError("BLOCKED_CONFIG", str(error)) from error
    linear = _tracked_json(repository_root, LINEAR_PATH, TRACKED_SHA[LINEAR_PATH])
    bank = _tracked_json(repository_root, BANK_PATH, TRACKED_SHA[BANK_PATH])
    test = _tracked_json(repository_root, TEST_BINDING_PATH, TRACKED_SHA[TEST_BINDING_PATH])
    provenance = config["provenance"]
    if (provenance["predecessor_linear_config"]["git_blob_sha256"] != TRACKED_SHA[LINEAR_PATH]
            or provenance["accepted_linear_test_binding"]["git_blob_sha256"] != TRACKED_SHA[TEST_BINDING_PATH]
            or bank.get("adapter_config_sha256") != TRACKED_SHA[LINEAR_PATH]
            or bank.get("adapter_config_path") != str(LINEAR_PATH).replace("\\", "/")
            or test.get("bank_binding_config_path") != str(BANK_PATH).replace("\\", "/")
            or test.get("validation_result_sha256") != provenance["accepted_linear_test_binding"]["validation_result_sha256"]
            or test.get("validation_runtime_identity_sha256") != provenance["accepted_linear_test_binding"]["validation_runtime_identity_sha256"]
            or bank.get("manifest_sha256") != config["split"]["manifest_sha256"]
            or linear.get("manifest", {}).get("split_identity_sha256") != config["split"]["split_identity_sha256"]
            or bank.get("t001d_plan_id") is None):
        _fail("BLOCKED_LINEAGE", "T004A, predecessor, bank binding, and Validation identities differ")
    for source in SOURCE_ORDER:
        spec = test.get("finalized_adapter_artifacts", {}).get(source, {}).get("FULL_LINEAR", {})
        if (spec.get("sha256") != config["ols_control"]["exact_artifact_sha256_by_source"][source]
                or spec.get("relative_path") != config["ols_control"]["relative_artifact_path_by_source"][source]):
            _fail("BLOCKED_LINEAGE", f"{source} accepted Full Linear provenance differs")
        for key in ("model_id", "revision"):
            if linear["sources"][source][key] != config["sources"][source][key]:
                _fail("BLOCKED_LINEAGE", f"{source} model identity differs")
        if linear["sources"][source]["tap_name"] != config["sources"][source]["tap_name"]:
            _fail("BLOCKED_LINEAGE", f"{source} tap identity differs")
    for key in ("model_id", "revision", "tap_name"):
        if linear["target"][key] != config["target"][key]:
            _fail("BLOCKED_LINEAGE", "target CLIP identity differs")
    for key in ("model_id", "revision", "connector_input_shape", "connector_output_shape"):
        if linear["target_vlm"][key] != config["target_vlm"][key]:
            _fail("BLOCKED_LINEAGE", "TinyLLaVA identity differs")
    return config, linear, bank, test


def _ols(validation_root: Path, test: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, AcceptedOLS]:
    result_path = _artifact(validation_root, "result.json", "T002D Validation result")
    if _sha_file(result_path) != expected["result"]:
        _fail("BLOCKED_RESULT_HASH", "T002D Validation result SHA differs from accepted identity")
    result = _json_file(result_path)
    if (result.get("schema_id") != "vea.p2.query_adapter_linear_validation_runtime.v1"
            or result.get("status") != "PASS_QUERY_ADAPTER_LINEAR_VALIDATION"
            or result.get("runtime_identity_sha256") != test["validation_runtime_identity_sha256"]
            or result.get("t001d_plan_id") != expected["plan"]
            or any(type(result.get(key)) is not int or result[key] != 0 for key in
                   ("test_scientific_access", "test_metric_count", "generation_call_count", "connector_vision_call_count"))):
        _fail("BLOCKED_RESULT", "T002D Validation result status, runtime, plan, or access counters differ")
    manifest = result.get("fit_artifacts")
    if not isinstance(manifest, dict):
        _fail("BLOCKED_RESULT", "T002D fit-artifact manifest missing")
    output: dict[str, AcceptedOLS] = {}
    for source in SOURCE_ORDER:
        spec = test["finalized_adapter_artifacts"][source]["FULL_LINEAR"]
        relative = spec["relative_path"]
        sha = expected["ols"][source]
        path = _artifact(validation_root, relative, f"{source} Full Linear")
        if manifest.get(relative) != sha or _sha_file(path) != sha:
            _fail("BLOCKED_OLS_HASH", f"{source} Full Linear manifest or artifact SHA differs")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"runtime_identity", "weights", "source_mean", "target_mean"}:
                    _fail("BLOCKED_SHAPE", f"{source} Full Linear fields differ")
                identity = archive["runtime_identity"]
                if identity.shape != () or identity.dtype.kind not in "US" or str(identity.item()) != test["validation_runtime_identity_sha256"]:
                    _fail("BLOCKED_OLS_IDENTITY", f"{source} Full Linear persisted runtime identity differs")
                output[source] = AcceptedOLS(
                    _array(archive, "weights", (768, 768), source),
                    _array(archive, "source_mean", (768,), source),
                    _array(archive, "target_mean", (768,), source), sha,
                )
        except LowRankRuntimeBindingError:
            raise
        except (OSError, ValueError, EOFError, KeyError) as error:
            raise LowRankRuntimeBindingError("BLOCKED_SHAPE", f"{source} Full Linear cannot be safely read") from error
    return output


def _banks(run_root: Path, plan: Mapping[str, Any], plan_id: str, linear: Mapping[str, Any],
           patch_shape: tuple[int, int], expected: Mapping[str, Any]
           ) -> tuple[dict[str, tuple[_Unit, ...]], tuple[str, ...]]:
    try:
        rows, counts = preflight._validate_plan(plan, linear, preflight.PRODUCTION_EXPECTATIONS)
        identities = preflight._validate_identities(plan, linear)
    except preflight.BankPreflightError as error:
        code = ("BLOCKED_SPLIT" if "SPLIT" in error.classification else
                "BLOCKED_IMAGE_ORDER" if "IMAGE_ORDER" in error.classification else
                "BLOCKED_IDENTITY" if "MODEL_IDENTITY" in error.classification else "BLOCKED_PLAN")
        raise LowRankRuntimeBindingError(code, f"T001D plan/split/model identity invalid: {error}") from error
    except (KeyError, TypeError, ValueError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_PLAN", f"T001D plan/split/model identity invalid: {error}") from error
    if counts != {"train": 800, "validation": 100, "test": 100}:
        _fail("BLOCKED_SPLIT", "T001D partition counts differ")
    split_by_id = {row["image_id"]: row["split"] for row in rows}
    manifest_ids = tuple(row["image_id"] for row in rows)
    train_ids = tuple(row["image_id"] for row in rows if row["split"] == "train")
    all_units: dict[str, tuple[_Unit, ...]] = {}
    for bank in BANK_ORDER:
        try:
            planned = preflight._ordered_units(plan, bank)
        except preflight.BankPreflightError as error:
            raise LowRankRuntimeBindingError("BLOCKED_PLAN", str(error)) from error
        units: list[_Unit] = []
        ids: list[str] = []
        digest_units: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        seen_unit_ids: set[str] = set()
        for unit in planned:
            unit_ids = unit.get("image_ids")
            relative = unit.get("artifact_relative_path")
            uid = unit.get("unit_id")
            if (unit.get("encoder") != bank or not isinstance(unit_ids, list) or not unit_ids
                    or not isinstance(uid, str) or uid in seen_unit_ids
                    or not isinstance(relative, str) or relative in seen_paths):
                _fail("BLOCKED_PLAN", f"{bank} unit identity/IDs invalid or duplicate")
            seen_paths.add(relative)
            seen_unit_ids.add(uid)
            if any(image_id not in split_by_id for image_id in unit_ids):
                _fail("BLOCKED_SPLIT", f"{bank} unit contains an unapproved image ID")
            splits = tuple(split_by_id[image_id] for image_id in unit_ids)
            if "train" in splits and "test" in splits:
                _fail("BLOCKED_TEST_GUARD", f"{bank} unit mixes Train and Test")
            n = len(unit_ids)
            if "train" in splits and n * patch_shape[0] > core.MAX_CHUNK_ROWS:
                _fail("BLOCKED_CHUNK_CAP", f"{bank} Train-bearing NPZ unit exceeds 4096 patch rows")
            shape = {"patches": [n, *patch_shape], "globals": [n, patch_shape[1]]}
            if bank == "TARGET_CLIP":
                shape["connectors"] = [n, 196, 1536]
            if unit.get("shape") != shape or unit.get("dtype") != {key: "float32" for key in shape}:
                _fail("BLOCKED_SHAPE", f"{bank} unit shape/dtype metadata differs")
            path = _artifact(run_root, relative, f"{bank} unit")
            sha = _sha_file(path)  # Opaque bytes only, including Test-only units.
            units.append(_Unit(bank, tuple(unit_ids), splits, path, sha, plan_id))
            ids.extend(unit_ids)
            digest_units.append({"unit_id": uid, "artifact_relative_path": relative, "artifact_sha256": sha})
        if tuple(ids) != manifest_ids or len(set(ids)) != 1000:
            _fail("BLOCKED_IMAGE_ORDER", f"{bank} image order differs from approved manifest")
        digest_material = {
            "bank": bank, "encoder_identity": identities[bank], "t001d_plan_id": plan_id,
            "manifest_sha256": plan["manifest_sha256"], "ordered_units": digest_units,
            "ordered_image_ids": ids, "patch_shape": list(patch_shape), "patch_dtype": "float32",
            "connector_shape": [196, 1536] if bank == "TARGET_CLIP" else None,
            "connector_dtype": "float32" if bank == "TARGET_CLIP" else None,
        }
        digest = hashlib.sha256(t001d.canonical_json_bytes(digest_material)).hexdigest()
        if digest != expected["banks"][bank]:
            _fail("BLOCKED_BANK_HASH", f"{bank} bank identity differs from accepted binding")
        all_units[bank] = tuple(units)
    return all_units, train_ids


def _bind(repository_root: str | Path, t001d_run_root: str | Path, t002d_validation_root: str | Path,
          *, fixture: Mapping[str, Any] | None = None) -> BoundLowRankInputs:
    repository = _root(repository_root, "repository_root")
    run_root = _root(t001d_run_root, "t001d_run_root")
    validation_root = _root(t002d_validation_root, "t002d_validation_root")
    config, linear, bank, test = _lineage(repository)
    try:
        plan, plan_id = t001d.load_bound_plan(run_root)
    except (OSError, ValueError, TypeError, t001d.HarnessError) as error:
        raise LowRankRuntimeBindingError("BLOCKED_PLAN", "T001D bound plan unavailable") from error
    expected = {
        "plan": bank["t001d_plan_id"], "manifest": bank["manifest_sha256"], "banks": bank["banks"],
        "result": test["validation_result_sha256"],
        "ols": {source: test["finalized_adapter_artifacts"][source]["FULL_LINEAR"]["sha256"] for source in SOURCE_ORDER},
    }
    if fixture is not None:
        expected = fixture["expected"]
    if plan_id != expected["plan"] or plan.get("manifest_sha256") != expected["manifest"]:
        _fail("BLOCKED_PLAN", "T001D plan or manifest SHA differs from accepted binding")
    patch_shape = (196, 768) if fixture is None else tuple(fixture["patch_shape"])
    units, train_ids = _banks(run_root, plan, plan_id, linear, patch_shape, expected)
    ols = _ols(validation_root, test, expected)
    factories = {source: TrainChunkFactory(source, units[source], units["TARGET_CLIP"], train_ids, patch_shape)
                 for source in SOURCE_ORDER}
    return BoundLowRankInputs(config, ols, factories, {
        "schema_id": SCHEMA_ID, "t001d_plan_id": plan_id,
        "manifest_sha256": expected["manifest"], "bank_identity_sha256": dict(expected["banks"]),
        "validation_result_sha256": expected["result"],
        "validation_runtime_identity_sha256": test["validation_runtime_identity_sha256"],
        "ols_artifact_sha256": dict(expected["ols"]),
        "train_image_count": len(train_ids), "test_scientific_access": 0,
        "test_payload_open_count": 0, "test_only_units_hashed_as_opaque_bytes": True,
        "float32_latency_evidence": False,
        "numeric_note": "T004B apply_low_rank accumulates float64 and returns float32; no float32 adapter-only latency was measured.",
    })


def bind_low_rank_runtime(repository_root: str | Path, t001d_run_root: str | Path,
                          t002d_validation_root: str | Path) -> BoundLowRankInputs:
    """Bind accepted real identities without fitting, evaluating, or opening Test tensors."""
    return _bind(repository_root, t001d_run_root, t002d_validation_root)


def _bind_synthetic_fixture_for_tests(repository_root: str | Path, t001d_run_root: str | Path,
                                      t002d_validation_root: str | Path, *,
                                      fixture: Mapping[str, Any]) -> BoundLowRankInputs:
    """Private offline seam: fixture hashes/geometry cannot reach the public entry point."""
    return _bind(repository_root, t001d_run_root, t002d_validation_root, fixture=fixture)
