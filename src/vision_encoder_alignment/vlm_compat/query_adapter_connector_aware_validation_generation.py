"""Validation-only frozen generation for exact, finalized T003E affine lanes."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_runtime_binding as binding_module
from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_validation_runtime as validation_runtime
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as transport


CONFIG_PATH = Path("configs/vlm_compat/p2_connector_aware_validation_generation_v1.json")
CONFIG_SHA256 = "26fa1a597a4895ec6f0883f2e8bea5e5795bf4f189eb86a12aa6512e7d0cffde"
CANONICAL_CONFIG_SHA256 = "85b1f32a397bf54fff9808d378050fc83cbf3123678a4dc536698a6fa124147b"
SCHEMA_ID = "vea.p2.connector_aware_validation_generation.v1"
ROW_SCHEMA_ID = "vea.p2.connector_aware_validation_generation_row.v1"
RESULT_SCHEMA_ID = "vea.p2.connector_aware_validation_generation_result.v1"
T003E_RESULT_SHA256 = "2147b41ec525f387c0cc8d434d6b3cd748ed224ef39053fc9c6a9e7372963a10"
T003E_RUNTIME_IDENTITY = "6e0e274a47813101e560d8f8dc1dbf40ad416b89abd7e6f6d1688646f4b466ca"
LANE_ORDER = (
    "SIGLIP2/OLS_REFERENCE", "SIGLIP2/CONNECTOR_AWARE_SELECTED",
    "DINOV3/OLS_REFERENCE", "DINOV3/CONNECTOR_AWARE_SELECTED",
)
GENERATION_CONTRACT = {
    "prompt": "What is shown in this image? Answer briefly.",
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 32,
    "native_target_reference": True,
    "fit_authorized": False,
    "selection_authorized": False,
}


class ValidationGenerationError(RuntimeError):
    """A frozen identity, Validation boundary, or publication contract failed."""


def _fail(message: str) -> None:
    raise ValidationGenerationError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationGenerationError("generation value is not finite JSON") from error


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValidationGenerationError(f"required file unavailable: {path.name}") from error


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationGenerationError(f"JSON unavailable: {path.name}") from error
    if not isinstance(value, dict):
        _fail(f"JSON object required: {path.name}")
    return value


def load_config(repository_root: str | Path) -> dict[str, Any]:
    path = Path(repository_root).resolve() / CONFIG_PATH
    if _sha_file(path) != CONFIG_SHA256:
        _fail("T003F generation config SHA-256 mismatch")
    value = _json(path)
    if _sha_bytes(_canonical_bytes(value)) != CANONICAL_CONFIG_SHA256:
        _fail("T003F generation config canonical identity mismatch")
    if (value.get("schema_id") != SCHEMA_ID or value.get("experiment_id") != core.EXPERIMENT_ID
            or value.get("study_role") != "development_validation_only_not_confirmatory_test"
            or value.get("t003a_config_sha256") != core.CONFIG_SHA256
            or value.get("t003c_binding_sha256") != validation_runtime.BINDING_SHA256
            or value.get("t003e", {}).get("result_sha256") != T003E_RESULT_SHA256
            or value.get("t003e", {}).get("runtime_identity_sha256") != T003E_RUNTIME_IDENTITY
            or value.get("t003e", {}).get("device") != "cuda:0"
            or value.get("validation") != {"image_count": 100, "split": "validation", "test_access_authorized": False}
            or value.get("generation") != GENERATION_CONTRACT
            or value.get("generation_plan") != {"native": 100, "adapted": 400, "total": 500}):
        _fail("T003F frozen configuration contract differs")
    lanes = value.get("lanes")
    if not isinstance(lanes, dict) or tuple(lanes) != LANE_ORDER:
        _fail("T003F lane order differs")
    for key, lane in lanes.items():
        if not isinstance(lane, dict) or lane.get("source") not in core.SOURCES or lane.get("lane") not in (
            "OLS_REFERENCE", "CONNECTOR_AWARE_SELECTED"
        ) or lane.get("artifact_condition") not in core.CONDITIONS[1:] or lane.get("selected_epoch") not in range(9):
            _fail(f"T003F lane contract differs: {key}")
        digest = lane.get("affine_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            _fail(f"T003F affine identity is invalid: {key}")
    if lanes["DINOV3/OLS_REFERENCE"]["affine_sha256"] != lanes["DINOV3/CONNECTOR_AWARE_SELECTED"]["affine_sha256"]:
        _fail("DINOV3 duplicate selected-artifact identity differs")
    return value


class GeneratorProtocol(Protocol):
    device: str
    native_calls: int
    adapted_calls: int

    def generate_native(self, image_path: str | Path) -> Any: ...
    def generate_adapted(self, image_path: str | Path, patch: np.ndarray) -> Any: ...


@dataclass(frozen=True)
class LoadedLane:
    key: str
    source: str
    lane: str
    condition: str
    selected_epoch: int
    affine_sha256: str
    adapter: core.AffineAdapter


@dataclass(frozen=True)
class ValidationImage:
    path: Path
    relative_path: str


def _safe_relative(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        _fail("selected artifact path escapes T003E root")
    return candidate


def verify_t003e_selected_artifacts(config: Mapping[str, Any], t003e_root: str | Path) -> dict[str, LoadedLane]:
    """Read and verify only T003E result plus its finalized selected artifacts."""
    root = Path(t003e_root)
    if not root.is_absolute() or not root.is_dir():
        _fail("T003E output root must be an explicit existing absolute directory")
    result_path = root / "result.json"
    if _sha_file(result_path) != config["t003e"]["result_sha256"]:
        _fail("T003E result SHA-256 mismatch")
    result = _json(result_path)
    if result.get("runtime_identity_sha256") != config["t003e"]["runtime_identity_sha256"]:
        _fail("T003E runtime identity mismatch")
    if any(result.get(key) != value for key, value in config["t003e"]["required_zero_counts"].items()):
        _fail("T003E result records forbidden Test, generation, or vision activity")
    loaded: dict[str, LoadedLane] = {}
    for key in LANE_ORDER:
        expected = config["lanes"][key]
        try:
            spec = result["sources"][expected["source"]]["selected_artifacts"][expected["artifact_condition"]]
            relative = spec["relative_path"]
        except (KeyError, TypeError) as error:
            raise ValidationGenerationError(f"T003E selected artifact is missing: {key}") from error
        if (spec.get("source") != expected["source"] or spec.get("condition") != expected["artifact_condition"]
                or spec.get("selected_epoch") != expected["selected_epoch"]
                or spec.get("affine_npz_sha256") != expected["affine_sha256"]):
            _fail(f"T003E selected artifact metadata differs: {key}")
        folder = _safe_relative(root, str(relative)).parent
        adapter, metadata = validation_runtime.load_selected_artifact(
            folder, runtime_identity=config["t003e"]["runtime_identity_sha256"]
        )
        if metadata.get("affine_npz_sha256") != expected["affine_sha256"]:
            _fail(f"T003E selected artifact bytes differ: {key}")
        loaded[key] = LoadedLane(key, expected["source"], expected["lane"], expected["artifact_condition"],
                                 expected["selected_epoch"], expected["affine_sha256"], adapter)
    if loaded["DINOV3/OLS_REFERENCE"].affine_sha256 != loaded["DINOV3/CONNECTOR_AWARE_SELECTED"].affine_sha256:
        _fail("DINOV3 duplicate artifact verification failed")
    return loaded


def load_validation_image_paths(
    t001d_run_root: str | Path, inputs: binding_module.BoundInputs
) -> list[ValidationImage]:
    """Resolve exactly the immutable Validation images; Test rows never enter output."""
    root = Path(t001d_run_root)
    if not root.is_absolute() or not root.is_dir():
        _fail("T001D run root must be an explicit existing absolute directory")
    try:
        plan, plan_id = transport.load_bound_plan(root)
        dataset_root = Path(plan["roots"]["dataset_root"]).resolve()
        rows = [row for row in plan["manifest_rows"] if row.get("split") == "validation"]
    except (KeyError, OSError, transport.HarnessError) as error:
        raise ValidationGenerationError("T001D Validation manifest is unavailable") from error
    ids = tuple(inputs.validation[core.SOURCES[0]].source_image_ids)
    if len(rows) != 100 or tuple(str(row.get("image_id")) for row in rows) != ids:
        _fail("T001D Validation image order/count differs from T003C-bound inputs")
    paths: list[ValidationImage] = []
    for row in rows:
        path = (dataset_root / str(row.get("relative_path"))).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError:
            _fail("Validation image path escapes immutable dataset root")
        if not path.is_file() or _sha_file(path) != row.get("source_sha256"):
            _fail("Validation image identity differs")
        paths.append(ValidationImage(path=path, relative_path=str(row["relative_path"])))
    return paths


def _atomic_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    if path.exists() or partial.exists():
        _fail(f"output collision: {path.name}")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(partial, path)
    except BaseException:
        raise


def _tokens(output: Any, *, adapted: bool) -> tuple[list[int], str, int, int, int, bool]:
    try:
        tokens = [int(value) for value in output.token_ids]
        text = str(output.decoded_text)
        vision, connector, injection = int(output.vision_calls), int(output.connector_calls), int(output.injection_calls)
        restored = bool(output.injection_restored)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValidationGenerationError("generation boundary output is malformed") from error
    if not tokens or not text or connector < 1 or not restored:
        _fail("generation output lacks required frozen-boundary evidence")
    if adapted and (vision != 0 or injection != 1):
        _fail("adapted generation did not bypass vision with one restored injection")
    if not adapted and (vision < 1 or injection != 0):
        _fail("native generation did not use target vision path")
    return tokens, text, vision, connector, injection, restored


def _lane_patch(lane: LoadedLane, batch: core.FeatureBatch, index: int, device: str) -> np.ndarray:
    adapter = lane.adapter.to(device)
    with torch.inference_mode():
        value = adapter(batch.source[index:index + 1].to(device)).detach().cpu().numpy()
    if value.shape != (1, 196, 768) or value.dtype != np.float32 or not np.isfinite(value).all():
        _fail("same-image adapted patch contract differs")
    return np.ascontiguousarray(value)


def _validate_validation_batch(batch: core.FeatureBatch, expected_ids: tuple[str, ...]) -> None:
    if (batch.split != "validation" or tuple(batch.source_image_ids) != expected_ids
            or tuple(batch.target_image_ids) != expected_ids
            or tuple(batch.connector_image_ids) != expected_ids):
        _fail("Test or mismatched image order requested")
    for value, shape in ((batch.source, (196, 768)), (batch.target, (196, 768)),
                         (batch.target_connector, (196, 1536))):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.ndim != 3 or tuple(value.shape[1:]) != shape
                or value.shape[0] != len(expected_ids) or not bool(torch.isfinite(value).all())):
            _fail("Validation tensor contract differs")


def _aggregate(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    native = {row["image_id"]: row for row in rows if row["lane"] == "NATIVE"}
    expected = config["validation"]["image_count"]
    if len(native) != expected:
        _fail("native Validation row count differs")
    lanes: dict[str, Any] = {}
    for key in LANE_ORDER:
        members = [row for row in rows if row["lane_key"] == key]
        if len(members) != expected:
            _fail(f"adapted Validation row count differs: {key}")
        lanes[key] = {
            "native_exact_match_count": sum(int(row["native_exact_match"]) for row in members),
            "mean_token_unigram_f1": float(sum(float(row["token_unigram_f1"]) for row in members) / expected),
            "generation_success_count": sum(int(row["generation_success"]) for row in members),
            "injection_restoration_count": sum(int(row["injection_restored"]) for row in members),
        }
    pairwise: dict[str, Any] = {}
    for source in core.SOURCES:
        control = {row["image_id"]: row for row in rows if row["lane_key"] == f"{source}/OLS_REFERENCE"}
        aware = {row["image_id"]: row for row in rows if row["lane_key"] == f"{source}/CONNECTOR_AWARE_SELECTED"}
        if set(control) != set(native) or set(aware) != set(native):
            _fail(f"within-source row identity differs: {source}")
        pairwise[source] = {
            "generated_text_equality_count": sum(control[key]["decoded_text"] == aware[key]["decoded_text"] for key in native),
            "token_equality_count": sum(control[key]["token_ids"] == aware[key]["token_ids"] for key in native),
            "duplicate_affine_identity": config["lanes"][f"{source}/OLS_REFERENCE"]["affine_sha256"]
                == config["lanes"][f"{source}/CONNECTOR_AWARE_SELECTED"]["affine_sha256"],
        }
    return {"per_lane": lanes, "within_source": pairwise}


def run_validation_generation(
    *, config: Mapping[str, Any], inputs: binding_module.BoundInputs, lanes: Mapping[str, LoadedLane],
    generator: GeneratorProtocol, image_paths: Sequence[str | Path | ValidationImage], output_root: str | Path,
    device: str, fake: bool = False,
) -> dict[str, Any]:
    """Generate native plus four frozen adapted lanes on Validation only and publish result last."""
    if config.get("schema_id") != SCHEMA_ID or tuple(config.get("lanes", {})) != LANE_ORDER:
        _fail("generation configuration is not frozen")
    if set(lanes) != set(LANE_ORDER) or set(inputs.validation) != set(core.SOURCES):
        _fail("frozen lane or Validation source set differs")
    if inputs.provenance.get("policy") != binding_module.DENIED_POLICY:
        _fail("T003C Test/generation policy drift")
    if not fake:
        device = validation_runtime.canonical_real_device(device)
        if inputs.provenance.get("synthetic_fixture") is True or inputs.provenance.get("bank_preflight_status") != binding_module.bank_preflight.SUCCESS_STATUS:
            _fail("real generation requires verified T003C production inputs")
    elif device != "cpu" or inputs.provenance.get("synthetic_fixture") is not True:
        _fail("fake generation requires explicit CPU synthetic inputs")
    if generator.device != device:
        _fail("generator device identity differs")
    if not fake:
        anchor = getattr(generator, "anchor_config", None)
        if any(getattr(anchor, key, object()) != value for key, value in GENERATION_CONTRACT.items()
               if key in {"prompt", "do_sample", "num_beams", "max_new_tokens"}):
            _fail("TinyLLaVA prompt/decoding contract differs")
    ids = tuple(inputs.validation[core.SOURCES[0]].source_image_ids)
    count = len(ids)
    if len(image_paths) != count or (not fake and count != config["validation"]["image_count"]):
        _fail("Validation image count/path contract differs")
    resolved_paths = [
        item if isinstance(item, ValidationImage) else ValidationImage(Path(item), Path(item).as_posix())
        for item in image_paths
    ]
    if len({item.relative_path for item in resolved_paths}) != count:
        _fail("Validation image path uniqueness differs")
    for source in core.SOURCES:
        batch = inputs.validation[source]
        _validate_validation_batch(batch, ids)
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        _fail("generation output root must be new or empty")
    identity = _sha_bytes(_canonical_bytes({
        "schema_id": RESULT_SCHEMA_ID, "config_sha256": CONFIG_SHA256,
        "t003e_result_sha256": config["t003e"]["result_sha256"],
        "t003e_runtime_identity_sha256": config["t003e"]["runtime_identity_sha256"],
        "validation_image_ids": list(ids), "fake": fake,
    }))
    rows: list[dict[str, Any]] = []
    for index, (image_id, image) in enumerate(zip(ids, resolved_paths, strict=True)):
        native = generator.generate_native(image.path)
        tokens, text, vision, connector, injection, restored = _tokens(native, adapted=False)
        rows.append({"schema_id": ROW_SCHEMA_ID, "runtime_identity_sha256": identity,
                     "t003e_result_sha256": config["t003e"]["result_sha256"],
                     "t003e_runtime_identity_sha256": config["t003e"]["runtime_identity_sha256"],
                     "image_id": image_id, "relative_path": image.relative_path, "lane": "NATIVE",
                     "lane_key": None, "source": "TARGET_CLIP", "method": "NATIVE", "artifact_sha256": None,
                     "prompt_decoding_identity": _sha_bytes(_canonical_bytes(config["generation"])),
                     "token_ids": tokens, "decoded_text": text, "generation_success": True,
                     "injection_restored": restored, "vision_calls": vision, "connector_calls": connector,
                     "injection_calls": injection})
        for key in LANE_ORDER:
            lane = lanes[key]
            patch = _lane_patch(lane, inputs.validation[lane.source], index, device)
            adapted = generator.generate_adapted(image.path, patch)
            atokens, atext, avision, aconnector, ainjection, arestored = _tokens(adapted, adapted=True)
            metric = transport.generation_metrics(atokens, tokens)
            rows.append({"schema_id": ROW_SCHEMA_ID, "runtime_identity_sha256": identity,
                         "t003e_result_sha256": config["t003e"]["result_sha256"],
                         "t003e_runtime_identity_sha256": config["t003e"]["runtime_identity_sha256"],
                         "image_id": image_id, "relative_path": image.relative_path, "lane": lane.lane,
                         "lane_key": key, "source": lane.source, "method": lane.condition,
                         "artifact_sha256": lane.affine_sha256,
                         "prompt_decoding_identity": _sha_bytes(_canonical_bytes(config["generation"])),
                         "token_ids": atokens, "decoded_text": atext, "generation_success": True,
                         "injection_restored": arestored, "vision_calls": avision, "connector_calls": aconnector,
                         "injection_calls": ainjection, "native_exact_match": int(metric["exact_match"]),
                         "token_unigram_f1": float(metric["token_unigram_F1"])})
    keys = [(row["image_id"], row["lane_key"] or "NATIVE") for row in rows]
    if len(set(keys)) != len(keys) or len(rows) != count * 5:
        _fail("generation row uniqueness/count contract differs")
    if generator.native_calls != count or generator.adapted_calls != count * 4:
        _fail("native/adapted generation boundary call counts differ")
    aggregate = _aggregate(rows, {**config, "validation": {**config["validation"], "image_count": count}} if fake else config)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_new(output / "rows.jsonl", b"".join(_canonical_bytes(row) + b"\n" for row in rows))
    _atomic_new(output / "aggregate.json", _canonical_bytes(aggregate))
    result = {"schema_id": RESULT_SCHEMA_ID, "status": "PASS_VALIDATION_GENERATION",
              "runtime_identity_sha256": identity, "validation_image_count": count,
              "generation_counts": {"native": count, "adapted": count * 4, "total": count * 5},
              "production_generation_plan": config["generation_plan"], "aggregate": aggregate,
              "test_access_count": 0, "test_metric_count": 0, "generation_is_validation_only": True,
              "t003e_result_sha256": config["t003e"]["result_sha256"],
              "t003e_runtime_identity_sha256": config["t003e"]["runtime_identity_sha256"]}
    _atomic_new(output / "result.json", _canonical_bytes(result))
    return result
