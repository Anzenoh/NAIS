"""Deterministic Train/Validation execution for the frozen connector-aware design.

The public runner consumes T003C-bound inputs and a frozen connector-only
boundary. It has no Test, vision, chat, or generation entry point.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_runtime_binding as binding_module
from vision_encoder_alignment.vlm_compat import query_adapter_linear as predecessor


SCHEMA_ID = "vea.p2.query_adapter_connector_aware_validation_runtime.v1"
ARTIFACT_SCHEMA_ID = "vea.p2.query_adapter_connector_aware_selected_affine.v1"
BINDING_SHA256 = "f0336604f4d4f37350d2745d8b54edeb1cc19f75fe27d5475609d81402946614"
CANONICAL_BINDING_SHA256 = "69ea2d95485fea1a5926271684245e1fec4a38f65499a62cf99b51746470601d"
CANONICAL_ANCHOR_SHA256 = "f63bc94d0722361fd9a4d913e56976e5cfc5c3dab613009fb4c808ef7fab6ad3"
ANCHOR_PATH = Path("configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json")
GRADIENT_CONDITIONS = core.CONDITIONS[1:]
METRIC_FIELDS = (
    "L_patch", "L_connector", "L_total", "patch_cosine", "patch_nmse", "patch_r2",
    "same_image_recall_at_1", "same_image_recall_at_5", "connector_cosine", "connector_nmse",
)


class ValidationError(RuntimeError):
    """A frozen runtime contract or publication check failed."""


def _fail(message: str) -> None:
    raise ValidationError(message)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError("runtime result is not finite JSON") from error


def load_accepted_inputs(repository_root: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read only the three T003D-named config paths and verify frozen hashes."""
    root = Path(repository_root).resolve()
    observed: list[dict[str, Any]] = []
    for relative, expected in (
        (core.CONFIG_PATH, core.CONFIG_SHA256),
        (binding_module.BINDING_PATH, BINDING_SHA256),
        (ANCHOR_PATH, "78242f4062fe748826aad0b51ae50ddce1c4bcbb06143396e87d9a862585ac8a"),
    ):
        try:
            payload = (root / relative).read_bytes()
            if _sha_bytes(payload) != expected:
                _fail(f"accepted config SHA-256 mismatch: {relative.name}")
            value = json.loads(payload)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValidationError(f"accepted config unavailable: {relative.name}") from error
        if not isinstance(value, dict):
            _fail(f"accepted config is not an object: {relative.name}")
        observed.append(value)
    design, bound_config, anchor = observed
    core.validate_config(design)
    if (bound_config.get("schema_id") != binding_module.SCHEMA_ID
            or bound_config.get("experiment_id") != core.EXPERIMENT_ID
            or bound_config.get("connector_aware_config_sha256") != core.CONFIG_SHA256
            or bound_config.get("policy") != binding_module.DENIED_POLICY):
        _fail("T003C binding identity or policy differs")
    if (anchor.get("schema_id") != "vea.p2.vlm_anchor.v1"
            or anchor.get("model_id") != design["connector_autograd_boundary"]["model_id"]
            or anchor.get("revision") != design["connector_autograd_boundary"]["revision"]
            or anchor.get("local_files_only") is not True):
        _fail("frozen TinyLLaVA anchor identity or local-only policy differs")
    return design, bound_config, anchor


class ConnectorOnly(Protocol):
    connector: Any
    device: str
    dtype: Any
    generation_calls: int
    vision_calls: int

    def apply(self, patches: np.ndarray, *, batch_size: int) -> np.ndarray: ...


def _connector_clean(connector: ConnectorOnly) -> None:
    if type(connector.generation_calls) is not int or connector.generation_calls != 0:
        _fail("generation calls must remain zero")
    if type(connector.vision_calls) is not int or connector.vision_calls != 0:
        _fail("vision calls must remain zero")


def _check_connector(connector: ConnectorOnly, device: str, fake: bool) -> core.FrozenDifferentiableConnector:
    if not isinstance(device, str) or not device or connector.device != device or connector.dtype != torch.float16:
        _fail("connector device or float16 contract mismatch")
    if not fake and device == "cpu":
        _fail("real connector-aware execution requires an explicit accelerator device")
    if not isinstance(connector.connector, torch.nn.Module):
        _fail("connector must be the loaded frozen Torch module")
    module = connector.connector
    if module.training or any(parameter.requires_grad for parameter in module.parameters()):
        _fail("connector must already be frozen in eval mode")
    for tensor in module.state_dict().values():
        if tensor.device != torch.device(device) or (tensor.is_floating_point() and tensor.dtype != torch.float16):
            _fail("connector state device or dtype mismatch")
    _connector_clean(connector)
    return core.FrozenDifferentiableConnector(module)


def _on_device(batch: core.FeatureBatch, device: str) -> core.FeatureBatch:
    return core.FeatureBatch(
        batch.source.to(device), batch.target.to(device), batch.target_connector.to(device),
        tuple(batch.source_image_ids), tuple(batch.target_image_ids), tuple(batch.connector_image_ids),
        batch.split, batch.synthetic_fixture,
    )


def _fresh_adapter(arrays: Mapping[str, np.ndarray], device: str) -> core.AffineAdapter:
    adapter = core.AffineAdapter.from_full_linear(arrays["weights"], arrays["source_mean"], arrays["target_mean"])
    if not np.array_equal(adapter.b.detach().cpu().numpy(), arrays["bias"]):
        _fail("T003C-derived Full Linear bias differs")
    return adapter.to(device)


def _zero_step_parity(adapter: core.AffineAdapter, batch: core.FeatureBatch,
                      boundary: core.FrozenDifferentiableConnector, inference: ConnectorOnly) -> None:
    candidate = adapter(batch.source[:1])
    training = boundary(candidate).detach().cpu().numpy()
    inferred = inference.apply(candidate.detach().cpu().numpy(), batch_size=1)
    if inferred.shape != training.shape or inferred.dtype != np.float32 or not np.isfinite(inferred).all():
        _fail("zero-step inference connector output contract mismatch")
    if not np.allclose(training, inferred, rtol=2e-3, atol=2e-3):
        _fail("zero-step training/inference connector parity failure")
    boundary.assert_unchanged()
    _connector_clean(inference)


def _finite_record(record: Mapping[str, Any]) -> None:
    for key in METRIC_FIELDS:
        value = record.get(key)
        if type(value) not in (float, int) or not np.isfinite(value):
            _fail(f"Validation metric missing or nonfinite: {key}")


def _evaluate(
    adapter: core.AffineAdapter, batch: core.FeatureBatch,
    boundary: core.FrozenDifferentiableConnector, normalizers: core.TrainNormalizers,
    *, source: str, condition: str, epoch: int, config: Mapping[str, Any],
    train_patch_mean: np.ndarray, train_connector_mean: np.ndarray,
) -> dict[str, Any]:
    if batch.split != "validation" or condition not in core.CONDITIONS or epoch not in range(9):
        _fail("Validation evaluation identity mismatch")
    loss_condition = "PATCH_ONLY_CONTROL" if condition == "OLS_REFERENCE" else condition
    weight = 0.0 if condition == "OLS_REFERENCE" else core.condition_weight(condition, config)
    chunks: list[np.ndarray] = []
    projected_chunks: list[np.ndarray] = []
    losses = {key: 0.0 for key in ("L_patch", "L_connector", "L_total")}
    width = config["gradient_training"]["batch_size_images"]
    with torch.enable_grad():
        for start in range(0, batch.source.shape[0], width):
            stop = min(start + width, batch.source.shape[0])
            candidate = adapter(batch.source[start:stop])
            selected = core.compute_losses(
                candidate, batch.target[start:stop], batch.target_connector[start:stop],
                boundary, normalizers, loss_condition, config,
            )
            projected = boundary(candidate)
            chunks.append(candidate.detach().cpu().numpy())
            projected_chunks.append(projected.detach().cpu().numpy())
            for key in losses:
                losses[key] += float(selected[key].detach().item()) * (stop - start)
    count = batch.source.shape[0]
    losses = {key: value / count for key, value in losses.items()}
    predicted = np.concatenate(chunks, axis=0)
    projected = np.concatenate(projected_chunks, axis=0)
    target = batch.target.detach().cpu().numpy()
    target_connector = batch.target_connector.detach().cpu().numpy()
    patch = predecessor.patch_metrics(predicted, target, train_patch_mean)
    retrieval = predecessor.same_image_retrieval(
        predicted, target, batch.source_image_ids, batch.target_image_ids,
    )
    connector = predecessor.patch_metrics(projected, target_connector, train_connector_mean)
    record = {
        "split": "validation", "source": source, "condition": condition, "epoch": epoch,
        "connector_loss_weight": weight, **losses,
        "patch_cosine": float(patch["cosine"]), "patch_nmse": float(patch["nmse"]),
        "patch_r2": float(patch["r2"]),
        "same_image_recall_at_1": float(retrieval["recall_at_1"]),
        "same_image_recall_at_5": float(retrieval["recall_at_5"]),
        "connector_cosine": float(connector["cosine"]),
        "connector_nmse": float(connector["nmse"]),
    }
    _finite_record(record)
    boundary.assert_unchanged()
    return record


def _snapshot(adapter: core.AffineAdapter) -> dict[str, np.ndarray]:
    return {name: value.detach().cpu().numpy().copy() for name, value in adapter.state_dict().items()}


def _save_selected(
    root: Path, *, source: str, condition: str, checkpoint: Mapping[str, Any],
    state: Mapping[str, np.ndarray], runtime_identity: str, guard: core.AccessGuard,
) -> dict[str, Any]:
    guard.require_clean()
    folder = root / "selected" / source.lower() / condition.lower()
    if folder.exists():
        _fail("selected artifact output collision")
    folder.mkdir(parents=True)
    archive = folder / "affine.npz"
    buffer = io.BytesIO()
    np.savez(buffer, W=state["W"], b=state["b"])
    payload = buffer.getvalue()
    with archive.open("xb") as stream:
        stream.write(payload)
    metadata = {
        "schema_id": ARTIFACT_SCHEMA_ID, "config_sha256": core.CONFIG_SHA256,
        "binding_sha256": BINDING_SHA256, "runtime_identity_sha256": runtime_identity,
        "source": source, "condition": condition, "selected_epoch": checkpoint["epoch"],
        "affine_npz_sha256": _sha_bytes(payload), "test_access_count": 0,
        "test_metric_count": 0, "generation_calls": 0,
    }
    with (folder / "metadata.json").open("xb") as stream:
        stream.write(_json_bytes(metadata))
    restored, loaded = load_selected_artifact(folder, runtime_identity=runtime_identity)
    if loaded != metadata or any(not np.array_equal(value, state[name]) for name, value in _snapshot(restored).items()):
        _fail("selected artifact save/reload mismatch")
    return {"relative_path": (folder / "affine.npz").relative_to(root).as_posix(), **metadata}


def load_selected_artifact(directory: str | Path, *, runtime_identity: str) -> tuple[core.AffineAdapter, dict[str, Any]]:
    folder = Path(directory)
    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        archive = folder / "affine.npz"
        payload = archive.read_bytes()
        if (metadata.get("schema_id") != ARTIFACT_SCHEMA_ID
                or metadata.get("config_sha256") != core.CONFIG_SHA256
                or metadata.get("binding_sha256") != BINDING_SHA256
                or metadata.get("runtime_identity_sha256") != runtime_identity
                or metadata.get("source") not in core.SOURCES
                or metadata.get("condition") not in GRADIENT_CONDITIONS
                or type(metadata.get("selected_epoch")) is not int
                or metadata["selected_epoch"] not in range(9)
                or any(metadata.get(key) != 0 for key in ("test_access_count", "test_metric_count", "generation_calls"))
                or metadata.get("affine_npz_sha256") != _sha_bytes(payload)):
            _fail("selected artifact metadata or SHA-256 mismatch")
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            if set(arrays.files) != {"W", "b"}:
                _fail("selected artifact numeric fields mismatch")
            adapter = core.AffineAdapter(arrays["W"], arrays["b"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as error:
        if isinstance(error, ValidationError):
            raise
        raise ValidationError("selected artifact cannot be safely reloaded") from error
    return adapter, metadata


def run_validation(
    *, config: Mapping[str, Any], binding: Mapping[str, Any], anchor: Mapping[str, Any],
    inputs: binding_module.BoundInputs, inference_connector: ConnectorOnly,
    output_root: str | Path, device: str, fake: bool = False,
) -> dict[str, Any]:
    """Run exactly the frozen 0..8 Train/Validation budget and publish result last."""
    core.validate_config(config)
    if _sha_bytes(_json_bytes(binding)) != CANONICAL_BINDING_SHA256:
        _fail("T003C binding config in memory differs from accepted bytes")
    if _sha_bytes(_json_bytes(anchor)) != CANONICAL_ANCHOR_SHA256:
        _fail("TinyLLaVA anchor config in memory differs from accepted bytes")
    if inputs.provenance.get("binding") != binding or binding.get("policy") != binding_module.DENIED_POLICY:
        _fail("T003C bound input or Test/generation policy drift")
    if (anchor.get("model_id") != config["connector_autograd_boundary"]["model_id"]
            or anchor.get("revision") != config["connector_autograd_boundary"]["revision"]):
        _fail("connector model/revision differs from accepted design")
    output = Path(output_root).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        _fail("output root must be new or empty")
    boundary = _check_connector(inference_connector, device, fake)
    if set(inputs.train) != set(core.SOURCES) or set(inputs.validation) != set(core.SOURCES):
        _fail("bound source set differs")
    if set(inputs.full_linear_initialization) != set(core.SOURCES):
        _fail("Full Linear source set differs")
    if fake:
        if device != "cpu" or inputs.provenance.get("synthetic_fixture") is not True:
            _fail("fake mode requires an explicit CPU synthetic fixture")
    elif (inputs.provenance.get("synthetic_fixture") is True
          or inputs.provenance.get("bank_preflight_status") != binding_module.bank_preflight.SUCCESS_STATUS):
        _fail("real mode requires verified T003C production inputs")
    for split, count in (("train", 800), ("validation", 100)):
        expected_ids = tuple(getattr(inputs, split)[core.SOURCES[0]].source_image_ids)
        for source in core.SOURCES:
            batch = getattr(inputs, split)[source]
            if tuple(batch.source_image_ids) != expected_ids:
                _fail("cross-source Train/Validation image order differs")
            if not fake and (batch.synthetic_fixture or batch.source.shape[0] != count):
                _fail("real mode requires frozen 800/100 image counts")
            if fake and not batch.synthetic_fixture:
                _fail("fake mode requires only synthetic batches")
    identity_material = {
        "schema_id": SCHEMA_ID, "config_sha256": core.CONFIG_SHA256,
        "binding_sha256": BINDING_SHA256,
        "anchor_config_sha256": config["provenance"]["tinyllava_anchor_config"]["sha256"],
        "t001d_plan_id": binding["t001d"]["t001d_plan_id"],
        "manifest_sha256": binding["t001d"]["manifest_sha256"],
        "bank_identities": binding["t001d"]["banks"],
        "t002d_validation_result_sha256": binding["t002d"]["validation_result_sha256"],
        "t002d_runtime_identity_sha256": binding["t002d"]["validation_runtime_identity_sha256"],
        "full_linear_hashes": {source: binding["t002d"]["full_linear"][source]["sha256"] for source in core.SOURCES},
        "train_image_ids_sha256": {
            source: _sha_bytes(_json_bytes(list(inputs.train[source].source_image_ids))) for source in core.SOURCES
        },
        "validation_image_ids_sha256": {
            source: _sha_bytes(_json_bytes(list(inputs.validation[source].source_image_ids))) for source in core.SOURCES
        },
        "device": device, "fake": fake,
    }
    runtime_identity = _sha_bytes(_json_bytes(identity_material))
    guard = core.AccessGuard()
    source_results: dict[str, Any] = {}
    for source in core.SOURCES:
        train = _on_device(inputs.train[source], device)
        held_validation = _on_device(inputs.validation[source], device)
        core.validate_train_validation_pair(train, held_validation, config)
        normalizers = core.compute_train_normalizers(train, config)
        train_patch_mean = train.target.mean(dim=(0, 1)).detach().cpu().numpy()
        train_connector_mean = train.target_connector.mean(dim=(0, 1)).detach().cpu().numpy()
        arrays = inputs.full_linear_initialization[source]
        reference = _fresh_adapter(arrays, device)
        _zero_step_parity(reference, train, boundary, inference_connector)
        initial_state = _snapshot(reference)
        ols = _evaluate(reference, held_validation, boundary, normalizers, source=source,
                        condition="OLS_REFERENCE", epoch=0, config=config,
                        train_patch_mean=train_patch_mean, train_connector_mean=train_connector_mean)
        if any(not np.array_equal(value, initial_state[name]) for name, value in _snapshot(reference).items()):
            _fail("OLS_REFERENCE parameters changed")
        conditions: dict[str, Any] = {}
        selected_states: dict[str, Mapping[str, np.ndarray]] = {}
        for condition in GRADIENT_CONDITIONS:
            adapter = _fresh_adapter(arrays, device)
            if any(not np.array_equal(value, initial_state[name]) for name, value in _snapshot(adapter).items()):
                _fail("gradient condition did not start from the accepted Full Linear initialization")
            optimizer = core.make_optimizer(adapter, config)
            records = []
            states = {}
            for epoch in range(9):
                if epoch:
                    core.train_epoch(adapter, train, boundary, normalizers, condition,
                                     optimizer, epoch, config, guard)
                records.append(_evaluate(adapter, held_validation, boundary, normalizers,
                                         source=source, condition=condition, epoch=epoch, config=config,
                                         train_patch_mean=train_patch_mean,
                                         train_connector_mean=train_connector_mean))
                states[epoch] = _snapshot(adapter)
                guard.require_clean()
                _connector_clean(inference_connector)
            selected = core.select_checkpoint(records, source, condition, config)
            conditions[condition] = {"validation_records": records, "selected_checkpoint": selected}
            selected_states[condition] = states[selected["epoch"]]
        winner = core.select_connector_weight(
            [conditions[condition]["selected_checkpoint"] for condition in core.CONDITIONS[2:]],
            source, config,
        )
        source_results[source] = {
            "normalizers": {"D_patch": float(normalizers.patch.item()),
                            "D_connector": float(normalizers.connector.item())},
            "OLS_REFERENCE": ols, "gradient_conditions": conditions,
            "selected_connector_aware": winner,
            "_selected_states": selected_states,
        }
    guard.require_clean()
    _connector_clean(inference_connector)
    boundary.assert_unchanged()
    output.mkdir(parents=True, exist_ok=True)
    for source in core.SOURCES:
        source_result = source_results[source]
        states = source_result.pop("_selected_states")
        artifacts = {}
        for condition in GRADIENT_CONDITIONS:
            artifacts[condition] = _save_selected(
                output, source=source, condition=condition,
                checkpoint=source_result["gradient_conditions"][condition]["selected_checkpoint"],
                state=states[condition], runtime_identity=runtime_identity, guard=guard,
            )
        source_result["selected_artifacts"] = artifacts
    result = {
        **identity_material, "runtime_identity_sha256": runtime_identity,
        "status": "PASS_CONNECTOR_AWARE_VALIDATION", "sources": source_results,
        "test_access_count": 0, "test_metric_count": 0,
        "generation_calls": 0, "vision_calls": 0,
    }
    payload = _json_bytes(result)
    with (output / "result.json").open("xb") as stream:
        stream.write(payload)
    return result
