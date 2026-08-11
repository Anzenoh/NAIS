#!/usr/bin/env python3
"""Internal, single-stage executor for VEA-P2-T001D.

Only argparse and the standard library are imported before the immutable plan
and requested stage have been validated.  Runtime/model imports stay inside the
stage handlers.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_STOP_REQUESTED = False
_CURRENT_UNIT: Mapping[str, Any] | None = None


class StageInterrupted(RuntimeError):
    pass


def _unit_started(core, run_root: Path, plan_id: str, unit: Mapping[str, Any]) -> None:
    global _CURRENT_UNIT
    _CURRENT_UNIT = unit
    core.append_event(
        run_root,
        core.progress_event(
            plan_id,
            "UNIT_STARTED",
            stage=unit["stage"],
            unit_id=unit["unit_id"],
            encoder=unit["encoder"],
            condition=unit["condition"],
            split=unit["split"],
            chunk_index=unit["chunk_index"],
            image_id=unit["image_id"],
        ),
    )


def _stop_after_unit(core, run_root: Path, plan_id: str, stage: str) -> None:
    if _STOP_REQUESTED:
        core.append_event(run_root, core.progress_event(plan_id, "INTERRUPTED", stage=stage))
        raise StageInterrupted(f"stop requested after atomic unit in {stage}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute exactly one validated T001D stage.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    return parser


def _bootstrap(args: argparse.Namespace):
    run_root = args.run_root.resolve()
    plan_path = args.plan.resolve()
    if plan_path != run_root / "run_plan.json":
        raise ValueError("--plan must name the run root's immutable run_plan.json")
    raw = plan_path.read_bytes()
    document = json.loads(raw)
    repository = Path(document["roots"]["repository_root"]).resolve()
    source = str(repository / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as core

    plan, plan_id = core.load_bound_plan(run_root)
    if plan != document or args.stage not in core.STAGES[1:7]:
        raise core.PlanMismatchError("stage or plan identity is outside the immutable DAG")
    runtime_bindings = plan.get("runtime_bindings")
    if isinstance(runtime_bindings, dict):
        overlay = os.path.normpath(os.path.abspath(str(runtime_bindings["main_overlay"])))
        main_root = os.path.normpath(os.path.abspath(str(runtime_bindings["main_runtime_root"])))
        observed_paths = [os.path.normpath(os.path.abspath(entry)) for entry in sys.path if entry]
        if args.stage == "01_DINOV3_FEATURE_BANK":
            if any(path == overlay or path.startswith(main_root + os.sep) for path in observed_paths):
                raise core.ContractError("DINO stage inherited main-lane site packages")
        elif overlay not in observed_paths:
            raise core.ContractError("main-lane stage is missing the approved TinyLLaVA overlay")
    lock_path = run_root / "coordinator.lock"
    if not lock_path.is_file():
        raise core.AmbiguousLockError("stage worker requires the live coordinator lock")
    lock = core.strict_json_load(lock_path)
    if lock.get("plan_id") != plan_id or lock.get("host") != socket.gethostname() or lock.get("pid") != os.getppid():
        raise core.AmbiguousLockError("stage worker is not a child of the bound coordinator lock owner")
    if core.process_liveness(lock["host"], int(lock["pid"]), lock.get("process_start_identity")) is not True:
        raise core.AmbiguousLockError("coordinator lock liveness cannot be proven by the stage worker")
    return core, plan, plan_id, run_root


def _units(plan: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    return [dict(unit) for unit in plan["expected_units"] if unit["stage"] == stage]


def _row_index(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["image_id"]: dict(row) for row in plan["manifest_rows"]}


def _image_path(plan: Mapping[str, Any], row: Mapping[str, Any], core) -> Path:
    root = Path(plan["roots"]["dataset_root"])
    candidate = (root / str(row["relative_path"])).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise core.ContractError("manifest image path escapes the dataset root") from error
    if not candidate.is_file() or core.sha256_file(candidate) != row["source_sha256"]:
        raise core.ContractError(f"dataset source SHA mismatch: {row['image_id']}")
    return candidate


def _npz_bytes(**arrays: Any) -> bytes:
    import numpy as np

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _validate_npz(payload: bytes, required: Mapping[str, tuple[tuple[int, ...], str]]) -> None:
    import numpy as np

    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(required):
            raise ValueError(f"NPZ fields differ: expected={sorted(required)} observed={sorted(archive.files)}")
        for name, (shape, dtype) in required.items():
            value = archive[name]
            if value.shape != shape or value.dtype.name != dtype:
                raise ValueError(f"NPZ field contract mismatch: {name} {value.shape} {value.dtype}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"NPZ field contains non-finite values: {name}")


def _publish_npz(core, run_root: Path, plan_id: str, unit: Mapping[str, Any], arrays: Mapping[str, Any], started: float) -> None:
    import numpy as np

    bound_arrays = dict(arrays)
    bound_arrays["plan_id"] = np.asarray(plan_id, dtype="U64")
    required = {name: (tuple(value.shape), value.dtype.name) for name, value in bound_arrays.items()}
    core.atomic_publish_unit(
        run_root,
        unit,
        _npz_bytes(**bound_arrays),
        plan_id,
        elapsed_seconds=time.perf_counter() - started,
        validate_payload=lambda payload, ignored: _validate_npz(payload, required),
    )
    _stop_after_unit(core, run_root, plan_id, str(unit["stage"]))


def _load_npz(run_root: Path, unit: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    with np.load(run_root / unit["artifact_relative_path"], allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True, order="C") for name in archive.files}


def _canonical_batch(plan: Mapping[str, Any], unit: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]], core):
    import numpy as np
    from vision_encoder_alignment.extractors.base import CanonicalImageBatch
    from vision_encoder_alignment.vlm_compat.dinov3_raw_injection import canonical_rgb_nhwc

    decoded = [canonical_rgb_nhwc(_image_path(plan, rows[image_id], core)) for image_id in unit["image_ids"]]
    pixels = np.concatenate(decoded, axis=0)
    if pixels.shape != (len(unit["image_ids"]), 224, 224, 3) or pixels.dtype != np.uint8:
        raise core.ContractError("canonical image batch must be uint8 [N,224,224,3]")
    return CanonicalImageBatch(tuple(unit["image_ids"]), pixels)


def _snapshot(cache_root: Path, model_id: str, revision: str) -> Path:
    return cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision


def _feature_stage(core, plan, plan_id, run_root, stage: str) -> tuple[int, int]:
    import numpy as np

    rows = _row_index(plan)
    pending = [unit for unit in _units(plan, stage) if not core.validate_unit_completion(run_root, unit, plan_id)]
    if not pending:
        return 0, 0
    repository = Path(plan["roots"]["repository_root"])
    cache = Path(plan["roots"]["cache_root"])
    model_loads = 0 if stage == "01_DINOV3_FEATURE_BANK" else 1
    worker_invocations = 0
    if stage == "01_DINOV3_FEATURE_BANK":
        from vision_encoder_alignment.vlm_compat import dinov3_raw_injection as raw

        bindings = raw.validate_dinov3_roots(
            repository_root=repository,
            runtime_root=plan["roots"]["dino_runtime_root"],
            snapshot_root=plan["roots"]["dino_snapshot_root"],
        )
        context = raw.apple_dinov3_interpreter_compatibility(bindings)
        context.__enter__()
        try:
            extractor = raw._construct_dinov3_extractor(
                bindings,
                profile_path=repository / raw.DINO_PROFILE,
                runtime_profile_path=repository / raw.DINO_RUNTIME_PROFILE,
            )
            tap = raw.DINO_TAP
            for unit in pending:
                _unit_started(core, run_root, plan_id, unit)
                started = time.perf_counter()
                batch = _canonical_batch(plan, unit, rows, core)
                result = extractor.extract(batch, tap_names=(tap,))
                worker_invocations += 1
                model_loads += 1
                patches = np.array(result.get_tap(tap).patch, dtype=np.float32, copy=True, order="C")
                patches = core.validate_patch_batch(patches, name="DINOV3 patches", batch_size=len(unit["image_ids"]))
                globals_ = np.ascontiguousarray(patches.mean(axis=1, dtype=np.float32))
                ids = np.asarray(unit["image_ids"], dtype="U96")
                _publish_npz(core, run_root, plan_id, unit, {"image_ids": ids, "patches": patches, "globals": globals_}, started)
        finally:
            try:
                del extractor
            except UnboundLocalError:
                pass
            context.__exit__(None, None, None)
    else:
        from vision_encoder_alignment.extractors.siglip2 import SigLIP2ExtractorConfig, SigLIP2VisionEncoderExtractor

        spec = plan["identity_bindings"]["encoders"]["SIGLIP2"]
        extractor = SigLIP2VisionEncoderExtractor(
            SigLIP2ExtractorConfig(
                snapshot_path=_snapshot(cache, spec["model_id"], spec["revision"]),
                runtime_profile_path=repository / spec["runtime_profile_path"],
                execution_selector="apple_main_cuda_fp32_v0",
            )
        )
        try:
            for unit in pending:
                _unit_started(core, run_root, plan_id, unit)
                started = time.perf_counter()
                batch = _canonical_batch(plan, unit, rows, core)
                result = extractor.extract(batch, tap_names=(spec["tap_name"],))
                patches = np.array(result.get_tap(spec["tap_name"]).patch, dtype=np.float32, copy=True, order="C")
                patches = core.validate_patch_batch(patches, name="SIGLIP2 patches", batch_size=len(unit["image_ids"]))
                globals_ = np.ascontiguousarray(patches.mean(axis=1, dtype=np.float32))
                ids = np.asarray(unit["image_ids"], dtype="U96")
                _publish_npz(core, run_root, plan_id, unit, {"image_ids": ids, "patches": patches, "globals": globals_}, started)
        finally:
            del extractor
    return model_loads, worker_invocations


def _load_tinyllava(plan: Mapping[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vision_encoder_alignment.vlm_compat import tinyllava_native_replay as native

    repository = Path(plan["roots"]["repository_root"])
    target = plan["identity_bindings"]["target_vlm"]
    config_path = repository / target["config_path"]
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != target["config_sha256"]:
        raise RuntimeError("TinyLLaVA configuration SHA mismatch")
    config = native.load_anchor_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after the coordinator GPU gate")
    torch_dtype = torch.float16
    common = {
        "revision": config.revision,
        "cache_dir": plan["roots"]["cache_root"],
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch_dtype, **common)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=plan["roots"]["cache_root"],
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
        use_fast=False,
        model_max_length=model.config.tokenizer_model_max_length,
        padding_side=model.config.tokenizer_padding_side,
    )
    model.to(device=plan["device"], dtype=torch_dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("TinyLLaVA must remain frozen")
    vision = model.get_vision_tower() if callable(getattr(model, "get_vision_tower", None)) else native._resolve_component(
        model, label="vision tower", paths=("vision_tower", "model.vision_tower")
    )
    connector = native._resolve_component(
        model, label="connector", paths=("connector", "model.connector", "mm_projector", "model.mm_projector")
    )
    native._resolve_runtime_config(model, vision, config)
    return torch, native, config, model, tokenizer, vision, connector


def _target_bank_stage(core, plan, plan_id, run_root) -> tuple[int, int]:
    import numpy as np
    from vision_encoder_alignment.vlm_compat.training_free_alignment_pilot import capture_native_calibration_features

    pending = [unit for unit in _units(plan, "03_TARGET_CLIP_AND_NATIVE_CONNECTOR_BANK") if not core.validate_unit_completion(run_root, unit, plan_id)]
    if not pending:
        return 0, 0
    rows = _row_index(plan)
    torch, native, config, model, tokenizer, vision, connector = _load_tinyllava(plan)
    try:
        with torch.inference_mode():
            for unit in pending:
                _unit_started(core, run_root, plan_id, unit)
                started = time.perf_counter()
                patches, connectors = [], []
                for image_id in unit["image_ids"]:
                    captured = capture_native_calibration_features(
                        model=model,
                        vision_tower=vision,
                        connector=connector,
                        image_path=_image_path(plan, rows[image_id], core),
                        dtype="float16",
                    )
                    patches.append(captured.patch[0])
                    connectors.append(captured.connector[0])
                patch = core.validate_patch_batch(np.stack(patches).astype(np.float32), name="TARGET_CLIP patches")
                connector_bank = np.ascontiguousarray(np.stack(connectors), dtype=np.float32)
                if connector_bank.shape != (len(unit["image_ids"]), 196, 1536) or not np.isfinite(connector_bank).all():
                    raise core.ContractError("native connector bank contract mismatch")
                arrays = {
                    "image_ids": np.asarray(unit["image_ids"], dtype="U96"),
                    "patches": patch,
                    "globals": np.ascontiguousarray(patch.mean(axis=1, dtype=np.float32)),
                    "connectors": connector_bank,
                }
                _publish_npz(core, run_root, plan_id, unit, arrays, started)
    finally:
        del model, tokenizer, vision, connector
        torch.cuda.empty_cache()
    return 1, 0


def _bank_units(plan: Mapping[str, Any], encoder: str) -> list[Mapping[str, Any]]:
    stage = {
        "DINOV3": "01_DINOV3_FEATURE_BANK",
        "SIGLIP2": "02_SIGLIP2_FEATURE_BANK",
        "TARGET_CLIP": "03_TARGET_CLIP_AND_NATIVE_CONNECTOR_BANK",
    }[encoder]
    return [unit for unit in plan["expected_units"] if unit["stage"] == stage]


def _load_bank(plan: Mapping[str, Any], run_root: Path, encoder: str) -> dict[str, Any]:
    import numpy as np

    chunks = [_load_npz(run_root, unit) for unit in _bank_units(plan, encoder)]
    result = {
        "image_ids": np.concatenate([chunk["image_ids"] for chunk in chunks]),
        "patches": np.concatenate([chunk["patches"] for chunk in chunks]),
        "globals": np.concatenate([chunk["globals"] for chunk in chunks]),
    }
    if encoder == "TARGET_CLIP":
        result["connectors"] = np.concatenate([chunk["connectors"] for chunk in chunks])
    expected = np.asarray([row["image_id"] for row in plan["manifest_rows"]], dtype="U96")
    if not np.array_equal(result["image_ids"], expected):
        raise RuntimeError(f"{encoder} bank order differs from immutable manifest")
    return result


def _anchor_stage(core, plan, plan_id, run_root) -> tuple[int, int]:
    import numpy as np

    pending = [unit for unit in _units(plan, "04_ANCHOR_REGISTRY") if not core.validate_unit_completion(run_root, unit, plan_id)]
    if not pending:
        return 0, 0
    indices_by_id = {row["image_id"]: index for index, row in enumerate(plan["manifest_rows"])}
    indices = np.asarray([indices_by_id[image_id] for image_id in plan["anchor_image_ids"]], dtype=np.int64)
    target = _load_bank(plan, run_root, "TARGET_CLIP")
    banks = {encoder: _load_bank(plan, run_root, encoder) for encoder in ENCODERS_FROM_CORE(core)}
    for unit in pending:
        _unit_started(core, run_root, plan_id, unit)
        started = time.perf_counter()
        name = unit["unit_id"].split("-")[-1].upper()
        encoder = "TARGET_CLIP" if name == "TARGET_DICTIONARY" else name
        bank = banks[encoder]
        arrays = {
            "image_ids": np.asarray(plan["anchor_image_ids"], dtype="U96"),
            "patches": np.ascontiguousarray(target["patches"][indices] if name == "TARGET_DICTIONARY" else bank["patches"][indices]),
            "globals": np.ascontiguousarray(bank["globals"][indices]),
        }
        if name == "TARGET_DICTIONARY":
            arrays["connectors"] = np.ascontiguousarray(target["connectors"][indices])
        _publish_npz(core, run_root, plan_id, unit, arrays, started)
    return 0, 0


def ENCODERS_FROM_CORE(core) -> tuple[str, ...]:
    return tuple(core.ENCODERS)


def _anchor_array(plan, run_root, name):
    unit = next(unit for unit in plan["expected_units"] if unit["unit_id"] == f"04_anchor_registry-{name.lower()}")
    return _load_npz(run_root, unit)


def _connector_output(torch, native, connector, array, device):
    tensor = torch.from_numpy(array)
    prepared = native._prepare_for_connector(tensor, connector).to(device=device)
    output = connector(prepared)
    return output.detach().to(device="cpu", dtype=torch.float32).numpy().copy(order="C")


def _transport_stage(core, plan, plan_id, run_root) -> tuple[int, int]:
    import numpy as np

    pending = [unit for unit in _units(plan, "05_TRANSPORT_AND_CONNECTOR_VALIDATION") if not core.validate_unit_completion(run_root, unit, plan_id)]
    if not pending:
        return 0, 0
    torch, native, config, model, tokenizer, vision, connector = _load_tinyllava(plan)
    target = _load_bank(plan, run_root, "TARGET_CLIP")
    target_anchor = _anchor_array(plan, run_root, "target_dictionary")["patches"]
    target_connector_anchor = _anchor_array(plan, run_root, "target_dictionary")["connectors"]
    indices = {image_id: index for index, image_id in enumerate(target["image_ids"].tolist())}
    try:
        with torch.inference_mode():
            for encoder in core.ENCODERS:
                source = _load_bank(plan, run_root, encoder)
                source_anchor = _anchor_array(plan, run_root, encoder)["patches"]
                for unit in [item for item in pending if item["encoder"] == encoder]:
                    _unit_started(core, run_root, plan_id, unit)
                    started = time.perf_counter()
                    selected = np.asarray([indices[image_id] for image_id in unit["image_ids"]], dtype=np.int64)
                    raw_patch = np.ascontiguousarray(source["patches"][selected], dtype=np.float32)
                    native_patch = np.ascontiguousarray(target["patches"][selected], dtype=np.float32)
                    transported = core.apply_transport_condition(unit["condition"], raw_patch, source_anchor, target_anchor)
                    transported = core.validate_patch_batch(transported, name="transported", batch_size=len(selected))
                    connector_output = np.ascontiguousarray(
                        _connector_output(torch, native, connector, transported, plan["device"]), dtype=np.float32
                    )
                    native_connector = np.ascontiguousarray(target["connectors"][selected], dtype=np.float32)
                    if connector_output.shape != (len(selected), 196, 1536) or not np.isfinite(connector_output).all():
                        raise core.ContractError("connector output contract mismatch")
                    if encoder == "TARGET_CLIP" and unit["condition"] == "RAW":
                        if not np.array_equal(transported, native_patch) or not np.array_equal(connector_output, native_connector):
                            raise core.ContractError("TARGET_CLIP RAW patch/connector self-control failed")
                    metric_rows = []
                    for metric_index, image_id in enumerate(unit["image_ids"]):
                        patch_metrics = core.representation_metrics(
                            np.ascontiguousarray(transported[metric_index : metric_index + 1]),
                            np.ascontiguousarray(native_patch[metric_index : metric_index + 1]),
                            target_anchor,
                        )
                        connector_metrics = core.representation_metrics(
                            np.ascontiguousarray(connector_output[metric_index : metric_index + 1]),
                            np.ascontiguousarray(native_connector[metric_index : metric_index + 1]),
                            target_connector_anchor,
                        )
                        metric_rows.append(
                            {
                                "image_id": image_id,
                                **{f"patch_{key}": value for key, value in patch_metrics.items()},
                                **{f"connector_{key}": value for key, value in connector_metrics.items()},
                            }
                        )
                    arrays = {
                        "image_ids": np.asarray(unit["image_ids"], dtype="U96"),
                        "transported": transported,
                        "connector": connector_output,
                        "native_patch": native_patch,
                        "native_connector": native_connector,
                        "per_image_metrics_json": np.asarray(
                            json.dumps(metric_rows, sort_keys=True, separators=(",", ":")), dtype="U32768"
                        ),
                    }
                    _publish_npz(core, run_root, plan_id, unit, arrays, started)
                del source
    finally:
        del model, tokenizer, vision, connector
        torch.cuda.empty_cache()
    return 1, 0


def _transported_for(plan, run_root, image_id, encoder, condition):
    units = [
        unit for unit in plan["expected_units"]
        if unit["stage"] == "05_TRANSPORT_AND_CONNECTOR_VALIDATION"
        and unit["encoder"] == encoder and unit["condition"] == condition and image_id in unit["image_ids"]
    ]
    if len(units) != 1:
        raise RuntimeError("transport unit lookup is not unique")
    data = _load_npz(run_root, units[0])
    index = data["image_ids"].tolist().index(image_id)
    return data["transported"][index : index + 1]


def _generation_stage(core, plan, plan_id, run_root) -> tuple[int, int]:
    import numpy as np
    from vision_encoder_alignment.vlm_compat.training_free_alignment_pilot import _capture, _validate_capture

    pending = [unit for unit in _units(plan, "06_HELD_OUT_GENERATION") if not core.validate_unit_completion(run_root, unit, plan_id)]
    if not pending:
        return 0, 0
    rows = _row_index(plan)
    torch, native, config, model, tokenizer, vision, connector = _load_tinyllava(plan)
    try:
        with torch.inference_mode():
            for unit in pending:
                _unit_started(core, run_root, plan_id, unit)
                started = time.perf_counter()
                image_id = unit["image_ids"][0]
                image_path = _image_path(plan, rows[image_id], core)
                if unit["encoder"] is None:
                    state, tokens, text = _capture(model, vision, connector, tokenizer, image_path, config, 32)
                    _validate_capture(state, "float16", injected=False)
                    injection_calls = 0
                else:
                    patch = core.validate_patch_batch(
                        _transported_for(plan, run_root, image_id, unit["encoder"], unit["condition"]),
                        name="generation replacement",
                        batch_size=1,
                    )
                    injection = native.ReplayInjection(
                        model, torch.from_numpy(patch), connector, prepare_patch=native._prepare_for_connector
                    )
                    with injection:
                        state, tokens, text = _capture(model, vision, connector, tokenizer, image_path, config, 32)
                    _validate_capture(state, "float16", injected=True)
                    if injection.invocations != 1 or not injection.restored:
                        raise core.ContractError("generation injection did not run once and restore encode_images")
                    injection_calls = injection.invocations
                flat_tokens = [int(value) for row in tokens for value in row]
                record = {
                    "schema_id": core.UNIT_SCHEMA,
                    "plan_id": plan_id,
                    "unit_id": unit["unit_id"],
                    "image_id": image_id,
                    "encoder": unit["encoder"],
                    "condition": unit["condition"] if unit["encoder"] else "NATIVE",
                    "token_ids": flat_tokens,
                    "decoded_text": text,
                    "generation_success": True,
                    "injection_calls": injection_calls,
                }
                if unit["encoder"] is not None:
                    native_unit = next(
                        item for item in plan["expected_units"]
                        if item["stage"] == "06_HELD_OUT_GENERATION"
                        and item["encoder"] is None
                        and item["image_ids"] == [image_id]
                    )
                    native_record = json.loads((run_root / native_unit["artifact_relative_path"]).read_text(encoding="utf-8"))
                    record.update(core.generation_metrics(flat_tokens, native_record["token_ids"]))
                    record["token_ids_equal_to_native"] = flat_tokens == native_record["token_ids"]
                    record["decoded_text_equal_to_native"] = text == native_record["decoded_text"]
                    record["native_answer"] = native_record["decoded_text"]
                    record["condition_answer"] = text
                    if unit["encoder"] == "TARGET_CLIP" and unit["condition"] == "RAW":
                        if not record["token_ids_equal_to_native"] or not record["decoded_text_equal_to_native"]:
                            raise core.ContractError("TARGET_CLIP RAW generation self-control failed")
                payload = core.canonical_json_bytes(record)
                core.atomic_publish_unit(
                    run_root,
                    unit,
                    payload,
                    plan_id,
                    elapsed_seconds=time.perf_counter() - started,
                    validate_payload=lambda observed, ignored: json.loads(observed),
                )
                _stop_after_unit(core, run_root, plan_id, str(unit["stage"]))
    finally:
        del model, tokenizer, vision, connector
        torch.cuda.empty_cache()
    return 1, 0


def execute_stage(
    core,
    plan: Mapping[str, Any],
    plan_id: str,
    run_root: Path,
    stage: str,
    *,
    fake_loader: Callable[[str], Any] | None = None,
    fake_processor: Callable[[Any, Mapping[str, Any]], bytes] | None = None,
) -> dict[str, Any]:
    """Execute one stage; injected fakes prove persistent one-load behavior."""

    pending = [unit for unit in _units(plan, stage) if not core.validate_unit_completion(run_root, unit, plan_id)]
    if not pending:
        return {
            "stage": stage,
            "completed_new": 0,
            "model_load_count": 0,
            "stage_worker_invocation_count": 1,
            "extractor_worker_invocation_count": 0,
        }
    core.append_event(run_root, core.progress_event(plan_id, "STAGE_STARTED", stage=stage))
    completed = 0
    if fake_loader is not None or fake_processor is not None:
        if fake_loader is None or fake_processor is None:
            raise ValueError("fake loader and processor must be supplied together")
        if stage == "01_DINOV3_FEATURE_BANK":
            for unit in pending:
                model = fake_loader(stage)
                try:
                    _unit_started(core, run_root, plan_id, unit)
                    core.atomic_publish_unit(
                        run_root,
                        unit,
                        fake_processor(model, unit),
                        plan_id,
                        elapsed_seconds=0.0,
                    )
                    completed += 1
                    _stop_after_unit(core, run_root, plan_id, stage)
                finally:
                    close = getattr(model, "close", None)
                    if callable(close):
                        close()
            loads, workers = completed, completed
        else:
            model = fake_loader(stage)
            try:
                for unit in pending:
                    _unit_started(core, run_root, plan_id, unit)
                    core.atomic_publish_unit(
                        run_root,
                        unit,
                        fake_processor(model, unit),
                        plan_id,
                        elapsed_seconds=0.0,
                    )
                    completed += 1
                    _stop_after_unit(core, run_root, plan_id, stage)
            finally:
                close = getattr(model, "close", None)
                if callable(close):
                    close()
            loads, workers = 1, 1
    elif stage in {"01_DINOV3_FEATURE_BANK", "02_SIGLIP2_FEATURE_BANK"}:
        loads, workers = _feature_stage(core, plan, plan_id, run_root, stage)
        completed = len(pending)
    elif stage == "03_TARGET_CLIP_AND_NATIVE_CONNECTOR_BANK":
        loads, workers = _target_bank_stage(core, plan, plan_id, run_root)
        completed = len(pending)
    elif stage == "04_ANCHOR_REGISTRY":
        loads, workers = _anchor_stage(core, plan, plan_id, run_root)
        completed = len(pending)
    elif stage == "05_TRANSPORT_AND_CONNECTOR_VALIDATION":
        loads, workers = _transport_stage(core, plan, plan_id, run_root)
        completed = len(pending)
    else:
        loads, workers = _generation_stage(core, plan, plan_id, run_root)
        completed = len(pending)
    core.append_event(
        run_root,
        core.progress_event(
            plan_id,
            "STAGE_COMPLETED",
            stage=stage,
            completed_new=completed,
            model_load_count=loads,
            stage_worker_invocation_count=1,
            extractor_worker_invocation_count=workers,
        ),
    )
    return {
        "stage": stage,
        "completed_new": completed,
        "model_load_count": loads,
        "stage_worker_invocation_count": 1,
        "extractor_worker_invocation_count": workers,
    }


def _resource_failure(error: BaseException) -> bool:
    message = str(error).lower()
    signatures = (
        "cuda out of memory",
        "cuda allocation",
        "nvml",
        "gpu availability",
    )
    return any(signature in message for signature in signatures)


def main(argv: Sequence[str] | None = None) -> int:
    global _STOP_REQUESTED
    args = build_parser().parse_args(argv)
    core, plan, plan_id, run_root = _bootstrap(args)
    def request_stop(signum, frame) -> None:
        global _STOP_REQUESTED
        _STOP_REQUESTED = True

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, request_stop)
    try:
        result = execute_stage(core, plan, plan_id, run_root, args.stage)
        if _STOP_REQUESTED:
            core.append_event(run_root, core.progress_event(plan_id, "INTERRUPTED", stage=args.stage))
            return 130
        print(json.dumps({"status": "PASS", "plan_id": plan_id, **result}, sort_keys=True, separators=(",", ":")))
        return 0
    except StageInterrupted as error:
        print(json.dumps({"status": "PARTIAL", "plan_id": plan_id, "cause": str(error)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 130
    except BaseException as error:
        retryable = _resource_failure(error)
        classification = "RETRYABLE_FAILURE" if retryable else "NONRETRYABLE_FAILURE"
        current_fields = {} if _CURRENT_UNIT is None else {
            key: _CURRENT_UNIT.get(key)
            for key in ("unit_id", "encoder", "condition", "split", "chunk_index", "image_id")
        }
        core.append_event(
            run_root,
            core.progress_event(
                plan_id, classification, stage=args.stage, cause=type(error).__name__, **current_fields
            ),
        )
        recorded_error: BaseException = error
        if retryable:
            recorded_error = core.IncompleteRunError(str(error))
            recorded_error.__cause__ = error
        record = core.failure_record(
            recorded_error,
            plan_id,
            None,
        )
        coordinates = {} if _CURRENT_UNIT is None else {
            key: _CURRENT_UNIT.get(key)
            for key in ("unit_id", "encoder", "condition", "split", "chunk_index", "image_id", "image_ids")
        }
        completed = [
            unit["unit_id"] for unit in plan["expected_units"]
            if core.marker_path(run_root, unit["unit_id"]).exists()
        ]
        core.write_failure_record(
            run_root,
            {
                **record,
                "stage": args.stage,
                "host": socket.gethostname(),
                "coordinates": coordinates,
                "last_valid_unit": completed[-1] if completed else None,
                "exact_log": "coordinator-captured stage-worker stdout/stderr",
            },
        )
        print(json.dumps(record, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return core.RETRYABLE_EXIT_CODE if retryable else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
