#!/usr/bin/env python3
"""Plan, preflight, execute, resume, or validate P1 feature extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from vision_encoder_alignment.p1_feature_extraction import (  # noqa: E402
    EXIT_EXECUTION,
    EXIT_NOT_READY,
    EXIT_SUCCESS,
    FeatureExtractionError,
    GLOBAL_TAPS,
    MODEL_ORDER,
    PATCH_TAPS,
    atomic_write_json,
    complete_run,
    execution_plan,
    initial_run_state,
    initialize_run_artifacts,
    load_extraction_profile,
    offline_environment,
    ordered_batches,
    plan_shards,
    reconcile_resume,
    run_command,
    sha256_file,
    stream_manifest,
    validate_mode,
    validate_completion_candidate,
    validate_non_link_path,
    validate_output_root,
    validate_run,
    verify_repository_inputs,
    write_execution_metrics,
    write_failure_evidence,
    write_shard,
)
from vision_encoder_alignment.p1_runtime_preflight import (  # noqa: E402
    PreflightError,
    checkpoint_snapshot_files,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight"):
        item = sub.add_parser(name, allow_abbrev=False)
        item.add_argument("--profile", required=True)
        item.add_argument("--repository-root", required=True)
        item.add_argument("--output-json")
        if name == "preflight":
            item.add_argument("--dataset-root", required=True)
            item.add_argument("--cache-root", required=True)
    for name in ("run", "resume"):
        item = sub.add_parser(name, allow_abbrev=False)
        item.add_argument("--profile", required=True)
        item.add_argument("--repository-root", required=True)
        item.add_argument("--dataset-root", required=True)
        item.add_argument("--cache-root", required=True)
        item.add_argument("--output-parent", required=True)
        item.add_argument("--run-id", required=True)
        item.add_argument("--mode", choices=("smoke", "full"), required=True)
        item.add_argument("--batch-size", type=int, required=True)
    validate = sub.add_parser("validate", allow_abbrev=False)
    validate.add_argument("--profile", required=True)
    validate.add_argument("--run-root", required=True)
    validate.add_argument("--allow-incomplete", action="store_true")
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS, allow_abbrev=False)
    worker.add_argument("--profile", required=True)
    worker.add_argument("--repository-root", required=True)
    worker.add_argument("--dataset-root", required=True)
    worker.add_argument("--cache-root", required=True)
    worker.add_argument("--run-root", required=True)
    worker.add_argument("--model", choices=MODEL_ORDER, required=True)
    worker.add_argument("--mode", choices=("smoke", "full"), required=True)
    worker.add_argument("--batch-size", type=int, required=True)
    worker.add_argument("--start-shard", type=int, default=0)
    return parser


def _canonical_output(value: Mapping[str, Any]) -> bytes:
    from vision_encoder_alignment.p1_feature_extraction import canonical_json_bytes
    return canonical_json_bytes(value)


def _write_output(raw: bytes, output: str) -> None:
    """Publish one plan/preflight record without replacing existing bytes."""

    path = Path(output)
    if (
        not path.is_absolute()
        or os.fspath(path) != os.path.normpath(os.fspath(path))
        or path.exists() or path.is_symlink()
    ):
        raise FeatureExtractionError("evidence output is unavailable or already exists", 4)
    validate_non_link_path(
        path.parent, expected_kind="directory", label="evidence output parent",
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _emit(value: Mapping[str, Any], output: str | None = None) -> None:
    raw = _canonical_output(value)
    if output:
        _write_output(raw, output)
    sys.stdout.buffer.write(raw)


def _emit_error(value: Mapping[str, Any], output: str | None) -> None:
    """Emit a canonical error without letting evidence publication change its exit."""

    raw = _canonical_output(value)
    if output:
        try:
            _write_output(raw, output)
        except (FeatureExtractionError, OSError):
            pass
    sys.stdout.buffer.write(raw)


def _snapshot(cache_root: Path, model_id: str, revision: str) -> Path:
    return cache_root / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision


def _validate_dinov3_contract(profile: Any, repository: Path, cache_root: Path) -> None:
    from vision_encoder_alignment.dinov3_extraction_boundary import (
        DINOV3_SELECTOR,
        load_dinov3_extractor_profile,
    )
    from vision_encoder_alignment.production_runtime_contract import (
        load_production_runtime_profile,
        validate_profile_source_bindings,
    )
    from vision_encoder_alignment.runtime_boundary import runtime_python_path
    from vision_encoder_alignment.snapshot_boundary import validate_apple_hf_snapshot

    lane = profile.document["runtime"]["dinov3"]
    extractor_profile = load_dinov3_extractor_profile(
        repository / lane["extractor_profile"]["path"],
        execution_selector=DINOV3_SELECTOR,
    )
    runtime_profile = load_production_runtime_profile(repository / lane["profile"]["path"])
    validate_profile_source_bindings(runtime_profile, repository)
    configured = Path(lane["interpreter"])
    expected = runtime_python_path(
        runtime_profile.runtime_root,
        interpreter_layout=runtime_profile.interpreter_layout,
    )
    if configured != expected and configured != expected.with_name("python3"):
        raise FeatureExtractionError("DINOv3 interpreter contract mismatch", 4)
    model = next(item for item in profile.models if item["registry_key"] == MODEL_ORDER[2])
    snapshot = _snapshot(cache_root, model["model_id"], model["revision"])
    validate_apple_hf_snapshot(
        snapshot,
        cache_root=cache_root,
        model_id=model["model_id"],
        revision=model["revision"],
        files=extractor_profile.checkpoint_files,
    )


def _validate_dataset(rows: Sequence[Any], dataset_root: Path) -> None:
    from PIL import Image
    root = validate_non_link_path(
        dataset_root, expected_kind="directory", label="dataset root",
    )
    for row in rows:
        path = validate_non_link_path(
            root.joinpath(*row.relative_path.split("/")),
            expected_kind="file", label="dataset image",
        )
        if not path.is_relative_to(root):
            raise FeatureExtractionError("dataset image is missing or escapes root", 4)
        if sha256_file(path) != row.source_sha256:
            raise FeatureExtractionError("dataset source SHA mismatch")
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB"); rgb.load()
                decoded = hashlib.sha256(rgb.tobytes()).hexdigest()
        except Exception as exc:
            raise FeatureExtractionError("dataset image decode failed", 4) from exc
        if decoded != row.decoded_rgb_sha256:
            raise FeatureExtractionError("dataset decoded-RGB SHA mismatch")


def _preflight(
    profile: Any,
    repository: Path,
    dataset_root: Path,
    cache_root: Path,
    *,
    dataset_validator: Callable[[Sequence[Any], Path], None] = _validate_dataset,
    interpreter_is_file: Callable[[Path], bool] | None = None,
    host_name: Callable[[], str] = platform.node,
    gpu_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    dinov3_validator: Callable[[Any, Path, Path], None] = _validate_dinov3_contract,
) -> dict[str, Any]:
    repository = validate_non_link_path(
        repository, expected_kind="directory", label="repository root",
    )
    verify_repository_inputs(profile, repository)
    manifest = repository / profile.document["dataset"]["manifest"]["path"]
    rows = stream_manifest(manifest, expected_sha256=profile.document["dataset"]["manifest"]["sha256"])
    dataset = validate_non_link_path(
        dataset_root, expected_kind="directory", label="dataset root",
    )
    dataset_validator(rows, dataset)
    cache = validate_non_link_path(
        cache_root, expected_kind="directory", label="checkpoint cache root",
    )
    if cache.is_relative_to(repository) or repository.is_relative_to(cache):
        raise FeatureExtractionError("checkpoint cache root is unsafe", 4)
    for model in profile.models:
        try:
            files = checkpoint_snapshot_files(
                cache,
                model_id=model["model_id"],
                revision=model["revision"],
            )
        except PreflightError as exc:
            raise FeatureExtractionError(
                f"checkpoint snapshot unavailable: {model['registry_key']}", 4,
            ) from exc
        if not files:
            raise FeatureExtractionError(f"checkpoint snapshot unavailable: {model['registry_key']}", 4)
    dinov3_validator(profile, repository, cache)
    runtime = profile.document["runtime"]
    for lane in (runtime["main"], runtime["dinov3"]):
        interpreter = Path(lane["interpreter"])
        if interpreter_is_file is None:
            validate_non_link_path(
                interpreter, expected_kind="file", label="runtime interpreter",
            )
            validate_non_link_path(
                interpreter.parent.parent, expected_kind="directory", label="runtime root",
            )
        elif not interpreter_is_file(interpreter):
            raise FeatureExtractionError("runtime interpreter unavailable", 4)
    if host_name() != "apple":
        raise FeatureExtractionError("preflight host is not apple", EXIT_NOT_READY)
    gpu = gpu_runner(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, shell=False,
    )
    if gpu.returncode != 0 or gpu.stdout.strip() != "0, NVIDIA GeForce RTX 5090":
        raise FeatureExtractionError("RTX 5090 preflight failed", EXIT_NOT_READY)
    return {
        "classification": "P1_FEATURE_EXTRACTION_PREFLIGHT_VALID",
        "dataset_image_count": 1000, "model_count": 3,
        "operation_counts": {
            "cache_write": 0,
            "credential_read": 0,
            "model_load": 0,
            "network": 0,
            "package": 0,
        },
        "ready_for_p1_extraction": False, "ready_for_smoke_execution": True,
        "schema_id": "vea.p1.feature_extraction_preflight.v0",
    }


def _canonical_batch(rows: Sequence[Any], dataset_root: Path) -> Any:
    import numpy as np
    from PIL import Image
    images = []
    for row in rows:
        with Image.open(dataset_root.joinpath(*row.relative_path.split("/"))) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            short = min(width, height)
            if width <= height:
                size = (224, (height * 224 + short // 2) // short)
            else:
                size = ((width * 224 + short // 2) // short, 224)
            resized = rgb.resize(size, Image.Resampling.BICUBIC)
            left, top = (size[0] - 224) // 2, (size[1] - 224) // 2
            cropped = resized.crop((left, top, left + 224, top + 224))
            images.append(np.ascontiguousarray(np.asarray(cropped), dtype=np.uint8))
    return np.ascontiguousarray(np.stack(images), dtype=np.uint8)


def _extractor(model_key: str, profile: Any, repository: Path, cache_root: Path) -> Any:
    from vision_encoder_alignment.extractors import (
        CLIPExtractorConfig, CLIPVisionEncoderExtractor,
        DINOv3ExtractorConfig, DINOv3VisionEncoderExtractor,
        SigLIP2ExtractorConfig, SigLIP2VisionEncoderExtractor,
    )
    model = next(item for item in profile.models if item["registry_key"] == model_key)
    snapshot = _snapshot(cache_root, model["model_id"], model["revision"])
    if model_key == MODEL_ORDER[0]:
        return CLIPVisionEncoderExtractor(CLIPExtractorConfig(
            snapshot, execution_selector="apple_main_cuda_fp32_v0",
            runtime_profile_path=repository / profile.document["runtime"]["main"]["profile"]["path"],
        ))
    if model_key == MODEL_ORDER[1]:
        return SigLIP2VisionEncoderExtractor(SigLIP2ExtractorConfig(
            snapshot, execution_selector="apple_main_cuda_fp32_v0",
            runtime_profile_path=repository / profile.document["runtime"]["main"]["profile"]["path"],
        ))
    return DINOv3VisionEncoderExtractor(DINOv3ExtractorConfig(
        repository / profile.document["runtime"]["dinov3"]["extractor_profile"]["path"],
        snapshot_root=snapshot,
            runtime_root=Path(profile.document["runtime"]["dinov3"]["interpreter"]).parent.parent,
        execution_selector="dinov3_apple_cuda_fp32_v1",
        apple_runtime_profile_path=repository / profile.document["runtime"]["dinov3"]["profile"]["path"],
    ))


def _worker(args: argparse.Namespace) -> int:
    import numpy as np
    from vision_encoder_alignment.extractors.base import CanonicalImageBatch
    profile = load_extraction_profile(args.profile)
    repository = validate_non_link_path(
        args.repository_root, expected_kind="directory", label="repository root",
    )
    rows_all = stream_manifest(repository / profile.document["dataset"]["manifest"]["path"], expected_sha256=profile.document["dataset"]["manifest"]["sha256"])
    count = validate_mode(profile, args.mode, args.batch_size)
    rows = rows_all[:count]
    plans = plan_shards(tuple(row.image_id for row in rows))
    model = next(item for item in profile.models if item["registry_key"] == args.model)
    run_root = validate_non_link_path(
        args.run_root, expected_kind="directory", label="worker run root",
    )
    shard_dir = run_root / "models" / args.model
    started = time.perf_counter()
    dataset_root = validate_non_link_path(
        args.dataset_root, expected_kind="directory", label="worker dataset root",
    )
    cache_root = validate_non_link_path(
        args.cache_root, expected_kind="directory", label="worker checkpoint cache root",
    )
    extractor = _extractor(args.model, profile, repository, cache_root)
    peak_allocated = 0
    peak_reserved = 0
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        torch = None
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for plan in plans[args.start_shard:]:
        shard_rows = rows[plan.start:plan.stop]
        collected = {f"backbone.{tap}.global_mean": [] for tap in GLOBAL_TAPS}
        collected.update({f"backbone.{tap}.patch": [] for tap in PATCH_TAPS})
        for batch_rows in ordered_batches(shard_rows, args.batch_size):
            pixels = _canonical_batch(batch_rows, dataset_root)
            result = extractor.extract(CanonicalImageBatch(tuple(row.image_id for row in batch_rows), pixels), tap_names=GLOBAL_TAPS)
            for feature in result.tap_features:
                collected[f"backbone.{feature.tap.name}.global_mean"].append(feature.global_mean)
                if feature.tap.name in PATCH_TAPS:
                    collected[f"backbone.{feature.tap.name}.patch"].append(feature.patch)
        tensors = {key: np.ascontiguousarray(np.concatenate(values, axis=0), dtype=np.float32) for key, values in collected.items()}
        write_shard(shard_dir, run_id=run_root.name, model=model, lane_id=model["lane_id"], shard=plan, tensors=tensors, created_at_utc=created)
    if torch is not None and torch.cuda.is_available():
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    write_execution_metrics(
        shard_dir / "execution_metrics.json", run_id=run_root.name, model=model,
        image_count=len(rows), elapsed_seconds=time.perf_counter() - started,
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
    )
    return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_WORKER_ERROR_SCHEMA_ID = "vea.p1.feature_extraction_error.v0"
_WORKER_ERROR_FIELDS = frozenset((
    "error", "exit_code", "ready_for_p1_extraction", "schema_id",
))
_WORKER_ERROR_MAX_BYTES = 512
_WORKER_DIAGNOSTIC_MAX_CHARS = 160
_WORKER_DIAGNOSTIC_SENSITIVE = re.compile(
    r"(?:token|secret|password|credential|api[_ -]?key)", re.IGNORECASE,
)


def _worker_error_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate key")
        document[key] = value
    return document


def _reject_worker_json_constant(value: str) -> None:
    raise ValueError(f"non-finite constant: {value}")


def _bounded_worker_failure_diagnostic(stdout: str, returncode: int) -> str:
    """Return only a canonical, safe child error classification or the fallback."""

    fallback = "worker_process_failed"
    if not isinstance(stdout, str):
        return fallback
    try:
        raw = stdout.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return fallback
    if not raw or len(raw) > _WORKER_ERROR_MAX_BYTES:
        return fallback
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_worker_error_pairs,
            parse_constant=_reject_worker_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError):
        return fallback
    if not isinstance(document, dict) or set(document) != _WORKER_ERROR_FIELDS:
        return fallback
    if (
        document["schema_id"] != _WORKER_ERROR_SCHEMA_ID
        or document["ready_for_p1_extraction"] is not False
        or type(document["exit_code"]) is not int
        or document["exit_code"] != returncode
    ):
        return fallback
    message = document["error"]
    if (
        not isinstance(message, str)
        or not 0 < len(message) <= _WORKER_DIAGNOSTIC_MAX_CHARS
        or any(not (" " <= character <= "~") for character in message)
        or "/" in message
        or "\\" in message
        or re.search(r"(?:^[A-Za-z]:|^~)", message) is not None
        or _WORKER_DIAGNOSTIC_SENSITIVE.search(message) is not None
    ):
        return fallback
    try:
        if raw != _canonical_output(document):
            return fallback
    except FeatureExtractionError:
        return fallback
    return message


def _record_failure_without_masking(
    run_root: Path, *, state: Mapping[str, Any], stage: str,
    exception_class: str, sanitized_message: str,
    original: BaseException, model: Mapping[str, Any] | None = None,
    shard: Any | None = None, raw_return_code: int | None = None,
) -> None:
    try:
        write_failure_evidence(
            run_root, state=state, stage=stage, model=model, shard=shard,
            exception_class=exception_class, stable_exit_code=EXIT_EXECUTION,
            raw_return_code=raw_return_code,
            sanitized_message=sanitized_message, created_at_utc=_utc_now(),
        )
    except Exception as evidence_error:
        add_note = getattr(original, "add_note", None)
        if add_note is not None:
            add_note(
                "failure evidence publication failed without overwrite fallback: "
                f"{type(evidence_error).__name__}"
            )


def _progress_marker(event: str, model: str, classification: str, *, exit_code: int | None = None) -> None:
    marker = f"{event} model={model} classification={classification}"
    if exit_code is not None:
        marker += f" exit_code={exit_code}"
    print(marker, file=sys.stderr, flush=True)


def _run_or_resume(args: argparse.Namespace, *, resume: bool) -> int:
    profile = load_extraction_profile(args.profile)
    repository = validate_non_link_path(
        args.repository_root, expected_kind="directory", label="repository root",
    )
    verify_repository_inputs(profile, repository)
    rows_all = stream_manifest(repository / profile.document["dataset"]["manifest"]["path"], expected_sha256=profile.document["dataset"]["manifest"]["sha256"])
    count = validate_mode(profile, args.mode, args.batch_size)
    rows = rows_all[:count]
    dataset_root = validate_non_link_path(
        args.dataset_root, expected_kind="directory", label="dataset root",
    )
    cache_root = validate_non_link_path(
        args.cache_root, expected_kind="directory", label="checkpoint cache root",
    )
    protected = [repository, dataset_root, cache_root]
    if os.name != "nt":
        protected.extend((
            "/home/chltjsdh/vea/runtime/p1_apple",
            "/home/chltjsdh/vea/artifacts/p1_t002",
        ))
    if resume:
        run_root = validate_non_link_path(
            Path(args.output_parent) / args.run_id,
            expected_kind="directory", label="resume run root",
        )
        state = initial_run_state(profile, run_id=args.run_id, mode=args.mode, batch_size=args.batch_size, image_ids=[row.image_id for row in rows])
        try:
            progress = reconcile_resume(run_root, state, profile)
        except FeatureExtractionError as exc:
            _record_failure_without_masking(
                run_root, state=state, stage="model_free_validation",
                exception_class="ResumeReconciliationError",
                sanitized_message="resume_reconciliation_failed", original=exc,
            )
            raise
    else:
        run_root = validate_output_root(
            args.output_parent, args.run_id, protected_roots=protected,
            require_parent=(profile.document["output"]["parent"] if os.name != "nt" else None),
        )
        run_root.mkdir(mode=0o700)
        state = initialize_run_artifacts(
            run_root, profile, repository, run_id=args.run_id, mode=args.mode,
            batch_size=args.batch_size, image_ids=[row.image_id for row in rows],
        )
        progress = {model: 0 for model in MODEL_ORDER}
    env = offline_environment(repository)
    runtime = profile.document["runtime"]
    interpreters = (runtime["main"]["interpreter"], runtime["main"]["interpreter"], runtime["dinov3"]["interpreter"])
    if os.name != "nt":
        for interpreter in dict.fromkeys(interpreters):
            path = validate_non_link_path(
                interpreter, expected_kind="file", label="runtime interpreter",
            )
            validate_non_link_path(
                path.parent.parent, expected_kind="directory", label="runtime root",
            )
    plans = plan_shards(tuple(row.image_id for row in rows))
    by_model = {item["registry_key"]: item for item in profile.models}
    for model, interpreter in zip(MODEL_ORDER, interpreters, strict=True):
        metrics = run_root / "models" / model / "execution_metrics.json"
        if progress[model] == len(plans) and metrics.is_file() and not metrics.is_symlink():
            continue
        command = [interpreter, "-B", str(Path(__file__).resolve()), "_worker",
                   "--profile", str(Path(args.profile).resolve()), "--repository-root", str(repository),
                   "--dataset-root", str(dataset_root), "--cache-root", str(cache_root),
                   "--run-root", str(run_root), "--model", model, "--mode", args.mode,
                   "--batch-size", str(args.batch_size), "--start-shard", str(progress[model])]
        _progress_marker("model_start", model, "execution_started")
        try:
            result = run_command(command, env)
        except OSError as exc:
            failure = FeatureExtractionError(f"{model} worker launch failed", EXIT_EXECUTION)
            _record_failure_without_masking(
                run_root, state=state, stage="worker_launch",
                model=by_model[model], shard=plans[min(progress[model], len(plans) - 1)],
                exception_class="WorkerLaunchError",
                sanitized_message="worker_launch_failed", original=failure,
            )
            _progress_marker("model_failure", model, "worker_launch_failed", exit_code=EXIT_EXECUTION)
            raise failure from exc
        if result.returncode != 0:
            failure = FeatureExtractionError(f"{model} worker failed", EXIT_EXECUTION)
            _record_failure_without_masking(
                run_root, state=state, stage="worker_process",
                model=by_model[model], shard=plans[min(progress[model], len(plans) - 1)],
                exception_class="WorkerProcessFailure", raw_return_code=result.returncode,
                sanitized_message=_bounded_worker_failure_diagnostic(
                    result.stdout, result.returncode,
                ),
                original=failure,
            )
            _progress_marker("model_failure", model, "worker_process_failed", exit_code=EXIT_EXECUTION)
            raise failure
        try:
            observed = reconcile_resume(run_root, state, profile)
            if observed[model] != len(plans):
                raise FeatureExtractionError("worker artifact reconciliation incomplete", EXIT_EXECUTION)
            progress = observed
        except FeatureExtractionError as exc:
            failure = FeatureExtractionError(
                f"{model} worker artifact reconciliation failed", EXIT_EXECUTION,
            )
            _record_failure_without_masking(
                run_root, state=state, stage="worker_artifact_reconciliation",
                model=by_model[model], shard=plans[min(progress[model], len(plans) - 1)],
                exception_class="WorkerArtifactReconciliationError",
                sanitized_message="worker_artifact_reconciliation_failed",
                original=failure,
            )
            _progress_marker("model_failure", model, "worker_artifact_reconciliation_failed", exit_code=EXIT_EXECUTION)
            raise failure from exc
        _progress_marker("model_complete", model, "execution_complete", exit_code=EXIT_SUCCESS)
    try:
        candidate = complete_run(run_root, profile, publish_manifest=False)
    except FeatureExtractionError as exc:
        _record_failure_without_masking(
            run_root, state=state, stage="completion",
            exception_class="CompletionError", sanitized_message="completion_failed",
            original=exc,
        )
        raise
    try:
        result = validate_completion_candidate(run_root, profile, candidate)
    except FeatureExtractionError as exc:
        _record_failure_without_masking(
            run_root, state=state, stage="model_free_validation",
            exception_class="ModelFreeValidationError",
            sanitized_message="model_free_validation_failed", original=exc,
        )
        raise
    try:
        atomic_write_json(run_root / "run_manifest.json", candidate)
    except FeatureExtractionError as exc:
        _record_failure_without_masking(
            run_root, state=state, stage="completion",
            exception_class="RunManifestPublicationError",
            sanitized_message="run_manifest_publication_failed", original=exc,
        )
        raise
    _emit(result)
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "_worker":
            return _worker(args)
        profile = load_extraction_profile(args.profile)
        if args.command == "plan":
            result = execution_plan(profile, args.repository_root)
            _emit(result, args.output_json)
            return 0
        if args.command == "preflight":
            result = _preflight(
                profile, Path(args.repository_root), Path(args.dataset_root),
                Path(args.cache_root),
            )
            _emit(result, args.output_json)
            return 0
        if args.command == "run":
            return _run_or_resume(args, resume=False)
        if args.command == "resume":
            return _run_or_resume(args, resume=True)
        result = validate_run(args.run_root, profile, allow_incomplete=args.allow_incomplete)
        _emit(result)
        return 0
    except FeatureExtractionError as exc:
        _emit_error(
            {"error": str(exc), "exit_code": exc.exit_code, "ready_for_p1_extraction": False, "schema_id": "vea.p1.feature_extraction_error.v0"},
            getattr(args, "output_json", None),
        )
        return exc.exit_code
    except (OSError, ValueError) as exc:
        _emit_error(
            {"error": str(exc), "exit_code": 4, "ready_for_p1_extraction": False, "schema_id": "vea.p1.feature_extraction_error.v0"},
            getattr(args, "output_json", None),
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
