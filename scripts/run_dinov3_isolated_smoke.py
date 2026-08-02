"""Run the deterministic DINOv3 persistent-runtime process/artifact smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

import numpy as np

from vision_encoder_alignment.feature_artifacts import (
    ARTIFACT_SCHEMA_ID,
    FeatureArtifactMetadata,
    canonical_tensor_keys,
    read_feature_artifact,
    sha256_file,
    verify_sha256,
    write_feature_artifact,
)
from vision_encoder_alignment.runtime_boundary import (
    WORKER_REQUEST_SCHEMA_ID,
    RuntimeProfile,
    WorkerRequest,
    invoke_isolated_worker,
    load_runtime_profile,
    resolve_default_runtime_root,
    validate_runtime_root,
)


IMAGE_IDS = ("image-b", "image-a")
TAP_NAMES = ("embedding_output", "block_06")
PATCH_SHAPE = (2, 4, 3)
GLOBAL_SHAPE = (2, 3)
CREATED_AT_UTC = "2026-07-20T00:00:00Z"

_INBOUND_VALUE_HASHES = (
    (
        "7eb4ca01077988fa5de9ffff007bd66097cc410ee7b919afecb87a2554a85f90",
        "3f19a90ad27c0b4dee2ef959aa840a55530e34f4b4742ca1fbed21eda7d991c9",
    ),
    (
        "eb897d1b2e4dafb832512bffc8e38e037df31e61262833142062b7329f8ebe1c",
        "e6b72e4fc33d85501da61775976d1601ab74ff8227b26646b61fceccd1ec6ccb",
    ),
)
_INBOUND_FILE_HASHES = (
    "31c2f1c7aab1bfddd9aa025a406c8425418155ff7b57339365b5139fe1ea69df",
    "ac88cb5d2a8a8a28779ec6444270db0e8352953ff2edd8a7d634d15c60268865",
)
_OUTBOUND_VALUE_HASHES = (
    (
        "45a99655901702d55ab6284a18aed6a5e16677181d16c7a7517b68c2ae2c0c7a",
        "a8ed788140a07f4db935b3205c81a27c38be1aa885bf659220704528a591e010",
    ),
    (
        "1b089fcdd28fee09782f771ed7ac440a0f171296f270978c79d0bd979f615367",
        "45cab7d97e645faad748045c80472b7c176d441272669c941a108dfc9294f014",
    ),
)
_OUTBOUND_FILE_HASHES = (
    "19621cacf073aa0433e5e502472bdb968c467427e2be390f363caa475746fdfa",
    "c9115946aab2a02d1d2cc6081aaad01589c7809efb1ba27b31f08e6981b8b533",
)


class SmokeValidationError(RuntimeError):
    """Raised when the process/artifact smoke violates its exact contract."""


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _synthetic_tensors(offset: int) -> tuple[np.ndarray, np.ndarray]:
    base = np.arange(24, dtype=np.float32).reshape(PATCH_SHAPE)
    patch = np.ascontiguousarray(base + np.float32(offset), dtype=np.float32)
    global_mean = np.ascontiguousarray(
        patch.mean(axis=1, dtype=np.float32), dtype=np.float32
    )
    return patch, global_mean


def _artifact_inventory(root: Path) -> tuple[tuple[str, int, str], ...]:
    if not root.exists():
        return ()
    return tuple(
        sorted(
            (
                path.relative_to(root).as_posix(),
                path.stat().st_size,
                sha256_file(path),
            )
            for path in root.rglob("*")
            if path.is_file()
        )
    )


def _write_inbound_artifacts(directory: Path, run_id: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for shard_index, (tap_name, offset) in enumerate(
        zip(TAP_NAMES, (1000, 1100), strict=True)
    ):
        patch, global_mean = _synthetic_tensors(offset)
        patch_key, global_key = canonical_tensor_keys(tap_name)
        sidecar_name = f"shard-{shard_index:05d}.json"
        tensor_name = f"shard-{shard_index:05d}.safetensors"
        metadata = FeatureArtifactMetadata(
            schema_id=ARTIFACT_SCHEMA_ID,
            run_id=run_id,
            model_key="dinov3_vit_b16",
            tap_namespace="backbone",
            tap_name=tap_name,
            patch_tensor_key=patch_key,
            global_mean_tensor_key=global_key,
            image_ids=list(IMAGE_IDS),
            grid_height=2,
            grid_width=2,
            patch_count=4,
            hidden_dimension=3,
            dtype="float32",
            shard_index=shard_index,
            safetensors_file_name=tensor_name,
            sidecar_file_name=sidecar_name,
            checksum_algorithm="sha256",
            checksum_value="",
            created_at_utc=CREATED_AT_UTC,
            run_status="incomplete",
        )
        completed = write_feature_artifact(directory, metadata, patch, global_mean)
        tensor_path = directory / completed.safetensors_file_name
        if completed.checksum_value != _INBOUND_FILE_HASHES[shard_index]:
            raise SmokeValidationError("inbound Safetensors checksum mismatch")
        entries.append(
            {
                "checksum": completed.checksum_value,
                "tap_name": tap_name,
                "patch_values_sha256": _sha256_array(patch),
                "global_values_sha256": _sha256_array(global_mean),
                "sidecar": completed.sidecar_file_name,
                "tensor": completed.safetensors_file_name,
            }
        )
        if entries[-1]["patch_values_sha256"] != _INBOUND_VALUE_HASHES[shard_index][0]:
            raise SmokeValidationError("inbound patch-value SHA-256 mismatch")
        if entries[-1]["global_values_sha256"] != _INBOUND_VALUE_HASHES[shard_index][1]:
            raise SmokeValidationError("inbound global-value SHA-256 mismatch")
        if sha256_file(tensor_path) != completed.checksum_value:
            raise SmokeValidationError("inbound file SHA-256 mismatch")
    return {
        "image_ids": list(IMAGE_IDS),
        "tap_order": list(TAP_NAMES),
        "patch_shape": list(PATCH_SHAPE),
        "global_shape": list(GLOBAL_SHAPE),
        "dtype": "float32",
        "entries": entries,
    }


def _validate_outbound_artifacts(directory: Path, run_id: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for shard_index, (tap_name, offset) in enumerate(
        zip(TAP_NAMES, (0, 100), strict=True)
    ):
        sidecar_name = f"shard-{shard_index:05d}.json"
        loaded = read_feature_artifact(directory / sidecar_name)
        patch, global_mean = _synthetic_tensors(offset)
        patch_key, global_key = canonical_tensor_keys(tap_name)
        metadata = loaded.metadata
        expected = {
            "schema_id": ARTIFACT_SCHEMA_ID,
            "run_id": run_id,
            "model_key": "dinov3_vit_b16",
            "tap_namespace": "backbone",
            "tap_name": tap_name,
            "patch_tensor_key": patch_key,
            "global_mean_tensor_key": global_key,
            "image_ids": list(IMAGE_IDS),
            "grid_height": 2,
            "grid_width": 2,
            "patch_count": 4,
            "hidden_dimension": 3,
            "dtype": "float32",
            "shard_index": shard_index,
            "safetensors_file_name": f"shard-{shard_index:05d}.safetensors",
            "sidecar_file_name": sidecar_name,
            "checksum_algorithm": "sha256",
            "checksum_value": _OUTBOUND_FILE_HASHES[shard_index],
            "created_at_utc": CREATED_AT_UTC,
            "run_status": "complete",
        }
        if metadata.to_dict() != expected:
            raise SmokeValidationError("outbound metadata mismatch")
        np.testing.assert_array_equal(loaded.patch, patch)
        np.testing.assert_array_equal(loaded.global_mean, global_mean)
        np.testing.assert_array_equal(
            loaded.global_mean, loaded.patch.mean(axis=1, dtype=np.float32)
        )
        tensor_path = directory / metadata.safetensors_file_name
        result = verify_sha256(tensor_path, metadata.checksum_value)
        if result is not None:
            raise SmokeValidationError("verify_sha256 success must return None")
        patch_hash = _sha256_array(loaded.patch)
        global_hash = _sha256_array(loaded.global_mean)
        if patch_hash != _OUTBOUND_VALUE_HASHES[shard_index][0]:
            raise SmokeValidationError("outbound patch-value SHA-256 mismatch")
        if global_hash != _OUTBOUND_VALUE_HASHES[shard_index][1]:
            raise SmokeValidationError("outbound global-value SHA-256 mismatch")
        entries.append(
            {
                "checksum": metadata.checksum_value,
                "tap_name": tap_name,
                "patch_values_sha256": patch_hash,
                "global_values_sha256": global_hash,
                "sidecar": sidecar_name,
                "tensor": metadata.safetensors_file_name,
            }
        )
    return {
        "image_ids": list(IMAGE_IDS),
        "tap_order": list(TAP_NAMES),
        "patch_shape": list(PATCH_SHAPE),
        "global_shape": list(GLOBAL_SHAPE),
        "dtype": "float32",
        "entries": entries,
    }


def _repository_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or len(value) != 40:
        raise SmokeValidationError("unable to read repository HEAD")
    return value


def _response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "to_dict"):
        value = response.to_dict()
    elif isinstance(response, dict):
        value = dict(response)
    else:
        raise SmokeValidationError("worker returned an unsupported response object")
    if not isinstance(value, dict):
        raise SmokeValidationError("worker response serialization is invalid")
    return value


def _canonical_repeat_projection(
    response: dict[str, Any], inbound: dict[str, Any], outbound: dict[str, Any]
) -> bytes:
    value = {
        "response": response,
        "inbound": inbound,
        "outbound": outbound,
    }
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def run_boundary_smoke(
    *,
    profile_path: Path,
    runtime_root: Path | None = None,
    repository_root: Path | None = None,
    worker_script: Path | None = None,
    operation: str = "runtime_self_test",
    run_id: str = "t007a2-boundary-smoke",
    timeout_seconds: float = 600.0,
    profile_loader: Callable[[str | Path], RuntimeProfile] = load_runtime_profile,
    runtime_validator: Callable[..., Any] = validate_runtime_root,
    worker_invoker: Callable[..., Any] = invoke_isolated_worker,
    head_reader: Callable[[Path], str] = _repository_head,
) -> dict[str, Any]:
    """Run the boundary twice and return path-free deterministic evidence."""

    repository = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    profile_file = profile_path.resolve(strict=True)
    profile = profile_loader(profile_file)
    runtime = (
        resolve_default_runtime_root(profile)
        if runtime_root is None
        else runtime_root
    ).resolve(strict=True)
    worker = (
        repository / "scripts" / "dinov3_isolated_worker.py"
        if worker_script is None
        else worker_script
    ).resolve(strict=True)
    manifest = runtime_validator(runtime, profile, repository_root=repository)
    artifacts_root = repository / "artifacts"
    artifacts_before = _artifact_inventory(artifacts_root)
    runtime_before = _artifact_inventory(runtime)
    repository_head = head_reader(repository)

    workspace_path: Path | None = None
    repetitions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vea_t007a2_boundary_") as temporary:
        workspace_path = Path(temporary).resolve(strict=True)
        for repeat_index in range(2):
            repeat_root = workspace_path / f"repeat-{repeat_index}"
            inbound_directory = repeat_root / "inbound"
            outbound_directory = repeat_root / "outbound"
            repeat_root.mkdir()
            inbound = _write_inbound_artifacts(inbound_directory, run_id)
            request = WorkerRequest(
                schema_id=WORKER_REQUEST_SCHEMA_ID,
                operation=operation,
                run_id=run_id,
                repository_head=repository_head,
                runtime_profile_sha256=profile.source_sha256,
                workspace_relative_inbound_artifact_path="inbound",
                workspace_relative_outbound_artifact_path="outbound",
                deterministic_seed=0,
                expected_preprocessing_profile_id=(
                    profile.manual_preprocessing_profile_id
                ),
            )
            response_object = worker_invoker(
                runtime_root=runtime,
                repository_root=repository,
                profile_path=profile_file,
                worker_script=worker,
                workspace_root=repeat_root,
                request=request,
                timeout_seconds=timeout_seconds,
            )
            response = _response_dict(response_object)
            if response.get("status") not in {"ok", "success"}:
                raise SmokeValidationError("worker did not report success")
            if response.get("run_id") != run_id or response.get("operation") != operation:
                raise SmokeValidationError("worker response correlation mismatch")
            if response.get("socket_attempt_count") != 0:
                raise SmokeValidationError("worker reported a network attempt")
            if response.get("model_weight_byte_count") != 0:
                raise SmokeValidationError("worker reported model-weight bytes")
            if response.get("torchvision_use_count") != 0:
                raise SmokeValidationError("worker reported forbidden vision use")
            outbound = _validate_outbound_artifacts(outbound_directory, run_id)
            if response.get("inbound_checksums") not in (
                None,
                [entry["checksum"] for entry in inbound["entries"]],
            ):
                raise SmokeValidationError("worker inbound checksums mismatch")
            if response.get("outbound_checksums") not in (
                None,
                [entry["checksum"] for entry in outbound["entries"]],
            ):
                raise SmokeValidationError("worker outbound checksums mismatch")
            repetitions.append(
                {
                    "response": response,
                    "inbound": inbound,
                    "outbound": outbound,
                    "projection_sha256": _sha256_array(
                        np.frombuffer(
                            _canonical_repeat_projection(response, inbound, outbound),
                            dtype=np.uint8,
                        )
                    ),
                }
            )
        if repetitions[0]["projection_sha256"] != repetitions[1]["projection_sha256"]:
            raise SmokeValidationError("repeated boundary result is not deterministic")

    if workspace_path is None or workspace_path.exists():
        raise SmokeValidationError("worker workspace cleanup failed")
    if _artifact_inventory(artifacts_root) != artifacts_before:
        raise SmokeValidationError("repository artifacts changed during smoke")
    if _artifact_inventory(runtime) != runtime_before:
        raise SmokeValidationError("persistent runtime changed during smoke")
    runtime_validator(runtime, profile, repository_root=repository)

    markers = list(repetitions[0]["response"].get("markers", ()))
    markers.extend(
        [
            "DINOV3_PERSISTENT_REPEAT_DETERMINISM_OK",
            "DINOV3_PERSISTENT_RUNTIME_BOUNDARY_OK",
        ]
    )
    return {
        "status": "PASS",
        "runtime_id": profile.runtime_id,
        "runtime_manifest_sha256": getattr(manifest, "source_sha256", None),
        "operation": operation,
        "repeat_count": 2,
        "repeat_deterministic": True,
        "workspace_cleaned": True,
        "repository_artifacts_unchanged": True,
        "persistent_runtime_unchanged": True,
        "projection_sha256": repetitions[0]["projection_sha256"],
        "inbound": repetitions[0]["inbound"],
        "outbound": repetitions[0]["outbound"],
        "worker_response": repetitions[0]["response"],
        "markers": markers,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        evidence = run_boundary_smoke(
            profile_path=args.profile,
            runtime_root=args.runtime_root,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"DINOV3_PERSISTENT_RUNTIME_BOUNDARY_FAILED:{type(exc).__name__}", file=sys.stderr)
        return 1
    for marker in evidence["markers"]:
        print(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
