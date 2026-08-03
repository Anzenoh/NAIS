"""Strict, offline Phase 1 feature-extraction planning and artifact boundary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, NoReturn


PROFILE_SCHEMA_ID = "vea.p1.feature_extraction_profile.v0"
PLAN_SCHEMA_ID = "vea.p1.feature_extraction_plan.v0"
PREFLIGHT_SCHEMA_ID = "vea.p1.feature_extraction_preflight.v0"
SHARD_SCHEMA_ID = "vea.p1.feature_shard.v0"
MODEL_MANIFEST_SCHEMA_ID = "vea.p1.feature_model_manifest.v0"
RUN_STATE_SCHEMA_ID = "vea.p1.feature_run_state.v0"
RUN_MANIFEST_SCHEMA_ID = "vea.p1.feature_run_manifest.v0"
VALIDATION_SCHEMA_ID = "vea.p1.feature_validation.v0"
ENVIRONMENT_SCHEMA_ID = "vea.p1.feature_environment.v0"
EXECUTION_METRICS_SCHEMA_ID = "vea.p1.feature_execution_metrics.v0"
FAILURE_SCHEMA_ID = "vea.p1.feature_failure.v0"
FAILURE_STAGES = (
    "worker_launch", "worker_process", "worker_artifact_reconciliation",
    "completion", "model_free_validation",
)
WORKER_FAILURE_STAGES = FAILURE_STAGES[:3]
ORCHESTRATION_FAILURE_STAGES = FAILURE_STAGES[3:]

MODEL_ORDER = ("clip_vit_b16", "siglip2_base_p16_224", "dinov3_vit_b16")
GLOBAL_TAPS = (
    "embedding_output", "block_01", "block_02", "block_03", "block_04",
    "block_05", "block_06", "block_07", "block_08", "block_09",
    "block_10", "block_11", "block_12", "final_norm",
)
PATCH_TAPS = ("block_03", "block_06", "block_09", "block_12")
PATCH_DEPTHS = (0.25, 0.5, 0.75, 1.0)
EXPECTED_GRID = (14, 14)
EXPECTED_PATCH_COUNT = 196
EXPECTED_HIDDEN_SIZE = 768
SHARD_IMAGE_COUNT = 25
PROFILE_RELATIVE_PATH = "configs/extraction_profiles/p1_1000_apple_fp32_v0.json"
DEFAULT_OUTPUT_PARENT = "/home/chltjsdh/vea/artifacts/p1_t003"
RUN_ID_RE = re.compile(
    r"^[0-9]{8}T[0-9]{6}Z_p1_1000_three_encoder_[0-9a-f]{7,12}$"
)
RUN_ID_PATTERN = r"^[0-9]{8}T[0-9]{6}Z_p1_1000_three_encoder_[0-9a-f]{7,12}$"
PRODUCTION_SOURCE_BASELINE = "e518a51ba83f70b42611cc4386a9f55dc9e97de3"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SENSITIVE_RE = re.compile(r"(?:token|secret|password|credential|api[_-]?key)", re.I)

EXIT_SUCCESS = 0
EXIT_NOT_READY = 2
EXIT_CONTRACT = 3
EXIT_RESOURCE = 4
EXIT_EXECUTION = 5
EXIT_ARTIFACT = 6


class FeatureExtractionError(RuntimeError):
    """Fail-closed error carrying a stable CLI exit classification."""

    def __init__(self, message: str, exit_code: int = EXIT_CONTRACT) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _fail(message: str, exit_code: int = EXIT_CONTRACT) -> NoReturn:
    raise FeatureExtractionError(message, exit_code)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON value: {value}")


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FeatureExtractionError("value is not finite JSON") from exc


def strict_json_bytes(raw: bytes, *, label: str, canonical: bool = True) -> dict[str, Any]:
    if not raw or raw.startswith(b"\xef\xbb\xbf"):
        _fail(f"{label}: empty JSON or UTF-8 BOM")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except FeatureExtractionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise FeatureExtractionError(f"{label}: malformed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        _fail(f"{label}: expected an object")
    if canonical and raw != canonical_json_bytes(value):
        _fail(f"{label}: noncanonical JSON bytes")
    return value


def load_strict_json(path: str | Path, *, label: str, canonical: bool = True) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"{label}: expected a regular non-link file", EXIT_RESOURCE)
    try:
        return strict_json_bytes(candidate.read_bytes(), label=label, canonical=canonical)
    except OSError as exc:
        raise FeatureExtractionError(f"{label}: unavailable", EXIT_RESOURCE) from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FeatureExtractionError("required file is unavailable", EXIT_RESOURCE) from exc
    return digest.hexdigest()


def _exact(value: object, keys: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(f"{path}: missing or unknown fields")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{path}: expected a canonical non-empty string")
    if any(ord(character) < 32 for character in value):
        _fail(f"{path}: control character")
    return value


def _sha(value: object, path: str) -> str:
    text = _string(value, path)
    if SHA_RE.fullmatch(text) is None:
        _fail(f"{path}: expected lowercase SHA-256")
    return text


def safe_relative_path(value: object, path: str = "path") -> str:
    text = _string(value, path)
    posix, windows = PurePosixPath(text), PureWindowsPath(text)
    if (
        "\\" in text or posix.is_absolute() or windows.is_absolute()
        or windows.drive or posix.as_posix() != text
        or any(part in {"", ".", ".."} or ":" in part for part in text.split("/"))
    ):
        _fail(f"{path}: unsafe relative path")
    return text


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ExtractionProfile:
    source_path: Path
    source_sha256: str
    document: Mapping[str, Any]

    @property
    def models(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.document["models"])

    @property
    def shard_image_count(self) -> int:
        return int(self.document["output"]["shard_image_count"])


@dataclass(frozen=True, slots=True)
class ManifestRow:
    image_id: str
    relative_path: str
    source_sha256: str
    decoded_rgb_sha256: str
    selection_index: int


@dataclass(frozen=True, slots=True)
class ShardPlan:
    index: int
    start: int
    stop: int
    image_ids: tuple[str, ...]

    @property
    def basename(self) -> str:
        return f"shard-{self.index:05d}"


_PROFILE_KEYS = {
    "canonical_geometry", "checkpoint_plan", "dataset", "execution", "models",
    "output", "policies", "profile_id", "runtime", "schema_id",
    "source_bindings", "tensors",
}

EXPECTED_MODELS = (
    {"lane_id": "primary_lane", "model_id": "openai/clip-vit-base-patch16", "registry_key": "clip_vit_b16", "revision": "57c216476eefef5ab752ec549e440a49ae4ae5f3"},
    {"lane_id": "primary_lane", "model_id": "google/siglip2-base-patch16-224", "registry_key": "siglip2_base_p16_224", "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"},
    {"lane_id": "dinov3_isolated_lane", "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m", "registry_key": "dinov3_vit_b16", "revision": "5931719e67bbdb9737e363e781fb0c67687896bc"},
)
EXPECTED_GEOMETRY = {
    "antialiasing": "pillow_bicubic_implicit_v0", "channel_order": "RGB",
    "crop_height": 224, "crop_width": 224,
    "geometry_id": "canonical_224_center_crop_v0", "resize_shorter_side": 224,
}
EXPECTED_RUNTIME = {
    "dinov3": {
        "extractor_profile": {"path": "configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json", "sha256": "74ba6f23d31910450c53e6b35bba6218e78f0edd281c9069593ea58faee62645"},
        "interpreter": "/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated/bin/python",
        "profile": {"path": "configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json", "profile_id": "dinov3-apple-cuda-fp32-v1", "sha256": "86069cd7396c7d254d13c63b4e5ea0cf0120a0a31f181c708945273d15825c15"},
    },
    "main": {
        "interpreter": "/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3",
        "profile": {"path": "configs/runtime_profiles/p1_apple_main_cuda_fp32_v0.json", "profile_id": "p1-apple-main-cuda-fp32-v0", "sha256": "7e578ea3451e595e243e0e6a39058b0eb12a8641ce433bc60b6039b23e2cbdf5"},
    },
}
EXPECTED_SOURCE_BINDINGS = (
    {"path": "scripts/dinov3_extraction_worker.py", "sha256": "d1e9ff2b61f9feab120e464966ac106efb7373aa03be2807d402ed6f48e8a1c5"},
    {"path": "src/vision_encoder_alignment/dinov3_extraction_boundary.py", "sha256": "38cc3030215ed419cc90b16ee5e4562c99fe2f978276cb8e22e9e4f6b0583287"},
    {"path": "src/vision_encoder_alignment/extractors/clip.py", "sha256": "0ec0980c9761a73597244b03116de093c71b568cc0d8755951821b4f448eb82d"},
    {"path": "src/vision_encoder_alignment/extractors/dinov3.py", "sha256": "3527d5645e98c66f01c1a42b8eb1415055ba89d81bcbc5b5d087d1526449cb10"},
    {"path": "src/vision_encoder_alignment/extractors/siglip2.py", "sha256": "0267d9eaa51d6c6b686339d9d9e700f782d2ec3b5c16c20d3df16e8fa17cfd5e"},
    {"path": "src/vision_encoder_alignment/runtime_boundary.py", "sha256": "cb72edbf0f889991cac164703892a4e2d6c52b062b07ef193668da5c08f560cc"},
)


def load_extraction_profile(path: str | Path) -> ExtractionProfile:
    candidate = validate_non_link_path(
        path, expected_kind="file", label="extraction profile",
    )
    raw = candidate.read_bytes()
    root = _exact(strict_json_bytes(raw, label="extraction_profile"), _PROFILE_KEYS, "profile")
    if root["schema_id"] != PROFILE_SCHEMA_ID or root["profile_id"] != "p1-1000-apple-fp32-v0":
        _fail("profile: unsupported identity")
    dataset = _exact(root["dataset"], {"manifest", "metadata", "profile"}, "profile.dataset")
    expected_dataset = {
        "profile": ("configs/datasets/p1_1000_profile.template.json", "8cc9fc42c39c1df7ddd0547aa9b836116dd798a58e22d541118707c15770f48b"),
        "manifest": ("data/manifests/p1_1000.jsonl", "1819bbc6235dd4805d71642e63e73112ce2e918135ec94f46493077698be92da"),
        "metadata": ("data/manifests/p1_1000.metadata.json", "61d530e4a9928b70fa9382af772706834b43ef7d63be2bbed36dee493815a521"),
    }
    for name, (path_value, digest) in expected_dataset.items():
        keys = {"path", "sha256", "row_count"} if name == "manifest" else {"path", "sha256"}
        row = _exact(dataset[name], keys, f"profile.dataset.{name}")
        if safe_relative_path(row["path"]) != path_value or _sha(row["sha256"], "sha256") != digest:
            _fail(f"profile.dataset.{name}: immutable identity mismatch")
        if name == "manifest" and row["row_count"] != 1000:
            _fail("profile.dataset.manifest.row_count: expected 1000")
    checkpoint = _exact(root["checkpoint_plan"], {"path", "sha256"}, "profile.checkpoint_plan")
    if checkpoint != {
        "path": "configs/model_snapshots/p1_three_encoder_checkpoint_plan.template.json",
        "sha256": "86dcb8072d9e8670a714e2564e06d41dd6e22c75c5ce553b4e7b9e0624647ae6",
    }:
        _fail("profile.checkpoint_plan: immutable identity mismatch")
    models = root["models"]
    if not isinstance(models, list) or models != list(EXPECTED_MODELS):
        _fail("profile.models: exact model ID, revision, lane, and order required")
    if any(REVISION_RE.fullmatch(item["revision"]) is None for item in models):
        _fail("profile.models: exact 40-hex revision required")
    execution = _exact(root["execution"], {"full", "model_order", "sequential_lanes", "smoke"}, "profile.execution")
    if execution["model_order"] != list(MODEL_ORDER) or execution["sequential_lanes"] is not True:
        _fail("profile.execution: model order must be sequential")
    if execution["smoke"] != {"batch_size": 1, "image_count": 10}:
        _fail("profile.execution.smoke: expected 10 images and batch size 1")
    if execution["full"] != {"batch_size": None, "batch_size_state": "unresolved_requires_t003b_acceptance", "image_count": 1000}:
        _fail("profile.execution.full: batch size must remain unresolved")
    tensors = _exact(root["tensors"], {"global_taps", "hidden_size", "patch_count", "patch_grid", "patch_normalized_depths", "patch_taps"}, "profile.tensors")
    if (
        tensors["global_taps"] != list(GLOBAL_TAPS)
        or tensors["patch_taps"] != list(PATCH_TAPS)
        or tuple(tensors["patch_normalized_depths"]) != PATCH_DEPTHS
        or tuple(tensors["patch_grid"]) != EXPECTED_GRID
        or tensors["patch_count"] != EXPECTED_PATCH_COUNT
        or tensors["hidden_size"] != EXPECTED_HIDDEN_SIZE
    ):
        _fail("profile.tensors: tap, depth, shape, or order mismatch")
    if root["canonical_geometry"] != EXPECTED_GEOMETRY:
        _fail("profile.canonical_geometry: exact geometry required")
    policies = _exact(root["policies"], {"compute_dtype", "local_files_only", "mixed_precision", "network_allowed", "ready_for_p1_extraction", "ready_for_p1_t003b", "storage_dtype", "trust_remote_code"}, "profile.policies")
    if policies != {
        "compute_dtype": "float32", "local_files_only": True,
        "mixed_precision": False, "network_allowed": False,
        "ready_for_p1_extraction": False, "ready_for_p1_t003b": False,
        "storage_dtype": "float32", "trust_remote_code": False,
    }:
        _fail("profile.policies: expected fail-closed FP32 policy")
    output = _exact(root["output"], {"no_overwrite", "parent", "run_id_pattern", "shard_image_count"}, "profile.output")
    if output != {"no_overwrite": True, "parent": DEFAULT_OUTPUT_PARENT, "run_id_pattern": RUN_ID_PATTERN, "shard_image_count": SHARD_IMAGE_COUNT}:
        _fail("profile.output: output contract mismatch")
    runtime = _exact(root["runtime"], {"dinov3", "main"}, "profile.runtime")
    if runtime != EXPECTED_RUNTIME:
        _fail("profile.runtime: exact interpreter/profile identity required")
    bindings = root["source_bindings"]
    if not isinstance(bindings, list) or len(bindings) != 6:
        _fail("profile.source_bindings: expected six bindings")
    paths: list[str] = []
    for index, item in enumerate(bindings):
        row = _exact(item, {"path", "sha256"}, f"profile.source_bindings[{index}]")
        paths.append(safe_relative_path(row["path"]))
        _sha(row["sha256"], "source_binding.sha256")
    if bindings != list(EXPECTED_SOURCE_BINDINGS):
        _fail("profile.source_bindings: exact path/hash identity required")
    _reject_sensitive(root)
    return ExtractionProfile(candidate, hashlib.sha256(raw).hexdigest(), _freeze(root))


def _reject_sensitive(value: object, path: str = "profile") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if SENSITIVE_RE.search(str(key)):
                _fail(f"{path}.{key}: sensitive field forbidden")
            _reject_sensitive(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(f"{path}: non-finite")
    elif isinstance(value, str) and (PureWindowsPath(value).is_absolute() or (PurePosixPath(value).is_absolute() and not value.startswith("/home/chltjsdh/vea/"))):
        _fail(f"{path}: unauthorized absolute path")


def verify_repository_inputs(
    profile: ExtractionProfile, repository_root: str | Path, *,
    lstat_probe: Callable[[Path], os.stat_result] = os.lstat,
) -> None:
    root = validate_non_link_path(
        repository_root, expected_kind="directory", label="repository root",
        lstat_probe=lstat_probe,
    )
    rows: list[Mapping[str, Any]] = []
    dataset = profile.document["dataset"]
    rows.extend((dataset["profile"], dataset["manifest"], dataset["metadata"], profile.document["checkpoint_plan"]))
    runtime = profile.document["runtime"]
    rows.extend((runtime["main"]["profile"], runtime["dinov3"]["profile"], runtime["dinov3"]["extractor_profile"]))
    rows.extend(profile.document["source_bindings"])
    for row in rows:
        relative = safe_relative_path(row["path"])
        candidate = root.joinpath(*relative.split("/"))
        candidate = validate_non_link_path(
            candidate, expected_kind="file", label=f"repository input {relative}",
            lstat_probe=lstat_probe,
        )
        if not candidate.is_relative_to(root):
            _fail(f"repository input unavailable or unsafe: {relative}", EXIT_RESOURCE)
        if sha256_file(candidate) != row["sha256"]:
            _fail(f"repository input SHA mismatch: {relative}")


def stream_manifest(path: str | Path, *, expected_sha256: str, expected_rows: int = 1000) -> tuple[ManifestRow, ...]:
    candidate = validate_non_link_path(
        path, expected_kind="file", label="dataset manifest",
    )
    if sha256_file(candidate) != expected_sha256:
        _fail("dataset manifest SHA mismatch")
    rows: list[ManifestRow] = []
    try:
        with candidate.open("rb") as stream:
            for index, raw in enumerate(stream):
                if not raw.endswith(b"\n") or b"\r" in raw or raw == b"\n":
                    _fail(f"manifest row {index}: noncanonical JSONL")
                value = strict_json_bytes(raw, label=f"manifest row {index}")
                if value.get("selection_index") != index:
                    _fail(f"manifest row {index}: selection order mismatch")
                rows.append(ManifestRow(
                    image_id=_string(value.get("image_id"), "image_id"),
                    relative_path=safe_relative_path(value.get("relative_path")),
                    source_sha256=_sha(value.get("source_sha256"), "source_sha256"),
                    decoded_rgb_sha256=_sha(value.get("decoded_rgb_sha256"), "decoded_rgb_sha256"),
                    selection_index=index,
                ))
    except OSError as exc:
        raise FeatureExtractionError("dataset manifest read failed", EXIT_RESOURCE) from exc
    if len(rows) != expected_rows or len({row.image_id for row in rows}) != expected_rows:
        _fail(f"manifest: expected {expected_rows} unique ordered rows")
    return tuple(rows)


def ordered_batches(rows: Sequence[ManifestRow], batch_size: int) -> tuple[tuple[ManifestRow, ...], ...]:
    if type(batch_size) is not int or batch_size <= 0:
        _fail("batch size must be a positive integer")
    return tuple(tuple(rows[start:start + batch_size]) for start in range(0, len(rows), batch_size))


def plan_shards(image_ids: Sequence[str], shard_size: int = SHARD_IMAGE_COUNT) -> tuple[ShardPlan, ...]:
    if type(shard_size) is not int or shard_size <= 0:
        _fail("shard size must be positive")
    ids = tuple(image_ids)
    if len(ids) != len(set(ids)) or not ids:
        _fail("image IDs must be non-empty and unique")
    return tuple(
        ShardPlan(index, start, min(start + shard_size, len(ids)), ids[start:start + shard_size])
        for index, start in enumerate(range(0, len(ids), shard_size))
    )


def validate_mode(profile: ExtractionProfile, mode: str, batch_size: int | None) -> int:
    if mode == "smoke":
        if batch_size != 1:
            _fail("smoke mode requires batch size 1")
        return 10
    if mode != "full":
        _fail("mode must be smoke or full")
    if type(batch_size) is not int or batch_size <= 0:
        _fail("full mode requires an explicitly approved positive batch size", EXIT_NOT_READY)
    return 1000


def lane_command_plan(profile: ExtractionProfile, repository_root: str | Path) -> tuple[tuple[str, ...], ...]:
    root = str(Path(repository_root).resolve())
    runtime = profile.document["runtime"]
    main = runtime["main"]["interpreter"]
    return (
        (main, "-B", f"{root}/scripts/extract_features.py", "_worker", "--model", MODEL_ORDER[0]),
        (main, "-B", f"{root}/scripts/extract_features.py", "_worker", "--model", MODEL_ORDER[1]),
        (main, "-B", f"{root}/scripts/extract_features.py", "_worker", "--model", MODEL_ORDER[2]),
    )


def _approved_repository_commit(repository_root: str | Path | None = None) -> str:
    approved = os.environ.get("VEA_APPROVED_REPO_COMMIT", "").lower()
    if REVISION_RE.fullmatch(approved) is None:
        _fail("VEA_APPROVED_REPO_COMMIT is missing or noncanonical")
    if repository_root is not None:
        result = subprocess.run(
            ["git", "-C", str(Path(repository_root).resolve()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        actual = result.stdout.strip().lower()
        if result.returncode != 0 or actual != approved:
            _fail("approved repository commit does not match HEAD")
    return approved


def offline_environment(repository_root: str | Path) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "0", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PIP_NO_INDEX": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path(repository_root).resolve() / "src"),
        "TRANSFORMERS_OFFLINE": "1",
        "VEA_APPROVED_REPO_COMMIT": _approved_repository_commit(repository_root),
    }


def execution_plan(profile: ExtractionProfile, repository_root: str | Path) -> dict[str, Any]:
    verify_repository_inputs(profile, repository_root)
    return {
        "commands": [list(command) for command in lane_command_plan(profile, repository_root)],
        "full_batch_size": None,
        "full_batch_size_state": "unresolved_requires_t003b_acceptance",
        "global_taps": list(GLOBAL_TAPS), "model_order": list(MODEL_ORDER),
        "network_allowed": False, "patch_taps": list(PATCH_TAPS),
        "profile_id": profile.document["profile_id"],
        "profile_sha256": profile.source_sha256,
        "ready_for_p1_extraction": False, "ready_for_p1_t003b": False,
        "schema_id": PLAN_SCHEMA_ID, "shard_image_count": SHARD_IMAGE_COUNT,
        "smoke_batch_size": 1, "smoke_image_count": 10,
    }


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        junction = getattr(path, "is_junction", None)
        return bool((junction is not None and junction()) or getattr(path.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except OSError:
        return False


def _stat_is_link_or_reparse(value: os.stat_result) -> bool:
    return bool(
        stat.S_ISLNK(value.st_mode)
        or getattr(value, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def validate_non_link_path(
    path: str | Path, *, expected_kind: str, label: str,
    lstat_probe: Callable[[Path], os.stat_result] = os.lstat,
) -> Path:
    """Validate every existing component without resolving through links.

    This narrows path substitution risk but does not claim to eliminate TOCTOU.
    The caller must still use atomic no-clobber publication for outputs.
    """
    raw = os.fspath(path)
    candidate = Path(raw)
    if (
        expected_kind not in {"directory", "file"}
        or not candidate.is_absolute()
        or raw != os.path.normpath(raw)
    ):
        _fail(f"{label}: expected a canonical absolute path", EXIT_RESOURCE)
    current = Path(candidate.anchor)
    try:
        root_stat = lstat_probe(current)
        if _stat_is_link_or_reparse(root_stat):
            _fail(f"{label}: link or reparse component rejected", EXIT_RESOURCE)
        for component in candidate.parts[1:]:
            current = current / component
            observed = lstat_probe(current)
            if _stat_is_link_or_reparse(observed):
                _fail(f"{label}: link or reparse component rejected", EXIT_RESOURCE)
    except FeatureExtractionError:
        raise
    except OSError as exc:
        raise FeatureExtractionError(f"{label}: path component unavailable", EXIT_RESOURCE) from exc
    lexical = Path(os.path.abspath(raw))
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FeatureExtractionError(f"{label}: unavailable", EXIT_RESOURCE) from exc
    if resolved != lexical:
        _fail(f"{label}: canonical resolution mismatch", EXIT_RESOURCE)
    if expected_kind == "directory" and not stat.S_ISDIR(lstat_probe(candidate).st_mode):
        _fail(f"{label}: expected directory", EXIT_RESOURCE)
    if expected_kind == "file" and not stat.S_ISREG(lstat_probe(candidate).st_mode):
        _fail(f"{label}: expected regular file", EXIT_RESOURCE)
    return candidate


def validate_output_root(
    output_parent: str | Path, run_id: str, *, protected_roots: Sequence[str | Path],
    require_parent: str | None = None,
    lstat_probe: Callable[[Path], os.stat_result] = os.lstat,
) -> Path:
    if RUN_ID_RE.fullmatch(run_id) is None:
        _fail("run ID is noncanonical")
    resolved_parent = validate_non_link_path(
        output_parent, expected_kind="directory", label="output parent",
        lstat_probe=lstat_probe,
    )
    if require_parent is not None and resolved_parent.as_posix() != require_parent:
        _fail("output parent does not match the production policy")
    final = resolved_parent / run_id
    if final.exists() or final.is_symlink():
        _fail("run root already exists")
    for raw in protected_roots:
        protected_path = Path(raw)
        if not protected_path.exists():
            _fail("protected root is unavailable", EXIT_RESOURCE)
        protected = validate_non_link_path(
            protected_path, expected_kind="directory", label="protected root",
            lstat_probe=lstat_probe,
        )
        if final == protected or final.is_relative_to(protected) or protected.is_relative_to(final):
            _fail("output run root overlaps a protected root")
    return final


def atomic_write(path: str | Path, payload: bytes) -> Path:
    destination = Path(path)
    parent = validate_non_link_path(
        destination.parent, expected_kind="directory", label="publication parent",
    )
    if (
        not destination.is_absolute()
        or os.fspath(destination) != os.path.normpath(os.fspath(destination))
        or destination.parent != parent
    ):
        _fail("publication destination is noncanonical", EXIT_RESOURCE)
    if destination.exists() or destination.is_symlink():
        _fail(f"refusing to overwrite: {destination.name}")
    if not destination.parent.is_dir():
        _fail("output parent is unavailable", EXIT_RESOURCE)
    stage: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        stage = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(stage, destination)
        except FileExistsError as exc:
            _fail(f"refusing to overwrite: {destination.name}")
        except OSError as exc:
            raise FeatureExtractionError("atomic no-clobber publication unavailable", EXIT_RESOURCE) from exc
        stage.unlink(); stage = None
        return destination
    finally:
        if stage is not None:
            stage.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return atomic_write(path, canonical_json_bytes(value))


def tensor_keys() -> tuple[str, ...]:
    return tuple(
        [f"backbone.{tap}.global_mean" for tap in GLOBAL_TAPS]
        + [f"backbone.{tap}.patch" for tap in PATCH_TAPS]
    )


def validate_tensor_mapping(tensors: Mapping[str, Any], image_count: int) -> None:
    import numpy as np
    if tuple(tensors) != tensor_keys() and set(tensors) != set(tensor_keys()):
        _fail("unexpected tensor key set", EXIT_ARTIFACT)
    for key in tensor_keys():
        value = tensors[key]
        expected = (image_count, EXPECTED_PATCH_COUNT, EXPECTED_HIDDEN_SIZE) if key.endswith(".patch") else (image_count, EXPECTED_HIDDEN_SIZE)
        if not isinstance(value, np.ndarray) or value.shape != expected or value.dtype != np.float32:
            _fail(f"tensor shape/dtype mismatch: {key}", EXIT_ARTIFACT)
        if not np.isfinite(value).all():
            _fail(f"non-finite tensor: {key}", EXIT_ARTIFACT)
    for tap in PATCH_TAPS:
        patch = tensors[f"backbone.{tap}.patch"]
        global_mean = tensors[f"backbone.{tap}.global_mean"]
        if not np.allclose(global_mean, patch.mean(axis=1, dtype=np.float32), rtol=1e-5, atol=1e-6):
            _fail(f"global mean mismatch: {tap}", EXIT_ARTIFACT)


def write_shard(
    directory: str | Path, *, run_id: str, model: Mapping[str, Any], lane_id: str,
    shard: ShardPlan, tensors: Mapping[str, Any], created_at_utc: str,
) -> dict[str, Any]:
    import numpy as np
    from safetensors.numpy import save_file
    validate_tensor_mapping(tensors, len(shard.image_ids))
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    tensor_path = root / f"{shard.basename}.safetensors"
    sidecar_path = root / f"{shard.basename}.json"
    if tensor_path.exists() or sidecar_path.exists():
        _fail("shard output collision")
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{shard.basename}.", suffix=".safetensors.tmp", dir=root)
    os.close(descriptor); temp = Path(temp_name)
    try:
        save_file({key: np.ascontiguousarray(tensors[key]) for key in tensor_keys()}, str(temp))
        with temp.open("rb+") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temp, tensor_path)
        except FileExistsError:
            _fail("shard tensor output collision", EXIT_ARTIFACT)
        except OSError as exc:
            raise FeatureExtractionError("atomic tensor publication unavailable", EXIT_RESOURCE) from exc
        temp.unlink()
    finally:
        temp.unlink(missing_ok=True)
    tensor_hashes = {
        key: hashlib.sha256(np.ascontiguousarray(tensors[key]).tobytes(order="C")).hexdigest()
        for key in tensor_keys()
    }
    sidecar = {
        "created_at_utc": created_at_utc, "dtype": "float32",
        "end_index_exclusive": shard.stop, "hidden_size": EXPECTED_HIDDEN_SIZE,
        "image_ids": list(shard.image_ids), "lane_id": lane_id,
        "model_id": model["model_id"], "model_key": model["registry_key"],
        "patch_count": EXPECTED_PATCH_COUNT, "revision": model["revision"],
        "run_id": run_id, "safetensors_file": tensor_path.name,
        "safetensors_sha256": sha256_file(tensor_path), "schema_id": SHARD_SCHEMA_ID,
        "shard_index": shard.index, "start_index": shard.start,
        "tensor_keys": list(tensor_keys()), "tensor_sha256": tensor_hashes,
    }
    atomic_write_json(sidecar_path, sidecar)
    return {"relative_sidecar": sidecar_path.name, "sha256": sha256_file(sidecar_path), "shard_index": shard.index}


def _load_safetensors(path: Path) -> Mapping[str, Any]:
    try:
        from safetensors.numpy import load_file
        return load_file(str(path))
    except Exception as exc:
        raise FeatureExtractionError("unable to read Safetensors shard", EXIT_ARTIFACT) from exc


def validate_shard(sidecar_path: str | Path, *, expected: ShardPlan, model: Mapping[str, Any], lane_id: str, run_id: str) -> dict[str, Any]:
    path = Path(sidecar_path)
    value = load_strict_json(path, label="shard sidecar")
    required = {"created_at_utc", "dtype", "end_index_exclusive", "hidden_size", "image_ids", "lane_id", "model_id", "model_key", "patch_count", "revision", "run_id", "safetensors_file", "safetensors_sha256", "schema_id", "shard_index", "start_index", "tensor_keys", "tensor_sha256"}
    _exact(value, required, "shard")
    exact = {
        "schema_id": SHARD_SCHEMA_ID, "run_id": run_id,
        "model_key": model["registry_key"], "model_id": model["model_id"],
        "revision": model["revision"], "lane_id": lane_id,
        "shard_index": expected.index, "start_index": expected.start,
        "end_index_exclusive": expected.stop, "image_ids": list(expected.image_ids),
        "patch_count": EXPECTED_PATCH_COUNT, "hidden_size": EXPECTED_HIDDEN_SIZE,
        "dtype": "float32", "tensor_keys": list(tensor_keys()),
        "safetensors_file": f"{expected.basename}.safetensors",
    }
    for key, expected_value in exact.items():
        if value[key] != expected_value:
            _fail(f"shard {expected.index} identity mismatch: {key}", EXIT_ARTIFACT)
    tensor_path = path.parent / value["safetensors_file"]
    if tensor_path.is_symlink() or not tensor_path.is_file() or sha256_file(tensor_path) != value["safetensors_sha256"]:
        _fail("shard Safetensors checksum mismatch", EXIT_ARTIFACT)
    tensors = _load_safetensors(tensor_path)
    validate_tensor_mapping(tensors, len(expected.image_ids))
    for key in tensor_keys():
        import numpy as np
        digest = hashlib.sha256(np.ascontiguousarray(tensors[key]).tobytes(order="C")).hexdigest()
        if value["tensor_sha256"].get(key) != digest:
            _fail(f"tensor checksum mismatch: {key}", EXIT_ARTIFACT)
    return value


def initial_run_state(profile: ExtractionProfile, *, run_id: str, mode: str, batch_size: int, image_ids: Sequence[str], repository_head: str | None = None) -> dict[str, Any]:
    if RUN_ID_RE.fullmatch(run_id) is None:
        _fail("run ID is noncanonical")
    count = validate_mode(profile, mode, batch_size)
    if len(image_ids) != count:
        _fail(f"{mode} mode requires exactly {count} ordered images")
    head = _approved_repository_commit() if repository_head is None else repository_head
    if REVISION_RE.fullmatch(head) is None:
        _fail("repository head is noncanonical")
    plan = extraction_plan_document(profile, run_id=run_id, mode=mode, batch_size=batch_size, image_ids=image_ids, repository_head=head)
    environment = environment_document(profile, repository_head=head)
    return {
        "batch_size": batch_size, "image_ids": list(image_ids), "mode": mode,
        "model_order": list(MODEL_ORDER), "profile_sha256": profile.source_sha256,
        "production_source_baseline": PRODUCTION_SOURCE_BASELINE,
        "repository_head": head,
        "immutable_artifacts": {
            "dataset_manifest.snapshot.jsonl": profile.document["dataset"]["manifest"]["sha256"],
            "environment.json": hashlib.sha256(canonical_json_bytes(environment)).hexdigest(),
            "extraction_plan.json": hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
        },
        "ready_for_p1_extraction": False, "run_id": run_id,
        "schema_id": RUN_STATE_SCHEMA_ID, "status": "incomplete",
    }


def environment_document(profile: ExtractionProfile, *, repository_head: str | None = None) -> dict[str, Any]:
    head = _approved_repository_commit() if repository_head is None else repository_head
    return {
        "cuda_visible_devices": "0",
        "network_allowed": False,
        "offline_variables": {
            "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
            "PIP_NO_INDEX": "1", "TRANSFORMERS_OFFLINE": "1",
        },
        "profile_sha256": profile.source_sha256,
        "python_interpreters": {
            "dinov3_isolated_lane": profile.document["runtime"]["dinov3"]["interpreter"],
            "primary_lane": profile.document["runtime"]["main"]["interpreter"],
        },
        "ready_for_p1_extraction": False,
        "production_source_baseline": PRODUCTION_SOURCE_BASELINE,
        "repository_head": head,
        "schema_id": ENVIRONMENT_SCHEMA_ID,
    }


def extraction_plan_document(
    profile: ExtractionProfile, *, run_id: str, mode: str,
    batch_size: int, image_ids: Sequence[str], repository_head: str | None = None,
) -> dict[str, Any]:
    head = _approved_repository_commit() if repository_head is None else repository_head
    count = validate_mode(profile, mode, batch_size)
    ids = tuple(image_ids)
    if len(ids) != count or len(set(ids)) != count:
        _fail(f"{mode} mode requires exactly {count} unique ordered images")
    plans = plan_shards(ids)
    return {
        "batch_size": batch_size,
        "canonical_geometry": dict(profile.document["canonical_geometry"]),
        "mode": mode,
        "models": [dict(model) for model in profile.models],
        "profile_sha256": profile.source_sha256,
        "ready_for_p1_extraction": False,
        "production_source_baseline": PRODUCTION_SOURCE_BASELINE,
        "repository_head": head,
        "run_id": run_id,
        "schema_id": PLAN_SCHEMA_ID,
        "selected_image_ids": list(ids),
        "shard_image_count": SHARD_IMAGE_COUNT,
        "shards": [
            {"end_index_exclusive": plan.stop, "image_ids": list(plan.image_ids),
             "shard_index": plan.index, "start_index": plan.start}
            for plan in plans
        ],
        "source_bindings": [dict(item) for item in profile.document["source_bindings"]],
    }


def initialize_run_artifacts(
    run_root: str | Path, profile: ExtractionProfile, repository_root: str | Path,
    *, run_id: str, mode: str, batch_size: int, image_ids: Sequence[str],
) -> dict[str, Any]:
    root = validate_non_link_path(
        run_root, expected_kind="directory", label="new run root",
    )
    if any(root.iterdir()):
        _fail("new run root must be an empty non-link directory", EXIT_RESOURCE)
    repository = validate_non_link_path(
        repository_root, expected_kind="directory", label="repository root",
    )
    repository_head = _approved_repository_commit(repository)
    manifest_relative = profile.document["dataset"]["manifest"]["path"]
    manifest_path = repository.joinpath(*manifest_relative.split("/"))
    raw_manifest = manifest_path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != profile.document["dataset"]["manifest"]["sha256"]:
        _fail("dataset manifest snapshot identity mismatch")
    state = initial_run_state(profile, run_id=run_id, mode=mode, batch_size=batch_size, image_ids=image_ids, repository_head=repository_head)
    atomic_write(root / "dataset_manifest.snapshot.jsonl", raw_manifest)
    atomic_write_json(root / "environment.json", environment_document(profile, repository_head=repository_head))
    atomic_write_json(root / "extraction_plan.json", extraction_plan_document(profile, run_id=run_id, mode=mode, batch_size=batch_size, image_ids=image_ids, repository_head=repository_head))
    atomic_write_json(root / "run_state.json", state)
    return state


def _validate_control_artifacts(root: Path, state: Mapping[str, Any], profile: ExtractionProfile) -> None:
    repository_head = state.get("repository_head")
    if state.get("profile_sha256") != profile.source_sha256 or not isinstance(repository_head, str) or REVISION_RE.fullmatch(repository_head) is None:
        _fail("run control identity mismatch", EXIT_ARTIFACT)
    expected_state = initial_run_state(
        profile, run_id=str(state.get("run_id")), mode=str(state.get("mode")),
        batch_size=state.get("batch_size"), image_ids=state.get("image_ids", ()), repository_head=repository_head,
    )
    if dict(state) != expected_state:
        _fail("run state fields or values mismatch", EXIT_ARTIFACT)
    immutable = state.get("immutable_artifacts")
    if not isinstance(immutable, Mapping):
        _fail("run immutable-artifact identity missing", EXIT_ARTIFACT)
    expected_plan = extraction_plan_document(
        profile, run_id=str(state["run_id"]), mode=str(state["mode"]),
        batch_size=state["batch_size"], image_ids=state["image_ids"], repository_head=repository_head,
    )
    expected_json = {
        "environment.json": environment_document(profile, repository_head=repository_head),
        "extraction_plan.json": expected_plan,
    }
    for name, document in expected_json.items():
        path = root / name
        if load_strict_json(path, label=name) != document or sha256_file(path) != immutable.get(name):
            _fail(f"immutable control artifact mismatch: {name}", EXIT_ARTIFACT)
    snapshot = root / "dataset_manifest.snapshot.jsonl"
    if snapshot.is_symlink() or not snapshot.is_file() or sha256_file(snapshot) != immutable.get("dataset_manifest.snapshot.jsonl"):
        _fail("dataset manifest snapshot mismatch", EXIT_ARTIFACT)
    if immutable.get("dataset_manifest.snapshot.jsonl") != profile.document["dataset"]["manifest"]["sha256"]:
        _fail("dataset manifest snapshot profile mismatch", EXIT_ARTIFACT)


def write_execution_metrics(
    path: str | Path, *, run_id: str, model: Mapping[str, Any], image_count: int,
    elapsed_seconds: float, peak_allocated_bytes: int, peak_reserved_bytes: int,
) -> dict[str, Any]:
    if not isinstance(elapsed_seconds, (int, float)) or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        _fail("invalid elapsed metric")
    if type(peak_allocated_bytes) is not int or type(peak_reserved_bytes) is not int or min(peak_allocated_bytes, peak_reserved_bytes) < 0:
        _fail("invalid GPU memory metric")
    value = {
        "elapsed_seconds": float(elapsed_seconds), "image_count": image_count,
        "lane_id": model["lane_id"], "model_key": model["registry_key"],
        "peak_gpu_allocated_bytes": peak_allocated_bytes,
        "peak_gpu_reserved_bytes": peak_reserved_bytes,
        "ready_for_p1_extraction": False, "run_id": run_id,
        "schema_id": EXECUTION_METRICS_SCHEMA_ID,
    }
    atomic_write_json(path, value)
    return value


def _validate_metrics(path: Path, *, state: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    value = load_strict_json(path, label="execution_metrics")
    if set(value) != {"elapsed_seconds", "image_count", "lane_id", "model_key", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "ready_for_p1_extraction", "run_id", "schema_id"}:
        _fail("execution metrics fields mismatch", EXIT_ARTIFACT)
    if value["schema_id"] != EXECUTION_METRICS_SCHEMA_ID or value["run_id"] != state["run_id"] or value["model_key"] != model["registry_key"] or value["lane_id"] != model["lane_id"] or value["image_count"] != len(state["image_ids"]) or value["ready_for_p1_extraction"] is not False:
        _fail("execution metrics identity mismatch", EXIT_ARTIFACT)
    if type(value["elapsed_seconds"]) not in (int, float) or not math.isfinite(value["elapsed_seconds"]) or value["elapsed_seconds"] < 0:
        _fail("execution metrics elapsed mismatch", EXIT_ARTIFACT)
    if type(value["peak_gpu_allocated_bytes"]) is not int or type(value["peak_gpu_reserved_bytes"]) is not int or min(value["peak_gpu_allocated_bytes"], value["peak_gpu_reserved_bytes"]) < 0:
        _fail("execution metrics memory mismatch", EXIT_ARTIFACT)
    return value


def write_failure_evidence(
    run_root: str | Path, *, state: Mapping[str, Any], stage: str,
    exception_class: str, sanitized_message: str, created_at_utc: str,
    stable_exit_code: int = EXIT_EXECUTION,
    model: Mapping[str, Any] | None = None,
    shard: ShardPlan | None = None,
    raw_return_code: int | None = None,
) -> Path:
    root = Path(run_root)
    validate_non_link_path(root, expected_kind="directory", label="failure run root")
    if stage not in FAILURE_STAGES:
        _fail("unsupported failure stage")
    worker_scope = stage in WORKER_FAILURE_STAGES
    if worker_scope != (model is not None and shard is not None):
        _fail("failure stage/scope mismatch")
    if stable_exit_code != EXIT_EXECUTION:
        _fail("stable failure exit code must be EXIT_EXECUTION")
    if raw_return_code is not None and (
        type(raw_return_code) is not int
        or not -(2**31) <= raw_return_code <= 2**31 - 1
    ):
        _fail("raw worker return code is outside the canonical range")
    failures = root / "failures"
    try:
        failures.mkdir()
    except FileExistsError:
        pass
    if not failures.is_dir() or _is_link(failures):
        _fail("failure evidence root mismatch", EXIT_ARTIFACT)
    existing = sorted(failures.glob("attempt-*.json"))
    if any(path.name != f"attempt-{index:04d}.json" for index, path in enumerate(existing, 1)):
        _fail("failure evidence inventory mismatch", EXIT_ARTIFACT)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", exception_class):
        _fail("unsafe exception class")
    message = _string(sanitized_message, "sanitized_message")
    if "/" in message or "\\" in message or SENSITIVE_RE.search(message):
        _fail("failure message is not sanitized")
    attempt = len(existing) + 1
    value = {
        "attempt_index": attempt, "created_at_utc": created_at_utc,
        "end_index_exclusive": shard.stop if shard is not None else None,
        "exception_class": exception_class,
        "failure_scope": "worker" if worker_scope else "orchestration",
        "failure_stage": stage,
        "image_ids": list(shard.image_ids) if shard is not None else None,
        "lane_id": model["lane_id"] if model is not None else None,
        "model_key": model["registry_key"] if model is not None else None,
        "raw_return_code": raw_return_code,
        "ready_for_p1_extraction": False,
        "run_id": state["run_id"], "sanitized_message": message,
        "schema_id": FAILURE_SCHEMA_ID,
        "shard_index": shard.index if shard is not None else None,
        "stable_exit_code": stable_exit_code,
        "start_index": shard.start if shard is not None else None,
    }
    path = failures / f"attempt-{attempt:04d}.json"
    atomic_write_json(path, value)
    return path


def _failure_inventory(root: Path, state: Mapping[str, Any], profile: ExtractionProfile) -> set[str]:
    directory = root / "failures"
    inventory: set[str] = set()
    if not directory.exists():
        return inventory
    if not directory.is_dir() or _is_link(directory):
        _fail("failure evidence root mismatch", EXIT_ARTIFACT)
    models = {model["registry_key"]: model for model in profile.models}
    plans = plan_shards(state["image_ids"])
    files = sorted(directory.iterdir())
    for index, path in enumerate(files, 1):
        if path.name != f"attempt-{index:04d}.json" or not path.is_file() or path.is_symlink():
            _fail("failure evidence is not append-only canonical", EXIT_ARTIFACT)
        value = load_strict_json(path, label="failure evidence")
        required = {
            "attempt_index", "created_at_utc", "end_index_exclusive",
            "exception_class", "failure_scope", "failure_stage", "image_ids",
            "lane_id", "model_key", "raw_return_code",
            "ready_for_p1_extraction", "run_id", "sanitized_message",
            "schema_id", "shard_index", "stable_exit_code", "start_index",
        }
        if set(value) != required:
            _fail("failure evidence fields mismatch", EXIT_ARTIFACT)
        if value.get("schema_id") != FAILURE_SCHEMA_ID or value.get("attempt_index") != index or value.get("run_id") != state["run_id"] or value.get("ready_for_p1_extraction") is not False:
            _fail("failure evidence identity mismatch", EXIT_ARTIFACT)
        exception_class = value.get("exception_class")
        message = value.get("sanitized_message")
        created = value.get("created_at_utc")
        exit_code = value.get("stable_exit_code")
        if (
            not isinstance(exception_class, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", exception_class) is None
            or not isinstance(message, str) or not message or "/" in message or "\\" in message
            or SENSITIVE_RE.search(message) is not None
            or not isinstance(created, str) or not created.endswith("Z")
            or exit_code != EXIT_EXECUTION
        ):
            _fail("failure evidence safety mismatch", EXIT_ARTIFACT)
        stage = value.get("failure_stage")
        scope = value.get("failure_scope")
        raw_return_code = value.get("raw_return_code")
        if stage not in FAILURE_STAGES or scope != ("worker" if stage in WORKER_FAILURE_STAGES else "orchestration"):
            _fail("failure evidence stage/scope mismatch", EXIT_ARTIFACT)
        if raw_return_code is not None and (
            type(raw_return_code) is not int
            or not -(2**31) <= raw_return_code <= 2**31 - 1
        ):
            _fail("failure evidence raw return code mismatch", EXIT_ARTIFACT)
        if scope == "worker":
            model = models.get(value.get("model_key"))
            shard_index = value.get("shard_index")
            if model is None or value.get("lane_id") != model["lane_id"] or type(shard_index) is not int or not 0 <= shard_index < len(plans):
                _fail("failure evidence model/shard mismatch", EXIT_ARTIFACT)
            plan = plans[shard_index]
            if value.get("start_index") != plan.start or value.get("end_index_exclusive") != plan.stop or value.get("image_ids") != list(plan.image_ids):
                _fail("failure evidence image range mismatch", EXIT_ARTIFACT)
        elif any(value.get(key) is not None for key in (
            "model_key", "lane_id", "shard_index", "start_index",
            "end_index_exclusive", "image_ids", "raw_return_code",
        )):
            _fail("orchestration failure context must be null", EXIT_ARTIFACT)
        inventory.add(f"failures/{path.name}")
    return inventory


def reconcile_resume(run_root: str | Path, expected_state: Mapping[str, Any], profile: ExtractionProfile) -> dict[str, int]:
    root = validate_non_link_path(
        run_root, expected_kind="directory", label="resume run root",
    )
    if (root / "run_manifest.json").exists():
        _fail("resume requires an incomplete non-link run")
    allowed_root_names = {
        "dataset_manifest.snapshot.jsonl", "environment.json", "extraction_plan.json",
        "failures", "models", "run_state.json",
    }
    if {path.name for path in root.iterdir()} - allowed_root_names:
        _fail("finalization residue blocks automatic resume", EXIT_ARTIFACT)
    actual = load_strict_json(root / "run_state.json", label="run_state")
    if actual != dict(expected_state):
        _fail("resume identity mismatch")
    _validate_control_artifacts(root, actual, profile)
    _failure_inventory(root, actual, profile)
    plans = plan_shards(actual["image_ids"])
    progress: dict[str, int] = {}
    by_model = {item["registry_key"]: item for item in profile.models}
    for model_key in MODEL_ORDER:
        model = by_model[model_key]; lane = model["lane_id"]
        directory = root / "models" / model_key
        next_index = 0
        if directory.exists():
            if _is_link(directory) or any(path.is_dir() for path in directory.iterdir()):
                _fail("unexpected nested model directory", EXIT_ARTIFACT)
            actual_names = {path.name for path in directory.iterdir()}
            for plan in plans:
                sidecar = directory / f"{plan.basename}.json"
                tensor = directory / f"{plan.basename}.safetensors"
                if sidecar.exists() or tensor.exists():
                    if not sidecar.is_file() or not tensor.is_file():
                        _fail("partial shard blocks resume", EXIT_ARTIFACT)
                    validate_shard(sidecar, expected=plan, model=model, lane_id=lane, run_id=actual["run_id"])
                    next_index += 1
                else:
                    break
            allowed = {f"{plan.basename}{suffix}" for plan in plans[:next_index] for suffix in (".json", ".safetensors")}
            metrics = directory / "execution_metrics.json"
            if metrics.exists():
                if next_index != len(plans):
                    _fail("metrics published before model completion", EXIT_ARTIFACT)
                _validate_metrics(metrics, state=actual, model=model); allowed.add(metrics.name)
            elif next_index == len(plans):
                _fail("complete shards without trustworthy metrics require manual review", EXIT_ARTIFACT)
            if actual_names != allowed:
                _fail("unexpected or nonconsecutive shard blocks resume", EXIT_ARTIFACT)
        progress[model_key] = next_index
    return progress


def build_model_manifest(run_root: Path, profile: ExtractionProfile, model: Mapping[str, Any], plans: Sequence[ShardPlan]) -> dict[str, Any]:
    model_key = model["registry_key"]
    shards = []
    for plan in plans:
        relative = f"models/{model_key}/{plan.basename}.json"
        sidecar = run_root / relative
        validate_shard(sidecar, expected=plan, model=model, lane_id=model["lane_id"], run_id=run_root.name)
        shards.append({"relative_path": relative, "sha256": sha256_file(sidecar), "shard_index": plan.index})
    return {
        "image_count": sum(len(plan.image_ids) for plan in plans),
        "image_ids": [image_id for plan in plans for image_id in plan.image_ids],
        "lane_id": model["lane_id"], "model_id": model["model_id"],
        "model_key": model_key, "revision": model["revision"],
        "run_id": run_root.name, "schema_id": MODEL_MANIFEST_SCHEMA_ID,
        "shards": shards,
    }


def complete_run(
    run_root: str | Path, profile: ExtractionProfile, *, publish_manifest: bool = True,
) -> dict[str, Any]:
    root = validate_non_link_path(
        run_root, expected_kind="directory", label="completion run root",
    )
    if (root / "run_manifest.json").exists() or (root / "run_manifest.json").is_symlink():
        _fail("run manifest output collision", EXIT_ARTIFACT)
    state = load_strict_json(root / "run_state.json", label="run_state")
    if state.get("status") != "incomplete" or state.get("profile_sha256") != profile.source_sha256:
        _fail("run state mismatch", EXIT_ARTIFACT)
    _validate_control_artifacts(root, state, profile)
    failures = _failure_inventory(root, state, profile)
    plans = plan_shards(state["image_ids"])
    model_entries = []
    for model in profile.models:
        _validate_metrics(root / "models" / model["registry_key"] / "execution_metrics.json", state=state, model=model)
        manifest = build_model_manifest(root, profile, model, plans)
        path = root / "models" / model["registry_key"] / "model_manifest.json"
        atomic_write_json(path, manifest)
        model_entries.append({"model_key": model["registry_key"], "relative_path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    validation = {
        "checks": ["inventory", "model_identity", "shard_checksum", "tensor_contract", "image_order", "metrics"],
        "image_count": len(state["image_ids"]), "model_count": 3,
        "ready_for_p1_extraction": False, "run_id": root.name,
        "schema_id": VALIDATION_SCHEMA_ID, "status": "validated",
    }
    validation_root = root / "validation"
    if validation_root.exists():
        _fail("validation output collision", EXIT_ARTIFACT)
    validation_root.mkdir()
    validation_path = validation_root / "artifact_validation.json"
    atomic_write_json(validation_path, validation)
    manifest = {
        "batch_size": state["batch_size"], "image_count": len(state["image_ids"]),
        "image_ids": state["image_ids"], "mode": state["mode"],
        "model_manifests": model_entries, "model_order": list(MODEL_ORDER),
        "profile_sha256": profile.source_sha256, "ready_for_p1_extraction": False,
        "required_artifacts": {
            "dataset_manifest.snapshot.jsonl": sha256_file(root / "dataset_manifest.snapshot.jsonl"),
            "environment.json": sha256_file(root / "environment.json"),
            "extraction_plan.json": sha256_file(root / "extraction_plan.json"),
            "validation/artifact_validation.json": sha256_file(validation_path),
        },
        "failure_evidence_count": len(failures),
        "run_id": root.name, "schema_id": RUN_MANIFEST_SCHEMA_ID,
        "status": "complete",
    }
    # Validate the candidate in memory and all referenced files before publication.
    _validate_complete_document(root, profile, manifest)
    if publish_manifest:
        atomic_write_json(root / "run_manifest.json", manifest)
    return manifest


def _validate_complete_document(root: Path, profile: ExtractionProfile, run: Mapping[str, Any]) -> None:
    if run.get("schema_id") != RUN_MANIFEST_SCHEMA_ID or run.get("status") != "complete" or run.get("ready_for_p1_extraction") is not False:
        _fail("run manifest status mismatch", EXIT_ARTIFACT)
    if run.get("profile_sha256") != profile.source_sha256 or run.get("model_order") != list(MODEL_ORDER):
        _fail("run manifest identity mismatch", EXIT_ARTIFACT)
    count = validate_mode(profile, str(run.get("mode")), run.get("batch_size"))
    if run.get("image_count") != count or len(run.get("image_ids", [])) != count:
        _fail("run manifest image count mismatch", EXIT_ARTIFACT)
    plans = plan_shards(run["image_ids"])
    entries = run.get("model_manifests")
    if not isinstance(entries, list) or len(entries) != 3:
        _fail("run manifest requires three model manifests", EXIT_ARTIFACT)
    by_model = {item["registry_key"]: item for item in profile.models}
    required = run.get("required_artifacts")
    expected_required = {
        "dataset_manifest.snapshot.jsonl": sha256_file(root / "dataset_manifest.snapshot.jsonl"),
        "environment.json": sha256_file(root / "environment.json"),
        "extraction_plan.json": sha256_file(root / "extraction_plan.json"),
        "validation/artifact_validation.json": sha256_file(root / "validation" / "artifact_validation.json"),
    }
    if required != expected_required:
        _fail("required artifact identity mismatch", EXIT_ARTIFACT)
    validation = load_strict_json(root / "validation" / "artifact_validation.json", label="artifact_validation")
    if validation != {"checks": ["inventory", "model_identity", "shard_checksum", "tensor_contract", "image_order", "metrics"], "image_count": count, "model_count": 3, "ready_for_p1_extraction": False, "run_id": root.name, "schema_id": VALIDATION_SCHEMA_ID, "status": "validated"}:
        _fail("artifact validation evidence mismatch", EXIT_ARTIFACT)
    _validate_control_artifacts(root, load_strict_json(root / "run_state.json", label="run_state"), profile)
    failure_inventory = _failure_inventory(root, load_strict_json(root / "run_state.json", label="run_state"), profile)
    if run.get("failure_evidence_count") != len(failure_inventory):
        _fail("failure evidence count mismatch", EXIT_ARTIFACT)
    inventory: set[str] = {"run_state.json", "run_manifest.json", *expected_required, *failure_inventory}
    for index, entry in enumerate(entries):
        if entry.get("model_key") != MODEL_ORDER[index]:
            _fail("model manifest order mismatch", EXIT_ARTIFACT)
        relative = safe_relative_path(entry.get("relative_path"))
        path = root / relative
        if sha256_file(path) != entry.get("sha256"):
            _fail("model manifest checksum mismatch", EXIT_ARTIFACT)
        model_manifest = load_strict_json(path, label="model_manifest")
        expected_model = by_model[MODEL_ORDER[index]]
        if model_manifest != build_model_manifest(root, profile, expected_model, plans):
            _fail("model manifest content mismatch", EXIT_ARTIFACT)
        inventory.add(relative)
        metrics = f"models/{MODEL_ORDER[index]}/execution_metrics.json"
        _validate_metrics(root / metrics, state=load_strict_json(root / "run_state.json", label="run_state"), model=expected_model)
        inventory.add(metrics)
        for plan in plans:
            inventory.add(f"models/{MODEL_ORDER[index]}/{plan.basename}.json")
            inventory.add(f"models/{MODEL_ORDER[index]}/{plan.basename}.safetensors")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    # run_manifest may not exist while validating before final atomic publication.
    expected_inventory = inventory if (root / "run_manifest.json").exists() else inventory - {"run_manifest.json"}
    if actual != expected_inventory:
        _fail("run file inventory mismatch", EXIT_ARTIFACT)


def validate_completion_candidate(
    run_root: str | Path, profile: ExtractionProfile, candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Perform model-free validation before the last run-manifest publication."""
    root = validate_non_link_path(
        run_root, expected_kind="directory", label="candidate validation run root",
    )
    if (root / "run_manifest.json").exists() or (root / "run_manifest.json").is_symlink():
        _fail("candidate validation requires an unpublished run manifest", EXIT_ARTIFACT)
    _validate_complete_document(root, profile, candidate)
    return {
        "classification": "P1_FEATURE_EXTRACTION_COMPLETE_VALIDATED",
        "image_count": candidate["image_count"], "model_count": 3,
        "ready_for_p1_extraction": False, "run_id": candidate["run_id"],
        "schema_id": VALIDATION_SCHEMA_ID,
    }


def validate_run(run_root: str | Path, profile: ExtractionProfile, *, allow_incomplete: bool = False) -> dict[str, Any]:
    root = validate_non_link_path(
        run_root, expected_kind="directory", label="validation run root",
    )
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        state = load_strict_json(root / "run_state.json", label="run_state")
        if not allow_incomplete:
            _fail("incomplete run rejected", EXIT_ARTIFACT)
        progress = reconcile_resume(root, state, profile)
        return {
            "classification": "P1_FEATURE_EXTRACTION_INCOMPLETE",
            "progress": progress, "ready_for_p1_extraction": False,
            "schema_id": VALIDATION_SCHEMA_ID,
        }
    run = load_strict_json(manifest_path, label="run_manifest")
    _validate_complete_document(root, profile, run)
    return {
        "classification": "P1_FEATURE_EXTRACTION_COMPLETE_VALIDATED",
        "image_count": run["image_count"], "model_count": 3,
        "ready_for_p1_extraction": False, "run_id": run["run_id"],
        "schema_id": VALIDATION_SCHEMA_ID,
    }


CommandRunner = Callable[[Sequence[str], Mapping[str, str]], subprocess.CompletedProcess[str]]


def run_command(arguments: Sequence[str], environment: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(arguments), env=dict(environment), capture_output=True, text=True, check=False, shell=False)


__all__ = [
    "EXIT_ARTIFACT", "EXIT_CONTRACT", "EXIT_EXECUTION", "EXIT_NOT_READY",
    "EXIT_RESOURCE", "EXIT_SUCCESS", "ExtractionProfile", "FeatureExtractionError",
    "FAILURE_STAGES", "GLOBAL_TAPS", "MODEL_ORDER", "PATCH_DEPTHS", "PATCH_TAPS", "PROFILE_RELATIVE_PATH",
    "RUN_ID_RE", "SHARD_IMAGE_COUNT", "ShardPlan", "atomic_write", "atomic_write_json",
    "canonical_json_bytes", "complete_run", "environment_document", "execution_plan", "extraction_plan_document", "initial_run_state", "initialize_run_artifacts",
    "lane_command_plan", "load_extraction_profile", "load_strict_json",
    "offline_environment", "ordered_batches", "plan_shards", "reconcile_resume",
    "run_command", "safe_relative_path", "sha256_file", "stream_manifest",
    "strict_json_bytes", "tensor_keys", "validate_mode", "validate_non_link_path", "validate_output_root",
    "validate_completion_candidate", "validate_run", "validate_shard", "validate_tensor_mapping",
    "verify_repository_inputs", "write_execution_metrics", "write_failure_evidence", "write_shard",
]
