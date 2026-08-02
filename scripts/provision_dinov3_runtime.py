"""Provision and verify the exact persistent DINOv3 isolated runtime.

The module intentionally imports only the Python standard library at module import
time.  Network access is limited to the ``provision`` command; ``verify`` is a
read-only, offline operation.  A fresh runtime is built in a uniquely-owned sibling
staging directory and is published with a same-parent rename only after complete
verification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


PROFILE_SCHEMA_ID = "vea.dinov3_isolated_runtime_profile.v0"
RUNTIME_MANIFEST_SCHEMA_ID = "vea.dinov3_persistent_runtime.v0"
RUNTIME_ID = "dinov3-vitb16-r5931719-py312-torch230-tf4560-manual-v1"
TARGET_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
TARGET_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
MANUAL_PROFILE_ID = "dinov3-manual-fp32-imagenet-v1"
PROFILE_SHA256 = '52fc589cc47127c4c764f9631f17777f31f9784f7d4b92426f5b31ac4e3a2163'
CURRENT_FREEZE_SHA256 = (
    "42ded209c200cf18b422e0626083776bf1567c5151eacf0d5f895a7bcc17e109"
)
CANDIDATE_FREEZE_SHA256 = (
    "995687ed934c0a94b52a040af048f87bc8de4e5089475f70565ad37f1f33ecd9"
)
EXPECTED_CONFLICT_SHA256 = (
    "dd98f3eba507ab782d4550192672d0f9fbd56e85c82478a0cea8611c775798a2"
)
EXPECTED_CONFLICT_COUNT = 8
EXPECTED_ACTIVE_REQUIREMENT_COUNT = 18
MAX_METADATA_RESPONSE_BYTES = 2 * 1024 * 1024
MINIMUM_FREE_BYTES = 4 * 1024**3

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXACT_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*(?:\+[A-Za-z0-9.]+)?$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_CONFLICT_RE = re.compile(r"^.+ has requirement .+, but you have .+\.$")
_SECRET_NAME_RE = re.compile(
    r"(?:^|_)(?:api_?keys?|access_?key(?:_id)?|auth|cookies?|credentials?|"
    r"github_?pat|passwords?|passwds?|private_?keys?|secrets?|tokens?)(?:$|_)",
    re.I,
)

EXPECTED_WHEELS: dict[str, tuple[str, int, str]] = {
    "huggingface_hub-0.36.2-py3-none-any.whl": (
        "huggingface-hub",
        566395,
        "48f0c8eac16145dfce371e9d2d7772854a4f591bcb56c9cf548accf531d54270",
    ),
    "tokenizers-0.22.2-cp39-abi3-win_amd64.whl": (
        "tokenizers",
        2747786,
        "c9ea31edff2968b44a88f97d784c2f16dc0729b8b143ed004699ebca91f05c48",
    ),
    "transformers-4.56.0-py3-none-any.whl": (
        "transformers",
        11607938,
        "bacf539c38dd850690856881c4974321af93a22f2ee96bcc994741a2121d8e71",
    ),
}
EXPECTED_METADATA: dict[str, tuple[int, str]] = {
    "config.json": (
        744,
        "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec",
    ),
    "preprocessor_config.json": (
        585,
        "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
    ),
}
EXPECTED_CANDIDATE_LOCAL_PACKAGES = {
    "huggingface-hub": "0.36.2",
    "pip": "23.2.1",
    "tokenizers": "0.22.2",
    "transformers": "4.56.0",
}
EXPECTED_CURRENT_PACKAGES = {
    "Pillow": "10.3.0",
    "huggingface-hub": "0.22.2",
    "numpy": "1.26.4",
    "pytest": "8.4.2",
    "safetensors": "0.4.3",
    "tokenizers": "0.19.1",
    "torch": "2.3.0",
    "transformers": "4.40.1",
}
EXPECTED_CURRENT_SOURCES = {'configs/models.yaml': '2c75cc26a27febc633b2bad479779327f35abbf36160df2805f43a4e33f48244',
 'pyproject.toml': '45f3ab3f1d3267c51f82babf47fe2f53011192ccf317494931aba70478ba9c7e',
 'scripts/dinov3_extraction_worker.py': '08cfdbbbd9e99120adb7c660e4c25f162f766b7516451c6a4a410f316a11defd',
 'src/vision_encoder_alignment/dinov3_extraction_boundary.py': 'fe2ba7035c8852bdd3ed44f75b01926c1987aca1f71bae94accbb6578eaec54d',
 'src/vision_encoder_alignment/dinov3_production_identity.py': '01e60a8e8434695c845467c361e3b129ba54a1231287e8fa4f59f983fc91d5c4',
 'src/vision_encoder_alignment/extractors/__init__.py': 'f7ccf421524355c7ea11bb9ed3e7b6d83fa25e953d7266f3a36a4e7bcb95fc49',
 'src/vision_encoder_alignment/extractors/base.py': 'ecf512c1830fc1735392b010a69312bc62914e44c4b5c125a55d2b89889eae82',
 'src/vision_encoder_alignment/extractors/clip.py': '0ec0980c9761a73597244b03116de093c71b568cc0d8755951821b4f448eb82d',
 'src/vision_encoder_alignment/extractors/dinov3.py': '97c781592265c9e600ba0ad257528c313c471e23c2d22f27905d2e42a872f0d1',
 'src/vision_encoder_alignment/extractors/siglip2.py': '0267d9eaa51d6c6b686339d9d9e700f782d2ec3b5c16c20d3df16e8fa17cfd5e',
 'src/vision_encoder_alignment/feature_artifacts/__init__.py': '5615ef70615d23cfd3af945f8803f3a3bb7883329a7460ccee8f9888a7ee965d',
 'src/vision_encoder_alignment/feature_artifacts/checksum.py': '306077306a4fe8fe7f5084fcdda5d1999869f73038db2389a34ac5281803a81a',
 'src/vision_encoder_alignment/feature_artifacts/io.py': 'c86e2c2566c98d112e05e12b0f3704f21794f3fc69542e19972d788df4536b2d',
 'src/vision_encoder_alignment/feature_artifacts/schema.py': 'f35059e0c0baf61929b00c92993c849cb7d601b3d0a61c037fb686cdc3fc3aa2',
 'src/vision_encoder_alignment/feature_artifacts/validation.py': '2f52f7aaf73287a0fd061666c1412d6cf2e1546209cf715ac96a1cc776235b36',
 'src/vision_encoder_alignment/production_runtime_contract.py': 'bfd41c817a98768ef6d9e6f5db996420ff4017583165c6afb2f8bc587c3b6010',
 'src/vision_encoder_alignment/runtime_boundary.py': '3233ed503581af6f9e9fb54993e85783546a0ef251b02dbe943faec6724b0063'}
EXPECTED_INSTALLED_SOURCES: dict[str, tuple[int, str]] = {
    "transformers/image_transforms.py": (
        41500,
        "163556a472733191afe8cd86e08348d599bd88eb2370a3747ca687b1c14a6691",
    ),
    "transformers/image_processing_utils_fast.py": (
        27737,
        "bbde838e3f91b9dc579db57b5d7ad68edcffd9df65b51b28a74aa8ffc116f4dd",
    ),
    "transformers/image_utils.py": (
        37821,
        "9046fc74a50458ad84d99e2fac279e09bf9987a1c3db9f7242a7beb4634763c6",
    ),
    "transformers/models/auto/configuration_auto.py": (
        53343,
        "5ae3551d7ba745e899af5664286e5f3ca2346a415f6390153cc112984f23c6e7",
    ),
    "transformers/models/auto/image_processing_auto.py": (
        38879,
        "34a4639959e6513d8471bb62a3d8b51cf0d1e10d95c29f8cbeb2fa027f4b94f2",
    ),
    "transformers/models/auto/modeling_auto.py": (
        96469,
        "a25d5b7c28aa8f2b79a4334323f8759416b4afb8ad0271a8c87bf3f2360de8d5",
    ),
    "transformers/models/dinov3_vit/configuration_dinov3_vit.py": (
        7421,
        "1ac7cb889e2314e8cadf9b0bed43d42cfb05dce6d730f719be4380c8a10a8a46",
    ),
    "transformers/models/dinov3_vit/image_processing_dinov3_vit_fast.py": (
        4167,
        "5a84a6a7283b77e0a0b000851150dd9d04dc54a11baa3d2cd3e2603d4669ec80",
    ),
    "transformers/models/dinov3_vit/modeling_dinov3_vit.py": (
        22688,
        "00652d90ff8c7ae32b047ebe2acd2271f97e03f768cf570ac051045a4e0d5cf5",
    ),
}


class ProvisioningError(RuntimeError):
    """A fail-closed provision or verification error with a safe classification."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


class RuntimeCollisionError(ProvisioningError):
    """Raised when an existing final root is not exact-valid."""

    def __init__(self, message: str = "existing runtime root is not exact-valid") -> None:
        super().__init__("DINOV3_PERSISTENT_RUNTIME_COLLISION", message)


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Validated immutable profile plus the hash of its canonical repository bytes."""

    path: Path
    sha256: str
    values: Mapping[str, Any]

    @property
    def runtime_id(self) -> str:
        return str(self.values["runtime_id"])


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class _BoundedResponseWriter:
    """Delegate binary writes while enforcing the authorized response-body cap."""

    def __init__(self, stream: Any, maximum_bytes: int) -> None:
        self._stream = stream
        self._maximum_bytes = maximum_bytes

    def write(self, payload: bytes) -> int:
        if self._stream.tell() + len(payload) > self._maximum_bytes:
            raise ProvisioningError(
                "METADATA_RESPONSE_TOO_LARGE",
                "metadata response exceeded the authorized body cap",
            )
        return int(self._stream.write(payload))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@dataclass(frozen=True, slots=True)
class ProvisionPlan:
    profile_path: Path
    repository_root: Path
    runtime_root: Path
    staging_parent: Path
    staging_prefix: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    runtime_id: str
    runtime_manifest_sha256: str
    repository_profile_sha256: str
    candidate_freeze_sha256: str
    inherited_conflict_sha256: str


Runner = Callable[..., CommandResult]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProvisioningError("INVALID_JSON", f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ProvisioningError("INVALID_JSON", f"non-finite JSON number: {value}")


def _load_canonical_json(path: Path, *, classification: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ProvisioningError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvisioningError(classification, "unable to read canonical JSON") from exc
    if not isinstance(value, dict):
        raise ProvisioningError(classification, "JSON root must be an object")
    try:
        canonical = _canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ProvisioningError(classification, "JSON is not canonicalizable") from exc
    if raw != canonical:
        raise ProvisioningError(
            classification,
            "JSON must use sorted keys, UTF-8, two-space indentation, and one final LF",
        )
    return value, raw


def _expect_keys(value: object, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProvisioningError("INVALID_PROFILE", f"{path} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ProvisioningError(
            "INVALID_PROFILE",
            f"{path} fields mismatch; missing={missing!r}, unknown={unknown!r}",
        )
    return value


def _expect_list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProvisioningError("INVALID_PROFILE", f"{path} must be an array")
    return value


def _expect_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProvisioningError("INVALID_PROFILE", f"{path} must be a nonempty string")
    return value


def _expect_int(value: object, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProvisioningError("INVALID_PROFILE", f"{path} must be an integer >= {minimum}")
    return value


def _expect_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise ProvisioningError("INVALID_PROFILE", f"{path} must be Boolean")
    return value


def _expect_sha256(value: object, path: str) -> str:
    text = _expect_string(value, path)
    if _SHA256_RE.fullmatch(text) is None:
        raise ProvisioningError("INVALID_PROFILE", f"{path} must be lowercase SHA-256")
    return text


def _expect_exact_version(value: object, path: str) -> str:
    text = _expect_string(value, path)
    if _EXACT_VERSION_RE.fullmatch(text) is None or any(c in text for c in "<>=!~,*"):
        raise ProvisioningError("INVALID_PROFILE", f"{path} must be one exact version")
    return text


def _expect_relative_path(value: object, path: str) -> str:
    text = _expect_string(value, path)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
        or text.startswith(("/", "\\"))
    ):
        raise ProvisioningError("INVALID_PROFILE", f"{path} must be a safe relative path")
    return text


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _validate_profile(value: dict[str, Any]) -> None:
    top = _expect_keys(
        value,
        {
            "allowed_worker_operations",
            "candidate",
            "canonical_input",
            "current_base",
            "dependency_validation",
            "external_layout",
            "forbidden_packages",
            "installed_sources",
            "manual_preprocessing",
            "metadata_files",
            "model_weights",
            "policies",
            "runtime_id",
            "runtime_root",
            "schema_id",
            "target",
            "wheel_files",
            "worker_protocol",
        },
        "profile",
    )
    if top["schema_id"] != PROFILE_SCHEMA_ID or top["runtime_id"] != RUNTIME_ID:
        raise ProvisioningError("INVALID_PROFILE", "schema or runtime identity mismatch")

    operations = _expect_list(top["allowed_worker_operations"], "allowed_worker_operations")
    if operations != ["artifact_boundary_self_test", "runtime_self_test"]:
        raise ProvisioningError("INVALID_PROFILE", "worker operation allowlist mismatch")

    candidate = _expect_keys(
        top["candidate"],
        {"expected_freeze", "local_packages", "python", "system_site_packages"},
        "candidate",
    )
    candidate_freeze = _expect_keys(
        candidate["expected_freeze"], {"bytes", "line_count", "sha256"}, "candidate.expected_freeze"
    )
    if (
        _expect_int(candidate_freeze["bytes"], "candidate.expected_freeze.bytes", minimum=1)
        != 5206
        or _expect_int(
            candidate_freeze["line_count"], "candidate.expected_freeze.line_count", minimum=1
        )
        != 275
        or _expect_sha256(candidate_freeze["sha256"], "candidate.expected_freeze.sha256")
        != CANDIDATE_FREEZE_SHA256
    ):
        raise ProvisioningError("INVALID_PROFILE", "candidate freeze identity mismatch")
    packages = _expect_keys(
        candidate["local_packages"], set(EXPECTED_CANDIDATE_LOCAL_PACKAGES), "candidate.local_packages"
    )
    for name, expected in EXPECTED_CANDIDATE_LOCAL_PACKAGES.items():
        if _expect_exact_version(packages[name], f"candidate.local_packages.{name}") != expected:
            raise ProvisioningError("INVALID_PROFILE", f"candidate package mismatch: {name}")
    python = _expect_keys(candidate["python"], {"major", "minor", "version"}, "candidate.python")
    if (
        _expect_int(python["major"], "candidate.python.major") != 3
        or _expect_int(python["minor"], "candidate.python.minor") != 12
        or _expect_exact_version(python["version"], "candidate.python.version") != "3.12.0"
        or not _expect_bool(candidate["system_site_packages"], "candidate.system_site_packages")
    ):
        raise ProvisioningError("INVALID_PROFILE", "candidate Python policy mismatch")

    canonical_input = _expect_keys(
        top["canonical_input"],
        {"channel_order", "dtype", "height", "layout", "random_augmentation", "width"},
        "canonical_input",
    )
    if canonical_input != {
        "channel_order": "RGB",
        "dtype": "uint8",
        "height": 224,
        "layout": "NHWC",
        "random_augmentation": False,
        "width": 224,
    }:
        raise ProvisioningError("INVALID_PROFILE", "canonical input mismatch")

    current = _expect_keys(
        top["current_base"],
        {"expected_freeze", "packages", "python", "source_files", "torch"},
        "current_base",
    )
    current_freeze = _expect_keys(
        current["expected_freeze"], {"bytes", "line_count", "sha256"}, "current_base.expected_freeze"
    )
    if (
        current_freeze != {
            "bytes": 5206,
            "line_count": 275,
            "sha256": CURRENT_FREEZE_SHA256,
        }
        or current["python"] != "3.12.0"
    ):
        raise ProvisioningError("INVALID_PROFILE", "current runtime identity mismatch")
    current_packages = _expect_keys(
        current["packages"], set(EXPECTED_CURRENT_PACKAGES), "current_base.packages"
    )
    for name, expected in EXPECTED_CURRENT_PACKAGES.items():
        if _expect_exact_version(current_packages[name], f"current_base.packages.{name}") != expected:
            raise ProvisioningError("INVALID_PROFILE", f"current package mismatch: {name}")
    source_rows = _expect_list(current["source_files"], "current_base.source_files")
    observed_current_sources: dict[str, str] = {}
    for index, row_value in enumerate(source_rows):
        row = _expect_keys(row_value, {"path", "sha256"}, f"current_base.source_files[{index}]")
        source_path = _expect_relative_path(row["path"], f"current_base.source_files[{index}].path")
        if source_path in observed_current_sources:
            raise ProvisioningError("INVALID_PROFILE", "duplicate current source path")
        observed_current_sources[source_path] = _expect_sha256(
            row["sha256"], f"current_base.source_files[{index}].sha256"
        )
    if observed_current_sources != EXPECTED_CURRENT_SOURCES:
        raise ProvisioningError("INVALID_PROFILE", "current source identity mismatch")
    torch_profile = _expect_keys(
        current["torch"],
        {
            "candidate_local_distribution",
            "cuda_available",
            "cuda_device_count",
            "metadata_version",
            "runtime_version",
        },
        "current_base.torch",
    )
    if torch_profile != {
        "candidate_local_distribution": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "metadata_version": "2.3.0",
        "runtime_version": "2.3.0+cpu",
    }:
        raise ProvisioningError("INVALID_PROFILE", "shared torch identity mismatch")

    dependency = _expect_keys(
        top["dependency_validation"],
        {
            "active_requirement_count",
            "expected_inherited_conflict_count",
            "expected_inherited_conflict_sha256",
            "expected_new_conflict_count",
            "expected_unsatisfied_active_requirement_count",
            "inherited_conflict_hash_encoding",
            "inherited_conflict_hash_join",
            "requirement_owners",
        },
        "dependency_validation",
    )
    if (
        dependency["active_requirement_count"] != EXPECTED_ACTIVE_REQUIREMENT_COUNT
        or dependency["expected_inherited_conflict_count"] != EXPECTED_CONFLICT_COUNT
        or dependency["expected_inherited_conflict_sha256"] != EXPECTED_CONFLICT_SHA256
        or dependency["expected_new_conflict_count"] != 0
        or dependency["expected_unsatisfied_active_requirement_count"] != 0
        or dependency["inherited_conflict_hash_encoding"] != "UTF-8"
        or dependency["inherited_conflict_hash_join"]
        != "sorted unique lines joined by LF without a trailing LF"
        or dependency["requirement_owners"]
        != ["huggingface-hub", "tokenizers", "transformers"]
    ):
        raise ProvisioningError("INVALID_PROFILE", "dependency contract mismatch")

    layout = _expect_keys(
        top["external_layout"],
        {"manifest", "metadata_directory", "venv_directory", "venv_python", "wheelhouse_directory"},
        "external_layout",
    )
    for key, expected in {
        "manifest": "runtime-manifest.json",
        "metadata_directory": "metadata",
        "venv_directory": "venv",
        "venv_python": "venv/Scripts/python.exe",
        "wheelhouse_directory": "wheelhouse",
    }.items():
        if _expect_relative_path(layout[key], f"external_layout.{key}") != expected:
            raise ProvisioningError("INVALID_PROFILE", f"layout mismatch: {key}")

    forbidden = _expect_list(top["forbidden_packages"], "forbidden_packages")
    if forbidden != sorted(set(forbidden)) or forbidden != [
        "directml",
        "nvidia-*",
        "opencv-python",
        "torch",
        "torchaudio",
        "torchvision",
        "triton",
    ]:
        raise ProvisioningError("INVALID_PROFILE", "forbidden package list mismatch")

    installed_rows = _expect_list(top["installed_sources"], "installed_sources")
    observed_sources: dict[str, tuple[int, str]] = {}
    for index, row_value in enumerate(installed_rows):
        row = _expect_keys(row_value, {"bytes", "path", "sha256"}, f"installed_sources[{index}]")
        source_path = _expect_relative_path(row["path"], f"installed_sources[{index}].path")
        if source_path in observed_sources:
            raise ProvisioningError("INVALID_PROFILE", "duplicate installed source path")
        observed_sources[source_path] = (
            _expect_int(row["bytes"], f"installed_sources[{index}].bytes", minimum=1),
            _expect_sha256(row["sha256"], f"installed_sources[{index}].sha256"),
        )
    if observed_sources != EXPECTED_INSTALLED_SOURCES:
        raise ProvisioningError("INVALID_PROFILE", "installed source identities mismatch")

    manual = _expect_keys(
        top["manual_preprocessing"],
        {
            "expected_input_sha256",
            "expected_output_sha256",
            "formula",
            "image_generation",
            "image_mean",
            "image_std",
            "input_shape",
            "output_dtype",
            "output_shape",
            "profile_id",
            "rescale_factor_dtype",
            "rescale_factor_fp32",
            "rescale_factor_json",
        },
        "manual_preprocessing",
    )
    image_generation = _expect_keys(
        manual["image_generation"],
        {
            "checker_blue",
            "checker_green",
            "checker_red",
            "checker_tile_size",
            "gradient_blue",
            "gradient_green",
            "gradient_red",
            "indices_dtype",
            "stack_order",
        },
        "manual_preprocessing.image_generation",
    )
    if image_generation != {
        "checker_blue": "where(checker_base == 0, (7*x) % 256, (11*y) % 256)",
        "checker_green": "(3*x + 5*y) % 256",
        "checker_red": "checker_base",
        "checker_tile_size": 8,
        "gradient_blue": "(x + y) % 256",
        "gradient_green": "y % 256",
        "gradient_red": "x % 256",
        "indices_dtype": "uint16",
        "stack_order": ["gradient", "checker"],
    }:
        raise ProvisioningError("INVALID_PROFILE", "deterministic input formula mismatch")
    if (
        manual["profile_id"] != MANUAL_PROFILE_ID
        or manual["expected_input_sha256"]
        != "b0fa2f6e3b2fbca50cde7e4a2dc3a65a0f788a184aec382365e520d54b468411"
        or manual["expected_output_sha256"]
        != "c0ebe2af6db2e61d052dc58272379beef749a4c5392d4e2204a11862cf172526"
        or manual["image_mean"] != [0.485, 0.456, 0.406]
        or manual["image_std"] != [0.229, 0.224, 0.225]
        or manual["input_shape"] != [2, 224, 224, 3]
        or manual["output_shape"] != [2, 3, 224, 224]
        or manual["output_dtype"] != "float32"
        or manual["rescale_factor_dtype"] != "float32"
        or manual["rescale_factor_json"] != 0.00392156862745098
        or manual["rescale_factor_fp32"] != 0.003921568859368563
    ):
        raise ProvisioningError("INVALID_PROFILE", "manual preprocessing identity mismatch")
    expected_formula = [
        "astype(float32, copy=True)",
        "multiply np.float32(rescale_factor)",
        "subtract np.asarray(mean, dtype=float32)",
        "divide by np.asarray(std, dtype=float32)",
        "transpose NHWC to NCHW",
        "ascontiguousarray float32",
        "finite assertion",
    ]
    if manual["formula"] != expected_formula:
        raise ProvisioningError("INVALID_PROFILE", "manual preprocessing operation order mismatch")

    metadata_rows = _expect_list(top["metadata_files"], "metadata_files")
    observed_metadata: dict[str, tuple[int, str]] = {}
    for index, row_value in enumerate(metadata_rows):
        row = _expect_keys(row_value, {"bytes", "filename", "sha256"}, f"metadata_files[{index}]")
        filename = _safe_basename(row["filename"], f"metadata_files[{index}].filename")
        observed_metadata[filename] = (
            _expect_int(row["bytes"], f"metadata_files[{index}].bytes", minimum=1),
            _expect_sha256(row["sha256"], f"metadata_files[{index}].sha256"),
        )
    if observed_metadata != EXPECTED_METADATA:
        raise ProvisioningError("INVALID_PROFILE", "metadata identity mismatch")

    weights = _expect_keys(
        top["model_weights"], {"expected_present", "forbidden_filenames", "transferred_bytes"}, "model_weights"
    )
    if (
        _expect_bool(weights["expected_present"], "model_weights.expected_present")
        or weights["forbidden_filenames"] != ["model.safetensors", "pytorch_model.bin"]
        or weights["transferred_bytes"] != 0
    ):
        raise ProvisioningError("INVALID_PROFILE", "model weight policy mismatch")

    policies = _expect_keys(
        top["policies"],
        {
            "cpu_only",
            "image_processor_runtime_object_allowed",
            "local_metadata_only_after_provision",
            "model_checkpoint_allowed",
            "network_allowed_during_verify",
            "system_site_packages",
            "torchvision_allowed",
        },
        "policies",
    )
    if policies != {
        "cpu_only": True,
        "image_processor_runtime_object_allowed": False,
        "local_metadata_only_after_provision": True,
        "model_checkpoint_allowed": False,
        "network_allowed_during_verify": False,
        "system_site_packages": True,
        "torchvision_allowed": False,
    }:
        raise ProvisioningError("INVALID_PROFILE", "runtime policies mismatch")

    root = _expect_keys(
        top["runtime_root"],
        {
            "default_template",
            "environment_variable",
            "relative_parts",
            "repository_relation",
            "staging_name_prefix",
        },
        "runtime_root",
    )
    if (
        root["environment_variable"] != "LOCALAPPDATA"
        or root["relative_parts"] != ["VEA", "runtimes", RUNTIME_ID]
        or root["repository_relation"] != "outside_repository"
        or root["default_template"] != f"%LOCALAPPDATA%/VEA/runtimes/{RUNTIME_ID}"
        or root["staging_name_prefix"] != f".{RUNTIME_ID}.staging-"
    ):
        raise ProvisioningError("INVALID_PROFILE", "runtime root policy mismatch")

    target = _expect_keys(top["target"], {"evidence_revision", "model_id", "registry_key"}, "target")
    if target != {
        "evidence_revision": TARGET_REVISION,
        "model_id": TARGET_MODEL_ID,
        "registry_key": "dinov3_vit_b16",
    }:
        raise ProvisioningError("INVALID_PROFILE", "target identity mismatch")

    wheel_rows = _expect_list(top["wheel_files"], "wheel_files")
    observed_wheels: dict[str, tuple[str, int, str]] = {}
    for index, row_value in enumerate(wheel_rows):
        row = _expect_keys(
            row_value,
            {"bytes", "distribution", "filename", "sha256", "version"},
            f"wheel_files[{index}]",
        )
        filename = _safe_basename(row["filename"], f"wheel_files[{index}].filename")
        distribution = _expect_string(row["distribution"], f"wheel_files[{index}].distribution")
        version = _expect_exact_version(row["version"], f"wheel_files[{index}].version")
        if version not in filename:
            raise ProvisioningError("INVALID_PROFILE", "wheel filename/version mismatch")
        observed_wheels[filename] = (
            distribution,
            _expect_int(row["bytes"], f"wheel_files[{index}].bytes", minimum=1),
            _expect_sha256(row["sha256"], f"wheel_files[{index}].sha256"),
        )
    if observed_wheels != EXPECTED_WHEELS:
        raise ProvisioningError("INVALID_PROFILE", "wheel identity mismatch")
    if sum(row[1] for row in observed_wheels.values()) != 14922119:
        raise ProvisioningError("INVALID_PROFILE", "wheel payload mismatch")

    worker = _expect_keys(
        top["worker_protocol"], {"request_schema_id", "response_schema_id"}, "worker_protocol"
    )
    if worker != {
        "request_schema_id": "vea.dinov3_isolated_worker_request.v0",
        "response_schema_id": "vea.dinov3_isolated_worker_response.v0",
    }:
        raise ProvisioningError("INVALID_PROFILE", "worker protocol mismatch")


def _safe_basename(value: object, path: str) -> str:
    text = _expect_string(value, path)
    if PurePosixPath(text).name != text or PureWindowsPath(text).name != text:
        raise ProvisioningError("INVALID_PROFILE", f"{path} must be a basename")
    return text


def load_profile(path: str | Path) -> RuntimeProfile:
    """Load the exact canonical profile and return an immutable representation."""

    profile_path = Path(path).resolve(strict=True)
    values, raw = _load_canonical_json(profile_path, classification="INVALID_PROFILE")
    _validate_profile(values)
    digest = _sha256_bytes(raw)
    if digest != PROFILE_SHA256:
        raise ProvisioningError("INVALID_PROFILE", "repository profile hash mismatch")
    return RuntimeProfile(
        path=profile_path,
        sha256=digest,
        values=_freeze(values),
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_reparse_point(path: Path) -> bool:
    try:
        result = path.lstat()
    except FileNotFoundError:
        return False
    if path.is_symlink():
        return True
    attributes = getattr(result, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _reject_reparse_chain(path: Path, *, stop_at: Path | None = None) -> None:
    current = path
    stop = stop_at.resolve(strict=False) if stop_at is not None else None
    while True:
        if current.exists() and _is_reparse_point(current):
            raise ProvisioningError("PATH_ESCAPE", "runtime path contains a symlink or reparse point")
        if stop is not None and current.resolve(strict=False) == stop:
            break
        if current.parent == current:
            break
        current = current.parent


def resolve_default_runtime_root(
    profile: RuntimeProfile, *, environment: Mapping[str, str] | None = None
) -> Path:
    """Resolve the profile default without creating directories."""

    env = os.environ if environment is None else environment
    variable = str(profile.values["runtime_root"]["environment_variable"])
    base = env.get(variable)
    if not base:
        raise ProvisioningError("DEFAULT_ROOT_UNAVAILABLE", f"{variable} is not set")
    parts = tuple(str(item) for item in profile.values["runtime_root"]["relative_parts"])
    result = Path(base).joinpath(*parts)
    if not result.is_absolute():
        raise ProvisioningError("DEFAULT_ROOT_UNAVAILABLE", "default runtime root is not absolute")
    return result.resolve(strict=False)


def _validate_runtime_root_path(
    runtime_root: Path,
    repository_root: Path,
    *,
    allow_owned_staging: bool = False,
) -> Path:
    if not runtime_root.is_absolute():
        raise ProvisioningError("INVALID_RUNTIME_ROOT", "runtime root must be absolute")
    lexical = Path(os.path.abspath(runtime_root))
    repository = repository_root.resolve(strict=True)
    _reject_reparse_chain(lexical.parent)
    if lexical.exists() and _is_reparse_point(lexical):
        raise ProvisioningError("PATH_ESCAPE", "runtime root is a symlink or reparse point")
    resolved = lexical.resolve(strict=False)
    if (
        resolved == repository
        or _is_relative_to(resolved, repository)
        or _is_relative_to(repository, resolved)
    ):
        raise ProvisioningError("INVALID_RUNTIME_ROOT", "runtime root must be outside repository")
    staging_prefix = f".{RUNTIME_ID}.staging-"
    staging_suffix = (
        lexical.name[len(staging_prefix) :]
        if lexical.name.startswith(staging_prefix)
        else ""
    )
    recognized_staging = (
        allow_owned_staging
        and len(staging_suffix) == 32
        and all(character in "0123456789abcdef" for character in staging_suffix)
    )
    if lexical.name != RUNTIME_ID and not recognized_staging:
        raise ProvisioningError(
            "INVALID_RUNTIME_ROOT",
            "runtime root basename must match the runtime ID",
        )
    return resolved


def plan_provision(
    profile_path: str | Path,
    runtime_root: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    _allow_owned_staging: bool = False,
) -> ProvisionPlan:
    """Build a side-effect-free provision plan for tests and CLI use."""

    profile = load_profile(profile_path)
    repository = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    requested = (
        resolve_default_runtime_root(profile, environment=environment)
        if runtime_root is None
        else Path(runtime_root)
    )
    final_root = _validate_runtime_root_path(
        requested,
        repository,
        allow_owned_staging=_allow_owned_staging,
    )
    return ProvisionPlan(
        profile_path=profile.path,
        repository_root=repository,
        runtime_root=final_root,
        staging_parent=final_root.parent,
        staging_prefix=str(profile.values["runtime_root"]["staging_name_prefix"]),
    )


def _normalized_environment(*, offline: bool = False) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if _SECRET_NAME_RE.search(name):
            environment.pop(name, None)
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PIP_CONFIG_FILE",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_INDEX_URL",
        "PIP_TRUSTED_HOST",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1" if offline else "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    if offline:
        environment.update(
            {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "PIP_NO_INDEX": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    return environment


def run_command(
    args: Sequence[str | os.PathLike[str]],
    *,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 300.0,
    accepted_returncodes: frozenset[int] = frozenset({0}),
) -> CommandResult:
    """Run one argument-list command without a shell and capture text output."""

    command = tuple(os.fspath(item) for item in args)
    try:
        completed = subprocess.run(
            command,
            cwd=None if cwd is None else os.fspath(cwd),
            env=None if environment is None else dict(environment),
            capture_output=True,
            check=False,
            shell=False,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvisioningError("SUBPROCESS_FAILED", "subprocess could not complete") from exc
    result = CommandResult(command, completed.returncode, completed.stdout, completed.stderr)
    if result.returncode not in accepted_returncodes:
        raise ProvisioningError(
            "SUBPROCESS_FAILED", f"subprocess returned exit code {result.returncode}"
        )
    return result


def _normalize_freeze(stdout: str) -> bytes:
    normalized = stdout.replace("\r\n", "\n").replace("\r", "\n")
    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    return normalized.encode("utf-8")


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_pip_conflicts(result: CommandResult) -> tuple[str, ...]:
    lines: set[str] = set()
    for stream in (result.stdout, result.stderr):
        for raw_line in stream.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if _CONFLICT_RE.fullmatch(line):
                lines.add(line)
    return tuple(sorted(lines))


def _conflict_sha256(conflicts: Sequence[str]) -> str:
    return _sha256_bytes("\n".join(sorted(set(conflicts))).encode("utf-8"))


def _current_conflicts(*, runner: Runner = run_command) -> tuple[str, ...]:
    result = runner(
        [sys.executable, "-m", "pip", "check"],
        environment=_normalized_environment(offline=True),
        timeout=120.0,
        accepted_returncodes=frozenset({0, 1}),
    )
    conflicts = _parse_pip_conflicts(result)
    if (
        result.returncode != 1
        or len(conflicts) != EXPECTED_CONFLICT_COUNT
        or _conflict_sha256(conflicts) != EXPECTED_CONFLICT_SHA256
    ):
        raise ProvisioningError("CURRENT_ENVIRONMENT_MISMATCH", "current conflict set mismatch")
    return conflicts


_CURRENT_TORCH_CODE = textwrap.dedent(
    r"""
    import importlib.metadata as md
    import json
    import torch

    print(json.dumps({
        "candidate_local_distribution": False,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "metadata_version": md.version("torch"),
        "runtime_version": str(torch.__version__),
    }, sort_keys=True, separators=(",", ":")))
    """
).strip()


def _validate_current_base(
    profile: RuntimeProfile,
    repository_root: Path,
    *,
    runner: Runner = run_command,
    include_conflicts: bool = True,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if platform_python_version() != "3.12.0":
        raise ProvisioningError("CURRENT_ENVIRONMENT_MISMATCH", "current Python mismatch")
    for relative, expected in EXPECTED_CURRENT_SOURCES.items():
        candidate = repository_root / Path(relative)
        if not candidate.is_file() or _sha256_file(candidate) != expected:
            raise ProvisioningError("CURRENT_SOURCE_MISMATCH", f"source hash mismatch: {relative}")
    for name, expected in EXPECTED_CURRENT_PACKAGES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProvisioningError("CURRENT_ENVIRONMENT_MISMATCH", f"missing package: {name}") from exc
        if actual != expected:
            raise ProvisioningError("CURRENT_ENVIRONMENT_MISMATCH", f"package mismatch: {name}")
    freeze_result = runner(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        environment=_normalized_environment(offline=True),
        timeout=120.0,
    )
    freeze = _normalize_freeze(freeze_result.stdout)
    if len(freeze) != 5206 or freeze.count(b"\n") != 275 or _sha256_bytes(freeze) != CURRENT_FREEZE_SHA256:
        raise ProvisioningError("CURRENT_ENVIRONMENT_MISMATCH", "current freeze mismatch")
    torch_result = runner(
        [sys.executable, "-I", "-c", _CURRENT_TORCH_CODE],
        environment=_normalized_environment(offline=True),
        timeout=180.0,
    )
    torch_facts = _parse_single_json_output(
        torch_result, "CURRENT_ENVIRONMENT_MISMATCH"
    )
    expected_torch = dict(profile.values["current_base"]["torch"])
    if torch_facts != expected_torch:
        raise ProvisioningError("CURRENT_ENVIRONMENT_MISMATCH", "shared torch mismatch")
    conflicts = _current_conflicts(runner=runner) if include_conflicts else ()
    return torch_facts, conflicts


def platform_python_version() -> str:
    return ".".join(str(value) for value in sys.version_info[:3])


def _candidate_python(runtime_root: Path, profile: RuntimeProfile) -> Path:
    relative = Path(str(profile.values["external_layout"]["venv_python"]))
    candidate = runtime_root / relative
    if not candidate.is_file():
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "candidate Python is missing")
    if not _is_relative_to(candidate.resolve(strict=True), runtime_root.resolve(strict=True)):
        raise ProvisioningError("PATH_ESCAPE", "candidate Python escapes runtime root")
    return candidate


_CANDIDATE_IDENTITY_CODE = textwrap.dedent(
    r"""
    import importlib.metadata as md
    import importlib.util
    import json
    import pathlib
    import re
    import sys

    def canonical(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    prefix = pathlib.Path(sys.prefix).resolve()
    local = {}
    for distribution in md.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        location = pathlib.Path(distribution.locate_file("")).resolve()
        try:
            location.relative_to(prefix)
        except ValueError:
            continue
        local[canonical(name)] = distribution.version

    import torch
    result = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "local_packages": dict(sorted(local.items())),
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch_metadata": md.version("torch"),
        "torch_module_inside_venv": False,
        "torch_runtime": str(torch.__version__),
        "torchvision_distribution_present": False,
        "torchvision_module_spec_present": importlib.util.find_spec("torchvision") is not None,
    }
    torch_path = pathlib.Path(torch.__file__).resolve()
    try:
        torch_path.relative_to(prefix)
    except ValueError:
        pass
    else:
        result["torch_module_inside_venv"] = True
    try:
        md.distribution("torchvision")
    except md.PackageNotFoundError:
        pass
    else:
        result["torchvision_distribution_present"] = True
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    """
).strip()


_DEPENDENCY_CODE = textwrap.dedent(
    r"""
    import importlib.metadata as md
    import json
    import re
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    owners = ("huggingface-hub", "tokenizers", "transformers")
    environment = default_environment()
    environment["extra"] = ""
    active = []
    unsatisfied = []
    for owner in owners:
        distribution = md.distribution(owner)
        for raw in distribution.requires or ():
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            name = re.sub(r"[-_.]+", "-", requirement.name).lower()
            try:
                version = md.version(requirement.name)
            except md.PackageNotFoundError:
                version = None
            satisfied = version is not None and (
                not requirement.specifier or requirement.specifier.contains(version, prereleases=True)
            )
            row = {
                "name": name,
                "owner": owner,
                "requirement": str(requirement),
                "satisfied": bool(satisfied),
                "version": version,
            }
            active.append(row)
            if not satisfied:
                unsatisfied.append(row)
    print(json.dumps({"active": active, "unsatisfied": unsatisfied}, sort_keys=True, separators=(",", ":")))
    """
).strip()


_SOURCE_CODE = textwrap.dedent(
    r"""
    import hashlib
    import importlib.util
    import json
    import pathlib

    paths = json.loads(__import__("sys").argv[1])
    spec = importlib.util.find_spec("transformers")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("transformers package missing")
    package = pathlib.Path(tuple(spec.submodule_search_locations)[0]).resolve()
    root = package.parent
    result = []
    for relative in paths:
        path = (root / pathlib.PurePosixPath(relative)).resolve()
        path.relative_to(root)
        data = path.read_bytes()
        result.append({"bytes": len(data), "path": relative, "sha256": hashlib.sha256(data).hexdigest()})
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    """
).strip()


def _parse_single_json_output(result: CommandResult, classification: str) -> Any:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ProvisioningError(classification, "candidate probe output was not one JSON line")
    try:
        return json.loads(lines[0], parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ProvisioningError) as exc:
        raise ProvisioningError(classification, "candidate probe returned invalid JSON") from exc


def _validate_candidate(
    runtime_root: Path,
    profile: RuntimeProfile,
    current_conflicts: tuple[str, ...],
    *,
    runner: Runner = run_command,
) -> dict[str, Any]:
    python = _candidate_python(runtime_root, profile)
    environment = _normalized_environment(offline=True)
    identity_result = runner(
        [python, "-I", "-c", _CANDIDATE_IDENTITY_CODE],
        environment=environment,
        timeout=180.0,
    )
    identity = _parse_single_json_output(identity_result, "RUNTIME_IDENTITY_MISMATCH")
    expected_identity = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "local_packages": EXPECTED_CANDIDATE_LOCAL_PACKAGES,
        "python": "3.12.0",
        "torch_metadata": "2.3.0",
        "torch_module_inside_venv": False,
        "torch_runtime": "2.3.0+cpu",
        "torchvision_distribution_present": False,
        "torchvision_module_spec_present": False,
    }
    if identity != expected_identity:
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "candidate package/provenance mismatch")

    freeze_result = runner(
        [python, "-m", "pip", "freeze", "--all"],
        environment=environment,
        timeout=120.0,
    )
    freeze = _normalize_freeze(freeze_result.stdout)
    if (
        len(freeze) != int(profile.values["candidate"]["expected_freeze"]["bytes"])
        or freeze.count(b"\n")
        != int(profile.values["candidate"]["expected_freeze"]["line_count"])
        or _sha256_bytes(freeze) != CANDIDATE_FREEZE_SHA256
    ):
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "candidate freeze mismatch")

    dependency_result = runner(
        [python, "-I", "-c", _DEPENDENCY_CODE],
        environment=environment,
        timeout=120.0,
    )
    dependency = _parse_single_json_output(dependency_result, "DEPENDENCY_PROFILE_MISMATCH")
    if (
        not isinstance(dependency, dict)
        or set(dependency) != {"active", "unsatisfied"}
        or not isinstance(dependency["active"], list)
        or len(dependency["active"]) != EXPECTED_ACTIVE_REQUIREMENT_COUNT
        or dependency["unsatisfied"] != []
    ):
        raise ProvisioningError("DEPENDENCY_PROFILE_MISMATCH", "active requirement closure mismatch")

    check_result = runner(
        [python, "-m", "pip", "check"],
        environment=environment,
        timeout=120.0,
        accepted_returncodes=frozenset({0, 1}),
    )
    candidate_conflicts = _parse_pip_conflicts(check_result)
    if check_result.returncode != 1 or candidate_conflicts != current_conflicts:
        raise ProvisioningError("DEPENDENCY_PROFILE_MISMATCH", "candidate added a dependency conflict")

    source_paths = sorted(EXPECTED_INSTALLED_SOURCES)
    source_result = runner(
        [python, "-I", "-c", _SOURCE_CODE, json.dumps(source_paths, separators=(",", ":"))],
        environment=environment,
        timeout=120.0,
    )
    source_rows = _parse_single_json_output(source_result, "RUNTIME_IDENTITY_MISMATCH")
    observed_sources: dict[str, tuple[int, str]] = {}
    if not isinstance(source_rows, list):
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "source probe result is not an array")
    for row in source_rows:
        if not isinstance(row, dict) or set(row) != {"bytes", "path", "sha256"}:
            raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "invalid source probe row")
        observed_sources[str(row["path"])] = (int(row["bytes"]), str(row["sha256"]))
    if observed_sources != EXPECTED_INSTALLED_SOURCES:
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "installed source hash mismatch")

    return {
        "active_requirement_count": len(dependency["active"]),
        "candidate_freeze_bytes": len(freeze),
        "candidate_freeze_line_count": freeze.count(b"\n"),
        "candidate_freeze_sha256": _sha256_bytes(freeze),
        "candidate_identity": identity,
        "inherited_conflict_count": len(candidate_conflicts),
        "inherited_conflict_sha256": _conflict_sha256(candidate_conflicts),
        "installed_sources": [dict(row) for row in source_rows],
        "new_conflict_count": 0,
        "unsatisfied_active_requirement_count": 0,
    }


def _validate_exact_directory(
    directory: Path,
    expected: Mapping[str, tuple[int, str]],
    *,
    classification: str,
) -> None:
    if not directory.is_dir() or _is_reparse_point(directory):
        raise ProvisioningError(classification, "required runtime directory is missing or unsafe")
    entries = {item.name: item for item in directory.iterdir()}
    if set(entries) != set(expected):
        raise ProvisioningError(classification, "runtime directory inventory mismatch")
    for filename, (size, digest) in expected.items():
        path = entries[filename]
        if (
            not path.is_file()
            or _is_reparse_point(path)
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise ProvisioningError(classification, f"runtime file identity mismatch: {filename}")


def _manifest_value(
    profile: RuntimeProfile,
    current_torch: Mapping[str, Any],
    candidate: Mapping[str, Any],
    provisioned_at_utc: str,
) -> dict[str, Any]:
    return {
        "candidate_freeze": {
            "bytes": int(candidate["candidate_freeze_bytes"]),
            "line_count": int(candidate["candidate_freeze_line_count"]),
            "sha256": str(candidate["candidate_freeze_sha256"]),
        },
        "candidate_local_packages": dict(EXPECTED_CANDIDATE_LOCAL_PACKAGES),
        "current_base_freeze_sha256": CURRENT_FREEZE_SHA256,
        "dependency_validation": {
            "active_requirement_count": int(candidate["active_requirement_count"]),
            "inherited_conflict_count": int(candidate["inherited_conflict_count"]),
            "inherited_conflict_sha256": str(candidate["inherited_conflict_sha256"]),
            "new_conflict_count": int(candidate["new_conflict_count"]),
            "unsatisfied_active_requirement_count": int(
                candidate["unsatisfied_active_requirement_count"]
            ),
        },
        "forbidden_candidate_package_count": 0,
        "installed_sources": [
            {
                "bytes": int(row["bytes"]),
                "path": str(row["path"]),
                "sha256": str(row["sha256"]),
            }
            for row in profile.values["installed_sources"]
        ],
        "manual_preprocessing_profile_id": MANUAL_PROFILE_ID,
        "metadata_files": [
            {"bytes": size, "filename": name, "sha256": digest}
            for name, (size, digest) in sorted(EXPECTED_METADATA.items())
        ],
        "model_weight_present": False,
        "network_independent_verification_status": "VERIFIED",
        "provisioned_at_utc": provisioned_at_utc,
        "python_version": "3.12.0",
        "repository_profile_sha256": profile.sha256,
        "runtime_id": profile.runtime_id,
        "schema_id": RUNTIME_MANIFEST_SCHEMA_ID,
        "shared_torch": dict(current_torch),
        "torchvision": {
            "distribution_count": 0,
            "module_count": 0,
        },
        "wheel_files": [
            {
                "bytes": size,
                "distribution": distribution,
                "filename": name,
                "sha256": digest,
                "version": EXPECTED_CANDIDATE_LOCAL_PACKAGES[distribution],
            }
            for name, (distribution, size, digest) in sorted(EXPECTED_WHEELS.items())
        ],
    }


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_manifest_shape(manifest: dict[str, Any]) -> None:
    expected_keys = {
        "candidate_freeze",
        "candidate_local_packages",
        "current_base_freeze_sha256",
        "dependency_validation",
        "forbidden_candidate_package_count",
        "installed_sources",
        "manual_preprocessing_profile_id",
        "metadata_files",
        "model_weight_present",
        "network_independent_verification_status",
        "provisioned_at_utc",
        "python_version",
        "repository_profile_sha256",
        "runtime_id",
        "schema_id",
        "shared_torch",
        "torchvision",
        "wheel_files",
    }
    if set(manifest) != expected_keys:
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "external manifest fields mismatch")
    if not _valid_utc_timestamp(manifest["provisioned_at_utc"]):
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "invalid provision timestamp")
    for key in manifest:
        if _SECRET_NAME_RE.search(key):
            raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "secret-like manifest field")


def _write_canonical_json(path: Path, value: object) -> None:
    data = _canonical_json_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scan_for_forbidden_runtime_files(runtime_root: Path) -> None:
    forbidden_names = {
        ".netrc",
        ".hf_token",
        "auth.json",
        "credentials",
        "credentials.ini",
        "credentials.json",
        "model.safetensors",
        "pytorch_model.bin",
        "stored_tokens",
        "stored_tokens.json",
        "token",
        "tokens.json",
    }
    for item in runtime_root.rglob("*"):
        if _is_reparse_point(item):
            raise ProvisioningError("PATH_ESCAPE", "runtime contains a symlink or reparse point")
        if item.is_file() and item.name.casefold() in forbidden_names:
            raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "forbidden runtime file present")
        if item.is_dir() and item.name.startswith("models--facebook--dinov3"):
            raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "model cache present")


def verify_runtime(
    profile_path: str | Path,
    runtime_root: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
    runner: Runner = run_command,
    _allow_owned_staging: bool = False,
) -> VerificationResult:
    """Read-only, network-free verification of one complete persistent runtime."""

    profile = load_profile(profile_path)
    plan = plan_provision(
        profile_path,
        runtime_root,
        repository_root=repository_root,
        _allow_owned_staging=_allow_owned_staging,
    )
    root = plan.runtime_root
    if not root.is_dir() or _is_reparse_point(root):
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "runtime root is missing or unsafe")
    top_level = {item.name for item in root.iterdir()}
    if top_level != {"metadata", "runtime-manifest.json", "venv", "wheelhouse"}:
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "runtime top-level layout mismatch")

    metadata_expected = {name: value for name, value in EXPECTED_METADATA.items()}
    wheel_expected = {
        name: (size, digest) for name, (_, size, digest) in EXPECTED_WHEELS.items()
    }
    _validate_exact_directory(
        root / "metadata", metadata_expected, classification="RUNTIME_IDENTITY_MISMATCH"
    )
    _validate_exact_directory(
        root / "wheelhouse", wheel_expected, classification="RUNTIME_IDENTITY_MISMATCH"
    )
    _scan_for_forbidden_runtime_files(root)

    manifest_path = root / "runtime-manifest.json"
    manifest, manifest_raw = _load_canonical_json(
        manifest_path, classification="RUNTIME_IDENTITY_MISMATCH"
    )
    _validate_manifest_shape(manifest)
    if manifest["schema_id"] != RUNTIME_MANIFEST_SCHEMA_ID:
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "runtime manifest schema mismatch")

    current_torch, current_conflicts = _validate_current_base(
        profile, plan.repository_root, runner=runner, include_conflicts=True
    )
    candidate = _validate_candidate(
        root, profile, current_conflicts, runner=runner
    )
    expected_manifest = _manifest_value(
        profile,
        current_torch,
        candidate,
        str(manifest["provisioned_at_utc"]),
    )
    if manifest != expected_manifest:
        raise ProvisioningError("RUNTIME_IDENTITY_MISMATCH", "runtime manifest content mismatch")
    return VerificationResult(
        runtime_id=profile.runtime_id,
        runtime_manifest_sha256=_sha256_bytes(manifest_raw),
        repository_profile_sha256=profile.sha256,
        candidate_freeze_sha256=str(candidate["candidate_freeze_sha256"]),
        inherited_conflict_sha256=str(candidate["inherited_conflict_sha256"]),
    )


def _create_parent_directories(parent: Path) -> tuple[Path, ...]:
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    _reject_reparse_chain(current)
    created: list[Path] = []
    for directory in reversed(missing):
        directory.mkdir()
        created.append(directory)
    return tuple(created)


def _require_free_space(path: Path) -> None:
    existing = path
    while not existing.exists():
        if existing.parent == existing:
            raise ProvisioningError("INSUFFICIENT_FREE_SPACE", "runtime volume is unavailable")
        existing = existing.parent
    if not existing.is_dir() or _is_reparse_point(existing):
        raise ProvisioningError("INVALID_RUNTIME_ROOT", "runtime parent is not a safe directory")
    try:
        free = shutil.disk_usage(existing).free
    except OSError as exc:
        raise ProvisioningError(
            "INSUFFICIENT_FREE_SPACE", "unable to determine runtime-volume free space"
        ) from exc
    if free < MINIMUM_FREE_BYTES:
        raise ProvisioningError(
            "INSUFFICIENT_FREE_SPACE", "runtime volume has less than 4 GiB free"
        )


def _remove_created_empty_parents(created: Sequence[Path]) -> None:
    for directory in reversed(created):
        try:
            directory.rmdir()
        except OSError:
            break


def _cleanup_owned_staging(staging: Path, parent: Path, prefix: str, nonce: str) -> None:
    expected_name = prefix + nonce
    lexical = Path(os.path.abspath(staging))
    parent_lexical = Path(os.path.abspath(parent))
    if lexical.parent != parent_lexical or lexical.name != expected_name:
        raise ProvisioningError("UNSAFE_CLEANUP_REFUSED", "staging cleanup target is not owned")
    if not lexical.exists():
        return
    if _is_reparse_point(lexical):
        raise ProvisioningError("UNSAFE_CLEANUP_REFUSED", "staging root is a reparse point")
    shutil.rmtree(lexical)


def _cleanup_owned_work_directory(work: Path, staging: Path) -> None:
    lexical = Path(os.path.abspath(work))
    staging_lexical = Path(os.path.abspath(staging))
    if lexical.parent != staging_lexical or lexical.name != ".provision-work":
        raise ProvisioningError("UNSAFE_CLEANUP_REFUSED", "work cleanup target is not owned")
    if not lexical.exists():
        return
    if _is_reparse_point(lexical):
        raise ProvisioningError("UNSAFE_CLEANUP_REFUSED", "work directory is a reparse point")
    shutil.rmtree(lexical)


def _download_wheels(
    wheelhouse: Path,
    pip_cache: Path,
    *,
    runner: Runner = run_command,
) -> None:
    wheelhouse.mkdir()
    pip_cache.mkdir(parents=True)
    environment = _normalized_environment(offline=False)
    environment.update(
        {
            "PIP_CACHE_DIR": os.fspath(pip_cache),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
        }
    )
    runner(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            "--no-input",
            "--cache-dir",
            pip_cache,
            "download",
            "--dest",
            wheelhouse,
            "--only-binary=:all:",
            "--no-deps",
            "--index-url",
            "https://pypi.org/simple",
            "huggingface-hub==0.36.2",
            "tokenizers==0.22.2",
            "transformers==4.56.0",
        ],
        environment=environment,
        timeout=600.0,
    )
    expected = {name: (size, digest) for name, (_, size, digest) in EXPECTED_WHEELS.items()}
    _validate_exact_directory(
        wheelhouse, expected, classification="WHEELHOUSE_IDENTITY_MISMATCH"
    )


def _download_metadata(metadata: Path, hub_cache: Path) -> None:
    try:
        metadata.mkdir()
        hub_cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProvisioningError(
            "METADATA_DOWNLOAD_FAILED", "unable to create task-owned metadata staging"
        ) from exc
    if any(hub_cache.iterdir()):
        raise ProvisioningError(
            "METADATA_DOWNLOAD_FAILED", "task-owned Hub cache must start empty"
        )
    endpoint_override = os.environ.get("HF_ENDPOINT")
    if endpoint_override and endpoint_override.rstrip("/") != "https://huggingface.co":
        raise ProvisioningError("UNAUTHORIZED_ENDPOINT", "HF_ENDPOINT override is not allowed")
    previous_dont_write_bytecode = sys.dont_write_bytecode
    hub_file_download: Any | None = None
    original_http_get: Any | None = None
    original_hf_transfer: bool | None = None
    sys.dont_write_bytecode = True
    try:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub import file_download as hub_file_download
        except Exception as exc:
            raise ProvisioningError(
                "METADATA_DOWNLOAD_FAILED", "Hugging Face Hub is unavailable"
            ) from exc
        original_http_get = hub_file_download.http_get
        original_hf_transfer = bool(hub_file_download.HF_HUB_ENABLE_HF_TRANSFER)

        def bounded_http_get(url: str, temp_file: Any, **kwargs: Any) -> None:
            expected_size = kwargs.get("expected_size")
            if (
                type(expected_size) is not int
                or expected_size < 0
                or expected_size > MAX_METADATA_RESPONSE_BYTES
            ):
                raise ProvisioningError(
                    "METADATA_RESPONSE_TOO_LARGE",
                    "metadata response size is missing or exceeds the authorized cap",
                )
            original_http_get(
                url,
                _BoundedResponseWriter(temp_file, MAX_METADATA_RESPONSE_BYTES),
                **kwargs,
            )

        hub_file_download.HF_HUB_ENABLE_HF_TRANSFER = False
        hub_file_download.http_get = bounded_http_get
        for filename in sorted(EXPECTED_METADATA):
            try:
                downloaded = Path(
                    hf_hub_download(
                        repo_id=TARGET_MODEL_ID,
                        filename=filename,
                        revision=TARGET_REVISION,
                        token=True,
                        cache_dir=os.fspath(hub_cache),
                        local_files_only=False,
                    )
                )
            except ProvisioningError:
                raise
            except Exception as exc:
                raise ProvisioningError(
                    "METADATA_DOWNLOAD_FAILED", "exact metadata access failed"
                ) from exc
            size, digest = EXPECTED_METADATA[filename]
            if (
                not downloaded.is_file()
                or downloaded.stat().st_size != size
                or _sha256_file(downloaded) != digest
            ):
                raise ProvisioningError(
                    "METADATA_IDENTITY_MISMATCH", "downloaded metadata mismatch"
                )
            destination = metadata / filename
            with downloaded.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
    finally:
        if hub_file_download is not None and original_http_get is not None:
            hub_file_download.http_get = original_http_get
        if hub_file_download is not None and original_hf_transfer is not None:
            hub_file_download.HF_HUB_ENABLE_HF_TRANSFER = original_hf_transfer
        sys.dont_write_bytecode = previous_dont_write_bytecode
    _validate_exact_directory(
        metadata, EXPECTED_METADATA, classification="METADATA_IDENTITY_MISMATCH"
    )


def _create_and_install_venv(
    staging: Path,
    *,
    runner: Runner = run_command,
) -> None:
    environment = _normalized_environment(offline=True)
    # Build under a short, same-volume sibling path. Pip can otherwise exceed
    # legacy MAX_PATH while unpacking Transformers beneath the descriptive
    # staging name. A Windows venv remains relocatable within the same volume
    # because its interpreter discovers pyvenv.cfg adjacent to the executable.
    with tempfile.TemporaryDirectory(
        prefix=".vea-venv-", dir=staging.parent
    ) as build_root_text:
        build_root = Path(build_root_text)
        venv = build_root / "venv"
        runner(
            [sys.executable, "-m", "venv", "--system-site-packages", venv],
            environment=environment,
            timeout=300.0,
        )
        python = venv / "Scripts" / "python.exe"
        if not python.is_file():
            raise ProvisioningError(
                "VENV_CREATION_FAILED", "candidate Python was not created"
            )
        runner(
            [
                python,
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "--no-input",
                "install",
                "--no-index",
                "--no-deps",
                "--no-compile",
                "--find-links",
                staging / "wheelhouse",
                "huggingface-hub==0.36.2",
                "tokenizers==0.22.2",
                "transformers==4.56.0",
            ],
            environment=environment,
            timeout=600.0,
        )
        venv.rename(staging / "venv")


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def provision_runtime(
    profile_path: str | Path,
    runtime_root: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
    runner: Runner = run_command,
) -> str:
    """Provision a fresh root or return the exact-valid idempotent status marker."""

    profile = load_profile(profile_path)
    plan = plan_provision(
        profile_path,
        runtime_root,
        repository_root=repository_root,
    )
    final = plan.runtime_root
    if final.exists() or final.is_symlink():
        try:
            verify_runtime(
                profile_path,
                final,
                repository_root=plan.repository_root,
                runner=runner,
            )
        except ProvisioningError as exc:
            raise RuntimeCollisionError() from exc
        return "DINOV3_PERSISTENT_RUNTIME_ALREADY_VALID"

    current_torch, current_conflicts = _validate_current_base(
        profile, plan.repository_root, runner=runner, include_conflicts=True
    )
    _require_free_space(plan.staging_parent)
    created_parents = _create_parent_directories(plan.staging_parent)
    nonce = uuid.uuid4().hex
    staging = plan.staging_parent / f"{plan.staging_prefix}{nonce}"
    try:
        staging.mkdir(exist_ok=False)
    except Exception:
        _remove_created_empty_parents(created_parents)
        raise
    try:
        work = staging / ".provision-work"
        work.mkdir()
        _download_wheels(staging / "wheelhouse", work / "pip-cache", runner=runner)
        # Hugging Face's nested cache layout can exceed the legacy Windows path
        # limit beneath the deliberately descriptive staging-root name. Keep the
        # cache task-owned and explicit, but use the short system temp root and
        # remove it immediately after the exact metadata files are copied.
        with tempfile.TemporaryDirectory(prefix="vea_t007a2_hf_") as hub_cache:
            _download_metadata(staging / "metadata", Path(hub_cache))
        _cleanup_owned_work_directory(work, staging)
        _create_and_install_venv(staging, runner=runner)
        candidate = _validate_candidate(
            staging, profile, current_conflicts, runner=runner
        )
        manifest = _manifest_value(profile, current_torch, candidate, _timestamp_now())
        _write_canonical_json(staging / "runtime-manifest.json", manifest)
        verify_runtime(
            profile_path,
            staging,
            repository_root=plan.repository_root,
            runner=runner,
            _allow_owned_staging=True,
        )
        if final.exists() or final.is_symlink():
            raise RuntimeCollisionError("runtime root appeared before publication")
        try:
            staging.rename(final)
        except FileExistsError as exc:
            raise RuntimeCollisionError("runtime root appeared during publication") from exc
    except Exception:
        _cleanup_owned_staging(staging, plan.staging_parent, plan.staging_prefix, nonce)
        _remove_created_empty_parents(created_parents)
        raise
    return "DINOV3_PERSISTENT_RUNTIME_PROVISIONED"


def _default_profile_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "runtime_profiles" / "dinov3_isolated_manual_v0.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("print-default-root", "provision", "verify"),
    )
    parser.add_argument("--profile", type=Path, default=_default_profile_path())
    parser.add_argument("--runtime-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile = load_profile(args.profile)
        repository = Path(__file__).resolve().parents[1]
        if args.command == "print-default-root":
            root = (
                resolve_default_runtime_root(profile)
                if args.runtime_root is None
                else _validate_runtime_root_path(args.runtime_root, repository)
            )
            print(root)
            return 0
        if args.command == "verify":
            verify_runtime(
                profile.path,
                args.runtime_root,
                repository_root=repository,
            )
            print("DINOV3_PERSISTENT_RUNTIME_VERIFY_OK")
            return 0
        marker = provision_runtime(
            profile.path,
            args.runtime_root,
            repository_root=repository,
        )
        print(marker)
        return 0
    except ProvisioningError as exc:
        print(f"{exc.classification}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
