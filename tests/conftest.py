"""Shared deterministic fixtures for contract tests."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, NoReturn

import numpy as np
import pytest

from vision_encoder_alignment.feature_artifacts import (
    ARTIFACT_SCHEMA_ID,
    FeatureArtifactMetadata,
    canonical_tensor_keys,
)


_DINOV3_LIVE_PROFILE_RELATIVE_PATH = (
    "configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json"
)
_DINOV3_LIVE_PROFILE_SHA256 = (
    "60d0d0d7d37c6b4d47997e09619204f9e1eddb96118d5ee3528d9e6fcd5fe829"
)
_DINOV3_REGISTRY_PIN_PROFILE_SHA256 = (
    "3281b94e188230c521e674cae301768a335dd635e7401b51acafeae053a137bc"
)
_DINOV3_LIVE_PROFILE_BYTES = 12_506
_DINOV3_REGISTRY_PIN_PROFILE_BYTES = 8_885
_DINOV3_RUNTIME_ID = "dinov3-vitb16-r5931719-py312-torch230-tf4560-manual-v1"
_DINOV3_SNAPSHOT_ID = "dinov3-vitb16-r5931719-safetensors-v1"
_DINOV3_REGISTRY_PIN_HASHES = {
    "configs/runtime_profiles/dinov3_isolated_manual_v0.json": (
        "1fe6ba3ef691e97cfb1cb8f3ffc7af7a4590f0bf75de0686c3536df4828900c4"
    ),
    "configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json": (
        "e7181b80924f3a95d64a83a3ec2fee88a872de70330656ea6c88a0fe2ad11308"
    ),
    "scripts/provision_dinov3_runtime.py": (
        "b2d679732b77f567bd74d1bbde5cdbaa0a42543abcffcec5219180774d86cd9e"
    ),
    "src/vision_encoder_alignment/snapshot_boundary.py": (
        "19eaaa8abee768cc325588654471dabca3fa96c24ac2e4937b0b9897751a7c1c"
    ),
    "scripts/acquire_dinov3_snapshot.py": (
        "e185c7536a03b0101e3638a1a1f8a9afc07ce63e1bea5cde5596bcec4894f0a1"
    ),
    "runtime/runtime-manifest.json": (
        "3b17f1158a9d9db4b6213904825528b0a593839be44fe05cc6336514a154d80a"
    ),
    "snapshot/snapshot-manifest.json": (
        "d5de6075023ca0b7a5d2f4fb1fa04add8f489f9d9613671c9cb475b3045caa25"
    ),
    _DINOV3_LIVE_PROFILE_RELATIVE_PATH: _DINOV3_REGISTRY_PIN_PROFILE_SHA256,
}
_DINOV3_REGISTRY_PIN_SOURCE_FILES = [
    {
        "path": "configs/models.yaml",
        "sha256": "2c75cc26a27febc633b2bad479779327f35abbf36160df2805f43a4e33f48244",
    },
    {
        "path": "pyproject.toml",
        "sha256": "45f3ab3f1d3267c51f82babf47fe2f53011192ccf317494931aba70478ba9c7e",
    },
    {
        "path": "src/vision_encoder_alignment/extractors/__init__.py",
        "sha256": "6c49c2800964c6b5d934aa55708c833da5d8fb1bafff1dabb344140f632838f6",
    },
    {
        "path": "src/vision_encoder_alignment/extractors/clip.py",
        "sha256": "e2540777a9e72ccec50992c44d2cfbfb034acd42d40cad59fbc6902ea755f45b",
    },
    {
        "path": "src/vision_encoder_alignment/extractors/dinov3.py",
        "sha256": "c7f7556e0e868adb6f2324cc063853abee816af21f35fc164a17326485c888f2",
    },
    {
        "path": "src/vision_encoder_alignment/extractors/siglip2.py",
        "sha256": "9988510a726e313146e454264b010a8f1c0784e26b1e274f04d571d39980c06c",
    },
]


def pytest_configure(config: pytest.Config) -> None:
    """Keep same-named unit/integration modules isolated during collection."""

    if config.option.importmode == "prepend":
        config.option.importmode = "importlib"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_non_finite(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        value[key] = item
    return value


def _strict_json(payload: bytes) -> dict[str, Any]:
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_non_finite,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _replace_assignment_value(payload: bytes, name: str, expression: str) -> bytes:
    """Replace one top-level Python assignment value with exact source text."""

    text = payload.decode("utf-8")
    tree = ast.parse(text)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(matches) == 1
    value = matches[0].value
    assert value.end_lineno is not None and value.end_col_offset is not None
    lines = text.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: value.lineno - 1]) + value.col_offset
    end = sum(len(line) for line in lines[: value.end_lineno - 1]) + value.end_col_offset
    return (text[:start] + expression + text[end:]).encode("utf-8")


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    assert path.is_file() and not path.is_symlink()
    assert path.read_bytes() == payload


@pytest.fixture(scope="session")
def dinov3_live_production_profile_path() -> Path:
    """Return the actual exact live production-v1 extractor profile."""

    repository = Path(__file__).resolve().parents[1]
    live_profile = repository / _DINOV3_LIVE_PROFILE_RELATIVE_PATH
    live_bytes = live_profile.read_bytes()
    live = _strict_json(live_bytes)
    assert len(live_bytes) == _DINOV3_LIVE_PROFILE_BYTES
    assert _sha256(live_bytes) == _DINOV3_LIVE_PROFILE_SHA256
    assert live["schema_id"] == "vea.dinov3_isolated_extractor_profile.v1"
    assert isinstance(live["production_sources"], list)
    assert len(live["production_sources"]) == 18
    return live_profile.resolve()


def reconstruct_dinov3_registry_pin_graph(graph_root: Path) -> Path:
    """Reconstruct the exact historical registry-pin graph at an external root."""

    repository = Path(__file__).resolve().parents[1]
    live_profile = repository / _DINOV3_LIVE_PROFILE_RELATIVE_PATH
    live_bytes = live_profile.read_bytes()
    assert len(live_bytes) == _DINOV3_LIVE_PROFILE_BYTES
    assert _sha256(live_bytes) == _DINOV3_LIVE_PROFILE_SHA256
    local = Path(os.environ["LOCALAPPDATA"])
    live_runtime_manifest = (
        local / "VEA" / "runtimes" / _DINOV3_RUNTIME_ID / "runtime-manifest.json"
    )
    live_snapshot_manifest = (
        local / "VEA" / "snapshots" / _DINOV3_SNAPSHOT_ID / "snapshot-manifest.json"
    )
    live = _strict_json(live_bytes)

    graph_root = graph_root.resolve()
    graph_root.mkdir(parents=True, exist_ok=False)
    graph_repository = graph_root / "repository"
    graph_runtime = graph_root / "runtime"
    graph_snapshot = graph_root / "snapshot"

    source_rows = live["production_sources"]
    assert [row["path"] for row in source_rows] == sorted(row["path"] for row in source_rows)
    assert len({row["path"] for row in source_rows}) == 18
    for row in source_rows:
        source = repository / row["path"]
        payload = source.read_bytes()
        assert _sha256(payload) == row["sha256"]
        _write_exact(graph_repository / row["path"], payload)

    runtime_relative = "configs/runtime_profiles/dinov3_isolated_manual_v0.json"
    runtime_value = _strict_json((repository / runtime_relative).read_bytes())
    runtime_value["current_base"]["source_files"] = copy.deepcopy(
        _DINOV3_REGISTRY_PIN_SOURCE_FILES
    )
    _write_exact(graph_repository / runtime_relative, _canonical_json(runtime_value))

    snapshot_relative = "configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json"
    snapshot_value = _strict_json((repository / snapshot_relative).read_bytes())
    snapshot_value["persistent_runtime"]["profile_sha256"] = (
        _DINOV3_REGISTRY_PIN_HASHES[runtime_relative]
    )
    _write_exact(graph_repository / snapshot_relative, _canonical_json(snapshot_value))

    provisioner_relative = "scripts/provision_dinov3_runtime.py"
    provisioner = (repository / provisioner_relative).read_bytes()
    provisioner = _replace_assignment_value(
        provisioner,
        "PROFILE_SHA256",
        f'"{_DINOV3_REGISTRY_PIN_HASHES[runtime_relative]}"',
    )
    provisioner = _replace_assignment_value(
        provisioner,
        "EXPECTED_CURRENT_SOURCES",
        """{
    "configs/models.yaml": "2c75cc26a27febc633b2bad479779327f35abbf36160df2805f43a4e33f48244",
    "pyproject.toml": "45f3ab3f1d3267c51f82babf47fe2f53011192ccf317494931aba70478ba9c7e",
    "src/vision_encoder_alignment/extractors/__init__.py": (
        "6c49c2800964c6b5d934aa55708c833da5d8fb1bafff1dabb344140f632838f6"
    ),
    "src/vision_encoder_alignment/extractors/clip.py": (
        "e2540777a9e72ccec50992c44d2cfbfb034acd42d40cad59fbc6902ea755f45b"
    ),
    "src/vision_encoder_alignment/extractors/dinov3.py": (
        "c7f7556e0e868adb6f2324cc063853abee816af21f35fc164a17326485c888f2"
    ),
    "src/vision_encoder_alignment/extractors/siglip2.py": (
        "9988510a726e313146e454264b010a8f1c0784e26b1e274f04d571d39980c06c"
    ),
}""",
    )
    _write_exact(graph_repository / provisioner_relative, provisioner)

    boundary_relative = "src/vision_encoder_alignment/snapshot_boundary.py"
    boundary = _replace_assignment_value(
        (repository / boundary_relative).read_bytes(),
        "PERSISTENT_RUNTIME_PROFILE_SHA256",
        f'''(
    "{_DINOV3_REGISTRY_PIN_HASHES[runtime_relative]}"
)''',
    )
    _write_exact(graph_repository / boundary_relative, boundary)

    acquisition_relative = "scripts/acquire_dinov3_snapshot.py"
    acquisition = (repository / acquisition_relative).read_bytes()
    for name, digest in (
        ("EXPECTED_PROFILE_SHA256", _DINOV3_REGISTRY_PIN_HASHES[snapshot_relative]),
        ("EXPECTED_RUNTIME_PROFILE_SHA256", _DINOV3_REGISTRY_PIN_HASHES[runtime_relative]),
        (
            "EXPECTED_REGISTRY_SHA256",
            "2c75cc26a27febc633b2bad479779327f35abbf36160df2805f43a4e33f48244",
        ),
    ):
        acquisition = _replace_assignment_value(
            acquisition,
            name,
            f'''(
    "{digest}"
)''',
        )
    _write_exact(graph_repository / acquisition_relative, acquisition)

    runtime_manifest = _strict_json(live_runtime_manifest.read_bytes())
    runtime_manifest["repository_profile_sha256"] = _DINOV3_REGISTRY_PIN_HASHES[
        runtime_relative
    ]
    _write_exact(
        graph_runtime / "runtime-manifest.json", _canonical_json(runtime_manifest)
    )

    snapshot_manifest = _strict_json(live_snapshot_manifest.read_bytes())
    snapshot_manifest["repository_profile_sha256"] = _DINOV3_REGISTRY_PIN_HASHES[
        snapshot_relative
    ]
    _write_exact(
        graph_snapshot / "snapshot-manifest.json", _canonical_json(snapshot_manifest)
    )

    registry_profile = copy.deepcopy(live)
    registry_profile["schema_id"] = "vea.dinov3_isolated_extractor_profile.v0"
    del registry_profile["production_sources"]
    registry_profile["identities"].update(
        {
            "runtime_profile_sha256": _DINOV3_REGISTRY_PIN_HASHES[runtime_relative],
            "snapshot_profile_sha256": _DINOV3_REGISTRY_PIN_HASHES[snapshot_relative],
            "persistent_runtime_manifest_sha256": _DINOV3_REGISTRY_PIN_HASHES[
                "runtime/runtime-manifest.json"
            ],
            "external_snapshot_manifest_sha256": _DINOV3_REGISTRY_PIN_HASHES[
                "snapshot/snapshot-manifest.json"
            ],
        }
    )
    ordered_registry_profile: dict[str, Any] = {}
    for key, value in registry_profile.items():
        if key == "tap_catalog":
            continue
        ordered_registry_profile[key] = value
        if key == "tap_mapping":
            ordered_registry_profile["tap_catalog"] = registry_profile["tap_catalog"]
    registry_profile_bytes = (
        json.dumps(ordered_registry_profile, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    assert len(registry_profile_bytes) == _DINOV3_REGISTRY_PIN_PROFILE_BYTES
    _write_exact(
        graph_repository / _DINOV3_LIVE_PROFILE_RELATIVE_PATH,
        registry_profile_bytes,
    )

    graph_paths = {
        **{
            relative: graph_repository / relative
            for relative in _DINOV3_REGISTRY_PIN_HASHES
            if not relative.startswith(("runtime/", "snapshot/"))
        },
        "runtime/runtime-manifest.json": graph_runtime / "runtime-manifest.json",
        "snapshot/snapshot-manifest.json": graph_snapshot / "snapshot-manifest.json",
    }
    assert set(graph_paths) == set(_DINOV3_REGISTRY_PIN_HASHES)
    assert all(
        _sha256(graph_paths[relative].read_bytes()) == digest
        for relative, digest in _DINOV3_REGISTRY_PIN_HASHES.items()
    )
    return graph_root


@pytest.fixture(scope="session")
def dinov3_registry_pin_graph_path(
    tmp_path_factory: pytest.TempPathFactory,
    dinov3_live_production_profile_path: Path,
) -> Path:
    """Yield one exact external historical registry-pin graph."""

    directory = tmp_path_factory.mktemp("dinov3-registry-pin-v0")
    graph_root = reconstruct_dinov3_registry_pin_graph(directory / "graph")
    try:
        yield graph_root
    finally:
        if directory.exists():
            shutil.rmtree(directory)
        assert not directory.exists()
        assert _sha256(dinov3_live_production_profile_path.read_bytes()) == (
            _DINOV3_LIVE_PROFILE_SHA256
        )


@pytest.fixture(scope="session")
def dinov3_registry_pin_profile_path(dinov3_registry_pin_graph_path: Path) -> Path:
    """Return the external exact historical v0 extractor profile."""

    path = (
        dinov3_registry_pin_graph_path
        / "repository"
        / _DINOV3_LIVE_PROFILE_RELATIVE_PATH
    ).resolve()
    payload = path.read_bytes()
    assert len(payload) == _DINOV3_REGISTRY_PIN_PROFILE_BYTES
    assert _sha256(payload) == _DINOV3_REGISTRY_PIN_PROFILE_SHA256
    return path


@pytest.fixture(scope="session")
def dinov3_candidate_profile_path(
    tmp_path_factory: pytest.TempPathFactory,
    dinov3_live_production_profile_path: Path,
) -> Path:
    """Copy the exact live production-v1 profile outside the repository."""

    repository = Path(__file__).resolve().parents[1]
    live_bytes = dinov3_live_production_profile_path.read_bytes()
    assert len(live_bytes) == _DINOV3_LIVE_PROFILE_BYTES
    assert _sha256(live_bytes) == _DINOV3_LIVE_PROFILE_SHA256
    directory = tmp_path_factory.mktemp("dinov3-candidate-v1")
    candidate_path = (directory / "extractor-profile.json").resolve()
    assert candidate_path.is_absolute()
    assert not candidate_path.is_relative_to(repository.resolve())
    candidate_path.write_bytes(live_bytes)
    assert candidate_path.is_file() and not candidate_path.is_symlink()
    assert candidate_path.read_bytes() == live_bytes
    try:
        yield candidate_path
    finally:
        if directory.exists():
            shutil.rmtree(directory)
        assert not directory.exists()
        assert _sha256(dinov3_live_production_profile_path.read_bytes()) == (
            _DINOV3_LIVE_PROFILE_SHA256
        )


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def model_registry(repository_root: Path) -> dict[str, object]:
    from vision_encoder_alignment.config import load_model_registry

    return load_model_registry(repository_root / "configs" / "models.yaml")


@pytest.fixture
def smoke_config(
    repository_root: Path, model_registry: dict[str, object]
) -> dict[str, object]:
    from vision_encoder_alignment.config import load_experiment_config

    return load_experiment_config(
        repository_root / "configs" / "smoke_cpu.yaml",
        model_registry,
    )


@pytest.fixture
def feature_metadata() -> FeatureArtifactMetadata:
    patch_key, global_key = canonical_tensor_keys("block_03")
    return FeatureArtifactMetadata(
        schema_id=ARTIFACT_SCHEMA_ID,
        run_id="20260712T000000Z_mock",
        model_key="clip_vit_b16",
        tap_namespace="backbone",
        tap_name="block_03",
        patch_tensor_key=patch_key,
        global_mean_tensor_key=global_key,
        image_ids=["image-b", "image-a"],
        grid_height=14,
        grid_width=14,
        patch_count=196,
        hidden_dimension=8,
        dtype="float32",
        shard_index=0,
        safetensors_file_name="shard-00000.safetensors",
        sidecar_file_name="shard-00000.json",
        checksum_algorithm="sha256",
        checksum_value="",
        created_at_utc="2026-07-12T00:00:00Z",
        run_status="incomplete",
    )


@pytest.fixture
def feature_tensors() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(20260712)
    patch = generator.standard_normal((2, 196, 8)).astype(np.float32)
    global_mean = patch.mean(axis=1, dtype=np.float32)
    return patch, global_mean
