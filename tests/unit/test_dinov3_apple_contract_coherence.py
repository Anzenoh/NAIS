from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vision_encoder_alignment.dinov3_extraction_boundary import (
    DINOv3AppleInvocationPlan,
    DINOv3ExtractionBoundaryError,
    DINOV3_SELECTOR,
    LEGACY_EXECUTION_SELECTOR,
    MODEL_ID,
    MODEL_KEY,
    REVISION,
    invoke_dinov3_extraction_worker,
    load_dinov3_extractor_profile,
    validate_dinov3_extraction_request,
)
from vision_encoder_alignment.dinov3_production_identity import validate_production_sources
from vision_encoder_alignment.extractors.base import CanonicalImageBatch, ExtractorValidationError
import vision_encoder_alignment.extractors.dinov3 as extractor_module
from vision_encoder_alignment.production_runtime_contract import (
    ProductionRuntimeContractError,
    load_production_runtime_profile,
    validate_profile_source_bindings,
)
from vision_encoder_alignment.snapshot_boundary import load_snapshot_profile


ROOT = Path(__file__).resolve().parents[2]
APPLE_EXTRACTOR = ROOT / "configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json"
APPLE_RUNTIME = ROOT / "configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json"
LEGACY_EXTRACTOR = ROOT / "configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json"
SNAPSHOT = ROOT / "configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json"


def _canonical_write(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _apple_request(profile_sha256: str) -> object:
    return validate_dinov3_extraction_request(
        {
            "expected_worker_policy": dict(extractor_module._APPLE_EXPECTED_WORKER_POLICY),
            "extractor_profile_sha256": profile_sha256,
            "image_ids": ["image-1"],
            "input_dtype": "uint8",
            "input_file_sha256": "1" * 64,
            "input_shape": [1, 224, 224, 3],
            "model_id": MODEL_ID,
            "model_key": MODEL_KEY,
            "operation": "extract_canonical_features",
            "raw_input_sha256": "2" * 64,
            "requested_tap_names": ["final_norm"],
            "revision": REVISION,
            "run_id": "apple-unit",
            "schema_id": "vea.dinov3_extraction_request.v0",
            "workspace_relative_input_path": "input.npy",
            "workspace_relative_output_directory": "output",
        },
        execution_selector=DINOV3_SELECTOR,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_id="vea.dinov3_isolated_extractor_profile.v1"),
        lambda value: value["runtime_profile"].update(selector=LEGACY_EXECUTION_SELECTOR),
        lambda value: value["policies"].update(device="cpu"),
        lambda value: value["policies"].update(dtype="float16"),
        lambda value: value["policies"].update(network_allowed=True),
    ],
)
def test_apple_profile_is_selector_specific_and_fail_closed(
    tmp_path: Path, mutate: object
) -> None:
    value = json.loads(APPLE_EXTRACTOR.read_text(encoding="utf-8"))
    mutate(value)  # type: ignore[operator]
    candidate = tmp_path / "apple.json"
    _canonical_write(candidate, value)
    with pytest.raises(DINOv3ExtractionBoundaryError):
        load_dinov3_extractor_profile(candidate, execution_selector=DINOV3_SELECTOR)


def test_profile_selector_and_request_policy_are_explicit() -> None:
    profile = load_dinov3_extractor_profile(APPLE_EXTRACTOR, execution_selector=DINOV3_SELECTOR)
    with pytest.raises(DINOv3ExtractionBoundaryError):
        load_dinov3_extractor_profile(APPLE_EXTRACTOR)
    request = _apple_request(profile.sha256)
    altered = request.to_dict()  # type: ignore[union-attr]
    altered["expected_worker_policy"]["cpu_fallback"] = True
    with pytest.raises(DINOv3ExtractionBoundaryError):
        validate_dinov3_extraction_request(altered, execution_selector=DINOV3_SELECTOR)


def test_apple_context_never_enters_legacy_runtime_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = load_dinov3_extractor_profile(APPLE_EXTRACTOR, execution_selector=DINOV3_SELECTOR)
    runtime_profile = load_production_runtime_profile(APPLE_RUNTIME)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("legacy runtime helper was reachable")

    monkeypatch.setattr(extractor_module, "load_runtime_profile", forbidden)
    monkeypatch.setattr(extractor_module, "resolve_default_runtime_root", forbidden)
    monkeypatch.setattr(extractor_module, "validate_runtime_root", forbidden)
    monkeypatch.setattr(
        extractor_module,
        "_resolve_root",
        lambda value, field, repository: (
            Path(runtime_profile.runtime_root) if field.endswith("runtime_root") else snapshot
        ),
    )
    monkeypatch.setattr(extractor_module, "runtime_python_path", lambda *args, **kwargs: Path("/bin/python"))
    monkeypatch.setattr(
        extractor_module,
        "validate_apple_hf_snapshot",
        lambda *args, **kwargs: SimpleNamespace(
            source_sha256=profile.identities.external_snapshot_manifest_sha256
        ),
    )
    context = extractor_module._load_extractor_context(
        SimpleNamespace(
            execution_selector=DINOV3_SELECTOR,
            profile_path=APPLE_EXTRACTOR,
            apple_runtime_profile_path=APPLE_RUNTIME,
            runtime_root=Path(runtime_profile.runtime_root),
            snapshot_root=snapshot,
        )
    )
    assert context.profile.sha256 == profile.sha256


def test_legacy_context_still_enters_legacy_runtime_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = load_dinov3_extractor_profile(LEGACY_EXTRACTOR)
    runtime = tmp_path / "runtime"
    snapshot = tmp_path / "snapshot"
    runtime.mkdir()
    snapshot.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(extractor_module, "validate_production_sources", lambda *args: ())
    monkeypatch.setattr(
        extractor_module,
        "load_runtime_profile",
        lambda path: SimpleNamespace(source_sha256=profile.identities.runtime_profile_sha256),
    )
    monkeypatch.setattr(
        extractor_module,
        "load_snapshot_profile",
        lambda path: SimpleNamespace(source_sha256=profile.identities.snapshot_profile_sha256),
    )
    monkeypatch.setattr(
        extractor_module,
        "validate_runtime_root",
        lambda *args, **kwargs: calls.append("runtime")
        or SimpleNamespace(
            source_sha256=profile.identities.persistent_runtime_manifest_sha256,
            candidate_freeze_sha256=profile.identities.persistent_freeze_sha256,
        ),
    )
    monkeypatch.setattr(
        extractor_module,
        "validate_snapshot_root",
        lambda *args, **kwargs: SimpleNamespace(
            source_sha256=profile.identities.external_snapshot_manifest_sha256
        ),
    )
    identity_by_name = {
        "models.yaml": profile.identities.registry_sha256,
        "dinov3_isolated_manual_v0.json": profile.identities.runtime_profile_sha256,
        "dinov3_vit_b16_r5931719_v0.json": profile.identities.snapshot_profile_sha256,
        "dinov3_vit_b16_r5931719_semantics_v0.json": profile.identities.semantic_profile_sha256,
        "VEA-P0-T007A-4-dinov3-pretrained-runtime-semantic-probe.json": profile.identities.semantic_evidence_sha256,
    }
    monkeypatch.setattr(
        extractor_module, "sha256_file", lambda path: identity_by_name[Path(path).name]
    )
    extractor_module._load_extractor_context(
        SimpleNamespace(
            execution_selector=LEGACY_EXECUTION_SELECTOR,
            profile_path=LEGACY_EXTRACTOR,
            apple_runtime_profile_path=None,
            runtime_root=runtime,
            snapshot_root=snapshot,
        )
    )
    assert calls == ["runtime"]


def test_extract_forwards_exact_apple_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = load_dinov3_extractor_profile(APPLE_EXTRACTOR, execution_selector=DINOV3_SELECTOR)
    extractor = object.__new__(extractor_module.DINOv3VisionEncoderExtractor)
    extractor._config = SimpleNamespace(
        profile_path=APPLE_EXTRACTOR,
        execution_selector=DINOV3_SELECTOR,
        timeout_seconds=1.0,
    )
    extractor._context = extractor_module._ExtractorContext(
        profile=profile,
        runtime_root=ROOT,
        snapshot_root=ROOT,
        repository_root=ROOT,
        worker_script=ROOT / "scripts/dinov3_extraction_worker.py",
        descriptor=extractor_module._build_descriptor(),
    )
    extractor._last_worker_response = None
    captured: dict[str, object] = {}

    def stop_after_capture(**kwargs: object) -> object:
        captured.update(kwargs)
        raise DINOv3ExtractionBoundaryError("test_stop", "stop")

    monkeypatch.setattr(extractor_module, "invoke_dinov3_extraction_worker", stop_after_capture)
    batch = CanonicalImageBatch(("image-1",), np.zeros((1, 224, 224, 3), dtype=np.uint8))
    with pytest.raises(ExtractorValidationError, match="dinov3_worker.test_stop"):
        extractor._extract(batch, tap_names=("final_norm",))
    assert captured["execution_selector"] == DINOV3_SELECTOR


def test_apple_invoke_uses_shell_free_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    snapshot = tmp_path / "snapshot"
    workspace = tmp_path / "workspace"
    for directory in (runtime, snapshot, workspace):
        directory.mkdir()
    (workspace / "input.npy").write_bytes(b"input")
    profile = load_dinov3_extractor_profile(APPLE_EXTRACTOR, execution_selector=DINOV3_SELECTOR)
    runtime_profile = replace(
        load_production_runtime_profile(APPLE_RUNTIME), runtime_root=runtime.as_posix()
    )
    request = _apple_request(profile.sha256)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "vision_encoder_alignment.dinov3_extraction_boundary.load_production_runtime_profile",
        lambda path: runtime_profile,
    )
    monkeypatch.setattr(
        "vision_encoder_alignment.dinov3_extraction_boundary.validate_profile_source_bindings",
        lambda *args: (),
    )
    monkeypatch.setattr(
        "vision_encoder_alignment.dinov3_extraction_boundary.validate_apple_hf_snapshot",
        lambda *args, **kwargs: SimpleNamespace(
            source_sha256=profile.identities.external_snapshot_manifest_sha256
        ),
    )

    def plan(**kwargs: object) -> DINOv3AppleInvocationPlan:
        captured["plan"] = kwargs
        return DINOv3AppleInvocationPlan(
            command=(str(runtime / "bin/python"), "-B", "worker.py"),
            environment={"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1"},
            interpreter=runtime / "bin/python",
            selector=DINOV3_SELECTOR,
            device="cuda:0",
            dtype="float32",
        )

    def run(command: object, **kwargs: object) -> object:
        captured["run"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "vision_encoder_alignment.dinov3_extraction_boundary.build_dinov3_apple_invocation_plan",
        plan,
    )
    monkeypatch.setattr("vision_encoder_alignment.dinov3_extraction_boundary.subprocess.run", run)
    sentinel = object()
    monkeypatch.setattr(
        "vision_encoder_alignment.dinov3_extraction_boundary.read_dinov3_extraction_response",
        lambda *args, **kwargs: sentinel,
    )
    assert invoke_dinov3_extraction_worker(
        runtime_root=runtime,
        snapshot_root=snapshot,
        repository_root=ROOT,
        extractor_profile_path=APPLE_EXTRACTOR,
        worker_script=ROOT / "scripts/dinov3_extraction_worker.py",
        workspace_root=workspace,
        request=request,  # type: ignore[arg-type]
        timeout_seconds=1,
        execution_selector=DINOV3_SELECTOR,
    ) is sentinel
    assert captured["plan"]["runtime_profile"] is runtime_profile  # type: ignore[index]
    assert captured["run"]["shell"] is False  # type: ignore[index]
    assert captured["run"]["env"]["CUDA_VISIBLE_DEVICES"] == "0"  # type: ignore[index]


def test_known_contract_error_keeps_stable_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = extractor_module.DINOv3ExtractorConfig(LEGACY_EXTRACTOR)
    monkeypatch.setattr(
        extractor_module,
        "_load_extractor_context",
        lambda value: (_ for _ in ()).throw(
            ProductionRuntimeContractError("detail", "SOURCE_BINDING_MISMATCH")
        ),
    )
    with pytest.raises(
        ExtractorValidationError,
        match=r"dinov3_contract\.SOURCE_BINDING_MISMATCH: contract validation failed",
    ):
        extractor_module.DINOv3VisionEncoderExtractor(config)


def _source_rows(profile: object) -> tuple[tuple[str, str, str], ...]:
    return tuple((row.path, row.role, row.sha256) for row in profile.production_sources)  # type: ignore[union-attr]


def test_final_dual_profile_and_runtime_bindings_match() -> None:
    apple = load_dinov3_extractor_profile(APPLE_EXTRACTOR, execution_selector=DINOV3_SELECTOR)
    legacy = load_dinov3_extractor_profile(LEGACY_EXTRACTOR)
    runtime = load_production_runtime_profile(APPLE_RUNTIME)
    assert _source_rows(apple) == _source_rows(legacy)
    assert _source_rows(apple) == tuple(
        (row.path, row.role, row.sha256) for row in validate_production_sources(ROOT, APPLE_EXTRACTOR)
    )
    assert _source_rows(legacy) == tuple(
        (row.path, row.role, row.sha256) for row in validate_production_sources(ROOT, LEGACY_EXTRACTOR)
    )
    assert validate_profile_source_bindings(runtime, ROOT) == runtime.source_bindings
    assert apple.identities.runtime_profile_sha256 == runtime.source_sha256
    assert apple.document.runtime_profile.sha256 == runtime.source_sha256
    assert runtime.ready_for_p1_extraction is False
