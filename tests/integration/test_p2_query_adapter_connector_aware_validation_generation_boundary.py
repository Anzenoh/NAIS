"""Offline boundary tests for the T003F Validation generation CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_query_adapter_connector_aware_validation_generation as cli
from vision_encoder_alignment.vlm_compat import (
    query_adapter_connector_aware_validation_generation as runtime,
)


ROOT = Path(__file__).resolve().parents[2]


def test_fake_mode_is_cpu_only_and_never_loads_real_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        calls.append("called")
        raise AssertionError("real runtime access is forbidden in fake mode")

    monkeypatch.setattr(cli.binding_module, "preflight_runtime", forbidden)
    monkeypatch.setattr(cli.generation_boundary, "load_frozen_tinyllava_generator", forbidden)

    assert cli.main(["--fake-smoke", str((tmp_path / "out").resolve())]) == 0
    assert calls == []


def test_real_mode_rejects_bare_cuda_before_runtime_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        calls.append("called")
        raise AssertionError("bare cuda must fail before runtime access")

    monkeypatch.setattr(runtime, "verify_t003e_selected_artifacts", forbidden)
    monkeypatch.setattr(cli.binding_module, "preflight_runtime", forbidden)

    assert cli.main(
        [
            "--t001d-run-root",
            str((tmp_path / "t001d").resolve()),
            "--t002d-validation-root",
            str((tmp_path / "t002d").resolve()),
            "--t003e-output-root",
            str((tmp_path / "t003e").resolve()),
            "--cache-root",
            str((tmp_path / "cache").resolve()),
            "--output-root",
            str((tmp_path / "out").resolve()),
            "--device",
            "cuda",
        ]
    ) == 2
    assert calls == []


def test_real_mode_forwards_exact_cuda_to_generator_loader_without_loading_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    for name in ("t001d", "t002d", "t003e", "cache"):
        (tmp_path / name).mkdir()

    class Inputs:
        provenance = {"bank_preflight_status": "PASS"}

    def fake_preflight(*_args: object, **_kwargs: object) -> Inputs:
        return Inputs()

    def stop_loader(*_args: object, **kwargs: object) -> object:
        observed["device"] = kwargs["device"]
        raise RuntimeError("stop before model loading")

    monkeypatch.setattr(runtime, "verify_t003e_selected_artifacts", lambda *_args: {})
    monkeypatch.setattr(cli.binding_module, "preflight_runtime", fake_preflight)
    monkeypatch.setattr(runtime, "load_validation_image_paths", lambda *_args: [tmp_path / "one.png"])
    monkeypatch.setattr(cli.generation_boundary, "load_frozen_tinyllava_generator", stop_loader)

    assert cli.main(
        [
            "--t001d-run-root",
            str((tmp_path / "t001d").resolve()),
            "--t002d-validation-root",
            str((tmp_path / "t002d").resolve()),
            "--t003e-output-root",
            str((tmp_path / "t003e").resolve()),
            "--cache-root",
            str((tmp_path / "cache").resolve()),
            "--output-root",
            str((tmp_path / "out").resolve()),
            "--device",
            "cuda:0",
        ]
    ) == 2
    assert observed == {"device": "cuda:0"}
