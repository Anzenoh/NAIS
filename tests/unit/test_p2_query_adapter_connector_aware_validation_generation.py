"""CPU-only T003F generation binding and Validation-boundary checks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts import run_query_adapter_connector_aware_validation_generation as cli
from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_validation_generation as runtime


ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path):
    config = runtime.load_config(ROOT)
    inputs, lanes, paths = cli._fake_inputs(config)
    generator = cli.FakeGenerator()
    result = runtime.run_validation_generation(config=config, inputs=inputs, lanes=lanes, generator=generator,
                                               image_paths=paths, output_root=tmp_path / "out", device="cpu", fake=True)
    return config, inputs, lanes, generator, result, tmp_path / "out"


def test_exact_frozen_configuration_and_production_plan() -> None:
    config = runtime.load_config(ROOT)
    assert config["t003e"]["result_sha256"] == runtime.T003E_RESULT_SHA256
    assert config["t003e"]["runtime_identity_sha256"] == runtime.T003E_RUNTIME_IDENTITY
    assert config["generation_plan"] == {"native": 100, "adapted": 400, "total": 500}
    assert config["generation"]["prompt"] == "What is shown in this image? Answer briefly."
    assert config["lanes"]["SIGLIP2/CONNECTOR_AWARE_SELECTED"]["artifact_condition"] == "PATCH_CONNECTOR_4P0"
    assert config["lanes"]["DINOV3/OLS_REFERENCE"]["affine_sha256"] == config["lanes"]["DINOV3/CONNECTOR_AWARE_SELECTED"]["affine_sha256"]


def test_fake_rows_are_unique_validation_only_and_aggregate_recomputes(tmp_path: Path) -> None:
    config, _, _, generator, result, output = _run(tmp_path)
    assert result["generation_counts"] == {"native": 2, "adapted": 8, "total": 10}
    assert result["production_generation_plan"] == config["generation_plan"]
    assert generator.native_calls == 2 and generator.adapted_calls == 8
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len({(row["image_id"], row["lane_key"] or "NATIVE") for row in rows}) == 10
    assert {row["lane"] for row in rows} == {"NATIVE", "OLS_REFERENCE", "CONNECTOR_AWARE_SELECTED"}
    assert all(row["t003e_result_sha256"] == runtime.T003E_RESULT_SHA256 for row in rows)
    assert all(row["generation_success"] is True and row["injection_restored"] is True for row in rows)
    adapted = [row for row in rows if row["lane"] != "NATIVE"]
    assert all(row["vision_calls"] == 0 and row["injection_calls"] == 1 for row in adapted)
    native = [row for row in rows if row["lane"] == "NATIVE"]
    assert all(row["vision_calls"] >= 1 and row["injection_calls"] == 0 for row in native)
    assert result["aggregate"]["within_source"]["DINOV3"]["duplicate_affine_identity"] is True
    assert result["aggregate"]["within_source"]["DINOV3"]["generated_text_equality_count"] == 2
    assert result["test_access_count"] == result["test_metric_count"] == 0
    assert result["generation_is_validation_only"] is True


def test_same_image_patch_injection_and_output_publication_order(tmp_path: Path) -> None:
    config = runtime.load_config(ROOT)
    inputs, lanes, paths = cli._fake_inputs(config)
    captured = []
    class RecordingGenerator(cli.FakeGenerator):
        def generate_adapted(self, image_path, patch):
            captured.append((Path(image_path).stem, patch.copy()))
            return super().generate_adapted(image_path, patch)
    output = tmp_path / "ordered"
    runtime.run_validation_generation(config=config, inputs=inputs, lanes=lanes, generator=RecordingGenerator(),
                                      image_paths=paths, output_root=output, device="cpu", fake=True)
    assert [path.name for path in (output / "rows.jsonl", output / "aggregate.json", output / "result.json")] == [
        "rows.jsonl", "aggregate.json", "result.json"
    ]
    assert len(captured) == 8
    assert all(patch.shape == (1, 196, 768) for _, patch in captured)
    assert np.array_equal(captured[0][1], inputs.validation["SIGLIP2"].source[0:1].numpy())


@pytest.mark.parametrize("kind", ["test", "unknown", "order", "nonfinite"])
def test_test_or_invalid_validation_input_fails_closed(kind: str, tmp_path: Path) -> None:
    config = runtime.load_config(ROOT)
    inputs, lanes, paths = cli._fake_inputs(config)
    batch = inputs.validation["SIGLIP2"]
    values = dict(inputs.validation)
    if kind == "test":
        values["SIGLIP2"] = replace(batch, split="test")
    elif kind == "unknown":
        values["SIGLIP2"] = replace(batch, split="unknown")
    elif kind == "order":
        values["SIGLIP2"] = replace(batch, target_image_ids=tuple(reversed(batch.target_image_ids)))
    else:
        source = batch.source.clone(); source[0, 0, 0] = float("nan")
        values["SIGLIP2"] = replace(batch, source=source)
    with pytest.raises(runtime.ValidationGenerationError):
        runtime.run_validation_generation(config=config, inputs=replace(inputs, validation=values), lanes=lanes,
                                          generator=cli.FakeGenerator(), image_paths=paths,
                                          output_root=tmp_path / kind, device="cpu", fake=True)


def test_malformed_t003e_result_and_output_collision_fail_closed(tmp_path: Path, monkeypatch) -> None:
    config = runtime.load_config(ROOT)
    root = tmp_path / "t003e"; root.mkdir()
    (root / "result.json").write_text("{}", encoding="utf-8")
    original_sha_file = runtime._sha_file
    monkeypatch.setattr(
        runtime,
        "_sha_file",
        lambda path: config["t003e"]["result_sha256"] if Path(path).name == "result.json" else original_sha_file(path),
    )
    with pytest.raises(runtime.ValidationGenerationError, match="runtime identity"):
        runtime.verify_t003e_selected_artifacts(config, root)
    monkeypatch.setattr(runtime, "_sha_file", original_sha_file)
    _, inputs, lanes, _, _, output = _run(tmp_path)
    with pytest.raises(runtime.ValidationGenerationError, match="output root"):
        runtime.run_validation_generation(config=config, inputs=inputs, lanes=lanes, generator=cli.FakeGenerator(),
                                          image_paths=["synthetic-0.bin", "synthetic-1.bin"], output_root=output,
                                          device="cpu", fake=True)
