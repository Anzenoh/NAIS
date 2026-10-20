"""Offline T004J frozen generation and adversarial boundary tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_query_adapter_low_rank_validation_generation as cli
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_generation as runtime

ROOT = Path(__file__).resolve().parents[2]


def fixture(tmp_path: Path):
    config = runtime.load_config(ROOT)
    inputs = cli.fake_inputs(tmp_path)
    return config, inputs


def run(tmp_path: Path):
    config, inputs = fixture(tmp_path)
    generator = cli.FakeGenerator()
    result = runtime.run_diagnostic(config=config, ids=inputs[0], images=inputs[1],
                                    arrays=inputs[2], ols=inputs[3], factors=inputs[4],
                                    hashes=inputs[5], ranks=inputs[6], provenance=inputs[7],
                                    generator=generator, output_root=tmp_path / "out",
                                    device="cpu", fake=True)
    return config, inputs, generator, result, tmp_path / "out"


def test_frozen_config_and_500_call_plan() -> None:
    config = runtime.load_config(ROOT)
    assert config["generation_plan"] == {"native": 100, "adapted": 400, "total": 500}
    assert config["lane_order"] == list(runtime.LANES)
    assert config["selection_status"] == "NO_RANK_ACCEPTED"
    assert config["t004f_result_sha256"] == runtime.T004F_SHA
    assert config["t004h_result_sha256"] == runtime.T004H_SHA
    assert all(config[key] == 0 for key in runtime.ZERO)


def test_config_byte_drift_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "configs/vlm_compat"
    destination.mkdir(parents=True)
    source = ROOT / runtime.CONFIG_PATH
    (destination / source.name).write_bytes(source.read_bytes() + b" ")
    with pytest.raises(runtime.DiagnosticError, match="config SHA"):
        runtime.load_config(tmp_path)


def test_fake_rows_same_query_and_independent_publication(tmp_path: Path) -> None:
    config, inputs, generator, result, output = run(tmp_path)
    rows = runtime.read_rows(output / "rows.jsonl")
    assert result["generation_counts"] == {"native": 2, "adapted": 8, "total": 10}
    assert generator.native_calls == 2 and generator.adapted_calls == 8
    assert len(rows) == len({(row["image_id"], row["lane_key"]) for row in rows}) == 10
    assert [row["lane_key"] for row in rows[:5]] == list(runtime.LANES)
    assert runtime.validate_rows(rows, inputs[0], inputs[1], inputs[5], inputs[6], config) == result["aggregate"]
    assert result["rows_sha256"] == runtime.sha_file(output / "rows.jsonl")
    assert result["aggregate_sha256"] == runtime.sha_file(output / "aggregate.json")
    assert all(row["image_id"] == inputs[0][index // 5] for index, row in enumerate(rows))
    assert all(row["vision_calls"] == 0 and row["injection_calls"] == 1
               for row in rows if row["lane_key"] != "NATIVE")
    assert all(row["injection_restored"] for row in rows)
    assert result["aggregate"]["within_source"]["SIGLIP2"]["token_equality_count"] == 2
    assert result["test_generation_count"] == 0


@pytest.mark.parametrize("mutation", ["subset", "order", "duplicate", "tokens", "prompt", "counters", "rank", "field"])
def test_row_adversaries_rejected(tmp_path: Path, mutation: str) -> None:
    config, inputs, _, _, output = run(tmp_path)
    rows = runtime.read_rows(output / "rows.jsonl")
    if mutation == "subset":
        rows.pop()
    elif mutation == "order":
        rows[1], rows[2] = rows[2], rows[1]
    elif mutation == "duplicate":
        rows[2] = copy.deepcopy(rows[1])
    elif mutation == "tokens":
        rows[1]["token_ids"] = [999]
    elif mutation == "prompt":
        rows[1]["prompt_decoding_identity"] = "0" * 64
    elif mutation == "counters":
        rows[1]["injection_restored"] = False
    elif mutation == "rank":
        rows[1]["same_image_rank"] = 99
    else:
        rows[1]["unsupported"] = "bad"
    with pytest.raises(runtime.DiagnosticError):
        runtime.validate_rows(rows, inputs[0], inputs[1], inputs[5], inputs[6], config)


def test_wrong_split_ids_hash_and_output_collision(tmp_path: Path) -> None:
    config, inputs = fixture(tmp_path)
    kwargs = dict(config=config, ids=inputs[0], images=inputs[1], arrays=inputs[2],
                  ols=inputs[3], factors=inputs[4], hashes=inputs[5], ranks=inputs[6],
                  provenance=inputs[7], generator=cli.FakeGenerator(),
                  output_root=tmp_path / "out", device="cpu", fake=True)
    with pytest.raises(runtime.DiagnosticError):
        runtime.run_diagnostic(**{**kwargs, "ids": ("test-0", "test-1")})
    with pytest.raises(runtime.DiagnosticError):
        runtime.run_diagnostic(**{**kwargs, "images": list(reversed(inputs[1]))})
    bad = list(inputs[1])
    bad[0] = runtime.ValidationImage(bad[0].image_id, bad[0].path, bad[0].relative_path, "0" * 64)
    with pytest.raises(runtime.DiagnosticError):
        runtime.run_diagnostic(**{**kwargs, "images": bad})
    output = tmp_path / "occupied"
    output.mkdir()
    with pytest.raises(runtime.DiagnosticError):
        runtime.run_diagnostic(**{**kwargs, "output_root": output})


def test_prompt_sha_and_injection_restore_fail_closed(tmp_path: Path) -> None:
    config, inputs = fixture(tmp_path)
    kwargs = dict(config=config, ids=inputs[0], images=inputs[1], arrays=inputs[2],
                  ols=inputs[3], factors=inputs[4], hashes=inputs[5], ranks=inputs[6],
                  provenance=inputs[7], output_root=tmp_path / "out", device="cpu", fake=True)
    changed = copy.deepcopy(config)
    changed["generation"]["prompt"] = "different"
    with pytest.raises(runtime.DiagnosticError, match="config"):
        runtime.run_diagnostic(**{**kwargs, "config": changed, "generator": cli.FakeGenerator()})
    changed = copy.deepcopy(config)
    changed["t004f_result_sha256"] = "0" * 64
    with pytest.raises(runtime.DiagnosticError, match="config"):
        runtime.run_diagnostic(**{**kwargs, "config": changed, "generator": cli.FakeGenerator()})
    changed = copy.deepcopy(config)
    changed["t004h_result_sha256"] = "0" * 64
    with pytest.raises(runtime.DiagnosticError, match="config"):
        runtime.run_diagnostic(**{**kwargs, "config": changed, "generator": cli.FakeGenerator()})

    class BrokenGenerator(cli.FakeGenerator):
        def generate_adapted(self, image_path, patch):
            value = super().generate_adapted(image_path, patch)
            return type(value)(value.token_ids, value.decoded_text, 0, 1, 1, False)

    with pytest.raises(runtime.DiagnosticError, match="counter"):
        runtime.run_diagnostic(**{**kwargs, "generator": BrokenGenerator()})
    assert (tmp_path / "out").is_dir()
    assert not (tmp_path / "out/result.json").exists()


def test_bad_factor_or_generation_boundary_preserves_incomplete_run(tmp_path: Path) -> None:
    config, inputs = fixture(tmp_path)
    malformed = dict(inputs[4])
    factor = malformed["SIGLIP2"]
    malformed["SIGLIP2"] = core.LowRankFactors(256, factor.U[:-1], factor.V, factor.bias,
                                                factor.scalar_count, 196, 768, True)
    kwargs = dict(config=config, ids=inputs[0], images=inputs[1], arrays=inputs[2],
                  ols=inputs[3], factors=malformed, hashes=inputs[5], ranks=inputs[6],
                  provenance=inputs[7], generator=cli.FakeGenerator(),
                  output_root=tmp_path / "out", device="cpu", fake=True)
    with pytest.raises(core.LowRankLinearError):
        runtime.run_diagnostic(**kwargs)
    assert (tmp_path / "out").is_dir()
    assert not (tmp_path / "out" / "result.json").exists()
    with pytest.raises(runtime.DiagnosticError):
        runtime.run_diagnostic(**kwargs)


def test_malformed_missing_repeated_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    for payload in ("{", "", "{}\n{}\n"):
        path.write_text(payload, encoding="utf-8")
        if payload == "{}\n{}\n":
            assert len(runtime.read_rows(path)) == 2
        else:
            with pytest.raises(runtime.DiagnosticError):
                runtime.read_rows(path)
    config, inputs, _, _, output = run(tmp_path)
    rows = runtime.read_rows(output / "rows.jsonl")
    rows[-1] = rows[0]
    with pytest.raises(runtime.DiagnosticError):
        runtime.validate_rows(rows, inputs[0], inputs[1], inputs[5], inputs[6], config)
    path.write_text('{"image_id":"one","image_id":"two"}\n', encoding="utf-8")
    with pytest.raises(runtime.DiagnosticError, match="malformed"):
        runtime.read_rows(path)


def h_fixture(tmp_path: Path):
    ids = tuple(f"validation-{i:03d}" for i in range(100))
    ordered_sha = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    provenance = {"t001d_plan_id": "p", "manifest_sha256": "m",
                  "bank_identity_sha256": {"SIGLIP2": "a", "DINOV3": "b", "TARGET_CLIP": "c"},
                  "validation_result_sha256": "v",
                  "ols_artifact_sha256": {"SIGLIP2": "s", "DINOV3": "d"}}
    factors = {source: {rank: hashlib.sha256(f"{source}-{rank}".encode()).hexdigest()
                        for rank in core.RANK_GRID} for source in ("SIGLIP2", "DINOV3")}
    sources = []
    for source in ("SIGLIP2", "DINOV3"):
        conditions = []
        for name, rank in [("OLS_REFERENCE", None), *((f"LOW_RANK_{rank}", rank) for rank in core.RANK_GRID)]:
            conditions.append({"condition": name, "rank": rank,
                               "factor_sha256": None if rank is None else factors[source][rank],
                               "selection_status": "UNSELECTED",
                               "per_image": [{"image_id": image_id, "same_image_rank": index + 1}
                                             for index, image_id in enumerate(ids)]})
        sources.append({"source": source, "validation_image_count": 100,
                        "ordered_image_id_sha256": ordered_sha, "conditions": conditions})
    value = {"schema_id": "vea.p2.query_adapter_low_rank_validation_metrics.v1",
             "status": "PASS_VALIDATION_PATCH_IDENTITY_ONLY", "split": "validation",
             "config_sha256": core.CONFIG_SHA256, "t004f_result_sha256": runtime.T004F_SHA,
             "t001d_plan_id": "p", "manifest_sha256": "m",
             "bank_identity_sha256": provenance["bank_identity_sha256"],
             "t002d_validation_result_sha256": "v",
             "ols_artifact_sha256": provenance["ols_artifact_sha256"],
             "source_order": ["SIGLIP2", "DINOV3"], "rank_grid_ordered": list(core.RANK_GRID),
             "validation_image_count": 100, "feature_shape": [100, 196, 768],
             "feature_dtype": "float32", "selection_status": "UNSELECTED",
             "test_scientific_access": 0, "test_payload_open_count": 0,
             "test_metric_count": 0, "generation_call_count": 0, "connector_call_count": 0,
             "ordered_image_id_sha256": ordered_sha, "sources": sources}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return ids, SimpleNamespace(provenance=provenance), factors, value, path


def test_t004h_hash_condition_order_and_pairing(tmp_path: Path) -> None:
    ids, bound, factors, value, path = h_fixture(tmp_path)
    sha = runtime.sha_file(path)
    paired = runtime.h_metrics(tmp_path, bound, ids, factors, expected_sha=sha)
    assert paired["SIGLIP2"]["LOW_RANK_256"][ids[0]] == 1
    with pytest.raises(runtime.DiagnosticError, match="SHA"):
        runtime.h_metrics(tmp_path, bound, ids, factors, expected_sha="0" * 64)
    value["t004f_result_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(runtime.DiagnosticError, match="t004f"):
        runtime.h_metrics(tmp_path, bound, ids, factors, expected_sha=runtime.sha_file(path))
    value["t004f_result_sha256"] = runtime.T004F_SHA
    value["sources"][0]["conditions"][1]["factor_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(runtime.DiagnosticError, match="factor"):
        runtime.h_metrics(tmp_path, bound, ids, factors, expected_sha=runtime.sha_file(path))
    value["sources"][0]["conditions"][1]["factor_sha256"] = factors["SIGLIP2"][8]
    value["sources"][0]["conditions"][0]["per_image"][0]["image_id"] = "test-0"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(runtime.DiagnosticError, match="pairing"):
        runtime.h_metrics(tmp_path, bound, ids, factors, expected_sha=runtime.sha_file(path))


def test_image_manifest_rejects_escape_and_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image = dataset / "ok.jpg"
    image.write_bytes(b"jpeg")
    rows = [{"split": "validation", "image_id": "validation-0", "relative_path": "ok.jpg",
             "source_sha256": runtime.sha_file(image)}]
    plan = {"roots": {"dataset_root": str(dataset)}, "manifest_rows": rows, "manifest_sha256": "m"}
    monkeypatch.setattr(runtime.transport, "load_bound_plan", lambda _root: (plan, "p"))
    provenance = {"t001d_plan_id": "p", "manifest_sha256": "m"}
    assert runtime.validation_images(tmp_path, ("validation-0",), provenance)[0].path == image
    rows[0]["relative_path"] = "../escape.jpg"
    with pytest.raises(runtime.DiagnosticError, match="unsafe"):
        runtime.validation_images(tmp_path, ("validation-0",), provenance)
    rows[0]["relative_path"] = "ok.jpg"
    rows[0]["source_sha256"] = "0" * 64
    with pytest.raises(runtime.DiagnosticError, match="SHA"):
        runtime.validation_images(tmp_path, ("validation-0",), provenance)
    rows[0]["split"] = "test"
    with pytest.raises(runtime.DiagnosticError, match="Validation manifest"):
        runtime.validation_images(tmp_path, ("validation-0",), provenance)
