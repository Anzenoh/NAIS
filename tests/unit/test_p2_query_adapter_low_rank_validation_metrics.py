"""Synthetic Validation-only metric and artifact controls for T004G."""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_runtime_binding as binding
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_train_fit_runtime as train_fit
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_validation_metrics_runtime as runtime


CONFIG = Path("configs/vlm_compat/p2_query_adapter_low_rank_linear_v1.json")


def _fixture(tmp_path: Path):
    config = core.load_low_rank_config(CONFIG)
    rng = np.random.default_rng(43)
    train_x = rng.normal(size=(12, 2, 4)).astype(np.float32)
    valid_x = rng.normal(size=(6, 2, 4)).astype(np.float32)
    W = np.diag([2, 1, 0, 0]).astype(np.float32)
    offset = np.array([1, -2, 0.5, 3], dtype=np.float32)
    train_y = (train_x @ W + offset).astype(np.float32)
    valid_y = (valid_x @ W + offset).astype(np.float32)
    train_ids = tuple(f"train-{index}" for index in range(12))
    valid_ids = tuple(f"valid-{index}" for index in range(6))
    ols = {}
    train_factories = {}
    valid_factories = {}
    for source in binding.SOURCE_ORDER:
        ols[source] = binding.AcceptedOLS(W, train_x.mean((0, 1)), train_y.mean((0, 1)),
                                          config["ols_control"]["exact_artifact_sha256_by_source"][source])

        def train_factory(x=train_x, y=train_y):
            for start in (0, 6):
                yield core.TrainChunk(x[start:start + 6].copy(), y[start:start + 6].copy(),
                                      train_ids[start:start + 6], train_ids[start:start + 6], "train")

        def validation_factory(x=valid_x, y=valid_y):
            for start in (0, 3):
                yield binding.ValidationChunk(x[start:start + 3].copy(), y[start:start + 3].copy(),
                                              valid_ids[start:start + 3])

        train_factories[source] = train_factory
        valid_factories[source] = validation_factory
    provenance = {
        "schema_id": binding.SCHEMA_ID, "t001d_plan_id": "a" * 64,
        "manifest_sha256": config["split"]["manifest_sha256"],
        "bank_identity_sha256": {bank: "b" * 64 for bank in binding.BANK_ORDER},
        "validation_result_sha256": config["provenance"]["accepted_linear_test_binding"]["validation_result_sha256"],
        "validation_runtime_identity_sha256": "c" * 64,
        "ols_artifact_sha256": config["ols_control"]["exact_artifact_sha256_by_source"],
        "train_image_count": 800, "test_scientific_access": 0, "test_payload_open_count": 0,
        "test_only_units_hashed_as_opaque_bytes": True, "float32_latency_evidence": False,
    }
    train_bound = binding.BoundLowRankInputs(config, ols, train_factories, provenance)
    fit_root = tmp_path / "fit"
    train_fit._run_synthetic_bound_for_tests(train_bound, fit_root,
                                             shape=core.SyntheticTestShape(2, 4, 12), ranks=(1, 2))
    validation_bound = binding.BoundLowRankValidationInputs(
        config, ols, valid_factories,
        {**provenance, "schema_id": "vea.p2.query_adapter_low_rank_validation_binding.v1",
         "validation_image_count": 100})
    return validation_bound, fit_root, valid_x, valid_y, valid_ids


def _run(bound, fit_root, output):
    return runtime._run_synthetic_fixture_for_tests(bound, fit_root, output,
                                                    images=6, patches=2, dim=4, ranks=(1, 2))


def _manifest_check(bound, fit_root):
    return runtime._fit_manifest(fit_root, bound,
                                 result_sha=binding._sha_file(fit_root / "result.json"),
                                 dim=4, patches=2, ranks=(1, 2), synthetic=True)


def test_synthetic_ols_and_low_rank_metrics_match_independent_control(tmp_path: Path) -> None:
    bound, fit_root, x, native, ids = _fixture(tmp_path)
    output = tmp_path / "metrics"
    result = _run(bound, fit_root, output)
    assert result["status"] == runtime.STATUS and result["split"] == "validation"
    assert result["validation_image_count"] == 6 and result["source_order"] == list(binding.SOURCE_ORDER)
    assert result["t004f_result_sha256"] == binding._sha_file(fit_root / "result.json")
    assert result["selection_status"] == "UNSELECTED"
    assert result["connector_metrics"] == result["generation_metrics"] == "NOT_EVALUATED"
    assert result["float32_adapter_latency"] == "NOT_EVALUATED"
    assert result["test_scientific_access"] == result["test_metric_count"] == 0
    assert result["generation_call_count"] == result["connector_call_count"] == 0
    assert set(output.iterdir()) == {output / "result.json"}
    for source in result["sources"]:
        assert [condition["condition"] for condition in source["conditions"]] == [
            "OLS_REFERENCE", "LOW_RANK_1", "LOW_RANK_2"]
        assert all(condition["selection_status"] == "UNSELECTED" for condition in source["conditions"])
        assert all([row["image_id"] for row in condition["per_image"]] == list(ids)
                   for condition in source["conditions"])
        ols = source["conditions"][0]
        np.testing.assert_allclose([row["patch_cosine"] for row in ols["per_image"]], 1.0, atol=1e-7)
        np.testing.assert_allclose([row["patch_nmse"] for row in ols["per_image"]], 0.0, atol=1e-12)
        assert ols["patch_metrics"]["recall_at_1"] == 1.0
        assert source["conditions"][2]["factor_sha256"] == binding._sha_file(
            fit_root / f"factors/{source['source'].lower()}_rank_002.npz")
    # Independent image-weighted cosine/NMSE calculation for the rank-one map.
    factor_path = fit_root / "factors/siglip2_rank_001.npz"
    with np.load(factor_path, allow_pickle=False) as archive:
        predicted = ((x.astype(np.float64) @ archive["U"].astype(np.float64)) @
                     archive["V"].astype(np.float64).T + archive["bias"].astype(np.float64)).astype(np.float32)
    flat_a, flat_b = predicted.reshape(6, -1).astype(np.float64), native.reshape(6, -1).astype(np.float64)
    independent_cosine = np.sum(flat_a * flat_b, axis=1) / (
        np.linalg.norm(flat_a, axis=1) * np.linalg.norm(flat_b, axis=1))
    np.testing.assert_allclose(result["sources"][0]["conditions"][1]["patch_metrics"]["cosine"],
                               independent_cosine.mean(), atol=1e-12)


@pytest.mark.parametrize("change", ["report_sha", "source_order", "rank", "missing", "status", "counter", "bank"])
def test_stale_or_malformed_fit_manifest_rejected(tmp_path: Path, change: str) -> None:
    bound, fit_root, *_ = _fixture(tmp_path)
    report = fit_root / "result.json"
    original_sha = binding._sha_file(report)
    data = json.loads(report.read_text(encoding="utf-8"))
    if change == "report_sha":
        report.write_text(report.read_text(encoding="utf-8") + " ", encoding="utf-8")
        expected = original_sha
    else:
        if change == "source_order":
            data["source_order"] = ["DINOV3", "SIGLIP2"]
        elif change == "rank":
            data["sources"][0]["ranks"][1]["rank"] = 1
        elif change == "missing":
            data["sources"][0]["ranks"].pop()
        elif change == "status":
            data["sources"][0]["ranks"][0]["status"] = "DEGENERATE_RANK"
        elif change == "counter":
            data["test_scientific_access"] = 1
        else:
            data["bank_identity_sha256"]["TARGET_CLIP"] = "0" * 64
        report.write_text(json.dumps(data), encoding="utf-8")
        expected = binding._sha_file(report)
    with pytest.raises(runtime.LowRankValidationError):
        runtime._fit_manifest(fit_root, bound, result_sha=expected,
                              dim=4, patches=2, ranks=(1, 2), synthetic=True)


@pytest.mark.parametrize("change", ["physical_sha", "extra", "missing", "dtype", "shape", "nan", "duplicate", "path"])
def test_bad_factor_bytes_or_metadata_rejected(tmp_path: Path, change: str) -> None:
    bound, fit_root, *_ = _fixture(tmp_path)
    report = fit_root / "result.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    artifact = data["sources"][0]["ranks"][0]["artifact"]
    path = fit_root / artifact["relative_path"]
    if change == "path":
        data["sources"][0]["ranks"][1]["artifact"]["relative_path"] = artifact["relative_path"]
        report.write_text(json.dumps(data), encoding="utf-8")
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if change == "extra":
            arrays["W"] = np.zeros((4, 4), np.float32)
        elif change == "missing":
            del arrays["bias"]
        elif change == "dtype":
            arrays["U"] = arrays["U"].astype(np.float64)
        elif change == "shape":
            arrays["V"] = arrays["V"][:-1]
        elif change == "nan":
            arrays["bias"] = np.full_like(arrays["bias"], np.nan)
        elif change == "duplicate":
            with zipfile.ZipFile(path) as archive:
                member = archive.read("U.npy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "a") as archive:
                    archive.writestr("U.npy", member)
        if change not in ("physical_sha", "duplicate"):
            np.savez(path, **arrays)
        if change == "physical_sha":
            path.write_bytes(path.read_bytes() + b"tamper")
        else:
            artifact["sha256"] = binding._sha_file(path)
            report.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((runtime.LowRankValidationError, binding.LowRankRuntimeBindingError)):
        _manifest_check(bound, fit_root)


def test_wrong_or_duplicate_validation_ids_and_zero_cosine_fail(tmp_path: Path) -> None:
    bound, fit_root, x, y, ids = _fixture(tmp_path)
    def wrong_factory():
        yield binding.ValidationChunk(x, y, ids[:-1] + (ids[-2],))
    changed = replace(bound, validation_factories={**bound.validation_factories, "SIGLIP2": wrong_factory})
    with pytest.raises(runtime.LowRankValidationError, match="image IDs"):
        _run(changed, fit_root, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(linear.QueryAdapterLinearError, match="cosine denominator"):
        runtime._condition(np.zeros_like(y), y, ids, bound.ols["SIGLIP2"].target_mean,
                           label="zero", rank=None, factor_sha=None)


def test_cross_source_validation_order_mismatch_blocks_publish(tmp_path: Path) -> None:
    bound, fit_root, x, y, ids = _fixture(tmp_path)
    def reversed_factory():
        yield binding.ValidationChunk(x[::-1].copy(), y[::-1].copy(), tuple(reversed(ids)))
    changed = replace(bound, validation_factories={**bound.validation_factories,
                                                    "DINOV3": reversed_factory})
    output = tmp_path / "misordered"
    with pytest.raises(runtime.LowRankValidationError, match="IDs or native target"):
        _run(changed, fit_root, output)
    assert not output.exists()


def test_existing_output_is_untouched(tmp_path: Path) -> None:
    bound, fit_root, *_ = _fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("keep", encoding="utf-8")
    with pytest.raises(runtime.LowRankValidationError, match="synthetic roots"):
        _run(bound, fit_root, output)
    assert (output / "keep").read_text(encoding="utf-8") == "keep"


def _mixed_unit(tmp_path: Path, *, changed: str | None = None):
    splits = ("test", "train", "validation", "test", "validation", "train", "validation", "test")
    ids = tuple(f"mix-{index}" for index in range(len(splits)))
    path = tmp_path / "mixed.npz"
    patches = np.stack([np.full((2, 4), index + 1, np.float32) for index in range(len(splits))])
    if changed == "dtype":
        patches = patches.astype(np.float64)
    if changed == "shape":
        patches = patches[:, :, :-1]
    arrays = {"image_ids": np.asarray(ids, dtype="U96"), "patches": patches,
              "globals": np.full((8, 4), 999, np.float32),
              "connectors": np.full((1,), 999, np.float32),
              "plan_id": np.asarray("a" * 64, dtype="U64")}
    if changed == "extra":
        arrays["rogue"] = np.zeros(1, np.float32)
    (np.savez_compressed if changed == "compressed" else np.savez)(path, **arrays)
    unit = binding._Unit("TARGET_CLIP", ids, splits, path, hashlib.sha256(path.read_bytes()).hexdigest(), "a" * 64)
    return unit


def test_mixed_npz_decodes_only_validation_patch_ranges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unit = _mixed_unit(tmp_path)
    with unit.path.open("rb") as stream:
        names = frozenset(f"{name}.npy" for name in ("image_ids", "patches", "globals", "connectors", "plan_id"))
        members = binding._stored_members(stream, names)
        data_start = binding._patch_data_offset(stream, members["patches.npy"], (8, 2, 4))
        forbidden = [(data_start + index * 32, data_start + (index + 1) * 32)
                     for index, split in enumerate(unit.splits) if split != "validation"]
        forbidden += [(members[name].data_offset, members[name].data_offset + members[name].size)
                      for name in ("globals.npy", "connectors.npy")]
    observed = []
    original = binding._read_exact_at

    def watched(stream, offset, size):
        observed.append((offset, offset + size))
        return original(stream, offset, size)

    monkeypatch.setattr(binding, "_read_exact_at", watched)
    rows = binding._select_validation_rows(unit, (2, 4))
    assert [image_id for image_id, _ in rows] == ["mix-2", "mix-4", "mix-6"]
    assert [float(value[0, 0]) for _, value in rows] == [3.0, 5.0, 7.0]
    assert all(end <= low or start >= high for start, end in observed for low, high in forbidden)
    with pytest.raises(binding.LowRankRuntimeBindingError) as error:
        binding._select_partition_rows(unit, (2, 4), "test")
    assert error.value.classification == "BLOCKED_TEST_GUARD"


@pytest.mark.parametrize("change", ["extra", "compressed", "dtype", "shape"])
def test_validation_npz_unknown_layout_or_shape_rejected(tmp_path: Path, change: str) -> None:
    unit = _mixed_unit(tmp_path, changed=change)
    with pytest.raises(binding.LowRankRuntimeBindingError):
        binding._select_validation_rows(unit, (2, 4))


def test_validation_unit_hash_drift_blocks_before_payload_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unit = _mixed_unit(tmp_path)
    ids = tuple(image_id for image_id, split in zip(unit.image_ids, unit.splits, strict=True)
                if split == "validation")
    factory = binding.ValidationChunkFactory("SIGLIP2", (unit,), (unit,), ids, (2, 4))
    unit.path.write_bytes(unit.path.read_bytes() + b"changed")
    monkeypatch.setattr(binding, "_select_validation_rows",
                        lambda *args: pytest.fail("payload reader called after SHA drift"))
    with pytest.raises(binding.LowRankRuntimeBindingError) as error:
        list(factory())
    assert error.value.classification == "BLOCKED_ARTIFACT_HASH"


def test_unchanged_train_api_still_refuses_test() -> None:
    assert list(binding.bind_low_rank_runtime.__code__.co_varnames[:3]) == [
        "repository_root", "t001d_run_root", "t002d_validation_root"]
    with pytest.raises(binding.LowRankRuntimeBindingError) as error:
        binding.BoundLowRankInputs({}, {}, {}, {}).split("test")
    assert error.value.classification == "BLOCKED_TEST_GUARD"
    with pytest.raises(binding.LowRankRuntimeBindingError) as error:
        binding.BoundLowRankValidationInputs({}, {}, {}, {}).split("test")
    assert error.value.classification == "BLOCKED_TEST_GUARD"
