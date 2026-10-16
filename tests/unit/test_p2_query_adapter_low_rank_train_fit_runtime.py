"""Offline compact-output and fail-closed checks for T004E."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_linear as core
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_runtime_binding as binding
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_train_fit_runtime as runtime


CONFIG = Path("configs/vlm_compat/p2_query_adapter_low_rank_linear_v1.json")


def synthetic_bound(*, drift_replay: bool = False, zero_spectrum: bool = False) -> binding.BoundLowRankInputs:
    config = core.load_low_rank_config(CONFIG)
    rng = np.random.default_rng(43)
    source_array = rng.normal(size=(12, 2, 4)).astype(np.float32)
    ids = tuple(f"offline-{index:02d}" for index in range(12))
    ols: dict[str, binding.AcceptedOLS] = {}
    factories = {}
    for source in runtime.SOURCE_ORDER:
        W = np.zeros((4, 4), dtype=np.float32) if zero_spectrum else np.diag([2, 1, 0, 0]).astype(np.float32)
        target_array = (source_array @ W + np.array([1, -2, 0.5, 3], np.float32)).astype(np.float32)
        ols[source] = binding.AcceptedOLS(W, source_array.mean((0, 1)), target_array.mean((0, 1)),
                                          config["ols_control"]["exact_artifact_sha256_by_source"][source])
        calls = [0]

        def factory(s=source_array, t=target_array, counter=calls):
            counter[0] += 1
            for start in (0, 6):
                values = s[start:start + 6].copy()
                if drift_replay and counter[0] == 2 and start == 0:
                    values[0, 0, 0] += 1
                yield core.TrainChunk(values, t[start:start + 6].copy(), ids[start:start + 6],
                                      ids[start:start + 6], "train")

        factories[source] = factory
    provenance = {
        "schema_id": binding.SCHEMA_ID, "t001d_plan_id": "a" * 64,
        "manifest_sha256": config["split"]["manifest_sha256"],
        "bank_identity_sha256": {bank: "b" * 64 for bank in binding.BANK_ORDER},
        "validation_result_sha256": config["provenance"]["accepted_linear_test_binding"]["validation_result_sha256"],
        "validation_runtime_identity_sha256": "c" * 64,
        "ols_artifact_sha256": config["ols_control"]["exact_artifact_sha256_by_source"],
        "train_image_count": 800, "test_scientific_access": 0,
        "test_payload_open_count": 0, "test_only_units_hashed_as_opaque_bytes": True,
        "float32_latency_evidence": False, "numeric_note": "float64 application; no latency claim",
    }
    return binding.BoundLowRankInputs(config, ols, factories, provenance)


def run_fixture(bound: binding.BoundLowRankInputs, output: Path):
    return runtime._run_synthetic_bound_for_tests(
        bound, output, shape=core.SyntheticTestShape(2, 4, 12), ranks=(1, 2, 4))


def test_synthetic_fit_publishes_only_compact_factors_and_complete_provenance(tmp_path: Path) -> None:
    output = tmp_path / "new-result"
    result = run_fixture(synthetic_bound(), output)
    assert result["schema_id"] == runtime.SCHEMA_ID
    assert result["status"] == runtime.STATUS and result["synthetic_test_only"] is True
    assert result["config_sha256"] == core.CONFIG_SHA256
    assert result["source_order"] == ["SIGLIP2", "DINOV3"]
    assert result["rank_grid_ordered"] == [1, 2, 4]
    assert result["train_image_count"] == 12 and result["train_row_count"] == 24
    assert result["test_scientific_access"] == result["test_payload_open_count"] == 0
    assert result["validation_patch_access"] == result["generation_call_count"] == 0
    assert result["float32_latency_evidence"] is False
    assert {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()} == {
        "result.json", "factors/siglip2_rank_001.npz", "factors/siglip2_rank_002.npz",
        "factors/dinov3_rank_001.npz", "factors/dinov3_rank_002.npz",
    }
    for source in result["sources"]:
        assert source["train_image_count"] == 12 and source["train_row_count"] == 24
        assert source["positive_eigenvalue_count"] == 2
        assert [record["status"] for record in source["ranks"]] == ["FITTED", "FITTED", "DEGENERATE_RANK"]
        assert source["ranks"][2]["artifact"] is None
        for record in source["ranks"][:2]:
            artifact = record["artifact"]
            path = output / artifact["relative_path"]
            assert runtime._sha_file(path) == artifact["sha256"]
            assert artifact["scalar_count"] == 8 * record["rank"] + 4
            assert artifact["theoretical_raw_parameter_bytes"] == artifact["scalar_count"] * 4
            with np.load(path, allow_pickle=False) as archive:
                assert set(archive.files) == {"U", "V", "bias"}
                assert archive["U"].dtype == archive["V"].dtype == archive["bias"].dtype == np.float32
    assert output.joinpath("result.json").is_file()


def test_zero_spectrum_records_all_ranks_without_artifacts(tmp_path: Path) -> None:
    result = run_fixture(synthetic_bound(zero_spectrum=True), tmp_path / "zero")
    assert all(source["positive_eigenvalue_count"] == 0 for source in result["sources"])
    assert all(source["candidate_disposition"] == "NO_RANK_ACCEPTED" for source in result["sources"])
    assert all(rank["status"] == "DEGENERATE_RANK" and rank["artifact"] is None
               for source in result["sources"] for rank in source["ranks"])
    assert [path.name for path in (tmp_path / "zero").iterdir()] == ["result.json"]


@pytest.mark.parametrize("change", ["result", "bank", "ols", "test_access", "test_payload", "source_order", "config"])
def test_bound_identity_drift_rejected(change: str) -> None:
    bound = synthetic_bound()
    provenance = copy.deepcopy(bound.provenance)
    ols = dict(bound.ols)
    config = copy.deepcopy(bound.config)
    if change == "result":
        provenance["validation_result_sha256"] = "0" * 64
    elif change == "bank":
        provenance["bank_identity_sha256"]["TARGET_CLIP"] = "bad"
    elif change == "ols":
        ols["SIGLIP2"] = replace(ols["SIGLIP2"], artifact_sha256="0" * 64)
    elif change == "test_access":
        provenance["test_scientific_access"] = 1
    elif change == "test_payload":
        provenance["test_payload_open_count"] = 1
    elif change == "source_order":
        ols = dict(reversed(list(ols.items())))
    else:
        config["adapter"]["rank_grid_ordered"] = [8]
    changed = replace(bound, provenance=provenance, ols=ols, config=config)
    with pytest.raises((runtime.LowRankTrainFitError, core.LowRankLinearError)):
        runtime._validate_bound(changed)


def test_changed_replay_fails_without_published_output(tmp_path: Path) -> None:
    output = tmp_path / "replay"
    with pytest.raises(core.LowRankLinearError, match="changed between passes"):
        run_fixture(synthetic_bound(drift_replay=True), output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["shape", "dtype", "nonfinite", "rank", "duplicate"])
def test_bad_compact_factors_or_rank_grid_fail_closed(change: str) -> None:
    bound = synthetic_bound()
    result = core.fit_low_rank_rrr(bound.train_factory("SIGLIP2"), bound.ols["SIGLIP2"].weights,
                                   bound.ols["SIGLIP2"].source_mean, bound.ols["SIGLIP2"].target_mean,
                                   bound.config, synthetic_test_shape=core.SyntheticTestShape(2, 4, 12),
                                   synthetic_test_ranks=(1, 2, 4))
    by_rank = dict(result.by_rank)
    item = by_rank[1]
    assert isinstance(item, core.LowRankFactors)
    if change == "shape":
        by_rank[1] = replace(item, U=item.U[:-1])
    elif change == "dtype":
        by_rank[1] = replace(item, V=item.V.astype(np.float64))
    elif change == "nonfinite":
        by_rank[1] = replace(item, bias=np.full_like(item.bias, np.nan))
    elif change == "rank":
        by_rank[1] = replace(item, rank=3)
    else:
        with pytest.raises(runtime.LowRankTrainFitError, match="duplicate"):
            runtime._run_synthetic_bound_for_tests(bound, Path.cwd() / "never-publish",
                                                    shape=core.SyntheticTestShape(2, 4, 12), ranks=(1, 1))
        return
    with pytest.raises(runtime.LowRankTrainFitError):
        runtime._validate_fit(replace(result, by_rank=by_rank), (1, 2, 4), 12, 2, 4, True, bound.config)


@pytest.mark.parametrize("change", ["extra", "missing", "dtype", "shape", "nan", "duplicate_member"])
def test_saved_npz_tampering_rejected(tmp_path: Path, change: str) -> None:
    output = tmp_path / "stored"
    result = run_fixture(synthetic_bound(), output)
    artifact = result["sources"][0]["ranks"][0]["artifact"]
    path = output / artifact["relative_path"]
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
    else:
        import warnings
        import zipfile
        with zipfile.ZipFile(path) as archive:
            member = archive.read("U.npy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("U.npy", member)
    if change != "duplicate_member":
        np.savez(path, **arrays)
    item = core.LowRankFactors(1, arrays["U"] if change not in ("dtype", "shape") else np.zeros((4, 1), np.float32),
                               np.zeros((4, 1), np.float32), np.zeros(4, np.float32), 12, 2, 4, True)
    with pytest.raises(runtime.LowRankTrainFitError):
        runtime._verify_npz(path, item, runtime._sha_file(path))


def test_output_collision_is_preserved(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(runtime.LowRankTrainFitError, match="fresh"):
        runtime._run_synthetic_bound_for_tests(synthetic_bound(), output,
                                               shape=core.SyntheticTestShape(2, 4, 12), ranks=(1, 2, 4))
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_duplicate_factor_path_fails_without_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_factor_relative_path", lambda source, rank: "factors/duplicate.npz")
    output = tmp_path / "duplicate"
    with pytest.raises(runtime.LowRankTrainFitError, match="duplicate"):
        run_fixture(synthetic_bound(), output)
    assert not output.exists()


def test_forbidden_split_api_is_never_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Test or Validation split requested")

    monkeypatch.setattr(binding.BoundLowRankInputs, "split", forbidden)
    result = run_fixture(synthetic_bound(), tmp_path / "train-only")
    assert result["validation_patch_access"] == result["test_scientific_access"] == 0
    assert result["generation_call_count"] == 0
