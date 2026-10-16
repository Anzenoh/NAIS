"""Offline identity, stream, and Test-isolation checks for T004C."""

from __future__ import annotations

import hashlib
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_bank_preflight as preflight
from vision_encoder_alignment.vlm_compat import query_adapter_linear as linear
from vision_encoder_alignment.vlm_compat import query_adapter_linear_validation_runtime as validation
from vision_encoder_alignment.vlm_compat import query_adapter_low_rank_runtime_binding as binding
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


ROOT = Path(__file__).resolve().parents[2]
PATCH_SHAPE = (2, 768)


class Fixture:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.run = tmp_path / "run"
        self.validation = tmp_path / "validation"
        self.run.mkdir()
        self.validation.mkdir()
        self.plan_id = "a" * 64
        self.linear = json.loads((ROOT / binding.LINEAR_PATH).read_text(encoding="utf-8"))
        self.test = json.loads((ROOT / binding.TEST_BINDING_PATH).read_text(encoding="utf-8"))
        self.plan = preflight.synthetic_t001d_plan(self.linear, preflight.PRODUCTION_EXPECTATIONS, chunks=4)
        self.plan["expected_units"] = []
        bounds = [(0, 400), (400, 800), (800, 900), (900, 1000)]
        self.unit_paths: dict[tuple[str, int], Path] = {}
        self.hashes: dict[str, list[dict[str, str]]] = {}
        self.ids: dict[str, list[str]] = {}
        for bank in binding.BANK_ORDER:
            self.hashes[bank] = []
            self.ids[bank] = []
            for index, (start, stop) in enumerate(bounds):
                ids = [row["image_id"] for row in self.plan["manifest_rows"][start:stop]]
                count = len(ids)
                stage = preflight.REQUIRED_STAGES[bank]
                relative = f"features/{bank.lower()}/chunk_{index:04d}.npz"
                path = self.run / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shape = {"patches": [count, *PATCH_SHAPE], "globals": [count, PATCH_SHAPE[1]]}
                if bank == "TARGET_CLIP":
                    shape["connectors"] = [count, 196, 1536]
                unit = {
                    "unit_id": f"{stage.lower()}-{index:04d}", "stage": stage,
                    "artifact_relative_path": relative, "encoder": bank, "condition": None,
                    "split": None, "image_ids": ids, "shape": shape,
                    "dtype": {key: "float32" for key in shape}, "chunk_index": index, "image_id": None,
                }
                self.plan["expected_units"].append(unit)
                self.ids[bank].extend(ids)
                self.unit_paths[(bank, index)] = path
                if index == 3:
                    path.write_bytes(b"opaque Test-only artifact bytes")
                else:
                    arrays = {
                        "image_ids": np.asarray(ids, dtype="U96"),
                        "patches": np.full((count, *PATCH_SHAPE), float(index + binding.BANK_ORDER.index(bank)), dtype=np.float32),
                        "globals": np.zeros((count, 768), dtype=np.float32),
                        "plan_id": np.asarray(self.plan_id, dtype="U64"),
                    }
                    if bank == "TARGET_CLIP":
                        arrays["connectors"] = np.zeros((1,), dtype=np.float32)
                    np.savez(path, **arrays)
                self.hashes[bank].append({"unit_id": unit["unit_id"], "artifact_relative_path": relative,
                                          "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.plan["expected_unit_count"] = len(self.plan["expected_units"])
        monkeypatch.setattr(binding.t001d, "load_bound_plan", lambda root: (self.plan, self.plan_id))
        identities = preflight._validate_identities(self.plan, self.linear)
        bank_hashes = {}
        for bank in binding.BANK_ORDER:
            material = {
                "bank": bank, "encoder_identity": identities[bank], "t001d_plan_id": self.plan_id,
                "manifest_sha256": self.plan["manifest_sha256"], "ordered_units": self.hashes[bank],
                "ordered_image_ids": self.ids[bank], "patch_shape": list(PATCH_SHAPE), "patch_dtype": "float32",
                "connector_shape": [196, 1536] if bank == "TARGET_CLIP" else None,
                "connector_dtype": "float32" if bank == "TARGET_CLIP" else None,
            }
            bank_hashes[bank] = hashlib.sha256(t001d.canonical_json_bytes(material)).hexdigest()
        self.ols_paths: dict[str, Path] = {}
        ols_hashes = {}
        for source in binding.SOURCE_ORDER:
            relative = self.test["finalized_adapter_artifacts"][source]["FULL_LINEAR"]["relative_path"]
            path = self.validation / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            fit = linear.FullLinearFit(
                weights=np.eye(768, dtype=np.float32),
                source_mean=np.zeros(768, dtype=np.float32),
                target_mean=np.ones(768, dtype=np.float32),
                rank=768,
                residual_sum_squares=np.zeros(768, dtype=np.float32),
            )
            np.savez(path, runtime_identity=np.asarray(self.test["validation_runtime_identity_sha256"], dtype="U64"),
                     **validation._fit_arrays(fit))
            self.ols_paths[source] = path
            ols_hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.result = {
            "schema_id": "vea.p2.query_adapter_linear_validation_runtime.v1",
            "status": "PASS_QUERY_ADAPTER_LINEAR_VALIDATION",
            "runtime_identity_sha256": self.test["validation_runtime_identity_sha256"],
            "t001d_plan_id": self.plan_id,
            "test_scientific_access": 0, "test_metric_count": 0,
            "generation_call_count": 0, "connector_vision_call_count": 0,
            "fit_artifacts": {
                self.test["finalized_adapter_artifacts"][source]["FULL_LINEAR"]["relative_path"]: ols_hashes[source]
                for source in binding.SOURCE_ORDER
            },
        }
        self.write_result()
        self.fixture = {"patch_shape": PATCH_SHAPE, "expected": {
            "plan": self.plan_id, "manifest": self.plan["manifest_sha256"],
            "banks": bank_hashes, "result": hashlib.sha256((self.validation / "result.json").read_bytes()).hexdigest(),
            "ols": ols_hashes,
        }}

    def write_result(self) -> None:
        (self.validation / "result.json").write_text(json.dumps(self.result), encoding="utf-8")

    def bind(self) -> binding.BoundLowRankInputs:
        return binding._bind_synthetic_fixture_for_tests(ROOT, self.run, self.validation, fixture=self.fixture)

    def rehash_bank(self, bank: str) -> None:
        identities = preflight._validate_identities(self.plan, self.linear)
        for index in range(4):
            path = self.unit_paths[(bank, index)]
            self.hashes[bank][index]["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        material = {
            "bank": bank, "encoder_identity": identities[bank], "t001d_plan_id": self.plan_id,
            "manifest_sha256": self.plan["manifest_sha256"], "ordered_units": self.hashes[bank],
            "ordered_image_ids": self.ids[bank], "patch_shape": list(PATCH_SHAPE), "patch_dtype": "float32",
            "connector_shape": [196, 1536] if bank == "TARGET_CLIP" else None,
            "connector_dtype": "float32" if bank == "TARGET_CLIP" else None,
        }
        self.fixture["expected"]["banks"][bank] = hashlib.sha256(t001d.canonical_json_bytes(material)).hexdigest()

    def materialize_test_only_unit_for_mixed_fixture(self, bank: str) -> None:
        unit = next(unit for unit in self.plan["expected_units"]
                    if unit["encoder"] == bank and unit["chunk_index"] == 3)
        ids = unit["image_ids"]
        arrays = {
            "image_ids": np.asarray(ids, dtype="U96"),
            "patches": np.full((len(ids), *PATCH_SHAPE), 3 + binding.BANK_ORDER.index(bank), dtype=np.float32),
            "globals": np.zeros((len(ids), 768), dtype=np.float32),
            "plan_id": np.asarray(self.plan_id, dtype="U64"),
        }
        if bank == "TARGET_CLIP":
            arrays["connectors"] = np.zeros((1,), dtype=np.float32)
        np.savez(self.unit_paths[(bank, 3)], **arrays)
        self.rehash_bank(bank)

    def rewrite_ols(self, source: str, arrays: dict[str, np.ndarray]) -> None:
        path = self.ols_paths[source]
        np.savez(path, **arrays)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = self.test["finalized_adapter_artifacts"][source]["FULL_LINEAR"]["relative_path"]
        self.fixture["expected"]["ols"][source] = sha
        self.result["fit_artifacts"][relative] = sha
        self.write_result()
        self.fixture["expected"]["result"] = hashlib.sha256((self.validation / "result.json").read_bytes()).hexdigest()


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    return Fixture(tmp_path, monkeypatch)


def blocked(code: str, function) -> None:
    with pytest.raises(binding.LowRankRuntimeBindingError) as raised:
        function()
    assert raised.value.classification == code


def test_accepted_synthetic_lineage_train_factory_and_no_test_payload(fake: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    real_load = binding.np.load
    opened: list[Path] = []

    def guarded_load(path, *args, **kwargs):
        p = Path(path)
        assert p not in {fake.unit_paths[(bank, 3)] for bank in binding.BANK_ORDER}
        assert kwargs.get("allow_pickle") is False
        opened.append(p)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(binding.np, "load", guarded_load)
    bound = fake.bind()
    assert list(bound.ols) == list(binding.SOURCE_ORDER)
    assert all(value.weights.shape == (768, 768) for value in bound.ols.values())
    assert bound.provenance["test_payload_open_count"] == 0
    assert bound.provenance["float32_latency_evidence"] is False
    for source in binding.SOURCE_ORDER:
        first = list(bound.train_factory(source)())
        second = list(bound.train_factory(source)())
        assert len(first) == len(second) == 1
        assert sum(len(chunk.source_image_ids) for chunk in first) == 800
        assert all(chunk.split == "train" and chunk.source.shape[1:] == PATCH_SHAPE for chunk in first)
        assert all(chunk.source.shape[0] * chunk.source.shape[1] <= 4096 for chunk in first)
        assert [chunk.source_image_ids for chunk in first] == [chunk.source_image_ids for chunk in second]
        assert all(np.array_equal(a.source, b.source) and np.array_equal(a.target, b.target)
                   for a, b in zip(first, second, strict=True))
    assert not any(path in opened for path in (fake.unit_paths[(bank, 3)] for bank in binding.BANK_ORDER))
    blocked("BLOCKED_TEST_GUARD", lambda: bound.split("test"))
    blocked("BLOCKED_TEST_GUARD", lambda: bound.train_factory("TARGET_CLIP"))


def test_missing_absolute_roots_and_public_fixture_guard(fake: Fixture, tmp_path: Path) -> None:
    blocked("BLOCKED_ROOT", lambda: binding.bind_low_rank_runtime(ROOT, "relative", fake.validation))
    blocked("BLOCKED_ROOT", lambda: binding.bind_low_rank_runtime(ROOT, tmp_path / "absent", fake.validation))
    blocked("BLOCKED_PLAN", lambda: binding.bind_low_rank_runtime(ROOT, fake.run, fake.validation))


def test_tracked_sha_and_result_drift(fake: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(binding.TRACKED_SHA, binding.BANK_PATH, "0" * 64)
    blocked("BLOCKED_TRACKED_SHA", fake.bind)
    monkeypatch.setitem(binding.TRACKED_SHA, binding.BANK_PATH,
                        "d2742c93f8eeae286d490fc229ff57fa8cd6701aab81dadde656ff23433129be")
    fake.result["test_metric_count"] = 1
    fake.write_result()
    blocked("BLOCKED_RESULT_HASH", fake.bind)
    fake.fixture["expected"]["result"] = hashlib.sha256((fake.validation / "result.json").read_bytes()).hexdigest()
    blocked("BLOCKED_RESULT", fake.bind)


def test_ols_hash_payload_and_identity(fake: Fixture) -> None:
    fake.ols_paths["SIGLIP2"].write_bytes(b"corrupted")
    blocked("BLOCKED_OLS_HASH", fake.bind)


def test_historical_six_field_ols_schema_binds_both_sources_as_copies(fake: Fixture) -> None:
    bound = fake.bind()
    expected = {"runtime_identity", "weights", "source_mean", "target_mean", "rank", "residual_sum_squares"}
    assert set(bound.ols) == set(binding.SOURCE_ORDER)
    assert set(binding.AcceptedOLS.__dataclass_fields__) == {
        "weights", "source_mean", "target_mean", "artifact_sha256"
    }
    for source in binding.SOURCE_ORDER:
        with np.load(fake.ols_paths[source], allow_pickle=False) as archive:
            assert set(archive.files) == expected
            assert set(archive.zip.namelist()) == {f"{field}.npy" for field in expected}
            assert archive["rank"].shape == () and int(archive["rank"].item()) == 768
            assert archive["residual_sum_squares"].shape == (768,)
            stored_weights = archive["weights"]
            assert np.array_equal(bound.ols[source].weights, stored_weights)
            assert np.array_equal(bound.ols[source].source_mean, archive["source_mean"])
            assert np.array_equal(bound.ols[source].target_mean, archive["target_mean"])
            stored_weights[0, 0] = -999
            assert bound.ols[source].weights[0, 0] == 1


@pytest.mark.parametrize("rank", [0, 767])
def test_rank_deficient_empty_residuals_allowed(fake: Fixture, rank: int) -> None:
    source = "DINOV3"
    with np.load(fake.ols_paths[source], allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["rank"] = np.asarray(rank, dtype=np.int64)
    arrays["residual_sum_squares"] = np.empty((0,), dtype=np.float32)
    fake.rewrite_ols(source, arrays)
    assert np.array_equal(fake.bind().ols[source].weights, arrays["weights"])


@pytest.mark.parametrize("change", ["missing_rank", "missing_residuals", "extra", "unsafe_extra"])
def test_ols_member_allowlist_fail_closed(fake: Fixture, change: str) -> None:
    source = "SIGLIP2"
    with np.load(fake.ols_paths[source], allow_pickle=False) as archive:
        arrays = dict(archive)
    if change == "missing_rank":
        del arrays["rank"]
    elif change == "missing_residuals":
        del arrays["residual_sum_squares"]
    else:
        arrays["unexpected" if change == "extra" else "../unexpected"] = np.zeros(1, dtype=np.float32)
    fake.rewrite_ols(source, arrays)
    blocked("BLOCKED_SHAPE", fake.bind)


def test_ols_duplicate_zip_member_rejected(fake: Fixture) -> None:
    source = "SIGLIP2"
    path = fake.ols_paths[source]
    with zipfile.ZipFile(path) as archive:
        duplicate = archive.read("rank.npy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("rank.npy", duplicate)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    relative = fake.test["finalized_adapter_artifacts"][source]["FULL_LINEAR"]["relative_path"]
    fake.fixture["expected"]["ols"][source] = sha
    fake.result["fit_artifacts"][relative] = sha
    fake.write_result()
    fake.fixture["expected"]["result"] = hashlib.sha256((fake.validation / "result.json").read_bytes()).hexdigest()
    blocked("BLOCKED_SHAPE", fake.bind)


@pytest.mark.parametrize("rank", [
    np.asarray([768], dtype=np.int64),
    np.asarray(768.0, dtype=np.float64),
    np.asarray(True, dtype=np.bool_),
    np.asarray(-1, dtype=np.int64),
    np.asarray(769, dtype=np.int64),
])
def test_ols_invalid_rank_rejected(fake: Fixture, rank: np.ndarray) -> None:
    source = "SIGLIP2"
    with np.load(fake.ols_paths[source], allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["rank"] = rank
    fake.rewrite_ols(source, arrays)
    blocked("BLOCKED_SHAPE", fake.bind)


@pytest.mark.parametrize("residuals,rank", [
    (np.zeros((1,), dtype=np.float32), 768),
    (np.zeros((768, 1), dtype=np.float32), 768),
    (np.zeros((768,), dtype=np.float64), 768),
    (np.full((768,), np.nan, dtype=np.float32), 768),
    (np.full((768,), np.inf, dtype=np.float32), 768),
    (np.zeros((768,), dtype=np.float32), 767),
])
def test_ols_invalid_residuals_rejected(fake: Fixture, residuals: np.ndarray, rank: int) -> None:
    source = "DINOV3"
    with np.load(fake.ols_paths[source], allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["rank"] = np.asarray(rank, dtype=np.int64)
    arrays["residual_sum_squares"] = residuals
    fake.rewrite_ols(source, arrays)
    blocked("BLOCKED_SHAPE", fake.bind)


def test_ols_fit_manifest_hash_drift_rejected(fake: Fixture) -> None:
    relative = fake.test["finalized_adapter_artifacts"]["DINOV3"]["FULL_LINEAR"]["relative_path"]
    fake.result["fit_artifacts"][relative] = "0" * 64
    fake.write_result()
    fake.fixture["expected"]["result"] = hashlib.sha256((fake.validation / "result.json").read_bytes()).hexdigest()
    blocked("BLOCKED_OLS_HASH", fake.bind)


@pytest.mark.parametrize("field,value", [
    ("weights", np.zeros((767, 768), dtype=np.float32)),
    ("source_mean", np.zeros(768, dtype=np.float64)),
    ("target_mean", np.full(768, np.nan, dtype=np.float32)),
])
def test_ols_array_rejections(fake: Fixture, field: str, value: np.ndarray) -> None:
    source = "SIGLIP2"
    path = fake.ols_paths[source]
    arrays = dict(np.load(path, allow_pickle=False))
    arrays[field] = value
    np.savez(path, **arrays)
    relative = fake.test["finalized_adapter_artifacts"][source]["FULL_LINEAR"]["relative_path"]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    fake.fixture["expected"]["ols"][source] = sha
    fake.result["fit_artifacts"][relative] = sha
    fake.write_result()
    fake.fixture["expected"]["result"] = hashlib.sha256((fake.validation / "result.json").read_bytes()).hexdigest()
    blocked("BLOCKED_SHAPE", fake.bind)


def test_ols_persisted_runtime_identity(fake: Fixture) -> None:
    source = "SIGLIP2"
    path = fake.ols_paths[source]
    arrays = dict(np.load(path, allow_pickle=False))
    arrays["runtime_identity"] = np.asarray("0" * 64, dtype="U64")
    np.savez(path, **arrays)
    relative = fake.test["finalized_adapter_artifacts"][source]["FULL_LINEAR"]["relative_path"]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    fake.fixture["expected"]["ols"][source] = sha
    fake.result["fit_artifacts"][relative] = sha
    fake.write_result()
    fake.fixture["expected"]["result"] = hashlib.sha256((fake.validation / "result.json").read_bytes()).hexdigest()
    blocked("BLOCKED_OLS_IDENTITY", fake.bind)


def test_unsafe_relative_and_symlink_path(fake: Fixture, tmp_path: Path) -> None:
    blocked("BLOCKED_UNSAFE_PATH", lambda: binding._artifact(fake.validation, "../outside.npz", "OLS"))
    blocked("BLOCKED_UNSAFE_PATH", lambda: binding._artifact(fake.validation, "/outside.npz", "OLS"))
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"x")
    link = fake.validation / "link.npz"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("local symlink creation unavailable")
    blocked("BLOCKED_UNSAFE_PATH", lambda: binding._artifact(fake.validation, "link.npz", "OLS"))


def test_plan_duplicate_misorder_split_and_mixed_test_fail(fake: Fixture) -> None:
    fake.plan["expected_units"][0]["image_ids"][0] = fake.plan["expected_units"][0]["image_ids"][1]
    blocked("BLOCKED_IMAGE_ORDER", fake.bind)
    fake.plan["expected_units"][0]["image_ids"][0] = "fake-0000"
    fake.plan["expected_units"][0]["image_ids"].reverse()
    blocked("BLOCKED_IMAGE_ORDER", fake.bind)
    fake.plan["expected_units"][0]["image_ids"].reverse()
    fake.plan["manifest_rows"][0]["split"] = "test"
    blocked("BLOCKED_SPLIT", fake.bind)


def test_mixed_train_test_unit_selects_train_without_numpy_open(fake: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep split counts 800/100/100 while moving one Test image into a Train unit.
    rows = fake.plan["manifest_rows"]
    rows[0]["split"], rows[900]["split"] = rows[900]["split"], rows[0]["split"]
    for bank in binding.BANK_ORDER:
        fake.materialize_test_only_unit_for_mixed_fixture(bank)
    real_load = binding.np.load

    def guard(path, *args, **kwargs):
        assert Path(path) not in set(fake.unit_paths.values())
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(binding.np, "load", guard)
    bound = fake.bind()
    expected = tuple(row["image_id"] for row in rows if row["split"] == "train")
    for source in binding.SOURCE_ORDER:
        first = list(bound.train_factory(source)())
        second = list(bound.train_factory(source)())
        ids = tuple(image_id for chunk in first for image_id in chunk.source_image_ids)
        assert ids == expected
        assert first[0].source_image_ids == second[0].source_image_ids
        assert all(chunk.source.shape[0] * chunk.source.shape[1] <= 4096 for chunk in first)


@pytest.mark.parametrize("kind", ["reverse", "duplicate"])
def test_cross_source_payload_order_and_duplicates_rejected(fake: Fixture, kind: str) -> None:
    path = fake.unit_paths[("DINOV3", 0)]
    arrays = dict(np.load(path, allow_pickle=False))
    ids = arrays["image_ids"].copy()
    if kind == "reverse":
        ids[[0, 1]] = ids[[1, 0]]
    else:
        ids[0] = ids[1]
    arrays["image_ids"] = ids
    np.savez(path, **arrays)
    fake.rehash_bank("DINOV3")
    bound = fake.bind()
    blocked("BLOCKED_IMAGE_ORDER", lambda: list(bound.train_factory("DINOV3")()))


def test_train_payload_dtype_shape_nonfinite_and_replay_drift(fake: Fixture) -> None:
    bound = fake.bind()
    assert len(list(bound.train_factory("SIGLIP2")())) == 1
    path = fake.unit_paths[("SIGLIP2", 0)]
    arrays = dict(np.load(path, allow_pickle=False))
    arrays["patches"][0, 0, 0] = np.nan
    np.savez(path, **arrays)
    blocked("BLOCKED_ARTIFACT_HASH", lambda: list(bound.train_factory("SIGLIP2")()))
    fake.rehash_bank("SIGLIP2")
    bound = fake.bind()
    blocked("BLOCKED_SHAPE", lambda: list(bound.train_factory("SIGLIP2")()))


@pytest.mark.parametrize("kind", ["shape", "dtype"])
def test_train_payload_bad_shape_or_dtype(fake: Fixture, kind: str) -> None:
    path = fake.unit_paths[("DINOV3", 0)]
    arrays = dict(np.load(path, allow_pickle=False))
    arrays["patches"] = (arrays["patches"][:, :, :-1] if kind == "shape"
                         else arrays["patches"].astype(np.float64))
    np.savez(path, **arrays)
    fake.rehash_bank("DINOV3")
    bound = fake.bind()
    blocked("BLOCKED_SHAPE", lambda: list(bound.train_factory("DINOV3")()))


def test_second_pass_values_drift_rejected(fake: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    bound = fake.bind()
    factory = bound.train_factory("SIGLIP2")
    list(factory())
    original = factory._rows

    def altered(units):
        for index, (image_id, value) in enumerate(original(units)):
            if units is factory._source_units and index == 0:
                value = value.copy()
                value[0, 0] += 1
            yield image_id, value

    monkeypatch.setattr(factory, "_rows", altered)
    blocked("BLOCKED_STREAM_REPLAY", lambda: list(factory()))


def test_chunk_row_ceiling(fake: Fixture) -> None:
    bound = fake.bind()
    assert all(chunk.source.shape[0] * chunk.source.shape[1] <= 4096
               for chunk in bound.train_factory("DINOV3")())
    fake.fixture["patch_shape"] = (20, 768)
    blocked("BLOCKED_CHUNK_CAP", fake.bind)
