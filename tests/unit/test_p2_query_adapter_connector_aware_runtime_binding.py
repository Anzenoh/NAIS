"""Offline identity and payload rejection tests for T003C."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.vlm_compat import query_adapter_connector_aware_runtime_binding as binding
from vision_encoder_alignment.vlm_compat import query_adapter_linear_validation_runtime as validation
from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as t001d


ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_plan(root: Path, manifest: str = "synthetic-manifest") -> str:
    plan = {"schema_id": t001d.PLAN_SCHEMA, "manifest_sha256": manifest,
            "manifest_split_counts": {"train": 2, "validation": 1, "test": 0}}
    payload = t001d.canonical_json_bytes(plan)
    plan_id = hashlib.sha256(payload).hexdigest()
    (root / "run_plan.json").write_bytes(payload)
    state = root / "state"; state.mkdir(exist_ok=True)
    (state / "plan_binding.json").write_text(json.dumps({"schema_id": t001d.STATE_SCHEMA, "plan_id": plan_id}), encoding="utf-8")
    return plan_id


def _write_fit(path: Path, identity: str, *, field: str | None = None, value=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "runtime_identity": np.asarray(identity, dtype="U64"),
        "weights": np.eye(768, dtype=np.float32),
        "source_mean": np.zeros(768, dtype=np.float32),
        "target_mean": np.ones(768, dtype=np.float32),
    }
    if field is not None:
        arrays[field] = value
    np.savez_compressed(path, **arrays)


class Fixture:
    def __init__(self, tmp_path: Path) -> None:
        self.run = tmp_path / "t001d"; self.run.mkdir()
        self.val = tmp_path / "t002d"; self.val.mkdir()
        self.plan = _write_plan(self.run)
        self.accepted = binding.load_binding(ROOT)
        self.identity = self.accepted["t002d"]["validation_runtime_identity_sha256"]
        self.paths = {source: self.val / self.accepted["t002d"]["full_linear"][source]["relative_path"]
                      for source in binding.core.SOURCES}
        for path in self.paths.values():
            _write_fit(path, self.identity)
        self.result = {
            "runtime_identity_sha256": self.identity,
            "t001d_plan_id": self.accepted["t001d"]["t001d_plan_id"],
            "test_scientific_access": 0, "test_metric_count": 0, "generation_call_count": 0,
            "fit_artifacts": {},
        }
        self.ids = {"train": ["train-0", "train-1"], "validation": ["val-0"]}
        self.banks = {}
        for name in (*binding.core.SOURCES, "TARGET_CLIP"):
            self.banks[name] = {
                "train": np.zeros((2, 196, 768), dtype=np.float32),
                "validation": np.zeros((1, 196, 768), dtype=np.float32),
                "ids": {key: list(ids) for key, ids in self.ids.items()},
            }
        self.banks["TARGET_CLIP"]["connectors"] = {
            "train": np.zeros((2, 196, 1536), dtype=np.float32),
            "validation": np.zeros((1, 196, 1536), dtype=np.float32),
        }

    def refresh(self) -> dict:
        hashes = {source: _sha(path) for source, path in self.paths.items() if path.exists()}
        self.result["fit_artifacts"] = {
            self.accepted["t002d"]["full_linear"][source]["relative_path"]: digest
            for source, digest in hashes.items()
        }
        result_path = self.val / "result.json"
        result_path.write_text(json.dumps(self.result), encoding="utf-8")
        return {"plan": self.plan, "manifest": "synthetic-manifest", "result": _sha(result_path),
                "full_linear": hashes}

    def report(self, _config: Path, _root: Path) -> dict:
        bank = self.accepted["t001d"]
        return {
            "status": binding.bank_preflight.SUCCESS_STATUS,
            "adapter_experiment_id": validation.linear.EXPERIMENT_ID,
            "adapter_config_sha256": validation.load_bank_binding(ROOT / binding.BANK_PATH)["adapter_config_sha256"],
            "t001d_plan_id": bank["t001d_plan_id"], "manifest_sha256": bank["manifest_sha256"],
            "cross_bank_image_order_equal": True, "manifest_order_equal": True,
            "all_required_units_valid": True,
            "banks": {name: {"bank_identity_sha256": digest} for name, digest in bank["banks"].items()},
        }

    def bind(self, *, expectations: dict | None = None):
        return binding._bind_synthetic_fixture_for_tests(
            ROOT, self.run, self.val, expected_counts=(2, 1),
            synthetic_preflight=self.report, synthetic_bank_loader=lambda _: self.banks,
            expected_hashes=self.refresh() if expectations is None else expectations,
        )


def test_exact_schema_and_copied_provenance() -> None:
    value = binding.load_binding(ROOT)
    assert value["schema_id"] == binding.SCHEMA_ID
    assert value["implementation_predecessor"] == binding.BASELINE
    assert value["policy"] == binding.DENIED_POLICY
    assert value["t001d"]["banks"] == validation.load_bank_binding(ROOT / binding.BANK_PATH)["banks"]


@pytest.mark.parametrize("source", list(binding.SOURCE_HASHES))
def test_any_accepted_source_sha_drift_rejected(monkeypatch, source: Path) -> None:
    monkeypatch.setitem(binding.SOURCE_HASHES, source, "0" * 64)
    with pytest.raises(binding.RuntimeBindingError, match="source SHA-256 drift"):
        binding.load_binding(ROOT)


def test_absolute_path_rejected_in_committed_binding(tmp_path: Path, monkeypatch) -> None:
    value = binding.load_binding(ROOT)
    value["runtime_roots"]["t001d"] = "C:\\private\\bank"
    for relative in (*binding.SOURCE_HASHES, binding.BINDING_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    (tmp_path / binding.BINDING_PATH).write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(binding.RuntimeBindingError, match="absolute path"):
        binding.load_binding(tmp_path)


def test_synthetic_success_is_train_validation_only(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    bound = fixture.bind()
    assert set(bound.train) == set(binding.core.SOURCES)
    assert set(bound.validation) == set(binding.core.SOURCES)
    assert bound.train["SIGLIP2"].source.shape == (2, 196, 768)
    assert bound.full_linear_initialization["DINOV3"]["bias"].shape == (768,)
    assert bound.provenance["policy"] == binding.DENIED_POLICY
    with pytest.raises(binding.RuntimeBindingError, match="Test"):
        bound.split("test")
    assert not hasattr(binding, "generate")


def test_wrong_plan_and_manifest_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    expectations = fixture.refresh()
    expectations["plan"] = "0" * 64
    with pytest.raises(binding.RuntimeBindingError, match="plan or manifest"):
        fixture.bind(expectations=expectations)
    expectations["plan"] = fixture.plan
    expectations["manifest"] = "wrong"
    with pytest.raises(binding.RuntimeBindingError, match="plan or manifest"):
        fixture.bind(expectations=expectations)


@pytest.mark.parametrize("source,split", [("SIGLIP2", "train"), ("DINOV3", "validation")])
def test_source_target_order_mismatch_rejected(tmp_path: Path, source: str, split: str) -> None:
    fixture = Fixture(tmp_path)
    fixture.banks[source]["ids"][split] = list(reversed(fixture.ids[split])) if split == "train" else ["wrong"]
    with pytest.raises(binding.RuntimeBindingError, match="image ID order"):
        fixture.bind()


def test_validation_result_sha_and_runtime_identity_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    expectations = fixture.refresh()
    expectations["result"] = "0" * 64
    with pytest.raises(binding.RuntimeBindingError, match="result SHA"):
        fixture.bind(expectations=expectations)
    fixture.result["runtime_identity_sha256"] = "0" * 64
    with pytest.raises(binding.RuntimeBindingError, match="runtime identity"):
        fixture.bind()


@pytest.mark.parametrize("field", ["test_scientific_access", "test_metric_count", "generation_call_count"])
def test_nonzero_historical_activity_rejected(tmp_path: Path, field: str) -> None:
    fixture = Fixture(tmp_path)
    fixture.result[field] = 1
    with pytest.raises(binding.RuntimeBindingError, match="Test or generation"):
        fixture.bind()


@pytest.mark.parametrize("source", list(binding.core.SOURCES))
def test_missing_or_tampered_full_linear_rejected(tmp_path: Path, source: str) -> None:
    fixture = Fixture(tmp_path)
    expectations = fixture.refresh()
    fixture.paths[source].write_bytes(b"tampered")
    with pytest.raises(binding.RuntimeBindingError, match="artifact identity"):
        fixture.bind(expectations=expectations)
    fixture.paths[source].unlink()
    with pytest.raises(binding.RuntimeBindingError, match="artifact identity"):
        fixture.bind(expectations=expectations)


@pytest.mark.parametrize("field,value", [
    ("weights", np.zeros((767, 768), dtype=np.float32)),
    ("source_mean", np.zeros(768, dtype=np.float64)),
    ("target_mean", np.full(768, np.nan, dtype=np.float32)),
])
def test_invalid_full_linear_array_rejected(tmp_path: Path, field: str, value: np.ndarray) -> None:
    fixture = Fixture(tmp_path)
    _write_fit(fixture.paths["SIGLIP2"], fixture.identity, field=field, value=value)
    with pytest.raises(binding.RuntimeBindingError, match="shape, dtype, or finiteness"):
        fixture.bind()


def test_safe_np_load_and_both_hashes_checked(tmp_path: Path, monkeypatch) -> None:
    fixture = Fixture(tmp_path)
    actual = binding.np.load
    calls = []
    def safe_load(*args, **kwargs):
        calls.append((Path(args[0]).name, kwargs.get("allow_pickle")))
        return actual(*args, **kwargs)
    monkeypatch.setattr(binding.np, "load", safe_load)
    fixture.bind()
    assert sorted(calls) == sorted([(path.name, False) for path in fixture.paths.values()])
    expectations = fixture.refresh()
    expectations["full_linear"]["DINOV3"] = "0" * 64
    with pytest.raises(binding.RuntimeBindingError, match="artifact identity"):
        fixture.bind(expectations=expectations)
