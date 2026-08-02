from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vision_encoder_alignment.p1_feature_extraction import (
    EXIT_ARTIFACT,
    EXIT_EXECUTION,
    FeatureExtractionError,
    GLOBAL_TAPS,
    MODEL_ORDER,
    PATCH_TAPS,
    atomic_write_json,
    complete_run,
    initial_run_state,
    initialize_run_artifacts,
    load_extraction_profile,
    offline_environment,
    plan_shards,
    stream_manifest,
    validate_run,
    write_execution_metrics,
    write_shard,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/extraction_profiles/p1_1000_apple_fp32_v0.json"
EXTRACT = ROOT / "scripts/extract_features.py"
VALIDATE = ROOT / "scripts/validate_p1_feature_extraction.py"
RUNBOOK = ROOT / "reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md"
RUN_ID = "20260730T120000Z_p1_1000_three_encoder_e518a51"


@pytest.fixture(autouse=True)
def _supported_no_clobber_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe_source = tmp_path / ".link-source"
    probe_target = tmp_path / ".link-target"
    probe_source.write_bytes(b"probe")
    try:
        os.link(probe_source, probe_target)
        probe_target.unlink()
        return
    except OSError:
        pass

    def exclusive_copy(source, destination):
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with open(source, "rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
                outgoing.write(incoming.read())
                outgoing.flush()
                os.fsync(outgoing.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    monkeypatch.setattr(os, "link", exclusive_copy)


def _load_cli():
    spec = importlib.util.spec_from_file_location("p1_extract_cli", EXTRACT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensors(count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for index, tap in enumerate(GLOBAL_TAPS):
        result[f"backbone.{tap}.global_mean"] = np.full((count, 768), index, np.float32)
    for index, tap in enumerate(PATCH_TAPS):
        patch = np.full((count, 196, 768), index + 1, np.float32)
        result[f"backbone.{tap}.patch"] = patch
        result[f"backbone.{tap}.global_mean"] = patch.mean(axis=1, dtype=np.float32)
    return result


def _fake_worker(profile, command: list[str], *, corrupt: bool = False) -> subprocess.CompletedProcess[str]:
    def option(name: str) -> str:
        return command[command.index(name) + 1]

    run_root = Path(option("--run-root"))
    model_key = option("--model")
    start = int(option("--start-shard"))
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    model = next(item for item in profile.models if item["registry_key"] == model_key)
    for plan in plan_shards(state["image_ids"])[start:]:
        tensors = _tensors(len(plan.image_ids))
        if corrupt:
            tensors.pop(next(iter(tensors)))
        write_shard(
            run_root / "models" / model_key,
            run_id=run_root.name,
            model=model,
            lane_id=model["lane_id"],
            shard=plan,
            tensors=tensors,
            created_at_utc="2026-07-30T03:00:00Z",
        )
    write_execution_metrics(
        run_root / "models" / model_key / "execution_metrics.json",
        run_id=run_root.name, model=model, image_count=len(state["image_ids"]),
        elapsed_seconds=0.5, peak_allocated_bytes=1000,
        peak_reserved_bytes=2000,
    )
    return subprocess.CompletedProcess(command, 0, "", "")


def _arguments(tmp_path: Path, command: str = "run") -> list[str]:
    for name in ("dataset", "cache", "outputs"):
        (tmp_path / name).mkdir(exist_ok=True)
    return [
        command,
        "--profile", str(PROFILE),
        "--repository-root", str(ROOT),
        "--dataset-root", str(tmp_path / "dataset"),
        "--cache-root", str(tmp_path / "cache"),
        "--output-parent", str(tmp_path / "outputs"),
        "--run-id", RUN_ID,
        "--mode", "smoke",
        "--batch-size", "1",
    ]


def test_plan_cli_is_deterministic_and_model_free() -> None:
    command = [sys.executable, "-B", str(EXTRACT), "plan", "--profile", str(PROFILE), "--repository-root", str(ROOT)]
    first = subprocess.run(command, capture_output=True, check=False)
    second = subprocess.run(command, capture_output=True, check=False)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["ready_for_p1_extraction"] is False


def test_minimal_environment_child_import_succeeds() -> None:
    command = [sys.executable, "-B", "-c", "import vision_encoder_alignment.p1_feature_extraction"]
    result = subprocess.run(command, env=offline_environment(ROOT), capture_output=True, text=True, check=False, shell=False)
    assert result.returncode == 0, result.stderr


def test_preflight_failure_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "_preflight", lambda *args: (_ for _ in ()).throw(cli.FeatureExtractionError("not ready", 3)))
    dataset = tmp_path / "dataset"; dataset.mkdir()
    cache = tmp_path / "cache"; cache.mkdir()
    rc = cli.main(["preflight", "--profile", str(PROFILE), "--repository-root", str(ROOT), "--dataset-root", str(dataset), "--cache-root", str(cache)])
    assert rc == 3


def test_preflight_resource_error_publishes_matching_output_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        cli, "_preflight",
        lambda *args: (_ for _ in ()).throw(
            cli.FeatureExtractionError("checkpoint snapshot unavailable: clip_vit_b16", 4),
        ),
    )
    dataset = tmp_path / "dataset"; dataset.mkdir()
    cache = tmp_path / "cache"; cache.mkdir()
    output = tmp_path / "preflight-error.json"
    rc = cli.main([
        "preflight", "--profile", str(PROFILE), "--repository-root", str(ROOT),
        "--dataset-root", str(dataset), "--cache-root", str(cache),
        "--output-json", str(output),
    ])
    stdout = capsysbinary.readouterr().out
    assert rc == 4
    assert output.read_bytes() == stdout
    assert len(stdout) == 153
    assert json.loads(stdout) == {
        "error": "checkpoint snapshot unavailable: clip_vit_b16",
        "exit_code": 4,
        "ready_for_p1_extraction": False,
        "schema_id": "vea.p1.feature_extraction_error.v0",
    }


def test_preflight_error_output_collision_preserves_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        cli, "_preflight",
        lambda *args: (_ for _ in ()).throw(cli.FeatureExtractionError("not ready", 4)),
    )
    dataset = tmp_path / "dataset"; dataset.mkdir()
    cache = tmp_path / "cache"; cache.mkdir()
    output = tmp_path / "competing.json"; output.write_bytes(b"existing-bytes")
    rc = cli.main([
        "preflight", "--profile", str(PROFILE), "--repository-root", str(ROOT),
        "--dataset-root", str(dataset), "--cache-root", str(cache),
        "--output-json", str(output),
    ])
    assert rc == 4
    assert output.read_bytes() == b"existing-bytes"
    assert json.loads(capsysbinary.readouterr().out)["error"] == "not ready"


def test_repository_input_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    profile = load_extraction_profile(PROFILE)
    repository = tmp_path / "repository"; repository.mkdir()
    rows = [
        profile.document["dataset"]["profile"],
        profile.document["dataset"]["manifest"],
        profile.document["dataset"]["metadata"],
        profile.document["checkpoint_plan"],
        profile.document["runtime"]["main"]["profile"],
        profile.document["runtime"]["dinov3"]["profile"],
        profile.document["runtime"]["dinov3"]["extractor_profile"],
        *profile.document["source_bindings"],
    ]
    for row in rows:
        target = repository / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / row["path"], target)
    (repository / profile.document["source_bindings"][0]["path"]).write_bytes(b"corrupt")
    result = subprocess.run(
        [sys.executable, "-B", str(EXTRACT), "plan", "--profile", str(PROFILE),
         "--repository-root", str(repository)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 3


def test_preflight_success_with_synthetic_local_resources(tmp_path: Path) -> None:
    cli = _load_cli(); profile = load_extraction_profile(PROFILE)
    dataset = tmp_path / "dataset"; dataset.mkdir()
    cache = tmp_path / "cache"; cache.mkdir()
    for model in profile.models:
        snapshot = cli._snapshot(cache, model["model_id"], model["revision"])
        snapshot.mkdir(parents=True)
        (snapshot / "synthetic-metadata.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], bool]] = []

    def gpu_runner(arguments, **kwargs):
        calls.append((tuple(arguments), kwargs["shell"]))
        return subprocess.CompletedProcess(arguments, 0, "0, NVIDIA GeForce RTX 5090\n", "")

    result = cli._preflight(
        profile,
        ROOT,
        dataset,
        cache,
        dataset_validator=lambda rows, root: None,
        interpreter_is_file=lambda path: True,
        host_name=lambda: "apple",
        gpu_runner=gpu_runner,
    )
    assert result["classification"] == "P1_FEATURE_EXTRACTION_PREFLIGHT_VALID"
    assert calls == [(('nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader,nounits'), False)]


def test_preflight_empty_snapshot_is_resource_failure(tmp_path: Path) -> None:
    cli = _load_cli(); profile = load_extraction_profile(PROFILE)
    dataset = tmp_path / "dataset"; dataset.mkdir()
    cache = tmp_path / "cache"; cache.mkdir()
    with pytest.raises(cli.FeatureExtractionError, match="checkpoint snapshot unavailable") as error:
        cli._preflight(
            profile, ROOT, dataset, cache,
            dataset_validator=lambda rows, root: None,
            interpreter_is_file=lambda path: True,
            host_name=lambda: "apple",
        )
    assert error.value.exit_code == 4


def test_fake_three_encoder_smoke_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli(); profile = load_extraction_profile(PROFILE)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def runner(command, environment):
        calls.append((list(command), dict(environment)))
        return _fake_worker(profile, list(command))

    monkeypatch.setattr(cli, "run_command", runner)
    assert cli.main(_arguments(tmp_path)) == 0
    run_root = tmp_path / "outputs" / RUN_ID
    assert validate_run(run_root, profile)["image_count"] == 10
    assert [call[0][call[0].index("--model") + 1] for call in calls] == list(MODEL_ORDER)
    assert all("-I" not in call[0] and call[0][1] == "-B" for call in calls)
    assert all(call[1]["HF_HUB_OFFLINE"] == "1" and call[1]["CUDA_VISIBLE_DEVICES"] == "0" for call in calls)
    result = subprocess.run([sys.executable, "-B", str(VALIDATE), "--profile", str(PROFILE), "--run-root", str(run_root)], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert json.loads(result.stdout)["classification"] == "P1_FEATURE_EXTRACTION_COMPLETE_VALIDATED"
    files = {path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_file()}
    assert all("/shards/" not in path for path in files)
    assert all(f"models/{model}/execution_metrics.json" in files for model in MODEL_ORDER)


@pytest.mark.parametrize("return_code", [-9, 300, 9])
def test_worker_failure_leaves_incomplete_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, return_code: int) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "run_command", lambda command, environment: subprocess.CompletedProcess(command, return_code, "", "failed"))
    assert cli.main(_arguments(tmp_path)) == EXIT_EXECUTION
    run_root = tmp_path / "outputs" / RUN_ID
    assert (run_root / "run_state.json").is_file()
    assert not (run_root / "run_manifest.json").exists()
    failures = list((run_root / "failures").glob("attempt-*.json"))
    assert len(failures) == 1
    evidence = json.loads(failures[0].read_text(encoding="utf-8"))
    assert evidence["model_key"] == MODEL_ORDER[0]
    assert evidence["failure_stage"] == "worker_process"
    assert evidence["stable_exit_code"] == EXIT_EXECUTION
    assert evidence["raw_return_code"] == return_code


def _canonical_child_error(error: str, *, exit_code: int = 4) -> str:
    return json.dumps({
        "error": error,
        "exit_code": exit_code,
        "ready_for_p1_extraction": False,
        "schema_id": "vea.p1.feature_extraction_error.v0",
    }, sort_keys=True, separators=(",", ":")) + "\n"


def _worker_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, stdout: str, stderr: str = "",
) -> dict[str, object]:
    cli = _load_cli()
    monkeypatch.setattr(
        cli, "run_command",
        lambda command, environment: subprocess.CompletedProcess(command, 4, stdout, stderr),
    )
    assert cli.main(_arguments(tmp_path)) == EXIT_EXECUTION
    path = next((tmp_path / "outputs" / RUN_ID / "failures").glob("attempt-*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def test_worker_failure_preserves_safe_canonical_child_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _worker_failure_evidence(
        tmp_path, monkeypatch,
        stdout=_canonical_child_error("checkpoint snapshot unavailable: clip_vit_b16"),
        stderr="arbitrary stderr with token=do-not-persist",
    )
    assert evidence["sanitized_message"] == "checkpoint snapshot unavailable: clip_vit_b16"
    assert evidence["stable_exit_code"] == EXIT_EXECUTION
    assert evidence["raw_return_code"] == 4
    assert "token=do-not-persist" not in json.dumps(evidence, sort_keys=True)
    assert set(evidence) == {
        "attempt_index", "created_at_utc", "end_index_exclusive", "exception_class",
        "failure_scope", "failure_stage", "image_ids", "lane_id", "model_key",
        "raw_return_code", "ready_for_p1_extraction", "run_id", "sanitized_message",
        "schema_id", "shard_index", "stable_exit_code", "start_index",
    }


@pytest.mark.parametrize(
    "case",
    ("malformed", "extra", "missing", "exit_mismatch", "path", "secret", "oversized", "multiline"),
)
def test_worker_failure_rejects_unsafe_or_noncanonical_child_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    document: dict[str, object] = {
        "error": "checkpoint snapshot unavailable: clip_vit_b16",
        "exit_code": 4,
        "ready_for_p1_extraction": False,
        "schema_id": "vea.p1.feature_extraction_error.v0",
    }
    if case == "malformed":
        stdout = "{not-json}\n"
    else:
        if case == "extra":
            document["unexpected"] = "x"
        elif case == "missing":
            document.pop("schema_id")
        elif case == "exit_mismatch":
            document["exit_code"] = 3
        elif case == "path":
            document["error"] = "/home/chltjsdh/vea/cache"
        elif case == "secret":
            document["error"] = "token rejected"
        elif case == "oversized":
            document["error"] = "x" * 161
        elif case == "multiline":
            document["error"] = "first line\nsecond line"
        stdout = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    evidence = _worker_failure_evidence(tmp_path, monkeypatch, stdout=stdout)
    assert evidence["sanitized_message"] == "worker_process_failed"
    assert evidence["stable_exit_code"] == EXIT_EXECUTION
    assert evidence["raw_return_code"] == 4


def test_worker_failure_noncanonical_stdout_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _canonical_child_error("checkpoint snapshot unavailable: clip_vit_b16").strip()
    evidence = _worker_failure_evidence(tmp_path, monkeypatch, stdout=document + "\n\n")
    assert evidence["sanitized_message"] == "worker_process_failed"


def test_resume_continues_only_missing_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli(); profile = load_extraction_profile(PROFILE)
    rows = stream_manifest(ROOT / profile.document["dataset"]["manifest"]["path"], expected_sha256=profile.document["dataset"]["manifest"]["sha256"])
    args = _arguments(tmp_path, "resume")
    run_root = tmp_path / "outputs" / RUN_ID; run_root.mkdir()
    state = initialize_run_artifacts(run_root, profile, ROOT, run_id=RUN_ID, mode="smoke", batch_size=1, image_ids=[row.image_id for row in rows[:10]])
    first = profile.models[0]; plan = plan_shards(state["image_ids"])[0]
    write_shard(run_root / "models" / first["registry_key"], run_id=RUN_ID, model=first, lane_id=first["lane_id"], shard=plan, tensors=_tensors(10), created_at_utc="2026-07-30T03:00:00Z")
    write_execution_metrics(
        run_root / "models" / first["registry_key"] / "execution_metrics.json",
        run_id=RUN_ID, model=first, image_count=10, elapsed_seconds=1.0,
        peak_allocated_bytes=1, peak_reserved_bytes=2,
    )
    starts: dict[str, int] = {}

    def runner(command, environment):
        command = list(command); model = command[command.index("--model") + 1]
        starts[model] = int(command[command.index("--start-shard") + 1])
        return _fake_worker(profile, command)

    monkeypatch.setattr(cli, "run_command", runner)
    assert cli.main(args) == 0
    assert starts == {MODEL_ORDER[1]: 0, MODEL_ORDER[2]: 0}


def test_worker_launch_failure_writes_one_stage_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "run_command", lambda *args: (_ for _ in ()).throw(OSError("launch")))
    assert cli.main(_arguments(tmp_path)) == EXIT_EXECUTION
    run_root = tmp_path / "outputs" / RUN_ID
    records = list((run_root / "failures").glob("attempt-*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text(encoding="utf-8"))["failure_stage"] == "worker_launch"
    assert not (run_root / "run_manifest.json").exists()


def test_failure_evidence_publication_error_does_not_replace_worker_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        cli, "run_command",
        lambda command, environment: subprocess.CompletedProcess(command, -9, "", "failed"),
    )
    monkeypatch.setattr(
        cli, "write_failure_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(FeatureExtractionError("evidence collision", EXIT_ARTIFACT)),
    )
    assert cli.main(_arguments(tmp_path)) == EXIT_EXECUTION
    assert not (tmp_path / "outputs" / RUN_ID / "run_manifest.json").exists()


def test_partial_pair_failure_preserves_competing_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli()

    def runner(command, environment):
        command = list(command)
        run_root = Path(command[command.index("--run-root") + 1])
        model = command[command.index("--model") + 1]
        partial = run_root / "models" / model / "shard-00000.safetensors"
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial-competitor")
        return subprocess.CompletedProcess(command, -9, "", "failed")

    monkeypatch.setattr(cli, "run_command", runner)
    assert cli.main(_arguments(tmp_path)) == EXIT_EXECUTION
    run_root = tmp_path / "outputs" / RUN_ID
    partial = run_root / "models" / MODEL_ORDER[0] / "shard-00000.safetensors"
    assert partial.read_bytes() == b"partial-competitor"
    record = json.loads(next((run_root / "failures").glob("attempt-*.json")).read_text(encoding="utf-8"))
    assert record["stable_exit_code"] == EXIT_EXECUTION and record["raw_return_code"] == -9
    assert not (run_root / "run_manifest.json").exists()


def test_completion_failure_writes_orchestration_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli(); profile = load_extraction_profile(PROFILE)
    monkeypatch.setattr(cli, "run_command", lambda command, environment: _fake_worker(profile, list(command)))
    monkeypatch.setattr(cli, "complete_run", lambda *args, **kwargs: (_ for _ in ()).throw(FeatureExtractionError("completion collision", EXIT_ARTIFACT)))
    assert cli.main(_arguments(tmp_path)) == EXIT_ARTIFACT
    run_root = tmp_path / "outputs" / RUN_ID
    record = json.loads(next((run_root / "failures").glob("attempt-*.json")).read_text(encoding="utf-8"))
    assert record["failure_stage"] == "completion" and record["failure_scope"] == "orchestration"
    assert not (run_root / "run_manifest.json").exists()


def test_model_free_validation_failure_writes_orchestration_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli(); profile = load_extraction_profile(PROFILE)
    monkeypatch.setattr(cli, "run_command", lambda command, environment: _fake_worker(profile, list(command)))
    monkeypatch.setattr(cli, "validate_completion_candidate", lambda *args: (_ for _ in ()).throw(FeatureExtractionError("validation", EXIT_ARTIFACT)))
    assert cli.main(_arguments(tmp_path)) == EXIT_ARTIFACT
    run_root = tmp_path / "outputs" / RUN_ID
    record = json.loads(next((run_root / "failures").glob("attempt-*.json")).read_text(encoding="utf-8"))
    assert record["failure_stage"] == "model_free_validation"
    assert not (run_root / "run_manifest.json").exists()


def test_corrupt_existing_shard_blocks_resume(tmp_path: Path) -> None:
    profile = load_extraction_profile(PROFILE)
    rows = stream_manifest(ROOT / profile.document["dataset"]["manifest"]["path"], expected_sha256=profile.document["dataset"]["manifest"]["sha256"])
    run_root = tmp_path / "outputs" / RUN_ID; run_root.mkdir(parents=True)
    state = initialize_run_artifacts(run_root, profile, ROOT, run_id=RUN_ID, mode="smoke", batch_size=1, image_ids=[row.image_id for row in rows[:10]])
    model = profile.models[0]; plan = plan_shards(state["image_ids"])[0]
    shard_root = run_root / "models" / model["registry_key"]
    write_shard(shard_root, run_id=RUN_ID, model=model, lane_id=model["lane_id"], shard=plan, tensors=_tensors(10), created_at_utc="2026-07-30T03:00:00Z")
    (shard_root / f"{plan.basename}.safetensors").write_bytes(b"corrupt")
    result = subprocess.run([sys.executable, "-B", str(EXTRACT), *_arguments(tmp_path, "resume")], capture_output=True, text=True, check=False)
    assert result.returncode == EXIT_ARTIFACT


def test_full_mode_requires_explicit_batch_size(tmp_path: Path) -> None:
    command = [sys.executable, "-B", str(EXTRACT), "run", "--profile", str(PROFILE), "--repository-root", str(ROOT), "--dataset-root", str(tmp_path), "--cache-root", str(tmp_path), "--output-parent", str(tmp_path), "--run-id", RUN_ID, "--mode", "full"]
    assert subprocess.run(command, capture_output=True, check=False).returncode == 2


def test_smoke_mode_rejects_non_ten_image_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli(); profile = load_extraction_profile(PROFILE)
    rows = stream_manifest(ROOT / profile.document["dataset"]["manifest"]["path"], expected_sha256=profile.document["dataset"]["manifest"]["sha256"])
    monkeypatch.setattr(cli, "stream_manifest", lambda *args, **kwargs: rows[:9])
    monkeypatch.setattr(cli, "run_command", lambda *args: pytest.fail("worker must not run"))
    assert cli.main(_arguments(tmp_path)) != 0


def test_output_collision_rejected_before_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli(); (tmp_path / "dataset").mkdir(); (tmp_path / "cache").mkdir(); (tmp_path / "outputs").mkdir(); (tmp_path / "outputs" / RUN_ID).mkdir()
    monkeypatch.setattr(cli, "run_command", lambda *args: pytest.fail("worker must not run"))
    assert cli.main(_arguments(tmp_path)) != 0
    assert not (tmp_path / "outputs" / RUN_ID / "failures").exists()


def test_validator_rejects_incomplete_by_default(tmp_path: Path) -> None:
    profile = load_extraction_profile(PROFILE); run_root = tmp_path / RUN_ID; run_root.mkdir()
    initialize_run_artifacts(run_root, profile, ROOT, run_id=RUN_ID, mode="smoke", batch_size=1, image_ids=[f"i{x}" for x in range(10)])
    result = subprocess.run([sys.executable, "-B", str(VALIDATE), "--profile", str(PROFILE), "--run-root", str(run_root)], capture_output=True, check=False)
    assert result.returncode == EXIT_ARTIFACT


def test_no_shell_true_or_network_client_in_execution_code() -> None:
    source = EXTRACT.read_text(encoding="utf-8") + (ROOT / "src/vision_encoder_alignment/p1_feature_extraction.py").read_text(encoding="utf-8")
    assert "shell=True" not in source
    for forbidden in ("requests.", "urllib.request", "hf_hub_download", "curl ", "wget ", "ssh "):
        assert forbidden not in source


def test_runbook_has_approval_and_stop_boundaries() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    for required in ("ACCEPTED_PASS", "10-image", "tmux", "READY_FOR_P1_EXTRACTION=false", "Do not run full extraction", "independent approval", "execution_metrics.json", "run_manifest.json"):
        assert required in text
    assert 'MAIN_PYTHON=/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3' in text
    assert 'DINO_PYTHON=/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated/bin/python3' in text
    assert 'models/*/shard-*.json' in text and 'failures/attempt-*.json' in text
    for required in ("VEA_DATASET_ROOT", "VEA_CHECKPOINT_CACHE", "VEA_T003B_EVIDENCE_ROOT", "assert_nonlink_existing"):
        assert required in text
    assert "02-plan.stdout.json" in text and "03-preflight.stdout.json" in text
    assert '> "$EVIDENCE_ROOT/02-plan.stdout.json"' not in text
    assert '> "$EVIDENCE_ROOT/03-preflight.stdout.json"' not in text
    mkdir_index = text.index('mkdir -m 700 -- "$EVIDENCE_ROOT"')
    for precheck in (
        'test "$(hostname)" = apple', 'VEA_APPROVED_REPO_COMMIT',
        'git rev-parse HEAD', 'git diff --cached --name-only',
        'git diff --name-only', 'git ls-files --others --exclude-standard', 'diff --check',
        'assert_nonlink_existing "$R1R3_TASK"',
        'assert_nonlink_existing "$R1R3_REPORT"',
        'assert_nonlink_existing "$PROFILE"',
        'scripts/extract_features.py',
        'src/vision_encoder_alignment/p1_feature_extraction.py',
        'src/vision_encoder_alignment/p1_runtime_preflight.py',
        'scripts/inventory_p1_checkpoint_cache.py',
        'scripts/validate_p1_feature_extraction.py',
        'scripts/validate_p1_apple_runtime.py',
        'assert_nonlink_existing "$DATASET_ROOT"',
        'assert_nonlink_existing "$CACHE_ROOT"',
        'assert_nonlink_existing "$OUTPUT_PARENT"',
        'assert_nonlink_existing "$T002D_EVIDENCE_ROOT"',
        'assert_nonlink_existing "$MAIN_PYTHON"',
        'assert_nonlink_existing "$DINO_PYTHON"',
        'test ! -e "$EVIDENCE_ROOT"',
    ):
        assert text.index(precheck) < mkdir_index
    assert "empty or placeholder path rejected" in text
    assert 'case "$path" in /*)' in text
    assert 'canonical=$(realpath -e -- "$path")' in text
    task_hash_check = text.index(
        'printf \'%s  %s\\n\' "$VEA_R1R3_TASK_SHA256" "$R1R3_TASK" | sha256sum --check --strict'
    )
    sbr1_acceptance_check = text.index(
        "grep -F 'VEA-P1-T002D-SBR1 = ACCEPTED_PASS' \"$R1R3_TASK\" >/dev/null"
    )
    assert task_hash_check < sbr1_acceptance_check < mkdir_index
    assert 'SBR1_REPORT=' not in text
    assert '"$SBR1_REPORT"' not in text
    assert text.count('--output-json "$EVIDENCE_ROOT/02-plan.json"') == 1
    assert text.count('--output-json "$EVIDENCE_ROOT/03-preflight.json"') == 1
    manifest_block = text[text.index('(cd "$EVIDENCE_ROOT" && sha256sum'):]
    for name in (
        "00-runtime-live.json", "01-checkpoint-inventory.json", "02-plan.json",
        "03-preflight.json", "04-readiness.txt",
    ):
        assert name in manifest_block
    assert "--mode full" not in text
    stage_a = text[text.index("## 3. Stage A"):text.index("## 4. Stage B")]
    assert "Stop here and submit Stage A evidence for independent approval." in stage_a
    for artifact in (
        "00-runtime-live.json", "01-checkpoint-inventory.json", "02-plan.json",
        "03-preflight.json", "04-readiness.txt", "05-preflight-manifest.txt",
    ):
        assert artifact in stage_a
    assert 'test "${#VEA_APPROVED_REPO_COMMIT}" -eq 40' in text
    assert 'case "$VEA_APPROVED_REPO_COMMIT" in *[!0-9a-f]*) exit 2;; esac' in text
    assert 'test "$(git rev-parse HEAD)" = "$VEA_APPROVED_REPO_COMMIT"' in text
    assert 'test "$(git branch --show-current)" = main' not in text
    assert "e518a51ba83f70b42611cc4386a9f55dc9e97de3" not in text
    assert 'APPROVED_COMMIT_PREFIX=${VEA_APPROVED_REPO_COMMIT:0:7}' in text
    assert 'three_encoder_${APPROVED_COMMIT_PREFIX}' in text


def test_execution_never_claims_extraction_readiness(tmp_path: Path) -> None:
    profile = load_extraction_profile(PROFILE); run_root = tmp_path / RUN_ID; run_root.mkdir()
    state = initial_run_state(profile, run_id=RUN_ID, mode="smoke", batch_size=1, image_ids=[f"i{x}" for x in range(10)])
    assert state["ready_for_p1_extraction"] is False
