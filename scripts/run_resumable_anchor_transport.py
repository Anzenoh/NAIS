#!/usr/bin/env python3
"""Coordinator CLI for the immutable VEA-P2-T001D run plan."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _repository_src(repository_root: Path) -> Path:
    return repository_root / "src"


def _import_core(repository_root: Path):
    source = str(_repository_src(repository_root))
    if source not in sys.path:
        sys.path.insert(0, source)
    from vision_encoder_alignment.vlm_compat import resumable_anchor_transport as core

    return core


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--dinov3-runtime-root", required=True, type=Path)
    parser.add_argument("--dinov3-snapshot-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--reuse-root", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, execute, resume, verify, and finalize VEA-P2-T001D."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "resume", "verify", "status", "finalize"):
        child = subparsers.add_parser(name)
        _common(child)
    return parser


def _resolved_existing_directory(value: Path, label: str) -> Path:
    resolved = value.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} must be an existing directory: {resolved}")
    return resolved


def _repository_commit(repository_root: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    commit = process.stdout.strip().lower()
    if process.returncode != 0 or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("repository root does not resolve to an exact Git commit")
    return commit


def _bindings(args: argparse.Namespace, core) -> tuple[dict, list[dict], str, str, str]:
    repository = _resolved_existing_directory(args.repository_root, "repository root")
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    if not config_path.is_file() or not manifest_path.is_file():
        raise ValueError("configuration and manifest must be regular files")
    config = core.load_config(config_path)
    core.validate_identity_sources(config, repository)
    config_sha = core.sha256_file(config_path)
    manifest_sha = core.sha256_file(manifest_path)
    if config["manifest"]["sha256"] != manifest_sha:
        raise core.ContractError("manifest hash differs from immutable configuration")
    rows = core.read_manifest(manifest_path, manifest_sha)
    return config, rows, _repository_commit(repository), config_sha, manifest_sha


def _root_bindings(args: argparse.Namespace) -> dict[str, str]:
    return {
        "dataset_root": str(args.dataset_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "repository_root": str(args.repository_root.resolve()),
        "dino_runtime_root": str(args.dinov3_runtime_root.resolve()),
        "dino_snapshot_root": str(args.dinov3_snapshot_root.resolve()),
        "run_root": str(args.run_root.resolve()),
        "reuse_root": str(args.reuse_root.resolve()) if args.reuse_root else None,
    }


def _validate_existing(args: argparse.Namespace, core):
    config, rows, commit, config_sha, manifest_sha = _bindings(args, core)
    plan, plan_id = core.load_bound_plan(args.run_root.resolve())
    core.validate_resume_bindings(
        plan,
        repository_commit=commit,
        config_sha256=config_sha,
        manifest_sha256=manifest_sha,
        roots=_root_bindings(args),
        device=args.device,
    )
    if plan["identity_bindings"]["encoders"] != config["encoders"]:
        raise core.PlanMismatchError("current encoder/profile identity differs from immutable plan")
    if plan["identity_bindings"]["target_vlm"] != config["target_vlm"]:
        raise core.PlanMismatchError("current target VLM identity differs from immutable plan")
    return plan, plan_id


def _plan(args: argparse.Namespace, core) -> dict:
    config, rows, commit, config_sha, manifest_sha = _bindings(args, core)
    for value, label in (
        (args.dataset_root, "dataset root"),
        (args.cache_root, "cache root"),
        (args.dinov3_runtime_root, "DINOv3 runtime root"),
        (args.dinov3_snapshot_root, "DINOv3 snapshot root"),
    ):
        _resolved_existing_directory(value, label)
    if args.reuse_root is not None:
        _resolved_existing_directory(args.reuse_root, "reuse root")
    dataset = args.dataset_root.resolve()
    for row in rows:
        candidate = (dataset / row["relative_path"]).resolve()
        try:
            candidate.relative_to(dataset)
        except ValueError as error:
            raise core.ContractError(f"manifest path escapes dataset root: {row['image_id']}") from error
        if not candidate.is_file() or core.sha256_file(candidate) != row["source_sha256"]:
            raise core.ContractError(f"dataset source SHA mismatch: {row['image_id']}")
    parent = args.run_root.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(parent).free
    plan = core.build_run_plan(
        config,
        rows,
        repository_commit=commit,
        config_sha256=config_sha,
        manifest_sha256=manifest_sha,
        dataset_root=str(args.dataset_root),
        cache_root=str(args.cache_root),
        repository_root=str(args.repository_root),
        dino_runtime_root=str(args.dinov3_runtime_root),
        dino_snapshot_root=str(args.dinov3_snapshot_root),
        run_root=str(args.run_root),
        device=args.device,
        storage_available_bytes=available,
        reuse_root=str(args.reuse_root) if args.reuse_root else None,
    )
    plan_id = core.initialize_run(args.run_root.resolve(), plan)
    return {
        "status": "PASS",
        "command": "plan",
        "plan_id": plan_id,
        "expected_unit_count": plan["expected_unit_count"],
        "expected_generation_count": plan["expected_generation_count"],
        "storage_required_gib": plan["storage_preflight"]["required_with_30_percent_margin_gib"],
        "storage_available_gib": round(available / (1024**3), 6),
    }


def _stage_interpreter(plan: dict, stage: str) -> str:
    if stage != "01_DINOV3_FEATURE_BANK":
        return sys.executable
    runtime_root = Path(plan["roots"]["dino_runtime_root"])
    candidates = (runtime_root / "bin" / "python3", runtime_root / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise ValueError("approved DINOv3 runtime has no bin/python3 or bin/python interpreter")


def _execute(args: argparse.Namespace, core, *, resume: bool) -> dict:
    plan, plan_id = _validate_existing(args, core)
    current = core.verify_run(args.run_root.resolve())
    if not resume and current["completed_unit_count"]:
        raise core.ContractError("run requires a valid unstarted plan; use resume")
    if current["completed_unit_count"] == 0:
        available = shutil.disk_usage(args.run_root.resolve()).free
        required = int(plan["storage_preflight"]["required_with_30_percent_margin_bytes"])
        if available < required:
            raise core.ContractError(
                f"storage preflight no longer passes before first GPU stage: required={required} available={available}"
            )
    worker_script = Path(plan["roots"]["repository_root"]) / "scripts" / "p2_anchor_transport_stage_worker.py"
    stop_requested = False

    def request_stop(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)
    try:
        with core.CoordinatorLock(args.run_root.resolve(), plan_id):
            quarantined = core.recover_orphans(args.run_root.resolve(), plan, plan_id)
            reuse = core.reuse_exact_feature_units(args.run_root.resolve(), plan, plan_id)
            event = "RESUMED" if resume else "RUN_STARTED"
            core.append_event(
                args.run_root.resolve(),
                core.progress_event(plan_id, event, quarantined_paths=quarantined, reuse=reuse),
            )
            for stage in core.STAGES[1:7]:
                if stop_requested:
                    core.append_event(args.run_root.resolve(), core.progress_event(plan_id, "INTERRUPTED", stage=stage))
                    raise core.RunInterruptedError("stop requested before next stage")
                command = [
                    _stage_interpreter(plan, stage),
                    str(worker_script),
                    "--stage",
                    stage,
                    "--plan",
                    str(args.run_root.resolve() / "run_plan.json"),
                    "--run-root",
                    str(args.run_root.resolve()),
                ]
                process = subprocess.run(command, check=False)
                if process.returncode == core.RETRYABLE_EXIT_CODE:
                    raise core.IncompleteRunError(f"stage worker reported retryable resource failure: {stage}")
                if process.returncode == 130:
                    raise core.RunInterruptedError(f"stage worker stopped after its current atomic unit: {stage}")
                if process.returncode != 0:
                    raise core.ContractError(f"stage worker failed with exit code {process.returncode}: {stage}")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return {"status": "PASS", "command": "resume" if resume else "run", **core.verify_run(args.run_root.resolve())}


def _status(run_root: Path, core) -> dict:
    plan, plan_id = core.load_bound_plan(run_root)
    stages = []
    for stage in core.STAGES:
        units = [unit for unit in plan["expected_units"] if unit["stage"] == stage]
        completed = sum(core.validate_unit_completion(run_root, unit, plan_id) for unit in units)
        stages.append({"stage": stage, "completed": completed, "expected": len(units)})
    return {"status": "PASS", "plan_id": plan_id, "stages": stages}


def _exact_recovery_arguments(args: argparse.Namespace) -> list[str]:
    values = []
    for option, attribute in (
        ("--config", "config"),
        ("--manifest", "manifest"),
        ("--dataset-root", "dataset_root"),
        ("--cache-root", "cache_root"),
        ("--repository-root", "repository_root"),
        ("--dinov3-runtime-root", "dinov3_runtime_root"),
        ("--dinov3-snapshot-root", "dinov3_snapshot_root"),
        ("--run-root", "run_root"),
        ("--device", "device"),
    ):
        values.extend((option, str(getattr(args, attribute))))
    if args.reuse_root:
        values.extend(("--reuse-root", str(args.reuse_root)))
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    core = _import_core(args.repository_root.resolve())
    try:
        if args.command == "plan":
            output = _plan(args, core)
        elif args.command == "run":
            output = _execute(args, core, resume=False)
        elif args.command == "resume":
            output = _execute(args, core, resume=True)
        elif args.command == "verify":
            _validate_existing(args, core)
            output = core.verify_run(args.run_root.resolve(), require_final=False)
        elif args.command == "status":
            _validate_existing(args, core)
            output = _status(args.run_root.resolve(), core)
        else:
            _validate_existing(args, core)
            output = core.finalize_run(args.run_root.resolve())
        print(json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    except BaseException as error:
        plan_id = None
        try:
            _, plan_id = core.load_bound_plan(args.run_root.resolve())
        except BaseException:
            pass
        recovery = core.format_recovery_command(
            [sys.executable, str(Path(__file__).resolve())],
            _exact_recovery_arguments(args),
        )
        record = core.failure_record(error, plan_id, recovery)
        try:
            path = core.write_failure_record(args.run_root.resolve(), record)
            record["failure_record"] = str(path)
        except BaseException:
            pass
        print(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False), file=sys.stderr)
        return int(record["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
