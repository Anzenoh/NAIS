#!/usr/bin/env python3
"""Standalone structural, artifact, setup, and runtime validator for Apple P1."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping, Sequence

from vision_encoder_alignment.production_runtime_contract import ProductionRuntimeContractError
from vision_encoder_alignment.p1_apple_runtime_setup import (
    AppleRuntimeSetupError,
    atomic_write_json,
    build_operation_plan,
    canonical_json_bytes,
    execute_operation_plan,
    live_host_identity,
    load_setup_plan,
    load_strict_json,
    structural_result,
    validate_artifact_root,
    validate_runtime_probe_document,
)


EXIT_BY_CLASSIFICATION = {
    "NONCANONICAL_JSON": 2,
    "SETUP_CONTRACT_INVALID": 2,
    "PROFILE_BINDING_MISMATCH": 2,
    "REQUIREMENTS_MISMATCH": 2,
    "UNSAFE_PATH": 2,
    "UNSAFE_RUNTIME_ROOT": 2,
    "ARTIFACT_MISMATCH": 3,
    "RUNTIME_PROBE_MISMATCH": 4,
    "CUDA_RUNTIME_UNAVAILABLE": 4,
    "PACKAGE_IDENTITY_MISMATCH": 4,
    "LANE_ISOLATION_FAILURE": 4,
    "OUTPUT_EXISTS": 5,
    "SETUP_EXECUTION_FAILED": 6,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--mode",
        choices=("structural", "artifacts", "runtime-probe", "runtime-live", "setup-dry-run", "setup-execute"),
        default="structural",
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--python-root")
    parser.add_argument("--work-root")
    parser.add_argument("--probe-json")
    parser.add_argument("--output-json")
    parser.add_argument("--transcript")
    return parser


def _required(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(args, name.replace("-", "_")) is None]
    if missing:
        raise AppleRuntimeSetupError(
            "missing required arguments: " + ", ".join(f"--{name}" for name in missing)
        )


def _result_with_artifacts(plan: Any, artifact_root: str) -> dict[str, Any]:
    result = structural_result(plan)
    result.update(validate_artifact_root(plan, artifact_root))
    result["artifact_validation_performed"] = True
    result["classification"] = "APPLE_OFFLINE_ARTIFACT_SET_VALID"
    return result


_LIVE_PROBE = r'''import importlib.metadata as m, importlib.util, json, pathlib, re, site, sys
import torch
prefix = pathlib.Path(sys.prefix).resolve()
base_prefix = pathlib.Path(sys.base_prefix).resolve()
cfg_path = prefix / "pyvenv.cfg"
cfg_text = cfg_path.read_text(encoding="utf-8")
cfg = {}
for line in cfg_text.splitlines():
  if not line or "=" not in line:
    raise RuntimeError("malformed pyvenv.cfg")
  key, value = line.split("=", 1)
  key, value = key.strip().lower(), value.strip().lower()
  if not key or not value or key in cfg:
    raise RuntimeError("malformed or duplicate pyvenv.cfg setting")
  cfg[key] = value
if cfg.get("include-system-site-packages") != "false":
  raise RuntimeError("system site packages are not disabled")
if site.ENABLE_USER_SITE is not False:
  raise RuntimeError("user site packages are not disabled")
site_paths = sorted({str(pathlib.Path(value).resolve()) for value in site.getsitepackages()})
if len(site_paths) != 1:
  raise RuntimeError("site-packages identity is ambiguous")
installed = {}
metadata_paths = {}
for distribution in m.distributions():
  raw_name = distribution.metadata.get("Name")
  raw_version = distribution.version
  if not isinstance(raw_name, str) or not raw_name.strip() or raw_name != raw_name.strip():
    raise RuntimeError("distribution Name metadata is missing or malformed")
  if not isinstance(raw_version, str) or not raw_version.strip() or raw_version != raw_version.strip():
    raise RuntimeError("distribution version metadata is missing or malformed")
  name = re.sub(r"[-_.]+", "-", raw_name).lower()
  if not name or name in installed:
    raise RuntimeError("normalized distribution name collision")
  installed[name] = raw_version
  metadata_paths[name] = str(pathlib.Path(distribution.locate_file("")).resolve())
payload = {
  "autocast_enabled": bool(torch.is_autocast_enabled()),
  "base_prefix": str(base_prefix),
  "cuda_available": bool(torch.cuda.is_available()),
  "cudnn_version": str(torch.backends.cudnn.version()),
  "distribution_metadata_paths": metadata_paths,
  "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
  "installed_packages": installed,
  "interpreter": str(pathlib.Path(sys.executable).resolve()),
  "offline": all(__import__("os").environ.get(k) == "1" for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_DATASETS_OFFLINE","PIP_NO_INDEX")),
  "python_version": __import__("platform").python_version(),
  "pyvenv_cfg": cfg_text,
  "runtime_root": str(prefix),
  "selected_device_index": 0,
  "site_packages": site_paths[0],
  "sys_path": [str(pathlib.Path(value).resolve()) for value in sys.path if value],
  "system_site_packages": cfg["include-system-site-packages"] == "true",
  "torch_cuda_build": torch.version.cuda,
  "torch_runtime_version": torch.__version__,
  "torchvision_distribution_present": "torchvision" in installed,
  "torchvision_imported_module_present": "torchvision" in sys.modules,
  "torchvision_module_spec_present": importlib.util.find_spec("torchvision") is not None,
  "user_site_enabled": False,
  "visible_gpu_count": torch.cuda.device_count(),
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))'''


LiveProbeRunner = Callable[
    [Sequence[str], Mapping[str, str]], subprocess.CompletedProcess[str]
]


def _run_live_probe(
    argv: Sequence[str], environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        env=dict(environment),
        shell=False,
    )


def _collect_live_probe(
    plan: Any,
    *,
    runner: LiveProbeRunner = _run_live_probe,
    host_probe: Callable[[], dict[str, str]] = live_host_identity,
) -> dict[str, Any]:
    host = host_probe()
    lanes: dict[str, Any] = {}
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for name, lane in (("main", plan.main), ("dinov3", plan.dinov3)):
        interpreter = f"{lane.runtime_root}/bin/python"
        result = runner((interpreter, "-I", "-B", "-c", _LIVE_PROBE), environment)
        if result.returncode != 0:
            raise AppleRuntimeSetupError("live lane probe failed", "RUNTIME_PROBE_MISMATCH")
        try:
            lane_probe = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AppleRuntimeSetupError("live lane probe returned invalid JSON", "RUNTIME_PROBE_MISMATCH") from exc
        lanes[name] = lane_probe
    return {"host": host, "lanes": lanes, "schema_id": "vea.p1.apple_runtime_probe.v1"}


def _ensure_absent_target(path: str, label: str) -> None:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise AppleRuntimeSetupError(f"{label} is not absolute", "UNSAFE_RUNTIME_ROOT")
    current = candidate
    while current != current.parent:
        if current.is_symlink():
            raise AppleRuntimeSetupError(f"{label} has a symlink component", "UNSAFE_RUNTIME_ROOT")
        current = current.parent
    if candidate.exists() or candidate.is_symlink():
        raise AppleRuntimeSetupError(f"{label} already exists", "OUTPUT_EXISTS")


def _validate_execute_host() -> None:
    host = live_host_identity()
    if host != {
        "architecture": "x86_64",
        "glibc": "2.35",
        "hostname": "apple",
        "operating_system": "Linux",
    }:
        raise AppleRuntimeSetupError("execute host identity mismatch", "RUNTIME_PROBE_MISMATCH")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if result.returncode != 0 or result.stdout.strip() != "0, NVIDIA GeForce RTX 5090":
        raise AppleRuntimeSetupError("execute GPU identity mismatch", "RUNTIME_PROBE_MISMATCH")


def _execute(plan: Any, args: argparse.Namespace) -> dict[str, Any]:
    _required(args, "artifact-root", "python-root", "work-root", "output-json", "transcript")
    _validate_execute_host()
    validate_artifact_root(plan, args.artifact_root)
    for value, label in (
        (args.python_root, "Python root"),
        (args.work_root, "work root"),
        (plan.main.runtime_root, "main runtime root"),
        (plan.dinov3.runtime_root, "DINOv3 runtime root"),
        (args.output_json, "result output"),
        (args.transcript, "transcript output"),
    ):
        _ensure_absent_target(value, label)
    commands = build_operation_plan(
        plan,
        artifact_root=args.artifact_root,
        python_root=args.python_root,
        work_root=args.work_root,
    )
    transcript_path = Path(args.transcript)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = transcript_path.open("x", encoding="utf-8", newline="\n")
    except FileExistsError as exc:
        raise AppleRuntimeSetupError("transcript already exists", "OUTPUT_EXISTS") from exc

    def runner(argv: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
        command_cwd = args.work_root if argv[0].endswith("/configure") else None
        result = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            env=dict(env),
            cwd=command_cwd,
            shell=False,
        )
        handle.write(json.dumps({"argv": list(argv), "returncode": result.returncode}, sort_keys=True) + "\n")
        handle.write(result.stdout)
        handle.write(result.stderr)
        handle.flush()
        os.fsync(handle.fileno())
        return result

    try:
        execute_operation_plan(commands, runner=runner)
        handle.write("APPLE_RUNTIME_INSTALLATION_COMPLETE_VALIDATION_REQUIRED\n")
    finally:
        handle.close()
    result = {
        "classification": "APPLE_RUNTIME_INSTALLED_POST_VALIDATION_REQUIRED",
        "command_count": len(commands),
        "installation_executed": True,
        "ready_for_p1_extraction": False,
        "runtime_validation_required": True,
        "schema_id": "vea.p1.apple_runtime_setup_execution.v1",
        "target_id": "lab_apple_rtx5090",
    }
    atomic_write_json(args.output_json, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_setup_plan(args.repository_root)
        if args.mode == "structural":
            result = structural_result(plan)
        elif args.mode == "artifacts":
            _required(args, "artifact-root")
            result = _result_with_artifacts(plan, args.artifact_root)
        elif args.mode == "runtime-probe":
            _required(args, "probe-json")
            result = validate_runtime_probe_document(plan, load_strict_json(args.probe_json))
        elif args.mode == "runtime-live":
            result = validate_runtime_probe_document(plan, _collect_live_probe(plan))
        elif args.mode == "setup-dry-run":
            _required(args, "artifact-root", "python-root", "work-root")
            commands = build_operation_plan(
                plan,
                artifact_root=args.artifact_root,
                python_root=args.python_root,
                work_root=args.work_root,
            )
            result = structural_result(plan)
            result.update(
                {
                    "classification": "APPLE_RUNTIME_SETUP_DRY_RUN",
                    "command_argv": [list(command) for command in commands],
                    "command_count": len(commands),
                }
            )
        else:
            result = _execute(plan, args)
        if args.output_json and args.mode != "setup-execute":
            atomic_write_json(args.output_json, result)
        sys.stdout.buffer.write(canonical_json_bytes(result))
        return 0
    except AppleRuntimeSetupError as exc:
        error = {
            "classification": exc.classification,
            "message": str(exc),
            "ready_for_p1_extraction": False,
            "schema_id": "vea.p1.apple_runtime_setup_error.v1",
        }
        sys.stderr.buffer.write(canonical_json_bytes(error))
        return EXIT_BY_CLASSIFICATION.get(exc.classification, 2)
    except ProductionRuntimeContractError as exc:
        error = {
            "classification": exc.classification,
            "message": str(exc),
            "ready_for_p1_extraction": False,
            "schema_id": "vea.p1.apple_runtime_setup_error.v1",
        }
        sys.stderr.buffer.write(canonical_json_bytes(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
