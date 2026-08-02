"""Collect read-only A100/Slurm v0 or direct-host v1 runtime evidence."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from vision_encoder_alignment.p1_runtime_preflight import (  # noqa: E402
    CommandRunner,
    DIRECT_HOST_TEMPLATE_SCHEMA_ID,
    PreflightError,
    StorageProbe,
    SystemProbe,
    collect_direct_host_preflight,
    collect_runtime_preflight,
    direct_host_summary,
    load_canonical_json,
    load_direct_host_template,
    load_runtime_template,
    runtime_summary,
    subprocess_runner,
    system_probe,
    system_storage_probe,
    utc_now,
    write_json_output,
    write_text_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Phase 1 target-specific runtime preflight evidence.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--vea-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--timestamp-utc")
    return parser


ProbeFactory = Callable[[str | Path], SystemProbe]


def _parse_args(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> tuple[argparse.Namespace | None, int | None]:
    try:
        return parser.parse_args(argv), None
    except SystemExit as exc:
        return None, 0 if exc.code == 0 else 3


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: CommandRunner = subprocess_runner,
    probe_factory: ProbeFactory = system_probe,
    storage_probe: StorageProbe = system_storage_probe,
    environment: Mapping[str, str] | None = None,
) -> int:
    args, parse_exit = _parse_args(build_parser(), argv)
    if args is None:
        assert parse_exit is not None
        return parse_exit
    try:
        profile = load_canonical_json(args.profile, label="runtime_template")
        probe = probe_factory(args.repository_root)
        current_environment = dict(os.environ if environment is None else environment)
        timestamp = args.timestamp_utc or utc_now()
        if profile.get("schema_id") == DIRECT_HOST_TEMPLATE_SCHEMA_ID:
            template = load_direct_host_template(args.profile)
            evidence, exit_code = collect_direct_host_preflight(
                template,
                vea_root=args.vea_root,
                repository_root=args.repository_root,
                environment=current_environment,
                runner=runner,
                probe=probe,
                storage_space=storage_probe(args.vea_root),
                timestamp_utc=timestamp,
            )
            summary = direct_host_summary(evidence)
        else:
            template = load_runtime_template(args.profile)
            evidence, exit_code = collect_runtime_preflight(
                template,
                vea_root=args.vea_root,
                repository_root=args.repository_root,
                environment=current_environment,
                runner=runner,
                probe=probe,
                timestamp_utc=timestamp,
            )
            summary = runtime_summary(evidence)
        write_json_output(args.output, evidence)
        if args.summary_output:
            write_text_output(args.summary_output, summary)
        for line in summary:
            print(line)
        return exit_code
    except PreflightError as exc:
        print(f"P1_RUNTIME_PREFLIGHT_FAILED[{exc.classification}]: {exc}", file=sys.stderr)
        return 3 if exc.classification in {
            "invalid_json", "schema_error", "unsafe_invocation", "unsafe_path",
            "unsafe_command", "unsafe_operation", "path_leakage",
            "secret_leakage", "output_collision",
        } else 4


if __name__ == "__main__":
    raise SystemExit(main())
