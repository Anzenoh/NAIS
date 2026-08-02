"""Validate canonical direct-host evidence without optional research packages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from vision_encoder_alignment.p1_runtime_preflight import (  # noqa: E402
    PreflightError,
    decide_direct_host_preflight,
    direct_host_decision_summary,
    load_canonical_json,
    load_direct_host_template,
    utc_now,
    validate_direct_host_evidence,
    write_json_output,
    write_text_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 direct-GPU-host preflight evidence."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--runtime-evidence", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--timestamp-utc")
    return parser


def _parse_args(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> tuple[argparse.Namespace | None, int | None]:
    try:
        return parser.parse_args(argv), None
    except SystemExit as exc:
        return None, 0 if exc.code == 0 else 3


def main(argv: Sequence[str] | None = None) -> int:
    args, parse_exit = _parse_args(build_parser(), argv)
    if args is None:
        assert parse_exit is not None
        return parse_exit
    try:
        template = load_direct_host_template(args.profile)
        evidence = load_canonical_json(
            args.runtime_evidence, label="direct_host_evidence"
        )
        validate_direct_host_evidence(
            evidence, template, repository_root=args.repository_root
        )
        decision = decide_direct_host_preflight(
            evidence,
            template,
            timestamp_utc=args.timestamp_utc or utc_now(),
            repository_root=args.repository_root,
        )
        summary = direct_host_decision_summary(decision)
        write_json_output(args.output, decision)
        if args.summary_output:
            write_text_output(args.summary_output, summary)
        for line in summary:
            print(line)
        return 0 if decision["ready_to_author_apple_runtime_setup"] else 2
    except PreflightError as exc:
        print(
            f"P1_DIRECT_HOST_PREFLIGHT_VALIDATION_FAILED[{exc.classification}]: {exc}",
            file=sys.stderr,
        )
        return 3 if exc.classification in {
            "invalid_json", "schema_error", "unsafe_invocation", "unsafe_path",
            "unsafe_operation", "path_leakage", "secret_leakage",
            "output_collision",
        } else 4


if __name__ == "__main__":
    raise SystemExit(main())
