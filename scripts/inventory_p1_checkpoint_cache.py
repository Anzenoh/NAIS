"""Inventory exact-revision checkpoint files without loading model weights."""

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
    inventory_checkpoint_cache,
    inventory_summary,
    load_canonical_json,
    load_checkpoint_plan,
    utc_now,
    validate_checkpoint_inventory,
    verify_inventory_against_cache,
    write_json_output,
    write_text_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only inventory of the external checkpoint cache.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--expected-inventory")
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
        plan = load_checkpoint_plan(args.plan, args.registry)
        evidence = inventory_checkpoint_cache(
            plan,
            cache_root=args.cache_root,
            repository_root=args.repository_root,
            timestamp_utc=args.timestamp_utc or utc_now(),
        )
        if args.expected_inventory:
            expected = load_canonical_json(args.expected_inventory, label="expected_inventory")
            validate_checkpoint_inventory(expected, plan)
            verify_inventory_against_cache(
                expected,
                plan,
                cache_root=args.cache_root,
                repository_root=args.repository_root,
            )
        write_json_output(args.output, evidence)
        if args.summary_output:
            write_text_output(args.summary_output, inventory_summary(evidence))
        for line in inventory_summary(evidence):
            print(line)
        return 0 if evidence["status"] == "COMPLETE" else 2
    except PreflightError as exc:
        print(f"P1_CHECKPOINT_INVENTORY_FAILED[{exc.classification}]: {exc}", file=sys.stderr)
        return 3 if exc.classification in {
            "invalid_json", "schema_error", "unsafe_invocation", "unsafe_path",
            "unsafe_operation", "root_escape", "credential_discovery",
            "path_leakage", "secret_leakage", "output_collision",
        } else 4


if __name__ == "__main__":
    raise SystemExit(main())
