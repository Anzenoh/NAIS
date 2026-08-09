#!/usr/bin/env python3
"""CLI for the frozen TinyLLaVA native replay harness."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_encoder_alignment.vlm_compat.tinyllava_native_replay import (  # noqa: E402
    load_anchor_config,
    run_native_replay,
    write_blocker_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or run one offline frozen TinyLLaVA native replay.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-config-only",
        action="store_true",
        help="validate the pinned JSON using only the standard library",
    )
    mode.add_argument("--run", action="store_true", help="run one Apple CUDA replay")
    parser.add_argument("--config", required=True, help="pinned anchor JSON")
    parser.add_argument("--image-path", help="single input image")
    parser.add_argument("--cache-root", help="existing offline Hugging Face hub root")
    parser.add_argument("--output-dir", help="fresh evidence directory")
    parser.add_argument("--device", help="Apple CUDA device, for example cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        help="runtime floating dtype",
    )
    return parser


def _require_run_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("image_path", "cache_root", "output_dir", "device", "dtype")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("--run requires " + ", ".join("--" + name.replace("_", "-") for name in missing))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate_config_only:
        try:
            config = load_anchor_config(args.config)
        except Exception as exc:
            print(f"STATUS=BLOCKED_INVALID_ANCHOR_CONFIG ERROR={type(exc).__name__}")
            return 2
        print(f"STATUS=PASS_CONFIG MODEL={config.model_id} REVISION={config.revision}")
        return 0

    _require_run_arguments(parser, args)
    output_was_present = Path(args.output_dir).exists()
    try:
        result = run_native_replay(
            config_path=args.config,
            image_path=args.image_path,
            cache_root=args.cache_root,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
            repository_root=ROOT,
        )
    except Exception as exc:
        evidence = None if output_was_present else write_blocker_result(args.output_dir, exc)
        suffix = f" RESULT={evidence.name}" if evidence is not None else ""
        classification = getattr(exc, "classification", "BLOCKED_RUNTIME_FAILURE")
        print(f"STATUS={classification} ERROR={type(exc).__name__}{suffix}")
        return 2
    print(
        "STATUS=" + str(result["status"])
        + " IMAGE_COUNT=1 RESULT=result.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
