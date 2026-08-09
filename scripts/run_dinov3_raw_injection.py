#!/usr/bin/env python3
"""CLI for one offline DINOv3 raw block-11 injection smoke."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_encoder_alignment.vlm_compat.dinov3_raw_injection import (  # noqa: E402
    run_dinov3_raw_injection_smoke,
    write_blocker_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one offline DINOv3 raw injection smoke.")
    parser.add_argument("--tinyllava-config", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True, choices=("float16", "bfloat16", "float32"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_was_present = Path(args.output_dir).exists()
    try:
        result = run_dinov3_raw_injection_smoke(
            tinyllava_config=args.tinyllava_config,
            image_path=args.image_path,
            cache_root=args.cache_root,
            repository_root=args.repository_root,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
        )
    except Exception as exc:
        evidence = None if output_was_present else write_blocker_result(
            args.output_dir, stage="runtime", error=exc
        )
        suffix = "" if evidence is None else " RESULT=result.json"
        print(f"STATUS={getattr(exc, 'classification', 'BLOCKED_DINOV3_RAW_INJECTION')} ERROR={type(exc).__name__}{suffix}")
        return 2
    print(f"STATUS={result['status']} IMAGE_COUNT=1 RESULT=result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
