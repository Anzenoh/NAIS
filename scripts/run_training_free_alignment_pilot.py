#!/usr/bin/env python3
"""Run the bounded VEA-P2-T001C Apple pilot."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic 8/4 training-free alignment pilot")
    parser.add_argument("--tinyllava-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--dinov3-runtime-root", required=True)
    parser.add_argument("--dinov3-snapshot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True, choices=("float16", "bfloat16", "float32"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Path(args.repository_root)
    sys.path.insert(0, str(repository / "src"))
    from vision_encoder_alignment.vlm_compat.training_free_alignment_pilot import (
        run_training_free_alignment_pilot,
        write_failure_result,
    )

    try:
        result = run_training_free_alignment_pilot(
            tinyllava_config=args.tinyllava_config,
            manifest=args.manifest,
            dataset_root=args.dataset_root,
            cache_root=args.cache_root,
            repository_root=args.repository_root,
            dinov3_runtime_root=args.dinov3_runtime_root,
            dinov3_snapshot_root=args.dinov3_snapshot_root,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
        )
    except Exception as exc:
        traceback.print_exc()
        evidence = write_failure_result(args.output_dir, error=exc)
        stage = getattr(exc, "stage", "unclassified")
        image_id = getattr(exc, "image_id", None)
        condition = getattr(exc, "condition", None)
        print(
            "BLOCKED_TRAINING_FREE_ALIGNMENT_PILOT "
            f"stage={stage} image_id={image_id} condition={condition} evidence={evidence}"
        )
        return 1
    print(f"{result['status']} result={Path(args.output_dir) / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
