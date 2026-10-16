"""Explicit-root CLI for the frozen Train-only low-rank fitting runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vision_encoder_alignment.vlm_compat.query_adapter_low_rank_train_fit_runtime import (
    run_low_rank_train_fit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--t001d-run-root", type=Path, required=True)
    parser.add_argument("--t002d-validation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_low_rank_train_fit(args.repository_root, args.t001d_run_root,
                                        args.t002d_validation_root, args.output_root)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 1
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
