#!/usr/bin/env python3
"""Model-free validator for a P1 feature-extraction run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_encoder_alignment.p1_feature_extraction import (  # noqa: E402
    FeatureExtractionError,
    canonical_json_bytes,
    load_extraction_profile,
    validate_run,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate_run(
            args.run_root, load_extraction_profile(args.profile),
            allow_incomplete=args.allow_incomplete,
        )
        sys.stdout.buffer.write(canonical_json_bytes(result))
        return 0
    except FeatureExtractionError as exc:
        sys.stderr.buffer.write(canonical_json_bytes({
            "error": str(exc), "exit_code": exc.exit_code,
            "ready_for_p1_extraction": False,
            "schema_id": "vea.p1.feature_extraction_error.v0",
        }))
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
