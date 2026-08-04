#!/usr/bin/env python3
"""Independently verify a sealed Phase 1 analysis package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_encoder_alignment.analysis.artifacts import (  # noqa: E402
    ArtifactError,
    verify_review_bundle,
    verify_sealed_package,
)
from vision_encoder_alignment.analysis.config import canonical_json_bytes  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--package")
    target.add_argument("--review-bundle")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    try:
        if args.review_bundle:
            result = verify_review_bundle(args.review_bundle)
            observed = result["review_bundle_sha256"]
            kind = "review_bundle"
        else:
            result = verify_sealed_package(args.package)
            observed = result["sealed_package_sha256"]
            kind = "sealed_package"
        if args.expected_sha256 and observed != args.expected_sha256:
            raise ArtifactError(f"{kind} checksum differs from expected")
        sys.stdout.buffer.write(canonical_json_bytes({
            "schema_id": "vea.p1.package_verification_cli.v1",
            "status": "PASS",
            "verified_kind": kind,
            **result,
        }))
        return 0
    except (ArtifactError, OSError) as exc:
        sys.stderr.buffer.write(canonical_json_bytes({
            "schema_id": "vea.p1.package_verification_cli.v1",
            "status": "FAIL",
            "first_root_cause": str(exc).replace("\r", " ").replace("\n", " ")[:500],
        }))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
