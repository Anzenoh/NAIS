#!/usr/bin/env python3
"""Portable header/checksum-only validator for the accepted P1 analysis input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_encoder_alignment.analysis.access_control import AccessError, validate_analysis_input  # noqa: E402
from vision_encoder_alignment.analysis.config import ConfigError, canonical_json_bytes, load_analysis_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--config", default=str(ROOT / "configs" / "analysis" / "p1_analysis_campaign_v1.json"))
    args = parser.parse_args(argv)
    try:
        config = load_analysis_config(args.config)
        result = validate_analysis_input(
            run_root=args.run_root,
            evidence_root=args.evidence_root,
            repository_root=args.repository_root,
            config=config,
        )
        document = {
            "schema_id": "vea.p1.analysis_input_validation.v1",
            "status": "PASS",
            "run_id": config.document["input"]["run_id"],
            "verified_file_count": result.verified_file_count,
            "model_count": len(result.shards),
            "shard_count": sum(len(value) for value in result.shards.values()),
            "tensor_header_count": sum(len(shard.tensor_keys) for values in result.shards.values() for shard in values),
            "decoded_tensor_value_count": 0,
            "input_checksum_manifest_sha256": result.checksum_manifest_sha256,
            "analysis_config_sha256": config.sha256,
        }
        sys.stdout.buffer.write(canonical_json_bytes(document))
        return 0
    except (AccessError, ConfigError, OSError) as exc:
        document = {
            "schema_id": "vea.p1.analysis_input_validation.v1",
            "status": "FAIL",
            "failure_class": "BLOCKED_INPUT_IDENTITY",
            "first_root_cause": str(exc).replace("\r", " ").replace("\n", " ")[:500],
            "decoded_tensor_value_count": 0,
        }
        sys.stderr.buffer.write(canonical_json_bytes(document))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
