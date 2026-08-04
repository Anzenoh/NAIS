#!/usr/bin/env python3
"""Compute explicit synthetic/local matrix similarity; not a P1 shard reader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_encoder_alignment.metrics import linear_cka, rsa_cosine_spearman, ucentered_cka  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-npy", required=True)
    parser.add_argument("--target-npy", required=True)
    args = parser.parse_args(argv)
    source = np.load(args.source_npy, allow_pickle=False)
    target = np.load(args.target_npy, allow_pickle=False)
    result = {
        "schema_id": "vea.metric.layer_similarity.v1",
        "linear_cka": linear_cka(source, target),
        "ucentered_cka": ucentered_cka(source, target),
        "rsa_cosine_spearman": rsa_cosine_spearman(source, target),
    }
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
