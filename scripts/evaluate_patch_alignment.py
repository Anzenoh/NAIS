#!/usr/bin/env python3
"""Evaluate explicit patch arrays; the sealed campaign owns P1 shard access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_encoder_alignment.metrics import fit_position_template, linear_cka, patch_coordinate_retrieval, patch_state  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--target-train", required=True)
    parser.add_argument("--source-evaluation", required=True)
    parser.add_argument("--target-evaluation", required=True)
    parser.add_argument("--state", choices=("X", "W", "D"), required=True)
    args = parser.parse_args(argv)
    load = lambda path: np.load(path, allow_pickle=False)
    source_template = fit_position_template(load(args.source_train))
    target_template = fit_position_template(load(args.target_train))
    source = patch_state(load(args.source_evaluation), args.state, template=source_template)
    target = patch_state(load(args.target_evaluation), args.state, template=target_template)
    result = {
        "schema_id": "vea.metric.patch_alignment.v1",
        "state": args.state,
        "linear_cka": linear_cka(source.reshape(-1, source.shape[-1]), target.reshape(-1, target.shape[-1])),
        "retrieval": patch_coordinate_retrieval(source, target),
    }
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
