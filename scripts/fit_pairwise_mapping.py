#!/usr/bin/env python3
"""Fit one explicit train/validation ridge mapping from non-P1 NumPy fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_encoder_alignment.metrics import alignment_metrics, select_ridge_mapping  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("source-train", "target-train", "source-validation", "target-validation"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--relative-alphas", default="1e-4,1e-3,1e-2,1e-1,1")
    args = parser.parse_args(argv)
    load = lambda path: np.load(path, allow_pickle=False)
    xtr, ytr = load(args.source_train), load(args.target_train)
    xv, yv = load(args.source_validation), load(args.target_validation)
    mapping, table = select_ridge_mapping(
        xtr, ytr, xv, yv,
        relative_alphas=tuple(float(item) for item in args.relative_alphas.split(",")),
        bootstrap_count=200,
        seed=20260804,
        source_name="source", target_name="target",
    )
    result = {
        "schema_id": "vea.metric.pairwise_mapping.v1",
        "selection": table,
        "validation": alignment_metrics(yv, mapping.transform(xv), target_train_mean=mapping.target_mean),
    }
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
