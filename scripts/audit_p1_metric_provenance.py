#!/usr/bin/env python3
"""Bounded provenance audit for the exact metric and CLI paths named by T004A."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PATHS = (
    "src/vision_encoder_alignment/metrics/cka.py",
    "src/vision_encoder_alignment/metrics/procrustes.py",
    "src/vision_encoder_alignment/metrics/linear_alignment.py",
    "src/vision_encoder_alignment/metrics/patch_correspondence.py",
    "scripts/compute_layer_similarity.py",
    "scripts/fit_pairwise_mapping.py",
    "scripts/evaluate_patch_alignment.py",
)


def _git(repository: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--baseline-checkout", required=True)
    parser.add_argument("--dirty-source-checkout", required=True)
    parser.add_argument("--baseline-commit", default="27688e67b0c639a81d07c31b7992dc8a09341d8d")
    args = parser.parse_args(argv)
    baseline = Path(args.baseline_checkout).resolve()
    dirty = Path(args.dirty_source_checkout).resolve()
    rows = []
    for path in PATHS:
        baseline_bytes = _git(baseline, "show", f"{args.baseline_commit}:{path}")
        dirty_path = dirty / path
        dirty_bytes = dirty_path.read_bytes()
        history = _git(baseline, "log", "--all", "--format=%H", "--", path).decode().splitlines()
        rows.append({
            "path": path,
            "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            "baseline_relation": "exact accepted extraction baseline explicit stub",
            "dirty_source_sha256": hashlib.sha256(dirty_bytes).hexdigest(),
            "dirty_source_relation": "uncommitted source-checkout candidate",
            "history_commits": history,
            "schema_compatibility": "metric core only; no vea.p1.feature_shard.v0 reader",
            "test_coverage": "direct dirty-source Phase 0 unit tests only for CKA/Procrustes/ridge; none establishes the T004A campaign boundary",
            "disposition": "REUSE_AFTER_REVIEW" if path.endswith(("cka.py", "procrustes.py", "linear_alignment.py")) else "REIMPLEMENT",
        })
    document = {
        "schema_id": "vea.p1.metric_provenance_audit.v1",
        "task_id": "VEA-P1-T004A",
        "baseline_commit": args.baseline_commit,
        "baseline_tree": _git(baseline, "rev-parse", f"{args.baseline_commit}^{{tree}}").decode().strip(),
        "target_schema_id": "vea.p1.feature_shard.v0",
        "legacy_schema_id": "vea.feature_artifact.v0",
        "bounded_paths": list(PATHS),
        "entries": rows,
        "conclusion": "The accepted baseline contains stubs. Reviewed dirty-source numerical kernels are references only until adapted; no prior reader or CLI satisfies the 18-key blind campaign contract.",
    }
    sys.stdout.write(json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
