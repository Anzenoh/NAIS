# VEA-P2-T001B-R2 coherent snapshot materialization report

- Status: `PASS` (offline Tier 0–1 only)
- Baseline: `1f6135d5c0542368bf294071454b244a746c80d6`
- Branch: `codex/vea-p2-t001b-r2-coherent-snapshot`
- Manifest: `1129` canonical bytes; SHA-256 `ec13f3f45b2b0976319122d123f3bacdd583ed13c68f29eec6a79462e4e7d6ec`
- Final Apple extractor profile SHA-256: `b0d8c3fee92a783aa873efedcadab68c48bcde8aac458d767bb5fa77cac1cb8d`
- P1 JSON and Python bindings: exact match to the final Apple profile SHA

## Validation

- `py -3.12 -m py_compile scripts/materialize_dinov3_snapshot_from_hf_cache.py` — exit 0
- `py -3.12 -m pytest -q tests/unit/test_p2_dinov3_snapshot_materialization.py tests/unit/test_p2_dinov3_raw_injection.py tests/unit/test_p1_feature_extraction.py` — exit 0; 96 passed
- `py -3.12 scripts/materialize_dinov3_snapshot_from_hf_cache.py --help` — exit 0
- `py -3.12 scripts/run_dinov3_raw_injection.py --help` — exit 0
- `git diff --check` — exit 0
- Bounded live-binding audit — old Apple profile SHA absent from `configs`/`src`; both new bindings exact

No real snapshot bytes, model, dataset, Apple, GPU, CUDA, credential, package,
download, calibration, or benchmark execution occurred. The historical P0 run
report named by the task was absent at the exact baseline; the named acquisition
implementation and protected snapshot boundary supplied the required patterns.
