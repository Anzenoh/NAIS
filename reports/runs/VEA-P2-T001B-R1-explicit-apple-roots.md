# VEA-P2-T001B-R1 explicit Apple roots run report

- Status: `PASS` (offline Tier 0–1 only)
- Baseline: `9f1af0d6cb6f18c9e78e2e72deaa1d0da14e68ac`
- Branch: `codex/vea-p2-t001b-r1-explicit-apple-roots`
- Scope: explicit Apple DINO runtime/snapshot binding, bounded interpreter compatibility, regression tests, and corrected Apple runbook
- Previous tests: 8
- Final tests: 17

## Validation

- `py -3.12 -m py_compile src/vision_encoder_alignment/vlm_compat/dinov3_raw_injection.py scripts/run_dinov3_raw_injection.py` — exit 0
- `py -3.12 tests/unit/test_p2_dinov3_raw_injection.py` — exit 0; 17 passed
- `py -3.12 scripts/run_dinov3_raw_injection.py --help` — exit 0; both explicit DINO root arguments present
- `git diff --check` — exit 0

No Apple, GPU, model, checkpoint, CUDA, network-download, installation,
calibration, or benchmark execution was performed.
