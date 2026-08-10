# VEA-P2-T001B-R3 Apple worker response identity report

- Status: `PASS` (offline Tier 0–1 only)
- Baseline: `6fb258299a08244d1922559e798a50a1e4fee5d0`
- Branch: `codex/vea-p2-t001b-r3-apple-worker-response`
- Worker SHA-256: `152a75c7f26ba1d56c0c3e829442109b687f4ee2e09157773e4c4279a544f3f8`
- Apple runtime-profile SHA-256: `51af286a73d987ba34f8c5f54b96b91083191f7e51d62d6b02b7503459cd717f`
- Apple extractor-profile SHA-256: `771fba9fa683065d628109856385405e14d87cdaea38c5b448d0aba8fb6508bc`
- Legacy extractor-profile SHA-256: `4ade57fb0653a01ca769f726fc8612e0195f27bf36cd4dd504fe44feb5c809a7`

## Validation

- `py -3.12 -m py_compile scripts/dinov3_extraction_worker.py src/vision_encoder_alignment/p1_feature_extraction.py` — exit 0
- Assigned seven-file pytest command — exit 0; 216 passed
- `git diff --check` — exit 0
- Standard-library live identity and allowlist audit — PASS

No worker subprocess, Apple, CUDA, model, dataset, network acquisition,
snapshot materialization, raw-injection smoke, package installation, or
calibration execution occurred.
