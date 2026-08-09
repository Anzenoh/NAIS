# Run Report — VEA-P2-T001A-R3

## Status

`PASS`

## Scope

- Baseline: `07e13ce8e344a8a15ba52244d8f549bfbac1bcf1`
- Selected branch: `codex/vea-p2-t001a-r3-20260809-abi`
- Changed scope: exactly the five paths authorized by VEA-P2-T001A-R3.

## Corrected ABI behavior

The pinned `chat()` adapter now passes the image path as `str`, uses
`temperature=0` with `top_p=None`, and does not pass `do_sample`, a generation
config, a processor, or a PIL image. The model loads before its tokenizer; the
tokenizer uses `use_fast=False` plus the required pinned model-config length and
padding-side values. The model's native `chat()` owns image processing.

## Validation

| Check | Command | Result |
|---|---|---|
| Compile | `py -3.12 -m compileall src/vision_encoder_alignment/vlm_compat scripts/run_tinyllava_native_replay.py` | PASS |
| Unit tests | `py -3.12 tests/unit/test_p2_tinyllava_native_replay.py` | 17 tests PASS |
| Config-only | `py -3.12 scripts/run_tinyllava_native_replay.py --validate-config-only --config configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json` | PASS |
| CLI help | `py -3.12 scripts/run_tinyllava_native_replay.py --help` | PASS |
| Diff check | `git diff --check` | PASS |

No real model, package installation, CUDA, or Apple execution occurred.

## Remaining requirement

After independent review of the pushed R3 commit, run the corrected existing
Apple one-image runbook immediately. TinyLLaVA runtime ABI and CUDA replay remain
unverified until that user-run validation.
