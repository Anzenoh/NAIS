# VEA-P2-T001C-R2 implementation and offline-validation record

## Status

`PASS_OFFLINE_IMPLEMENTATION` from exact baseline
`dfb00a3ef62bdf3f4bf806063db685859bf100a8` on branch
`codex/vea-p2-t001c-r2-preserve-injection-batch-axis`.

This record contains fake/CPU Tier 0–1 evidence only. Apple, CUDA, real models,
datasets, packages, external snapshots, materialization, and the scientific
experiment were not accessed or executed. `.codex/PROJECT_STATE.json` was absent
at the accepted baseline and was not created.

## Confirmed root cause fixed

R1 selected held-out DINO features with scalar indexing and therefore reduced
`[12,196,768]` to `[196,768]`. R2 range-slices source indices 8 through 11 so
each held-out image remains `[1,196,768]`. RAW uses that batch directly,
CALIBRATED applies the unchanged moment transform, and LINEAR applies the
unchanged Procrustes transform to CALIBRATED; all retain the same exact shape.

A dedicated pre-injection boundary now rejects any array unless it is exactly
`[1,196,768]`, float32, finite, C contiguous, and writable. Validation occurs
before `torch.from_numpy`, ReplayInjection, chat, or generation and never reshapes
or modifies a feature. The fake ReplayInjection connector observes 3-D input and
returns `[1,196,1536]`, matching the protected one-image raw-injection ABI.

R1 calibration capture, splits, transforms, metrics, artifacts, chained failure
evidence, and accounting remain unchanged: DINO extraction `1`, TinyLLaVA load
`1`, calibration feature calls `8`, calibration generations `0`, held-out
generations `16`.

## Apple runbook correction

The runbook targets the R2 branch and fresh R2 paths. Before the shared-GPU gate,
dependency validation uses `importlib.metadata` only and imports neither Torch
nor other runtime modules. It performs no `torch.cuda.*`, CUDA initialization,
model load, or model forward before two successful idle/memory checks. The child
bash still runs one pilot and the current tmux parent always reports its alive
status.

## Assigned Tier 0–1 validation

```text
py -3.12 -m py_compile src/vision_encoder_alignment/vlm_compat/training_free_alignment_pilot.py
exit 0

py -3.12 -m pytest -q tests/unit/test_p2_training_free_alignment_pilot.py tests/unit/test_p2_dinov3_raw_injection.py
51 passed in 1.12s
exit 0

py -3.12 scripts/run_training_free_alignment_pilot.py --help
exit 0

git diff --check
exit 0
```

Next action after independent review: execute the corrected R2 Apple runbook
exactly once. Do not rerun R1 or precede R2 with a diagnostic model execution.
