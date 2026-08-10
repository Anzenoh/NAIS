# VEA-P2-T001C-R1 implementation and offline-validation record

## Status

`PASS_OFFLINE_IMPLEMENTATION` from exact baseline
`d571f7704364aa38a09ecd49e0cf2a674dd711d4` on branch
`codex/vea-p2-t001c-r1-direct-calibration-capture`.

This is fake/CPU Tier 0–1 evidence only. Apple, CUDA, real models, datasets,
packages, external snapshots, materialization, and the scientific experiment
were not accessed or executed.

## Recovery implemented

- Replaced eight calibration chat/generation captures with the pinned model
  runtime module's `load_image` and `process_images`, followed by one direct
  `model.encode_images` call per calibration image.
- Direct capture uses `vision_tower._image_processor` and the loaded model config,
  validates exact native shapes/dtype/finite values and one vision/connector call,
  proves exact equality between the returned value and connector-hook output,
  removes all hooks in `finally`, and retains owned CPU feature copies only.
- Calibration accounting is eight feature calls and zero language generations.
  The four held-out conditions and exactly 16 test generations are unchanged.
- Added bounded contexts for input validation, DINO extraction, TinyLLaVA load,
  calibration capture, alignment fit, held-out generation, and artifact publication.
- Failure JSON records nullable image/path/condition plus a maximum eight-entry
  chained root cause. The CLI prints the full traceback first, without storing it
  in bounded JSON.
- Updated the Apple procedure to one child bash for preflight and one corrected
  pilot, with no new tmux and mandatory parent-alive terminal lines.

## Assigned Tier 0–1 validation

```text
py -3.12 -m py_compile src/vision_encoder_alignment/vlm_compat/training_free_alignment_pilot.py scripts/run_training_free_alignment_pilot.py
exit 0

py -3.12 -m pytest -q tests/unit/test_p2_training_free_alignment_pilot.py tests/unit/test_p2_dinov3_raw_injection.py
42 passed in 1.05s
exit 0

py -3.12 scripts/run_training_free_alignment_pilot.py --help
exit 0

git diff --check
exit 0
```

Fake accounting proves DINO batches `1`, TinyLLaVA loads `1`, calibration native
feature calls `8`, calibration generation calls `0`, and held-out generation
calls `16`. The chained failure test preserves stage, image, basename, condition,
top-level error, and the synthetic connector/OOM root chain while excluding a
traceback from JSON.

Next action after independent review: execute the corrected Apple runbook exactly
once. Do not rerun the failed baseline or precede it with a diagnostic model run.
