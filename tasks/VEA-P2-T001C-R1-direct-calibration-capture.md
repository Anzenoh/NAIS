# VEA-P2-T001C-R1 — Direct calibration capture and exact failure evidence

- Mode: recovery
- Baseline: `d571f7704364aa38a09ecd49e0cf2a674dd711d4`
- Branch: `codex/vea-p2-t001c-r1-direct-calibration-capture`
- Commit: `fix(vlm-compat): VEA-P2-T001C-R1 remove calibration generation dependency`

## Recovery objective

Remove all language generation from the eight native calibration captures. Use
the already loaded pinned TinyLLaVA model class's runtime module and require its
callable `load_image` and `process_images`. Apply those functions to the same
local image path, `vision_tower._image_processor`, and model config, then call
the original model instance's `encode_images(image_tensor)` exactly once.

Install native vision and connector hooks around that call. Require exactly one
vision call and one connector call; validate frozen-model, dtype, finite-value,
`[1,196,768]` patch, and `[1,196,1536]` connector contracts. Prove the returned
connector tensor is exactly equal to the hook output, remove every hook in
`finally`, copy results to owned CPU arrays, and retain no calibration GPU tensor.
Calibration must not invoke chat, generation, tokenizer coding, token decoding,
or language-model forward.

## Preserved experiment

Keep eight calibration and four held-out images, one 12-image DINO extraction,
one TinyLLaVA load, exact condition order `NATIVE`, `RAW`, `CALIBRATED`,
`LINEAR`, and exactly 16 held-out generations with 32 tokens. Preserve manifest
selection, split/leakage checks, both closed-form transformations, metrics/NMSE,
parameter dtypes, frozen connector, six artifacts, result schema, and scientific
PASS semantics.

Successful accounting is exact:

```text
dinov3_batch_extractions = 1
tinyllava_model_loads = 1
calibration_native_feature_calls = 8
calibration_generation_calls = 0
test_generation_calls = 16
```

Successful metadata states that calibration uses pinned native preprocessing and
direct `encode_images`, without language generation.

## Failure evidence

Every runtime operation is bounded by one of `input_validation`,
`dinov3_extraction`, `tinyllava_load`, `calibration_native_capture`,
`alignment_fit`, `held_out_generation`, and `artifact_publication`. Image work
records the manifest image ID and basename; held-out work also records condition.

Failure JSON contains schema, blocked status, stage, nullable image/path/condition,
top-level type/message, and a cause chain of at most eight `{type,message}`
entries. It contains no traceback or tensor/environment dump. The CLI prints the
full traceback to the worker log before writing bounded JSON so the first run is
actionable without a diagnostic rerun.

## Validation and delivery

Use fake/CPU-only tests for native preprocessing, exact direct call/output,
hook cleanup, zero calibration generations, 16 held-out generations, unchanged
scientific contracts, bounded chained failures, traceback separation, and import
safety. Run only the assigned two-module pytest, compile, CLI help, and diff check.

Modify only the implementation, CLI, focused test, existing Apple runbook, this
task, and the R1 run report. Do not modify validated native/DINO paths, contracts,
profiles, dependencies, manifest, or metrics. Do not access Apple, CUDA, models,
datasets, packages, snapshots, or begin another task during Codex validation.

After PASS, commit only those paths, push the R1 branch, and stop. Independent
review is followed by exactly one corrected Apple execution in a separate child
bash while the user's current tmux parent remains alive.
