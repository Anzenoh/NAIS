# VEA-P2-T001D-R3 — Run DINO orchestration in the main lane

## Objective

Run Stage 01 JPEG decoding and orchestration in the approved main Python while
retaining the approved isolated DINO runtime for every model forward.

## Recovery contract

- Every real stage-worker command uses `MAIN_PY` from the immutable validated
  runtime binding; fake/CPU plans retain the current test interpreter.
- Stage 01 strips the main overlay and main-runtime entries from inherited
  `PYTHONPATH`, then checks its interpreter, `PIL`, `numpy`, and both DINO roots
  before any unit begins.
- The existing extractor continues to invoke the isolated DINO boundary once
  per pending 16-image chunk.
- The retained first chunk is atomically published and becomes the smoke; a
  resumed attempt validates and skips it without another forward or rewrite.

## Preserved scope

No encoder, extractor boundary, model/runtime profile, formula, data selection,
chunking, unit inventory, storage, artifact schema, metric, generation, or
checkpoint behavior is changed.
