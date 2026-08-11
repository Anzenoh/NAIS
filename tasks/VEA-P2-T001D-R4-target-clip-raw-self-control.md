# VEA-P2-T001D-R4 — TARGET_CLIP RAW self-control and exact feature reuse

- Mode: `recovery`
- Baseline: `9f488cc46f1711c3f43e8ed24561198757988927`
- Branch: `codex/vea-p2-t001d-r4-target-clip-raw-self-control`
- Validation: offline fake/CPU Tier 0–1

## Objective

Replace the invalid comparison between Stage 03 persisted connector output and
a separately loaded, differently batched Stage 05 connector output. Preserve
strict integrity by checking persisted patch identity, effective connector input
identity, and duplicate live output identity in one paired connector call.

Reuse the completed R3 Stage 01–03 banks only after exact read-only inspection,
without encoder re-execution or mutation of the R3 evidence root. Stage 04 stays
recomputable and the first TARGET_CLIP/RAW Stage 05 unit remains an atomic,
resumable checkpoint.

## Frozen contracts

Encoder/model identities, manifest ordering, feature values, anchor selection,
transport and generation formulas, scientific metrics, condition ordering,
chunking, artifact/marker schemas, storage accounting, and the `2345`-unit
inventory remain unchanged. No tolerance or new aggregate metric is introduced.

## Acceptance record

- Three TARGET_CLIP/RAW controls have distinct fail-fast classifications.
- Same-call live connector validation uses exactly one paired invocation.
- Stage 03 connector difference is recorded but remains metric-only.
- Shared inspection/publication proof yields `193` completed, `189` eligible,
  and `189` reused units with zero feature-model forwards.
- Fake interruption/resume does not rewrite or repeat the first Stage 05 unit.
- Assigned validation passed: `141` tests, compilation, both CLI helps, and
  `git diff --check`.
- Apple/CUDA/model execution is deferred to independent review and one fresh R4
  run using the immutable R3 reuse root.
