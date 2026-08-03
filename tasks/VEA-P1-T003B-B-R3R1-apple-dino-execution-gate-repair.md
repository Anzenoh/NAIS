# VEA-P1-T003B-B-R3R1 — Apple DINO execution-gate repair

## Scope

Repair the bounded Apple DINO execution gates from baseline
`b40e74689f063e5cdaacc6005c9d12fe81c02078` without executing Apple, GPU,
dataset, checkpoint, model, Stage A, smoke, or extraction operations.

## Required outcomes

- Pin the Apple extractor profile to canonical snapshot identity
  `d1326e9358930d826eaae6c18b44e384fa38b854140304951210347d7ecdac0d`.
- Require Stage A to compare the validated snapshot identity to the extractor
  profile before it can become smoke-ready.
- Use only `dinov3_isolated/bin/python` in all live Apple DINO execution
  contracts; retain main-lane `bin/python3`.
- Preserve the worker unless the real Apple-selector success regression proves
  a production defect.
- Keep extraction readiness false and preserve all no-overwrite, offline,
  SHA-256, containment, and process-isolation boundaries.

## Delivery boundary

Only the R3R1 allowlisted implementation, test, task, and run-report paths may
change.  Validation is the assigned offline Tier 0–1 command set only.  Git
delivery is one commit on `codex/vea-p1-t003b-b-r3r1` with message
`fix(extraction): VEA-P1-T003B-B-R3R1 close Apple DINO execution gates`.

`READY_FOR_P1_EXTRACTION=false`.
