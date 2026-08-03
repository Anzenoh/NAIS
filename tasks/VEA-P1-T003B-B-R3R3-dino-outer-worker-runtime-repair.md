# VEA-P1-T003B-B-R3R3 — Separate DINO orchestration and model runtimes

## Mode

`implementation`

## Baseline and delivery

- Baseline commit: `e324566e98266089552dfff551f62a5c05d9c6ff`
- Baseline tree: `05ccfb6d70e4487dbacb8d37f2d16b6a6a632a9e`
- Predecessor ref: `origin/codex/vea-p1-t003b-b-r3r2`
- Task branch: `codex/vea-p1-t003b-b-r3r3`
- Commit message: `fix(extraction): VEA-P1-T003B-B-R3R3 separate DINO orchestration and model runtimes`

## Goal

Run all three outer extraction workers and shared Pillow/NumPy preprocessing with
the exact main runtime. Keep the actual DINO model-loading and forward child in
the exact isolated DINO runtime. Make the live worker argv consume the same
authoritative prefixes emitted by `lane_command_plan()`.

## Required behavior

1. Outer CLIP, SigLIP2, and DINOv3 command prefixes use main Python, preserve
   model order and `-B`, omit `-I`, and own exactly one `--model` pair.
2. `_run_or_resume()` consumes those prefixes without duplicating `--model`.
3. Both runtime interpreters and roots are independently validated before launch.
4. The DINO extractor still derives `runtime_root` from the isolated interpreter;
   its child invocation and sanitized offline boundary remain unchanged.
5. The contract and runbook distinguish orchestration/preprocessing from the
   model-bearing DINO child and preserve every existing stop line.
6. The extraction profile and both runtime profiles remain byte-unchanged.
7. `READY_FOR_P1_EXTRACTION=false` remains the only readiness state.

## Validation boundary

- Compile only changed Python implementation and tests.
- Run the eight assigned focused test files.
- Run the complete repository suite only after the focused gate passes.
- Run `git -c core.whitespace=cr-at-eol diff --check` and exact scope checks.
- Every required check must exit `0`; otherwise stop without commit or push.

## Prohibited operations

No Apple, GPU, model, dataset, checkpoint, network, install, smoke, resume,
cleanup, extraction, profile mutation, merge, amend, rebase, or force-push.

## Status

Execution is governed by the active control task. A validation failure is
reported without weakening a protected identity fixture or expanding scope.
