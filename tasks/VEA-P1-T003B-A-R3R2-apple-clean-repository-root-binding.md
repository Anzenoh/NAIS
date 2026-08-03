# VEA-P1-T003B-A-R3R2 — Parameterize the clean Apple repository root

## Scope

This bounded implementation replaces the Stage A runbook's fixed Apple
repository assignment with a required user-supplied canonical clean repository
root. It preserves the accepted R3R1 commit, cleanliness, path-validation,
binding, evidence, and Stage A/B separation contracts.

## Required change

- Require `VEA_REPO_ROOT` with no default or fallback.
- Assign exactly `REPO_ROOT="$VEA_REPO_ROOT"` in the initial setup block.
- Continue to validate that root as an existing non-link directory before
  evidence creation and before approved-commit and Git-cleanliness operations.

## Boundaries

- Baseline: `464ebf3fdeaf53a16d04838fc67aed725fe5d08c`
- Delivery branch: `codex/vea-p1-t003b-a-r3r2`
- Delivery contains only the runbook, its nine-test contract file, and these two
  compact records.
- Apple, GPU, dataset, checkpoint, model, Stage A, and Stage B access remain
  prohibited. `READY_FOR_P1_EXTRACTION=false`.
