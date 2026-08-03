# VEA-P1-T003B-A-R3R1 — Verify the Stage A runbook contract

## Scope

This validation-only delivery adds the omitted dedicated static contract test for
the committed R3 Stage A runbook.  It does not modify the runbook, production
code, profiles, manifests, or prior tests.

## Baseline and boundaries

- Baseline: `8fd93a830b203d92c1669609cdce5dc2bc673922`
- Delivery branch: `codex/vea-p1-t003b-a-r3r1`
- Required result: eight new static tests and the existing focused `87 + 66 + 36`
  tests pass in an isolated clean candidate.
- Stage A, Stage B, Apple, GPU, dataset, checkpoint, and model access remain
  prohibited. `READY_FOR_P1_EXTRACTION=false`.

## Contract categories

1. Required, lowercase, 40-character approved commit and exact `HEAD` equality.
2. Staged, unstaged, untracked, and whitespace cleanliness before evidence creation.
3. No executable fixed-branch gate or obsolete executable commit identity.
4. Extraction-profile SHA-256 binding.
5. Six exact source bindings to current Git blobs.
6. Six exact Stage A implementation bindings, including the runtime validator.
7. Exact `00` through `05` Stage A evidence contract and stop line.
8. Separate Stage B approval, approved-commit run-ID derivation, and no full mode.

## Delivery set

- `tests/unit/test_p1_t003b_stage_a_runbook.py`
- `tasks/VEA-P1-T003B-A-R3R1-stage-a-runbook-contract-validation.md`
- `reports/runs/VEA-P1-T003B-A-R3R1-stage-a-runbook-contract-validation.md`
