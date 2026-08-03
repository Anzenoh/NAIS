# VEA-P1-T003B-B-R3R2 — Apple smoke failure-observability repair

## Status

`PASS` for the offline runbook/test repair.  This does not authorize Apple,
Stage A, smoke, resume, cleanup, or extraction.

## Independent-review finding and repair

R3R1 reported smoke-execution failure, but a later smoke-validator, required
artifact, hashing, review-tar, readiness, or final-manifest failure could exit
without the required authoritative fd 4 failure record.  R3R2 adds one guarded
`ERR` trap across both Stage B and final evidence.  It captures `$?` before any
other command, prevents recursion, preserves the original nonzero exit, and
uses the current bounded report/evidence path.  The trap is disarmed only after
successful `15-smoke-evidence-manifest.txt` creation.

The static regression examines the actual two Bash blocks and checks handler
placement, report-path progression, artifact-check exit preservation, one
failure record, and final disarm/PASS order.  It fails on the R3R1 baseline.

## Scope and preservation

- Baseline: `41553f9e00f51626e5c2632378b3592e379a7775`.
- Branch: `codex/vea-p1-t003b-b-r3r2`.
- Changed paths: this runbook, its unit test, this Task, and this Run Report.
- Production source, profiles, identities, contracts, and worker tests: unchanged.
- Original dirty repository and all prior worktrees were preserved.

## Validation

| Command | Exit | Result |
| --- | ---: | --- |
| `py -3 -B -m compileall -q scripts/extract_features.py scripts/dinov3_extraction_worker.py src/vision_encoder_alignment/p1_feature_extraction.py tests/unit/test_p1_t003b_stage_a_runbook.py` | 0 | pass |
| Assigned 14-file Tier 1 pytest command | 0 | 441 passed, 0 skipped |
| `git -c core.whitespace=cr-at-eol diff --check` | 0 | pass |

## Delivery and readiness

- Commit message: `fix(extraction): VEA-P1-T003B-B-R3R2 close post-smoke failure reporting`.
- Push disposition is recorded in the final Git delivery response.
- Apple/GPU/model/dataset/checkpoint/network/install/download operations: `0`.
- `READY_FOR_P1_EXTRACTION=false`.
