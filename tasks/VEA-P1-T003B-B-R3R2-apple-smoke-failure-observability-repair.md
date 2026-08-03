# VEA-P1-T003B-B-R3R2 — Apple smoke failure-observability repair

## Scope

From baseline `41553f9e00f51626e5c2632378b3592e379a7775`, repair only the
runbook's post-smoke failure reporting and its static contract test.  No Apple,
GPU, dataset, checkpoint, model, Stage A, smoke, resume, cleanup, or extraction
operation is authorized.

## Required boundary

One `ERR`-trap failure handler remains armed from Stage B smoke launch through
successful creation of `15-smoke-evidence-manifest.txt`.  It records exactly one
`final_status=failed`, the original nonzero exit code, and the current bounded
report/evidence path to authoritative fd 4, without retry or mutation.

`extraction_status=complete` remains intermediate.  The handler is disarmed
only immediately before the sole final `final_status=PASS` / `exit_code=0`.

## Delivery

Only the four R3R2 allowlisted paths may change.  Required offline validation
is the assigned Tier 0 compile command and Tier 1 focused pytest command.
Commit message: `fix(extraction): VEA-P1-T003B-B-R3R2 close post-smoke failure reporting`.

`READY_FOR_P1_EXTRACTION=false`.
