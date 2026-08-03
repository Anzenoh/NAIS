# VEA-P1-T003B-B-R2 — Align Apple DINO Runtime and Snapshot Contracts

## Mode

`implementation`

## Baseline and delivery

- baseline: `8fbc4c16c9a8a3802d1fdc1e8fb7b64738482980`
- branch: `codex/vea-p1-t003b-b-r2`
- commit message: `fix(extraction): VEA-P1-T003B-B-R2 align Apple DINO runtime and snapshot contracts`

## Contract

Repair only the bounded Apple DINO Phase 1 boundary. Apple uses the exact
`<runtime_root>/bin/python` layout and the pinned
`huggingface_cache_snapshot_v1` model/revision cache layout. The shared snapshot
validator must check the exact three pinned files, containment, links, sizes, and
SHA-256 identities without network, writes, credentials, package operations, or
model loading. Preserve the legacy Windows runtime and external snapshot contract.

Stage A must validate the exact Stage B runtime/snapshot path before smoke and
report all prohibited-operation counts as zero. Repository evidence must bind an
explicitly approved actual commit; the historical production-source baseline must
remain separately named. Execution progress is bounded stderr-only metadata, and
canonical stdout JSON remains authoritative.

The runbook must require a fresh Stage A evidence root and a fresh smoke run after
independent acceptance. It must never suggest retry, resume, output reuse, full
extraction, cleanup, or access to Apple during this implementation Task.

## Validation

Run only the Tier 0 compile command and the 13 explicitly assigned Tier 1 test
files in `.codex/ACTIVE_TASK.md`, followed by whitespace and scoped Git checks.
Keep `READY_FOR_P1_EXTRACTION=false` and do not begin another task.
