# VEA-P1-T004A -- Sealed Phase 1 analysis campaign

Status: implementation and deterministic offline validation assigned.

## Objective

Implement the frozen encoder-only T004--T007 campaign over the accepted T003B feature artifacts without modifying or re-extracting them. Preserve image pairing, split by image ID, keep every encoder frozen, and compare equivalent recorded representation stages.

## Fixed identities

- Baseline commit: `27688e67b0c639a81d07c31b7992dc8a09341d8d`
- Baseline tree: `b6fb3ef7fc99e0a4f16c351867f2da1fea100760`
- Input run: `20260803T192848Z_p1_1000_three_encoder_27688e67b0c6`
- Input checksum manifest SHA-256: `44e531e242e434d7f9aa5c901da694bf29088cf2a7cae0fad3d6a87ea2a1386d`
- Image manifest SHA-256: `1819bbc6235dd4805d71642e63e73112ce2e918135ec94f46493077698be92da`
- Split SHA-256: `6c299a79b22b99b4c68c3de471b9d5c7c756b6bf4a0f78b831a97e30a0a10c5f`

## Required behavior

The canonical config and analysis contract freeze input validation, train/validation discovery, exact-node mapping, patch states and controls, authorized linear extension, one logical test unseal, conclusion rules, deterministic reports, payload sealing, external delivery identity, and ending 252-file checksum equality.

The Windows-local lane records the same-setting repeat exactly as `NOT_EXECUTED_LOCAL_RESOURCE_UNAVAILABLE`. It separately reports checksum integrity and sets `checksum_integrity_is_repeat=false`.

## Acceptance

All assigned offline tests, complete repository tests, syntax checks, diff checks, bounded input header/checksum validation, commit allowlist checks, exact push verification, clean exact-commit campaign execution, package verification, and ending input checksums must pass. Scientific null or negative results are accepted outcomes.
