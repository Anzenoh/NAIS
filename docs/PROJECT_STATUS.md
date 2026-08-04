# Project Status

- Status date: 2026-08-04
- Current phase: `P1 -- sealed encoder-only analysis implementation`
- Overall status: `T004A_IMPLEMENTATION_VALIDATION`
- Last verified input: `VEA-P1-T003B` accepted 1,000-image three-encoder feature run at extraction commit `27688e67b0c639a81d07c31b7992dc8a09341d8d`
- Active task: `VEA-P1-T004A`
- Source of truth: this repository, tracked contracts, and the accepted bounded evidence roots

## Accepted baseline

- Phase 0 and Phase 1 are encoder-only; encoders remain frozen.
- Canonical image geometry is shared and numeric normalization remains model-specific.
- The fixed comparison is CLIP ViT-B/16, SigLIP2 Base Patch16-224, and DINOv3 ViT-B/16.
- Image pairing is preserved and splitting is by image ID.
- Scientific metric strength is not a software PASS condition.

## Frozen T004--T007 analysis

- The canonical configuration is `configs/analysis/p1_analysis_campaign_v1.json`.
- Normative analysis, access, inference, resume, reporting, and seal rules are in `docs/contracts/P1_ANALYSIS_CAMPAIGN_CONTRACT.md`.
- Development reads train/validation only; one logical test unseal follows atomic freeze.
- Windows-local repeat status is `NOT_EXECUTED_LOCAL_RESOURCE_UNAVAILABLE`; checksum preservation is a separate control and never a repeat.
- The only authorized extension is the declared linear canonical family.

## Open decisions

1. Empirical interpretation bands remain open until supported by frozen null and confidence distributions.
2. A future same-setting repeat requires the separately accepted extraction runtimes, checkpoints, source images, and CUDA host; no substitute is authorized.

## Next required action

Complete only `tasks/VEA-P1-T004A-phase1-analysis-campaign.md`. Do not begin another task from this status file.

## Update rule

Update this file only through an explicit task after evidence is reviewed. Do not mark a task complete based solely on narrative; use commands, test output, manifests, checksums, and exact commit evidence.
