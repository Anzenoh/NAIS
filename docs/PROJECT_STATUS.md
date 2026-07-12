# Project Status

- Status date: 2026-07-12
- Current phase: `P0 — specification and repository bootstrap`
- Overall status: `NOT_STARTED`
- Last verified task: none
- Active task: `VEA-P0-T001`
- Source of truth: this repository and its tracked contracts

## Accepted baseline

- Encoder-only research comes before any LLM integration.
- Encoders remain frozen in Phase 0 and Phase 1.
- Canonical image geometry is shared; numeric normalization remains model-specific.
- The first comparison set is CLIP ViT-B/16, SigLIP2 Base Patch16-224, and DINOv3 ViT-B/16.
- Global features are scanned across all comparable layers before storing patch features at selected depths.
- A high or low scientific metric is an experimental result, not a software PASS condition.

## Open decisions

1. Exact local Python and package baseline.
2. Exact checkpoint revision hashes.
3. Local path and license status for the smoke image manifest.
4. Local path and selection procedure for the 1,000-image manifest.
5. Final bulk-extraction runtime and GPU.
6. Empirical thresholds for similarity interpretation; these must not be invented before baseline distributions exist.

## Next required action

Execute `tasks/VEA-P0-T001-repository-audit.md`. Do not start extractor implementation before the audit is reviewed.

## Update rule

Update this file only through an explicit task after evidence has been reviewed. Do not mark a task complete based solely on Codex's narrative; use commands, test output, manifests, and diffs.
