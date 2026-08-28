# Project Status

- Status date: 2026-08-28
- Current phase: `Phase 2 - query-conditioned encoder-to-VLM adapter compatibility`
- Documentation state: synchronized by `VEA-P2-T001E-R1R`
- Source of truth: this repository and its tracked contracts

## Accepted state

- Phase 0/1 encoder-only representation analysis remains the historical
  foundation; its scientific contracts and accepted results remain valid.
- `VEA-P2-T001D` anchor transport is retained as a historical/comparison
  baseline, not the primary current adapter method.
- No direct learned Phase 2 adapter experiment has yet been accepted.
- The validated `p1_1000` feature banks and image-level 800/100/100 split are
  reusable by default.
- Source encoders, target CLIP, TinyLLaVA connector, and TinyLLaVA LLM remain
  frozen; only an adapter may be fitted or trained.
- The first adapter experiment configuration and hyperparameters remain `OPEN`.

## Next required action

Freeze the first Phase 2 adapter experiment configuration and hyperparameters
in a separately reviewed Task before implementation or execution.

## Update rule

Update this file only through an explicit task after evidence has been reviewed.
Software PASS and scientific performance remain separate judgments.
