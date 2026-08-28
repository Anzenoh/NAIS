# Project Status

- Status date: 2026-08-29
- Current phase: `Phase 2 - query-conditioned encoder-to-VLM adapter compatibility`
- Documentation state: synchronized by `VEA-P2-T001E-R1R`
- First linear experiment configuration: frozen by `VEA-P2-T002A` at
  `configs/vlm_compat/p2_query_adapter_linear_v1.json`
- Source of truth: this repository and its tracked contracts

## Accepted state

- Phase 0/1 encoder-only representation analysis remains the historical
  foundation; its scientific contracts and accepted results remain valid.
- `VEA-P2-T001D` anchor transport is retained as a historical/comparison
  baseline, not the primary current adapter method.
- No direct Phase 2 adapter implementation or scientific experiment has yet run.
- The validated `p1_1000` feature banks and image-level 800/100/100 split are
  reusable by default.
- Source encoders, target CLIP, TinyLLaVA connector, and TinyLLaVA LLM remain
  frozen; only an adapter may be fitted or trained.
- The first direct linear experiment now freezes both source lanes, four linear
  methods, Ridge validation selection, deterministic generation, and numeric
  precision. Future low-rank, MLP, and spatial settings remain `OPEN`.

## Next required action

Implement the frozen VEA Phase 2 linear adapter configuration with synthetic/unit validation only; do not start the real feature-bank, connector, or LLM run in the implementation Task.

## Update rule

Update this file only through an explicit task after evidence has been reviewed.
Software PASS and scientific performance remain separate judgments.
