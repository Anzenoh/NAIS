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
- `VEA-P2-T002B` implemented and synthetic/unit validated the pure NumPy direct
  linear-adapter numerical core. No real adapter fitting or Test evaluation has run.
- `VEA-P2-T002C` implemented the read-only, memory-bounded T001D feature-bank
  binding preflight. Real feature-bank identity remains unaccepted until the
  user runs it against the actual completed T001D run root.
- Source encoders, target CLIP, TinyLLaVA connector, and TinyLLaVA LLM remain
  frozen; only an adapter may be fitted or trained.
- The first direct linear experiment now freezes both source lanes, four linear
  methods, Ridge validation selection, deterministic generation, and numeric
  precision. Future low-rank, MLP, and spatial settings remain `OPEN`.

## Next required action

Run the VEA-P2-T002C feature-bank preflight against the completed real T001D
run root on the Apple host and submit the bounded JSON result for review.

TinyLLaVA connector/LLM runtime integration remains deferred.

## Update rule

Update this file only through an explicit task after evidence has been reviewed.
Software PASS and scientific performance remain separate judgments.
