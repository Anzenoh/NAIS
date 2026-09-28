# Project Status

- Status date: 2026-09-28
- Current phase: `Phase 2 - query-conditioned encoder-to-VLM adapter compatibility`
- Documentation state: `VEA-P2-T002F` independently accepted; state closed by
  `VEA-P2-T002G`
- First linear experiment configuration: frozen by `VEA-P2-T002A` at
  `configs/vlm_compat/p2_query_adapter_linear_v1.json`
- Source of truth: this repository and its tracked contracts

## Accepted state

- Phase 0/1 encoder-only representation analysis remains the historical
  foundation; its scientific contracts and accepted results remain valid.
- `VEA-P2-T001D` anchor transport is retained as a historical/comparison
  baseline, not the primary current adapter method.
- `VEA-P2-T002B` implemented and synthetic/unit validated the pure NumPy direct
  linear-adapter numerical core. `VEA-P2-T002C` implemented the read-only,
  memory-bounded T001D feature-bank binding preflight. These are historical
  implementation gates, not the current Test result.
- Source encoders, target CLIP, TinyLLaVA connector, and TinyLLaVA LLM remain
  frozen; only an adapter may be fitted or trained.
- The first direct linear experiment now freezes both source lanes, four linear
  methods, Ridge validation selection, deterministic generation, and numeric
  precision. Future low-rank, MLP, and spatial settings remain `OPEN`.

## Next required action

Design and approve the next bounded Phase 2 experiment using Train/Validation
without Test access. This state closure does not select or start that
experiment. Canonical integration merge remains `UNVERIFIED`, and historical
Codex test counts remain `UNRECOVERED`.

## T002E real-Test result recorded by T002F

- T002E commit `39fbbccd654aa6d0726802b5aaa1377a86f81f6b` has the
  verified parent `ccde89c3c89c410c69b3492a6ed53d360d161de9` and a
  six-path implementation diff. Its historical remote branch resolves to that
  commit.
- The hash-verified T002E result and generation records report 100 Test images,
  100 native and 800 adapted successful generations. Each image has all nine
  configurations; adapted rows agree with the result JSON.
- The T002F record of this software/runtime execution boundary received an
  independent `PASS`, recorded by T002G. Scientifically, Full Linear
  substantially improves on Calibration and Orthogonal Procrustes across the
  reported alignment and generation measures; selected Ridge is practically
  tied with Full Linear.
  Linear alignment recovers substantial but incomplete TinyLLaVA functional
  compatibility. Exact matches to native generations remain low.
- Original Apple artifact paths are not recovered here. Current validator
  evidence paths are runtime provenance, not replacements for those paths.

## Update rule

Update this file only through an explicit task after evidence has been reviewed.
Software PASS and scientific performance remain separate judgments.
