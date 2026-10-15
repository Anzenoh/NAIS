# Project Status

- Status date: 2026-10-15
- Current phase: `Phase 2 - query-conditioned encoder-to-VLM adapter compatibility`
- Documentation state: `VEA-P2-T002F` independently accepted and closed by
  `VEA-P2-T002G`; T003H connector-aware study closure awaits independent review
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
- The first direct linear experiment freezes both source lanes, four linear
  methods, Ridge validation selection, deterministic generation, and numeric
  precision. The tested same-capacity connector-aware affine objective is
  `NOT_PROMOTED` over OLS; future low-rank, MLP, and spatial settings remain
  `OPEN`.

## Next required action

Submit T003H for independent review. A later task may design and freeze a
bounded low-rank linear Train/Validation experiment, including rank grid,
fitting/parameterization, parameter and latency budgets, Validation selection,
frozen-generation evaluation, scientific acceptance criteria, and untouched-Test
policy. No values or method are selected here, and no new Test access is
authorized. Canonical integration merge remains `UNVERIFIED`, and historical
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

## T003E/T003G Validation connector-aware study

- T003E reported `PASS_T003E_REAL_APPLE_TRAIN_VALIDATION` at implementation
  commit `97a617ab2a11a34512c591c344161b877f28b8c1`, result SHA-256
  `2147b41ec525f387c0cc8d434d6b3cd748ed224ef39053fc9c6a9e7372963a10`,
  and runtime identity
  `6e0e274a47813101e560d8f8dc1dbf40ad416b89abd7e6f6d1688646f4b466ca`.
  Test access/metrics/generation/vision were `0/0/0/0`. For SIGLIP2, OLS to
  connector-aware selected (`lambda=4`, epoch 7) changed Validation R@1
  `0.89 -> 0.92`, connector cosine
  `0.48697282446433066 -> 0.5056841280687281`, and patch cosine
  `0.4628383531145029 -> 0.449191858608662`. DINOV3 selected
  (`lambda=0.25`, epoch 0) retained the OLS/patch-only affine SHA-256
  `918872025697964e4126675a107f5db00b4efbed222c73b6c30ec40c9bf37e3f`.
- T003F implementation commit `0b3b7a2ee135e1e85cf459248066f5b53a313ef9`
  was independently reviewed; that implementation task did not run real
  generation. The user-supplied T003G Validation-only launcher report states
  `PASS_T003G_REAL_VALIDATION_GENERATION`, run
  `VEA-P2-T003G-REAL-APPLE-20261001T164130Z-631682`, with 100 Validation images,
  100 native plus 400 adapted generations, runtime/verification exits `0/0`,
  structural verification `PASS`, and Test access/metrics `0/0`.
- The reported T003G SIGLIP2 OLS Token F1/native exact match were
  `0.577915913`/`1/100`; connector-aware values were `0.569035705`/`0/100`.
  Their text/token equality was `18/100`, and F1 delta was `-0.008880208`.
  DINOV3 OLS and connector-aware each had F1 `0.594797657` and exact match
  `1/100`; text/token equality was `100/100`, consistent with the byte-identical
  affine identity. This tested same-capacity objective does not establish
  improved downstream functional compatibility over OLS.
- T003G hashes and structural `PASS` are user-supplied execution output; T003H
  did not fetch, independently rehash, regenerate, or review raw per-image files.
  T002E held-out Test and T003G Validation are different image partitions;
  their absolute metrics are not directly comparable or valid for future method
  selection.

## Update rule

Update this file only through an explicit task after evidence has been reviewed.
Software PASS and scientific performance remain separate judgments.
