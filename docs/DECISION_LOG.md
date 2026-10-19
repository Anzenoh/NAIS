# Decision Log

## Accepted decisions

| ID | Decision | Rationale | Status |
|---|---|---|---|
| VEA-D001 | Run encoder-only analysis before any LLM integration | Isolates representation differences from projector and language-model adaptation | Accepted |
| VEA-D002 | Freeze all encoder weights in Phase 0 and Phase 1 | Prevents training from masking native representation relationships | Accepted |
| VEA-D003 | Share image geometry but keep model-specific numeric normalization | Preserves spatial correspondence without forcing an out-of-distribution normalization | Accepted |
| VEA-D004 | Start with CLIP ViT-B/16, SigLIP2 Base Patch16-224, and DINOv3 ViT-B/16 | Controls patch geometry and approximate scale while varying training objectives | Accepted |
| VEA-D005 | Scan global features across all comparable layers before storing selected patch layers | Reduces storage and identifies informative layer pairs first | Accepted |
| VEA-D006 | Use CPU-capable smoke validation and an explicit NVIDIA CUDA environment for bulk extraction | Avoids making unsupported local AMD GPU paths a prerequisite | Accepted |
| VEA-D007 | Separate software acceptance from scientific hypothesis confirmation | Null or surprising findings remain valid research outcomes | Accepted |
| VEA-D008 | Pin exact checkpoint revisions before full runs | Prevents silent model drift | Accepted |
| VEA-D009 | Use safe tensor shards with JSON manifests as the canonical generated feature format | Provides portable, non-pickle tensor storage and explicit image ordering | Accepted |
| VEA-D010 | Keep tested logic in a Python package; notebooks are visualization clients only | Improves reproducibility and testability | Accepted |
| VEA-D011 | Permit the isolated `vea.p2.vlm_compatibility_pilot.v0` frozen native replay boundary | Tests serialization and connector-boundary software compatibility without changing Phase 0/1 conclusions or training any parameter | Accepted |
| VEA-D012 | Make the same-Query query-conditioned adapter the primary Phase 2 research lane | Directly tests whether `Adapter(E_source(x))` can restore the target feature and downstream behavior for the current Query | Accepted |
| VEA-D013 | Retain T001D anchor transport as historical/comparison evidence only | It selects or combines other-image target features and is not a direct transformation of the current Query source feature | Accepted |
| VEA-D014 | Permit fitting or training only the adapter while source encoders, target CLIP, TinyLLaVA connector, and TinyLLaVA LLM remain frozen | Isolates the minimum transformation required without adapting the target VLM | Accepted |
| VEA-D015 | Evaluate every finalized adapter at patch, connector, same-image identity, and actual generation levels | Tensor similarity alone did not establish functional compatibility; validation-rejected candidates need not receive Test generation | Accepted |
| VEA-D016 | Reuse the immutable `p1_1000` 800/100/100 image-level split and validated feature banks by default | Preserves pairing and prevents unnecessary extraction or patch-level leakage | Accepted |
| VEA-D017 | Freeze `vea.p2.query_adapter_linear.v1` with source order SIGLIP2 then DINOV3 and independent per-source adapters | Establishes the first implementation-ready same-Query direct adapter experiment without a shared adapter | Accepted |
| VEA-D018 | Compare CALIBRATED, ORTHOGONAL_PROCRUSTES, FULL_LINEAR, and RIDGE in the first direct experiment | Covers the first four approved linear-complexity baselines while excluding low-rank and nonlinear methods | Accepted |
| VEA-D019 | Search normalized Ridge lambdas `[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]` independently per source | Makes regularization scale explicit through `alpha=lambda_normalized*N_train_patch_rows` | Accepted |
| VEA-D020 | Select Ridge on Validation by R@1, then R@5, connector cosine, then smallest normalized lambda | Provides a deterministic functional-first hierarchy with no Test-informed selection | Accepted |
| VEA-D021 | Reuse the exact T001D deterministic generation settings and skip Validation generation during Ridge search | Makes all eight finalized Test configurations comparable without generation for every lambda candidate | Accepted |
| VEA-D022 | Use float64 fitting/solves and similarity metrics with float32 stored adapter parameters and adapted features | Preserves numerical stability while retaining the validated injection storage boundary | Accepted |

## Study disposition

| ID | Decision | Rationale | Status |
|---|---|---|---|
| VEA-D023 | Do not promote the tested same-capacity connector-aware affine objective over OLS; plan a separately approved bounded low-rank linear Train/Validation design only | T003E SIGLIP2 Validation connector/R@1 proxies improved, but user-supplied T003G Validation generation reports lower Token F1 and native exact match; DINOV3 selected unchanged OLS initialization and identical generation. No Test-informed selection or new Test access follows | NOT_PROMOTED |
| VEA-D024 | Record `NO_RANK_ACCEPTED` for both sources under the frozen T004A six-rank Validation R@1 gate; retain the accepted OLS controls | User-supplied T004H rounded Validation R@1: SIGLIP2 OLS `0.89`, best rank 256 `0.86` below minimum `0.87`; DINOV3 OLS `0.87`, best rank 256 `0.84` below minimum `0.85`. All other frozen ranks are lower. The necessary criterion fails before conjunctive qualification; connector, generation and latency remain `NOT_EVALUATED`. No threshold/config change, Test-informed selection, or claim about frozen-VLM generation follows | NO_RANK_ACCEPTED; REVIEW_PENDING |

## Open decisions

| ID | Question | Required before |
|---|---|---|
| VEA-O001 | Exact Python and dependency versions | First implementation task |
| VEA-O002 | Smoke image dataset root and immutable ten-image manifest | First real-checkpoint smoke |
| VEA-O003 | 1,000-image selection policy and manifest | Phase 1 extraction |
| VEA-O004 | Bulk CUDA host and batch size | Phase 1 extraction |
| VEA-O005 | Empirical similarity interpretation bands | Final Phase 1 analysis, after null and repeat distributions |
| VEA-O007 | MLP bottleneck size; the T004A study's low-rank rank grid is frozen | Future nonlinear-adapter configuration |
| VEA-O008 | MLP optimizer, learning rate, epochs, and early stopping | Future optimizer-trained adapter configuration |
| VEA-O009 | MLP patch/connector multi-objective loss and connector-loss weight | Future nonlinear adapter configuration |
| VEA-O010 | Hard parameter and latency budgets for future studies; T004A's low-rank study budgets are frozen | Future complexity comparison |
| VEA-O012 | Semantic generation metric | Separately approved optional-metric Task |

Resolved by `VEA-P2-T002A`: VEA-O006 and VEA-O011. The Ridge portion of
VEA-O007 and the source-order portion of VEA-O010 are also resolved for the
first linear experiment by VEA-D017 through VEA-D022.
T004A separately resolved its six-rank grid, parameter ceiling and latency
criterion without resolving MLP or broader future-study settings. VEA-D024
records the necessary R@1 gate result; it does not replace VEA-D023 or revise
the T004A configuration. Earlier runs and T004G `UNSELECTED` output are not
invalidated or rewritten.

## Change procedure

A new decision or reversal requires:

1. an explicit task permitting this file to change;
2. the old and new behavior;
3. the evidence or constraint motivating the change;
4. affected contracts, tests, and artifacts;
5. a migration or invalidation note for prior runs.
