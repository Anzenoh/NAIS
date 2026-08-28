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

## Open decisions

| ID | Question | Required before |
|---|---|---|
| VEA-O001 | Exact Python and dependency versions | First implementation task |
| VEA-O002 | Smoke image dataset root and immutable ten-image manifest | First real-checkpoint smoke |
| VEA-O003 | 1,000-image selection policy and manifest | Phase 1 extraction |
| VEA-O004 | Bulk CUDA host and batch size | Phase 1 extraction |
| VEA-O005 | Empirical similarity interpretation bands | Final Phase 1 analysis, after null and repeat distributions |
| VEA-O006 | Exact Phase 2 adapter experiment ID | First Phase 2 adapter configuration Task |
| VEA-O007 | Ridge lambda grid, low-rank rank grid, and MLP bottleneck size | Relevant adapter-family implementation |
| VEA-O008 | Optimizer, learning rate, epochs, and early stopping | First optimizer-trained adapter configuration |
| VEA-O009 | Exact patch loss, connector auxiliary loss, and connector-loss weight | First Phase 2 adapter configuration Task |
| VEA-O010 | Parameter budget, latency budget, and DINOv3 versus SigLIP2 execution order | First Phase 2 adapter configuration Task |
| VEA-O011 | Exact generation prompt, temperature, top_p, and max_new_tokens | First Phase 2 generation configuration |
| VEA-O012 | Semantic generation metric | Separately approved optional-metric Task |

## Change procedure

A new decision or reversal requires:

1. an explicit task permitting this file to change;
2. the old and new behavior;
3. the evidence or constraint motivating the change;
4. affected contracts, tests, and artifacts;
5. a migration or invalidation note for prior runs.
