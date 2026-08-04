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
| VEA-D011 | Freeze a canonical T004--T007 train/validation/test campaign with exactly one logical test unseal | Prevents test-driven layer, rank, endpoint, or conclusion changes | Accepted |
| VEA-D012 | Authorize only the declared linear canonical extension; exclude MLP, cross-attention, projectors, and language models | Keeps Phase 1 encoder-only and makes the hypothesis boundary auditable | Accepted |
| VEA-D013 | Record Windows-local repeat unavailability separately from checksum preservation | Prevents integrity checks from being misrepresented as repeat evidence | Accepted |
| VEA-D014 | Seal results with a non-self-referential internal manifest and external archive checksum | Makes scientific payload bytes deterministic and independently verifiable | Accepted |

## Open decisions

| ID | Question | Required before |
|---|---|---|
| VEA-O001 | Exact Python and dependency versions | First implementation task |
| VEA-O002 | Smoke image dataset root and immutable ten-image manifest | First real-checkpoint smoke |
| VEA-O003 | 1,000-image selection policy and manifest | Phase 1 extraction |
| VEA-O004 | Bulk CUDA host and batch size | Phase 1 extraction |
| VEA-O005 | Empirical similarity interpretation bands | Remains open; use null calibration and confidence intervals rather than invented universal bands |

## Change procedure

A new decision or reversal requires:

1. an explicit task permitting this file to change;
2. the old and new behavior;
3. the evidence or constraint motivating the change;
4. affected contracts, tests, and artifacts;
5. a migration or invalidation note for prior runs.
