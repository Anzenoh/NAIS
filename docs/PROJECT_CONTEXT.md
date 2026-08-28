# Project Context

## 1. Research problem

Different VLM and MLLM systems use vision encoders trained with different objectives and sometimes different architectures. Their hidden states may differ in token count, channel dimension, layer depth, normalization, spatial organization, and semantic emphasis.

Phase 0 and Phase 1 established whether those representations share recoverable
structure. The current Phase 2 problem is query-conditioned encoder-to-VLM
compatibility:

```text
A_s(E_s(x)) -> E_t(x)
```

The adapter must directly transform the source feature of the same Query image
into the target encoder feature space. The research question is what minimum
transformation complexity restores functional compatibility with the frozen
target VLM.

## 2. Primary objective

For the same input images, compare layer-wise representations from heterogeneous vision encoders and determine:

1. which layer pairs exhibit global or local structural correspondence;
2. whether correspondence is explained by orthogonal, linear, low-rank, or nonlinear transformations;
3. whether global similarity also exists at patch and spatial levels;
4. which measurements predict cross-encoder transfer or reconstruction;
5. which information appears common versus encoder-specific.

## 3. Initial candidate encoders

The Phase 1 candidate set is:

- `openai/clip-vit-base-patch16`;
- `google/siglip2-base-patch16-224`;
- `facebook/dinov3-vitb16-pretrain-lvd1689m`.

These are candidates, not silently interchangeable aliases. Exact checkpoint revisions must be pinned before a full experiment. Access requirements and model configuration must be verified and recorded.

## 4. Current phase

### Phase 0 — Infrastructure and contract validation

- audit the repository and execution environment;
- establish a tested package and configuration structure;
- implement mock-based schemas and metric tests;
- add real model extractors one at a time;
- perform a 10-image smoke test.

### Phase 1 — Controlled representation analysis

- use a fixed 1,000-image manifest;
- extract global features from all comparable layers;
- extract patch features from selected depths;
- compute calibrated similarity and mapping metrics;
- run pairing-shuffle and synthetic controls;
- generate layer-pair and minimum-transform-complexity reports.

### Phase 2 - Query-conditioned adapter compatibility

- target TinyLLaVA native CLIP pre-connector patch features `[N,196,768]`;
- source candidates SigLIP2 `final_norm` and DINOv3 block 11 after removing
  CLS 1 plus register tokens 4, each `[N,196,768]`;
- source encoders, target CLIP, connector, and LLM frozen;
- adapter-only fitting or training;
- patch, connector, identity, and frozen-LLM generation evaluation for every
  finalized method.

## 5. Historical non-goals for Phase 0 and Phase 1

- LLM integration;
- multimodal projector comparison;
- instruction tuning;
- encoder fine-tuning;
- end-to-end MLLM training;
- multi-encoder feature fusion;
- claiming a universal representation before held-out tests exist;
- treating a high CKA value alone as proof of functional interchangeability.

These restrictions remain binding for Phase 0/1. Phase 2 does not permit model
fine-tuning; it permits only adapter fitting plus frozen downstream evaluation.

## 6. Expected research outputs

- a layer-to-layer similarity atlas for each encoder pair;
- held-out Procrustes and ridge mapping results;
- patch correspondence and spatial error measurements;
- a map of the simplest transformation that reaches each measured performance level;
- documented failure modes and encoder-specific information;
- a justified specification for a later common-space adapter.
- a Phase 2 comparison identifying the minimum direct adapter complexity that
  restores functional compatibility with the frozen target VLM.

## 7. Known execution context

- Primary development host: Windows.
- Local GPU: AMD Radeon RX 6950 XT.
- Phase 0 smoke tests should be CPU-capable.
- Bulk extraction is expected to use an explicitly configured NVIDIA CUDA environment such as Colab or a laboratory server.
- DirectML and unsupported ROCm paths are not part of v0 unless a later task approves them.
