# Experiment Protocol v0

## 1. Purpose

Establish whether three frozen ViT-B/16-class encoders exhibit reproducible layer-wise, linear, and spatial correspondence under controlled input geometry.

## 2. Candidate models

The model registry is `configs/models.yaml`. The initial keys are:

- `clip_vit_b16`;
- `siglip2_base_p16_224`;
- `dinov3_vit_b16`.

Exact model revision hashes must be resolved and pinned before the 1,000-image run. DINOv3 access or license requirements must be handled by the user, not accepted automatically by Codex.

## 3. Execution stages

### P0-A — Mock and contract tests

No network or real checkpoint is required.

- validate configuration parsing;
- validate feature manifest schemas;
- test special-token handling with synthetic model outputs;
- test metric implementations on mathematically controlled arrays;
- test image-level split isolation.

### P0-B — One-model real smoke tests

Add and validate one extractor at a time:

1. CLIP;
2. SigLIP2;
3. DINOv3.

Each smoke test uses two images first, then ten images. Do not proceed to the next model while required checks for the current model fail.

### P0-C — Three-model smoke run

- one common ten-image manifest;
- CPU-compatible execution;
- batch size 1 by default;
- FP32 compute and storage unless the task explicitly tests another dtype;
- all global taps;
- patch tensors at normalized depths 0.25, 0.50, 0.75, and 1.00;
- artifact and manifest validation.

### P1-A — Controlled 1,000-image extraction

- one immutable natural-image manifest;
- exact checkpoint revisions pinned;
- bulk extraction in an explicit CUDA environment;
- FP16 feature storage permitted after FP32 smoke equivalence is measured;
- all global taps and selected patch taps.

### P1-B — Global representation atlas

For every encoder pair:

- paired linear CKA across all comparable layer pairs;
- shuffled-image CKA control;
- same-encoder deterministic repeat control;
- bootstrap confidence intervals;
- layer heatmaps and machine-readable tables.

### P1-C — Pairwise mapping

For selected layer pairs:

- orthogonal Procrustes when dimensions allow;
- ridge linear mapping;
- held-out R², normalized MSE, cosine similarity, and retrieval;
- both mapping directions;
- image-level split isolation.

### P1-D — Patch and spatial correspondence

- patch-level feature-space or streaming CKA;
- coordinate retrieval after explicitly defined alignment;
- exact-coordinate accuracy;
- within-one-patch accuracy;
- mean Manhattan and Euclidean grid error;
- qualitative maps only after quantitative artifacts exist.

## 4. Canonical preprocessing

- RGB decode;
- bicubic resize of shorter side to 224;
- center crop 224 × 224;
- shared geometry output retained or hashed;
- model-specific value scaling and normalization after geometry;
- no random augmentation in the baseline extraction.

## 5. Data manifests

A JSONL manifest contains at minimum:

```json
{"image_id":"stable-id","relative_path":"relative/path.jpg","split":"smoke"}
```

Rules:

- paths are relative to a configured dataset root;
- image IDs are stable and unique;
- the manifest order is immutable within a run;
- unavailable or corrupt images are errors, not silently skipped;
- raw images are not copied into generated artifact directories.

## 6. Required controls

- identical feature input should produce CKA approximately 1 within numeric tolerance;
- a known orthogonal transform should be recovered by Procrustes;
- shuffled image pairing must be reported as a null reference;
- random Gaussian features of matched shape must be available as a synthetic null;
- same-encoder repeated extraction must quantify runtime nondeterminism;
- an extraction with different geometry must not be merged with the canonical run.

## 7. Phase gates

### Gate P0

Pass only when:

- all mock/unit checks pass;
- each real extractor has an independent successful smoke report;
- the combined ten-image artifact passes schema and shape validation;
- no model-specific assumption is hidden in shared analysis code.

### Gate P1 extraction

Pass only when:

- checkpoint revisions and environment are pinned;
- manifest and preprocessing hashes are recorded;
- all 1,000 images are represented for every encoder;
- tensor counts, image IDs, and taps are consistent;
- no NaN or Inf exists.

The gate does not require similarity to be high.
