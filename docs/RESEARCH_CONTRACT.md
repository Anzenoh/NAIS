# Research Contract v0

## 1. Canonical notation

For image `x`, encoder `E_i`, and comparable block output `l`:

```text
H_i^l(x) ∈ R^(P_i × D_i)
```

- `P_i`: number of canonical patch tokens after special-token removal;
- `D_i`: hidden dimension;
- `H_i^l`: raw vision-backbone block output before native pooling, multimodal projection, token merging, or LLM alignment.

A global representation for that tap is defined as the mean of canonical patch tokens:

```text
g_i^l(x) = mean_patch(H_i^l(x))
```

Native CLS, attention-pool, projected, or pooled outputs are different feature classes. They may be stored, but they must never be silently compared as if they were `g_i^l`.

## 2. Comparable tap semantics

The canonical block tap is the output after a complete transformer block, including its residual updates, and before any model-level final pooling or projection.

Each extractor must expose explicit tap names:

- `embedding_output`;
- `block_01` through `block_N`;
- `final_norm`, when present;
- optional native outputs under a separate namespace such as `native_pooled`.

If a library's `hidden_states` semantics differ, the extractor must document and test the mapping. Layer numbering is one-based in reports and must include the underlying zero-based library index in metadata.

## 3. Canonical image geometry

For v0 spatial comparison:

1. decode the same RGB image once;
2. apply one shared deterministic geometry transform;
3. produce the same 224 × 224 pixel field for every encoder;
4. only then apply each encoder's documented numeric scaling and channel normalization.

The default geometry is bicubic resize of the shorter side to 224 followed by a 224 × 224 center crop. The exact interpolation implementation and antialias setting must be recorded. A later native-preprocessing experiment must use a different experiment ID.

## 4. Patch-token contract

- Canonical patch tokens represent the row-major spatial grid after removing non-spatial special or register tokens.
- Do not remove tokens by an unverified fixed offset.
- Derive model-specific token structure from configuration and validated extractor logic.
- For the v0 224/p16 candidate set, the expected spatial grid is 14 × 14 and the canonical patch count is 196. Runtime output must still be asserted rather than assumed.
- Record normalized patch-center coordinates and grid shape in metadata.

## 5. Pairing and split contract

- The same ordered image manifest must be used for every encoder in a compared run.
- Every artifact records image IDs and manifest hash.
- Train, validation, and test partitions are created by image ID before any patch flattening.
- All patches from one image remain in the same partition.
- Mapping parameters, means, variances, or hyperparameters are fit only on the training partition.
- The test partition is evaluated once per finalized configuration.

## 6. Feature classes

The following classes must remain distinct:

1. raw block patch tokens;
2. mean-pooled raw block tokens;
3. final-normalized tokens;
4. native pooled or CLS features;
5. projected image-text embedding;
6. token-merger or resampler output;
7. multimodal-projector or LLM-aligned tokens.

A comparison crossing feature classes requires an explicitly named experiment and rationale.

## 7. Scope invariants

During Phase 0 and Phase 1:

- encoders are in evaluation mode and gradients are disabled;
- encoder weights are unchanged;
- no LLM or multimodal projector is loaded;
- no labels are required for representation extraction;
- no adapter is trained before extraction, schemas, controls, and baseline metrics are validated;
- all model and dataset substitutions require approval.

## 8. Scientific interpretation

- CKA measures relational similarity, not direct interchangeability.
- Reconstruction or mapping quality must be reported on held-out images.
- Global alignment does not imply patch-level or spatial alignment.
- Similarity values require paired, shuffled, and synthetic reference distributions.
- Directional mappings `A → B` and `B → A` may differ and must be reported separately.
- Failure to confirm a hypothesis is a valid research outcome.

## 9. Valid run requirements

A scientific run is valid only when it records:

- Git commit and dirty state;
- environment and hardware metadata;
- exact model IDs and revisions;
- dataset manifest and hash;
- preprocessing configuration and hash;
- tap semantics and actual tensor shapes;
- random seeds and split manifest;
- artifact schema version;
- validation results and any deviations.

## 10. Phase 2 frozen-VLM compatibility pilot

The explicitly named `vea.p2.vlm_compatibility_pilot.v0` experiment may load a
frozen vision tower, frozen multimodal connector, and frozen LLM for checkpoint
inference only. It may capture and reinject patch features immediately before
the connector. No optimizer, backward pass, fine-tuning, or parameter update is
permitted. Raw patch tokens and connector outputs remain separate feature
classes. This pilot is separate from, and does not revise, any Phase 0 or Phase
1 scientific claim; a software replay PASS does not establish encoder
interchangeability.

## 11. Phase 2 query-conditioned adapter contract

This section is separate from the frozen native-replay pilot above. Phase 0 and
Phase 1 remain encoder-only historical contracts. Phase 2 permits fitting or
training only an adapter that implements the same-Query relation:

```text
Adapter(E_source(x)) -> E_target(x)
```

Normative requirements:

- the source feature and target feature belong to the same Query image `x`;
- every source vision encoder is frozen;
- target TinyLLaVA CLIP is frozen;
- the TinyLLaVA connector is frozen;
- the TinyLLaVA LLM is frozen;
- only adapter parameters may be fitted or trained, and trainable-parameter
  verification must prove that boundary;
- source and target feature identities, taps, shapes, image IDs, and ordering
  are checked before fitting or evaluation;
- the immutable `p1_1000` split is by image ID: Train 800, Validation 100,
  Test 100, seed 43, with no patch-level leakage;
- fitting statistics and weights use Train; model and hyperparameter selection
  uses Train/Validation only; Test remains held out for finalized evaluation;
- raw-source, adapted-target-like, connector-output, and LLM-output feature
  classes remain explicit and are never silently mixed;
- every finalized method uses identical generation prompt and decoding settings.

Any identity, split, feature-class, or frozen-state mismatch fails closed.
Closed-form Calibration, Procrustes, and Ridge mappings count as adapter-family
baselines even when no optimizer or backward pass is used.

## 12. Phase 2 interpretation boundary

`VEA-P2-T001D` anchor transport remains valid historical/comparison evidence,
but it is not the primary direct adapter because it selects or combines target
features from other images. Its established qualitative result is retained:
tensor or connector similarity optimum and downstream retrieval or generation
optimum did not necessarily coincide. Tensor similarity alone therefore cannot
establish functional VLM compatibility.
