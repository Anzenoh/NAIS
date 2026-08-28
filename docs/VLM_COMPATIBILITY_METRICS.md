# Phase 2 VLM Compatibility Metrics

## 1. Evaluation reference and aggregation

The TinyLLaVA native CLIP path is the target reference. Each finalized adapter
is evaluated on the same held-out image IDs and with identical downstream
configuration. Aggregate by image where patch count would otherwise overweight
an image; confidence intervals or resampling, when used, operate by image ID.

Existing formulas and leakage rules in `METRIC_DEFINITIONS.md` remain
authoritative where they apply. This document adds the Phase 2 evaluation
hierarchy without redefining incompatible metric semantics.

## 2. Patch feature metrics

Compare adapted target-like patch tensors with native CLIP pre-connector patch
tensors for the same Query image:

- cosine similarity, aggregated per image and then across images;
- NMSE using the training-target reference mean defined in
  `METRIC_DEFINITIONS.md`;
- R-squared (`R^2`) using training-derived reference statistics.

Report tensor shape, sample count, dtype, aggregation unit, and split.

## 3. Connector metrics

Pass adapted and native target patch tensors through the same frozen TinyLLaVA
connector. Compare the distinct connector-output feature class with:

- cosine similarity;
- NMSE.

Aggregate per image before dataset summaries where appropriate.

## 4. Same-image identity metrics

Using the approved cosine-based cross-image comparison, measure whether each
adapted Query is matched to the native target representation of the same image:

- same-image R@1;
- same-image R@5.

The retrieval gallery contains held-out evaluation images only and preserves
image-ID pairing and ordering.

## 5. Frozen-LLM generation metrics

Run actual generation through the frozen connector and frozen TinyLLaVA LLM.
For every finalized method report:

- generated text;
- native-path exact match after the explicitly frozen comparison normalization;
- Token F1 between adapted-path and native-path generated token sequences,
  computed from token precision and recall with their harmonic mean.

Prompt, temperature, `top_p`, `max_new_tokens`, tokenizer, stopping behavior,
and all other decoding settings must be identical across finalized methods.
Their exact values remain `OPEN` until separately approved.

## 6. Interpretation

No patch or connector similarity value alone establishes functional VLM
compatibility. Patch, connector, same-image identity, and generation results
must be considered together. Every finalized method in the final comparison
receives generation evaluation; validation-rejected hyperparameter candidates
do not require Test generation.

Logit KL, top-k token overlap, and a semantic generation metric are optional and
non-mandatory until separately approved. No semantic metric is selected here.

## 7. First linear experiment generation freeze

For `vea.p2.query_adapter_linear.v1`, the authoritative configuration freezes:

```text
prompt: "What is shown in this image? Answer briefly."
max_new_tokens: 32
num_beams: 1
do_sample: false
top_p: null
temperature: 0
```

These exactly reuse the deterministic T001D settings. Native generation occurs
once for each of 100 Test images; eight finalized adapted source-method
configurations each generate for all 100 Test images, for 900 planned
generations total. Ridge lambda candidates receive no Validation generation.
Future experiments require their own separately approved exact configuration.
