# Frozen VLM Compatibility Pilot Contract v0

Experiment ID: `vea.p2.vlm_compatibility_pilot.v0`

## Scope

This Phase 2 infrastructure pilot permits one exact-checkpoint inference path:
capture the native ordered patch-token tensor immediately before TinyLLaVA's
multimodal connector, serialize and reload that tensor, and inject it again at
the same pre-connector boundary.

The vision tower, multimodal connector, and LLM are frozen. Every parameter has
`requires_grad=False`; the model is in evaluation mode. The pilot creates no
optimizer, performs no backward pass, and makes no parameter update.

## Feature boundary

- Pre-connector raw patch tokens and post-connector LLM-aligned outputs are
  distinct feature classes.
- Capture and injection occur only at the explicit pre-connector patch-feature
  boundary.
- Injection may temporarily replace one loaded model instance's `encode_images`
  method, must call the original frozen connector, and must restore the method.
- Downloaded model source and cache files are immutable inputs.

## Interpretation

This experiment does not revise Phase 0 or Phase 1 metrics or scientific
conclusions. A software `PASS_NATIVE_REPLAY` establishes deterministic native
feature replay for one pinned VLM path only. It does not imply that another
encoder is interchangeable with CLIP, that cross-encoder injection is valid, or
that representation similarity is functionally sufficient.
