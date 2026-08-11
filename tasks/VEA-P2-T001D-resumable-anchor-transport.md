# VEA-P2-T001D — Resumable universal anchor transport

## Objective

Implement, without executing scientifically, the immutable-plan and atomic
resume harness for the fixed three-encoder nonparametric anchor-transport
experiment against the frozen TinyLLaVA CLIP connector and LLM.

## Frozen experiment

- Manifest: immutable `p1_1000` repository manifest in manifest order.
- Anchors: first 128 train rows by ascending `selection_index`.
- Evaluation: every validation and test row, disjoint by image ID.
- Sources, in order: TinyLLaVA-native TARGET_CLIP, approved SigLIP2 `final_norm`,
  approved DINOv3 block 11 with five prefix tokens removed.
- Conditions, identically dispatched: RAW, SAMPLE_RMS, GLOBAL_1NN,
  GLOBAL_TOP8, POSITIONAL_1NN, POSITIONAL_TOP8.
- Feature/transport unit: at most 16 images. Generation unit: one image and one
  source-condition, with NATIVE generated once per test image.
- Expected generation units: `100 * (1 + 3 * 6) = 1,900`.

## Software definition of done

- Canonical immutable `run_plan.json` and SHA-256 `plan_id` bind repository,
  config, manifest, source/profile/tap identities, paths, units, and storage.
- All 2,345 planned units publish atomically with exact plan-bound done markers.
- Resume skips valid units, quarantines task-owned orphans, and fails closed on
  corruption, identity mismatch, or an ambiguous lock.
- One coordinator enforces sequential model residency; heavy imports occur only
  in the validated stage worker after the GPU gate.
- TARGET_CLIP RAW is an exact patch, connector, token-ID, and decoded-text
  self-control.
- Final CPU aggregation publishes exactly seven named artifacts only when every
  unit is exact-valid.
- Fake interruption after completed generation work resumes without rewriting
  that work and produces the same final `result.json` as an uninterrupted run.
- Only the assigned Tier 0 checks run locally; the real experiment remains a
  separate, independently reviewed Apple action.

## Non-goals

No model/data acquisition, package installation, Apple/CUDA execution, learned
projection, regression, SVD, optimizer, gradient, model-specific adapter,
second target VLM, significance test, merge, or downstream task is in scope.
