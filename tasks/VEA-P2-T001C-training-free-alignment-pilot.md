# VEA-P2-T001C — 12-image training-free DINO-to-CLIP alignment pilot

- Mode: implementation
- Baseline: `9ab2fb5391438b36c94fc201e8c73b9bcadb11c7`
- Baseline branch: `origin/codex/vea-p2-t001b-r3-apple-worker-response`
- Task branch: `codex/vea-p2-t001c-training-free-alignment-pilot`
- Commit: `feat(vlm-compat): VEA-P2-T001C add training-free alignment pilot`
- Delivery: PASS-only commit and `git push -u origin HEAD`

## Goal and scientific question

Implement one deterministic Apple experiment fitting two closed-form,
training-free transformations from frozen DINOv3 block-11 patch features to
frozen TinyLLaVA CLIP block-11 patch features. Determine whether either mapping
recovers connector-level numerical agreement and deterministic LLM behavior.
This 12-image pilot is bounded feasibility evidence, not a statistical result.

## Immutable identities and split

- CLIP: TinyLLaVA native CLIP ViT-B/16 `hidden_states[-2]`, patch-only
  `[image,196,768]`, then its original frozen connector and LLM.
- DINO: `facebook/dinov3-vitb16-pretrain-lvd1689m` revision
  `5931719e67bbdb9737e363e781fb0c67687896bc`, `hidden_states[11]`, remove
  one CLS and four register tokens, producing `[image,196,768]`.
- Reuse the validated R3 extraction and injection paths without modifying them.
- From immutable `data/manifests/p1_1000.jsonl`, select the first eight `train`
  rows and first four `test` rows by `selection_index`, preserving order.
- Partitions are unique and disjoint at image level; flatten patches only after
  partitioning. Reject validation rows, missing/substituted/hash-mismatched files,
  and development image `000000000139.jpg`.

## Conditions and transforms

The exact condition order is `NATIVE`, `RAW`, `CALIBRATED`, `LINEAR`.

`CALIBRATED` uses training rows only, float64 accumulation, population standard
deviation (`ddof=0`), and denominator floor `1e-6`:

```text
scale = std(Y) / max(std(X), 1e-6)
bias = mean(Y) - mean(X) * scale
A(X) = X * scale + bias
```

Store scale and bias in float32. `LINEAR` applies orthogonal Procrustes after
calibration, again fitting training rows only in float64:

```text
U,S,Vt = SVD((A(X)-mean(A(X))).T @ (Y-mean(Y)))
R = U @ Vt
B(X) = (A(X)-mean(A(X))) @ R + mean(Y)
```

Store `R` float32, allow reflection, and perform no determinant correction.
No optimization, gradient, clipping, pooling, permutation, regularization,
whitening, CORAL, Ridge, OLS, MLP, or hyperparameter search is permitted.

## Exact runtime

Validate and decode all 12 sources; run one isolated DINO batch for all 12;
wait for that worker to exit; load TinyLLaVA once; capture eight calibration
native patch/connector results with `max_new_tokens=1`; fit both transforms; then
run four held-out images in the four fixed conditions with exactly 16 generation
calls and `max_new_tokens=32`. Use prompt `What is shown in this image? Answer
briefly.`, beams 1, sampling false, `top_p=None`, and temperature 0.

DINO and TinyLLaVA may never coexist on GPU. Every replacement bypasses CLIP,
uses the original frozen connector, restores `encode_images`, never mutates a
source tensor, and crosses NumPy-to-Torch using a writable contiguous copy.

## Metrics and aggregation

For RAW, CALIBRATED, and LINEAR, compare the effective post-cast connector input
and connector output with the same image's NATIVE values. Record flattened
float64 cosine, norm ratio, and task-defined NMSE. Patch NMSE uses the eight-image
calibration CLIP patch mean; connector NMSE uses the calibration native connector
mean. Record generation success, token equality, text equality, token count, and
both answers. For every condition/metric report mean, median, min, max, and count
improved over RAW using the correct higher/lower direction. Different answers or
low similarity remain a software PASS.

## Artifact and CLI contract

A fresh output directory contains exactly:

```text
result.json
per_image.csv
summary.csv
paired_patch_features.safetensors
alignment_parameters.safetensors
test_connector_outputs.safetensors
```

The paired file stores calibration/test CLIP and DINO patches. Parameters stores
moment scale/bias, calibrated/CLIP/DINO train moments, and Procrustes rotation.
Connector outputs stores native/raw/calibrated/linear held-out outputs.
`result.json` uses schema `vea.p2.t001c.training_free_alignment_pilot.v1` and
status `PASS_TRAINING_FREE_ALIGNMENT_PILOT`, with identities, selections/hashes,
definitions, shapes/dtypes, metrics, tokens/text, call counts, artifacts, and
stage timings.

The CLI requires only `--tinyllava-config`, `--manifest`, `--dataset-root`,
`--cache-root`, `--repository-root`, `--dinov3-runtime-root`,
`--dinov3-snapshot-root`, `--output-dir`, `--device`, and `--dtype`. It requires
CUDA for the real run, remains import-safe for help, has no image-count or fit
override, does no network/install, and writes bounded structured failure evidence.

## Offline validation and protected scope

Use CPU/fake tests to prove split/hash/leakage, exact fits and metrics, condition
order, artifact/schema contracts, writable copies, one DINO batch, one model
load, exactly 16 test generations, scientific-failure software PASS, and import
safety. Run only the assigned compile, two targeted test modules, CLI help, and
`git diff --check` checks.

Modify only the implementation, CLI, focused test, Apple runbook, run report,
and this task. Do not modify the R3 raw/native modules or shared extractor,
worker, boundary, profiles, manifest, metrics, or dependencies. Codex must not
access Apple, CUDA, models, datasets, network, packages, or materialization.
Do not begin SigLIP2, MLP, another prompt, expanded evaluation, or another task.
