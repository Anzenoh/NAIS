# Phase 1 Analysis Campaign Contract v1

Schema: `vea.p1.analysis_campaign.v1`

## Scope and immutable identity

This contract covers encoder-only analysis of the accepted run
`20260803T192848Z_p1_1000_three_encoder_27688e67b0c6` at extraction commit
`27688e67b0c639a81d07c31b7992dc8a09341d8d` and tree
`b6fb3ef7fc99e0a4f16c351867f2da1fea100760`. CLIP, SigLIP2, and DINOv3 keep their
accepted revisions, FP32 14-by-14 patch geometry, 768-wide representations, common
ordered 1,000-image manifest, and image-ID split (800 train, 100 validation, 100 test;
seed 43; split SHA-256 `6c299a79b22b99b4c68c3de471b9d5c7c756b6bf4a0f78b831a97e30a0a10c5f`).
Encoders remain frozen and no model, projector, LLM, MLP, nonlinear adapter, or
cross-attention component is loaded or fitted.

The canonical machine contract is `configs/analysis/p1_analysis_campaign_v1.json`.
Duplicate or unknown JSON fields, mutable defaults, non-finite values, inconsistent
identities, or a non-canonical encoding fail closed. The master seed is `20260804`; each
sub-seed is the big-endian unsigned first 64 bits of SHA-256 over
`"20260804:<canonical-label>"`.

## Blind access and split isolation

Before value access, the reader verifies the four-file T003B evidence manifest, the
252-entry full-run checksum list, the exact inventory, every sidecar, and every
Safetensors header, dtype, key, shape, byte span, and file boundary. Opaque whole-file
hashing and header parsing do not count as value access.

Value access uses `safe_open(...).get_slice(key)[start:stop]` only. It splits
non-contiguous authorized rows into exact spans and also splits a full-shard span so no
whole-key slice is used. Before `T007_TEST_UNSEAL`, only train and validation rows may be
decoded. Test NaN/Inf, all-zero, norm, and variance checks occur only in the single test
event. Every decoded span is appended to `data_access_ledger.jsonl`. The legacy
`vea.feature_artifact.v0` whole-shard loader is not an analysis input route.

The first shard is a mandatory mixed fixture: 20 train, 3 validation, and 2 test images.
Any pre-freeze test value read is `BLOCKED_LEAKAGE`. After unseal, only frozen test
endpoints may read values; refit, reselection, changed code/config/input, endpoint
removal, and a second logical unseal are forbidden.

## Frozen numerical definitions

- Linear CKA uses column-centered feature-space FP64 moments. Patch CKA never constructs
  a patch-row sample Gram matrix.
- `CKA_u` uses zero-diagonal U-centering and unbiased HSIC for `n >= 4` as a sensitivity
  endpoint. RSA is tie-aware Spearman correlation of upper-triangle cosine-distance RDMs.
- The spectrum-matched null independently replaces each centered encoder matrix's left
  singular vectors with a seeded Haar basis projected off the constant vector while
  retaining its singular values and right singular vectors.
- Covariance eigenvalues, entropy effective rank, participation ratio, mean off-diagonal
  cosine anisotropy, norm summaries, and cosine kNN overlap at 5, 10, 20, and 50 are
  secondary geometry endpoints.
- Procrustes is centered, square, orthogonal, train-fitted SVD. Ridge is centered and
  uses `alpha_absolute = alpha_relative * trace(Xc.T @ Xc) / source_width`. Alpha is
  selected once on validation. A condition number above `1e12`, SVD failure, undefined
  denominator, zero norm, zero variance, or non-finite result fails the endpoint; no
  alternate solver or epsilon is substituted.
- Mapping R-squared is the macro mean over target dimensions. Each dimension's
  denominator uses that dimension's frozen train-target mean. NMSE uses the summed form
  with the same train mean. Cosine, per-image MSE, and cosine retrieval are separate.
- Patch states are `X[i,p]`, `W[i,p] = X[i,p] - G[i]`, and
  `D[i,p] = X[i,p] - G[i] - P_train[p] + mu_train`. Only `P_train` and `mu_train` are
  train-fitted; `G[i]` is the image's own within-image transform. Coordinates are
  row-major 14-by-14. Retrieval reports exact, Manhattan-at-most-one, top-five, MRR, and
  mean/median Manhattan and Euclidean error.
- Reduced-rank ridge is the rank-truncated SVD of a train-fitted ridge map. Ranks above
  `min(source width, target width, n_train - 1)` are invalid rather than clipped.
- Symmetric multiway alignment is regularized MAXVAR GCCA: encoder-balanced projection
  operators determine centered orthonormal `Z`; each encoder uses a train-fitted linear
  encoder and decoder. Shared variance and private residual are reported per encoder and
  as an encoder-balanced mean.

Constant dimensions and undefined metrics are recorded as undefined and force an
`INDETERMINATE` gate when primary; they are never coerced to zero. Stable sorting plus
canonical encoder/tap/grid order resolves ties.

## Development, selection, and inference

T004 builds 12-by-12 raw-block atlases and separate `embedding_output` and `final_norm`
tables. Train discovers the eight highest raw-block candidates per encoder pair;
validation selects exactly one by maximum CKA, then lexical tap order. T005 freezes both
directions of Procrustes and ridge for those exact nodes. T006 evaluates taps 03, 06, 09,
and 12 in `X`, `W`, and `D`; unavailable exact nodes are emitted only as
`NOT_APPLICABLE_PATCH_TAP_UNAVAILABLE`. The authorized confirmatory linear extension
uses a jointly compatible exact node for direct/composed maps, closed cycles,
reduced-rank maps, MAXVAR GCCA, shared/private decomposition, and a nested validation-only
learning curve.

The one-standard-error rule uses frozen image-block bootstrap draws at the validation
minimum and chooses the least complex eligible candidate, then canonical order. Ridge
complexity decreases with larger alpha; rank and latent complexity increase with size.
Test data never enters a mean, normal equation, template, alpha, node, rank, latent size,
null construction used for selection, or conclusion rule.

## Inference counts and decision rules

Global permutations are 1,000; global primary image bootstraps are 2,000; shuffled-refit
mapping nulls are 500; within-image patch permutations are 500; selected patch image
bootstraps are 1,000. Confidence is 95%. All bootstrap and permutation units are image
blocks; patches from one image never separate. Primary families use Holm-Bonferroni at
0.05. Counts cannot be reduced for a sealed campaign. Synthetic fixtures may use lower
counts but are structurally ineligible for a final seal.

Evidence gates are frozen before test:

- A: global, mapping, spatial, and canonical extension families confirm without strong
  counterevidence.
- B: global, mapping, and canonical extension confirm; spatial confirmation is absent or
  mixed.
- C: mapping or canonical extension confirms while global evidence does not.
- D: spatial evidence confirms while global and mapping do not jointly confirm.
- E: all frozen endpoints complete and no family confirms.
- `INDETERMINATE`: a primary endpoint is undefined/incomplete or mixed evidence meets no
  A-E gate.

There is no universal raw CKA threshold. Confirmation requires a positive
null-centered effect, predeclared family correction, and the relevant image-block
interval. Negative R-squared, low retrieval, asymmetry, or a null hypothesis result is a
valid scientific outcome, not a technical failure.

## Repeat and Windows-local execution

Apple execution requires the accepted exact-checkout repeat boundary. For the explicitly
authorized Windows-local lane, that model-inference resource is unavailable and the
same-setting repeat is exactly `NOT_EXECUTED_LOCAL_RESOURCE_UNAVAILABLE`. This is a
declared limitation, not a substituted control. Starting/ending 252-file checksum
identity is recorded separately as input preservation and must never be called a repeat.

## State, sealing, and bounded claims

The state sequence is exactly `PREFLIGHT -> BLIND_INPUT_INTEGRITY -> REPEAT_CONTROL ->
T004_DEV -> T005_DEV -> T006_DEV -> EXTENSION_DEV -> FREEZE -> T007_TEST_UNSEAL -> REPORT
-> SEAL -> VERIFY`. Stages publish atomically with a completion marker written last,
exclusive locking, no-clobber, exact-identity resume, and one logical test-unseal event.

At REPORT, immutable scientific members and sanitized ledger/state/event snapshots are
copied into payload staging. `payload_manifest.json` covers every archive file except
itself and `seal.json`; `seal.json` points to the payload-manifest SHA and contains no
outer archive SHA. Deterministic tar/gzip fixes paths, order, metadata, modes, IDs,
timestamps, and compression. The outer SHA appears only in sibling delivery evidence.
Verification re-hashes every covered member and finally proves the source 252-file
checksums are unchanged.

The report is limited to these three checkpoints, this COCO 1,000-image 224x224
condition, and encoder-only Phase 1. It makes no causal claim about pretraining
objectives, no universal encoder claim, and no nonlinear conclusion.
