# Metric Definitions v0

## 1. Common preprocessing for metrics

For a feature matrix `X ∈ R^(N × D)`:

- rows are samples in the same ordered manifest;
- columns are feature dimensions;
- center columns using means estimated from the appropriate partition;
- do not normalize or whiten unless the metric configuration explicitly says so;
- record dtype and accumulation dtype;
- use at least FP32 accumulation for reported metrics.

Patch rows may be flattened only after image-level partitioning.

## 2. Linear CKA

For column-centered matrices `X` and `Y`:

```text
CKA(X, Y) = ||XᵀY||_F² / (||XᵀX||_F · ||YᵀY||_F)
```

Requirements:

- use the feature-space formulation or an equivalent streaming covariance formulation;
- do not construct an `N × N` Gram matrix for large patch-level inputs;
- report sample count, dimensions, centering method, and accumulation dtype;
- report paired, shuffled, and synthetic-null values separately;
- do not infer interchangeability from CKA alone.

## 3. Orthogonal Procrustes

For compatible dimensions, fit on training data:

```text
W* = argmin_W ||XW - Y||_F², subject to WᵀW = I
```

Rules:

- center source and target using training-set means;
- solve using SVD of `XᵀY`;
- store `W`, source mean, target mean, and fit partition hash;
- evaluate only on held-out image IDs;
- report normalized residual, R², mean cosine similarity, and retrieval.

If dimensions differ, do not silently pad or truncate. Use the separately specified ridge mapping or an approved rectangular formulation.

## 4. Ridge linear mapping

Fit:

```text
W* = argmin_W ||XW - Y||_F² + λ||W||_F²
```

Rules:

- choose `λ` using training/validation data only;
- standardization statistics come from training data only;
- report source-to-target and target-to-source separately;
- report held-out R², normalized MSE, mean cosine similarity, and retrieval;
- preserve the image-level split when patch tokens are used.

Normalized MSE is:

```text
NMSE = sum(||Y_hat - Y||²) / sum(||Y - mean_train(Y)||²)
```

## 5. Cross-encoder retrieval

Given mapped or aligned source features and target features from the same test manifest:

- similarity: cosine unless explicitly changed;
- report top-1 and top-k matching-image accuracy;
- exclude training images;
- record whether retrieval is global-image or patch-level.

## 6. Patch spatial correspondence

For each source patch, retrieve a target patch within the same image after an explicitly fitted or identity alignment.

Report:

- exact coordinate top-1 accuracy;
- coordinate accuracy within Manhattan distance 1;
- mean Manhattan grid distance;
- mean Euclidean grid distance;
- top-k coordinate recall.

Do not compare patch coordinates directly when grids or canonical image fields differ. Such cases require a separately approved coordinate mapping or resampling protocol.

## 7. Confidence intervals

Use bootstrap resampling by image ID, not by individual patch row. Record:

- seed;
- number of bootstrap replicates;
- confidence level;
- estimator;
- whether the same resampled image indices were used across compared conditions.

## 8. Interpretation thresholds

No universal thresholds such as “CKA ≥ 0.7 is high” are approved in v0. Interpret values relative to:

- same-encoder repeat;
- shuffled pairing;
- synthetic random features;
- other encoder pairs and layer pairs;
- confidence intervals;
- held-out mapping performance.

Any later threshold must be introduced through `docs/DECISION_LOG.md` with empirical justification.

## 9. Frozen Phase 1 campaign additions

- Primary global similarity is centered linear CKA with image-permutation calibration; `CKA_u`, cosine-distance RSA, covariance spectrum, effective rank, anisotropy, and kNN overlap are sensitivity or descriptive measures.
- Directional alignment reports macro R2 and NMSE against the train-target mean, cosine summaries, Recall@1/5/10, per-image error, and image-block confidence intervals. Ridge alpha and reduced rank use the validation one-standard-error rule with the least-complex deterministic tie-break.
- Patch states are raw `X`, within-image centered `W`, and train-template double-centered `D = X - G_image - P_train + mu_train`. Spatial resampling units are whole images.
- The authorized extension is linear only: exact-node direct and composed maps, closed cycles, reduced-rank maps, symmetric regularized MAXVAR GCCA, shared/private reconstruction, and a nested validation learning curve.
- Undefined zero-variance or zero-norm endpoints are explicitly `INDETERMINATE`; no silent epsilon, solver, endpoint, or layer substitution is allowed.
