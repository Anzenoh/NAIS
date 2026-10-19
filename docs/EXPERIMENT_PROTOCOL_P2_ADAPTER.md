# Phase 2 Query-Conditioned Adapter Protocol

## 1. Purpose and research question

Phase 2 tests direct encoder-to-VLM compatibility while every pretrained model
component remains frozen. For the same Query image `x`, the required relation is:

```text
A_s(E_s(x)) -> E_t(x)
```

The research question is: what is the minimum transformation complexity needed
to restore functional compatibility with the frozen target VLM?

## 2. Feature boundaries

Target:

- TinyLLaVA native CLIP pre-connector patch feature;
- patch-only `[N,196,768]`.

Source candidates:

- SigLIP2 `final_norm`, patch-only `[N,196,768]`;
- DINOv3 block 11 after removing CLS 1 plus register tokens 4, patch-only
  `[N,196,768]`.

Raw-source, adapted-target-like, connector-output, and LLM-output tensors are
distinct feature classes. Identity, tap, shape, image-ID, and ordering checks
must pass before fitting or evaluation.

## 3. Frozen and fitted components

Frozen in evaluation mode:

- source vision encoder;
- target TinyLLaVA CLIP;
- TinyLLaVA connector;
- TinyLLaVA LLM.

Only the adapter may be fitted or trained. Calibration, Orthogonal Procrustes,
and Ridge are adapter-family baselines even when fitted in closed form without
an optimizer.

## 4. Dataset and partition roles

Reuse the immutable `p1_1000` image-level split:

| Partition | Images | Role |
|---|---:|---|
| Train | 800 | Fit statistics and adapter weights |
| Validation | 100 | Select finalized configuration without Test access |
| Test | 100 | Held-out evaluation of finalized configurations |

The split seed is 43. Partitioning is by image ID before patch flattening;
patch-level leakage is prohibited. Existing validated feature banks are reused
by default. New extraction requires a demonstrated identity, tap,
preprocessing, or artifact incompatibility.

## 5. Complexity ladder

Evaluate increasing complexity only after reviewed evidence:

1. Dimension-wise Calibration
2. Orthogonal Procrustes
3. Ridge / Full Linear
4. Low-rank Linear
5. Residual Bottleneck MLP
6. Spatial Residual Adapter only if token-wise methods remain insufficient

## 6. Fitting, selection, and Test separation

- Train supplies every fitted statistic, parameter, and weight.
- Validation selects hyperparameters and finalized configurations.
- Test is not used for fitting, threshold choice, early stopping, or model
  selection.
- Test evaluation is performed for finalized configurations only and follows
  the task-approved single-use protocol.
- Validation-rejected candidates do not require Test generation.

## 7. Frozen downstream evaluation

Every finalized method supports patch cosine, NMSE, and R-squared; connector
cosine and NMSE; same-image R@1 and R@5; and actual frozen-LLM generation with
native-path exact match and Token F1. The native CLIP path is the reference.
All finalized methods use identical prompt and decoding configuration.

`VEA-P2-T001D` anchor transport remains a historical/comparison baseline. It
does not satisfy the primary direct-adapter definition because it selects or
combines other-image target features. Its evidence also establishes that tensor
similarity alone is insufficient for functional compatibility claims.

## 8. Experiment gates

Before fitting:

- feature-bank and frozen-checkpoint identities pass;
- source/target shapes, image IDs, and ordering match;
- split isolation passes;
- only adapter parameters are trainable;
- applicable synthetic known-mapping controls pass.

Before Test:

- the exact configuration was selected using Train/Validation only;
- fitted artifact hash and reproducibility metadata are complete;
- prompt and decoding settings are frozen identically across methods.

Software acceptance and scientific performance are reported separately.

## 9. First frozen linear experiment

`configs/vlm_compat/p2_query_adapter_linear_v1.json` freezes the first direct
experiment, `vea.p2.query_adapter_linear.v1`. It includes both source lanes in
SIGLIP2, DINOV3 order; independent per-source adapters; CALIBRATED,
ORTHOGONAL_PROCRUSTES, FULL_LINEAR, and RIDGE; the nine-value normalized Ridge
grid; deterministic Validation selection; held-out Test evaluation; exact T001D
generation settings; and the float64-fit/float32-storage numeric policy.

The configuration freeze itself did not execute the experiment. Subsequent
T002E real-Test generation evidence is recorded by `VEA-P2-T002F` against
commit `39fbbccd654aa6d0726802b5aaa1377a86f81f6b`: 100 native and 800
adapted generations over 100 held-out images. The hash-verified generation rows
show the native vision/connector path and adapted injection/connector
path were called at their expected boundaries. This records software/runtime
execution evidence separately from the bounded scientific interpretation;
the T002E result record was independently accepted through T002F review and
T002G state closure.

## 9a. Connector-aware Validation study disposition

T003E reported real Apple Train/Validation fitting for the same-capacity
connector-aware affine objective, with Test access/metrics/generation/vision
`0/0/0/0`. The SIGLIP2 selected `lambda=4`, epoch 7 improved Validation R@1
`0.89 -> 0.92` and connector cosine
`0.48697282446433066 -> 0.5056841280687281` relative to OLS, while patch
cosine changed `0.4628383531145029 -> 0.449191858608662`. The DINOV3 selected
`lambda=0.25`, epoch 0 retained the OLS/patch-only affine identity.

T003F independently reviewed the Validation-generation implementation at
`0b3b7a2ee135e1e85cf459248066f5b53a313ef9` without real generation in
that task. User-supplied T003G execution output reports 100 Validation images,
100 native plus 400 adapted generations, Test access/metrics `0/0`, and a
structural `PASS`. SIGLIP2 connector-aware Token F1 and native exact match
declined relative to OLS; DINOV3 generation was identical. The tested
same-capacity objective is `NOT_PROMOTED` over OLS; this is not a universal
conclusion about connector loss. T003H did not independently inspect or rehash
the raw T003G files. T002E Test and T003G Validation metrics are from different
partitions and must not be compared as same-image results. Historical Test
results are not used for future method selection, and no new Test access is
authorized.

## 9b. Frozen T004A Low-rank Linear gate result

T004A separately froze the Low-rank Linear reduced-rank regression study at
`[8,16,32,64,128,256]`, with accepted same-run Full Linear OLS controls,
Train-only fitting, the 100-image Validation split, a maximum of 400,000 stored
float32 scalars per source, and the matched-device float32 adapter-only latency
criterion. T004B/C-R2/E/G implemented its numerical core, input binding,
Train-only fitting runner and Validation patch/identity runtime. None changes
the frozen T004A design.

The user-supplied T004F Train-only run at reviewed T004E commit
`8c0beb8a88cc5bdb1225c50ed62a7ea0349ef06e` reported six fitted ranks
per source and no Validation/Test metrics. The user-supplied T004H
Validation-only run at reviewed T004G commit
`a67d923b2b65c64f9782d70cd13f183312ed60fd` reported patch/identity
metrics for OLS and all six ranks on the same 100 Validation images per
source. T004I did not independently inspect or rehash the raw Apple files.

The frozen necessary R@1 criterion is candidate `>=` same-run OLS minus
`0.02` absolute. SIGLIP2 OLS `0.89` requires `0.87`, while the highest
low-rank R@1 is rank 256 at `0.86`; DINOV3 OLS `0.87` requires `0.85`, while
rank 256 reaches `0.84`. Every other frozen rank is lower. Since all selection
criteria are conjunctive, every rank is ineligible regardless of the
unmeasured criteria. The bounded eligibility verdict is `NO_RANK_ACCEPTED`
for each source; retain accepted OLS. T004G's original condition records
remain `UNSELECTED` and are not rewritten. This necessary-gate disposition
does not assert a frozen TinyLLaVA generation result or universal low-rank
failure. The 100-image R@1 granularity is `0.01`; rank 256 is one increment
below the minimum for each source.

Connector cosine/NMSE, native exact match, Token F1, fair float32 adapter-only
latency, paired bootstrap intervals and Test metrics are `NOT_EVALUATED` in
T004H. The frozen 1,500 Validation-generation plan is unchanged; no automatic
expensive follow-up is scheduled after this failed necessary gate. Any later
nonlinear or MLP design requires new approval and a separate configuration
freeze. Historical T002E Test numbers are not selectors for this study.

## 10. Remaining OPEN configuration fields

The first linear experiment does not assign future complexity settings. T004A
froze the rank grid, fitting, parameter and latency budgets, Validation
selection, generation plan and Test policy for its bounded Low-rank Linear
study. The following remain `OPEN` for separate future studies:

- MLP bottleneck size, optimizer, learning rate, epochs, and early stopping;
- MLP patch/connector multi-objective loss and connector-loss weight;
- parameter and latency budgets outside the bounded T004A study;
- spatial adapter architecture;
- semantic generation metric.

T003H assigned none of the later T004A low-rank values. T004I does not start
an MLP or spatial experiment, adjust thresholds, or authorize new Test access.

Original Apple artifact paths are not recovered in the T002F evidence record.
Current validator evidence paths do not replace those historical paths.
