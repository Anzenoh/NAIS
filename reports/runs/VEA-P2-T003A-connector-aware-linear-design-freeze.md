# Run Report — VEA-P2-T003A

## Status

`PASS` for the bounded design freeze and Tier 0 validation. The T003A delivery
remains pending independent review.

## Metadata

- Task ID: `VEA-P2-T003A`
- Run/report timestamp UTC: `2026-09-28 15:19:05 UTC`
- Repository root: user-selected P2 checkout (machine-local path omitted)
- Baseline commit: `296ff4fcdcfc23b489f166a283964fdd788091f1`
- Baseline remote branch: `codex/vea-p2-t002g-t002f-state-closure`, resolved
  to the exact baseline
- Git entry state: no actual tracked or staged diff; only the approved
  untracked `.codex/ACTIVE_TASK.md`
- Project state file: `NOT_PRESENT`
- Execution environment: Windows PowerShell and Python 3.12 standard library;
  configuration and Git validation only

## Goal

Freeze a Train/Validation-only, same-capacity affine experiment testing whether
a differentiable loss through the exact frozen TinyLLaVA connector improves
functional compatibility relative to accepted Full Linear OLS and a patch-only
gradient control.

## Summary

The new config is
`configs/vlm_compat/p2_query_adapter_connector_aware_linear_v1.json`, SHA-256
`4684a1ae43a124725df93d15c2287a30aa31ae38f6c58bd2f05b83e749f8494b`.
Its schema is `vea.p2.query_adapter_connector_aware_linear_config.v1` and its
experiment ID is `vea.p2.query_adapter_connector_aware_linear.v1`.

Each source, SIGLIP2 and DINOV3, receives an independent `Y_hat = X @ W + b`
adapter with `W[768,768]`, `b[768]`, and exactly 590592 trainable scalars.
Accepted Full Linear OLS initializes every gradient-trained condition and is
also the no-update reference. The five per-source conditions are
`OLS_REFERENCE`, `PATCH_ONLY_CONTROL`, `PATCH_CONNECTOR_0P25`,
`PATCH_CONNECTOR_1P0`, and `PATCH_CONNECTOR_4P0`. The connector-aware weight
grid is exactly `[0.25, 1.0, 4.0]`; only the separately named patch control
has weight zero.

Patch and connector squared-error losses use fixed, positive Train-derived
normalizers:

```text
D_patch     = mean_train((Y - mean_train(Y))^2)
L_patch     = mean_batch((Y_hat - Y)^2) / D_patch
D_connector = mean_train((C(Y) - mean_train(C(Y)))^2)
L_connector = mean_batch((C(Y_hat) - C(Y))^2) / D_connector
L_total     = L_patch + lambda_connector * L_connector
```

Reductions are elementwise means and loss accumulation is float32. The exact
pinned connector stays frozen in evaluation mode. A separate training boundary
must retain gradients through the float16 connector forward and accumulate its
output loss in float32. The existing inference-only connector method remains
unchanged.

All gradient conditions use AdamW, learning rate `3e-5`, betas `[0.9, 0.999]`,
epsilon `1e-8`, weight decay `0`, no scheduler, global gradient clip `1.0`,
eight images per batch, eight epochs, shuffle seed `43`, and no early stopping
or optimizer search. Validation runs at epochs `0..8`. Each condition selects
the lowest applicable Validation patch or combined loss, breaking exact ties
by earliest epoch. Each source then selects a connector-aware weight by highest
same-image R@1, R@5, connector cosine, then smallest weight. OLS and patch-only
controls remain mandatory. No performance threshold determines software PASS.

The immutable image split remains Train 800, Validation 100, Test 100. Test
access is unauthorized for this design and its Train/Validation execution;
the access counter must remain zero. Prior T002E Test is exposed for later
method development and remains historical comparison evidence only. Any future
confirmatory claim requires a separately frozen untouched-holdout policy.
Generation is not scheduled or authorized during fitting or selection.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `configs/vlm_compat/p2_query_adapter_connector_aware_linear_v1.json` | New frozen design | Fix all scientifically load-bearing choices for later implementation |
| `reports/runs/VEA-P2-T003A-connector-aware-linear-design-freeze.md` | This report | Record provenance, validation, and unresolved bindings |

## Validation evidence

| Requirement | Command/check | Exit status | Result |
|---|---|---:|---|
| Baseline and remote | `git rev-parse HEAD`; `git ls-remote --heads origin codex/vea-p2-t002g-t002f-state-closure` | 0 each | Both resolved to exact baseline |
| Entry scope | `git diff --quiet`; `git diff --cached --quiet`; `git ls-files --others --exclude-standard` | 0 each | Only approved task-control file untracked |
| JSON and frozen design | `py -3 -c $code` using standard-library `json`, `pathlib`, `hashlib` and assertions | 0 | Schema, conditions, capacity, losses, boundary, optimizer, selection, split, Test and generation policy, future gates all passed |
| Provenance | Same standard-library command recomputed SHA-256 for each referenced baseline file | 0 | Three of three hashes matched |
| Path and whitespace checks | Task-assigned Git commands | Recorded at delivery | See final Git handoff |

## Test and command transcript summary

```text
py -3 -c $code
CONFIG_SHA256 4684a1ae43a124725df93d15c2287a30aa31ae38f6c58bd2f05b83e749f8494b
PROVENANCE_SHA256 3/3 match
DESIGN_ASSERTIONS_PASS schema, conditions, capacity, losses, autograd, optimizer,
  selection, split, Test/generation prohibition, future gates
VALIDATION_EXIT=0
```

No pytest, model load, feature-bank or dataset access, raw T002E evidence,
Test access, GPU execution, generation, or runtime network access occurred.
No dependency was added.

## Artifacts

- Frozen JSON config at the path and checksum above.
- This Run Report. No fitted adapter or runtime artifact was created.

## Deviations

None.

## Unverified items

- Source and target feature-bank paths and hashes:
  `OPEN_PENDING_RUNTIME_IDENTITY_BINDING`.
- Accepted Full Linear fitted-artifact paths and hashes:
  `OPEN_PENDING_RUNTIME_IDENTITY_BINDING`.
- Runtime zero-step equivalence, connector gradients, and frozen-state checks
  await separately approved implementation and execution.
- T003A independent acceptance remains pending. Canonical integration merge
  remains `UNVERIFIED`; historical T002E Codex test counts remain `UNRECOVERED`.

## Remaining risks

The design has not been fit or empirically evaluated. Its scientific effect on
Validation functional compatibility is unknown. The future training boundary
must pass the frozen zero-step, gradient, model-state, identity, and Test-access
gates before any result can be interpreted.

## Suggested next action

Submit the pushed T003A design freeze for independent review.
