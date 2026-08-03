# VEA-P1-T003B-A-R3R2 — Apple clean repository root binding

## Status

`PASS_PENDING_INDEPENDENT_REVIEW`

- Baseline and candidate parent: `464ebf3fdeaf53a16d04838fc67aed725fe5d08c`
- The runbook now requires `VEA_REPO_ROOT` and assigns
  `REPO_ROOT="$VEA_REPO_ROOT"`; it has no default, fallback, or fixed Apple
  repository assignment.
- `READY_FOR_P1_EXTRACTION=false`

## Nine-category runbook result

The original eight R3R1 categories passed unchanged: approved commit, complete
pre-evidence Git cleanliness, stale identity removal, profile binding, source
bindings, implementation bindings, exact Stage A evidence/stop boundary, and
separate Stage B approval/run-ID/no-full-mode. The new ninth test aggregates
repository-root failures and passed: required environment gate, exact assignment,
no fixed/default root, non-link directory validation before evidence creation, and
approved-HEAD plus complete Git cleanliness after root selection.

## Isolated validation

All commands used `core.autocrlf=false`, `PIP_NO_INDEX=1`, offline Hub and
Transformers/Datasets variables, empty `CUDA_VISIBLE_DEVICES`, no bytecode,
no user site, disabled plugin autoload/cache, candidate `src` first, and an external
pytest temporary root.

| Command | Result | Exit |
|---|---:|---:|
| `pytest tests/unit/test_p1_t003b_stage_a_runbook.py` | `9 passed` | 0 |
| `pytest tests/unit/test_p1_runtime_preflight.py` | `87 passed` | 0 |
| `pytest tests/unit/test_p1_feature_extraction.py` | `66 passed` | 0 |
| `pytest tests/integration/test_p1_feature_extraction_cli.py` | `36 passed` | 0 |
| updated test in-memory compile | PASS | 0 |
| `git diff --check --ignore-space-at-eol` | PASS | 0 |

The required total is `198 passed`; the 533-node suite was not run. Production,
profile, manifest, and implementation-binding bytes were unchanged. The final
candidate/delivery source-tree equality is verified by the required staged-tree
comparison during delivery.

## Execution boundary

Apple, GPU, dataset, checkpoint, and model access were `false`. Stage A and Stage B
execution were both `false`; no tmux or runbook command was run. The original dirty
working tree is preserved by external snapshot and byte-digest comparison. Future
commit SHA values are intentionally not recorded here.
