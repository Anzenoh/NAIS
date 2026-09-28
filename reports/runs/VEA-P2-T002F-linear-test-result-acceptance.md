# Run Report — VEA-P2-T002F

## Status

`PASS` for the bounded evidence and documentation validation; delivery is
`REVIEW_PENDING`, not independently accepted.

## Metadata

- Task ID: `VEA-P2-T002F`
- Run/report timestamp UTC: `2026-09-28 14:17:26 UTC`
- Repository root: user-selected P2 checkout (machine-local path omitted)
- Git commit before work: `39fbbccd654aa6d0726802b5aaa1377a86f81f6b`
- Git dirty before work: `DIRTY_EXPECTED` — approved `AGENTS.md` unstaged bytes
  and untracked `.codex/ACTIVE_TASK.md`; other porcelain entries had no actual
  tracked or untracked content difference
- Execution environment: Windows PowerShell; Python 3.12 standard library;
  read-only artifact validation; no model, dataset, or GPU execution
- Project state file: `NOT_PRESENT`

## Goal

Bind the T002E implementation commit to its verified real-Test evidence and
synchronize the repository's Phase 2 result wording without rerunning T002E.

## Summary

The baseline commit's parent is
`ccde89c3c89c410c69b3492a6ed53d360d161de9`. Its six-path implementation
diff comprises:

1. `configs/vlm_compat/p2_query_adapter_linear_test_binding_v1.json`
2. `scripts/run_query_adapter_linear_test.py`
3. `src/vision_encoder_alignment/vlm_compat/query_adapter_linear_test_runtime.py`
4. `src/vision_encoder_alignment/vlm_compat/query_adapter_tinyllava_generation.py`
5. `tests/integration/test_p2_query_adapter_tinyllava_generation_boundary.py`
6. `tests/unit/test_p2_query_adapter_linear_test_runtime.py`

The historical remote T002E branch resolves to the baseline commit. The two
raw artifacts have the expected hashes:

| Artifact | SHA-256 |
|---|---|
| `result.json` | `2325d98ce8e694fc00887e02cb293cb7e64358cd5d88cc613a8dedc8f1a7c1ce` |
| `generation_records.jsonl` | `cc07ae2b3372e2b3b7351e269df09b07a46b2654565d9b48c3a9285f96826684` |

The staged `AGENTS.md` is the unchanged, approved entry residual with SHA-256
`a9e11839d000ed72fdbee87f1e7f8e848cc5bba476fe8a96721bcb602c3c061c`.

The generation JSONL parses as 900 valid rows for 100 distinct image IDs, with
exactly nine rows per image: one native and eight adapted. Each of the eight
source/method pairs has 100 rows. Relative paths and `(image_id, source,
method)` keys are unique. Every row has successful generation and restored
injection state. All native rows have `(vision, connector, injection)` calls
`(1, 1, 0)`; all adapted rows have `(0, 1, 1)`. Runtime identity is consistent
across every row and the result JSON. Result counts and the production plan
both equal 100 native, 800 adapted, 900 total. All 800 adapted row-level exact
match and Token F1 values agree with the result JSON; their independently
recomputed means agree for all eight configurations and all 16 aggregate
fields. The means are:

| Source | Method | Exact matches / 100 | Mean Token F1 | Same-image R@1 | Connector cosine |
|---|---|---:|---:|---:|---:|
| DINOv3 | Calibration | 0 | 0.338184 | 0.01 | 0.229 |
| DINOv3 | Orthogonal Procrustes | 0 | 0.393934 | 0.76 | 0.308 |
| DINOv3 | Full Linear | 0 | 0.585957 | 0.86 | 0.534 |
| DINOv3 | selected Ridge | 0 | 0.586269 | 0.86 | 0.534 |
| SigLIP2 | Calibration | 0 | 0.336103 | 0.02 | 0.246 |
| SigLIP2 | Orthogonal Procrustes | 0 | 0.483345 | 0.84 | 0.313 |
| SigLIP2 | Full Linear | 1 | 0.586525 | 0.86 | 0.489 |
| SigLIP2 | selected Ridge | 1 | 0.586915 | 0.85 | 0.489 |

The row validity and call-boundary findings establish recorded software/runtime
execution evidence. They do not prove semantic equivalence to native output.
Scientifically, Full Linear produces a large improvement over Calibration and
Orthogonal Procrustes in the reported alignment and generation measures;
selected Ridge is practically tied with Full Linear. Linear alignment recovers
substantial but incomplete TinyLLaVA functional compatibility. Exact-match
counts are especially limited.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `AGENTS.md` | Stage the pre-existing, user-approved bytes unchanged | Include approved operational protocol revision in the bounded delivery |
| `docs/PROJECT_STATUS.md` | Update stale state and record reviewed evidence boundary | Current status had said no real Test evaluation had run |
| `docs/EXPERIMENT_PROTOCOL_P2_ADAPTER.md` | Replace stale configuration-only execution claim | Distinguish T002A freeze from subsequent T002E execution |
| `docs/VLM_COMPATIBILITY_METRICS.md` | Replace stale planned-generation wording | Record T002E's observed 900 rows |
| `reports/runs/VEA-P2-T002F-linear-test-result-acceptance.md` | Add this run report | Preserve compact validation and interpretation evidence |

`docs/DECISION_LOG.md` required no synchronization: it records approved
experiment decisions and makes no stale T002E result claim.

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Identity and branch entry | `git rev-parse`, `git show-ref`, `git worktree list --porcelain`, `git ls-remote --heads origin` | 0; absent local branch is expected exit 1 | Baseline, parent, free task branch, historical remote verified | Git output |
| Actual residual scope | `git diff --name-only`; `git diff --cached --name-only`; `git ls-files --others --exclude-standard` | 0 each | Only approved `AGENTS.md` diff and task-control untracked file | Git output |
| Raw evidence integrity | `Get-FileHash -Algorithm SHA256`; Python `hashlib.sha256` | 0 | Both exact hashes match | Raw artifacts |
| Structure, boundaries, aggregate agreement | `py -3 -c $code` (standard-library `json`, `pathlib`, `collections`, `hashlib`, `math`; assertions described above) | 0 | 900/100/9, 100+800, 800/800 metric rows, 16/16 aggregates, all boundary assertions pass | Raw artifacts |
| Implementation lineage | `git diff-tree --no-commit-id --name-only -r 39fbbccd654aa6d0726802b5aaa1377a86f81f6b` | 0 | Exactly six paths listed above | Git commit |
| Pre-commit path and whitespace checks | `git diff --name-only`; `git ls-files --others --exclude-standard`; `git diff --check`; `git diff --cached --name-only`; `git diff --cached --check` | 0 each | Five staged allowlist paths; task-control file excluded; no whitespace errors | Git output |

## Test and command transcript summary

```text
Get-FileHash -Algorithm SHA256 -LiteralPath <each exact task evidence path>
  result:     2325d98ce8e694fc00887e02cb293cb7e64358cd5d88cc613a8dedc8f1a7c1ce
  generation: cc07ae2b3372e2b3b7351e269df09b07a46b2654565d9b48c3a9285f96826684
py -3 -c $code
  STRUCTURE 900 rows 100 images {9} rows/image 100 native 800 adapted
  BOUNDARIES native=(1,1,0), adapted=(0,1,1); success/restoration=all; paths and keys=unique
  ROW_CROSSCHECK 800/800 exact match and F1 agree with result JSON
  AGGREGATE_CROSSCHECK 8/8 configurations, 16/16 aggregate fields
  EXIT=0
```

No pytest, model load, dataset execution, GPU execution, or generation was run;
none is assigned to T002F. No runtime dependency was added.

## Artifacts

- This report; result schema `vea.p2.query_adapter_linear_test_runtime.v1`.
- Raw generation schema `vea.p2.query_adapter_linear_generation_unit.v1`.
- Raw evidence remains external and unchanged; no copy is committed.

## Deviations

None in the bounded validation. Post-commit Git delivery checks are reported
in the final handoff.

## Unverified items

- Historical Codex test counts: `UNRECOVERED`.
- Canonical integration merge: `UNVERIFIED`.
- Original Apple artifact paths: unrecovered; current validator paths are not
  substitutes for historical paths.
- Independent result acceptance: `REVIEW_PENDING`.

## Remaining risks

- Implementation: this report verifies the recorded rows and lineage, not a
  rerun of the frozen model stack.
- Environment: original Apple runtime artifact paths are not recovered.
- Scientific interpretation: nonzero residual mismatch and very low native
  exact match limit the functional-compatibility conclusion.

## Suggested next action

Submit the pushed T002F evidence for independent review. Do not merge, close
the Task, or begin another experiment as part of this delivery.
