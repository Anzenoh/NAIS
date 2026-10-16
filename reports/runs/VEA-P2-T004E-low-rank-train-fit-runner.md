# Run Report — VEA-P2-T004E

## Status

`PASS` for the bounded software implementation and mandatory Tier 0–1 synthetic checks. Independent review is pending; no real fit was run.

## Metadata

- Task ID: `VEA-P2-T004E`
- Run/report timestamp UTC: `2026-10-16T09:27:04Z`
- Repository root: verified local task workspace; machine-local absolute path omitted
- Git commit before work: `2d665b48504b2d4de5d5d016ad41da256d5a05e0`
- Git dirty before work: only approved untracked `.codex/ACTIVE_TASK.md`; tracked and staged diffs empty
- Execution environment: Windows PowerShell; Python 3.12.0, NumPy 1.26.4, pytest 8.4.2; CPU synthetic fixtures
- Branch: `codex/vea-p2-t004e-low-rank-train-fit-runner`
- Delivery: `COMMIT_AND_PUSH` after required PASS; review disposition `REVIEW_PENDING`

## Goal

Implement and synthetically verify a fail-closed Train-only runner that uses the accepted T004C-R2 binding and T004B numerical core to produce compact low-rank factor artifacts for the two frozen sources. Do not execute real fitting.

## Summary

The public API and CLI require four explicit absolute roots. The output destination must be fresh, absent, outside the repository and input roots, and have an existing parent. The runner calls only `bind_low_rank_runtime` for production inputs, validates the frozen config and provenance, then calls `fit_low_rank_rrr` once per source in SIGLIP2, DINOV3 order without production overrides. The T004B core retains its two-pass replay, Train mean, 4096-row, covariance, eigensolve, and compact-factor numerical controls.

Each nondegenerate rank is written as a factor-only `U`, `V`, `bias` float32 NPZ. Degenerate ranks receive metadata and no file. Files are reopened with `allow_pickle=False`, checked for exact members, shape, dtype, finiteness, SHA-256, array identity, and deterministic application parity. All files are prepared in a private sibling directory, then published by directory rename only after full verification; an existing output destination is rejected. The result JSON records frozen and accepted identities, every source/rank disposition, Train counts, factor shapes and hashes, theoretical parameter bytes, environment, and zero Test/Validation/generation counters. It makes no float32 latency claim because T004B application accumulates in float64.

The private synthetic seam accepts tiny offline geometry/ranks only in tests. The production CLI exposes no rank, split, model, Test, or synthetic override. No new runtime dependency was added.

T004D user-supplied external read-only observations reported real runtime binding and two Train stream passes as PASS at the R2 baseline: 800 ordered images, 156800 rows, 40 chunks per pass and source, with matching ordered image-ID digests. Its pasted final shell line was truncated at `VEA_P2_T004D_PREFLIGHT=PAS`, so this report does not claim independent final shell PASS proof. T004E's PASS below is solely local synthetic code validation. Real Apple RRR fitting remains unrun.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_train_fit_runtime.py` | New Train-only runner and compact artifact verification | Bind accepted identities, invoke reviewed core, publish immutable results |
| `scripts/run_query_adapter_low_rank_train_fit.py` | New explicit-root CLI | Expose only the approved production arguments |
| `tests/unit/test_p2_query_adapter_low_rank_train_fit_runtime.py` | New offline synthetic and adversarial tests | Verify result inventory, provenance, replay, compact storage, collisions, split isolation |
| `tests/integration/test_p2_query_adapter_low_rank_train_fit_cli.py` | New CLI boundary tests | Verify explicit roots, public binder route, forbidden overrides and bad-root rejection |
| `reports/runs/VEA-P2-T004E-low-rank-train-fit-runner.md` | This report | Record software evidence and limits |

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Git entry | `git rev-parse`, `git remote -v`, `git ls-remote`, worktree/lock checks, `git --no-optional-locks status --porcelain=v1 -uall`, both quiet diffs, untracked listing | 0 for positive gates; intended absent T004E ref probe 1 | Exact R2 baseline and remote, only Active Task residual, no locks | Transient command output |
| Tier 0 syntax | `py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_train_fit_runtime.py scripts/run_query_adapter_low_rank_train_fit.py tests/unit/test_p2_query_adapter_low_rank_train_fit_runtime.py tests/integration/test_p2_query_adapter_low_rank_train_fit_cli.py` | 0 | Four new Python files compiled | Command output |
| Tier 1 new suites | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_train_fit_runtime.py tests/integration/test_p2_query_adapter_low_rank_train_fit_cli.py` | 0 | 38 passed in 0.55s | New test files |
| Tier 1 predecessor regression | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py` | 0 | 82 passed in 8.32s | Existing tests |
| Negative controls | New test cases in the 38-test suite | 0 | Bad roots, accepted hash/identity drift, Test counters, stream replay, rank/shape/dtype/finiteness, malformed or duplicate NPZ members, duplicate paths, collisions, forbidden CLI options/split calls rejected | New test files |

## Test and command transcript summary

```text
py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_train_fit_runtime.py scripts/run_query_adapter_low_rank_train_fit.py tests/unit/test_p2_query_adapter_low_rank_train_fit_runtime.py tests/integration/test_p2_query_adapter_low_rank_train_fit_cli.py
exit 0
py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_train_fit_runtime.py tests/integration/test_p2_query_adapter_low_rank_train_fit_cli.py
exit 0; 38 passed in 0.55s
py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py
exit 0; 82 passed in 8.32s
```

## Artifacts

- This run report is the only repository artifact. Synthetic NPZ/result fixtures existed only in pytest temporary directories.
- Production result schema: `vea.p2.query_adapter_low_rank_train_fit_runtime.v1`; no real run ID or real factor checksum exists.
- No model weight, dataset, feature bank, real adapter, GPU result, Validation metric, or Test artifact was produced.

## Deviations

- `.codex/PROJECT_STATE.json` was absent at entry; no replacement was created.
- The runner requires an absent output path, a stricter reading of the task's fresh-output requirement; existing empty directories are rejected as well as populated ones.
- No frozen scientific design, numerical core, binder, or approved file-scope deviation.

## Unverified items

- Actual Apple Train RRR fitting and production output publication were not run. No real factors or scientific results were obtained.
- No Validation evaluation, fair float32 adapter-only timing, connector/LLM/generation, GPU, or Test access was performed. Full repository test suite was not run.
- The T004D user-supplied pasted final shell line was truncated; its exact final exit status was not established here.

## Remaining risks

- Implementation: real production resource use and filesystem behavior remain unverified by the tiny offline fixtures. Directory publication depends on same-filesystem rename semantics.
- Environment: the future Apple execution requires separately approved roots and a fresh external output destination.
- Scientific interpretation: synthetic mapping and storage checks imply no low-rank Validation fidelity, latency, rank selection, or Test outcome.

## Suggested next action

Submit the pushed T004E change for independent review. Do not start real fitting before separate approval.
