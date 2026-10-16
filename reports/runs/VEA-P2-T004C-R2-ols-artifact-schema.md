# Run Report — VEA-P2-T004C-R2

## Status

`PASS` — mandatory Tier 0–1 synthetic implementation checks; independent review pending.

## Metadata

- Task ID: VEA-P2-T004C-R2
- Run/report timestamp UTC: 2026-10-16T09:06:16Z
- Repository root: verified local task workspace; machine-local absolute path omitted
- Git commit before work: `35f45b0368c6e524d94856d6042a1bdc4e609924`
- Git dirty before work: only approved untracked `.codex/ACTIVE_TASK.md`; tracked and staged diffs empty
- Execution environment: Windows PowerShell, Python 3.12.0, NumPy 1.26.4, pytest 8.4.2; CPU synthetic fixtures

## Goal

Correct the low-rank binder's Full Linear NPZ member check to accept the authentic T002D producer schema while preserving accepted artifact hashes, lineage, Train-only access, and Test isolation.

## Summary

- The reviewed Apple T004D read-only preflight stopped at the binder's `SIGLIP2 Full Linear fields differ` check. The user confirmed six authentic NPZ members by name only: `rank.npy`, `residual_sum_squares.npy`, `runtime_identity.npy`, `source_mean.npy`, `target_mean.npy`, and `weights.npy`. T004D did not pass, and no real numeric NPZ payload was independently verified here.
- Historical `FullLinearFit` defines `weights`, `source_mean`, `target_mean`, `rank`, and `residual_sum_squares`; T002D `_fit_arrays()` serializes all five, and its `np.savez()` adds `runtime_identity`. The binder had allowed only four members. The production code change is confined to `_ols`: require exactly the six `.npy` names without duplicates, accept integer scalar rank 0..768, and accept finite float32 1-D residuals of length 0 or 768 (nonempty only at full rank). Neither extra field enters `AcceptedOLS` or selection.
- The synthetic OLS fixture now uses `FullLinearFit` plus the historical `_fit_arrays` serialization for both SIGLIP2 and DINOV3. Positive tests verify copied weights and means and empty residuals at deficient rank. Negative tests reject missing, extra, duplicate, unsafe-name, malformed rank/residual, persisted-identity, fit-manifest, and physical-hash drift.
- Existing result SHA, status/schema/runtime identity, Test/generation counters, accepted OLS SHA, fit manifest, path containment, `allow_pickle=False`, finite float32 weights/means, and the R1 selective Train reader remain in force. No real artifact, model, GPU, fit, generation, latency, or Test payload was accessed.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py` | Edited `_ols` only | Accept and validate exact historical six-member OLS archive |
| `tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py` | Edited | Producer-format fixture and focused positive/adversarial regressions |
| `reports/runs/VEA-P2-T004C-R2-ols-artifact-schema.md` | New | This evidence report |

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Git entry gates | root/Git/common dir, branch/HEAD, fetch/push origin, remote R1 baseline, absent R2 branch, worktrees, operations/locks, status, tracked/cached diff and untracked checks | 0 for successful checks; absent ref/operation probes 1 as expected | Exact R1 baseline; sole approved Active Task residual | Transient command output |
| Focused OLS tests | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py` | 0 | 38 passed in 7.78s | Edited unit test |
| Required syntax | `py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py` | 0 | Passed | Two edited Python files |
| Required binding/integration | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py` | 0 | 51 passed in 7.95s | Unit test and unchanged R1 integration test |
| Required T004B regression | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py` | 0 | 31 passed in 0.33s | Unchanged numerical-core test |

## Test and command transcript summary

```text
py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py
exit 0; 38 passed in 7.78s
py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py
exit 0
py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py
exit 0; 51 passed in 7.95s
py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py
exit 0; 31 passed in 0.33s
```

## Artifacts

- This report only. Synthetic `.npz` and result fixtures existed only in pytest temporary directories.
- No accepted artifact, model, feature bank, adapter, metric, Test, or GPU artifact was created or changed.
- No new runtime dependency.

## Git handoff

- Baseline: `35f45b0368c6e524d94856d6042a1bdc4e609924` on verified local workspace and origin R1 branch.
- R2 branch: `codex/vea-p2-t004c-r2-ols-artifact-schema`; delivery contract `COMMIT_AND_PUSH` after PASS, exact three paths listed above.
- Entry residual: approved untracked `.codex/ACTIVE_TASK.md`; no relevant write locks.
- Review disposition: `REVIEW_PENDING`; no merge or self-acceptance. Final commit and remote-ref verification are supplied in the transient task handoff.

## Deviations

- `.codex/PROJECT_STATE.json` was absent; no state file was invented or changed.
- No historical NPZ, T002D writer, frozen config, R1 selective reader, or old report was edited.

## Unverified items

- The six real NPZ member names were user-confirmed; actual `rank`, residual, weight, and mean array values/shapes/dtypes were not read in this task. The authentic T002D artifact may still fail at a later numeric check.
- Apple T004D remains blocked pending independent R2 review and a separately authorized read-only recheck with its expected branch and commit updated. No Apple command was run here.
- No real Train stream, fitting, Validation generation, latency, model/GPU, or Test access was performed.

## Remaining risks

- Implementation: the binder intentionally rejects any OLS member set or auxiliary-field shape outside the documented producer-compatible cases. The R1 file-integrity race caveat is unchanged.
- Environment: the real Apple artifact's numeric field types/shapes remain unverified; no full-repository regression claim is made.
- Scientific interpretation: synthetic schema tests establish no low-rank quality, native-output fidelity, or latency.

## Suggested next action

Submit the pushed R2 correction for independent review before a separately approved T004D read-only preflight with updated expected branch and commit.
