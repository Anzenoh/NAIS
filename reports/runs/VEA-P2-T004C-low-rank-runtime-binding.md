# Run Report — VEA-P2-T004C

## Status

`PASS` — required Tier 0–1 implementation checks. Delivery and independent review are separate.

## Metadata

- Task ID: VEA-P2-T004C
- Run/report timestamp UTC: 2026-10-15T14:44:18Z
- Repository root: verified local task workspace; machine-local absolute path omitted
- Git commit before work: `badada121d925c44972e6b66ae1ce93b61f05634`
- Git dirty before work: only the approved untracked `.codex/ACTIVE_TASK.md`; tracked and cached diffs empty
- Execution environment: Windows PowerShell; Python 3.12.0; NumPy 1.26.4; pytest 8.4.2; CPU synthetic fixtures

## Goal

Bind the frozen T004A low-rank design and T004B numerical core to accepted T001D bank and T002D Full Linear input identities, with a lazy Train-only handoff and no real experiment.

## Summary

- Added an import-safe, read-only `bind_low_rank_runtime(repository_root, t001d_run_root, t002d_validation_root)` API. All roots must be explicit existing absolute directories.
- Verified the canonical LF-normalized tracked SHA-256 of the T004A config, predecessor linear config, bank binding, and historical test-binding metadata. Physical T001D/T002D artifact bytes use separate SHA-256 checks. T004A config validation is delegated to `load_low_rank_config`.
- Reconstructed accepted bank digests from the bound T001D plan, ordered image IDs, model/tap/VLM identity projections, and every physical unit hash. Train patch arrays are loaded lazily, paired by image ID, finite float32, split by whole image, and bounded to 4096 patch rows per emitted chunk. A Train-bearing NPZ unit above that cap fails closed because NumPy cannot selectively decompress rows from it.
- Loaded only the two accepted Full Linear `.npz` weights and means after result, manifest, path-containment, SHA, dtype, shape, finite, and persisted runtime-identity checks. These NumPy arrays are copied out of the archives and exposed to a future authorized T004B caller. No fit was invoked.
- Test-only bank units are hashed as opaque bytes solely for accepted lineage. Their payloads are never opened. Any physical unit mixing Train and Test is rejected before `np.load`. A mixed Train/Validation unit may decompress Validation bytes but yields only Train rows; no Validation data is used for fitting or selection here.
- The T004B `apply_low_rank` float64 accumulation followed by float32 output supplies no evidence for a future float32 adapter-only latency criterion. No latency measurement was run.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py` | New module | Accepted lineage, OLS, Train stream, and Test guard |
| `tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py` | New tests | Synthetic identity, payload, path, split, replay, and guard adversaries |
| `tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py` | New tests | Public API and 4096-row stream contract |
| `reports/runs/VEA-P2-T004C-low-rank-runtime-binding.md` | New report | Reproducible implementation evidence |

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Entry Git gates | root/Git/common dir, branch/HEAD/origin/remote baseline, branch absence, worktrees, operations/locks, diff and untracked checks | 0 each | Exact T004B parent; only approved Active Task residual | Transient command output |
| Required syntax | `py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py` | 0 | Passed | Three allowlisted Python files |
| Required new tests | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py` | 0 | 21 passed in 3.60s | Two new test files |
| Required T004B regression | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py` | 0 | 31 passed in 0.34s | Existing T004B test file |
| Optional older binding suite | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_connector_aware_runtime_binding.py` | 1 | 3 passed, 16 failed at prior connector-aware accepted source SHA check; no new module appears in failing stack | Transient pytest output |

## Test and command transcript summary

```text
py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py
exit 0
py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py
exit 0; 21 passed in 3.60s
py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py
exit 0; 31 passed in 0.34s
py -3 -m pytest -q tests/unit/test_p2_query_adapter_connector_aware_runtime_binding.py
exit 1; 3 passed, 16 failed (pre-existing source SHA check)
```

## Artifacts

- This report. No feature, adapter, model, generation, metric, or GPU artifact was produced.
- Synthetic fixture `.npz` and JSON files existed only in pytest temporary directories; their identities simulate accepted external roots.
- No new runtime dependency.

## Git handoff

- Parent baseline: `badada121d925c44972e6b66ae1ce93b61f05634` on the verified local workspace and origin T004B branch.
- Task branch: `codex/vea-p2-t004c-low-rank-runtime-binding`; delivery contract: `COMMIT_AND_PUSH`, exact four paths in this report, after required PASS gates.
- Entry residual: approved untracked `.codex/ACTIVE_TASK.md`; no other entry content changes or relevant write locks.
- Review disposition: `REVIEW_PENDING`; no merge or self-acceptance. Final commit and remote-ref verification are recorded in the transient task handoff after delivery.

## Deviations

- `.codex/PROJECT_STATE.json` was absent; no file was invented or changed.
- The optional connector-aware suite failed at its existing source-file SHA check. This task did not alter that module or its accepted config, and its failures are outside the new binding path.

## Unverified items

- Actual T001D/T002D roots, physical accepted bank bytes, actual OLS artifact bytes, completion markers, and real Train stream were not accessed or verified in this task.
- Real fitting, Validation metrics, rank selection, generation, models, GPU, Test payload, and latency remain unrun and unauthorized here.
- Unit-wide NPZ decompression may include Validation tensor bytes if a Train/Validation unit is mixed; those rows are never emitted to the Train factory. Train/Test mixed units block before payload loading.

## Remaining risks

- Implementation: a real Train-bearing T001D unit larger than 4096 patch rows will block rather than load the oversized NPZ. Identity hashes are checked before and after Train payload reads; concurrent file replacement beyond those checks remains an external storage race.
- Environment: the older connector-aware binding suite currently fails independently on its source-file SHA gate in this checkout.
- Scientific interpretation: synthetic checks prove software boundaries only; they establish no low-rank adapter quality, native-output fidelity, or float32 latency.

## Suggested next action

Submit the pushed T004C change for independent review before a separately approved real Train-only RRR fitting and Validation preflight task.
