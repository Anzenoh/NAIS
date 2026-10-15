# Run Report — VEA-P2-T004C-R1

## Status

`PASS` — mandatory Tier 0–1 synthetic implementation checks; independent review pending.

## Metadata

- Task ID: VEA-P2-T004C-R1
- Run/report timestamp UTC: 2026-10-15T17:26:10Z
- Repository root: verified local task workspace; machine-local absolute path omitted
- Git commit before work: `d90fbbf225933f668838828b9df31683cb8d9a4c`
- Git dirty before work: only approved untracked `.codex/ACTIVE_TASK.md`; tracked and staged diffs empty
- Execution environment: Windows PowerShell, Python 3.12.0, NumPy 1.26.4, pytest 8.4.2, CPU synthetic fixtures

## Goal

Correct T004C's mixed Train/Test feature-unit incompatibility by reading only Train patch rows from accepted `np.savez` units, without consuming Test feature values or changing the frozen Test policy.

## Summary

- Independently reconstructed the approved manifest through T001D `build_unit_inventory(rows, 16)`: 1,000 image metadata rows (800 Train, 100 Validation, 100 Test), 63 feature units per bank, 51 Train/Test mixed units per bank, and 621 Train images within those mixed units. The legacy T004C mixed-unit guard therefore necessarily blocked the accepted layout; its targeted baseline rejection test passed before mutation.
- Replaced the mixed-unit rejection with a narrow reader for the T001D worker's `np.savez` ZIP_STORED layout. It validates end-of-central-directory and local ZIP64 records, exact member names, no encryption/compression, contiguous nonoverlapping member ranges, metadata IDs and plan ID, and the C-order little-endian float32 patch NPY header and approved shape. Unknown layouts fail closed.
- The reader seeks to manifest-selected Train row offsets and reads only those patch bytes. It does not load whole `patches.npy`, Test patch rows, globals, or connector tensor values. Test-only units remain opaque-hash-only. Bank and OLS lineage, safe paths, before/after artifact SHA, image pairing, replay digest, and 4096-row output cap remain in force.
- A realistic synthetic 16-image unit contains Train, Validation, and Test rows. Tests independently calculate forbidden Test patch and target connector byte ranges from the ZIP file and record every selective read; no read intersects those ranges. Two factory passes return the same ordered Train rows. Other synthetic tests reject compressed/encrypted/extra-member NPZ, wrong image/plan metadata, Fortran/shape/dtype errors, hash drift, and replay drift.
- No fit, real artifact load, model/GPU call, generation, Test metric, or policy change occurred. No new runtime dependency was added; the parser uses the Python standard library and existing NumPy.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py` | Edited | Selective Train-row NPZ reader for mixed units |
| `tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py` | Edited | Mixed-unit factory regression and guards |
| `tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py` | Edited | Realistic 16-image byte-range and adversarial NPZ tests |
| `reports/runs/VEA-P2-T004C-R1-mixed-unit-train-reader.md` | New | This evidence report |

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Git entry gates | root/Git/common dir, branch/HEAD/origin/remote baseline, R1 branch absence, worktrees, operations/locks, tracked/cached diff and untracked checks | 0 for successful gates; absent ref/operation probes 1 as expected | Exact T004C baseline; sole approved Active Task residual | Transient command output |
| Manifest reconstruction | Inline Python reading only manifest JSON metadata and calling `build_unit_inventory(rows, 16)` | 0 | 63 units/bank; 51 Train/Test mixed; 621 Train images in mixed units/bank | Transient command output |
| Legacy failure demonstration | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py::test_mixed_train_test_unit_refused_before_np_load` | 0 | 1 passed before implementation; confirms old guard rejects mixed units | Baseline test |
| Required syntax | `py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py` | 0 | Passed | Three Python paths above |
| Required binding tests | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py` | 0 | 31 passed in 4.20s | Two edited test paths above |
| Required numerical-core regression | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py` | 0 | 31 passed in 0.33s | Unchanged T004B test |

## Test and command transcript summary

```text
Manifest metadata reconstruction: exit 0; rows 1000, split 800/100/100,
  feature units per bank 63, Train/Test mixed units per bank 51,
  Train images inside mixed units per bank 621, max unit size 16.
Legacy mixed-unit rejection test before edit: exit 0; 1 passed.
Interim binding run after code edit: exit 1; 19 passed, 2 old-test assumptions failed.
Updated old tests for the selective reader, then final required commands:
compileall: exit 0.
binding unit + integration pytest: exit 0; 31 passed in 4.20s.
unchanged T004B numerical-core pytest: exit 0; 31 passed in 0.33s.
```

## Artifacts

- This report only. Synthetic `.npz` files lived in pytest temporary directories and were not committed.
- No model, feature-bank, adapter, metric, Test, or GPU artifact was produced.

## Git handoff

- Baseline: `d90fbbf225933f668838828b9df31683cb8d9a4c` on the verified local workspace and origin T004C branch.
- R1 branch: `codex/vea-p2-t004c-r1-mixed-unit-train-reader`; delivery contract `COMMIT_AND_PUSH` after PASS, exact four paths listed above.
- Entry residual: approved untracked `.codex/ACTIVE_TASK.md`; no relevant write locks.
- Review disposition: `REVIEW_PENDING`; no merge or self-acceptance. Final commit and remote-ref verification are provided in the transient task handoff.

## Deviations

- `.codex/PROJECT_STATE.json` was absent; no state file was invented or changed.
- No frozen config, predecessor module, or previous T004C report was edited.
- The previous T004C report records optional connector-aware suite failures at its own source SHA gate. That suite is outside the R1 reader path and was not rerun; this report makes no full-repository regression claim.

## Unverified items

- Physical accepted T001D/T002D roots, real unit bytes, real Full Linear artifacts, and a real Train/Validation runtime remain unaccessed and unverified.
- The supported layout is the observed T001D `np.savez` ZIP_STORED format with ZIP64 local headers. A different producer layout blocks pending separate review.
- No fitting, Validation selection, generation, latency, GPU, Test tensor, or Test evaluation was performed.

## Remaining risks

- Implementation: opaque whole-file SHA reads include incidental bytes of mixed units but never decode or expose Test numeric rows. A concurrent file replacement between hash checks remains a storage race.
- Environment: actual accepted unit layout has not been opened in this task; another NumPy/ZIP variant may fail closed.
- Scientific interpretation: synthetic access and identity checks establish no adapter quality, native-output fidelity, or float32 latency.

## Suggested next action

Submit the pushed T004C-R1 correction for independent review before any separately approved real Train-only runtime work.
