# Run Report — VEA-P2-T004G

## Status

`PASS` for the bounded software implementation and required synthetic Tier 0–1 checks. Independent review is pending. No real Validation evaluation was executed.

## Metadata

- Task ID: `VEA-P2-T004G`
- Run/report timestamp UTC: `2026-10-19T12:13:55Z`
- Repository root: verified local task workspace; machine-local absolute path omitted
- Git commit before work: `8c0beb8a88cc5bdb1225c50ed62a7ea0349ef06e`
- Git dirty before work: only approved untracked `.codex/ACTIVE_TASK.md`; tracked and staged diffs empty
- Execution environment: Windows PowerShell; Python 3.12.0, NumPy 1.26.4, pytest 8.4.2; CPU synthetic fixtures only
- Branch: `codex/vea-p2-t004g-validation-patch-identity`; delivery `COMMIT_AND_PUSH`; review `REVIEW_PENDING`

## Goal

Implement a read-only Validation patch and same-image identity evaluation path for the frozen low-rank grid and accepted Full Linear OLS controls, with no real Apple execution or rank selection.

## Summary

Added a distinct Validation binding that reuses the reviewed T004C-R2 ZIP_STORED/ZIP64 NPY parser and accepted T001D/T002D lineage. It decodes only manifest-approved Validation patch ranges; Train and Test patch values, globals, and connectors are not decoded. Whole-file bank hashes remain opaque. The existing Train-only public API and Test refusal remain unchanged.

The new explicit-root runner pins the user-supplied T004F result SHA-256, checks the exact T004E schema/status/provenance and all 12 frozen rank records, then checks every compact factor's path, hash, member inventory, scalar count, shape, dtype and finiteness. It uses the accepted OLS weights and Train means unchanged. Each source is evaluated against native TARGET_CLIP preconnector patches on the same ordered Validation IDs, with the existing equal-image-weighted `patch_metrics` and `same_image_retrieval` functions. The output contains two ordered source blocks, seven fixed conditions per source, per-image patch scores and retrieval ranks, aggregate cosine/NMSE/R² and R@1/R@5, identity hashes, and factor SHA provenance. It contains no raw feature tensor or machine-local absolute path. Output is staged and published only to a fresh external destination.

All conditions remain `UNSELECTED`. Connector, generation and float32 adapter latency fields are `NOT_EVALUATED`; there is no bootstrap or threshold qualification. The reviewed T004B `apply_low_rank` uses float64 arithmetic and supplies no fair float32 latency evidence. No new dependency was added.

The T004F Apple Train-only observations and result SHA in the Active Task are user-supplied external evidence: launcher `PASS_TRAIN_ONLY_LOW_RANK_FIT`, 800 Train images, six fitted ranks per source, 768 positive eigenvalues per source, artifact verification PASS, shell exit 0, and result SHA-256 `84549208e6e7a097e769b89b254bcf1bf939b1e29a3c06185a77f0afbdd858f3`. T004G did not read or independently rehash that Apple root. The PASS below is solely synthetic code validation, not real scientific performance.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py` | Add Validation-specific reader, factory and binding | Select only approved Validation patch byte ranges using the existing strict parser |
| `src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_validation_metrics_runtime.py` | New artifact verification and patch/identity evaluation runtime | Evaluate fixed OLS plus six ranks per source without selection |
| `scripts/run_query_adapter_low_rank_validation_metrics.py` | New five-root CLI | Expose only the approved production inputs/output |
| `tests/unit/test_p2_query_adapter_low_rank_validation_metrics.py` | New offline synthetic/adversarial tests | Verify metric controls, mixed-unit privacy, artifact and identity gates |
| `tests/integration/test_p2_query_adapter_low_rank_validation_metrics_cli.py` | New CLI boundary tests | Verify fixed arguments and root rejection |
| `reports/runs/VEA-P2-T004G-validation-patch-identity-metrics.md` | This report | Record evidence and limits |

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Git entry identity and scope | Root/Git/common, branch/HEAD, origin fetch/push and remote T004E, absent T004G local/remote branch, worktree, status, both quiet diffs, untracked listing, lock/operation probes | 0 for positive gates; absent-ref probe 1 as expected | Exact approved baseline, sole Active Task residual, no relevant locks/operations | Transient command output |
| Environment | `py -3 -c "import sys,numpy,pytest; print(sys.version.split()[0], numpy.__version__, pytest.__version__)"` | 0 | `3.12.0 1.26.4 8.4.2` | Command output |
| Tier 0 syntax | `py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_validation_metrics_runtime.py scripts/run_query_adapter_low_rank_validation_metrics.py tests/unit/test_p2_query_adapter_low_rank_validation_metrics.py tests/integration/test_p2_query_adapter_low_rank_validation_metrics_cli.py` | 0 | Five Python files compiled | Command output |
| Tier 1 new tests | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_validation_metrics.py tests/integration/test_p2_query_adapter_low_rank_validation_metrics_cli.py` | 0 | 40 passed in 1.21s | New tests |
| Tier 1 predecessor regression | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py tests/unit/test_p2_query_adapter_low_rank_train_fit_runtime.py tests/integration/test_p2_query_adapter_low_rank_train_fit_cli.py tests/unit/test_p2_query_adapter_linear.py` | 0 | 137 passed in 11.69s | Existing tests |
| Adversarial controls | Parametrized cases in new suite | 0 | Mixed Test/Train/Validation bytes, Test/global/connector byte exclusion, unknown NPZ layout, hash drift, wrong shape/dtype, result/factor SHA and manifest drift, missing/extra/duplicate NPZ members, fake/duplicate ranks, source/order mismatch, bad roots, output collision, zero cosine denominator, forbidden CLI options and Test split rejected | New tests |

## Test and command transcript summary

```text
py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_runtime_binding.py src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_validation_metrics_runtime.py scripts/run_query_adapter_low_rank_validation_metrics.py tests/unit/test_p2_query_adapter_low_rank_validation_metrics.py tests/integration/test_p2_query_adapter_low_rank_validation_metrics_cli.py
exit 0
py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_validation_metrics.py tests/integration/test_p2_query_adapter_low_rank_validation_metrics_cli.py
exit 0; 40 passed in 1.21s
py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py tests/unit/test_p2_query_adapter_low_rank_runtime_binding.py tests/integration/test_p2_query_adapter_low_rank_runtime_binding_contract.py tests/unit/test_p2_query_adapter_low_rank_train_fit_runtime.py tests/integration/test_p2_query_adapter_low_rank_train_fit_cli.py tests/unit/test_p2_query_adapter_linear.py
exit 0; 137 passed in 11.69s
```

## Artifacts

- This report is the only repository artifact. Synthetic fit NPZ and result fixtures existed only in pytest temporary directories.
- New production result schema: `vea.p2.query_adapter_low_rank_validation_metrics.v1`; no real T004G result ID or checksum exists.
- No feature tensor, dataset, model weight, real metric, GPU/connector/generation output, or Test artifact was produced.

## Deviations

- `.codex/PROJECT_STATE.json` was absent at entry and was not created.
- No scientific, API, scope, dependency or delivery deviation.

## Unverified items

- The real T004F result/factor bytes were not accessed in this task; their identities are pinned for a later separately approved run.
- Real Apple Validation metric values, connector metrics, native exact match, Token F1, fair float32 latency, selection and Test behavior were not measured. No full-repository test-suite claim is made.

## Remaining risks

- Implementation: production bank/factor I/O and output publication have not been exercised against real Apple files; synthetic fixtures cannot establish real filesystem throughput.
- Environment: a future real CPU-only run requires independently reviewed T004G code and separately approved exact roots and output destination.
- Scientific interpretation: patch and retrieval proxy metrics alone cannot promote a rank over OLS or establish frozen-VLM fidelity.

## Suggested next action

Submit the pushed T004G change for independent review before any separately approved real Validation assessment.
