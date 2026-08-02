# VEA-P1-T003A-R1R3 — Extraction Runner Failure Boundary and Apple Preflight Runbook Final Hardening

## 1. Final status

`PASS`

```text
VEA-P1-T003A-R1R3 = PASS
VEA-P1-T003A-R1R2 = SUPERSEDED_BLOCKED_INDEPENDENT_REVIEW
READY_FOR_VEA_P1_T003B_APPLE_PREFLIGHT_EXECUTION = false
READY_FOR_VEA_P1_T003B_10_IMAGE_SMOKE_EXECUTION = false
READY_FOR_P1_FULL_EXTRACTION = false
READY_FOR_P1_EXTRACTION = false
```

This is Codex-authored evidence and is not `ACCEPTED_PASS` until independent review verifies the sealed package.

## 2. Starting repository identity

- Repository: `G:\내 드라이브\2026NAIS`
- Branch: `main`
- HEAD: `e518a51ba83f70b42611cc4386a9f55dc9e97de3`
- Staged paths: `0`
- Starting Git-visible paths: `343`
- Starting `git diff --check`: exit `0`
- Existing dirty worktree was preserved; no reset, restore, checkout, stash, stage, commit, or push was performed.

The reconstructed starting status is preserved externally at `manifests/git-start-status.txt` under the task temporary root.

## 3. Predecessor disposition and package verification

The R1R2 implementation result is retained as predecessor input, but its review disposition is superseded by this bounded repair:

```text
VEA-P1-T003A-R1R2 = SUPERSEDED_BLOCKED_INDEPENDENT_REVIEW
```

Verified predecessor package:

- ZIP: `VEA_P1_T003A_R1R2_INDEPENDENT_REVIEW_20260730.zip`
- Bytes: `131207`
- SHA-256: `b5c5500bb75acef46b8842a8b1fa371fddebe8dcb026deb2c150042f86c582e4`
- Entries: `40`
- Manifested payloads: `39`
- Root manifest SHA-256: `9ab6c30e45317b60f7b09ef1898f381fdd438fe0b203d3903e3a8a5c4a331938`
- Missing, unmanifested, size mismatch, hash mismatch, duplicate, unsafe-name, and link findings: all `0`
- Detached SHA file SHA-256: `881efb0aec02f3faa25615fe271763a810d9a51ae5061235c015c7d475315256`
- Package identity SHA-256: `f8b59571073bea7e3d81d6b7babdc59e91cbe978927b9fe9fd8eae4c1479328e`

The predecessor ZIP, detached SHA, package identity, root manifest, and post-report repository seal are carried into the R1R3 review input.

## 4. Scope and changed files

Task-owned repository changes are limited to:

1. `tasks/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md`
2. `docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md`
3. `src/vision_encoder_alignment/p1_feature_extraction.py`
4. `scripts/extract_features.py`
5. `tests/unit/test_p1_feature_extraction.py`
6. `tests/integration/test_p1_feature_extraction_cli.py`
7. `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
8. `reports/runs/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md`

The standalone validator and production profile were inspected but not modified:

- `scripts/validate_p1_feature_extraction.py`: `7a0e8cfbde2ba8a4c8a76887d28f388d428341520ed404221b2ceba44f4ba4f0`
- `configs/extraction_profiles/p1_1000_apple_fp32_v0.json`: `98d2dfe8e711efdb6f33ac45e87e21318ec53db8b1cc8f6a0aed789babc58cf5`

## 5. Apple runbook corrections

The T003B runbook now separates two authorization stages.

- Stage A is read-only preflight and exact evidence retrieval.
- Stage B is the separately approved ten-image smoke only.
- It requires explicit `VEA_DATASET_ROOT`, `VEA_CHECKPOINT_CACHE`, and `VEA_T003B_EVIDENCE_ROOT` values instead of guessing machine paths.
- Host, Git, predecessor, source identity, and all path checks occur before the first evidence-directory creation.
- Plan and preflight each have exactly one `--output-json` destination and suppress stdout with `/dev/null`; misleading `.stdout.json` artifacts are forbidden.
- The Stage A manifest binds the exact `00` through `05` evidence set before retrieval.
- The runbook contains explicit stop and approval lines between Stage A and Stage B.
- It contains no full-extraction command, automatic retry, automatic resume, cleanup, model/checkpoint download, or dataset mutation.

## 6. All-component path safety

`validate_non_link_path` now validates the complete lexical component chain using an injectable `lstat` probe. It rejects:

- noncanonical or non-absolute lexical spellings;
- symlink, reparse-point, or junction-like components, including ancestors;
- strict-resolution mismatch;
- wrong expected path kind.

The guard is applied to profile and manifest inputs, repository/dataset/cache/runtime roots, worker paths, output parents, protected roots, run initialization/resume/completion/validation, failure-evidence roots, and atomic-publication parents. The contract explicitly records that the checks are fail-closed validation, not a claim of kernel-level TOCTOU elimination.

The Windows host could not create the requested real symlink probe. The single probe recorded `System.InvalidOperationException`, HResult `-2146233079`, and repository delta `0`. Injectable `lstat` tests cover leaf and ancestor reparse behavior deterministically.

## 7. Failure-evidence boundary

Failure evidence is canonical, append-only, no-overwrite, and numbered deterministically. Its schema distinguishes:

- stable orchestration exit code `5`;
- optional bounded signed 32-bit raw worker return code;
- worker-scoped context versus orchestration-scoped null context;
- stages `worker_launch`, `worker_process`, `worker_artifact_reconciliation`, `completion`, and `model_free_validation`.

Partial tensor/sidecar observations are retained in evidence. If evidence publication itself fails, the original execution exception remains primary and receives a diagnostic note; no alternate write location is attempted.

## 8. Resume, completion, and metrics behavior

- Resume rejects finalization residue and the ambiguous state where all shards exist but metrics do not.
- Worker reconciliation happens after a successful worker exit and has its own failure stage.
- Completion first produces a candidate without publishing the final manifest.
- Model-free validation runs against that candidate.
- The final manifest is atomically published last.
- Completion and model-free-validation failures receive orchestration-scoped evidence.

## 9. Contract clarification

The contract now defines the pre-initialization versus post-`run_state` failure boundary, the stable/raw exit distinction, partial-output handling, forbidden resume states, final-manifest publication ordering, all-component path validation, and the T003B Stage A/Stage B authorization boundary.

## 10. Validation environment

All validation was repository-local and offline. Task-owned temporary output was restricted to:

`G:\내 드라이브\VEA_P1_T003A_R1R3_TEMP_20260730`

Environment controls included external `TEMP`, `TMP`, `PYTHONPYCACHEPREFIX`, `PYTHONPATH=src`, Hugging Face/Transformers/Datasets offline flags, and an empty `CUDA_VISIBLE_DEVICES`.

## 11. Commands and exit codes

Final-source validation commands were:

1. `py -3 -B -m compileall -q src\vision_encoder_alignment scripts tests` — exit `0`.
2. `py -3 -B scripts\extract_features.py --profile <absolute-profile> --repository-root <absolute-repository> --mode plan --output-json <external-canonical-plan>` — exit `0`.
3. `py -3 -B -m pytest -q -p no:cacheprovider --basetemp <external-focused> tests\unit\test_p1_feature_extraction.py tests\integration\test_p1_feature_extraction_cli.py` — exit `0`.
4. Exact required fourteen-file legacy regression command with external `--basetemp` — exit `0`.
5. One external Windows symlink capability probe — capability unavailable; repository delta `0`.
6. `py -3 -B -m pytest -q -ra -p no:cacheprovider --basetemp <external-full>` — raw exit `1`, normalized exit interpretation `PASS`.
7. Static forbidden-operation inspection — exit `0`.
8. Protected-file start/end comparison — exit `0`.
9. `git diff --check` — exit `0`.
10. `git diff --cached --name-only` — exit `0`, output empty.

Raw logs are included in the independent-review package.

## 12. Focused and regression results

- Focused unit/integration gate: `89 passed`, `0 failed`, `0 skipped`, exit `0`.
- Exact fourteen-file regression gate: `384 passed`, `0 failed`, `0 skipped`, exit `0`.
- Canonical plan: exit `0`, bound profile SHA-256 matched, `full_batch_size=null`, extraction readiness remained false.

## 13. Full-suite raw and normalized result

The full suite was executed exactly once after the final source state and final compile.

```text
raw exit = 1
passed = 1280
failed = 4
skipped = 2
errors = 0
xfail = 0
deselected = 0
normalized Windows gate = PASS
```

Allowed failures, exactly:

1. `tests/integration/test_dinov3_identity_transaction.py::test_recovery_rejects_journal_temporary_symlink_without_overwrite`
2. `tests/unit/test_dinov3_identity_rebind.py::test_symlink_destination_is_rejected`
3. `tests/unit/test_dinov3_identity_rebind.py::test_transaction_root_symlink_is_rejected`
4. `tests/unit/test_dinov3_semantic_probe_profile.py::test_workspace_resolver_rejects_symlink_component`

Allowed skips, exactly:

1. `tests/unit/test_dinov3_runtime_profile.py::test_workspace_symlink_escape_rejected`
2. `tests/unit/test_dinov3_snapshot_profile.py::test_symlink_and_reparse_paths_rejected`

No new failure, skip, xfail, error, or deselection occurred.

## 14. Static and unauthorized-operation audit

Static inspection found:

- `shell=True`: `0`
- `os.replace`: `0`
- network/package-manager invocations: `0`
- isolated worker `-I` combined with injected `PYTHONPATH`: `0`
- inherited environment-copy worker launch: `0`
- hard-coded Apple dataset/cache roots: `0`
- duplicate plan/preflight redirections: `0`
- full-extraction command in the runbook: `0`
- automatic retry/resume/cleanup in the runbook: `0`
- atomic hard-link calls: `2`
- explicit `shell=False` calls: `2`
- isolated `-B` worker invocations: `4`

Actual unauthorized operation counts were all `0`: network, package installation, SSH/Apple access, dataset/image reads, checkpoint/model access, GPU/CUDA, extraction, and Git mutation.

## 15. Protected-file comparison

The protected manifest covers `365` Git-visible paths outside the seven implementation/task-owned paths and this report. Start/end comparison found:

```text
changed = 0
missing = 0
unexpected = 0
reparse findings = 0
staged paths = 0
```

The start manifest was materialized after active-task creation from the exact recorded precondition inventory and sealed identities; this timing deviation is disclosed below. No protected file changed afterward.

## 16. Git ending state

- Branch: `main`
- HEAD: `e518a51ba83f70b42611cc4386a9f55dc9e97de3`
- Staged paths: `0`
- Expected Git-visible paths after this report: `345`
- `git diff --check`: exit `0`
- No commit, push, branch change, reset, restore, checkout, stash, or staging operation occurred.

## 17. Evidence and package construction

The independent-review package is non-self-referential:

- the ZIP contains the task/report, changed sources/tests/runbook, unchanged bound validator/profile, predecessor package evidence, final logs, manifests, and repository seal;
- the root review manifest binds every ZIP payload other than itself;
- the ZIP is created once after repository sealing;
- the detached SHA-256 file is created after the ZIP;
- the package-identity JSON is created last and is not embedded in the ZIP.

Final ZIP, detached SHA, package identity, root manifest, repository seal, and current task/report SHA-256 values are recorded in the external package identity. They are intentionally not copied back into this report, avoiding post-seal report mutation.

## 18. Deviations

1. Diagnostic focused checks were run during implementation. The reported focused, regression, and full-suite results are final-source runs; the full suite was run exactly once after final source and compile.
2. The protected start manifest was materialized after active-task creation, reconstructed from the exact precondition status and separately verified source identities. It was not falsely presented as a byte snapshot taken earlier.
3. The full-suite raw exit remains `1` because of the exact four approved Windows symlink-environment failures. The normalized gate is separately and explicitly reported as `PASS`.
4. Real symlink creation was unavailable on this Windows host, so deterministic injected component probes provide the functional path-safety coverage.

## 19. Remaining risks

- Independent review has not yet accepted this result.
- No command was executed on `apple`; the Apple runbook remains authoring-only evidence.
- No dataset image, checkpoint, model, GPU, CUDA tensor, smoke extraction, or full extraction was exercised.
- Filesystem component checks cannot eliminate all cross-check/use races; the contract now avoids claiming that they do.

## 20. Readiness and next action

```text
READY_FOR_VEA_P1_T003B_APPLE_PREFLIGHT_EXECUTION = false
READY_FOR_VEA_P1_T003B_10_IMAGE_SMOKE_EXECUTION = false
READY_FOR_P1_FULL_EXTRACTION = false
READY_FOR_P1_EXTRACTION = false
```

Next action: submit the sealed R1R3 independent-review ZIP, detached SHA-256 file, and package-identity JSON for independent review. Do not execute T003B until that review explicitly approves the package.
