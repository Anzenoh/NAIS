# VEA-P1-T003B-B-R1R — Preserve bounded worker failure diagnostics

- Mode: `implementation`
- Task class: `STANDARD`
- Evidence level: `COMPACT`
- Package required: `false`
- Recommended Codex model: `GPT-5.6 Terra`
- Reasoning effort: `high`
- Goal: Preserve a safe, bounded child-worker error classification in canonical failure evidence so the next independently authorized smoke can expose the actual CLIP failure without leaking raw process output.
- Baseline: accepted `VEA-P1-T003B-A-R2` on `codex/vea-p1-t003b-a-r2 @ ff75f06e34ffd05e86844e07a3abb306d34d547f`
- Git delivery: `COMMIT_AND_PUSH`
- Task branch: `codex/vea-p1-t003b-b-r1r`
- Commit condition: `PASS_ONLY`
- Commit message: `fix(extraction): VEA-P1-T003B-B-R1R preserve worker failure diagnostics`
- Push target: `origin HEAD`

## Correction provenance

The predecessor `VEA-P1-T003B-B-R1` stopped before mutation because its Task specification incorrectly required `main @ e518a51ba83f70b42611cc4386a9f55dc9e97de3`.

The accepted Stage A successor is actually:

```text
branch=codex/vea-p1-t003b-a-r2
HEAD=ff75f06e34ffd05e86844e07a3abb306d34d547f
relation_to_e518a51=ahead_by_2
staged_paths=0
```

Do not switch back to `main` or `e518a51…`. That would omit the committed Stage A implementation and repair work. The prior B-R1 attempt changed, staged, committed, pushed, and executed nothing.

## Observed Stage B failure

```text
run_id=20260801T193902Z_p1_1000_three_encoder_e518a51
model_key=clip_vit_b16
lane_id=primary_lane
failure_stage=worker_process
raw_return_code=4
stable_exit_code=5
sanitized_message=worker_process_failed
completed_models=0
ready_for_p1_extraction=false
```

The control artifacts are internally consistent. The parent runner captures child stdout and stderr but currently discards them and writes only the generic message. Do not claim the underlying CLIP cause has already been identified.

## Preconditions

Before any repository mutation:

- Current branch is `codex/vea-p1-t003b-a-r2`.
- `git rev-parse HEAD` equals `ff75f06e34ffd05e86844e07a3abb306d34d547f`.
- `git merge-base --is-ancestor e518a51ba83f70b42611cc4386a9f55dc9e97de3 HEAD` succeeds.
- Staged path count is zero.
- `codex/vea-p1-t003b-b-r1r` does not exist locally or remotely.
- Existing unrelated dirty paths can remain untouched and safely separated.
- Verify these accepted source identities:

```text
ce709105c14595acbc585692185d624d8101b57f14f520ba527263de33ba2607 scripts/extract_features.py
2c754057f59ccdd4cfc07794816184c2392af583fe26e155183581e2f66f4ab8 src/vision_encoder_alignment/p1_feature_extraction.py
0bc992b5596c73e7e9cc5d2138f85296b4050d39e0290c535648444cdfdd2ed6 reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md
```

If any precondition fails, make no repository changes and report `BLOCKED`.

## Read

1. `AGENTS.md`
2. `.codex/ACTIVE_TASK.md`
3. `.codex/PROJECT_STATE.json`
4. `scripts/extract_features.py`
5. `src/vision_encoder_alignment/p1_feature_extraction.py`
6. `docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md`
7. `tests/integration/test_p1_feature_extraction_cli.py`
8. `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`

Do not scan historical tasks, reports, artifacts, or unrelated repository paths.

## Allowed changes

- `scripts/extract_features.py`
- `tests/integration/test_p1_feature_extraction_cli.py`
- `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
- `tasks/VEA-P1-T003B-B-R1R-worker-failure-diagnostic-preservation.md`
- `reports/runs/VEA-P1-T003B-B-R1R-worker-failure-diagnostic-preservation.md`
- `.codex/ACTIVE_TASK.md` only for the final `NO_ACTIVE_TASK` operational state

## Commit paths

- `scripts/extract_features.py`
- `tests/integration/test_p1_feature_extraction_cli.py`
- `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
- `tasks/VEA-P1-T003B-B-R1R-worker-failure-diagnostic-preservation.md`
- `reports/runs/VEA-P1-T003B-B-R1R-worker-failure-diagnostic-preservation.md`

Do not stage `.codex/ACTIVE_TASK.md` or unrelated dirty paths.

## Non-goals and protected scope

- Do not guess or directly fix the underlying CLIP runtime failure.
- Do not modify extractors, runtime profiles, checkpoint profiles, datasets, manifests, or `p1_feature_extraction.py`.
- Do not run Apple, CUDA, checkpoint, dataset, smoke, retry, resume, or full extraction.
- Do not install packages, access the network, download files, or access credentials.
- Do not switch to, reset to, or modify `main`.
- Do not create a review ZIP, detached SHA, package identity, repository-tree manifest, or full protected-tree hash comparison.
- Do not force-push, amend, rebase, merge, or open or approve a PR.

## Requirements

1. Add a bounded parser for a failed worker's captured stdout.
2. Accept child diagnostics only when stdout is one exact canonical JSON object with:
   - the exact expected error-document fields;
   - the expected schema ID;
   - `ready_for_p1_extraction=false`;
   - `exit_code` exactly equal to the subprocess return code;
   - one bounded, printable, single-line error string.
3. Reject diagnostics containing:
   - absolute paths or path separators;
   - control characters or multiple lines;
   - sensitive-key terms;
   - excessive length;
   - extra or missing fields;
   - mismatched exit codes;
   - malformed or noncanonical JSON.
4. For an accepted child error, store an actionable bounded value in the existing `sanitized_message` field.
5. Never add raw stdout or stderr to failure evidence.
6. For rejected or unavailable child diagnostics, preserve `worker_process_failed`.
7. Preserve:
   - parent stable exit code `5`;
   - bounded `raw_return_code`;
   - exact model, lane, shard, and image binding;
   - append-only evidence behavior;
   - no-overwrite behavior.
8. Never print or persist raw child stdout, stderr, traceback, dataset path, cache path, checkpoint path, credentials, environment dumps, or secret-like values.
9. Do not retry or resume a failed worker automatically.
10. Keep successful-worker and finalization behavior unchanged.
11. Add focused tests for:
    - valid canonical safe diagnostic propagation;
    - malformed JSON;
    - extra or missing fields;
    - exit-code mismatch;
    - path-bearing diagnostic;
    - secret-bearing diagnostic;
    - oversized or multiline diagnostic;
    - arbitrary nonempty stderr;
    - generic fallback;
    - unchanged stable and raw exit-code semantics.
12. Update the runbook SHA-256 binding for every changed Apple-transfer source.
13. The Run Report must record:
    - the B-R1 precondition-spec correction;
    - the observed Stage B failure;
    - changed paths;
    - validation commands and exit codes;
    - test count;
    - final Apple transfer-file SHA-256 values.

## Codex validation: Tier 0–1

```text
python -B -m compileall -q scripts/extract_features.py
python -B -m pytest -q tests/integration/test_p1_feature_extraction_cli.py
git -c core.whitespace=cr-at-eol diff --check
```

Also verify:

- `src/vision_encoder_alignment/p1_feature_extraction.py` remains at SHA-256 `2c754057f59ccdd4cfc07794816184c2392af583fe26e155183581e2f66f4ab8`.
- Changed paths are within the exact allowlist.
- The runbook pins the final changed source SHA rather than the pre-task SHA.
- Staged paths exactly equal the commit paths.

Do not run the full suite. If a focused-test dependency makes broader execution strictly necessary, stop and report rather than expanding scope.

## User validation: Tier 2–3

None during this Codex Task.

After independent commit review, ChatGPT will separately authorize Apple transfer, replacement Stage A validation, and one fresh Stage B smoke. Do not provide or execute those operations automatically.

## Git delivery

1. From the verified `ff75f06e34ffd05e86844e07a3abb306d34d547f` baseline, create:

```text
git switch -c codex/vea-p1-t003b-b-r1r
```

2. Implement only the allowed changes.
3. Run mandatory Tier 0–1 validation.
4. Stage only the exact commit paths.
5. Confirm the staged path list exactly matches the commit allowlist.
6. Commit with:

```text
fix(extraction): VEA-P1-T003B-B-R1R preserve worker failure diagnostics
```

7. Push with `git push -u origin HEAD`.
8. Set `.codex/ACTIVE_TASK.md` to `NO_ACTIVE_TASK` after delivery without staging it.

## Evidence and deliverables

- Bounded worker-diagnostic implementation
- Focused integration tests
- Updated Apple runbook source binding
- Compact Task and Run Report
- Baseline SHA, branch, commit SHA, and push result
- Final SHA-256 values for changed Apple-transfer files

## Done when

- Safe child diagnostics are preserved deterministically.
- Unsafe or malformed diagnostics fall back to the generic message.
- Raw stdout, stderr, and sensitive information are never persisted.
- Stable exit, raw return code, failure identity, and no-retry behavior remain intact.
- Mandatory validation passes.
- Exact commit paths are committed and pushed.
- Unauthorized operations and unexpected changes are zero.

## Stop conditions

- Baseline, ancestry, staged count, or source identity mismatch
- Existing local or remote task branch
- Required modification outside allowed paths
- Inability to test safe fallback behavior
- Need for Apple, CUDA, installation, network, credentials, retry, resume, or full extraction
- Unsafe Git separation

## Final response

Return at most 20 lines containing:

1. `Status`, Task ID, and model used
2. Changed paths
3. Tier 0–1 commands, exit codes, and test count
4. Final Apple transfer-file SHA-256 values
5. Baseline SHA and task branch
6. Exact commit message and commit SHA
7. Push result and remote branch
8. Staged count and `git diff --check`
9. Confirmation that Apple execution was not performed
10. Remaining next action: independent commit review

For `PASS` plus successful push, remind the user to submit Task ID, baseline SHA, commit SHA, and remote branch. For `PARTIAL` or `BLOCKED`, return the complete compact final response.

# 이번 작업 정리

## 작업 목적

승인된 Stage A 구현을 기준선으로 유지하면서 worker 실패 원인이 안전하게 기록되도록 보강한다.

## 핵심 수행 내용

- canonical child error만 제한적으로 해석한다.
- 안전한 진단만 failure evidence에 보존한다.
- 비정상적이거나 민감한 출력은 기존 generic 값으로 대체한다.
- focused test 후 새 전용 branch에 commit하고 push한다.

## 이번 작업에서 하지 않는 것

- `main @ e518a51…` 복귀
- CLIP 실패 원인 추측 또는 직접 수정
- Apple smoke, retry, resume 또는 full extraction

## 완료 기준

안전한 오류는 식별 가능하고 위험한 출력은 유출되지 않으며 기존 실패 계약이 유지된다.

## 다음 단계

pushed commit을 독립 검수한 뒤 Apple 전송 여부를 별도로 결정한다.

## 한눈에 보는 작업 요약

이전 명령은 오래된 기준선을 요구해 시작하지 못했습니다. 이번에는 승인된 최신 Stage A 코드를 그대로 이어받아, 다음 실패에서 안전한 원인만 확인할 수 있도록 기록 기능을 보강합니다.