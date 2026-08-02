# VEA-P1-T003B-B-R1R2 — Reconcile diagnostic task delivery identity

- Mode: `recovery`
- Task class: `STANDARD`
- Evidence level: `COMPACT`
- Package required: `false`
- Recommended Codex model: `GPT-5.6 Terra`
- Reasoning effort: `medium`
- Goal: Establish an unambiguous, append-only provenance record for diagnostic implementation commit `5d10ecc0ad844c33f17a99347de2c6aca76eac03` without modifying its implementation, tests, runbook, or Git history.
- Baseline: `codex/vea-p1-t003b-b-r1 @ 5d10ecc0ad844c33f17a99347de2c6aca76eac03`
- Git delivery: `COMMIT_AND_PUSH`
- Task branch: `codex/vea-p1-t003b-b-r1r2`
- Commit condition: `PASS_ONLY`
- Commit message: `docs(extraction): VEA-P1-T003B-B-R1R2 reconcile diagnostic task identity`
- Push target: `origin HEAD`

## Review disposition

Independent review found no blocking implementation defect in commit `5d10ecc0ad844c33f17a99347de2c6aca76eac03`.

The predecessor remains `PARTIAL_TASK_IDENTITY_MISMATCH` because its committed Task contract specifies:

```text
Task ID: VEA-P1-T003B-B-R1R
Task branch: codex/vea-p1-t003b-b-r1r
Commit message: fix(extraction): VEA-P1-T003B-B-R1R preserve worker failure diagnostics
```

but the actual immutable delivery is:

```text
Task label: VEA-P1-T003B-B-R1
Remote branch: origin/codex/vea-p1-t003b-b-r1
Commit message: fix(extraction): VEA-P1-T003B-B-R1 preserve worker failure diagnostics
Commit SHA: 5d10ecc0ad844c33f17a99347de2c6aca76eac03
```

Do not amend, rebase, force-push, rename, delete, or recreate the delivered commit or remote branch.

## Preconditions

- Current branch is `codex/vea-p1-t003b-b-r1`.
- `HEAD` equals `5d10ecc0ad844c33f17a99347de2c6aca76eac03`.
- `HEAD^` equals `ff75f06e34ffd05e86844e07a3abb306d34d547f`.
- `origin/codex/vea-p1-t003b-b-r1` resolves to the same HEAD.
- Staged path count is zero.
- The new task branch does not exist locally or remotely.
- The only permitted pre-task operational change is `.codex/ACTIVE_TASK.md`.
- If any condition fails, make no changes and report `BLOCKED`.

## Read

1. `AGENTS.md`
2. `.codex/ACTIVE_TASK.md`
3. `.codex/PROJECT_STATE.json`
4. `tasks/VEA-P1-T003B-B-R1-worker-failure-diagnostic-preservation.md`
5. `reports/runs/VEA-P1-T003B-B-R1-worker-failure-diagnostic-preservation.md`
6. `scripts/extract_features.py`
7. `tests/integration/test_p1_feature_extraction_cli.py`
8. `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`

Do not scan unrelated history, reports, artifacts, or repository paths.

## Allowed changes

- Rename:
  - from `tasks/VEA-P1-T003B-B-R1-worker-failure-diagnostic-preservation.md`
  - to `tasks/VEA-P1-T003B-B-R1R-worker-failure-diagnostic-preservation.md`
- Add `tasks/VEA-P1-T003B-B-R1R2-task-identity-reconciliation.md`
- Add `reports/runs/VEA-P1-T003B-B-R1R2-task-identity-reconciliation.md`
- `.codex/ACTIVE_TASK.md` only for the final unstaged `NO_ACTIVE_TASK` state

## Commit paths

- `tasks/VEA-P1-T003B-B-R1-worker-failure-diagnostic-preservation.md`
- `tasks/VEA-P1-T003B-B-R1R-worker-failure-diagnostic-preservation.md`
- `tasks/VEA-P1-T003B-B-R1R2-task-identity-reconciliation.md`
- `reports/runs/VEA-P1-T003B-B-R1R2-task-identity-reconciliation.md`

Do not stage `.codex/ACTIVE_TASK.md`.

## Non-goals and protected scope

- Do not modify `scripts/extract_features.py`, its tests, the Apple runbook, contracts, profiles, extractors, manifests, or existing B-R1 Run Report.
- Do not alter the bytes of the renamed B-R1R Task contract.
- Do not amend or rewrite commit `5d10ecc…`.
- Do not delete or rename the existing remote branch.
- Do not run Apple, CUDA, dataset, checkpoint, smoke, retry, resume, or full extraction.
- Do not install packages, use credentials, or create a review package.
- Do not merge or modify `main`.

## Requirements

1. Rename the mislocated Task record so its filename matches its internal `B-R1R` identity.
2. Preserve the renamed file byte-for-byte.
3. Preserve the existing B-R1 Run Report as immutable historical evidence of the actual delivery.
4. Create the R1R2 Task record and compact Run Report.
5. The R1R2 Run Report must record:
   - intended and actual predecessor identities;
   - baseline, parent, commit, and remote branch;
   - independent review result: implementation diff has no blocking defect;
   - predecessor disposition: `PARTIAL_TASK_IDENTITY_MISMATCH`;
   - exact source and protected-module SHA-256 values;
   - validation commands, exit codes, and test count;
   - confirmation that no implementation or Apple operation occurred.
6. Do not claim the predecessor was delivered under the `B-R1R` branch or commit message.
7. Do not change scientific readiness during this Task.

## Codex validation: Tier 0–1

```text
py -3 -B -m compileall -q scripts/extract_features.py
py -3 -B -m pytest -q tests/integration/test_p1_feature_extraction_cli.py
git -c core.whitespace=cr-at-eol diff --check
```

Also verify:

- focused tests still report `36 passed`;
- `scripts/extract_features.py` remains SHA-256 `95a5f9de7d9d1cd964417d478b3047ec1ff62f578e5c18fbab2176f954ee01d2`;
- `src/vision_encoder_alignment/p1_feature_extraction.py` remains SHA-256 `2c754057f59ccdd4cfc07794816184c2392af583fe26e155183581e2f66f4ab8`;
- the renamed Task record is byte-identical to its predecessor path;
- changed and staged paths exactly match the allowlist.

## User validation: Tier 2–3

None.

Apple transfer and replacement Stage A remain unauthorized until this recovery commit is independently reviewed.

## Git delivery

1. Create `codex/vea-p1-t003b-b-r1r2` from the verified baseline.
2. Perform only the allowed provenance changes.
3. Run mandatory validation.
4. Stage only the exact commit paths.
5. Commit with the exact configured message.
6. Push with `git push -u origin HEAD`.
7. Set `.codex/ACTIVE_TASK.md` to `NO_ACTIVE_TASK` without staging it.

## Done when

- Intended and actual predecessor identities are explicitly distinguishable.
- Historical commit and remote branch remain immutable.
- Implementation, test, runbook, and scientific bytes are unchanged.
- Required validation passes.
- Exact paths are committed and pushed.
- Unauthorized operations are zero.

## Stop conditions

- Git identity or parent mismatch
- Existing recovery branch
- Renamed Task bytes would change
- Any implementation or runbook modification
- Validation failure
- Need for history rewriting or Apple execution

## Final response

Return at most 20 lines containing:

1. Status, Task ID, and model used
2. Renamed and added paths
3. Validation commands, exit codes, and test count
4. Preserved source SHA-256 values
5. Baseline and parent SHA
6. Task branch, commit message, and commit SHA
7. Push result and remote branch
8. Staged count and `git diff --check`
9. Confirmation that implementation and Apple execution were unchanged
10. Next action: independent review

# 이번 작업 정리

## 작업 목적

진단 보강 구현을 다시 수행하지 않고, 실제 전달 커밋과 원래 Task 계약 사이의 식별자 불일치를 명확하게 복구한다.

## 핵심 수행 내용

- 잘못된 경로에 저장된 Task 기록을 올바른 `B-R1R` 경로로 이동한다.
- 실제 `B-R1` 전달 이력을 그대로 보존한다.
- intended identity와 actual identity의 관계를 새 correction report에 기록한다.
- 기존 focused test를 다시 확인한 뒤 correction commit만 push한다.

## 이번 작업에서 하지 않는 것

- 구현 수정
- Git history 재작성
- Apple 실행 또는 smoke 재시도

## 완료 기준

누가 보더라도 어떤 계약이 의도됐고 어떤 커밋이 실제 전달됐는지 혼동 없이 추적할 수 있어야 한다.

## 다음 단계

pushed correction commit을 독립 검수한다.

## 한눈에 보는 작업 요약

오류 기록 기능 자체는 정상으로 보입니다. 이번 작업은 코드가 아니라 작업 이름과 Git 이력이 서로 다르게 기록된 부분만 바로잡아, 이후 Apple 실행 결과를 신뢰할 수 있게 만드는 작업입니다.