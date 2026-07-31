# VEA-P1-T003B-A-R1 — SBR1 acceptance provenance gate repair

- Mode: `implementation`
- Task class: `STANDARD`
- Evidence level: `COMPACT`
- Package required: `false`
- Goal: T003B runbook이 불변 SBR1 Run Report에서 미래 승인 표식을 찾는 결함을 수정하고 회귀 test로 고정한다.
- Baseline: `VEA-P1-T002D-SBR1 = ACCEPTED_PASS`; `VEA-P1-T003A-R1R3 = PARTIAL / ACCEPTANCE_REOPENED`; `main` at `e518a51ba83f70b42611cc4386a9f55dc9e97de3`
- Git delivery: `COMMIT_AND_PUSH`
- Task branch: `codex/vea-p1-t003b-a-r1`
- Commit condition: `PASS_ONLY`
- Commit message: `fix(runbook): VEA-P1-T003B-A-R1 repair SBR1 acceptance gate`
- Push target: `origin HEAD`

## Preconditions

Before any repository modification:

- current branch: `main`
- HEAD: `e518a51ba83f70b42611cc4386a9f55dc9e97de3`
- staged paths: `0`
- `origin` is configured
- local task branch does not already exist
- existing dirty tree is captured and preserved
- non-allowlisted status entries must remain exactly unchanged

Verify these current file identities:

- R1R3 Task:
  `tasks/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md`
  SHA-256 `3b792f6792b18678ebd6dcc9bb0557bc8dba861f0278d407ebc58e1e9f42a101`
- SBR1 Run Report:
  `reports/runs/VEA-P1-T002D-SBR1-apple-source-binding-recovery-reconciliation.md`
  SHA-256 `8abfc27089b5a91b20a9c6a37dd042a7b0375914dbafd76b6433267155c8cbcd`
- T003B runbook:
  `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
  SHA-256 `416c3270ac222b6d8ef692957c2fb91562f5e18c95418083663e4a1b63abf36a`
- integration test:
  `tests/integration/test_p1_feature_extraction_cli.py`
  SHA-256 `4056c1c51b091923bffe348e5062667cc88e2fa134541fed0e9b5799c15a890c`

The SBR1 Run Report must retain:

- `Status = PASS`
- `Independent acceptance = pending`
- no `VEA-P1-T002D-SBR1 = ACCEPTED_PASS` marker

That absence is expected and is not a blocker.

A user-approved non-placeholder `.codex/ACTIVE_TASK.md` is the current contract.
`.codex/PROJECT_STATE.json` containing the previous `NO_ACTIVE_TASK` snapshot alone is not a blocker.

The following new paths must not already exist:

- `tasks/VEA-P1-T003B-A-R1-sbr1-acceptance-provenance-gate-repair.md`
- `reports/runs/VEA-P1-T003B-A-R1-sbr1-acceptance-provenance-gate-repair.md`

If any other precondition fails, make no repository modification and return `BLOCKED`.

## Read

1. `AGENTS.md`
2. `.codex/ACTIVE_TASK.md`
3. `.codex/PROJECT_STATE.json`
4. `tasks/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md`
5. `reports/runs/VEA-P1-T002D-SBR1-apple-source-binding-recovery-reconciliation.md`
6. `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
7. `tests/integration/test_p1_feature_extraction_cli.py`

Do not scan unrelated tasks, reports, handovers, artifacts or repository directories.

## Allowed changes

- `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
- `tests/integration/test_p1_feature_extraction_cli.py`
- `tasks/VEA-P1-T003B-A-R1-sbr1-acceptance-provenance-gate-repair.md`
- `reports/runs/VEA-P1-T003B-A-R1-sbr1-acceptance-provenance-gate-repair.md`

## Commit paths

Exactly the four Allowed changes paths.

Do not modify, stage or commit `.codex/ACTIVE_TASK.md` or `.codex/PROJECT_STATE.json`.
Do not stage unrelated dirty paths.

## Non-goals and protected scope

- Do not modify the historical SBR1 Run Report.
- Do not modify R1R3 Task, Run Report, review receipt, source or package evidence.
- Do not access Apple, dataset, checkpoint, model, GPU or evidence roots.
- Do not execute runbook Section 1–5.
- Do not execute Stage A, Stage B, smoke, resume or full extraction.
- Do not create a ZIP, detached SHA, package identity or repository-tree manifest.
- Do not install or download dependencies.
- Git network access is authorized only for normal task-branch push.
- Do not push `main`, force push, amend, rebase, merge or start another Task.

## Requirements

1. After preconditions pass, create `codex/vea-p1-t003b-a-r1` from the exact baseline.
2. Copy the current Active Task unchanged to:
   `tasks/VEA-P1-T003B-A-R1-sbr1-acceptance-provenance-gate-repair.md`.
3. Repair the runbook:
   - remove the operational `SBR1_REPORT` dependency used only for acceptance checking;
   - do not add or request an acceptance marker in the historical SBR1 Run Report;
   - after the existing successful R1R3 Task SHA-256 verification, require the exact marker
     `VEA-P1-T002D-SBR1 = ACCEPTED_PASS`
     from the hash-verified `R1R3_TASK`;
   - perform this marker check before any evidence-root creation or external mutation;
   - fail closed on missing file, identity mismatch or missing marker.
4. Preserve all existing Apple direct-host, Stage A/B, exact `00`–`05`, canonical JSON and explicit-stop boundaries.
5. Update the integration test to prove:
   - the accepted marker is not checked against the SBR1 Run Report;
   - the marker is checked against `R1R3_TASK`;
   - R1R3 Task SHA verification precedes the marker check;
   - the marker check precedes evidence-root creation;
   - Stage A/B and exact-artifact boundaries remain unchanged.
6. Create the compact Run Report at the allowed report path. Record:
   - baseline and branch;
   - before/after runbook SHA-256;
   - unchanged SBR1 Run Report SHA-256;
   - changed paths;
   - focused-test result;
   - `git diff --check`;
   - Stage A executed `false`;
   - package created `false`.
7. Verify non-allowlisted repository status is exactly unchanged.
8. Do not promote Stage A, Stage B or extraction readiness in this Task.

## Codex validation: Tier 0–1

- Independently calculate and compare the four precondition SHA-256 values.
- Run:

  `py -3 -B -m pytest -q -p no:cacheprovider tests/integration/test_p1_feature_extraction_cli.py`

- Run:

  `git -c core.whitespace=cr-at-eol diff --check`

- Confirm the SBR1 Run Report SHA-256 remains:
  `8abfc27089b5a91b20a9c6a37dd042a7b0375914dbafd76b6433267155c8cbcd`
- Confirm changed and staged paths exactly match the four-path allowlist before commit.
- Confirm non-allowlisted status entries match the pre-task snapshot.

## User validation: Tier 2–3

None in this Task.

After push, the user submits Task ID, baseline SHA, commit SHA and remote branch for independent review.
Stage A remains separately unauthorized.

## Git delivery

After all mandatory validation passes:

1. `git add --` exactly the four Commit paths.
2. Verify `git diff --cached --name-only` equals the allowlist.
3. Commit with:
   `fix(runbook): VEA-P1-T003B-A-R1 repair SBR1 acceptance gate`
4. Push with:
   `git push -u origin HEAD`
5. Record baseline SHA, commit SHA, push result and remote branch.

If push alone fails for auth or network reasons, preserve the valid local commit and return `PARTIAL_GIT_DELIVERY` with the exact retry command.

## Evidence and deliverables

- corrected T003B runbook
- focused regression test
- historical Task record
- compact Run Report
- baseline, branch, commit SHA and remote ref
- no sealed package

## Done when

- provenance gate uses the hash-verified R1R3 Task.
- historical SBR1 Run Report is byte-identical.
- focused test and `git diff --check` pass.
- unexpected and unauthorized changes are `0`.
- exact commit paths are committed and pushed.
- Stage A executed remains `false`.

## Stop conditions

- any accepted identity mismatch
- historical SBR1 Run Report modification required
- allowed path contains unexplained pre-existing changes
- new Task or Run Report target already exists
- scope expansion or protected-path modification required
- focused validation failure
- actual Apple or extraction execution required

## Final response

20 lines or fewer.

For PASS and successful push include:

- `Status: PASS`
- `Task ID: VEA-P1-T003B-A-R1`
- baseline SHA and task branch
- four changed paths
- focused-test result
- unchanged SBR1 Run Report SHA-256
- `Stage A executed: false`
- exact commit message and commit SHA
- `Push result: PASS`
- remote branch
- staged count and `git diff --check`
- next action: submit Task ID, baseline SHA, commit SHA and remote branch

For `PARTIAL`, `BLOCKED` or `PARTIAL_GIT_DELIVERY`, return the full compact result with the exact remaining blocker or recovery command.

Do not start Stage A, another Task or a merge.

# 이번 작업 정리

## 작업 목적

과거 실행 보고서를 변조하지 않고 T003B의 SBR1 승인 gate를 실행 가능하게 교정한다.

## 핵심 수행 내용

- runbook provenance gate 수정
- ordering·boundary 회귀 test 보강
- compact evidence 작성
- PASS일 때 전용 branch commit과 push

## 이번 작업에서 하지 않는 것

- Apple·dataset·checkpoint 실행
- Stage A, Stage B와 full extraction
- package 생성과 과거 보고서 수정

## 완료 기준

수정된 gate와 회귀 test가 통과하고 exact commit이 remote branch에 push된다.

## 한눈에 보는 작업 요약

과거 보고서에 미래 승인 문구를 억지로 추가하지 않고, 이미 해시로 보호되는 승인 Task에서 그 판정을 확인하도록 고치는 작업이다.