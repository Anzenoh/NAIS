# VEA-P1-T003B-A-R3 — Close the Apple Stage A Git graph

* Mode: `recovery`
* Task class: `STANDARD`
* Evidence level: `COMPACT`
* Package required: `false`
* Recommended Codex model: `GPT-5.6 Sol`
* Reasoning effort: `high`
* Goal: Produce one pushed descendant of the accepted R1R10 baseline whose clean checkout contains every repository-local input required by T003B Stage A and whose runbook validates a separately approved exact commit.
* Baseline: `368db0a865d3d81f35f260e3d3ce2a528f89b335`
* Accepted predecessor: `VEA-P1-T003B-B-R1R10 = ACCEPTED_PASS`
* Git delivery: `COMMIT_AND_PUSH`
* Task branch: `codex/vea-p1-t003b-a-r3`
* Commit condition: `PASS_ONLY`
* Commit message: `fix(preflight): VEA-P1-T003B-A-R3 close Stage A Git graph`
* Push target: `origin HEAD`

## Task identity correction

The historical Task and branch named `VEA-P1-T003B-A-R2` already exist at:

`ff75f06e34ffd05e86844e07a3abb306d34d547f`

That commit is the earlier Hugging Face snapshot-preflight repair and is an ancestor of the current baseline. It is not this Task's implementation result.

The duplicate R2 Stage-A-closure draft is:

`VOID_DUPLICATE_TASK_ID / BLOCKED_NO_COMMIT`

Do not reuse, delete, move, amend, or reinterpret the historical R2 branch or commit.

## Preconditions

Before source edits or branch creation:

1. Require:

   * `HEAD == 368db0a865d3d81f35f260e3d3ce2a528f89b335`;
   * staged-path count `0`;
   * `origin/codex/vea-p1-t003b-b-r1r10` resolves exactly to HEAD;
   * `codex/vea-p1-t003b-a-r3` is absent locally and remotely.
2. Do not require a particular current branch or fixed dirty-path count.
3. Record outside the repository a NUL-safe manifest of every dirty path, its type and byte identity.
4. Preserve every path outside the final allowed graph exactly.
5. Audit the complete Stage A graph and all missing sources before editing anything.
6. Do not stop at the first missing path.
7. If any required source cannot be obtained without guessing, return one aggregate `BLOCKED_STAGE_A_SOURCE_MISSING` result and create no branch.

The historical R2 branch being present is expected and is not a blocker.

## Read

1. `AGENTS.md`
2. `.codex/ACTIVE_TASK.md`
3. `.codex/PROJECT_STATE.json`
4. `docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md`
5. `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
6. Repository paths used by runbook Sections 1–3
7. `scripts/extract_features.py`
8. `src/vision_encoder_alignment/p1_runtime_preflight.py`
9. The Stage A profile and its structured references
10. Stage A-related tests reached from the selected scripts and imports

Use bounded path, structured-reference and Python-import traversal. Do not scan backlog, archives, unrelated reports, artifacts or the complete dirty tree.

## Confirmed baseline absences

These eight paths are absent from the baseline commit and are initial closure candidates:

* `configs/extraction_profiles/p1_1000_apple_fp32_v0.json`
* `configs/model_snapshots/p1_three_encoder_checkpoint_plan.template.json`
* `scripts/validate_p1_apple_runtime.py`
* `scripts/inventory_p1_checkpoint_cache.py`
* `scripts/validate_p1_feature_extraction.py`
* `src/vision_encoder_alignment/p1_feature_extraction.py`
* `tasks/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md`
* `reports/runs/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md`

Their absence from the baseline does not itself authorize invention or substitution.

For every missing candidate, record:

* referencing file, field, command or import;
* current source location;
* file type and SHA-256;
* whether a reachable Git blob exists;
* required or unrelated classification;
* final inclusion decision.

Current working-tree bytes may be used as candidates only when the path is concretely reached from the authorized Stage A graph and passes the clean-clone validation. Do not copy unidentified installed-package code.

## Path-set separation

Use three distinct sets.

### `STAGE_A_GRAPH_SET`

Only repository code, configuration, contracts, runbook and focused tests reached from:

* runbook Sections 1–3;
* the full-extraction contract;
* the Stage A extraction profile;
* structured profile references;
* local Python imports of the Stage A scripts;
* required package initializers;
* R1R3 provenance paths required by the runbook.

A path may enter only when it is absent or differs from the baseline and is reached by these rules. Dirty status alone is insufficient.

Freeze membership before candidate rebinding.

### `EVIDENCE_SET`

Exactly:

* `tasks/VEA-P1-T003B-A-R3-stage-a-git-closure.md`
* `reports/runs/VEA-P1-T003B-A-R3-stage-a-git-closure.md`

The Task record must preserve the approved Active Task content. The Run Report must not contain or predict the final commit SHA.

### `DELIVERY_SET`

`STAGE_A_GRAPH_SET ∪ EVIDENCE_SET`

Commit and tree-identity checks use the final `DELIVERY_SET`.

## Runbook repair

Update only the identity, cleanliness and reproducibility gates needed by Stage A.

1. Require:

   `VEA_APPROVED_REPO_COMMIT`

2. Before evidence-root creation:

   * require exactly 40 lowercase hexadecimal characters;
   * require `git rev-parse HEAD` to equal it;
   * reject every staged, unstaged and untracked path;
   * retain `git diff --check`.

3. Remove the operational requirement for branch `main`.

4. Remove every executable use of fixed commit `e518a51ba83f70b42611cc4386a9f55dc9e97de3`.

5. Do not embed the new commit SHA in the commit that creates it.

6. Derive the Stage B run-ID prefix from the already validated `VEA_APPROVED_REPO_COMMIT`; this consistency repair does not authorize Stage B.

7. Add `scripts/validate_p1_apple_runtime.py` to the pre-evidence-root Stage A implementation bindings.

8. Rebind the profile and implementation hashes to final Git blob bytes.

9. Preserve:

   * direct-host `apple`;
   * R1R3 and SBR1 provenance checks;
   * all-component path checks;
   * offline mode;
   * exact `00`–`05` Stage A evidence contract;
   * Stage A/Stage B approval separation;
   * false extraction readiness.

Do not weaken dataset, checkpoint, runtime, identity or path validation.

## Deterministic identity rebinding

Build the graph in an external candidate clone with:

`core.autocrlf=false`

Hash Git blob bytes, not source working-tree checkout bytes.

Use this acyclic order:

1. Finalize non-profile implementation, configuration and provenance files.
2. Rebind structured references and the extraction profile.
3. Hash the final profile blob.
4. Rebind the runbook to the final profile and implementation blobs.
5. Update only focused expected identities in tests.
6. Repeat the full binding calculation.
7. Require the second pass to change no digest or file.

If the binding graph is cyclic, return `BLOCKED_IDENTITY_CYCLE`; do not iterate toward an unstable fixed point.

## Candidate validation

Construct the candidate outside the source repository using only baseline Git bytes and the frozen graph candidates.

Validate in a second clean clone that cannot import from the original working tree.

Set:

* `core.autocrlf=false`
* `PYTHONDONTWRITEBYTECODE=1`
* `PYTHONNOUSERSITE=1`
* `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
* candidate `src` as the first Python path
* pytest cache disabled
* external pytest temporary directory

Require:

1. Every Stage A path is present in Git.
2. Every structured reference exists and matches.
3. Every imported `vision_encoder_alignment` module resolves below the clean clone.
4. No project module resolves from site-packages or the source working tree.
5. Changed JSON, YAML and TOML parse.
6. Changed Python compiles without writing bytecode.
7. The runbook requires the approved exact commit and a completely clean repository before evidence creation.
8. No executable `main` or `e518a51...` identity remains.
9. Stage B derives its commit prefix without receiving execution authorization.
10. Run:

    * `tests/unit/test_p1_runtime_preflight.py`
    * `tests/integration/test_p1_feature_extraction_cli.py`
    * an existing Stage A runbook contract test, or create `tests/unit/test_p1_t003b_stage_a_runbook.py` if none exists.
11. All focused tests pass.
12. `git diff --check --ignore-space-at-eol` passes.

Do not run the accepted R1R10 19-file or 533-node suite.

## Evidence without self-reference

After graph validation:

1. Record the final graph manifest, profile SHA, runbook SHA, structured-reference result, module-origin result and focused-test result.
2. Compare source and candidate blobs for `STAGE_A_GRAPH_SET`.
3. Generate the final Task and Run Report.
4. The Run Report may record graph-set blob equality but must not record:

   * the future commit SHA;
   * a claimed full-delivery tree OID computed before the report itself existed.
5. Add `EVIDENCE_SET`, freeze `DELIVERY_SET` and create the final disposable candidate tree.
6. Run parse, static and diff checks again for the completed delivery.
7. Record final full-tree equality only in the Codex final response.

This ordering prevents Task/Run Report self-reference.

## Source delivery

Proceed only after clean candidate validation passes.

1. Require every unrelated dirty path, type and byte identity to match the initial preservation manifest.
2. Require graph candidate blobs to match the validated external candidate.
3. Create `codex/vea-p1-t003b-a-r3` from the exact baseline.
4. Copy back only final `DELIVERY_SET` bytes.
5. Stage exactly `DELIVERY_SET`.
6. Require the staged path set to equal `DELIVERY_SET`.
7. Require the staged source tree OID to equal the final candidate tree OID.
8. Commit with the exact required message.
9. Require exactly one parent equal to the baseline.
10. Require staged count `0` after commit.
11. Push with `git push -u origin HEAD`.
12. Verify the remote R3 ref resolves exactly to the new commit.

If only push fails, retain the valid local commit and return `PARTIAL_GIT_DELIVERY` with the exact retry command.

## Evidence policy and non-goals

Do not create a review ZIP, detached SHA, package identity, repository-tree manifest or full protected-tree hash comparison.

Do not:

* reuse or delete historical R2;
* commit the entire dirty tree;
* synthesize missing production code;
* weaken tests or bindings;
* run Apple, GPU, CUDA, dataset, checkpoint or model access;
* execute Stage A, Stage B, smoke, resume or extraction;
* install or download dependencies;
* merge, rebase, amend, force push, reset, restore, clean or stash;
* commit `.codex/ACTIVE_TASK.md`;
* promote readiness.

Keep:

`READY_FOR_P1_EXTRACTION=false`

## Done when

* A clean checkout contains the complete Stage A repository graph.
* The runbook accepts an independently supplied exact commit and rejects every dirty repository state before evidence creation.
* All structured references and local imports resolve from Git alone.
* Focused validation passes.
* Candidate and source delivery trees match.
* Only `DELIVERY_SET` is committed and pushed.
* Every unrelated dirty byte remains unchanged.
* No external execution occurred.

## Final response

Within 20 lines include:

1. Task ID, model and status
2. baseline and final branch
3. historical R2 preservation confirmation
4. initial missing count and aggregate resolution
5. `STAGE_A_GRAPH_SET`, `EVIDENCE_SET` and `DELIVERY_SET` counts
6. profile and runbook SHA values
7. structured-reference and module-origin results
8. approved-commit and clean-repository gate result
9. focused test commands, counts and exits
10. graph blob and final tree equality
11. commit message and commit SHA
12. push result and remote branch
13. remaining dirty count and staged count
14. prohibited-operation confirmation
15. `Stage A executed: false`
16. `READY_FOR_P1_EXTRACTION=false`
17. next action: independent commit review before Apple synchronization

For any non-PASS result, return the complete aggregate missing, mismatch or identity-cycle set.

# 이번 작업 정리

## 작업 목적

깨끗한 Git checkout만으로 Apple Stage A preflight를 시작할 수 있는 실행 graph를 완결한다.

## 핵심 수행 내용

* Stage A에 실제로 도달하는 파일과 누락 source 전체 조사
* profile·runbook identity를 Git blob 기준으로 재결속
* 낡은 branch·commit gate를 승인 commit 입력 방식으로 교체
* clean clone에서 import와 focused validation 수행
* 검증된 graph와 evidence만 commit하고 push

## 이번 작업에서 하지 않는 것

* 역사적 R2 재사용 또는 삭제
* Apple·GPU·dataset·checkpoint 접근
* Stage A·Stage B·smoke·extraction 실행
* 533-node Windows 회귀 반복
* unrelated dirty tree commit

## 완료 기준

새 commit만 checkout한 환경에서 Stage A에 필요한 repository 파일과 identity 검사가 모두 재현된다.
