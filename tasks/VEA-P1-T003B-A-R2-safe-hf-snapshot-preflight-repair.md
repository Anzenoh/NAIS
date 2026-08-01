# VEA-P1-T003B-A-R2 — Safe Hugging Face snapshot preflight repair

- Mode: `recovery`
- Task class: `STANDARD`
- Evidence level: `COMPACT`
- Package required: `false`
- Recommended Codex model: `GPT-5.6 Terra`
- Reasoning effort: `high`
- Goal: 표준 Hugging Face cache의 안전한 snapshot-file symlink를 Stage A preflight가 허용하되, broken·directory·cache-escape link는 계속 차단하고 모든 preflight 실패 JSON을 보존한다.
- Baseline: `VEA-P1-T003B-A-R1` commit `9a5f64638687c9e5ef4d9c1c237facaac0a7f339`과 Apple audit SHA-256 `d3d2512696176638e6abc9013b31568e30ce7a4319e0869394e9df82959f7cce`
- Git delivery: `COMMIT_AND_PUSH`
- Task branch: `codex/vea-p1-t003b-a-r2`
- Commit condition: `PASS_ONLY`
- Commit message: `fix(extraction): VEA-P1-T003B-A-R2 repair HF snapshot preflight`
- Push target: `origin HEAD`

## Preconditions

- `codex/vea-p1-t003b-a-r1`이 commit `9a5f64638687c9e5ef4d9c1c237facaac0a7f339`을 가리켜야 한다.
- staged path는 `0`이어야 한다.
- 새 local·remote Task branch가 없어야 한다.
- branch 생성 전 전체 status를 기록하고 non-allowlisted dirty path를 그대로 보존한다.
- 다음 현재 bytes를 독립적으로 검증한다:
  - `audit.log`: `d3d2512696176638e6abc9013b31568e30ce7a4319e0869394e9df82959f7cce`
  - `docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md`: `72bcd1a6301c2604df17f8f734ada0807295d564f87460f8fed6c79e9eb7a833`
  - `scripts/extract_features.py`: `5c3d92c5ea03b1a4265855be5d21f5b782c41dbe9cb86cbe0a374185bce418f3`
  - `src/vision_encoder_alignment/p1_runtime_preflight.py`: `86d4e0a99a4c8d1f9c199d708ca355e14b2c63b9acd9088104a795d446c6e031`
  - `tests/unit/test_p1_runtime_preflight.py`: `27cd307f50599873483bc3f39eead1de9b3ed47cd748ce3e93a4654c355c3469`
  - `tests/integration/test_p1_feature_extraction_cli.py`: `c46ae853e3a477415ac816fb9cc3158064ac91ad09a9b8cf081293e7d430a85b`
  - `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`: `c254e815df03509e7a97c63fb3881087e18722ea5d55e6f3fdd31e26337a5896`
- `audit.log`는 `REPOSITORY_BAD_COUNT=0`, `PATH_BAD=0`, `DATASET_AUDIT_RC=0`, `PLAN_RC=0`, `PREFLIGHT_RC=4`, `stdout_bytes=153`을 포함해야 한다.
- 새 Task와 Run Report 경로가 없어야 한다.
- 불일치 시 수정하거나 우회하지 말고 `BLOCKED`.

## Read

1. `AGENTS.md`
2. `.codex/ACTIVE_TASK.md`
3. `.codex/PROJECT_STATE.json`
4. `audit.log`
5. `docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md`
6. `scripts/extract_features.py`
7. `src/vision_encoder_alignment/p1_feature_extraction.py`
8. `src/vision_encoder_alignment/p1_runtime_preflight.py`
9. `scripts/inventory_p1_checkpoint_cache.py`
10. `tests/unit/test_p1_runtime_preflight.py`
11. `tests/integration/test_p1_feature_extraction_cli.py`
12. `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
13. `reports/runs/VEA-P1-T003B-A-R1-sbr1-acceptance-provenance-gate-repair.md`

## Allowed changes

- `docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md`
- `scripts/extract_features.py`
- `src/vision_encoder_alignment/p1_runtime_preflight.py`
- `tests/unit/test_p1_runtime_preflight.py`
- `tests/integration/test_p1_feature_extraction_cli.py`
- `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
- `tasks/VEA-P1-T003B-A-R2-safe-hf-snapshot-preflight-repair.md`
- `reports/runs/VEA-P1-T003B-A-R2-safe-hf-snapshot-preflight-repair.md`

## Commit paths

Exactly the eight Allowed changes paths.

Do not modify, stage or commit `.codex/ACTIVE_TASK.md`, `.codex/PROJECT_STATE.json`,
`audit.log`, profile, manifest, checkpoint plan, model code or unrelated dirty paths.

## Non-goals and protected scope

- Apple 접속과 실제 dataset·checkpoint·model·GPU 사용 금지
- Stage A, Stage B, smoke, resume, full extraction 실행 금지
- checkpoint download, mutation, link materialization 또는 cache 재구성 금지
- CPU fallback, mixed precision, network와 package installation 금지
- profile·manifest·model revision 변경 금지
- review ZIP, detached SHA, package identity와 전체 tree manifest 생성 금지
- main push, force push, amend, rebase, merge와 다음 Task 시작 금지

## Requirements

1. exact baseline에서 `codex/vea-p1-t003b-a-r2`를 만들고 Active Task를 Task 기록 경로에 복사한다.
2. Run Report에 현재 실패를 다음처럼 분류한다:
   - repository/path/dataset/GPU/plan checks passed;
   - preflight exit `4`;
   - 153-byte canonical error는 `checkpoint snapshot unavailable: clip_vit_b16`과 일치;
   - inventory와 extraction preflight의 snapshot-link 정책 불일치.
3. `p1_runtime_preflight.py`의 기존 안전한 checkpoint inventory 동작을 하나의 공개된 bounded helper로 정리하고 inventory와 extraction preflight가 동일한 helper 또는 동일 authority를 사용하게 한다.
4. 허용되는 checkpoint leaf:
   - snapshot 내부의 일반 regular file;
   - strict resolution 결과가 같은 cache root 내부의 regular blob인 file symlink.
5. 반드시 거부할 대상:
   - snapshot 또는 하위 directory symlink/reparse;
   - broken link;
   - cache root 밖으로 나가는 link;
   - directory·device·socket 등 non-regular target;
   - 비어 있는 snapshot;
   - linked cache root 또는 linked snapshot root.
6. 단순히 `not path.is_symlink()`를 삭제하거나 `Path.is_file()`만 신뢰하는 약한 수정은 금지한다. `os.walk(..., followlinks=False)`, strict resolution, cache containment와 regular-file 판정을 명시적으로 검증한다.
7. runner에서는 해당 검증 실패를 안정적인 resource exit `4`로 매핑하며 절대경로·traceback·민감정보를 오류 JSON에 넣지 않는다.
8. `plan` 또는 `preflight`에 `--output-json`이 제공된 경우, 처리된 오류도 같은 canonical error schema로 해당 경로에 no-clobber 기록한다.
   - stdout JSON과 output JSON은 동일해야 한다.
   - 기존 destination은 절대 덮어쓰지 않는다.
   - 오류 증거 기록 실패가 원래 exit classification을 바꾸면 안 된다.
9. contract에 다음 경계를 명확히 기록한다:
   - dataset, cache root, snapshot directory와 모든 directory component는 non-link;
   - 표준 HF snapshot의 leaf-file symlink만 cache 내부 regular blob으로 strict-resolve되는 경우 허용;
   - 이는 root·directory link 허용으로 확대되지 않는다.
10. tests는 OS symlink 권한에 의존하지 않는 deterministic probe를 포함한다. 실제 symlink를 만들 수 있는 환경의 보조 test는 허용하지만 필수 test를 skip으로 대체하지 않는다.
11. 최소 회귀 사례:
   - safe HF leaf link PASS;
   - regular leaf PASS;
   - empty snapshot exit `4`;
   - broken, directory, escaping, non-regular link 각각 거부;
   - preflight resource failure의 `--output-json` 생성;
   - output collision no-clobber;
   - successful synthetic preflight의 기존 결과 유지.
12. runbook은 evidence-root 생성 전에 최소한 다음 실행 경로의 최종 SHA-256을 검사하도록 보강한다:
   - `scripts/extract_features.py`
   - `src/vision_encoder_alignment/p1_feature_extraction.py`
   - `src/vision_encoder_alignment/p1_runtime_preflight.py`
   - `scripts/inventory_p1_checkpoint_cache.py`
   - `scripts/validate_p1_feature_extraction.py`
13. Stage A 성공 evidence는 계속 정확히 `00`–`05`만 사용한다. 실패 시 생성된 canonical plan/preflight error JSON은 보존하고 즉시 중단한다.
14. Run Report에 before/after SHA-256, exact changed paths, Apple 전송 대상, post-transfer SHA 검증 명령과 Stage A 재검증 명령을 기록한다.
15. 모든 readiness 값은 계속 `false`로 유지한다.

## Codex validation: Tier 0–1

- 다음을 실행한다:

  `py -3 -B -m compileall -q src\vision_encoder_alignment\p1_runtime_preflight.py scripts\extract_features.py tests\unit\test_p1_runtime_preflight.py tests\integration\test_p1_feature_extraction_cli.py`

- 다음을 실행한다:

  `py -3 -B -m pytest -q -p no:cacheprovider tests\unit\test_p1_runtime_preflight.py tests\integration\test_p1_feature_extraction_cli.py`

- `git -c core.whitespace=cr-at-eol diff --check`
- changed/staged paths가 exact eight-path allowlist와 일치하는지 확인
- non-allowlisted status가 시작 snapshot과 정확히 같은지 확인
- model load, network, checkpoint write, Apple 실행 count가 모두 `0`인지 확인

## User validation: Tier 2–3

Codex는 실행하지 않는다.

독립 검수와 server file transfer 후 사용자가 runbook의 갱신된 read-only batch revalidation을 한 번 실행한다. Codex final response에는 그 정확한 명령과 전송 파일 SHA-256을 제공한다.

## Git delivery

필수 검증 통과 후에만:

1. exact eight Commit paths만 stage한다.
2. staged allowlist가 정확히 일치하는지 확인한다.
3. 다음 메시지로 commit한다:

   `fix(extraction): VEA-P1-T003B-A-R2 repair HF snapshot preflight`

4. `git push -u origin HEAD`
5. baseline, commit SHA, push 결과와 remote branch를 기록한다.

Push만 인증·network 문제로 실패하면 local commit을 보존하고
`PARTIAL_GIT_DELIVERY`와 exact push command를 반환한다.

## Evidence and deliverables

- 안전한 HF snapshot validation 구현
- positive·negative regression tests
- 보강된 extraction contract와 Apple runbook
- compact Task record와 Run Report
- server transfer manifest와 user-run revalidation command
- baseline, branch, commit SHA와 remote ref

Do not create a review ZIP, detached SHA, package identity, repository-tree
manifest, or full protected-tree hash comparison. They are outside this Task's
evidence level. Use the pushed task commit, targeted validation, and compact Git evidence.

## Done when

- safe standard HF snapshot fixture가 preflight를 통과한다.
- unsafe link variants는 모두 stable exit `4`로 차단된다.
- 실패 시 `--output-json`에 canonical error JSON이 남는다.
- focused tests와 `git diff --check`가 통과한다.
- unexpected·unauthorized change는 `0`이다.
- exact commit이 remote branch에 push된다.
- Apple·Stage A·smoke 실행은 `0`이다.

## Stop conditions

- audit나 accepted source identity 불일치
- safe cache link를 cache 밖 link와 구분할 수 없음
- checkpoint bytes 또는 cache layout 수정 필요
- allowed scope 확대 필요
- 기존 dirty changes를 exact commit paths와 분리할 수 없음
- 필수 deterministic test 실패
- Apple 또는 실제 checkpoint 실행 필요

## Final response

20줄 이내로 다음을 포함한다:

1. status와 Task ID
2. changed paths
3. compile과 focused-test 결과
4. snapshot-link acceptance·rejection 결과
5. error `--output-json` 회귀 결과
6. before/after 핵심 SHA-256
7. baseline SHA와 task branch
8. commit message와 commit SHA
9. push result와 remote branch
10. staged count와 `git diff --check`
11. unauthorized operation count
12. server 전송 대상과 다음 단일 user validation

PASS + push 성공이면 사용자는 Task ID, baseline SHA, commit SHA, remote branch만 제출한다. PARTIAL 또는 BLOCKED이면 최종 출력 전문을 제출한다. Stage A나 다음 Task를 시작하지 않는다.

# 이번 작업 정리

## 작업 목적

정상적인 Hugging Face checkpoint cache를 안전하게 인식하고 preflight 실패 원인을 항상 증거로 남긴다.

## 핵심 수행 내용

- inventory와 extraction runner의 snapshot-link 정책 통일
- cache 내부 safe leaf link만 허용
- unsafe link와 cache escape 회귀 test 추가
- 실패 output JSON과 runbook identity gate 보강
- PASS 시 전용 branch commit과 push

## 이번 작업에서 하지 않는 것

- 실제 Apple preflight, 모델 실행, smoke와 full extraction
- checkpoint 또는 cache 내용 변경

## 완료 기준

안전한 HF cache fixture는 통과하고 위험한 link는 차단되며, 모든 필수 test와 Git delivery가 완료된다.

## 다음 단계

pushed commit을 독립 검수한 뒤 갱신 파일을 Apple에 전송하고 read-only Stage A batch revalidation을 한 번 수행한다.

## 한눈에 보는 작업 요약

체크포인트가 없는 것이 아니라 정상적인 Hugging Face 연결 파일을 잘못 누락으로 판단한 문제다. 안전한 내부 연결만 정확히 허용하고, 이후 오류는 숨겨지지 않도록 수정한다.