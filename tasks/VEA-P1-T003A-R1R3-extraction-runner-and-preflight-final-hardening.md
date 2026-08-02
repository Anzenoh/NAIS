# VEA-P1-T003A-R1R3 — Extraction Runner Failure Boundary and Apple Preflight Runbook Final Hardening

## Mode

`recovery`

## Repository

`G:\내 드라이브\2026NAIS`

## Goal

`VEA-P1-T003A-R1R2`의 구현과 증거를 보존하면서 독립 검토에서 확인된 T003B runbook 계약 불일치와 runner의 failure-evidence 경계를 제한적으로 수정한다. 수정 후에는 Windows offline evidence만 새로 만들고, 별도 독립 검토가 실제 Apple read-only preflight 진입 가능 여부를 판단할 수 있는 `VEA-P1-T003A-R1R3` review package를 생성한다.

이 Task는 Apple preflight, dataset image 접근, checkpoint body 접근, model load, forward pass, CUDA/GPU probe, 10-image smoke 또는 1,000-image extraction을 실행하지 않는다.

## Independent predecessor disposition

다음 상태를 소급 변경하지 않는다.

```text
VEA-P1-T002D = SUPERSEDED_PARTIAL
VEA-P1-T002D-SBR1 = ACCEPTED_PASS
VEA-P1-T003A = SUPERSEDED_BLOCKED
VEA-P1-T003A-R1 = SUPERSEDED_BLOCKED_PRECONDITION
VEA-P1-T003A-R1R = SUPERSEDED_BLOCKED_PRECONDITION
VEA-P1-T003A-R1R2 = BLOCKED_INDEPENDENT_REVIEW

READY_FOR_P1_T003B = false
READY_FOR_P1_EXTRACTION = false
```

R1R2의 self-reported `PASS`는 다음 이유로 `ACCEPTED_PASS`가 아니다.

1. active R1R2 Task는 `VEA_DATASET_ROOT`와 `VEA_CHECKPOINT_CACHE`를 사용자 제공 environment variable로 요구했지만 runbook은 두 경로를 하드코딩한다.
2. active R1R2 Task는 repository/hostname/branch/HEAD/staged/source identity를 evidence root 생성 전에 검사하도록 요구했지만 runbook은 evidence root를 먼저 생성한다.
3. active R1R2 Task는 repository, dataset, cache, runtime 및 output parent의 모든 symlink/reparse component를 거부하도록 요구했지만 runbook과 Python boundary는 final component 중심의 불충분한 검사를 사용한다.
4. active R1R2 Task는 `plan`과 `preflight`가 각각 canonical JSON 하나만 만들도록 요구했지만 runbook은 `--output-json`과 stdout redirect를 함께 사용해 각각 두 파일을 만든다.
5. runner는 worker return code가 음수이거나 1–255 밖이면 structured failure evidence를 쓰지 못할 수 있다.
6. runner의 `complete_run` 또는 parent-side validation failure는 structured orchestration evidence 없이 종료될 수 있다.
7. 제출된 review ZIP 내부는 일관되지만, 독립 검토 입력에는 R1R2 detached SHA file과 `package-identity.json`이 포함되지 않았다. R1R3는 새 package의 세 외부 파일을 모두 생성·검증하고 반환 대상으로 명시한다.

R1R2 Task, Run Report, review ZIP, detached SHA 및 package identity를 수정하거나 재생성하지 않는다.

## Preconditions

Repository file을 수정하기 전에 아래를 모두 read-only로 검증한다. 하나라도 실패하면 active R1R3 Task file도 만들지 말고 `BLOCKED_PRECONDITION`으로 종료한다.

### 1. Repository identity

```text
repository = G:\내 드라이브\2026NAIS
branch = main
HEAD = e518a51ba83f70b42611cc4386a9f55dc9e97de3
staged paths = 0
status entries = 343
git diff --check exit = 0
```

기존 dirty tree는 허용하지만 reset, restore, checkout, stash, clean, stage 또는 임의 정리를 수행하지 않는다. `git diff --check`의 기존 CRLF warning은 기록하되 exit code로 판정한다.

### 2. R1R2 package identity

다음 기존 외부 파일을 찾고 read-only로 검증한다.

```text
G:\내 드라이브\VEA_P1_T003A_R1R2_TEMP_20260730\review\
  VEA_P1_T003A_R1R2_INDEPENDENT_REVIEW_20260730.zip
  VEA_P1_T003A_R1R2_INDEPENDENT_REVIEW_20260730.zip.sha256.txt
  VEA_P1_T003A_R1R2_INDEPENDENT_REVIEW_20260730.package-identity.json
```

필수 ZIP identity:

```text
bytes = 131207
sha256 = b5c5500bb75acef46b8842a8b1fa371fddebe8dcb026deb2c150042f86c582e4
entry count = 40
manifested payload count = 39
manifest schema = vea.p1.t003a_r1r2.independent_review_manifest.v1
missing/unmanifested/bytes/SHA/duplicate/unsafe/symlink/CRC errors = 0
```

Detached SHA가 위 ZIP SHA와 일치하고, package identity가 다음을 모두 정확히 묶는지 검증한다.

- ZIP path, bytes, SHA
- detached SHA path, bytes, SHA
- root payload manifest SHA
- R1R2 Run Report path, bytes, SHA
- repository files modified after ZIP = 0
- ZIP payload files modified after ZIP = 0
- repository files modified after Run Report seal = 0

외부 detached SHA 또는 package identity가 없거나 서로 불일치하면 repository를 수정하지 말고 `BLOCKED_PRECONDITION_EVIDENCE_MISSING_OR_MISMATCH`로 종료한다. R1R2 package를 복구, 재작성 또는 추정하지 않는다.

### 3. Exact R1R2 repository identities

다음 current repository path의 bytes/SHA를 exact하게 확인한다.

| Path | Expected SHA-256 |
|---|---|
| `tasks/VEA-P1-T003A-R1R2-extraction-contract-runbook-correction-retry.md` | `1dcbf20fab387051170fc905ea0c7e8a49db713bc908c3a1ad6a0870f1d31c3c` |
| `reports/runs/VEA-P1-T003A-R1R2-extraction-contract-runbook-correction-retry.md` | `df8e3223ae83b0209880c51c6415fc23d5c723f43e51fc207b5c5d8e3f23d79c` |
| `reports/runs/VEA-P1-T003A-apple-full-extraction-authoring.md` | `429e6f386538b3ba6d1dc7c184c887e8a4bcec75d7c847445434bed2164c4600` |
| `configs/extraction_profiles/p1_1000_apple_fp32_v0.json` | `98d2dfe8e711efdb6f33ac45e87e21318ec53db8b1cc8f6a0aed789babc58cf5` |
| `docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md` | `221b31c64aa113a6505a6f0acd9ac74f931f2f124f2ea33e9425155735f2fc4d` |
| `src/vision_encoder_alignment/p1_feature_extraction.py` | `fd5dbc33da3ac5d7a2b38f8c474472056a8b0a558eae7b665f530130c5ff6507` |
| `scripts/extract_features.py` | `3a6bfa628fceadf4233726772a5bfd4e7a9361f932c46784f5cf41017ba7524b` |
| `scripts/validate_p1_feature_extraction.py` | `7a0e8cfb54798d857ddccdf6d8288d13f8587d3ff257be56c1ef49c8db1ba4f0` |
| `tests/unit/test_p1_feature_extraction.py` | `c2f05cb2dbd9ac2be1c9ea74533b39becd79fce678cadf36f40de45a7e0748e3` |
| `tests/integration/test_p1_feature_extraction_cli.py` | `41cc9aff13dead288ab29514b845ee1c78ebd2f7297d215b81a5078cc4ea1960` |
| `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md` | `b3448bcf94ca320f08e423571a3e871471be7c30485e1059fa96f8096fa633d4` |

R1R2 Task/Run Report와 original T003A Run Report는 read-only다.

### 4. R1R3 path absence

다음은 모두 없어야 한다.

```text
G:\내 드라이브\2026NAIS\tasks\
  VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md

G:\내 드라이브\2026NAIS\reports\runs\
  VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md

G:\내 드라이브\VEA_P1_T003A_R1R3_TEMP_20260730\
```

존재하면 overwrite, delete, rename, merge, reuse 또는 cleanup하지 말고 `BLOCKED_PRECONDITION_PATH_EXISTS`로 종료한다.

## Active Task creation

모든 precondition이 통과한 뒤 이 명령문 전체를 다음 파일에 verbatim 저장한다.

```text
tasks/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md
```

저장 직후 bytes/SHA를 기록하고 이 파일을 sole active implementation specification으로 사용한다.

## Read first

다음 순서로 읽는다.

1. `AGENTS.md`
2. `docs/RESEARCH_CONTRACT.md`
3. `docs/EXPERIMENT_PROTOCOL_V0.md`
4. `docs/REPRODUCIBILITY.md`
5. `docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md`
6. `configs/extraction_profiles/p1_1000_apple_fp32_v0.json`
7. `tasks/VEA-P1-T003A-apple-full-extraction-authoring.md`
8. `reports/runs/VEA-P1-T003A-apple-full-extraction-authoring.md`
9. `tasks/VEA-P1-T003A-R1R2-extraction-contract-runbook-correction-retry.md`
10. `reports/runs/VEA-P1-T003A-R1R2-extraction-contract-runbook-correction-retry.md`
11. `src/vision_encoder_alignment/p1_feature_extraction.py`
12. `scripts/extract_features.py`
13. `scripts/validate_p1_feature_extraction.py`
14. `tests/unit/test_p1_feature_extraction.py`
15. `tests/integration/test_p1_feature_extraction_cli.py`
16. `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
17. R1R2 review ZIP의 manifest, plan, validation logs 및 post-report seal
18. `tasks/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md`

## Source of truth

- Scientific identities, model order, taps, dtype, geometry, shard size 및 output parent: unchanged canonical profile
- Runtime/source/checkpoint/dataset identity: canonical profile과 repository의 exact bound files
- R1R2 starting bytes: R1R2 ZIP root manifest와 위 exact SHA table
- 이번 결함 및 보정 범위: active R1R3 Task
- 최종 판정 근거: R1R3 Run Report, raw logs, protected comparison, repository seal 및 review package

누락된 값은 추정하지 않는다.

## Allowed repository changes

다음 path만 생성 또는 수정할 수 있다.

```text
tasks/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md
reports/runs/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md
docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md
src/vision_encoder_alignment/p1_feature_extraction.py
scripts/extract_features.py
tests/unit/test_p1_feature_extraction.py
tests/integration/test_p1_feature_extraction_cli.py
reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md
```

`scripts/validate_p1_feature_extraction.py`는 read-only다. 변경이 반드시 필요하다고 판단되면 수정하지 말고 이유와 필요한 diff 범위를 보고한 뒤 `BLOCKED_SCOPE_EXPANSION_REQUIRED`로 종료한다.

## Protected files and roots

Allowed changes 밖의 repository 전체와 다음을 변경하지 않는다.

```text
configs/extraction_profiles/p1_1000_apple_fp32_v0.json
configs/datasets/**
configs/model_snapshots/**
configs/runtime_profiles/**
configs/extractor_profiles/**
data/manifests/**
scripts/validate_p1_apple_runtime.py
scripts/inventory_p1_checkpoint_cache.py
scripts/dinov3_extraction_worker.py
src/vision_encoder_alignment/extractors/**
src/vision_encoder_alignment/runtime_boundary.py
src/vision_encoder_alignment/dinov3_extraction_boundary.py
tests/integration/test_dinov3_identity_transaction.py
tests/unit/test_dinov3_identity_rebind.py
tests/unit/test_dinov3_semantic_probe_profile.py
tests/unit/test_dinov3_runtime_profile.py
tests/unit/test_dinov3_snapshot_profile.py
tasks/VEA-P1-T003A-R1R2-extraction-contract-runbook-correction-retry.md
reports/runs/VEA-P1-T003A-R1R2-extraction-contract-runbook-correction-retry.md
reports/runs/VEA-P1-T003A-apple-full-extraction-authoring.md
G:\내 드라이브\VEA_P1_T003A_R1R2_TEMP_20260730\
```

## External Task root

R1R3 log, pytest basetemp, manifest, probe 및 review output은 다음 신규 root에만 쓴다.

```text
G:\내 드라이브\VEA_P1_T003A_R1R3_TEMP_20260730\
```

Root는 precondition 완료 뒤 한 번만 생성한다. 실패 시 보존하며 자동 cleanup 또는 재사용하지 않는다.

## Network, dependency, Git, and execution policy

금지:

- network access
- SSH, SCP, remote shell 또는 Apple 접속
- package installation, upgrade 또는 removal
- dataset/model/checkpoint download
- dataset image 또는 checkpoint body 읽기
- model import를 통한 checkpoint load
- forward pass
- CUDA/GPU probe
- feature extraction
- Apple runtime write
- Git add, commit, branch, checkout, reset, restore, stash, clean, fetch, pull, push 또는 remote mutation
- 다음 Task 자동 시작

허용:

- repository source/config/manifest/profile read
- synthetic fixture를 이용한 offline unit/integration tests
- compile/test/Git read-only validation
- 외부 R1R3 root에 evidence 작성

## Non-goals

- Apple preflight 실행
- 10-image smoke 실행
- full extraction 실행
- full batch size 선택 또는 auto-tuning
- canonical profile, model revision, tap, geometry, dtype, dataset 또는 shard size 변경
- R1R2 결과를 소급해 `ACCEPTED_PASS`로 변경
- pre-existing symlink capability tests 수정, skip/xfail 추가 또는 global deselection
- unrelated refactor

## Implementation requirements

### 1. T003B runbook input contract

Runbook은 dataset/cache path를 하드코딩하지 않는다. 사용자가 실행 전에 다음 environment variable을 명시적으로 제공해야 한다.

```bash
: "${VEA_DATASET_ROOT:?set VEA_DATASET_ROOT to the Apple COCO val2017 root}"
: "${VEA_CHECKPOINT_CACHE:?set VEA_CHECKPOINT_CACHE to the Apple checkpoint cache root}"
: "${VEA_T003B_EVIDENCE_ROOT:?set VEA_T003B_EVIDENCE_ROOT to a fresh evidence root}"

DATASET_ROOT="$VEA_DATASET_ROOT"
CACHE_ROOT="$VEA_CHECKPOINT_CACHE"
EVIDENCE_ROOT="$VEA_T003B_EVIDENCE_ROOT"
```

다음을 거부한다.

- empty value
- literal placeholder
- relative path
- trailing ambiguity를 포함한 noncanonical path
- 존재하지 않는 dataset/cache parent
- 이미 존재하는 evidence root

Runbook은 R1R3가 독립적으로 `ACCEPTED_PASS`된 뒤에만 사용하도록 gate를 갱신한다.

### 2. Pre-evidence-root ordering

`mkdir "$EVIDENCE_ROOT"` 전에 다음 read-only 검사가 모두 끝나야 한다.

1. hostname `apple`
2. repository exact absolute path와 `.git`
3. branch `main`
4. HEAD `e518a51ba83f70b42611cc4386a9f55dc9e97de3`
5. staged paths 0
6. `git diff --check` exit 0
7. R1R3 Task와 Run Report 존재 및 exact accepted identity를 사용자가 제공한 승인 자료와 대조하도록 명시
8. T002D-SBR1 accepted runtime/source evidence 존재
9. canonical profile 및 bound source identities
10. main/DINO interpreter 존재와 안전성
11. dataset/cache/output/T002D evidence roots 존재와 안전성
12. evidence root가 absolute이고 아직 존재하지 않음

검사 실패 시 evidence root를 만들지 않는다.

### 3. Canonical non-link path verification

Python과 runbook 양쪽에 같은 fail-closed 의미를 구현한다.

- 입력 path를 `resolve()`한 뒤 `is_symlink()`만 검사하는 방식은 금지한다.
- 원래 입력 path의 root부터 leaf까지 존재하는 모든 component를 `lstat` 또는 동등한 방법으로 검사한다.
- symlink, junction 또는 reparse component가 하나라도 있으면 거부한다.
- 검사 뒤 canonical resolved path가 원래 absolute canonical path와 exact하게 일치해야 한다.
- final component뿐 아니라 ancestor component도 검사한다.
- repository, dataset root, checkpoint cache root, runtime interpreters와 그 runtime root, output parent, evidence parent 및 run root에 적용한다.
- missing path를 `resolve(strict=False)`로 안전하다고 간주하지 않는다.
- TOCTOU를 없앴다고 과장하지 않는다. 가능한 범위에서 검사하고, publication은 기존 atomic no-clobber를 유지한다.

Windows synthetic tests에서는 symlink capability가 있을 때 실제 ancestor symlink를 사용한다. capability가 없으면 새 production test를 skip하지 말고 injectable `lstat`/path-component probe로 deterministic하게 symlink/reparse 결과를 주입한다.

### 4. Single canonical plan and preflight evidence

Runbook에서 `plan`과 `preflight`는 각각 JSON file 하나만 만든다.

권장 형태:

```bash
"$MAIN_PYTHON" -B "$REPO_ROOT/scripts/extract_features.py" plan \
  ... \
  --output-json "$EVIDENCE_ROOT/02-plan.json" \
  > /dev/null

"$MAIN_PYTHON" -B "$REPO_ROOT/scripts/extract_features.py" preflight \
  ... \
  --output-json "$EVIDENCE_ROOT/03-preflight.json" \
  > /dev/null
```

다음 파일은 생성하지 않는다.

```text
02-plan.stdout.json
03-preflight.stdout.json
```

Preflight SHA manifest와 retrieval 목록은 실제 생성된 canonical file 전부와 정확히 일치해야 한다.

### 5. Stable worker failure classification

Raw child return code를 `stable_exit_code`로 그대로 기록하지 않는다.

- worker launch/process failure의 `stable_exit_code`는 repository-defined `EXIT_EXECUTION=5`다.
- raw child return code는 별도 optional field에 기록할 수 있지만 정수 범위, 민감정보 부재 및 canonical schema를 검증한다.
- negative signal-style return code, 0보다 큰 255 초과 값 및 일반 nonzero 값 모두 deterministic하게 처리한다.
- worker failure 때문에 failure-evidence writer 자체가 contract error로 바뀌어서는 안 된다.
- stdout/stderr body, traceback, credential 또는 absolute dataset/cache path를 failure evidence에 저장하지 않는다.

### 6. Worker and orchestration failure evidence

`run_state.json`이 안전하게 publish된 뒤 발생하는 worker 또는 parent-side orchestration failure는 가능한 경우 정확히 하나의 append-only canonical `failures/attempt-NNNN.json`을 남긴다.

최소 failure stage:

```text
worker_launch
worker_process
worker_artifact_reconciliation
completion
model_free_validation
```

Failure evidence는 기존 필드에 더해 stage를 명확히 구분한다. model/shard range를 알 수 있는 stage에서는 exact model/lane/shard/image range를 기록한다. Completion/validation stage처럼 단일 active shard가 없는 경우에는 schema가 이를 거짓 model/shard 값으로 가장하지 않도록 명시적 nullable 또는 orchestration scope를 사용하고 validator가 exact하게 검사한다.

다음을 보장한다.

- `run_manifest.json`은 만들지 않는다.
- 기존 artifact와 competing bytes는 유지한다.
- failure evidence를 쓰기 전에 strict `reconcile_resume()`가 partial pair 때문에 예외를 던져 원래 failure를 가리지 않게 한다.
- fully valid consecutive shard pair는 read-only로 확인한다.
- partial tensor/sidecar pair는 failed current shard로 분류하되 삭제하거나 덮어쓰지 않는다.
- completion 중 일부 model manifest 또는 validation directory가 만들어졌더라도 cleanup하지 않는다.
- 이러한 finalization residue는 automatic resume을 fail closed로 막고 manual independent review를 요구한다.
- failure evidence publication 자체가 불가능하면 원인을 보존하고 더 위험한 overwrite/cleanup fallback을 사용하지 않는다.

Run root 또는 `run_state.json`이 만들어지기 전의 initialization failure는 run-local evidence를 조작해 만들지 않는다. 이 경계는 contract와 CLI error JSON에 명시한다.

### 7. Resume and metrics honesty

Resume은 기존 exact rules를 유지한다.

- failure evidence 보존
- consecutive complete shard pair만 인정
- partial pair, unexpected nested directory, corrupt hash, early/competing manifest 거부
- complete run 자동 resume 금지

이미 모든 shard가 존재하지만 trustworthy whole-model metrics가 없는 상태에서 짧은 재실행 시간과 0개 처리량을 전체 모델 metrics처럼 기록하지 않는다. 이 경우 fail closed하거나, contract와 schema가 명시한 truthful resumed-invocation scope를 기록한다. 어느 방식을 선택했는지 contract, implementation 및 tests가 일치해야 한다.

### 8. Runbook preflight/smoke boundary

Runbook은 계속 두 단계로 나눈다.

#### Stage A: read-only preflight

- runtime-live validation
- exact dataset 1,000-row source/decoded-RGB validation
- exact checkpoint inventory
- canonical plan
- extraction preflight
- readiness false
- evidence SHA manifest
- stop

#### Stage B: separately approved smoke

Stage A evidence의 별도 독립 승인 전에는 실행하지 않는다.

- first 10 ordered images
- batch size 1
- FP32
- three lanes sequential
- canonical run ID
- model-free validator
- body-free review package
- readiness false
- stop

Full extraction, full batch-size selection, cleanup, automatic retry 및 automatic resume은 금지한다.

### 9. Contract alignment

`docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md`를 corrected implementation과 일치시킨다.

반드시 명시한다.

- all-component non-link verification
- stable exit code와 optional raw return code의 차이
- pre-initialization failure와 post-`run_state` structured evidence 경계
- worker 및 orchestration failure stage
- partial publication/finalization residue의 fail-closed 처리
- resume metrics의 truthful scope
- T003B preflight가 smoke와 분리되고 full extraction을 승인하지 않음

Scientific identity는 변경하지 않는다.

## Required tests

모든 새 test는 deterministic, offline, order-independent, synthetic이어야 한다. real dataset, checkpoint, model, CUDA 또는 network를 사용하지 않는다.

### 1. Focused unit tests

최소 다음을 검증한다.

1. repository root leaf symlink rejection
2. repository ancestor symlink/reparse rejection
3. dataset/cache/runtime/output ancestor link rejection
4. canonical non-link root acceptance
5. negative worker return code가 stable exit 5로 기록됨
6. 255 초과 return code가 stable exit 5로 기록됨
7. ordinary nonzero return code가 stable exit 5로 기록됨
8. partial tensor-only artifact 후 failure evidence 기록과 competitor preservation
9. partial sidecar-only artifact 후 failure evidence 기록과 competitor preservation
10. completion failure evidence
11. model-free validation failure evidence
12. failure-evidence append-only attempt numbering
13. finalization residue의 automatic resume rejection
14. all-shards/no-metrics resume가 거짓 전체 metrics를 만들지 않음
15. existing atomic no-clobber behavior 유지

### 2. Focused integration tests

최소 다음을 검증한다.

1. synthetic three-encoder smoke round trip
2. worker launch failure
3. negative-return worker process failure
4. mid-shard/partial-pair failure
5. completion collision
6. model-free validation failure
7. no run manifest on all failure paths
8. runbook requires the three user-provided environment variables
9. runbook rejects empty/relative/placeholder roots
10. all repository/predecessor/path checks precede evidence-root creation
11. runbook plan output count = 1
12. runbook preflight output count = 1
13. `.stdout.json` duplicate paths absent
14. preflight SHA manifest and retrieval inventory exact
15. preflight-to-smoke independent approval stop retained
16. full extraction command absent
17. network/package/model/CUDA real access absent

Focused tests:

```text
exit = 0
failed/error/skipped/xfail/deselected = 0
```

Passed count는 실제 값을 기록한다.

### 3. Existing R1R2 regressions

R1R2의 focused tests 전체와 원 T003A 14-file regression set을 외부 `--basetemp`로 실행한다.

14-file regression expected baseline:

```text
384 passed
failed/error/skipped/xfail/deselected = 0
```

실제 node count가 다르면 원인을 증명하지 않고 expected 값을 바꾸지 말고 `BLOCKED_REGRESSION_BASELINE_MISMATCH`로 종료한다.

### 4. Full offline suite

모든 final source 변경과 final compile 뒤 full `tests` suite를 정확히 한 번 실행한다.

Windows symlink capability probe가 unavailable이면 다음 exact raw set만 normalization할 수 있다.

Failures:

```text
tests/integration/test_dinov3_identity_transaction.py::test_recovery_rejects_journal_temporary_symlink_without_overwrite
tests/unit/test_dinov3_identity_rebind.py::test_symlink_destination_is_rejected
tests/unit/test_dinov3_identity_rebind.py::test_transaction_root_symlink_is_rejected
tests/unit/test_dinov3_semantic_probe_profile.py::test_workspace_resolver_rejects_symlink_component
```

Skips:

```text
tests/unit/test_dinov3_runtime_profile.py::test_workspace_symlink_escape_rejected
tests/unit/test_dinov3_snapshot_profile.py::test_symlink_and_reparse_paths_rejected
```

허용 조건:

```text
raw failed = exact 4 nodes above
raw skipped = exact 2 nodes above
errors = 0
xfail = 0
deselected = 0
all other nodes passed
five pre-existing symlink test files changed = 0
```

Symlink capability가 available이면 failed/skipped/error/xfail/deselected 모두 0이어야 한다. 새 R1R3 tests에는 이 normalization을 적용하지 않는다.

## Required validation

모든 command, stdout/stderr log, exit code, duration 및 summary를 외부 R1R3 root에 기록한다.

### 1. Final compile

```powershell
py -3 -B -m compileall -q src\vision_encoder_alignment scripts tests
```

필수 exit `0`.

### 2. Canonical plan

기존 profile로 plan을 실행한다.

필수:

```text
exit = 0
profile SHA = 98d2dfe8e711efdb6f33ac45e87e21318ec53db8b1cc8f6a0aed789babc58cf5
ready_for_p1_t003b = false
ready_for_p1_extraction = false
full_batch_size = null
```

### 3. Focused tests

R1R3 unit/integration tests와 기존 R1R2 focused tests를 실행한다.

### 4. 14-file regression

R1R2가 사용한 exact 14-file set을 변경 없이 실행한다.

### 5. Full suite

위 capability policy에 따라 raw와 normalized verdict를 모두 기록한다.

### 6. Static forbidden-operation inspection

수정된 production code, tests 및 runbook에서 다음을 검사한다.

```text
shell=True = 0
production os.replace = 0
network client/download path = 0
package install command = 0
-I plus PYTHONPATH dependency = 0
parent environment full copy = 0
hardcoded DATASET_ROOT = 0
hardcoded CACHE_ROOT = 0
plan stdout duplicate evidence = 0
preflight stdout duplicate evidence = 0
full extraction command in T003B runbook = 0
automatic retry/resume/cleanup = 0
```

`os.link`, `shell=False`, `-B`, allowlisted offline environment 및 atomic no-clobber는 유지한다.

### 7. Repository verification

Run Report 작성 전과 작성 후 각각 기록한다.

```text
git branch --show-current
git rev-parse HEAD
git status --short --untracked-files=all
git diff --stat
git diff --check
git diff --name-only
git diff --cached --name-only
```

필수:

```text
branch = main
HEAD = e518a51ba83f70b42611cc4386a9f55dc9e97de3
staged paths = 0
unexpected repository path = 0
protected changed/missing/unexpected/reparse = 0
git diff --check exit = 0
final status entries = 345
```

### 8. Protected comparison

Task 시작 시 allowed paths 밖의 regular file path/bytes/SHA와 symlink/reparse classification을 저장하고 종료 시 streaming compare한다.

특히 R1R2 Task, Run Report, package source, canonical profile, standalone validator 및 5개 pre-existing symlink test file은 exact unchanged여야 한다.

## Non-self-referential Run Report sealing and review packaging

다음 순서를 지킨다.

```text
final implementation and validation
-> protected comparison and pre-report Git evidence
-> R1R3 Run Report finalization
-> post-report repository seal
-> review-submission-manifest finalization
-> payload validation
-> ZIP creation exactly once
-> detached SHA creation
-> package-identity JSON creation
-> no further repository or ZIP-payload-source write
```

R1R3 Run Report에는 final ZIP 자체의 SHA를 사후 기록하지 않는다. Final ZIP identity는 post-report 외부 evidence, package identity 및 Codex final response에만 기록한다.

### Required review outputs

```text
G:\내 드라이브\VEA_P1_T003A_R1R3_TEMP_20260730\review\
  VEA_P1_T003A_R1R3_INDEPENDENT_REVIEW_20260730.zip
  VEA_P1_T003A_R1R3_INDEPENDENT_REVIEW_20260730.zip.sha256.txt
  VEA_P1_T003A_R1R3_INDEPENDENT_REVIEW_20260730.package-identity.json
```

ZIP 최소 포함:

- R1R2 Task와 Run Report
- R1R3 Task와 final Run Report
- corrected contract
- unchanged canonical profile
- corrected library/runner
- unchanged standalone validator
- corrected tests
- corrected T003B runbook
- R1R2 root manifest와 post-report seal
- R1R3 precondition/package reconciliation evidence
- canonical plan JSON
- final compile log
- focused full log
- 14-file regression full log
- symlink capability probe
- full-suite full raw log와 normalized verdict
- forbidden-operation inspection
- protected comparison
- pre-report Git evidence
- post-report repository seal
- root `review-submission-manifest.json`

ZIP payload에 dataset image, checkpoint body, model/runtime body, feature Safetensors 또는 pytest temp body를 넣지 않는다.

Root manifest는 모든 ZIP regular payload의 relative path, bytes 및 SHA-256을 exact하게 기록한다. Manifest 자신은 payload list에서 제외하고 ZIP root에 하나만 둔다.

ZIP 검증:

```text
missing = 0
unmanifested = 0
bytes mismatch = 0
SHA mismatch = 0
duplicate = 0
unsafe path = 0
symlink = 0
CRC error = 0
```

Detached SHA와 package identity는 ZIP 밖에 둔다. Package identity는 ZIP/detached SHA/root manifest/Run Report/post-report seal identity 및 seal 이후 modification count 0을 exact하게 기록한다.

## Deliverables

### Repository

```text
tasks/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md
reports/runs/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md
docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md
src/vision_encoder_alignment/p1_feature_extraction.py
scripts/extract_features.py
tests/unit/test_p1_feature_extraction.py
tests/integration/test_p1_feature_extraction_cli.py
reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md
```

### External

```text
evidence/precondition-reconciliation.json
evidence/canonical-plan.json
evidence/post-report-repository-seal.json
logs/*
manifests/*
review/VEA_P1_T003A_R1R3_INDEPENDENT_REVIEW_20260730.zip
review/VEA_P1_T003A_R1R3_INDEPENDENT_REVIEW_20260730.zip.sha256.txt
review/VEA_P1_T003A_R1R3_INDEPENDENT_REVIEW_20260730.package-identity.json
```

## Acceptance criteria

R1R3를 `PASS`로 보고하려면 모두 충족해야 한다.

1. repository branch/HEAD/staged/343-entry start baseline exact
2. R1R2 ZIP 131207 bytes / `b5c5500b…82e4` exact
3. R1R2 detached SHA와 package identity 검증 성공
4. R1R2 repository identities exact
5. R1R2 Task, Run Report, review package 및 original T003A evidence 변경 0
6. canonical profile 변경 0
7. runbook이 사용자 제공 dataset/cache/evidence variables를 요구
8. empty/relative/placeholder root 거부
9. 모든 identity/path check가 evidence root 생성보다 앞섬
10. Python과 runbook이 모든 path component의 symlink/reparse를 fail closed
11. plan canonical JSON 정확히 1개
12. preflight canonical JSON 정확히 1개
13. preflight SHA manifest/retrieval inventory exact
14. stable worker exit code가 raw return code와 분리됨
15. negative/oversized/nonzero return code에서도 failure evidence 작성
16. worker와 post-initialization orchestration failure evidence 구현
17. partial pair/competing bytes 보존
18. failure path에서 `run_manifest.json` 부재
19. finalization residue automatic resume 거부
20. resume metrics가 실제 처리 범위를 과장하지 않음
21. preflight와 smoke independent approval pause 유지
22. full extraction과 full batch selection 금지
23. final compile exit 0
24. canonical plan exit 0 및 readiness false
25. focused tests zero failure/skip/xfail/deselection
26. 14-file regression exact 384 pass
27. full suite가 exact symlink capability policy 충족
28. pre-existing symlink test file 변경 0
29. unauthorized operation 0
30. protected mutation 0
31. staged paths 0
32. final status entries 345
33. Run Report 완전
34. Run Report seal 뒤 repository modification 0
35. ZIP 생성 뒤 payload-source modification 0
36. R1R3 ZIP, detached SHA 및 package identity 검증 성공
37. `READY_FOR_P1_T003B=false`
38. `READY_FOR_P1_EXTRACTION=false`

## Stop conditions

다음이면 unsafe workaround 없이 `BLOCKED`로 종료한다.

- precondition identity 또는 R1R2 package 불일치
- R1R2 detached SHA/package identity 부재
- active R1R3 path 또는 external root 선존재
- allowed scope 밖 file 변경 필요
- canonical profile 또는 scientific identity 변경 필요
- real dataset/checkpoint/model/CUDA 접근 필요
- dependency 설치 또는 network 필요
- standalone validator 수정 필요
- protected file 변경
- focused test failure/skip/xfail/deselection
- 14-file regression이 384 pass가 아님
- full-suite result가 exact capability policy와 다름
- Git staged path 또는 unexpected path 발생
- non-self-referential package를 안전하게 만들 수 없음

BLOCKED 시 cleanup, restore, retry, skip/xfail 추가, expected count 수정, profile rebinding, Git mutation, Apple 접속 또는 T003B 실행을 수행하지 않는다.

## Required Run Report

다음에 작성한다.

```text
reports/runs/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md
```

포함:

1. Task ID, mode, final status
2. predecessor disposition
3. R1R2 ZIP/detached SHA/package identity verification
4. start branch/HEAD/status/staged
5. exact starting path identities
6. changed files
7. runbook input correction
8. pre-evidence-root ordering
9. all-component non-link validation
10. single plan/preflight output
11. stable/raw worker exit classification
12. worker/orchestration failure evidence
13. partial/finalization residue behavior
14. resume/metrics behavior
15. contract alignment
16. commands and exit codes
17. focused test full result
18. 14-file regression full result
19. symlink capability probe
20. full-suite raw and normalized result
21. static forbidden-operation inspection
22. unauthorized operation counts
23. protected comparison
24. pre-report Git evidence
25. deviations
26. remaining risks
27. non-self-referential packaging policy와 external seal/package filenames
28. readiness
29. next action

성공 상태:

```text
VEA-P1-T003A-R1R2 = SUPERSEDED_BLOCKED_INDEPENDENT_REVIEW
VEA-P1-T003A-R1R3 = PASS
READY_FOR_P1_T003B = false
READY_FOR_P1_EXTRACTION = false
```

Codex `PASS`는 독립 검토 전 `ACCEPTED_PASS`가 아니다.

## Required final response

다음 순서로 보고한다.

1. `Status`
2. `Predecessor disposition`
3. `R1R2 package verification`
4. `Summary`
5. `Changed files`
6. `Runbook corrections`
7. `Path safety`
8. `Failure evidence`
9. `Resume and metrics`
10. `Validation commands and exit codes`
11. `Focused and regression tests`
12. `Full-suite raw and normalized result`
13. `Protected-file comparison`
14. `Unauthorized operations`
15. `Git status`
16. `Post-report repository seal`
17. `Review ZIP`
18. `Detached SHA`
19. `Package identity`
20. `Deviations`
21. `Remaining risks`
22. `Readiness`
23. `Next action`

T003B를 시작하지 않는다.

# 이번 작업 정리

## 작업 목적

R1R2 결과를 되돌리지 않고, 독립 검토에서 확인된 T003B runbook의 입력·실행 순서·경로 안전성·중복 evidence 문제와 runner의 failure-evidence 누락 가능성을 제한적으로 보정한다.

## 핵심 수행 내용

- 사용자 제공 dataset/cache/evidence root 계약 적용
- evidence root 생성 전 모든 identity 검사
- 모든 path component의 symlink/reparse 거부
- plan/preflight 단일 canonical JSON
- stable worker exit 및 worker/orchestration failure evidence
- partial/finalization residue와 resume/metrics의 정직한 fail-closed 처리
- focused, 14-file regression 및 capability-normalized full-suite 검증
- 비순환 R1R3 review package 생성

## 이번 작업에서 하지 않는 것

- Apple 접속 또는 preflight
- dataset image/checkpoint body/model/CUDA 접근
- smoke/full extraction
- full batch size 결정
- scientific profile 변경
- R1R2 evidence 수정
- Git mutation

## 완료 기준

- runbook이 active R1R2 요구사항과 이번 보정 계약을 모두 충족
- failure path에서 overwrite 없이 canonical evidence 보존
- required tests와 Git/protected 검증 통과
- R1R3 ZIP, detached SHA, package identity 검증 성공
- `READY_FOR_P1_T003B=false`
- `READY_FOR_P1_EXTRACTION=false`

## 다음 단계

사용자가 R1R3 review ZIP, detached SHA, package-identity JSON과 Codex 출력을 반환하면 독립 검토한다. `VEA-P1-T003A-R1R3=ACCEPTED_PASS`가 확정된 뒤에만 별도 `VEA-P1-T003B` Task로 Apple read-only preflight를 실행한다. Preflight evidence가 다시 독립 승인되기 전에는 10-image smoke를 실행하지 않는다.

## 한눈에 보는 작업 요약

현재 결과는 ZIP 무결성은 통과했지만 runbook과 failure 처리 계약이 명세를 충족하지 않아 T003B로 갈 수 없다. R1R3는 이 실행 안전성만 고치고 다시 검증하며, 성공해도 실제 추출은 시작하지 않는다.
