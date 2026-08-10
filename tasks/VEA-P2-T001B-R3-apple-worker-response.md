# VEA-P2-T001B-R3 — Fix Apple DINO worker response identity

- Mode: `recovery`
- Task class: `STANDARD`
- Evidence level: `COMPACT`
- Package required: `false`
- Recommended Codex model: `GPT-5.6 Sol`
- Reasoning effort: `high`
- Goal: fix the deterministic Apple-worker response failure and coherently rebind only the resulting live source identities.
- Baseline: `6fb258299a08244d1922559e798a50a1e4fee5d0`
- Baseline branch: `origin/codex/vea-p2-t001b-r2-coherent-snapshot`
- Git delivery: `COMMIT_AND_PUSH`
- Task branch: `codex/vea-p2-t001b-r3-apple-worker-response`
- Commit condition: `PASS_ONLY`
- Commit message: `fix(vlm-compat): VEA-P2-T001B-R3 fix Apple worker response identity`
- Push target: `origin HEAD`

## Preconditions

- Read this Task from the Google Drive launch repository, but run no Git command there.
- Work only in a clean isolated repository at exact baseline `6fb258299a08244d1922559e798a50a1e4fee5d0`.
- Create a fresh isolated clone if the existing repository is not clean at that commit.
- The target task branch must not already exist locally or remotely.
- Do not run Apple, CUDA, a real model, a dataset, network access, package installation, or snapshot materialization.

Before editing, perform a bounded fixed-string search under `configs`, `src`, and `tests` for:

```text
d62295b174b980482a9ba6dfbcea9957dec038816fdf2b88dff244404646850f
d296bbbcf747d3d6d8c0a6ed9f48809cee058af4a6009b0342ab4ca55bab4223
b0d8c3fee92a783aa873efedcadab68c48bcde8aac458d767bb5fa77cac1cb8d
60d0d0d7d37c6b4d47997e09619204f9e1eddb96118d5ee3528d9e6fcd5fe829
```

Historical Task and Run Report references are immutable evidence and must not be changed.

If a required live binding exists outside the allowed paths, stop `BLOCKED` and report that exact path rather than weakening validation.

## Read

1. `AGENTS.md`
2. `scripts/dinov3_extraction_worker.py`
3. `src/vision_encoder_alignment/dinov3_extraction_boundary.py`
4. `src/vision_encoder_alignment/extractors/dinov3.py`
5. `configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json`
6. `configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json`
7. `configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json`
8. `configs/extraction_profiles/p1_1000_apple_fp32_v0.json`
9. `src/vision_encoder_alignment/p1_feature_extraction.py`
10. `tests/conftest.py`
11. `tests/unit/test_dinov3_extractor.py`
12. `tests/integration/test_dinov3_extractor_artifact.py`
13. `tests/unit/test_p2_dinov3_raw_injection.py`
14. `tests/unit/test_p2_dinov3_snapshot_materialization.py`
15. `reports/runbooks/VEA-P2-T001B-apple-dinov3-raw-injection.md`
16. this Task copied to `tasks/VEA-P2-T001B-R3-apple-worker-response.md`

## Allowed changes

- `scripts/dinov3_extraction_worker.py`
- `configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json`
- `configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json`
- `configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json`
- `configs/extraction_profiles/p1_1000_apple_fp32_v0.json`
- `src/vision_encoder_alignment/p1_feature_extraction.py`
- `tests/conftest.py`
- `tests/unit/test_dinov3_extractor.py`
- `reports/runbooks/VEA-P2-T001B-apple-dinov3-raw-injection.md`
- `reports/runs/VEA-P2-T001B-R3-apple-worker-response.md`
- `tasks/VEA-P2-T001B-R3-apple-worker-response.md`

## Commit paths

Commit only changed files from the eleven allowed paths.

Do not commit `.codex/ACTIVE_TASK.md`.

## Confirmed defect

In `scripts/dinov3_extraction_worker.py`, the Apple branch creates a production runtime profile but does not create a legacy `runtime_manifest`.

The response is nevertheless built with:

```python
runtime_manifest.candidate_freeze_sha256
runtime_manifest.source_sha256
```

This deterministically raises `UnboundLocalError` in the Apple execution path after successful extraction work.

## Requirements

### 1. Bounded worker correction

Add a small standard-library helper that resolves the two persistent identity fields.

Required semantics:

```text
Apple selector:
  persistent_freeze_sha256
    = profile.identities.persistent_freeze_sha256

  persistent_runtime_manifest_sha256
    = profile.identities.persistent_runtime_manifest_sha256

Legacy selector:
  persistent_freeze_sha256
    = validated runtime_manifest.candidate_freeze_sha256

  persistent_runtime_manifest_sha256
    = validated runtime_manifest.source_sha256
```

The legacy selector must reject a missing runtime manifest.

The Apple selector must succeed with no runtime-manifest object.

Use the two resolved values when constructing the response.

After the correction, the Apple response-building path must contain no access to an unassigned `runtime_manifest`.

### 2. Preserve the protocol

Do not change:

- response schema
- response field names
- expected markers
- model identity
- snapshot identity
- runtime policy
- preprocessing
- taps
- token removal
- model loading
- forward execution
- artifact writing
- RNG, network, torchvision, dtype, or device gates

Do not change `dinov3_extraction_boundary.py`, the public extractor, raw-injection implementation, snapshot materializer, or TinyLLaVA code.

### 3. Focused regression tests

Add tests to `tests/unit/test_dinov3_extractor.py` proving:

1. Apple response identity resolution works with `runtime_manifest=None`.
2. Apple values come exactly from `profile.identities`.
3. Legacy values come exactly from the validated runtime manifest.
4. Legacy mode rejects a missing runtime manifest.
5. The resulting identity mapping remains accepted by the existing extraction-response validator.

Do not mock away the identity-field selection being tested.

No real model, Torch, CUDA, worker subprocess, checkpoint, network, or dataset is permitted.

### 4. Coherent identity rebind

After changing the worker, compute the actual worker file SHA-256.

Update that exact SHA in:

```text
configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json
  source_bindings[path=scripts/dinov3_extraction_worker.py]

configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json
  production_sources[path=scripts/dinov3_extraction_worker.py]

configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json
  production_sources[path=scripts/dinov3_extraction_worker.py]

configs/extraction_profiles/p1_1000_apple_fp32_v0.json
  source_bindings[path=scripts/dinov3_extraction_worker.py]

src/vision_encoder_alignment/p1_feature_extraction.py
  exact worker source binding
```

Then compute the actual new Apple runtime-profile SHA-256 and update both of these Apple extractor fields:

```text
identities.runtime_profile_sha256
runtime_profile.sha256
```

Compute the resulting actual Apple extractor-profile SHA-256 and update:

```text
configs/extraction_profiles/p1_1000_apple_fp32_v0.json
  runtime.dinov3.extractor_profile.sha256

src/vision_encoder_alignment/p1_feature_extraction.py
  exact Apple extractor-profile SHA
```

Also update the P1 JSON and Python runtime-profile references to the actual new Apple runtime-profile SHA.

Do not guess any resulting SHA in advance.

Keep this existing external snapshot-manifest identity unchanged:

```text
ec13f3f45b2b0976319122d123f3bacdd583ed13c68f29eec6a79462e4e7d6ec
```

Do not alter the legacy profile's historical runtime, snapshot, freeze, or manifest identities. Only its current worker production-source row and resulting test constants may change.

Update only the exact live-profile SHA constants in:

```text
tests/conftest.py
tests/unit/test_dinov3_extractor.py
```

Do not weaken, delete, skip, or broadly regenerate identity assertions.

### 5. Runbook

Update the existing Apple runbook to:

- use branch `codex/vea-p2-t001b-r3-apple-worker-response`
- retain the reviewed-commit placeholder
- reuse the already materialized snapshot idempotently
- run exactly one raw-injection smoke
- preserve the current parent-shell safety and GPU waiting
- add no new diagnostic/model run

## Protected scope

Do not modify:

- `src/vision_encoder_alignment/dinov3_extraction_boundary.py`
- `src/vision_encoder_alignment/extractors/dinov3.py`
- `src/vision_encoder_alignment/vlm_compat/dinov3_raw_injection.py`
- `scripts/run_dinov3_raw_injection.py`
- `src/vision_encoder_alignment/snapshot_boundary.py`
- `scripts/materialize_dinov3_snapshot_from_hf_cache.py`
- runtime or snapshot external files
- model/checkpoint/dataset bytes
- dependency files
- historical tasks or reports

Do not create a review ZIP, detached SHA, package identity, repository tree manifest, or protected-tree hash scan.

## Codex validation: Tier 0–1

Run:

```powershell
py -3.12 -m py_compile `
  scripts/dinov3_extraction_worker.py `
  src/vision_encoder_alignment/p1_feature_extraction.py

py -3.12 -m pytest -q `
  tests/unit/test_dinov3_extractor.py `
  tests/integration/test_dinov3_extractor_artifact.py `
  tests/unit/test_production_runtime_contract.py `
  tests/integration/test_apple_production_runtime_contract.py `
  tests/unit/test_p1_feature_extraction.py `
  tests/unit/test_p2_dinov3_raw_injection.py `
  tests/unit/test_p2_dinov3_snapshot_materialization.py

git diff --check
```

Also run one standard-library identity audit proving:

- worker SHA equals every updated live worker binding
- Apple runtime-profile SHA equals both Apple extractor references
- Apple extractor-profile SHA equals both P1 references
- external snapshot-manifest SHA remains `ec13f3...`
- legacy and Apple profiles both validate their current production-source worker row
- no unauthorized changed path exists

Do not run the worker, model, CUDA, Apple, network, snapshot materialization, or raw-injection smoke.

## User validation: Tier 2–3

After independent review, run the corrected Apple runbook exactly once.

The existing materialized snapshot must be reused through the materializer's `ALREADY_VALID` path. Do not delete or rebuild it manually.

## Git delivery

1. Create the task branch from the exact baseline.
2. Modify only allowed paths.
3. Run all Tier 0–1 validation.
4. Stage only exact changed commit paths.
5. Commit with:

```text
fix(vlm-compat): VEA-P2-T001B-R3 fix Apple worker response identity
```

6. Push with:

```bash
git push -u origin HEAD
```

Do not merge, amend, rebase, force-push, or push `main`.

## Done when

- Apple response construction no longer references an unassigned runtime manifest.
- Legacy response behavior remains unchanged.
- all live worker/runtime/extractor/P1 bindings equal actual file hashes
- focused tests pass
- exact allowed-path commit is pushed
- no Apple execution has occurred

## Stop conditions

- exact baseline unavailable
- target branch already exists
- isolated repository dirty
- required live binding exists outside the allowed paths
- shared boundary or extractor modification becomes necessary
- tests require a real model, CUDA, network, package, or dataset
- identity chain cannot be proven with targeted offline checks

## Final response

Within 20 lines report:

- Status and model/reasoning
- Baseline and branch
- Changed paths
- Defect correction summary
- Worker SHA
- Apple runtime-profile SHA
- Apple extractor-profile SHA
- Test command and passed count
- Compile and `git diff --check`
- Commit message and SHA
- Push result and remote branch
- Deviation or blocker
- Next action: independent review, then one Apple R3 runbook execution

For PASS + push, end with:

```text
Submit Task ID, baseline SHA, commit SHA, and remote branch to ChatGPT.
```

# 이번 작업 정리

## 작업 목적

Apple DINO worker가 정상 추론 후 응답을 만들 때 발생하는 미정의 변수 오류를 제거한다.

## 핵심 수행 내용

- Apple과 legacy의 persistent identity 출처를 명시적으로 분리
- worker 변경으로 영향을 받는 runtime·extractor·P1 hash만 재계산
- 실제 모델 없이 회귀 테스트
- 검증 후 commit과 push

## 이번 작업에서 하지 않는 것

Apple 실행, 모델 실행, snapshot 재생성, calibration 또는 benchmark를 수행하지 않는다.

## 완료 기준

Apple worker 응답 경로가 runtime manifest 없이 안전하게 구성되고 모든 연결 hash와 offline test가 일치한다.

## 다음 단계

독립 검수 후 기존 snapshot을 재사용하여 raw-injection smoke를 정확히 한 번 실행한다.

## 한눈에 보는 작업 요약

DINOv3 계산은 실행됐지만 결과 보고서를 만들 때 Apple 경로에 없는 변수를 읽어 실패했다. 그 보고 단계만 고치고 연결된 신분 확인 hash를 일관되게 갱신한다.