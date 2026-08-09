# VEA-P2-T001B-R2 — Offline coherent DINOv3 snapshot materialization

- Mode: `recovery`
- Task class: `STANDARD`
- Evidence level: `COMPACT`
- Package required: `false`
- Recommended Codex model: `GPT-5.6 Sol`
- Reasoning effort: `high`
- Goal: implement an offline, atomic materializer that converts the existing exact Hugging Face DINOv3 cache snapshot into the repository's validated external snapshot format and rebinds the live Apple profile chain coherently.
- Baseline: `1f6135d5c0542368bf294071454b244a746c80d6`
- Baseline branch: `origin/codex/vea-p2-t001b-r1-explicit-apple-roots`
- Git delivery: `COMMIT_AND_PUSH`
- Task branch: `codex/vea-p2-t001b-r2-coherent-snapshot`
- Commit condition: `PASS_ONLY`
- Commit message: `fix(vlm-compat): VEA-P2-T001B-R2 materialize coherent DINO snapshot`
- Push target: `origin HEAD`

## Preconditions

- Read this Task from the Google Drive launch repository, but run no Git command there.
- Work only in a clean isolated clone at exact baseline `1f6135d5c0542368bf294071454b244a746c80d6`.
- If the existing isolated repository is not clean at that commit, create a fresh GUID clone outside Google Drive.
- The task branch must not already exist locally or remotely.
- Do not access network, Hugging Face APIs, credentials, Apple, CUDA, a real model, or the dataset.

Before modifying files, run bounded searches only over `configs`, `src`, `scripts`, and `tests` for:

```text
ca8d17a601d520532319230379200c4727c9d62046e21ecb0407498019cf798a
7b83fd74033f359a5a7b4e7d052b8d1bd106c658355ae86499947d151b0176a9
```

Historical `tasks/` and `reports/runs/` references are immutable evidence and must not be changed.

If a live source/config binding outside the allowed paths must change, stop `BLOCKED` and report the exact path.

## Read

1. `AGENTS.md`
2. `src/vision_encoder_alignment/snapshot_boundary.py`
3. `scripts/acquire_dinov3_snapshot.py`
4. `configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json`
5. `configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json`
6. `configs/extraction_profiles/p1_1000_apple_fp32_v0.json`
7. `src/vision_encoder_alignment/p1_feature_extraction.py`
8. `src/vision_encoder_alignment/vlm_compat/dinov3_raw_injection.py`
9. `reports/runs/VEA-P0-T007A-3A-dinov3-snapshot-acquisition-and-offline-load.md`
10. `reports/runbooks/VEA-P2-T001B-apple-dinov3-raw-injection.md`
11. this Task copied to `tasks/VEA-P2-T001B-R2-coherent-snapshot-materialization.md`

Do not scan unrelated tasks, reports, artifacts, caches, or repository history.

## Allowed changes

- `scripts/materialize_dinov3_snapshot_from_hf_cache.py`
- `tests/unit/test_p2_dinov3_snapshot_materialization.py`
- `configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json`
- `configs/extraction_profiles/p1_1000_apple_fp32_v0.json`
- `src/vision_encoder_alignment/p1_feature_extraction.py`
- `reports/runbooks/VEA-P2-T001B-apple-dinov3-raw-injection.md`
- `reports/runs/VEA-P2-T001B-R2-coherent-snapshot-materialization.md`
- `tasks/VEA-P2-T001B-R2-coherent-snapshot-materialization.md`

## Commit paths

Commit only changed files from the eight allowed paths.

Do not commit `.codex/ACTIVE_TASK.md`.

## Root cause

The only Apple candidate is the Hugging Face cache directory:

```text
models--facebook--dinov3-vitb16-pretrain-lvd1689m/
  snapshots/5931719e67bbdb9737e363e781fb0c67687896bc
```

It contains the required model files but is not a valid external snapshot root.

The validated root must be:

```text
dinov3-vitb16-r5931719-safetensors-v1/
├── snapshot-manifest.json
└── files/
    ├── config.json
    ├── preprocessor_config.json
    └── model.safetensors
```

A second identity defect also exists:

```text
current snapshot profile SHA:
df90c58f2fd603b1e7979576270e67924bd84594eb08ad337edc3fdcdd2b80ac

current Apple extractor expected external manifest SHA:
ca8d17a601d520532319230379200c4727c9d62046e21ecb0407498019cf798a
```

The current validator requires the manifest's `repository_profile_sha256` to equal the current snapshot profile SHA.

Using the accepted acquisition provenance with the current snapshot profile must produce:

```text
expected canonical manifest bytes: 1129
expected canonical manifest SHA-256:
ec13f3f45b2b0976319122d123f3bacdd583ed13c68f29eec6a79462e4e7d6ec
```

## Requirements

### 1. Offline materializer CLI

Create:

```text
scripts/materialize_dinov3_snapshot_from_hf_cache.py
```

Commands:

```text
materialize
verify
```

Required arguments:

```text
--repository-root
--source-cache-root
--source-snapshot-root
--destination-root
--runtime-root
```

The script must be standard-library-only at import time and must never:

- access network
- import Hugging Face Hub
- read credentials
- install packages
- load Torch, Transformers, or the model
- modify the source cache
- modify the runtime or repository

Required success markers:

```text
DINOV3_CONTRACT_SNAPSHOT_MATERIALIZED
DINOV3_CONTRACT_SNAPSHOT_ALREADY_VALID
DINOV3_CONTRACT_SNAPSHOT_VERIFY_OK
```

### 2. Exact source validation

Select only:

```text
config.json
preprocessor_config.json
model.safetensors
```

Require exact identities from the current repository snapshot profile:

```text
config.json
bytes = 744
sha256 = 3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec

preprocessor_config.json
bytes = 585
sha256 = 960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7

model.safetensors
bytes = 342662192
sha256 = 9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b
```

Hugging Face source files may be symlink leaves, but each resolved target must:

- be a regular file
- resolve inside the supplied source cache root
- match exact byte count and SHA-256

Do not copy any README, tokenizer, lock, ref, blob inventory, or additional file.

### 3. Plain external snapshot

Destination root basename must be exactly:

```text
dinov3-vitb16-r5931719-safetensors-v1
```

The destination must be outside:

- repository root
- DINO runtime root
- source cache root

Destination files must be newly copied plain regular files.

Do not create:

- symlinks
- hard links
- junctions
- references back to the Hugging Face cache

### 4. Deterministic manifest

Construct canonical JSON through the existing `snapshot_boundary.canonical_json_bytes`.

Preserve the accepted acquisition provenance:

```text
provisioned_at_utc = 2026-07-20T17:53:45.969924Z
credential_backed_logical_request_count = 2
model_weight_transferred_bytes = 342662192
integrity_status = VERIFIED
network_independent_verification_status = VERIFIED
```

Set:

```text
repository_profile_sha256 =
SHA-256 of the current repository snapshot profile bytes
```

Require:

```text
manifest bytes = 1129
manifest SHA-256 =
ec13f3f45b2b0976319122d123f3bacdd583ed13c68f29eec6a79462e4e7d6ec
```

The materialization command must separately print:

```text
source_kind=huggingface_cache_exact_bytes
network_attempt_count=0
materialized_bytes=342663521
accepted_acquisition_provenance_preserved=true
```

Do not claim that the current materialization performed the two historical credential-backed requests.

### 5. Atomic and fail-closed publication

- Require at least 1 GiB free at the destination volume.
- Use a unique same-parent staging directory with the snapshot profile's staging prefix.
- Copy and fsync all three files.
- Write and fsync the manifest.
- Validate the complete staging root using existing `validate_snapshot_root(..., _allow_owned_staging=True)`.
- Publish through a same-parent atomic rename.
- Validate the final root again.
- Remove only the task-owned staging root on failure.
- Never overwrite or repair an existing invalid final root.
- If the final root already exists and is exact-valid, return `ALREADY_VALID` without rewriting bytes.

### 6. Coherent identity rebind

Update only:

```text
configs/extractor_profiles/
  dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json
```

Change only:

```text
identities.external_snapshot_manifest_sha256
```

from:

```text
ca8d17a601d520532319230379200c4727c9d62046e21ecb0407498019cf798a
```

to:

```text
ec13f3f45b2b0976319122d123f3bacdd583ed13c68f29eec6a79462e4e7d6ec
```

Then compute the new Apple extractor profile SHA-256 from final canonical bytes.

Rebind that computed SHA consistently in:

```text
configs/extraction_profiles/p1_1000_apple_fp32_v0.json

src/vision_encoder_alignment/p1_feature_extraction.py
```

Do not hardcode a guessed new Apple profile SHA in advance.

After the rebind:

- the two live references must equal the actual Apple profile SHA
- the old live Apple profile SHA must not remain under `configs` or `src`
- historical task/report evidence remains unchanged
- all JSON remains in its existing canonical format

### 7. Runbook

Update the existing T001B Apple runbook so one user execution performs:

```text
clone reviewed R2 commit
→ validate existing HF source snapshot
→ materialize or reuse exact contract snapshot
→ verify contract snapshot
→ verify runtimes and TinyLLaVA dependencies
→ wait for idle GPU
→ run exactly one R1 raw-injection smoke
→ verify and print result
```

Use destination:

```text
/home/chltjsdh/vea/snapshots/
  dinov3-vitb16-r5931719-safetensors-v1
```

All deterministic snapshot work must finish before GPU waiting.

The actual materialization and smoke must run in child worker processes. Failure must not terminate the user's tmux parent shell.

Do not download, install, create symlinks, or run calibration.

### 8. Tests

Create focused tests covering:

1. exact canonical manifest bytes and SHA
2. current snapshot profile SHA is embedded in the manifest
3. exact three-file allowlist
4. source symlink resolution remains inside the source cache
5. source escape is rejected
6. incorrect size or SHA is rejected before destination publication
7. destination contains only plain files and exact layout
8. existing invalid destination is not overwritten
9. existing exact destination is idempotently reused
10. injected copy/manifest/rename failure removes staging and leaves no final root
11. validator is run on staging and final roots
12. no network/API/runtime/model imports occur
13. Apple extractor manifest identity equals `ec13f3...`
14. P1 extraction profile and Python constant equal the actual new Apple profile SHA
15. R1 explicit-root raw-injection tests remain passing

## Protected scope and non-goals

Do not modify:

- `src/vision_encoder_alignment/snapshot_boundary.py`
- `scripts/acquire_dinov3_snapshot.py`
- `src/vision_encoder_alignment/vlm_compat/dinov3_raw_injection.py`
- `scripts/run_dinov3_raw_injection.py`
- DINO extractor, worker, runtime profile, snapshot profile, or model identity
- TinyLLaVA replay implementation
- dependency files

Do not perform:

- Apple, GPU, model, dataset, or checkpoint execution
- network or credential access
- package installation
- calibration, Ridge, CORAL, Linear, MLP, or benchmark
- main push, merge, amend, rebase, force push

Do not create a review ZIP, detached SHA, package identity, repository-tree manifest, or full protected-tree hash comparison.

## Codex validation: Tier 0–1

Run:

```powershell
py -3.12 -m py_compile `
  scripts/materialize_dinov3_snapshot_from_hf_cache.py

py -3.12 -m pytest -q `
  tests/unit/test_p2_dinov3_snapshot_materialization.py `
  tests/unit/test_p2_dinov3_raw_injection.py `
  tests/unit/test_p1_feature_extraction.py

py -3.12 scripts/materialize_dinov3_snapshot_from_hf_cache.py --help

py -3.12 scripts/run_dinov3_raw_injection.py --help

git diff --check
```

Also verify:

```text
changed paths ⊆ exact eight-path allowlist
staged paths = exact changed commit paths
old live external-manifest hash absent from current Apple profile
old live Apple-profile hash absent from configs/src
new profile bindings equal actual final file hashes
```

Do not run the materializer against real model files during Codex validation.

## User validation: Tier 2–3

After independent review, the user runs the corrected Apple runbook exactly once.

That one run may:

1. materialize or reuse the external snapshot
2. run one raw-injection smoke

No separate diagnostic or calibration run is authorized.

## Git delivery

1. Create the task branch from the exact baseline.
2. Modify only the allowed paths.
3. Run all Tier 0–1 validation.
4. Stage only exact changed commit paths.
5. Commit with:

   ```text
   fix(vlm-compat): VEA-P2-T001B-R2 materialize coherent DINO snapshot
   ```

6. Push with:

   ```bash
   git push -u origin HEAD
   ```

If push alone fails due to authentication or network, retain the valid local commit and report `PARTIAL_GIT_DELIVERY` with the exact push command.

## Done when

- an offline materializer can create the exact contract snapshot from the existing HF cache
- the manifest is deterministic and validates against the current snapshot profile
- the Apple extractor expects the new exact manifest SHA
- the P1 profile chain references the actual updated Apple extractor profile
- all targeted tests pass
- exactly the authorized paths are committed and pushed
- no Apple execution has occurred

## Stop conditions

- baseline or branch precondition mismatch
- live identity reference outside the allowed paths requires modification
- expected manifest SHA cannot be reproduced exactly
- source/destination safety cannot be proven
- shared validator or extractor modification becomes necessary
- real model, network, credential, Apple, or GPU access becomes necessary

## Final response

Within 20 lines return:

- Status and model/reasoning
- Baseline and branch
- Changed paths
- Manifest bytes and expected SHA test result
- Final Apple profile SHA and both rebound references
- Test commands and total passed count
- Compile, CLI help, and `git diff --check`
- Commit message and SHA
- Push result and remote branch
- Deviation or blocker
- Next action: independent review, then one corrected Apple runbook execution

For PASS + push, end with:

```text
Submit Task ID, baseline SHA, commit SHA, and remote branch to ChatGPT.
```

Do not execute Apple or start calibration.

# 이번 작업 정리

## 작업 목적

기존 DINOv3 cache를 검증된 외부 snapshot으로 원자적으로 구성하고 관련 identity hash를 일관되게 갱신한다.

## 핵심 수행 내용

- 네트워크 없는 snapshot materializer 구현
- 정확한 manifest 생성과 검증
- Apple extractor 및 P1 profile hash rebind
- 한 번 실행용 Apple runbook 갱신

## 이번 작업에서 하지 않는 것

모델 실행, GPU 사용, 다운로드, calibration, benchmark를 수행하지 않는다.

## 완료 기준

기존 cache만으로 유효한 snapshot을 만들 수 있고 모든 live profile binding과 targeted test가 일치한 commit이 push된다.

## 다음 단계

독립 검수 후 Apple에서 materialization과 one-image raw injection을 하나의 runbook으로 정확히 한 번 실행한다.

## 한눈에 보는 작업 요약

현재 모델 파일은 이미 있지만 저장 형식과 신분증 역할을 하는 hash가 맞지 않는다. 기존 파일을 다시 받지 않고 올바른 구조와 hash로 포장한 뒤 한 번의 smoke까지 이어지게 만든다.