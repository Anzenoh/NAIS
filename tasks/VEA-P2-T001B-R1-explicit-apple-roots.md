# VEA-P2-T001B-R1 — Bind explicit Apple DINO runtime and snapshot roots

- Mode: `recovery`
- Task class: `STANDARD`
- Evidence level: `COMPACT`
- Package required: `false`
- Recommended Codex model: `GPT-5.6 Sol`
- Reasoning effort: `high`
- Goal: remove all `LOCALAPPDATA` and legacy interpreter-layout fallbacks from the T001B Apple path by requiring and validating explicit DINOv3 runtime and snapshot roots.
- Baseline: `9f1af0d6cb6f18c9e78e2e72deaa1d0da14e68ac`
- Baseline branch: `origin/codex/vea-p2-t001b-dinov3-raw-smoke-20260809`
- Git delivery: `COMMIT_AND_PUSH`
- Task branch: `codex/vea-p2-t001b-r1-explicit-apple-roots`
- Commit condition: `PASS_ONLY`
- Commit message: `fix(vlm-compat): VEA-P2-T001B-R1 bind explicit Apple DINO roots`
- Push target: `origin HEAD`

## Preconditions

- Read this Task from the Google Drive launch repository, but run no Git command there.
- Work in a clean isolated clone at exact baseline `9f1af0d6cb6f18c9e78e2e72deaa1d0da14e68ac`.
- If the existing isolated clone is not clean at that commit, clone the exact baseline branch into a fresh GUID path outside Google Drive.
- Verify the target branch is absent locally and remotely.
- Do not install packages, access Hugging Face, load a model, use CUDA, or execute Apple validation.

## Read

1. `AGENTS.md`
2. `src/vision_encoder_alignment/vlm_compat/dinov3_raw_injection.py`
3. `scripts/run_dinov3_raw_injection.py`
4. `tests/unit/test_p2_dinov3_raw_injection.py`
5. `src/vision_encoder_alignment/extractors/dinov3.py`
6. `src/vision_encoder_alignment/dinov3_extraction_boundary.py`
7. `src/vision_encoder_alignment/runtime_boundary.py`
8. `src/vision_encoder_alignment/snapshot_boundary.py`
9. `scripts/extract_features.py`
   - read only `_snapshot()` and `_extractor()`
10. `configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json`
11. `configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json`
12. `reports/runbooks/VEA-P2-T001B-apple-dinov3-raw-injection.md`
13. this Task copied to `tasks/VEA-P2-T001B-R1-explicit-apple-roots.md`

Do not scan unrelated tasks, artifacts, caches, reports, or datasets.

## Allowed changes

- `src/vision_encoder_alignment/vlm_compat/dinov3_raw_injection.py`
- `scripts/run_dinov3_raw_injection.py`
- `tests/unit/test_p2_dinov3_raw_injection.py`
- `reports/runbooks/VEA-P2-T001B-apple-dinov3-raw-injection.md`
- `reports/runs/VEA-P2-T001B-R1-explicit-apple-roots.md`
- `tasks/VEA-P2-T001B-R1-explicit-apple-roots.md`

## Commit paths

Commit only changed files from the six allowed paths.

Do not commit `.codex/ACTIVE_TASK.md`.

## Root cause to close

The baseline T001B implementation constructs `DINOv3ExtractorConfig` without:

```text
runtime_root
snapshot_root
```

This causes two invalid fallbacks on Apple:

```text
runtime root
→ common linux_bin resolver expects <root>/venv/bin/python
→ actual accepted Apple venv is <root>/bin/python or bin/python3

snapshot root
→ resolve_default_snapshot_root()
→ requires Windows LOCALAPPDATA
→ LOCALAPPDATA is absent on Apple
```

The accepted P1 extraction path already avoids both fallbacks by passing explicit runtime and snapshot roots. T001B-R1 must follow the same principle.

## Requirements

### 1. Explicit CLI inputs

Add required CLI arguments:

```text
--dinov3-runtime-root
--dinov3-snapshot-root
```

They must be passed through to:

```python
run_dinov3_raw_injection_smoke(...)
extract_dinov3_block11(...)
```

Do not infer either root from `LOCALAPPDATA`, home directories, or an unrestricted filesystem search.

### 2. Root validation before output creation or GPU work

Add an import-safe validation helper that verifies:

```text
repository root:
  absolute existing non-link directory

DINO runtime root:
  absolute existing non-link directory
  exactly equals runtime_root in the pinned Apple DINO profile
  does not overlap the repository or snapshot root

DINO snapshot root:
  absolute existing non-link directory
  does not overlap the repository or runtime root
  validates through the existing snapshot profile and validate_snapshot_root()
```

The helper must run before:

- output-directory creation
- DINO model loading
- TinyLLaVA model loading
- CUDA allocation

No `LOCALAPPDATA` lookup is permitted in this path.

### 3. Exact Apple interpreter compatibility boundary

The accepted Apple runtime is a venv rooted at:

```text
/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated
```

Its interpreter is directly under:

```text
<runtime-root>/bin/python
```

or, when that executable is absent:

```text
<runtime-root>/bin/python3
```

Resolve the interpreter by requiring:

- regular executable file
- no symlink at the leaf
- child of the exact runtime root
- a subprocess identity probe proving `Path(sys.prefix).resolve()` equals the runtime root

Do not accept `<runtime-root>/venv/bin/python`.

### 4. Bounded compatibility context

Implement a context manager local to `dinov3_raw_injection.py`.

Within that context only, bind the `runtime_python_path` references used by:

```text
vision_encoder_alignment.runtime_boundary
vision_encoder_alignment.dinov3_extraction_boundary
vision_encoder_alignment.extractors.dinov3
```

The replacement must:

- handle only `interpreter_layout="linux_bin"`
- handle only the exact validated Apple DINO runtime root
- return the validated `<root>/bin/python` or `<root>/bin/python3`
- delegate every other root/layout to the original function
- restore all original references in `finally`
- restore them after success and exception

Do not edit the three shared production modules.

### 5. Explicit extractor construction

Construct `DINOv3ExtractorConfig` with all four exact bindings:

```text
profile_path
apple_runtime_profile_path
runtime_root
snapshot_root
```

The DINO extraction must still use:

```text
model = facebook/dinov3-vitb16-pretrain-lvd1689m
revision = 5931719e67bbdb9737e363e781fb0c67687896bc
tap = block_11
execution selector = dinov3_apple_cuda_fp32_v1
```

Do not modify the verified extractor, worker, runtime profile, snapshot profile, or identity graph.

### 6. Preserve raw-injection semantics

Do not change:

- canonical image geometry
- DINOv3 block-11 selection
- special-token removal
- raw NumPy-to-Torch conversion
- TinyLLaVA model or revision
- connector injection
- diagnostics
- result schema
- no-calibration policy

No centering, normalization, Ridge, CORAL, Procrustes, Linear, MLP, pooling, or token permutation may be added.

### 7. Apple runbook hardening

Update the existing T001B runbook so all deterministic failures occur before GPU waiting.

The runbook must:

1. clone the reviewed commit
2. verify main Python and DINO runtime root
3. determine the exact DINO interpreter under `<runtime-root>/bin`
4. find and validate the snapshot root before GPU waiting:
   - first test the exact P1 checkpoint-cache candidate

     ```text
     <HF_HUB_CACHE>/models--facebook--dinov3-vitb16-pretrain-lvd1689m/
       snapshots/5931719e67bbdb9737e363e781fb0c67687896bc
     ```

   - if that candidate is invalid, perform only a bounded read-only search below
     `/home/chltjsdh/vea/checkpoints` and `/home/chltjsdh/vea/snapshots`
     for directories named `dinov3-vitb16-r5931719-safetensors-v1`
   - validate every candidate with the repository snapshot profile
   - select the cache candidate when valid; otherwise require exactly one valid candidate
5. print the selected runtime interpreter and snapshot root
6. verify SentencePiece 0.2.0 and protobuf 4.25.3
7. wait for two idle-GPU checks
8. execute exactly one raw-injection smoke with both new CLI arguments
9. verify and print the result and hashes

The actual smoke must run in a child worker process. Failure must not terminate the tmux parent shell.

Do not download, install, modify runtime files, create symlinks, or patch repository source on Apple.

## Regression tests

Keep all existing tests and add focused coverage for:

1. CLI requires both new DINO root arguments
2. explicit roots are passed unchanged to extractor construction
3. execution succeeds with `LOCALAPPDATA` absent
4. no default runtime or snapshot resolver is called
5. runtime profile root mismatch is rejected
6. snapshot validation is called with the exact repository and runtime roots
7. `<runtime-root>/bin/python` is accepted
8. `<runtime-root>/venv/bin/python` is never selected
9. all patched function references are restored after success
10. all patched function references are restored after exception
11. raw-injection behavior and result schema remain unchanged

Tests must use fakes and temporary directories only. They must not require Torch, Transformers, CUDA, models, network, checkpoint bodies, or Apple.

## Non-goals and protected scope

Do not modify:

- `runtime_boundary.py`
- `snapshot_boundary.py`
- `dinov3_extraction_boundary.py`
- `extractors/dinov3.py`
- DINO worker
- runtime, extractor, or snapshot profiles
- model identities
- dependency files
- R4 native-replay implementation

Do not perform:

- package installation
- model/checkpoint download
- GPU or Apple execution
- calibration or benchmark work
- main push, merge, amend, rebase, or force-push

## Codex validation: Tier 0–1

Run:

```powershell
py -3.12 -m py_compile `
  src/vision_encoder_alignment/vlm_compat/dinov3_raw_injection.py `
  scripts/run_dinov3_raw_injection.py

py -3.12 tests/unit/test_p2_dinov3_raw_injection.py

py -3.12 scripts/run_dinov3_raw_injection.py --help

git diff --check
```

Require:

- all previous 8 tests pass
- all new root-binding tests pass
- unexpected changed paths = 0
- staged paths are an exact subset of the six allowed paths

Do not claim Apple, GPU, model, runtime, or snapshot validation.

## User validation: Tier 2–3

After independent review, execute the corrected Apple runbook exactly once.

No additional diagnostic or process-local shim is permitted before that run.

## Git delivery

1. Create the Task branch from exact baseline.
2. Modify only allowed paths.
3. Run all Tier 0–1 checks.
4. Stage only exact commit paths.
5. Commit with:

   ```text
   fix(vlm-compat): VEA-P2-T001B-R1 bind explicit Apple DINO roots
   ```

6. Push with:

   ```bash
   git push -u origin HEAD
   ```

If push alone fails for network or authentication, preserve the local commit and return `PARTIAL_GIT_DELIVERY` with the exact push command.

Do not create a review ZIP, detached SHA, repository-tree manifest, or sealed package.

## Done when

- T001B reads neither runtime nor snapshot location from `LOCALAPPDATA`
- runtime and snapshot roots are required explicit inputs
- the exact Apple interpreter is validated before model work
- the exact snapshot is validated before GPU waiting
- all shared production files remain unchanged
- targeted tests pass
- the correction commit is pushed

## Stop conditions

Stop without workaround if:

- exact baseline is unavailable
- target branch already exists
- isolated worktree is dirty
- the correction requires editing a protected production file
- tests require a real model, checkpoint, GPU, network, or package install
- root binding cannot be proven with focused fake tests

## Final response

Return no more than 20 lines:

1. Status
2. model and reasoning
3. baseline SHA
4. branch
5. changed paths
6. previous and final test counts
7. compile and CLI results
8. `git diff --check`
9. commit message
10. commit SHA
11. push result
12. remote branch
13. deviations or blocker
14. next action: independent review, then one corrected Apple smoke

For `PASS + push`, state:

```text
Submit Task ID, baseline SHA, commit SHA, and remote branch to ChatGPT.
```

Do not run Apple or start calibration.

# 이번 작업 정리

## 작업 목적

Apple DINO runtime과 snapshot 위치를 명시적으로 전달해 Windows 기본 경로로 내려가는 두 결함을 모두 제거한다.

## 핵심 수행 내용

- runtime root와 snapshot root를 필수 CLI 인자로 추가
- 실제 `<runtime-root>/bin/python*` 검증
- 기존 snapshot contract로 실행 전 검증
- process-local compatibility context의 안전한 복원
- Apple runbook의 GPU 전 preflight 강화

## 이번 작업에서 하지 않는 것

공통 production boundary 수정, runtime 변경, package 설치, calibration, benchmark를 수행하지 않는다.

## 완료 기준

명시적 두 root만으로 T001B가 구성되며 모든 offline regression과 Git delivery가 통과한다.

## 다음 단계

독립 검수 후 수정된 runbook으로 one-image raw-injection smoke를 정확히 한 번 실행한다.

## 한눈에 보는 작업 요약

이번에는 오류가 발생할 때마다 환경 변수를 덧붙이지 않는다. DINO runtime과 checkpoint 위치를 처음부터 정확히 넘기고, GPU를 기다리기 전에 둘 다 검증한다.