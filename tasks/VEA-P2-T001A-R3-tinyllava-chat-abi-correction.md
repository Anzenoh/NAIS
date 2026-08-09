# VEA-P2-T001A-R3 — Correct pinned TinyLLaVA chat ABI and unblock Apple replay

- Mode: `recovery`
- Task class: `STANDARD`
- Evidence level: `COMPACT`
- Package required: `false`
- Recommended Codex model: `GPT-5.6 Terra`
- Reasoning effort: `high`
- Goal: correct only the pinned TinyLLaVA `chat()` integration defects found during independent review, add exact ABI regression coverage, and leave the repository ready for immediate Apple one-image validation.
- Baseline branch: `origin/codex/vea-p2-t001a-r2-20260809-7f3c`
- Baseline SHA: `07e13ce8e344a8a15ba52244d8f549bfbac1bcf1`
- Primary task branch: `codex/vea-p2-t001a-r3-20260809-abi`
- Authorized fallback branch: `codex/vea-p2-t001a-r3-20260809-abi-b`
- Git delivery: `COMMIT_AND_PUSH`
- Commit condition: `PASS_ONLY`
- Commit message: `fix(vlm-compat): VEA-P2-T001A-R3 match pinned TinyLLaVA chat ABI`
- Push target: `origin HEAD`

## Preconditions and repository selection

The Google Drive repository is launch-only.

```text
G:\내 드라이브\2026NAIS
```

Read this Active Task from that path, but do not run Git commands there and do not repair or clean its Git metadata.

Preferred implementation repository:

```text
C:\Users\Administrator\AppData\Local\OpenAI\Codex\isolated\NAIS\vea-p2-t001a-r2-4f6524cff704403d99fb5eb2f6062cf2
```

Use the preferred repository only if all of the following hold:

```text
it exists
origin is Anzenoh/NAIS
HEAD is 07e13ce8e344a8a15ba52244d8f549bfbac1bcf1
working tree is clean
index is clean
```

If any preferred-repository condition fails, create a fresh independent clone under:

```text
%LOCALAPPDATA%\OpenAI\Codex\isolated\NAIS\
```

Use a fresh GUID directory and clone only:

```text
codex/vea-p2-t001a-r2-20260809-7f3c
```

Judge clone success by the process exit code and final repository checks. Normal Git progress written to stderr is not a failure.

In the selected repository:

1. fetch the baseline branch
2. verify that the remote baseline resolves exactly to:

   ```text
   07e13ce8e344a8a15ba52244d8f549bfbac1bcf1
   ```

3. verify the working tree is clean
4. use the primary R3 branch if absent locally and remotely
5. use the authorized fallback branch only if the primary branch already exists
6. stop only if both authorized R3 branches already exist

Create the selected R3 branch directly from the exact baseline SHA.

## Read

Read only:

1. `AGENTS.md`
2. `tasks/VEA-P2-T001A-R2-isolated-tinyllava-native-replay.md`
3. `configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json`
4. `src/vision_encoder_alignment/vlm_compat/tinyllava_native_replay.py`
5. `tests/unit/test_p2_tinyllava_native_replay.py`
6. `reports/runbooks/VEA-P2-T001A-R2-apple-native-replay.md`
7. `reports/runs/VEA-P2-T001A-R2-isolated-tinyllava-native-replay.md`
8. this Task copied to:
   `tasks/VEA-P2-T001A-R3-tinyllava-chat-abi-correction.md`

Do not inspect unrelated tasks, artifacts, packages, datasets, or model caches.

## Allowed changes

- `src/vision_encoder_alignment/vlm_compat/tinyllava_native_replay.py`
- `tests/unit/test_p2_tinyllava_native_replay.py`
- `reports/runbooks/VEA-P2-T001A-R2-apple-native-replay.md`
- `reports/runs/VEA-P2-T001A-R3-tinyllava-chat-abi-correction.md`
- `tasks/VEA-P2-T001A-R3-tinyllava-chat-abi-correction.md`

## Commit paths

Commit only the five allowed paths above that are actually changed or created.

## Protected scope

Do not modify:

- `configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json`
- `configs/runtime_profiles/p2_tinyllava_overlay.requirements.txt`
- `scripts/run_tinyllava_native_replay.py`
- `src/vision_encoder_alignment/vlm_compat/__init__.py`
- research contracts or decision logs
- feature serialization logic
- connector comparison thresholds
- replay injection and restoration logic, except only where required to pass the exact ABI tests
- any Phase 0 or Phase 1 implementation

## Non-goals

Do not:

- download or load TinyLLaVA
- install packages
- use CUDA or Apple
- modify the remote Hugging Face source
- implement DINOv3 or SigLIP2 injection
- implement calibration, CORAL, Ridge, Procrustes, Linear, or MLP
- change prompts, model identity, revision, feature dimensions, thresholds, or result schema
- refactor unrelated code
- create a new runbook by copying the existing runbook
- create ZIP or sealed evidence

## Exact pinned ABI contract

Treat the pinned model's interface as exactly:

```python
def chat(
    self,
    prompt,
    tokenizer=None,
    image=None,
    max_new_tokens=512,
    num_beams=1,
    top_p=None,
    temperature=0,
):
    ...
```

The pinned model internally derives:

```python
do_sample = temperature > 0
```

The pinned model expects `image` to be a local-path or URL string and opens it internally.

The pinned model uses its vision tower's own image processor.

## Required source correction

### 1. Replace the generic chat adapter

Replace the existing generic `_chat_kwargs()` behavior with a pinned-model-specific adapter.

It must verify that `model.chat` accepts these required parameters:

```text
prompt
tokenizer
image
max_new_tokens
num_beams
top_p
temperature
```

It must return exactly:

```python
{
    "prompt": config.prompt,
    "tokenizer": tokenizer,
    "image": str(image_path),
    "max_new_tokens": config.max_new_tokens,
    "num_beams": config.num_beams,
    "top_p": None,
    "temperature": 0,
}
```

It must not pass:

```text
do_sample
generation_config
image_processor
PIL.Image
```

If a required parameter is absent, raise `RuntimeABIError`.

### 2. Remove the external processor path

From `run_native_replay()`:

- remove `AutoProcessor`
- remove `AutoProcessor.from_pretrained(...)`
- remove local `PIL.Image` loading
- do not open the image before calling `model.chat`
- pass the same `str(image_file)` to both native and replay calls

Do not modify the image itself.

### 3. Match the official tokenizer initialization

Load the model first.

Then load the tokenizer using:

```text
model ID and pinned revision
cache root
local_files_only=True
use_fast=False
model.config.tokenizer_model_max_length
model.config.tokenizer_padding_side
```

Fail with `RuntimeABIError` if either required tokenizer config field is absent.

Do not add a new tokenizer, processor, or model dependency.

### 4. Preserve all valid R2 behavior

Do not alter:

- model freezing
- `torch.inference_mode()`
- vision and connector hooks
- safetensors save and reload
- exact patch serialization comparison
- temporary `encode_images` replacement
- vision-tower bypass accounting
- connector-output comparison
- token-ID comparison
- decoded-text comparison
- `finally` restoration
- result and blocker schemas

## Required regression tests

Keep all existing tests and add focused tests using a fake model whose `chat()` has the exact pinned signature.

The new tests must verify:

1. `_chat_kwargs()` succeeds with the pinned signature
2. returned `image` is `str`
3. returned `image` equals `str(image_path)`
4. `temperature == 0`
5. `top_p is None`
6. `num_beams == 1`
7. `max_new_tokens == 32`
8. the key set contains no `do_sample`
9. the key set contains no `generation_config`
10. the key set contains no `image_processor`
11. a fake `chat()` missing `temperature` is rejected with `RuntimeABIError`

The complete test file must remain runnable with the standard library only. No Torch, Transformers, Pillow, Safetensors, model, network, or CUDA may be required.

## Apple runbook correction

Modify the existing runbook only where necessary for R3 execution.

Required edits:

- heading identifies `VEA-P2-T001A-R3`
- `TASK_BRANCH` is the actual selected R3 branch
- repository and output labels use `t001a-r3`

Do not alter the runtime, download, offline execution, result validation, or submission procedure.

## Run Report

Create:

```text
reports/runs/VEA-P2-T001A-R3-tinyllava-chat-abi-correction.md
```

Record only:

- baseline and selected branch
- exact five-path scope
- corrected ABI behavior
- test count and commands
- no real model or GPU execution
- remaining Apple validation requirement

## Codex validation

Use CPython 3.11 or 3.12 already installed on the system.

Run:

```powershell
py -3.12 -m compileall `
  src/vision_encoder_alignment/vlm_compat `
  scripts/run_tinyllava_native_replay.py

py -3.12 tests/unit/test_p2_tinyllava_native_replay.py

py -3.12 scripts/run_tinyllava_native_replay.py `
  --validate-config-only `
  --config configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json

py -3.12 scripts/run_tinyllava_native_replay.py --help

git diff --check
```

If `py -3.12` is unavailable, use an already installed supported 3.11 interpreter. Do not install Python or packages.

Mandatory result:

```text
all previous 15 tests pass
all new ABI tests pass
config-only validation passes
CLI help passes
git diff --check passes
unexpected changed paths = 0
```

Do not claim real TinyLLaVA or CUDA validation.

## User validation

After independent review of the pushed R3 commit, execute the corrected existing Apple runbook immediately.

No additional implementation Task is required before that one-image Apple execution.

## Git delivery

Before staging, verify:

```text
HEAD descends directly from baseline 07e13ce8...
current branch is the selected R3 branch
unexpected changed paths = 0
```

Stage only the exact allowed changed paths.

Commit with exactly:

```text
fix(vlm-compat): VEA-P2-T001A-R3 match pinned TinyLLaVA chat ABI
```

Push:

```bash
git push -u origin HEAD
```

If push alone fails because of authentication or network:

- preserve the local commit
- do not repeat implementation or tests
- return `PARTIAL_GIT_DELIVERY`
- provide the exact push command

## Done when

- only the pinned `chat()` ABI integration has been corrected
- external `AutoProcessor` and local PIL image path are removed
- tokenizer loading follows the pinned model configuration
- all old and new offline tests pass
- the existing runbook points to the R3 branch
- the R3 commit is pushed
- no actual model, package, CUDA, or Apple execution has occurred

## Stop conditions

Stop only if:

- the exact R2 baseline commit cannot be obtained
- both authorized R3 branches already exist
- the selected isolated repository is dirty before the Task
- one of the five allowed paths is insufficient
- mandatory offline tests cannot pass without an additional dependency
- real model loading or GPU execution is required to complete the code correction

Do not stop because of the Google Drive repository's lock, dirty state, HEAD, R1 worktree, or R1 branch.

## Final response

Return no more than 20 lines:

1. `Status`
2. model and reasoning
3. repository path
4. baseline SHA
5. selected task branch
6. changed paths
7. old and new test count
8. compile result
9. config-only result
10. CLI-help result
11. `git diff --check`
12. commit message
13. commit SHA
14. push result
15. remote branch
16. deviation or blocker
17. next action: independent review followed immediately by Apple runbook execution

For `PASS + push`, state:

```text
Submit Task ID, baseline SHA, commit SHA, and remote branch to ChatGPT.
```

Do not merge, execute Apple, or begin cross-encoder injection.

# 이번 작업 정리

## 작업 목적

R2의 replay 구조는 유지하고, 실제 TinyLLaVA 실행을 막는 `chat()` 호출 방식만 정확한 pinned ABI에 맞춘다.

## 핵심 수행 내용

- PIL 객체를 이미지 경로 문자열로 교체
- `do_sample` 전달을 제거하고 `temperature=0` 사용
- 불필요한 `AutoProcessor` 제거
- 공식 설정에 맞는 tokenizer 초기화
- exact ABI regression test 추가
- 기존 Apple runbook을 R3 branch로 연결

## 이번 작업에서 하지 않는 것

새 모델, adapter, calibration, benchmark, GPU 실행, 구조 refactor를 추가하지 않는다.

## 완료 기준

최소 다섯 경로만 변경하고 모든 offline 검증과 push를 완료하여, 독립 검수 직후 Apple one-image 실행으로 넘어갈 수 있어야 한다.