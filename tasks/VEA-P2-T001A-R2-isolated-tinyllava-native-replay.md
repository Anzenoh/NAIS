# VEA-P2-T001A-R2 — Isolated clone and frozen TinyLLaVA native replay harness

- Mode: `implementation`
- Task class: `STANDARD`
- Evidence level: `COMPACT`
- Package required: `false`
- Recommended Codex model: `GPT-5.6 Sol`
- Reasoning effort: `high`
- Goal: create an independent clean clone from the exact accepted baseline, then implement and offline-test a frozen TinyLLaVA native patch-feature replay harness without using or repairing the current Google Drive repository's Git metadata.
- Repository remote: `https://github.com/Anzenoh/NAIS.git`
- Accepted baseline branch: `codex/vea-p1-t003b-a-r3r2`
- Accepted baseline SHA: `8fbc4c16c9a8a3802d1fdc1e8fb7b64738482980`
- Git delivery: `COMMIT_AND_PUSH`
- Primary task branch: `codex/vea-p2-t001a-r2-20260809-7f3c`
- Authorized collision fallback: `codex/vea-p2-t001a-r2-20260809-7f3c-b`
- Commit condition: `PASS_ONLY`
- Commit message: `feat(vlm-compat): VEA-P2-T001A-R2 add isolated TinyLLaVA replay harness`
- Push target: `origin HEAD`

## 1. Critical execution rule

The launch repository is:

```text
G:\내 드라이브\2026NAIS
```

It is not the implementation repository for this Task.

It may contain:

- `HEAD=464ebf3fdeaf53a16d04838fc67aed725fe5d08c`
- staged, unstaged, or untracked files
- `.git\packed-refs.lock`
- incomplete R1 worktree metadata
- incomplete local branch `codex/vea-p2-t001a-r1`
- incomplete directory `G:\내 드라이브\2026NAIS-p2-t001a-r1`

None of those conditions is a blocker for this R2 Task.

In the launch repository, Codex must not run:

```text
git status
git fetch
git branch
git switch
git checkout
git worktree
git prune
git reset
git restore
git clean
git stash
git add
git commit
git push
```

Codex must not:

- delete `.git\packed-refs.lock`
- repair source Git metadata
- remove the incomplete R1 worktree
- remove the R1 branch
- modify any source repository file other than reading `.codex\ACTIVE_TASK.md`

The launch repository is used only to read this approved task contract.

All Git operations, implementation, validation, commit, and push occur in a new independent clone.

## 2. Isolated clone bootstrap

### 2.1 Source task read

Read the complete file:

```text
G:\내 드라이브\2026NAIS\.codex\ACTIVE_TASK.md
```

Retain its exact content for creation of the task-history file in the target clone.

Do not invoke Git to locate the source repository.

### 2.2 Select an isolated parent

Use the first writable path in this order:

```text
%LOCALAPPDATA%\OpenAI\Codex\isolated\NAIS
%TEMP%\OpenAI-Codex-isolated\NAIS
```

The selected parent must:

- exist or be safely creatable
- be writable
- not be inside `G:\내 드라이브`
- not be inside the launch repository
- not be the incomplete R1 directory
- not be a symlink or junction to the launch repository

Create a fresh target path using a GUID:

```text
<isolated-parent>\vea-p2-t001a-r2-<32-hex-guid>
```

Never reuse an existing target path.

### 2.3 Clone with bounded retry

Run an independent HTTPS clone:

```powershell
git -c core.autocrlf=false clone `
  --no-tags `
  --single-branch `
  --branch codex/vea-p1-t003b-a-r3r2 `
  https://github.com/Anzenoh/NAIS.git `
  <fresh-target-path>
```

If the first clone attempt fails:

1. preserve the failed target path
2. record the command, exit code, and stderr
3. create a different GUID target path
4. retry once

Do not repair or reuse the failed clone.

If both attempts fail, return:

```text
BLOCKED_EXTERNAL_GIT_CLONE
```

This is an external delivery blocker, not an implementation defect.

### 2.4 Baseline validation

In the successful target clone, verify:

```powershell
git rev-parse HEAD
git remote get-url origin
git status --porcelain=v1 --untracked-files=all
git diff --check
```

Required:

```text
HEAD = 8fbc4c16c9a8a3802d1fdc1e8fb7b64738482980
origin = https://github.com/Anzenoh/NAIS.git
status output = empty
git diff --check = PASS
```

Set target-local Git configuration:

```powershell
git config core.autocrlf false
git config core.safecrlf true
```

Do not inspect ancestry against the launch repository.

### 2.5 Task branch selection

Check the primary branch remotely:

```powershell
git ls-remote --exit-code --heads origin `
  refs/heads/codex/vea-p2-t001a-r2-20260809-7f3c
```

Interpretation:

- exit `2` with no matching ref: primary is available
- exit `0`: use the authorized fallback branch
- any other transport failure: retry once, then report external Git connectivity failure

If primary exists, check:

```text
codex/vea-p2-t001a-r2-20260809-7f3c-b
```

If both already exist, stop with:

```text
BLOCKED_REMOTE_BRANCH_COLLISION
```

Create only the selected branch:

```powershell
git switch -c <selected-task-branch>
```

Record the selected branch in the Run Report and final response.

## 3. Instruction loading in the target clone

As the first target repository change, create:

```text
tasks/VEA-P2-T001A-R2-isolated-tinyllava-native-replay.md
```

Its content must preserve the full approved Task contract from the source
`.codex\ACTIVE_TASK.md`.

Then read only:

1. `AGENTS.md`
2. `tasks/VEA-P2-T001A-R2-isolated-tinyllava-native-replay.md`
3. `docs/INDEX.md`
4. `docs/PROJECT_CONTEXT.md`
5. `docs/PROJECT_STATUS.md`
6. `docs/RESEARCH_CONTRACT.md`
7. `docs/DECISION_LOG.md`
8. `docs/VALIDATION_PLAN.md`
9. `docs/REPRODUCIBILITY.md`
10. `docs/contracts/FEATURE_ARTIFACT_SCHEMA.md`
11. `docs/templates/RUN_REPORT_TEMPLATE.md`
12. `pyproject.toml`
13. `src/vision_encoder_alignment/feature_artifacts/io.py`
14. `src/vision_encoder_alignment/extractors/clip.py`
15. nearest directly relevant unit-test conventions only

Do not scan historical packages, large reports, artifacts, model caches, datasets, or unrelated tasks.

## 4. Allowed changes

- `docs/RESEARCH_CONTRACT.md`
- `docs/DECISION_LOG.md`
- `docs/contracts/VLM_COMPATIBILITY_PILOT_V0.md`
- `configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json`
- `configs/runtime_profiles/p2_tinyllava_overlay.requirements.txt`
- `src/vision_encoder_alignment/vlm_compat/__init__.py`
- `src/vision_encoder_alignment/vlm_compat/tinyllava_native_replay.py`
- `scripts/run_tinyllava_native_replay.py`
- `tests/unit/test_p2_tinyllava_native_replay.py`
- `reports/runbooks/VEA-P2-T001A-R2-apple-native-replay.md`
- `reports/runs/VEA-P2-T001A-R2-isolated-tinyllava-native-replay.md`
- `tasks/VEA-P2-T001A-R2-isolated-tinyllava-native-replay.md`

## 5. Commit paths

Commit only changed files from the exact allowed list.

Do not add or modify:

- `.codex/`
- `AGENTS.md`
- `pyproject.toml`
- `configs/models.yaml`
- existing extractor implementation
- existing test files
- datasets, checkpoints, caches, generated feature tensors
- machine-specific configuration files

## 6. Non-goals

Do not implement or execute:

- DINOv3 injection
- SigLIP2 injection
- raw cross-encoder swap
- diagonal calibration
- CORAL
- Ridge
- Procrustes
- Linear adapter fitting
- MLP fitting
- projector training
- LLM training or fine-tuning
- synthetic numeric benchmark
- VQA accuracy evaluation
- Apple GPU execution
- Hugging Face model download
- package installation
- source repository cleanup
- R1 worktree or branch recovery

## 7. Phase 2 pilot contract

Append a minimal section to `docs/RESEARCH_CONTRACT.md` and record one decision in
`docs/DECISION_LOG.md`.

Create:

```text
docs/contracts/VLM_COMPATIBILITY_PILOT_V0.md
```

The contract must explicitly define:

- experiment ID: `vea.p2.vlm_compatibility_pilot.v0`
- frozen vision tower
- frozen multimodal connector
- frozen LLM
- no optimizer
- no backward pass
- no parameter update
- explicit pre-connector patch-feature capture
- explicit pre-connector patch-feature injection
- raw patch tokens and connector outputs remain distinct feature classes
- this experiment does not revise Phase 0 or Phase 1 metric conclusions
- a software PASS does not imply encoder interchangeability

Do not broadly rewrite the existing Research Contract.

## 8. Pinned anchor configuration

Create:

```text
configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json
```

Required values:

```text
schema_id = vea.p2.vlm_anchor.v1
model_id = jiajunlong/TinyLLaVA-OpenELM-450M-CLIP-0.55B
revision = 0bff7b92d5b9ef4520c61b1f7872d7d1e13537a6
trust_remote_code = true
local_files_only = true
expected_vision_model = openai/clip-vit-base-patch16
expected_vision_feature_layer = -2
expected_vision_feature_select_strategy = patch
expected_connector_type = mlp2x_gelu
expected_image_size = 224
expected_patch_size = 16
expected_patch_count = 196
expected_vision_hidden_size = 768
expected_connector_output_size = 1536
expected_vision_tower_frozen = true
prompt = What is shown in this image? Answer briefly.
max_new_tokens = 32
do_sample = false
num_beams = 1
```

Reject unknown fields and runtime mismatches.

Do not substitute another checkpoint, revision, vision tower, connector, tokenizer, or prompt.

## 9. Runtime overlay declaration

Create:

```text
configs/runtime_profiles/p2_tinyllava_overlay.requirements.txt
```

Exact content:

```text
sentencepiece==0.2.0
```

This is an Apple runtime overlay declaration only.

Do not modify `pyproject.toml`.

Do not install it during Codex validation.

## 10. Import-safe implementation

Implement:

```text
src/vision_encoder_alignment/vlm_compat/tinyllava_native_replay.py
```

Requirements:

- no model load at import time
- no Torch import at import time
- no Transformers import at import time
- no Safetensors import at import time
- no GPU allocation at import time
- no network access at import time
- no cache mutation at import time
- typed dataclasses or equivalent explicit structures
- explicit exception classes
- deterministic JSON output
- runtime dependencies imported only inside runtime functions

The real runtime loader must:

- load exactly the configured model ID and revision
- use `local_files_only=True`
- use the configured `trust_remote_code=True`
- avoid `device_map`
- avoid Accelerate requirements
- avoid FlashAttention
- avoid quantization
- call `eval()`
- set every parameter to `requires_grad_(False)`
- assert trainable parameter count is zero
- create no optimizer
- perform no backward operation

If required TinyLLaVA interfaces are absent, classify:

```text
BLOCKED_TINYLLAVA_RUNTIME_ABI
```

## 11. Native capture path

Use the pinned model's own `chat` path for the native generation.

During native generation:

1. register a forward hook on the loaded `vision_tower`
2. capture its selected output immediately before the connector
3. register a forward hook on the original connector
4. temporarily wrap `model.generate` only to record returned token IDs
5. call the original `model.chat`
6. restore every hook and wrapped method in `finally`

Assert:

```text
patch shape = [1,196,768]
connector output shape = [1,196,1536]
all values finite
vision tower call count >= 1
connector call count >= 1
```

Record:

- actual resolved runtime configuration
- tensor dtype
- device
- hidden-state selection metadata exposed by the model
- patch tensor hash
- connector tensor hash
- generated token IDs
- decoded output text

Do not label the runtime feature as a project `block_XX` tap unless the remote model
explicitly exposes and verifies that mapping.

## 12. Safetensors serialization

Save the native pre-connector patch tensor as:

```text
captured_patch_features.safetensors
```

Then reload it before replay.

Assert:

```text
shape exact match
dtype exact match
maximum absolute difference = 0
all values finite
```

Record file size and SHA-256.

Refuse to overwrite an existing output directory or artifact.

## 13. Replay injection path

Implement a context manager that temporarily replaces only the loaded model instance's:

```text
encode_images
```

The temporary replay method must:

- ignore the placeholder image tensor
- use the reloaded `[1,196,768]` patch tensor
- move it to the connector's actual device and dtype
- call the original frozen connector
- bypass the vision tower
- edit no downloaded source file
- use no direct `inputs_embeds` injection
- restore the original bound method in `finally`

During replay:

- use the same image path
- use the same prompt
- use the same tokenizer
- use the same generation arguments
- capture connector output
- capture generated token IDs
- capture decoded text

Required comparison:

```text
replay vision tower call count = 0
replay connector call count >= 1
original encode_images restored = true
connector max_abs_diff <= 1e-5
connector cosine >= 0.999999
generated token IDs exact match
decoded text exact match
```

Successful runtime classification:

```text
PASS_NATIVE_REPLAY
```

## 14. CLI

Implement:

```text
scripts/run_tinyllava_native_replay.py
```

Required modes:

```text
--validate-config-only
--run
```

Required run arguments:

```text
--config
--image-path
--cache-root
--output-dir
--device
--dtype
```

Requirements:

- exactly one image per T001A run
- image path must exist and be a regular file
- output directory must not exist
- `--help` must require no Torch, Transformers, Safetensors, CUDA, or model import
- `--validate-config-only` must use only the standard library
- failed runtime assertions return nonzero
- when possible, runtime failure still writes a structured blocker result
- stdout contains only a compact status summary

## 15. Result artifacts

Write under the fresh output directory:

```text
result.json
feature_manifest.json
captured_patch_features.safetensors
```

`result.json` must include:

```text
schema_id = vea.p2.t001a.native_replay.v1
status
repository commit
model ID
model revision
runtime package versions
CUDA identity
checkpoint snapshot identity
image path basename
image SHA-256
runtime-resolved model configuration
trainable parameter count
patch shape
connector shape
native call counts
replay call counts
serialization comparison
connector comparison
native token IDs
replay token IDs
native decoded text
replay decoded text
restoration status
artifact relative paths
artifact sizes
artifact SHA-256 values
```

Do not record:

- tokens or credentials
- model weights
- private cache directory listings
- full machine-specific dataset paths in publication-facing summaries

## 16. Apple runbook

Create:

```text
reports/runbooks/VEA-P2-T001A-R2-apple-native-replay.md
```

The runbook must use a fresh independent Apple clone, not `git worktree`.

Known Apple resources:

```text
target host = apple
GPU target = lab_apple_rtx5090
main Python = /home/chltjsdh/vea/runtime/p1_apple/main/bin/python3
dataset root = /home/chltjsdh/vea/datasets/coco2017/images/val2017
HF home = /home/chltjsdh/vea/checkpoints/huggingface
```

The runbook must separate:

1. clone the selected task branch into a fresh `/home/chltjsdh/vea/p2_repos/...` path
2. verify exact task commit and clean status
3. inspect Python, Torch, Transformers, CUDA, Safetensors, Pillow, and SentencePiece
4. if SentencePiece is absent, install only the pinned package into:

   ```text
   /home/chltjsdh/vea/runtime/p2_tinyllava_overlay/site-packages
   ```

   using:

   ```text
   pip install --no-deps --target <overlay> -r <overlay requirements>
   ```

5. do not modify the existing P1 main virtual environment
6. download only the exact TinyLLaVA revision in a separately labeled network stage
7. verify the existing audited CLIP processor snapshot is locally resolvable
8. enable Hugging Face and Transformers offline mode
9. deterministically select the lexicographically first COCO JPG using Python
10. run one-image native replay
11. validate `PASS_NATIVE_REPLAY`
12. print result path and artifact SHA-256 values

Do not include DINOv3 injection in this runbook.

## 17. Codex validation

Select a Python interpreter satisfying:

```text
>=3.11,<3.13
```

Try compatible installed interpreters without installing one.

Run only in the isolated target clone:

```powershell
python -m compileall `
  src/vision_encoder_alignment/vlm_compat `
  scripts/run_tinyllava_native_replay.py

python tests/unit/test_p2_tinyllava_native_replay.py

python scripts/run_tinyllava_native_replay.py `
  --validate-config-only `
  --config configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json

python scripts/run_tinyllava_native_replay.py --help

git diff --check
```

The test file must use standard-library `unittest`.

Tests must use fakes or mocks and must not require:

- pytest
- Torch
- Transformers
- Safetensors
- SentencePiece
- CUDA
- network
- checkpoints
- datasets

Test at least:

- strict config validation
- invalid model identity rejection
- patch-shape rejection
- connector-shape rejection
- finite-value rejection
- fake serialization exact round trip
- native call accounting
- replay bypasses vision tower
- original method restored after success
- original method restored after exception
- result-schema construction
- runtime dependencies absent at import time
- CLI help and config validation are import-safe

Do not claim real model or GPU validation.

## 18. Git delivery

Before staging:

- verify current repository is the isolated clone
- verify current branch is the selected R2 branch
- verify merge base and HEAD started at the exact accepted baseline
- verify no unexpected paths changed
- recheck that the remote selected branch still does not exist

Stage only exact allowed changed paths.

Verify:

```powershell
git diff --cached --name-only
git diff --cached --check
```

Commit with exactly:

```text
feat(vlm-compat): VEA-P2-T001A-R2 add isolated TinyLLaVA replay harness
```

Push:

```powershell
git push -u origin HEAD
```

If push alone fails because of authentication, network, or transient remote error:

- preserve the local commit
- do not repeat implementation or tests
- return `PARTIAL_GIT_DELIVERY`
- report the exact clone path
- report the local commit SHA
- report the exact push command

Do not create a ZIP, detached package, repository tree manifest, or full protected-tree hash scan.

## 19. Conditions that are explicitly not blockers

Do not stop because of:

- source `packed-refs.lock`
- source dirty state
- source HEAD `464ebf3...`
- source ancestry mismatch
- incomplete R1 worktree
- incomplete R1 local branch
- R1 target directory
- missing pytest
- missing Torch or Transformers in the Codex test interpreter
- missing SentencePiece in the Apple runtime during Codex implementation
- absent dataset manifest for this one-image infrastructure replay

## 20. Remaining true stop conditions

Stop only for:

- Git executable unavailable
- neither isolated parent is writable
- remote clone fails twice
- cloned accepted branch does not resolve to exact baseline SHA
- both authorized R2 remote branch names already exist
- no Python interpreter in the supported range
- source ACTIVE_TASK cannot be read
- a protected, unlisted repository path must be modified
- standard-library unit validation cannot pass after bounded correction
- real checkpoint, GPU, package installation, or non-Git network access is required during Codex validation

## 21. Final response

Return no more than 20 lines:

1. `Status`
2. model and reasoning
3. successful clone path
4. clone attempt count
5. accepted baseline SHA
6. selected task branch
7. changed paths
8. validation commands and results
9. commit message
10. commit SHA
11. push result
12. remote branch
13. staged path count
14. `git diff --check`
15. deviations
16. remaining risks
17. next action: independent commit review before Apple execution

For `PASS + push`, state:

```text
Submit Task ID, baseline SHA, commit SHA, and remote branch to ChatGPT.
```

Do not clean the source repository, run Apple validation, merge, or start the next Task.

# 이번 작업 정리

## 작업 목적

현재 Google Drive 저장소의 lock, dirty state, HEAD, R1 worktree 상태를 전혀 사용하지 않고, accepted baseline을 새로 clone하여 TinyLLaVA native feature replay 코드를 구현한다.

## 핵심 수행 내용

- Google Drive 밖에 독립 clone 생성
- frozen TinyLLaVA patch capture 구현
- Safetensors 저장·재로드
- connector 앞 replay injection 구현
- standard-library fake 기반 검증
- Apple RTX 5090 실행 runbook 작성
- 전용 branch commit과 push

## 이번 작업에서 하지 않는 것

source lock 삭제, R1 정리, DINOv3 주입, calibration, adapter 학습, 실제 GPU 실행을 하지 않는다.

## 완료 기준

독립 clone에서 모든 offline 검증이 통과하고, 허용 경로만 commit되어 remote task branch로 push된다.