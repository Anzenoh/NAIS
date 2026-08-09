# VEA-P2-T001B — DINOv3 raw block-11 injection one-image smoke

- Mode: `implementation`
- Task class: `STANDARD`
- Evidence level: `COMPACT`
- Package required: `false`
- Recommended Codex model: `GPT-5.6 Terra`
- Reasoning effort: `high`
- Goal: implement a one-image smoke that sequentially extracts frozen DINOv3 block-11 patch tokens and injects them without calibration into the frozen TinyLLaVA CLIP connector.
- Baseline: `66ec9ed2f7f4719b36e803179516b25292a0a3c0`
- Git delivery: `COMMIT_AND_PUSH`
- Task branch: `codex/vea-p2-t001b-dinov3-raw-smoke-20260809`
- Commit condition: `PASS_ONLY`
- Commit message: `feat(vlm-compat): VEA-P2-T001B add raw DINOv3 injection smoke`
- Push target: `origin HEAD`

## Preconditions

- Read this Task from the launch repository, but do not run Git commands in the Google Drive repository.
- Prefer the existing isolated repository only if it is clean at exact baseline `66ec9ed2f7f4719b36e803179516b25292a0a3c0`.
- Otherwise clone `origin/codex/vea-p2-t001a-r4-fp64-cosine` into a fresh GUID path outside Google Drive.
- The new task branch must not already exist locally or remotely.
- Do not download models, install packages, use CUDA, or run real inference during Codex validation.

## Read

1. `AGENTS.md`
2. `src/vision_encoder_alignment/vlm_compat/tinyllava_native_replay.py`
3. `scripts/run_tinyllava_native_replay.py`
4. `src/vision_encoder_alignment/extractors/base.py`
5. `src/vision_encoder_alignment/extractors/dinov3.py`
6. `src/vision_encoder_alignment/dinov3_extraction_boundary.py`
7. `scripts/extract_features.py`
   - inspect only `_canonical_batch` and `_extractor`
8. `configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json`
9. `configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json`
10. `tests/unit/test_p2_tinyllava_native_replay.py`
11. this Task copied to `tasks/VEA-P2-T001B-dinov3-raw-injection-smoke.md`

## Allowed changes

- `src/vision_encoder_alignment/vlm_compat/dinov3_raw_injection.py`
- `scripts/run_dinov3_raw_injection.py`
- `tests/unit/test_p2_dinov3_raw_injection.py`
- `reports/runbooks/VEA-P2-T001B-apple-dinov3-raw-injection.md`
- `reports/runs/VEA-P2-T001B-dinov3-raw-injection-smoke.md`
- `tasks/VEA-P2-T001B-dinov3-raw-injection-smoke.md`

## Protected scope

Do not modify:

- `tinyllava_native_replay.py`
- existing DINOv3 extractor, worker, profiles, or runtime contracts
- existing CLI or tests
- model, prompt, connector, threshold, result, or Phase 0/1 contracts
- dependency files

## Required experiment identity

### TinyLLaVA anchor

Use the existing pinned anchor config unchanged.

```text
model = jiajunlong/TinyLLaVA-OpenELM-450M-CLIP-0.55B
revision = 0bff7b92d5b9ef4520c61b1f7872d7d1e13537a6
native feature = CLIP hidden_states[-2], patch-only
expected native patch shape = [1,196,768]
```

### DINOv3 source

Use only the existing verified Apple DINOv3 extractor.

```text
model = facebook/dinov3-vitb16-pretrain-lvd1689m
revision = 5931719e67bbdb9737e363e781fb0c67687896bc
tap = block_11
source = hidden_states[11]
special tokens removed = CLS 1 + registers 4
expected raw patch shape = [1,196,768]
expected raw patch dtype = float32
execution selector = dinov3_apple_cuda_fp32_v1
```

Do not use block 5, block 12, or final norm.

## Requirements

### 1. Import-safe module

Create:

```text
src/vision_encoder_alignment/vlm_compat/dinov3_raw_injection.py
```

At import time it must not:

- import Torch, Transformers, Safetensors, or DINO runtime packages
- load models
- allocate CUDA
- access network or cache
- modify files

Reuse the validated R4 replay primitives from `tinyllava_native_replay.py` without modifying that file.

### 2. Exact canonical geometry

Decode one RGB image and apply the exact P1 geometry:

```text
resize shorter side to 224
Pillow bicubic
center crop 224×224
uint8 RGB NHWC
```

The implementation must match the algorithm in `scripts/extract_features.py::_canonical_batch`.

Do not use the DINO image processor, resize inside the DINO worker, or introduce a second geometry.

### 3. Sequential DINO extraction

Before loading TinyLLaVA:

1. construct one `CanonicalImageBatch`
2. instantiate the existing verified Apple DINOv3 extractor
3. request only `block_11`
4. wait for its isolated worker to exit
5. obtain exactly one `[1,196,768]` float32 patch tensor
6. verify shape, dtype, finiteness, model ID, revision, tap, and token-removal policy

DINOv3 and TinyLLaVA must never be resident on GPU simultaneously.

### 4. Raw means no alignment

Before the CLIP connector, do not apply:

- centering
- feature normalization
- mean or variance matching
- learned or closed-form mapping
- Ridge, Procrustes, CORAL, Linear, or MLP
- patch permutation or pooling

Only these operations are allowed:

```text
NumPy → Torch conversion
device movement
cast to the frozen connector parameter dtype
```

### 5. Native and raw paths

Load the same frozen TinyLLaVA path validated in R4.

For the same image and prompt, run exactly:

```text
native CLIP path once
raw DINOv3 injection path once
```

The raw path must:

- temporarily replace only the loaded model instance's `encode_images`
- pass DINOv3 block-11 patch tokens through the original frozen connector
- bypass the CLIP vision tower
- restore `encode_images` in `finally`

Assertions:

```text
native vision calls >= 1
raw DINOv3 vision calls = 0
raw connector calls >= 1
encode_images restored = true
raw connector output shape = [1,196,1536]
all recorded tensors finite
```

Do not require native and DINOv3 outputs, token IDs, or text to match.

### 6. Diagnostics, not pass gates

Record without applying acceptance thresholds:

- native CLIP patch mean, std, L2 norm
- DINOv3 raw patch mean, std, L2 norm
- native-vs-DINO patch cosine in float64
- patch norm ratio
- native connector output statistics
- DINOv3 connector output statistics
- connector-output cosine in float64
- native answer and token IDs
- DINOv3 raw answer and token IDs
- answer equality boolean

A low cosine or different answer is a scientific result, not a software failure.

### 7. Artifacts

Write only to a fresh output directory:

```text
result.json
input_patch_features.safetensors
```

The Safetensors file must contain:

```text
native_clip_block_11
dinov3_block_11
```

Successful result:

```text
schema_id = vea.p2.t001b.dinov3_raw_injection.v1
status = PASS_DINOV3_RAW_INJECTION_SMOKE
```

Include exact repository commit, model IDs, revisions, tap identities, image SHA-256, geometry identity, shapes, dtypes, call counts, diagnostics, outputs, artifact sizes, and SHA-256 values.

On infrastructure failure, write a small structured `result.json` containing stage, classification, error type, and message.

### 8. CLI

Create:

```text
scripts/run_dinov3_raw_injection.py
```

Required arguments:

```text
--tinyllava-config
--image-path
--cache-root
--repository-root
--output-dir
--device
--dtype
```

The CLI must:

- accept exactly one image
- require a fresh output directory
- require CUDA for the Apple run
- use no network
- return nonzero for infrastructure or contract failure
- print one compact status line

### 9. Apple runbook

Create one safe Apple runbook that:

- clones the reviewed task commit into a fresh repository
- checks both main and DINO isolated Python paths
- checks existing SentencePiece 0.2.0 and protobuf 4.25.3 overlay
- enables existing offline HF caches
- downloads or installs nothing
- waits until no compute PID exists and at least 24,000 MiB is free for two checks
- runs exactly one image
- uses an external worker process so the tmux parent shell cannot exit
- prints `result.json` status, native answer, DINOv3 raw answer, and artifact hashes

Do not include calibration or a multi-image benchmark.

## Codex validation

Run only offline Tier 0–1:

```powershell
py -3.12 -m py_compile `
  src/vision_encoder_alignment/vlm_compat/dinov3_raw_injection.py `
  scripts/run_dinov3_raw_injection.py

py -3.12 tests/unit/test_p2_dinov3_raw_injection.py

py -3.12 scripts/run_dinov3_raw_injection.py --help

git diff --check
```

Tests must use fakes and cover:

- exact canonical geometry
- exact DINO model, revision, profile, and `block_11`
- `[1,196,768]` float32 finite contract
- only dtype/device conversion before connector
- CLIP vision bypass and method restoration
- differing native/raw answers still produce successful result
- result schema
- runtime imports absent at module import time
- CLI help import safety

Do not run real models, CUDA, network, dataset, or Apple validation.

## Git delivery

- Create the task branch from exact baseline.
- Stage only the six allowed paths.
- Commit only after all required checks pass.
- Push using `git push -u origin HEAD`.
- Do not merge, amend, rebase, force-push, or create a package.

## Done when

- DINOv3 block-11 raw injection is implemented without modifying validated R4 code.
- No alignment or calibration exists in the path.
- All offline checks pass.
- The task branch is pushed.
- The Apple one-image execution has not been performed by Codex.

## Final response

Within 20 lines return:

- Status
- baseline and branch
- changed paths
- test count and commands
- commit message and SHA
- push result and remote branch
- `git diff --check`
- deviation or blocker
- next action: independent review, then one Apple raw-injection smoke

# 이번 작업 정리

## 작업 목적

DINOv3 block-11 patch를 아무 보정 없이 CLIP용 connector에 넣을 수 있는지 한 이미지에서 확인한다.

## 핵심 수행 내용

- 기존 verified DINOv3 extractor 재사용
- DINOv3와 TinyLLaVA를 순차 실행
- native CLIP과 raw DINOv3 출력 기록
- 학습·보정 없는 직접 주입

## 이번 작업에서 하지 않는 것

Calibration, Ridge, CORAL, Linear, MLP, 다중 이미지 평가를 수행하지 않는다.

## 완료 기준

한 이미지 raw-injection 실행 도구와 안전한 Apple runbook이 offline test를 통과해 remote branch에 push된다.

## 다음 단계

독립 검수 후 Apple에서 one-image raw-injection smoke를 정확히 한 번 실행한다.

## 한눈에 보는 작업 요약

먼저 DINOv3 feature를 그대로 CLIP connector에 꽂아 기술적으로 실행되는지 확인한다. 성능 비교와 보정은 이 smoke가 끝난 뒤에만 진행한다.