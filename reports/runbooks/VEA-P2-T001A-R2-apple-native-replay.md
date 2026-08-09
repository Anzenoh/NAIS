# VEA-P2-T001A-R2 Apple native replay runbook

This runbook is the user-run Tier 2 validation for target `lab_apple_rtx5090` on
host `apple`. Run it only after independent review supplies the exact task
commit SHA. It clones a fresh repository and does not use `git worktree`.

## 1. Fresh task-commit clone

Log in to `apple`, then run:

```bash
set -euo pipefail
TASK_BRANCH='codex/vea-p2-t001a-r2-20260809-7f3c'
TASK_COMMIT_SHA='<REVIEWED_TASK_COMMIT_SHA>'
REPO="/home/chltjsdh/vea/p2_repos/vea-p2-t001a-r2-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$REPO"
git clone --no-tags --single-branch --branch "$TASK_BRANCH" \
  https://github.com/Anzenoh/NAIS.git "$REPO"
cd "$REPO"
test "$(git rev-parse HEAD)" = "$TASK_COMMIT_SHA"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
printf 'repository=%s\ncommit=%s\n' "$REPO" "$TASK_COMMIT_SHA"
```

## 2. Runtime and CUDA preflight

```bash
set -euo pipefail
PY='/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3'
OVERLAY='/home/chltjsdh/vea/runtime/p2_tinyllava_overlay/site-packages'
test -x "$PY"
"$PY" - <<'PY'
import importlib.metadata as metadata
import platform
import torch
print('python=' + platform.python_version())
for name in ('torch', 'transformers', 'safetensors', 'Pillow', 'sentencepiece'):
    try:
        print(name + '=' + metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name + '=NOT_INSTALLED')
print('cuda_runtime=' + str(torch.version.cuda))
print('cuda_available=' + str(torch.cuda.is_available()))
print('cuda_device_count=' + str(torch.cuda.device_count()))
if torch.cuda.is_available():
    print('cuda_device_0=' + torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
PY
```

If and only if SentencePiece is absent, install the single pinned overlay
without modifying the P1 environment:

```bash
set -euo pipefail
mkdir -p "$OVERLAY"
if ! PYTHONPATH="$OVERLAY" "$PY" -c 'import sentencepiece' >/dev/null 2>&1; then
  "$PY" -m pip install --no-deps --target "$OVERLAY" \
    -r "$REPO/configs/runtime_profiles/p2_tinyllava_overlay.requirements.txt"
fi
PYTHONPATH="$OVERLAY" "$PY" -c \
  'import sentencepiece; print("sentencepiece=" + sentencepiece.__version__)'
```

Do not install any other package, enable FlashAttention, or alter the main P1
virtual environment. Any remaining import or CUDA incompatibility is
`BLOCKED_TINYLLAVA_RUNTIME_ABI`.

## 3. One-time pinned TinyLLaVA download (network stage)

This is the only model download stage. Do not download another model or
revision.

```bash
set -euo pipefail
export HF_HOME='/home/chltjsdh/vea/checkpoints/huggingface'
export PYTHONPATH="$OVERLAY"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='jiajunlong/TinyLLaVA-OpenELM-450M-CLIP-0.55B',
    revision='0bff7b92d5b9ef4520c61b1f7872d7d1e13537a6',
    cache_dir='/home/chltjsdh/vea/checkpoints/huggingface/hub',
    local_files_only=False,
)
print('tinyllava_snapshot=' + path)
PY
```

## 4. Offline processor check and deterministic image selection

Verify that the already audited CLIP processor snapshot resolves locally. This
command is offline and must not acquire CLIP files.

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='openai/clip-vit-base-patch16',
    revision='57c216476eefef5ab752ec549e440a49ae4ae5f3',
    cache_dir='/home/chltjsdh/vea/checkpoints/huggingface/hub',
    local_files_only=True,
)
print('clip_processor_snapshot=' + path)
PY
DATASET_ROOT='/home/chltjsdh/vea/datasets/coco2017/images/val2017'
IMAGE_PATH="$($PY - "$DATASET_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
images = sorted(path for path in root.glob('*.jpg') if path.is_file())
if not images:
    raise SystemExit('no COCO JPG found')
print(images[0])
PY
)"
test -f "$IMAGE_PATH"
printf 'image_basename=%s\n' "$(basename "$IMAGE_PATH")"
```

## 5. Offline one-image native replay

```bash
set -euo pipefail
OUTPUT="/home/chltjsdh/vea/p2_outputs/vea-p2-t001a-r2-$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${OUTPUT}.log"
test ! -e "$OUTPUT"
export HF_HOME='/home/chltjsdh/vea/checkpoints/huggingface'
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$OVERLAY:$REPO/src"
"$PY" "$REPO/scripts/run_tinyllava_native_replay.py" \
  --run \
  --config "$REPO/configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json" \
  --image-path "$IMAGE_PATH" \
  --cache-root "$HF_HOME/hub" \
  --output-dir "$OUTPUT" \
  --device cuda:0 \
  --dtype float16 | tee "$LOG"
```

## 6. Result verification and submission

```bash
set -euo pipefail
"$PY" - "$OUTPUT" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
assert result['schema_id'] == 'vea.p2.t001a.native_replay.v1'
assert result['status'] == 'PASS_NATIVE_REPLAY'
assert result['replay_call_counts']['vision_tower'] == 0
assert result['restoration_status']['encode_images_restored'] is True
for name in ('result.json', 'feature_manifest.json', 'captured_patch_features.safetensors'):
    path = root / name
    print(f'{name} bytes={path.stat().st_size} sha256={sha256(path.read_bytes()).hexdigest()}')
print('result_path=' + str(root / 'result.json'))
PY
printf 'log_path=%s\n' "$LOG"
```

Submit `result.json`, `feature_manifest.json`, the three printed SHA-256 values,
and the log. Do not run DINOv3 or cross-encoder injection.
