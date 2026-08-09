# VEA-P2-T001B Apple DINOv3 raw-injection smoke

Run this once, only after independent review supplies the reviewed task commit.
It uses a fresh clone, existing offline caches and runtimes, and no downloads or
installation.

```bash
set -euo pipefail
TASK_BRANCH='codex/vea-p2-t001b-dinov3-raw-smoke-20260809'
TASK_COMMIT_SHA='<REVIEWED_TASK_COMMIT_SHA>'
REPO="/home/chltjsdh/vea/p2_repos/vea-p2-t001b-$(date -u +%Y%m%dT%H%M%SZ)"
MAIN_PY='/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3'
DINO_PY='/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated/bin/python3'
OVERLAY='/home/chltjsdh/vea/runtime/p2_tinyllava_overlay/site-packages'
HF_HOME='/home/chltjsdh/vea/checkpoints/huggingface'
test ! -e "$REPO"
test -x "$MAIN_PY" && test -x "$DINO_PY"
git clone --no-tags --single-branch --branch "$TASK_BRANCH" https://github.com/Anzenoh/NAIS.git "$REPO"
cd "$REPO"
test "$(git rev-parse HEAD)" = "$TASK_COMMIT_SHA"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
PYTHONPATH="$OVERLAY" "$MAIN_PY" - <<'PY'
import importlib.metadata as m
assert m.version('sentencepiece') == '0.2.0'
assert m.version('protobuf') == '4.25.3'
print('sentencepiece=0.2.0 protobuf=4.25.3')
PY
```

Enable the existing offline cache and choose one deterministic image:

```bash
set -euo pipefail
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
DATASET_ROOT='/home/chltjsdh/vea/datasets/coco2017/images/val2017'
IMAGE_PATH="$($MAIN_PY - "$DATASET_ROOT" <<'PY'
from pathlib import Path
import sys
images = sorted(path for path in Path(sys.argv[1]).glob('*.jpg') if path.is_file())
if not images:
    raise SystemExit('no COCO JPG found')
print(images[0])
PY
)"
test -f "$IMAGE_PATH"
```

Wait for two consecutive idle GPU checks (no compute PID and at least 24,000 MiB
free), then use an external worker while the tmux parent waits for it:

```bash
set -euo pipefail
wait_for_idle_gpu() {
  local good=0
  while [ "$good" -lt 2 ]; do
    local pids free
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' || true)"
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [ -z "$pids" ] && [ "$free" -ge 24000 ]; then good=$((good + 1)); else good=0; fi
    sleep 10
  done
}
wait_for_idle_gpu
OUTPUT="/home/chltjsdh/vea/p2_outputs/vea-p2-t001b-$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${OUTPUT}.log"
SESSION="vea-p2-t001b-$(date -u +%H%M%S)"
test ! -e "$OUTPUT"
tmux new-session -d -s "$SESSION" "bash -lc 'set -euo pipefail; setsid \"$MAIN_PY\" \"$REPO/scripts/run_dinov3_raw_injection.py\" --tinyllava-config \"$REPO/configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json\" --image-path \"$IMAGE_PATH\" --cache-root \"$HF_HOME/hub\" --repository-root \"$REPO\" --output-dir \"$OUTPUT\" --device cuda:0 --dtype float16 > \"$LOG\" 2>&1 & child=\$!; wait \$child'"
tmux attach -t "$SESSION"
```

After the worker exits, validate and submit only the smoke evidence:

```bash
set -euo pipefail
"$MAIN_PY" - "$OUTPUT" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
assert result['schema_id'] == 'vea.p2.t001b.dinov3_raw_injection.v1'
assert result['status'] == 'PASS_DINOV3_RAW_INJECTION_SMOKE'
assert result['raw_dinov3']['clip_vision_calls'] == 0
assert result['raw_dinov3']['encode_images_restored'] is True
print('native_answer=' + result['native']['answer'])
print('dinov3_raw_answer=' + result['raw_dinov3']['answer'])
for name in ('result.json', 'input_patch_features.safetensors'):
    path = root / name
    print(f'{name} sha256={sha256(path.read_bytes()).hexdigest()}')
PY
printf 'result=%s log=%s\n' "$OUTPUT/result.json" "$LOG"
```

Do not run calibration, a multi-image benchmark, or another model injection.
