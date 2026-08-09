# VEA-P2-T001B-R1 Apple DINOv3 raw-injection smoke

Run this once, only after independent review supplies the reviewed task commit.
It uses a fresh clone, existing offline caches and runtimes, and no downloads or
installation. All deterministic root, snapshot, interpreter, and package checks
finish before GPU waiting.

```bash
set -euo pipefail
TASK_BRANCH='codex/vea-p2-t001b-r1-explicit-apple-roots'
TASK_COMMIT_SHA='<REVIEWED_TASK_COMMIT_SHA>'
REPO="/home/chltjsdh/vea/p2_repos/vea-p2-t001b-r1-$(date -u +%Y%m%dT%H%M%SZ)"
MAIN_PY='/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3'
DINO_RUNTIME_ROOT='/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated'
OVERLAY='/home/chltjsdh/vea/runtime/p2_tinyllava_overlay/site-packages'
HF_HOME='/home/chltjsdh/vea/checkpoints/huggingface'
HF_HUB_CACHE="$HF_HOME/hub"
test ! -e "$REPO"
git clone --no-tags --single-branch --branch "$TASK_BRANCH" https://github.com/Anzenoh/NAIS.git "$REPO"
cd "$REPO"
test "$(git rev-parse HEAD)" = "$TASK_COMMIT_SHA"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -x "$MAIN_PY"
test -d "$DINO_RUNTIME_ROOT" && test ! -L "$DINO_RUNTIME_ROOT"
DINO_PY="$(PYTHONPATH="$REPO/src" "$MAIN_PY" - "$DINO_RUNTIME_ROOT" <<'PY'
from pathlib import Path
import sys
from vision_encoder_alignment.vlm_compat.dinov3_raw_injection import (
    resolve_apple_dinov3_interpreter,
)
print(resolve_apple_dinov3_interpreter(Path(sys.argv[1])))
PY
)"
test "$DINO_PY" = "$DINO_RUNTIME_ROOT/bin/python" -o "$DINO_PY" = "$DINO_RUNTIME_ROOT/bin/python3"
printf 'dinov3_interpreter=%s\n' "$DINO_PY"
```

Resolve and fully validate the explicit snapshot before any GPU wait. The search
is read-only and bounded to the two named roots; symlink directories are not
followed. The exact P1 checkpoint-cache candidate is tested first.

```bash
set -euo pipefail
DINO_SNAPSHOT_ROOT="$(PYTHONPATH="$REPO/src" "$MAIN_PY" - "$REPO" "$DINO_RUNTIME_ROOT" "$HF_HUB_CACHE" <<'PY'
import os
from pathlib import Path
import sys
from vision_encoder_alignment.vlm_compat.dinov3_raw_injection import validate_dinov3_roots

repository = Path(sys.argv[1])
runtime = Path(sys.argv[2])
hub = Path(sys.argv[3])
revision = '5931719e67bbdb9737e363e781fb0c67687896bc'
cache_candidate = hub / 'models--facebook--dinov3-vitb16-pretrain-lvd1689m' / 'snapshots' / revision
target_name = 'dinov3-vitb16-r5931719-safetensors-v1'

def valid(candidate: Path) -> bool:
    try:
        validate_dinov3_roots(
            repository_root=repository,
            runtime_root=runtime,
            snapshot_root=candidate,
        )
        return True
    except Exception:
        return False

if valid(cache_candidate):
    selected = cache_candidate.resolve()
else:
    candidates = []
    for search_root in (Path('/home/chltjsdh/vea/checkpoints'), Path('/home/chltjsdh/vea/snapshots')):
        if not search_root.is_dir() or search_root.is_symlink():
            continue
        for current, directories, _files in os.walk(search_root, followlinks=False):
            directories[:] = [name for name in directories if not (Path(current) / name).is_symlink()]
            if target_name in directories:
                candidate = Path(current) / target_name
                if valid(candidate):
                    candidates.append(candidate.resolve())
                directories.remove(target_name)
    unique = sorted(set(candidates), key=lambda path: path.as_posix())
    if len(unique) != 1:
        raise SystemExit(f'expected exactly one valid DINO snapshot, found {len(unique)}')
    selected = unique[0]
print(selected)
PY
)"
test -n "$DINO_SNAPSHOT_ROOT"
printf 'dinov3_snapshot_root=%s\n' "$DINO_SNAPSHOT_ROOT"
PYTHONPATH="$OVERLAY" "$MAIN_PY" - <<'PY'
import importlib.metadata as m
assert m.version('sentencepiece') == '0.2.0'
assert m.version('protobuf') == '4.25.3'
print('sentencepiece=0.2.0 protobuf=4.25.3')
PY
```

Enable the existing offline cache and choose one deterministic image, still
before GPU waiting:

```bash
set -euo pipefail
export HF_HOME HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
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
free), then run exactly one smoke in a child process. The tmux parent records the
child exit code and remains alive even when the child fails.

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
OUTPUT="/home/chltjsdh/vea/p2_outputs/vea-p2-t001b-r1-$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${OUTPUT}.log"
SESSION="vea-p2-t001b-r1-$(date -u +%H%M%S)"
test ! -e "$OUTPUT"
tmux new-session -d -s "$SESSION" "bash -lc 'set +e; PYTHONPATH=\"$OVERLAY\" setsid \"$MAIN_PY\" \"$REPO/scripts/run_dinov3_raw_injection.py\" --tinyllava-config \"$REPO/configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json\" --image-path \"$IMAGE_PATH\" --cache-root \"$HF_HUB_CACHE\" --repository-root \"$REPO\" --dinov3-runtime-root \"$DINO_RUNTIME_ROOT\" --dinov3-snapshot-root \"$DINO_SNAPSHOT_ROOT\" --output-dir \"$OUTPUT\" --device cuda:0 --dtype float16 > \"$LOG\" 2>&1 & child=\$!; wait \$child; rc=\$?; printf \"child_exit_code=%s\\n\" \"\$rc\"; exec bash'"
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
