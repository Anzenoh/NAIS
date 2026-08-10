# VEA-P2-T001B-R3 Apple worker-response raw-injection smoke

Run this once after independent review supplies the reviewed R3 commit. It uses
the existing offline Hugging Face cache, requires idempotent reuse of the exact
contract snapshot, and then runs one raw-injection smoke. It never downloads or installs.

```bash
set -euo pipefail
TASK_BRANCH='codex/vea-p2-t001b-r3-apple-worker-response'
TASK_COMMIT_SHA='<REVIEWED_TASK_COMMIT_SHA>'
REPO="/home/chltjsdh/vea/p2_repos/vea-p2-t001b-r3-$(date -u +%Y%m%dT%H%M%SZ)"
MAIN_PY='/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3'
DINO_RUNTIME_ROOT='/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated'
OVERLAY='/home/chltjsdh/vea/runtime/p2_tinyllava_overlay/site-packages'
HF_HOME='/home/chltjsdh/vea/checkpoints/huggingface'
HF_HUB_CACHE="$HF_HOME/hub"
REVISION='5931719e67bbdb9737e363e781fb0c67687896bc'
SOURCE_SNAPSHOT_ROOT="$HF_HUB_CACHE/models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/$REVISION"
DINO_SNAPSHOT_ROOT='/home/chltjsdh/vea/snapshots/dinov3-vitb16-r5931719-safetensors-v1'
DATASET_ROOT='/home/chltjsdh/vea/datasets/coco2017/images/val2017'
test ! -e "$REPO"
git clone --no-tags --single-branch --branch "$TASK_BRANCH" https://github.com/Anzenoh/NAIS.git "$REPO"
cd "$REPO"
test "$(git rev-parse HEAD)" = "$TASK_COMMIT_SHA"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -x "$MAIN_PY"
```

Create one external controller. It launches both materialization and smoke as
child worker processes. Every failure drops into a live tmux parent shell.

```bash
set -euo pipefail
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SESSION="vea-p2-t001b-r3-${STAMP}"
OUTPUT="/home/chltjsdh/vea/p2_outputs/vea-p2-t001b-r3-${STAMP}"
LOG="${OUTPUT}.log"
MATERIALIZE_LOG="${OUTPUT}.materialize.log"
CONTROLLER="${OUTPUT}.controller.sh"
test ! -e "$OUTPUT" && test ! -e "$LOG" && test ! -e "$CONTROLLER"
export REPO MAIN_PY DINO_RUNTIME_ROOT OVERLAY HF_HOME HF_HUB_CACHE
export SOURCE_SNAPSHOT_ROOT DINO_SNAPSHOT_ROOT DATASET_ROOT OUTPUT LOG MATERIALIZE_LOG
cat > "$CONTROLLER" <<'BASH'
#!/usr/bin/env bash
set -uo pipefail
hold() {
  printf 'STOP: %s\n' "$1"
  exec bash
}
run_child() {
  setsid "$@" &
  local child=$!
  wait "$child"
  CHILD_RC=$?
  printf 'child_pid=%s child_exit_code=%s\n' "$child" "$CHILD_RC"
  return "$CHILD_RC"
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

# Validate the exact existing HF cache bytes before publication.
if ! "$MAIN_PY" - "$REPO" "$HF_HUB_CACHE" "$SOURCE_SNAPSHOT_ROOT" "$DINO_SNAPSHOT_ROOT" "$DINO_RUNTIME_ROOT" <<'PY'
import importlib.util
from pathlib import Path
import sys
script = Path(sys.argv[1]) / 'scripts/materialize_dinov3_snapshot_from_hf_cache.py'
spec = importlib.util.spec_from_file_location('vea_materializer_preflight', script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
plan = module.plan_materialization(
    repository_root=sys.argv[1], source_cache_root=sys.argv[2],
    source_snapshot_root=sys.argv[3], destination_root=sys.argv[4],
    runtime_root=sys.argv[5],
)
selected = module.validate_source_files(plan)
assert set(selected) == {'config.json', 'preprocessor_config.json', 'model.safetensors'}
print('DINOV3_HF_SOURCE_SNAPSHOT_VALID')
PY
then hold 'HF source snapshot validation failed'; fi

# Materialize or idempotently reuse the exact contract snapshot in a child.
if ! run_child "$MAIN_PY" "$REPO/scripts/materialize_dinov3_snapshot_from_hf_cache.py" materialize \
  --repository-root "$REPO" \
  --source-cache-root "$HF_HUB_CACHE" \
  --source-snapshot-root "$SOURCE_SNAPSHOT_ROOT" \
  --destination-root "$DINO_SNAPSHOT_ROOT" \
  --runtime-root "$DINO_RUNTIME_ROOT" > "$MATERIALIZE_LOG" 2>&1
then cat "$MATERIALIZE_LOG"; hold 'snapshot materialization failed'; fi
cat "$MATERIALIZE_LOG"
if ! grep -Fxq 'DINOV3_CONTRACT_SNAPSHOT_ALREADY_VALID' "$MATERIALIZE_LOG"; then
  hold 'existing contract snapshot was not reused through ALREADY_VALID'
fi

if ! "$MAIN_PY" "$REPO/scripts/materialize_dinov3_snapshot_from_hf_cache.py" verify \
  --repository-root "$REPO" \
  --source-cache-root "$HF_HUB_CACHE" \
  --source-snapshot-root "$SOURCE_SNAPSHOT_ROOT" \
  --destination-root "$DINO_SNAPSHOT_ROOT" \
  --runtime-root "$DINO_RUNTIME_ROOT"
then hold 'contract snapshot verification failed'; fi

# Validate both runtimes and TinyLLaVA dependencies before GPU waiting.
DINO_PY="$(PYTHONPATH="$REPO/src" "$MAIN_PY" - "$DINO_RUNTIME_ROOT" <<'PY'
from pathlib import Path
import sys
from vision_encoder_alignment.vlm_compat.dinov3_raw_injection import resolve_apple_dinov3_interpreter
print(resolve_apple_dinov3_interpreter(Path(sys.argv[1])))
PY
)" || hold 'DINO runtime validation failed'
if [ "$DINO_PY" != "$DINO_RUNTIME_ROOT/bin/python" ] && [ "$DINO_PY" != "$DINO_RUNTIME_ROOT/bin/python3" ]; then
  hold 'unexpected DINO interpreter'
fi
printf 'dinov3_interpreter=%s\ndinov3_snapshot_root=%s\n' "$DINO_PY" "$DINO_SNAPSHOT_ROOT"
if ! PYTHONPATH="$OVERLAY" "$MAIN_PY" - <<'PY'
import importlib.metadata as m
assert m.version('sentencepiece') == '0.2.0'
assert m.version('protobuf') == '4.25.3'
print('sentencepiece=0.2.0 protobuf=4.25.3')
PY
then hold 'TinyLLaVA dependency validation failed'; fi

IMAGE_PATH="$($MAIN_PY - "$DATASET_ROOT" <<'PY'
from pathlib import Path
import sys
images = sorted(path for path in Path(sys.argv[1]).glob('*.jpg') if path.is_file())
if not images:
    raise SystemExit('no COCO JPG found')
print(images[0])
PY
)" || hold 'deterministic image selection failed'

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

if ! PYTHONPATH="$OVERLAY" run_child "$MAIN_PY" "$REPO/scripts/run_dinov3_raw_injection.py" \
  --tinyllava-config "$REPO/configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json" \
  --image-path "$IMAGE_PATH" --cache-root "$HF_HUB_CACHE" --repository-root "$REPO" \
  --dinov3-runtime-root "$DINO_RUNTIME_ROOT" --dinov3-snapshot-root "$DINO_SNAPSHOT_ROOT" \
  --output-dir "$OUTPUT" --device cuda:0 --dtype float16 > "$LOG" 2>&1
then cat "$LOG"; hold 'raw-injection smoke failed'; fi
cat "$LOG"

if ! "$MAIN_PY" - "$OUTPUT" <<'PY'
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
then hold 'smoke evidence validation failed'; fi
printf 'PASS result=%s log=%s materialize_log=%s\n' "$OUTPUT/result.json" "$LOG" "$MATERIALIZE_LOG"
exec bash
BASH
chmod 700 "$CONTROLLER"
tmux new-session -d -s "$SESSION" "bash '$CONTROLLER'"
tmux attach -t "$SESSION"
```

Do not run calibration, a multi-image benchmark, another injection, download,
installation, runtime mutation, or symlink creation.
