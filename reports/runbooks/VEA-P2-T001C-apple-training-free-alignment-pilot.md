# VEA-P2-T001C Apple training-free alignment pilot

Run this exactly once, after independent review supplies the reviewed T001C commit.
Start in the existing tmux parent shell. The commands do not create or attach a
tmux session, replace the parent shell, download, install, delete, or rebuild a snapshot.

```bash
TASK_BRANCH='codex/vea-p2-t001c-training-free-alignment-pilot'
TASK_COMMIT_SHA='<REVIEWED_TASK_COMMIT_SHA>'
REPO="/home/chltjsdh/vea/p2_repos/vea-p2-t001c-$(date -u +%Y%m%dT%H%M%SZ)"
MAIN_PY='/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3'
DINO_RUNTIME_ROOT='/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated'
OVERLAY='/home/chltjsdh/vea/runtime/p2_tinyllava_overlay/site-packages'
HF_HOME='/home/chltjsdh/vea/checkpoints/huggingface'
HF_HUB_CACHE="$HF_HOME/hub"
REVISION='5931719e67bbdb9737e363e781fb0c67687896bc'
SOURCE_SNAPSHOT_ROOT="$HF_HUB_CACHE/models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/$REVISION"
DINO_SNAPSHOT_ROOT='/home/chltjsdh/vea/snapshots/dinov3-vitb16-r5931719-safetensors-v1'
DATASET_ROOT='/home/chltjsdh/vea/datasets/coco2017/images/val2017'
vea_t001c_clone_preflight() {
  test ! -e "$REPO" || { printf 'STOP: repository path exists\n'; return 1; }
  git clone --no-tags --single-branch --branch "$TASK_BRANCH" https://github.com/Anzenoh/NAIS.git "$REPO" || return 1
  cd "$REPO" || return 1
  test "$(git rev-parse HEAD)" = "$TASK_COMMIT_SHA" || { printf 'STOP: commit mismatch\n'; return 1; }
  test -z "$(git status --porcelain=v1 --untracked-files=all)" || { printf 'STOP: dirty clone\n'; return 1; }
  test -x "$MAIN_PY" || { printf 'STOP: main runtime missing\n'; return 1; }
}
vea_t001c_clone_preflight
CLONE_PREFLIGHT_RC=$?
printf 'clone_preflight_exit_code=%s\n' "$CLONE_PREFLIGHT_RC"
```

If `CLONE_PREFLIGHT_RC` is nonzero, stop in the current parent shell. Otherwise,
run the following preflight block. It verifies the existing snapshot; it never invokes
the materializer's `materialize` action.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
vea_t001c_environment_preflight() {
"$MAIN_PY" "$REPO/scripts/materialize_dinov3_snapshot_from_hf_cache.py" verify \
  --repository-root "$REPO" \
  --source-cache-root "$HF_HUB_CACHE" \
  --source-snapshot-root "$SOURCE_SNAPSHOT_ROOT" \
  --destination-root "$DINO_SNAPSHOT_ROOT" \
  --runtime-root "$DINO_RUNTIME_ROOT" || { printf 'STOP: snapshot verification failed\n'; return 1; }

DINO_PY="$(PYTHONPATH="$REPO/src" "$MAIN_PY" - "$DINO_RUNTIME_ROOT" <<'PY'
from pathlib import Path
import sys
from vision_encoder_alignment.vlm_compat.dinov3_raw_injection import resolve_apple_dinov3_interpreter
print(resolve_apple_dinov3_interpreter(Path(sys.argv[1])))
PY
)" || return 1
case "$DINO_PY" in
  "$DINO_RUNTIME_ROOT/bin/python"|"$DINO_RUNTIME_ROOT/bin/python3") ;;
  *) printf 'STOP: unexpected DINO interpreter\n'; return 1 ;;
esac
printf 'main_interpreter=%s\ndinov3_interpreter=%s\ndinov3_snapshot=%s\n' "$MAIN_PY" "$DINO_PY" "$DINO_SNAPSHOT_ROOT"

PYTHONPATH="$OVERLAY" "$MAIN_PY" - <<'PY' || { printf 'STOP: dependencies unavailable\n'; return 1; }
import importlib.metadata as m
assert m.version('sentencepiece') == '0.2.0'
assert m.version('protobuf') == '4.25.3'
import safetensors, torch, transformers
print('TinyLLaVA offline dependencies valid')
PY

PYTHONPATH="$REPO/src" "$MAIN_PY" - "$REPO/data/manifests/p1_1000.jsonl" "$DATASET_ROOT" <<'PY' || { printf 'STOP: selected source validation failed\n'; return 1; }
from pathlib import Path
import sys
from vision_encoder_alignment.vlm_compat.training_free_alignment_pilot import select_pilot_rows, validate_selected_sources
selection = select_pilot_rows(sys.argv[1])
paths = validate_selected_sources(selection, sys.argv[2])
for partition, rows, selected in (
    ('calibration', selection.calibration, paths[:8]), ('test', selection.test, paths[8:])
):
    for row, path in zip(rows, selected, strict=True):
        print(partition, row.selection_index, row.image_id, path, row.source_sha256)
PY
}
vea_t001c_environment_preflight
ENVIRONMENT_PREFLIGHT_RC=$?
printf 'environment_preflight_exit_code=%s\n' "$ENVIRONMENT_PREFLIGHT_RC"
```

Only when both preflight exit codes are zero, wait for two consecutive idle checks and
launch the pilot exactly once as one child process. A failure returns control to
the current tmux parent; do not rerun it as a diagnostic.

```bash
good=0
while [ "$good" -lt 2 ]; do
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' || true)"
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [ -z "$pids" ] && [ "${free:-0}" -ge 24000 ]; then good=$((good + 1)); else good=0; fi
  sleep 10
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="/home/chltjsdh/vea/p2_outputs/vea-p2-t001c-${STAMP}"
LOG="${OUTPUT}.log"
vea_t001c_execute_once() {
test ! -e "$OUTPUT" && test ! -e "$LOG" || { printf 'STOP: output path exists\n'; return 1; }
setsid env PYTHONPATH="$OVERLAY" "$MAIN_PY" "$REPO/scripts/run_training_free_alignment_pilot.py" \
  --tinyllava-config "$REPO/configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json" \
  --manifest "$REPO/data/manifests/p1_1000.jsonl" \
  --dataset-root "$DATASET_ROOT" --cache-root "$HF_HUB_CACHE" --repository-root "$REPO" \
  --dinov3-runtime-root "$DINO_RUNTIME_ROOT" --dinov3-snapshot-root "$DINO_SNAPSHOT_ROOT" \
  --output-dir "$OUTPUT" --device cuda:0 --dtype float16 >"$LOG" 2>&1 &
CHILD_PID=$!
wait "$CHILD_PID"
CHILD_RC=$?
printf 'child_pid=%s child_exit_code=%s\n' "$CHILD_PID" "$CHILD_RC"
cat "$LOG"
return "$CHILD_RC"
}
vea_t001c_execute_once
PILOT_RC=$?
printf 'pilot_exit_code=%s\n' "$PILOT_RC"
```

Do not start another execution if `PILOT_RC` is nonzero. For a zero exit code,
print the bounded evidence and verify all six artifacts:

```bash
"$MAIN_PY" - "$OUTPUT" "$PILOT_RC" <<'PY'
import csv, hashlib, json
from pathlib import Path
import sys
root, exit_code = Path(sys.argv[1]), int(sys.argv[2])
assert exit_code == 0
names = ('result.json','per_image.csv','summary.csv','paired_patch_features.safetensors','alignment_parameters.safetensors','test_connector_outputs.safetensors')
assert {p.name for p in root.iterdir()} == set(names)
result = json.loads((root/'result.json').read_text(encoding='utf-8'))
assert result['schema_id'] == 'vea.p2.t001c.training_free_alignment_pilot.v1'
assert result['status'] == 'PASS_TRAINING_FREE_ALIGNMENT_PILOT'
assert result['runtime_call_counts']['test_generation_calls'] == 16
print((root/'summary.csv').read_text(encoding='utf-8'))
with (root/'per_image.csv').open(newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
for image_id in dict.fromkeys(row['image_id'] for row in rows):
    print('answers image_id=' + image_id)
    for row in [item for item in rows if item['image_id'] == image_id]:
        print(row['condition'] + ': ' + row['condition_answer'])
for name in names:
    path = root/name
    print(f'{name} bytes={path.stat().st_size} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}')
print(f'result={root / "result.json"} exit_code={exit_code}')
PY
```

Stop here. Do not run SigLIP2, MLP, Ridge, CORAL, a second prompt, calibration-only
diagnostics, expanded evaluation, or any second experiment.
