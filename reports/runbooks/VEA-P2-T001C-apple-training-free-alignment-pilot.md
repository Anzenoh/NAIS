# VEA-P2-T001C-R2 Apple batch-preserving alignment pilot

Run this once from the user's existing tmux parent shell after independent review
supplies the reviewed R2 commit. The parent commands below apply no `set -e`,
`exit`, `exec`, ERR/EXIT trap, or tmux create/attach operation. Preflight and the
single pilot execution run inside one separate child bash.

```bash
TASK_BRANCH='codex/vea-p2-t001c-r2-preserve-injection-batch-axis'
TASK_COMMIT_SHA='<REVIEWED_TASK_COMMIT_SHA>'
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPO="/home/chltjsdh/vea/p2_repos/vea-p2-t001c-r2-${STAMP}"
OUTPUT="/home/chltjsdh/vea/p2_outputs/vea-p2-t001c-r2-${STAMP}"
WORKER="/home/chltjsdh/vea/p2_outputs/vea-p2-t001c-r2-${STAMP}.worker.sh"
WORKER_LOG="/home/chltjsdh/vea/p2_outputs/vea-p2-t001c-r2-${STAMP}.worker.log"
PILOT_LOG="/home/chltjsdh/vea/p2_outputs/vea-p2-t001c-r2-${STAMP}.pilot.log"
MAIN_PY='/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3'
DINO_RUNTIME_ROOT='/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated'
OVERLAY='/home/chltjsdh/vea/runtime/p2_tinyllava_overlay/site-packages'
HF_HOME='/home/chltjsdh/vea/checkpoints/huggingface'
HF_HUB_CACHE="$HF_HOME/hub"
REVISION='5931719e67bbdb9737e363e781fb0c67687896bc'
SOURCE_SNAPSHOT_ROOT="$HF_HUB_CACHE/models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/$REVISION"
DINO_SNAPSHOT_ROOT='/home/chltjsdh/vea/snapshots/dinov3-vitb16-r5931719-safetensors-v1'
DATASET_ROOT='/home/chltjsdh/vea/datasets/coco2017/images/val2017'

export TASK_BRANCH TASK_COMMIT_SHA REPO OUTPUT WORKER_LOG PILOT_LOG MAIN_PY
export DINO_RUNTIME_ROOT OVERLAY HF_HOME HF_HUB_CACHE SOURCE_SNAPSHOT_ROOT
export DINO_SNAPSHOT_ROOT DATASET_ROOT

if [ -e "$REPO" ] || [ -e "$OUTPUT" ] || [ -e "$WORKER" ] || [ -e "$WORKER_LOG" ] || [ -e "$PILOT_LOG" ]; then
  printf 'STOP: one or more fresh R2 paths already exist\n'
  WORKER_EXIT_CODE=125
  VEA_P2_EXECUTION=FAILED
  printf 'WORKER_EXIT_CODE=%s\nVEA_P2_EXECUTION=%s\nTMUX_PARENT_SHELL=ALIVE\n' "$WORKER_EXIT_CODE" "$VEA_P2_EXECUTION"
else
  cat >"$WORKER" <<'BASH'
#!/usr/bin/env bash
set -uo pipefail

worker_main() {
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

  git clone --no-tags --single-branch --branch "$TASK_BRANCH" https://github.com/Anzenoh/NAIS.git "$REPO" || return 10
  cd "$REPO" || return 11
  test "$(git rev-parse HEAD)" = "$TASK_COMMIT_SHA" || { printf 'STOP: reviewed commit mismatch\n'; return 12; }
  test -z "$(git status --porcelain=v1 --untracked-files=all)" || { printf 'STOP: reviewed clone is dirty\n'; return 13; }
  test -x "$MAIN_PY" || { printf 'STOP: main runtime missing\n'; return 14; }

  "$MAIN_PY" "$REPO/scripts/materialize_dinov3_snapshot_from_hf_cache.py" verify \
    --repository-root "$REPO" \
    --source-cache-root "$HF_HUB_CACHE" \
    --source-snapshot-root "$SOURCE_SNAPSHOT_ROOT" \
    --destination-root "$DINO_SNAPSHOT_ROOT" \
    --runtime-root "$DINO_RUNTIME_ROOT" || { printf 'STOP: existing snapshot verification failed\n'; return 15; }

  DINO_PY="$(PYTHONPATH="$REPO/src" "$MAIN_PY" - "$DINO_RUNTIME_ROOT" <<'PY'
from pathlib import Path
import sys
from vision_encoder_alignment.vlm_compat.dinov3_raw_injection import resolve_apple_dinov3_interpreter
print(resolve_apple_dinov3_interpreter(Path(sys.argv[1])))
PY
)" || return 16
  case "$DINO_PY" in
    "$DINO_RUNTIME_ROOT/bin/python"|"$DINO_RUNTIME_ROOT/bin/python3") ;;
    *) printf 'STOP: unexpected DINO interpreter\n'; return 17 ;;
  esac
  printf 'main_interpreter=%s\ndinov3_interpreter=%s\ndinov3_snapshot=%s\n' "$MAIN_PY" "$DINO_PY" "$DINO_SNAPSHOT_ROOT"

  PYTHONPATH="$OVERLAY" "$MAIN_PY" - <<'PY' || { printf 'STOP: TinyLLaVA dependency metadata unavailable\n'; return 18; }
import importlib.metadata as m
assert m.version('sentencepiece') == '0.2.0'
assert m.version('protobuf') == '4.25.3'
for distribution in ('safetensors', 'torch', 'transformers'):
    print(f'{distribution}={m.version(distribution)}')
print('TinyLLaVA dependency metadata valid; no runtime module imported')
PY

  PYTHONPATH="$REPO/src" "$MAIN_PY" - "$REPO/data/manifests/p1_1000.jsonl" "$DATASET_ROOT" <<'PY' || { printf 'STOP: selected source validation failed\n'; return 19; }
import sys
from vision_encoder_alignment.vlm_compat.training_free_alignment_pilot import select_pilot_rows, validate_selected_sources
selection = select_pilot_rows(sys.argv[1])
paths = validate_selected_sources(selection, sys.argv[2])
for partition, rows, selected in (
    ('calibration', selection.calibration, paths[:8]), ('test', selection.test, paths[8:])
):
    for row, path in zip(rows, selected, strict=True):
        print(partition, row.selection_index, row.image_id, path.name, row.source_sha256)
PY

  good=0
  while [ "$good" -lt 2 ]; do
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' || true)"
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [ -z "$pids" ] && [ "${free:-0}" -ge 24000 ]; then good=$((good + 1)); else good=0; fi
    sleep 10
  done

  PYTHONPATH="$OVERLAY" "$MAIN_PY" "$REPO/scripts/run_training_free_alignment_pilot.py" \
    --tinyllava-config "$REPO/configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json" \
    --manifest "$REPO/data/manifests/p1_1000.jsonl" \
    --dataset-root "$DATASET_ROOT" --cache-root "$HF_HUB_CACHE" --repository-root "$REPO" \
    --dinov3-runtime-root "$DINO_RUNTIME_ROOT" --dinov3-snapshot-root "$DINO_SNAPSHOT_ROOT" \
    --output-dir "$OUTPUT" --device cuda:0 --dtype float16 >"$PILOT_LOG" 2>&1
  pilot_rc=$?
  cat "$PILOT_LOG"
  if [ "$pilot_rc" -ne 0 ]; then
    if [ -f "$OUTPUT/result.json" ]; then
      printf '%s\n' '----- precise failure result -----'
      "$MAIN_PY" -m json.tool "$OUTPUT/result.json"
    else
      printf 'STOP: pilot failed without result.json; inspect full traceback above\n'
    fi
    return "$pilot_rc"
  fi

  "$MAIN_PY" - "$OUTPUT" <<'PY' || return 20
import csv, hashlib, json
from pathlib import Path
import sys
root = Path(sys.argv[1])
names = ('result.json','per_image.csv','summary.csv','paired_patch_features.safetensors','alignment_parameters.safetensors','test_connector_outputs.safetensors')
assert {path.name for path in root.iterdir()} == set(names)
result = json.loads((root/'result.json').read_text(encoding='utf-8'))
assert result['schema_id'] == 'vea.p2.t001c.training_free_alignment_pilot.v1'
assert result['status'] == 'PASS_TRAINING_FREE_ALIGNMENT_PILOT'
assert result['calibration_capture']['language_generation'] is False
assert result['runtime_call_counts'] == {
    'dinov3_batch_extractions': 1,
    'tinyllava_model_loads': 1,
    'calibration_native_feature_calls': 8,
    'calibration_generation_calls': 0,
    'test_generation_calls': 16,
}
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
print(f'result={root / "result.json"}')
PY
}

worker_main
BASH

  chmod 700 "$WORKER"
  setsid bash "$WORKER" >"$WORKER_LOG" 2>&1 &
  WORKER_PID=$!
  wait "$WORKER_PID"
  WORKER_EXIT_CODE=$?
  cat "$WORKER_LOG"
  if [ "$WORKER_EXIT_CODE" -eq 0 ]; then VEA_P2_EXECUTION=PASS; else VEA_P2_EXECUTION=FAILED; fi
  printf 'WORKER_EXIT_CODE=%s\nVEA_P2_EXECUTION=%s\nTMUX_PARENT_SHELL=ALIVE\n' "$WORKER_EXIT_CODE" "$VEA_P2_EXECUTION"
fi
```

If the worker fails, use the printed traceback and bounded `result.json`; do not
run a diagnostic-only or second pilot. Stop here without SigLIP2, MLP, Ridge,
CORAL, another prompt, snapshot rebuild, download, or installation.
