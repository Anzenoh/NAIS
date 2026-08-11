# VEA-P2-T001D-R1 persistent Apple runbook

Run this only after independent review of the pushed R1 commit, from the user's
existing tmux parent shell. The parent blocks deliberately do not enable strict
shell options, replace or terminate the shell, install traps, or create/attach
tmux. Strict behavior is confined to the generated child scripts.

## 1. Persistent paths, reviewed checkout, and main-lane metadata preflight

```bash
REMOTE_URL='<reviewed repository URL>'
REVIEWED_BRANCH='codex/vea-p2-t001d-r1-real-run-readiness'
REVIEWED_COMMIT='<40-character reviewed R1 commit>'
REPO_PARENT='/home/chltjsdh/vea/p2_repos'
RUN_PARENT='/home/chltjsdh/vea/p2_runs'
RUN_TOKEN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
TASK_REPOSITORY="$REPO_PARENT/VEA-P2-T001D-R1-$RUN_TOKEN"
RUN_ROOT="$RUN_PARENT/VEA-P2-T001D-R1-$RUN_TOKEN"
LAUNCH_LOG_ROOT="$RUN_PARENT/launcher-$RUN_TOKEN"

MAIN_PY='/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3'
MAIN_OVERLAY='/home/chltjsdh/vea/runtime/p2_tinyllava_overlay/site-packages'
TRANSFORMERS_CACHE='/home/chltjsdh/vea/runtime/p2_transformers_cache'
HF_MODULES_CACHE='/home/chltjsdh/vea/runtime/p2_hf_modules'
HF_HOME='/home/chltjsdh/vea/checkpoints/huggingface'
HF_HUB_CACHE='/home/chltjsdh/vea/checkpoints/huggingface/hub'
DATASET_ROOT='<existing immutable p1_1000 dataset root>'
CACHE_ROOT="$HF_HUB_CACHE"
DINOV3_RUNTIME_ROOT='<existing approved DINOv3 runtime root>'
DINOV3_SNAPSHOT_ROOT='<existing materialized DINOv3 snapshot root>'
DEVICE='cuda:0'
GPU_INDEX='0'

HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
DO_NOT_TRACK=1
PYTHONNOUSERSITE=1
PYTHONPATH="$MAIN_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export MAIN_PY MAIN_OVERLAY TRANSFORMERS_CACHE HF_MODULES_CACHE HF_HOME HF_HUB_CACHE
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_HUB_DISABLE_TELEMETRY DO_NOT_TRACK
export PYTHONNOUSERSITE PYTHONPATH DATASET_ROOT CACHE_ROOT DINOV3_RUNTIME_ROOT DINOV3_SNAPSHOT_ROOT DEVICE GPU_INDEX
export TASK_REPOSITORY RUN_ROOT LAUNCH_LOG_ROOT REVIEWED_COMMIT REVIEWED_BRANCH

mkdir -p "$REPO_PARENT" "$RUN_PARENT" "$LAUNCH_LOG_ROOT"
git clone --branch "$REVIEWED_BRANCH" --single-branch "$REMOTE_URL" "$TASK_REPOSITORY"
CLONE_RC=$?
OBSERVED_BRANCH="$(git -C "$TASK_REPOSITORY" branch --show-current 2>/dev/null)"
OBSERVED_COMMIT="$(git -C "$TASK_REPOSITORY" rev-parse HEAD 2>/dev/null)"
OBSERVED_DIRTY="$(git -C "$TASK_REPOSITORY" status --porcelain 2>/dev/null)"
if [ "$CLONE_RC" -eq 0 ] && [ "$OBSERVED_BRANCH" = "$REVIEWED_BRANCH" ] && \
   [ "$OBSERVED_COMMIT" = "$REVIEWED_COMMIT" ] && [ -z "$OBSERVED_DIRTY" ]; then
  CHECKOUT_PREFLIGHT=PASS
else
  CHECKOUT_PREFLIGHT=FAILED
fi
printf 'CHECKOUT_PREFLIGHT=%s\n' "$CHECKOUT_PREFLIGHT"

"$MAIN_PY" - <<'PY'
from importlib import metadata
from pathlib import Path
import os
required_files = [Path(os.environ["MAIN_PY"])]
required_dirs = [
    Path(os.environ["MAIN_OVERLAY"]),
    Path(os.environ["TRANSFORMERS_CACHE"]),
    Path(os.environ["HF_MODULES_CACHE"]),
    Path(os.environ["HF_HOME"]),
    Path(os.environ["HF_HUB_CACHE"]),
]
assert all(path.is_file() for path in required_files), "approved main interpreter file is unavailable"
assert all(path.is_dir() for path in required_dirs), "approved overlay/cache directory is unavailable"
for package in ("torch", "transformers", "safetensors", "Pillow"):
    print(f"PACKAGE_METADATA {package}={metadata.version(package)}")
print("MAIN_RUNTIME_METADATA_PREFLIGHT=PASS")
PY
RUNTIME_PREFLIGHT_RC=$?
printf 'RUNTIME_PREFLIGHT_EXIT_CODE=%s\n' "$RUNTIME_PREFLIGHT_RC"
```

Proceed only when checkout and runtime metadata preflight both pass. No Torch,
Transformers, Safetensors, Pillow, model, or CUDA module has been imported.

## 2. Plan exactly once, before GPU waiting

```bash
PLAN_STDOUT="$LAUNCH_LOG_ROOT/plan-$RUN_TOKEN.stdout.json"
PLAN_STDERR="$LAUNCH_LOG_ROOT/plan-$RUN_TOKEN.stderr.log"
"$MAIN_PY" "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" plan \
  --config "$TASK_REPOSITORY/configs/vlm_compat/p2_universal_anchor_transport_v0.json" \
  --manifest "$TASK_REPOSITORY/data/manifests/p1_1000.jsonl" \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --repository-root "$TASK_REPOSITORY" \
  --dinov3-runtime-root "$DINOV3_RUNTIME_ROOT" \
  --dinov3-snapshot-root "$DINOV3_SNAPSHOT_ROOT" \
  --run-root "$RUN_ROOT" \
  --device "$DEVICE" \
  >"$PLAN_STDOUT" 2>"$PLAN_STDERR"
PLAN_RC=$?
printf 'PLAN_EXIT_CODE=%s\nPLAN_STDOUT=%s\nPLAN_STDERR=%s\n' "$PLAN_RC" "$PLAN_STDOUT" "$PLAN_STDERR"
```

Proceed only when `PLAN_RC=0`. The repository and run root are persistent and
must remain at the exact planned paths. The first retained chunk is each
encoder's smoke; do not launch a throwaway model smoke.

## 3. Generate the established exclusive-or-stable-shared GPU gate

```bash
GPU_GATE_SCRIPT="$LAUNCH_LOG_ROOT/gpu-gate-$RUN_TOKEN.sh"
cat >"$GPU_GATE_SCRIPT" <<'VEA_GPU_GATE'
#!/usr/bin/env bash
set -u
QUERY_FAILURES=0
while true; do
  FREE_VALUES=()
  UTIL_VALUES=()
  PID_SETS=()
  SAMPLE_FAILED=0
  SAMPLE=0
  while [ "$SAMPLE" -lt 3 ]; do
    MEMORY_UTIL="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"
    QUERY_RC=$?
    COMPUTE_ROWS="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"
    COMPUTE_RC=$?
    if [ "$QUERY_RC" -ne 0 ] || [ "$COMPUTE_RC" -ne 0 ] || [ -z "$MEMORY_UTIL" ]; then
      SAMPLE_FAILED=1
      break
    fi
    FREE_VALUES+=("$(printf '%s' "$MEMORY_UTIL" | awk -F, '{gsub(/ /,"",$1); print $1}')")
    UTIL_VALUES+=("$(printf '%s' "$MEMORY_UTIL" | awk -F, '{gsub(/ /,"",$2); print $2}')")
    PID_SETS+=("$(printf '%s\n' "$COMPUTE_ROWS" | awk 'NF {gsub(/ /,""); print}' | sort -n | paste -sd, -)")
    SAMPLE=$((SAMPLE + 1))
    if [ "$SAMPLE" -lt 3 ]; then sleep 30; fi
  done
  if [ "$SAMPLE_FAILED" -eq 1 ]; then
    QUERY_FAILURES=$((QUERY_FAILURES + 1))
    if [ "$QUERY_FAILURES" -ge 3 ]; then
      printf 'GPU_GATE=QUERY_FAILURE\n'
      exit 75
    fi
    sleep 15
    continue
  fi
  QUERY_FAILURES=0
  MIN_FREE="$(printf '%s\n' "${FREE_VALUES[@]}" | sort -n | head -1)"
  MAX_FREE="$(printf '%s\n' "${FREE_VALUES[@]}" | sort -n | tail -1)"
  MEAN_UTIL=$(((UTIL_VALUES[0] + UTIL_VALUES[1] + UTIL_VALUES[2]) / 3))
  FINAL_UTIL="${UTIL_VALUES[2]}"
  STABLE_PIDS=0
  if [ "${PID_SETS[0]}" = "${PID_SETS[1]}" ] && [ "${PID_SETS[1]}" = "${PID_SETS[2]}" ]; then STABLE_PIDS=1; fi
  if [ "$STABLE_PIDS" -eq 1 ] && [ -z "${PID_SETS[2]}" ] && [ "$MIN_FREE" -ge 24000 ]; then
    printf 'GPU_GATE=EXCLUSIVE\n'
    exit 0
  fi
  CURRENT_USER_OWNS_PID=0
  IFS=',' read -r -a PIDS <<<"${PID_SETS[2]}"
  for PID in "${PIDS[@]}"; do
    if [ -n "$PID" ] && [ "$(ps -o user= -p "$PID" 2>/dev/null | awk '{$1=$1};1')" = "$(id -un)" ]; then
      CURRENT_USER_OWNS_PID=1
    fi
  done
  SWING=$((MAX_FREE - MIN_FREE))
  if [ "$STABLE_PIDS" -eq 1 ] && [ "$MIN_FREE" -ge 12000 ] && \
     [ "$CURRENT_USER_OWNS_PID" -eq 0 ] && [ "$SWING" -le 512 ] && \
     [ "$MEAN_UTIL" -le 50 ] && [ "$FINAL_UTIL" -le 60 ]; then
    printf 'GPU_GATE=STABLE_SHARED\n'
    exit 0
  fi
  printf 'GPU_GATE=BUSY min_free=%s swing=%s mean_util=%s final_util=%s pids=%s\n' \
    "$MIN_FREE" "$SWING" "$MEAN_UTIL" "$FINAL_UTIL" "${PID_SETS[2]}"
  sleep 30
done
VEA_GPU_GATE
chmod 700 "$GPU_GATE_SCRIPT"
```

## 4. Generate and detach the resource-only resume supervisor

```bash
SUPERVISOR="$LAUNCH_LOG_ROOT/supervisor-$RUN_TOKEN.sh"
SUPERVISOR_LOG="$LAUNCH_LOG_ROOT/supervisor-$RUN_TOKEN.log"
WORKER_LOG="$LAUNCH_LOG_ROOT/worker-$RUN_TOKEN.log"
export GPU_GATE_SCRIPT SUPERVISOR_LOG WORKER_LOG
cat >"$SUPERVISOR" <<'VEA_T001D_SUPERVISOR'
#!/usr/bin/env bash
set -u -o pipefail
COMMON=(
  --config "$TASK_REPOSITORY/configs/vlm_compat/p2_universal_anchor_transport_v0.json"
  --manifest "$TASK_REPOSITORY/data/manifests/p1_1000.jsonl"
  --dataset-root "$DATASET_ROOT"
  --cache-root "$CACHE_ROOT"
  --repository-root "$TASK_REPOSITORY"
  --dinov3-runtime-root "$DINOV3_RUNTIME_ROOT"
  --dinov3-snapshot-root "$DINOV3_SNAPSHOT_ROOT"
  --run-root "$RUN_ROOT"
  --device "$DEVICE"
)
ATTEMPT=0
RESOURCE_RESUMES=0
COMMAND=run
WORKER_EXIT_CODE=1
TEE_EXIT_CODE=1
VERIFY_EXIT_CODE=1
FINALIZE_EXIT_CODE=1
VEA_P2_EXECUTION=FAILED
while [ "$ATTEMPT" -lt 4 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  GATE_LOG="$LAUNCH_LOG_ROOT/gate-attempt-${ATTEMPT}-$RUN_TOKEN.log"
  bash "$GPU_GATE_SCRIPT" >"$GATE_LOG" 2>&1
  GATE_RC=$?
  if [ "$GATE_RC" -ne 0 ]; then
    WORKER_EXIT_CODE=75
    TEE_EXIT_CODE=0
  else
    "$MAIN_PY" "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" "$COMMAND" "${COMMON[@]}" 2>&1 | tee -a "$WORKER_LOG"
    PIPE_CODES=("${PIPESTATUS[@]}")
    WORKER_EXIT_CODE="${PIPE_CODES[0]}"
    TEE_EXIT_CODE="${PIPE_CODES[1]}"
  fi
  VERIFY_JSON="$LAUNCH_LOG_ROOT/verify-attempt-${ATTEMPT}-$RUN_TOKEN.json"
  VERIFY_STDERR="$LAUNCH_LOG_ROOT/verify-attempt-${ATTEMPT}-$RUN_TOKEN.stderr.log"
  "$MAIN_PY" "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" verify "${COMMON[@]}" \
    >"$VERIFY_JSON" 2>"$VERIFY_STDERR"
  VERIFY_EXIT_CODE=$?
  if [ "$WORKER_EXIT_CODE" -eq 0 ] && [ "$TEE_EXIT_CODE" -eq 0 ] && \
     [ "$VERIFY_EXIT_CODE" -eq 0 ] && grep -q '"complete_units":true' "$VERIFY_JSON"; then
    VEA_P2_EXECUTION=PASS
    break
  fi
  if [ "$WORKER_EXIT_CODE" -eq 75 ] && [ "$RESOURCE_RESUMES" -lt 3 ]; then
    RESOURCE_RESUMES=$((RESOURCE_RESUMES + 1))
    COMMAND=resume
    continue
  fi
  if [ "$WORKER_EXIT_CODE" -eq 75 ] || [ "$WORKER_EXIT_CODE" -eq 130 ]; then
    VEA_P2_EXECUTION=PARTIAL
  else
    VEA_P2_EXECUTION=FAILED
  fi
  break
done
if [ "$VEA_P2_EXECUTION" = PASS ]; then
  FINALIZE_JSON="$LAUNCH_LOG_ROOT/finalize-$RUN_TOKEN.json"
  FINALIZE_STDERR="$LAUNCH_LOG_ROOT/finalize-$RUN_TOKEN.stderr.log"
  "$MAIN_PY" "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" finalize "${COMMON[@]}" \
    >"$FINALIZE_JSON" 2>"$FINALIZE_STDERR"
  FINALIZE_EXIT_CODE=$?
  FINAL_VERIFY_JSON="$LAUNCH_LOG_ROOT/final-verify-$RUN_TOKEN.json"
  FINAL_VERIFY_STDERR="$LAUNCH_LOG_ROOT/final-verify-$RUN_TOKEN.stderr.log"
  "$MAIN_PY" "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" verify "${COMMON[@]}" \
    >"$FINAL_VERIFY_JSON" 2>"$FINAL_VERIFY_STDERR"
  FINAL_VERIFY_EXIT_CODE=$?
  if [ "$FINALIZE_EXIT_CODE" -ne 0 ] || [ "$FINAL_VERIFY_EXIT_CODE" -ne 0 ] || \
     ! grep -q '"missing_final_artifacts":\[\]' "$FINAL_VERIFY_JSON"; then
    VEA_P2_EXECUTION=FAILED
  fi
fi
STATUS_JSON="$LAUNCH_LOG_ROOT/status-$RUN_TOKEN.json"
STATUS_STDERR="$LAUNCH_LOG_ROOT/status-$RUN_TOKEN.stderr.log"
"$MAIN_PY" "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" status "${COMMON[@]}" \
  >"$STATUS_JSON" 2>"$STATUS_STDERR"
printf 'RESOURCE_RESUMES=%s\nVERIFY_EXIT_CODE=%s\nFINALIZE_EXIT_CODE=%s\n' \
  "$RESOURCE_RESUMES" "$VERIFY_EXIT_CODE" "$FINALIZE_EXIT_CODE"
printf 'EXACT_MANUAL_RESUME=%q %q resume' "$MAIN_PY" "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py"
printf ' %q' "${COMMON[@]}"
printf '\nRESULT_PATH=%s\n' "$RUN_ROOT/final/result.json"
if [ -d "$RUN_ROOT/final" ]; then sha256sum "$RUN_ROOT"/final/*; fi
printf 'WORKER_EXIT_CODE=%s\n' "$WORKER_EXIT_CODE"
printf 'TEE_EXIT_CODE=%s\n' "$TEE_EXIT_CODE"
printf 'VEA_P2_EXECUTION=%s\n' "$VEA_P2_EXECUTION"
printf 'TMUX_PARENT_SHELL=ALIVE\n'
VEA_T001D_SUPERVISOR
chmod 700 "$SUPERVISOR"

bash -n "$GPU_GATE_SCRIPT"
GPU_GATE_SYNTAX_RC=$?
bash -n "$SUPERVISOR"
SUPERVISOR_SYNTAX_RC=$?
printf 'GPU_GATE_SYNTAX_RC=%s\nSUPERVISOR_SYNTAX_RC=%s\n' "$GPU_GATE_SYNTAX_RC" "$SUPERVISOR_SYNTAX_RC"
if [ "$GPU_GATE_SYNTAX_RC" -eq 0 ] && [ "$SUPERVISOR_SYNTAX_RC" -eq 0 ]; then
  nohup setsid bash "$SUPERVISOR" >"$SUPERVISOR_LOG" 2>&1 < /dev/null &
  SUPERVISOR_PID=$!
else
  SUPERVISOR_PID='NOT_STARTED'
fi
printf 'SUPERVISOR_PID=%s\nSUPERVISOR_LOG=%s\nWORKER_LOG=%s\nRUN_ROOT=%s\n' \
  "$SUPERVISOR_PID" "$SUPERVISOR_LOG" "$WORKER_LOG" "$RUN_ROOT"
```

Launch only when both syntax codes are zero. The detached child survives SSH
disconnect and never retries a non-75 result. Exit `130` remains `PARTIAL` for
manual resume. Monitor the unique persistent log with
`tail -f "$SUPERVISOR_LOG"`. The child report ends with exactly:

```text
WORKER_EXIT_CODE=...
TEE_EXIT_CODE=...
VEA_P2_EXECUTION=PASS|PARTIAL|FAILED
TMUX_PARENT_SHELL=ALIVE
```
