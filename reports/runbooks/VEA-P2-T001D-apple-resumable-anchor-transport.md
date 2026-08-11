# VEA-P2-T001D Apple resumable anchor-transport runbook

Run this only after independent review of the pushed T001D commit. Start in the
user's existing tmux parent shell. The parent section deliberately uses no
`set -e`, shell replacement, error/return traps, tmux creation, or tmux attach.
It does not download or install packages and never removes an earlier run root.

## 1. Parent-shell bindings and fresh reviewed clone

```bash
REMOTE_URL='<reviewed repository URL>'
REVIEWED_COMMIT='<40-character reviewed T001D commit>'
DATASET_ROOT='<existing immutable p1_1000 dataset root>'
CACHE_ROOT='<existing offline Hugging Face cache root>'
DINOV3_RUNTIME_ROOT='<existing approved DINOv3 runtime root>'
DINOV3_SNAPSHOT_ROOT='<existing materialized DINOv3 snapshot root>'
DEVICE='cuda:0'

TASK_REPOSITORY="$(mktemp -d "${TMPDIR:-/tmp}/vea-p2-t001d-repo.XXXXXXXX")"
RUN_PARENT='<existing persistent evidence parent>'
RUN_ROOT="${RUN_PARENT}/VEA-P2-T001D-$(date -u +%Y%m%dT%H%M%SZ)-${REVIEWED_COMMIT}"

git clone --no-checkout "$REMOTE_URL" "$TASK_REPOSITORY"
CLONE_RC=$?
if [ "$CLONE_RC" -eq 0 ]; then
  git -C "$TASK_REPOSITORY" checkout --detach "$REVIEWED_COMMIT"
  CHECKOUT_RC=$?
else
  CHECKOUT_RC=1
fi
OBSERVED_COMMIT="$(git -C "$TASK_REPOSITORY" rev-parse HEAD 2>/dev/null)"
if [ "$CHECKOUT_RC" -ne 0 ] || [ "$OBSERVED_COMMIT" != "$REVIEWED_COMMIT" ]; then
  printf 'VEA_P2_PREFLIGHT=FAILED reviewed commit mismatch\n'
fi
```

Proceed only when the clone and exact-commit comparison both pass. Create a new
run root; never reuse the path of an earlier plan. `plan` performs only Git,
JSON, SHA-256, dataset-file, path, and free-space checks. It imports no model
runtime, Torch, Pillow, Safetensors, Transformers, or CUDA library.

```bash
mkdir -p "$RUN_PARENT"
python3 "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" plan \
  --config "$TASK_REPOSITORY/configs/vlm_compat/p2_universal_anchor_transport_v0.json" \
  --manifest "$TASK_REPOSITORY/data/manifests/p1_1000.jsonl" \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --repository-root "$TASK_REPOSITORY" \
  --dinov3-runtime-root "$DINOV3_RUNTIME_ROOT" \
  --dinov3-snapshot-root "$DINOV3_SNAPSHOT_ROOT" \
  --run-root "$RUN_ROOT" \
  --device "$DEVICE" \
  >"${RUN_PARENT}/t001d-plan.stdout.log" \
  2>"${RUN_PARENT}/t001d-plan.stderr.log"
PLAN_RC=$?
printf 'PLAN_EXIT_CODE=%s\n' "$PLAN_RC"
```

Proceed only when `PLAN_RC=0`. The retained first chunk of each encoder is the
smoke; do not run a separate model smoke.

## 2. Established GPU gate

Use the site's previously approved exclusive-or-stable-shared gate command. It
must return zero only after the same GPU is acceptable in three consecutive
checks and must not import Torch or initialize CUDA. Bind it here:

```bash
GPU_GATE_COMMAND='<existing approved gate command and fixed GPU selector>'
```

Do not replace this with a new heuristic. `nvidia-smi`/NVML observation by the
approved gate is pre-runtime metadata inspection; the coordinator starts only
after the gate returns zero.

## 3. Detached supervisor with bounded resource resumes

The following child script owns the real run. It retries only structured exit
code `75`, re-runs the approved GPU gate before each retry, verifies after every
interrupted attempt, and permits at most three automatic resource resumes.

```bash
SUPERVISOR="$RUN_PARENT/VEA-P2-T001D-supervisor-${REVIEWED_COMMIT}.sh"
SUPERVISOR_LOG="$RUN_PARENT/VEA-P2-T001D-supervisor-${REVIEWED_COMMIT}.log"
WORKER_LOG="$RUN_PARENT/VEA-P2-T001D-worker-${REVIEWED_COMMIT}.log"

cat >"$SUPERVISOR" <<'VEA_T001D_SUPERVISOR'
#!/usr/bin/env bash
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
TEE_EXIT_CODE=0
VEA_P2_EXECUTION=FAILED

while [ "$ATTEMPT" -lt 4 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  bash -lc "$GPU_GATE_COMMAND"
  GATE_RC=$?
  if [ "$GATE_RC" -ne 0 ]; then
    WORKER_EXIT_CODE=75
  else
    python3 "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" "$COMMAND" "${COMMON[@]}" \
      > >(tee -a "$WORKER_LOG") 2> >(tee -a "$WORKER_LOG" >&2)
    WORKER_EXIT_CODE=$?
    TEE_EXIT_CODE=0
  fi

  python3 "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" verify "${COMMON[@]}" \
    >"$RUN_ROOT/logs/verify-attempt-${ATTEMPT}.json" \
    2>"$RUN_ROOT/logs/verify-attempt-${ATTEMPT}.stderr.log"
  VERIFY_EXIT_CODE=$?

  if [ "$WORKER_EXIT_CODE" -eq 0 ] && \
     grep -q '"complete_units":true' "$RUN_ROOT/logs/verify-attempt-${ATTEMPT}.json"; then
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
  python3 "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" finalize "${COMMON[@]}" \
    >"$RUN_ROOT/logs/finalize.stdout.json" \
    2>"$RUN_ROOT/logs/finalize.stderr.log"
  FINALIZE_RC=$?
  if [ "$FINALIZE_RC" -ne 0 ]; then
    VEA_P2_EXECUTION=FAILED
    WORKER_EXIT_CODE=$FINALIZE_RC
  fi
fi

python3 "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py" status "${COMMON[@]}" \
  >"$RUN_ROOT/logs/final-status.json" 2>"$RUN_ROOT/logs/final-status.stderr.log"
if [ -f "$RUN_ROOT/final/result.json" ]; then
  sha256sum "$RUN_ROOT"/final/*
  printf 'RESULT_PATH=%s\n' "$RUN_ROOT/final/result.json"
fi
printf 'RESOURCE_RESUMES=%s\n' "$RESOURCE_RESUMES"
printf 'EXACT_RESUME_COMMAND=python3 %q resume' "$TASK_REPOSITORY/scripts/run_resumable_anchor_transport.py"
printf ' %q' "${COMMON[@]}"
printf '\n'
printf 'WORKER_EXIT_CODE=%s\n' "$WORKER_EXIT_CODE"
printf 'TEE_EXIT_CODE=%s\n' "$TEE_EXIT_CODE"
printf 'VEA_P2_EXECUTION=%s\n' "$VEA_P2_EXECUTION"
printf 'TMUX_PARENT_SHELL=ALIVE\n'
VEA_T001D_SUPERVISOR

chmod 700 "$SUPERVISOR"
export TASK_REPOSITORY RUN_ROOT DATASET_ROOT CACHE_ROOT DINOV3_RUNTIME_ROOT DINOV3_SNAPSHOT_ROOT DEVICE
export GPU_GATE_COMMAND WORKER_LOG
setsid bash "$SUPERVISOR" >"$SUPERVISOR_LOG" 2>&1 < /dev/null &
SUPERVISOR_PID=$!
printf 'SUPERVISOR_PID=%s\nSUPERVISOR_LOG=%s\nRUN_ROOT=%s\n' "$SUPERVISOR_PID" "$SUPERVISOR_LOG" "$RUN_ROOT"
```

The detached child survives an SSH disconnect. From the same existing tmux
parent, monitor with `tail -f "$SUPERVISOR_LOG"`. A non-75 failure is never
retried. Inspect `final/status`, aggregate rows, hashes, and the exact resume
command printed at the end. The supervisor always terminates its report with:

```text
WORKER_EXIT_CODE=...
TEE_EXIT_CODE=...
VEA_P2_EXECUTION=PASS|PARTIAL|FAILED
TMUX_PARENT_SHELL=ALIVE
```
