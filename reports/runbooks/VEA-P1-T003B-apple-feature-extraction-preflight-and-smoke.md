# VEA-P1-T003B — Apple preflight and separately approved 10-image smoke

## Authority and stop line

Use this runbook only after an independent review assigns
`VEA-P1-T003A-R1R3=ACCEPTED_PASS` and confirms
`VEA-P1-T002D-SBR1=ACCEPTED_PASS`. It applies to direct-host `apple` and not to the
school A100/Slurm system. Stage A preflight and Stage B smoke require separate
independent approvals.

```text
READY_FOR_P1_EXTRACTION=false
Do not run full extraction.
```

Do not download, install, repair, clean up, retry, resume, select a full batch size, or
access secrets. On any failure preserve all existing bytes and stop.

## 1. Dedicated tmux and user-supplied roots

```bash
tmux new-session -s vea_p1_t003b_apple
set -euo pipefail
umask 077

: "${VEA_DATASET_ROOT:?set VEA_DATASET_ROOT to the Apple COCO val2017 root}"
: "${VEA_CHECKPOINT_CACHE:?set VEA_CHECKPOINT_CACHE to the Apple checkpoint cache root}"
: "${VEA_T003B_EVIDENCE_ROOT:?set VEA_T003B_EVIDENCE_ROOT to a fresh evidence root}"
: "${VEA_R1R3_TASK_SHA256:?set from the independently accepted R1R3 package identity}"
: "${VEA_R1R3_REPORT_SHA256:?set from the independently accepted R1R3 package identity}"

REPO_ROOT=/home/chltjsdh/vea/repo
DATASET_ROOT="$VEA_DATASET_ROOT"
CACHE_ROOT="$VEA_CHECKPOINT_CACHE"
EVIDENCE_ROOT="$VEA_T003B_EVIDENCE_ROOT"
OUTPUT_PARENT=/home/chltjsdh/vea/artifacts/p1_t003
T002D_EVIDENCE_ROOT=/home/chltjsdh/vea/artifacts/p1_t002d_runtime_setup
PROFILE="$REPO_ROOT/configs/extraction_profiles/p1_1000_apple_fp32_v0.json"
R1R3_TASK="$REPO_ROOT/tasks/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md"
R1R3_REPORT="$REPO_ROOT/reports/runs/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md"
MAIN_PYTHON=/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3
DINO_PYTHON=/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated/bin/python3
MAIN_RUNTIME_ROOT=/home/chltjsdh/vea/runtime/p1_apple/main
DINO_RUNTIME_ROOT=/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated

reject_placeholder() {
  case "$1" in
    ''|*[![:print:]]*|*'<'*|*'>'*|*PLACEHOLDER*|*CHANGE_ME*|*REPLACE_ME*)
      printf '%s\n' 'empty or placeholder path rejected' >&2; return 2 ;;
  esac
}

assert_nonlink_existing() {
  path=$1
  kind=$2
  reject_placeholder "$path"
  case "$path" in /*) ;; *) printf '%s\n' 'relative path rejected' >&2; return 2;; esac
  canonical=$(realpath -e -- "$path")
  test "$canonical" = "$path"
  current=/
  old_ifs=$IFS
  IFS=/
  set -- $path
  IFS=$old_ifs
  for component do
    test -n "$component" || continue
    current="${current%/}/$component"
    test ! -L "$current"
  done
  case "$kind" in
    directory) test -d "$path" ;;
    file) test -f "$path" ;;
    executable) test -f "$path" && test -x "$path" ;;
    *) return 2 ;;
  esac
}
```

## 2. All read-only checks before evidence-root creation

The following entire block must succeed before `mkdir "$EVIDENCE_ROOT"`. The
component walk rejects every symlink ancestor and leaf on Apple; `realpath -e` must
equal the original canonical absolute spelling. Missing paths are not treated as safe.
This narrows path substitution risk but does not eliminate TOCTOU.

```bash
test "$(hostname)" = apple
assert_nonlink_existing "$REPO_ROOT" directory
assert_nonlink_existing "$REPO_ROOT/.git" directory
cd "$REPO_ROOT"
test "$(git branch --show-current)" = main
test "$(git rev-parse HEAD)" = e518a51ba83f70b42611cc4386a9f55dc9e97de3
test -z "$(git diff --cached --name-only)"
git -c core.whitespace=cr-at-eol diff --check

assert_nonlink_existing "$R1R3_TASK" file
assert_nonlink_existing "$R1R3_REPORT" file
printf '%s  %s\n' "$VEA_R1R3_TASK_SHA256" "$R1R3_TASK" | sha256sum --check --strict
printf '%s  %s\n' "$VEA_R1R3_REPORT_SHA256" "$R1R3_REPORT" | sha256sum --check --strict
grep -F 'VEA-P1-T002D-SBR1 = ACCEPTED_PASS' "$R1R3_TASK" >/dev/null

assert_nonlink_existing "$PROFILE" file
printf '%s  %s\n' 98d2dfe8e711efdb6f33ac45e87e21318ec53db8b1cc8f6a0aed789babc58cf5 "$PROFILE" | sha256sum --check --strict
while read -r expected relative; do
  assert_nonlink_existing "$REPO_ROOT/$relative" file
  printf '%s  %s\n' "$expected" "$REPO_ROOT/$relative" | sha256sum --check --strict
done <<'SOURCE_BINDINGS'
08cfdbbbd9e99120adb7c660e4c25f162f766b7516451c6a4a410f316a11defd scripts/dinov3_extraction_worker.py
fe2ba7035c8852bdd3ed44f75b01926c1987aca1f71bae94accbb6578eaec54d src/vision_encoder_alignment/dinov3_extraction_boundary.py
0ec0980c9761a73597244b03116de093c71b568cc0d8755951821b4f448eb82d src/vision_encoder_alignment/extractors/clip.py
97c781592265c9e600ba0ad257528c313c471e23c2d22f27905d2e42a872f0d1 src/vision_encoder_alignment/extractors/dinov3.py
0267d9eaa51d6c6b686339d9d9e700f782d2ec3b5c16c20d3df16e8fa17cfd5e src/vision_encoder_alignment/extractors/siglip2.py
3233ed503581af6f9e9fb54993e85783546a0ef251b02dbe943faec6724b0063 src/vision_encoder_alignment/runtime_boundary.py
SOURCE_BINDINGS

assert_nonlink_existing "$DATASET_ROOT" directory
assert_nonlink_existing "$CACHE_ROOT" directory
assert_nonlink_existing "$OUTPUT_PARENT" directory
assert_nonlink_existing "$T002D_EVIDENCE_ROOT" directory
assert_nonlink_existing "$MAIN_RUNTIME_ROOT" directory
assert_nonlink_existing "$MAIN_PYTHON" executable
assert_nonlink_existing "$DINO_RUNTIME_ROOT" directory
assert_nonlink_existing "$DINO_PYTHON" executable

reject_placeholder "$EVIDENCE_ROOT"
case "$EVIDENCE_ROOT" in /*) ;; *) exit 2;; esac
test "$(dirname -- "$EVIDENCE_ROOT")" = "$(realpath -e -- "$(dirname -- "$EVIDENCE_ROOT")")"
assert_nonlink_existing "$(dirname -- "$EVIDENCE_ROOT")" directory
test "$(basename -- "$EVIDENCE_ROOT")" != .
test "$(basename -- "$EVIDENCE_ROOT")" != ..
test ! -e "$EVIDENCE_ROOT"
test ! -L "$EVIDENCE_ROOT"

mkdir -m 700 -- "$EVIDENCE_ROOT"
```

## 3. Stage A — read-only preflight

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PIP_NO_INDEX=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_ROOT/src"

"$MAIN_PYTHON" -B "$REPO_ROOT/scripts/validate_p1_apple_runtime.py" \
  --mode runtime-live --repository-root "$REPO_ROOT" \
  --output-json "$EVIDENCE_ROOT/00-runtime-live.json"

"$MAIN_PYTHON" -B "$REPO_ROOT/scripts/inventory_p1_checkpoint_cache.py" \
  --plan "$REPO_ROOT/configs/model_snapshots/p1_three_encoder_checkpoint_plan.template.json" \
  --registry "$REPO_ROOT/configs/models.yaml" --cache-root "$CACHE_ROOT" \
  --repository-root "$REPO_ROOT" --output "$EVIDENCE_ROOT/01-checkpoint-inventory.json"

"$MAIN_PYTHON" -B "$REPO_ROOT/scripts/extract_features.py" plan \
  --profile "$PROFILE" --repository-root "$REPO_ROOT" \
  --output-json "$EVIDENCE_ROOT/02-plan.json" \
  > /dev/null

"$MAIN_PYTHON" -B "$REPO_ROOT/scripts/extract_features.py" preflight \
  --profile "$PROFILE" --repository-root "$REPO_ROOT" \
  --dataset-root "$DATASET_ROOT" --cache-root "$CACHE_ROOT" \
  --output-json "$EVIDENCE_ROOT/03-preflight.json" \
  > /dev/null

printf '%s\n' 'READY_FOR_P1_EXTRACTION=false' > "$EVIDENCE_ROOT/04-readiness.txt"
(cd "$EVIDENCE_ROOT" && sha256sum \
  00-runtime-live.json 01-checkpoint-inventory.json 02-plan.json \
  03-preflight.json 04-readiness.txt) > "$EVIDENCE_ROOT/05-preflight-manifest.txt"
```

Exactly these six Stage A files (`00` through `05`) are retrieved. Files named
`02-plan.stdout.json` or `03-preflight.stdout.json` must not exist.

Stop here and submit Stage A evidence for independent approval. Preflight success is
not smoke approval.

## 4. Stage B — separately approved 10-image smoke

Continue only after explicit independent approval of all Stage A evidence. Use one fresh
canonical ID. The smoke is the first 10 ordered images, batch size 1, sequential
CLIP/SigLIP2/DINOv3 lanes, and FP32 only.

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_p1_1000_three_encoder_e518a51"
case "$RUN_ID" in ????????T??????Z_p1_1000_three_encoder_[0-9a-f]*) ;; *) exit 2;; esac
test ! -e "$OUTPUT_PARENT/$RUN_ID"
test ! -L "$OUTPUT_PARENT/$RUN_ID"

"$MAIN_PYTHON" -B "$REPO_ROOT/scripts/extract_features.py" run \
  --profile "$PROFILE" --repository-root "$REPO_ROOT" \
  --dataset-root "$DATASET_ROOT" --cache-root "$CACHE_ROOT" \
  --output-parent "$OUTPUT_PARENT" --run-id "$RUN_ID" \
  --mode smoke --batch-size 1 > "$EVIDENCE_ROOT/10-smoke-run.json"

"$MAIN_PYTHON" -B "$REPO_ROOT/scripts/validate_p1_feature_extraction.py" \
  --profile "$PROFILE" --run-root "$OUTPUT_PARENT/$RUN_ID" \
  > "$EVIDENCE_ROOT/11-smoke-validation.json"

test -f "$OUTPUT_PARENT/$RUN_ID/run_state.json"
test -f "$OUTPUT_PARENT/$RUN_ID/dataset_manifest.snapshot.jsonl"
test -f "$OUTPUT_PARENT/$RUN_ID/environment.json"
test -f "$OUTPUT_PARENT/$RUN_ID/extraction_plan.json"
test -f "$OUTPUT_PARENT/$RUN_ID/validation/artifact_validation.json"
test -f "$OUTPUT_PARENT/$RUN_ID/run_manifest.json"
for model in clip_vit_b16 siglip2_base_p16_224 dinov3_vit_b16; do
  test -f "$OUTPUT_PARENT/$RUN_ID/models/$model/execution_metrics.json"
  test -f "$OUTPUT_PARENT/$RUN_ID/models/$model/model_manifest.json"
done
test -z "$(find "$OUTPUT_PARENT/$RUN_ID/models" -type d -name shards -print -quit)"
```

`run_manifest.json` must be the final run artifact. Do not run `resume` automatically.
If execution fails, preserve `failures/attempt-*.json`, every partial pair, and all
finalization residue, then stop for independent review.

## 5. Final evidence and body-free review package

```bash
find "$OUTPUT_PARENT/$RUN_ID" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$EVIDENCE_ROOT/12-run-file-sha256.txt"

find "$OUTPUT_PARENT/$RUN_ID" -type f ! -name '*.safetensors' -print0 | sort -z | \
  tar --null -T - -czf "$EVIDENCE_ROOT/13-review-bundle-no-tensors.tar.gz"

printf '%s\n' 'READY_FOR_P1_EXTRACTION=false' > "$EVIDENCE_ROOT/14-readiness.txt"
(cd "$EVIDENCE_ROOT" && sha256sum 10-smoke-run.json 11-smoke-validation.json \
  12-run-file-sha256.txt 13-review-bundle-no-tensors.tar.gz 14-readiness.txt) \
  > "$EVIDENCE_ROOT/15-smoke-evidence-manifest.txt"
```

Retrieve Stage A files `00`–`05`, Stage B files `10`–`15`, and these small run
artifacts only: `run_state.json`, `dataset_manifest.snapshot.jsonl`, `environment.json`,
`extraction_plan.json`, `validation/artifact_validation.json`, `run_manifest.json`,
`models/*/execution_metrics.json`, `models/*/model_manifest.json`,
`models/*/shard-*.json`, and `failures/attempt-*.json` when present.

Do not retrieve dataset, checkpoint, model, or feature tensor bodies. Stop immediately
after retrieval. Do not clean up, retry, resume, choose a full batch size, or run the
1,000-image extraction.

```text
READY_FOR_P1_EXTRACTION=false
```
