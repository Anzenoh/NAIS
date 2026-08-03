# Phase 1 Full Feature Extraction Contract

## Scope and readiness

This contract defines deterministic, offline, encoder-only extraction for the accepted
Apple RTX 5090 runtime. It does not authorize a dataset transfer, checkpoint download,
model execution, smoke run, or full run. T003B must independently approve preflight
before one ten-image smoke, and a later task must accept that smoke before a full run.
Every artifact keeps `ready_for_p1_extraction=false`.

The authoritative profile is
`configs/extraction_profiles/p1_1000_apple_fp32_v0.json`. Its complete production graph
is exact: dataset and checkpoint-plan identities; model IDs, revisions, lane assignments,
and order; geometry and tensor contract; runtime interpreters, profile paths/IDs/hashes;
all six source-binding paths/hashes; output policy; and canonical run-ID regex. A
well-formed profile with any changed graph value is rejected.

JSON artifacts are canonical UTF-8 without BOM: sorted compact keys, finite values, and
one final LF. Duplicate or unknown fields, credentials, machine-local Windows paths,
and unresolved identities are invalid.

## Immutable scientific inputs

Manifest rows are consumed in `selection_index` order without sorting or sampling.
Every image is a regular non-link file below the explicit dataset root and must match
both source-byte and decoded-RGB SHA-256. Shared geometry is RGB conversion, Pillow
bicubic shorter-side resize to 224, and deterministic 224 by 224 center crop; numeric
normalization remains encoder-specific.

The immutable model revisions are CLIP
`57c216476eefef5ab752ec549e440a49ae4ae5f3`, SigLIP2
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, and DINOv3
`5931719e67bbdb9737e363e781fb0c67687896bc`. Encoders remain frozen in inference mode.
Compute and storage are FP32. Patch geometry is 14 by 14, 196 patches, hidden size 768.
Global means cover embedding output, blocks 01 through 12, and final norm. Patch tensors
are retained only at blocks 03, 06, 09, and 12.

## Runtime and child import policy

Execution is sequential: CLIP and SigLIP2 use
`/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3`; DINOv3 uses
`/home/chltjsdh/vea/runtime/p1_apple/dinov3_isolated/bin/python`. Children are launched
as argument arrays with `-B`, `shell=False`, and an allowlisted offline environment that
sets repository `src` as `PYTHONPATH`. `-I` is forbidden because isolated mode ignores
that explicit import path. DINOv3 remains in its isolated lane. Network fallback, CPU
fallback, autocast, mixed precision, training, and checkpoint mutation are forbidden.

Every repository, dataset root, cache root, runtime interpreter/runtime root, output parent,
evidence parent, and run-root path is supplied as a canonical absolute path. Validation
walks the original path from its filesystem root through every existing component with
`lstat`-equivalent observations and rejects symbolic links, junctions, and reparse
points at either ancestors or the leaf, except for the narrow snapshot-leaf rule below.
Resolution must equal the original canonical
absolute path exactly. A missing path is never treated as safe via non-strict
resolution. These checks narrow path-substitution risk; they do not claim to eliminate
TOCTOU, so immutable publication still uses atomic no-clobber creation.

Hugging Face cache inspection has one narrow exception: a checkpoint snapshot leaf may
be a file symlink only when strict resolution remains below the same validated cache
root and the resolved target is a regular file. Cache-root, model, `snapshots`, snapshot,
and every other directory component remain non-link canonical directories. Directory,
broken, escaping, device, socket, and other non-regular link targets are rejected.
Inventory and extraction preflight share this bounded inspection authority; a snapshot
with no accepted regular leaf is a local resource failure (stable exit `4`).

Smoke mode is exactly the first ten ordered rows at batch size 1. Full mode is all 1,000
rows and requires a separately approved positive batch size. The canonical run ID is
`YYYYMMDDTHHMMSSZ_p1_1000_three_encoder_<7-to-12-lowercase-hex-git-prefix>`; suffixes
such as `_smoke10` are invalid.

## Exact artifact layout

The run root is a fresh directory under `/home/chltjsdh/vea/artifacts/p1_t003`:

```text
run_state.json
dataset_manifest.snapshot.jsonl
environment.json
extraction_plan.json
models/<model_key>/execution_metrics.json
models/<model_key>/model_manifest.json
models/<model_key>/shard-00000.safetensors
models/<model_key>/shard-00000.json
failures/attempt-0001.json                 # only after a failed attempt
validation/artifact_validation.json
run_manifest.json                          # always published last
```

There is no nested `shards/` directory. Each shard contains at most 25 consecutive
images. Sidecars bind run/model/lane/revision, ordered image IDs, boundaries, shapes,
dtype, tensor-byte hashes, and Safetensors SHA-256. Each model's metrics record image
count, elapsed seconds, and peak CUDA allocated/reserved bytes.

## Immutable no-clobber publication

Immutable artifacts are written to a unique sibling temporary file, flushed and
`fsync`ed, then published with `os.link(temporary, final)`. This is atomic
create-if-absent: a competing destination wins and is never overwritten. `os.replace`,
delete-and-retry, merge, truncation, and overwrite fallback are forbidden. If hard-link
publication is unavailable, execution fails closed. Temporary residue is removed only
when it is task-owned. `run_state.json` is an immutable initial control document in this
contract; progress is derived from validated inventory rather than rewriting it.

`run_manifest.json` is checked for collision before finalization and is published only
after all controls, shards, metrics, model manifests, and
`validation/artifact_validation.json` validate. It is the last run artifact.

## Failure evidence and resume

A failure after immutable `run_state.json` publication appends at most one next
canonical `failures/attempt-NNNN.json`. Stages are `worker_launch`, `worker_process`,
`worker_artifact_reconciliation`, `completion`, and `model_free_validation`. Worker
stages bind the exact model, lane, shard, and image range. Completion and model-free
validation use an explicit orchestration scope with those fields null rather than
inventing an active shard. Every record uses repository-defined stable exit code `5`.
An optional raw worker return code is a separate bounded integer; negative signal-style,
ordinary nonzero, and greater-than-255 values never replace the stable code.
Tracebacks, credentials, absolute data/cache paths, and mutable free-form dumps are not
stored. Failure records are preserved by resume and completion.

The evidence writer does not run strict resume reconciliation first: valid completed
pairs are observed without mutating them, while a tensor-only or sidecar-only current
pair is preserved as failure residue. Evidence-publication failure does not authorize
overwrite, deletion, cleanup, or a second fallback publication and cannot replace the
original failure classification. A pre-initialization failure, before `run_state.json`
exists, is represented only by the CLI error JSON; no fictitious run-local evidence is
created.

Resume is explicit and only accepts an incomplete run whose run state, manifest
snapshot, environment, plan, profile, source graph, mode, batch size, run ID, ordered
image IDs, and shard plan are exact. It validates each consecutive shard and any metrics
before continuing at the first missing shard. Nested directories, partial pairs,
unexpected files, out-of-order shards, early metrics, corrupt data, or an existing
complete manifest fail closed. Existing valid artifacts are never rewritten.
Model manifests or validation-directory residue from interrupted finalization blocks
automatic resume and requires independent manual review. If all shards exist but the
trustworthy whole-model execution metrics do not, resume fails closed rather than
publishing a short no-work invocation as whole-model metrics.

## Model-free validation and stable exits

The independent validator loads no model. It validates exact file inventory, canonical
JSON, immutable control hashes, model/revision/lane bindings, image order, shard
boundaries, checksums, exact tensor keys, FP32 shapes, finite values, patch/global-mean
agreement, metrics, preserved failure evidence, model manifests, validation evidence,
and last-published run manifest. Incomplete runs fail by default;
`--allow-incomplete` reports only validated progress and false readiness.

Stable exits are: `0` complete/valid; `2` not ready; `3` contract or identity error;
`4` local resource unavailable; `5` worker/runtime failure; `6` artifact validation
failure. Windows symlink limitations are an authoring-test normalization only and do
not weaken the Apple production execution contract.

## T003B authorization boundary

T003B Stage A is read-only preflight: runtime-live, the exact 1,000-row dataset and
decoded-RGB check, checkpoint inventory, one canonical plan JSON, one canonical
preflight JSON, false readiness, hashes, and stop. The evidence root is created only
after all repository, predecessor, runtime, and all-component path checks pass. Stage B
is a separately approved first-ten-image, batch-one, FP32 smoke followed by model-free
validation and another stop. Neither stage authorizes full extraction, full-batch-size
selection, cleanup, automatic retry, or automatic resume.
