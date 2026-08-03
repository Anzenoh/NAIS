# VEA-P1-T003B-B-R2 — Runtime/Snapshot Contract Repair Run Report

## Status

`PASS_PENDING_GIT_DELIVERY`

## Baseline and isolation

- baseline: `8fbc4c16c9a8a3802d1fdc1e8fb7b64738482980`
- task branch: `codex/vea-p1-t003b-b-r2`
- original dirty working tree: preserved; implementation and validation occurred in external worktrees
- network operations: baseline remote verification only; final push pending
- Apple/GPU/dataset/checkpoint/model/package/smoke/extraction operations: `0`

## Repair summary

- Apple Linux interpreter layout is `<runtime_root>/bin/python`; legacy Windows remains `venv/Scripts/python.exe`.
- `huggingface_cache_snapshot_v1` is resolved from exact cache root, model ID, and revision and checks all pinned DINO files before use.
- Stage A, extractor, invocation boundary, and worker use the shared validator.
- Apple worker success identities no longer dereference a legacy-only runtime manifest.
- approved repository commit and production-source baseline are distinct evidence fields.
- bounded `model_start`, `model_complete`, and `model_failure` markers use stderr only.
- runbook preserves a terminal fd and requires fresh evidence and smoke roots.

## Validation

- Tier 0 `py -3 -B -m py_compile`/assigned compile targets: exit `0` during implementation; final compile recorded before delivery.
- focused contract subset: `101 passed`, exit `0`.
- recovery subset: `113 passed`, exit `0`.
- assigned Tier 1 suite before final coverage additions: `431 passed, 2 expected Windows symlink-capability skips`, exit `0`.
- final assigned Tier 1 suite, whitespace check, staged allowlist, commit, and push: recorded in the compact final response because the commit cannot contain its own SHA or future push result.

## Deviations and risks

- The first external worktree inherited Windows CRLF checkout semantics. It was retained untouched after diagnosis; validation was repeated in a second LF-materialized external worktree matching Git blob identities.
- Two pre-existing Windows capability skips remain: directory symlink and file symlink creation are unavailable on the external G: filesystem.
- No Apple synchronization or runtime execution occurred. Independent acceptance remains required.

```text
READY_FOR_P1_EXTRACTION=false
```
