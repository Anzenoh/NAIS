# VEA-P2-T001D-R2 supervisor environment readiness

- Status: local offline runbook-contract validation PASS.
- Baseline: `ad86127c5e7d23c43a3abbec317c10ece4e6c5a7`.
- Branch: `codex/vea-p2-t001d-r2-supervisor-env-readiness`.
- Scope: only the persistent Apple runbook and its two contract-test files;
  scientific implementation, plan identity, checkpoint behavior, transport,
  metrics, generation, storage, and resume behavior are unchanged.
- Child closure: `RUN_TOKEN`, `GPU_INDEX`, `GPU_GATE_SCRIPT`, `MAIN_PY`, task and
  run roots, logs, dataset/cache/DINO roots, and device bindings are explicitly
  exported before the relevant child launch.
- Prelaunch: directory, reviewed clone/branch/commit/cleanliness, exact local
  snapshot/root presence, metadata-only main-runtime validation, immutable plan,
  child-script creation, and both `bash -n` checks gate `nohup setsid`.
- Resource checks: exact local TinyLLaVA, CLIP, and SigLIP2 revision snapshot
  directories, approved DINO runtime/snapshot roots, dataset root, manifest, and
  main interpreter are checked without importing a model runtime.
- Real Apple/CUDA/model/dataset/checkpoint/package activity: 0.

## Delivery evidence

`py -3.12 -m pytest -q tests/unit/test_p2_resumable_anchor_transport.py
tests/integration/test_p2_resumable_anchor_transport_contract.py`: `67 passed in
76.92s`.

Focused parsed-here-document audit:

```text
GPU gate unresolved external variables = 0
supervisor unresolved external variables = 0
RUN_TOKEN exported = true
prelaunch failure paths reaching nohup = 0
parent-shell forbidden operations = 0
```

`git diff --check`: PASS. Pending exact-path commit and push.
