# VEA-P2-T001D-R3 main-lane DINO orchestration

- Status: Tier 0–1 fake/CPU validation PASS.
- Baseline: `a3d4cc57c0f2edde8639d718a3f2ef0c3dfe4275`.
- Branch: `codex/vea-p2-t001d-r3-main-lane-dino-orchestration`.
- Correction: all real stage orchestration commands derive the approved
  `/home/chltjsdh/vea/runtime/p1_apple/main/bin/python3` from the immutable plan.
- Stage 01 retains a sanitized `PYTHONPATH` without the TinyLLaVA overlay or
  paths under the main runtime root. Main-interpreter native site packages remain
  available for canonical JPEG decoding and NumPy orchestration.
- DINO forward isolation remains in the protected extractor boundary: one
  isolated DINO interpreter worker, model load, and forward per pending chunk.
- Preflight uses standard-library module-spec inspection for `PIL` and `numpy`
  with the overlay excluded, plus accepted metadata-only package checks,
  without importing either module.
- Scientific formulas, identities, 2,345-unit inventory, storage, checkpoint,
  generation, metric, and resume contracts are unchanged.
- Real Apple/CUDA/model/dataset/checkpoint/package activity: 0.

## R2 failed-run disposition

- Plan valid.
- Completed units: `0`.
- Failure evidence retained.
- Not eligible for resume under the R3 commit.
- No reusable scientific unit exists.

R3 Apple validation must create a new plan ID and a new persistent run root.

## Delivery evidence

- Assigned pytest: `134 passed in 76.64s` on the final rerun.
- Focused first-chunk audit: `3 passed, 79 deselected in 0.22s`.
- Stage 01 orchestration interpreter: `MAIN_PY`.
- Stage 01 main overlay in inherited `PYTHONPATH`: `false`.
- `PIL` and `numpy` module specs: `available` under injected import-light
  preflight; missing-module cases fail before plan/unit execution.
- DINO model boundary: isolated runtime.
- First chunk completed atomically: `true`.
- First chunk re-executed after resume: `false`; its artifact bytes and marker
  timestamps remained unchanged.
- Resume accounting for the one remaining fake chunk: one model load, one
  extractor-worker invocation, one stage-worker invocation.
- Assigned compile, both CLI-help commands, and `git diff --check`: PASS.

Pending exact-path commit and push.
