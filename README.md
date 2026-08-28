# Vision Encoder Alignment

A research codebase for reproducible comparison and functional alignment of frozen vision-encoder representations.

## Current scope

Phase 0 and Phase 1 established the encoder-only representation-analysis
foundation: shared image geometry, model-specific normalization, validated
feature banks, image-level splits, and held-out alignment metrics.

Current work is Phase 2 query-conditioned encoder-to-VLM adapter compatibility.
An adapter may be fitted to transform a source feature from the same Query image
into the TinyLLaVA native CLIP pre-connector feature space. Source encoders,
target CLIP, the TinyLLaVA connector, and the TinyLLaVA LLM remain frozen.
Frozen connector and LLM generation evaluation is permitted only under the
Phase 2 contracts.

## Start here

1. Read `AGENTS.md`.
2. Read `docs/INDEX.md` and `docs/PROJECT_STATUS.md`.
3. Execute tasks in `tasks/BACKLOG.md` order.
4. After each Codex task, review the generated file under `reports/runs/` before starting the next task.

## Repository state

See `docs/PROJECT_STATUS.md` for the accepted state and next reviewed task.
