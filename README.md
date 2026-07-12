# Vision Encoder Alignment

A research codebase for analyzing whether heterogeneous vision encoders learn structurally related representations and for determining the minimum transformation complexity required to align them.

## Current scope

The initial scope is encoder-only:

- frozen vision encoders;
- canonical image geometry;
- layer-wise global and patch feature extraction;
- CKA, Procrustes, ridge mapping, retrieval, and spatial correspondence;
- reproducible manifests and artifact validation.

An LLM or multimodal projector is not part of the current implementation phase.

## Start here

1. Read `AGENTS.md`.
2. Read `docs/INDEX.md` and `docs/PROJECT_STATUS.md`.
3. Execute tasks in `tasks/BACKLOG.md` order.
4. After each Codex task, review the generated file under `reports/runs/` before starting the next task.

## Repository state

This starter kit intentionally contains research and workflow contracts before implementation. The first task is `tasks/VEA-P0-T001-repository-audit.md`.
