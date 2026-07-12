# Task Backlog

Tasks are executed sequentially unless an explicit review approves parallel work.

## Phase 0 — Repository and contract foundation

| Task ID | Goal | Dependency | Status |
|---|---|---|---|
| VEA-P0-T001 | Audit repository and runtime; produce an evidence-based implementation plan | starter kit | READY |
| VEA-P0-T002 | Create Python package, dependency baseline, config validation, and mock artifact contract tests | T001 PASS | BLOCKED |
| VEA-P0-T003 | Implement mathematically verified CKA, Procrustes, ridge, and split utilities | T002 PASS | BLOCKED |
| VEA-P0-T004 | Implement unified extractor interface and mock extractor | T002 PASS | BLOCKED |
| VEA-P0-T005 | Add and validate CLIP real extractor | T004 PASS | BLOCKED |
| VEA-P0-T006 | Add and validate SigLIP2 real extractor | T005 PASS | BLOCKED |
| VEA-P0-T007 | Add and validate DINOv3 real extractor | T006 PASS and user access | BLOCKED |
| VEA-P0-T008 | Execute and validate combined ten-image CPU smoke | T005–T007 PASS and manifest | BLOCKED |

## Phase 1 — Controlled analysis

| Task ID | Goal | Dependency | Status |
|---|---|---|---|
| VEA-P1-T001 | Create immutable 1,000-image manifest and split | P0 gate PASS | BLOCKED |
| VEA-P1-T002 | Pin environment and checkpoint revisions | P0 gate PASS | BLOCKED |
| VEA-P1-T003 | Execute full feature extraction | P1-T001/T002 PASS | BLOCKED |
| VEA-P1-T004 | Compute global layer-wise CKA and controls | P1-T003 PASS | BLOCKED |
| VEA-P1-T005 | Fit and evaluate pairwise Procrustes/ridge mappings | P1-T004 PASS | BLOCKED |
| VEA-P1-T006 | Compute patch and spatial correspondence | P1-T005 PASS | BLOCKED |
| VEA-P1-T007 | Produce Phase 1 analysis report and next-method recommendation | P1-T004–T006 PASS | BLOCKED |
