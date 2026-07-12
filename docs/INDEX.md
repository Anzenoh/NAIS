# Documentation Index

This file is the navigation entry point for Codex and human reviewers.

| Document | Purpose | Read when |
|---|---|---|
| `PROJECT_CONTEXT.md` | Research motivation, objectives, non-goals, and candidate encoders | Onboarding and planning |
| `PROJECT_STATUS.md` | Latest verified phase, completed work, open decisions, and next task | Before every task |
| `RESEARCH_CONTRACT.md` | Invariants that define a valid experiment | Before extraction, metrics, or adapter work |
| `EXPERIMENT_PROTOCOL_V0.md` | Phase 0/1 scientific procedure and gates | Extraction and analysis tasks |
| `METRIC_DEFINITIONS.md` | Exact metric semantics and anti-leakage rules | Metric implementation and review |
| `VALIDATION_PLAN.md` | Software validation levels and PASS/PARTIAL/BLOCKED criteria | Every implementation task |
| `REPRODUCIBILITY.md` | Required environment, checkpoint, dataset, and run metadata | Environment and execution tasks |
| `DECISION_LOG.md` | Accepted and open architectural/research decisions | Planning or scope changes |
| `TASK_WORKFLOW.md` | Human–ChatGPT–Codex execution and review process | Before issuing tasks |
| `contracts/FEATURE_ARTIFACT_SCHEMA.md` | Canonical tensor, manifest, tap, and storage contract | Extractor and artifact work |
| `templates/TASK_TEMPLATE.md` | Template for new task specifications | Writing a Codex task |
| `templates/RUN_REPORT_TEMPLATE.md` | Mandatory implementation evidence format | Completing a task |
| `templates/REVIEW_TEMPLATE.md` | Human/ChatGPT review verdict format | Reviewing completed work |

## Minimal read sets

### Repository or environment task

- `PROJECT_STATUS.md`
- `VALIDATION_PLAN.md`
- `REPRODUCIBILITY.md`

### Feature extractor task

- `RESEARCH_CONTRACT.md`
- `EXPERIMENT_PROTOCOL_V0.md`
- `contracts/FEATURE_ARTIFACT_SCHEMA.md`
- `VALIDATION_PLAN.md`
- `REPRODUCIBILITY.md`

### Metric task

- `RESEARCH_CONTRACT.md`
- `METRIC_DEFINITIONS.md`
- `VALIDATION_PLAN.md`

### Scope or research design change

- `PROJECT_CONTEXT.md`
- `RESEARCH_CONTRACT.md`
- `DECISION_LOG.md`
- `PROJECT_STATUS.md`
