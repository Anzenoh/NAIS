# VEA-P0-T001 — Repository and Runtime Audit

## Mode

`investigation-only`

## Goal

Produce a verified baseline of the current repository and local execution environment, then propose the exact implementation plan for `VEA-P0-T002` without writing implementation code or downloading models.

## Why this task exists

Python, package, OS, Git, storage, and accelerator facts are currently unresolved. Dependency and package decisions must be based on observed evidence rather than assumptions.

## Read first

- `AGENTS.md`
- `docs/INDEX.md`
- `docs/PROJECT_STATUS.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/VALIDATION_PLAN.md`
- `docs/REPRODUCIBILITY.md`
- `docs/DECISION_LOG.md`
- `configs/models.yaml`

## Current evidence

- Primary development host is expected to be Windows.
- Local GPU is expected to be an AMD Radeon RX 6950 XT.
- CPU smoke capability is required.
- No implementation baseline has been verified.
- Exact Python and package versions are unknown.
- Network access is not approved for this task.

## Scope

### Allowed actions

- inspect repository files and Git state;
- run read-only environment commands;
- inspect available Python executables and package versions;
- inspect disk space and hardware/runtime visibility;
- identify existing code or conflicts if the repository is not empty;
- create one run report under `reports/runs/`;
- propose a concrete file tree, dependency set, validation commands, and task split for `VEA-P0-T002`.

### Allowed file changes

- `reports/runs/<timestamp>_VEA-P0-T001.md` only.

### Non-goals

- do not create `pyproject.toml`;
- do not create source or test code;
- do not install or upgrade packages;
- do not download datasets or model checkpoints;
- do not authenticate to Hugging Face or accept licenses;
- do not modify any contract, config, status, or backlog file;
- do not test DirectML, ROCm, CUDA, or model inference beyond reporting already visible environment facts.

## Required observations

Record exact commands and results for:

1. repository root and `git status --short`;
2. tracked/untracked top-level structure;
3. OS and architecture;
4. available Python launch commands and versions;
5. `pip` association with the selected Python;
6. installed versions, if present, of PyTorch, Transformers, NumPy, Pillow, PyYAML, Safetensors, Pytest, Ruff, and Mypy;
7. whether PyTorch reports CUDA, MPS, or another explicitly supported backend;
8. CPU, RAM, GPU, and free disk space using commands available on the host;
9. any path, encoding, or Windows-specific risks;
10. whether files in this starter kit are intact and internally consistent at a high level.

Do not expose secrets, full environment-variable dumps, usernames, or unrelated personal paths. Redact user-specific path prefixes in the report when feasible.

## Deliverables

Create one report from `docs/templates/RUN_REPORT_TEMPLATE.md` containing:

1. verified environment table;
2. repository inventory and conflict assessment;
3. proposed Python version range;
4. proposed minimal runtime and development dependencies with reasons;
5. proposed package/file tree for `VEA-P0-T002`;
6. proposed exact validation commands for `VEA-P0-T002`;
7. blockers and decisions requiring the user's approval;
8. a recommended next Codex task boundary.

## Validation

### Required

- confirm that only the run report changed;
- run `git diff --check` if Git is available;
- include `git status --short` after writing the report;
- verify no network download or package installation occurred.

### Optional

- calculate checksums of protected starter-kit files if a standard local command is available.

## Stop conditions

Return `BLOCKED` rather than improvising if:

- the repository root cannot be established;
- required starter-kit files are missing;
- read-only environment commands are prohibited;
- writing the run report would overwrite an existing file;
- unrelated working-tree changes make the repository state impossible to interpret.

## Required final response

Use the format required by `AGENTS.md`. Do not implement the proposed plan in this task.
