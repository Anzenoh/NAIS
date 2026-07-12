# AGENTS.md — Vision Encoder Alignment Research

## 1. Repository purpose

This repository implements reproducible, encoder-only experiments for comparing and aligning hidden representations from heterogeneous vision encoders.

The current research phase is **representation extraction and analysis**. An LLM, multimodal projector, instruction tuning, and end-to-end MLLM training are outside the current phase unless a later approved task explicitly changes the research contract.

## 2. Instruction and context loading

Before changing files, do the following in order:

1. Read the active task specification under `tasks/`.
2. Read `docs/INDEX.md`.
3. Read every document listed in the task's `Read first` section.
4. Inspect the nearest implementation and tests relevant to the task.
5. Check `docs/PROJECT_STATUS.md` for the latest verified state.

Do not indiscriminately load generated artifacts or unrelated documents. If the task conflicts with `docs/RESEARCH_CONTRACT.md`, stop and report the conflict. Do not silently reinterpret the contract.

The following files are protected and may be edited only when the active task explicitly lists them in scope:

- `AGENTS.md`
- `docs/RESEARCH_CONTRACT.md`
- `docs/METRIC_DEFINITIONS.md`
- `docs/contracts/FEATURE_ARTIFACT_SCHEMA.md`
- `configs/models.yaml`

## 3. Non-negotiable research rules

- Keep all vision encoders frozen in Phase 0 and Phase 1.
- Do not introduce an LLM, multimodal projector, caption decoder, or instruction-tuning path in Phase 0 or Phase 1.
- Compare equivalent representation stages. Never mix raw backbone hidden states with pooled, projected, merged, or LLM-aligned tokens without an explicitly named experiment.
- Use one canonical image geometry for cross-encoder spatial comparison, followed by each model's documented numeric normalization.
- Preserve image pairing and split by image ID. Never split patch rows independently across train and test.
- Never fabricate images, labels, model revisions, layer counts, token counts, metrics, command output, or successful test results.
- Runtime-discovered facts must be recorded in manifests or run reports rather than silently hard-coded.
- A software task passes when its implementation and validation contract pass. A scientific hypothesis does not have to be confirmed for the task to pass.

## 4. Network, model, data, and secrets

- Do not download datasets or model checkpoints unless the active task explicitly permits network use and names the resource.
- Do not accept a gated model license or terms on the user's behalf.
- Do not enable `trust_remote_code=True` unless the task explicitly permits it and the code path has been reviewed.
- Never commit model weights, datasets, feature tensors, cache directories, API keys, access tokens, environment files containing secrets, or machine-specific absolute paths.
- Use environment variables for credentials. Do not print credential values.
- If a requested model or dataset is unavailable, report `BLOCKED`; do not substitute another one without approval.

## 5. Implementation rules

- Prefer a small typed Python package over notebook-only logic.
- Notebooks may visualize results but must call tested library functions for extraction and metrics.
- Use `pathlib`, explicit configuration, deterministic seeds, and structured metadata.
- Avoid hidden global state, import-time downloads, and import-time GPU allocation.
- Keep model-specific behavior inside extractor adapters. Shared analysis code must consume the canonical feature contract.
- Do not infer patch tokens by blindly dropping the first token. Use model configuration and model-specific validation, then assert the canonical patch count.
- Avoid quadratic `N x N` Gram matrices for large patch-level CKA. Use feature-space or streaming formulations defined in `docs/METRIC_DEFINITIONS.md`.
- Use safe tensor formats for generated feature artifacts. Do not use pickle as the canonical interchange format.
- New behavior requires tests. Bug fixes require a regression test when feasible.
- Add no dependency unless it is needed by the active task. Explain every new runtime dependency in the run report.

## 6. Git and scope control

- Inspect `git status` before editing.
- Do not overwrite unrelated user changes.
- Keep one task ID to one bounded change set.
- Do not rename or reorganize unrelated files.
- Do not commit, push, create branches, or open pull requests unless explicitly requested.
- Generated outputs belong under `artifacts/` and are ignored by Git. Human-readable run reports belong under `reports/runs/`.

## 7. Validation behavior

- Run the smallest relevant checks first, then broader checks only when justified.
- Report the exact commands executed and their exit status.
- Distinguish required checks from optional checks.
- If a required check cannot run, the result cannot be `PASS`.
- Do not claim GPU validation when only CPU validation ran, or real-checkpoint validation when only mocks ran.
- Preserve raw logs or summaries needed to reproduce a failure.

## 8. Required final response for every implementation task

Return these sections:

1. `Status`: `PASS`, `PARTIAL`, or `BLOCKED`.
2. `Summary`: what changed and why.
3. `Changed files`: exact paths.
4. `Validation`: commands, exit status, and key results.
5. `Artifacts`: generated report or artifact paths.
6. `Deviations`: anything that differs from the task specification.
7. `Remaining risks`: unresolved technical or scientific risks.
8. `Next action`: one concrete next step; do not begin it unless requested.

Also create a run report using `docs/templates/RUN_REPORT_TEMPLATE.md` unless the task explicitly says not to.
