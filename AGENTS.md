# AGENTS.md — Vision Encoder Alignment Research

## 1. Repository purpose

This repository implements reproducible experiments for comparing and aligning
hidden representations from heterogeneous frozen vision encoders. Phase 0 and
Phase 1 are encoder-only. Phase 2 permits query-conditioned adapter-only fitting
and evaluation through a frozen TinyLLaVA connector and frozen LLM under the
approved Phase 2 contracts.

## 2. Instruction and context loading

Before changing files, do the following in order:

1. Read `AGENTS.md`, `.codex/PROJECT_STATE.json`, and the user-approved,
   non-placeholder `.codex/ACTIVE_TASK.md`.
2. Treat the Active Task as the sole implementation contract. Report state drift;
   stale `NO_ACTIVE_TASK` alone is not a veto.
3. Read additional source, tests, contracts and history only at paths explicitly
   named by the Active Task. `docs/INDEX.md`, `docs/PROJECT_STATUS.md` and
   `tasks/` are not default reads.

Do not invent a missing Active Task or alter state to make it agree. Missing or
ambiguous approval blocks implementation. An explicitly requested read-only
investigation may instead use the exact task file supplied by the user.

Do not indiscriminately load generated artifacts or unrelated documents. If the task conflicts with `docs/RESEARCH_CONTRACT.md`, stop and report the conflict. Do not silently reinterpret the contract.

The following files are protected and may be edited only when the active task explicitly lists them in scope:

- `AGENTS.md`
- `docs/RESEARCH_CONTRACT.md`
- `docs/METRIC_DEFINITIONS.md`
- `docs/EXPERIMENT_PROTOCOL_P2_ADAPTER.md`
- `docs/VLM_COMPATIBILITY_METRICS.md`
- `docs/contracts/FEATURE_ARTIFACT_SCHEMA.md`
- `docs/contracts/VLM_ADAPTER_COMPATIBILITY_V1.md`
- `configs/models.yaml`

## 3. Non-negotiable research rules

- Keep all vision encoders frozen in Phase 0 and Phase 1.
- Do not introduce an LLM, multimodal projector, caption decoder, or instruction-tuning path in Phase 0 or Phase 1.
- In Phase 2, only the adapter may be fitted or trained. Source encoders,
  target CLIP, the TinyLLaVA connector, and the TinyLLaVA LLM remain frozen.
- Phase 2 may evaluate adapted features through the frozen target connector and
  may run frozen-LLM generation. It never permits connector, LLM, target-CLIP,
  or source-encoder fine-tuning.
- A direct Phase 2 adapter must use the same Query image's source feature; it
  must not present another image's target feature as its primary output.
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

- Apply the approved Active Task's Git entry/handoff checks; verify actual
  root/Git/common dirs, branch/HEAD/baseline, delivery and each Git exit code.
- Only for a documented false-dirty anomaly, require expected HEAD, both
  `git diff --quiet` / `git diff --cached --quiet` exiting 0, and
  `git ls-files --others --exclude-standard` exiting 0 with no paths before
  treating porcelain as advisory. Never ignore actual content differences.
- Preserve unrelated changes. Unexplained diffs/untracked paths, identity drift
  or relevant unresolved write locks block mutation. Approved exact, non-conflicting
  residuals are DIRTY_EXPECTED, not CLEAN; read-only work preserves entry state.
- No automatic stash/reset/restore/clean, lock deletion, index refresh, worktree
  removal/prune, force push, amend, rebase or merge to clear a gate.
- Commit/push/branch/PR actions require authorization and exact task paths/delivery.
  Keep one task ID to one bounded change set; no unrelated staging or reorganization.
- Reuse verified workspaces; serialize writes sharing a common Git directory.
  Phantom porcelain alone does not justify a new clone or workspace migration.
- Generated outputs belong in ignored `artifacts/`; reports in `reports/runs/`.
  Delivery, independent acceptance, Task closure and next-task approval are separate.

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

Include one compact Git handoff: root/Git/common dirs, branch/baseline/HEAD,
delivery and verification, residual paths, relevant locks, and Task/state/review
disposition. No extra handoff document; never self-accept. Keep machine-local
paths in transient output, not committed reports (see §4).

Also create a run report using `docs/templates/RUN_REPORT_TEMPLATE.md` unless the task explicitly says not to.