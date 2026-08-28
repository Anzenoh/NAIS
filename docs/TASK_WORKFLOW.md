# Human–ChatGPT–Codex Task Workflow

## 1. Roles

### Human researcher

- selects research direction and approves scope changes;
- supplies licensed datasets/checkpoints and execution access;
- runs or supervises Codex;
- returns evidence for review;
- decides whether to commit accepted work.

### ChatGPT planning/review thread

- converts research intent into bounded task specifications;
- checks tasks against repository contracts;
- reviews Codex evidence;
- assigns `PASS`, `PARTIAL`, or `BLOCKED`;
- writes the next task only after the current state is understood.

### Codex

- reads repository instructions and the active task;
- inspects before editing;
- implements only the approved scope;
- runs required checks;
- writes a run report with exact evidence;
- does not make unapproved research decisions.

## 2. Task lifecycle

1. **Discuss** the next research or engineering objective in ChatGPT.
2. **Write** one task file using `docs/templates/TASK_TEMPLATE.md`.
3. **Baseline** the repository with a clean or understood `git status`.
4. **Execute** the task in Codex by referencing exact paths.
5. **Collect** the run report, changed files, diff, and test output.
6. **Review** against the task and contracts using `REVIEW_TEMPLATE.md`.
7. **Classify** the result as PASS, PARTIAL, or BLOCKED.
8. **Commit or revise** only after review.
9. **Update** `PROJECT_STATUS.md` through an explicit task.
10. **Issue** the next bounded task.

## 3. Task size

A good task has one externally verifiable objective. Split work when it combines more than one of the following:

- new model support;
- new artifact schema;
- new metric;
- full dataset execution;
- visualization/reporting;
- environment migration;
- research-contract change.

For example, implement and validate CLIP extraction before adding SigLIP2 and DINOv3.

## 4. Prompt structure

Every Codex command should identify:

- Task ID and mode;
- exact goal;
- files to read first;
- current evidence or reproduction steps;
- allowed scope;
- non-goals;
- constraints;
- deliverables;
- validation commands or observable acceptance criteria;
- stop conditions;
- required final report.

Do not rely on “continue the project” or “make it work.”

## 5. Planning versus implementation

Use an investigation-only task when any of these are unknown:

- repository state;
- runtime or dependency compatibility;
- external access;
- model output semantics;
- likely file impact;
- acceptance-test feasibility.

The investigation task must not silently transition into implementation. Once the plan is reviewed, issue a separate implementation task.

## 6. Evidence bundle returned for review

At minimum provide:

- generated `reports/runs/<run>.md`;
- `git status --short`;
- `git diff --stat`;
- the relevant diff or changed files;
- exact required test output;
- generated manifests for artifact-related tasks;
- any error log for PARTIAL or BLOCKED work.

## 7. Review rule

Review the submitted evidence, not the agent's confidence. If the material is insufficient to verify a requirement, mark that requirement unverified. Do not infer missing results.

## 8. Recommended branch/commit discipline

- one task per branch or clearly isolated working-tree change;
- one accepted task per commit;
- commit message includes the task ID;
- generated artifacts remain untracked;
- protected contract changes are isolated from implementation changes where possible.

## 9. Git cleanliness authority

`VEA-OPS-GIT-001` demonstrated a filesystem/location false-dirty effect in the
G: worktree. A worktree is Task-clean only when all four content-level checks
pass:

1. `git rev-parse HEAD` equals the exact expected Task baseline;
2. `git diff --quiet` exits 0;
3. `git diff --cached --quiet` exits 0;
4. `git ls-files --others --exclude-standard` returns no paths.

`git status` and porcelain remain useful diagnostic information. Nonempty
porcelain alone is not a blocker in an environment where the documented
filesystem/location effect applies and all four authoritative checks pass.
No semantic diff, staged diff, untracked path, or baseline mismatch may be
ignored. `git diff-files --quiet` is not the sole cleanliness authority in this
environment.

`git update-index --refresh` is a diagnostic operation, not routine cleanup,
and requires explicit Task authorization. A new clone or repository migration
is not required solely because of phantom porcelain modifications. The
controlled evidence implicated filesystem location; it did not prove that
Google Drive software itself was the causal mechanism.
