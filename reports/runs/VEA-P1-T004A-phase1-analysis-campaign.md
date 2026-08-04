# VEA-P1-T004A compact run report

Status: implementation validation in progress; post-commit campaign evidence is external to Git.

## Identity and provenance

- Exact baseline commit/tree: `27688e67b0c639a81d07c31b7992dc8a09341d8d` / `b6fb3ef7fc99e0a4f16c351867f2da1fea100760`
- Branch: `codex/vea-p1-t004a`
- Remote: `https://github.com/Anzenoh/NAIS.git`
- Bounded provenance audit: `reports/runs/VEA-P1-T004A-metric-provenance-audit.json`
- Input validation: 252 files, 120 shards, 2,160 tensor headers, zero decoded tensor values before commit
- Input checksum manifest: `44e531e242e434d7f9aa5c901da694bf29088cf2a7cae0fad3d6a87ea2a1386d`

## Implementation

Added strict canonical configuration, fail-closed P1 reader and access ledger, tested global/mapping/patch/canonical metrics, monotonic hashed stage publication, one-time freeze/test gate, dependency-free reports, deterministic non-self-referential package sealing, verification CLIs, and Windows-local repeat semantics.

## Validation record

Exact command families are the ACTIVE_TASK Tier 0--2 compile, focused pytest, named miniature byte-identity test, Bash syntax, complete offline pytest, and `git -c core.whitespace=cr-at-eol diff --check`. The implementation environment uses the existing Windows Python 3.12 launcher because the WindowsApps `python3` alias is not an executable runtime; no package was installed.

The exact task commit is the commit containing this report and is resolved post-commit with `git rev-parse HEAD^{commit}`. The real scientific result, seal identities, final six-line block, package verification, and ending input checksum evidence live only under the bounded external result root.

## Blocking validation evidence (2026-08-04)

- Focused T004A suite: `24 passed`; deterministic miniature package identity: `1 passed`; Bash syntax: exit 0.
- Required complete offline suite: exit 1, `28 failed, 742 passed`.
- First root cause: the exact protected baseline file `scripts/dinov3_extraction_worker.py` does not contain `snapshot_files = snapshot_files_directory(snapshot_root)`, while the exact baseline test `tests/integration/test_dinov3_extractor_artifact.py::test_worker_source_guard_order_local_snapshot_and_no_mutation_clients` requires it.
- Independent reproduction: the single named test exits 1 with the same assertion under a short Windows temp path, so it is not caused by the long-path failures later in the complete-suite output.
- Both paths are protected from T004A changes. Mandatory validation therefore blocks staging, commit, push, feature-value access, and the real campaign. The isolated worktree remains resumable and no source tensors were decoded.
- Continuation audit 2: authority hash `f96cd720c258e03d266114c518198625ecaa5687c6e421c174954210476dcd68`, baseline/remote ref `27688e67b0c639a81d07c31b7992dc8a09341d8d`, and the protected-path failure are unchanged. The exact isolated single-test command again exited 1 (`1 failed in 0.35s`). No files are staged.
- Continuation audit 3: the same authority hash, exact local/remote baseline, and protected-path assertion remain unchanged. The isolated test again exited 1 (`1 failed in 0.35s`). This is the third consecutive identical blocking audit; task scope still forbids the only relevant source/test changes, so no in-scope progress to commit or execute exists.
