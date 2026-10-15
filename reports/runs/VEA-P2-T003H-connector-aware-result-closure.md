# Run Report — VEA-P2-T003H

## Status

`PASS` for the bounded documentation and validation contract, subject to
independent review. Scientific disposition: `NOT_PROMOTED` for the tested
same-capacity connector-aware affine objective over OLS.

## Metadata

- Task ID: `VEA-P2-T003H`
- Run/report timestamp UTC: `2026-10-15T11:12:05Z`
- Repository root: repository-relative `.` (`NAIS_P2`)
- Git commit before work: `0b3b7a2ee135e1e85cf459248066f5b53a313ef9`
- Git dirty before work: only untracked `.codex/ACTIVE_TASK.md`, the approved
  task-control residual; tracked and staged diffs empty
- Execution environment: Windows PowerShell; text and Git checks only; no
  package, GPU, model, dataset, training, or generation execution
- Git delivery: approved `COMMIT_AND_PUSH` on
  `codex/vea-p2-t003h-connector-aware-result-closure`; final handoff records
  actual commit and remote-head verification

## Goal

Synchronize repository documents with T003E/T003G evidence, without changing
code or starting low-rank work.

## Summary

Recorded the T003E Train/Validation proxy improvements and T003G Validation
generation evidence, including their differing SIGLIP2 and DINOV3 outcomes.
The tested same-capacity connector-aware affine objective is `NOT_PROMOTED`
over OLS; this does not reject connector loss generally. Low-rank linear work
is a future design-and-freeze action only. Existing T002E held-out Test evidence
is preserved; T002E and T003G absolute metrics are from different partitions.

## Evidence and limits

- T003E: `PASS_T003E_REAL_APPLE_TRAIN_VALIDATION`; implementation commit
  `97a617ab2a11a34512c591c344161b877f28b8c1`; result SHA-256
  `2147b41ec525f387c0cc8d434d6b3cd748ed224ef39053fc9c6a9e7372963a10`;
  runtime identity
  `6e0e274a47813101e560d8f8dc1dbf40ad416b89abd7e6f6d1688646f4b466ca`;
  Test access/metrics/generation/vision `0/0/0/0`. SIGLIP2 OLS to
  connector-aware selected (`lambda=4`, epoch 7): Validation R@1
  `0.89 -> 0.92`, connector cosine
  `0.48697282446433066 -> 0.5056841280687281`, patch cosine
  `0.4628383531145029 -> 0.449191858608662`. DINOV3 selected
  (`lambda=0.25`, epoch 0) has OLS/patch-only affine SHA-256
  `918872025697964e4126675a107f5db00b4efbed222c73b6c30ec40c9bf37e3f`.
- T003F: implementation commit
  `0b3b7a2ee135e1e85cf459248066f5b53a313ef9` was independently reviewed;
  that implementation task did not execute real generation.
- T003G: user-supplied Validation-only launcher output states
  `PASS_T003G_REAL_VALIDATION_GENERATION`, run
  `VEA-P2-T003G-REAL-APPLE-20261001T164130Z-631682`, at T003F implementation
  commit; 100 Validation images; 100 native plus 400 adapted = 500 generations;
  runtime/verification exits `0/0`; structural verification `PASS`; Test
  access/metrics `0/0`. Supplied rows SHA-256
  `303af396dea0e7d7a5c5931a2deed58e9caa96eb85d0f9d21de299aba66fa254`,
  aggregate SHA-256
  `25c088e333ecd2931ef8f57b1c26ef610928d05c7b6a07d4439a83c1bd45711e`,
  result SHA-256
  `8f462bce3d24a8da09c9afb7f1aa2b303778245d2b65e9ca211ddf7c098a2118`.
  SIGLIP2 OLS F1 `0.577915913`, native exact `1/100`; connector-aware F1
  `0.569035705`, native exact `0/100`; text/token equality `18/100`; F1 delta
  `-0.008880208`. DINOV3 OLS and connector-aware each F1 `0.594797657` and
  native exact `1/100`, with text/token equality `100/100` and byte-identical
  affine identity.
- T003G hashes and structural `PASS` above are user-supplied execution output.
  T003H did not fetch, independently rehash, regenerate, or review raw JSONL,
  result, or per-image files. No Test-informed method selection or new Test
  access is authorized.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `docs/PROJECT_STATUS.md` | Added study status and evidence boundary | Synchronize current repository state |
| `docs/DECISION_LOG.md` | Added one `NOT_PROMOTED` study disposition | Record bounded scientific decision |
| `docs/EXPERIMENT_PROTOCOL_P2_ADAPTER.md` | Added Validation study and future-design gate | Preserve split and experiment boundaries |
| `docs/VLM_COMPATIBILITY_METRICS.md` | Added Validation generation interpretation | Keep proxy and generation claims distinct |
| `reports/runs/VEA-P2-T003H-connector-aware-result-closure.md` | Added this run report | Record evidence, checks, and limits |

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Git entry identity and baseline | `git rev-parse`, `git branch`, `git ls-remote`, `git worktree list` | 0 | T003F HEAD and remote at exact baseline; T003H branch free before creation | Git command output |
| Git entry content and locks | `git status`, `git diff`, `git ls-files`, PowerShell lock listing | 0 | Only approved Active Task residual; no relevant lock | Git command output |
| Evidence and wording | Deterministic text assertions | 0 | Required hashes, metrics, split limits, and wording present | This report and four docs |
| Five-path allowlist and whitespace | `git diff --check` and path assertion | 0 | Only five approved paths | Git command output |
| Staged allowlist and whitespace | `git diff --cached --check` and path assertion | 0 | Only five approved paths | Git command output |
| Postcommit scope and remote proof | `git diff baseline..HEAD`, `git ls-remote` | 0 | Exact five paths; remote HEAD matches local | Final handoff |

## Test and command transcript summary

```text
Entry: git rev-parse --show-toplevel / --absolute-git-dir / --git-common-dir;
git branch --show-current; git rev-parse HEAD; git remote -v;
git status --porcelain=v1 --untracked-files=all; git diff --name-status;
git diff --cached --name-status; git ls-files --others --exclude-standard;
git worktree list --porcelain; git branch --list <T003H branch>;
git ls-remote --heads origin <T003F branch> <T003H branch>;
git diff --quiet; git diff --cached --quiet; PowerShell .git lock listing.
All entry gates passed. The approved task-control file was the sole residual.
PowerShell Get-Content -Raw assertions over the five allowed files checked the
exact hashes, metrics, split language, disposition, and evidence limits. Two
initial literal checks exited 1 on a required phrase split across Markdown
lines; the whitespace-normalized check exited 0 without document changes.
git diff --check; git diff --cached --name-only; git diff --cached --check;
git diff --name-only; git ls-files --others --exclude-standard.
The exact five-path allowlist and sole Active Task residual assertion exited 0.
No pytest or full suite was run, as the task specifies text-only validation.
```

## Artifacts

- This report is the only new artifact; no generated tensor, checkpoint,
  dataset, JSONL, model output, or package artifact was produced.
- T003G run ID and user-supplied checksums are recorded in Evidence and limits;
  the raw files were not part of this handoff.

## Deviations

- `.codex/PROJECT_STATE.json` was absent at entry. The approved task expressly
  permits this file to be missing; no state was invented or changed.
- The previous PowerShell `Get-Date -AsUTC` probe was unsupported by this host;
  `[DateTime]::UtcNow` supplied the report timestamp. A `git rev-parse` commit
  expression with PowerShell-sensitive syntax failed; direct HEAD and remote
  identity checks supplied the required baseline proof.
- No other deviation from the approved task.

## Unverified items

- Raw T003G rows, aggregate, and result files were not independently accessed,
  hashed, or inspected; structural `PASS` is user-supplied.
- Canonical integration merge remains `UNVERIFIED`; historical Codex test
  counts remain `UNRECOVERED`.
- ChatGPT Project files, including `04_PROJECT_LIVE_STATE.md`, may now be stale
  and require separate approved synchronization.

## Remaining risks

- Implementation: none identified for this bounded documentation task.
- Environment: raw T003G artifacts are unavailable in this handoff for
  independent per-image review.
- Scientific interpretation: Validation generation gives no held-out Test
  claim for this objective; other connector-loss designs remain untested here.

## Suggested next action

Submit T003H for independent review.
