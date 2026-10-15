# Run Report — VEA-P2-T004A

## Status

`PASS` for Tier-0 design/config validation. Delivery is `COMMIT_AND_PUSH`;
independent review remains pending and this task is not self-accepted.

## Metadata

- Task ID: `VEA-P2-T004A`
- Run/report timestamp UTC: `2026-10-15T11:30:05Z`
- Repository root: repository-relative `.` (`NAIS_P2`)
- Git commit before work: `f941d931b7f6f46aeb7f947a133a08bc9c0bff5a`
- Git dirty before work: only untracked `.codex/ACTIVE_TASK.md`, the approved
  task-control residual; tracked and staged diffs empty
- Execution environment: Windows PowerShell and Python 3 standard library;
  design/config and text validation only; no added dependency
- Task branch: `codex/vea-p2-t004a-low-rank-linear-design-freeze`

## Goal

Freeze a reproducible Train/Validation-only reduced-rank regression study of
minimum same-Query adapter complexity relative to Full Linear OLS.

## Summary

The new JSON config freezes the source and target identities, accepted
Train-fitted T002D `FULL_LINEAR` artifact SHA for each source, immutable
`p1_1000` split, closed-form RRR from the accepted OLS weight and Train
covariance, six ranks, factor-only storage, hard parameter and planned latency
budgets, same-run Validation generation, selection and null-result rules, and
zero Test access. External artifact roots and feature-bank paths remain
`OPEN_PENDING_RUNTIME_IDENTITY_BINDING`. No adapter was fitted or evaluated.

The accepted predecessor linear config's canonical Git-blob SHA-256 is
`28835f065f9a00d756316ced0a1d5ad3e74a3709d88a83abc47ddbe1d6f142a1`.
The accepted linear binding's canonical Git-blob SHA-256 is
`7688f7f14bdf6058f314e8b23ab60cc7ce9ca59dc48df4844b328ed7844662d3`.
The binding pins Validation result SHA-256
`10777d3aa4b19683beecb3b78ae5cea21d20538ae126db9a65daf434aa604ac9`
and the SIGLIP2/DINOV3 `FULL_LINEAR` artifact SHA-256 values
`0661e426aa62623283c971925d2df93fe6621217d746f8583c3158bc31984974`
and `71728c32e2343f9e8e23f45d524142671f10a3eae0b3bbe59c40f8b5a26cecc7`.
These are predecessor identities, not newly generated artifacts.

## Scientific design and numeric derivation

For each source, the accepted OLS weight `W` and Train means `mu_X,mu_Y`
are bound by the accepted artifact SHA. In future approved fitting, float64
streaming Train rows yield `Sxx=Xc.T@Xc`, `M=W.T@Sxx@W`; top-r eigenvectors
of `M` form `V_r`, with `U_r=W@V_r` and
`b_r=mu_Y-mu_X@(U_r@V_r.T)`. The applied map is
`(x@U_r)@V_r.T+b_r`. This is reduced-rank regression; directly truncating
`SVD(W)` would solve a different objective. The config defines finite/PSD
guards, deterministic eigenspace tie and sign handling, and ineligible
`DEGENERATE_RANK` candidates.

| Rank | Stored scalars `1536r+768` | Raw float32 bytes | Per-patch MACs `1536r` |
|---:|---:|---:|---:|
| 8 | 13,056 | 52,224 | 12,288 |
| 16 | 25,344 | 101,376 | 24,576 |
| 32 | 49,920 | 199,680 | 49,152 |
| 64 | 99,072 | 396,288 | 98,304 |
| 128 | 197,376 | 789,504 | 196,608 |
| 256 | 393,984 | 1,575,936 | 393,216 |

Full Linear is `768²+768=590592` scalars, 2,362,368 raw float32 bytes and
589,824 per-patch MACs, plus the same 768 bias additions. Rank 256 reduces
stored scalar count by approximately 33.29% and stays under the 400,000
scalar/source ceiling. Actual serialized adapter bytes are future runtime
measurements; no measured size or latency is claimed here.

Future latency profiling uses one image/196 patches, float32, identical
device, adapter-only timing, 50 warmups and 200 timed iterations in each of
three repeats, with CUDA synchronization when applicable. A candidate must
have p50 no more than `1.10 ×` matched OLS p50. Future generation covers
six ranks plus OLS for both sources on the same 100 Validation images, with
native once/image: `100 + (2 × 7 × 100) = 1500` planned generations. It
requires separate approval; none were run here.

Per source, the smallest rank qualifies only when all five predeclared
Validation conditions pass: mean native-reference Token F1 within 0.02
absolute of same-run OLS, R@1 within 0.02 absolute, native exact count at
least `max(0, OLS_count-1)` out of 100, p50 ratio at most 1.10, and at most
400,000 stored scalars. Otherwise `NO_RANK_ACCEPTED` retains OLS. The
criterion is exploratory engineering, not proof of equivalence. Paired
per-image deltas and 95% image bootstrap intervals (seed 43, 2,000 resamples)
are descriptive and cannot change selection. Native-reference F1 measures
fidelity to the frozen CLIP path, not independent caption correctness.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `configs/vlm_compat/p2_query_adapter_low_rank_linear_v1.json` | New design config | Freeze exact RRR and Validation-only study |
| `reports/runs/VEA-P2-T004A-low-rank-linear-design-freeze.md` | New run report | Record derivation, provenance and validation limits |

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Git identity and baseline | `git rev-parse`, `git ls-remote`, `git worktree list` | 0 | T003H local/remote exact baseline; new branch free | Git command output |
| Git entry content and locks | `git status`, `git diff`, `git ls-files`, PowerShell lock listing | 0 | Sole approved Active Task residual; no lock | Git command output |
| Canonical predecessor lineage | `py -3 -c` with `git show HEAD:<path>` SHA-256 checks | 0 | Canonical Git blobs match predecessor config and contract bindings | Git command output |
| Tier-0 design assertions | PowerShell here-string piped to `py -3 -` | 0 | `TIER0 PASS 130 assertions`; formulas, identities, counts, budgets, selector and Test restrictions | Config and command output |
| Config SHA-256 | Python `hashlib.sha256(path.read_bytes())` | 0 | `7cde52b7a9c11da630475abf99375cf4bf08007f2a1d526532e3aaa8b9a1eb1f` | Config |
| Exact allowed paths and whitespace | `git diff --check`, `git diff --cached --check`, path assertion | 0 | Two approved paths only | Git command output |
| Commit/push scope and remote proof | `git diff baseline..HEAD`, `git ls-remote` | 0 | Exact two paths; remote head matches | Final handoff |

## Test and command transcript summary

```text
Entry: git rev-parse --show-toplevel; git rev-parse --absolute-git-dir;
git rev-parse --git-common-dir; git branch --show-current; git rev-parse HEAD;
git remote -v; git ls-remote --heads origin <T003H> <T004A>;
git branch --list <T004A>; git worktree list --porcelain;
git status --porcelain=v1 --untracked-files=all; git diff --name-status;
git diff --cached --name-status; git ls-files --others --exclude-standard;
git diff --quiet; git diff --cached --quiet; PowerShell .git lock listing.
All entry gates exited 0; only the approved Active Task was untracked.
Tier 0: @' <standard-library JSON assertion program> '@ | py -3 -
Exit 0: TIER0 PASS 130 assertions
CONFIG_SHA256 7cde52b7a9c11da630475abf99375cf4bf08007f2a1d526532e3aaa8b9a1eb1f
No pytest, package installation, adapter fitting, dataset/model load, GPU,
inference, generation, or Test access occurred.
```

## Artifacts

- Design config SHA-256:
  `7cde52b7a9c11da630475abf99375cf4bf08007f2a1d526532e3aaa8b9a1eb1f`.
- This report is the only other new artifact. No ZIP, tensor, checkpoint,
  dataset, generated result, or package was produced.

## Deviations

- `.codex/PROJECT_STATE.json` was absent at entry; the task permits absence.
- The canonical Git-blob SHA differs from the Windows worktree byte SHA for
  existing predecessor files because of checkout line endings. Canonical
  lineage and Git content checks passed; no predecessor file was changed.
- `docs/PROJECT_STATUS.md` still says T003H awaits review, while the T004A
  task states T003H was independently reviewed. This state drift is reported
  and left untouched under T004A's exact two-path allowlist.
- No scientific or implementation decision deviates from the approved task.

## Unverified items

- External feature-bank and accepted OLS artifact roots remain
  `OPEN_PENDING_RUNTIME_IDENTITY_BINDING`; raw files were not loaded or
  rehashed in T004A.
- No latency, generation, metric, or statistical result was measured.
- Older ChatGPT Project Live State may be stale and was not edited.
- Canonical integration merge remains `UNVERIFIED`; historical Codex test
  counts remain `UNRECOVERED`.

## Remaining risks

- Implementation: future code must reproduce the frozen eigenvalue tie/PSD
  policy, artifact identity checks and bounded-memory fit exactly.
- Environment: future runtime bindings and matched hardware timing are open.
- Scientific interpretation: the exploratory Validation thresholds do not
  establish equivalence or a Test claim; a future confirmatory Test would
  require a separately approved untouched-holdout policy.

## Suggested next action

Submit pushed T004A for independent review.
