# Run Report — VEA-P2-T004B

## Status

`PASS` for the bounded Tier 0–1 numerical-core implementation and synthetic
validation. Git delivery is `COMMIT_AND_PUSH`; independent review remains
pending, and this task is not self-accepted.

## Metadata

- Task ID: `VEA-P2-T004B`
- Run/report timestamp UTC: `2026-10-15T12:15:25Z`
- Repository root: repository-relative `.` (`NAIS_P2`)
- Git commit before work: `c52ebd4c982081d616945efe71a9440630575b24`
- Git dirty before work: only untracked `.codex/ACTIVE_TASK.md`; tracked and
  staged diffs empty
- Execution environment: Windows PowerShell, Python 3.12.0, NumPy 1.26.4,
  pytest 8.4.2; CPU synthetic fixtures only; no new dependency
- Branch: `codex/vea-p2-t004b-low-rank-linear-numerical-core`

## Goal

Implement and synthetically validate the exact Train-only closed-form
reduced-rank regression core frozen in T004A, without real feature banks or
model runtime.

## Summary

Added a pure NumPy module for exact T004A config loading, frozen-field and
canonical digest validation, production parameter counts, two-pass
bounded-memory paired Train streaming, finite/identity/mean/row-count checks,
float64 source covariance and RRR eigensolve, deterministic tied-eigenspace
canonicalization, explicit degenerate ranks, and compact float32 factor
application. The module stores no dense OLS weight in a selected low-rank
result. Small dimensions and ranks require an explicit synthetic-test
geometry; production defaults to `[N,196,768]`, 800 Train images, 156,800
patch rows, and the six frozen ranks.

The unchanged authoritative config is
`configs/vlm_compat/p2_query_adapter_low_rank_linear_v1.json` from reviewed
T004A commit `c52ebd4c982081d616945efe71a9440630575b24`, normalized-file
SHA-256 `7cde52b7a9c11da630475abf99375cf4bf08007f2a1d526532e3aaa8b9a1eb1f`.
External feature-bank/OLS artifact identity binding remains future work.
Passing synthetic arrays does not verify accepted real artifacts.

## Numerical control

On a seeded synthetic rank-two affine mapping, the compact rank-one Train
prediction objective was `612.255217219561`; an independent SVD of
`Xc @ W` gave tail objective `612.255217219560`. The rank-two map had
maximum absolute reconstruction error `4.42114711063e-07` against the
synthetic target, with two positive eigenvalues. These are synthetic control
values, not Validation or model performance. Tests also exercised a
full-numerical-rank fixture, chunk independence, rank-cut ties, PSD tolerance,
degenerate spectra, and fail-closed replay changes.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_linear.py` | New pure NumPy RRR core | Implement exact Train-only frozen numeric behavior |
| `tests/unit/test_p2_query_adapter_low_rank_linear.py` | New deterministic synthetic tests | Verify analytic, streaming, tie, PSD, geometry and failure boundaries |
| `reports/runs/VEA-P2-T004B-low-rank-linear-numerical-core.md` | New report | Record checks, provenance and limits |

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Git entry identity, remote and residual | `git rev-parse`, `git ls-remote`, `git --no-optional-locks status`, `git diff --quiet`, `git diff --cached --quiet`, `git ls-files`, lock listing | 0 | T004A local/remote exact baseline, T004B branch free, only Active Task residual, no lock | Git command output |
| Tier 0 compile | `py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_linear.py tests/unit/test_p2_query_adapter_low_rank_linear.py` | 0 | Both files compiled | Python command output |
| Tier 1 new suite | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py` | 0 | 31 passed | pytest output |
| Tier 1 focused regression | `py -3 -m pytest -q tests/unit/test_p2_query_adapter_linear.py` | 0 | 17 passed | pytest output |
| Synthetic numeric control | PowerShell here-string piped to `py -3 -` | 0 | Rank-one objective matches independent prediction-SVD tail; rank-two max error `4.42114711063e-07` | Command output |
| Exact staged scope and whitespace | `git diff --check`, `git diff --cached --check`, exact path assertion | 0 | Three approved paths only | Git command output |
| Commit/push scope and remote proof | `git diff baseline..HEAD`, `git ls-remote` | 0 | Exact three paths and matching remote head | Final handoff |

## Test and command transcript summary

```text
Entry: git rev-parse --show-toplevel; git rev-parse --absolute-git-dir;
git rev-parse --git-common-dir; git branch --show-current; git rev-parse HEAD;
git remote -v; git ls-remote --heads origin <T004A> <T004B>;
git branch --list <T004B>; git worktree list --porcelain;
git --no-optional-locks status --porcelain=v1 --untracked-files=all;
git diff --quiet; git diff --cached --quiet;
git ls-files --others --exclude-standard; PowerShell .git lock/operation listing.
All required Git entry gates exited 0.
python -c "import sys,numpy,pytest; ..." exited 1 on this Windows shell;
py -3 -c "import sys,numpy,pytest; ..." exited 0: 3.12.0 1.26.4 8.4.2.
py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_low_rank_linear.py tests/unit/test_p2_query_adapter_low_rank_linear.py
Exit 0.
py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_linear.py
Exit 0: 31 passed in 0.36s.
py -3 -m pytest -q tests/unit/test_p2_query_adapter_linear.py
Exit 0: 17 passed in 0.28s.
PowerShell synthetic-control here-string | py -3 -
Exit 0: rank1_objective=612.255217219561
independent_prediction_svd_tail=612.255217219560
rank2_max_abs_error=4.42114711063e-07 positive_eigenvalues=2.
```

## Artifacts

- This report is the only non-code artifact. No ZIP, feature tensor,
  checkpoint, dataset, real-model output or package was produced.
- No runtime run ID or real-result checksum exists for T004B.

## Deviations

- `.codex/PROJECT_STATE.json` was absent at entry; the task permits absence.
- The Windows `python` launcher exited 1. The installed `py -3` launcher
  ran the required equivalent compileall and pytest commands successfully.
- No frozen scientific behavior or file-scope deviation.

## Unverified items

- External feature-bank manifest, accepted OLS artifact SHA and model
  identities were not verified against raw runtime files. This pure numerical
  core assumes a future approved caller performs that binding.
- No real Train, Validation or Test feature bank was loaded. No GPU,
  TinyLLaVA, generation, latency measurement or winner selection ran.
- `docs/PROJECT_STATUS.md` and older Project Live State may remain stale;
  neither was in the allowed edit scope.

## Remaining risks

- Implementation: future runtime integration must verify external bank/OLS
  identities and preserve the exact paired Train stream before calling this
  core. Synthetic tests do not establish real-data correctness.
- Environment: CPU-only validation here gives no GPU/runtime evidence.
- Scientific interpretation: no Validation performance or Test claim follows
  from these synthetic numerical controls.

## Suggested next action

Submit pushed T004B for independent review.
