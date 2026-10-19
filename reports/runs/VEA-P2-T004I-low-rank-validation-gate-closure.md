# Run Report — VEA-P2-T004I

## Status

`PASS` for the bounded documentation and deterministic Tier-0 gate check. Independent review remains `REVIEW_PENDING`; this report does not self-accept the study.

## Metadata

- Task ID: `VEA-P2-T004I`
- Run/report timestamp UTC: `2026-10-19T16:00:00Z`
- Repository root: verified local task workspace; machine-local absolute path omitted
- Git commit before work: `a67d923b2b65c64f9782d70cd13f183312ed60fd`
- Git dirty before work: only approved untracked `.codex/ACTIVE_TASK.md`; tracked and staged diffs empty
- Execution environment: Windows PowerShell, Python standard library only for the gate check; no Apple, model, data or GPU runtime
- Branch: `codex/vea-p2-t004i-low-rank-validation-closure`; delivery `COMMIT_AND_PUSH`; review `REVIEW_PENDING`

## Goal

Record user-supplied T004F/T004H evidence and apply the already-frozen T004A same-run Validation R@1 necessary criterion to all six low-rank candidates per source, preserving unmeasured downstream fields.

## Summary

T004A's unchanged config freezes ranks `[8,16,32,64,128,256]`, R@1 candidate minimum `OLS - 0.02` absolute, conjunctive qualification, and `NO_RANK_ACCEPTED` with accepted OLS retained when none qualifies. A standard-library `Decimal` check parsed the 14 rows in the approved Active Task, verified both minima, all twelve R@1 deltas, six failures per source, and the rank-256 `-0.03` difference from OLS. No threshold, rank grid, factor, T004G output or experimental config was changed.

The resulting bounded eligibility verdict is `NO_RANK_ACCEPTED` for SIGLIP2 and DINOV3. This is a documentation-level application of a necessary gate; the T004G runtime's original per-condition status remains `UNSELECTED`. It is not an observed connector, generation or latency result and does not establish universal low-rank failure.

## Evidence provenance

- **T004F Train-only fit, user supplied:** reviewed T004E commit `8c0beb8a88cc5bdb1225c50ed62a7ea0349ef06e`; run `VEA-P2-T004F-REAL-APPLE-20261019T113611Z-253652`; result SHA-256 `84549208e6e7a097e769b89b254bcf1bf939b1e29a3c06185a77f0afbdd858f3`. Launcher reported 800 Train images per source, all six ranks `FITTED`, 768 positive eigenvalues per source, artifact manifest verification `PASS`, shell exit `0`; no Validation/Test metric or generation.
- **T004H Validation patch/identity, user supplied:** reviewed T004G commit `a67d923b2b65c64f9782d70cd13f183312ed60fd`; run `VEA-P2-T004H-REAL-APPLE-20261019T141806Z-259343`; result SHA-256 `ee4c36afd3597a4e737a3af33e5a27a7c944bad282331c2fa2a0e3558fa70fe2`. Supplied verifier reported `OUTPUT_STRUCTURE_VERIFY=PASS`, `ALL_RANKS=UNSELECTED`, `TEST_NUMERIC_EVALUATION=NOT_RUN`, `CONNECTOR_GENERATION_LATENCY=NOT_EVALUATED`, `VEA_P2_T004H_REAL_VALIDATION_PATCH_IDENTITY=PASS`, `T004H_SHELL_EXIT=0`.
- T004I did **not** access either Apple root, independently rehash either result JSON, or recompute metrics from per-image arrays. The values below are rounded user-supplied console summaries. The deterministic check verifies their arithmetic against the frozen policy, not the raw observations.

## Supplied same-run Validation table

All patch metrics and retrieval values below refer to the same 100 Validation image IDs per source. The reference is native TARGET_CLIP preconnector patches for the same Query images; OLS is each source's accepted Full Linear control.

| Source | Condition | Patch cosine | Patch NMSE | Patch R² | R@1 | R@5 |
|---|---|---:|---:|---:|---:|---:|
| SIGLIP2 | OLS_REFERENCE | 0.462842 | 0.863400 | 0.136600 | 0.890 | 0.980 |
| SIGLIP2 | LOW_RANK_8 | 0.419152 | 0.904354 | 0.095646 | 0.130 | 0.340 |
| SIGLIP2 | LOW_RANK_16 | 0.427904 | 0.896340 | 0.103660 | 0.270 | 0.520 |
| SIGLIP2 | LOW_RANK_32 | 0.437880 | 0.887121 | 0.112879 | 0.440 | 0.780 |
| SIGLIP2 | LOW_RANK_64 | 0.448120 | 0.877450 | 0.122550 | 0.710 | 0.890 |
| SIGLIP2 | LOW_RANK_128 | 0.456768 | 0.869154 | 0.130846 | 0.810 | 0.970 |
| SIGLIP2 | LOW_RANK_256 | 0.461477 | 0.864650 | 0.135350 | 0.860 | 0.980 |
| DINOV3 | OLS_REFERENCE | 0.507949 | 0.813225 | 0.186775 | 0.870 | 0.980 |
| DINOV3 | LOW_RANK_8 | 0.461682 | 0.861145 | 0.138855 | 0.120 | 0.320 |
| DINOV3 | LOW_RANK_16 | 0.470888 | 0.851873 | 0.148127 | 0.230 | 0.560 |
| DINOV3 | LOW_RANK_32 | 0.481471 | 0.841085 | 0.158915 | 0.430 | 0.830 |
| DINOV3 | LOW_RANK_64 | 0.492282 | 0.829814 | 0.170186 | 0.640 | 0.940 |
| DINOV3 | LOW_RANK_128 | 0.501130 | 0.820472 | 0.179528 | 0.790 | 0.970 |
| DINOV3 | LOW_RANK_256 | 0.506261 | 0.815003 | 0.184997 | 0.840 | 0.980 |

## Frozen R@1 gate calculation

| Source | OLS R@1 | Required minimum | Rank | Candidate R@1 | Delta to OLS | R@1 gate |
|---|---:|---:|---:|---:|---:|---|
| SIGLIP2 | 0.890 | 0.870 | 8 | 0.130 | -0.760 | FAIL |
| SIGLIP2 | 0.890 | 0.870 | 16 | 0.270 | -0.620 | FAIL |
| SIGLIP2 | 0.890 | 0.870 | 32 | 0.440 | -0.450 | FAIL |
| SIGLIP2 | 0.890 | 0.870 | 64 | 0.710 | -0.180 | FAIL |
| SIGLIP2 | 0.890 | 0.870 | 128 | 0.810 | -0.080 | FAIL |
| SIGLIP2 | 0.890 | 0.870 | 256 | 0.860 | -0.030 | FAIL |
| DINOV3 | 0.870 | 0.850 | 8 | 0.120 | -0.750 | FAIL |
| DINOV3 | 0.870 | 0.850 | 16 | 0.230 | -0.640 | FAIL |
| DINOV3 | 0.870 | 0.850 | 32 | 0.430 | -0.440 | FAIL |
| DINOV3 | 0.870 | 0.850 | 64 | 0.640 | -0.230 | FAIL |
| DINOV3 | 0.870 | 0.850 | 128 | 0.790 | -0.080 | FAIL |
| DINOV3 | 0.870 | 0.850 | 256 | 0.840 | -0.030 | FAIL |

Both rank-256 values are one `0.01` increment below the respective minimum on 100 Validation images. All lower ranks fail by larger margins. Since the T004A criteria are conjunctive, none can qualify regardless of the remaining unmeasured criteria. The accepted OLS controls are retained. The frozen 1,500 Validation-generation plan is unchanged, but an expensive follow-up is not scheduled by default after the necessary gate has failed.

## Changed files

| Path | Change | Reason |
|---|---|---|
| `docs/PROJECT_STATUS.md` | Update bounded study status and next action | Remove stale future-low-rank planning text and record user-supplied execution/gate limits |
| `docs/DECISION_LOG.md` | Add study disposition VEA-D024 and clarify OPEN fields | Preserve VEA-D023 while recording no qualifying T004A rank |
| `docs/EXPERIMENT_PROTOCOL_P2_ADAPTER.md` | Add T004A/T004F/T004H gate section | Keep frozen criteria and later-study boundaries explicit |
| `docs/VLM_COMPATIBILITY_METRICS.md` | Add compact Validation metric/gate summary | Separate patch/identity evidence from unmeasured functional metrics |
| `reports/runs/VEA-P2-T004I-low-rank-validation-gate-closure.md` | This report | Preserve supplied table, arithmetic and provenance limits |

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Git entry | Root/Git/common, baseline HEAD and origin T004G, T004I branch absence, worktree, status, quiet diffs, untracked paths, operations/locks | 0 for positive gates; absent-ref probe 1 as expected | Exact baseline, only approved Active Task residual, no relevant locks | Transient output |
| Frozen gate arithmetic | PowerShell here-string containing standard-library `json`, `re`, `decimal`, `pathlib` script piped to `py -3 -` | 0 | 14 approved rows parsed; six failures per source; rank 256 deltas `-0.030` each; `NO_RANK_ACCEPTED` | Command output |
| Full rounded-table transcription | Standard-library `re`/`pathlib` comparison of Active Task and report tables via `py -3 -` | 0 | Exactly 14 matching rows and 70 matching rounded numeric values | Command output |
| Text/provenance assertions | Corrected documentation-only standard-library `pathlib` check via `py -3 -` | 0 | Five paths, six complete identifiers, `NO_RANK_ACCEPTED` and `NOT_EVALUATED` in all five, one VEA-D024 row, Validation/Test boundary and supplied-evidence limits | Transient output |
| Precommit Git whitespace and frozen config | `git diff --check`; `git diff --cached --check`; `git diff -- configs/vlm_compat/p2_query_adapter_low_rank_linear_v1.json` | 0 each | No whitespace errors or frozen-config diff | Transient output |
| Exact staged scope and postcommit remote | Five-path staged allowlist, baseline-to-HEAD diff/whitespace, matching pushed remote ref | Verified during delivery after report draft | Final Git handoff supplies commit and remote proof | Transient output |

## Test and command transcript summary

```text
PowerShell here-string | py -3 -
Reads only .codex/ACTIVE_TASK.md and the frozen T004A JSON; parses 14 supplied rows
with re.fullmatch, Decimal; asserts frozen rank grid, OLS - 0.02 criterion,
conjunction, six strict failures per source and rank-256 delta -0.030.
exit 0; APPROVED_TABLE_ROWS=14 GATE_CHECK=PASS
SIGLIP2: OLS=0.890 MIN=0.870 R@1=[0.130,0.270,0.440,0.710,0.810,0.860]
DINOV3: OLS=0.870 MIN=0.850 R@1=[0.120,0.230,0.430,0.640,0.790,0.840]
PowerShell here-string | py -3 -: compare all seven fields in each approved
source/condition table row against the report; exit 0,
REPORT_TABLE_MATCH=PASS ROWS=14 EXACT_ROUNDED_VALUES=70.
First text-check invocation: exit 1 from a Windows path-key mismatch in the
transient check script; no document failed an assertion. Corrected invocation:
exit 0; TEXT_ASSERTIONS=PASS FIVE_PATHS=5 IDS=6 NO_RANK_ACCEPTED=5
NOT_EVALUATED=5 D024=1.
git diff --check: exit 0.
git diff --cached --check: exit 0.
git diff -- configs/vlm_compat/p2_query_adapter_low_rank_linear_v1.json:
exit 0, empty output.
```

## Artifacts

- This report is the only new artifact. It contains rounded supplied summaries, no raw Apple payload or per-image array.
- T004F/T004H run IDs and result hashes above are provenance references, not independently rehashed T004I artifacts.
- No review ZIP, data/model artifact or downstream metric was produced.

## Deviations

- `.codex/PROJECT_STATE.json` was absent at entry; it was not created.
- No frozen threshold, rank grid, Test policy or five-path scope deviation.

## Unverified items

- Raw T004F/T004H result JSONs, factor files and per-image Validation rows were not available or independently rehashed/recomputed in this task.
- Connector cosine/NMSE, native exact match, Token F1, fair float32 adapter latency, paired bootstrap intervals and Test metrics are `NOT_EVALUATED` in T004H. No frozen-TinyLLaVA generation inference follows from this gate.
- Canonical integration merge remains unverified; no full software test suite was run for this documentation task.

## Remaining risks

- Implementation: no code changed; the documentation depends on supplied rounded summaries rather than raw Apple bytes.
- Environment: future independently approved execution must verify its own exact inputs and outputs.
- Scientific interpretation: 100-image R@1 resolution is `0.01`; the gate excludes these six frozen ranks under T004A, not all possible low-rank designs. No threshold adjustment or MLP escalation is implied.

## Suggested next action

Submit T004I for independent review. A later hypothesis or MLP design needs separate explicit approval and configuration freeze.
