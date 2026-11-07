# Run Report — VEA-P2-T004M

## Status

`PASS` for bounded tooling implementation and synthetic Tier-0/1 validation. Independent review is `REVIEW_PENDING`. Real human evaluation is `NOT_RUN`; the rank decision remains `NO_RANK_ACCEPTED`.

## Metadata

- Task ID: `VEA-P2-T004M`
- Run/report timestamp UTC: `2026-11-07T12:22:07Z`
- Repository root: verified independent local workspace; machine-specific absolute path omitted
- Git commit before work: `be61fcc4c4458303ba78dbd731cc11318c54b118`
- Git dirty before work: `DIRTY_EXPECTED`, solely unchanged untracked `.codex/ACTIVE_TASK.md`; tracked and staged diff exit 0
- Execution environment: Windows PowerShell, Python 3.12, CPU synthetic fixtures; optional local Node syntax check inside the test suite
- Branch: `codex/vea-p2-t004m-blinded-human-audit`; approved delivery `COMMIT_AND_PUSH / PASS_ONLY`

## Goal

Implement an offline, method-blinded human assessment of actual-image description correctness and a read-only analyzer for exported ratings of the already-generated T004K Validation result. No human scoring, model generation, GPU run, fitting, Test access, rank promotion or merge occurs in this task.

## Summary

The frozen protocol pins T004K result SHA-256 `6d526c7382811cb79a78c58cd2bc6802ee793117d21ea669ff03cbcb94cddff0`, 100 Validation images, five captions per image, seed 43, image-grounded 0–5 ratings and the exploratory role. The existing T004J viewer now exposes a reusable read-only loader that checks the result SHA, rows and aggregate hashes, the accepted Validation manifest and image bytes, 100 ordered IDs, production status, and `NO_RANK_ACCEPTED`. Its original builder behavior and tests remain intact.

The blind builder requires explicit roots and the pinned result SHA. It derives a deterministic independent A–E permutation from seed 43 plus each image ID, embeds only approved JPEG bytes and anonymous captions in offline HTML, and writes the true lane mapping to a separate `assignment.json` in a fresh external directory. The HTML contains no lane identities or numeric generation metrics. It shows one image, five independent score selectors, the full rubric, coverage, navigation/filtering, bookmarks, notes and strict JSON backup import/export. Browser state is namespaced by the T004K result SHA and mapping SHA. The private mapping file must remain separate from the assessor. This is organizational/UI blinding, not tamper-proof anonymity.

The analyzer reuses the verified T004K loader and validates the exact assignment, ordered IDs, full A–E bijection, rubric/mapping hashes and every exported score. An incomplete export yields anonymous coverage and `INCOMPLETE_NO_FINAL_INFERENCE` only. A complete 500-slot export yields per-lane 0–5 distributions, mean/median, native absolute image-grounded quality, and within-image OLS-versus-low-rank deltas with win/tie/loss counts and descriptive paired-image percentile bootstrap intervals (2,000 resamples, seed 43). It makes no significance, equivalence, non-inferiority or rank-selection claim. No new runtime dependency was added.

### Frozen score rubric

Every anonymous caption is judged against the actual displayed image, including empty or whitespace captions. `Unrated` is null, distinct from score `0`.

| Score | Definition |
|---:|---|
| 5 | virtually all essential objects/scene/actions correct; no meaningful hallucination. |
| 4 | largely correct with only minor omissions or errors. |
| 3 | main scene broadly correct but notable mistake or omission. |
| 2 | partial match but major misunderstanding or hallucination. |
| 1 | very limited overlap; most essential content wrong. |
| 0 | main objects/scene wrong or description nearly unrelated. |

## Changed files

| Path | Change | Reason |
|---|---|---|
| `configs/vlm_compat/p2_low_rank_blinded_human_audit_v1.json` | New frozen protocol | Pins identities, rubric, seed and exploratory limits |
| `scripts/build_low_rank_generation_viewer.py` | Minimal loader refactor | Reuses complete T004J/T004K read-only verification |
| `scripts/build_low_rank_blinded_human_viewer.py` | New offline builder | Separates anonymous HTML from private assignment map |
| `scripts/summarize_low_rank_blinded_human_scores.py` | New analyzer | Validates backups and computes descriptive paired summaries |
| `tests/unit/test_p2_query_adapter_low_rank_generation_viewer.py` | Added regression | Verifies the existing builder and shared loader guards |
| `tests/unit/test_p2_query_adapter_low_rank_blinded_human_audit.py` | New synthetic tests | Covers blinding, import/export validation and paired calculations |
| `reports/runs/VEA-P2-T004M-blinded-human-audit-tooling.md` | New run report | Records exact scope, validation and limits |

## Validation evidence

| Requirement | Command/check | Exit status | Result | Evidence path |
|---|---|---:|---|---|
| Tier 0 compile | `py -3 -m compileall -q` on three scripts and two tests | 0 | All five files compile | Changed Python paths above |
| Tier 1 T004M | `py -3 -m pytest -q` on existing viewer and new blind-audit tests | 0 | 31 passed | Test paths above |
| T004J regression | `py -3 -m pytest -q` on frozen-generation unit, CLI and viewer tests | 0 | 25 passed | Prescribed regression tests |
| Browser syntax | Optional `node --check` inside T004M test | 0 | Extracted offline JS parses on local Node 24 | New blind-audit test |
| Identity and policy | Synthetic adversaries | 0 | Wrong SHA/rubric/mapping, Test ID, old namespace, malformed score, path collision and privacy leakage rejected | New blind-audit test |

## Test and command transcript summary

```text
py -3 -m compileall -q scripts/build_low_rank_generation_viewer.py scripts/build_low_rank_blinded_human_viewer.py scripts/summarize_low_rank_blinded_human_scores.py tests/unit/test_p2_query_adapter_low_rank_generation_viewer.py tests/unit/test_p2_query_adapter_low_rank_blinded_human_audit.py
exit 0

py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_generation_viewer.py tests/unit/test_p2_query_adapter_low_rank_blinded_human_audit.py
31 passed in 0.59s; exit 0

py -3 -m pytest -q tests/unit/test_p2_query_adapter_low_rank_validation_generation.py tests/integration/test_p2_query_adapter_low_rank_validation_generation_cli.py tests/unit/test_p2_query_adapter_low_rank_generation_viewer.py
25 passed in 1.74s; exit 0
```

During refinement, two test runs failed on test-local issues: the first two-image summary test compared its synthetic count to the frozen production count, and a later path-collision assertion expected the wrong error wording. The synthetic helper now uses its verified fixture ID count while production entry still requires 100 IDs; the assertion was corrected. The mandatory runs above supersede those intermediate failures.

## Artifacts

- Tracked protocol SHA-256: `4cb15b1faf6e3fcbc06518ab8c81b67a478549a580af6b7a9810e702bc1630b6`.
- Future external files: `blinded_viewer.html`, private `assignment.json`, anonymized score backup JSON, optional descriptive analysis JSON.
- No T004K production file, actual human score, generated HTML, private mapping, image data or score export was created or committed by this task.

## Deviations

- `.codex/PROJECT_STATE.json` was absent; the Active Task permits reporting this without changing it.
- No implementation deviation.

## Unverified items

- Real T004K Apple files and Validation JPEGs were not available to this Windows task; the pinned production preflight was not executed against them.
- Real human evaluation `NOT_RUN`; Apple `NOT_RUN`; Test `NOT_ACCESSED`; GPU/model/fitting/generation `NOT_RUN`.
- Browser interaction was not manually audited on the assessor's machine. The local JS syntax and Python export contract were checked synthetically.

## Remaining risks

- Implementation: file-browser storage policy and human workflow need independent review; backups remain essential when localStorage is unavailable.
- Environment: the later Apple run must supply the exact reviewed T004K roots and approved Validation image manifest.
- Scientific interpretation: one rater gives a descriptive first pass only; no inter-rater reliability or population-level accuracy is established. Human scores cannot reverse the frozen Validation retrieval gate or promote LOW_RANK_256.

## Suggested next action

Submit the pushed T004M commit for independent review before any real human audit.
