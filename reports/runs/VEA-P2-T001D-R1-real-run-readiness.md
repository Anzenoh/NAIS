# VEA-P2-T001D-R1 real-run readiness report

- Status: local fake/CPU Tier 0–1 PASS; no scientific execution.
- Baseline: `e9afe5820dd03149811f06f0f0daa374d1c459e2`.
- Branch: `codex/vea-p2-t001d-r1-real-run-readiness`.
- `.codex/PROJECT_STATE.json`: absent at the accepted baseline; not created.
- Optional reuse binding: planned `null` now compares directly with observed
  `null`; explicit roots are resolved only when both sides are paths. Null/path
  and unequal path/path pairs fail closed.
- Previous storage requirement: 13,195,152,589 bytes (12.288943 GiB).
- Corrected storage requirement: 22,276,378,676 bytes (20.746494 GiB), including
  16,904,018,176 durable bytes, a 231,657,728-byte largest partial allowance,
  then the 30% safety margin.
- Corrected accounting covers three patch/global banks, native connector bank,
  all four anchor registries, four Stage 05 tensor copies, fixed-width image and
  metric metadata, generations, markers, progress, failures, and final output.
- Stage 04: all four units and done markers bind the exact ordered 128 anchor
  image IDs; total unit count remains 2,345.
- Runtime binding: real CUDA plans bind the approved main interpreter, overlay,
  cache roots, offline variables, and package metadata versions.
- Environment isolation: injected subprocess coverage removes the main overlay
  and main-runtime paths from DINO while retaining the overlay for main stages.
- DINO accounting: 63 fake one-forward chunks report 63 model loads, one stage
  worker, and 63 extractor workers.
- Apple policy: persistent repositories under `/home/chltjsdh/vea/p2_repos` and
  runs under `/home/chltjsdh/vea/p2_runs`; no plan-bound path uses `/tmp`.
- Supervisor: generated scripts receive `bash -n`; launch uses `nohup setsid`,
  `/dev/null` stdin, unique logs, real pipeline statuses, and only code 75 is
  automatically resumed at most three times. Code 130 remains partial.
- No-reuse fake flow: interruption after 500 units, partial verify, resume of
  1,845 units, complete verify, finalize, and status PASS. Previously completed
  generation bytes and mtimes remained unchanged; resumed and uninterrupted
  `result.json` were byte-identical under the same plan ID.
- Assigned pytest: `116 passed in 90.37s` on the final clean-scope rerun.
- Tier 0 import safety: assigned `py_compile` and both CLI `--help` commands
  passed; `git diff --check` passed.
- Real Apple/CUDA/model/dataset/checkpoint execution: 0.
- Network use: target-branch absence precondition and eventual authorized Git
  delivery only; no experiment runtime network access.

## Delivery evidence

Pending exact-path commit and push.
