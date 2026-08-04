# VEA-P1-T004A one-shot analysis runbook

The checked-in entry points accept the exact task commit and tree at execution time; they intentionally do not embed a future commit identity. No installer, download, model load, re-extraction, or stage-by-stage invocation is part of this runbook.

## Required arguments

- `--expected-commit <40hex>`
- `--expected-tree <40hex>`
- `--repository-root <clean detached checkout>`
- `--run-root <accepted T003B run root>`
- `--evidence-root <accepted T003B evidence root>`
- `--result-parent <bounded external result parent>`
- `--python <existing accepted Python>`
- `--platform-mode apple|windows_local`
- `--resume-or-start --run-all`

The wrapper rejects sourcing before changing shell options, verifies the exact clean checkout, exports the frozen deterministic environment, and replaces itself with the Python orchestrator. An unchanged command resumes the deterministic result identity. A failed status preserves the first root cause and evidence path.

## Windows-local execution

Use the existing Python 3.12 runtime from a clean exact-commit checkout. The local lane never loads a model and records repeat control as `NOT_EXECUTED_LOCAL_RESOURCE_UNAVAILABLE`. The normal successful output is exactly six lines ending in the external sealed-package SHA-256.

## Verification

Run `scripts/verify_p1_analysis_package.py --package <sealed_package.tar.gz>`. Confirm `delivery_summary.json` records one test unseal, the canonical config/input/payload identities, `PASS_IDENTICAL_252_FILES`, and `checksum_integrity_is_repeat=false`.
