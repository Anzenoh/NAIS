# VEA-P1-T003B-A-R2 — Safe Hugging Face snapshot preflight repair

## Status

`PASS` is proposed locally. This task performed no Apple, dataset, checkpoint,
model, GPU, Stage A, Stage B, smoke, resume, or full-extraction operation.

## Baseline and audit classification

- Baseline branch/commit: `codex/vea-p1-t003b-a-r1` / `9a5f64638687c9e5ef4d9c1c237facaac0a7f339`.
- Task branch: `codex/vea-p1-t003b-a-r2`.
- Accepted audit: `audit.log` SHA-256
  `d3d2512696176638e6abc9013b31568e30ce7a4319e0869394e9df82959f7cce`.
- The audit records repository/path/dataset/GPU/plan success, then resource exit
  `4` and the 153-byte canonical error `checkpoint snapshot unavailable:
  clip_vit_b16`. The root cause was an inventory/extraction mismatch on standard
  Hugging Face snapshot leaf links, not a readiness claim.

## Repair

- `checkpoint_snapshot_files` is the shared bounded authority for cache inventory
  and extraction preflight.
- Cache and snapshot directory components must be canonical non-links. A snapshot
  leaf may resolve strictly only to a regular file below that same cache root.
- Broken, directory/non-regular, escaping, linked-directory, and empty-snapshot
  cases fail closed. Runner mapping is the sanitized resource exit `4`.
- Plan/preflight error output now writes the same canonical JSON to an available
  `--output-json` path; a collision preserves the competing bytes and original exit.

## Changed paths and SHA-256

| Path | Before | After |
| --- | --- | --- |
| `docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md` | `72bcd1a6301c2604df17f8f734ada0807295d564f87460f8fed6c79e9eb7a833` | `9a62e51d2c4e05bc2e1937454da9ccb66484c08fdc3b1b3ef11b5e055469df45` |
| `scripts/extract_features.py` | `5c3d92c5ea03b1a4265855be5d21f5b782c41dbe9cb86cbe0a374185bce418f3` | `ce709105c14595acbc585692185d624d8101b57f14f520ba527263de33ba2607` |
| `src/vision_encoder_alignment/p1_runtime_preflight.py` | `86d4e0a99a4c8d1f9c199d708ca355e14b2c63b9acd9088104a795d446c6e031` | `dcd5dce50b9e5e2e4ac7c505efac751e9f621fd1a780495484fb02758873a13e` |
| `tests/unit/test_p1_runtime_preflight.py` | `27cd307f50599873483bc3f39eead1de9b3ed47cd748ce3e93a4654c355c3469` | `5249a272cc2c83341f6fac040687b6be17bbfad3b9d463f6838a7c704f687a71` |
| `tests/integration/test_p1_feature_extraction_cli.py` | `c46ae853e3a477415ac816fb9cc3158064ac91ad09a9b8cf081293e7d430a85b` | `a0b0f1aa332027afbfe659489359773a8b38011b6a354ebaff49e2cb03d2ac0e` |
| `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md` | `c254e815df03509e7a97c63fb3881087e18722ea5d55e6f3fdd31e26337a5896` | `0bc992b5596c73e7e9cc5d2138f85296b4050d39e0290c535648444cdfdd2ed6` |

The task record is an unchanged byte-for-byte copy of `.codex/ACTIVE_TASK.md`:
`85d1dba069d6b1f038ee5eff46e737626007db904fec81494867c165a88f7f0b`.

## Tier 0–1 validation

| Command | Exit/result |
| --- | --- |
| `py -3 -B -m compileall -q src\vision_encoder_alignment\p1_runtime_preflight.py scripts\extract_features.py tests\unit\test_p1_runtime_preflight.py tests\integration\test_p1_feature_extraction_cli.py` | `0` |
| `py -3 -B -m pytest -q -p no:cacheprovider tests\unit\test_p1_runtime_preflight.py tests\integration\test_p1_feature_extraction_cli.py` | `0`; `113 passed` |
| `git -c core.whitespace=cr-at-eol diff --check` | `0` |

The test set covers a deterministic safe HF leaf link, regular leaf, empty snapshot
resource failure, broken/directory/escaping link rejection, linked snapshot directory
rejection, matching canonical error output, collision no-clobber, and the existing
synthetic preflight success path. No test requires symlink creation permission.

## Scope, delivery, and user-only validation

The initial non-allowlisted dirty-path snapshot was retained unchanged; only the eight
allowed paths are to be staged. Unauthorized operation counts are all zero.

User-only transfer target: `/home/chltjsdh/vea/repo` on `apple`. After transferring
the pushed commit, run the pre-evidence-root SHA checks in the updated runbook section
2 (the five Stage A implementation paths use `sha256sum --check --strict`), then run
that runbook's read-only Stage A commands once. Do not run Stage B or extraction.

`READY_FOR_P1_EXTRACTION=false`.
