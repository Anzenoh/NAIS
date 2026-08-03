# VEA-P1-T003B-B-R3R1 — Apple DINO execution-gate repair

## Status

`PASS` for the offline implementation and Tier 0–1 gate.  This report does not
authorize Apple execution, Stage A, smoke, or extraction.

## Baseline and preservation

- Baseline: `b40e74689f063e5cdaacc6005c9d12fe81c02078`
  (`origin/codex/vea-p1-t003b-b-r2`).
- Delivery branch: `codex/vea-p1-t003b-b-r3r1`.
- Original repository was recorded before isolation (branch
  `codex/vea-p1-t003b-a-r3r1`, HEAD
  `464ebf3fdeaf53a16d04838fc67aed725fe5d08c`, staged `0`, with its dirty
  bytes preserved).  The prior local R3 worktree was not modified.
- Google Drive reported metadata-only worktree modifications despite matching
  Git blobs and `git diff --quiet`; LF byte checks were used for the isolated
  candidate.  No original-repository byte was changed.

## Repair and final identities

1. The prior R3 allowlist omitted `p1_feature_extraction.py`; R3R1 includes it.
2. Apple canonical snapshot identity is
   `d1326e9358930d826eaae6c18b44e384fa38b854140304951210347d7ecdac0d`.
3. Final Apple extractor profile SHA-256:
   `74ba6f23d31910450c53e6b35bba6218e78f0edd281c9069593ea58faee62645`.
4. Final P1 extraction profile SHA-256:
   `f5d941d02f014a314c7b9d8e26e55c2bf85e6dbb53acd9e844efe61458767f99`.
5. Final Stage A implementation SHA-256 values are
   `scripts/extract_features.py` =
   `ab735371603091df0eedaa34d12eee02d35f03f2df8b31f9c35a24e94fe6a9a0`
   and `p1_feature_extraction.py` =
   `f252f1199f43894eda0a1fa872e053493b2f7e00d2ef1897ace5d7a3c4dba7b8`.
6. Rebinding order was Apple profile, P1 source/profile, Stage A source, then
   runbook bindings.  No accepted live binding retains the former snapshot
   identity, former Apple profile SHA, or DINO `bin/python3` path.

## Worker result

The new real-`run_worker()` Apple-selector regression reached unmocked response
construction, response validation, `_write_response`, and JSON publication.
It mocked only test hardware/Torch, checkpoint snapshot validation, model loading,
and forward execution; legacy runtime/snapshot validators raise if reached.
The worker passed unchanged, so no conditional worker/profile change was made.

## Validation

| Command | Exit | Result |
| --- | ---: | --- |
| `py -3 -B -m compileall -q scripts/extract_features.py scripts/dinov3_extraction_worker.py src/vision_encoder_alignment/p1_feature_extraction.py` | 0 | pass |
| Assigned 14-file Tier 1 pytest command | 0 | 440 passed, 0 skipped |
| `git -c core.whitespace=cr-at-eol diff --check` | 0 | pass |

## Operations and readiness

- Apple/GPU/model/dataset/checkpoint/network/install/download operations: `0`.
- `READY_FOR_P1_EXTRACTION=false`.
- Planned commit message:
  `fix(extraction): VEA-P1-T003B-B-R3R1 close Apple DINO execution gates`.
- Push result is reported with the final Git delivery response.
