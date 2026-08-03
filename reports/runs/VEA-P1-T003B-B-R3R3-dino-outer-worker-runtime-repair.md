# VEA-P1-T003B-B-R3R3 Run Report

## Status

`BLOCKED`

The bounded implementation candidate is complete, but the mandatory focused gate
cannot exit zero because the exact accepted baseline contains a protected DINOv3
profile SHA binding mismatch. No commit or push was performed.

## Git and isolation evidence

- Baseline HEAD: `e324566e98266089552dfff551f62a5c05d9c6ff`
- Baseline tree: `05ccfb6d70e4487dbacb8d37f2d16b6a6a632a9e`
- `origin/codex/vea-p1-t003b-b-r3r2`: exact baseline HEAD
- Task branch was absent locally and remotely at preflight.
- Candidate branch: `codex/vea-p1-t003b-b-r3r3` (uncommitted and unpushed)
- Candidate root: separate LF-semantics clean clone outside the source repository
- Original dirty repository status: `319` NUL-safe entries, SHA-256
  `34e86c327e6b9d53674c2fb8bb08686ca9aad1bddc983dd066931cd133041584`
  before and after candidate work.

## Candidate behavior

- `lane_command_plan()` emits CLIP, SigLIP2, and DINOv3 outer prefixes with exact
  main Python, `-B`, model order, and one owned `--model` pair.
- `_run_or_resume()` consumes those exact prefixes while independently validating
  main and isolated DINO interpreters and runtime roots.
- DINO extractor configuration remains bound to the isolated DINO interpreter
  root; the actual model child boundary is unchanged.
- Contract and runbook now distinguish main-runtime orchestration/preprocessing
  from the isolated model-bearing DINO child.

## Candidate changed paths

1. `docs/contracts/P1_FULL_FEATURE_EXTRACTION_CONTRACT.md`
2. `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`
3. `scripts/extract_features.py`
4. `src/vision_encoder_alignment/p1_feature_extraction.py`
5. `tests/integration/test_p1_feature_extraction_cli.py`
6. `tests/unit/test_p1_feature_extraction.py`
7. `tests/unit/test_p1_t003b_stage_a_runbook.py`
8. `tasks/VEA-P1-T003B-B-R3R3-dino-outer-worker-runtime-repair.md`
9. `reports/runs/VEA-P1-T003B-B-R3R3-dino-outer-worker-runtime-repair.md`

No profile, runtime, configuration, DINO model implementation, or invocation
boundary path changed. `tests/integration/test_dinov3_isolated_boundary.py` did
not require a byte change and is omitted from the candidate diff.

## Final candidate identities

- `scripts/extract_features.py`: `29725` bytes,
  `d7b869b83d2a05fd48e5863c10c8ba6890abb038c8cc74247b011d8bee84857b`
- `src/vision_encoder_alignment/p1_feature_extraction.py`: `64321` bytes,
  `3d81e229363fa31bb6fabe3be4c3e7b5dc63cc3e2195553c7ec5ab83768531c3`
- Stage A runbook: `14800` bytes,
  `f760bde631d7044020249bbdb363c458e45c172c098d9fe0f2dc3865f4f415c3`

## Byte-unchanged profiles

- Extraction profile: `3996` bytes,
  `f5d941d02f014a314c7b9d8e26e55c2bf85e6dbb53acd9e844efe61458767f99`
- Main runtime profile: `13938` bytes,
  `7e578ea3451e595e243e0e6a39058b0eb12a8641ce433bc60b6039b23e2cbdf5`
- DINO runtime profile: `14353` bytes,
  `86069cd7396c7d254d13c63b4e5ea0cf0120a0a31f181c708945273d15825c15`

## Validation

Configured interpreter: `py -3`, Python `3.12.0`. Offline variables and empty
`CUDA_VISIBLE_DEVICES` were set; temp, bytecode, and pytest roots were external.

1. Changed implementation/unit/CLI early gate: exit `0`, `108 passed`.
2. Changed Python/test compile: exit `0`.
3. Assigned eight-file focused gate: raw exit `1`, `222 passed`, `1 skipped`,
   `31 errors`.
4. `git -c core.whitespace=cr-at-eol diff --check`: exit `0`.
5. Complete repository suite: not run because the active task requires it only
   after the focused gate passes.

The one skip is the existing Windows directory-symlink capability skip in
`test_workspace_symlink_escape_rejected`. Every error is fixture setup fallout
from the same protected identity mismatch; no candidate implementation assertion
failed.

## Blocking baseline identity

The baseline Git blob and working bytes of
`configs/extractor_profiles/dinov3_vit_b16_r5931719_isolated_v0.json` are exactly
`12506` bytes with SHA-256
`82f804ac4440d2a198585e78e968683da53fc27851d5a4ca360f0bd821e7d2b0`.
The baseline `tests/conftest.py` fixture instead requires
`60d0d0d7d37c6b4d47997e09619204f9e1eddb96118d5ee3528d9e6fcd5fe829`.
Both paths are protected and outside this task's allowlist. Changing either would
expand scope and violate the stop conditions, so the test was not weakened.

## Operations and unresolved items

- Apple/GPU/model/dataset/checkpoint/install/download/smoke/extraction: `0`
- Authorized Git origin identity verification: performed; commit/push: `0`
- Staged paths: `0`
- Full batch size and exclusive-window duration remain unresolved pending a
  repaired smoke and separately authorized DINO child memory/timing calibration.
- `READY_FOR_P1_EXTRACTION=false`

## Next action

Independently reconcile the protected baseline profile-to-fixture SHA identity in
a separately authorized task, then rerun this unchanged candidate's focused gate.
