# VEA-P2-T001D-R4 run report

## Result

Status: PASS (offline Tier 0–1 only)

Baseline `9f488cc46f1711c3f43e8ed24561198757988927` was implemented on
`codex/vea-p2-t001d-r4-target-clip-raw-self-control`. No Apple, CUDA, model,
dataset, checkpoint, package, credential, or external-resource network activity
occurred during validation; network use is limited to the required Git push.

## Contract evidence

- The invalid cross-stage connector bitwise check was removed.
- TARGET_CLIP/RAW now requires exact persisted-patch identity, exact effective
  connector-input identity, and exact duplicate outputs from one paired live
  connector invocation. The RAW half is published as the Stage 05 output.
- The unchanged Stage 03 connector remains only the reference for the existing
  cosine, norm-ratio, and NMSE metrics. Its measured maximum absolute difference
  is retained per image in a non-aggregated audit object.
- The shared read-only reuse inspector validates plan binding, ordered manifest,
  identities, chunking, schemas, exact Stage 01–03 units, artifacts, markers,
  and hashes before the publication path consumes its eligible unit list.
- A fake R3-like root proved `193` completed units, `189` eligible and reused
  feature units, zero repeated encoder forwards/extractor calls, recomputable
  four-unit Stage 04, and no mutation of source evidence.
- The fake interruption/resume flow preserved the first TARGET_CLIP/RAW Stage 05
  artifact, marker bytes, marker timestamp, and single connector execution.
- Expected immutable unit inventory remains `2345`; scientific formula changes:
  `0`; real runtime activity: `0`.

## Required audit

```text
eligible_feature_units = 189
reused_feature_units = 189
reused_encoder_forwards = 0
patch_self_control = PASS
effective_input_self_control = PASS
same_call_connector_self_control = PASS
persisted_connector_difference_is_metric_only = true
first_stage05_reexecuted = false
expected_unit_count = 2345
scientific_formula_change_count = 0
real_runtime_activity = 0
```

## Validation

- `py -3.12 -m py_compile ...`: PASS
- Assigned four-file pytest command: PASS, `141 passed in 96.89s`
- Coordinator and stage-worker `--help`: PASS
- `git diff --check`: PASS

One initial invocation of the same pytest command was terminated by the local
tool timeout at 64 seconds without a test result; the complete rerun above is the
authoritative validation result.
