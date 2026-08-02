# VEA-P1-T003B-B-R1 — Preserve bounded worker failure diagnostics

## Status

`PASS` is proposed locally. No Apple, CUDA, smoke, retry, resume, full extraction,
dataset, checkpoint, model, network, package, or credential operation was performed.

## Baseline and observed failure

- Accepted baseline: `codex/vea-p1-t003b-a-r2` at
  `ff75f06e34ffd05e86844e07a3abb306d34d547f`.
- Observed independent Stage B result: CLIP `worker_process`, raw return `4`,
  stable parent exit `5`, generic `worker_process_failed`, zero completed models.
- This repair does not claim the underlying CLIP cause is known.

## Repair and changed paths

- `scripts/extract_features.py`: accept only one exact canonical child error document;
  preserve its bounded safe `error` as `sanitized_message`, otherwise use the prior
  `worker_process_failed` fallback. Raw stdout/stderr is never retained.
- `tests/integration/test_p1_feature_extraction_cli.py`: safe propagation, malformed,
  extra/missing field, exit mismatch, path, secret, oversized, multiline,
  noncanonical, arbitrary-stderr, fallback, and exit-semantics cases.
- `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`:
  final `extract_features.py` SHA binding.
- `tasks/VEA-P1-T003B-B-R1-worker-failure-diagnostic-preservation.md`: active-task copy.

## Validation

| Command | Exit/result |
| --- | --- |
| `python -B -m compileall -q scripts/extract_features.py` | `9009`; `python` command unavailable on this Windows host |
| `py -3 -B -m compileall -q scripts/extract_features.py` | `0` (equivalent local Python launcher) |
| `py -3 -B -m pytest -q tests/integration/test_p1_feature_extraction_cli.py` | `0`; `36 passed` |
| `git -c core.whitespace=cr-at-eol diff --check` | `0` |

`src/vision_encoder_alignment/p1_feature_extraction.py` remained exactly
`2c754057f59ccdd4cfc07794816184c2392af583fe26e155183581e2f66f4ab8`.

## Final Apple transfer SHA-256 values

```text
95a5f9de7d9d1cd964417d478b3047ec1ff62f578e5c18fbab2176f954ee01d2 scripts/extract_features.py
2c754057f59ccdd4cfc07794816184c2392af583fe26e155183581e2f66f4ab8 src/vision_encoder_alignment/p1_feature_extraction.py
dcd5dce50b9e5e2e4ac7c505efac751e9f621fd1a780495484fb02758873a13e src/vision_encoder_alignment/p1_runtime_preflight.py
1e3d5b9b95315a61af436f61f32b1fb184fb2c9d3e0b53118437d3c8b3ecabd5 scripts/inventory_p1_checkpoint_cache.py
7a0e8cfb54798d857ddccdf6d8288d13f8587d3ff257be56c1ef49c8db1ba4f0 scripts/validate_p1_feature_extraction.py
```

Only the five task commit paths are to be staged. Existing unrelated dirty paths were
preserved. The next action is independent commit review; Apple transfer and another
Stage A/Stage B authorization remain outside this task.
