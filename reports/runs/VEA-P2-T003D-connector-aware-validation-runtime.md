# VEA-P2-T003D Run Report — Connector-aware Validation runtime

## Status and delivery

- Status: PASS for the assigned offline implementation and validation; independent review pending.
- Accepted T003C baseline and pushed remote: `d040bf1cc5ea92448e72de7efabdb298a67447e9`.
- Delivery branch: `codex/vea-p2-t003d-connector-aware-validation-runtime`.
- Commit message: `feat(p2): VEA-P2-T003D implement connector-aware validation runtime`.
- Result schema: `vea.p2.query_adapter_connector_aware_validation_runtime.v1`.
- Selected artifact metadata schema: `vea.p2.query_adapter_connector_aware_selected_affine.v1`.

## Frozen inputs

- Accepted T003A design config SHA-256: `4684a1ae43a124725df93d15c2287a30aa31ae38f6c58bd2f05b83e749f8494b`.
- Accepted T003C runtime binding config SHA-256: `f0336604f4d4f37350d2745d8b54edeb1cc19f75fe27d5475609d81402946614`.
- Pinned TinyLLaVA anchor config SHA-256: `78242f4062fe748826aad0b51ae50ddce1c4bcbb06143396e87d9a862585ac8a`.
- T003C supplies exact T001D plan/manifest/bank identities, T002D Validation identity, and both accepted Full Linear initializations. The new runtime does not refit Full Linear.

## Runtime and CLI semantics

Real mode requires explicit `--t001d-run-root`, `--t002d-validation-root`, `--cache-root`, `--output-root`, and `--device`. It checks that the output root is new or empty, invokes T003C `preflight_runtime()` before the existing local-files-only frozen TinyLLaVA connector loader, and performs one bounded zero-step training/inference connector parity check per source before updates. The loaded model and connector remain frozen; only affine `W` and `b` enter each AdamW optimizer.

For each source, the runtime computes Train normalizers once, evaluates `OLS_REFERENCE` without an optimizer, initializes all four gradient conditions independently from the same Full Linear arrays, trains with the accepted T003B epoch routine for exactly epochs 1–8, and records Validation at epochs 0–8. It computes the frozen loss and patch, retrieval, and connector metrics using Train target means where required. Accepted T003B selectors choose checkpoints and the connector-aware weight. Each selected affine is saved as numeric `affine.npz` plus hashed JSON metadata and reloaded before final `result.json` publication. The result records all Validation rows, selections, normalizers, artifact hashes, runtime/input identities, and zero Test, generation, and vision counters. Metrics are evidence; software PASS applies no performance threshold.

`--fake-smoke OUTPUT_ROOT` creates small deterministic CPU tensors and a parameter-free connector fixture entirely in memory. It calls neither a real model loader nor T003C runtime-root preflight. No machine-local path is committed.

## Assigned validation

| Command | Exit | Result |
| --- | ---: | --- |
| `py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_connector_aware_validation_runtime.py scripts/run_query_adapter_connector_aware_validation.py tests/unit/test_p2_query_adapter_connector_aware_validation_runtime.py tests/integration/test_p2_query_adapter_connector_aware_validation_cli.py` | 0 | New files compile. |
| `PYTHONPATH=src py -3 -m pytest -q tests/unit/test_p2_query_adapter_connector_aware_validation_runtime.py tests/integration/test_p2_query_adapter_connector_aware_validation_cli.py` | 0 | 17 passed, deterministic synthetic CPU only. |
| `PYTHONPATH=src py -3 -m pytest -q tests/unit/test_p2_query_adapter_connector_aware_linear.py tests/unit/test_p2_query_adapter_connector_aware_runtime_binding.py tests/integration/test_p2_query_adapter_connector_aware_runtime_binding_contract.py tests/integration/test_p2_query_adapter_tinyllava_connector_boundary.py` | 0 | 45 passed. |

An initial focused run had one test assertion that expected different error wording; the runtime rejected the drift as intended. The assertion was corrected and the focused suite passed. No full suite was run. No dependency was installed or added.

## Access boundary and remaining real-runtime work

No real Apple GPU, model, feature bank, dataset, Train/Validation root, Test evidence, generation, or chat path was accessed. Only temporary synthetic fixtures were executed. Real T003E execution will require externally supplied T001D and T002D roots, a local pinned model cache, a new output root, and an explicit accelerator device. Connector parity, memory use, and scientific metrics on those real inputs remain unverified. T003D remains pending independent review; no merge, self-acceptance, or T003E work was performed.
