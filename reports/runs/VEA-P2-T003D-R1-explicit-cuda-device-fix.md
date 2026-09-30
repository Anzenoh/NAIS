# VEA-P2-T003D-R1 Run Report — Explicit indexed CUDA device fix

## Status

- Status: PASS for the assigned CPU-only recovery implementation and validation; independent review pending.
- Baseline: `66e249b9648a149f5fe589eb8039ec8ed857da8b`.
- Delivery branch: `codex/vea-p2-t003d-r1-explicit-cuda-device-fix`.

## Observed defect and correction

The T003E Apple run reached the frozen CUDA connector load and then stopped before any Train/Validation update with `connector state device or dtype mismatch`. The T003D CLI accepted bare `--device cuda`. PyTorch resolves model tensors placed with that value to `cuda:0`, while `torch.device("cuda")` and `torch.device("cuda:0")` are distinct identities: the former has no index and the latter has index zero. The state check therefore compared unequal device objects.

T003D-R1 adds one deterministic `canonical_real_device()` boundary. Real mode now accepts only `mps` or an exact indexed CUDA string such as `cuda:0` or `cuda:1`; bare `cuda`, negative, noninteger, padded, malformed, CPU, and unrelated device strings fail closed. The CLI canonicalizes the value before T003C preflight and model load, forwards the exact indexed value to the connector loader, and the runtime uses the same guard before exact connector-state checks. No index is selected or substituted automatically.

## Scientific contracts

The repair changes only device syntax and exact device identity checking. It does not alter the Train/Validation split, data, adapter capacity, conditions or weights, optimizer, epochs, losses, normalizers, selectors, result schema, artifact schema, or Test/generation/vision counters.

## Validation

| Command | Exit | Result |
| --- | ---: | --- |
| `py -3 -m compileall -q src/vision_encoder_alignment/vlm_compat/query_adapter_connector_aware_validation_runtime.py scripts/run_query_adapter_connector_aware_validation.py tests/unit/test_p2_query_adapter_connector_aware_validation_runtime.py tests/integration/test_p2_query_adapter_connector_aware_validation_cli.py` | 0 | Changed files compile. |
| `PYTHONPATH=src py -3 -m pytest -q tests/unit/test_p2_query_adapter_connector_aware_validation_runtime.py tests/integration/test_p2_query_adapter_connector_aware_validation_cli.py` | 0 | 25 passed. Includes CPU-only CUDA identity parsing, bare/malformed CUDA rejection before preflight/model loading, exact `cuda:0` forwarding, deterministic fake smoke, and preserved MPS behavior. |
| `PYTHONPATH=src py -3 -m pytest -q tests/unit/test_p2_query_adapter_connector_aware_linear.py tests/unit/test_p2_query_adapter_connector_aware_runtime_binding.py tests/integration/test_p2_query_adapter_connector_aware_runtime_binding_contract.py tests/integration/test_p2_query_adapter_tinyllava_connector_boundary.py` | 0 | 45 passed. |

No real model, GPU, data, network, Test evidence, generation, or Train/Validation execution occurred. No dependency was installed or changed.

## Future execution

T003E must use `CUDA_VISIBLE_DEVICES=0` with `--device cuda:0`. Real CUDA tensor/device behavior, model parity, resource use, and scientific metrics remain outside this CPU-only repair and require the separately authorized T003E execution. T003D-R1 remains pending independent review; no merge, self-acceptance, or T003E work was performed.
