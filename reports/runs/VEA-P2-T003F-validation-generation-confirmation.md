# VEA-P2-T003F — Validation Generation Confirmation

## Status

Implementation validation passed. This report records the software contract only; no
real TinyLLaVA model, GPU, dataset, T003E artifact root, Test partition, or generation
execution was used.

## Frozen predecessor identities

- T003E result SHA-256:
  `2147b41ec525f387c0cc8d434d6b3cd748ed224ef39053fc9c6a9e7372963a10`
- T003E runtime identity SHA-256:
  `6e0e274a47813101e560d8f8dc1dbf40ad416b89abd7e6f6d1688646f4b466ca`
- T003A source-config SHA-256:
  `4684a1ae43a124725df93d15c2287a30aa31ae38f6c58bd2f05b83e749f8494b`
- T003C binding SHA-256:
  `f0336604f4d4f37350d2745d8b54edeb1cc19f75fe27d5475609d81402946614`

## Frozen lanes and plan

| Source | Lane | Condition | Epoch | Affine SHA-256 |
| --- | --- | --- | ---: | --- |
| SIGLIP2 | OLS_REFERENCE | PATCH_ONLY_CONTROL | 0 | `a103505426fbb890a891ffed80c7b022b2fcddc24d69192d025c6967438b5d0f` |
| SIGLIP2 | CONNECTOR_AWARE_SELECTED | PATCH_CONNECTOR_4P0 | 7 | `c316f550b551bd8b35e82341366ae156bb947e1e493f27b212b7f6d56e0a17cf` |
| DINOV3 | OLS_REFERENCE | PATCH_ONLY_CONTROL | 0 | `918872025697964e4126675a107f5db00b4efbed222c73b6c30ec40c9bf37e3f` |
| DINOV3 | CONNECTOR_AWARE_SELECTED | PATCH_CONNECTOR_0P25 | 0 | `918872025697964e4126675a107f5db00b4efbed222c73b6c30ec40c9bf37e3f` |

DINOV3 retains the accepted identical affine identity for both lanes. The frozen
production plan is 100 native plus 400 adapted generations, for 500 total on the
100-image Validation partition.

The reused T002E prompt is `What is shown in this image? Answer briefly.` with
`do_sample=false`, `num_beams=1`, and `max_new_tokens=32`.

## Implementation

The runtime verifies the frozen configuration, T003E result and artifact identities,
and the T003C production input binding before a real model call. It accepts only the
Validation partition, checks same-image affine patches, reuses the accepted native
and injection generation boundary, emits deterministic JSONL rows before aggregate
JSON and result JSON, and records exact-match and existing Token F1 aggregates.

The included CPU-only fake tests exercise smaller synthetic counts while separately
checking the frozen 100/400/500 production arithmetic. They do not load models,
datasets, or artifacts.

## Validation

- `py -3 -m compileall -q` on the four changed Python and test files: exit 0.
- `py -3 -m pytest -q tests/unit/test_p2_query_adapter_connector_aware_validation_generation.py tests/integration/test_p2_query_adapter_connector_aware_validation_generation_boundary.py tests/integration/test_p2_query_adapter_tinyllava_generation_boundary.py`: exit 0, 14 passed.

## Changed paths

- `configs/vlm_compat/p2_connector_aware_validation_generation_v1.json`
- `src/vision_encoder_alignment/vlm_compat/query_adapter_connector_aware_validation_generation.py`
- `scripts/run_query_adapter_connector_aware_validation_generation.py`
- `tests/unit/test_p2_query_adapter_connector_aware_validation_generation.py`
- `tests/integration/test_p2_query_adapter_connector_aware_validation_generation_boundary.py`
- `reports/runs/VEA-P2-T003F-validation-generation-confirmation.md`

## Deviations and remaining risks

There are no implementation deviations. Real `cuda:0` execution remains pending
independent review and a separate approved task. Its result will be Validation
development evidence only and cannot establish a Test claim.
