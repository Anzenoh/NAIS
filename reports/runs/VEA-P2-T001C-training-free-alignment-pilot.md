# VEA-P2-T001C implementation and offline-validation record

## Status

`PASS_OFFLINE_IMPLEMENTATION` at baseline
`9ab2fb5391438b36c94fc201e8c73b9bcadb11c7` on branch
`codex/vea-p2-t001c-training-free-alignment-pilot`.

This record covers implementation and assigned Tier 0–1 validation only. No
Apple, CUDA, model, dataset, network, package, snapshot materialization, or
scientific pilot execution occurred.

## Frozen selection

Calibration, in manifest order:

1. `000000206411.jpg` (`selection_index=0`)
2. `000000312263.jpg` (`selection_index=1`)
3. `000000049060.jpg` (`selection_index=2`)
4. `000000161008.jpg` (`selection_index=3`)
5. `000000572408.jpg` (`selection_index=4`)
6. `000000230450.jpg` (`selection_index=5`)
7. `000000571804.jpg` (`selection_index=6`)
8. `000000328030.jpg` (`selection_index=7`)

Held-out test, in manifest order:

1. `000000172946.jpg` (`selection_index=12`)
2. `000000127270.jpg` (`selection_index=15`)
3. `000000001818.jpg` (`selection_index=27`)
4. `000000365886.jpg` (`selection_index=28`)

The offline tests prove deterministic image-level selection, disjointness,
development-image rejection, source-hash enforcement, and rejection of flattened
patch rows as fit inputs.

## Implemented contract

- One 12-image call through the validated R3 DINO extractor boundary, followed
  by one TinyLLaVA load.
- Eight native calibration captures and closed-form float64 population-moment
  calibration with `1e-6` denominator floor and float32 parameters.
- Float64 orthogonal Procrustes `R=U@Vt`, reflection allowed, stored float32.
- Four held-out images in exact `NATIVE`, `RAW`, `CALIBRATED`, `LINEAR` order:
  exactly 16 held-out generations through the original frozen connector.
- Task-defined patch/connector cosine, norm ratio, NMSE, functional comparisons,
  and four-image summaries with improvement relative to RAW.
- Exact six-artifact publisher, structured failure evidence, import-safe CLI,
  and a one-execution Apple runbook that keeps the current tmux parent alive.

## Assigned Tier 0–1 results

```text
py -3.12 -m py_compile ...
exit 0

py -3.12 -m pytest -q tests/unit/test_p2_training_free_alignment_pilot.py tests/unit/test_p2_dinov3_raw_injection.py
37 passed in 1.20s
exit 0

py -3.12 scripts/run_training_free_alignment_pilot.py --help
exit 0

git diff --check
exit 0
```

The real scientific status `PASS_TRAINING_FREE_ALIGNMENT_PILOT` is intentionally
not claimed here. After independent review, execute the Apple runbook once.
