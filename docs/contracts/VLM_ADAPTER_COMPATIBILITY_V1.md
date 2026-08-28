# VLM Adapter Compatibility Contract v1

## 1. Normative direct-adapter relation

A Phase 2 direct adapter MUST implement:

```text
Adapter(E_source(x))
```

as a function of the same Query image `x` whose frozen target feature is
`E_target(x)`. A target feature belonging to another image MUST NOT be presented
as the primary direct adapter output. Anchor transport remains a separately
labeled historical/comparison baseline.

## 2. Frozen and trainable boundary

The following components MUST remain frozen and in evaluation mode:

- source vision encoder;
- target TinyLLaVA CLIP;
- TinyLLaVA connector;
- TinyLLaVA LLM.

Only adapter parameters MAY be fitted or trained. Runs MUST verify the
trainable-parameter set and MUST fail if any frozen component can update.
Closed-form adapter-family fitting is permitted without implying optimizer
training.

## 3. Feature identity contract

Before fitting or evaluation, the run MUST verify and record:

- exact source and target model/checkpoint identities;
- source and target feature-bank artifact identities and hashes;
- exact taps and preprocessing identities;
- source and target tensor shapes;
- exact image-ID sets and ordering.

The current target is TinyLLaVA native CLIP pre-connector patch-only
`[N,196,768]`. Current source candidates are SigLIP2 `final_norm` and DINOv3
block 11 after removal of CLS 1 plus register tokens 4, both patch-only
`[N,196,768]`.

## 4. Split and selection contract

The immutable `p1_1000` split MUST be used by image ID: Train 800, Validation
100, Test 100, seed 43. All patches from one image MUST remain in one partition.

- fitting statistics and adapter weights use Train;
- fitting and model/hyperparameter selection use Train/Validation only;
- Test is held out from fitting and selection;
- finalized configurations receive held-out Test evaluation.

## 5. Downstream evaluation contract

Every finalized method MUST support patch, connector, same-image identity, and
actual frozen-LLM generation evaluation under
`VLM_COMPATIBILITY_METRICS.md`. The native CLIP path is the reference. Prompt,
tokenizer, decoding, and stopping settings MUST be identical across finalized
methods.

## 6. Feature-class boundaries

Artifacts and reports MUST explicitly distinguish:

1. raw-source encoder features;
2. adapted-target-like features;
3. frozen connector-output features;
4. frozen LLM-output tokens and generated text.

No result may silently compare or relabel one class as another.

## 7. Fail-closed conditions

The run MUST stop before fitting or evaluation on any:

- checkpoint, bank, tap, preprocessing, shape, image-ID, or ordering mismatch;
- Train/Validation/Test leakage or Test-informed selection;
- unexpected trainable parameter outside the adapter;
- changed frozen component state;
- prompt or decoding mismatch across finalized methods;
- use of another image's target feature as the claimed direct adapter output.
