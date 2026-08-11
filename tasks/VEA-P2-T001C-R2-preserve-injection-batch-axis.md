# VEA-P2-T001C-R2 — Preserve held-out injection batch axis

- Mode: recovery
- Baseline: `dfb00a3ef62bdf3f4bf806063db685859bf100a8`
- Branch: `codex/vea-p2-t001c-r2-preserve-injection-batch-axis`
- Commit: `fix(vlm-compat): VEA-P2-T001C-R2 preserve injection batch axis`

## Confirmed defect and bounded fix

The R1 Apple run completed the first held-out NATIVE path and failed at RAW.
Indexing a `[12,196,768]` DINO feature array with `dino[8 + test_index]`
dropped the singleton batch axis and passed `[196,768]` into replay. TinyLLaVA
then received a two-dimensional connector input and could not combine its
one-dimensional image result with two-dimensional language embeddings.

For each held-out image, select the source only with an explicit range slice:

```text
start = 8 + test_index
raw_batch = dino[start:start + 1]  # [1,196,768]
```

Derive every replacement from this same batch:

```text
RAW        = raw_batch
CALIBRATED = moment_calibration(raw_batch)
LINEAR     = procrustes(CALIBRATED)
```

Before NumPy-to-Torch conversion and ReplayInjection, a dedicated fail-fast
boundary requires an actual NumPy array with exact shape `[1,196,768]`, float32
dtype, finite values, C-contiguous layout, and writable storage. It performs no
reshape, squeeze, flatten, pooling, broadcasting, duplication, or feature change.
The previous `[196,768]` value must fail before chat or generation.

## Preserved R1 contracts

Keep the direct `load_image → process_images → encode_images` calibration path,
eight calibration images, four held-out images, one DINO extraction, one VLM
load, eight calibration feature calls, zero calibration generations, and exactly
16 held-out generations. Preserve fixed splits/hashes, condition order, alignment
formulas, float64 fit/float32 parameter rules, metrics and NMSE, token/text
comparisons, six artifacts, prompt/settings, and bounded chained failure evidence.

Do not modify the validated raw-injection or native-replay implementations,
extractor/worker/boundaries/profiles, CLI, manifest, metrics, or dependencies.

## Regression and Apple procedure

CPU/fake tests prove index 8 and all four held-out slices remain `[1,196,768]`;
the legacy 2-D form is rejected before generation; all three replacement results
pass the exact boundary; dtype/layout/writable/finite violations fail; and a fake
ReplayInjection connector observes 3-D input and emits `[1,196,1536]`. Existing
R1 science, accounting, failure, and import-safety tests remain mandatory.

The R2 Apple runbook starts from the existing tmux parent, performs preflight and
one pilot in a child bash, verifies rather than rebuilds the snapshot, validates
12 source hashes, and waits for a fully idle shared GPU. Before that gate it may
perform only CPU/file/package-metadata checks: no Torch import, `torch.cuda.*`,
CUDA context, model load, or forward. After the gate it runs exactly one pilot
and prints either success artifacts or the existing precise failure evidence.

During Codex validation do not access Apple, CUDA, models, datasets, packages,
snapshots, or execute the experiment. Run only the assigned compile, two targeted
test modules, CLI help, and diff check. Commit the five allowed paths, push the
R2 branch, and stop for independent review.
