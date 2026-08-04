# Reproducibility Requirements

## 1. Environment record

Every real-checkpoint run records:

- UTC timestamp and run ID;
- operating system and version;
- Python version;
- Git commit and dirty working-tree status;
- CPU model and RAM;
- GPU model, driver, runtime, and visible device list;
- PyTorch, Transformers, NumPy, Pillow, and artifact-library versions;
- relevant deterministic and backend flags;
- complete dependency lock or freeze output.

## 2. Model record

For each encoder:

- registry key;
- repository/model ID;
- exact revision or commit hash;
- resolved model class and vision component path;
- model configuration hash;
- parameter dtype and compute dtype;
- evaluation-mode confirmation;
- gradient-disabled confirmation;
- access/license status as supplied by the user;
- cache path recorded without embedding secrets.

Do not run the 1,000-image extraction with an unpinned `main` revision.

## 3. Dataset record

- dataset name and version if applicable;
- user-provided dataset root outside Git;
- immutable JSONL manifest;
- manifest SHA-256;
- unique image count;
- corrupt/missing file validation result;
- split manifest and hash;
- no silent skips or replacements.

## 4. Preprocessing record

- geometry transform name and parameters;
- interpolation and antialias settings;
- output resolution;
- model-specific scaling, mean, and standard deviation;
- serialized configuration hash;
- optional hash of the shared pre-normalization image tensor for smoke validation.

## 5. Run identity

Recommended format:

```text
YYYYMMDDTHHMMSSZ_<experiment_id>_<git_short_sha>
```

All artifacts, manifests, tables, and reports from one run must reference the same run ID.

## 6. Randomness

Record separate seeds for:

- dataset subset selection;
- train/validation/test split;
- shuffled-pair control;
- bootstrap resampling;
- model or metric initialization, when applicable.

Seeds do not replace deterministic validation. Record observed nondeterminism for repeated extraction.

## 7. Storage and portability

- generated dense features use the versioned contract in `contracts/FEATURE_ARTIFACT_SCHEMA.md`;
- use relative paths in manifests where possible;
- use checksums for shards;
- do not use pickle as the canonical interchange format;
- incomplete runs must be marked and must not be consumed by default analysis commands.

## 8. Sealed Phase 1 analysis

- The result identity is derived from the accepted input run ID, full analysis commit, and canonical config SHA-256.
- Set `PYTHONHASHSEED=0` and all declared NumPy/BLAS thread variables before importing NumPy. The campaign uses explicit FP64 accumulation over FP32 source features and fixed PCG64 label-derived seeds.
- Resume requires unchanged code, config, input, prior-stage inventories, and fitted bytes. Completed stages are no-clobber.
- Exactly one logical test-unseal event is allowed. After it, reselection, refit, config mutation, code mutation, and non-test value reads are rejected.
- The deterministic archive fixes ordering, paths, modes, ownership fields, timestamps, compression, JSON, and newlines. Its outer SHA-256 stays outside the archive.
- Windows-local repeat status is `NOT_EXECUTED_LOCAL_RESOURCE_UNAVAILABLE`; starting/ending 252-file equality is preservation evidence only.
