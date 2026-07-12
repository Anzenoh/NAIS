# Feature Artifact Schema v0

Schema ID: `vea.feature_artifact.v0`

## 1. Directory layout

```text
artifacts/runs/<run_id>/
├── run_manifest.json
├── dataset_manifest.snapshot.jsonl
├── environment.json
├── models/
│   └── <model_key>/
│       ├── model_manifest.json
│       ├── shard-00000.safetensors
│       ├── shard-00000.json
│       └── ...
└── validation/
    └── artifact_validation.json
```

The generated directory is not committed to Git.

## 2. Run manifest

Required fields:

```json
{
  "schema_id": "vea.feature_artifact.v0",
  "run_id": "...",
  "status": "complete",
  "experiment_id": "...",
  "created_at_utc": "...",
  "git_commit": "...",
  "git_dirty": false,
  "dataset_manifest_sha256": "...",
  "preprocess_config_sha256": "...",
  "image_count": 10,
  "model_keys": ["clip_vit_b16"],
  "storage_dtype": "float32",
  "shard_size": 10,
  "seed": 20260712
}
```

Allowed status values:

- `incomplete`;
- `complete`;
- `invalid`.

Analysis commands must reject `incomplete` and `invalid` runs by default.

## 3. Model manifest

Required fields include:

- schema ID and run ID;
- model registry key;
- exact model ID and revision;
- resolved implementation class;
- resolved vision component path;
- model configuration hash;
- compute and parameter dtype;
- device;
- evaluation-mode and gradient-disabled flags;
- canonical grid and patch count;
- hidden dimension;
- actual tap names and library indices;
- special-token policy and observed non-patch token count;
- normalization parameters;
- shard list and checksums.

## 4. Tensor keys

Each Safetensors shard uses keys of the following form:

```text
<tapping_namespace>.<tap_name>.patch
<tapping_namespace>.<tap_name>.global_mean
```

Canonical namespace: `backbone`.

Examples:

```text
backbone.block_03.patch
backbone.block_03.global_mean
backbone.final_norm.patch
backbone.final_norm.global_mean
```

Native pooled or projected outputs, when explicitly requested, use a separate namespace and must not reuse canonical keys.

## 5. Tensor shapes

For shard batch size `B`, patch count `P`, and hidden dimension `D`:

- patch tensor: `[B, P, D]`;
- global mean tensor: `[B, D]`.

The JSON shard sidecar contains the ordered `image_ids` for the first dimension and the SHA-256 checksum of the Safetensors file.

## 6. Dtype rules

- Phase 0 smoke default: FP32 compute and storage.
- Phase 1 storage may use FP16 after an approved equivalence check.
- Metrics use at least FP32 accumulation.
- Quantized feature storage is not part of v0.

## 7. Tap naming

- one-based report names: `block_01`, `block_02`, ...;
- manifest also records the zero-based implementation index;
- `embedding_output` is separate from blocks;
- `final_norm` is separate from the final block output;
- missing taps are errors when required by the experiment configuration.

## 8. Patch ordering and coordinates

- patch ordering is row-major over the canonical grid;
- model-specific special or register tokens are removed before canonical storage;
- normalized patch-center coordinates are derivable from `grid_height` and `grid_width` and are recorded or deterministically reconstructible;
- for v0 224/p16 models, expected `P = 196`, but runtime validation is mandatory.

## 9. Artifact validity checks

A complete artifact must satisfy:

- manifest and shard checksums match;
- every model has the same ordered image IDs;
- each required tap exists exactly once per image;
- tensor first dimension equals sidecar image count;
- global means numerically match patch means within configured tolerance;
- no NaN or Inf;
- dimensions are constant within a model and tap;
- run status is `complete` only after all checks pass.

## 10. Security and portability

- no pickle-based canonical artifacts;
- no credentials or absolute secret-bearing paths;
- no raw image bytes in feature shards;
- all schema changes require a new schema ID or documented backward-compatible revision.
