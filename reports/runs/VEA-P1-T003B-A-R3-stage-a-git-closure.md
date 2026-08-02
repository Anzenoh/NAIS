# VEA-P1-T003B-A-R3 — Stage A Git closure

## Status

`PASS_PENDING_INDEPENDENT_REVIEW`

- Baseline: `368db0a865d3d81f35f260e3d3ce2a528f89b335`
- Delivery branch: `codex/vea-p1-t003b-a-r3`
- Historical R2 `ff75f06e34ffd05e86844e07a3abb306d34d547f`: preserved and not reused
- Stage A executed: `false`
- Apple/GPU/dataset/checkpoint/model access: `false`
- `READY_FOR_P1_EXTRACTION=false`

## Aggregate source audit

The complete bounded traversal covered runbook Sections 1–3, the full-extraction
contract, the extraction profile and its 13 direct structured references, Stage A
Python imports, package initializers, and the three focused test files. Seventeen
required paths were absent from the baseline. All seventeen had regular authoritative
working-tree candidates; none had an identical blob reachable from an existing ref.
No source was invented or loaded from an installed package.

| Required baseline absence | Initial source SHA-256 | Final candidate SHA-256 |
|---|---|---|
| `configs/datasets/p1_1000_profile.template.json` | `8cc9fc42c39c1df7ddd0547aa9b836116dd798a58e22d541118707c15770f48b` | same |
| `configs/extraction_profiles/p1_1000_apple_fp32_v0.json` | `98d2dfe8e711efdb6f33ac45e87e21318ec53db8b1cc8f6a0aed789babc58cf5` | `a0ef8cfe3bf8e6aed95c2ce2fcc32c6d90cfb169e8593ecc1954862afe96563f` |
| `configs/model_snapshots/p1_three_encoder_checkpoint_plan.template.json` | `86dcb8072d9e8670a714e2564e06d41dd6e22c75c5ce553b4e7b9e0624647ae6` | same |
| `configs/runtime_profiles/p1_bulk_cuda_a100_preflight.template.json` | `96dcb81cf1851b82ac22a4d344666f15d2e28aba127d03710c20bef785e150c2` | same |
| `configs/runtime_profiles/p1_bulk_cuda_rtx5090_direct_preflight.template.json` | `d3548c426497e524ddb76b42d2d248fc30bd99ac004339b8524039744f79e415` | same |
| `data/manifests/p1_1000.jsonl` | `1819bbc6235dd4805d71642e63e73112ce2e918135ec94f46493077698be92da` | same |
| `data/manifests/p1_1000.metadata.json` | `61d530e4a9928b70fa9382af772706834b43ef7d63be2bbed36dee493815a521` | same |
| `reports/runs/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md` | `58878c8ff78fc97b5f5a5cf20c40b0f09fd3fa085b52959276dfad1d3bc4d28c` | same |
| `scripts/collect_p1_bulk_runtime_preflight.py` | `66b0283740fcb11623b7d7e8cae0292f3d6be6e69b72ad3ac2f09095e173be55` | same |
| `scripts/inventory_p1_checkpoint_cache.py` | `1e3d5b9b95315a61af436f61f32b1fb184fb2c9d3e0b53118437d3c8b3ecabd5` | same |
| `scripts/validate_p1_apple_runtime.py` | `478ae09932379618893a34ddaaea7fa5a5ece2477d61360ebb2692b5ad9ba26f` | same |
| `scripts/validate_p1_direct_host_preflight.py` | `6369b8fdd8cd87427e0f91a46a29bc01c9cfd3c25be5c3f6c14a366e0fdd6ab8` | same |
| `scripts/validate_p1_feature_extraction.py` | `7a0e8cfb54798d857ddccdf6d8288d13f8587d3ff257be56c1ef49c8db1ba4f0` | same |
| `src/vision_encoder_alignment/p1_apple_runtime_setup.py` | `d2f5f1ee7da94780d00db26fbb0cbfc06e36fd8faa40ca833ffa751e53d38d08` | same |
| `src/vision_encoder_alignment/p1_feature_extraction.py` | `2c754057f59ccdd4cfc07794816184c2392af583fe26e155183581e2f66f4ab8` | `2a031c401289e408f18c31d58c3d2da473801bb871b286cf911f8f347cb6a3d4` |
| `tasks/VEA-P1-T003A-R1R3-extraction-runner-and-preflight-final-hardening.md` | `3b792f6792b18678ebd6dcc9bb0557bc8dba861f0278d407ebc58e1e9f42a101` | same |
| `tests/unit/test_p1_feature_extraction.py` | `385c3f463110682b4b3e6c44d666c8947835c35093f978af35f1a2b403e05ac6` | same |

The initially documented eight absences are included above. The traversal additionally
found the dataset profile and two manifest files, Apple runtime support, the unit
extraction test, two child-import scripts, and two runtime templates. All missing
candidates were reported together before the authoritative candidate was frozen.

## Frozen sets

`STAGE_A_GRAPH_SET` contains 20 paths:

- the 17 baseline absences above;
- `reports/runbooks/VEA-P1-T003B-apple-feature-extraction-preflight-and-smoke.md`;
- `src/vision_encoder_alignment/__init__.py`;
- `tests/integration/test_p1_feature_extraction_cli.py`.

`EVIDENCE_SET` contains exactly this Task record and this Run Report. Therefore the
final `DELIVERY_SET` contains 22 paths. The source package initializer was normalized
from BOM/CRLF to UTF-8/LF because the required Git whitespace gate rejected its
otherwise equivalent candidate bytes.

## Identity repair

- Extraction profile SHA-256: `a0ef8cfe3bf8e6aed95c2ce2fcc32c6d90cfb169e8593ecc1954862afe96563f`
- Runbook SHA-256: `58408952da337bffa1ac29b0b81fa7def8bfd5aec870afd40c21df44b57c1475`
- Direct structured references: `13/13` present and hash-matched
- Stage A implementation bindings: `6/6` present and hash-matched
- Second deterministic binding pass changed zero digests or files
- Strict module-origin audit: `16/16` project modules resolved below the clean clone

The runbook now requires a 40-character lowercase hexadecimal
`VEA_APPROVED_REPO_COMMIT`, requires `HEAD` equality, rejects staged, unstaged and
untracked paths, and retains `git diff --check` before evidence-root creation. It no
longer requires branch `main` or executes a fixed `e518a51...` identity. Stage B derives
only its run-ID prefix from the already validated approved commit and remains separately
unauthorized.

## Focused validation

All commands used `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, offline variables, candidate `src` first on
`PYTHONPATH`, disabled pytest cache, and an external basetemp.

| Command | Result | Exit |
|---|---:|---:|
| `pytest tests/unit/test_p1_runtime_preflight.py` | `87 passed` | 0 |
| `pytest tests/unit/test_p1_feature_extraction.py` | `66 passed` | 0 |
| `pytest tests/integration/test_p1_feature_extraction_cli.py` | `36 passed` | 0 |
| changed JSON/YAML/TOML parse and Python in-memory compile | PASS | 0 |
| `git diff --check --ignore-space-at-eol` | PASS | 0 |

The accepted 19-file/533-node Windows suite was not rerun. Two disposable preliminary
candidate checks exposed missing dynamic test imports and runtime-template path inputs;
the authoritative graph was rebuilt from the baseline after aggregating both sets.
Neither preliminary check nor final validation modified the source repository.

## Delivery boundary

The graph candidate used baseline Git bytes plus only the frozen graph candidates and
was validated in a second isolated clone. Graph blobs matched between the candidate and
validation checkout. The future delivery commit SHA and a pre-report full-delivery tree
OID are intentionally not recorded here. Independent commit review is required before
any Apple synchronization. Stage A, Stage B, smoke and extraction remain unauthorized.
