# VEA-P2-T001D implementation run report

- Status: Tier 0 implementation validation PASS; no scientific run.
- Baseline: `9bec9c62093bc06b13682259883713b96c69b2dc`
- Branch: `codex/vea-p2-t001d-resumable-anchor-transport`
- `.codex/PROJECT_STATE.json`: absent at the accepted baseline; not created.
- Manifest: `data/manifests/p1_1000.jsonl`, 1,000 rows, SHA-256
  `1819bbc6235dd4805d71642e63e73112ce2e918135ec94f46493077698be92da`.
- Resolved TARGET_CLIP: TinyLLaVA native `hidden_states[-2]`, patch-only
  `[N,196,768]`; target config SHA-256
  `78242f4062fe748826aad0b51ae50ddce1c4bcbb06143396e87d9a862585ac8a`.
- Resolved SigLIP2: `google/siglip2-base-patch16-224` at
  `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, `final_norm`, no special
  tokens, patch-only `[N,196,768]`.
- Resolved DINOv3: `facebook/dinov3-vitb16-pretrain-lvd1689m` at
  `5931719e67bbdb9737e363e781fb0c67687896bc`, block 11, remove CLS 1 plus
  register tokens 4, patch-only `[N,196,768]`.
- Plan inventory: 63 chunks per 1,000-image feature stage, four anchor-registry
  units, 252 validation/test transport units, and 1,900 one-generation units;
  2,345 atomic units total.
- Checkpoint contract: same-directory exclusive partial, file flush/fsync,
  post-write validation and SHA-256, atomic rename, directory fsync, then a
  separately atomic plan-bound marker. Valid units are immutable.
- Recovery contract: proven-dead locks and orphan task partials are quarantined;
  ambiguous locks, marker/artifact loss, and hash mismatch fail closed.
- Scientific contract: identical RAW, per-token SAMPLE_RMS, global 1NN/top-8,
  and positional 1NN/top-8 dispatch for TARGET_CLIP, SIGLIP2, and DINOv3; no
  fitting, optimizer, gradient, or encoder-specific adapter.
- Initial assigned pytest: `104 passed in 74.68s`.
- Final assigned pytest: `105 passed in 85.83s`.
- Real Apple/CUDA/model/dataset/network execution during Codex validation: 0.

## Resolved approved inputs

- `configs/extraction_profiles/p1_1000_apple_fp32_v0.json`
- `configs/runtime_profiles/p1_apple_main_cuda_fp32_v0.json`
- `configs/extractor_profiles/dinov3_vit_b16_r5931719_apple_cuda_fp32_v1.json`
- `configs/runtime_profiles/dinov3_apple_cuda_fp32_v1.json`
- `configs/model_snapshots/dinov3_vit_b16_r5931719_v0.json`
- `configs/vlm_compat/tinyllava_openelm_clip_anchor_v0.json`

## Final evidence

- `py -3.12 -m py_compile` for the core, coordinator, and worker: exit 0.
- Assigned four-file pytest command: exit 0, 105 passed.
- Coordinator `--help`: exit 0 and import-safe.
- Stage-worker `--help`: exit 0 and import-safe.
- Fake E2E: interrupted after 500 valid units (including 55 generation units),
  orphan partial quarantined, resumed 1,845 units, previously completed
  generation bytes and mtimes unchanged, and resumed `result.json` byte-equal
  to the 2,345-unit uninterrupted counterfactual under the same plan ID.
- Fake real-runtime counters: Apple 0, CUDA 0, model 0, dataset 0, network 0.
- `git diff --check`: exit 0.
- Changed-path review: exactly the nine T001D allowlisted paths; protected
  existing-file modification count 0.
- Commit identity and push result: pending the PASS-only delivery step.
