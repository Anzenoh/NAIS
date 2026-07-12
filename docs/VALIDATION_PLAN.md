# Validation Plan

## 1. Validation levels

### L0 — Static and unit validation

No network or model checkpoint required.

- formatting and linting;
- type checks where configured;
- config/schema validation;
- deterministic synthetic tests for CKA, Procrustes, ridge, split isolation, and token handling.

### L1 — Mock integration validation

- mock extractor returns canonical tensors and manifests;
- artifact writer and reader round-trip safely;
- interrupted or incomplete shards are detected;
- duplicate image IDs and mismatched tap shapes fail loudly.

### L2 — Real single-model smoke validation

- explicit checkpoint and revision;
- two then ten real images;
- evaluation mode and no gradients;
- canonical tap semantics verified;
- patch grid and token removal verified;
- deterministic repeat measured;
- no NaN or Inf.

### L3 — Cross-model smoke validation

- same ten-image manifest for all encoders;
- all required taps exist;
- image order and hashes agree;
- canonical patch count agrees for the v0 candidate set;
- a small paired metric computation completes.

### L4 — Full experimental run

- immutable 1,000-image manifest;
- exact environment and revisions pinned;
- complete feature counts;
- controls and confidence intervals;
- machine-readable results and human-readable report.

## 2. Required mathematical unit tests

- `CKA(X, X)` is 1 within tolerance for non-degenerate deterministic `X`.
- Linear CKA is invariant to a deterministic orthogonal rotation of features within tolerance.
- Procrustes recovers a known orthogonal transform on held-out synthetic data.
- Ridge recovers a known noisy linear map without using test data for fitting.
- Image-level split tests prove that no image ID appears in multiple partitions.
- A special-token fixture proves that only verified non-spatial tokens are removed.
- A streaming metric matches a small in-memory reference implementation.

## 3. Status semantics

### PASS

All required deliverables exist, all required checks ran and passed, and no contract violation is known.

### PARTIAL

Useful scoped work was completed, but at least one required deliverable or check is missing. State exactly what remains. Do not use PARTIAL to hide a failed required check.

### BLOCKED

The task cannot be validly completed because required input, access, environment support, or a decision is missing, or because continuing would violate a contract.

## 4. Evidence rules

A narrative claim is not evidence by itself. Evidence includes:

- file diff;
- command and exit code;
- test output;
- artifact manifest;
- deterministic tensor shape/value checks;
- environment and revision metadata.

If a required command was not run, say `NOT RUN`. Never infer success.

## 5. Scientific outcomes versus software acceptance

A metric being unexpectedly low, high, asymmetric, or null does not fail the software task if the implementation and controls are correct. Conversely, a plausible-looking figure does not pass a task when the code, split, or artifact contract is invalid.
