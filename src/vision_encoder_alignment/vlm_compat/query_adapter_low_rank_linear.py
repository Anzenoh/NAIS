"""Pure NumPy, Train-only reduced-rank regression for the frozen T004A design.

No feature bank or accepted OLS artifact is loaded here. A later runtime task
must verify those external identities before passing paired Train chunks.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CONFIG_SHA256 = "7cde52b7a9c11da630475abf99375cf4bf08007f2a1d526532e3aaa8b9a1eb1f"
CONFIG_CANONICAL_JSON_SHA256 = "8c7a671fb9c4d526a9f34a021ea73d5bd0c2be0c5ffe08092256b61908b83f45"
RANK_GRID = (8, 16, 32, 64, 128, 256)
PRODUCTION_PATCHES = 196
PRODUCTION_DIM = 768
PRODUCTION_IMAGES = 800
PRODUCTION_ROWS = 156800
MAX_CHUNK_ROWS = 4096
FLOAT64_EPSILON = np.finfo(np.float64).eps


class LowRankLinearError(ValueError):
    """A frozen configuration, stream, or numerical invariant was violated."""


@dataclass(frozen=True)
class TrainChunk:
    """One paired, ordered set of Train images."""

    source: np.ndarray
    target: np.ndarray
    source_image_ids: tuple[str, ...]
    target_image_ids: tuple[str, ...]
    split: str


@dataclass(frozen=True)
class SyntheticTestShape:
    """Explicit small-fixture pathway; never a production geometry override."""

    patches_per_image: int
    feature_dim: int
    expected_train_images: int


@dataclass(frozen=True)
class LowRankFactors:
    """Compact adapted-target-like map; contains no dense OLS weight."""

    rank: int
    U: np.ndarray
    V: np.ndarray
    bias: np.ndarray
    scalar_count: int
    patches_per_image: int
    feature_dim: int
    synthetic_test_only: bool
    input_feature_class: str = "raw_source_preconnector_patch"
    output_feature_class: str = "adapted_target_like_preconnector_patch"


@dataclass(frozen=True)
class DegenerateRank:
    rank: int
    status: str = "DEGENERATE_RANK"


@dataclass(frozen=True)
class RRRResult:
    by_rank: Mapping[int, LowRankFactors | DegenerateRank]
    train_image_count: int
    train_row_count: int
    positive_eigenvalue_count: int
    synthetic_test_only: bool


def _field(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise LowRankLinearError(f"frozen config drift: {path}")
        current = current[key]
    return current


def _expect(value: Mapping[str, Any], path: str, expected: Any) -> None:
    if _field(value, path) != expected:
        raise LowRankLinearError(f"frozen config drift: {path}")


def validate_low_rank_config(value: Any) -> dict[str, Any]:
    """Reject changes to scientifically material T004A design fields."""

    if not isinstance(value, Mapping):
        raise LowRankLinearError("config must be an object")
    config = copy.deepcopy(dict(value))
    expected = {
        "schema_id": "vea.p2.query_adapter_low_rank_linear_config.v1",
        "experiment_id": "vea.p2.query_adapter_low_rank_linear.v1",
        "task_id": "VEA-P2-T004A",
        "design_only": True,
        "provenance.predecessor_linear_config.path": "configs/vlm_compat/p2_query_adapter_linear_v1.json",
        "provenance.predecessor_linear_config.git_blob_sha256": "28835f065f9a00d756316ced0a1d5ad3e74a3709d88a83abc47ddbe1d6f142a1",
        "provenance.accepted_linear_test_binding.git_blob_sha256": "7688f7f14bdf6058f314e8b23ab60cc7ce9ca59dc48df4844b328ed7844662d3",
        "provenance.accepted_linear_test_binding.validation_result_sha256": "10777d3aa4b19683beecb3b78ae5cea21d20538ae126db9a65daf434aa604ac9",
        "split.id": "p1_1000",
        "split.manifest_sha256": "1819bbc6235dd4805d71642e63e73112ce2e918135ec94f46493077698be92da",
        "split.split_identity_sha256": "6c299a79b22b99b4c68c3de471b9d5c7c756b6bf4a0f78b831a97e30a0a10c5f",
        "split.seed": 43,
        "split.unit": "image_id_before_patch_flattening",
        "split.patches_per_image": PRODUCTION_PATCHES,
        "split.expected_train_patch_rows": PRODUCTION_ROWS,
        "split.train.image_count": PRODUCTION_IMAGES,
        "split.validation.image_count": 100,
        "split.test.image_count": 100,
        "split.test.access_authorized": False,
        "split.ordered_same_image_source_target_ids_required": True,
        "split.patch_level_split_forbidden": True,
        "sources.order": list(("SIGLIP2", "DINOV3")),
        "sources.independent_adapter_per_source": True,
        "sources.shared_adapter": False,
        "sources.same_query_image_required": True,
        "sources.SIGLIP2.model_id": "google/siglip2-base-patch16-224",
        "sources.SIGLIP2.revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "sources.SIGLIP2.tap_name": "final_norm",
        "sources.SIGLIP2.patch_shape": [PRODUCTION_PATCHES, PRODUCTION_DIM],
        "sources.DINOV3.model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "sources.DINOV3.revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
        "sources.DINOV3.tap_name": "block_11",
        "sources.DINOV3.patch_shape": [PRODUCTION_PATCHES, PRODUCTION_DIM],
        "target.model_id": "openai/clip-vit-base-patch16",
        "target.revision": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
        "target.patch_shape": [PRODUCTION_PATCHES, PRODUCTION_DIM],
        "target.frozen": True,
        "target_vlm.source_encoders_frozen": True,
        "target_vlm.target_clip_frozen": True,
        "target_vlm.connector_frozen": True,
        "target_vlm.llm_frozen": True,
        "feature_bank_policy.runtime_source_bank_paths_and_hashes": "OPEN_PENDING_RUNTIME_IDENTITY_BINDING",
        "feature_bank_policy.runtime_target_bank_path_and_hash": "OPEN_PENDING_RUNTIME_IDENTITY_BINDING",
        "ols_control.method": "FULL_LINEAR",
        "ols_control.accepted_fit": "T002D_Train_fitted_OLS_per_source",
        "ols_control.refit_for_control": False,
        "ols_control.weight_shape": [PRODUCTION_DIM, PRODUCTION_DIM],
        "ols_control.means": ["source_mean", "target_mean"],
        "ols_control.exact_artifact_sha256_by_source.SIGLIP2": "0661e426aa62623283c971925d2df93fe6621217d746f8583c3158bc31984974",
        "ols_control.exact_artifact_sha256_by_source.DINOV3": "71728c32e2343f9e8e23f45d524142671f10a3eae0b3bbe59c40f8b5a26cecc7",
        "ols_control.artifact_root_and_external_runtime_paths": "OPEN_PENDING_RUNTIME_IDENTITY_BINDING",
        "ols_control.comparison": "same_run_Validation_OLS_only",
        "ols_control.historical_T002E_Test_metric_as_selector_forbidden": True,
        "adapter.family": "closed_form_reduced_rank_regression",
        "adapter.formula": "A_r(x) = (x @ U_r) @ V_r.T + b_r",
        "adapter.input_feature_class": "raw_source_preconnector_patch",
        "adapter.output_feature_class": "adapted_target_like_preconnector_patch",
        "adapter.input_patch_shape": [PRODUCTION_PATCHES, PRODUCTION_DIM],
        "adapter.output_patch_shape": [PRODUCTION_PATCHES, PRODUCTION_DIM],
        "adapter.U_shape": [PRODUCTION_DIM, "r"],
        "adapter.V_shape": [PRODUCTION_DIM, "r"],
        "adapter.b_shape": [PRODUCTION_DIM],
        "adapter.rank_grid_ordered": list(RANK_GRID),
        "adapter.rank_search_beyond_grid_forbidden": True,
        "adapter.stored_factor_dtype": "float32",
        "adapter.stored_bias_dtype": "float32",
        "adapter.stored_adapted_patch_dtype": "float32",
        "adapter.selected_artifact_dense_W_forbidden": True,
        "adapter.residual_or_full_dense_weight_stored_in_selected_artifact_forbidden": True,
        "adapter.nonlinear_layers_forbidden": True,
        "adapter.spatial_or_token_mixing_forbidden": True,
        "adapter.connector_aware_loss_forbidden": True,
        "adapter.optimizer": None,
        "adapter.new_ridge_search_forbidden": True,
        "closed_form_fit.partition": "train_only",
        "closed_form_fit.method": "RRR_from_accepted_OLS_and_Train_covariance_not_SVD_of_W",
        "closed_form_fit.mean_identity_check_rtol": 1e-5,
        "closed_form_fit.mean_identity_check_atol": 1e-6,
        "closed_form_fit.Xc": "X - mu_X",
        "closed_form_fit.Yc": "Y - mu_Y",
        "closed_form_fit.Sxx": "Xc.T @ Xc",
        "closed_form_fit.Sxx_accumulation": "streaming_Train_patch_rows_after_Train_mean_pass",
        "closed_form_fit.M": "W.T @ Sxx @ W",
        "closed_form_fit.eigendecomposition": "symmetric_eigh_float64",
        "closed_form_fit.V_r": "top_r_orthonormal_eigenvectors_of_M_descending_eigenvalue",
        "closed_form_fit.U_r": "W @ V_r",
        "closed_form_fit.b_r": "mu_Y - mu_X @ (U_r @ V_r.T)",
        "closed_form_fit.max_patch_rows_per_streaming_chunk": MAX_CHUNK_ROWS,
        "closed_form_fit.accumulation_dtype": "float64",
        "closed_form_fit.eigensolve_dtype": "float64",
        "closed_form_fit.psd_tolerance": "for_each_of_Sxx_and_M: 64 * float64_epsilon * max(1, max_abs_eigenvalue)",
        "closed_form_fit.eigenvalue_below_negative_tolerance": "FAIL_CLOSED",
        "closed_form_fit.eigenvalue_in_negative_tolerance_band": "CLAMP_TO_ZERO",
        "closed_form_fit.eigenvalue_tie": "adjacent_descending_eigenvalues_with_absolute_difference_at_most_psd_tolerance",
        "closed_form_fit.tie_policy": "within_tied_eigenspace_use_projector_then_project_standard_basis_vectors_in_ascending_coordinate_order_with_two_pass_modified_Gram_Schmidt",
        "closed_form_fit.sign_policy": "make_first_maximum_absolute_component_positive; lowest_coordinate_index_breaks_component_tie",
        "closed_form_fit.rank_exceeds_positive_eigenvalue_count": "DEGENERATE_RANK_ineligible_for_selection",
        "closed_form_fit.zero_spectrum": "all_ranks_DEGENERATE_RANK_then_NO_RANK_ACCEPTED",
        "closed_form_fit.validation_or_test_statistics_for_fit_forbidden": True,
        "complexity.full_linear_scalar_count_per_source": 590592,
        "complexity.low_rank_scalar_formula": "1536*r + 768",
        "complexity.low_rank_scalar_counts_ordered": [13056, 25344, 49920, 99072, 197376, 393984],
        "complexity.hard_max_stored_float32_scalars_per_source": 400000,
        "latency.hard_max_candidate_to_matched_OLS_p50_ratio": 1.10,
        "validation.partition": "validation",
        "validation.proxy_only_selection_forbidden": True,
        "validation.generation.future_separate_approval_required": True,
        "validation.generation.planned_total_generations": 1500,
        "selection.partition": "validation_only",
        "selection.no_qualifying_rank_result": "NO_RANK_ACCEPTED",
        "selection.no_qualifying_rank_action": "retain_accepted_OLS_control",
        "test_policy.design_and_train_validation_development_access": 0,
        "test_policy.design_and_train_validation_development_metrics": 0,
        "test_policy.design_and_train_validation_development_generation": 0,
        "test_policy.design_and_train_validation_development_selection": 0,
        "test_policy.new_test_access_authorized": False,
        "test_policy.future_confirmatory_test_requires_separately_approved_untouched_holdout_policy": True,
        "execution_authorization.fit_authorized_now": False,
        "execution_authorization.dataset_or_model_load_authorized_now": False,
        "execution_authorization.GPU_authorized_now": False,
        "execution_authorization.generation_authorized_now": False,
        "execution_authorization.test_authorized_now": False,
    }
    for path, frozen in expected.items():
        _expect(config, path, frozen)
    try:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LowRankLinearError("frozen config is not canonical JSON data") from exc
    if hashlib.sha256(canonical).hexdigest() != CONFIG_CANONICAL_JSON_SHA256:
        raise LowRankLinearError("frozen config drift: canonical T004A design")
    return config


def load_low_rank_config(path: str | Path) -> dict[str, Any]:
    """Load the exact T004A file; tolerate only checkout LF/CRLF differences."""

    try:
        raw = Path(path).read_bytes()
        normalized = raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(normalized).hexdigest() != CONFIG_SHA256:
            raise LowRankLinearError("frozen config SHA-256 mismatch")
        value = json.loads(normalized.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LowRankLinearError("low-rank config is unavailable or invalid JSON") from exc
    return validate_low_rank_config(value)


def production_parameter_count(rank: int, config: Mapping[str, Any]) -> int:
    """Return the exact frozen factor/bias scalar count for a production rank."""

    validated = validate_low_rank_config(config)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in RANK_GRID:
        raise LowRankLinearError("rank is not in the frozen production grid")
    count = 2 * PRODUCTION_DIM * rank + PRODUCTION_DIM
    index = RANK_GRID.index(rank)
    if count != validated["complexity"]["low_rank_scalar_counts_ordered"][index]:
        raise LowRankLinearError("frozen parameter count mismatch")
    if count > validated["complexity"]["hard_max_stored_float32_scalars_per_source"]:
        raise LowRankLinearError("frozen scalar ceiling exceeded")
    return count


def _real_float64(value: Any, label: str, *, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise LowRankLinearError(f"{label} is not a regular numeric array") from exc
    if array.ndim != ndim:
        raise LowRankLinearError(f"{label} must have {ndim} dimensions")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise LowRankLinearError(f"{label} must be finite real numeric data")
    try:
        converted = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LowRankLinearError(f"{label} cannot be converted to float64") from exc
    if not np.isfinite(converted).all():
        raise LowRankLinearError(f"{label} contains NaN or Inf")
    return converted


def _geometry(shape: SyntheticTestShape | None) -> tuple[int, int, int, bool]:
    if shape is None:
        return PRODUCTION_PATCHES, PRODUCTION_DIM, PRODUCTION_IMAGES, False
    if not isinstance(shape, SyntheticTestShape):
        raise LowRankLinearError("synthetic test geometry must be explicit")
    values = (shape.patches_per_image, shape.feature_dim, shape.expected_train_images)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values):
        raise LowRankLinearError("synthetic test geometry must contain positive integers")
    return *values, True


def _digest_chunk(
    chunk: TrainChunk, source: np.ndarray, target: np.ndarray, source_dtype: np.dtype[Any], target_dtype: np.dtype[Any]
) -> str:
    digest = hashlib.sha256()
    identity = [chunk.split, chunk.source_image_ids, chunk.target_image_ids, source.shape, str(source_dtype), str(target_dtype)]
    digest.update(json.dumps(identity, separators=(",", ":")).encode("utf-8"))
    digest.update(memoryview(np.ascontiguousarray(source)).cast("B"))
    digest.update(memoryview(np.ascontiguousarray(target)).cast("B"))
    return digest.hexdigest()


def _checked_chunk(chunk: Any, patches: int, dim: int) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    if not isinstance(chunk, TrainChunk) or chunk.split != "train":
        raise LowRankLinearError("every chunk must be an explicit TrainChunk with split='train'")
    try:
        source_array = np.asarray(chunk.source)
        target_array = np.asarray(chunk.target)
    except (TypeError, ValueError) as exc:
        raise LowRankLinearError("paired Train chunk is not regular numeric data") from exc
    if source_array.ndim != 3 or target_array.shape != source_array.shape:
        raise LowRankLinearError("paired Train chunk shape mismatch")
    batch, observed_patches, observed_dim = source_array.shape
    if batch <= 0 or (observed_patches, observed_dim) != (patches, dim):
        raise LowRankLinearError("Train chunk geometry mismatch")
    if batch * observed_patches > MAX_CHUNK_ROWS:
        raise LowRankLinearError("Train chunk exceeds 4096 flattened rows")
    source_ids = chunk.source_image_ids
    target_ids = chunk.target_image_ids
    if (
        not isinstance(source_ids, tuple)
        or not isinstance(target_ids, tuple)
        or source_ids != target_ids
        or len(source_ids) != batch
        or any(not isinstance(item, str) or not item for item in source_ids)
        or len(set(source_ids)) != batch
    ):
        raise LowRankLinearError("paired Train image-ID order or uniqueness mismatch")
    source = _real_float64(source_array, "source", ndim=3)
    target = _real_float64(target_array, "target", ndim=3)
    return source, target, source_ids, _digest_chunk(chunk, source, target, source_array.dtype, target_array.dtype)


def _symmetric_eigh(matrix: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, float]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise LowRankLinearError(f"{label} must be a finite square matrix")
    symmetric = (matrix + matrix.T) * 0.5
    if not np.isfinite(symmetric).all():
        raise LowRankLinearError(f"{label} symmetrization is non-finite")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise LowRankLinearError(f"{label} eigensolve is non-finite")
    tolerance = 64 * FLOAT64_EPSILON * max(1.0, float(np.max(np.abs(eigenvalues))))
    if np.any(eigenvalues < -tolerance):
        raise LowRankLinearError(f"{label} has materially negative eigenvalues")
    # Numerical zeros include the configured negative tolerance band.
    eigenvalues[np.abs(eigenvalues) <= tolerance] = 0.0
    return eigenvalues, eigenvectors, tolerance


def _signed(vector: np.ndarray) -> np.ndarray:
    pivot = int(np.argmax(np.abs(vector)))  # first index wins exact magnitude ties
    return -vector if vector[pivot] < 0 else vector


def _canonical_tied_basis(eigenspace: np.ndarray) -> np.ndarray:
    """Canonicalize a whole tied eigenspace before any rank-cut slicing."""

    dimension, multiplicity = eigenspace.shape
    projector = eigenspace @ eigenspace.T
    basis: list[np.ndarray] = []
    threshold = 64 * FLOAT64_EPSILON * dimension
    for coordinate in range(dimension):
        candidate = projector[:, coordinate].copy()
        for _ in range(2):  # frozen two-pass modified Gram-Schmidt
            for previous in basis:
                candidate -= previous * np.dot(previous, candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= threshold:
            continue
        basis.append(_signed(candidate / norm))
        if len(basis) == multiplicity:
            break
    if len(basis) != multiplicity:
        raise LowRankLinearError("tied eigenspace projector is numerically degenerate")
    result = np.column_stack(basis)
    if not np.allclose(result.T @ result, np.eye(multiplicity), rtol=0.0, atol=1e-10):
        raise LowRankLinearError("canonical tied basis is not orthonormal")
    return result


def _canonical_positive_basis(eigenvalues: np.ndarray, eigenvectors: np.ndarray, tolerance: float) -> np.ndarray:
    order = np.argsort(-eigenvalues, kind="stable")
    positive = order[eigenvalues[order] > 0.0]
    if len(positive) == 0:
        return np.empty((eigenvectors.shape[0], 0), dtype=np.float64)
    values = eigenvalues[positive]
    vectors = eigenvectors[:, positive]
    groups: list[np.ndarray] = []
    start = 0
    for index in range(1, len(positive) + 1):
        if index < len(positive) and abs(float(values[index - 1] - values[index])) <= tolerance:
            continue
        group = vectors[:, start:index]
        groups.append(_signed(group[:, 0].copy())[:, None] if group.shape[1] == 1 else _canonical_tied_basis(group))
        start = index
    return np.column_stack(groups)


def _float32_factor(value: np.ndarray, label: str) -> np.ndarray:
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise LowRankLinearError(f"{label} cannot be stored as finite float32")
    result.flags.writeable = False
    return result


def fit_low_rank_rrr(
    chunks_factory: Callable[[], Iterable[TrainChunk]],
    accepted_ols_W: Any,
    accepted_mu_X: Any,
    accepted_mu_Y: Any,
    config: Mapping[str, Any],
    *,
    synthetic_test_shape: SyntheticTestShape | None = None,
    synthetic_test_ranks: Sequence[int] | None = None,
) -> RRRResult:
    """Derive compact factors from a repeatable paired Train stream.

    The caller supplies accepted OLS arrays; their external artifact SHA and
    feature-bank identities remain a later runtime-binding gate.
    """

    validated = validate_low_rank_config(config)
    if not callable(chunks_factory):
        raise LowRankLinearError("Train chunks must come from a re-iterable factory")
    patches, dim, expected_images, synthetic = _geometry(synthetic_test_shape)
    if synthetic:
        if synthetic_test_ranks is None:
            raise LowRankLinearError("synthetic test ranks must be explicit")
        ranks = tuple(synthetic_test_ranks)
    else:
        if synthetic_test_ranks is not None:
            raise LowRankLinearError("production ranks cannot override the frozen grid")
        ranks = RANK_GRID
    if not ranks or any(isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0 or rank > dim for rank in ranks):
        raise LowRankLinearError("duplicate, unsupported, or malformed ranks")
    if len(set(ranks)) != len(ranks):
        raise LowRankLinearError("duplicate, unsupported, or malformed ranks")
    W = _real_float64(accepted_ols_W, "accepted OLS W", ndim=2).copy()
    mu_x = _real_float64(accepted_mu_X, "accepted source mean", ndim=1).copy()
    mu_y = _real_float64(accepted_mu_Y, "accepted target mean", ndim=1).copy()
    if W.shape != (dim, dim) or mu_x.shape != (dim,) or mu_y.shape != (dim,):
        raise LowRankLinearError("accepted OLS weight or mean shape mismatch")
    source_sum = np.zeros(dim, dtype=np.float64)
    target_sum = np.zeros(dim, dtype=np.float64)
    seen: set[str] = set()
    digests: list[str] = []
    image_count = 0
    row_count = 0
    for chunk in chunks_factory():
        x, y, ids, digest = _checked_chunk(chunk, patches, dim)
        if seen.intersection(ids):
            raise LowRankLinearError("duplicate Train image ID across chunks")
        seen.update(ids)
        digests.append(digest)
        image_count += len(ids)
        row_count += x.shape[0] * patches
        if image_count > expected_images:
            raise LowRankLinearError("Train image count exceeds expected count")
        source_sum += x.sum(axis=(0, 1), dtype=np.float64)
        target_sum += y.sum(axis=(0, 1), dtype=np.float64)
    expected_rows = expected_images * patches
    if image_count != expected_images or row_count != expected_rows or not digests:
        raise LowRankLinearError("Train image or flattened row count mismatch")
    if not np.allclose(source_sum / row_count, mu_x, rtol=validated["closed_form_fit"]["mean_identity_check_rtol"], atol=validated["closed_form_fit"]["mean_identity_check_atol"]):
        raise LowRankLinearError("accepted source mean differs from Train stream")
    if not np.allclose(target_sum / row_count, mu_y, rtol=validated["closed_form_fit"]["mean_identity_check_rtol"], atol=validated["closed_form_fit"]["mean_identity_check_atol"]):
        raise LowRankLinearError("accepted target mean differs from Train stream")

    covariance = np.zeros((dim, dim), dtype=np.float64)
    replay_images = 0
    replay_rows = 0
    replay_chunks = 0
    for chunk in chunks_factory():
        x, _y, ids, digest = _checked_chunk(chunk, patches, dim)
        if replay_chunks >= len(digests) or digest != digests[replay_chunks]:
            raise LowRankLinearError("Train stream changed between passes")
        replay_chunks += 1
        replay_images += len(ids)
        replay_rows += x.shape[0] * patches
        centered = x.reshape(-1, dim) - mu_x
        covariance += centered.T @ centered
    if replay_chunks != len(digests) or replay_images != image_count or replay_rows != row_count:
        raise LowRankLinearError("Train stream changed length between passes")
    covariance = (covariance + covariance.T) * 0.5
    _symmetric_eigh(covariance, "Sxx")
    M = W.T @ covariance @ W
    M = (M + M.T) * 0.5
    eigenvalues, eigenvectors, tolerance = _symmetric_eigh(M, "M")
    basis = _canonical_positive_basis(eigenvalues, eigenvectors, tolerance)
    positive_count = basis.shape[1]
    results: dict[int, LowRankFactors | DegenerateRank] = {}
    for rank in ranks:
        if rank > positive_count:
            results[rank] = DegenerateRank(rank)
            continue
        V = basis[:, :rank]
        U = W @ V
        bias = mu_y - mu_x @ (U @ V.T)
        scalar_count = 2 * dim * rank + dim
        if scalar_count > validated["complexity"]["hard_max_stored_float32_scalars_per_source"]:
            raise LowRankLinearError("stored scalar ceiling exceeded")
        if not synthetic and scalar_count != production_parameter_count(rank, validated):
            raise LowRankLinearError("production factor count differs from frozen grid")
        results[rank] = LowRankFactors(
            rank=rank,
            U=_float32_factor(U, "U"),
            V=_float32_factor(V, "V"),
            bias=_float32_factor(bias, "bias"),
            scalar_count=scalar_count,
            patches_per_image=patches,
            feature_dim=dim,
            synthetic_test_only=synthetic,
        )
    return RRRResult(results, image_count, row_count, positive_count, synthetic)


def apply_low_rank(features: Any, factors: LowRankFactors) -> np.ndarray:
    """Map raw-source patches to float32 adapted-target-like patches."""

    if not isinstance(factors, LowRankFactors):
        raise LowRankLinearError("a nondegenerate compact rank is required")
    dim, rank = factors.feature_dim, factors.rank
    if (
        factors.U.shape != (dim, rank)
        or factors.V.shape != (dim, rank)
        or factors.bias.shape != (dim,)
        or any(item.dtype != np.float32 for item in (factors.U, factors.V, factors.bias))
        or factors.scalar_count != 2 * dim * rank + dim
        or any(not np.isfinite(item).all() for item in (factors.U, factors.V, factors.bias))
    ):
        raise LowRankLinearError("compact factor shape, dtype, or count mismatch")
    if not factors.synthetic_test_only and (dim, factors.patches_per_image, rank) not in {
        (PRODUCTION_DIM, PRODUCTION_PATCHES, candidate) for candidate in RANK_GRID
    }:
        raise LowRankLinearError("production compact factor geometry or rank mismatch")
    source = _real_float64(features, "raw-source patches", ndim=3)
    if source.shape[0] <= 0 or source.shape[1:] != (factors.patches_per_image, dim):
        raise LowRankLinearError("raw-source patch geometry mismatch")
    result = (source @ factors.U.astype(np.float64)) @ factors.V.astype(np.float64).T
    result += factors.bias.astype(np.float64)
    adapted = np.array(result, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(adapted).all():
        raise LowRankLinearError("adapted-target-like patches are non-finite")
    return adapted
