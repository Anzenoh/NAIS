"""Tested representation-alignment metric API."""

from .canonical_space import (
    MultiwayLatentModel,
    apply_affine,
    closed_cycle_error,
    compose_linear_maps,
    fit_multiway_maxvar,
    select_multiway_dimension,
)
from .cka import (
    cka_image_bootstrap,
    linear_cka,
    linear_cka_from_grams,
    linear_cka_from_moments,
    permutation_calibration,
    rsa_cosine_spearman,
    spectrum_matched_row_null,
    ucentered_cka,
)
from .linear_alignment import (
    RidgeMapping,
    alignment_metrics,
    bootstrap_mapping_metric,
    fit_ridge_mapping,
    reduced_rank_mapping,
    select_reduced_rank_mapping,
    select_ridge_mapping,
)
from .patch_correspondence import (
    PositionTemplate,
    fit_position_template,
    patch_state,
    streaming_patch_cka,
)
from .procrustes import ProcrustesMapping, fit_orthogonal_procrustes
from .representation_geometry import covariance_geometry, knn_overlap
from .retrieval import cosine_retrieval, patch_coordinate_retrieval
from .statistics import MetricError

__all__ = [
    "MetricError", "MultiwayLatentModel", "PositionTemplate", "ProcrustesMapping",
    "RidgeMapping", "alignment_metrics", "apply_affine", "bootstrap_mapping_metric",
    "cka_image_bootstrap", "closed_cycle_error", "compose_linear_maps",
    "cosine_retrieval", "covariance_geometry", "fit_multiway_maxvar",
    "fit_orthogonal_procrustes", "fit_position_template", "fit_ridge_mapping",
    "knn_overlap", "linear_cka", "linear_cka_from_grams",
    "linear_cka_from_moments", "patch_coordinate_retrieval", "patch_state",
    "permutation_calibration", "reduced_rank_mapping", "rsa_cosine_spearman",
    "select_reduced_rank_mapping",
    "select_multiway_dimension", "select_ridge_mapping", "spectrum_matched_row_null",
    "streaming_patch_cka", "ucentered_cka",
]
