import numpy as np

from vision_encoder_alignment.metrics import covariance_geometry, knn_overlap


def test_geometry_and_knn_positive_control():
    x = np.random.default_rng(1).normal(size=(40, 8))
    geometry = covariance_geometry(x)
    assert 1 <= geometry["effective_rank"] <= 8
    assert knn_overlap(x, x, k_values=(5,))["overlap_at_5"] == 1.0
