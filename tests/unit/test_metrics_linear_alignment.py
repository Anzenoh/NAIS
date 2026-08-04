import numpy as np
import pytest

from vision_encoder_alignment.metrics import MetricError, fit_orthogonal_procrustes, fit_ridge_mapping, select_reduced_rank_mapping


def test_ridge_beats_procrustes_for_affine_nonorthogonal_map():
    generator = np.random.default_rng(5)
    x = generator.normal(size=(160, 10))
    weights = np.diag(np.linspace(0.2, 2.0, 10))
    y = x @ weights + 3.0 + generator.normal(scale=0.01, size=x.shape)
    ridge = fit_ridge_mapping(x[:120], y[:120], relative_alpha=1e-6)
    proc = fit_orthogonal_procrustes(x[:120], y[:120])
    assert np.mean((ridge.transform(x[120:]) - y[120:]) ** 2) < np.mean((proc.transform(x[120:]) - y[120:]) ** 2)


def test_reduced_rank_uses_one_standard_error_and_is_deterministic():
    generator = np.random.default_rng(8)
    x = generator.normal(size=(90, 8))
    y = x @ generator.normal(size=(8, 7)) + generator.normal(scale=0.1, size=(90, 7))
    mapping = fit_ridge_mapping(x[:60], y[:60], relative_alpha=1e-3)
    first, table = select_reduced_rank_mapping(mapping, x[60:], y[60:], ranks=(2, 4, 6), bootstrap_count=50, seed=9)
    second, _ = select_reduced_rank_mapping(mapping, x[60:], y[60:], ranks=(2, 4, 6), bootstrap_count=50, seed=9)
    assert first.rank == second.rank
    assert sum(bool(row["selected"]) for row in table) == 1


def test_ill_conditioned_unregularized_fit_rejected():
    x = np.column_stack((np.arange(20.0), np.arange(20.0)))
    with pytest.raises(MetricError):
        fit_ridge_mapping(x, x, relative_alpha=0.0, condition_number_limit=1e8)
