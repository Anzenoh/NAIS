import numpy as np

from vision_encoder_alignment.metrics import closed_cycle_error, fit_multiway_maxvar, fit_ridge_mapping


def test_multiway_shared_space_and_closed_cycle():
    generator = np.random.default_rng(2)
    z = generator.normal(size=(60, 4))
    values = [z @ generator.normal(size=(4, 7)) for _ in range(3)]
    model = fit_multiway_maxvar(values, encoder_names=("a", "b", "c"), latent_dimension=4, relative_alpha=1e-3)
    assert model.shared_private(values)["encoder_balanced_shared_variance_fraction"] > 0.8
    maps = [
        fit_ridge_mapping(values[0], values[1], relative_alpha=1e-3),
        fit_ridge_mapping(values[1], values[2], relative_alpha=1e-3),
        fit_ridge_mapping(values[2], values[0], relative_alpha=1e-3),
    ]
    assert closed_cycle_error(values[0], maps)["cycle_nmse"] < 0.1
