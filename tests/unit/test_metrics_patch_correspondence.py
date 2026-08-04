import numpy as np

from vision_encoder_alignment.metrics import fit_position_template, linear_cka, patch_state


def test_position_only_fixture_raw_high_and_double_centered_collapses():
    generator = np.random.default_rng(3)
    position = generator.normal(size=(16, 5))
    x = position[None] + generator.normal(scale=0.01, size=(30, 16, 5))
    y = position[None] @ generator.normal(size=(5, 5)) + generator.normal(scale=0.01, size=(30, 16, 5))
    xt = fit_position_template(x[:20])
    yt = fit_position_template(y[:20])
    raw = linear_cka(x[20:].reshape(-1, 5), y[20:].reshape(-1, 5))
    xd = patch_state(x[20:], "D", template=xt).reshape(-1, 5)
    yd = patch_state(y[20:], "D", template=yt).reshape(-1, 5)
    assert raw > linear_cka(xd, yd)


def test_patch_permutation_destroys_correspondence():
    generator = np.random.default_rng(10)
    x = generator.normal(size=(12, 16, 6))
    permuted = x[:, generator.permutation(16)]
    assert linear_cka(x.reshape(-1, 6), x.reshape(-1, 6)) > linear_cka(x.reshape(-1, 6), permuted.reshape(-1, 6))
