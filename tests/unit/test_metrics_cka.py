import numpy as np
import pytest

from vision_encoder_alignment.metrics import MetricError, linear_cka, permutation_calibration, ucentered_cka
from tests._p1_analysis_fixtures import paired


def test_identical_and_orthogonal_are_one():
    x, y = paired()
    assert linear_cka(x, x) == pytest.approx(1.0)
    assert linear_cka(x, y) == pytest.approx(1.0)
    assert ucentered_cka(x, x) == pytest.approx(1.0)


def test_independent_and_permuted_controls_collapse():
    generator = np.random.default_rng(11)
    x = generator.normal(size=(120, 10))
    y = generator.normal(size=(120, 10))
    assert linear_cka(x, y) < 0.25
    calibrated = permutation_calibration(x, x, permutations=31, seed=4)
    assert calibrated["observed"] > calibrated["null_mean"]


def test_zero_denominator_rejected():
    with pytest.raises(MetricError):
        linear_cka(np.ones((8, 3)), np.ones((8, 2)))
