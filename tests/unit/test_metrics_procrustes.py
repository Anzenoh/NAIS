import numpy as np
import pytest

from vision_encoder_alignment.metrics import MetricError, fit_orthogonal_procrustes
from tests._p1_analysis_fixtures import paired


def test_known_orthogonal_transform_is_recovered():
    x, y = paired(n=80, d=9)
    model = fit_orthogonal_procrustes(x, y)
    assert np.mean((model.transform(x) - y) ** 2) < 1e-24


def test_singular_source_is_rejected():
    with pytest.raises(MetricError):
        fit_orthogonal_procrustes(np.ones((10, 4)), np.arange(40).reshape(10, 4))
