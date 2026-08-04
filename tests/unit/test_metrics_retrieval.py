import numpy as np

from vision_encoder_alignment.metrics import cosine_retrieval, patch_coordinate_retrieval


def test_exact_global_and_patch_retrieval():
    x = np.eye(12)
    result = cosine_retrieval(x, x, recall_at=(1, 5, 10))
    assert result["recall_at_1"] == 1.0
    patch = np.eye(196, dtype=np.float64)[None]
    spatial = patch_coordinate_retrieval(patch, patch)
    assert spatial["exact_coordinate_top1"] == 1.0
