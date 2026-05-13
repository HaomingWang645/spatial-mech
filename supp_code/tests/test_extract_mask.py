import numpy as np

from spatial_subspace.extract import mask_to_patch_coverage, pool_object_vector


def test_mask_coverage_uniform_object():
    H, W = 28, 28
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[:14, :14] = 2  # object_id 1 (stored as id+1)
    cov = mask_to_patch_coverage(mask, (2, 2), object_ids=[1])
    assert cov[1].shape == (2, 2)
    assert cov[1][0, 0] == 1.0
    assert cov[1][0, 1] == 0.0


def test_pool_object_respects_threshold():
    gh, gw, D = 2, 2, 4
    visual = np.arange(gh * gw * D, dtype=np.float32).reshape(gh, gw, D)
    coverage = np.array([[1.0, 0.0], [0.1, 0.4]])
    out = pool_object_vector(visual, coverage, threshold=0.3)
    # Only (0, 0) and (1, 1) pass the threshold.
    expected = (1.0 * visual[0, 0] + 0.4 * visual[1, 1]) / (1.0 + 0.4)
    assert np.allclose(out, expected)


def test_pool_returns_none_when_no_patch_passes():
    gh, gw, D = 2, 2, 3
    visual = np.zeros((gh, gw, D), dtype=np.float32)
    coverage = np.full((gh, gw), 0.1)
    assert pool_object_vector(visual, coverage, threshold=0.3) is None
