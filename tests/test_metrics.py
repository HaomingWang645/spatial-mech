import numpy as np

from spatial_subspace.metrics import (
    pairwise_distance_spearman,
    procrustes_align,
    procrustes_error,
    r2,
)


def test_r2_perfect_is_one():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((20, 3))
    assert abs(r2(y, y) - 1.0) < 1e-6


def test_procrustes_identity():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((10, 3))
    aligned, s, _R, _t = procrustes_align(y, y)
    assert np.allclose(aligned, y, atol=1e-6)
    assert abs(s - 1.0) < 1e-6


def test_procrustes_recovers_rigid_and_scale():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((20, 3))
    theta = 0.7
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pred = 2.5 * y @ R.T + np.array([1.0, -1.0, 0.5])
    err = procrustes_error(pred, y)
    assert err < 1e-6


def test_pairwise_distance_spearman_identity():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((8, 3))
    rho = pairwise_distance_spearman(y, y)
    assert rho > 0.9999
