import numpy as np

from spatial_subspace.labels import (
    distance_rank_order,
    normalized_pairwise_distances,
    pairwise_distances,
    per_scene_normalized_coords,
)


def test_normalized_coords_unit_ball():
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((10, 3)) * 5
    n = per_scene_normalized_coords(coords)
    assert np.linalg.norm(n, axis=1).max() <= 1 + 1e-5
    assert np.allclose(n.mean(axis=0), 0, atol=1e-6)


def test_pairwise_distances_symmetric():
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((5, 3))
    d = pairwise_distances(coords)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0)


def test_normalized_pairwise_max_is_one():
    rng = np.random.default_rng(2)
    coords = rng.standard_normal((5, 3))
    d = normalized_pairwise_distances(coords)
    assert abs(d.max() - 1.0) < 1e-6


def test_rank_order_is_permutation():
    rng = np.random.default_rng(3)
    coords = rng.standard_normal((6, 3))
    ranks = distance_rank_order(coords)
    assert set(ranks.tolist()) == set(range(len(ranks)))
