"""Synthetic regression test for probes: when the first 3 dims of the feature
are the object coordinates plus noise, the linear probe should explain most
of the variance."""
import numpy as np

from spatial_subspace.labels import per_scene_normalized_coords
from spatial_subspace.probes import (
    fit_linear_probe,
    fit_pairwise_distance_probe,
    fit_pca_linear,
    scene_split,
)


def _make_synthetic(n_scenes=30, hidden_dim=64, seed=0):
    rng = np.random.default_rng(seed)
    scenes = []
    vecs = []
    labels_raw = []
    scene_ids = []
    for s in range(n_scenes):
        n_obj = rng.integers(3, 7)
        coords = rng.uniform(-4, 4, size=(n_obj, 3))
        coords[:, 2] = 0.0
        norm = per_scene_normalized_coords(coords)
        for i in range(n_obj):
            vec = np.concatenate(
                [norm[i], rng.normal(scale=0.05, size=hidden_dim - 3)]
            ).astype(np.float32)
            vecs.append(vec)
            labels_raw.append(norm[i])
            scene_ids.append(f"s{s}")
    return (
        np.array(scene_ids),
        np.stack(vecs).astype(np.float32),
        np.stack(labels_raw).astype(np.float32),
    )


def test_linear_probe_recovers_coords():
    scene_ids, X, y = _make_synthetic()
    train_s, test_s = scene_split(scene_ids, 0.8, 0)
    tr = np.isin(scene_ids, train_s)
    te = np.isin(scene_ids, test_s)
    res = fit_linear_probe(X[tr], y[tr], X[te], y[te])
    assert res.r2 > 0.9, f"expected R² > 0.9, got {res.r2}"
    assert res.extras["procrustes"] < 0.05


def test_pca_linear_low_k_suffices():
    scene_ids, X, y = _make_synthetic()
    train_s, test_s = scene_split(scene_ids, 0.8, 0)
    tr = np.isin(scene_ids, train_s)
    te = np.isin(scene_ids, test_s)
    res = fit_pca_linear(X[tr], y[tr], X[te], y[te], k=8)
    assert res.r2 > 0.8


def test_pairwise_probe_runs_and_produces_finite_metrics():
    # Euclidean distance is not linear in concatenated features, so the
    # absolute value of pairwise-probe R² is an empirical quantity, not a
    # correctness invariant. Just check the probe runs end-to-end.
    scene_ids, X, y = _make_synthetic()
    train_s, test_s = scene_split(scene_ids, 0.8, 0)
    _unique, scene_of = np.unique(scene_ids, return_inverse=True)
    tr_ids = np.where(np.isin(_unique, train_s))[0]
    te_ids = np.where(np.isin(_unique, test_s))[0]
    res = fit_pairwise_distance_probe(X, y, scene_of, tr_ids, te_ids)
    assert np.isfinite(res.r2)
    assert np.isfinite(res.extras["spearman"])
    assert res.extras["n_train_pairs"] > 0
    assert res.extras["n_test_pairs"] > 0
