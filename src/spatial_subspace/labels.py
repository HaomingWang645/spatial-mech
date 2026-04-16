"""Label normalization for spatial probing (plan §3.7).

Three conventions, chosen to sidestep the monocular scale ambiguity:
  - per-scene normalized coordinates (centroid-subtract, divide by bbox diagonal)
  - normalized pairwise distances   (d_ij / d_max within scene)
  - distance rank order             (Spearman-compatible, maximally invariant)
Absolute (x, y, z) is available from the raw scene metadata as a secondary
diagnostic after Procrustes alignment.
"""
from __future__ import annotations

import numpy as np


def per_scene_normalized_coords(coords: np.ndarray) -> np.ndarray:
    """coords: (N, 3). Centered and scaled so the bbox diagonal is 1."""
    c = coords - coords.mean(axis=0, keepdims=True)
    lo = c.min(axis=0)
    hi = c.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if diag < 1e-8:
        return c
    return c / diag


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """coords: (N, 3). Returns (N, N) Euclidean distance matrix."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def normalized_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    d = pairwise_distances(coords)
    dmax = float(d.max())
    if dmax < 1e-8:
        return d
    return d / dmax


def distance_rank_order(coords: np.ndarray) -> np.ndarray:
    """(N*(N-1)/2,) integer ranks of unique pairs, ascending."""
    d = pairwise_distances(coords)
    iu, ju = np.triu_indices(d.shape[0], k=1)
    pairs = d[iu, ju]
    return np.argsort(np.argsort(pairs))
