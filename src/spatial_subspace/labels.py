"""Label normalization for spatial probing (plan §3.7).

Three conventions, chosen to sidestep the monocular scale ambiguity:
  - per-scene normalized coordinates (centroid-subtract, divide by bbox diagonal)
  - normalized pairwise distances   (d_ij / d_max within scene)
  - distance rank order             (Spearman-compatible, maximally invariant)
Absolute (x, y, z) is available from the raw scene metadata as a secondary
diagnostic after Procrustes alignment.

For the Tier C free6dof probing alternative (camera motion + per-object depth),
this module also exposes extrinsic-derived labels: ``rotation_to_axis_angle``,
``camera_delta_6d``, and ``object_depth_in_camera``.
"""
from __future__ import annotations

import math

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


# ---------------------------------------------------------------------------
# Extrinsic-derived labels for camera-motion / depth probing
# ---------------------------------------------------------------------------


def rotation_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → Rodrigues vector r = θ · axis.

    Uses the antisymmetric-trace formula. The θ≈π edge case collapses to the
    zero vector (the axis is indeterminate from the antisymmetric part); our
    per-frame camera deltas in the free6dof trajectory are small so this does
    not come up in practice.
    """
    R = np.asarray(R, dtype=np.float64)
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    theta = math.acos(c)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    v = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return v * (theta / n)


def camera_delta_6d(E_prev: np.ndarray, E_curr: np.ndarray) -> np.ndarray:
    """Relative camera pose from frame t-1 to frame t as a 6-vector.

    Inputs are 4x4 world-to-camera extrinsics (``p_cam = E[:3,:3] @ p_world
    + E[:3,3]``). Returns ``[tx, ty, tz, rx, ry, rz]`` where (tx,ty,tz) is the
    translation of the relative transform ``E_curr · E_prev⁻¹`` and (rx,ry,rz)
    is its axis-angle rotation.
    """
    E_prev = np.asarray(E_prev, dtype=np.float64)
    E_curr = np.asarray(E_curr, dtype=np.float64)
    E_rel = E_curr @ np.linalg.inv(E_prev)
    t_rel = E_rel[:3, 3]
    r_rel = rotation_to_axis_angle(E_rel[:3, :3])
    return np.concatenate([t_rel, r_rel]).astype(np.float64)


def object_depth_in_camera(centroid_world: np.ndarray, E: np.ndarray) -> float:
    """z-coordinate of a world point in the camera frame of extrinsic ``E``."""
    E = np.asarray(E, dtype=np.float64)
    p = np.asarray(centroid_world, dtype=np.float64)
    p_cam = E[:3, :3] @ p + E[:3, 3]
    return float(p_cam[2])
