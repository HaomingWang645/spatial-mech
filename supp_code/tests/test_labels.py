import math

import numpy as np

from spatial_subspace.labels import (
    camera_delta_6d,
    distance_rank_order,
    normalized_pairwise_distances,
    object_depth_in_camera,
    pairwise_distances,
    per_scene_normalized_coords,
    rotation_to_axis_angle,
)
from spatial_subspace.render.tier_c import look_at


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


def _extrinsic_from_look_at(eye, target, roll=0.0):
    R, t = look_at(np.asarray(eye), np.asarray(target), roll=roll)
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = t
    return E


def test_rotation_to_axis_angle_identity_is_zero():
    assert np.allclose(rotation_to_axis_angle(np.eye(3)), 0.0)


def test_rotation_to_axis_angle_known_z_rotation():
    theta = 0.3
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    r = rotation_to_axis_angle(R)
    # rotation is about +z by θ
    assert np.allclose(r, [0.0, 0.0, theta], atol=1e-8)


def test_camera_delta_is_zero_for_identical_extrinsics():
    E = _extrinsic_from_look_at([5.0, 0.0, 3.0], [0.0, 0.0, 0.5])
    d = camera_delta_6d(E, E)
    assert np.allclose(d, 0.0, atol=1e-8)


def test_camera_delta_translation_nonzero_when_eye_moves():
    E0 = _extrinsic_from_look_at([5.0, 0.0, 3.0], [0.0, 0.0, 0.5])
    E1 = _extrinsic_from_look_at([5.0, 1.0, 3.0], [0.0, 0.0, 0.5])
    d = camera_delta_6d(E0, E1)
    # The eye translated; the relative-transform translation must be non-zero.
    assert np.linalg.norm(d[:3]) > 0.5
    # Pure translation between the two cameras keeps the rotation small (they
    # both re-aim at the origin so there's some rotation, but it's bounded).
    assert np.linalg.norm(d[3:]) < np.linalg.norm(d[:3])


def test_object_depth_in_camera_matches_projection_depth():
    # Camera at (5, 0, 0) looking at origin → forward is -x. A point at the
    # origin is 5 units in front of the camera.
    E = _extrinsic_from_look_at([5.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    depth = object_depth_in_camera([0.0, 0.0, 0.0], E)
    assert abs(depth - 5.0) < 1e-6
    # A point beyond the origin (farther from the camera) has larger depth.
    farther = object_depth_in_camera([-2.0, 0.0, 0.0], E)
    assert farther > depth
