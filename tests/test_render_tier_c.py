import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_subspace.render.common import generate_3d_scene
from spatial_subspace.render.tier_c import (
    look_at,
    project,
    render_tier_c,
    sample_trajectory,
)
from spatial_subspace.scene import Scene
from spatial_subspace.utils import load_yaml


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tier_c.yaml"
CONFIG_FREE6DOF = Path(__file__).resolve().parents[1] / "configs" / "tier_c_free6dof.yaml"


def test_look_at_world_origin_projects_to_image_center():
    eye = np.array([5.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    R, t = look_at(eye, target, up=np.array([0.0, 0.0, 1.0]))

    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    proj = project(np.array([0.0, 0.0, 0.0]), R, t, K)
    assert proj is not None
    u, v, depth = proj
    assert abs(u - 50.0) < 1e-6
    assert abs(v - 50.0) < 1e-6
    assert abs(depth - 5.0) < 1e-6


def test_project_depth_far_object_smaller_apparent_radius():
    """Two points at the same image position but different depth — the closer
    one has a larger apparent size."""
    eye = np.array([0.0, 0.0, -10.0])  # camera looking +z
    target = np.array([0.0, 0.0, 0.0])
    R, t = look_at(eye, target, up=np.array([0.0, 1.0, 0.0]))
    K = np.array([[400.0, 0.0, 200.0], [0.0, 400.0, 200.0], [0.0, 0.0, 1.0]])

    near = project(np.array([0.0, 0.0, -5.0]), R, t, K)  # 5 units in front
    far = project(np.array([0.0, 0.0, 0.0]), R, t, K)    # 10 units in front
    assert near is not None and far is not None
    assert near[2] < far[2]
    # apparent radius scales as 1/depth
    r_near = K[0, 0] * 0.5 / near[2]
    r_far = K[0, 0] * 0.5 / far[2]
    assert r_near > r_far


def test_project_returns_none_for_point_behind_camera():
    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 5.0])
    R, t = look_at(eye, target, up=np.array([0.0, 1.0, 0.0]))
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    behind = project(np.array([0.0, 0.0, -3.0]), R, t, K)
    assert behind is None


def test_sample_trajectory_length_and_orbit_radius():
    cfg = load_yaml(CONFIG)
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    poses = sample_trajectory(scene, cfg, rng, traj_idx=0)
    assert len(poses) == cfg["n_frames"]
    # All eyes should be at the configured altitude
    altitudes = [p[0][2] for p in poses]
    assert all(abs(a - cfg["trajectory"]["altitudes"][0]) < 1e-6 for a in altitudes)


def test_two_trajectories_are_meaningfully_different():
    cfg = load_yaml(CONFIG)
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    poses_a = sample_trajectory(scene, cfg, rng, traj_idx=0)
    poses_b = sample_trajectory(scene, cfg, rng, traj_idx=1)
    eyes_a = np.array([p[0] for p in poses_a])
    eyes_b = np.array([p[0] for p in poses_b])
    # The two trajectories should differ on average by at least 1 world unit
    assert np.linalg.norm(eyes_a - eyes_b, axis=1).mean() > 1.0


def test_render_tier_c_from_existing_scene(tmp_path):
    cfg = load_yaml(CONFIG)
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    out = render_tier_c(scene, cfg, tmp_path, rng, traj_idx=0)

    assert out.tier == "C"
    assert out.scene_id == f"{scene.scene_id}_t0"
    assert out.extras["base_scene_id"] == scene.scene_id
    assert out.extras["trajectory_idx"] == 0
    assert len(out.frames) == cfg["n_frames"]
    assert all(f.camera.kind == "perspective" for f in out.frames)

    scene_dir = tmp_path / out.scene_id
    assert (scene_dir / "scene.json").exists()
    for i in range(cfg["n_frames"]):
        assert (scene_dir / "frames" / f"{i:03d}.png").exists()
        assert (scene_dir / "masks" / f"{i:03d}.png").exists()

    # Round-trip
    loaded = Scene.load(scene_dir)
    assert loaded.tier == "C"
    assert loaded.extras["base_scene_id"] == scene.scene_id
    assert len(loaded.objects) == len(scene.objects)


def test_render_tier_c_object_visibility_across_orbit(tmp_path):
    """Across the full orbit, every object should be visible in at least one frame."""
    cfg = load_yaml(CONFIG)
    rng = random.Random(7)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    out = render_tier_c(scene, cfg, tmp_path, rng, traj_idx=0)
    scene_dir = tmp_path / out.scene_id

    seen = set()
    for f in out.frames:
        m = np.array(Image.open(scene_dir / f.mask_path))
        seen |= set(int(x) for x in np.unique(m)) - {0}
    expected = {o.object_id + 1 for o in scene.objects}
    assert seen == expected, f"missing objects in mask: {expected - seen}"


def test_free6dof_trajectory_varies_radius_altitude_and_roll():
    cfg = load_yaml(CONFIG_FREE6DOF)
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    poses = sample_trajectory(scene, cfg, rng, traj_idx=0)
    assert len(poses) == cfg["n_frames"]

    eyes = np.array([p[0] for p in poses])
    targets = np.array([p[1] for p in poses])
    rolls = np.array([p[2] for p in poses])

    # Altitude must actually vary (orbit mode keeps it constant).
    assert eyes[:, 2].std() > 0.05, "free6dof altitude should drift"
    # Distance from scene center (radius) should also vary.
    radii = np.linalg.norm(eyes[:, :2] - eyes[:, :2].mean(axis=0), axis=1)
    assert radii.std() > 0.1, "free6dof orbit radius should drift"
    # Look-at should drift, not stay locked at scene center.
    assert targets.std(axis=0).sum() > 0.05, "free6dof look-at should drift"
    # Roll should vary frame-to-frame and be bounded by config.
    roll_max = math.radians(cfg["trajectory"]["free6dof"]["roll_max_degrees"])
    assert rolls.std() > 0.0
    assert np.max(np.abs(rolls)) <= roll_max + 1e-9


def test_free6dof_every_frame_has_at_least_one_visible_object(tmp_path):
    """Visibility-repair must guarantee ≥1 object per frame in the rendered mask."""
    cfg = load_yaml(CONFIG_FREE6DOF)
    rng = random.Random(3)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    out = render_tier_c(scene, cfg, tmp_path, rng, traj_idx=0)
    scene_dir = tmp_path / out.scene_id

    for f in out.frames:
        m = np.array(Image.open(scene_dir / f.mask_path))
        ids = set(int(x) for x in np.unique(m)) - {0}
        assert ids, f"frame {f.frame_id} has no visible object in mask"
    assert out.extras["trajectory_mode"] == "free6dof"


def test_free6dof_two_trajectories_are_different():
    cfg = load_yaml(CONFIG_FREE6DOF)
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    poses_a = sample_trajectory(scene, cfg, rng, traj_idx=0)
    poses_b = sample_trajectory(scene, cfg, rng, traj_idx=1)
    eyes_a = np.array([p[0] for p in poses_a])
    eyes_b = np.array([p[0] for p in poses_b])
    assert np.linalg.norm(eyes_a - eyes_b, axis=1).mean() > 1.0


def test_render_tier_c_two_trajectories_produce_different_first_frames(tmp_path):
    cfg = load_yaml(CONFIG)
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    out_a = render_tier_c(scene, cfg, tmp_path, rng, traj_idx=0)
    out_b = render_tier_c(scene, cfg, tmp_path, rng, traj_idx=1)
    assert out_a.scene_id != out_b.scene_id

    img_a = np.array(Image.open(tmp_path / out_a.scene_id / out_a.frames[0].image_path))
    img_b = np.array(Image.open(tmp_path / out_b.scene_id / out_b.frames[0].image_path))
    # Different trajectories → first frames should not be pixel-identical
    assert not np.array_equal(img_a, img_b)
