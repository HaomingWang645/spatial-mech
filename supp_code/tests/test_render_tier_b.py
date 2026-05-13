import random
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_subspace.render.common import generate_3d_scene
from spatial_subspace.render.tier_b import render_tier_b, sample_trajectory
from spatial_subspace.scene import Scene
from spatial_subspace.utils import load_yaml


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tier_b.yaml"


def test_sample_trajectory_length_and_bounds():
    cfg = load_yaml(CONFIG)
    rng = random.Random(0)
    traj = sample_trajectory(cfg, rng)
    assert len(traj) == cfg["n_frames"]
    half = cfg["trajectory"]["window_world_size"] / 2
    x_lo, x_hi = cfg["working_volume"]["x"]
    y_lo, y_hi = cfg["working_volume"]["y"]
    for cx, cy, h in traj:
        assert h == half  # zoom_sigma = 0 in the default config
        # Clamping: window must stay inside the working volume
        assert cx - h >= x_lo - 1e-6
        assert cx + h <= x_hi + 1e-6
        assert cy - h >= y_lo - 1e-6
        assert cy + h <= y_hi + 1e-6


def test_render_tier_b_from_existing_scene(tmp_path):
    cfg = load_yaml(CONFIG)
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    out = render_tier_b(scene, cfg, tmp_path, rng)

    assert out.tier == "B"
    assert out.scene_id == scene.scene_id
    assert len(out.frames) == cfg["n_frames"]
    assert all(f.camera.kind == "orthographic" for f in out.frames)

    scene_dir = tmp_path / out.scene_id
    assert (scene_dir / "scene.json").exists()
    for i in range(cfg["n_frames"]):
        assert (scene_dir / "frames" / f"{i:03d}.png").exists()
        assert (scene_dir / "masks" / f"{i:03d}.png").exists()

    loaded = Scene.load(scene_dir)
    assert loaded.tier == "B"
    assert len(loaded.frames) == cfg["n_frames"]
    assert len(loaded.objects) == len(scene.objects)


def test_tier_b_trajectory_actually_moves(tmp_path):
    cfg = load_yaml(CONFIG)
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    out = render_tier_b(scene, cfg, tmp_path, rng)

    # The camera centers should not all be identical — random walk has nonzero variance.
    centers = [
        (f.camera.extrinsics[0][3], f.camera.extrinsics[1][3]) for f in out.frames
    ]
    cxs = [-c[0] for c in centers]  # extrinsics store -cx
    cys = [-c[1] for c in centers]
    assert max(cxs) - min(cxs) > 0.1
    assert max(cys) - min(cys) > 0.1


def test_tier_b_object_visibility_per_frame(tmp_path):
    """Most frames should contain at least one object somewhere in the mask;
    every object should be visible in at least one frame across the whole video."""
    cfg = load_yaml(CONFIG)
    rng = random.Random(7)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    out = render_tier_b(scene, cfg, tmp_path, rng)
    scene_dir = tmp_path / out.scene_id

    seen_objects = set()
    for f in out.frames:
        m = np.array(Image.open(scene_dir / f.mask_path))
        ids_in_frame = set(int(x) for x in np.unique(m)) - {0}
        seen_objects |= ids_in_frame

    expected = {o.object_id + 1 for o in scene.objects}
    # At least 80% of objects should appear somewhere in the video
    assert len(seen_objects) >= 0.8 * len(expected), (
        f"only {len(seen_objects)}/{len(expected)} objects visible across {cfg['n_frames']} frames"
    )


def test_temporal_shuffle_permutes_frames(tmp_path):
    cfg = load_yaml(CONFIG)
    cfg["temporal_shuffle"] = True
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    out = render_tier_b(scene, cfg, tmp_path, rng)
    # frame_id values should still cover [0, n_frames) but the list order may differ
    fids = [f.frame_id for f in out.frames]
    assert sorted(fids) == list(range(cfg["n_frames"]))
