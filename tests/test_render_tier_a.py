import random
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_subspace.render.common import generate_3d_scene
from spatial_subspace.render.tier_a import render_scene, render_tier_a
from spatial_subspace.scene import Scene
from spatial_subspace.utils import load_yaml


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tier_a.yaml"


def test_generate_3d_scene_has_no_frames():
    cfg = load_yaml(CONFIG)
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    assert scene.tier == "3D"
    assert scene.frames == []
    assert cfg["min_objects"] <= len(scene.objects) <= cfg["max_objects"]
    # Genuine 3D: every object's centroid sits strictly above the floor
    # (cz = floor_z + size > 0 for any positive size).
    for o in scene.objects:
        assert o.centroid[2] > 0.0
        assert o.bbox_min[2] == cfg.get("floor_z", 0.0)
        assert o.bbox_max[2] > o.bbox_min[2]


def test_generate_3d_scene_objects_dont_overlap_in_xy():
    """Pairwise check: d(i,j) ≥ r_i + r_j + min_gap (new contract)."""
    cfg = load_yaml(CONFIG)
    rng = random.Random(1)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    gap = float(cfg.get("min_gap", cfg.get("min_separation", 0.2)))
    sizes = cfg["sizes"]
    for i in range(len(scene.objects)):
        for j in range(i + 1, len(scene.objects)):
            a, b = scene.objects[i], scene.objects[j]
            d2 = (a.centroid[0] - b.centroid[0]) ** 2 + (a.centroid[1] - b.centroid[1]) ** 2
            r_i = float(sizes[a.size])
            r_j = float(sizes[b.size])
            need = r_i + r_j + gap
            assert d2 >= need ** 2 - 1e-9, (
                f"objects {i} (r={r_i}) and {j} (r={r_j}) overlap: "
                f"d={d2 ** 0.5:.3f} < {need:.3f}"
            )


def test_generate_3d_scene_shape_color_pairs_are_unique():
    """Every (shape, color) combination must appear at most once per scene."""
    cfg = load_yaml(CONFIG)
    for seed in range(8):
        rng = random.Random(seed)
        scene = generate_3d_scene(cfg, rng, tier="3D")
        combos = [(o.shape, o.color) for o in scene.objects]
        assert len(set(combos)) == len(combos), (
            f"seed={seed}: duplicate (shape, color) pair in {combos}"
        )


def test_render_tier_a_from_existing_scene(tmp_path):
    cfg = load_yaml(CONFIG)
    rng = random.Random(0)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    out = render_tier_a(scene, cfg, tmp_path)
    assert out.tier == "A"
    assert len(out.frames) == 1
    assert out.scene_id == scene.scene_id  # 3D ground truth survives the render

    scene_dir = tmp_path / out.scene_id
    assert (scene_dir / "scene.json").exists()
    assert (scene_dir / "frames" / "000.png").exists()
    assert (scene_dir / "masks" / "000.png").exists()

    # Round-trip the saved scene.
    loaded = Scene.load(scene_dir)
    assert loaded.tier == "A"
    assert len(loaded.frames) == 1
    assert len(loaded.objects) == len(scene.objects)


def test_render_tier_a_does_not_mutate_input_scene(tmp_path):
    cfg = load_yaml(CONFIG)
    rng = random.Random(2)
    scene = generate_3d_scene(cfg, rng, tier="3D")
    orig_tier = scene.tier
    orig_frames = list(scene.frames)
    _ = render_tier_a(scene, cfg, tmp_path)
    assert scene.tier == orig_tier
    assert scene.frames == orig_frames


def test_render_scene_single_step(tmp_path):
    cfg = load_yaml(CONFIG)
    rng = random.Random(0)
    scene = render_scene(cfg, rng, tmp_path)
    assert scene.tier == "A"
    assert len(scene.frames) == 1

    scene_dir = tmp_path / scene.scene_id
    assert (scene_dir / "scene.json").exists()
    assert (scene_dir / "frames" / "000.png").exists()
    assert (scene_dir / "masks" / "000.png").exists()

    mask = np.array(Image.open(scene_dir / "masks" / "000.png"))
    object_ids_in_mask = set(int(x) for x in np.unique(mask)) - {0}
    expected = {o.object_id + 1 for o in scene.objects}
    assert object_ids_in_mask <= expected
    assert len(object_ids_in_mask) == len(scene.objects)


def test_render_multiple_scenes_distinct_ids(tmp_path):
    cfg = load_yaml(CONFIG)
    rng = random.Random(42)
    ids = set()
    for _ in range(5):
        scene = render_scene(cfg, rng, tmp_path)
        ids.add(scene.scene_id)
    assert len(ids) == 5
