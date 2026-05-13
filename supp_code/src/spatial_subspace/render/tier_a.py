"""Tier A — single BEV image rendered from a canonical 3D scene.

Pipeline is two-stage by design:

    generate_3d_scene(cfg, rng)         -> Scene with objects, frames=[]
    render_tier_a(scene, cfg, out_dir)  -> Scene with tier="A", one Frame

Tier B/C add their own renderers that consume the same 3D Scenes, so
downstream probes see identical ground truth across tiers.

BEV is orthographic by definition so we render with PIL rather than full
Blender — this gives pixel-exact segmentation masks and ~1000× faster
generation. The underlying scene is still genuinely 3D (objects sit on a
floor with real z extents); Tier A just projects it orthographically and
discards z.

Shadows. A strictly vertical sun would hide shadows under the object from
a top-down view, so we simulate a fixed oblique sun: azimuth ``SUN_AZ_DEG``
(measured from +x counter-clockwise, so 135° = NW) and elevation
``SUN_ELEV_DEG``. Each object's shadow is its own ground footprint shifted
along the sun-opposite direction by ``height / tan(elevation)``, drawn on
the floor before any object is drawn so shadows never occlude objects.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ..scene import Camera, Frame, Scene
from ..utils import ensure_dir, load_yaml, set_seed
from .common import generate_3d_scene


SUN_AZ_DEG = 135.0     # sun at NW → shadows cast toward SE
SUN_ELEV_DEG = 55.0    # elevation; shadow length = height / tan(elev)
SHADOW_RGB = (85, 85, 85)


def _shadow_offset_world(height: float) -> tuple[float, float]:
    """Ground-plane offset of the shadow of a point at altitude ``height``."""
    length = height / math.tan(math.radians(SUN_ELEV_DEG))
    # Shadow direction is opposite to the sun direction.
    az = math.radians(SUN_AZ_DEG + 180.0)
    return length * math.cos(az), length * math.sin(az)


def _world_to_image(
    xyz: tuple[float, float, float],
    world_range: tuple[float, float],
    image_size: int,
) -> tuple[float, float]:
    """Orthographic top-down: world (x, y) → image (col, row). +y world = up."""
    x, y, _ = xyz
    lo, hi = world_range
    u = (x - lo) / (hi - lo) * image_size
    v = (1.0 - (y - lo) / (hi - lo)) * image_size
    return u, v


def _radius_px(size_world: float, world_range: tuple[float, float], image_size: int) -> float:
    lo, hi = world_range
    return size_world / (hi - lo) * image_size


def render_tier_a(scene: Scene, cfg: dict[str, Any], out_dir: Path) -> Scene:
    """Render a BEV orthographic view of an existing 3D Scene.

    Returns a new Scene (same ``scene_id`` and object list) with ``tier="A"``
    and one Frame appended. Writes the image, mask, and enriched ``scene.json``
    into ``out_dir/<scene_id>/``.
    """
    image_size = int(cfg["image_size"])
    bg = int(cfg["background_gray"])
    x_lo, x_hi = cfg["working_volume"]["x"]
    y_lo, y_hi = cfg["working_volume"]["y"]
    assert x_lo == y_lo and x_hi == y_hi, "Tier A assumes a square working volume"
    world_range = (float(x_lo), float(x_hi))

    img = Image.new("RGB", (image_size, image_size), (bg, bg, bg))
    mask = Image.new("L", (image_size, image_size), 0)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    # Pass 1: shadows on the ground — drawn before any object so objects sit
    # on top of their own and others' shadows.
    for obj in scene.objects:
        size_world = float(cfg["sizes"][obj.size])
        r = _radius_px(size_world, world_range, image_size)
        sdx, sdy = _shadow_offset_world(2.0 * size_world)
        sc_world = (obj.centroid[0] + sdx, obj.centroid[1] + sdy, 0.0)
        su, sv = _world_to_image(sc_world, world_range, image_size)
        sbbox = (su - r, sv - r, su + r, sv + r)
        if obj.shape == "cube":
            draw_img.rectangle(sbbox, fill=SHADOW_RGB)
        else:
            draw_img.ellipse(sbbox, fill=SHADOW_RGB)

    # Pass 2: objects (image + mask).
    for obj in scene.objects:
        u, v = _world_to_image(obj.centroid, world_range, image_size)
        size_world = float(cfg["sizes"][obj.size])
        r = _radius_px(size_world, world_range, image_size)
        rgb = tuple(cfg["colors"][obj.color])
        bbox = (u - r, v - r, u + r, v + r)
        mid = obj.object_id + 1  # 0 reserved for background
        if obj.shape == "cube":
            draw_img.rectangle(bbox, fill=rgb)
            draw_mask.rectangle(bbox, fill=mid)
        elif obj.shape in ("sphere", "cylinder"):
            draw_img.ellipse(bbox, fill=rgb)
            draw_mask.ellipse(bbox, fill=mid)
        else:
            raise ValueError(f"unknown shape: {obj.shape}")

    scene_dir = ensure_dir(out_dir / scene.scene_id)
    ensure_dir(scene_dir / "frames")
    ensure_dir(scene_dir / "masks")
    img.save(scene_dir / "frames" / "000.png")
    mask.save(scene_dir / "masks" / "000.png")

    fx = fy = image_size / (x_hi - x_lo)
    cx = cy = image_size / 2.0
    K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    E = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    new_frame = Frame(
        frame_id=0,
        image_path="frames/000.png",
        mask_path="masks/000.png",
        camera=Camera(intrinsics=K, extrinsics=E, kind="orthographic"),
    )

    scene_out = Scene(
        scene_id=scene.scene_id,
        tier="A",
        objects=scene.objects,
        frames=[*scene.frames, new_frame],
        qa=list(scene.qa),
        extras={
            **scene.extras,
            "image_size": image_size,
            "world_range": list(world_range),
        },
    )
    scene_out.save(scene_dir)
    return scene_out


def render_scene(cfg: dict[str, Any], rng: random.Random, out_dir: Path) -> Scene:
    """Single-step convenience: sample a 3D scene and render its Tier A view."""
    scene = generate_3d_scene(cfg, rng, tier="3D")
    return render_tier_a(scene, cfg, out_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render Tier A scenes")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--scenes-in",
        type=str,
        default=None,
        help="Directory of pre-generated 3D scenes (from generate_scenes.py). "
             "If set, --n-scenes is ignored.",
    )
    p.add_argument("--n-scenes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    out = ensure_dir(Path(args.out))
    set_seed(args.seed)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_):
            return x

    if args.scenes_in is not None:
        scenes_dir = Path(args.scenes_in)
        scene_dirs = sorted(
            d for d in scenes_dir.iterdir() if d.is_dir() and (d / "scene.json").exists()
        )
        for d in tqdm(scene_dirs, desc="Tier A (render existing)"):
            render_tier_a(Scene.load(d), cfg, out)
    else:
        n = int(args.n_scenes if args.n_scenes is not None else cfg.get("n_scenes", 5000))
        rng = random.Random(args.seed)
        for _ in tqdm(range(n), desc="Tier A (sample + render)"):
            render_scene(cfg, rng, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
