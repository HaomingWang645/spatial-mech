"""Scene-sampling logic shared across tiers.

The 3D scene is the canonical unit: we sample a physical object layout once,
and every tier renders a different view of the same scene (BEV for Tier A,
panning BEV crops for Tier B, ego-centric perspective video for Tier C).
Keeping the underlying 3D ground truth identical across tiers is what makes
cross-tier comparisons meaningful — any difference in probe quality must
come from the view, not from a different underlying world.
"""
from __future__ import annotations

import random
import uuid
from typing import Any

from ..scene import Object3D, Scene
from .qa import generate_qa


def new_scene_id(prefix: str = "s") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _floor_z(cfg: dict[str, Any]) -> float:
    # Accept both the new top-level key and the legacy working_volume.z
    if "floor_z" in cfg:
        return float(cfg["floor_z"])
    return float(cfg.get("working_volume", {}).get("z", 0.0))


def sample_scene_contents(cfg: dict[str, Any], rng: random.Random) -> list[Object3D]:
    """Sample a physical 3D object layout.

    Objects sit on a floor at ``floor_z``. Horizontal placement uses rejection
    sampling against ``min_separation``. Each object's vertical extent is
    [floor_z, floor_z + 2*size]; the centroid sits at floor_z + size, which
    gives spheres/cubes/cylinders a physically plausible resting pose.
    """
    n = rng.randint(cfg["min_objects"], cfg["max_objects"])
    x_lo, x_hi = cfg["working_volume"]["x"]
    y_lo, y_hi = cfg["working_volume"]["y"]
    floor_z = _floor_z(cfg)
    shapes: list[str] = cfg["shapes"]
    colors: list[str] = list(cfg["colors"].keys())
    sizes: list[str] = list(cfg["sizes"].keys())
    min_sep = float(cfg["min_separation"])

    xy_positions: list[tuple[float, float]] = []
    tries = 0
    while len(xy_positions) < n and tries < 5000:
        tries += 1
        x = rng.uniform(x_lo, x_hi)
        y = rng.uniform(y_lo, y_hi)
        if all(
            (x - px) ** 2 + (y - py) ** 2 >= min_sep ** 2 for px, py in xy_positions
        ):
            xy_positions.append((x, y))
    if len(xy_positions) < n:
        raise RuntimeError(
            f"could not place {n} objects in working volume after {tries} tries"
        )

    used: set[tuple[str, str]] = set()
    objects: list[Object3D] = []
    for i, (x, y) in enumerate(xy_positions):
        shape = rng.choice(shapes)
        color = rng.choice(colors)
        for _ in range(20):
            if (shape, color) not in used:
                break
            shape = rng.choice(shapes)
            color = rng.choice(colors)
        used.add((shape, color))
        size_name = rng.choice(sizes)
        s = float(cfg["sizes"][size_name])
        cz = floor_z + s
        objects.append(
            Object3D(
                object_id=i,
                shape=shape,
                color=color,
                size=size_name,
                centroid=(x, y, cz),
                bbox_min=(x - s, y - s, floor_z),
                bbox_max=(x + s, y + s, floor_z + 2 * s),
            )
        )
    return objects


def generate_3d_scene(
    cfg: dict[str, Any],
    rng: random.Random,
    tier: str = "3D",
) -> Scene:
    """Sample a canonical 3D Scene with no rendered frames.

    The returned Scene has objects, ground-truth-derived QA, and ``frames=[]``.
    Per-tier renderers consume this Scene and add Frame entries.
    """
    objects = sample_scene_contents(cfg, rng)
    return Scene(
        scene_id=new_scene_id("s"),
        tier=tier,
        objects=objects,
        frames=[],
        qa=generate_qa(objects, rng),
        extras={
            "floor_z": _floor_z(cfg),
            "working_volume": cfg["working_volume"],
        },
    )
