"""Scene-sampling logic shared across tiers.

The 3D scene is the canonical unit: we sample a physical object layout once,
and every tier renders a different view of the same scene (BEV for Tier A,
panning BEV crops for Tier B, ego-centric perspective video for Tier C).
Keeping the underlying 3D ground truth identical across tiers is what makes
cross-tier comparisons meaningful — any difference in probe quality must
come from the view, not from a different underlying world.

Invariants enforced by ``sample_scene_contents``:
  - **Unique (shape, color) combinations.** Every object in a scene has a
    distinct (shape, color) pair — sampled without replacement from the
    Cartesian product, so uniqueness is guaranteed rather than attempted.
  - **No overlap.** Horizontal placement uses rejection sampling against the
    actual object radii: d(i,j) ≥ r_i + r_j + ``min_gap``. The old global
    ``min_separation`` was smaller than 2·max_radius, so two large objects
    could overlap; the per-pair check fixes that.
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


def _min_gap(cfg: dict[str, Any]) -> float:
    # ``min_gap`` is the new name; ``min_separation`` is kept as a fallback
    # for older configs but now means edge-to-edge gap, not centre-to-centre
    # distance. 0.2 is the default if neither is set.
    if "min_gap" in cfg:
        return float(cfg["min_gap"])
    if "min_separation" in cfg:
        return float(cfg["min_separation"])
    return 0.2


def sample_scene_contents(cfg: dict[str, Any], rng: random.Random) -> list[Object3D]:
    """Sample a physical 3D object layout.

    Each object rests on the floor at ``floor_z``: its centroid sits at
    ``floor_z + size`` and its bbox spans ``[floor_z, floor_z + 2*size]``
    vertically (a physically plausible pose for cubes/spheres/cylinders of
    radius/half-extent ``size``).

    Guarantees:
      - Every object has a unique (shape, color) pair. If
        ``max_objects`` > |shapes| × |colors|, it is silently clamped.
      - No two objects overlap horizontally: for every pair (i, j),
        d(i, j) ≥ r_i + r_j + min_gap.
    """
    x_lo, x_hi = cfg["working_volume"]["x"]
    y_lo, y_hi = cfg["working_volume"]["y"]
    floor_z = _floor_z(cfg)
    shapes: list[str] = list(cfg["shapes"])
    colors: list[str] = list(cfg["colors"].keys())
    sizes: list[str] = list(cfg["sizes"].keys())
    min_gap = _min_gap(cfg)

    # Step 1: sample (shape, color) combinations without replacement.
    combos_all = [(s, c) for s in shapes for c in colors]
    n_requested = rng.randint(cfg["min_objects"], cfg["max_objects"])
    n = min(n_requested, len(combos_all))
    combos = rng.sample(combos_all, n)

    # Step 2: assign sizes. Placing larger objects first makes rejection
    # sampling succeed more often in tight working volumes.
    size_names = [rng.choice(sizes) for _ in range(n)]
    radii = [float(cfg["sizes"][sn]) for sn in size_names]
    order = sorted(range(n), key=lambda i: -radii[i])

    # Step 3: per-object rejection sampling against actual radii.
    placed: dict[int, tuple[float, float]] = {}
    for i in order:
        r_i = radii[i]
        lo_x, hi_x = x_lo + r_i, x_hi - r_i
        lo_y, hi_y = y_lo + r_i, y_hi - r_i
        if lo_x >= hi_x or lo_y >= hi_y:
            raise RuntimeError(
                f"object radius {r_i} exceeds working volume; shrink sizes or "
                f"enlarge working_volume"
            )
        for _ in range(5000):
            x = rng.uniform(lo_x, hi_x)
            y = rng.uniform(lo_y, hi_y)
            ok = True
            for j, (px, py) in placed.items():
                need = r_i + radii[j] + min_gap
                if (x - px) ** 2 + (y - py) ** 2 < need * need:
                    ok = False
                    break
            if ok:
                placed[i] = (x, y)
                break
        else:
            raise RuntimeError(
                f"could not place object {i} (r={r_i}) after 5000 tries; "
                f"reduce max_objects or enlarge working_volume"
            )

    # Step 4: assemble objects in canonical order (0..n-1).
    objects: list[Object3D] = []
    for i in range(n):
        x, y = placed[i]
        s = radii[i]
        cz = floor_z + s
        shape, color = combos[i]
        objects.append(
            Object3D(
                object_id=i,
                shape=shape,
                color=color,
                size=size_names[i],
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
