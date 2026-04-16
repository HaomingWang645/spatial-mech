#!/usr/bin/env python
"""Sample canonical 3D scenes — tier-agnostic ground truth.

This is the first stage of the data pipeline. It samples physical 3D object
layouts and writes each one as a ``scene.json`` file with ``frames=[]``.
Downstream tier renderers (Tier A BEV, Tier B panning BEV, Tier C ego video)
load these scenes and add their own views.

Example:
    python scripts/generate_scenes.py \
        --config configs/tier_a.yaml --out data/scenes_3d --n-scenes 5000
    python scripts/render_tier_a.py \
        --config configs/tier_a.yaml --scenes-in data/scenes_3d --out data/tier_a
"""
import _bootstrap  # noqa: F401

import argparse
import random
from pathlib import Path

from spatial_subspace.render.common import generate_3d_scene
from spatial_subspace.utils import ensure_dir, load_yaml, set_seed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-scenes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tier",
        default="3D",
        help="Tier label written to scene.json. Per-tier renderers overwrite this.",
    )
    args = p.parse_args()

    cfg = load_yaml(args.config)
    n = int(args.n_scenes if args.n_scenes is not None else cfg.get("n_scenes", 5000))
    out = ensure_dir(Path(args.out))
    set_seed(args.seed)
    rng = random.Random(args.seed)

    try:
        from tqdm import tqdm
        it = tqdm(range(n), desc="3D scenes")
    except ImportError:
        it = range(n)

    for _ in it:
        scene = generate_3d_scene(cfg, rng, tier=args.tier)
        scene_dir = ensure_dir(out / scene.scene_id)
        scene.save(scene_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
