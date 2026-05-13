#!/usr/bin/env python
"""Generate a multi-trajectory-length person-walk dataset.

Each base scene gets N trajectories, each with a different frame count (the
default is 16 / 32 / 64 frames). Output structure is:

    <out>/
        <scene_id>_t0/   # n_frames = frame_counts[0]
            scene.json, frames/, masks/
        <scene_id>_t1/   # n_frames = frame_counts[1]
        <scene_id>_t2/   # n_frames = frame_counts[2]
        ...

All trajectories of the same base scene share the same canonical 3D layout
(same object positions / shapes / colours).

Usage:
    python scripts/generate_person_walk_dataset.py \
        --config configs/tier_c_person_walk.yaml \
        --out data/tier_c_person_walk \
        --n-scenes 100 --frame-counts 16,32,64 --seed 0
"""
import _bootstrap  # noqa: F401

import argparse
import random
from pathlib import Path

from spatial_subspace.render.common import generate_3d_scene
from spatial_subspace.render.tier_c import render_tier_c
from spatial_subspace.utils import ensure_dir, load_yaml, set_seed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-scenes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--frame-counts", default="16,32,64",
        help="Comma-separated frame counts; one trajectory per value.",
    )
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    out_root = ensure_dir(Path(args.out))
    set_seed(args.seed)
    frame_counts = [int(x) for x in args.frame_counts.split(",")]

    try:
        from tqdm import tqdm
        outer = tqdm(range(args.n_scenes), desc="person-walk scenes")
    except ImportError:
        outer = range(args.n_scenes)

    master_rng = random.Random(args.seed)
    n_ok = 0
    for i in outer:
        base_seed = master_rng.randint(0, 2**31 - 1)
        scene_rng = random.Random(base_seed)
        scene = generate_3d_scene(cfg, scene_rng, tier="3D")
        for traj_idx, n_frames in enumerate(frame_counts):
            cfg_this = dict(cfg)
            cfg_this["n_frames"] = n_frames
            traj_rng = random.Random(base_seed + traj_idx * 12345)
            try:
                render_tier_c(scene, cfg_this, out_root, traj_rng, traj_idx=traj_idx)
            except Exception as e:
                print(f"[skip] scene {scene.scene_id} t{traj_idx} (n={n_frames}): "
                      f"{type(e).__name__}: {e}")
        n_ok += 1
    print(f"generated {n_ok} base scenes × {len(frame_counts)} trajectories → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
